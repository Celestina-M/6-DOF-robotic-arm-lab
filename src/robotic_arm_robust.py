# 文件: src/robotic_arm_robust.py
"""
鲁棒控制模块

针对动力学模型失配（质量/质心摄动、摩擦未建模、外扰）的鲁棒控制器：

1. SlidingModeController —— 滑模控制（Slotine-Li 型滑模面 + 边界层饱和）
   控制律（e = q_desired - q_actual）：
       q̇r  = q̇_desired + Λ·e
       q̈r  = q̈_desired + Λ·ė
       s   = q̇ - q̇r                     （滑模面；注意 s = -(ė + Λe)，
                                            符号反了 -K·s 就是正反馈）
       τ   = M(q)·q̈r + C(q,q̇)·q̇r + G(q) - K·s - ρ·sat(s/φ)
   前四项为名义动力学逆补偿（基于控制器持有的名义模型），
   -K·s 保证滑模面收敛，-ρ·sat(s/φ) 为鲁棒切换项：
   只要 ρ 不小于总扰动（模型误差 + 外扰）上界，即可保证误差有界收敛。
   边界层 φ 用饱和函数替代符号函数，消除抖振、将高频切换平滑为
   边界层内的线性反馈，代价是稳态误差 O(φ·ρ/K) 量级。

2. RobustComputedTorqueController —— 鲁棒计算力矩（CTC + 鲁棒饱和项）
   在普通 CTC 的基础上叠加切换项 -ρ·sat(s/φ)，
   用于隔离"鲁棒项本身"的贡献，便于消融对比。

接口与 ComputedTorqueController 完全一致（compute_control 五参签名），
可直接注册进 RobotSystem.controllers 走 execute_trajectory 统一管线。
"""

import numpy as np


def _sat(x, phi):
    """边界层饱和函数：|x|<=φ 时 x/φ，否则 sign(x)"""
    return np.clip(x / phi, -1.0, 1.0)


class SlidingModeController:
    """积分滑模控制器（Slotine-Li 型滑模面 + 积分抗扰 + 边界层饱和抑抖振）

    控制律（e = q_desired - q_actual，ż = e）：
        q̇r  = q̇_desired + Λ·e + Γ·z
        q̈r  = q̈_desired + Λ·ė + Γ·e
        s   = q̇ - q̇r                      （滑模面；s = -(ė + Λe + Γz)，
                                             符号反了 -K·s 就是正反馈）
        τ   = M̂·q̈r + Ĉ·q̇r + Ĝ + F̂·q̇ - K·s - ρ·sat(s/φ)

    三个关键设计点（本仓库调试记录，改动需谨慎）：
    1. 名义补偿必须显式使用 Ĉ(q,q̇)·q̇r，不能用
       inverse_dynamics(q, q̇, q̈r) 代替（后者是 C·q̇，相差 -C·s，
       C 非正定，小惯量关节上会变成负阻尼）。
    2. 名义补偿必须包含摩擦项 F̂·q̇：本模型的摩擦量级可达数 N·m
       （关节2 名义 5·q̇），失配实验中摩擦 ×1.5 后若不补偿，
       扰动将远超 ρ，直接发散。
    3. 滑模面带积分项 Γ·z：慢变/常值扰动（重力误差、摩擦残差）
       由积分器吸收，ρ 只需覆盖快变残差，可取得较小，
       从而满足离散时间稳定约束 dt·λ_max(M̂⁻¹(K + ρ/φ + ΓM̂)) < 2。
    """

    def __init__(self, robotic_dynamics, lam, K, rho, gamma=0.0,
                 boundary=0.05):
        """
        参数:
            robotic_dynamics: 名义动力学模型（RoboticArmDynamics 实例）。
                注意：应绑定"名义模型"而非被控对象——模型失配正是由
                名义模型与被控对象的差异构造的（见 perturbation.py）。
            lam: 滑模面带宽 Λ（标量或逐关节数组），决定误差收敛速度
            K:   滑模面收敛增益（标量或数组）
            rho: 鲁棒切换增益（标量或数组），应 ≥ 快变扰动上界估计
            gamma: 积分增益 Γ（标量或数组），用于吸收常值/慢变扰动；
                0 表示退化为纯 Slotine-Li 滑模
            boundary: 边界层厚度 φ。小 → 接近理想滑模（抖振强、等效
                小信号增益 ρ/φ 大，受离散稳定约束）；大 → 平滑但鲁棒性弱。
        """
        self.arm = robotic_dynamics
        n = robotic_dynamics.num_joints
        self.lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), (n,))
        self.K = np.broadcast_to(np.asarray(K, dtype=np.float64), (n,))
        self.rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n,))
        self.gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (n,))
        self.boundary = float(boundary)
        self._z = np.zeros(n)   # 积分器状态 ż = e

    def compute_control(self, q_desired, q_dot_desired, q_ddot_desired,
                        q_actual, q_dot_actual):
        """与 ComputedTorqueController 相同的五参签名"""
        q_desired = np.asarray(q_desired, dtype=np.float64)
        q_dot_desired = np.asarray(q_dot_desired, dtype=np.float64)
        q_ddot_desired = np.asarray(q_ddot_desired, dtype=np.float64)
        q_actual = np.asarray(q_actual, dtype=np.float64)
        q_dot_actual = np.asarray(q_dot_actual, dtype=np.float64)

        # 跟踪误差（注意符号约定：e = qd - q 时滑模面必须取 s = q̇ - q̇r，
        # 若写成 s = ė + Λe，则 -K·s 会变成正反馈而发散）
        e = q_desired - q_actual
        e_dot = q_dot_desired - q_dot_actual

        # 积分器更新（ż = e，与仿真同一步长离散）
        self._z = self._z + e * getattr(self, '_dt', 0.02)

        # 参考速度/加速度（含积分抗扰项）
        q_dot_r = q_dot_desired + self.lam * e + self.gamma * self._z
        q_ddot_r = q_ddot_desired + self.lam * e_dot + self.gamma * e

        # 滑模面：实际速度偏离参考速度的程度
        s = q_dot_actual - q_dot_r

        # 名义逆补偿：M̂·q̈r + Ĉ·q̇r + Ĝ + F̂·q̇
        M = self.arm.mass_matrix(q_actual)
        C = self.arm.coriolis_matrix(q_actual, q_dot_actual)
        G = self.arm.gravity_vector(q_actual)
        F = np.asarray(self.arm.friction_coeffs, dtype=np.float64)
        tau_nominal = M @ q_ddot_r + C @ q_dot_r + G + F * q_dot_actual

        # 滑模面收敛项 + 鲁棒切换项
        tau = tau_nominal - self.K * s \
            - self.rho * _sat(s, self.boundary)
        return tau

    def reset(self):
        """清空积分器（新轨迹执行前调用）"""
        self._z = np.zeros_like(self._z)


class RobustComputedTorqueController:
    """鲁棒计算力矩：CTC + 切换项（用于消融对比，隔离鲁棒项贡献）"""

    def __init__(self, robotic_dynamics, Kp, Kd, rho, lam=None,
                 boundary=0.05):
        """
        参数:
            robotic_dynamics: 名义动力学模型
            Kp, Kd: CTC 的 PD 增益（同 ComputedTorqueController）
            rho:    鲁棒切换增益
            lam:    滑模面带宽（仅用于构造切换项 s=ė+Λe，默认取 Kd）
            boundary: 边界层厚度
        """
        self.arm = robotic_dynamics
        n = robotic_dynamics.num_joints
        self.Kp = np.asarray(Kp, dtype=np.float64)
        self.Kd = np.asarray(Kd, dtype=np.float64)
        self.rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n,))
        self.lam = (np.broadcast_to(np.asarray(Kd, dtype=np.float64), (n,))
                    if lam is None
                    else np.broadcast_to(np.asarray(lam, dtype=np.float64), (n,)))
        self.boundary = float(boundary)

    def compute_control(self, q_desired, q_dot_desired, q_ddot_desired,
                        q_actual, q_dot_actual):
        q_desired = np.asarray(q_desired, dtype=np.float64)
        q_dot_desired = np.asarray(q_dot_desired, dtype=np.float64)
        q_ddot_desired = np.asarray(q_ddot_desired, dtype=np.float64)
        q_actual = np.asarray(q_actual, dtype=np.float64)
        q_dot_actual = np.asarray(q_dot_actual, dtype=np.float64)

        error = q_desired - q_actual
        error_dot = q_dot_desired - q_dot_actual
        # 滑模面 s = q̇ - q̇r（q̇r = q̇d + Λe），符号约定与滑模控制器一致：
        # 只有 s = q̇ - q̇r 时，-ρ·sat(s) 才是抑制偏离的负反馈
        q_dot_r = q_dot_desired + self.lam * error
        s = q_dot_actual - q_dot_r

        q_ddot_cmd = q_ddot_desired + self.Kp * error + self.Kd * error_dot
        tau = self.arm.inverse_dynamics(q_actual, q_dot_actual, q_ddot_cmd)
        tau = tau - self.rho * _sat(s, self.boundary)
        return tau

    def reset(self):
        pass
