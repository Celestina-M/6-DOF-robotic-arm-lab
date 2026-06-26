# 文件：src/robotic_arm_optimization.py
"""
优化控制模块（MPC + ILC，ILC 为重写版）

ILC 重写要点（针对原版"越学越发散"的根因）：
1. 删除重复的 ILCController 定义（原文件里有两个同名类，后者覆盖前者，
   且签名与 robotic_system.py 的调用不兼容）。现在只保留一个。
2. 前馈初始化：第一轮控制量用逆动力学基于"参考轨迹"算出，
   u_0[t] = inverse_dynamics(q_ref[t], q_dot_ref[t], q_ddot_ref[t])，
   而非只给重力。这样第一轮误差就很小，后续学习是在好起点上微调。
3. 学习律时序对齐（发散主因）：力矩 u[t] 影响的是 t 之后的状态，
   因此用"下一时刻误差" e[t+1] 来修正 u[t]，而不是同时刻 e[t]。
   采用 PD 型：u_{k+1}[t] = u_k[t] + Kp * e[t+1] + Kd * edot[t+1]。
4. 接口兼容 robotic_system.py：
   - 构造：ILCController(arm, learning_rate=0.3) 必须可用。
   - update(q_traj, q_dot_traj, q_ref_traj, q_dot_ref_traj=None)：
     第四个参数可选，不传时用 np.gradient 从 q_ref 数值微分得到参考速度，
     这样 robotic_system.py 的三参数调用 update(q_traj, dq_traj, q_ref) 也能跑。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from robotic_arm_dynamics import RoboticArmDynamics


# ============================================================
# MPC 控制器（保持不变，仅随文件一起保留）
# ============================================================
class MPCController:
    def __init__(self, arm_dynamics, horizon=10, dt=0.1):
        self.arm = arm_dynamics
        self.horizon = horizon
        self.dt = dt
        self.num_joints = arm_dynamics.num_joints
        self.Q = np.diag([100, 100, 100, 50, 50, 50])
        self.R = np.diag([1, 1, 1, 1, 1, 1])
        self.u_min = -np.array([50, 50, 50, 25, 25, 25])
        self.u_max = np.array([50, 50, 50, 25, 25, 25])

    def compute(self, q_current, q_dot_current, q_ref_trajectory):
        u_init = np.zeros(self.horizon * self.num_joints)

        def cost_function(u_sequence):
            u_sequence = u_sequence.reshape(self.horizon, self.num_joints)
            cost = 0
            q = q_current.copy()
            q_dot = q_dot_current.copy()
            for k in range(self.horizon):
                q_ddot = self.arm.forward_dynamics(q, q_dot, u_sequence[k])
                q_dot = q_dot + q_ddot * self.dt
                q = q + q_dot * self.dt
                if k < len(q_ref_trajectory):
                    q_error = q - q_ref_trajectory[k]
                else:
                    q_error = q - q_ref_trajectory[-1]
                cost += q_error.T @ self.Q @ q_error
                cost += u_sequence[k].T @ self.R @ u_sequence[k]
            return cost

        bounds = []
        for _ in range(self.horizon):
            for i in range(self.num_joints):
                bounds.append((self.u_min[i], self.u_max[i]))

        result = minimize(cost_function, u_init, method='SLSQP',
                          bounds=bounds, options={'maxiter': 100, 'disp': False})
        return result.x[:self.num_joints]


def test_mpc():
    """测试MPC轨迹跟踪"""
    arm = RoboticArmDynamics(num_joints=6)
    mpc = MPCController(arm, horizon=10, dt=0.1)
    t = np.arange(0, 5.0, 0.1)
    q_ref = np.array([0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6) for ti in t])

    q = np.zeros(6); q_dot = np.zeros(6)
    q_history = [q.copy()]
    for i in range(len(t) - mpc.horizon):
        u = mpc.compute(q, q_dot, q_ref[i:i + mpc.horizon])
        q_ddot = arm.forward_dynamics(q, q_dot, u)
        q_dot = q_dot + q_ddot * 0.1
        q = q + q_dot * 0.1
        q_history.append(q.copy())
    q_history = np.array(q_history)

    tracking_error = np.mean(np.linalg.norm(q_history - q_ref[:len(q_history)], axis=1))
    print(f"平均跟踪误差: {tracking_error:.4f} rad")


# ============================================================
# ILC 控制器（重写版，唯一定义）
# ============================================================
class ILCController:
    def __init__(self, arm_dynamics, learning_rate=0.3, dt=0.01,
                 Kp=None, Kd=None, u_limit=150.0,
                 use_qfilter=True, qfilter_cutoff=5.0):
        """
        迭代学习控制器（PD 型，时序对齐，逆动力学前馈初始化，Q-filter）

        参数:
            arm_dynamics: RoboticArmDynamics 实例
            learning_rate: 标量学习率。当未显式给出 Kp/Kd 时，
                           Kp 默认 = learning_rate，Kd 默认 = 0.1 * learning_rate。
                           （保留此参数是为兼容 robotic_system.py 的
                            ILCController(arm, learning_rate=0.3) 调用方式）
            dt: 时间步长，用于无参考速度时的 np.gradient 数值微分，
                以及 Q-filter 的采样频率
            Kp, Kd: 可选，PD 型学习增益（标量或长度 num_joints 的数组）。
                    不给则由 learning_rate 推出。
            u_limit: 力矩限幅，防止个别点数值异常导致发散
            use_qfilter: 是否启用 Q-filter（零相位低通滤波）。
                         ILC 只对"可重复的误差"有效，但高频噪声分量会被一轮轮
                         累积放大，在长轨迹（步数多）上尤其明显，会导致误差
                         先降后升直至发散。Q-filter 每轮对更新后的力矩序列做一次
                         零相位低通，滤掉高频累积成分，只保留低频可学习的修正量。
                         短轨迹可关，长轨迹强烈建议开。
            qfilter_cutoff: Q-filter 截止频率 (Hz)，越低越稳但响应越慢，默认 5Hz。
        """
        self.arm = arm_dynamics
        self.num_joints = arm_dynamics.num_joints
        self.dt = dt
        self.learning_rate = learning_rate

        # Kp/Kd 缺省时由 learning_rate 推导：D 项取 P 项的 1/10（经验值）
        self.Kp = float(learning_rate) if Kp is None else Kp
        self.Kd = 0.1 * float(learning_rate) if Kd is None else Kd
        self.Kp = np.asarray(self.Kp, dtype=float)
        self.Kd = np.asarray(self.Kd, dtype=float)

        self.u_limit = u_limit

        self.use_qfilter = use_qfilter
        self.qfilter_cutoff = qfilter_cutoff

        self.u_prev = None          # 上一轮控制力矩序列 (T, num_joints)
        self.e_prev = None          # 上一轮位置误差序列 (T, num_joints)

    def _qfilter(self, u_seq):
        """
        Q-filter：对每个关节的力矩序列做零相位低通滤波 (filtfilt)。
        filtfilt 前向+反向各滤一次，零相位、不引入时间延迟，
        把 ILC 累积的高频分量滤掉，只留下低频可学习的修正量。
        """
        fs = 1.0 / self.dt
        nyq = fs / 2.0
        wn = min(self.qfilter_cutoff / nyq, 0.99)  # 归一化截止频率 < 1
        b, a = butter(2, wn)                        # 2 阶巴特沃斯低通
        u_filt = np.zeros_like(u_seq)
        T = len(u_seq)
        for j in range(self.num_joints):
            u_filt[:, j] = filtfilt(b, a, u_seq[:, j], padlen=min(15, T - 1))
        return u_filt

    def update(self, q_trajectory, q_dot_trajectory,
               q_ref_trajectory, q_dot_ref_trajectory=None):
        """
        一次完整迭代后更新控制序列。

        参数:
            q_trajectory:        本轮实际关节角度轨迹 (T, num_joints)
            q_dot_trajectory:    本轮实际关节速度轨迹 (T, num_joints)
            q_ref_trajectory:    参考关节角度轨迹 (T, num_joints)
            q_dot_ref_trajectory: 参考关节速度轨迹 (T, num_joints)，可选。
                                  不传时用 np.gradient(q_ref, dt) 数值微分得到。
                                  （兼容 robotic_system.py 的三参数调用）

        返回:
            u_new: 更新后的控制力矩序列 (T, num_joints)
        """
        q_trajectory = np.asarray(q_trajectory, dtype=float)
        q_dot_trajectory = np.asarray(q_dot_trajectory, dtype=float)
        q_ref_trajectory = np.asarray(q_ref_trajectory, dtype=float)
        T = len(q_ref_trajectory)

        # 参考速度：未提供则用数值微分补出（沿时间轴）
        if q_dot_ref_trajectory is None:
            q_dot_ref_trajectory = np.gradient(q_ref_trajectory, self.dt, axis=0)
        else:
            q_dot_ref_trajectory = np.asarray(q_dot_ref_trajectory, dtype=float)

        # 位置误差 e 与速度误差 edot
        e = q_ref_trajectory - q_trajectory               # (T, num_joints)
        edot = q_dot_ref_trajectory - q_dot_trajectory    # (T, num_joints)

        # ---------- 前馈初始化（仅第一轮）----------
        # 用参考轨迹做逆动力学，得到能"大致跟住"参考的前馈力矩。
        # 参考加速度用 np.gradient 从参考速度数值微分得到。
        if self.u_prev is None:
            q_ddot_ref = np.gradient(q_dot_ref_trajectory, self.dt, axis=0)
            self.u_prev = np.zeros((T, self.num_joints))
            for t in range(T):
                self.u_prev[t] = self.arm.inverse_dynamics(
                    q_ref_trajectory[t], q_dot_ref_trajectory[t], q_ddot_ref[t]
                )

        # ---------- PD 型学习律（时序对齐）----------
        # 力矩 u[t] 影响 t 之后的状态，故用下一时刻误差 e[t+1] 修正 u[t]。
        # 末点没有 t+1，用自身误差兜底（也可保持不变，这里取自身更简单稳定）。
        e_next = np.roll(e, -1, axis=0)
        edot_next = np.roll(edot, -1, axis=0)
        e_next[-1] = e[-1]
        edot_next[-1] = edot[-1]

        u_new = self.u_prev + self.Kp * e_next + self.Kd * edot_next

        # Q-filter：滤掉本轮更新引入的高频累积分量（长轨迹收敛的关键）。
        # 序列太短时 filtfilt 会报错，故加长度保护。
        if self.use_qfilter and len(u_new) > 10:
            u_new = self._qfilter(u_new)

        # 力矩限幅，防止个别异常点放大发散
        u_new = np.clip(u_new, -self.u_limit, self.u_limit)

        self.u_prev = u_new
        self.e_prev = e
        return u_new

    def get_convergence_metric(self):
        """返回上一轮的平均跟踪误差（用于画收敛曲线）"""
        if self.e_prev is not None:
            return np.mean(np.linalg.norm(self.e_prev, axis=1))
        return float('inf')


def test_ilc():
    """测试 ILC 学习（重写版）"""
    arm = RoboticArmDynamics(num_joints=6)
    dt = 0.01
    # learning_rate 调到 0.2（更稳），Q-filter cutoff 3Hz（长轨迹滤得更狠）
    ilc = ILCController(arm, learning_rate=0.2, dt=dt,
                        use_qfilter=True, qfilter_cutoff=3.0)

    t = np.arange(0, 5.0, dt)

    # 参考轨迹：位置与速度（解析给出，最准）
    q_ref = np.array([0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6) for ti in t])
    q_dot_ref = np.array([
        0.5 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * ti) * np.ones(6) for ti in t
    ])

    # 轻量 PD 反馈增益。
    # 关键说明：本动力学模型的逆/正动力学在 Joint 4 上不完全自洽
    # （C 项是两层数值微分嵌套，小惯量的 Joint 4 受影响最大），
    # 导致纯开环前馈在 5 秒长轨迹上 Joint 4 会单调漂移（开环末端误差可达 ~1.1 rad）。
    # 这种"随时间累积的系统性漂移"不是 ILC 能学掉的（ILC 学的是可重复误差），
    # 纯开环 ILC 反而会在追漂移的过程中被高频化、发散。
    # 工程上 ILC 标准用法就是"前馈学习 + 反馈保稳定"：
    # 反馈把漂移压住，ILC 在其上学习精细前馈。两者叠加即可稳定收敛。
    Kp_fb = np.array([20., 20., 20., 20., 20., 20.])
    Kd_fb = np.array([4., 4., 4., 4., 4., 4.])

    # 前馈初始化提到循环前：让第一轮就用上逆动力学前馈力矩
    q_ddot_ref = np.gradient(q_dot_ref, dt, axis=0)
    ilc.u_prev = np.array([
        arm.inverse_dynamics(q_ref[k], q_dot_ref[k], q_ddot_ref[k])
        for k in range(len(t))
    ])

    num_iterations = 20
    errors = []
    print("开始 ILC 学习（重写版，前馈学习 + PD 反馈）...")

    q_trajectory = None
    for iteration in range(num_iterations):
        # 初始状态对齐参考起点（正弦参考 t=0 速度非零，避免初始速度缺口）
        q = q_ref[0].copy()
        q_dot = q_dot_ref[0].copy()
        q_traj = [q.copy()]
        q_dot_traj = [q_dot.copy()]

        u_sequence = ilc.u_prev   # ILC 学习到的前馈力矩

        for i in range(len(t) - 1):
            # 总力矩 = ILC 前馈 + PD 反馈
            # 前馈负责跟踪，反馈压住模型不自洽导致的漂移
            fb = Kp_fb * (q_ref[i] - q) + Kd_fb * (q_dot_ref[i] - q_dot)
            u = u_sequence[i] + fb
            q_ddot = arm.forward_dynamics(q, q_dot, u)
            # 半隐式 Euler：先更新速度，再用新速度更新位置
            q_dot = q_dot + q_ddot * dt
            q = q + q_dot * dt
            q_traj.append(q.copy())
            q_dot_traj.append(q_dot.copy())

        q_trajectory = np.array(q_traj)
        q_dot_trajectory = np.array(q_dot_traj)

        # 更新控制序列（显式传参考速度）
        ilc.update(q_trajectory, q_dot_trajectory, q_ref, q_dot_ref)
        error = ilc.get_convergence_metric()
        errors.append(error)
        print(f"迭代 {iteration+1}: 误差 = {error:.4f} rad")

    # 画收敛曲线 + 最终轨迹
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(errors, marker='o')
    plt.xlabel('Iteration'); plt.ylabel('Tracking Error (rad)')
    plt.title('ILC Learning Curve (Rewritten)'); plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(6):
        plt.plot(t, q_trajectory[:, i], label=f'Joint {i+1}')
    plt.plot(t, q_ref[:, 0], 'k--', linewidth=2, label='Reference')
    plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)')
    plt.title(f'Final Trajectory (Iteration {num_iterations})')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_mpc()
    test_ilc()