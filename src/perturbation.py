# 文件: src/perturbation.py
"""
动力学参数摄动模块

用于"模型失配"实验：对被控对象（plant）的动力学参数施加随机摄动，
控制器与状态估计器仍使用名义模型，从而量化模型误差的影响。

典型用法（配合 RobotSystem）：
    import copy
    system = RobotSystem(mode='dynamics')       # 控制器绑定名义模型
    plant = copy.deepcopy(system.arm)           # 复制一份作为被控对象
    apply_parameter_perturbation(plant, rel=0.2, seed=42)
    system.arm = plant                          # 物理真值用摄动模型
    # system.estimator_model 保持名义模型（默认在 __init__ 绑定）
"""
import numpy as np


def apply_parameter_perturbation(arm, rel=0.2, seed=None,
                                 mass_factors=None, com_factors=None):
    """
    对 RoboticArmDynamics 实例施加逐连杆参数摄动（就地修改）。

    参数:
        arm: RoboticArmDynamics 实例（将作为被控对象）
        rel: 相对摄动幅度，逐连杆因子采样自 [1-rel, 1+rel] 均匀分布
        seed: 随机种子
        mass_factors / com_factors: 显式指定逐连杆因子（优先级高于随机采样）

    返回:
        (mass_factors, com_factors) 实际使用的因子，供记录/复现
    """
    rng = np.random.default_rng(seed)
    n = arm.num_joints

    if mass_factors is None:
        mass_factors = 1.0 + rng.uniform(-rel, rel, n)
    if com_factors is None:
        com_factors = 1.0 + rng.uniform(-rel, rel, n)

    arm.link_masses = [m * f for m, f in zip(arm.link_masses, mass_factors)]
    arm.link_coms = [[c * f for c in com]
                     for com, f in zip(arm.link_coms, com_factors)]

    # 按 __init__ 中相同的圆柱模型重建惯性张量，保持参数自洽
    arm.link_inertias = []
    for m, l in zip(arm.link_masses, arm.link_lengths):
        r = max(l * 0.1, 0.02)
        I_axial = 0.5 * m * r**2
        I_radial = (1 / 12) * m * (3 * r**2 + l**2)
        I_axial = max(I_axial, 0.01)
        I_radial = max(I_radial, 0.01)
        arm.link_inertias.append(np.diag([I_radial, I_radial, I_axial]))

    return mass_factors, com_factors


def apply_friction_mismatch(arm, scale=1.5):
    """
    摩擦失配：放大被控对象的粘性摩擦系数（控制器仍用名义值）。
    模拟真实机械臂摩擦辨识不准的常见情形。
    """
    arm.friction_coeffs = arm.friction_coeffs * scale
