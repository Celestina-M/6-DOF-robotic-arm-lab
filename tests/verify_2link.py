"""
DH 正向运动学手写公式验证

用标准 DH 变换矩阵逐连杆手写累乘，独立复算末端位姿，
与 robotic_arm_new.RoboticArm.forward_kinematics 的输出对比。
两者完全一致即证明 FK 实现的 DH 约定与矩阵乘法顺序正确。

运行: python tests/verify_2link.py
"""

import numpy as np
import sys
import os

# 与其他测试统一：把仓库根目录加入 sys.path 后以 src 包方式导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.robotic_arm_new import RoboticArm


def dh_matrix(theta, d, a, alpha):
    """标准 DH 变换矩阵（与 src/robotic_arm_new.py 中的约定一致）"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0]
    ])


def handwritten_fk(joint_angles, dh_params):
    """手写 FK：逐连杆累乘 DH 矩阵"""
    T = np.eye(4)
    for i in range(len(joint_angles)):
        theta_offset, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(theta_offset + joint_angles[i], d, a, alpha)
    return T


def verify_math():
    arm = RoboticArm(num_joints=6)

    # 多组位形覆盖零位、正限位附近、混合符号，避免单点验证的偶然性
    test_configs = [
        np.zeros(6),
        [0.5, 0.5, -0.5, 0.0, 0.5, 0.0],
        [-1.2, 0.8, 1.0, -0.6, 1.5, -2.0],
        [np.pi / 4, -np.pi / 3, np.pi / 6, np.pi / 2, -np.pi / 4, np.pi / 3],
    ]

    print("--- DH 正向运动学验证（代码 vs 手写公式） ---")
    all_pass = True
    for idx, q in enumerate(test_configs):
        arm.set_joint_angles(q)
        T_code = arm.forward_kinematics()
        T_hand = handwritten_fk(np.asarray(q, dtype=np.float64), arm.dh_params)

        err = np.max(np.abs(T_code - T_hand))
        ok = np.allclose(T_code, T_hand, atol=1e-12)
        all_pass &= ok
        pos = T_code[:3, 3]
        print(f"用例 {idx + 1}: q = {np.round(q, 3)}")
        print(f"  末端位置: {np.round(pos, 6)}")
        print(f"  最大矩阵元素误差: {err:.3e}  {'✓ 通过' if ok else '✗ 失败'}")

    print()
    if all_pass:
        print("✅ 全部通过：FK 实现与手写 DH 公式完全一致。")
    else:
        print("❌ 存在不一致，请检查 DH 参数或矩阵乘法顺序。")
    return all_pass


if __name__ == "__main__":
    success = verify_math()
    sys.exit(0 if success else 1)
