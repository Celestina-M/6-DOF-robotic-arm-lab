"""
逆运动学测试
"""

import numpy as np
import sys
import os

# 与其他测试统一：把仓库根目录加入 sys.path 后以 src 包方式导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.robotic_arm_new import RoboticArm

def test_ik_success_rate():
    """
    测试IK成功率
    目标：≥80%
    """
    arm = RoboticArm(num_joints=6)

    num_tests = 100
    success_count = 0
    errors = []

    print("逆运动学成功率测试：")
    print(f"测试次数：{num_tests}")

    for i in range(num_tests):
        # 生成随机目标（工作空间内）
        target = generate_random_target_in_workspace(arm)

        # IK求解
        success, angles = arm.inverse_kinematics(target, method='jacobian')
        if success:
            success_count += 1
            # 验证精度
            arm.set_joint_angles(angles)
            actual = arm.get_end_effector_position()
            error = np.linalg.norm(actual - target)
            errors.append(error)

    success_rate = success_count / num_tests * 100
    avg_error = np.mean(errors) if errors else float('inf')

    print(f"成功率: {success_rate:.1f}%")
    print(f"平均误差: {avg_error*100:.2f} cm")
    print(f"目标成功率: ≥80%, 误差＜2 cm")
    print(f"达标: {'√' if success_rate >= 80 and avg_error < 0.02 else '×'}")

def generate_random_target_in_workspace(arm):
    """
    在工作空间内生成随机目标位置
    """
    #简单方法：生成随机关节角度，然后计算FK
    random_angles = np.random.uniform(-np.pi/2, np.pi/2, 6)
    arm.set_joint_angles(random_angles)
    target_pos = arm.get_end_effector_position()
    return target_pos

if __name__ == "__main__":
    test_ik_success_rate()