import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from robotic_arm_new import RoboticArm


def benchmark_forward_kinematics(arm):
    print("\n" + "="*45)
    print("      正向运动学性能基准测试 (UR5 Params)")
    print("="*45)

    num_tests = 50000
    tests_angles = np.random.uniform(-np.pi, np.pi, (num_tests, 6))

    # Numba 预热 (第一次编译)
    _ = arm.forward_kinematics(tests_angles[0])

    start_time = time.time()
    for angles in tests_angles:
        arm.forward_kinematics(angles)
    elapsed_time = time.time() - start_time

    ops_per_second = num_tests / elapsed_time

    print(f"总测试次数: {num_tests}")
    print(f"总耗时: {elapsed_time:.4f} 秒")
    print(f"性能：{ops_per_second:.0f} ops/second")
    print(f"目标：≥10,000 ops/second")
    print(f"达标：{'√' if ops_per_second >= 10000 else '×'}")
    return ops_per_second


def test_accuracy(arm):
    print("\n--- 精度测试 ---")

    # 全零位姿理论值 (robotic_arm_new.py DH 参数 [theta, d, a, alpha])：
    # a2=-0.425, a3=-0.3922 → X = -(0.425+0.3922) = -0.8172
    # alpha1=pi/2, alpha4=pi/2 → Y = -(d4+d6) = -(0.1333+0.0996) = -0.2329
    # Z = d1 - d5 = 0.1625 - 0.0997 = 0.0628
    test_cases = [
        {
            'name': 'UR5全零位姿',
            'angles': [0, 0, 0, 0, 0, 0],
            'expected': [-0.8172, -0.2329, 0.0628],
            'tolerance': 0.0001
        }
    ]

    for i, case in enumerate(test_cases):
        actual_pos = arm.forward_kinematics(case['angles'])[:3, 3]
        expected_pos = np.array(case['expected'])
        error = np.linalg.norm(actual_pos - expected_pos)

        print(f"用例 {i+1} [{case['name']}]:")
        print(f"  期望: {expected_pos}")
        print(f"  实际: {actual_pos.round(4)}")
        print(f"  误差: {error * 1000:.6f} mm")
        print(f"  达标：{'√' if error <= case['tolerance'] else '×'}")


if __name__ == "__main__":
    my_arm = RoboticArm(num_joints=6)
    benchmark_forward_kinematics(my_arm)
    test_accuracy(my_arm)
