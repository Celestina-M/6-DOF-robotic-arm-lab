import numpy as np
import trajectory
from robotic_arm_new import RoboticArm

if __name__ == "__main__":
    # 格式：[theta(关节角), d, a, alpha]，其中 d/a/alpha 为 UR5 固定几何参数
    start = [
            [0,          0.1625, 0,      0      ],  # Joint 1
            [0,          0,      0,     -np.pi/2],  # Joint 2
            [0,          0,      0.425,  0      ],  # Joint 3
            [0,          0,      0.3922, 0      ],  # Joint 4
            [0,          0.1333, 0,     -np.pi/2],  # Joint 5
            [0,          0.0997, 0,      np.pi/2]   # Joint 6
        ]
    endd = [
            [0,          0.1625, 0,      0      ],  # Joint 1
            [0,          0,      0,     -np.pi/2],  # Joint 2
            [-np.pi/2,   0,      0.425,  0      ],  # Joint 3: 关节角从 0 转到 -90°
            [0,          0,      0.3922, 0      ],  # Joint 4
            [0,          0.1333, 0,     -np.pi/2],  # Joint 5
            [0,          0.0997, 0,      np.pi/2]   # Joint 6
        ]

    # 提取每行第 0 列作为关节角度
    start_angles = [row[0] for row in start]
    end_angles   = [row[0] for row in endd]

    arm = RoboticArm(num_joints=6)
    planner = trajectory.TrajectoryPlanner(robotic_arm_new=arm)
    traj = planner.plan_joint_trajectory(start_angles, end_angles, duration=2.0, method='cubic', num_points=50)
    print(f"轨迹规划完成，共 {len(traj)} 个轨迹点")
    print(f"起点角度: {[round(a, 4) for a in traj[0]['position']]}")
    print(f"终点角度: {[round(a, 4) for a in traj[-1]['position']]}")