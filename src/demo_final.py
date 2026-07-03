# 文件: demo_final.py
"""
6-DOF 机械臂仿真系统 —— 最终答辩演示脚本

包含四个演示模块：
  demo_kinematics()  运动学：正向/逆向运动学 + 可视化
  demo_trajectory()  轨迹规划：三种插值方法对比
  demo_control()     动力学控制：CTC vs PID 追踪精度对比
  demo_learning()    学习算法：ILC 迭代学习 (+ 可选 BC 行为克隆)

用法：
  python demo_final.py            # 运行全部演示（含可视化窗口）
  直接运行即可，按顺序展示各模块能力。

说明：
  - 各演示函数均可独立调用。
  - SHOW_PLOTS 控制是否弹出可视化窗口，答辩时设 True，
    快速验证功能时可设 False 只看数值结果。
"""
import numpy as np
import matplotlib
# macOS 上使用 TkAgg 后端，避免 3D 渲染的 SystemError
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from robotic_system import RobotSystem

# 是否弹出可视化窗口（答辩演示设 True）
SHOW_PLOTS = True


def demo_kinematics():
    """演示 1：运动学（正向 + 逆向）"""
    print("\n" + "=" * 55)
    print("演示 1：运动学 (Forward / Inverse Kinematics)")
    print("=" * 55)

    system = RobotSystem(mode='kinematics')

    # --- 正向运动学：给定关节角，求末端位置 ---
    angles = [0.5, 0.5, -0.5, 0.0, 0.5, 0.0]
    system.arm.set_joint_angles(angles)
    pos = system.arm.get_end_effector_position()
    print(f"[正向运动学]")
    print(f"  关节角度: {np.round(angles, 3)}")
    print(f"  末端位置: {pos.round(4)} (m)")

    # --- 逆向运动学：给定末端位置，反解关节角 ---
    target = [0.3, 0.1, 0.4]
    success, sol = system.arm.inverse_kinematics(target, initial_guess=np.zeros(6))
    print(f"\n[逆向运动学]")
    print(f"  目标位置: {target}")
    print(f"  求解成功: {success}")
    if success:
        system.arm.set_joint_angles(sol)
        reached = system.arm.get_end_effector_position()
        err = np.linalg.norm(np.array(target) - reached)
        print(f"  求解角度: {sol.round(3)}")
        print(f"  实际到达: {reached.round(4)}, 误差: {err*1000:.3f} mm")

    # --- 可视化 ---
    if SHOW_PLOTS:
        system.visualizer.draw_arm(angles)
        system.visualizer.ax.set_title("Demo 1: Kinematics")
        plt.show()


def demo_trajectory():
    """演示 2：轨迹规划（三种插值方法对比）"""
    print("\n" + "=" * 55)
    print("演示 2：轨迹规划 (Trajectory Planning)")
    print("=" * 55)

    system = RobotSystem(mode='kinematics')
    start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end = [np.pi/2, np.pi/4, -np.pi/4, np.pi/2, 0.0, 0.0]
    duration = 2.0
    methods = ['linear', 'cubic', 'quintic']

    trajectories = {}
    for method in methods:
        traj = system.plan_trajectory(start, end, method=method,
                                      duration=duration, num_points=100)
        trajectories[method] = traj
        print(f"  [{method:8s}] 轨迹点数: {len(traj)}")

    print("\n  三种方法特性：")
    print("    linear  - 速度恒定，起停有冲击")
    print("    cubic   - 速度连续，端点加速度非零")
    print("    quintic - 速度/加速度两端归零，最平滑（推荐用于动力学追踪）")

    # --- 可视化：关节 0 的位置/速度/加速度曲线 ---
    if SHOW_PLOTS:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        colors = {'linear': 'red', 'cubic': 'green', 'quintic': 'blue'}
        for method in methods:
            traj = trajectories[method]
            t = [pt['time'] for pt in traj]
            p = [pt['position'][0] for pt in traj]
            v = [pt['velocity'][0] for pt in traj]
            a = [pt['acceleration'][0] for pt in traj]
            axes[0].plot(t, p, label=method, color=colors[method], linewidth=2)
            axes[1].plot(t, v, label=method, color=colors[method], linewidth=2)
            axes[2].plot(t, a, label=method, color=colors[method], linewidth=2)
        axes[0].set_title("Demo 2: Joint 0 Position"); axes[0].set_ylabel("rad")
        axes[1].set_title("Velocity"); axes[1].set_ylabel("rad/s")
        axes[2].set_title("Acceleration"); axes[2].set_ylabel("rad/s²")
        axes[2].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        plt.show()


def demo_control():
    """演示 3：动力学控制（CTC vs PID 精度对比）"""
    print("\n" + "=" * 55)
    print("演示 3：动力学控制 (Computed Torque vs PID)")
    print("=" * 55)

    system = RobotSystem(mode='dynamics')
    start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end = [0.5, 0.5, -0.5, 0.0, 0.5, 0.0]

    # 计算力矩控制 (CTC)：含完整动力学前馈，精度高、数值稳定
    traj_ctc = system.plan_trajectory(start, end, method='quintic',
                                      duration=2.0, num_points=100)
    res_ctc = system.execute_trajectory(traj_ctc, controller_name='computed_torque')
    rmse_ctc = np.sqrt(np.mean(np.array([r['error'] for r in res_ctc])**2))
    print(f"  [计算力矩控制 CTC] RMSE = {rmse_ctc:.6f} rad")

    # PID：无模型信息，需更小步长保证数值稳定（num_points 加倍 -> dt 减半）
    traj_pid = system.plan_trajectory(start, end, method='quintic',
                                      duration=2.0, num_points=200)
    res_pid = system.execute_trajectory(traj_pid, controller_name='pid')
    rmse_pid = np.sqrt(np.mean(np.array([r['error'] for r in res_pid])**2))
    print(f"  [PID 控制]          RMSE = {rmse_pid:.6f} rad")

    print(f"\n  对比结论：")
    print(f"    CTC 利用动力学模型做前馈线性化，精度比 PID 高约 "
          f"{rmse_pid/rmse_ctc:.0f} 倍；")
    print(f"    PID 无模型信息，存在重力引起的稳态误差，且对积分步长更敏感。")

    # --- 可视化：关节 0 追踪曲线 ---
    if SHOW_PLOTS:
        plt.figure(figsize=(10, 5))
        t_ctc = [r['time'] for r in res_ctc]
        actual = [r['position'][0] for r in res_ctc]
        desired = [r['q_desired'][0] for r in res_ctc]
        plt.plot(t_ctc, desired, 'k--', linewidth=2, label='Desired')
        plt.plot(t_ctc, actual, 'b-', linewidth=2, label='CTC Actual')
        plt.title(f"Demo 3: CTC Tracking (Joint 0), RMSE={rmse_ctc:.4f} rad")
        plt.xlabel("Time (s)"); plt.ylabel("Angle (rad)")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


def demo_learning():
    """演示 4：学习算法（ILC 迭代学习 + 可选 BC 行为克隆）"""
    print("\n" + "=" * 55)
    print("演示 4：学习算法 (Iterative Learning Control)")
    print("=" * 55)

    system = RobotSystem(mode='dynamics')

    # --- ILC 迭代学习：前馈学习 + PD 反馈 ---
    traj = system.plan_trajectory([0]*6, [0.5, 0.5, -0.5, 0.0, 0.5, 0.0],
                                  method='quintic', duration=2.0, num_points=100)
    print("  [ILC 迭代学习] 前馈学习 + PD 反馈，误差逐轮下降：")
    system.train_ilc_trajectory(traj, iterations=10)

    # --- BC 行为克隆（若已训练模型）---
    print("\n  [行为克隆 BC]")
    if 'behavioral_cloning' in system.controllers:
        import os
        if os.path.exists('bc_policy.pth'):
            print("    检测到已训练模型 bc_policy.pth，策略已加载。")
            print("    （运行 robotic_arm_imitation.py 可重新训练）")
        else:
            print("    未检测到 bc_policy.pth。")
            print("    先运行 python src/robotic_arm_imitation.py 训练模型。")
    else:
        print("    模仿学习模块未加载（通常是未安装 PyTorch）。")


if __name__ == "__main__":
    print("6-DOF 机械臂仿真系统 - 最终答辩演示")
    print("=" * 55)

    demo_kinematics()
    demo_trajectory()
    demo_control()
    demo_learning()

    print("\n" + "=" * 55)
    print("演示完成！")
    print("=" * 55)