# 文件: demo_final.py
"""
6-DOF 机械臂仿真系统

包含六个演示模块：
  demo_kinematics()  运动学：正向/逆向运动学 + 可视化
  demo_trajectory()  轨迹规划：三种插值方法对比
  demo_control()     动力学控制：CTC vs PID 追踪精度对比
  demo_learning()    学习算法：ILC 迭代学习 (+ 可选 BC 行为克隆)
  demo_impedance()   笛卡尔阻抗控制：柔顺让步 + 撤力回归（力控制范式）
  demo_animation()   机械臂运动动画：沿 quintic 轨迹实时播放

用法：
  python demo_final.py            # 运行全部演示（含可视化窗口）
  直接运行即可，按顺序展示各模块能力。

说明：
  - 各演示函数均可独立调用。
  - SHOW_PLOTS 控制是否弹出可视化窗口
    快速验证功能时可设 False 只看数值结果。
"""
import numpy as np
import matplotlib
# macOS 上使用 TkAgg 后端，避免 3D 渲染的 SystemError
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from robotic_system import RobotSystem

# 是否弹出可视化窗口
SHOW_PLOTS = True


def _close_visualizer_canvas(system):
    """
    关闭 RobotSystem 初始化时 Visualizer 自动创建的空 3D 画布。
    仅在"只画 2D 曲线、不使用 Visualizer 3D 显示"的演示中调用
    """
    import matplotlib.pyplot as plt
    fig = getattr(system.visualizer, 'fig', None)
    if fig is not None:
        plt.close(fig)


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
    _close_visualizer_canvas(system)   # 本演示只画 2D 曲线，关掉多余的空 3D 画布
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
    _close_visualizer_canvas(system)   # 本演示只画 2D 曲线，关掉多余的空 3D 画布
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
    _close_visualizer_canvas(system)   # 本演示不画 3D，关掉多余的空画布

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


def demo_impedance():
    """演示 5：笛卡尔阻抗控制（力控制范式：柔顺让步 + 自动回归）"""
    print("\n" + "=" * 55)
    print("演示 5：笛卡尔阻抗控制 (Cartesian Impedance Control)")
    print("=" * 55)

    from robotic_arm_dynamics import RoboticArmDynamics, CartesianImpedanceController

    arm = RoboticArmDynamics(num_joints=6)
    controller = CartesianImpedanceController(
        arm, K_p=[300.0, 300.0, 300.0], K_d=[35.0, 35.0, 35.0]
    )

    # 初始姿态与目标：末端沿 x 方向偏移 +10 cm
    q0 = np.array([0.0, 0.3, -0.5, 0.0, 0.3, 0.0])
    T0 = arm.forward_kinematics(q0)
    x_start = T0[0:3, 3]
    x_target = x_start + np.array([0.10, 0.0, 0.0])
    print(f"  初始末端位置: {x_start.round(4)}")
    print(f"  目标末端位置: {x_target.round(4)}")
    print(f"  外力扰动: t=1.2~1.5s 在 z 方向施加 -20 N")
    print(f"  仿真中（动力学计算较慢，约需十几秒）...")

    # 2.5 秒仿真：t=1.2~1.5s 施加 -20N 外力（z 方向）
    dt = 0.01
    t_arr = np.arange(0, 2.5, dt)
    q = q0.copy()
    q_dot = np.zeros(6)
    x_hist = []
    for t in t_arr:
        tau = controller.compute_control(x_target, q, q_dot)
        if 1.2 <= t <= 1.5:
            J = arm.jacobian(q)
            tau = tau + J.T @ np.array([0.0, 0.0, -20.0])
        qdd = arm.forward_dynamics(q, q_dot, tau)
        # 半隐式 Euler
        q_dot = q_dot + qdd * dt
        q = q + q_dot * dt
        T = arm.forward_kinematics(q)
        x_hist.append(T[0:3, 3].copy())
    x_hist = np.array(x_hist)

    final_err = np.linalg.norm(x_hist[-1] - x_target)
    idx_before = int(1.1 / dt)
    idx_during = int(1.4 / dt)
    z_push = (x_hist[idx_during, 2] - x_hist[idx_before, 2]) * 100
    print(f"\n  外力期间末端被推开约 {abs(z_push):.1f} cm（柔顺让步）")
    print(f"  撤力后回到目标，最终位置误差: {final_err*1000:.1f} mm")
    print(f"  结论: 末端表现为虚拟弹簧-阻尼系统 —— 受外力让步、撤力自动回归。")
    print(f"        与 CTC/ILC 的位置控制不同，这是力控制范式，适合接触类任务。")

    # --- 可视化：XYZ 位置曲线 + 外力扰动区间 ---
    if SHOW_PLOTS:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            axes[i].plot(t_arr, x_hist[:, i], label=f'Actual {labels[i]}')
            axes[i].axhline(x_target[i], color='r', linestyle='--',
                            label=f'Target {labels[i]}')
            axes[i].axvspan(1.2, 1.5, alpha=0.2, color='orange',
                            label='External force')
            axes[i].set_ylabel(f'{labels[i]} (m)')
            axes[i].legend(loc='best'); axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[0].set_title("Demo 5: Cartesian Impedance Control")
        axes[2].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()


def demo_animation():
    """演示 6：机械臂运动动画（实时播放）"""
    print("\n" + "=" * 55)
    print("演示 6：机械臂运动动画 (Real-time Animation)")
    print("=" * 55)

    system = RobotSystem(mode='kinematics')

    # 规划一条平滑的 quintic 轨迹，让机械臂从零位运动到目标姿态
    start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end = [0.8, 0.5, -0.5, 0.5, 0.5, 0.0]
    traj = system.plan_trajectory(start, end, method='quintic',
                                  duration=3.0, num_points=90)
    print(f"  规划轨迹: {len(traj)} 个点，quintic 插值，时长 3.0 s")

    if SHOW_PLOTS:
        print("  正在播放机械臂运动动画，请查看弹出的窗口...")
        # animate 会弹出实时播放窗口，展示机械臂沿轨迹平滑运动
        system.animate(traj, speed=1.0)
    else:
        print("  (SHOW_PLOTS=False，跳过动画播放)")


if __name__ == "__main__":
    print("6-DOF 机械臂仿真系统 - 最终答辩演示")
    print("=" * 55)

    demo_kinematics()
    demo_trajectory()
    demo_control()
    demo_learning()
    demo_impedance()
    demo_animation()

    print("\n" + "=" * 55)
    print("演示完成！")
    print("=" * 55)