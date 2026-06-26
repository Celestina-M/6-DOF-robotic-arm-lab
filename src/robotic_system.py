# 文件: src/robotic_system.py
"""
机械臂系统统一接口
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# 导入底层物理与控制模块
from robotic_arm_new import RoboticArm
from robotic_arm_dynamics import RoboticArmDynamics, PIDController, ComputedTorqueController
from trajectory import TrajectoryPlanner
from visualizer import Visualizer

#  进阶控制器导入
try:
    from robotic_arm_optimization import MPCController, ILCController
    from robotic_arm_imitation import BehavioralCloningAgent
    from robotic_arm_reinforcement import PPOAgent, RobotArmEnv
    ADVANCED_CTRL_AVAILABLE = True
except ImportError:
    print("提示: 未检测到进阶控制模块文件，将仅启用基础控制 (PID/CTC)。")
    ADVANCED_CTRL_AVAILABLE = False


class RobotSystem:
    """
    机械臂系统统一接口：整合运动学、动力学、轨迹规划、优化控制、AI控制与可视化
    """
    def __init__(self, mode='kinematics'):
        """初始化系统"""
        if mode == 'kinematics':
            self.arm = RoboticArm(num_joints=6)
        elif mode == 'dynamics':
            self.arm = RoboticArmDynamics(num_joints=6)
        else:
            raise ValueError("mode 必须是 'kinematics' 或 'dynamics'")

        self.mode = mode
        self.planner = TrajectoryPlanner(self.arm)
        self.visualizer = Visualizer(self.arm)

        self.controllers = {}
        if mode == 'dynamics':
            self._initialize_controllers()

    def _initialize_controllers(self):
        """初始化各类控制器"""
        # 1. 经典控制
        self.controllers['pid'] = PIDController(
            Kp=[100, 100, 100, 50, 50, 50],
            Ki=[10, 10, 10, 5, 5, 5],
            Kd=[20, 20, 20, 10, 10, 10]
        )
        self.controllers['computed_torque'] = ComputedTorqueController(
            robotic_dynamics=self.arm,
            Kp=[200, 200, 200, 100, 100, 100],
            Kd=[40, 40, 40, 20, 20, 20]
        )

        # 2. 进阶控制 (MPC, ILC, 模仿学习, 强化学习)
        if ADVANCED_CTRL_AVAILABLE:
            self.controllers['mpc'] = MPCController(self.arm, horizon=10, dt=0.01)
            self.controllers['ilc'] = ILCController(self.arm, learning_rate=0.2)

            # 模仿学习加载
            state_dim_bc = self.arm.num_joints * 4
            bc_agent = BehavioralCloningAgent(state_dim=state_dim_bc, action_dim=self.arm.num_joints)
            if os.path.exists('bc_policy.pth'):
                bc_agent.load('bc_policy.pth')
            self.controllers['behavioral_cloning'] = bc_agent

            # 强化学习加载
            ppo_agent = PPOAgent(state_dim=12, action_dim=self.arm.num_joints, hidden_dim=128)
            self.controllers['ppo'] = ppo_agent

    def plan_trajectory(self, start, end, method='cubic', **kwargs):
        """规划轨迹"""
        return self.planner.plan_joint_trajectory(start, end, method=method, **kwargs)

    def execute_trajectory(self, trajectory, controller_name='pid'):
        """执行常规轨迹追踪"""
        if self.mode != 'dynamics':
            raise ValueError("动力学模式才能执行轨迹")
        if controller_name not in self.controllers:
            raise KeyError(f"未找到控制器: {controller_name}")
        if controller_name in ['ilc', 'ppo']:
            raise ValueError(f"{controller_name} 是回合制学习算法，请使用专门的 train_ 接口。")

        controller = self.controllers[controller_name]
        results = []

        current_q = np.array(trajectory[0]['position'])
        current_dq = np.zeros(self.arm.num_joints)
        dt = trajectory[1]['time'] - trajectory[0]['time'] if len(trajectory) > 1 else 0.01

        for i, point in enumerate(trajectory):
            q_desired = np.array(point['position'])
            dq_desired = np.array(point.get('velocity', np.zeros(self.arm.num_joints)))
            ddq_desired = np.array(point.get('acceleration', np.zeros(self.arm.num_joints)))

            # 核心多态调用
            if controller_name == 'pid':
                tau = controller.compute_control(q_desired, current_q, current_dq)
            elif controller_name == 'computed_torque':
                tau = controller.compute_control(q_desired, dq_desired, ddq_desired, current_q, current_dq)
            elif controller_name == 'mpc':
                horizon = controller.horizon
                end_idx = min(i + horizon, len(trajectory))
                q_ref_horizon = [pt['position'] for pt in trajectory[i:end_idx]]
                while len(q_ref_horizon) < horizon:
                    q_ref_horizon.append(trajectory[-1]['position'])
                tau = controller.compute(current_q, current_dq, np.array(q_ref_horizon))
            elif controller_name == 'behavioral_cloning':
                state = np.concatenate([current_q, current_dq, q_desired, dq_desired])
                tau = controller.predict(state)

            # 动力学步进
            ddq_actual = self.arm.forward_dynamics(current_q, current_dq, tau)
            next_dq = current_dq + ddq_actual * dt
            next_q = current_q + current_dq * dt

            results.append({
                'time': point.get('time', 0.0),
                'position': current_q.copy(),
                'q_desired': q_desired.copy(),
                'error': (q_desired - current_q).copy()
            })

            current_q, current_dq = next_q, next_dq

        return results

    def train_ilc_trajectory(self, trajectory, iterations=10,
                             Kp_fb=20.0, Kd_fb=4.0):
        """
        ILC 迭代学习训练接口（前馈学习 + PD 反馈）

        与旧版的区别（对齐重写后的 ILCController）：
        1. 加 PD 反馈叠加：本动力学模型逆/正动力学在小惯量关节(如 Joint4)上
           不完全自洽，纯开环前馈在长轨迹上会单调漂移直至发散。
           ILC 学不掉这种随时间累积的系统漂移，故用"前馈学习 + 反馈保稳定"：
           总力矩 = ILC前馈 + PD反馈。
        2. 初始状态对齐参考速度：避免参考起点速度非零时的初始速度缺口。
        3. 循环前用逆动力学做前馈初始化：第一轮即有好起点。
        4. 显式传参考速度给 update（trajectory 无 velocity 时用 np.gradient 补）。

        参数:
            trajectory: 轨迹点列表，每点含 'position'，可选 'velocity'/'acceleration'
            iterations: 迭代回合数
            Kp_fb, Kd_fb: PD 反馈增益（标量或长度 num_joints 的数组）
        """
        if self.mode != 'dynamics':
            raise ValueError("需要动力学模式")

        controller = self.controllers['ilc']
        dt = trajectory[1]['time'] - trajectory[0]['time'] if len(trajectory) > 1 else 0.01

        # 参考位置
        q_ref = np.array([pt['position'] for pt in trajectory])

        # 参考速度：轨迹自带就用，否则用 np.gradient 从位置数值微分补出
        if all('velocity' in pt for pt in trajectory):
            q_dot_ref = np.array([pt['velocity'] for pt in trajectory])
        else:
            q_dot_ref = np.gradient(q_ref, dt, axis=0)

        # 参考加速度（用于前馈初始化）：自带就用，否则微分参考速度
        if all('acceleration' in pt for pt in trajectory):
            q_ddot_ref = np.array([pt['acceleration'] for pt in trajectory])
        else:
            q_ddot_ref = np.gradient(q_dot_ref, dt, axis=0)

        # 前馈初始化：让第一轮就用上逆动力学前馈力矩
        controller.dt = dt   # 确保 ILC 的 dt 与本轨迹一致（Q-filter / 微分需要）
        controller.u_prev = np.array([
            self.arm.inverse_dynamics(q_ref[k], q_dot_ref[k], q_ddot_ref[k])
            for k in range(len(trajectory))
        ])

        Kp_fb = np.asarray(Kp_fb, dtype=float)
        Kd_fb = np.asarray(Kd_fb, dtype=float)

        print(f"\n开始 ILC 迭代学习，共 {iterations} 个回合（前馈 + PD 反馈）...")
        q_traj = None
        for iteration in range(iterations):
            # 初始状态对齐参考起点（位置与速度都对齐，消除初始缺口）
            current_q = q_ref[0].copy()
            current_dq = q_dot_ref[0].copy()
            q_traj, dq_traj = [current_q.copy()], [current_dq.copy()]

            u_seq = controller.u_prev   # ILC 学到的前馈力矩

            for i in range(len(trajectory) - 1):
                # 总力矩 = ILC 前馈 + PD 反馈
                # 前馈负责跟踪，反馈压住模型不自洽导致的漂移
                fb = Kp_fb * (q_ref[i] - current_q) + Kd_fb * (q_dot_ref[i] - current_dq)
                tau = u_seq[i] + fb

                ddq_actual = self.arm.forward_dynamics(current_q, current_dq, tau)
                # 半隐式 Euler：先更新速度，再用新速度更新位置
                current_dq = current_dq + ddq_actual * dt
                current_q = current_q + current_dq * dt
                q_traj.append(current_q.copy())
                dq_traj.append(current_dq.copy())

            # 显式传参考速度（比 update 内部 np.gradient 更准）
            controller.update(np.array(q_traj), np.array(dq_traj), q_ref, q_dot_ref)
            print(f"   - 第 {iteration+1:02d} 回合 | 误差: {controller.get_convergence_metric():.6f} rad")

        return [{'time': pt.get('time', 0.0), 'position': q_traj[i]}
                for i, pt in enumerate(trajectory)]

    def animate(self, trajectory, speed=1.0):
        return self.visualizer.animate(trajectory, speed)


# ==========================================
# 综合系统测试
# ==========================================
if __name__ == "__main__":
    print("-" * 50)
    print("机械臂综合系统集成测试启动")
    print("-" * 50)

    system = RobotSystem(mode='dynamics')

    # 1. 轨迹规划
    start_pos = [0, 0, 0, 0, 0, 0]
    end_pos = [0.5, 0.5, -0.5, 0, 0.5, 0]
    print("正在规划五次多项式轨迹...")
    traj = system.plan_trajectory(start_pos, end_pos, method='quintic', duration=2.0, num_points=100)

    # 2. 执行计算力矩法 (CTC) 追踪
    print("\n[运行] 执行 Computed Torque Control 轨迹追踪...")
    results_ctc = system.execute_trajectory(traj, controller_name='computed_torque')

    # 3. 结果验证与分析
    errors_np = np.array([res['error'] for res in results_ctc])
    rmse = np.sqrt(np.mean(errors_np**2))
    print(f"执行完成！CTC追踪 RMSE: {rmse:.6f} rad")

    # 4. 可视化
    print("\n正在生成仿真物理动画...")
    system.animate(results_ctc, speed=1.0)