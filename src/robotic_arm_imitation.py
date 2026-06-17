# 文件: src/robotic_arm_imitation.py
"""
模仿学习模块
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from robotic_arm_dynamics import RoboticArmDynamics

class TrajectoryDataset(Dataset):
    def __init__(self, states, actions):
        """
        轨迹数据集

        参数:
            states: 状态数组 (N, state_dim)
            actions: 动作数组 (N, action_dim)
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        策略网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """前向传播"""
        return self.network(state)

class BehavioralCloningAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 learning_rate=1e-3):
        """
        行为克隆智能体

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
        """
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, dataset, epochs=100, batch_size=64):
        """
        训练策略网络

        参数:
            dataset: TrajectoryDataset实例
            epochs: 训练轮数
            batch_size: 批大小

        返回:
            losses: 训练损失历史
        """
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True)
        losses = []

        print("开始训练...")

        for epoch in range(epochs):
            epoch_loss = 0

            for states, actions in dataloader:
                # 前向传播
                predicted_actions = self.policy(states)

                # 计算损失
                loss = self.criterion(predicted_actions, actions)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def predict(self, state):
        """
        预测动作

        参数:
            state: 状态

        返回:
            action: 预测的动作
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.policy(state_tensor).squeeze(0).numpy()
        return action

    def save(self, path):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path))

# 数据收集
def collect_expert_demonstrations(arm, num_trajectories=10):
    """
    收集专家演示数据

    参数:
        arm: RoboticArmDynamics实例
        num_trajectories: 轨迹数量

    返回:
        states: 状态数组
        actions: 动作数组
    """
    from robotic_arm_dynamics import ComputedTorqueController

    # 使用计算力矩控制器作为"专家"
    kp = [200, 200, 200, 100, 100, 100]
    kd = [40, 40, 40, 20, 20, 20]
    expert = ComputedTorqueController(arm, kp, kd)

    states = []
    actions = []

    print(f"收集 {num_trajectories} 条专家演示...")

    for traj_idx in range(num_trajectories):
        # 随机目标轨迹
        t_total = 5.0
        dt = 0.01
        t = np.arange(0, t_total, dt)

        # 随机频率和幅度
        freq = np.random.uniform(0.1, 0.3)
        amp = np.random.uniform(0.3, 0.7)

        q0 = np.zeros(6)
        q_dot0 = np.zeros(6)

        q = q0.copy()
        q_dot = q_dot0.copy()

        for ti in t:
            # 目标轨迹
            q_d = amp * np.sin(2 * np.pi * freq * ti) * np.ones(6)
            q_dot_d = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * ti) * np.ones(6)
            q_ddot_d = -amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * ti) * np.ones(6)

            # 专家控制
            u = expert.compute_control(q_d, q_dot_d, q_ddot_d, q, q_dot)

            # 记录状态和动作
            state = np.concatenate([q, q_dot, q_d, q_dot_d])
            states.append(state)
            actions.append(u)

            # 更新状态
            q_ddot = arm.forward_dynamics(q, q_dot, u)
            q_dot = q_dot + q_ddot * dt
            q = q + q_dot * dt

        if (traj_idx + 1) % 2 == 0:
            print(f"完成 {traj_idx + 1}/{num_trajectories} 条轨迹")

    return np.array(states), np.array(actions)

# 测试模仿学习
def test_imitation_learning():
    """测试模仿学习"""
    arm = RoboticArmDynamics(num_joints=6)

    # 收集数据
    states, actions = collect_expert_demonstrations(arm, num_trajectories=20)

    print(f"\n数据集大小: {len(states)} 个样本")
    print(f"状态维度: {states.shape[1]}")
    print(f"动作维度: {actions.shape[1]}")

    # 创建数据集
    dataset = TrajectoryDataset(states, actions)

    # 创建智能体
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    agent = BehavioralCloningAgent(state_dim, action_dim)

    # 训练
    losses = agent.train(dataset, epochs=100, batch_size=64)

    # 绘制学习曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Behavioral Cloning Training')
    plt.grid(True)
    plt.show()

    # 测试学习到的策略
    print("\n测试学习到的策略...")

    q0 = np.zeros(6)
    q_dot0 = np.zeros(6)

    q = q0.copy()
    q_dot = q_dot0.copy()

    t_total = 5.0
    dt = 0.01
    t = np.arange(0, t_total, dt)

    q_trajectory = [q.copy()]

    for ti in t[:-1]:
        # 目标轨迹
        q_d = 0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6)
        q_dot_d = 0.5 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * ti) * np.ones(6)

        # 使用学习到的策略
        state = np.concatenate([q, q_dot, q_d, q_dot_d])
        u = agent.predict(state)

        # 更新状态
        q_ddot = arm.forward_dynamics(q, q_dot, u)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt

        q_trajectory.append(q.copy())

    q_trajectory = np.array(q_trajectory)

    # 绘制结果
    plt.figure(figsize=(12, 8))

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(t, q_trajectory[:, i], label='Learned Policy')
        q_ref = [0.5 * np.sin(2 * np.pi * 0.2 * ti) for ti in t]
        plt.plot(t, q_ref, '--', label='Reference')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Joint {i+1} (rad)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 保存模型
    agent.save('bc_policy.pth')
    print("\n模型已保存到 bc_policy.pth")

if __name__ == "__main__":
    test_imitation_learning()