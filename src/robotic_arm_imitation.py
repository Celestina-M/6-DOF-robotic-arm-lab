# 文件: src/robotic_arm_imitation.py
"""
模仿学习模块（改进版）

相比原版的主要改动：
1. 输入/输出归一化：状态各分量与力矩量级差异很大，归一化后训练更稳、最终 loss 更低。
2. 初始状态随机化：专家演示不再都从零位出发，扩大状态分布覆盖，提升泛化。
3. 半隐式 Euler 积分：先更新速度再用新速度更新位置，对机械系统比显式 Euler 稳定。
4. 归一化统计量随模型一起保存/加载，predict 时自动反归一化回真实力矩量级。
5. torch.load 显式指定 weights_only，避免新版 PyTorch 的兼容性警告。
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
        轨迹数据集（接收已归一化的数据）

        参数:
            states: 状态数组 (N, state_dim)，应为归一化后的值
            actions: 动作数组 (N, action_dim)，应为归一化后的值
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

        # 归一化统计量（在 fit_normalizer 中根据训练数据计算）
        # 初始化为单位变换（mean=0, std=1），未拟合时等价于不归一化
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_std = np.ones(state_dim, dtype=np.float32)
        self.action_mean = np.zeros(action_dim, dtype=np.float32)
        self.action_std = np.ones(action_dim, dtype=np.float32)
        self._normalizer_fitted = False

    # ---------- 归一化相关 ----------
    def fit_normalizer(self, states, actions, eps=1e-6):
        """
        根据原始（未归一化）训练数据计算均值和标准差。
        eps 防止某些维度方差为 0 时除零。
        """
        self.state_mean = states.mean(axis=0).astype(np.float32)
        self.state_std = (states.std(axis=0) + eps).astype(np.float32)
        self.action_mean = actions.mean(axis=0).astype(np.float32)
        self.action_std = (actions.std(axis=0) + eps).astype(np.float32)
        self._normalizer_fitted = True

    def normalize_states(self, states):
        return (states - self.state_mean) / self.state_std

    def normalize_actions(self, actions):
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions_norm):
        return actions_norm * self.action_std + self.action_mean

    # ---------- 训练 ----------
    def train(self, dataset, epochs=100, batch_size=64):
        """
        训练策略网络

        参数:
            dataset: TrajectoryDataset实例（数据应已归一化）
            epochs: 训练轮数
            batch_size: 批大小

        返回:
            losses: 训练损失历史（归一化空间的 MSE）
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

    # ---------- 推理 ----------
    def predict(self, state):
        """
        预测动作（接收原始状态，返回原始力矩量级的动作）

        参数:
            state: 原始（未归一化）状态

        返回:
            action: 预测的动作（真实力矩量级）
        """
        state = np.asarray(state, dtype=np.float32)
        # 归一化输入
        state_norm = self.normalize_states(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
            action_norm = self.policy(state_tensor).squeeze(0).numpy()
        # 反归一化回真实力矩量级
        return self.denormalize_actions(action_norm)

    # ---------- 持久化 ----------
    def save(self, path):
        """保存模型（含归一化统计量）"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'normalizer_fitted': self._normalizer_fitted,
        }, path)

    def load(self, path):
        """加载模型（含归一化统计量）

        weights_only=False：因为 checkpoint 里包含 numpy 数组等非张量对象。
        仅加载你自己保存的可信文件。
        """
        checkpoint = torch.load(path, weights_only=False)
        # 兼容旧格式（只存了 state_dict 的情况）
        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.state_mean = checkpoint['state_mean']
            self.state_std = checkpoint['state_std']
            self.action_mean = checkpoint['action_mean']
            self.action_std = checkpoint['action_std']
            self._normalizer_fitted = checkpoint.get('normalizer_fitted', True)
        else:
            # 旧版只有 state_dict，没有归一化参数（退化为不归一化）
            self.policy.load_state_dict(checkpoint)
            print("警告：加载的是旧格式模型，无归一化统计量，predict 将不做归一化。")


# 数据收集
def collect_expert_demonstrations(arm, num_trajectories=10):
    """
    收集专家演示数据

    改进点：
    - 初始关节角度随机化（不再都从零位出发），扩大状态分布覆盖。
    - 使用半隐式 Euler 积分（先更新速度，再用新速度更新位置）。

    参数:
        arm: RoboticArmDynamics实例
        num_trajectories: 轨迹数量

    返回:
        states: 状态数组 (N, state_dim)，原始未归一化
        actions: 动作数组 (N, action_dim)，原始未归一化
    """
    from robotic_arm_dynamics import ComputedTorqueController

    # 使用计算力矩控制器作为"专家"
    kp = [200, 200, 200, 100, 100, 100]
    kd = [40, 40, 40, 20, 20, 20]
    expert = ComputedTorqueController(arm, kp, kd)

    states = []
    actions = []

    print(f"收集 {num_trajectories} 条专家演示...")

    rng = np.random.default_rng()

    for traj_idx in range(num_trajectories):
        # 随机目标轨迹
        t_total = 5.0
        dt = 0.01
        t = np.arange(0, t_total, dt)

        # 随机频率和幅度
        freq = rng.uniform(0.1, 0.3)
        amp = rng.uniform(0.3, 0.7)

        # 初始状态随机化：让策略见到更多样的出发位置
        q = rng.uniform(-0.3, 0.3, 6)
        q_dot = np.zeros(6)

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

            # 半隐式 Euler：先更新速度，再用新速度更新位置
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

    # 收集数据（原始未归一化）
    states, actions = collect_expert_demonstrations(arm, num_trajectories=20)

    print(f"\n数据集大小: {len(states)} 个样本")
    print(f"状态维度: {states.shape[1]}")
    print(f"动作维度: {actions.shape[1]}")
    print(f"力矩量级（std）: {actions.std(axis=0).round(2)}")

    # 创建智能体
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    agent = BehavioralCloningAgent(state_dim, action_dim)

    # 关键：先用原始数据拟合归一化统计量，再归一化后喂给 dataset
    agent.fit_normalizer(states, actions)
    states_norm = agent.normalize_states(states)
    actions_norm = agent.normalize_actions(actions)
    dataset = TrajectoryDataset(states_norm, actions_norm)

    # 训练
    losses = agent.train(dataset, epochs=100, batch_size=64)

    # 绘制学习曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (normalized space)')
    plt.title('Behavioral Cloning Training (Normalized)')
    plt.grid(True)
    plt.show()

    # 测试学习到的策略
    print("\n测试学习到的策略...")

    q = np.zeros(6)
    q_dot = np.zeros(6)

    t_total = 5.0
    dt = 0.01
    t = np.arange(0, t_total, dt)

    q_trajectory = [q.copy()]

    for ti in t[:-1]:
        # 目标轨迹
        q_d = 0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6)
        q_dot_d = 0.5 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * ti) * np.ones(6)

        # 使用学习到的策略（predict 内部会自动归一化输入、反归一化输出）
        state = np.concatenate([q, q_dot, q_d, q_dot_d])
        u = agent.predict(state)

        # 半隐式 Euler 更新
        q_ddot = arm.forward_dynamics(q, q_dot, u)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt

        q_trajectory.append(q.copy())

    q_trajectory = np.array(q_trajectory)

    # 绘制结果（含跟踪误差量化）
    q_ref_full = np.array([
        0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6) for ti in t
    ])
    tracking_rmse = np.sqrt(np.mean((q_trajectory - q_ref_full) ** 2))
    print(f"闭环跟踪 RMSE: {tracking_rmse:.4f} rad")

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, q_trajectory[:, i], label='Learned Policy')
        plt.plot(t, q_ref_full[:, i], '--', label='Reference')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Joint {i+1} (rad)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 保存模型（含归一化统计量）
    agent.save('bc_policy.pth')
    print("\n模型已保存到 bc_policy.pth")


if __name__ == "__main__":
    test_imitation_learning()