# 文件: src/robotic_arm_reinforcement.py
"""
强化学习模块
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from robotic_arm_dynamics import RoboticArmDynamics

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Actor-Critic网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(ActorCritic, self).__init__()

        # Actor网络 (策略)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic网络 (价值函数)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 动作标准差 (可学习)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """前向传播"""
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

    def get_action(self, state):
        """采样动作"""
        action_mean, value = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        """
        PPO智能体

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: PPO裁剪参数
            epochs: 每次更新的训练轮数
        """
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs

        # 经验缓冲
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def save(self, path):
        """保存策略网络与优化器状态（支持断点续训）"""
        import torch
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """加载策略网络与优化器状态"""
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        self.policy.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)

        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        """存储转移"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        """更新策略"""
        # 计算回报和优势
        returns = []
        advantages = []

        R = 0
        for reward, value, done in zip(reversed(self.rewards),
                                       reversed(self.values),
                                       reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - value)

        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.epochs):
            # 重新计算动作概率和价值
            action_means, values = self.policy(states)
            std = torch.exp(self.policy.log_std)
            dist = Normal(action_means, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空缓冲
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        return actor_loss.item(), critic_loss.item()

# 环境包装器
class RobotArmEnv:
    def __init__(self, arm, use_gravity_comp=False, shaped_reward=False):
        """
        机械臂环境

        参数:
            arm: RoboticArmDynamics实例
            use_gravity_comp: 动作叠加重力补偿项（tau = π(a) + G(q)）。
                端到端力矩 RL 的探索-重力失配修复：初始探索噪声 std≈1 N·m，
                而重力力矩达 10~30 N·m，策略永远无法发现"抬起手臂"的动作，
                训练完全不收敛（实测 100 回合末端距离始终 ~0.9 m）。
            shaped_reward: 奖励整形（距离差分 + 到达奖励），
                替代稀疏的 -distance 累积奖励。
        """
        self.arm = arm
        self.use_gravity_comp = use_gravity_comp
        self.shaped_reward = shaped_reward
        self.state_dim = 12  # q (6) + q_dot (6)
        self.action_dim = 6  # 关节力矩

        # 目标位置
        self.target = np.array([0.5, 0.3, 0.4])

        # 时间步
        self.dt = 0.01
        self.max_steps = 500
        self.current_step = 0

        # 状态
        self.q = np.zeros(6)
        self.q_dot = np.zeros(6)
        self._prev_dist = None

    def reset(self):
        """重置环境"""
        self.q = np.random.uniform(-0.5, 0.5, 6)
        self.q_dot = np.zeros(6)
        self.current_step = 0
        self.arm.set_joint_angles(self.q)
        self._prev_dist = np.linalg.norm(
            self.arm.get_end_effector_position() - self.target)

        return self._get_state()

    def step(self, action):
        """执行一步"""
        # 应用动作 (力矩)，可选叠加重力补偿
        tau = np.clip(action, -50, 50)
        if self.use_gravity_comp:
            tau = tau + self.arm.gravity_vector(self.q)

        # 动力学仿真
        q_ddot = self.arm.forward_dynamics(self.q, self.q_dot, tau)
        self.q_dot = self.q_dot + q_ddot * self.dt
        self.q = self.q + self.q_dot * self.dt

        # 计算奖励
        self.arm.set_joint_angles(self.q)
        current_pos = self.arm.get_end_effector_position()
        distance = np.linalg.norm(current_pos - self.target)

        if self.shaped_reward:
            # 距离差分（每步最多 ±几厘米，量纲友好）+ 到达奖励
            reward = (self._prev_dist - distance) * 10.0
            if distance < 0.05:
                reward += 10.0
        else:
            reward = -distance  # 距离越小，奖励越大
        self._prev_dist = distance

        # 检查终止条件
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (distance < 0.05)

        return self._get_state(), reward, done, {}

    def _get_state(self):
        """获取状态"""
        return np.concatenate([self.q, self.q_dot])

# 测试PPO
def test_ppo():
    """测试PPO训练"""
    arm = RoboticArmDynamics(num_joints=6)
    env = RobotArmEnv(arm)

    # 创建智能体
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        lr=3e-4
    )

    # 训练
    num_episodes = 100
    episode_rewards = []

    print("开始PPO训练...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储转移
            agent.store_transition(state, action, log_prob, reward, value, done)

            state = next_state
            episode_reward += reward

        # 更新策略
        actor_loss, critic_loss = agent.update()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")

    # 绘制学习曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Progress')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_ppo()