# 文件: src/exp2_ppo.py
"""
实验 2 补充：PPO 点到点任务的训练（断点续训）与含噪鲁棒性评估

任务：RobotArmEnv 末端到达目标点 (0.5, 0.3, 0.4)，奖励 = -距离
训练：100 回合，每 5 回合存检查点（ppo_policy.pth + ppo_train_log.csv）
评估：确定性策略（actor 均值），反馈模式 {ideal, raw, kalman}
      × 噪声 {5e-4, 1e-3, 5e-3} × 5 种子，指标为回合末端执行器到目标的距离

用法:
    python src/exp2_ppo.py train
    python src/exp2_ppo.py eval [--workers 8]
    python src/exp2_ppo.py plot
"""
import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from robotic_arm_dynamics import RoboticArmDynamics
from robotic_arm_reinforcement import PPOAgent, RobotArmEnv
from sensing import SensingPipeline
from exp1_noise_sweep import _load_done_keys, save_csv

CKPT = 'ppo_policy.pth'
LOG = 'docs/ppo_train_log.csv'
EVAL_CSV = 'docs/exp2_ppo_eval.csv'
NOISE_LEVELS = [5e-4, 1e-3, 5e-3]


def train(total_episodes=100, save_every=5, batch_episodes=4):
    """训练：每 batch_episodes 回合累积一批样本再做一次 PPO 更新。

    失效案例教训：单回合（~500 样本、强相关）即时更新 + 10 epochs
    会在 30 回合后摧毁策略（实测距离 1.12→1.28 发散），批量更新是标准修复。
    """
    import torch
    arm = RoboticArmDynamics(num_joints=6)
    env = RobotArmEnv(arm, use_gravity_comp=True, shaped_reward=True)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     hidden_dim=128, lr=1e-4, epochs=5)

    start_ep = 0
    if os.path.exists(CKPT):
        agent.load(CKPT)
        if os.path.exists(LOG):
            import csv
            with open(LOG, encoding='utf-8-sig') as f:
                start_ep = sum(1 for _ in csv.DictReader(f))
        print(f"从第 {start_ep} 回合续训", flush=True)

    torch.manual_seed(0)
    np.random.seed(0)

    for ep in range(start_ep, total_episodes):
        state = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, log_prob, reward,
                                   value, done)
            state = next_state
            ep_reward += reward
        if (ep + 1) % batch_episodes == 0 or ep == total_episodes - 1:
            agent.update()

        final_dist = np.linalg.norm(
            arm.get_end_effector_position() - env.target)
        save_csv([dict(episode=ep, reward=ep_reward,
                       final_dist=final_dist)], LOG, append=True)
        if (ep + 1) % save_every == 0:
            agent.save(CKPT)
            print(f"  回合 {ep+1}/{total_episodes} 奖励 {ep_reward:.2f} "
                  f"末端距离 {final_dist:.3f}（已存检查点）", flush=True)
    agent.save(CKPT)
    print("训练完成")


def eval_once(mode, noise_std, seed):
    """单次评估回合：确定性策略 + 传感通道，返回末端最终距离 (m)"""
    import torch
    arm = RoboticArmDynamics(num_joints=6)
    env = RobotArmEnv(arm, use_gravity_comp=True, shaped_reward=True)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     hidden_dim=128)
    agent.load(CKPT)
    agent.policy.eval()

    np.random.seed(seed)
    torch.manual_seed(seed)
    pipe = SensingPipeline(dt=env.dt, mode=mode, noise_std=noise_std,
                           seed=seed, process_noise=1.0)

    state = env.reset()
    pipe.reset(q_init=env.q.copy())
    ddq_pred = np.zeros(6)
    for _ in range(env.max_steps):
        q_fb, dq_fb = pipe.process(env.q, env.q_dot, ddq_pred=ddq_pred)
        obs = np.concatenate([q_fb, dq_fb])
        with torch.no_grad():
            action = agent.policy.actor(
                torch.FloatTensor(obs)).numpy()
        _, _, done, _ = env.step(action)
        if pipe.mode == 'kalman' and pipe.input_aware:
            ddq_pred = arm.forward_dynamics(q_fb, dq_fb,
                                            np.clip(action, -50, 50))
        if done:
            break
    return float(np.linalg.norm(
        arm.get_end_effector_position() - env.target))


def _worker(task):
    return task['key'], eval_once(**task['kwargs'])


def evaluate(n_seeds=5, max_workers=8):
    done = _load_done_keys(EVAL_CSV, ['mode', 'noise', 'seed'])
    tasks = [dict(key=('ideal', 0.0, 0),
                  kwargs=dict(mode='ideal', noise_std=0.0, seed=0))]
    for mode in ('raw', 'kalman'):
        for noise in NOISE_LEVELS:
            for seed in range(n_seeds):
                tasks.append(dict(key=(mode, noise, seed),
                                  kwargs=dict(mode=mode, noise_std=noise,
                                              seed=seed)))
    tasks = [t for t in tasks
             if (t['key'][0], repr(t['key'][1]), str(t['key'][2])) not in done]
    if not tasks:
        print("评估已全部完成。")
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"共 {len(tasks)} 次评估，并行执行中...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, t): t['key'] for t in tasks}
        for fut in as_completed(futs):
            key, dist = fut.result()
            mode, noise, seed = key
            rows.append(dict(mode=mode, noise=noise, seed=seed,
                             final_dist=dist))
    save_csv(rows, EVAL_CSV, append=True)
    print("评估完成")


def plot():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 训练曲线
    log = pd.read_csv(LOG)
    ax = axes[0]
    ax.plot(log['episode'], log['reward'], alpha=0.3, color='#7570b3')
    ax.plot(log['episode'],
            log['reward'].rolling(10, min_periods=1).mean(),
            color='#7570b3', label='10-episode moving avg')
    ax.set_xlabel('episode')
    ax.set_ylabel('total reward')
    ax.set_title('PPO training curve (reaching task)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 含噪评估
    df = pd.read_csv(EVAL_CSV)
    ax = axes[1]
    colors = {'raw': '#d95f02', 'kalman': '#1b9e77'}
    ideal = df[df['mode'] == 'ideal']['final_dist'].iloc[0]
    ax.axhline(ideal, color='#666666', ls='--', lw=1.2,
               label=f'ideal ({ideal:.3f} m)')
    width = 0.35
    for k, mode in enumerate(['raw', 'kalman']):
        g = df[df['mode'] == mode].groupby('noise')['final_dist']
        x = np.arange(len(NOISE_LEVELS)) + (k - 0.5) * width
        ax.bar(x, g.mean().reindex(NOISE_LEVELS), width,
               yerr=g.std().reindex(NOISE_LEVELS), capsize=3,
               color=colors[mode], alpha=0.85, label=mode)
    ax.set_xticks(np.arange(len(NOISE_LEVELS)))
    ax.set_xticklabels([f'{n:g}' for n in NOISE_LEVELS])
    ax.set_xlabel('measurement noise std (rad)')
    ax.set_ylabel('final distance to target (m)')
    ax.set_title('PPO policy robustness (deterministic eval, 5 seeds)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = 'docs/exp2_ppo.png'
    fig.savefig(out, dpi=300)
    print(f"已保存: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'eval', 'plot'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    if args.action == 'train':
        train(total_episodes=args.episodes)
    elif args.action == 'eval':
        evaluate(max_workers=args.workers)
    else:
        plot()
