# 文件: src/exp2_all_controllers.py
"""
实验 2：全控制器含噪公平比较

Part A 追踪型控制器（PID / CTC / ILC / BC）：
    反馈模式 {ideal(基线), raw, kalman} × 噪声 {5e-4, 1e-3, 5e-3} × 5 种子
Part B MPC（算力约束下降配）：
    horizon=5, maxiter=20, 50 点轨迹, 噪声 5e-3, 单种子
    同时记录单步平均耗时（实现代价维度）
PPO 为点到点任务，单独见 exp2_ppo.py

用法:
    python src/exp2_all_controllers.py run [--workers 8]
    python src/exp2_all_controllers.py plot
"""
import argparse
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from robotic_system import RobotSystem
from exp1_noise_sweep import _load_done_keys, save_csv

START = [0, 0, 0, 0, 0, 0]
END = [0.5, 0.5, -0.5, 0, 0.5, 0]

# 追踪型控制器配置：名称 -> 轨迹点数（PID 需 dt≈0.01）
TRACKING = {
    'pid': 200,
    'computed_torque': 100,
    'sliding_mode': 100,
    'robust_ctc': 100,
    'ilc': 100,
    'behavioral_cloning': 100,
}
LABELS = {'pid': 'PID', 'computed_torque': 'CTC',
          'sliding_mode': 'SMC', 'robust_ctc': 'RCTC',
          'ilc': 'ILC', 'behavioral_cloning': 'BC', 'mpc': 'MPC'}
NOISE_LEVELS = [5e-4, 1e-3, 5e-3]
CSV_PATH = 'docs/exp2_all_controllers.csv'


def run_once(controller_name, mode, noise_std, seed):
    """单次含噪追踪实验，返回 (RMSE, 单步平均耗时秒)；发散 RMSE=inf"""
    npts = TRACKING[controller_name]
    system = RobotSystem(mode='dynamics')
    system.set_sensing(mode=mode, noise_std=noise_std, seed=seed,
                       process_noise=1.0)
    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=npts)
    q_ref = np.array([pt['position'] for pt in traj])

    t0 = time.time()
    if controller_name == 'ilc':
        results = system.train_ilc_trajectory(traj, iterations=10)
        q_act = np.array([r['position'] for r in results])
    else:
        results = system.execute_trajectory(traj,
                                            controller_name=controller_name)
        q_act = np.array([r['position'] for r in results])
    wall = (time.time() - t0) / len(traj)

    err = q_ref - q_act
    rmse = np.sqrt(np.mean(err**2))
    if rmse >= 1.0:
        rmse = np.inf
    return rmse, wall


def run_mpc(mode, noise_std, seed):
    """MPC 降配评估：horizon=5, maxiter=20, 50 点"""
    from robotic_arm_optimization import MPCController
    system = RobotSystem(mode='dynamics')
    system.controllers['mpc'] = MPCController(system.arm, horizon=5,
                                              dt=0.04, maxiter=20)
    system.set_sensing(mode=mode, noise_std=noise_std, seed=seed,
                       process_noise=1.0)
    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=50)
    t0 = time.time()
    results = system.execute_trajectory(traj, controller_name='mpc')
    wall = (time.time() - t0) / len(traj)
    err = np.array([r['error'] for r in results])
    rmse = np.sqrt(np.mean(err**2))
    if rmse >= 1.0:
        rmse = np.inf
    return rmse, wall


def _worker(task):
    kind = task['kwargs'].pop('kind')
    if kind == 'mpc':
        r, w = run_mpc(**task['kwargs'])
    else:
        r, w = run_once(**task['kwargs'])
    return task['key'], (r, w)


def _parallel(tasks, max_workers=8):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, t): t['key'] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            key, r = fut.result()
            results[key] = r
            if i % 10 == 0 or i == len(tasks):
                eta = (time.time() - t0) / i * (len(tasks) - i)
                print(f"  进度 {i}/{len(tasks)} (ETA {eta/60:.1f} min)",
                      flush=True)
    return results


def run(n_seeds=5, max_workers=8):
    done = _load_done_keys(CSV_PATH, ['controller', 'mode', 'noise', 'seed'])
    t0 = time.time()
    tasks = []

    for ctrl in TRACKING:
        label = LABELS[ctrl]
        if (label, 'ideal', '0.0', '0') not in done:
            tasks.append(dict(key=(label, 'ideal', 0.0, 0),
                              kwargs=dict(kind='tracking',
                                          controller_name=ctrl, mode='ideal',
                                          noise_std=0.0, seed=0)))
        for mode in ('raw', 'kalman'):
            for noise in NOISE_LEVELS:
                for seed in range(n_seeds):
                    if (label, mode, repr(noise), str(seed)) in done:
                        continue
                    tasks.append(dict(
                        key=(label, mode, noise, seed),
                        kwargs=dict(kind='tracking', controller_name=ctrl,
                                    mode=mode, noise_std=noise, seed=seed)))

    # MPC 定量评估说明：
    # 本实现的 MPC（SLSQP + 数值雅可比 + 数值科氏矩阵）单步求解耗时
    # 约 12~60s（热启动后仍 ~10s），一条轨迹的仿真需数十分钟，
    # 不具备实时性——这本身就是跨范式比较中"实现代价"维度的数据。
    # 因此 MPC 不进含噪定量矩阵，论文中以单步耗时测量值呈现。
    # for mode, noise in [('ideal', 0.0), ('raw', 5e-3), ('kalman', 5e-3)]:
    #     tasks.append(...)

    if tasks:
        print(f"共 {len(tasks)} 次运行（跳过已完成），分批并行执行中...",
              flush=True)
        batch_size = 30
        for bstart in range(0, len(tasks), batch_size):
            batch = tasks[bstart:bstart + batch_size]
            results = _parallel(batch, max_workers=max_workers)
            rows = []
            for t in batch:
                label, mode, noise, seed = t['key']
                r, w = results[t['key']]
                rows.append(dict(controller=label, mode=mode, noise=noise,
                                 seed=seed, rmse=r, wall_per_step=w))
            save_csv(rows, CSV_PATH, append=True)
            print(f"  批次完成 {bstart + len(batch)}/{len(tasks)}，已落盘",
                  flush=True)
    else:
        print("全部运行已完成，无需重跑。")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


def plot():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    df = pd.read_csv(CSV_PATH)
    df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'raw': '#d95f02', 'kalman': '#1b9e77'}
    ctrl_order = ['PID', 'CTC', 'RCTC', 'SMC', 'ILC', 'BC', 'MPC']

    # 左图：噪声 5e-3 下各控制器 RMSE（raw vs kalman，均值±std）
    ax = axes[0]
    noise_focus = 5e-3
    width = 0.35
    xlabels, xi = [], 0
    for ctrl in ctrl_order:
        sub = df[(df['controller'] == ctrl)]
        ideal_row = sub[sub['mode'] == 'ideal']
        if len(ideal_row):
            ax.axhline(ideal_row['rmse'].iloc[0], color='#bbbbbb',
                       ls=':', lw=0.8)
        for k, mode in enumerate(['raw', 'kalman']):
            d = sub[(sub['mode'] == mode) & (sub['noise'] == noise_focus)]
            d = d[np.isfinite(d['rmse'])]
            if len(d) == 0:
                continue
            x = xi + (k - 0.5) * width
            ax.bar(x, d['rmse'].mean(), width,
                   yerr=d['rmse'].std() if len(d) > 1 else 0,
                   capsize=3, color=colors[mode], alpha=0.85)
        xlabels.append(ctrl)
        xi += 1
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yscale('log')
    ax.set_ylabel('tracking RMSE (rad)')
    ax.set_title(f'All controllers under noise std={noise_focus} rad\n'
                 f'(dotted lines = noise-free baselines)')
    ax.grid(alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors['raw'], label='raw'),
                       Patch(color=colors['kalman'],
                             label='input-aware Kalman')], fontsize=9)

    # 右图：RMSE-噪声曲线（每控制器 raw/kalman 两条）
    ax = axes[1]
    markers = {'PID': 'o', 'CTC': 's', 'RCTC': 'v', 'SMC': '*',
               'ILC': '^', 'BC': 'D'}
    for ctrl in ['PID', 'CTC', 'RCTC', 'SMC', 'ILC', 'BC']:
        sub = df[df['controller'] == ctrl]
        for mode, ls in [('raw', '--'), ('kalman', '-')]:
            g = sub[sub['mode'] == mode]
            g = g[np.isfinite(g['rmse'])].groupby('noise')['rmse'].mean()
            if len(g) == 0:
                continue
            ax.plot(g.index, g.values, marker=markers[ctrl], ls=ls,
                    label=f'{ctrl} ({mode})', markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('measurement noise std (rad)')
    ax.set_ylabel('tracking RMSE (rad)')
    ax.set_title('RMSE vs. noise (5 seeds mean)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7.5, ncol=2)

    fig.tight_layout()
    out = 'docs/exp2_all_controllers.png'
    fig.savefig(out, dpi=300)
    print(f"已保存: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'plot'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    if args.action == 'run':
        run(n_seeds=args.seeds, max_workers=args.workers)
    else:
        plot()
