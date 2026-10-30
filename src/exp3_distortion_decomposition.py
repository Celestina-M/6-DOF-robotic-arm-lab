# 文件: src/exp3_distortion_decomposition.py
"""
实验 3：失真类型分解 —— 定位主导误差源

失真类型（单独 + 组合）：
    noise    : 位置测量高斯噪声 std=0.005 rad
    quant    : 编码器量化 2π/2^17 rad（17 位）
    bias     : 常值零偏 0.002 rad
    param    : 质量/质心逐连杆 ±20% 摄动（仅被控对象，控制器用名义模型）
    friction : 被控对象粘性摩擦 ×1.5（控制器用名义值）
    combined : 以上全部叠加

用法:
    python src/exp3_distortion_decomposition.py run [--workers 8]
    python src/exp3_distortion_decomposition.py plot
"""
import argparse
import copy
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
from perturbation import apply_parameter_perturbation, apply_friction_mismatch
from exp1_noise_sweep import _load_done_keys, save_csv

START = [0, 0, 0, 0, 0, 0]
END = [0.5, 0.5, -0.5, 0, 0.5, 0]
CONTROLLERS = {
    'pid': ('PID', 200),
    'computed_torque': ('CTC', 100),
}

NOISE_STD = 0.005
QUANT_RES = 2 * np.pi / 2**17
BIAS = 0.002
PARAM_REL = 0.2
FRICTION_SCALE = 1.5

DISTORTIONS = ['noise', 'quant', 'bias', 'param', 'friction', 'combined']
CSV_PATH = 'docs/exp3_distortion_decomposition.csv'


def run_once(controller_name, mode, distortion, seed):
    """单次失真实验，返回 RMSE (rad)；发散返回 np.inf"""
    label, npts = CONTROLLERS[controller_name]
    system = RobotSystem(mode='dynamics')
    # 控制器在 _initialize_controllers 中绑定名义模型；
    # 被控对象单独复制并摄动（estimator_model 保持名义）
    plant = copy.deepcopy(system.arm)

    has_noise = distortion in ('noise', 'combined')
    has_quant = distortion in ('quant', 'combined')
    has_bias = distortion in ('bias', 'combined')
    if distortion in ('param', 'combined'):
        apply_parameter_perturbation(plant, rel=PARAM_REL, seed=seed)
    if distortion in ('friction', 'combined'):
        apply_friction_mismatch(plant, scale=FRICTION_SCALE)
    system.arm = plant

    system.set_sensing(mode=mode,
                       noise_std=NOISE_STD if has_noise else 0.0,
                       resolution=QUANT_RES if has_quant else None,
                       bias=BIAS if has_bias else None,
                       seed=seed)
    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=npts)
    res = system.execute_trajectory(traj, controller_name=controller_name)
    err = np.array([r['error'] for r in res])
    rmse = np.sqrt(np.mean(err**2))
    return rmse if rmse < 1.0 else np.inf


def _worker(task):
    return task['key'], run_once(**task['kwargs'])


def _parallel(tasks, max_workers=8):
    """本模块的并行执行器（worker 必须引用本模块的 run_once）"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, t): t['key'] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            key, r = fut.result()
            results[key] = r
            if i % 20 == 0 or i == len(tasks):
                eta = (time.time() - t0) / i * (len(tasks) - i)
                print(f"  进度 {i}/{len(tasks)} (ETA {eta/60:.1f} min)",
                      flush=True)
    return results


def run(n_seeds=10, max_workers=8):
    done = _load_done_keys(CSV_PATH, ['controller', 'mode', 'distortion',
                                      'seed'])
    t0 = time.time()
    tasks = []
    for ctrl, (label, _) in CONTROLLERS.items():
        # 零失真基线（ideal，一次）
        key = (label, 'ideal', 'none', 0)
        if (label, 'ideal', 'none', '0') not in done:
            tasks.append(dict(key=key, kwargs=dict(
                controller_name=ctrl, mode='ideal', distortion='none',
                seed=0)))
        for mode in ('raw', 'kalman'):
            for dist in DISTORTIONS:
                for seed in range(n_seeds):
                    if (label, mode, dist, str(seed)) in done:
                        continue
                    tasks.append(dict(key=(label, mode, dist, seed),
                                      kwargs=dict(controller_name=ctrl,
                                                  mode=mode, distortion=dist,
                                                  seed=seed)))

    if tasks:
        print(f"共 {len(tasks)} 次运行（跳过已完成），分批并行执行中...",
              flush=True)
        batch_size = 40
        for bstart in range(0, len(tasks), batch_size):
            batch = tasks[bstart:bstart + batch_size]
            results = _parallel(batch, max_workers=max_workers)
            rows = []
            for t in batch:
                label, mode, dist, seed = t['key']
                rows.append(dict(controller=label, mode=mode,
                                 distortion=dist, seed=seed,
                                 rmse=results[t['key']]))
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

    df = pd.read_csv(CSV_PATH)
    df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')
    df.loc[~np.isfinite(df['rmse']), 'rmse'] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {'raw': '#d95f02', 'kalman': '#1b9e77'}
    width = 0.38

    for ax, ctrl in zip(axes, ['PID', 'CTC']):
        sub = df[df['controller'] == ctrl]
        ideal = sub[sub['mode'] == 'ideal']['rmse'].iloc[0]
        ax.axhline(ideal, color='#666666', ls='--', lw=1.2,
                   label=f'ideal ({ideal:.2e})')
        for k, mode in enumerate(['raw', 'kalman']):
            means, stds = [], []
            for dist in DISTORTIONS:
                d = sub[(sub['mode'] == mode)
                        & (sub['distortion'] == dist)]['rmse']
                means.append(d.mean())
                stds.append(d.std())
            x = np.arange(len(DISTORTIONS)) + (k - 0.5) * width
            ax.bar(x, means, width, yerr=stds, capsize=3,
                   color=colors[mode], alpha=0.85,
                   label=f'{mode} (10 seeds)')
        ax.set_yscale('log')
        ax.set_xticks(np.arange(len(DISTORTIONS)))
        ax.set_xticklabels(DISTORTIONS, rotation=20, fontsize=9)
        ax.set_title(ctrl)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=9)
    axes[0].set_ylabel('tracking RMSE (rad)')
    fig.suptitle('Distortion type decomposition: which distortion dominates? '
                 '(mean ± std, 10 seeds)')
    fig.tight_layout()
    out = 'docs/exp3_distortion_decomposition.png'
    fig.savefig(out, dpi=300)
    print(f"已保存: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'plot'])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    if args.action == 'run':
        run(n_seeds=args.seeds, max_workers=args.workers)
    else:
        plot()
