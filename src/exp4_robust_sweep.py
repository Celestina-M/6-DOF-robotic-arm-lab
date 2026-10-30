# 文件: src/exp4_robust_sweep.py
"""
实验 4：鲁棒控制——模型失配扫描

回答的问题：随着动力学失配加剧（质量/质心摄动 ↑ + 摩擦失配 ↑），
名义 CTC、鲁棒 CTC（RCTC）、积分滑模（SMC）的追踪精度分别如何退化？

回路构造（与 perturbation.py 一致）：控制器/估计器绑定名义模型，
被控对象替换为摄动后的 plant，二者之差即模型失配。
反馈统一为 kalman（5e-3）——SMC 滑模面含速度项，raw 差分噪声会被
切换项放大，kalman 下对比最干净（见 tests/test_robust_control.py）。

用法:
    python src/exp4_robust_sweep.py run [--seeds 2] [--workers 6]
    python src/exp4_robust_sweep.py plot
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

# (质量/质心摄动幅度 rel, 摩擦倍数)
LEVELS = [(0.0, 1.0), (0.1, 1.25), (0.2, 1.5),
          (0.3, 2.0), (0.4, 3.0), (0.5, 4.0)]
CONTROLLERS = ('computed_torque', 'robust_ctc', 'sliding_mode')
LABELS = {'computed_torque': 'CTC', 'robust_ctc': 'RCTC',
          'sliding_mode': 'SMC'}
CSV_PATH = 'docs/exp4_robust_sweep.csv'
NOISE_STD = 5e-3


def run_once(controller_name, rel, fric, seed):
    """单次失配追踪实验，返回 RMSE (rad)"""
    system = RobotSystem(mode='dynamics')
    plant = copy.deepcopy(system.arm)
    apply_parameter_perturbation(plant, rel=rel, seed=42)
    apply_friction_mismatch(plant, scale=fric)
    system.arm = plant
    system.set_sensing(mode='kalman', noise_std=NOISE_STD, seed=seed,
                       process_noise=1.0)

    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=100)
    q_ref = np.array([pt['position'] for pt in traj])
    results = system.execute_trajectory(traj, controller_name=controller_name)
    q_act = np.array([r['position'] for r in results])
    rmse = np.sqrt(np.mean((q_ref - q_act) ** 2))
    if rmse >= 1.0:
        rmse = np.inf
    return float(rmse)


def _worker(task):
    key, kwargs = task
    return key, run_once(**kwargs)


def run(n_seeds=2, max_workers=6):
    done = _load_done_keys(CSV_PATH, ['controller', 'rel', 'fric', 'seed'])
    tasks = []
    for ctrl in CONTROLLERS:
        for rel, fric in LEVELS:
            for seed in range(n_seeds):
                key = (ctrl, str(rel), str(fric), str(seed))
                if key in done:
                    continue
                tasks.append((key, dict(controller_name=ctrl, rel=rel,
                                        fric=fric, seed=seed)))

    if not tasks:
        print("全部运行已完成，无需重跑。")
        return

    print(f"共 {len(tasks)} 次运行（跳过已完成），并行执行中...", flush=True)
    t0 = time.time()
    results = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            key, rmse = fut.result()
            results[key] = rmse
            if i % 6 == 0 or i == len(tasks):
                eta = (time.time() - t0) / i * (len(tasks) - i)
                print(f"  进度 {i}/{len(tasks)} (ETA {eta/60:.1f} min)",
                      flush=True)

    rows = [dict(controller=LABELS[k[0]], rel=k[1], fric=k[2],
                 seed=k[3], rmse=results[k]) for k in results]
    save_csv(rows, CSV_PATH, append=True)
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

    colors = {'CTC': '#7570b3', 'RCTC': '#1b9e77', 'SMC': '#d95f02'}
    markers = {'CTC': 's', 'RCTC': 'v', 'SMC': '*'}
    level_labels = [f'{int(r*100)}%/+{int((f-1)*100)}%' for r, f in LEVELS]
    x = np.arange(len(LEVELS))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.26
    for k, ctrl in enumerate(['CTC', 'RCTC', 'SMC']):
        means, stds = [], []
        for rel, fric in LEVELS:
            d = df[(df['controller'] == ctrl)
                   & (df['rel'] == rel) & (df['fric'] == fric)]
            d = d[np.isfinite(d['rmse'])]
            means.append(d['rmse'].mean() if len(d) else np.nan)
            stds.append(d['rmse'].std() if len(d) > 1 else 0.0)
        ax.bar(x + (k - 1) * width, means, width,
               yerr=stds, capsize=3, color=colors[ctrl], alpha=0.85,
               label=ctrl)
        ax.plot(x + (k - 1) * width, means, marker=markers[ctrl],
                color=colors[ctrl], lw=1, ls='--', ms=7)

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_xlabel('model mismatch level (mass/COM perturbation / friction scale)')
    ax.set_ylabel('tracking RMSE (rad)')
    ax.set_yscale('log')
    ax.set_title(f'Robust control under model mismatch\n'
                 f'(kalman feedback, noise std={NOISE_STD} rad)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    fig.tight_layout()
    out = 'docs/exp4_robust_sweep.png'
    fig.savefig(out, dpi=300)
    print(f"已保存: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'plot'])
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    if args.action == 'run':
        run(n_seeds=args.seeds, max_workers=args.workers)
    else:
        plot()
