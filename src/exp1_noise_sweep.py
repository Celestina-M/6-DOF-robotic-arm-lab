# 文件: src/exp1_noise_sweep.py
"""
实验 1：状态估计方法对比 —— RMSE-噪声水平扫描 + 卡尔曼稳定性热力图

用法:
    python src/exp1_noise_sweep.py sweep          # 噪声扫描（多种子蒙特卡洛）
    python src/exp1_noise_sweep.py heatmap        # 卡尔曼参数稳定性热力图
    python src/exp1_noise_sweep.py sweep --quick  # 快速模式（少种子，调试用）

结果保存: docs/exp1_*.csv
"""
import argparse
import os
import sys
import time

# 多进程下限制每个 worker 的线程数，避免 32 核上线程超额订阅
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from robotic_system import RobotSystem

START = [0, 0, 0, 0, 0, 0]
END = [0.5, 0.5, -0.5, 0, 0.5, 0]

# 控制器 -> (名称, 轨迹点数)。PID 在 dt≈0.01 才稳定，CTC 用 0.02
CONTROLLERS = {
    'pid': ('PID', 200),
    'computed_torque': ('CTC', 100),
}

NOISE_LEVELS = [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
FEEDBACK_MODES = ['raw', 'kalman']


def run_once(controller_name, mode, noise_std, seed, process_noise=1.0,
             measurement_noise=None, input_aware=True):
    """单次含噪追踪实验，返回 RMSE (rad)；发散返回 np.inf"""
    label, npts = CONTROLLERS[controller_name]
    system = RobotSystem(mode='dynamics')
    system.set_sensing(mode=mode, noise_std=noise_std, seed=seed,
                       process_noise=process_noise,
                       measurement_noise=measurement_noise,
                       input_aware=input_aware)
    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=npts)
    res = system.execute_trajectory(traj, controller_name=controller_name)
    err = np.array([r['error'] for r in res])
    rmse = np.sqrt(np.mean(err**2))
    return rmse if rmse < 1.0 else np.inf   # RMSE≥1 判为发散


def _worker(task):
    """多进程工作函数：task 为参数字典，返回 (task_key, rmse)"""
    r = run_once(**task['kwargs'])
    return task['key'], r


def _parallel(tasks, max_workers=None):
    """并行执行任务列表，返回 {key: rmse}，带进度输出"""
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


def _load_done_keys(path, key_fields):
    """断点续跑：读取已有 CSV，返回已完成 key 集合"""
    import csv
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                done.add(tuple(row[k] for k in key_fields))
    return done


def sweep(n_seeds=10, max_workers=16):
    """噪声扫描：控制器 × 反馈模式 × 噪声水平 × 多种子（可断点续跑）"""
    out = 'docs/exp1_noise_sweep.csv'
    done = _load_done_keys(out, ['controller', 'mode', 'noise', 'seed'])
    t0 = time.time()
    tasks, rows = [], []

    for ctrl, (label, _) in CONTROLLERS.items():
        candidates = [dict(key=(label, 'ideal', 0.0, 0),
                           kwargs=dict(controller_name=ctrl, mode='ideal',
                                       noise_std=0.0, seed=0))]
        for mode in FEEDBACK_MODES:
            for noise in NOISE_LEVELS[1:]:
                for seed in range(n_seeds):
                    candidates.append(dict(
                        key=(label, mode, noise, seed),
                        kwargs=dict(controller_name=ctrl, mode=mode,
                                    noise_std=noise, seed=seed)))
        for c in candidates:
            label_, mode_, noise_, seed_ = c['key']
            if (label_, mode_, repr(noise_), str(seed_)) in done:
                continue
            tasks.append(c)

    if tasks:
        print(f"共 {len(tasks)} 次运行（跳过已完成），分批并行执行中...",
              flush=True)
        batch_size = 40
        for bstart in range(0, len(tasks), batch_size):
            batch = tasks[bstart:bstart + batch_size]
            results = _parallel(batch, max_workers=max_workers)
            rows = []
            for t in batch:
                label, mode, noise, seed = t['key']
                rows.append(dict(controller=label, mode=mode, noise=noise,
                                 seed=seed, rmse=results[t['key']]))
            save_csv(rows, out, append=True)
            print(f"  批次完成 {bstart + len(batch)}/{len(tasks)}，已落盘",
                  flush=True)
    else:
        print("全部运行已完成，无需重跑。")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


def heatmap(max_workers=16, input_aware=True):
    """卡尔曼 (Q, R) 参数稳定性热力图：固定噪声 0.005（可断点续跑）"""
    tag = 'ia' if input_aware else 'noinput'
    out = f'docs/exp1_kalman_heatmap_{tag}.csv'
    done = _load_done_keys(out, ['controller', 'process_noise',
                                 'measurement_noise'])
    pn_grid = [1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0]
    mn_grid = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    t0 = time.time()
    tasks = []
    for ctrl, (label, _) in CONTROLLERS.items():
        for pn in pn_grid:
            for mn in mn_grid:
                if (label, repr(pn), repr(mn)) in done:
                    continue
                tasks.append(dict(
                    key=(label, pn, mn),
                    kwargs=dict(controller_name=ctrl, mode='kalman',
                                noise_std=0.005, seed=42,
                                process_noise=pn, measurement_noise=mn,
                                input_aware=input_aware)))

    if tasks:
        print(f"共 {len(tasks)} 次运行（跳过已完成），分批并行执行中...",
              flush=True)
        batch_size = 40
        for bstart in range(0, len(tasks), batch_size):
            batch = tasks[bstart:bstart + batch_size]
            results = _parallel(batch, max_workers=max_workers)
            rows = []
            for t in batch:
                label, pn, mn = t['key']
                r = results[t['key']]
                rows.append(dict(controller=label, process_noise=pn,
                                 measurement_noise=mn,
                                 rmse='inf' if np.isinf(r) else f'{r:.6f}',
                                 stable=0 if np.isinf(r) else 1))
            save_csv(rows, out, append=True)
            print(f"  批次完成 {bstart + len(batch)}/{len(tasks)}，已落盘",
                  flush=True)
    else:
        print("全部运行已完成，无需重跑。")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


def save_csv(rows, path, append=False):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not (append and os.path.exists(path))
    with open(path, 'a' if append else 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"已保存: {path} ({'追加' if append else '写入'} {len(rows)} 行)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['sweep', 'heatmap'])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--no-input', action='store_true',
                        help='热力图用无输入卡尔曼（对照组）')
    args = parser.parse_args()

    n = 3 if args.quick else args.seeds
    if args.experiment == 'sweep':
        sweep(n_seeds=n, max_workers=args.workers)
    else:
        heatmap(max_workers=args.workers, input_aware=not args.no_input)
