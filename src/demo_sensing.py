# 文件: src/demo_sensing.py
"""
统一传感通道演示：ideal / raw / kalman 三种反馈模式下的追踪性能对比

输出：
1. 终端 RMSE 对照表（PID 与 CTC × 三种反馈模式）
2. 图：CTC 在三种反馈模式下的关节 1 追踪曲线（含噪声放大区）

运行: python src/demo_sensing.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from robotic_system import RobotSystem

NOISE_STD = 0.005      # 位置测量噪声 (rad)
SEED = 42
START = [0, 0, 0, 0, 0, 0]
END = [0.5, 0.5, -0.5, 0, 0.5, 0]

CONFIGS = [
    ('ideal',  {}),
    ('raw',    {}),
    ('kalman', dict(process_noise=1.0)),                    # 输入感知（默认开启）
    ('kalman', dict(process_noise=1.0, input_aware=False)), # 无输入对照
]


def run(controller_name, sensing_mode, npts, **kwargs):
    system = RobotSystem(mode='dynamics')
    system.set_sensing(mode=sensing_mode, noise_std=NOISE_STD, seed=SEED, **kwargs)
    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=npts)
    return traj, system.execute_trajectory(traj, controller_name=controller_name)


def rmse(results):
    err = np.array([r['error'] for r in results])
    return np.sqrt(np.mean(err**2))


def main():
    print(f"含噪追踪对比 | 位置测量噪声 std = {NOISE_STD} rad, seed = {SEED}")
    print(f"{'控制器':<18}{'反馈模式':<22}{'RMSE (rad)':>14}  状态")
    print('-' * 64)

    table = {}
    for ctrl, npts, label in [('pid', 200, 'PID'),
                              ('computed_torque', 100, 'CTC')]:
        for mode, extra in CONFIGS:
            ia = extra.get('input_aware', True)
            tag = mode if mode != 'kalman' else ('kalman (input-aware)' if ia
                                                 else 'kalman (no-input)')
            _, res = run(ctrl, mode, npts, **extra)
            r = rmse(res)
            table[(label, tag)] = (res, r)
            status = '稳定' if r < 1 else '发散'
            print(f"{label:<20}{tag:<24}{r:>12.6f}  {status}")

    # 图：CTC 三种稳定模式的关节 1 追踪误差曲线
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag in ['ideal', 'raw', 'kalman (input-aware)']:
        res, r = table[('CTC', tag)]
        t = [p['time'] for p in res]
        e = [p['error'][0] for p in res]
        ax.plot(t, e, label=f"{tag} (RMSE={r:.2e})")
    ax.set_xlabel('t (s)')
    ax.set_ylabel('joint 1 tracking error (rad)')
    ax.set_title(f'CTC tracking under sensing modes (noise std={NOISE_STD} rad)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join('docs', 'sensing_modes_comparison.png')
    os.makedirs('docs', exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"\n图已保存: {out}")


if __name__ == '__main__':
    main()
