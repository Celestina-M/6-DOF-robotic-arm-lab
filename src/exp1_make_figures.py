# 文件: src/exp1_make_figures.py
"""
实验 1 出图：从 docs/exp1_*.csv 生成论文核心图
1. RMSE-噪声水平曲线（含均值±标准差带，20 种子）
2. 卡尔曼 (Q, R) 参数闭环稳定性热力图

运行: python src/exp1_make_figures.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUT = os.path.join('docs')


def fig_noise_sweep():
    df = pd.read_csv(os.path.join(OUT, 'exp1_noise_sweep.csv'))
    df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {'raw': '#d95f02', 'kalman': '#1b9e77'}
    labels = {'raw': 'raw measurement', 'kalman': 'input-aware Kalman'}

    for ax, ctrl in zip(axes, ['PID', 'CTC']):
        sub = df[df['controller'] == ctrl]
        # ideal 基线
        ideal = sub[sub['mode'] == 'ideal']['rmse'].iloc[0]
        ax.axhline(ideal, color='#666666', ls='--', lw=1.2,
                   label=f'ideal ({ideal:.2e})')
        for mode in ['raw', 'kalman']:
            d = sub[sub['mode'] == mode]
            g = d.groupby('noise')['rmse']
            x = g.mean().index.to_numpy()
            mean = g.mean().to_numpy()
            std = g.std().to_numpy()
            # 发散的种子记为 inf，统计稳定性
            div = d[np.isinf(d['rmse'])].groupby('noise').size()
            ax.plot(x, mean, 'o-', color=colors[mode], label=labels[mode])
            ax.fill_between(x, mean - std, mean + std,
                            color=colors[mode], alpha=0.2)
            for n, c in div.items():
                ax.plot(n, mean[list(x).index(n)] if n in x else np.nan,
                        marker='x', color='red', ms=10, mew=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('measurement noise std (rad)')
        ax.set_title(ctrl)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=9)
    axes[0].set_ylabel('tracking RMSE (rad)')
    fig.suptitle('Tracking RMSE vs. measurement noise (20 seeds, mean ± std)')
    fig.tight_layout()
    path = os.path.join(OUT, 'exp1_rmse_vs_noise.png')
    fig.savefig(path, dpi=300)
    print(f"已保存: {path}")


def fig_heatmap():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    files = {'input-aware': 'exp1_kalman_heatmap_ia.csv',
             'no-input': 'exp1_kalman_heatmap_noinput.csv'}

    for col, (tag, fname) in enumerate(files.items()):
        df = pd.read_csv(os.path.join(OUT, fname))
        pn_vals = sorted(df['process_noise'].unique())
        mn_vals = sorted(df['measurement_noise'].unique())
        n_stable_total = 0

        for row, ctrl in enumerate(['PID', 'CTC']):
            ax = axes[row, col]
            sub = df[df['controller'] == ctrl]
            Z = np.full((len(mn_vals), len(pn_vals)), np.nan)
            S = np.zeros((len(mn_vals), len(pn_vals)))
            for _, r in sub.iterrows():
                i = mn_vals.index(r['measurement_noise'])
                j = pn_vals.index(r['process_noise'])
                S[i, j] = r['stable']
                if r['stable'] == 1:
                    Z[i, j] = float(r['rmse'])
            n_stable_total += int(S.sum())
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad(color='#d0d0d0')
            masked = np.ma.masked_invalid(np.log10(Z))
            im = ax.imshow(masked, cmap=cmap, origin='lower', aspect='auto',
                           extent=[-0.5, len(pn_vals)-0.5, -0.5,
                                   len(mn_vals)-0.5])
            for i in range(len(mn_vals)):
                for j in range(len(pn_vals)):
                    if S[i, j] == 0:
                        ax.text(j, i, '✕', ha='center', va='center',
                                color='#777777', fontsize=11)
                    else:
                        ax.text(j, i, f'{Z[i, j]:.0e}', ha='center',
                                va='center',
                                color='white' if masked[i, j] < np.nanmedian(masked)
                                else 'black', fontsize=6)
            ax.set_xticks(range(len(pn_vals)))
            ax.set_xticklabels([f'{v:g}' for v in pn_vals], fontsize=8)
            ax.set_yticks(range(len(mn_vals)))
            ax.set_yticklabels([f'{v:g}' for v in mn_vals], fontsize=8)
            ax.set_title(f'{ctrl} — {tag} '
                         f'({int(S.sum())}/{S.size} stable)', fontsize=10)
            if tag == 'input-aware' and 1.0 in pn_vals and 5e-3 in mn_vals:
                ax.plot(pn_vals.index(1.0), mn_vals.index(5e-3), 'r*', ms=14)
            fig.colorbar(im, ax=ax, label='log10(RMSE)')

    for ax in axes[1, :]:
        ax.set_xlabel('process noise q')
    for ax in axes[:, 0]:
        ax.set_ylabel('measurement noise r')
    fig.suptitle('Kalman closed-loop stability & RMSE (noise std = 0.005 rad; '
                 '✕ = diverged, ★ = default)')
    fig.tight_layout()
    path = os.path.join(OUT, 'exp1_kalman_stability_heatmap.png')
    fig.savefig(path, dpi=300)
    print(f"已保存: {path}")


if __name__ == '__main__':
    fig_noise_sweep()
    fig_heatmap()
