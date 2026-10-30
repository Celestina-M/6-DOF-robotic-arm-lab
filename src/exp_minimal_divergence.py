# -*- coding: utf-8 -*-
"""图1：无输入卡尔曼闭环发散——最小双积分器复现 + 增广系统特征值。

系统: q_ddot = u, PD 控制 u = kp*(qd-q) - kd*q_dot_est
观测: y = q + noise (仅位置)
对照: 无输入卡尔曼 (预测步不含 u) vs 输入感知卡尔曼 (预测步含 u)
输出: docs/exp_minimal_divergence.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    import os
    os.environ.setdefault(_v, '2')

DT = 0.02
STEPS = 200
KP, KD = 400.0, 80.0       # 单独稳定的 PD 增益（无滤波时极点在单位圆内）
SIGMA = 1e-3               # 测量噪声 std
Q_KF = np.diag([1e-10, 1e-8])
R_KF = SIGMA ** 2
TARGET = 1.0


def make_kalman(dt, input_aware):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    B = np.array([0.5 * dt * dt, dt])  # 加速度输入矩阵
    return F, H, B


def run(input_aware, seed=0):
    rng = np.random.default_rng(seed)
    F, H, B = make_kalman(DT, input_aware)
    x = np.zeros(2)                    # 真值 [q, qd]
    xhat = np.zeros(2)                 # 估计
    P = np.diag([SIGMA ** 2, 0.01])
    u_prev = 0.0
    hist_q, hist_qhat, hist_err = [], [], []
    for _ in range(STEPS):
        y = x[0] + rng.normal(0, SIGMA)
        # 滤波: 预测
        xp = F @ xhat + (B * u_prev if input_aware else 0.0)
        Pp = F @ P @ F.T + Q_KF
        # 更新
        S = H @ Pp @ H.T + R_KF
        K = Pp @ H.T / S
        xhat = xp + K.flatten() * (y - H @ xp)
        P = (np.eye(2) - K @ H) @ Pp
        # PD 控制(用估计)
        u = KP * (TARGET - xhat[0]) - KD * xhat[1]
        # 对象推进
        x[1] += u * DT
        x[0] += x[1] * DT
        u_prev = u
        hist_q.append(x[0])
        hist_qhat.append(xhat[0])
        hist_err.append(abs(x[0] - xhat[0]))
    return np.array(hist_q), np.array(hist_qhat), np.array(hist_err)


def augmented_eigenvalues():
    """增广闭环 [x; e] (对象2维+观测误差2维) 的离散特征值 (无输入情形)。

    真实: x+ = F x + B u,  u = -Kc xhat,  xhat = x - e
    误差: e+ = (I - K H)(F e - B u)   ... 展开为线性系统
    """
    F, H, B = make_kalman(DT, False)
    Kc = np.array([KP, KD])            # 控制增益行向量(对 xhat)
    # 稳态卡尔曼增益(数值迭代收敛)
    P = np.eye(2)
    for _ in range(2000):
        Pp = F @ P @ F.T + Q_KF
        S = H @ Pp @ H.T + R_KF
        K = Pp @ H.T / S
        P = (np.eye(2) - K @ H) @ Pp
    # x+ = F x + B(-Kc xhat) = F x - B Kc (x - e) = (F - B Kc) x + B Kc e
    # xhat+ = xp + K(y - H xp), xp = F xhat (无输入!)
    # e = x - xhat => e+ = F x + B u - F xhat - K(F x + B u - F xhat)(仅H行)
    # 数值构造: 状态 z=[x; xhat], 用符号化一步推进
    n = 2
    A = np.zeros((4, 4))
    # x+ = F x - B Kc xhat
    A[:n, :n] = F
    A[:n, n:] = -np.outer(B, Kc)
    # xhat+ = xp + K (y - H xp),  xp = F xhat,  y = H x + noise
    #     = F xhat + K H x - K H F xhat = K H x + (I - K H) F xhat
    A[n:, :n] = K @ H
    A[n:, n:] = (np.eye(2) - K @ H) @ F
    return np.linalg.eigvals(A)


def main():
    q_no, qh_no, e_no = run(input_aware=False)
    q_ia, qh_ia, e_ia = run(input_aware=True)
    ev = augmented_eigenvalues()
    print('增广系统特征值模:', np.round(np.abs(ev), 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.arange(STEPS) * DT
    ax1.semilogy(t, np.abs(q_no - TARGET) + 1e-12, color='#d95f02',
                 lw=1.4, label='no-input Kalman (diverges)')
    ax1.semilogy(t, np.abs(q_ia - TARGET) + 1e-12, color='#1b9e77',
                 lw=1.4, label='input-aware Kalman (stable)')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('|tracking error| (log scale)')
    ax1.set_title('Closed-loop with position-only Kalman feedback\n'
                  '(double integrator, PD control, both stable w/o filter)')
    ax1.legend()
    ax1.grid(alpha=0.3, which='both')

    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.9,
             label='unit circle')
    ax2.scatter(ev.real, ev.imag, s=90, color='#d95f02', zorder=5,
                label='closed-loop eigenvalues')
    for i, e in enumerate(ev):
        ax2.annotate(f'|λ|={abs(e):.3f}', (e.real, e.imag),
                     textcoords='offset points', xytext=(8, 8),
                     fontsize=9)
    ax2.set_xlabel('Re')
    ax2.set_ylabel('Im')
    ax2.set_title('Augmented closed-loop eigenvalues\n'
                  '(plant 2-dim + observer 2-dim, no-input Kalman)')
    ax2.set_aspect('equal')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('docs/exp_minimal_divergence.png', dpi=200)
    print('已保存: docs/exp_minimal_divergence.png')


if __name__ == '__main__':
    main()
