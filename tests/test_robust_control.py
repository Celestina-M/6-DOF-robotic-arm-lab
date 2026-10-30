# 文件: tests/test_robust_control.py
"""
鲁棒控制验证：模型失配 + 传感噪声下的追踪对比

构造方式（与 perturbation.py 的说明一致）：
    控制器/估计器绑定名义模型；被控对象替换为摄动后的 plant，
    二者之差即模型失配，正好构成鲁棒控制要对付的扰动回路。

对比：CTC（无鲁棒项） vs RCTC（CTC+切换项） vs SMC（积分滑模）
反馈：kalman 模式（SMC 的滑模面含速度项，raw 差分速度噪声会被
      切换项放大；kalman 估计速度下鲁棒控制器优势最干净）
断言：各失配等级下 SMC 与 RCTC 的 RMSE 均优于 CTC。

运行: python tests/test_robust_control.py
"""

import copy
import os
import sys

import numpy as np

# 与其他测试统一：把 src 目录加入 sys.path 后按扁平方式导入
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))

from robotic_system import RobotSystem
from perturbation import apply_parameter_perturbation, apply_friction_mismatch

START = [0, 0, 0, 0, 0, 0]
END = [0.5, 0.5, -0.5, 0, 0.5, 0]

# (质量/质心摄动幅度, 摩擦倍数)
MISMATCH_LEVELS = [(0.2, 1.5), (0.4, 3.0)]


def run_case(controller_name, rel, fric_scale, seed=0):
    """构造失配回路并执行一次追踪，返回 RMSE (rad)"""
    system = RobotSystem(mode='dynamics')
    plant = copy.deepcopy(system.arm)
    apply_parameter_perturbation(plant, rel=rel, seed=42)
    apply_friction_mismatch(plant, scale=fric_scale)
    system.arm = plant
    # 控制器与估计器仍持有 __init__ 时绑定的名义模型（鲁棒回路）

    system.set_sensing(mode='kalman', noise_std=5e-3, seed=seed,
                       process_noise=1.0)

    traj = system.plan_trajectory(START, END, method='quintic',
                                  duration=2.0, num_points=100)
    q_ref = np.array([pt['position'] for pt in traj])
    results = system.execute_trajectory(traj, controller_name=controller_name)
    q_act = np.array([r['position'] for r in results])
    return float(np.sqrt(np.mean((q_ref - q_act) ** 2)))


def main():
    ok = True
    for rel, fric in MISMATCH_LEVELS:
        print("=" * 62)
        print(f"  失配等级：质量/质心 ±{int(rel*100)}% + 摩擦 ×{fric}"
              f" + kalman(5e-3)")
        print("=" * 62)

        rmse = {}
        for ctrl in ('computed_torque', 'robust_ctc', 'sliding_mode'):
            r = run_case(ctrl, rel, fric)
            rmse[ctrl] = r
            print(f"  {ctrl:16s} RMSE = {r:.5f} rad")

        ctc, smc, rctc = (rmse['computed_torque'], rmse['sliding_mode'],
                          rmse['robust_ctc'])
        improve = (ctc - smc) / ctc * 100
        print(f"  SMC 相对 CTC 改善 {improve:.1f}%，"
              f"RCTC 改善 {(ctc - rctc) / ctc * 100:.1f}%")

        if not (smc < ctc and rctc < ctc):
            print(f"  [失败] 该失配等级下鲁棒控制器未优于名义 CTC")
            ok = False
        print()

    print("[通过] 鲁棒控制验证通过" if ok else "[失败] 鲁棒控制验证未通过")
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
