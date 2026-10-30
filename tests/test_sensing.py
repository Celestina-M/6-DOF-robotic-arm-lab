# 文件: tests/test_sensing.py
"""
统一传感与状态估计模块测试

覆盖：
1. ideal 模式原值直通（回归保护：默认行为与旧版完全一致）
2. raw 模式噪声统计特性
3. 编码器量化正确性
4. 卡尔曼估计的速度方差显著低于差分速度
5. 随机种子可复现
6. execute_trajectory 在 ideal 模式下回归旧版 CTC 追踪精度
7. 含噪条件下 kalman 模式追踪精度优于 raw 模式（固定种子，确定性）

运行: python tests/test_sensing.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensing import SensorModel, StateEstimator, SensingPipeline
from robotic_arm_dynamics import JointKalmanFilter

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [通过] {name}")
    else:
        FAIL += 1
        print(f"  [失败] {name}")


def test_ideal_passthrough():
    print("测试 1: ideal 模式原值直通")
    pipe = SensingPipeline(dt=0.02, mode='ideal')
    q = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1])
    dq = np.array([0.01] * 6)
    q_fb, dq_fb = pipe.process(q, dq)
    check("位置原值直通", np.array_equal(q_fb, q))
    check("速度原值直通", np.array_equal(dq_fb, dq))


def test_raw_noise_statistics():
    print("测试 2: raw 模式噪声统计特性")
    sensor = SensorModel(noise_std=0.005, seed=42)
    q_true = np.zeros(6)
    samples = np.array([sensor.measure(q_true) for _ in range(2000)])
    std_empirical = samples.std(axis=0).mean()
    check(f"经验标准差 {std_empirical:.5f} ≈ 0.005",
          abs(std_empirical - 0.005) < 5e-4)
    check("均值近似无偏", abs(samples.mean()) < 5e-4)


def test_quantization():
    print("测试 3: 编码器量化")
    res = 2 * np.pi / 2**17   # 17 位编码器
    sensor = SensorModel(noise_std=0.0, resolution=res, seed=1)
    q_true = np.array([0.123456789] * 6)
    q_meas = sensor.measure(q_true)
    check("量化后落在网格上",
          np.allclose(q_meas / res, np.round(q_meas / res)))
    check("量化误差小于半个步长",
          np.all(np.abs(q_meas - q_true) <= res / 2 + 1e-12))


def test_kalman_smooths_velocity():
    print("测试 4: 卡尔曼速度估计优于差分")
    dt, noise = 0.02, 0.005
    # 真实信号：匀加速运动
    N = 200
    t = np.arange(N) * dt
    q_true = 0.5 * t**2
    dq_true = t.copy()

    rng = np.random.default_rng(42)
    q_meas = q_true + rng.normal(0, noise, N)

    # 差分速度
    dq_diff = np.gradient(q_meas, dt)
    # 卡尔曼速度（CV 需加大过程噪声以跟踪加速信号，CA 默认即可）
    kf_cv = JointKalmanFilter(dt=dt, process_noise=1.0, measurement_noise=noise)
    kf_ca = JointKalmanFilter(dt=dt, process_noise=0.01, measurement_noise=noise,
                              model='ca')
    dq_cv = np.array([kf_cv.update(z)[1] for z in q_meas])
    dq_ca = np.array([kf_ca.update(z)[1] for z in q_meas])

    # 丢弃前 20 步收敛段
    s = slice(20, None)
    rmse_diff = np.sqrt(np.mean((dq_diff[s] - dq_true[s])**2))
    rmse_cv = np.sqrt(np.mean((dq_cv[s] - dq_true[s])**2))
    rmse_ca = np.sqrt(np.mean((dq_ca[s] - dq_true[s])**2))
    print(f"    速度 RMSE: 差分 {rmse_diff:.4f} | CV {rmse_cv:.4f} | CA {rmse_ca:.4f}")
    check("CV 卡尔曼优于差分", rmse_cv < rmse_diff * 0.7)
    check("CA 卡尔曼优于差分", rmse_ca < rmse_diff * 0.7)


def test_seed_reproducibility():
    print("测试 5: 随机种子可复现")
    s1 = SensorModel(noise_std=0.005, seed=42)
    s2 = SensorModel(noise_std=0.005, seed=42)
    a = np.array([s1.measure(np.zeros(6)) for _ in range(50)])
    b = np.array([s2.measure(np.zeros(6)) for _ in range(50)])
    check("同种子输出完全一致", np.array_equal(a, b))


def _run_ctc(system):
    traj = system.plan_trajectory([0]*6, [0.5, 0.5, -0.5, 0, 0.5, 0],
                                  method='quintic', duration=2.0, num_points=100)
    results = system.execute_trajectory(traj, controller_name='computed_torque')
    err = np.array([r['error'] for r in results])
    return np.sqrt(np.mean(err**2))


def test_execute_regression():
    print("测试 6: ideal 模式下 execute_trajectory 回归旧版精度")
    from robotic_system import RobotSystem
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # 找 bc_policy.pth
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    system = RobotSystem(mode='dynamics')
    rmse = _run_ctc(system)
    print(f"    CTC RMSE = {rmse:.6f} rad（论文基准 7.86e-4）")
    check("CTC RMSE 与论文基准一致", abs(rmse - 7.86e-4) < 1e-4)


def test_noisy_tracking_improvement():
    print("测试 7: 含噪条件下输入感知卡尔曼优于原始测量（CTC, noise=0.005）")
    from robotic_system import RobotSystem
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    results = {}
    for mode in ('raw', 'kalman'):
        system = RobotSystem(mode='dynamics')
        system.set_sensing(mode=mode, noise_std=0.005, seed=42,
                           process_noise=1.0)
        traj = system.plan_trajectory([0]*6, [0.5, 0.5, -0.5, 0, 0.5, 0],
                                      method='quintic', duration=2.0,
                                      num_points=100)
        res = system.execute_trajectory(traj, controller_name='computed_torque')
        err = np.array([r['error'] for r in res])
        results[mode] = np.sqrt(np.mean(err**2))
        check("含噪仿真数值稳定", np.isfinite(results[mode]) and results[mode] < 1)
    print(f"    CTC RMSE: raw {results['raw']:.6f} | kalman {results['kalman']:.6f}")
    check("kalman 模式 RMSE 更低", results['kalman'] < results['raw'])


def test_input_unaware_kalman_diverges():
    """对照实验：无输入卡尔曼在闭环下发散（分离原理不成立的实证）"""
    print("测试 8: 无输入卡尔曼闭环发散（受控负面结果）")
    from robotic_system import RobotSystem
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    system = RobotSystem(mode='dynamics')
    system.set_sensing(mode='kalman', noise_std=0.005, seed=42,
                       process_noise=1.0, input_aware=False)
    traj = system.plan_trajectory([0]*6, [0.5, 0.5, -0.5, 0, 0.5, 0],
                                  method='quintic', duration=2.0, num_points=100)
    res = system.execute_trajectory(traj, controller_name='computed_torque')
    err = np.array([r['error'] for r in res])
    rmse = np.sqrt(np.mean(err**2))
    print(f"    无输入卡尔曼 RMSE = {rmse:.3e}（预期 >> 1）")
    check("无输入卡尔曼如预期发散", rmse > 1.0)


if __name__ == '__main__':
    test_ideal_passthrough()
    test_raw_noise_statistics()
    test_quantization()
    test_kalman_smooths_velocity()
    test_seed_reproducibility()
    test_execute_regression()
    test_noisy_tracking_improvement()
    test_input_unaware_kalman_diverges()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    sys.exit(1 if FAIL else 0)
