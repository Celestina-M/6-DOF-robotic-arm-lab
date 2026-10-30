# 文件: src/sensing.py
"""
统一传感与状态估计模块

设计目的：
    将"传感器模型"与"状态估计器"从单个控制器内部上移到集成层，
    使所有控制器在含噪测量条件下接受完全一致的反馈信号，保证横向比较的公平性。

三层结构：
    1. SensorModel      —— 传感器模型：高斯噪声 + 编码器量化 + 可选零偏
                           （只测位置，符合真实编码器；速度必须由估计得到）
    2. StateEstimator   —— 状态估计器：每关节一个卡尔曼滤波器（CV/CA 模型），
                           由带噪位置观测估计位置与速度
    3. SensingPipeline  —— 传感管线：组合二者，对控制器暴露统一接口
                           mode='ideal' / 'raw' / 'kalman'

使用方式（在 RobotSystem 内部调用，见 robotic_system.py）：
    pipeline = SensingPipeline(dt=0.02, mode='kalman',
                               noise_std=0.005, resolution=1.5e-4, seed=42)
    pipeline.reset(q_init)
    q_fb, dq_fb = pipeline.process(q_true, dq_true)   # 控制器只准用这两个值
"""
import numpy as np

from robotic_arm_dynamics import JointKalmanFilter


class SensorModel:
    """
    关节位置传感器模型（编码器）。

    测量链：q_meas = quantize(q_true + bias) + 高斯噪声
    - noise_std : 高斯测量噪声标准差 (rad)，0 表示无噪声
    - resolution: 编码器量化步长 (rad)，None 表示不量化
                  例：17 位编码器 ≈ 2π/2^17 ≈ 4.8e-5 rad
    - bias      : 常值零偏 (rad)，标量或逐关节数组，None 表示无零偏
    速度不可直接测量（真实编码器只输出位置脉冲计数）。
    """

    def __init__(self, num_joints=6, noise_std=0.0, resolution=None,
                 bias=None, seed=None):
        self.num_joints = num_joints
        self.noise_std = float(noise_std)
        self.resolution = resolution
        if bias is None:
            self.bias = np.zeros(num_joints)
        else:
            self.bias = np.broadcast_to(np.asarray(bias, dtype=float),
                                        (num_joints,)).copy()
        self.rng = np.random.default_rng(seed)

    def measure(self, q_true):
        """输入真实关节角，返回带噪量化后的位置测量值"""
        q = np.asarray(q_true, dtype=float) + self.bias
        if self.resolution is not None:
            q = np.round(q / self.resolution) * self.resolution
        if self.noise_std > 0.0:
            q = q + self.rng.normal(0.0, self.noise_std, size=self.num_joints)
        return q


class StateEstimator:
    """
    多关节状态估计器：由带噪位置观测估计 (q, q_dot)。

    每关节一个独立的 JointKalmanFilter（CV 或 CA 模型）。
    process_noise / measurement_noise 为卡尔曼调参旋钮，
    其中 measurement_noise 应与 SensorModel 的 noise_std 同量级。
    """

    def __init__(self, num_joints=6, dt=0.01, model='cv',
                 process_noise=0.01, measurement_noise=0.05):
        self.num_joints = num_joints
        self.filters = [
            JointKalmanFilter(dt=dt, process_noise=process_noise,
                              measurement_noise=measurement_noise,
                              model=model)
            for _ in range(num_joints)
        ]

    def estimate(self, q_meas, u_acc=None):
        """
        输入一次带噪位置观测，返回 (估计位置, 估计速度)

        u_acc: 逐关节模型预测加速度（输入感知模式），None 时退化
        """
        q_est = np.zeros(self.num_joints)
        dq_est = np.zeros(self.num_joints)
        for i, kf in enumerate(self.filters):
            u = None if u_acc is None else u_acc[i]
            q_est[i], dq_est[i] = kf.update(q_meas[i], u_acc=u)
        return q_est, dq_est

    def reset(self, q_init=None):
        q_init = np.zeros(self.num_joints) if q_init is None else q_init
        for i, kf in enumerate(self.filters):
            kf.reset(q_init=q_init[i])


class SensingPipeline:
    """
    传感管线：对控制器暴露统一的反馈接口 process(q_true, dq_true)。

    mode:
        'ideal'  —— 理想反馈：直接使用真实状态（等价于旧版行为）
        'raw'    —— 原始测量：位置用带噪测量，速度由相邻测量值差分得到
        'kalman' —— 滤波反馈：位置/速度均由卡尔曼滤波器从带噪测量估计
    """

    MODES = ('ideal', 'raw', 'kalman')

    def __init__(self, dt, mode='ideal', num_joints=6,
                 noise_std=0.0, resolution=None, bias=None, seed=None,
                 kalman_model='cv', process_noise=1.0,
                 measurement_noise=None, input_aware=True):
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES}")
        self.mode = mode
        self.dt = dt
        self.num_joints = num_joints
        # 输入感知：把上一步控制力矩的模型预测加速度注入滤波预测步。
        # 无输入滤波器将控制效果当作未知扰动，估计误差被控制量驱动，
        # 闭环下存在失稳风险（观测器不知控制输入，分离原理不成立）。
        self.input_aware = input_aware

        self.sensor = None
        self.estimator = None
        if mode in ('raw', 'kalman'):
            self.sensor = SensorModel(num_joints=num_joints,
                                      noise_std=noise_std,
                                      resolution=resolution,
                                      bias=bias, seed=seed)
        if mode == 'kalman':
            # measurement_noise 缺省时与传感器实际噪声对齐；
            # 下限 5e-4 rad：R 再小会让滤波增益过大，对增益裕度有限的
            # 控制回路（如本平台的 PID）存在闭环失稳风险（实验证实
            # 稳定边界在 R∈(1e-4, 1e-3) 之间）
            mn = noise_std if measurement_noise is None else measurement_noise
            self._kf_process_noise = process_noise
            self._kf_measurement_noise = max(mn, 5e-4)
            self.estimator = StateEstimator(
                num_joints=num_joints, dt=dt, model=kalman_model,
                process_noise=self._kf_process_noise,
                measurement_noise=self._kf_measurement_noise)

        self._prev_q_meas = None   # raw 模式差分用

    def set_dt(self, dt):
        """
        对齐仿真步长。卡尔曼滤波器的 F/Q 矩阵在构造时依赖 dt，
        步长变化必须重建估计器，否则滤波增益与实际步长失配
        （旧实现只改管道 dt 属性，曾导致闭环发散）。
        """
        if abs(dt - self.dt) < 1e-12:
            return
        self.dt = dt
        if self.estimator is not None:
            first = self.estimator.filters[0]
            self.estimator = StateEstimator(
                num_joints=self.num_joints, dt=dt, model=first.model,
                process_noise=self._kf_process_noise,
                measurement_noise=self._kf_measurement_noise)

    def reset(self, q_init=None):
        """每个回合/实验开始前调用，清空滤波器与差分历史"""
        q_init = np.zeros(self.num_joints) if q_init is None else q_init
        if self.estimator is not None:
            self.estimator.reset(q_init)
        self._prev_q_meas = None

    def process(self, q_true, dq_true, ddq_pred=None):
        """
        输入仿真真实状态，返回控制器可用的反馈 (q_fb, dq_fb)。
        控制器在含噪实验中只准使用返回值，不得再访问真实状态。

        ddq_pred: 上一步控制力矩经动力学模型（以反馈状态求值）预测的
                  加速度，仅 kalman 模式且 input_aware=True 时使用。
        """
        if self.mode == 'ideal':
            return np.asarray(q_true, dtype=float), np.asarray(dq_true, dtype=float)

        q_meas = self.sensor.measure(q_true)

        if self.mode == 'raw':
            if self._prev_q_meas is None:
                dq_fb = np.zeros(self.num_joints)
            else:
                dq_fb = (q_meas - self._prev_q_meas) / self.dt
            self._prev_q_meas = q_meas.copy()
            return q_meas, dq_fb

        # mode == 'kalman'
        u = ddq_pred if self.input_aware else None
        return self.estimator.estimate(q_meas, u_acc=u)
