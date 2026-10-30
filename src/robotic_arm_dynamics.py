"""
机械臂动力学模块
"""
import numpy as np
from numba import njit
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from robotic_arm_new import RoboticArm


@njit
def fast_christoffel_computation(num_joints, dM_dq_arr, q_dot):

    C = np.zeros((num_joints, num_joints))
    for i in range(num_joints):
        for j in range(num_joints):
            for k in range(num_joints):
                # 查表直接获取偏导数
                c_ijk = 0.5 * (dM_dq_arr[k, i, j] + dM_dq_arr[j, i, k] - dM_dq_arr[i, j, k])
                C[i, j] += c_ijk * q_dot[k]
    return C

class RoboticArmDynamics(RoboticArm):
    def __init__(self, num_joints=6):
        """初始化机械臂动力学模型
        参数:
            num_joints: 机械臂关节数量 (默认为 6)
        """
        super().__init__(num_joints)
        #self.link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.link_lengths = []
        for i in range(self.num_joints):
            a = abs(self.dh_params[i, 2])   # a 参数
            d = abs(self.dh_params[i, 1])   # d 参数
            length = a if a > 1e-6 else (d if d > 1e-6 else 0.05)  # 都为0时给最小值
            self.link_lengths.append(length)
        # 连杆质量
        self.link_masses = [2.0, 3.0, 2.5, 1.5, 1.0, 0.5]
        # 连杆质心位置 (相对于关节坐标系)
        self.link_coms = [
            [0, 0, self.link_lengths[0]/2],  # Link 1
            [self.link_lengths[1]/2, 0, 0],  # Link 2
            [self.link_lengths[2]/2, 0, 0], # Link 3
            [0, 0, self.link_lengths[3]/2],  # Link 4
            [0, 0, 0], # Link 5
            [0, 0, self.link_lengths[5]/2]  # Link 6
        ]
        #连杆惯性张量（简化为对角矩阵 ）
        self.link_inertias = []
        for i, (m, l) in enumerate(zip(self.link_masses, self.link_lengths)):
            # 圆柱模型：考虑半径，半径假设为连杆长度的 1/10，最小 0.02m
            r = max(l * 0.1, 0.02)
            # 沿轴方向: (1/2)mr², 垂直方向: (1/12)m(3r² + L²)
            I_axial   = 0.5 * m * r**2
            I_radial  = (1/12) * m * (3*r**2 + l**2)
            # 至少保证 0.01 kg·m²
            I_axial  = max(I_axial,  0.01)
            I_radial = max(I_radial, 0.01)
            self.link_inertias.append(np.diag([I_radial, I_radial, I_axial]))
        # 重力加速度
        self.gravity = np.array([0, 0, -9.81])

        # 关节摩擦系数 
        self.friction_coeffs = np.array([2.0, 5.0, 4.0, 1.0, 1.0, 0.5])

    def mass_matrix(self, q):
        """计算质量矩阵 M(q)
        参数:
            q: 关节角度列表 (长度应为 num_joints)
        返回:
            M: 质量矩阵 (num_joints x num_joints)
        """
        M = np.zeros((self.num_joints, self.num_joints))

        # 对每个连杆 k 计算其对整体质量矩阵的贡献
        for k in range(self.num_joints):
            # 获取第 k 个连杆的质量和惯性张量
            m_k = self.link_masses[k]
            I_k = self.link_inertias[k]

            # 获取第 k 个连杆的雅可比矩阵 (3 x num_joints)
            # 注意：是连杆 k，计算它受所有前面关节的影响
            J_vk = self._velocity_jacobian(q, k)
            J_wk = self._angular_velocity_jacobian(q, k)

            # 矩阵乘法直接累加贡献： J_v.T * J_v 是 (6x3) @ (3x6) = (6x6) 的矩阵
            M += m_k * (J_vk.T @ J_vk) + (J_wk.T @ I_k @ J_wk)

        # 给每个关节加一个最小转子惯量（模拟电机转子），防止 M 奇异
        rotor_inertia = np.array([0.5, 0.5, 0.3, 0.1, 0.1, 0.05])
        M += np.diag(rotor_inertia)        

        return M
    def coriolis_matrix(self, q, q_dot):
        """计算科氏力矩阵 C(q, q_dot)
        参数:
            q: 关节角度列表
            q_dot: 关节速度列表
        返回:
            C: 科氏力矩阵 (num_joints x num_joints)
        """
        #如果当前各关节速度都极小，直接返回 0 矩阵
        #if np.max(np.abs(q_dot)) < 0.05:
        #    return np.zeros((self.num_joints, self.num_joints))
        
        C = np.zeros((self.num_joints, self.num_joints))
        epsilon = 1e-3 #1e-6

        # 1. 预计算：将质量矩阵 M 对每个关节角度 q_k 的偏导数先算出来
        # dM_dq 会保存 6 个 6x6 的矩阵，避免在三层循环里重复计算 
        dM_dq = []
        for k in range(self.num_joints):
            q_plus = q.copy()
            q_plus[k] += epsilon
            M_plus = self.mass_matrix(q_plus)

            q_minus = q.copy()
            q_minus[k] -= epsilon
            M_minus = self.mass_matrix(q_minus)

            # 数值求导得到偏导数矩阵 ∂M/∂q_k
            dM_dq_k = (M_plus - M_minus) / (2 * epsilon)
            dM_dq.append(dM_dq_k)
        
        # 把 Python 列表转换成形状为 (6, 6, 6) 的 3D Numpy 数组
        dM_dq_arr = np.array(dM_dq)
        
        C = fast_christoffel_computation(self.num_joints, dM_dq_arr, q_dot)
        # 2. 根据克里斯托菲尔符号组装科氏力矩阵
        #for i in range(self.num_joints):
            #for j in range(self.num_joints):
                #for k in range(self.num_joints):
                    # 查表直接获取偏导数，公式：c_ijk = 0.5 * ( ∂M_ij/∂q_k + ∂M_ik/∂q_j - ∂M_jk/∂q_i )
                    #c_ijk = 0.5 * (dM_dq[k][i, j] + dM_dq[j][i, k] - dM_dq[i][j, k])
                    #C[i, j] += c_ijk * q_dot[k]

        return C
        
    def _get_link_com_position(self, q, link_idx):
        """获取连杆质心在世界坐标系下的位置"""
        transforms = self.get_all_joint_transforms(q)
        T_i = transforms[link_idx]
        R_i = T_i[0:3, 0:3]                          # 旋转矩阵
        p_i = T_i[0:3, 3]                            # 关节原点位置
        com_local = np.array(self.link_coms[link_idx])
        return p_i + R_i @ com_local                 # 质心世界坐标
    
    def gravity_vector(self, q):
        """计算重力向量 G(q)
        参数:
            q: 关节角度列表
        返回:
            G: 重力向量 (num_joints,)
            实现重力项计算
        """
        G = np.zeros(self.num_joints)

        # 对每个关节计算重力贡献
        for i in range(self.num_joints):
            # 获得连杆质心的雅可比矩阵
            J_com = self._velocity_jacobian(q, i)
            # 重力矩阵 = J^T * F_gravity
            m_i = self.link_masses[i]
            F_gravity = m_i * self.gravity

            G += J_com.T @ F_gravity

        return G
    def forward_dynamics(self, q, q_dot,tau):
        """
        正向动力学：给定力矩，计算加速度

        参数:
            q: 关节角度列表
            q_dot: 关节速度列表
            tau: 关节力矩列表
            返回:
            q_ddot: 关节加速度列表
            实现正向动力学计算
            公式：q'' = M^(-1)(tau - C*q' - G)
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)

        # 考虑摩擦力矩
        friction = self.friction_coeffs * q_dot

        rhs = tau - C @ q_dot - G - friction

        # 计算加速度
            # 先检查条件数，防止奇异
        cond = np.linalg.cond(M)
        if cond > 1e8:
            import warnings
            warnings.warn(f"质量矩阵接近奇异，条件数={cond:.2e}，使用 pinv 兜底")
            return np.linalg.pinv(M) @ rhs
    
        return np.linalg.solve(M, rhs)
    
    def inverse_dynamics(self, q, q_dot, q_ddot):
        """
        逆向动力学：给定期望的加速度，计算所需的力矩

        参数:
            q: 关节角度列表
            q_dot: 关节速度列表
            q_ddot: 期望的关节加速度列表
            返回:
            tau: 关节力矩列表
            实现逆向动力学计算
            公式：tau = M*q'' + C*q' + G
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)

        # 计算所需的力矩
        tau = M @ q_ddot + C @ q_dot + G 

        # 添加摩擦补偿
        tau += self.friction_coeffs * q_dot

        return tau
    
    def simulate(self, q0, q_dot0, tau_func, t_span, dt=0.01, use_fixed_step=False):
        """
        模拟机械臂运动

        参数:
            q0: 初始关节角度列表
            q_dot0: 初始关节速度列表
            tau_func: 力矩函数，接受 (q, q_dot, t) 作为输入，返回力矩列表
            t: 时间数组
            返回:
            q_traj: 关节角度轨迹 (len(t) x num_joints)
            实现机械臂运动模拟
        """
        # 计数器
        step_count = [0]

        def dynamics_fn(t, state):
            """状态方程"""
            q = state[:self.num_joints]
            q_dot = state[self.num_joints:]
            # 显示计算进度
            step_count[0] += 1
            if step_count[0] % 50 == 0:
                print(f"动力学仿真计算中... 虚拟时间推进到了 t = {t:.3f} 秒")

            #计算力矩
            tau = tau_func(q, q_dot, t).copy()


            #软限位
            k_spring = 10.0
            k_damp   = 3.0
            for i, (lo, hi) in enumerate(self.joint_limits):
                if q[i] < lo:
                    tau[i] += k_spring * (lo - q[i]) - k_damp * q_dot[i]
                elif q[i] > hi:
                    tau[i] += k_spring * (hi - q[i]) - k_damp * q_dot[i]
    
            #计算加速度
            q_ddot = self.forward_dynamics(q, q_dot, tau)
            #返回状态导数
            return np.concatenate([q_dot, q_ddot])
        
        #初始状态
        state0 = np.concatenate([q0, q_dot0])
            
        if use_fixed_step:
            # 半隐式 Euler（Symplectic Euler）：先更新速度，再用新速度更新位置
            # 对机械系统比普通 Euler 稳定得多
            t_arr = np.arange(t_span[0], t_span[1], dt)
            n = self.num_joints
            states = np.zeros((len(t_arr), len(state0)))
            states[0] = state0

            for idx in range(len(t_arr) - 1):
                t_cur = t_arr[idx]
                q     = states[idx, :n].copy()
                q_dot = states[idx, n:].copy()

                # 计算力矩
                tau = tau_func(q, q_dot, t_cur).copy()

                # 软限位
                k_spring, k_damp = 10.0, 3.0
                for i, (lo, hi) in enumerate(self.joint_limits):
                    if q[i] < lo:
                        tau[i] += k_spring * (lo - q[i]) - k_damp * q_dot[i]
                    elif q[i] > hi:
                        tau[i] += k_spring * (hi - q[i]) - k_damp * q_dot[i]

                # 计算加速度
                q_ddot = self.forward_dynamics(q, q_dot, tau)

                # NaN 保护
                if not np.all(np.isfinite(q_ddot)):
                    print(f"警告：t={t_cur:.3f}s 时仿真发散，停止积分")
                    states = states[:idx+1]
                    t_arr = t_arr[:idx+1]
                    break

                # 半隐式 Euler：先更新速度，再用新速度更新位置
                q_dot_new = q_dot + dt * q_ddot
                q_new     = q     + dt * q_dot_new

                # 更新进度
                step_count[0] += 1
                if step_count[0] % 50 == 0:
                    print(f"动力学仿真计算中... 虚拟时间推进到了 t = {t_cur:.3f} 秒")

                states[idx+1, :n] = q_new
                states[idx+1, n:] = q_dot_new

            q     = states[:, :n]
            q_dot = states[:, n:]
            return t_arr, q, q_dot
        
        else:
            # Radau 用于计算力矩控制（无状态控制器）
            t_eval = np.arange(t_span[0], t_span[1], dt)
            result = solve_ivp(
                dynamics_fn, t_span, state0,
                method='Radau', t_eval=t_eval,
                rtol=1e-3, atol=1e-5, max_step=dt
            )
            if not result.success:
                print(f"警告：ODE求解未完全收敛：{result.message}")
            return result.t, result.y[:self.num_joints].T, result.y[self.num_joints:].T

    
        #数值积分
        #states = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, method='RK45', max_step=dt)

        #q = states[:, :self.num_joints]
        #q_dot = states[:, self.num_joints:]

        #return t, q, q_dot

    def _velocity_jacobian(self, q, link_idx):
        """计算连杆速度雅可比矩阵"""
        #简化实现：使用数值微分
        J = np.zeros((3, self.num_joints))
        epsilon = 1e-4 #1e-6

        for i in range(self.num_joints):
            q_plus = q.copy(); q_plus[i] += epsilon
            q_minus = q.copy(); q_minus[i] -= epsilon
        
            # ← 改为质心位置，不是关节原点
            pos_plus  = self._get_link_com_position(q_plus,  link_idx)
            pos_minus = self._get_link_com_position(q_minus, link_idx)
        
            J[:, i] = (pos_plus - pos_minus) / (2 * epsilon)
        return J

    def _angular_velocity_jacobian(self, q, link_idx):
        """计算连杆角速度雅可比矩阵"""
        #简化实现
        J = np.zeros((3, self.num_joints))
        #实现角速度雅可比矩阵
        
        transforms = self.get_all_joint_transforms(q)
    
        for i in range(link_idx + 1):   # 只有 link_idx 之前的关节有贡献
            T_i = transforms[i]
            J[:, i] = T_i[0:3, 2]      # z 轴方向就是第3列的前3个元素
        return J

    def _com_jacobian(self, q, link_idx):
        """计算质心雅可比矩阵"""
        #简化实现
        return self._velocity_jacobian(q, link_idx)
    
    def _get_link_position(self, q, link_idx):
        """获得指定连杆的位置 (x, y, z)"""
        # 调用robotics_arm_new中的方法，获取所有关节的 4x4 变换矩阵列表
        transforms = self.get_all_joint_transforms(q)
        
        # 提取目标连杆的变换矩阵
        T_i = transforms[link_idx]
        
        # 变换矩阵的第 4 列的前 3 个元素，就是该连杆在世界坐标系下的位置 [x, y, z]
        return T_i[0:3, 3]
    
    
class JointKalmanFilter:
    """
    单关节卡尔曼滤波器
    观测: z = [位置]（速度由滤波器估计出来）

    用途：传感器只能测位置，速度是估计出的，比直接数值微分更平滑、抗噪

    model:
        'cv' —— 匀速模型，状态 x = [位置, 速度]
        'ca' —— 匀加速模型，状态 x = [位置, 速度, 加速度]
                追踪加速度非零的轨迹（如五次多项式）时滞后更小，
                代价是噪声略大、计算量略高
    """
    def __init__(self, dt=0.01, process_noise=0.01, measurement_noise=0.05,
                 model='cv'):
        self.dt = dt
        self.model = model

        if model == 'cv':
            # 状态转移矩阵 F (匀速模型)
            self.F = np.array([[1.0, dt ],
                               [0.0, 1.0]])
            # 观测矩阵 H：只能测到位置
            self.H = np.array([[1.0, 0.0]])
            # 过程噪声协方差 Q（连续白噪声加速度模型离散化）
            q = process_noise
            self.Q = q * np.array([[dt**4/4, dt**3/2],
                                   [dt**3/2, dt**2  ]])
            self._dim = 2
        elif model == 'ca':
            # 匀加速模型
            self.F = np.array([[1.0, dt,  dt**2/2],
                               [0.0, 1.0, dt     ],
                               [0.0, 0.0, 1.0    ]])
            self.H = np.array([[1.0, 0.0, 0.0]])
            # 连续白噪声加加速度 (jerk) 模型离散化
            q = process_noise
            self.Q = q * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                   [dt**4/8,  dt**3/3, dt**2/2],
                                   [dt**3/6,  dt**2/2, dt     ]])
            self._dim = 3
        else:
            raise ValueError("model 必须是 'cv' 或 'ca'")

        # 观测噪声协方差 R：传感器噪声水平
        self.R = np.array([[measurement_noise**2]])
        self._mn = measurement_noise

        # 已知控制输入（加速度）的输入矩阵 B：x_pred = F x + B * a
        # 仅 CV 模型支持"输入感知"：把控制器上一步施加力矩对应的模型预测
        # 加速度作为已知输入注入预测步。无输入滤波器把控制效果当作未知扰动，
        # 估计误差会被控制量持续驱动，闭环下有失稳风险（分离原理不成立，
        # 因为观测器不知道控制输入）。
        self.B = np.array([[dt**2 / 2], [dt]])

        # 初始状态和协方差
        self.x = np.zeros(self._dim)
        self.P = np.eye(self._dim) * 1.0     # 初始不确定性

    def update(self, q_measured, u_acc=None):
        """
        输入一次位置观测，返回 (滤波后的位置, 估计的速度)

        参数:
            q_measured: 位置观测值
            u_acc: 上一步控制输入对应的模型预测加速度（输入感知模式），
                   None 时退化为普通无输入滤波器
        """
        # === 预测步 ===
        x_pred = self.F @ self.x
        if u_acc is not None:
            if self._dim != 2:
                raise ValueError("输入感知(u_acc)仅支持 CV 模型")
            x_pred = x_pred + (self.B @ np.atleast_1d(u_acc)).ravel()
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # === 更新步 ===
        z = np.array([q_measured])
        y = z - self.H @ x_pred                              # 残差
        S = self.H @ P_pred @ self.H.T + self.R              # 残差协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S)             # 卡尔曼增益

        self.x = x_pred + (K @ y).flatten()
        self.P = (np.eye(self._dim) - K @ self.H) @ P_pred

        return self.x[0], self.x[1]   # 滤波后的位置和速度

    def reset(self, q_init=0.0):
        """
        重置滤波器。

        初始协方差取"位置已知（= 测量噪声水平）、速度/加速度较不确定"，
        而非旧版的 P = I：初始位置确实已知（复位时给定 q_init），
        P 过大的前几步会产生 ~50 倍的速度增益，速度估计被测量噪声猛推，
        闭环下会激发控制器与估计器互相追逐直至发散。
        """
        self.x = np.zeros(self._dim)
        self.x[0] = q_init
        P0 = np.eye(self._dim) * 0.01      # 速度/加速度初始不确定度 0.1
        P0[0, 0] = max(self._mn**2, 1e-8)  # 位置初始不确定度 = 观测噪声
        self.P = P0

class PIDController:
    def __init__(self, Kp, Ki, Kd, num_joints=6, integral_limit=30.0, error_threshold=0.1, dt=0.01,
                 use_kalman=False,                    
                 process_noise=0.01,                  
                 measurement_noise=0.05):
        """PID控制器初始化
        参数:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
            integral_limit: 积分项限幅的最大绝对值 (积分限幅 Anti-Windup)
            error_threshold: 启动积分的误差阈值 (积分分离)
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        
        self.integral_limit = integral_limit
        self.error_threshold = error_threshold
        # 积分项累计
        self.integral = np.zeros(num_joints)
        # 上一次误差
        self.prev_error = np.zeros(num_joints)
        # 时间步长
        self.dt = dt

        # 卡尔曼滤波器（每个关节独立）
        self.use_kalman = use_kalman
        if use_kalman:
            self.kf_list = [
                JointKalmanFilter(dt=dt,
                                  process_noise=process_noise,
                                  measurement_noise=measurement_noise)
                for _ in range(num_joints)
            ]


    def compute_control(self, q_desired, q_actual, q_dot_actual, q_dot_desired=None):
        """计算闭环控制力矩       
        参数:
            q_desired: 期望关节角度列表
            q_current: 当前关节角度列表
            q_dot_current: 当前关节速度列表
            dt: 时间步长
            返回:
            tau: 控制力矩列表
        """
        # 如果没有提供期望速度（定点控制），则默认为 0
        if q_dot_desired is None:
            q_dot_desired = np.zeros_like(q_actual)

        # 卡尔曼滤波：用观测的 q 估计更平滑的 q 和 q_dot
        if self.use_kalman:
            q_filt = np.zeros_like(q_actual)
            q_dot_filt = np.zeros_like(q_dot_actual)
            for i, kf in enumerate(self.kf_list):
                q_filt[i], q_dot_filt[i] = kf.update(q_actual[i])
        else:
            q_filt = q_actual
            q_dot_filt = q_dot_actual

        # 用滤波后的值计算误差
        error = q_desired - q_filt

        for i in range(len(error)):
            if abs(error[i]) < self.error_threshold:
                self.integral[i] += error[i] * self.dt

        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        error_dot = q_dot_desired - q_dot_filt   # ← 用滤波速度

        tau_feedback = (self.Kp * error + 
                        self.Ki * self.integral + 
                        self.Kd * error_dot)
        
        self.prev_error = error
        return tau_feedback

    def reset(self, q_init=None):
        self.integral = np.zeros_like(self.integral)
        self.prev_error = np.zeros_like(self.prev_error)
        if self.use_kalman:
            for i, kf in enumerate(self.kf_list):
                init_val = q_init[i] if q_init is not None else 0.0
                kf.reset(q_init=init_val)
    

class ComputedTorqueController:
    def __init__(self, robotic_dynamics, Kp, Kd):
        """计算力矩控制器初始化
        参数:
            robotic_dynamics: 机械臂动力学模型实例
            Kp: 比例增益列表
            Kd: 微分增益列表
        """
        self.arm = robotic_dynamics
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)

    def compute_control(self, q_desired, q_dot_desired, q_ddot_desired, q_actual, q_dot_actual):
        """计算控制力矩
        参数:
            q_desired: 期望关节角度列表
            q_dot_desired: 期望关节速度列表
            q_ddot_desired: 期望关节加速度列表
            q_actual: 当前关节角度列表
            q_dot_actual: 当前关节速度列表
            返回:
            tau: 控制力矩列表

            公式：tau = M(q) * (q_ddot_desired + Kp*(q_desired - q_actual) + Kd*(q_dot_desired - q_dot_actual)) + C(q, q_dot)*q_dot + G(q)
        """
        #位置误差和速度误差
        error = q_desired - q_actual
        error_dot = q_dot_desired - q_dot_actual

        #期望加速度（PD控制律）
        q_ddot_cmd = q_ddot_desired + self.Kp * error + self.Kd * error_dot

        #计算所需的力矩（逆向动力学）
        tau = self.arm.inverse_dynamics(q_actual, q_dot_actual, q_ddot_cmd)

        return tau

class GravityCompensationPIDController:
    def __init__(self, robotic_dynamics, pid_controller):
        """前馈+反馈组合控制器初始化
        参数:
            robotic_dynamics: 机械臂动力学模型实例 (用于计算 G)
            pid_controller: 包含完整PID逻辑的控制器实例
        """
        self.arm = robotic_dynamics
        self.pid = pid_controller

    def compute_control(self, q_desired, q_actual, q_dot_actual, q_dot_desired=None):
        """计算总控制力矩
        公式：tau = G(q) + PID_output
        """
        # 1. 前馈：计算当前姿态下的重力补偿力矩
        tau_gravity = self.arm.gravity_vector(q_actual)

        # 2. 反馈：计算 PID 闭环补偿力矩
        tau_pid = self.pid.compute_control(q_desired, q_actual, q_dot_actual, q_dot_desired)

        # 3. 总控制力矩
        tau_total = tau_gravity + tau_pid
        
        return tau_total

class CartesianImpedanceController:
    """
    笛卡尔空间阻抗控制器
    
    让末端在3D空间表现得像一个 弹簧-阻尼 系统：
        F = K_p * (x_desired - x_actual) - K_d * x_dot_actual
    然后通过雅可比转置 tau = J^T * F 转成关节力矩
    
    特点：
    - 末端"软"地被拉向目标点，遇到外力会让步（适合接触任务）
    - 不需要精确的逆运动学，直接在笛卡尔空间施力
    - 配合重力补偿一起用，效果最佳
    """
    def __init__(self, robotic_dynamics, 
                 K_p=None, K_d=None,
                 use_gravity_comp=True):
        """
        参数:
            robotic_dynamics: 动力学模型实例（用于算雅可比和重力）
            K_p: 笛卡尔空间刚度矩阵 (3,)，单位 N/m，越大越"硬"
            K_d: 笛卡尔空间阻尼矩阵 (3,)，单位 N·s/m
            use_gravity_comp: 是否启用重力补偿
        """
        self.arm = robotic_dynamics
        # 默认刚度：x/y/z 三个方向各 200 N/m（中等软度）
        self.K_p = np.array(K_p) if K_p is not None else np.array([200.0, 200.0, 200.0])
        # 默认阻尼：建议 K_d ≈ 2*sqrt(K_p)，临界阻尼避免振荡
        self.K_d = np.array(K_d) if K_d is not None else np.array([28.0, 28.0, 28.0])
        self.use_gravity_comp = use_gravity_comp

    def compute_control(self, x_desired, q_actual, q_dot_actual):
        """
        参数:
            x_desired: 末端期望位置 (3,) [x, y, z]
            q_actual: 当前关节角度 (6,)
            q_dot_actual: 当前关节速度 (6,)
        返回:
            tau: 关节力矩 (6,)
        """
        # 1. 计算末端当前位置（用父类的正向运动学）
        T_end = self.arm.forward_kinematics(q_actual)
        x_actual = T_end[0:3, 3]

        # 2. 计算末端线速度雅可比 J (3x6)
        J = self.arm.jacobian(q_actual)
        
        # 3. 末端当前速度 = J @ q_dot
        x_dot_actual = J @ q_dot_actual

        # 4. 笛卡尔空间的虚拟弹簧-阻尼力 F = K_p * Δx - K_d * v
        F = self.K_p * (x_desired - x_actual) - self.K_d * x_dot_actual

        # 5. 通过雅可比转置映射到关节力矩 tau = J^T * F
        tau = J.T @ F

        # 6. 加上重力补偿（不补偿的话机械臂会因重力下垂）
        if self.use_gravity_comp:
            tau += self.arm.gravity_vector(q_actual)

        return tau
    
#测试代码
def test_freefall():
    #创建动力学模型
    arm = RoboticArmDynamics(num_joints=6)

    #测试1:重力补偿
    print("测试1：重力补偿")
    q = np.array([0, np.pi/4, -np.pi/4, 0, 0, 0])
    G = arm.gravity_vector(q)
    print(f"重力力矩：{G}")

    #测试2：自由落体仿真
    print("\n测试2：自由落体仿真")
    q0 = np.array([0, np.pi/4, 0, 0, 0, 0])
    q_dot0 = np.zeros(6)

    def zero_torque(q, q_dot, t):
        return np.zeros(6)
    
    t, q_hist, q_dot_hist = arm.simulate(
        q0, q_dot0, zero_torque, [0, 1.0], dt=0.01, 
        use_fixed_step=True
    )            
    print(f"仿真时间：{len(t)}个时间步")
    print(f"最终角度：{q_hist[-1]}")

def test_pid_controller():
    """测试重力补偿 + PID闭环控制器"""
    arm = RoboticArmDynamics(num_joints=6)

    # PID参数 
    Kp = [1.0, 5.0, 4.0, 0.5, 0.5, 0.5]
    Ki = [0.1, 0.5, 0.4, 0.05, 0.05, 0.05]    
    Kd = [1.5, 3.0, 2.5, 0.3, 0.3, 0.3]

    # 启用卡尔曼滤波
    pure_pid = PIDController(Kp, Ki, Kd, 
                             integral_limit=30.0, error_threshold=0.1, dt=0.01,
                             use_kalman=True,             # ← 开启
                             process_noise=0.01,
                             measurement_noise=0.02)
    
    controller = GravityCompensationPIDController(arm, pure_pid)

    q_target = np.array([0.0, 0.3, -0.3, 0.0, 0.2, 0.0])
    q0 = np.array([0.0, 0.1, -0.1, 0.0, 0.1, 0.0])
    q_dot0 = np.zeros(6)

    # 模拟带噪声的传感器读数
    noise_std = 0.005   # 5毫弧度的位置噪声
    rng = np.random.default_rng(seed=42)

    # 定义控制函数，内部添加观测噪声
    def control_torque(q, q_dot, t):
        # 在真实 q 上加观测噪声
        q_noisy = q + rng.normal(0, noise_std, size=q.shape)
        q_dot_noisy = q_dot + rng.normal(0, noise_std*10, size=q_dot.shape)
        return controller.compute_control(q_target, q_noisy, q_dot_noisy)
    
    pure_pid.reset(q_init=q0)   # 用 q0 初始化卡尔曼

    print("重力补偿+PID(带卡尔曼滤波)闭环仿真计算中...")
    t, q_hist, q_dot_hist = arm.simulate(
        q0, q_dot0, control_torque, [0, 5.0], dt=0.01,
        use_fixed_step=True
    )

    # 绘制结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(t, q_hist[:, i], label=f'Joint {i+1}')
        plt.axhline(q_target[i], color='r', linestyle='--', label='Target')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title(f'Joint {i+1} Angle')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 计算稳态误差
    final_error = np.abs(q_hist[-1] - q_target)
    print(f"最终误差：{final_error}")
    print(f"最大误差：{np.max(final_error):.6f} rad")



#测试代码
def compare_controllers():
    """比较计算力矩控制器和重力补偿控制器"""
    arm = RoboticArmDynamics(num_joints=6)

    #目标轨迹（正弦波）
    def trajectory(t):
        q = 0.5 * np.sin(2 * np.pi * 0.2 * t) * np.ones(6)
        q_dot = 0.5 * 2 * np.pi  * 0.2 * np.cos(2 * np.pi* 0.2 * t) * np.ones(6)
        q_ddot = -0.5 * (2 * np.pi * 0.2)**2 * np.sin(2 * np.pi * 0.2 * t) * np.ones(6)
        return q, q_dot, q_ddot
    
    #控制器参数
    Kp = [200, 200, 200, 100, 100, 100]
    Kd = [40, 40, 40, 20, 20, 20]

    #测试计算力矩控制
    print("测试计算力矩控制器...")
    ct_controller = ComputedTorqueController(arm, Kp, Kd)

    #q0 = np.zeros(6)
    #q_dot0 = np.zeros(6)

    def ct_control(q, q_dot, t):
        q_d, q_dot_d, q_ddot_d = trajectory(t)
        return ct_controller.compute_control(q_d, q_dot_d, q_ddot_d, q, q_dot)
    
    q0, _, _ = trajectory(0)
    q_dot0 = np.zeros(6)
    t, q_ct, q_dot_ct = arm.simulate(q0, q_dot0, ct_control, [0, 1.0], dt=0.02)

    #计算跟踪误差
    errors = []
    for i, ti in enumerate(t):
        q_d, _, _ = trajectory(ti)
        errors.append(np.linalg.norm(q_ct[i] - q_d))

    print(f"计算力矩控制 - 平均误差：{np.mean(errors):.4f} rad")
    print(f"计算力矩控制 - 最大误差：{np.max(errors):.4f} rad")

    #绘制结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, q_ct[:, 0], label = 'Actual')
    q_desired = [trajectory(ti)[0][0] for ti in t]
    plt.plot(t, q_desired, '--', label='Desired')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.legend()
    plt.grid(True)
    plt.title('Computed Torque Control')

    plt.subplot(2, 1, 2)
    plt.plot(t, errors)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (rad)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_impedance_controller():
    """测试笛卡尔阻抗控制：末端被虚拟弹簧拉向目标，遇外力会让步"""
    arm = RoboticArmDynamics(num_joints=6)
    
    # 阻抗参数：x/y/z 方向刚度
    K_p = [300.0, 300.0, 300.0]   # N/m
    K_d = [35.0,  35.0,  35.0]    # N·s/m，约 2*sqrt(K_p)
    
    controller = CartesianImpedanceController(arm, K_p=K_p, K_d=K_d)
    
    # 初始姿态（接近"home"位置但稍微弯曲）
    q0 = np.array([0.0, 0.3, -0.5, 0.0, 0.3, 0.0])
    q_dot0 = np.zeros(6)
    
    # 先算出初始末端位置
    T0 = arm.forward_kinematics(q0)
    x_start = T0[0:3, 3]
    print(f"初始末端位置: {x_start.round(4)}")
    
    # 目标点：在初始位置 x 方向偏移 +10cm
    x_target = x_start + np.array([0.10, 0.0, 0.0])
    print(f"目标末端位置: {x_target.round(4)}")
    
    # 模拟一个外力扰动：t=2.5s 时在 z 方向施加 -20 N 持续 0.5 秒
    def control_torque(q, q_dot, t):
        tau = controller.compute_control(x_target, q, q_dot)
        
        # 外力扰动模拟（直接给关节力矩加一个干扰项）
        if 2.5 <= t <= 3.0:
            # 通过 J^T 把外力映射到关节空间
            J = arm.jacobian(q)
            F_ext = np.array([0.0, 0.0, -20.0])   # z 方向向下推 20N
            tau += J.T @ F_ext
        
        return tau
    
    print("阻抗控制仿真计算中...")
    t, q_hist, q_dot_hist = arm.simulate(
        q0, q_dot0, control_torque, [0, 5.0], dt=0.01,
        use_fixed_step=True
    )
    
    # 计算每个时刻的末端位置轨迹
    x_hist = np.zeros((len(t), 3))
    for i in range(len(t)):
        T = arm.forward_kinematics(q_hist[i])
        x_hist[i] = T[0:3, 3]
    
    # 绘制结果
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t, x_hist[:, i], label=f'Actual {labels[i]}')
        axes[i].axhline(x_target[i], color='r', linestyle='--', label=f'Target {labels[i]}')
        axes[i].axvspan(2.5, 3.0, alpha=0.2, color='orange', label='External force')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f'{labels[i]} position (m)')
        axes[i].set_title(f'End-Effector {labels[i]} Position')
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()
    
    # 最终位置
    print(f"最终末端位置: {x_hist[-1].round(4)}")
    print(f"位置误差: {np.linalg.norm(x_hist[-1] - x_target):.4f} m")


if __name__ == "__main__":
    # ===== 0. 验证质量矩阵健康度 =====
    print("========== 验证质量矩阵 ==========")
    arm = RoboticArmDynamics()
    M = arm.mass_matrix(np.zeros(6))
    print("质量矩阵对角线:", np.diag(M).round(4))
    print("条件数:", np.linalg.cond(M))
    print()
    # ===== 1.单独验证重力补偿方向 =====
    print("========== 验证重力向量符号 ==========")
    arm_test = RoboticArmDynamics(num_joints=6)
    q_test = np.array([0, np.pi/4, 0, 0, 0, 0])
    G = arm_test.gravity_vector(q_test)
    print(f"重力向量 G = {G.round(4)}")
    print(f"Joint 2 (抬起 45°) 的重力矩 G[1] = {G[1]:.4f}")
    print(f"期望：G[1] > 0（需要正力矩抵抗重力下压）")
    print(f"结果：{'✓ 符号正确' if G[1] > 0 else '✗ 符号反了，需要取负'}")
    print()

    print("========== 运行自由落体测试 ==========")
    test_freefall()
    
    print("\n========== 运行重力补偿+PID测试 ==========")
    test_pid_controller()
    
    print("\n========== 运行计算力矩控制器测试 ==========")
    compare_controllers()

    print("\n========== 运行笛卡尔阻抗控制测试 ==========")
    test_impedance_controller()