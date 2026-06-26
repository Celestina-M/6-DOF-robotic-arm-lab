import numpy as np
import time
from numba import njit

# 核心计算函数 (保持在类外进行 Numba 加速) 
@njit
def fast_fk_computation(joint_angles, dh_params):
    """齐次变换矩阵连乘逻辑"""
    T = np.eye(4, dtype=np.float64)
    # 强制确保 T 是 float64 类型
    for i in range(len(joint_angles)):
        # dh_params 结构: [theta_offset, d, a, alpha]
        theta_offset, d, a, alpha = dh_params[i]
        theta = theta_offset + joint_angles[i]
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        # 所有的 0 和 1 都要写成 0.0 和 1.0
        # 这样 Numba 才能正确识别这是一个全浮点数的二维数组
        T_i = np.array([
            [ct,  -st*ca,  st*sa,  a*ct],
            [st,   ct*ca, -ct*sa,  a*st],
            [0.0,  sa,     ca,     d   ],
            [0.0,  0.0,    0.0,    1.0 ]
        ], dtype=np.float64)# 显式指定 dtype 是一种保险做法
        T = T @ T_i
    return T

class RoboticArm:
    def __init__(self, num_joints=6):
        """初始化机械臂：设定 UR5 标准参数"""
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints)
        # 关节限制
        self.joint_limits = [(-np.pi, np.pi)] * num_joints
        
        # UR5 标准 DH 参数 [theta_offset, d, a, alpha]
        self.dh_params = np.array([
            [0, 0.1625,  0,       np.pi/2 ],  # Joint 1
            [0, 0,      -0.425,   0       ],  # Joint 2: 上臂长度
            [0, 0,      -0.3922,  0       ],  # Joint 3: 前臂长度
            [0, 0.1333,  0,       np.pi/2 ],  # Joint 4
            [0, 0.0997,  0,      -np.pi/2 ],  # Joint 5
            [0, 0.0996,  0,       0       ]   # Joint 6
        ], dtype=np.float64)

    def forward_kinematics(self, joint_angles=None):
        """调用加速后的计算函数"""
        if joint_angles is None:
            joint_angles = self.joint_angles

        # 转换为 numpy 数组确保兼容性
        joint_angles = np.array(joint_angles, dtype=np.float64)
        return fast_fk_computation(joint_angles, self.dh_params)

    def set_joint_angles(self, angles):
        """设置角度"""
        self.joint_angles = np.clip(np.array(angles), -np.pi, np.pi)

    def get_end_effector_position(self):
        """快速获取末端 [x, y, z]"""
        T = self.forward_kinematics()
        return T[:3, 3]
    

    # 动力学补充
    
    def get_all_joint_transforms(self, joint_angles=None):
        """
        获取每一个关节在世界坐标系下的 4x4 变换矩阵。
        动力学计算各连杆速度、角速度的前置条件
        """
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        transforms = []
        T = np.eye(4, dtype=np.float64)
        
        for i in range(self.num_joints):
            theta_offset, d, a, alpha = self.dh_params[i]
            theta = theta_offset + joint_angles[i]
            
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            # 当前关节自身的变换矩阵
            T_i = np.array([
                [ct,  -st*ca,  st*sa,  a*ct],
                [st,   ct*ca, -ct*sa,  a*st],
                [0.0,  sa,     ca,     d   ],
                [0.0,  0.0,    0.0,    1.0 ]
            ], dtype=np.float64)
            
            # 累乘得到在世界坐标系下的绝对位姿
            T = T @ T_i
            transforms.append(T.copy()) # 存放每个关节的位姿
            
        return transforms

    # 逆运动学相关方法
    # 1: 实现数值微分计算雅可比矩阵

    def jacobian(self, joint_angles=None, epsilon=1e-6):
        """
        计算雅可比矩阵 (数值微分法 - 中心差分)
        joint_angles: 可选的关节角度输入，如果为 None 则使用当前状态
        epsilon: 微小扰动量，用于数值微分
        返回: 雅可比矩阵 J (3 x num_joints)
        实现微分计算的雅可比矩阵
        """
        if joint_angles is None:
            joint_angles = self.joint_angles.copy()
        else:
            joint_angles = np.array(joint_angles, dtype=np.float64).copy()

        # 初始化雅可比矩阵 (假设末端位置是3D [x,y,z]，所以是 3 x num_joints)
        J = np.zeros((3, self.num_joints))

        # 对每个关节进行数值微分
        for i in range(self.num_joints):
            # 保存原始角度
            original_angle = joint_angles[i]

            # 正向扰动
            joint_angles[i] = original_angle + epsilon
            # 假设 forward_kinematics 返回完整变换矩阵，提取位置部分 [:, 3] 的前三个元素
            pos_plus = self.forward_kinematics(joint_angles)[:3, 3] # 修正：直接取位置

            # 负向扰动
            joint_angles[i] = original_angle - epsilon
            pos_minus = self.forward_kinematics(joint_angles)[:3, 3] # 修正：直接取位置

            # 计算偏导数 (中心差分公式)
            J[:, i] = (pos_plus - pos_minus) / (2 * epsilon)

            # 恢复原始角度
            joint_angles[i] = original_angle

        return J
    
    # 2: 实现逆运动学求解器入口

    def inverse_kinematics(self, target_position, initial_guess=None, 
                           method='jacobian', max_iterations=100, tolerance=1e-3):
        """
        逆运动学求解统一接口
        参数：
        - target_position: 目标末端位置 [x, y, z]
        - initial_guess: 可选的初始关节角度猜测 (长度为 num_joints)
        - method: 'jacobian' 或 'optimization'，选择求解方法
        - max_iterations: 最大迭代次数 (仅对雅可比方法有效)
        - tolerance: 收敛容忍度 (仅对雅可比方法有效)
        返回：      (success: bool 是否成功求解, joint_angles: np.array 求解的关节角度)
        实现逆运动学求解
        """
        target_position = np.array(target_position)

        # 初始猜测
        theta = self.joint_angles.copy() if initial_guess is None else np.array(initial_guess).copy()
        # 使用当前角度或随机初始化

        if method == 'jacobian':
            return self._ik_jacobian(target_position, theta, max_iterations, tolerance)
        elif method == 'optimization':
            return self._ik_optimization(target_position, theta)
        else:
            raise ValueError(f"未知方法: {method}")
        
    # 3: 基于雅可比矩阵的 IK 求解
    def _ik_jacobian(self, target_pos, theta, max_iter, tol):
        """
        雅可比迭代求解器（阻尼最小二乘法 DLS）

        原理：在当前关节角附近，末端位置变化 ≈ J · Δθ。
        我们想让末端朝目标移动 error，于是反解 Δθ，迭代逼近。
        用 DLS 而非纯伪逆，是为了在奇异点附近保持数值稳定、避免 Δθ 爆炸。

        状态保护：迭代过程中会反复调用 self.set_joint_angles 改写 arm 的内部
        joint_angles。若求解失败却不恢复，arm 的内部状态会停留在失败时的角度，
        污染后续的 get_end_effector_position()、以及不传 initial_guess 时的下一次
        IK（它会以当前角度为初始猜测）。因此：求解前存档，仅在成功时保留新状态，
        失败时回滚到求解前的角度。
        """
        damping = 0.05          # 阻尼系数 λ：越大越稳但收敛越慢，越小越快但奇异点附近易发散
        theta = theta.copy()    # 拷贝一份，避免污染外部传入的初始猜测
        saved_angles = self.joint_angles.copy()   # 存档：求解前的 arm 状态

        for _ in range(max_iter):
            # 1. 用当前角度算出末端实际位置
            self.set_joint_angles(theta)
            current_pos = self.get_end_effector_position()

            # 2. 计算位置误差（目标 - 当前），即末端还需移动的向量
            error = target_pos - current_pos

            # 3. 误差足够小则认为收敛，返回成功（保留收敛后的 arm 状态）
            if np.linalg.norm(error) < tol:
                return True, theta

            # 4. 计算当前位姿的雅可比矩阵 (3 x num_joints)
            J = self.jacobian(theta)

            # 5. 阻尼最小二乘求解 Δθ：
            #    Δθ = Jᵀ (J Jᵀ + λ²I)⁻¹ · error
            #    这里 (J Jᵀ) 是 3x3 小矩阵，加 λ²I 保证可逆（奇异点也不会崩）。
            #    用 np.linalg.solve 解线性方程组，比先求逆再相乘更稳更快。
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            delta_theta = J.T @ np.linalg.solve(JJt, error)

            # 6. 自适应步长限制：单步更新过大时按比例缩小，
            #    防止远离目标时步子太猛、越过目标来回震荡而无法收敛
            step = np.linalg.norm(delta_theta)
            if step > 0.3:
                delta_theta = delta_theta * (0.3 / step)

            # 7. 更新关节角度
            theta = theta + delta_theta

            # 8. 应用关节物理限制，防止超出机械臂活动范围
            for i, (min_angle, max_angle) in enumerate(self.joint_limits):
                theta[i] = np.clip(theta[i], min_angle, max_angle)

        # 超过最大迭代次数仍未收敛：回滚 arm 内部状态，返回失败
        # （theta 仍作为"当前最接近的解"返回，供调用者参考，但不污染 arm 状态）
        self.set_joint_angles(saved_angles)
        return False, theta
    
# 任务 4: 基于优化的 IK 求解
    def _ik_optimization(self, target_pos, theta):
        """
        使用 scipy.optimize 的数值优化求解 IK
        """
        from scipy.optimize import minimize

        saved_angles = self.joint_angles.copy()

        def objective(angles):
            """目标函数：位置误差的平方和"""
            self.set_joint_angles(angles)
            return np.sum((target_pos - self.get_end_effector_position())**2)

        # 关节限制作为优化的边界
        bounds = self.joint_limits

        # 执行优化
        result = minimize(
            objective,
            theta,
            method='SLSQP', # 序列最小二乘规划算法，适合带边界的非线性优化
            bounds=bounds,
            options={'maxiter': 100}
        )

        # 检查结果
        if result.success and result.fun < 1e-4:
            self.set_joint_angles(result.x)
            return True, result.x
        else:
            self.set_joint_angles(saved_angles)  # 优化失败时恢复原始状态
            return False, result.x
# 测试代码 
if __name__ == "__main__":
    arm = RoboticArm(num_joints=6)
    
    # 1. 精度验证
    arm.set_joint_angles([0, 0, 0, 0, 0, 0])
    pos = arm.get_end_effector_position()
    print(f"验证：全零位姿末端坐标: {pos.round(4)}")
    
    # 2. 性能预热
    arm.forward_kinematics([0]*6)
    
    # 3. 正式测速
    print("正在进行10000次正向运动学测速...")
    num_tests = 10000
    start = time.time()
    for _ in range(num_tests):
        arm.forward_kinematics(np.random.uniform(-np.pi, np.pi, 6))
    elapsed = time.time() - start
    print(f"优化后性能: {num_tests / elapsed:.0f} ops/sec")

    # 4. 逆运动学简单验证
    #    显式从零位出发，避免被前面测速循环或其它调用留下的状态影响
    print("\n逆运动学测试:")
    target = [0.3, 0.1, 0.4]
    success, sol = arm.inverse_kinematics(target, initial_guess=np.zeros(6))
    print(f"求解状态: {success}, 目标: {target}, 结果位置: {arm.get_end_effector_position().round(4)}")

    # ===== 工作空间自检 =====
    #    每个点都显式从零位出发求解，保证各点相互独立、不串联污染
    print("\n===== 工作空间自检 =====")
    arm.set_joint_angles(np.zeros(6))
    print("零位:", arm.get_end_effector_position().round(4))
    for z in [0.1, 0.2, 0.3, 0.4]:
        ok, sol = arm.inverse_kinematics([0.3, 0.0, z], initial_guess=np.zeros(6))
        arm.set_joint_angles(sol)
        print(f"z={z}: success={ok}, 实际={arm.get_end_effector_position().round(4)}")