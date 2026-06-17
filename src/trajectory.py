import numpy as np
from scipy.interpolate import CubicSpline
from robotic_arm_new import RoboticArm

class TrajectoryPlanner:
    def __init__(self, robotic_arm_new):
        """
        初始化轨迹规划器
        参数:
        robotic_arm: RoboticArm实例
        """
        self.robotic_arm = robotic_arm_new

    def plan_joint_trajectory(self, start_angles, end_angles, duration=2.0, method='cubic', num_points=50):
        """
        规划从起始关节角度到目标关节角度的轨迹
        参数:
        start_angles: 起始关节角度
        end_angles: 目标关节角度
        duration: 轨迹持续时间（秒）
        method: 插值方法 ('cubic' , 'linear','quintic')
        num_points: 轨迹点数量
        返回:
        trajectory: 规划好的轨迹点列表，每个点包含：
            time: 时间
            position：关节位置
            velocity：关节速度
            acceleration：关节加速度
        实现三种插值方法：线性、三次样条、五次样条，并计算速度和加速度
        """
        start_angles = np.array(start_angles)
        end_angles = np.array(end_angles)
        
        if method == 'linear':
            return self._linear_interpolation(start_angles, end_angles, duration, num_points)
        elif method == 'cubic':
            return self._cubic_interpolation(start_angles, end_angles, duration, num_points)
        elif method == 'quintic':
            return self._quintic_interpolation(start_angles, end_angles, duration, num_points)
        else:
            raise ValueError(f"未知插值方法：请选择 'linear', 'cubic' 或 'quintic'")
    def _linear_interpolation(self, start, end, duration, num_points):
        """
        线性插值方法
        TODO: 实现线性插值，计算速度和加速度
        q(t) = start + (end - start) * (t / duration)
        """
        trajectory = []
        times = np.linspace(0, duration, num_points)
        
        for t in times:
            #归一化时间
            s = t / duration
            #位置计算
            position = (1 - s) * start + s * end
            #速度和加速度计算
            velocity = (end - start) / duration
            acceleration = np.zeros_like(start)
            
            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration
            })
        
        return trajectory

    def _cubic_interpolation(self, start, end, duration, num_points):
        """
        三次样条插值
        边界条件: 起点和终点速度为零
        """
        times = np.linspace(0, duration, num_points)
        key_times = [0, duration]

        # 每个关节只构建一次样条，共 num_joints 次
        splines = [
            CubicSpline(key_times, [start[i], end[i]], bc_type='clamped')
            for i in range(len(start))
        ]

        # 一次性对所有时间点向量化求值，避免逐点调用
        positions     = np.column_stack([cs(times)    for cs in splines])
        velocities    = np.column_stack([cs(times, 1) for cs in splines])
        accelerations = np.column_stack([cs(times, 2) for cs in splines])

        return [
            {
                'time': t,
                'position':     positions[i],
                'velocity':     velocities[i],
                'acceleration': accelerations[i],
            }
            for i, t in enumerate(times)
        ]

    def _quintic_interpolation(self, start, end, duration, num_points):
        """五次多项式插值
        实现五次多项式插值，起点和终点的速度和加速度为零
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        trajectory = []
        times = np.linspace(0, duration, num_points)
        
        #五次多项式系数计算
        #边界条件：
        #q(0) = start, q'(0) = 0, q''(0) = 0
        #q(T) = end, q'(T) = 0, q''(T) = 0

        T= duration
        coeffs = np.zeros((len(start), 6))

        for i in range(len(start)):
            q0 = start[i]
            qf = end[i]

            #系数矩阵  
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                [0, 0, 2, 6*T, 12*T**2, 20*T**3]
            ])

            b = np.array([q0, 0, 0, qf, 0, 0])

            coeffs[i] = np.linalg.solve(A, b)


        #生成轨迹点
        for t in times:
            position = np.zeros(len(start))
            velocity = np.zeros(len(start))
            acceleration = np.zeros(len(start))

            for i in range(len(start)):
                a = coeffs[i]
                position[i] = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
                velocity[i] = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
                acceleration[i] = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3

            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration
            })
        
        return trajectory

    def plan_cartesian_line(self, start_pos, end_pos, num_points=50):
        """
        规划笛卡尔空间中的直线路径
        
        参数：
        start_pos: 起始位置 [x, y, z]
        end_pos: 目标位置 [x, y, z]
        num_points: 轨迹点数量
        返回：
        trajectory: 规划好的轨迹点列表，每个点包含：
            time: 时间
            position：末端位置 [x, y, z]
            velocity：末端速度 [vx, vy, vz]
            acceleration：末端加速度 [ax, ay, az]
        实现笛卡尔空间中的直线路径规划，计算末端位置、速度和加速度
        步骤：
        1. 在笛卡尔空间中插值
        2.对每个点求解IK
        3. 生成关节空间轨迹
        """

        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        trajectory = []
        last_angles = None  # 记录上一个成功解，用于 IK 的初始猜测

        # 在笛卡尔空间中线性插值
        for i in range(num_points):
            t = i/(num_points - 1)
            
            #线性插值
            target_pos = (1 - t) * start_pos + t * end_pos

            #求解IK
            success, angles = self.robotic_arm.inverse_kinematics(
            target_pos, initial_guess=last_angles)

             #如果成功，更新 last_angles

            if not success:
                print(f"警告: 第{i}个点IK求解失败")
                continue

            last_angles = angles
            trajectory.append({
                'time': t,
                'position': angles,
            })
        return trajectory

    def plan_cartesian_circle(self, center, radius, plane='xy', num_points=60):
        """
        规划笛卡尔空间中的圆形路径
        
        参数：
        center: 圆心位置 [x, y, z]
        radius: 圆半径
        plane: 圆所在平面 ('xy', 'yz', 'xz')
        num_points: 轨迹点数量
        返回：
        trajectory: 规划好的轨迹点列表，每个点包含：
            time: 时间
            position：末端位置 [x, y, z]
            velocity：末端速度 [vx, vy, vz]
            acceleration：末端加速度 [ax, ay, az]
        实现笛卡尔空间中的圆形路径规划，计算末端位置、速度和加速度
        """

        center = np.array(center)
        trajectory = []
        last_angles = None  # 记录上一个成功解，用于 IK 的初始猜测

        for i in range(num_points):
            angle = 2 * np.pi * i / num_points 

            #根据选择的平面计算圆上的点
            if plane == 'xy':
                target_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            elif plane == 'yz':
                target_pos = center + np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
            elif plane == 'xz':
                target_pos = center + np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
            else:
                raise ValueError("未知平面类型：请选择 'xy', 'yz' 或 'xz'")

            #求解IK
            success, angles = self.robotic_arm.inverse_kinematics(target_pos, initial_guess=last_angles)

            if success:
                last_angles = angles
                trajectory.append({
                    'time': angle,
                    'position': angles,
                })
        return trajectory 

    def get_star_waypoints(self, center, radius, plane='xy'):
            """获取五角星的5个顶点的笛卡尔坐标"""
            center = np.array(center)
            waypoints = []
            start_angle = np.pi / 2  # 从正上方开始
            
            for i in range(6): # 6个点为了闭合回到起点
                angle = start_angle + i * (4 * np.pi / 5)
                if plane == 'xy':
                    pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
                elif plane == 'yz':
                    pos = center + np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
                elif plane == 'xz':
                    pos = center + np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
                waypoints.append(pos)
                
            return waypoints

    def plan_cartesian_spiral(self, center, radius, z_start, z_end, turns=3, num_points=100):
            """规划笛卡尔空间中的 3D 螺旋路径"""
            center = np.array(center)
            trajectory = []
            last_angles = None          # 记录上一个成功解

            for i in range(num_points):
                t = i / (num_points - 1)
                angle = t * turns * 2 * np.pi
                target_pos = center + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    z_start + t * (z_end - z_start)
                ])
                # 用上一个解作为初始猜测（关键）
                success, angles = self.robotic_arm.inverse_kinematics(
                    target_pos, initial_guess=last_angles, max_iterations=300
                )
                if success:
                    last_angles = angles
                    trajectory.append({'time': t, 'position': angles})
                else:
                    print(f"警告: 螺旋线第{i}个点IK求解失败，位置: {target_pos}")
            return trajectory


    def plan_smooth_multisegment(self, cartesian_waypoints, total_duration, num_points=100):
            """多段轨迹平滑连接 (用于绘制五角星等)"""
            joint_waypoints = []
            valid_waypoints = []
            last_angles = None
            
            for pos in cartesian_waypoints:
                success, angles = self.robotic_arm.inverse_kinematics(pos, initial_guess=last_angles, max_iterations=300)
                if success:
                    last_angles = angles
                    joint_waypoints.append(angles)
                    valid_waypoints.append(pos)
                else:
                    print(f"警告：路径点 {pos} IK求解失败，已跳过该点。")

            if len(joint_waypoints) < 2:
                raise ValueError("有效的路径点不足，无法规划轨迹")

            joint_waypoints = np.array(joint_waypoints)
            num_waypoints = len(joint_waypoints)
            
            waypoint_times = np.linspace(0, total_duration, num_waypoints)
            cs = CubicSpline(waypoint_times, joint_waypoints, axis=0, bc_type='clamped')
            
            trajectory = []
            times = np.linspace(0, total_duration, num_points)
            
            for t in times:
                trajectory.append({
                    'time': t,
                    'position': cs(t),
                    'velocity': cs(t, 1),
                    'acceleration': cs(t, 2)
                })
                
            return trajectory


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 3D 轨迹可视化测试

def test_weekends():
    # 1. 实例化机械臂代码
    arm = RoboticArm(num_joints=6)
    
    # 2. 将机械臂传入轨迹规划器代码
    planner = TrajectoryPlanner(robotic_arm_new=arm)
    
    fig = plt.figure(figsize=(14, 6))

    # 绘制 3D 螺旋线 
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Weekends: 3D Spiral Track", fontsize=14)

    print("正在规划 3D 螺旋线 (可能需要几秒钟解算 IK)...")
    spiral_traj = planner.plan_cartesian_spiral(
        center=[0.3, 0.0, 0.0], radius=0.08, z_start=0.1, z_end=0.3, turns=4, num_points=100
    )

    spiral_x, spiral_y, spiral_z = [], [], []
    for pt in spiral_traj:
        # 调用 FK，提取 T 矩阵的前三行第四列作为 XYZ 位置
        T = planner.robotic_arm.forward_kinematics(pt['position'])
        pos = T[:3, 3] 
        spiral_x.append(pos[0])
        spiral_y.append(pos[1])
        spiral_z.append(pos[2])

    ax1.plot(spiral_x, spiral_y, spiral_z, color='blue', linewidth=2, label='Spiral Path')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.legend()

    # 绘制五角星 (平滑连接)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Weekends: Smooth Star Track", fontsize=14)

    print("正在规划五角星平滑轨迹...")
    star_center = [0.4, 0.0, 0.3] # 稍微调远一点，保证在 UR5 机械臂的舒适工作空间内
    star_waypoints = planner.get_star_waypoints(center=star_center, radius=0.15, plane='yz') 

    star_traj = planner.plan_smooth_multisegment(
        cartesian_waypoints=star_waypoints, total_duration=5.0, num_points=200
    )

    star_x, star_y, star_z = [], [], []
    for pt in star_traj:
        # 调用 FK，提取位置
        T = planner.robotic_arm.forward_kinematics(pt['position'])
        pos = T[:3, 3]
        star_x.append(pos[0])
        star_y.append(pos[1])
        star_z.append(pos[2])

    ax2.plot(star_x, star_y, star_z, color='red', linewidth=2, label='Smooth Trajectory')
    
    wp_x = [p[0] for p in star_waypoints]
    wp_y = [p[1] for p in star_waypoints]
    wp_z = [p[2] for p in star_waypoints]
    ax2.scatter(wp_x, wp_y, wp_z, color='black', s=50, zorder=5, label='Original Waypoints')

    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# 要求1：效率对比与曲线绘制测试脚本

def run_basic_assignment():
    # 1. 实例化机械臂和规划器
    arm = RoboticArm(num_joints=6)
    planner = TrajectoryPlanner(robotic_arm_new=arm)
    
    # 设定起始角度和目标角度 (6个关节)
    start_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end_angles = [np.pi/2, np.pi/4, -np.pi/4, np.pi/2, 0.0, 0.0]
    duration = 2.0
    num_points = 100
    methods = ['linear', 'cubic', 'quintic']
    
    
    # 要求 2: 对比计算效率 (测量运行时间)
    
    print("\n" + "="*40)
    print("要求 2: 计算效率对比")
    print("="*40)
    trajectories = {}
    
    for method in methods:
        # 为了让时间测量更准确，采取循环运行 500 次求平均值
        import time
        start_time = time.perf_counter()
        for _ in range(500):
            traj = planner.plan_joint_trajectory(start_angles, end_angles, duration, method, num_points)
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / 500) * 1000
        print(f"方法 [{method.ljust(7)}]: 平均耗时 {avg_time_ms:.4f} 毫秒")
        
        # 保存一份用于画图
        trajectories[method] = traj

    
    # 要求 3: 绘制位置-速度-加速度曲线 (以关节0为例)
    
    import matplotlib.pyplot as plt
    print("\n正在生成作业要求 3 的曲线图，请查看弹出的窗口...")
    
    # 提取关节0的数据的辅助函数
    def extract_joint_data(trajectory, joint_idx=0):
        t = [pt['time'] for pt in trajectory]
        p = [pt['position'][joint_idx] for pt in trajectory]
        v = [pt['velocity'][joint_idx] for pt in trajectory]
        a = [pt['acceleration'][joint_idx] for pt in trajectory]
        return t, p, v, a

    # 创建 3x1 的子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    colors = {'linear': 'red', 'cubic': 'green', 'quintic': 'blue'}
    
    for method in methods:
        t, p, v, a = extract_joint_data(trajectories[method], joint_idx=0)
        
        ax1.plot(t, p, label=method, color=colors[method], linewidth=2)
        ax2.plot(t, v, label=method, color=colors[method], linewidth=2)
        ax3.plot(t, a, label=method, color=colors[method], linewidth=2)

    # 设置图表标题和标签
    ax1.set_title('Joint 0: Position vs Time (Position Smoothness)', fontsize=14)
    ax1.set_ylabel('Position (rad)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title('Joint 0: Velocity vs Time (Velocity Smoothness)', fontsize=14)
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    ax3.set_title('Joint 0: Acceleration vs Time (Acceleration Smoothness)', fontsize=14)
    ax3.set_ylabel('Acceleration (rad/s^2)')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    
    run_basic_assignment()
    
    test_weekends()