# 6-DOF 机械臂仿真系统

## 项目简介

本项目实现了一个完整的 6 自由度（6-DOF）机械臂仿真系统，涵盖机器人学的完整技术栈：正向/逆向运动学、动力学建模、轨迹规划、多种控制策略，以及模仿学习与强化学习。系统以 UR5 机械臂的标准 DH 参数为物理模型，使用 Python 实现，并通过 Numba 加速核心计算。

## 功能特性

- ✅ **正向/逆向运动学**：基于 DH 参数的正向运动学（Numba 加速），阻尼最小二乘法（DLS）逆运动学求解
- ✅ **轨迹规划**：线性、三次样条、五次多项式三种关节空间插值，以及笛卡尔空间直线、圆、螺旋、多段平滑轨迹
- ✅ **动力学建模**：质量矩阵、科氏力矩阵、重力向量的完整拉格朗日动力学；正向/逆向动力学
- ✅ **多种控制策略**：PID（含卡尔曼滤波）、计算力矩控制（CTC）、模型预测控制（MPC）、迭代学习控制（ILC）、笛卡尔阻抗控制
- ✅ **学习算法**：模仿学习（行为克隆 BC）、强化学习（PPO）
- ✅ **3D 可视化**：基于 matplotlib 的 3D 实时机械臂显示与轨迹动画
- ✅ **GUI 控制界面**：基于 tkinter 的交互式关节控制与 IK 求解面板

## 安装说明

```bash
# 克隆仓库
git clone https://github.com/Celestina-M/6DOF-Robotic-Arm-SR.git
cd 6DOF-Robotic-Arm-SR

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**主要依赖**：NumPy、SciPy、Numba、PyTorch、matplotlib。GUI 界面需要 Tkinter 支持（macOS Homebrew Python 用户可能需额外执行 `brew install python-tk`）。

## 快速开始

```python
from robotic_system import RobotSystem
import numpy as np

# 创建系统（运动学模式）
system = RobotSystem(mode='kinematics')

# 正向运动学：设置关节角度，读取末端位置
system.arm.set_joint_angles([0.5, 0.5, -0.5, 0, 0.5, 0])
pos = system.arm.get_end_effector_position()
print(f"末端位置: {pos}")

# 逆运动学：给定目标位置求解关节角度
success, angles = system.arm.inverse_kinematics([0.3, 0.1, 0.4],
                                                 initial_guess=np.zeros(6))
print(f"IK 成功: {success}")

# 可视化
system.visualizer.draw_arm(angles)
```

动力学模式下可执行轨迹追踪：

```python
system = RobotSystem(mode='dynamics')
traj = system.plan_trajectory([0]*6, [0.5, 0.5, -0.5, 0, 0.5, 0],
                              method='quintic', duration=2.0, num_points=100)
results = system.execute_trajectory(traj, controller_name='computed_torque')
```

## 项目结构

```
6DOF-Robotic-Arm-SR/
├── src/                              # 源代码
│   ├── robotic_arm_new.py            # 运动学：FK / IK / 雅可比
│   ├── robotic_arm_dynamics.py       # 动力学 + 经典控制器(PID/CTC/阻抗)
│   ├── robotic_arm_optimization.py   # 优化控制(MPC / ILC)
│   ├── robotic_arm_imitation.py      # 模仿学习(行为克隆)
│   ├── robotic_arm_reinforcement.py  # 强化学习(PPO)
│   ├── trajectory.py                 # 轨迹规划
│   ├── visualizer.py                 # 3D 可视化
│   ├── gui_control.py                # GUI 控制界面
│   └── robotic_system.py             # 系统统一接口
├── tests/                            # 测试代码
├── docs/                             # 文档
├── examples/                         # 示例
└── README.md                         # 本文件
```

## 性能指标

| 指标 | 目标 | 实测 |
|------|------|------|
| 正向运动学吞吐 | ≥ 10,000 ops/sec | **~150,000 ops/sec** |
| 逆运动学成功率 | ≥ 80% | **100.0%**（平均误差 < 0.01 cm） |
| 计算力矩控制(CTC)追踪 RMSE | — | **0.0008 rad** |
| 模仿学习闭环跟踪 RMSE | — | **0.013 rad** |
| ILC 收敛(前馈+PD反馈) | 稳定收敛 | **~0.002 rad 量级** |
| 可视化帧率 | ≥ 30 FPS | **38.0 FPS** |

> 正向运动学吞吐经 Numba 编译加速，实测远超目标。控制器追踪精度在 UR5 参数、5 秒正弦参考轨迹下测得。逆运动学成功率在工作空间内可达目标点上测得（100 次随机测试）；可视化帧率为连续渲染 150 帧的实测均值。

## 已知说明

- **逆运动学**：采用阻尼最小二乘法，对工作空间边界或奇异位形附近的目标点可能求解失败（返回 `success=False`），属预期行为而非缺陷；建议对难解点提供合理的 `initial_guess`。
- **ILC 控制**：本动力学模型的逆/正动力学在小惯量关节上存在轻微数值不一致，纯开环 ILC 前馈在长轨迹上会漂移。系统采用"ILC 前馈学习 + PD 反馈"的标准组合以保证稳定收敛。
- **平台**：开发与测试环境为 macOS + Python 3.12。3D 可视化在 macOS 上需使用 TkAgg 后端。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- 作者：闵思睿(Sirui Min)
- 邮箱：yestmsr@gmail.com
- GitHub：https://github.com/Celestina-M