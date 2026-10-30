# 6-DOF Robotic Arm Lab

一个完整的 6 自由度（6-DOF）机械臂算法研究平台：涵盖机器人学全技术栈——正向/逆向运动学、动力学建模、轨迹规划、经典控制、鲁棒控制、优化控制与迭代学习，以及模仿学习（BC）和强化学习（PPO）。平台以 UR5 标准 DH 参数为物理模型，Python 实现，核心计算经 Numba 加速；内置传感噪声模型与动力学参数摄动模块，支持噪声/失配条件下的控制器公平对比实验。

## 功能特性

- **正向/逆向运动学**：基于 DH 参数的正向运动学（Numba 加速，~15 万次/秒），阻尼最小二乘法（DLS）逆运动学（工作空间内实测成功率 100%）
- **轨迹规划**：线性、三次样条、五次多项式关节空间插值；笛卡尔空间直线、圆、螺旋与多段平滑轨迹
- **动力学建模**：完整拉格朗日动力学（质量矩阵 / 科氏力矩阵 / 重力向量），正向/逆向动力学，含摩擦项
- **经典控制**：PID（可选内部卡尔曼滤波）、计算力矩控制（CTC）、重力补偿 PID、笛卡尔阻抗控制
- **鲁棒控制**：积分滑模控制（SMC，Slotine-Li 型滑模面 + 边界层饱和）、鲁棒计算力矩（RCTC = CTC + 切换项）；附质量/质心摄动与摩擦失配测试回路（`perturbation.py`）
- **优化控制**：模型预测控制（MPC）、迭代学习控制（ILC，前馈学习 + PD 反馈）
- **学习型控制**：模仿学习（行为克隆 BC）、强化学习（PPO，含批量更新训练管线）
- **传感与状态估计**：统一传感通道（ideal / raw / kalman），支持位置噪声、编码器量化、常值零偏；输入感知卡尔曼滤波
- **3D 可视化与 GUI**：matplotlib 实时机械臂显示与轨迹动画；tkinter 交互式关节控制与 IK 求解面板

## 安装

```bash
git clone https://github.com/Celestina-M/6-DOF-robotic-arm-lab.git
cd 6-DOF-robotic-arm-lab

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

**主要依赖**：NumPy、SciPy、Numba、PyTorch（CPU）、matplotlib、pandas。GUI 界面需要 Tkinter 支持。

## 快速开始

**注意**：`src/` 内模块采用扁平导入，运行任何脚本前请先进入 `src/` 目录（或在 `src/` 内启动 Python），否则会出现 `ModuleNotFoundError`。

```bash
cd src
python
```

```python
from robotic_system import RobotSystem
import numpy as np

# 运动学模式
system = RobotSystem(mode='kinematics')
system.arm.set_joint_angles([0.5, 0.5, -0.5, 0, 0.5, 0])
pos = system.arm.get_end_effector_position()
print(f"末端位置: {pos}")

# 逆运动学
success, angles = system.arm.inverse_kinematics([0.3, 0.1, 0.4],
                                                initial_guess=np.zeros(6))
print(f"IK 成功: {success}")

# 可视化
system.visualizer.draw_arm(angles)
```

动力学模式下的轨迹追踪：

```python
system = RobotSystem(mode='dynamics')
traj = system.plan_trajectory([0]*6, [0.5, 0.5, -0.5, 0, 0.5, 0],
                              method='quintic', duration=2.0, num_points=100)

# 名义控制器
results = system.execute_trajectory(traj, controller_name='computed_torque')
# 鲁棒控制器（滑模 / 鲁棒CTC 接口相同）
results = system.execute_trajectory(traj, controller_name='sliding_mode')

# 含噪反馈：kalman 状态估计 + 5e-3 位置噪声
system.set_sensing(mode='kalman', noise_std=5e-3, seed=0)
results = system.execute_trajectory(traj, controller_name='sliding_mode')
```

模型失配回路（鲁棒控制的标准测试工况）：

```python
import copy
from perturbation import apply_parameter_perturbation, apply_friction_mismatch

system = RobotSystem(mode='dynamics')
plant = copy.deepcopy(system.arm)                 # 被控对象
apply_parameter_perturbation(plant, rel=0.4, seed=42)   # 质量/质心 ±40%
apply_friction_mismatch(plant, scale=3.0)         # 摩擦 ×3
system.arm = plant
# 控制器与估计器仍持有名义模型（__init__ 时绑定），失配即构成扰动回路
```

## 运行测试与实验

```bash
# 测试（在仓库根目录执行，tests 内部自动处理导入路径）
python tests/test_inverse_kinematics.py   # IK 成功率（目标 ≥80%，实测 100%）
python tests/test_performance_3.py        # FK 精度与性能基准
python tests/test_sensing.py              # 传感噪声 / 卡尔曼估计（14 项）
python tests/verify_2link.py              # FK 与手写 DH 公式一致性验证
python tests/test_robust_control.py       # 失配下 SMC/RCTC vs CTC

# 实验（自动切换到仓库根目录，结果输出到 docs/，均支持断点续跑）
python src/exp1_noise_sweep.py sweep      # 实验1：传感噪声扫描 + 卡尔曼热力图
python src/exp2_all_controllers.py run    # 实验2：全控制器含噪公平比较
python src/exp2_ppo.py                    # 实验2：PPO 训练与评估
python src/exp3_distortion_decomposition.py  # 实验3：跟踪误差分解
python src/exp4_robust_sweep.py run       # 实验4：鲁棒控制失配扫描
python src/exp4_robust_sweep.py plot      # 生成 docs/exp4_robust_sweep.png
```

`docs/` 目录保存各实验的图表与 CSV 数据（见下文"实验结果"）。

## 项目结构

```
6-DOF-robotic-arm-lab/
├── src/                              # 源代码（扁平导入，脚本在 src/ 内运行）
│   ├── robotic_arm_new.py            # 运动学：FK / IK / 雅可比（Numba 加速）
│   ├── robotic_arm_dynamics.py       # 动力学 + 经典控制器(PID/CTC/阻抗/重力补偿)
│   ├── robotic_arm_robust.py         # 鲁棒控制：积分滑模 SMC / 鲁棒CTC
│   ├── robotic_arm_optimization.py   # 优化控制：MPC / ILC
│   ├── robotic_arm_imitation.py      # 模仿学习（行为克隆）
│   ├── robotic_arm_reinforcement.py  # 强化学习（PPO）
│   ├── sensing.py                    # 统一传感通道：噪声/量化/零偏/卡尔曼
│   ├── perturbation.py               # 动力学参数摄动（鲁棒性测试用）
│   ├── trajectory.py                 # 轨迹规划
│   ├── visualizer.py                 # 3D 可视化
│   ├── gui_control.py                # GUI 控制界面
│   ├── robotic_system.py             # 系统统一接口（控制器注册/执行管线）
│   ├── Main.py / demo_final.py / demo_sensing.py   # 演示脚本
│   ├── exp1_noise_sweep.py           # 实验1
│   ├── exp1_make_figures.py          # 实验1图表生成
│   ├── exp2_all_controllers.py       # 实验2：全控制器对比
│   ├── exp2_ppo.py                   # 实验2：PPO 训练/评估
│   ├── exp3_distortion_decomposition.py  # 实验3
│   └── exp4_robust_sweep.py          # 实验4：失配扫描
├── tests/                            # 测试代码（以脚本方式直接运行）
├── docs/                             # 技术文档与实验结果（图表 + CSV）
├── bc_policy.pth                     # 行为克隆训练权重
├── ppo_policy.pth                    # PPO 训练权重
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

## 实验结果（docs/）

| 图表 | 内容 |
|---|---|
| `exp1_rmse_vs_noise.png` / `exp1_noise_sweep.csv` | 噪声扫描：RMSE-噪声曲线（ideal/raw/kalman） |
| `exp1_kalman_stability_heatmap.png` | 卡尔曼参数热力图（含无输入发散的受控负面结果） |
| `sensing_modes_comparison.png` | 三种传感模式追踪对比 |
| `exp2_all_controllers.png` | 全控制器含噪对比（PID/CTC/RCTC/SMC/ILC/BC/MPC） |
| `exp2_ppo.png` / `ppo_train_log.csv` | PPO 训练曲线与评估 |
| `exp3_distortion_decomposition.png` | 跟踪误差按来源分解 |
| `exp4_robust_sweep.png` | **鲁棒性主图**：失配等级 vs 追踪 RMSE（CTC/RCTC/SMC） |

## 性能指标

| 指标 | 目标 | 实测 |
|------|------|------|
| 正向运动学吞吐 | ≥ 10,000 ops/sec | **~150,000 ops/sec**（Numba） |
| 逆运动学成功率 | ≥ 80% | **100.0%**（平均误差 < 0.01 cm，100 次随机目标） |
| CTC 追踪 RMSE（无失配） | — | **0.0008 rad**（5 s 正弦参考） |
| 模仿学习闭环跟踪 RMSE | — | **0.013 rad** |
| ILC 收敛（前馈+PD 反馈） | 稳定收敛 | **~0.002 rad 量级** |
| 鲁棒控制（40% 摄动+摩擦×3，kalman） | — | **SMC 追踪 RMSE 降低 30%，RCTC 降低 43%**（相对名义 CTC） |
| 可视化帧率 | ≥ 30 FPS | **38.0 FPS** |

## 已知说明

- **逆运动学**：阻尼最小二乘法对工作空间边界或奇异位形附近的目标可能失败（返回 `success=False`），属预期行为；难解点建议提供合理的 `initial_guess`。
- **ILC 控制**：本动力学模型的逆/正动力学在小惯量关节上存在轻微数值不一致，纯开环 ILC 前馈在长轨迹上会漂移，故采用"ILC 前馈学习 + PD 反馈"的标准组合。
- **滑模控制的反馈模式**：SMC 滑模面含速度项，raw 模式的差分速度噪声会被切换项放大，RMSE 反而劣于 kalman 模式；使用 SMC/RCTC 时建议搭配 `kalman` 反馈（`tests/test_robust_control.py` 即采用此组合）。
- **滑模增益整定**：需同时满足两个约束——滑模面符号约定（e = qd − q 时必须取 s = q̇ − q̇r，否则 −K·s 为正反馈）与离散时间稳定条件 `dt·λ_max(M̂⁻¹(K + ρ/φ)) < 2`（腕部关节惯量小，K 必须比大臂关节小约一个量级）。详见 `src/robotic_arm_robust.py` 模块注释。
- **MPC 实时性**：本实现（SLSQP + 数值雅可比）单步求解约 12~60 s，不具备实时性，作为"实现代价"维度的对照数据呈现。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

[MIT License](LICENSE)

## 联系方式

- 作者：闵思睿（Sirui Min）
- 邮箱：yestmsr@gmail.com
- GitHub：https://github.com/Celestina-M
