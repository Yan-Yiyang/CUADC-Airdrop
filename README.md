# CUADC 固定翼侦查与空投控制

**版本 0.2.1**。项目以 Python 实现遥测与图像对齐、YOLO/OCR 识别、目标定位和统计、航线规划、弹道预测与任务编排，支持真实环境及 PX4 SITL。

任务流程：起飞前预检与模型预热 → 等待起飞 → 侦查 → 排空识别与选靶 → 飞掠投放 → 按配置返航降落。真实任务默认由操作手通过 QGroundControl 上传并启动侦查；阶段开关决定采集、识别、模拟或真实释放及后续降落行为。

## 仓库结构

| 路径 | 职责 |
| --- | --- |
| `airdrop/` | 运行时、遥测、视频、感知、坐标、目标统计、弹道、任务与记录 |
| `tools/` | 模型文件整理、弹道拟合、文档检查和性能诊断 |
| `tests/` | 按资源需求分层的回归测试 |
| `docs/` | 使用手册、数据契约和专题说明 |
| `vendor/airdrop-sim/` | 独立仿真世界与飞机工具 |
| `vendor/airdrop-calibration/` | 独立离线相机标定工具 |
| `routes/`、`models/` | 操作手本地航线与模型文件；航线及权重不入库 |

## 项目背景与技术选型

本项目面向 CUADC 固定翼无人机侦察与打击项目。由于当前飞机设计和项目资源限制暂无法使用机载计算机，因此采用 PX4 负责自主飞行、MAVSDK 负责地面端任务交互与状态监测的架构。技术选型主要考虑通信带宽、赛场电磁环境、开发复杂度及后续扩展需求。

### 为什么暂不使用 ROS / ROS 2？

当前尚无机载计算机，暂不具备直接部署机载 ROS 节点的硬件条件。此外，若通过无线数传传输 ROS 2 话题，还需要考虑 DDS/RTPS 等通信协议带来的额外开销。

假设通过数传传输姿态（20 Hz）、位置与速度（10 Hz）、空速（10 Hz）、NED 原点 GPS（1 Hz）及任务状态（2 Hz），估算如下：

| 通信方式 | 估算带宽 |
| --- | ---: |
| 应用数据本身 | 约 4.6 KB/s |
| MAVLink 2 | 约 5.1 KB/s |
| ROS 2 / DDS / RTPS over UDP/IPv4 | 约 7.9 KB/s |

以上为量级估算，实际带宽取决于消息格式、序列化方式、协议封装、数据打包及通信配置，尚未计入全部控制交互和额外协议流量。参考：[MAVLink 帧格式](https://mavlink.io/en/guide/serialization.html)、[DDSI-RTPS 规范](https://www.omg.org/spec/DDSI-RTPS/2.5/PDF)。

现有常见数传设备的规格显示，其通讯速率一般在2.4KB/s~20KB/s，且一般**通信速率低于 8 KB/s 时采用 LoRa 模式，超过 8 KB/s 时采用 FHSS 高速模式**。ROS 2 估算所需的 7.9 KB/s 已接近 LoRa 模式的带宽边界，几乎没有余量供其他地面站操作、任务上传及额外遥测使用；而 FHSS 高速模式的抗干扰能力相对较弱，不利于应对赛场复杂的电磁环境。

因此，当前优先采用通信开销较小的 MAVLink 进行必要的遥测与任务交互，视频通过独立图传设备回传，对于不可避免的数据不同步，在地面设计算法对齐。

### 为什么使用 Mission，而不以 Offboard 作为主要控制模式？

MAVSDK 用于地面端与 PX4 交互，并不意味着必须使用 Offboard 模式。Offboard 模式对通信带宽和通信质量的要求同样较高，受限于上文所述带宽限制，本项目优先采用 PX4 Mission 执行预先规划的侦查航线，只能在比赛中生成的投弹航线根据计算结果在地面生成并一次性上发，减少对地面端持续发送控制设定值及稳定通信链路的依赖。

### 为什么选择 PX4，而不是 ArduPilot？

两者均能满足固定翼自主飞行需求。选择 PX4 主要是因为其模块化架构便于梳理系统职责、开展开发与调试，并可通过 uXRCE-DDS 等接口与 ROS 2 集成，为未来增加机载计算机及扩展机载自主决策能力预留技术基础。

### 为什么暂不使用云台？

固定向下安装的摄像头能够满足当前图像采集需求。云台可以调整相机视轴，在一定程度上降低坐标反解的复杂度，但普通舵机云台的指向精度有限，高性能伺服云台的成本又较高。是否采用仍需综合权衡定位收益、控制精度、成本与备赛进度。

## 开始使用

要求 Python ≥3.14，使用 uv 管理依赖。模型权重、相机标定和实际任务航线由使用者提供。

```bash
git clone --recurse-submodules https://github.com/Yan-Yiyang/CUADC-Airdrop.git
cd CUADC-Airdrop
uv sync
uv run python -m airdrop.run --help
uv run python -m airdrop.run mission --help
```

已有满足依赖的 Python 环境时，可在仓库根目录直接运行 `python -m airdrop.run --help`，无需重复安装。

- 运行配置、阶段开关和生命周期见 [运行说明](docs/runtime.md)。
- 设备配置、模型与标定输入、接口、文件格式和排障见 [项目手册](docs/handbook.md)。
- 子模块初始化与独立运行见 [工具仓库](docs/tool_repositories.md)。
- QGC 视频设置见 [视频说明](docs/qgc_video.md)，性能比较方法见 [性能诊断](docs/performance.md)。
- AI 执行约束和按需阅读索引见 [AGENTS.md](AGENTS.md)。

开发流程见[开发与发布](docs/development.md)，待完成的工程验证见[验证计划](docs/validation.md)。

## 验证与边界

```bash
uv run python -m pytest --test-tier fast
uv run python -m pytest --test-tier offline
uv run python -m airdrop.run check-docs
```

测试层级与平台用例见 [手册](docs/handbook.md)。离线合成回归、历史素材回放和仿真各自覆盖不同范围；它们不能证明真实摄像头定位精度、现场通信时延或物理挂架释放成功。模型和弹道参数需要按实际设备、载荷与场地标定。固定历史航迹不会响应回放中生成的新航线。
