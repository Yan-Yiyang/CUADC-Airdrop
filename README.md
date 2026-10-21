# CUADC 固定翼侦查与空投控制

**开发版本 3.0.0**。项目以 Python 实现遥测与图像对齐、YOLO/OCR 识别、目标定位和统计、航线规划、弹道预测与任务编排，支持真实环境及 PX4 SITL。

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

## 验证与边界

```bash
uv run python -m pytest --test-tier fast
uv run python -m pytest --test-tier offline
uv run python -m airdrop.run check-docs
```

测试层级与平台用例见 [手册](docs/handbook.md)。离线合成回归、历史素材回放和仿真各自覆盖不同范围；它们不能证明真实摄像头定位精度、现场通信时延或物理挂架释放成功。模型和弹道参数需要按实际设备、载荷与场地标定。固定历史航迹不会响应回放中生成的新航线。
