# AGENTS.md — airdrop 仓库 AI 智能体指南

本文件面向在本仓库工作的 AI 智能体，提供项目结构、常用命令、架构边界、写作规范与按需文档索引。动手前先确认用户任务与工作区现有改动；用户指令优先，实现事实以当前源码为准，按需阅读文档而非一次加载全部资料。

## 项目概览

- **项目与工具链**：`airdrop` 为 Python 包，要求 Python >= 3.14；依赖统一通过 `uv` 管理。
- **核心入口**：主仓命令行参数解析集中在 `airdrop/run.py`，全局配置集中在 `airdrop/config.py`；独立工具命令行由各子仓维护。

## 目录结构

| 路径 | 用途与边界 |
| --- | --- |
| `airdrop/` | 主包源码（遥测、视频、感知、坐标、目标统计、弹道、任务、记录等） |
| `airdrop/runtime/` | 实飞与仿真任务装配、阶段参数与生命周期库（不自行实现命令行解析） |
| `tools/` | 开发、标定、检查与数据处理库模块（不自行实现命令行解析） |
| `tests/` | pytest 测试套件 |
| `docs/` | 用户手册与按需查阅的专题文档 |
| `routes/`、`models/` | 航线与本地模型数据 |
| `vendor/` | 仿真、标定工具 Git 子模块（各自维护独立命令行、测试与指南） |
| `PX4-Autopilot-1.17.0/`、`qgroundcontrol-master/` | 第三方只读参考源码（严禁修改） |

## 常用检查命令

Windows 下优先使用仓库已有 `.venv`：

```powershell
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/ruff.exe format --check .
./.venv/Scripts/pyright.exe
./.venv/Scripts/python.exe -m pytest --test-tier fast
./.venv/Scripts/python.exe -m pytest --test-tier offline
./.venv/Scripts/python.exe -m pytest --test-tier platform
./.venv/Scripts/python.exe -m pytest --test-tier all --cov=airdrop --cov-branch --cov-report=term-missing:skip-covered
./.venv/Scripts/python.exe -m airdrop.run check-docs
```

| 测试层级 | 包含范围 | 说明 |
| --- | --- | --- |
| `fast`（默认） | 未标记快速回归 | 不启动 ffmpeg、WSL、GPU 或飞控 |
| `offline` | `fast` + `integration` | 含本地文件与离线跨模块测试 |
| `platform` | `offline` + `stream` / `network` / `wsl` | 依赖本机多媒体、网络或 WSL |
| `all` | `platform` + `realdata` / `sitl` | 需外部真实素材或完整仿真环境 |

- 标记与测试层级的映射在 `tests/conftest.py` 维护；pytest 原生 `-m` 仅作额外过滤，不会扩大层级。覆盖率默认关闭，按需显式传入 `--cov` 参数。
- `check-docs` 核验手册与代码引用、参数和产物名称一致性，不替代 ruff、pyright 或 pytest。
- 按改动影响范围运行检查，并如实说明未运行的环境专用用例或受限环境限制（如临时目录、进程或 socket 权限），区分环境限制与代码缺陷。

## 架构与运行不变量

- **不保留兼容层**：项目处于初始开发阶段，不保留旧接口、旧记录格式或历史输入的向前兼容层；缺失字段应明确失败或跳过，不得用猜测值静默降级。
- **重依赖惰性导入**：视觉、推理、仿真或多媒体等重依赖应在需要时惰性导入，保持轻量入口与离线测试可用。
- **在线与回放同构**：在线任务与正式回放共用缓存、预热与感知装配；回放恢复录制配置后仅应用显式覆盖。涉及缓存或感知改动须双向验证，离线回放不代表完整飞行验收。
- **平台分工**：主仓代码、Python 工具、测试与 QGC 在 Windows 运行；WSL 仅用于 PX4/Gazebo 仿真及仿真环境/机型配置，不将遥测客户端、识别/OCR 或回放迁入 WSL。

## 编码、文档与提交规范

- **语言与格式**：所有文档、代码注释、docstring 与提交摘要等文本内容统一使用中文（代码标识符、API 名称与命令除外）；类型标注使用现代写法（`X | None`、内置泛型）；文本文件统一使用 LF 换行。
- **常规工程注释与文档**：遵循常规软件工程开发规范（如 PEP 257）。注释、docstring 与项目文档应简洁客观地说明**当前设计意图、关键不变量、接口契约、参数语义与失败边界**；不写会话式叙事、历史排错流水账、临时计划或阶段代号，也不复述显而易见的代码逻辑。
- **禁止出现本机文件与本机环境信息**：代码注释、docstring、项目文档及 Git 提交信息中，**不得包含**本机绝对路径或个人目录、私有/忽略文件或临时脚本名称、本机特定录像批次或数据集名称、特定主机硬件/桌面环境参数（如具体型号、本机显存或内存占用、本机实测跑分），以及单机一次性运行结果。示例统一使用占位路径，设备或性能约束写成可复现前提与验证边界。
- **Git 提交**：一个逻辑改动一个提交，使用约定式提交前缀与简洁中文摘要（如 `feat(scope): ...`、`fix(scope): ...`、`perf(scope): ...`）。提交信息只描述**改动内容与工程原因**，不粘贴单机测试通过条数、一次性回放统计或环境相关验证段落。

## 操作边界

- **始终遵守**：保留用户已有改动；只编辑任务所需文件，避免无关重构；先查索引再按需阅读文档。
- **先征求用户同意**：安装新工具或依赖（仅使用 `uv`）；重生成生成产物；重写远端 Git 历史。
- **严禁执行**：使用 `pip`；修改 `PX4-Autopilot-1.17.0/` 或 `qgroundcontrol-master/`；将 `.local/`、本机资料或恢复备份提交至 Git；在 `airdrop/runtime/` 或 `tools/` 中自行实现命令行参数解析。

## 文档索引（按需阅读）

专题文档仅作按需参考，不构成额外全局开发规则；业务行为与接口以源码、配置及测试为准。

| 涉及主题 | 参考文档 |
| --- | --- |
| 模块入口、参数、文件格式与检查说明 | [`docs/handbook.md`](docs/handbook.md) |
| 统一任务入口、阶段开关与生命周期 | [`docs/runtime.md`](docs/runtime.md) |
| QGC 合成视频、只读显示状态与编码发布 | [`docs/qgc_video.md`](docs/qgc_video.md) |
| 性能计时、线程配置与 RTP 排障 | [`docs/performance.md`](docs/performance.md) |
| 标定与 OpenCV、输入契约 | [`docs/calibration_opencv.md`](docs/calibration_opencv.md)、[`docs/tool_contracts.md`](docs/tool_contracts.md) |
| 视频接收（HM30 / FFmpeg） | [`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md) |
| 感知与 OCR | [`docs/perception_ocr.md`](docs/perception_ocr.md) |
| 弹道数据记录与拟合 | [`docs/ballistics_fit.md`](docs/ballistics_fit.md) |
| 仿真环境与子模块管理 | [`docs/simulation_world.md`](docs/simulation_world.md)、[`vendor/airdrop-sim/README.md`](vendor/airdrop-sim/README.md)、[`docs/tool_repositories.md`](docs/tool_repositories.md) |
| 开发、提交、发布与待完成验证 | [`docs/development.md`](docs/development.md)、[`docs/validation.md`](docs/validation.md) |
