# AGENTS.md — airdrop 仓库 AI 智能体指南

本文件面向在本仓库工作的 AI 智能体，提供执行约束、代码入口和按需阅读索引。先确认用户当前任务与工作区已有改动，再读取相关文档和源码；不要一次加载全部资料。用户指令优先；实现事实以当前代码为准。

## 项目与目录

`airdrop` 是 Python 包。命令行入口为 `airdrop/run.py`；配置集中在 `airdrop/config.py`。
项目要求 Python 3.14 或更高版本；依赖通过 `uv` 管理，不使用 `pip`。
主仓命令行参数解析集中在 `airdrop/run.py`；独立工具参数由各子仓 CLI 维护；`airdrop/runtime/` 与 `tools/` 是库模块，不自行实现命令行解析。重依赖应在需要时惰性导入。

| 路径 | 用途 |
| --- | --- |
| `airdrop/` | 项目代码，按遥测、视频、感知、坐标、目标统计、弹道、任务、记录等模块组织 |
| `tools/` | 开发、校准、检查与数据处理工具 |
| `airdrop/runtime/` | real/SITL 任务装配、阶段参数与统一生命周期 |
| `tests/` | pytest 测试 |
| `docs/` | 用户手册与按需查阅的专题文档 |
| `routes/`、`models/` | 航线与本地模型数据 |
| `vendor/` | 可选仿真、标定工具 Git 子模块；各自有 CLI、测试和执行指南 |
| `PX4-Autopilot-1.17.0/`、`qgroundcontrol-master/` | 仅供参考的第三方源码 |

## 开发约定

- 保留用户已有改动；只编辑任务所需文件，避免无关重构。
- 项目处于初始开发阶段，不保留旧接口、旧记录格式或历史输入的向前兼容层；字段缺失应明确失败或跳过，不能用猜测值静默降级。
- 项目代码、Python 工具与测试、QGC 在 Windows 运行；WSL 仅用于 PX4/Gazebo 仿真及仿真环境/飞机配置，不将遥测客户端、识别/OCR 或回放迁入 WSL。
- 代码注释和 docstring 使用中文；类型标注使用现代写法，如 `X | None` 与内置泛型。
- 文本文件使用 LF。提交时一个逻辑改动一个提交，并使用 conventional 前缀和中文摘要。
- 安装工具或依赖前先取得用户同意；检查环境时优先使用仓库已有的 `.venv`。
- 不修改 PX4 与 QGroundControl 参考源码；不修改生成产物，除非任务明确要求重生成。
- 本机资料和恢复备份放在忽略的 `.local/`，不提交 Git。远端历史重写需明确授权。
- 阅读文档按需进行：先查本文件索引，再读任务涉及的手册章节、专题笔记和实际代码；不要求预先通读所有文档。

## 文档索引

- [`docs/runtime.md`](docs/runtime.md)：统一任务入口、阶段开关与资源生命周期。
- [`docs/qgc_video.md`](docs/qgc_video.md)：QGC 合成视频、只读显示状态与编码发布。
- [`docs/performance.md`](docs/performance.md)：性能计时、线程配置与 RTP 排障。
- [`docs/tool_repositories.md`](docs/tool_repositories.md)：子模块初始化、缺失处理、独立验证与固定版本升级。

- [`docs/handbook.md`](docs/handbook.md)：模块入口、参数、文件格式与开发检查信息。
- [`docs/calibration_opencv.md`](docs/calibration_opencv.md)：标定与 OpenCV 相关记录。
- [`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md)：视频接收相关记录。
- [`docs/perception_ocr.md`](docs/perception_ocr.md)：感知与 OCR 相关记录。
- [`docs/ballistics_fit.md`](docs/ballistics_fit.md)：数据记录与拟合相关信息。
- [`docs/simulation_world.md`](docs/simulation_world.md)、[`vendor/airdrop-sim/README.md`](vendor/airdrop-sim/README.md)：仿真资料。
- [`docs/tool_contracts.md`](docs/tool_contracts.md)：标定输入 DTO、相机文件及仿真 manifest 契约。

专题文档按任务需要阅读；它们不构成额外的通用开发规则。业务行为和接口以源码、配置及测试为准。

- [`docs/development.md`](docs/development.md)：开发、提交与发布流程。
- [`docs/validation.md`](docs/validation.md)：待完成的通用工程验证。

## 本地检查

Windows 下使用仓库虚拟环境中的工具：

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

默认 tier 是 `fast`，只运行未标记的快速回归，不启动 ffmpeg、WSL、GPU 或飞控。`offline` 加入 `integration`，`platform` 再加入 `stream`、`network` 与 `wsl`，`all` 还包含 `realdata` 与 `sitl`。marker 与 tier 的映射在 `tests/conftest.py` 中维护。pytest 原生 `-m` 是额外过滤，不会改变 tier；例如 `pytest --test-tier all -m realdata`。覆盖率默认关闭；需要完整分支覆盖时显式传入命令中的 `--cov` 参数。真实素材、仿真、网络或本地多媒体测试的范围受运行环境限制，pytest 结果不代表真实设备或外部素材已验证。

在受限环境中，临时目录、进程与 socket 权限可能影响测试；以现场错误为准，沿用各测试已有 fixture，并区分环境限制与代码缺陷。

`check-docs` 核验手册与代码引用、参数和产物名称的一致性；它不替代 pytest、ruff 或 pyright。
按改动影响范围运行检查，并如实说明未运行的环境专用用例。
