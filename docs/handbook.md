# CUADC固定翼无人机侦查与打击控制项目：项目手册（用户 / 代码审查者 / 维护者）

> 版本：`airdrop.__version__` = **0.1**。
> 本文所有**签名与默认值**来自机器自省（`python -m airdrop.run dump-api` 产出 `docs/api_reference.md`），
> **不是手抄**；字段语义、失败语义与流程说明由人工逐文件核对源码后写出，
> 关键处以 `文件:行号` 或 `模块.函数` 标注。
> 文中所有 `airdrop.*` 形式的引用都经过核验：能真正 `import` + `getattr` 解析到。
> §3 的示例由 `tests/test_handbook.py` **逐条真跑**（11 条），文档一致性由
> `python -m airdrop.run check-docs` 把关（参数齐全 / 引用可解析 / 产物文件名齐全 / 无编造参数名）。

## 0. 这份文档怎么用

### 0.1 三条阅读路径

| 你是 | 从哪读起 | 目的 |
| --- | --- | --- |
| **用户**（要飞一次任务 / 跑一次投放试验） | [§5 输入与输出](#5-输入与输出) → [§6 使用流程](#6-使用流程) → [§8 常见问题与排障](#8-常见问题与排障) | 知道要准备什么、敲哪条命令、产物在哪、出错看什么 |
| **代码审查者**（评估这份实现能不能用） | [§7 代码审查速查](#7-代码审查速查) → [§2 数据流与模块依赖](#2-数据流与模块依赖) → [§3 模块逐个说明](#3-模块逐个说明) | 找不变量、失败语义、注入点、边界条件 |
| **维护者**（改代码 / 加功能） | [§3 模块逐个说明](#3-模块逐个说明) → [§4 参数总表](#4-参数总表) → [§9 已知边界与未验证项](#9-已知边界与未验证项) | 知道改哪一处、哪些参数会影响什么、哪些结论没验证过 |

### 0.2 与其它文档的分工（冲突时以谁为准）

| 文档 | 讲什么 | 权威性 |
| --- | --- | --- |
| `README.md` | 怎么用：安装、四条入口、配置速览、安全提示 | 使用口径 |
| **本文档** `docs/handbook.md` | 全景手册：目录、数据流、逐模块调用方法、参数总表、输入输出、流程、审查清单、排障 | 使用 + 审查口径 |
| `AGENTS.md` | 给 AI 智能体的工作手册：环境约定、架构要点、关键易错点索引、验证方式 | 约定口径 |
| `docs/calibration_opencv.md` | 三步标定与 OpenCV 5.0 问题与易错点全记录（实测表、方程、判据） | **实测口径** |
| `docs/video_hm30_ffmpeg.md` | 图传链路为什么只有 ffmpeg 一个后端 + 逐项实测 | **实测口径** |
| `docs/perception_ocr.md` | 检测/OCR 细节：转正形态门限、Cls 权重命名、onnxruntime 前提 | **实测口径** |
| `docs/ballistics_fit.md` | 投放记录与反演：可辨识性判据、σ、留一验证 | **实测口径** |

**冲突时的优先级**：代码与 `docs/` 下的实测记录 > 本文档 > `AGENTS.md`/`README.md` 的概述性描述。
本文档刻意**不复制**专题笔记里的长篇实测结论，只给一行指针——要动那条链路前先读笔记。

### 0.3 一分钟了解这个项目

固定翼无人机（PX4 + MAVSDK）"先侦查后空投"：

```
起飞 → 预设侦查航线（约 1 分钟）→ 盘旋 hold
     → 感知（YOLO 检测 + OCR 读编号）→ 像素→NED 坐标解算 → 目标统计（聚类 + 选唯一）
     ├─ 有结果 → 飞掠航点（航向/高度/段长按配置）+ 降落航线，合并上传
     └─ 无结果 → 以备用点为目标生成同样航点
     → 飞掠段：弹道实时预测落点，距目标 ≤R 触发 gripper 投放；飞过目标未投则强制投放
     → 沿降落航线返航降落
全程：飞行目录（文本日志/遥测/检测/事件/投放/视频帧）写入磁盘，飞后可回放迭代
投放后：量出实际落点 → 反演弹道参数（质量是称出来的，反演识别的是 κ=Cd·A/m）
```

---

## 1. 仓库目录总览

### 1.1 顶层布局

| 路径 | 用途 | 关键文件 | 进 git？ |
| --- | --- | --- | --- |
| `airdrop/` | 唯一的运行时代码包（9 个子包，见 §3） | `__init__.py`（171 个公开名）、`config.py`（全部参数） | 是 |
| `airdrop/config.py` | **全部参数的唯一集中处**（frozen dataclass + 取值域校验） | `Config`、`Config.validated` | 是 |
| `tools/` | 一次性离线工具（**纯库模块**：文件内常量 = 默认值，暴露 `build_config(**覆盖)` / `main(**kwargs)`；命令行解析归 §6.2 的 `airdrop/run.py`） | `fetch_models.py`、`calibrate.py`、`fit_ballistics.py`、`make_world.py`、`dump_api.py`、`check_docs.py` | 是 |
| `examples/` | 七个入口示例（同样是纯库模块：常量 = 默认值；重依赖只在函数体内导入，所以 `import` 它们**不加载** cv2/mavsdk/torch） | `full_mission.py`、`replay_flight.py`、`sitl_mission.py` | 是 |
| `airdrop/run.py` | **唯一的命令行入口**：argparse 子命令 → 关键字覆盖到各入口的 `build_config`/`main`（见 §6.2） | `run.py`、`airdrop/_lazy.py`（PEP 562 惰性导出） | 是 |
| `tests/` | pytest 套件（20 个测试文件，全部离线；`realdata` 标记默认跳过） | `conftest.py`、`test_e2e.py`、`test_fit.py`、`test_plan.py`、`test_world.py` | 是 |
| `docs/` | 专题笔记（5 篇实测记录）+ 本手册 | `handbook.md`、`simulation_world.md`、`calibration_opencv.md`、`video_hm30_ffmpeg.md`、`perception_ocr.md`、`ballistics_fit.md` | 是 |
| `sim/` | **仿真模块**：世界（生成产物：两个 `.sdf` + 网格/贴图）+ 带下视相机的机型 + PX4 airframe + 装机/启动脚本；生成器是 `tools/make_world.py`，布局与启动见 [`simulation_world.md`](simulation_world.md)，模块说明见 [`../sim/README.md`](../sim/README.md) | `sim/worlds/cuadc/cuadc_recon_strike_r1.sdf`、`r2.sdf` | 是 |
| `routes/` | **操作手在 QGC 里画好的 `.plan` 航线**（`recon.plan` / `land.plan`），由配置按需导入 | `land.plan` | 是（航线是飞行输入，跟着版本走） |
| `models/` | 本地模型权重与字典（**字典入库、权重不入库**） | `best2.pt`、`ppocr/*.pth`、`ppocr/ppocrv6_dict.txt` | 字典是；权重否 |
| `.venv/` | 项目虚拟环境（uv 创建，Python 3.14.6，**里面没有 pip**） | — | 否 |
| `PX4-Autopilot-1.17.0/` | PX4 固件源码，**仅供查阅，禁止当项目代码修改** | — | 否 |
| `qgroundcontrol-master/` | QGC 地面站源码，同上（`MissionManager/LandingComplexItem.cc` 是 `.plan` 复杂项展开的权威依据） | — | 否 |
| `flights/` | **运行期产物**：`record.dir` 默认值，每次飞行一个子目录 | 见 §5.1 | 否（已 ignore） |
| `log/` | 运行期产物：`perception.crop_dir` 默认 `log/target`（裁剪排故图） | — | 否（已 ignore） |
| `.e2e-test-tmp/`、`.fit-test-tmp/`、`.calibrate-test-tmp/`、`.replay-test-tmp/`、`.pytest-tmp/`、`.plan-test-tmp/`、`.handbook-test-tmp/`、`.world-test-tmp/` | 测试用的**工作区内**临时目录（见 §5.6） | — | 否（已 ignore） |
| `.sitl-test-tmp/` | SITL 演练的临时产物（演练日志、机上任务备份、ulog 探查脚本） | — | 否（已 ignore） |
| `.idea/`、`.vscode/`、`.workbuddy/` | 编辑器/工具配置 | — | 否（已 ignore） |

### 1.2 `airdrop/` 包内布局

| 子包 / 模块 | 一句话职责 | 关键类型 |
| --- | --- | --- |
| `airdrop/telemetry/` | 遥测与控制：MAVSDK 工作线程、线程安全代理、任务级控制器 | `MavsdkThread`、`TelemetryBroker`、`TelemetrySnapshot`、`DroneController`、`MissionController`、`NedOrigin` |
| `airdrop/video/` | 图传接收：ffmpeg 拉流、帧-遥测时间对齐、环形缓冲 | `Hm30VideoSource`、`FrameTelemetryAligner`、`AlignedSample`、`AlignmentBuffer`、`AlignmentWriter`、`BufferedFrame` |
| `airdrop/record/` | 记录与回放：飞行目录写入磁盘、按原时间轴重放 | `FlightRecorder`、`EventLog`、`DropWriter`、`DetectionWriter`、`ReplayVideoSource`、`TelemetryPacer`、`FlightLog` |
| `airdrop/perception/` | 视频处理：YOLO 检测 +（OCR 读编号 \| 12 类直出） | `Detector`、`DetectionBatch`、`OpenCvPostProcess`、`OcrEngine`、`OcrWorkerPool`、`PerceptionWorker`、`Detection` |
| `airdrop/georef/` | 坐标处理：像素 → NED（视线与地面求交）→ WGS84 | `CameraModel`、`pixel_to_ned`、`cross_check_by_side`、`LLARef`、`wgs84_to_ned` |
| `airdrop/targeting/` | 目标统计：DBSCAN 聚类 + 按编号选唯一结果 | `TargetPoint`、`Cluster`、`TargetingResult`、`analyze` |
| `airdrop/ballistics/` | 弹道与投放：落点预测、投放判据、投放记录与参数反演 | `BallisticsModel`、`ReleaseJudge`、`DropRecord`、`fit_ballistics` |
| `airdrop/mission/` | 任务编排：状态机、任务项、QGC 航线解析、航线规划、坐标解算、主循环 | `MissionState`、`MissionRunner`、`MissionItem`、`load_plan`、`check_fixed_wing_landing`、`build_drop_mission`、`TargetTracker` |
| `airdrop/preflight.py` | 起飞前自检：载入模型（detector/ocr/camera）→ 视频自检（各项可关） | `Preflight`、`PreflightCheck`、`PreflightError`、`PreflightLike`、`PreflightConfig` |
| `airdrop/mission/items.py` | **任务项的唯一表示**（MAVLink 级：command/frame/params/位置）与 `MAV_CMD_*`/`MAV_FRAME_*` 常量 | `MissionItem`、`UNSET`、`command_name` |
| `airdrop/mission/plan_file.py` | QGC `.plan` 解析（含复杂项展开）+ 固定翼降落预检（纯 JSON，离线） | `load_plan`、`QgcPlan`、`PlanError`、`check_fixed_wing_landing` |
| `airdrop/config.py` | 全部参数（15 个 frozen dataclass） | `Config` 及 14 个片段 |

### 1.3 什么进 git、什么不进（实测口径）

用 `git check-ignore` 验过，`.gitignore` 的关键规则：

| 规则 | 覆盖的东西 |
| --- | --- |
| `models/**/*.pt` / `*.pth` / `*.onnx` | YOLO 与 PP-OCR 权重（约 177 MB）**不进库**；`models/**/.gitkeep` 保留 |
| `!models/**/.gitkeep` + 字典例外 | `models/ppocr/ppocrv6_dict.txt`、`ppocrv6_tiny_dict.txt`（约 100 KB）**进库**——它们决定 rec 的字符集，必须随版本走 |
| `log/`、`_cropdiag/`、`_e2e/`、`*.log` | 排故产物与文本日志 |
| `flights/` | **飞行记录产物**（`FlightRecorder` 的输出目录，一次飞行几十~几百 MB） |
| `.coverage*`、`coverage.xml`、`htmlcov/` | 覆盖率产物 |
| `.venv/`、`__pycache__/`、`*.py[cod]`、`.pytest_cache/` 等 | 环境与缓存 |
| `.replay-test-tmp/`、`.calibrate-test-tmp/`、`.e2e-test-tmp/`、`.fit-test-tmp/`、`.plan-test-tmp/`、`.handbook-test-tmp/`、`.world-test-tmp/`、`.pytest-tmp/`、`.sitl-test-tmp/`、`pytest-cache-files-*/` | 测试与演练临时目录（§5.6） |
| `PX4-Autopilot-1.17.0/`、`qgroundcontrol-master/`、`px4.tar.gz`、`qgc.tar.gz` | 参考源码与压缩包 |

⚠ **一处没有被 ignore、需要人工注意**：

* **`camera_calib.json`（标定产物）未被 ignore**——它是要跟着任务走的配置产物，
  是否入库由使用方决定（本仓库默认不入库、也不忽略）。

`flights/` 曾长期未被 ignore（`git status` 会列出整个飞行目录），现在已进 `.gitignore`；
`routes/` 相反是**要入库**的：`.plan` 是飞行输入，审查时要能看到"这次飞的是哪条航线"。

### 1.4 `models/` 目录实际内容（本地当前状态）

| 文件 | 大小 | 进 git | 用途 |
| --- | --- | --- | --- |
| `models/.gitkeep` | 0 | 是 | 占位 |
| `models/best2.pt` | 5.95 MB | 否 | **唯一可用的 YOLO 权重**（单类 `target`）；`best.pt` / `best1.pt` 已废弃（零检出） |
| `models/ppocr/PP-OCRv6_det_medium.pth` | 60.6 MB | 否 | OCR 检测模型（默认） |
| `models/ppocr/PP-OCRv6_rec_medium.pth` | 73.4 MB | 否 | OCR 识别模型（默认） |
| `models/ppocr/PP-OCRv6_{det,rec}_{small,tiny}.pth` | 1.9~20 MB | 否 | 备选尺度（`OcrEngineConfig` 可切） |
| `models/ppocr/ch_ptocr_mobile_v2.0_cls_mobile.pth` | 0.56 MB | 否 | 方向分类 TORCH 版（**文件名不许规范化**，见笔记） |
| `models/ppocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 0.56 MB | 否 | 方向分类 ONNX 版（默认用它，最快） |
| `models/ppocr/ppocrv6_dict.txt` | 70 KB | **是** | rec 字符集字典（固定识别字符集） |
| `models/ppocr/ppocrv6_tiny_dict.txt` | 30 KB | **是** | tiny 模型字典 |

取权重：`python -m airdrop.run fetch-models`（**只复制、不联网**；源路径是文件内常量，可由 `--source-dir` 覆盖）。
> Cls 权重命名与 onnxruntime 的 CUDA 前提见 [docs/perception_ocr.md](perception_ocr.md)。

---

## 2. 数据流与模块依赖

### 2.1 端到端流水线（实飞与回放共用）

```
 ┌──────────────┐        ┌──────────────┐
 │ SIYI 相机     │  RTSP  │ HM30 地面端   │  透明桥接 192.168.144.0/24
 │ 192.168.144.25│──────▶│ (以太网桥)    │
 └──────────────┘        └──────┬───────┘
                                │ rtsp://192.168.144.25:8554/main.264
                    ┌───────────▼─────────────────────────────┐
                    │ airdrop.video.source.Hm30VideoSource     │  1 个采集线程 + ffmpeg 子进程
                    │  · add_sink(逐帧，一帧不落)  · read()/latest()(允许丢帧)
                    └───────────┬─────────────────────────────┘
                                │ VideoFrame(index, image, timestamp, lag)
                    ┌───────────▼─────────────────────────────┐
                    │ airdrop.video.align.FrameTelemetryAligner│  拍摄时刻 = timestamp − lag
                    │  查 broker 历史（interpolate/nearest）    │
                    └───────────┬─────────────────────────────┘
                                │ AlignedSample(frame, snapshot, timestamp, lag, mode)
                    ┌───────────▼─────────────────────────────┐
                    │ airdrop.video.buffer.AlignmentBuffer     │  环形缓冲（jpeg/raw，默认 5400 帧）
                    │  AlignmentWriter 是标准 sink 接法         │  写者 1、读者 N（各自游标）
                    └───┬───────────────────────┬─────────────┘
      wait_new(逐帧)    │                       │ wait_new(逐帧)
        ┌───────────────▼──────────┐   ┌────────▼──────────────────────────┐
        │ perception.PerceptionWorker│  │ record.FlightRecorder              │
        │  YOLO 全帧检测             │  │  frames/%06d.jpg + frames_index.jsonl│
        │  ocr: 裁剪→去重→OCR 进程池 │  │  telemetry.jsonl（10Hz 节流）        │
        │  cls12: 类别直出编号       │  │  events.jsonl / drops.jsonl / flight.log│
        └───────────────┬───────────┘   └────────────────────────────────────┘
                        │ Detection(frame_index, capture_timestamp, pixel, code, telemetry)
        ┌───────────────▼───────────┐
        │ mission.TargetTracker      │  georef.pixel_to_ned（拍摄时刻遥测）
        │  + 边长互校（默认关）       │  + 去畸变只做一次（按 Detection.extra 判断）
        └───────────────┬───────────┘
                        │ TargetPoint(north_m, east_m, capture_timestamp, code, confidence)
        ┌───────────────▼───────────┐
        │ targeting.analyze          │  DBSCAN 聚类 → 类内编号众数 → 跨类 median/max 选唯一
        └───────────────┬───────────┘
                        │ TargetingResult(selected, clusters, noise, rejected)
        ┌───────────────▼───────────────────────────────────────────────┐
        │ mission.MissionRunner（INIT→PREFLIGHT→WAIT_AIRBORNE→RECON→…→DONE）│
        │  · mission.planner: 侦查航线 / 飞掠 [entry,exit] + 降落航线合并上传 │
        │  · telemetry.controller.DroneController: 上传/启动/hold/rtl/投放   │
        │  · ballistics.ReleaseJudge: 每拍预测落点 ≤R 即投；越过目标则强制投  │
        │  · on_drop → record.DropWriter: drops.jsonl（投放瞬间状态）        │
        └───────────────────────────────────────────────────────────────┘
                        │ （事后）
        ┌───────────────▼───────────┐
        │ 实测落点 impacts.jsonl     │ → ballistics.fit.fit_ballistics → ballistics_fit.json
        └───────────────────────────┘
```

关键点：**从 `AlignmentBuffer` 往右的所有环节，实飞与离线回放走的是同一条代码路径**——
回放时 `ReplayVideoSource` 顶替 `Hm30VideoSource`、`TelemetryPacer` 把录下来的遥测按
**原始时间戳**逐步灌进 broker（见 §3.3），于是对齐、解算、聚类、判据**一行都不用改**
（`airdrop.record.replay` 的模块 docstring 把这条设计意图写在最前面）。

### 2.2 模块依赖关系

| 模块 | 依赖谁 | 被谁依赖 |
| --- | --- | --- |
| `airdrop.config` | `airdrop.video.source`（仅取 `VideoConfig` 与 HM30 默认地址）、`airdrop.perception`/`airdrop.georef`（惰性导入，避免导入环） | 几乎所有模块 |
| `airdrop.telemetry.models` | 无（纯 dataclass + `time`/`uuid`） | telemetry / video / perception / record / mission |
| `airdrop.telemetry.broker` | `models` | video.align / record（recorder + replay）/ examples / tests |
| `airdrop.telemetry.mavsdk_thread` | `models`、`broker`、mavsdk | examples；`controller` 通过它下发指令 |
| `airdrop.telemetry.controller` | `models`、`mavsdk_thread`（经 `submit`）、`mavsdk` 的 `mission_raw`/`mission`/`gripper`/`action` | `mission.runner`（按 `MissionController` 协议） |
| `airdrop.video.source` | `VideoConfig`（本模块定义）、ffmpeg 子进程 | `record.replay`（同接口）、examples |
| `airdrop.video.align` | `broker`、`VideoFrame`、`AlignConfig` 语义 | `video.buffer.AlignmentWriter`、`record.replay` |
| `airdrop.video.buffer` | `align.AlignedSample`、cv2/numpy | `perception.pipeline`、`record.recorder` |
| `airdrop.record.recorder` | `broker`、`buffer`、`ballistics.drops.DropRecord` | examples（`on_event`/`on_drop`/`detections`） |
| `airdrop.record.replay` | `broker`、`buffer`（可选）、`video.source` 的接口形状 | examples.replay_flight、tests.test_replay |
| `airdrop.perception.detector` | ultralytics（**惰性导入**）、`georef` 相机矩阵（可选去畸变） | `perception.pipeline` |
| `airdrop.perception.cropproc` | rapidocr（惰性）、cv2 | `perception.pipeline`（OCR 进程内调用） |
| `airdrop.perception.ocr_worker` | `multiprocessing`、`cropproc` | `perception.pipeline`（`OcrWorkerPool`） |
| `airdrop.perception.pipeline` | `buffer`、`detector`、`ocr_worker` | `mission.targets.PerceptionTargetSource` |
| `airdrop.georef.camera` | numpy、cv2（去畸变）、标定 JSON | `georef.project`、`mission.targets`、examples |
| `airdrop.georef.project` | `camera`、numpy | `mission.targets`、`georef.__init__` |
| `airdrop.georef.geo` | pyproj（模块级建 transformer） | `mission.planner`、`ballistics.drops`、examples |
| `airdrop.targeting.cluster` | scikit-learn（DBSCAN） | `mission.targets`、examples |
| `airdrop.ballistics.model` | numpy 无关（纯 math）、`config.BallisticsConfig` | `ballistics.release`、`ballistics.drops`、`ballistics.fit` |
| `airdrop.ballistics.release` | `model`、`config.DropConfig`、`telemetry.models` | `mission.runner`（按 `ReleaseJudgeLike` 协议） |
| `airdrop.ballistics.drops` | `config`、`georef`（原点换算/姿态）、`model` | `ballistics.fit`、`record.recorder`、`mission.runner` |
| `airdrop.ballistics.fit` | scipy（`least_squares`）、`drops`、`model` | `tools.fit_ballistics` |
| `airdrop.mission.states` | `config`?（否，纯标准库） | `mission.runner` |
| `airdrop.mission.items` | 无（纯标准库；**不导入 telemetry**，避免成环） | `mission.planner`、`mission.plan_file`、`telemetry.controller`（仅类型标注 + 惰性导入 `command_name`）、`airdrop/__init__` |
| `airdrop.mission.plan_file` | `mission.items`（纯 JSON，不碰 MAVSDK/网络） | `mission.planner`、`tests/test_plan.py` |
| `airdrop.mission.planner` | `config`、`georef.geo`、`mission.items`、`mission.plan_file` | `mission.runner` |
| `airdrop.mission.targets` | `perception`（协议）、`georef`、`targeting` | `mission.runner`、examples |
| `airdrop.mission.runner` | `states`、`planner`、`broker`、`controller` 协议、`judge` 协议 | examples、`tools`（无） |

依赖方向是**单向**的：`config → telemetry/video/georef/targeting/ballistics → perception/record → mission → examples/tools`。
三个刻意的"反向"依赖都用**惰性导入**绕开：`config.PerceptionConfig.ocr_engine_config` /
`to_pipeline_config` 在函数内部导入 perception；`config.CameraConfig.load_model` 在函数内导入 georef；
`telemetry.controller` 只在**类型标注**下导入 `mission.items.MissionItem`，运行时在 `_verify_upload`
函数体内才导入 `command_name`（否则 `telemetry ↔ mission` 会成环）。
原因写在 `airdrop/config.py:153`、`:165`、`:198` 与 `airdrop/telemetry/controller.py` 的模块 docstring。

### 2.3 并发模型（谁拥有哪些线程/进程）

| 组件 | 形态 | 谁创建 | 生命周期 |
| --- | --- | --- | --- |
| `MavsdkThread` | **1 个线程**（内部跑独立 asyncio 事件循环 + supervisor 重连循环） | 调用方 `MavsdkThread.from_config(...)` + `.start()` | `.stop()` 显式结束；**必须**保证会话结束释放 `mavsdk_server`（§7 第 1 条） |
| `mavsdk_server` | **1 个外部进程**（mavsdk 库自己拉起，gRPC 端口固定 50051） | mavsdk 库 | 由 `_release_drone()` 收尾；不显式释放会留僵尸进程 |
| `Hm30VideoSource` | **1 个采集线程 + 1 个 ffmpeg 子进程** | `Hm30VideoSource(config)` 构造时起线程，`.start()` 起流 | `.stop()` 杀子进程并 join 线程 |
| `AlignmentBuffer` | 无自己的线程：写者是采集线程（经 `AlignmentWriter`），读者是各自线程/主线程 | 调用方 | 随进程 |
| `PerceptionWorker` | **1 个工作线程 + N 个 OCR 进程**（`ocr_workers`，默认 2） | `.start()` | `.stop()` 停线程并收进程池 |
| `FlightRecorder` | **2 个写盘线程**（`recorder-frames`、`recorder-telemetry`，daemon） | `.start(broker=…, buffer=…)` | `.stop()` join + 关文件 |
| `ReplayVideoSource` | **1 个回放线程**（`replay-video`） | `.start()` | `.stop()` |
| `MissionRunner` | **不自己起线程**：`run()` 是普通循环，`update()` 可单拍驱动 | 调用方 | 调用方决定 |
| `DroneController` | 无线程：所有指令经 `MavsdkThread.submit` 投到 MAVSDK 线程，再用一把 `RLock` 串行化 | 调用方 | 随进程 |

边界与约定：

* **跨线程只传不可变值或拷贝**：`TelemetrySnapshot` 是 dataclass、字段全标量，broker 里
  publish/get 都做浅拷贝；`BufferedFrame.image` 是拷贝出来的 ndarray（§7 第 9 条）。
* **跨进程只传图像 ndarray 与轻量结果**（`OcrRequest`/`OcrResult`），靠 `multiprocessing`
  的 pickle 管道；OCR 进程池是唯一的多进程组件（`airdrop/perception/ocr_worker.py:1` 的
  docstring 说明了"为什么单独开进程"以及复测前的保守默认）。
* **谁也不用共享内存**：没有全局可变状态；`config` 是 frozen dataclass，构造后不可变。
* **优先"每帧路径不阻塞"**：OCR 在独立进程池、事件/检测写盘在独立线程、ffmpeg 在子进程。

---

## 3. 模块逐个说明

每节给出：职责与关键类型 → **调用方法（签名）** → **最小可运行示例**（离线、不碰硬件；
本节所有示例都在 `tests/test_handbook.py` 里逐条真跑，11/11 通过）→ 注入点/协议 →
失败语义 → 易错点与指针。

### 3.1 `airdrop.telemetry` —— 遥测与控制

**职责**：把 MAVSDK 的六条遥测流合并成"一份最新飞机状态"，并把任务级动作（上传任务、启动、
hold、rtl、投放、查原点）包成同步方法。关键类型：`airdrop.telemetry.models.TelemetrySnapshot`、
`airdrop.telemetry.broker.TelemetryBroker`、`airdrop.telemetry.mavsdk_thread.MavsdkThread`、
`airdrop.telemetry.controller.DroneController`。

#### 3.1.1 `TelemetryBroker`（线程安全的进程内代理）

```python
TelemetryBroker(history_maxlen: int = 1200, history_interval: float = 0.1)
```

| 方法 | 说明 |
| --- | --- |
| `get_snapshot` | 最新快照的**副本**（可安全修改） |
| `get_snapshot_at(timestamp, mode='interpolate')` | 按时间查历史：`interpolate` 范围内线性内插 / 范围外外推，`nearest` 取最近；`mode` 只接受 `SUPPORTED_QUERY_MODES` 里的值 |
| `history_span` | 历史覆盖的 `(最早, 最新)`；空历史返回 None |
| `wait_history_until(timestamp, timeout=None)` | 阻塞直到历史覆盖到该时刻（对齐器用它等遥测追上） |
| `wait_next_snapshot(timeout=None, predicate=None)` | 阻塞等到满足 `predicate` 的下一条快照 |
| `subscribe(callback)` / `unsubscribe(callback)` | 新快照推送（回调在**发布线程**里调用，必须快） |
| `update_local_position_velocity(position_ned, velocity_ned)` 等 `update_*` | 送入一条 MAVSDK 风格的更新，返回合并后的快照 |
| `publish(snapshot)` | 直接发布一条**完整快照**（保留它自己的 `timestamp`、不节流）——回放用它 |
| `reset` | 清空快照与历史（重连前调用） |

**最小示例**（真跑通）：

```python
import time
from types import SimpleNamespace
from airdrop import TelemetryBroker

broker = TelemetryBroker(history_maxlen=1200, history_interval=0.0)
position = SimpleNamespace(north_m=100.0, east_m=200.0, down_m=-50.0)
velocity = SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0)
snapshot = broker.update_local_position_velocity(position, velocity)

print(snapshot.north_m, snapshot.east_m, snapshot.vx_m_s)  # 100.0 200.0 15.0
print(broker.get_snapshot_at(snapshot.timestamp, mode="nearest").north_m)  # 100.0
```

**失败语义**：不抛错。历史为空 → `history_span()` 返回 None、`get_snapshot_at` 返回 None、
`wait_*` 返回 False/None。**没有数据就是 None，不编一个默认值**。

**易错点**：`history_interval` 是**入库节流**（默认 0.1s ≈ 10Hz），历史最多 `history_maxlen`
条（默认 1200 条 ≈ 2 分钟）；最新快照与订阅推送不受节流影响。查询用
`bisect(key=operator.attrgetter("timestamp"))` 直接探测 deque（§7 第 2 条）。

#### 3.1.2 `MavsdkThread`（MAVSDK 工作线程）

```python
MavsdkThread.from_config(config, *, broker=None, thread_name="mavsdk-thread")
MavsdkThread(broker=None, system_address="udpin://0.0.0.0:14540", *,
             connect_timeout=30.0, wait_for_health=False, health_timeout=30.0,
             require_health_for_actions=False, origin_refresh_interval=5.0,
             reconnect=True, reconnect_delay=5.0,
             position_rate_hz=None, position_velocity_ned_rate_hz=None,
             attitude_rate_hz=None, thread_name="mavsdk-thread")
```

| 方法 | 说明 |
| --- | --- |
| `start(timeout=5.0)` / `stop(timeout=5.0)` | 起/停专用事件循环线程（幂等） |
| `connect(system_address=None, timeout=None)` | 连接飞控并等就绪（同步方法，任意线程可调） |
| `get_snapshot` / `get_snapshot_at(timestamp, mode='interpolate')` | 线程安全的快捷查询 |
| `get_gps_global_origin(timeout=10.0)` | 实时查询 NED 原点并写入快照 |
| `send_command(command, timeout=30.0)` | 发一条 `Command` 并等 `CommandResult` |
| `submit(coro)` | 把协程调度到 MAVSDK 事件循环，返回 `concurrent.futures.Future` |
| `require_drone()` | 返回已连接的 `System`；未连接时**抛出** |

**6 条核心流**：position / home / position_velocity_ned / attitude_euler /
attitude_quaternion / gps_global_origin，用 `asyncio.TaskGroup` 并行采集，**任一核心流结束即视为
断连**，交给 supervisor 重连。**两条"可选流"**（`mission_raw.mission_progress()` 任务进度、
`telemetry.flight_mode()` 飞行模式）：结束或报错只记日志、不影响会话——它们分别送入快照的
`mission_current`/`mission_total` 与 `flight_mode`，状态机靠这二者判断"任务是否飞完""是否真在任务模式"。
风估计也是可选流（没有它时投放判据按零风降级并记一次日志），但它是 MAVSDK `telemetry.wind()`。

#### 3.1.3 `DroneController`（任务级控制）

```python
DroneController.from_config(config, thread, *, on_event=None, timeout=30.0, sleep=time.sleep)
DroneController(thread, *, gripper_instance=0, gripper_enabled=True,
                release_settle_s=0.5, timeout=30.0, on_event=None, sleep=time.sleep)
```

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `upload_mission(items, /)` | `int` | 上传任务（覆盖飞控上的旧任务），返回任务项数；**走 `mission_raw`** 并**回读校验**（命令+位置逐项比对，不一致抛错） |
| `start_mission()` | None | 启动任务：**先 `set_current_mission_item(0)` 复位再启动**（否则同 CRC 的旧任务可能停在 `HOLD`） |
| `in_mission_mode()` | `bool` | 飞控**是否真的**在任务模式（读快照 `flight_mode`）；流未就绪 → 抛 `ControllerError` |
| `mission_finished()` | `bool` | 是否飞完（读快照 `mission_current == mission_total`）；进度流未就绪 → 抛 `ControllerError` |
| `hold()` / `rtl()` | None | 盘旋 / 返航 |
| `gripper_release()` | `bool` | 是否**真的发出了**投放指令（`GripperConfig.enabled=False` 时返回 False） |
| `request_origin()` | `NedOrigin \| None` | 实时查 NED 原点；未就绪返回 None |

**注入点/协议**：`airdrop.telemetry.controller.MissionController` 是 `Protocol`——状态机只依赖
它的八个方法，离线测试用假控制器（`tests/test_mission.py` 的 `FakeController`）。
`to_raw_item(item, index)` 是纯函数（`MissionItem` → MAVSDK raw 项），可离线断言；它给第 0 项置
`current=1`（全 0 会被 MAVSDK 判 `CURRENT_INVALID`，实测），`autocontinue` 传 **int**（传 bool 会被拒）。

**失败语义（贯穿全项目的约定）**：

* **命令**（上传/启动/hold/rtl/投放）失败 → 抛 `airdrop.telemetry.controller.ControllerError`；
* **查询**（`mission_finished` / `request_origin`）没有结果 → 返回 `False` / `None`，
  **不抛错**（"读到了、但确实还没有"）；而"快照里根本没有这个字段"（进度/模式流未就绪）算**查询失败**，
  抛 `ControllerError`，由 `MissionRunner._query` 记一次日志并按默认值继续；
* `MissionItem` 的"没用到"参数填 `float('nan')`（`UNSET`），**0 是有意义的值**（接受半径 0 = 用飞控默认、
  盘旋 0 秒）；位置是否合法只对 `is_positional` 的命令校验（`DO_LAND_START` 这类指令项不带坐标）。

#### 3.1.4 `DryRunController`（演练包装器）

```python
DryRunController(inner: MissionController, *, on_event=None)
```

**只拦投放**：`gripper_release()` 记事件 `drop_dry_run`、返回 True；航线/hold/rtl/查询全部
委托给 `inner`。用途：SITL 与"不装弹"演练能走完整状态机（SITL 里没有 gripper 硬件，
投放被拒会让状态机按失败进 ABORT）。**真机装弹时绝不能使用。**

> 线程/重连细节、`_release_drone()` 的必要性、可选流为什么不影响会话、以及"为什么任务项一律走
> `mission_raw`"（MAVSDK 的 `vehicle_action=LAND` 会把一项拆成两项、被 PX4 固定翼整条拒掉）→
> `AGENTS.md` 的"架构要点 / 关键约定"。

### 3.2 `airdrop.video` —— 图传接收、对齐与缓冲

#### 3.2.1 `Hm30VideoSource`（唯一的拉流后端：ffmpeg 子进程）

```python
Hm30VideoSource(config: VideoConfig | None = None)
open_hm30_video(config=None, *, autostart=True) -> Hm30VideoSource
```

| 方法 | 说明 |
| --- | --- |
| `add_sink(sink)` / `remove_sink(sink)` | 注册**每帧**回调（在采集线程里调用，**一帧不落**） |
| `start()` / `stop(timeout=5.0)` | 起动/停止拉流（幂等；stop 会杀 ffmpeg 子进程） |
| `wait_ready(timeout=10.0)` | 等到第一帧；超时 False |
| `read(timeout=None)` | 阻塞等**下一帧新画面**（实时路径，允许跳帧，跳帧计入 `stats.dropped`） |
| `latest()` | 立刻返回最新一帧（不等待） |
| `iter_frames(timeout=1.0)` | 迭代新帧，连续超时即结束 |
| `stats` | `airdrop.video.source.VideoStats`：`state` / `frames` / `dropped` / `reconnects` / `fps` / `last_error` |

**两条路别混用**：推理与写入磁盘要接 `add_sink`（经 `AlignmentWriter` 进缓冲）；`read()`/`latest()`
只保证"当前这一帧"，慢消费者会丢中间帧。

#### 3.2.2 `FrameTelemetryAligner`（按拍摄时刻取遥测）

```python
FrameTelemetryAligner(broker, lag=None, mode="interpolate",
                      max_extrapolation=None, max_wait=0.0, on_timeout="drop")
align(frame: VideoFrame | None) -> AlignedSample | None
```

`拍摄时刻 = frame.timestamp − lag`（`lag` 为 None 时采用帧自带的 `frame.lag`）。
返回 None 的四种情况：帧为 None；历史为空；查询时刻比历史新且等满 `max_wait` 仍未追上
（`on_timeout="drop"`）；超出 `max_extrapolation` 的硬保护。
`AlignedSample` 带 `extrapolated` / `offset`，如实标记这次查询是不是外推。

**最小示例**（真跑通；注意帧的"收到时间"必须比遥测晚 `lag` 秒，否则历史没覆盖拍摄时刻，
对齐器会等满 `max_wait` 后丢帧并在日志里警告）：

```python
import time
import numpy as np
from types import SimpleNamespace
from airdrop import (
    AlignmentBuffer,
    AlignmentWriter,
    FrameTelemetryAligner,
    TelemetryBroker,
    VideoFrame,
)

broker = TelemetryBroker()
broker.update_local_position_velocity(
    SimpleNamespace(north_m=100.0, east_m=200.0, down_m=-50.0),
    SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0),
)
aligner = FrameTelemetryAligner(broker, max_wait=0.5)
buffer = AlignmentBuffer(capacity=8, storage="jpeg")
writer = AlignmentWriter(buffer, aligner)  # 标准 sink 接法

frame = VideoFrame(
    index=1,
    image=np.zeros((180, 320, 3), dtype=np.uint8),
    timestamp=broker.get_snapshot().timestamp + 0.15,
    lag=0.15,
)
writer(frame)  # 逐帧回调里被调用
record = buffer.latest()
print(record.index, record.snapshot.north_m, writer.stats())  # 1 100.0 (1, 0)
```

#### 3.2.3 `AlignmentBuffer` / `AlignmentWriter`（环形缓冲）

```python
AlignmentBuffer(capacity=5400, *, storage="jpeg", jpeg_quality=80, max_bytes=None)
AlignmentWriter(buffer, aligner)      # 可调用对象：__call__(frame)
```

| 方法 | 说明 |
| --- | --- |
| `put(sample)` | 存入一条对齐结果（内部会拷贝图像），返回 `BufferedFrame` |
| `latest()` / `at(index)` / `latest_index()` / `indices()` | 取最新 / 按序号取 / 序号范围 |
| `iter_between(start=None, end=None, *, batch=16)` | **按时间区间分批迭代**（回看历史的正道） |
| `wait_new(after_index=0, timeout=None)` | 等一条更新记录，返回**最早**满足条件的那条（读者各自持游标） |
| `read_jpeg_bytes(index)` | 内部 API：取原始 jpeg 字节（`FlightRecorder` 零重编码写入磁盘用） |
| `clear()` | 清空（不影响已发出的记录对象） |
| `AlignmentWriter.stats()` | `(written, skipped)`：`skipped` 只因"该时刻取不到遥测"增加 |
| `BufferStats` | `storage` / `capacity` / `max_bytes` / `frames` / `put` / `evicted` / `bytes` / `last_index` |

默认 `storage="jpeg"`：720p 约 0.1~0.2 MB/帧，满载（`capacity_for(30, 180)` = 5400 帧）约
0.6~1 GiB；`storage="raw"` 无损但 5400 帧约 13.9 GiB。满了从**最旧**一端驱逐。
`capacity_for(fps=30.0, seconds=180.0)` 做换算；`raw_frame_bytes(width, height, channels=3)`
算未压缩一帧的字节数。

**易错点**：ffmpeg 后端的 `VideoFrame.image` 是**复用管道缓冲区的视图**——
`AlignmentBuffer` 内部已拷贝，但你若自己跨帧保存必须 `.copy()`；`raw` 模式读到的 image 是
只读视图。

### 3.3 `airdrop.record` —— 记录与回放

#### 3.3.1 `FlightRecorder`（飞行目录写入磁盘）

```python
FlightRecorder(config: Config, *, base_dir: str | Path | None = None)
start(*, broker: TelemetryBroker, buffer: AlignmentBuffer) -> FlightRecorder
stop(timeout: float = 5.0) -> None
stats() -> RecorderStats | None
events      -> EventLog          # events.jsonl 写入器（未启动时抛 RuntimeError）
detections  -> DetectionWriter   # detections.jsonl
drops       -> DropWriter        # drops.jsonl
flight_dir  -> Path | None
```

启动行为：在 `record.dir` 下建 `flights/<YYYYMMDD-HHMMSS>/`（重名自动追加 `-1`/`-2`）→
写 `config_snapshot.json` → 打开七个文件 → 给 root logger 挂 `FileHandler`（任何模块的
`logger.info` 都自动进 `flight.log`）→ 起 2 个写盘线程。**幂等**：重复 `start` 直接返回；
`stop` 后再 `start` 会报错（要新建实例）。

**最小的"录一次 + 回放"示例**（真跑通；用工作区内的临时目录，避免污染 `flights/`）：

```python
import json, shutil, time, uuid
from pathlib import Path
import cv2, numpy as np
from airdrop import (
    AlignmentBuffer,
    Config,
    FlightLog,
    FlightRecorder,
    ReplayVideoSource,
    TelemetryBroker,
    TelemetrySnapshot,
    load_broker_from_log,
)

work = Path(".fit-test-tmp") / uuid.uuid4().hex[:8]  # 任意可写目录
work.mkdir(parents=True, exist_ok=True)
config = Config()

recorder = FlightRecorder(config, base_dir=work)
recorder.start(broker=TelemetryBroker(), buffer=AlignmentBuffer(capacity=4))
recorder.events.emit("state", from_state="RECON", to_state="HOLD")
recorder.stop()
flight_dir = recorder.flight_dir  # work/<时间戳>/
print(sorted(p.name for p in flight_dir.iterdir()))
# ['config_snapshot.json', 'detections.jsonl', 'drops.jsonl', 'events.jsonl',
#  'flight.log', 'frames', 'frames_index.jsonl', 'telemetry.jsonl']

# 手工补一份最小素材（真飞时由写盘线程产出）：1 帧 + 3 条遥测
frames_dir = flight_dir / "frames"
frames_dir.mkdir(exist_ok=True)
base = time.time()
snapshots = []
for i, north in enumerate((0.0, 10.0, 20.0)):
    snap = TelemetrySnapshot(timestamp=base + i * 0.1)
    snap.north_m = north
    snapshots.append(snap)
(flight_dir / "telemetry.jsonl").write_text(
    "\n".join(json.dumps(s.as_dict()) for s in snapshots) + "\n", encoding="utf-8"
)
jpeg = cv2.imencode(".jpg", np.zeros((180, 320, 3), dtype=np.uint8))[1].tobytes()
(frames_dir / "000001.jpg").write_bytes(jpeg)
(flight_dir / "frames_index.jsonl").write_text(
    json.dumps(
        {
            "index": 1,
            "filename": "000001.jpg",
            "capture_timestamp": base,
            "received_timestamp": base + 0.15,
            "lag": 0.15,
            "extrapolated": False,
            "offset": 0.0,
            "bytes": len(jpeg),
        }
    )
    + "\n",
    encoding="utf-8",
)

log = FlightLog.open(flight_dir)
print(len(log.frames), sum(1 for _ in log.iter_telemetry()))  # 1 3
broker, pacer = load_broker_from_log(flight_dir)  # 遥测尚未灌入
pacer.publish_until(snapshots[-1].timestamp)
print(broker.get_snapshot().north_m)  # 20.0

received = []
source = ReplayVideoSource(flight_dir, speed=0.0)  # 0 = 全速
source.add_sink(received.append)
source.start()
try:
    source.wait_ready(timeout=5.0)
    while not received:
        time.sleep(0.05)
    print(received[0].index, abs(received[0].capture_timestamp - base) < 1e-9)  # 1 True
finally:
    source.stop()
    shutil.rmtree(work, ignore_errors=True)
```

#### 3.3.2 `ReplayVideoSource`（与实飞源同接口）

```python
ReplayVideoSource(flight_dir, *, speed=0.0, telemetry=None, buffer=None,
                  repeat=False, strict=True, tail_s=0.0)
```

`speed=0` 全速（批量回归）/ `1.0` 原速。与 `Hm30VideoSource` **同一套公开接口**
（`add_sink`/`read`/`latest`/`stats`/`iter_frames`/`stop`/`wait_ready`），换进去即可。
`sink` 路径一帧不落；`read()` 仍允许跳帧（计入 `stats.dropped`，回放里叫 `skipped` 的是
"素材缺文件/解码失败"）。`strict=True`（默认）下缺帧或 sink 抛异常会让回放以
`state="error"` 结束——做正式评估时宁可失败，也不要静默跳过。

#### 3.3.3 `load_broker_from_log` / `TelemetryPacer`（原时间轴回填）

```python
load_broker_from_log(flight_dir, *, broker=None, history_maxlen=0, ahead_s=0.5)
    -> tuple[TelemetryBroker, TelemetryPacer]
TelemetryPacer(broker, snapshots, *, ahead_s=0.5, tail_warn_s=5.0)
    · publish_until(timestamp) -> int      # 发布 ≤ timestamp + ahead_s 的快照
    · finish(capture_timestamp) -> int     # 按需把覆盖补到某帧：投递每帧前调用（尾部缺失时延伸）
    · cover_to() -> float                  # 当前已推进到的时间
```

刻意**不一次性灌完**：由回放源在投递每帧之前按帧时刻推进，使该帧拍摄时刻必然落在历史区间内
（内插命中），同时"遥测追不上"这类行为仍然可复现。`history_maxlen=0` 表示不限长，
`history_interval` 固定为 0（回填时间轴不能被采样节流改动）。

#### 3.3.4 `FlightLog` / `FrameRecord`（飞行目录只读视图）

```python
FlightLog.open(flight_dir) -> FlightLog      # 解析 frames_index.jsonl（遥测按需流式读）
FlightLog.time_span() -> tuple[float, float] | None
FlightLog.iter_telemetry() -> Iterator[TelemetrySnapshot]
```

异常：`airdrop.record.replay.FlightLogError`（目录不可用/缺遥测）、
`airdrop.record.replay.FrameIndexError`（`frames_index.jsonl` 损坏：时间戳不递增、字段缺失）。
`iter_telemetry()` 对**坏行只记 warning 跳过**（一段素材不该因最后一行被截断而整份作废）。

### 3.4 `airdrop.perception` —— 检测、裁剪转正与 OCR

| 文件 | 内容 |
| --- | --- |
| `airdrop.perception.models` | `PixelBox`、`Detection`（贯穿全项目的观测载体） |
| `airdrop.perception.detector` | `Detector`（YOLO 封装，`ultralytics` 惰性导入）、`DetectionBatch`、`DetectorConfig` |
| `airdrop.perception.cropproc` | `OpenCvPostProcess`（裁剪图 → 五边形定位 → 转正 → OCR）、`OcrEngine`（RapidOCR 薄封装）、`OcrEngineConfig`、`CropResult`、`OcrText`、`OcrOrientation` |
| `airdrop.perception.number` | `correct_ocr_number`：OCR 原始串 → 00–99 整数（`OCR_CHAR_MAP` 字符映射） |
| `airdrop.perception.ocr_worker` | `OcrWorkerPool`（OCR 独立进程池）、`OcrRequest`、`OcrResult`、`ocr_worker_main` |
| `airdrop.perception.pipeline` | `PerceptionWorker`（逐帧主循环）、`PerceptionStats`、`PerceptionConfigLike` |

#### 3.4.1 两个数据模型

`PixelBox` 是像素矩形，字段 `x1`、`y1`、`x2`、`y2`（浮点，允许越界），另有属性 `width`、`height`、`area`、`center` 与方法 `clipped(width, height)`、`expanded(ratio, width, height)`、`as_dict()`。

`Detection` 是一次检测的完整记录（字段与默认值）：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `frame_index` | 必填 | 帧号（来自缓冲，全局递增） |
| `capture_timestamp` | 必填 | **拍摄时刻**（不是收到时刻） |
| `pixel` | 必填 | 目标中心像素 `(x, y)` |
| `box` | 必填 | `PixelBox` |
| `confidence` | 必填 | 检测置信度 |
| `telemetry` | 必填 | 该拍摄时刻的 `TelemetrySnapshot` |
| `code` | `None` | 目标编号（`cls12` 来自 YOLO 类别，`ocr` 来自 OCR） |
| `side_px` | `0.0` | 目标边长像素（`ocr` 模式下由转正结果给出） |
| `raw_text` | `''` | OCR 原始串（纠错前） |
| `ocr_confidence` | `0.0` | OCR 置信度 |
| `mode` | `''` | 产生它的模式（`ocr` / `cls12`） |
| `extra` | 空字典 | 附加信息，检测器会写 `undistorted`（像素是否已在纠正后的图上） |

属性 `has_code`（`code is not None`）与 `as_dict()`（键：`frame_index`、`capture_timestamp`、`pixel`、`box`、`confidence`、`code`、`side_px`、`raw_text`、`ocr_confidence`、`mode`、`telemetry`、`extra`；`telemetry` 内嵌 `timestamp`、`latitude_deg`、`longitude_deg`、`absolute_altitude_m`、`north_m`、`east_m`、`down_m`、`roll_deg`、`pitch_deg`、`yaw_deg`，`extra` 为空时不出现）。`detections.jsonl` 与回放素材里写的就是这个字典。

#### 3.4.2 `Detector` —— YOLO 封装

```python
Detector(config: DetectorConfig | None = None)
DetectorConfig(model_path='models/best2.pt', device='0', conf_threshold=0.25,
               iou_threshold=0.45, imgsz=1280, max_detections=300,
               crop_expand_ratio=0.2, camera_matrix=None, dist_coeffs=None,
               undistort=True)
```

方法：

| 方法 | 说明 |
| --- | --- |
| `load()` | 惰性 `import ultralytics` 并加载权重；返回自身（可链式） |
| `warmup(width, height)` | 用一张空图跑一次，避免首帧卡顿；返回是否成功 |
| `prepare_undistort(width, height, *, force=False)` | 预计算 remap 映射表；**映射表锁定在标定分辨率上**，尺寸不符直接返回 `False` 并告警 |
| `undistort(image)` | 返回 `(纠正后图像, 是否真的纠正了)` |
| `detect(image, *, frame_index=0, capture_timestamp=0.0, telemetry=None, mode='', undistort=None)` | 返回 `DetectionBatch(image, detections, undistorted)` |
| `crop(batch, detection)` | 按 `crop_expand_ratio` 外扩取目标裁剪图（交给后处理） |
| `replace_config(**changes)` | 派生一个新检测器（改 `conf_threshold` 等做 A/B） |

三个必须知道的行为：

- **`device` 显式指定**（默认 `'0'`）。自动检测在本地曾误选 CPU，所以不给它机会。
- **去畸变用预计算 remap**（比逐帧 `undistort()` 快 3~5 倍）。只有 `camera_matrix` 非空时才会建表；建表用的分辨率与当前帧不一致就**跳过纠正**而不是用错内参扭画面。
- `detect` 会把"像素是否已被纠正"写进 `Detection.extra["undistorted"]`，供坐标解算判断要不要再纠正一次（见 3.5）。

#### 3.4.3 `OpenCvPostProcess` —— 裁剪图后处理

```python
OpenCvPostProcess(color='blue', ocr_conf_threshold=0.6, *, engine=None,
                  engine_config=None, target_side_px=300.0, max_upscale=10.0)
```

类常量（调门限前先读 [`docs/perception_ocr.md`](perception_ocr.md)）：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `HOUSE_APEX_ANGLE_DEG` | `60.0` | 五边形顶角期望值（等边三角形内角） |
| `HOUSE_APEX_ANGLE_TOL_DEG` | `15.0` | 顶角容差（实测近侧余量只有 1.4°） |
| `BLUE_S_MIN_LEVELS` | `(100, 60, 40, 20)` | 蓝色掩码饱和度逐级回退 |
| `RED_S_MIN_LEVELS` | `(100, 80, 60, 40, 20)` | 红色掩码饱和度逐级回退 |

处理顺序（`cropproc.py` 模块 docstring 有完整版）：

1. 去噪 → 等比放大到短边 `target_side_px`（上限 `max_upscale` 倍）；
2. 颜色掩码（HSV `inRange`），饱和度级别从高到低逐个回退；
3. 形态学 + 凸包 + `approxPolyDP` 确定性扫描 `epsilon=3..40` 找五边形；
4. 几何转正（**平行边法**为主、最小内角法兜底），顶角形态判据（60°±15°）拒绝假房子；
5. **彩图**整图 OCR（v6 检测模型对灰度图检不出文本框）；
6. 单数字框按行聚类拼两位 → 置信度加权（两位 ×1.15、单位 ×0.6）→ `correct_ocr_number` 纠错。

三个接口：

| 方法 | 说明 |
| --- | --- |
| `warmup()` | 预热 OCR；返回是否成功 |
| `edge_detection()` | **读 `self.image` 而不是参数**，返回 `(approx, ok)`，并设置 `self.s_min`（实际生效的饱和度级别） |
| `recognize(image)` | 完整后处理（含 OCR），返回 `CropResult` |
| `visualize(image)` | 返回 `(画了辅助线的图, CropResult)`，排障用 |

`CropResult` 字段：`number`（纠错后的编号）、`side_px`、`rectified`（转正后的图）、`raw_text`、`confidence`、`saturation_level`、`stage`（走到哪一步）、`house_ok`（形态判据是否通过）。

用 `edge_detection()` 单测几何时要注意：它是**实例方法且没有图像参数**，必须先 `post.image = image`（内部 `recognize`/`visualize` 会自己设）。

```python
"""五边形定位（纯几何，不需要 OCR 权重）。"""

import cv2
import numpy as np
from airdrop.perception import OpenCvPostProcess

image = np.zeros((320, 320, 3), dtype=np.uint8)
side, left, top = 100, 110, 180  # 正方形左上角
apex_y = int(top - side * (3**0.5) / 2)  # 等边三角形顶角
pentagon = np.array(
    [
        [left, top + side],
        [left + side, top + side],
        [left + side, top],
        [left + side // 2, apex_y],
        [left, top],
    ],
    dtype=np.int32,
)
cv2.fillPoly(image, [pentagon], (255, 0, 0))  # BGR：纯蓝
post = OpenCvPostProcess(color="blue")
post.image = image  # edge_detection 用 self.image
approx, ok = post.edge_detection()
print("五边形顶点 %d 个，ok=%s（s_min=%d）" % (len(approx), ok, post.s_min))
```

实测输出：`五边形顶点 5 个，ok=True（s_min=100）`。

#### 3.4.4 `OcrEngine` 与 `OcrEngineConfig`

`OcrEngine(config: OcrEngineConfig | None = None)`，方法：`ensure()`（惰性建 RapidOCR 实例）、`cls_providers()`（当前可用的方向分类后端）、`classify_orientation(image)`（返回 `OcrOrientation(label, score)`）、`warmup(size=320)`、`read_text(image)`（返回 `list[OcrText]`，每项 `box`/`text`/`score`）。

`OcrEngineConfig` 默认值（走 `airdrop.perception.cropproc`）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model_dir` | `'models/ppocr'` | 本地权重目录 |
| `det_model` | `'PP-OCRv6_det_medium.pth'` | 检测权重 |
| `rec_model` | `'PP-OCRv6_rec_medium.pth'` | 识别权重 |
| `rec_keys` | `'ppocrv6_dict.txt'` | 字符集字典（在 git 里） |
| `cls_engine` | `'onnx'` | 方向分类后端（实测最快） |
| `cls_onnx_model` | `'ch_ppocr_mobile_v2.0_cls_mobile.onnx'` | ONNX 版分类权重 |
| `cls_torch_model` | `'ch_ptocr_mobile_v2.0_cls_mobile.pth'` | TORCH 版分类权重（**注意这个错拼**） |
| `cls_use_cuda` | `True` | 分类是否用 CUDA |
| `cls_thresh` | `0.9` | 方向判定阈值 |
| `cls_autorotate` | `True` | 按判定把倒置文本行转正后再识别 |
| `use_cuda` | `True` | det/rec 是否用 CUDA |
| `device_id` | `0` | GPU 序号 |
| `text_score` | `0.1` | RapidOCR 的 `Global.text_score` |
| `rec_batch_num` | `6` | 识别批大小 |
| `log_level` | `'error'` | RapidOCR 日志级别 |

两个前提条件写在 [`docs/perception_ocr.md`](perception_ocr.md)：权重文件名必须保持 RapidOCR 那个错拼（`ch_ptocr_`…），`onnxruntime-gpu` 要用 CUDA 得在进程里**先 `import torch`**。

#### 3.4.5 `OcrWorkerPool` —— OCR 独立进程池

```python
OcrWorkerPool(workers=2, *, queue_size=500, config: dict | None = None)
# start() / stop(timeout=5.0) / submit(crop, *, frame_index, capture_timestamp, payload=None)
# poll(timeout=0.0) / drain()
```

- `submit` 返回任务号；请求/结果队列都**无界、不丢弃**（目标可能只清晰一瞬，漏一个请求就可能漏目标）。`queue_size` 不再是容量，而是**积压告警阈值**：未完成数（`submitted - results`）达到它时打一条 WARNING（跨阈值一次；回落到一半以下后再次涨上来会重新告警），提示降帧率或加 OCR worker。`PerceptionStats.ocr_dropped` 同步池侧的 `dropped` 计数——不丢弃策略下它恒为 0，保留作监控。
- OCR 放独立进程的理由：RapidOCR 单帧几十到几百毫秒，放主循环里会直接拖垮逐帧检测。进程参数经 `config`（字典）传给 `ocr_worker_main`。
- Windows 是 spawn 语义：**入口脚本必须放在 `if __name__ == "__main__":` 之下**，否则子进程会重复导入主模块。某些受限环境里命名管道不可用、进程池会起不来（见 §8）。

#### 3.4.6 `PerceptionWorker` —— 逐帧主循环

```python
PerceptionWorker(config: PerceptionConfigLike, *, buffer: AlignmentBuffer,
                 detector=None, pool=None, start_index=0)
# start() / stop(timeout=10.0) / poll_result(timeout=0.0) / drain_results()
# iter_results(timeout=0.5) / set_result_callback(callback)
```

- `buffer` 必须是 `AlignmentBuffer`（不是别的可迭代对象）：worker 用 `wait_new` 逐帧跟随，**一帧不落**。`start_index` 决定从哪一帧开始（断点续跑用）。
- `detector` / `pool` 可注入：传假对象即可离线测整条流水线（`tests/test_perception.py` 正是这么做的），这也是它们存在的理由。
- `PerceptionConfigLike` 只要求这些字段：`mode`、`model_path`、`device`、`conf_threshold`、`imgsz`、`target_color`、`ocr_conf_threshold`、`ocr_workers`、`ocr_queue_size`、`ocr_dedupe_s`、`ocr_dedupe_px`、`min_side_px`、`max_side_px`、`engine`（默认 `None`）。真实的参数类是 `PerceptionConfig`，字段更多（见 §4）。

两种模式：

| `mode` | 行为 |
| --- | --- |
| `'ocr'` | 检测 → 裁剪 → 送 OCR 进程池；编号来自 OCR（`correct_ocr_number` 纠错），边长来自转正结果 |
| `'cls12'` | 检测 → **类别直接当编号**，不裁剪、不启动 OCR 池；`extra` 里带类别信息 |

其余行为约定：

- **边长过滤**：`side_px` 不在 `min_side_px` 与 `max_side_px` 之间的检测被丢弃并计入 `invalid_side`（默认 10~400 px）。
- **OCR 逐帧送检**：默认不启用跨帧去重，避免运动模糊导致唯一清晰帧被跳过；显式打开 `ocr_dedupe_s` / `ocr_dedupe_px` 时，才按时间和像素窗口合并相邻重复请求。**OCR 请求、OCR 结果、检测结果三条队列都无界、不丢弃**：积压只告警（OCR 侧按 `ocr_queue_size`，检测结果侧按 2000 条阈值），不会因为"队列满"丢掉任何请求或结果；积压长期不降时调 `perception.ocr_workers` 或降检测帧率。
- **编号覆盖规则**：只有 `cls12` 的类别直出才写编号；OCR 结果回填时不会清掉已存在的编号（否则跨帧投票会被打乱）。
- 单帧异常不终止循环：记日志并写进 `PerceptionStats.last_error`。
- `PerceptionStats` 字段：`frames`（处理的帧数）、`detections`、`with_code`、`submitted`（送 OCR 次数）、`deduped`、`results`、`invalid_side`、`ocr_dropped`（队列丢弃数——不丢弃策略下恒为 0，保留作监控）、`lag_frames`、`last_error`。

```python
"""感知主循环（cls12 模式 + 注入假检测器）：缓冲 → 检测 → Detection。"""

import time
import numpy as np
from airdrop import (
    AlignedSample,
    AlignmentBuffer,
    Detection,
    DetectionBatch,
    PixelBox,
    TelemetrySnapshot,
    VideoFrame,
)
from airdrop.perception import PerceptionConfigLike, PerceptionWorker

buffer = AlignmentBuffer(capacity=4, storage="raw")
snapshot = TelemetrySnapshot(timestamp=time.time())
snapshot.north_m, snapshot.east_m, snapshot.down_m = 100.0, 200.0, -50.0
snapshot.vx_m_s = snapshot.vy_m_s = snapshot.vz_m_s = 0.0
for index in (1, 2, 3):
    frame = VideoFrame(
        index=index, image=np.zeros((180, 320, 3), dtype=np.uint8), timestamp=time.time(), lag=0.15
    )
    buffer.put(
        AlignedSample(
            frame=frame, snapshot=snapshot, timestamp=frame.timestamp, lag=0.15, mode="interpolate"
        )
    )


class _FakeDetector:  # 真项目里换成 Detector()
    def load(self):
        return self

    def detect(
        self,
        image,
        *,
        frame_index=0,
        capture_timestamp=0.0,
        telemetry=None,
        mode="",
        undistort=None,
    ):
        detection = Detection(
            frame_index=frame_index,
            capture_timestamp=capture_timestamp,
            pixel=(160.0, 90.0),
            box=PixelBox(150.0, 80.0, 170.0, 100.0),
            confidence=0.9,
            telemetry=telemetry,
            code=7,
            mode=mode,
        )
        return DetectionBatch(image=image, detections=(detection,), undistorted=False)


worker = PerceptionWorker(
    PerceptionConfigLike(mode="cls12"), buffer=buffer, detector=_FakeDetector()
)
worker.start()
try:
    deadline, results = time.time() + 5.0, []
    while time.time() < deadline and len(results) < 3:
        results.extend(worker.drain_results())
        time.sleep(0.05)
finally:
    worker.stop()
print("结果 %d 条，编号 %s" % (len(results), [d.code for d in results]))
```

实测输出：`结果 3 条，编号 [7, 7, 7]`。

### 3.5 `airdrop.georef` —— 像素到 NED 坐标

| 文件 | 内容 |
| --- | --- |
| `airdrop.georef.camera` | `CameraModel`、`default_camera_model`、`load_camera_model`、`rotation_x/y/z`、`euler_to_matrix` |
| `airdrop.georef.project` | 去畸变、像素 → 视线 → 地面交点、边长法估深与交叉验证 |
| `airdrop.georef.geo` | `LLARef`、`wgs84_to_ned`、`ned_to_wgs84`、`ned_distance` |

#### 3.5.1 `CameraModel`

```python
CameraModel(camera_matrix, dist_coeffs=<全零>, r_bc=<DEFAULT_R_BC>, t_bc=<零向量>,
            image_size=None, telemetry_lag=None, source=None)
```

| 成员 | 说明 |
| --- | --- |
| `camera_matrix` | 3×3 内参 |
| `dist_coeffs` | 畸变系数（OpenCV 顺序，默认全零） |
| `r_bc` | 相机系 → 机体系旋转（3×3），默认 `DEFAULT_R_BC`：光轴沿机体 z（朝下）、图像 x 向右、图像 y 向下 |
| `t_bc` | 相机在机体系里的安装位置（杆臂），默认零 |
| `image_size` | 标定分辨率 `(宽, 高)` |
| `telemetry_lag` | 标定出的链路延时（可为 `None`） |
| `source` | 来源说明（`'default'` / 标定文件路径等） |
| 属性/方法 | `fx`、`fy`、`cx`、`cy`、`focal_px`、`principal_point()`、`is_calibrated()`、`scaled(scale)` |

两个构造入口：

- `default_camera_model(width=1280, height=720)`：按视场角凑出的占位内参（`is_calibrated()` 为 `False`）。它只够"能跑通"，精度靠标定。
- `load_camera_model(path, *, fallback_size=(1280, 720))`：读 `tools/calibrate.py` 写出的标定文件；缺字段时用兜底尺寸补一个默认模型（不抛异常，便于缺标定也能起飞）。

#### 3.5.2 像素 → 坐标

| 函数 | 返回 |
| --- | --- |
| `undistort_pixel(pixel, camera)` | `(x, y)` 去畸变后的像素（无畸变时原样返回） |
| `pixel_to_ray_ned(pixel, *, camera, quaternion=None, euler_deg=None, undistort=True)` | 机体 NED 系下的单位视线向量 `(x, y, z)` |
| `pixel_to_ned(pixel, *, camera, ground_z, position_ned, quaternion=None, euler_deg=None, undistort=True)` | `GroundIntersection` |

`pixel_to_ned` 必需的两个"外部输入"是**飞机在原点 NED 系里的位置** `position_ned`（通常取快照的 `north_m`、`east_m`、`down_m`）与**地面高度** `ground_z`；姿态给四元数或欧拉角其中之一即可（两个都不给按单位姿态处理）。

`GroundIntersection` 字段：`ok`、`ned`（交点）、`reason`（失败原因）、`depth_m`（沿光轴的深度）、`distance_h_m`（相对飞机的水平距离）、`ray_ned`、`cam_origin_ned`（含杆臂修正后的相机位置）。失败语义是**显式返回 `ok=False` 且 `ned=None`**，绝不编一个坐标出来，`reason` 有两种：

- `视线朝上或接近水平` / `视线与地面夹角过小`：视线向下的分量 ≤ `MIN_DOWN_COS`（`0.05`），地面无限远或在背后；
- `解出的距离非正（…），检查 ground_z 与高度符号`：地面在相机上方，通常是 `ground_z` 或 `down_m` 符号弄反。

#### 3.5.3 边长法（独立第二来源）

目标物理边长已知时（`target_side_length_m`，默认 1.0 m），可以用像素边长独立估深度并与交点法互相验证：

| 函数 | 返回 |
| --- | --- |
| `estimate_depth_by_side(side_px, *, camera, side_m, quaternion=None, euler_deg=None)` | 深度估计，无法估算返回 `None` |
| `cross_check_by_side(*, side_px, side_m, depth_by_intersection_m, camera, tolerance=0.25, quaternion=None, euler_deg=None)` | `SideLengthCheck(ok, depth_by_side_m, depth_by_intersection_m, relative_error, reason)` |

`tolerance` 是相对误差门限（默认 0.25 = 25%）。**这个检查默认只当诊断**：`TargetTracker` 只记事件与计数，不剔点（见 §3.8.5）。

#### 3.5.4 姿态与经纬度

- `quaternion_to_matrix(w, x, y, z, *, normalize_input=True)`、`euler_to_matrix(roll_deg, pitch_deg, yaw_deg)`、`rotation_x/y/z(angle_deg)`、`normalize(vector)`。约定：欧拉角是**机体到 NED** 的旋转，顺序 roll → pitch → yaw。
- `LLARef(lon_deg, lat_deg, alt_m)`（注意字段顺序是**经度在前**）、`wgs84_to_ned(lon, lat, alt, ref)`、`ned_to_wgs84(ned, ref)`、`ned_distance(a, b)`。经纬度换算走 `pyproj`，`ref` 既可以是 `LLARef` 也可以是 `(lon, lat, alt)` 三元组。

**去畸变只做一次**（这条最容易出错）：检测器若已经 remap（检测器内参 `camera_matrix` 非空），像素就在纠正后的图上，坐标解算不能再来一次。`Detector.detect` 会把这件事写进 `Detection.extra["undistorted"]`，`TargetTracker(undistort=None)` 默认据此自动判断；手动调用 `pixel_to_ned` 时由 `undistort=` 显式控制。

```python
"""坐标解算：水平朝下看 + 主点像素 ⇒ 目标就在飞机正下方。"""

from airdrop import cross_check_by_side, default_camera_model, pixel_to_ned

camera = default_camera_model(320, 180)
result = pixel_to_ned(
    camera.principal_point(),
    camera=camera,
    ground_z=0.0,
    position_ned=(100.0, 200.0, -50.0),
    quaternion=(1.0, 0.0, 0.0, 0.0),
)
print("地面交点 NED:", result.ned, "深度 %.1f m" % result.depth_m)

# 边长法独立估深度：取与 50 m 自洽的像素边长做交叉验证
side_px = float(camera.camera_matrix[0, 0]) * 1.0 / result.depth_m
check = cross_check_by_side(
    side_px=side_px,
    side_m=1.0,
    depth_by_intersection_m=result.depth_m,
    camera=camera,
    quaternion=(1.0, 0.0, 0.0, 0.0),
)
print("边长法 ok=%s 相对误差 %.4f（side_px=%.2f）" % (check.ok, check.relative_error, side_px))
```

实测输出：`地面交点 NED: (100.0, 200.0, 0.0) 深度 50.0 m`；`边长法交叉验证 ok=True 相对误差 0.0000（side_px=5.54）`。

### 3.6 `airdrop.targeting` —— 目标统计（多点 → 唯一结果）

| 文件 | 内容 |
| --- | --- |
| `airdrop.targeting.models` | `TargetPoint`、`Cluster`、`TargetingResult` |
| `airdrop.targeting.cluster` | `cluster_points`、`select_cluster`、`analyze`（实飞与回放共用的唯一入口） |

三个模型：

| 类型 | 字段 |
| --- | --- |
| `TargetPoint` | `north_m`、`east_m`、`capture_timestamp`、`frame_index=-1`、`code=None`、`confidence=1.0`、`down_m=0.0` |
| `Cluster` | `label`、`members`（成员序号）、`north_m`、`east_m`、`code`、`confidence` |
| `TargetingResult` | `selection_rule`、`clusters`（全部候选类）、`selected`（唯一结果）、`noise`（DBSCAN 噪声点）、`rejected`（被剔除的点） |

`TargetingResult` 有 `code` / `north_m` / `east_m` 之类的透传属性（`selected` 为 `None` 时相应为 `None`），判空请用 `result.selected is None`。

```python
analyze(points: Sequence[TargetPoint], config: TargetingConfig) -> TargetingResult
cluster_points(points, config) -> (clusters, noise, rejected)
select_cluster(clusters, config) -> Cluster | None
```

流程与取舍（完整实测结论记在 `cluster.py` 的模块 docstring 里）：

1. **先排序**：输入按 `(capture_timestamp, frame_index)` 排序后再聚类——DBSCAN 的标签号依赖到达顺序，不排序结果不可复现。
2. **先剔病态点**：坐标非有限（georef 病态给出 `inf`/`nan`）的点进 `rejected` 并计数，不参与聚类。
3. **DBSCAN**（`eps_m` 默认 0.75、`min_samples` 默认 2）：两条实测语义必须记住——`eps` 边界是**闭区间**（恰好等于 `eps` 的两点相连）；`sample_weight` 是**绝对权重**，直接进核心点判据 `Σw ≥ min_samples`，所以按置信度加权时必须先归一到均值 1（代码里用 `MIN_WEIGHT=1e-6` 兜零）。
4. **类标签取类内编号众数**；坐标取类内均值。
5. **跨类选唯一结果**：`selection_rule='median'`（在**去重后的编号**上取**下中位**）或 `'max'`（取最大标签）。只有**带编号**的类能参与（`None` 取不了中位数）；`require_label=False` 时无编号的类仍留在 `clusters` 里供核对，但不会被选中。⚠ 中位数按**去重编号**取：同一编号可能分裂成多个类（同一目标看到多次、野点另成一类），按"类"取中位会让重复编号改变结果。
6. **结果唯一性**：同一标签出现多个类时按"成员多 → 置信度高 → 坐标"定序，保证结果稳定。

```python
"""目标统计：DBSCAN 聚类 + 类内编号众数。"""

from airdrop import TargetPoint, TargetingConfig, analyze

points = [
    TargetPoint(north_m=10.0, east_m=20.0, capture_timestamp=1.0, frame_index=1, code=56),
    TargetPoint(north_m=10.2, east_m=20.1, capture_timestamp=1.1, frame_index=2, code=56),
    TargetPoint(north_m=10.1, east_m=19.9, capture_timestamp=1.2, frame_index=3, code=12),
    TargetPoint(north_m=80.0, east_m=90.0, capture_timestamp=1.3, frame_index=4, code=7),
]
result = analyze(points, TargetingConfig(eps_m=0.75, min_samples=2))
print("选中编号 %s @ NED (%.2f, %.2f)" % (result.code, result.north_m, result.east_m))
print(
    "候选类 %d 个，噪声 %d，剔除 %d"
    % (len(result.clusters), len(result.noise), len(result.rejected))
)
```

实测输出：`选中编号 56 @ NED (10.10, 20.00)`；`候选类 1 个，噪声 1，剔除 0`（第四点离得太远，成了噪声；读 12 的那帧投给了 56，众数生效）。

### 3.7 `airdrop.ballistics` —— 弹道、投放判据、投放记录与反演

| 文件 | 内容 |
| --- | --- |
| `airdrop.ballistics.model` | `BallisticsModel`（二次阻力 + RK4）、`Impact`、`isa_air_density`、`wind_from_snapshot` |
| `airdrop.ballistics.release` | `ReleaseJudge`、`ReleaseDecision`、`heading_unit_vector` |
| `airdrop.ballistics.drops` | `DropRecord`、`DropSample`、`ImpactMeasurement` 及读写/配对/重算函数 |
| `airdrop.ballistics.fit` | `FitConfig`、`FitResult`、`DropResidual`、`fit_ballistics` |

#### 3.7.1 `BallisticsModel`

```python
BallisticsModel(config: BallisticsConfig)        # 参数见 §4.2
drag_factor(velocity_rel, height_m, *, ground_altitude_m=0.0) -> float
terminal_velocity(height_m=0.0, *, ground_altitude_m=0.0) -> float
predict_impact(position_ned, velocity_ned, *, ground_z=0.0,
               ground_altitude_m=0.0, wind=(0.0, 0.0, 0.0)) -> Impact
```

`Impact` 字段：`ok`、`ned`（落点）、`flight_time_s`、`speed_m_s`、`reason`。失败**不编落点**：起点已在地面以下 → `reason='below_ground'`；积分超过 `MAX_FLIGHT_TIME_S`（120 s）仍未落地 → `reason='timeout'`；两种都是 `ok=False`、`ned=None`。

模型要点：

- 阻力按空气相对速度算，风作为空气速度参与；**密度基准是真实海拔**：`ground_altitude_m` 是地面平面的海拔（飞行时由 `MissionRunner` 传 GPS 原点/地面点海拔，反演时取投放记录的 `origin.alt_m − ground_z`）。关闭 `air_density_isa`（默认）时在**投放（初始）海拔**上算一次 ISA 密度、全弹道共用；打开后按 `ground_altitude_m + 离地高度` **逐级**求值（贵 26~30%，高海拔/大落差时更准）。不再有"地面=海平面"的假设；记录缺原点时退回 0，反演报告会告警。
- 定步长 RK4（`rk_dt` 默认 0.005 s）；**落点用线性插值定在穿越地面的瞬间**，不是"第一个 z ≥ 地面的步"。
- `wind_from_snapshot(snapshot, config)`：`wind_source='telemetry'` 时从快照取风；**取不到返回 `None`**，刻意与"风确实是 0"区分开；调用方按零风降级并只记一次日志。

#### 3.7.2 `ReleaseJudge` —— 投放判据

```python
ReleaseJudge(config: DropConfig, ballistics: BallisticsModel, overfly_heading_deg: float,
             ground_z=0.0, ground_altitude_m=0.0, wind_ned=None, on_event=None, summary_hz=5.0)
update(snapshot, target_ned, *, ground_z=None, ground_altitude_m=None,
       wind_ned=None, now=None) -> ReleaseDecision
reset()
```

`ReleaseDecision` 字段：`should_release`、`reason`、`timestamp`、`predicted`（预测落点）、`horizontal_error_m`、`passed_target`、`approached`、`release_position`、`delay_s`。

判定逻辑：

1. 每拍用当前快照预测落点，**水平误差 ≤ `radius_m` 即投**（`reason='predict'`）；
2. **本次飞掠里先从目标前方接近过**（`approached=True`）且已越过目标、`force_after_pass=True` 时，执行强制投放（`reason='fallback'`）。⚠ "越过"只看沿航向的投影符号，没有 `approached` 前提的话，进入飞掠时飞机投影已在目标后方（例如刚结束盘旋、还在飞往入场点）会在第一拍假触发（实测 r2 世界架次 某架次：113 m 误差）；
3. **一次投放即锁存**：投过之后不再触发，`reset()` 才能重新武装（`approached` 也一并清零，每次飞掠各自重新计）；
4. `delay_s` 的状态前推是一阶近似（位置 += 速度 × 延迟），不做加速度二次项；
5. 摘要按 `SUMMARY_HZ`（5 Hz）落事件日志，触发瞬间写完整预测（落点/飞行时间/水平误差/前推后位置）。`on_event` 签名 `(kind, data)`，接 recorder 的标准写法是 `lambda kind, data: recorder.events.emit(kind, **data)`；事件写盘失败**不影响判据**。

#### 3.7.3 `drops` —— 投放记录与实测落点

| 名称 | 说明 |
| --- | --- |
| `DropRecord` | 一次投放的完整瞬间状态（见下表） |
| `DropSample` | 配对结果：`record` + `impact_ned` + `impact_source` + `label` |
| `ImpactMeasurement` | 实测落点：`index`、`lat_deg`/`lon_deg`/`alt_m` 或 `north_m`/`east_m`/`down_m`、`note` |
| `append_drop(path, record)` | 追加一条到 `drops.jsonl` |
| `load_drops(path)` / `load_impacts(path)` | 读回元组 |
| `match_impacts(records, measurements=())` | 按序号配对（缺失即报错，不猜） |
| `write_impact_template(path, records)` | 生成待填的实测落点模板 |
| `release_conditions(record, *, delay_s=None, offset_body_m=None)` | 求出释放瞬间的（位置, 速度） |
| `resolve_wind(record, *, scale=1.0)` | 取出该次投放的风（可缩放） |
| `predict_record_impact(record, model, *, delay_s=None, offset_body_m=None, wind_scale=1.0, ground_z=None, ground_altitude_m=None)` | 用给定模型重算落点，返回 `Impact`（密度基准缺省取记录原点海拔） |
| `attitude_matrix(record)` | 由四元数或欧拉角还原姿态矩阵（都没有则 `None`） |

`DropRecord` 字段分四组：

- 身份与时刻：`index`、`timestamp`、`reason`、`delay_s`；
- 投放瞬间状态：`position_ned`、`velocity_ned`、`ground_z`、`roll_deg`、`pitch_deg`、`yaw_deg`、`quaternion_wxyz`、`wind_ned`、`origin`、`target_ned`；
- 当时的预测：`release_position_ned`、`predicted_impact_ned`、`predicted_flight_time_s`、`predicted_error_m`；
- 参数与实测：`ballistics`（当时用的参数快照）、`impact_ned`、`impact_source`。

属性/方法：`from_snapshot(...)`（从遥测快照建记录）、`from_dict`/`as_dict`、`height_agl_m`、`ground_altitude_m`（地面海拔 = `origin.alt_m − ground_z`，密度的基准）、`horizontal_speed_m_s`、`heading_deg`、`euler_deg`。文件名常量：`DROPS_NAME='drops.jsonl'`、`IMPACTS_JSONL_NAME='impacts.jsonl'`、`IMPACTS_CSV_NAME='impacts.csv'`。

#### 3.7.4 `fit` —— 弹道参数反演

```python
fit_ballistics(samples: Sequence[DropSample], config: FitConfig | None = None) -> FitResult
FIT_PARAMETERS = ('drag_coefficient', 'cross_area_m2', 'release_delay_s', 'wind_scale')
```

`FitConfig` 默认只拟合阻力系数（`fit_drag_coefficient=True`，其余三个 `False`），并带上下界与这些开关：

`drag_coefficient_bounds=(0.05, 3.0)`、`cross_area_bounds=(0.0001, 0.5)`、`release_delay_bounds=(-1.0, 2.0)`、`wind_scale_bounds=(0.0, 3.0)`、`release_offset_body_m=(0, 0, 0)`、`max_condition_number=10000`、`allow_underdetermined=False`、`allow_ill_conditioned=False`、`leave_one_out=True`、`loss='linear'`、`f_scale=1.0`、`max_nfev=200`、`diff_step=1e-4`。

`FitResult` 关键字段：`ok`、`reliable`、`reason`、`fitted`（实际拟合的参数名）、`parameters`、`ballistics`（拟合后的参数对象）、`release_delay_s`、`wind_scale`、`drag_k_per_m`（κ = Cd·A/m）、`drag_area_m2`、`drag_area_sigma_m2`、`n_samples`、`n_equations`、`dof`、`rms_error_m`、`max_error_m`、`bias_north_m`、`bias_east_m`、`per_drop`（`DropResidual` 列表）、`sigma`、`correlation`、`condition_number`、`at_bound`、`leave_one_out_rms_m`、`leave_one_out_max_m`、`leave_one_out_errors_m`、`warnings`、`nfev`。

四条必须记住的结论（完整推导在 [`docs/ballistics_fit.md`](ballistics_fit.md)）：

- **质量是称出来的，不进反演**：数据能识别的只有 κ = Cd·A/m 一个数，所以质量填错多少，Cd 与 Cd·A 就跟着错多少，κ 不变。
- **同高同速下 Cd 与释放延迟不可辨识**（条件数实测 1.06e10，会被直接拒绝）：要分开就得把速度或高度拉开；有风反而容易分离。
- **密度基准取记录的原点海拔**（`origin.alt_m − ground_z`，见 §3.7.1；缺原点退回 0 并在 `FitResult.warnings` 告警）：同一站点自洽；改过密度基准（或换站点/海拔）后重跑旧数据，Cd 会按密度差等比变化。
- 有常数偏差时先看 `bias_north_m`/`bias_east_m`，别急着加参数。

运行入口：`python -m airdrop.run fit-ballistics`（见 §6.2；`tools/fit_ballistics.py` 里的常量就是默认值）。

```python
"""弹道 + 投放判据：预测落点、越过目标后强制投放。"""

import time
from airdrop import BallisticsModel, Config, DropConfig, ReleaseJudge, TelemetrySnapshot

model = BallisticsModel(Config().ballistics)
impact = model.predict_impact((0.0, 0.0, -20.0), (18.0, 0.0, 0.0), ground_z=0.0)
print("落点 %s，飞行时间 %.2fs" % (impact.ned, impact.flight_time_s))

judge = ReleaseJudge(DropConfig(radius_m=2.0), model, overfly_heading_deg=90.0)
snapshot = TelemetrySnapshot(timestamp=time.time())
snapshot.north_m, snapshot.east_m, snapshot.down_m = 0.0, -35.0, -20.0
snapshot.vx_m_s, snapshot.vy_m_s, snapshot.vz_m_s = 0.0, 18.0, 0.0
decision = judge.update(snapshot, (0.0, 0.0, 0.0), now=snapshot.timestamp)
print(
    "判据: should_release=%s reason=%s 误差 %.2f m"
    % (decision.should_release, decision.reason, decision.horizontal_error_m or -1.0)
)
```

实测输出：`落点 (34.5662493172948, 0.0, 0.0)，飞行时间 2.07s`；`判据: should_release=True reason=predict 误差 0.43 m`（同时 stderr 会有一条"遥测里没有风估计…按零风预测"的提示，属预期）。

```python
"""投放记录 + 反演：合成两条样本，反演出 Cd。"""

from dataclasses import replace
from airdrop import (
    BallisticsModel,
    Config,
    DropRecord,
    DropSample,
    FitConfig,
    fit_ballistics,
    predict_record_impact,
)

truth = replace(Config().ballistics, drag_coefficient=0.8)  # 真值 Cd=0.8
true_model = BallisticsModel(truth)
records = [
    DropRecord(
        index=i,
        timestamp=1000.0 + i,
        position_ned=(100.0, 200.0, -height),
        velocity_ned=(speed, 0.0, -1.0),
        ground_z=0.0,
        delay_s=0.05,
        ballistics=Config().ballistics,
    )  # 记录里存的是占位参数
    for i, (height, speed) in enumerate([(25.0, 12.0), (45.0, 20.0)], start=1)
]
samples = []
for record in records:
    impact = predict_record_impact(record, true_model, delay_s=0.05)
    assert impact.ok and impact.ned is not None
    samples.append(DropSample(record=record, impact_ned=impact.ned, label=f"#{record.index}"))

result = fit_ballistics(samples, FitConfig())
print("ok=%s reliable=%s（拟合参数 %s）" % (result.ok, result.reliable, list(result.fitted)))
print(
    "Cd=%.4f（真值 0.8），κ=Cd·A/m=%.5f，RMS=%.1e m"
    % (result.parameters["drag_coefficient"], result.drag_k_per_m, result.rms_error_m)
)
```

实测输出：`ok=True reliable=True（拟合参数 ['drag_coefficient']）`；`Cd=0.8000（真值 0.8），κ=Cd·A/m=0.00877，RMS=4.5e-13 m`。

### 3.8 `airdrop.mission` —— 状态机、航线、目标跟踪与主循环

| 文件 | 内容 |
| --- | --- |
| `airdrop.mission.items` | `MissionItem`（**MAVLink 级任务项**）、`MAV_CMD_*`/`MAV_FRAME_*` 常量、`UNSET`、`command_name` |
| `airdrop.mission.plan_file` | QGC `.plan` 解析（含 `fwLandingPattern` 复杂项展开）、`QgcPlan`、`PlanError`、`check_fixed_wing_landing` |
| `airdrop.mission.states` | `MissionState`、`TRANSITIONS`、`TERMINAL_STATES`、`MissionStateMachine`、`MissionTransition`、`InvalidTransition`、`emit_event` |
| `airdrop.mission.planner` | 纯函数：侦查航线、飞掠 entry/exit、飞掠+降落合并任务、WGS84↔NED |
| `airdrop.mission.targets` | `TargetTracker`（检测 → 目标点 → 统计）、`PerceptionTargetSource`（包成主循环要的两个回调） |
| `airdrop.mission.runner` | `MissionRunner`（主循环）、`MissionStats`、`MissionMonitor`、两个协议 |

#### 3.8.1 `states` —— 状态与合法转移

`MissionState`（`StrEnum`）：`INIT`、`PREFLIGHT`、`WAIT_AIRBORNE`、`RECON`、`HOLD_PROCESS`、`OVERFLY`、`LAND`、`DONE`、`ABORT`。

`TRANSITIONS` 是**唯一**的合法边表（改流程只改这一处）：

| 从 | 可到 |
| --- | --- |
| `INIT` | `PREFLIGHT`、`ABORT` |
| `PREFLIGHT` | `WAIT_AIRBORNE`、`ABORT` |
| `WAIT_AIRBORNE` | `RECON`、`ABORT` |
| `RECON` | `HOLD_PROCESS`、`ABORT` |
| `HOLD_PROCESS` | `OVERFLY`、`ABORT` |
| `OVERFLY` | `LAND`、`ABORT` |
| `LAND` | `DONE`、`ABORT` |
| `DONE` / `ABORT` | 无（`TERMINAL_STATES`） |

```python
MissionStateMachine(state=MissionState.INIT, on_event=None, history=[])
  .transition(to_state, *, reason='', now=None) -> MissionTransition
  .can_transition(to_state) -> bool
MissionTransition(from_state, to_state, timestamp, reason='')
InvalidTransition        # 非法转移抛它，且**状态不变**
emit_event(on_event, kind, data)   # on_event 为空时是空操作
```

每次成功转移记一条 `state` 事件（含 `from_state`/`to_state`/`reason`/`timestamp`），recorder 抓到的就是它。

#### 3.8.2 `items` —— 任务项的唯一表示（MAVLink 级）

```python
MissionItem(command, lat=0.0, lon=0.0, alt_m=0.0, frame=MAV_FRAME_GLOBAL_RELATIVE_ALT,
            param1=UNSET, param2=UNSET, param3=UNSET, param4=UNSET,
            autocontinue=True, do_jump_id=0)
  .waypoint(lat, lon, alt_m, *, acceptance_radius_m=None, frame=..., yaw_deg=None)
  .takeoff(lat, lon, alt_m, *, pitch_deg=15.0, yaw_deg=None, frame=...)
  .land(lat, lon, *, alt_m=0.0, frame=...)
  .from_waypoint(waypoint, *, land=False)
  .is_positional -> bool
  .as_dict() -> dict                 # 含 command_name，便于日志阅读
command_name(command) -> str         # 16 → 'NAV_WAYPOINT'；未知命令回十进制字符串
```

- **为什么不用 MAVSDK 的 `MissionItem`**：那层翻译会把 `vehicle_action=LAND` 的**一项拆成两项**
  （同坐标 `NAV_WAYPOINT` + `NAV_LAND`），PX4 固定翼的可行性检查随即拒掉**整条任务**
  （`the approach waypoint must be above the landing point`），而 `start_mission()` 仍回成功。
  任务项一律走 `mission_raw` 之后，这层翻译就不存在了（见 §7 第 31/32 条）。
- `UNSET`（`float('nan')`）表示"不指定"；**0 是有意义的值**——`NAV_WAYPOINT` 的 `param2=0` 表示
  "用飞控参数 `NAV_ACC_RAD`"，与"接受半径 0 米"不是一回事。
- 生成航点默认 `DEFAULT_WAYPOINT_ACCEPTANCE_M = 3.0`（米，与旧实现里 MAVSDK `UNSET` 的实际落值
  一致，换实现不能让"到点判据"悄悄变宽）。⚠ 固定翼上 3 米偏紧（见 §9.1）；QGC `.plan` 里的航点
  自带各自的 `param2`，**原样使用**。
- `is_positional` 只对带坐标的命令为真（`NAV_WAYPOINT`、`NAV_LOITER_UNLIM`、`NAV_LOITER_TIME`、
  `NAV_LAND`、`NAV_TAKEOFF`、`NAV_LOITER_TO_ALT`），`DO_LAND_START` 这类指令项不在内——
  它的位置字段是 0，**不能**拿去判"纬度非法"。
- 高度口径由 `frame` 决定：`MAV_FRAME_GLOBAL_RELATIVE_ALT`（默认）是相对起飞点，
  `MAV_FRAME_GLOBAL` 是海拔。
- 模块级常量：`MAV_CMD_NAV_WAYPOINT`(16)、`MAV_CMD_NAV_LOITER_UNLIM`(17)、`MAV_CMD_NAV_LOITER_TIME`(19)、
  `MAV_CMD_NAV_LAND`(21)、`MAV_CMD_NAV_TAKEOFF`(22)、`MAV_CMD_NAV_LOITER_TO_ALT`(31)、
  `MAV_CMD_DO_CHANGE_SPEED`(178)、`MAV_CMD_DO_LAND_START`(189)、`MAV_CMD_DO_SET_CAM_TRIGG_DIST`(206)、
  `MAV_CMD_IMAGE_STOP_CAPTURE`(2001)、`MAV_CMD_VIDEO_STOP_CAPTURE`(2501)；
  `MAV_FRAME_GLOBAL`(0)、`MAV_FRAME_MISSION`(2)、`MAV_FRAME_GLOBAL_RELATIVE_ALT`(3)、
  `MAV_FRAME_GLOBAL_TERRAIN_ALT`(10)（都在 §4.17 有表）。

#### 3.8.3 `plan_file` —— QGC `.plan` 解析 + 固定翼降落预检

```python
load_plan(path) -> QgcPlan            # 纯 JSON：不连飞控、不需要 MAVSDK
QgcPlan(path, items, vehicle_type, firmware_type, cruise_speed_m_s, home_position)
  .as_dict() -> dict                  # 条数 + 命令名序列 + 机身/固件类型 + home
check_fixed_wing_landing(items, *, preceding=None,
                         land_angle_deg=FW_DEFAULT_LAND_ANGLE_DEG) -> tuple[str, ...]
PlanError(ValueError)                 # 读不了 / 结构不认识 / 内容不合法（消息带文件路径与原因）
PLAN_FIRMWARE_PX4 = 12                # QGC mission.firmwareType
PLAN_VEHICLE_FIXED_WING = 1           # QGC mission.vehicleType
FW_DEFAULT_LAND_ANGLE_DEG = 8.0       # PX4 FW_LND_ANG 出厂默认
```

- **复杂项在本地展开**：QGC 把"固定翼降落航线"存成一个 `ComplexItem`。本模块照 QGC
  `LandingComplexItem::appendMissionItems`（`qgroundcontrol-master/src/MissionManager/LandingComplexItem.cc`）
  展开成 `DO_LAND_START` →（可选 `DO_CHANGE_SPEED`、停止拍照/录像）→ 进场项 → `NAV_LAND`；
  `useLoiterToAlt=True` 时进场项是 `NAV_LOITER_TO_ALT`（`param2` = 盘旋半径，顺时针为正），
  否则是 `NAV_WAYPOINT`。PX4 的 `DO_LAND_START` 是 `specifiesCoordinate: false`，所以**不带坐标**、
  `frame=MAV_FRAME_MISSION`、参数全 0。只支持 `fwLandingPattern`——VTOL 降落、测绘/结构航线
  **显式报错**，不猜着展开。
- **降落预检照 PX4 的判据**（`MissionFeasibility/FeasibilityChecker.cpp`）：紧前一项必须**严格高于**
  落点；下滑**斜率**（tan，垂直差/水平距离）≤ `tan(FW_LND_ANG + 0.1°)`（默认 8° ⇒ 上限约 0.142，
  即约 8.1°）；进场项只能是 `NAV_WAYPOINT` 或 `NAV_LOITER_TO_ALT`（后者按
  `sqrt(圆心距² − 半径²)` 修正水平距离，与 PX4 一致）；`LOITER_TO_ALT` 进场时落点必须在盘旋圈**外**。
  距离用与 PX4 `get_distance_to_next_waypoint` 同款的**球面 haversine**（半径 6371000，
  `_EARTH_RADIUS_M`）——刻意不换椭球/Geod，保证预检与飞控是同一把尺子。
  `preceding` 用来说明"`items` 之前还有一项"（飞掠段插在降落段前面时就是飞掠段的 exit）。
- 不合格的航线在**规划阶段**就带着原因失败（`PlanningError`），而不是上传后被飞控整条拒掉、
  飞机原地盘旋——2026-09 的 SITL 演练里为此空等了 15 分钟。

#### 3.8.4 `planner` —— 纯函数航线规划

```python
llaref_of(origin: NedOrigin) -> LLARef
waypoint_to_ned(waypoint, origin) -> (n, e, d)
overfly_positions(target_ned, *, heading_deg, leg_length_m) -> (entry_ned, exit_ned)
overfly_waypoints(target_ned, *, heading_deg, leg_length_m, altitude_m, origin) -> (entry, exit)
build_recon_mission(config) -> tuple[MissionItem, ...]
build_drop_mission(config, *, origin, target_ned=None) -> DropMissionPlan
```

- 飞掠段以目标为中心、沿配置航向前后各半段长生成 `[entry, exit]`，**方向与判据的"越过目标"一致**（符号不能反）。
- **每条腿的来源二选一**（`Config.validated()` 保证同时给两个来源会报错）：
  - 侦查段：`RoutesConfig.recon_plan`（QGC `.plan`，**原样使用**）或 `RoutesConfig.recon_route`
    （配置航点，`takeoff_first=True` 时首项做成 `MissionItem.takeoff`）；
  - 降落段：`RoutesConfig.land_plan`（QGC `.plan`，**飞掠段插在它前面**）或 `RoutesConfig.landing_route`
    （配置航点，`land_last=True` 时末项做成 `MissionItem.land`）。
- plan 模式下**不自动补**起飞/降落项：plan 里没有起飞项就只告警（飞控不会自动起飞，需要操作手先起飞）；
  补出来的项会留下"表面上能飞"的隐患。
- 降落段一定先过 `check_fixed_wing_landing`（`preceding` = 飞掠段 exit）：不合格抛 `PlanningError`。
- `build_drop_mission` 的目标来源：`target_ned` 给出时 `source='target'`；否则用 `RoutesConfig.backup_point`
  （`source='backup'`）；两者都没有 → `PlanningError`（**不猜**）。任务项 = 飞掠 2 项 + 降落段。
- `DropMissionPlan` 字段：`items`、`target_ned`、`source`、`heading_deg`、`entry`、`exit`、`tail_source`
  （`'route'` 或 `'plan:<文件>'`），另有 `overfly_count`/`landing_count` 与 `as_dict()`。
- 高度口径：`Waypoint.alt_m` 与 `MissionItem.alt_m`（`frame` 为相对高度时）都是**相对起飞点**的高度；
  飞掠高度直接取 `OverflyConfig.altitude_m`，不做高程换算。备用点换算成 NED 时海拔按"原点海拔 +
  相对高度"给（`GROUND_DOWN_M=0.0` 只用于 `waypoint_to_ned` 的 down 分量）。

#### 3.8.5 `targets` —— 检测 → 目标点 → 统计

```python
TargetTracker(config, camera, undistort=None, side_check_tolerance=0.25,
              reject_on_side_mismatch=False, on_event=None)
  .add(detection) -> TargetPoint | None      # 解算失败/缺字段返回 None
  .extend(detections) -> int                 # 成功条数
  .result() -> TargetingResult               # 内部就是 analyze(points, config.targeting)
  .reset()
  .ground_z(snapshot) -> float

PerceptionTargetSource(worker, tracker)
  .pump() -> int        # 抽干 worker 结果队列并送入 tracker
  .result() -> TargetingResult
  .busy() -> bool       # worker 或 tracker 还有活
```

行为约定：

- **缺姿态或位置的点不猜**：分别计入 `attitude_missing` / `no_fix` 后跳过（计数可从 tracker 的统计读到）。
- **边长交叉验证默认只当诊断**：超门限记 `side_check` 事件与计数，**不剔点**（默认 `reject_on_side_mismatch=False`；误判一个真实观测比多一个野点的代价大）。`DEFAULT_SIDE_TOLERANCE=0.25` 与 `cross_check_by_side` 的默认一致。
- `undistort=None` 时按 `Detection.extra["undistorted"]` 自动判断要不要去畸变（见 3.5.4）。
- `ground_z(snapshot)` 只依赖配置里的地面点参数；给不出时返回 0（原点高度面），由调用方决定记不记警告。

#### 3.8.6 `runner` —— 主循环

```python
MissionRunner(config, controller, broker, *, target_result=None, target_busy=None,
              release_judge=None, on_event=None, on_drop=None, preflight=None,
              clock=time.time, sleep=time.sleep)
  .run(*, max_ticks=None) -> MissionState     # 按 tick_hz 推进，直到终态/stop/拍数上限
  .update(now=None) -> MissionState           # 单拍
  .stop(reason='stopped')                     # 请求退出；未结束按 ABORT 处理
  .abort(reason)                              # 外部故障也可直接调
```

`preflight` 是 :class:`~airdrop.preflight.PreflightLike`（默认 `None` = 记一条 `preflight_skipped`
后放过，**正式入口必须注入**）。`update()` 对 `PREFLIGHT`/`WAIT_AIRBORNE` 这两个**瞬时状态**
允许在同一拍里连跳（条件当场满足时），但每一跳都照常记进历史与事件。

只读视图：`state`、`machine`、`history`（转移元组）、`stats`、`origin`、`plan`、`targeting_result`、`monitor`、`drops`、`stopped`。`MissionStats` 字段：`ticks`、`uploads`、`releases`、`aborts`、`errors`；`MissionMonitor` 字段：`armed`、`false_seen`、`true_seen`。

各状态的每拍行为：

| 状态 | 进入时 | 每拍判断 | 离开 |
| --- | --- | --- | --- |
| `INIT` | - | 要求遥测里有位置 → 查 NED 原点（**不上传任何任务**） | 遥测有效 + 原点就绪 → `PREFLIGHT`（`telemetry_ready`）；`init_max_s` 超时 → `ABORT`（`init_no_telemetry`/`init_no_origin`） |
| `PREFLIGHT` | - | 跑一次预检 `self._preflight.run()`：载入 detector/ocr/camera → 视频自检 | 全部通过 → `WAIT_AIRBORNE`（`preflight_ok`）；任一项失败 → `ABORT`（`preflight_failed:<check>`）；超 `preflight.max_s` → `preflight_timeout`；**没注入预检**（离线测试）→ 记 `preflight_skipped` 后放过 |
| `WAIT_AIRBORNE` | - | 什么都不下发；`in_air` 为主，取不到时用 `relative_altitude_m >= airborne_alt_m` 兜底 | 判定在空中 → 记 `airborne` 事件（带 `source`）→ `RECON`；`airborne_timeout_s`（默认 1800s）超时 → `ABORT`（`airborne_timeout`） |
| `RECON` | `recon_upload="auto"` 时上传并启动侦查航线；`"operator"`（默认，正式任务）**不上传**，记一条 `recon_waiting_operator` 后等操作手在 QGC 启动 | **先确认任务真在跑**（`_confirm_started`，见下）→ 链路看门狗 + 查询任务是否飞完 | 飞完 → `HOLD_PROCESS`（`recon_finished`）；`recon_max_s` 超时 → `ABORT`（`recon_timeout`） |
| `HOLD_PROCESS` | 下 `hold` | 读目标统计：有结果（`result.ok`）或到 `hold_process_max_s`，或已过 `hold_process_min_s` 且 `target_busy()` 为假 | 结束等待 → 建投放航线（无目标就用备用点）→ 上传启动 → `OVERFLY`；`PlanningError` → `ABORT` |
| `OVERFLY` | 记兜底上限 `land_max_s`；`judge.reset()` | **先确认任务真在跑** → 再跑投放判据：`should_release` 就投放 | 投放成功 → `LAND`；任务飞完却没投出去 → `ABORT`（`overfly_finished_without_release`，这是**失败**不是完成）；超时 → `overfly_timeout` |
| `LAND` | - | 查询任务是否飞完 | 飞完 → `DONE`；超时 → `land_timeout` |
| `ABORT` | 计数 + 下安全动作（`abort_action`） | - | 终态 |

**启动确认（`_confirm_started`）**：`upload_mission` + `start_mission` 之后，状态机在**有限时间**
（`MissionConfig.mission_start_timeout_s`，默认 20s）内要求 `in_mission_mode()` 为真；确认成功记一条
`mission_confirmed` 事件，超时则 `ABORT("mission_not_started")`。**"命令回成功"不等于"飞控进了任务模式"**
——飞控可能拒绝模式切换却仍然回 ACK，飞机继续盘旋（2026-09 SITL 里因此空等 15 分钟）。模式流不可用时
（`in_mission_mode()` 抛 `ControllerError`）**退化为不做确认**并只记一次日志：宁可退回旧行为，也不要
因为读不到模式就把任务判失败。

失败语义分五类（这条是审查重点）：

1. **命令失败**（上传/启动/hold/rtl/投放）→ 直接进 `ABORT`，原因写事件日志（`_fail` → `abort("<where>_failed")`）。
2. **查询失败**（`mission_finished`、`in_mission_mode`、`request_origin`）→ 只报**一次**日志与事件，然后照常继续，由状态超时兜底（20 Hz 每拍都报会掩盖真正原因）。
3. **启动未确认**（模式在 `mission_start_timeout_s` 内没进 `MISSION`）→ `ABORT`（`mission_not_started`）；模式流不可用时**不做这项判断**（只记一次告警）。
4. **链路异常**（遥测无效，或陈旧超过 `telemetry_stale_s`）→ `ABORT`（`telemetry_lost`）。
5. **投放记录失败不影响状态**：投放已经发生，记录写不进去只记日志与 `error` 事件（见 `_record_drop`）。
6. **自检失败**（模型/标定载入抛异常、视频自检窗口内帧数不够或分辨率与标定不一致）→ `ABORT`（`preflight_failed:<check>`）；预检整体超 `preflight.max_s` → `preflight_timeout`。**检查被关掉不算失败**，但会记一条 `preflight` 事件（`ok=null`）；**没注入预检对象**只记 `preflight_skipped` 并告警（正式入口必须注入）。
7. **等起飞超时**（`WAIT_AIRBORNE` 超过 `airborne_timeout_s`）→ `ABORT`（`airborne_timeout`）。

其他要点：

- `release_judge=None` 时不自动投放：记 `drop_skipped`（`reason='no_judge'`）直接转 `LAND`——只飞航线。
- 投放成功时把**判据看到的那份快照**一起写进 `DropRecord`（不是重新取一份），免得记录与决策之间插进一拍遥测导致残差归因错位；记录经 `on_drop` 回调交给 recorder 写入磁盘（`on_drop=recorder.drops.append`）。
- `MissionMonitor` 要求"**先见到一次 `False` 再认 `True`**"：任务进度是飞控侧的**状态**而不是"本次启动之后的事件"——新任务刚上传、还没启动时，它可能仍是上一条任务留下的"已飞完"（`current == total`）。
- 状态是**在某一拍结束时**进入的：测试里"进入新状态后要发生的事"需要再调一次 `update()`。
- 事件类型（runner 产生）：`origin`、`targeting`、`drop_plan`、`mission_confirmed`、`drop`、`drop_record`、`drop_skipped`、`abort_action`、`error`、`preflight_skipped`、`airborne`、`recon_waiting_operator`，加上状态机自己的 `state`（逐项检查的 `preflight` 事件由 `Preflight` 自己发，见 §5.2）。
- 主循环**不自己起线程**；`clock`/`sleep` 也是注入的，所以测试能把分钟级任务压成毫秒级确定性跑完。

```python
"""航线规划（纯函数）：侦查任务项 + 飞掠/降落合并任务。"""

from airdrop import (
    Config,
    GroundConfig,
    LLARef,
    MissionConfig,
    OverflyConfig,
    PreflightConfig,
    RoutesConfig,
    Waypoint,
    build_drop_mission,
    build_recon_mission,
    command_name,
)

config = Config(
    routes=RoutesConfig(
        recon_route=(
            Waypoint(lat=47.0, lon=8.0, alt_m=60.0),
            Waypoint(lat=47.001, lon=8.0, alt_m=60.0),
        ),
        backup_point=Waypoint(lat=47.05, lon=8.05, alt_m=0.0),
        landing_route=(Waypoint(lat=46.999, lon=8.01, alt_m=0.0),),
    ),
    overfly=OverflyConfig(heading_deg=90.0, altitude_m=20.0, leg_length_m=200.0),
    ground=GroundConfig(ground_point_alt=500.0),
    # 离线示例：侦查航线由示例自己上传（auto）；正式任务用默认的 operator（操作手在 QGC 启动）
    mission=MissionConfig(recon_upload="auto"),
    # 自检四项全关（示例里没有相机/模型）：**关掉 ≠ 通过**，只是离线跑通
    preflight=PreflightConfig(
        load_detector=False, load_ocr=False, load_camera=False, check_video=False
    ),
).validated()

items = build_recon_mission(config)
print("  侦查任务项 %d 个，首项命令 %s" % (len(items), command_name(items[0].command)))
origin = LLARef(lon_deg=8.0, lat_deg=47.0, alt_m=500.0)
plan = build_drop_mission(config, origin=origin, target_ned=(300.0, 0.0, 0.0))
print(
    "  投放任务：来源 %s，任务项 %d 个，目标 NED %s"
    % (plan.source, len(plan.items), plan.target_ned)
)
print("  飞掠 entry/exit: %.6f / %.6f" % (plan.entry.lat, plan.exit.lat))
```

实测输出：`  侦查任务项 2 个，首项命令 NAV_TAKEOFF`；`  投放任务：来源 target，任务项 3 个，目标 NED (300.0, 0.0, 0.0)`；`  飞掠 entry/exit: 47.002698 / 47.002698`（航向 90° 时 entry/exit 差在**经度**上，纬度与目标相同）。

```python
"""主循环：假控制器 + 假时钟（分钟级任务在毫秒级跑完）。"""

import time
from types import SimpleNamespace
from airdrop import MissionRunner, NedOrigin, TelemetryBroker


class _FakeClock:
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        return self.now


class _FakeController:  # 按 MissionController 协议实现
    def __init__(self):
        self.calls, self.finished = [], False

    def upload_mission(self, items, /):
        self.calls.append("upload_mission")
        return len(items)

    def start_mission(self):
        self.calls.append("start_mission")
        self.finished = False

    def in_mission_mode(self):  # 启动确认读它（真控制器读快照 flight_mode）
        return True

    def hold(self):
        self.calls.append("hold")

    def rtl(self):
        self.calls.append("rtl")

    def gripper_release(self):
        self.calls.append("gripper_release")
        return True

    def request_origin(self):
        return NedOrigin(lat_deg=47.0, lon_deg=8.0, alt_m=500.0)

    def mission_finished(self):
        return self.finished


clock = _FakeClock()
controller = _FakeController()
broker = TelemetryBroker()
broker.update_local_position_velocity(
    SimpleNamespace(north_m=0.0, east_m=-200.0, down_m=-40.0),
    SimpleNamespace(north_m_s=0.0, east_m_s=18.0, down_m_s=0.0),
)
broker.update_in_air(True)  # 假飞控报告"已离地"（WAIT_AIRBORNE 那道门读它）
runner = MissionRunner(config, controller, broker, clock=clock, sleep=lambda _s: None)
for _ in range(4):  # 4 拍走完 INIT → RECON
    runner.update()
    clock.now += 0.05
controller.finished = True  # 假飞控报告侦查任务飞完
runner.update()
print(
    "  状态 %s，上传 %d 次，历史 %d 条" % (runner.state, runner.stats.uploads, len(runner.history))
)
print(
    "  最后一次转移: %s → %s (%s)"
    % (runner.history[-1].from_state, runner.history[-1].to_state, runner.history[-1].reason)
)
```

实测输出（`config` 沿用 3.8.4 那份）：

```text
  状态 HOLD_PROCESS，上传 1 次，历史 4 条
  最后一次转移: RECON → HOLD_PROCESS (recon_finished)
```

（历史 4 条 = `PREFLIGHT` → `WAIT_AIRBORNE` → `RECON` → `HOLD_PROCESS`：第 1 拍里
`INIT` 拿到遥测与原点后就一路连跳到 `RECON`——`PREFLIGHT`/`WAIT_AIRBORNE` 是**瞬时**状态，
条件当场满足就不占满一拍；这两跳同样会记进历史与事件。）

#### 3.8.7 控制器协议（主循环看到的接口）

`MissionRunner` 只依赖这几件事（`airdrop.telemetry.controller` 里的 `MissionController` 协议；详见 §3.1）：

| 调用 | 语义 |
| --- | --- |
| `upload_mission(items, /)` | 上传任务项（`MissionItem` 序列），返回条数；失败抛 `ControllerError` |
| `start_mission()` | 复位到第 0 项并启动已上传的任务 |
| `in_mission_mode()` | 飞控是否真的在任务模式（启动确认用）；模式流未就绪抛 `ControllerError` |
| `hold()` / `rtl()` | 悬停 / 返航 |
| `gripper_release()` | 投放；返回 `False` 表示"没发出去"（也按失败处理） |
| `request_origin()` | 查 NED 原点；返回 `NedOrigin` 或 `None`（尚未就绪，**不是**失败） |
| `mission_finished()` | 查任务是否飞完（`mission_current == mission_total`）；进度流未就绪抛 `ControllerError` |

`MissionItem` 字段：`command`、`lat`、`lon`、`alt_m`、`frame`、`param1`~`param4`、`autocontinue`、`do_jump_id`；
"不指定"用 `UNSET`（`float('nan')`）而不是 `None`——**0 是有意义的值**（接受半径 0 = 用飞控默认、盘旋 0 秒）。
构造器 `MissionItem.waypoint/takeoff/land/from_waypoint` 覆盖规划器要用的三种项；`to_raw_item(item, index)`
把它转成 MAVSDK 的 raw 项（`airdrop.telemetry.controller`）。`DryRunController` 只拦投放（记 `drop_dry_run`
事件并返回 `True`），航线/hold/rtl/查询全部委托，供 SITL 与"不装弹"演练走完整状态机；**真机装弹时绝不能用**。

### 3.9 `airdrop.config` 与顶层导出面

`Config` 是唯一的参数集合：全部 frozen dataclass，15 个分块字段（`telemetry`、`video`、`align`、`perception`、`camera`、`ground`、`targeting`、`ballistics`、`drop`、`overfly`、`gripper`、`routes`、`mission`、`preflight`、`record`），逐字段说明见 §4。

```python
Config(...)  # 直接构造；分块各自有默认值
config.validated()  # 返回校验通过的同一个对象；不合法抛 ValueError
config.replace(**changes)  # 派生一个变体（改哪个传哪个，如 drop=DropConfig(radius_m=3.0)）
```

`validated()` 的校验清单（`airdrop/config.py` 的 `Config.validated`，行 381–462）：枚举字段取值（`mode`、`target_color`、`selection_rule`、`on_timeout`、`wind_source`、`abort_action`）、`ocr_workers >= 1`、`ocr_queue_size >= 1`、`imgsz >= 32`、`min_side_px < max_side_px`、去重参数 `>= 0`、`eps_m > 0`、`radius_m > 0`、各任务时长/频率为正、`hold_process_min_s <= hold_process_max_s`、gripper 参数 `>= 0`，最后调 `self.video.validated()`（传输方式与编码参数）。**校验只在调用 `validated()` 时发生**：直接构造的 `Config()` 不校验，装配函数里记得调。

顶层 `airdrop` 把常用名字重新导出（`airdrop.__all__` 共 176 个）。分组清单如下（要 `import` 什么都从这里找）：

- 分包与版本：`Config`、`__version__`
- 遥测：`TelemetryBroker`、`TelemetrySnapshot`、`MavsdkThread`、`DroneController`、`DryRunController`、`Command`、`CommandResult`、`ControllerError`、`MissionController`、`NedOrigin`、`to_raw_item`、`MISSION_TYPE_MISSION`
- 图传：`Hm30VideoSource`、`VideoFrame`、`VideoStats`、`open_hm30_video`、`FrameTelemetryAligner`、`AlignedSample`、`TelemetryPacer`、`SUPPORTED_QUERY_MODES`、`DEFAULT_TELEMETRY_LAG`、`HM30_CAMERA_IP`、`HM30_GROUND_IP`、`HM30_DEFAULT_RTSP`
- 缓冲：`AlignmentBuffer`、`AlignmentWriter`、`BufferedFrame`、`BufferStats`、`capacity_for`、`raw_frame_bytes`、`DEFAULT_BUFFER_CAPACITY`
- 记录与回放：`FlightRecorder`、`RecorderStats`、`DetectionWriter`、`DropWriter`、`EventLog`、`FlightLog`、`FlightLogError`、`FrameRecord`、`FrameIndexError`、`ReplayVideoSource`、`ReplayStats`、`load_broker_from_log`
- 感知：`Detector`、`DetectorConfig`、`DetectionBatch`、`Detection`、`PixelBox`、`PerceptionWorker`、`PerceptionStats`、`PerceptionConfigLike`、`OpenCvPostProcess`、`CropResult`、`OcrEngine`、`OcrEngineConfig`、`OcrWorkerPool`、`correct_ocr_number`
- 坐标：`CameraModel`、`LLARef`、`GroundIntersection`、`SideLengthCheck`、`default_camera_model`、`load_camera_model`、`undistort_pixel`、`pixel_to_ray_ned`、`pixel_to_ned`、`estimate_depth_by_side`、`cross_check_by_side`、`quaternion_to_matrix`、`wgs84_to_ned`、`ned_to_wgs84`、`ned_distance`
- 目标统计：`TargetPoint`、`Cluster`、`TargetingResult`、`analyze`、`cluster_points`、`select_cluster`
- 弹道与投放：`BallisticsModel`、`Impact`、`ReleaseJudge`、`ReleaseDecision`、`isa_air_density`、`wind_from_snapshot`、`heading_unit_vector`、`ZERO_WIND`、`SEA_LEVEL_DENSITY`、`MAX_FLIGHT_TIME_S`、`SUMMARY_HZ`、`UNSET`（同遥测）
- 投放记录与反演：`DropRecord`、`DropSample`、`ImpactMeasurement`、`DropResidual`、`FitConfig`、`FitResult`、`append_drop`、`load_drops`、`load_impacts`、`match_impacts`、`write_impact_template`、`release_conditions`、`resolve_wind`、`predict_record_impact`、`attitude_matrix`、`fit_ballistics`、`FIT_PARAMETERS`、`DROPS_NAME`、`IMPACTS_JSONL_NAME`、`IMPACTS_CSV_NAME`
- 任务：`MissionState`、`MissionStateMachine`、`MissionTransition`、`InvalidTransition`、`TRANSITIONS`、`TERMINAL_STATES`、`MissionRunner`、`MissionStats`、`MissionMonitor`、`TargetTracker`、`PerceptionTargetSource`、`DropMissionPlan`、`PlanningError`、`build_recon_mission`、`build_drop_mission`、`overfly_positions`、`overfly_waypoints`、`waypoint_to_ned`、`llaref_of`、`emit_event`、`DEFAULT_SIDE_TOLERANCE`、`GROUND_DOWN_M`
- 任务项与 QGC 航线：`MissionItem`、`UNSET`、`command_name`、`load_plan`、`QgcPlan`、`PlanError`、`check_fixed_wing_landing`、`FW_DEFAULT_LAND_ANGLE_DEG`、`PLAN_FIRMWARE_PX4`、`PLAN_VEHICLE_FIXED_WING`，以及 `MAV_CMD_*` / `MAV_FRAME_*` 常量（见 §4.17）
- 起飞前自检：`Preflight`、`PreflightCheck`、`PreflightError`、`PreflightLike`（见 §3.10）
- 参数类（全部在顶层可见）：`AlignConfig`、`BallisticsConfig`、`CameraConfig`、`DropConfig`、`GripperConfig`、`GroundConfig`、`MissionConfig`、`OverflyConfig`、`PerceptionConfig`、`PreflightConfig`、`RecordConfig`、`RoutesConfig`、`TargetingConfig`、`TelemetryConfig`、`VideoConfig`、`Waypoint`

约束：**顶层只导出稳定的公开接口**；子模块内部的私有函数（下划线开头）与新增实验代码不自动出现在这里，所以从顶层清单可以反推"这个项目对外承诺了什么"。判断某个名字是否在承诺范围内，直接看 `airdrop.__all__` 即可。

### 3.10 `airdrop.preflight` —— 起飞前自检（正式任务流程的第一步）

正式流程：`INIT`（遥测 + NED 原点）→ **`PREFLIGHT`**（载入模型 → 视频自检）→
`WAIT_AIRBORNE`（等飞机在空中）→ `RECON`（正式任务等操作手在 QGC 启动侦查航线）。

这些检查各自要碰外部资源（GPU/权重文件、ffmpeg 图传子进程、相机标定），而状态机必须
**离线可测**，所以每一步都做成**注入**：

```python
from airdrop import Preflight, PreflightCheck, PreflightConfig, PreflightError

preflight = Preflight(
    PreflightConfig(
        load_detector=True,
        load_ocr=True,
        load_camera=True,
        check_video=True,
        video_probe_s=5.0,
        video_min_frames=10,
        max_s=120.0,
    ),
    model_loaders={  # 键 = CHECK_NAMES = ("detector", "ocr", "camera")
        "detector": lambda: detector.load(),
        "ocr": lambda: worker.start(),
        "camera": lambda: config.camera.load_model(),  # CameraConfig.load_model()
    },
    video_source=source,  # 任何有 stats.frames（以及可选 latest()）的对象
    camera_model=camera,  # 有 width/height 时核对画面分辨率与标定一致
    on_event=lambda kind, data: recorder.events.emit(kind, **data),
)
checks = preflight.run()  # (PreflightCheck,)；幂等；失败抛 PreflightError
```

| 名字 | 作用 |
| --- | --- |
| `Preflight` | 默认实现：逐项检查（`detector` → `ocr` → `camera` → `video`），**每项只做一次**（`run()` 幂等） |
| `PreflightCheck` | 一项结果：`name` / `ok` / `detail`；**`ok is None` 表示该项被配置关掉（`.skipped`），不算通过** |
| `PreflightError` | 失败异常，消息带**检查名**与原因；状态机据此 `ABORT("preflight_failed:<check>")` |
| `PreflightLike` | 状态机眼里的协议（`run()`）；实现方可以是 `Preflight`，也可以是测试用的假对象 |

行为细节：

- **开着却没注入回调 = 装配错误**：那一项直接判失败并抛 `PreflightError`（绝不静默当成通过）；
- **关掉的项**记一条 `preflight` 事件（`detail='配置里关掉了（未检查）'`、`ok=null`）后跳过；
- 视频自检 = 在 `video_probe_s` 窗口内等够 `video_min_frames` 帧；若给了 `camera_model`，
  再核对最新帧尺寸与标定一致（`Hm30VideoSource` 与回放源都满足 `stats.frames` / `latest()`）；
- **每一项都发一条 `preflight` 事件**，正式任务起飞前逐项核对即可知道哪几项没有把关。

⚠ 复盘口径：**"关掉"与"通过"必须分得开**——`ok=None` 是"没把关"，`ok=True` 才是"查过且合格"。

## 4. 参数总表

全部参数集中在 `airdrop/config.py`，都是 frozen dataclass（不可变），分 16 个类型（15 个分块 + `Waypoint`）。本节按类型逐个列字段；"默认值"一列就是 `Config()` 的取值。

### 4.0 `Config` —— 顶层组合

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `telemetry` | TelemetryConfig | 默认实例 | 见 4.1 |
| `video` | VideoConfig | 默认实例 | 见 4.2 |
| `align` | AlignConfig | 默认实例 | 见 4.3 |
| `perception` | PerceptionConfig | 默认实例 | 见 4.4 |
| `camera` | CameraConfig | 默认实例 | 见 4.5 |
| `ground` | GroundConfig | 默认实例 | 见 4.6 |
| `targeting` | TargetingConfig | 默认实例 | 见 4.7 |
| `ballistics` | BallisticsConfig | 默认实例 | 见 4.8 |
| `drop` | DropConfig | 默认实例 | 见 4.9 |
| `overfly` | OverflyConfig | 默认实例 | 见 4.10 |
| `gripper` | GripperConfig | 默认实例 | 见 4.11 |
| `routes` | RoutesConfig | 默认实例 | 见 4.12 |
| `mission` | MissionConfig | 默认实例 | 见 4.13 |
| `record` | RecordConfig | 默认实例 | 见 4.14 |

方法：`validated()`（校验并返回自身，见 3.9）、`replace(**changes)`（派生变体）。

### 4.1 `TelemetryConfig` —— 遥测链路

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `system_address` | str | `'udpin://0.0.0.0:14540'` | MAVSDK 连接地址 |
| `connect_timeout` | float | `30.0` | 单次连接超时（秒） |
| `wait_for_health` | bool | `False` | 是否等健康检查通过再继续 |
| `health_timeout` | float | `30.0` | 健康检查超时 |
| `require_health_for_actions` | bool | `False` | 发指令前是否强制要求健康 |
| `origin_refresh_interval` | float | `5.0` | NED 原点刷新间隔 |
| `reconnect` | bool | `True` | 断连后是否自动重连 |
| `reconnect_delay` | float | `5.0` | 重连间隔基数 |
| `position_rate_hz` | float | `10.0` | 位置流请求频率 |
| `position_velocity_ned_rate_hz` | float | `10.0` | 位置+速度流请求频率 |
| `attitude_rate_hz` | float | `30.0` | 姿态流请求频率 |
| `history_interval` | float | `0.1` | 历史入库节流（10 Hz；最新快照不节流） |
| `history_maxlen` | int | `1200` | 历史容量（≈2 分钟） |

### 4.2 `VideoConfig` —— 图传链路

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `url` | str | `'rtsp://192.168.144.25:8554/main.264'` | HM30 相机地址 |
| `transport` | str | `'udp'` | `udp` / `tcp`（`SUPPORTED_TRANSPORTS`） |
| `width` | int | `1280` | 期望宽度（也是缓冲/标定的基准） |
| `height` | int | `720` | 期望高度 |
| `read_timeout` | float | `3.0` | 读帧超时 |
| `telemetry_lag` | float | `0.15` | 链路固定延时：拍摄时刻 = 收到时刻 − 该值 |
| `reconnect_delay` | float | `1.0` | 首次重连等待 |
| `max_reconnect_delay` | float | `10.0` | 重连退避上限 |
| `ffmpeg_executable` | str / None | `None` | 指定 ffmpeg 路径（默认用 `imageio-ffmpeg` 自带） |
| `ffmpeg_decoder` | str / None | `None` | 强制解码器（排障用） |
| `extra_input_args` | tuple | `()` | 追加到输入侧的 ffmpeg 参数 |
| `extra_output_args` | tuple | `()` | 追加到输出侧的 ffmpeg 参数 |

方法 `validated()` 校验传输方式与参数组合。

### 4.3 `AlignConfig` —— 帧-遥测对齐

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mode` | str | `'interpolate'` | 查询模式 `interpolate` / `nearest`（`SUPPORTED_QUERY_MODES`） |
| `max_wait` | float | `1.0` | 遥测追赶拍摄时刻的最长等待（秒） |
| `on_timeout` | str | `'drop'` | 超时动作（当前只有 `drop`，见 `ALIGN_TIMEOUT_ACTIONS`） |
| `max_extrapolation` | float / None | `None` | 外推硬保护：超过该秒数直接判失败 |

### 4.4 `PerceptionConfig` —— 检测与 OCR

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mode` | str | `'ocr'` | `ocr` / `cls12`（`PERCEPTION_MODES`） |
| `model_path` | str | `'models/best2.pt'` | YOLO 权重（`best.pt` / `best1.pt` 已废弃） |
| `device` | str | `'0'` | YOLO 设备，显式指定 |
| `conf_threshold` | float | `0.25` | 检测置信度门限 |
| `imgsz` | int | `1280` | 推理尺寸（校验要求 ≥ 32） |
| `models_dir` | str | `'models/ppocr'` | OCR 权重目录 |
| `det_model` | str | `'PP-OCRv6_det_medium.pth'` | 文本检测权重 |
| `rec_model` | str | `'PP-OCRv6_rec_medium.pth'` | 文本识别权重 |
| `rec_keys` | str | `'ppocrv6_dict.txt'` | 字符集字典 |
| `ocr_use_cuda` | bool | `True` | OCR 是否用 GPU |
| `target_color` | str | `'blue'` | `blue` / `red`（`TARGET_COLORS`） |
| `target_side_length_m` | float | `1.0` | 目标物理边长（边长法估深用） |
| `min_side_px` | float | `10.0` | 最小像素边长（小于则丢弃检测） |
| `max_side_px` | float | `400.0` | 最大像素边长（校验要求大于最小值） |
| `ocr_conf_threshold` | float | `0.6` | OCR 置信度门限 |
| `ocr_workers` | int | `2` | OCR 进程数（≥ 1） |
| `ocr_queue_size` | int | `500` | OCR 积压**告警阈值**（≥ 1；队列无界、不丢弃） |
| `ocr_dedupe_s` | float | `0.0` | 旧配置兼容字段，当前不启用去重 |
| `ocr_dedupe_px` | float | `0.0` | 旧配置兼容字段，当前不启用去重 |
| `save_crops` | bool | `False` | 裁剪图写入磁盘开关（⚠ 当前无消费者，见 4.17） |
| `crop_dir` | str | `'log/target'` | 裁剪图目录（⚠ 同上） |
| `max_crops` | int | `5000` | 裁剪图上限（⚠ 同上） |
| `lag_warn_frames` | int | `900` | 落后告警帧数（⚠ 同上） |

### 4.5 `CameraConfig` —— 相机与标定文件

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `calib_file` | str | `'camera_calib.json'` | 标定文件路径（`load_model()` 读它） |
| `undistort` | bool | `True` | ⚠ 当前无消费者，实际去畸变由检测器与坐标解算决定（见 3.5.4） |
| `fallback_width` | int | `1280` | 标定文件缺失时的兜底宽度 |
| `fallback_height` | int | `720` | 标定文件缺失时的兜底高度 |

方法 `load_model()`：按 `calib_file` 装配 `CameraModel`，文件缺失时用兜底尺寸生成占位模型。

### 4.6 `GroundConfig` —— 地面假设

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `flat` | bool | `True` | 平地假设（⚠ 当前无消费者，见 4.17） |
| `ground_point_lat` | float / None | `None` | 地面点纬度（⚠ 同上） |
| `ground_point_lon` | float / None | `None` | 地面点经度（⚠ 同上） |
| `ground_point_alt` | float / None | `None` | 地面点海拔，**唯一被使用的一个** |

方法 `ground_z(origin_alt_m)`：`ground_z = 原点海拔 − 地面点海拔`（NED 里向下为正，同高为 0）；任一参数缺失返回 `None`（**不猜**），调用方决定记不记警告。注意 `ground_point_lat`/`lon` 目前不参与计算，填了也只是备注。

### 4.7 `TargetingConfig` —— 聚类与选唯一结果

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `eps_m` | float | `0.75` | DBSCAN 邻域半径（米，需 > 0；边界是闭区间） |
| `min_samples` | int | `2` | 核心点最小样本数（含自身，权重按绝对值和计） |
| `selection_rule` | str | `'median'` | `median`（下中位）/ `max`（`SELECTION_RULES`） |
| `weight_by_confidence` | bool | `False` | 按置信度加权（**先归一到均值 1**，否则会整片剔点） |
| `require_label` | bool | `True` | 只让带编号的类参与选择 |

### 4.8 `BallisticsConfig` —— 弹道模型

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mass_kg` | float | `0.365` | 质量：**称重占位值，必须实测填入；不进反演** |
| `drag_coefficient` | float | `0.6` | 阻力系数（靠投放试验反演） |
| `cross_area_m2` | float | `0.004` | 迎风面积（几何量） |
| `gravity` | float | `9.80665` | 重力加速度 |
| `rk_dt` | float | `0.005` | RK4 定步长（秒） |
| `air_density_isa` | bool | `False` | `False`=按**投放海拔**算一次常密度（真实海拔基准）；`True`=按"地面海拔+离地高度"逐级 ISA |
| `wind_source` | str | `'telemetry'` | `telemetry` / `zero`（`WIND_SOURCES`） |

### 4.9 `DropConfig` —— 投放判据

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `radius_m` | float | `2.0` | 投放半径：预测落点水平误差 ≤ 它就投（需 > 0） |
| `delay_s` | float | `0.0` | 从判据触发到弹离开的延迟（状态前推用） |
| `force_after_pass` | bool | `True` | 越过目标后的强制投放（前提：本次飞掠里先从目标前方接近过，见 `ReleaseDecision.approached`） |
| `evaluation_hz` | float | `20.0` | 评估节拍口径（实际节拍来自 `MissionConfig.tick_hz`，默认也是 20 Hz） |

### 4.10 `OverflyConfig` —— 飞掠段

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `heading_deg` | float | `0.0` | 飞掠航向（0 = 正北，`% 360` 归一） |
| `altitude_m` | float | `20.0` | 飞掠相对高度 |
| `leg_length_m` | float | `200.0` | 段长：目标前后各半段（**需实验验证**） |

### 4.11 `GripperConfig` —— 投放机构

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | 是否启用投放指令 |
| `instance` | int | `0` | MAVSDK gripper 实例号（PX4 侧也要配） |
| `release_settle_s` | float | `0.5` | 投放后的稳定等待 |

### 4.12 `RoutesConfig` —— 航线

**每条腿二选一**：配置航点，或操作手的 QGC `.plan`。同时给两个来源 `validated()` 会报错
（静默让其中一个优先，比报错危险得多）。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `recon_route` | tuple | `()` | 侦查段航点；为空且没给 `recon_plan` → `PlanningError` |
| `backup_point` | Waypoint / None | `None` | 无目标时用的备用点（Q11 分支） |
| `landing_route` | tuple | `()` | 降落段航点，接在飞掠段之后；为空且没给 `land_plan` → `PlanningError` |
| `recon_plan` | str | `''` | 侦查段用的 QGC `.plan` 路径（**原样使用**，不自动补起飞项） |
| `land_plan` | str | `''` | 降落段用的 QGC `.plan` 路径（**飞掠段插在它前面**，含复杂项展开与降落预检） |
| `fw_land_angle_deg` | float | `8.0` | 飞控的 `FW_LND_ANG`：降落预检用它算允许的最大下滑角；改过飞控参数就要同步 |

### 4.13 `MissionConfig` —— 状态机与超时

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `tick_hz` | float | `20.0` | 主循环节拍 |
| `init_max_s` | float | `30.0` | 等遥测/原点的上限 |
| `recon_max_s` | float | `600.0` | 侦查段上限 |
| `hold_process_max_s` | float | `10.0` | 等待目标统计的上限 |
| `hold_process_min_s` | float | `2.0` | 最短等待时间（之后没活干就按无目标走） |
| `land_max_s` | float | `900.0` | 飞掠+降落整条任务的兜底上限 |
| `telemetry_stale_s` | float | `5.0` | 链路看门狗：遥测陈旧超过它就 ABORT |
| `mission_start_timeout_s` | float | `20.0` | 上传启动后确认"真进了 MISSION"的窗口；超时 → `ABORT('mission_not_started')` |
| `airborne_timeout_s` | float | `1800.0` | `WAIT_AIRBORNE` 等"飞机在空中"的上限（起飞由操作手决定，这里只是兜底）；超时 → `ABORT('airborne_timeout')` |
| `airborne_alt_m` | float | `5.0` | `in_air` 取不到时的兜底判据：`relative_altitude_m >= airborne_alt_m` 也算在空中 |
| `require_airborne` | bool | `True` | **地面演练/测试开关**：`False` 时 `WAIT_AIRBORNE` 的等待起飞检查**立即放行**（飞机停在地面也进侦查），并记 `airborne_skipped` 事件 + WARNING。**正式任务必须保持 `True`**——在停机坪上进侦查会让 PX4 在地面"追"第一个航点 |
| `recon_upload` | str | `'operator'` | 侦查航线谁上传：`'operator'`（默认，正式任务 = 操作手在 QGC 上传并启动）/ `'auto'`（本包上传并启动，**只用于自动测试**）（`RECON_UPLOAD_MODES`） |
| `abort_action` | str | `'hold'` | ABORT 时下的安全动作 `hold` / `rtl` / `none`（`ABORT_ACTIONS`） |
| `takeoff_first` | bool | `True` | 侦查任务首项带起飞动作（**只对 `recon_route` 生效**；plan 不自动补） |
| `land_last` | bool | `True` | 合并任务的最后一项做成降落（**只对 `landing_route` 生效**；plan 原样使用） |

校验要求 `hold_process_min_s <= hold_process_max_s`，其余时长与频率为正。

### 4.14 `PreflightConfig` —— 起飞前自检（正式任务流程的第一步）

正式流程：`INIT`（遥测 + NED 原点）→ `PREFLIGHT`（**载入模型 → 视频自检**）→
`WAIT_AIRBORNE`（等飞机在空中）→ `RECON`（侦查航线由操作手在 QGC 上传并启动）。
每一项都能单独关掉——测试/SITL 里常常没有相机、没有模型：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `load_detector` | bool | `True` | 载入 YOLO 权重（`model_loaders["detector"]`） |
| `load_ocr` | bool | `True` | 载入 OCR 权重与字典（`model_loaders["ocr"]`） |
| `load_camera` | bool | `True` | 载入相机标定（`model_loaders["camera"]`） |
| `check_video` | bool | `True` | 视频自检：窗口内帧数（+ 有标定时核对分辨率） |
| `video_probe_s` | float | `5.0` | 视频探测窗口（秒） |
| `video_min_frames` | int | `10` | 窗口内要求的最少帧数（至少 1） |
| `max_s` | float | `120.0` | 整个预检的总上限；超时 → `ABORT('preflight_timeout')` |

⚠ **关掉 ≠ 通过**：被关掉的项记一条 `preflight` 事件（`ok=null`），正式任务起飞前请核对那一条。

### 4.15 `RecordConfig` —— 记录

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dir` | str | `'flights'` | 飞行目录根（**已进 .gitignore**，见 §1.3） |
| `video` | bool | `True` | 是否记录视频（⚠ 当前无消费者，见 4.18） |
| `telemetry_hz` | float | `10.0` | 遥测写入磁盘节流 |
| `event_eval_hz` | float | `5.0` | ⚠ 当前无消费者（判据摘要频率实际取 `SUMMARY_HZ`） |

### 4.16 `Waypoint` —— 航点

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `lat` | float | 必填 | 纬度 |
| `lon` | float | 必填 | 经度 |
| `alt_m` | float | 必填 | **相对起飞点**的高度 |
| `acceptance_radius_m` | float | `0.0` | 到点判定半径（0 = 用飞控默认） |

`Waypoint` 没有默认值，必须显式写三个数；`acceptance_radius_m` 可省。

### 4.17 模块级常量与枚举

| 常量 | 位置 | 值 / 含义 |
| --- | --- | --- |
| `PERCEPTION_MODES` | `airdrop.config` | `{'ocr', 'cls12'}` |
| `TARGET_COLORS` | `airdrop.config` | `{'blue', 'red'}` |
| `SELECTION_RULES` | `airdrop.config` | `{'median', 'max'}` |
| `ALIGN_TIMEOUT_ACTIONS` | `airdrop.config` | `{'drop'}` |
| `WIND_SOURCES` | `airdrop.config` | `{'telemetry', 'zero'}` |
| `ABORT_ACTIONS` | `airdrop.config` | `{'hold', 'rtl', 'none'}` |
| `RECON_UPLOAD_MODES` | `airdrop.config` | `{'operator', 'auto'}`：侦查航线谁上传（`operator` = 操作手在 QGC 上传并启动） |
| `UNSET` | `airdrop.mission.items` | `float('nan')`，"不指定"；**0 是有意义的值** |
| `command_name(cmd)` | `airdrop.mission.items` | `16 → 'NAV_WAYPOINT'`；未知命令回十进制字符串 |
| `MAV_CMD_NAV_WAYPOINT` / `NAV_LOITER_UNLIM` / `NAV_LOITER_TIME` / `NAV_LAND` / `NAV_TAKEOFF` / `NAV_LOITER_TO_ALT` | `airdrop.mission.items` | `16` / `17` / `19` / `21` / `22` / `31` |
| `MAV_CMD_DO_CHANGE_SPEED` / `DO_LAND_START` / `DO_SET_CAM_TRIGG_DIST` | `airdrop.mission.items` | `178` / `189` / `206` |
| `MAV_CMD_IMAGE_STOP_CAPTURE` / `VIDEO_STOP_CAPTURE` | `airdrop.mission.items` | `2001` / `2501` |
| `MAV_FRAME_GLOBAL` / `MAV_FRAME_MISSION` / `MAV_FRAME_GLOBAL_RELATIVE_ALT` / `MAV_FRAME_GLOBAL_TERRAIN_ALT` | `airdrop.mission.items` | `0` / `2` / `3` / `10` |
| `DEFAULT_WAYPOINT_ACCEPTANCE_M` | `airdrop.mission.items` | `3.0`（生成航点的默认接受半径） |
| `DEFAULT_TAKEOFF_PITCH_DEG` | `airdrop.mission.items` | `15.0`（起飞项 `param1`） |
| `FW_DEFAULT_LAND_ANGLE_DEG` | `airdrop.mission.plan_file` | `8.0`（PX4 `FW_LND_ANG` 出厂默认） |
| `PLAN_FIRMWARE_PX4` / `PLAN_VEHICLE_FIXED_WING` | `airdrop.mission.plan_file` | `12` / `1`（QGC `.plan` 的 `firmwareType` / `vehicleType`） |
| `MISSION_TYPE_MISSION` | `airdrop.telemetry.controller` | `0`（MAVLink `mission_type`） |
| `SUPPORTED_QUERY_MODES` | `airdrop.telemetry.broker`、`airdrop.video.align` | `{'interpolate', 'nearest'}` |
| `SUPPORTED_TIMEOUT_ACTIONS` | `airdrop.video.align` | `{'drop'}` |
| `DEFAULT_TELEMETRY_LAG` | `airdrop.video.align`、`airdrop.video.source` | `0.15` |
| `HM30_CAMERA_IP` / `HM30_GROUND_IP` / `HM30_DEFAULT_RTSP` | `airdrop.video.source` | `192.168.144.25` / `192.168.144.12` / 默认 RTSP |
| `DEFAULT_BUFFER_CAPACITY` | `airdrop.video.buffer` | `5400`（= `capacity_for(30, 180)`） |
| `DEFAULT_JPEG_QUALITY` | `airdrop.video.buffer` | `80` |
| `SUPPORTED_STORAGE` | `airdrop.video.buffer` | `{'jpeg', 'raw'}` |
| `FRAMES_DIR` / `FRAMES_INDEX_NAME` / `TELEMETRY_NAME` / `EVENTS_NAME` / `DETECTIONS_NAME` | `airdrop.record.replay` | 素材文件名常量 |
| `MIN_DOWN_COS` | `airdrop.georef.project` | `0.05`（视线向下分量下限） |
| `OCR_CHAR_MAP` | `airdrop.perception.number` | OCR 字符纠错映射表 |
| `HOUSE_APEX_ANGLE_DEG` / `HOUSE_APEX_ANGLE_TOL_DEG` | `OpenCvPostProcess` 类属性 | `60.0` / `15.0` |
| `BLUE_S_MIN_LEVELS` / `RED_S_MIN_LEVELS` | `OpenCvPostProcess` 类属性 | 饱和度回退级别 |
| `ZERO_WIND` | `airdrop.ballistics.model` | `(0.0, 0.0, 0.0)` |
| `SEA_LEVEL_DENSITY` | `airdrop.ballistics.model` | `1.225` |
| `MAX_FLIGHT_TIME_S` | `airdrop.ballistics.model` | `120.0` |
| `SUMMARY_HZ` | `airdrop.ballistics.release` | `5.0`（判据摘要频率） |
| `DROPS_NAME` / `IMPACTS_JSONL_NAME` / `IMPACTS_CSV_NAME` | `airdrop.ballistics.drops` | `drops.jsonl` / `impacts.jsonl` / `impacts.csv` |
| `FIT_PARAMETERS` | `airdrop.ballistics.fit` | `('drag_coefficient', 'cross_area_m2', 'release_delay_s', 'wind_scale')` |
| `GROUND_DOWN_M` | `airdrop.mission.planner` | `0.0`（航点 down 分量按地面处理） |
| `DEFAULT_SIDE_TOLERANCE` | `airdrop.mission.targets` | `0.25` |
| `TRANSITIONS` / `TERMINAL_STATES` | `airdrop.mission.states` | 合法转移表 / 终态集合 |

### 4.18 改参数的规矩与"声明了但没接上"的字段

规矩：

1. **不要在业务代码里硬编码数值**——一律从 `Config` 取；需要变体时用 `Config.replace(**changes)` 或 `dataclasses.replace` 派生，别就地改默认值（frozen，改不了）。
2. 分块派生最常用：`config.replace(video=replace(config.video, telemetry_lag=0.2))`、`config.replace(mode...)`。
3. **装配后调 `validated()`**：直接构造不会校验。
4. 与硬件相关的量（`mass_kg`、`drag_coefficient`、`leg_length_m`、`telemetry_lag`）都是**待实测**的占位值，改动前先看 §9 与对应专题笔记。

以下字段在当前版本**已声明但没有消费者**（grep 全仓库确认，写在这里免得审查时误以为它们在生效）：

| 字段 | 现状 |
| --- | --- |
| `save_crops` / `crop_dir` / `max_crops` | 裁剪图写入磁盘的功能尚未接入，改这些参数没有效果 |
| `lag_warn_frames` | 落后告警改用别的口径计数（`PerceptionStats.lag_frames` 仍会统计） |
| `undistort`（相机分块） | 去畸变由检测器内参与坐标解算决定，该开关不生效 |
| `flat` / `ground_point_lat` / `ground_point_lon` | 地面高程只用 `ground_point_alt`；平地假设是隐含的 |
| `video`（记录分块） | 记录器是否存帧由装配时是否传缓冲决定，该开关不生效 |
| `event_eval_hz` | 判据摘要频率实际取 `SUMMARY_HZ`（5 Hz） |

## 5. 输入与输出

### 5.1 飞行目录（`flights/<YYYYMMDD-HHMMSS>/`）

一次飞行 = 一个目录，由 `record.recorder.FlightRecorder` 在 `start()` 时创建（重名自动追加 `-1`/`-2`）；
目录名时间戳来自 `time.strftime("%Y%m%d-%H%M%S")`，`ls` 排序即时间排序。

| 文件 | 谁写 | 内容 / 关键字段 | 谁读 | 能手改吗 |
| --- | --- | --- | --- | --- |
| `flight.log` | root logger 上的 `FileHandler`（`start()` 挂、`stop()` 摘，期间 root level 临时降到 INFO，业务模块无感） | 所有模块的 `logging` 输出 | 人 | ✅ 纯文本 |
| `telemetry.jsonl` | recorder 的 telemetry writer 线程，按 `RecordConfig.telemetry_hz`（默认 10Hz）节流；**只写有效快照** | 每行一份 `TelemetrySnapshot.as_dict()`（位置/速度/欧拉角/四元数/原点/风/**任务进度、飞行模式与是否在空中**…共 33 字段） | `record.replay.load_broker_from_log`、人 | ✅ 可删可裁；**别插乱时间戳**（回放按时间轴推进） |
| `detections.jsonl` | `PerceptionWorker` → `recorder.detections.append(Detection)` | `Detection.as_dict()`：`frame_index`、`capture_timestamp`、`pixel`、`box`、`confidence`、`code`、`side_px`、`raw_text`、`ocr_confidence`、`mode`、`extra`（含 `undistorted`/`ocr_stage`/`ocr_worker`）+ 精简遥测字段 | 人、脚本 | ✅ 追加字段不影响解析 |
| `events.jsonl` | 各模块经 `recorder.events.emit(kind, **data)`（线程安全、写完即 flush） | `{timestamp, kind, ...}`，`kind` 目录见 §5.2 | 人、复盘脚本 | ⚠ 建议只读（这是任务时间线） |
| `drops.jsonl` | `MissionRunner(on_drop=…)` → `recorder.drops.append(DropRecord)` | 投放瞬间**原始状态**：位置/速度/姿态（欧拉角+四元数）/风/离地高度/NED 原点/目标点/判据前推位置与预测落点（**数值不四舍五入**） | `tools/fit_ballistics.py`、人 | ⚠ 只读；实测落点写在 `impacts.jsonl`，别改这个文件 |
| `frames/%06d.jpg` | recorder 的 frame writer 线程（**零重编码**：把缓冲里的 jpeg 字节直接写入磁盘） | 逐帧 JPEG，文件名 = 缓冲序号 | `record.replay.ReplayVideoSource` | ⚠ 删帧会让回放 `strict=True` 报错 |
| `frames_index.jsonl` | 同上 | `index`、`filename`、`capture_timestamp`、`received_timestamp`、`lag`、`extrapolated`、`offset`、`bytes` | `record.replay.FlightLog` | ⚠ 只读（时间戳必须递增） |
| `config_snapshot.json` | `start()` 时立刻写 | `asdict(Config)` 全量（含视频地址、标定文件路径） | 人、审查 | ⚠ 只读（"当时怎么配的"证据） |
| `impacts.jsonl` / `impacts.csv` | **人**（现场量完落点后填） | `index` + 经纬度（`lat_deg`/`lon_deg`/`alt_m`）**或** NED（`north_m`/`east_m`/`down_m`） | `tools/fit_ballistics.py` | ✅ 本来就要手填（模板见 `write_impact_template`） |

### 5.2 `events.jsonl` 的 `kind` 目录

| `kind` | 谁发 | 关键字段 / 含义 |
| --- | --- | --- |
| `state` | `MissionStateMachine` | `from_state`、`to_state`、`reason`、`timestamp`——每次**成功**转移一条 |
| `origin` | `MissionRunner` | `lat_deg`/`lon_deg`/`alt_m`：NED 原点就绪 |
| `targeting` | `MissionRunner` | `reason`（`result`/`timeout`/`no_target`）、`elapsed_s`、`targeting`（`TargetingResult.as_dict()`） |
| `drop_plan` | `MissionRunner` | `DropMissionPlan.as_dict()`：任务项、目标 NED、来源（`target`/`backup`）、飞掠 entry/exit |
| `drop_check` | `ReleaseJudge` | 判据摘要，**按 5Hz 节流**（`SUMMARY_HZ`）：`should_release`、`reason`、`horizontal_error_m`、`impact_ned`、`release_position`… |
| `release` | `ReleaseJudge` | 触发瞬间的完整预测（含 `flight_time_s`、`impact_speed_m_s`） |
| `drop` | `MissionRunner` | 判据给的决策 + `target_ned` + `target_source`（**在发投放指令之前**记） |
| `drop_record` | `MissionRunner` | `DropRecord.as_dict()` 全量（与 `drops.jsonl` 同一份数据，进时间线） |
| `drop_skipped` | `MissionRunner` | `reason="no_judge"`：本架次没接判据，只飞航线 |
| `abort_action` | `MissionRunner` | `action`：`hold`/`rtl`/`none` |
| `error` | 各模块 | `module`、`where`、`message`：命令失败、链路陈旧、记录失败… |
| `mission_upload` | `DroneController` | `count` + `items`（每个任务项的 `as_dict()`：`command`/`command_name`/位置/`frame`/参数） |
| `mission_start` / `hold` / `rtl` / `gripper_release` / `gripper_skipped` | `DroneController` | 每条实际下发的指令（`mission_start` 表示"已复位到第 0 项并启动"；`gripper_release` 带 `instance`；`gripper_skipped` 表示 gripper 被禁用） |
| `mission_confirmed` | `MissionRunner` | `mode='MISSION'`：启动后确认飞控**真的**进了任务模式（`mission_start_timeout_s` 内） |
| `preflight` | `Preflight`（**每项一条**） | `{"check": "detector"/"ocr"/"camera"/"video", "ok": true/false/null, "detail": ...}`；**`ok=null` = 该项被配置关掉（没把关）** |
| `preflight_skipped` | `MissionRunner` | 没注入预检对象时记一条并告警（回放/离线测试档） |
| `airborne` | `MissionRunner` | `source`（`in_air` 或 `altitude`）：按哪个判据说"飞机在空中了" |
| `recon_waiting_operator` | `MissionRunner` | `note`：`recon_upload='operator'`，本包**不上传**侦查航线，等操作手在 QGC 上传并启动 |
| `drop_dry_run` | `DryRunController` | `count`：演练模式下第几次"投放只记日志" |
| `side_check` | `TargetTracker` | 边长法交叉验证超门限的诊断（**默认不剔点**） |
| `recorder_started` / `recorder_stopped` | `FlightRecorder` | 目录、频率、缓冲容量 / 各类计数 |

### 5.3 `camera_calib.json`（`tools/calibrate.py` 产出）

| 键 | 来源 | 是否权威 |
| --- | --- | --- |
| `camera_matrix`（3×3）、`dist_coeffs` | 步骤一：棋盘格多视图 `calibrateCamera` | ✅（质量看 `meta.intrinsics.rms_px`） |
| `image_size` | 同上 | ✅ 标定分辨率——换分辨率要用 `CameraModel.scaled()` 换算 |
| `R_bc` | 步骤三：手眼标定 `A X = X B` | ✅ 旋转以标定为准 |
| `t_bc` | **尺量常量**（`tools/calibrate.py` 顶部 `MEASURED_T_BC`） | ✅ 平移以尺量为准；标定估计值只在 `meta.extrinsics.lever_arm` 里供对比 |
| `telemetry_lag` | 步骤二：画面角速度 × 飞控角速度互相关 | ✅ 标定后回填 `VideoConfig.telemetry_lag` |
| `meta.intrinsics` | `rms_px`、`views`、`per_view_errors`、`pattern_size`、`square_size_m` | 诊断 |
| `meta.time_offset` | `lag_s`、`peak_correlation`、`samples`、`search_s`、`step_s`、`curve` | 诊断（相关系数低 → 激励不足，重采） |
| `meta.extrinsics` | `pairs`、`residual_rotation_deg`、`axis_spread_deg`、`t_bc_source`、`lever_arm`、`methods`、`note` | 诊断 |
| `meta.source_dir` | 用了哪个飞行目录 | 追溯 |

文件缺失/损坏时 `load_camera_model()` **不报错**，退回默认外参（`DEFAULT_R_BC` = 绕 z +90°）并 warning
——正式任务必须先标定，否则坐标会系统性偏。

### 5.4 投放试验的输入与输出（`tools/fit_ballistics.py`）

**输入**：`drops.jsonl`（实飞自动写）+ 实测落点（二选一写法）：

```jsonc
// impacts.jsonl —— 经纬度（现场 GPS，推荐）
{"index": 1, "lat_deg": 47.3978, "lon_deg": 8.5456, "alt_m": 500.0, "note": "手持 GPS"}
// 或 NED（相对**该架次**的 NED 原点）
{"index": 1, "north_m": 123.4, "east_m": -56.7}
```

也支持 `impacts.csv`（表头同名）。没有文件时工具**生成待填模板**；行填了一半会**显式报错**。

**输出** `ballistics_fit.json`：

| 键 | 内容 |
| --- | --- |
| `flights` | 每架次账目：`flight`、`drops`、`measured`、`missing`（哪些序号没量）、`impacts`（用了哪个测量文件） |
| `samples` | 每条样本：`label`（`架次#序号`）、`index`、`impact_ned`、`impact_source`（`gps`/`ned`/`record`）、`record`（`DropRecord.as_dict()` 全量） |
| `fit` | `FitResult.as_dict()`：`ok`/`reliable`/`reason`、`parameters`、`sigma`、`correlation`、`condition_number`、`drag_k_per_m`、`drag_area_m2`、`rms_error_m`、`bias_north/east_m`、`per_drop`（每条沿/横分解）、`leave_one_out_*`、`warnings` |

**退出码**：`0` 可信（`ok` 且 `reliable`）／`2` 收敛但不可信或不可辨识／`3` 还没有实测落点（已生成模板）／`1` 配置错误。

### 5.5 `models/` 目录

| 文件 | 用途 | 进 git |
| --- | --- | --- |
| `models/best2.pt` | YOLO 检测权重（**唯一可用**；`best.pt`/`best1.pt` 是废弃权重，零检出） | ❌（`*.pt` 忽略） |
| `models/ppocr/PP-OCRv6_det_medium.pth`、`PP-OCRv6_rec_medium.pth` | RapidOCR TORCH 引擎权重（det/rec） | ❌（`*.pth`） |
| `models/ppocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 方向分类 ONNX（默认 `cls_engine="onnx"`） | ❌（`*.onnx`） |
| `models/ppocr/ch_ptocr_mobile_v2.0_cls_mobile.pth` | 方向分类 TORCH 权重（**`pt` 是 RapidOCR 的拼写，别规范化**） | ❌ |
| `models/ppocr/*.txt` | 字符集字典——**入库**：它决定 rec 的字符集，跟着版本走才能复现识别结果 | ✅ |

取权重：`python -m airdrop.run fetch-models --source-dir <本地素材目录>`。

### 5.6 测试与临时目录

| 目录 | 谁用 | 说明 |
| --- | --- | --- |
| `.e2e-test-tmp/` | `tests/test_e2e.py` 的 `workdir` fixture | 手写合成飞行目录，用完即删 |
| `.fit-test-tmp/` | `tests/test_fit.py` 的 `workdir` fixture | 投放记录 / 测量文件 / 工具端到端 |
| `.calibrate-test-tmp/` | `tests/test_calibrate.py` 的 `workdir` fixture | 合成标定飞行目录（渲染棋盘格帧） |
| `.plan-test-tmp/` | `tests/test_plan.py` 的 `workdir` fixture | 现场造的 QGC `.plan`（含复杂项） |
| `.handbook-test-tmp/` | `tests/test_handbook.py` 的 `workdir` fixture | 手册 §3 示例真跑时的工作目录 |
| `.world-test-tmp/` | `tests/test_world.py` 的 `workdir` fixture | 生成到别处的赛区世界（含贴图复制） |
| `.replay-test-tmp/` | `tests/test_replay.py` | 手写的最小飞行目录 |
| `.sitl-test-tmp/` | **人工**（SITL 演练） | 演练日志、机上任务备份、ulog 探查脚本；不是测试 fixture |
| `.pytest-tmp/` | 历史遗留（`--basetemp` 落点） | 已可删除，无用 |

**为什么把临时目录放在工作区内**：某些受限环境里 pytest 的 basetemp 目录"创建后在别的进程里不可枚举、不可删除"
（`PermissionError: [WinError 5]`），相关用例会直接挂在 fixture 上。上表目录都已在 `.gitignore` 里。

⚠ **同一时刻只能有一个 `mavsdk_server`**（它固定占用 gRPC 50051）：一边跑示例/演练，一边想再起一个
`System()` 去查状态会争抢端口。调试时用
`System(mavsdk_server_address="localhost", port=50051)` **接已有的 server**（不再拉起第二个进程）。

---

## 6. 使用流程

### 6.1 安装与取权重

```bash
uv sync                                                # 依赖（清华镜像，见 pyproject.toml）
./.venv/Scripts/python.exe -m tools.fetch_models       # 把权重放进 models/
./.venv/Scripts/python.exe -m pytest -m "not realdata and not sitl and not stream"   # 冒烟：全部离线用例
```

### 6.2 入口清单：`python -m airdrop.run <子命令>`

**命令行解析集中在唯一一个运行模块** `airdrop/run.py`（标准库 argparse 子命令）。
`examples/*.py` 与 `tools/*.py` 都是**纯库模块**：文件顶部常量 = 默认值，暴露
`build_config(**覆盖)` 与 `main(**kwargs)`；命令行选项到配置走**关键字覆盖**
（各模块内部 `dataclasses.replace` 派生），默认值直接取入口模块的常量——`run.py` 里
**不重复写第二遍**。加子命令只需在 `airdrop/run.py` 的 `SUBCOMMANDS` 注册表里加一条。

⚠ **帮助信息不加载重库**：`--help`（顶层与每个子命令）与 `check-docs` 都不会加载
torch / cv2 / mavsdk / ultralytics / rapidocr / onnxruntime——重依赖只在 handler 真正跑起来时
由各模块的**函数体**导入，`airdrop/__init__.py` 与各子包都是 PEP 562 惰性导出
（`airdrop/_lazy.py`）。`tests/test_cli.py` 用干净子进程盯着这条不变量。

| 子命令 | 入口模块 | 前置条件 | 产物 / 输出 | 退出码 |
| --- | --- | --- | --- | --- |
| `python -m airdrop.run full-mission` | `examples/full_mission.py` | 飞控 + 图传 + `camera_calib.json` + 航线配置；`--dry-run` 可不装弹。**这是正式任务档**：`INIT→PREFLIGHT`（载入模型 + 视频自检）→`WAIT_AIRBORNE`→`RECON`（侦查航线由操作手在 QGC 上传并启动，`recon_upload="operator"`） | `flights/<ts>/` 一整套；终端打印任务与统计 | 0 成功 / 2 未到 DONE / 130 中断 / 1 异常 |
| `python -m airdrop.run sitl` | `examples/sitl_mission.py` | PX4 SITL 在 `udpin://0.0.0.0:14540`；`--land-plan` 指向的 `routes/land.plan` 要在（不给则用内置 `LANDING_ROUTE`）。**这是自动测试档**：自检全关、`recon_upload="auto"`（本包上传侦查航线）、**不等起飞**（`require_airborne=False`，见 §4.13） | 终端"演练记录"：状态轨迹 / 上传与投放次数 / 判据评估次数 / 飞掠计划；`flights/<ts>/` 记录 | 0 成功 / 2 未到 DONE / 130 中断 / 1 异常 |
| `python -m airdrop.run sitl-recon` | `tools/sitl_recon.py` | WSL 里起 SITL（`bash sim/run_sitl.sh r2`）+ Gazebo 自带的 RTP/H.264 图传（**不必用 HM30**）。**完全标准流程**：本脚本只扮演操作手（上传/启动「起飞任务」与「侦查航线」），其余交给 `MissionRunner` 状态机 + 真 `Preflight` + `PerceptionWorker`（OCR 进程池）+ `DryRunController`：监视侦查 → 空中出目标 → 飞掠投放 → 降落（**没有任何 offboard 设定点**）。测试期间临时把 `NAV_DLL_ACT` 置 0（跑完还原成 2）；`MIS_TKO_LAND_REQ` 已在仿真机型里永久置 0（理由见 `sim/airframes/4007_gz_rc_cessna_down_cam` 的注释——只飞侦查段，投弹航线要空中出结果才能生成） | 终端精度报告 + `.sitl-recon-tmp/report.json`（任务结果与分井误差、延时敏感性、反向诊断）+ `flights/<ts>/` 记录 | 0 成功 / 2 没遥测 / 3 没画面 / 4 没解锁 / 5 没进任务模式 / 6 没起飞 / 7 没飞完（仍出报告） / 1 异常 |
| `python -m airdrop.run replay` | `examples/replay_flight.py` | 一份飞行目录（`--flight`，不给则取 `flights/` 最新） | 终端：帧统计、目标点、聚类结果、飞掠航线（**不上传**） | 0（没结果也是 0，只是走备用点分支） |
| `python -m airdrop.run calibration-capture` | `examples/calibration_capture.py` | 图传（不给 `--rtsp-url` 用默认 HM30 地址）+ 飞控遥测 | 一个标定用飞行目录（姿态 50Hz、位置 20Hz、缓冲 300s） | 0；一帧没收到 = 1 |
| `python -m airdrop.run calibrate` | `tools/calibrate.py` | 上面那份标定目录（`--flight`） | `camera_calib.json`；终端打印每步残差/相关系数 | 0；`--strict` 且外参/时间差没标出来 = 2 |
| `python -m airdrop.run fit-ballistics` | `tools/fit_ballistics.py` | 投放试验的飞行目录 + `--impacts`（实测落点） | `ballistics_fit.json` + 终端对照表 + 可粘回的 `BallisticsConfig(...)` | 0 可信 / 2 不可信 / 3 缺测量 / 1 出错（§5.4） |
| `python -m airdrop.run basic` | `examples/basic_usage.py` | 飞控（只看遥测部分则不必） | 订阅推送、按时间戳查询与指令演示 | 0；15s 收不到遥测 = 1 |
| `python -m airdrop.run hm30-video` | `examples/hm30_video.py` | HM30 图传 | 链路统计；`--preview` 开窗、`--save-path` 存 mp4 | 0 |
| `python -m airdrop.run video-sync` | `examples/video_telemetry_sync.py` | HM30 图传 + 飞控 | 对齐 + 逐帧入缓冲 + 读者线程演示 | 0；首帧超时/一帧没收到 = 1 |
| `python -m airdrop.run fetch-models` | `tools/fetch_models.py` | 本地素材目录（`--source-dir`；**只复制、不联网**） | 权重复制进 `models/` | 0 |
| `python -m airdrop.run make-world` | `tools/make_world.py` | — | `sim/worlds/cuadc/` 里的两个 CUADC 赛区世界 + 网格/材质（`--out-dir` 会把贴图一起复制过去）；布局与 SITL 启动见 [`simulation_world.md`](simulation_world.md) | 0 成功 / 1 参数非法或缺资产 |
| `python -m airdrop.run make-backdrops` | `tools/make_backdrops.py` | —（离线；要 Pillow） | `sim/worlds/cuadc/materials/textures/` 里的目标区地面纹理 `ground_*.png` + 程序化航拍底图 `aerial_*.png`；`--parts ground` 只写前者（不覆盖抓来的真实航拍图） | 0 / 1 参数非法 |
| `python -m airdrop.run fetch-aerial` | `tools/fetch_aerial.py` | **联网**（USGS NAIP 影像服务；公有领域） | 用真实航拍影像覆盖 `aerial_*.png`（比赛区域外那块干扰底图） | 0 / 1 联网或下载失败 |
| `python -m airdrop.run preview-world` | `tools/preview_world.py` | —（要 Pillow；只读世界文件，不重新生成） | 俯视预览图 `sim/worlds/cuadc/preview.png`（带中文标注：天井编号/两位数、航拍块编号、图例；**标注只在这张 PNG 里**） | 0 / 1 世界文件读不了 |
| `python -m airdrop.run dump-api` | `tools/dump_api.py` | — | 把公开签名与默认值自省成 `docs/api_reference.md`（核对文档用） | 0 |
| `python -m airdrop.run check-docs` | `tools/check_docs.py` | — | 核验 `docs/handbook.md` 与代码是否一致 | 0 一致 / 1 有缺口 |

常用选项（每个子命令的完整选项与默认值看 `--help`；默认值直接取自入口模块的常量）：

| 选项 | 子命令 | 落到配置 |
| --- | --- | --- |
| `--system-address` | full-mission / sitl / basic / video-sync / calibration-capture | `TelemetryConfig.system_address` |
| `--rtsp-url` | full-mission / hm30-video / video-sync / calibration-capture | `VideoConfig.url` |
| `--telemetry-lag` | full-mission / video-sync / calibration-capture | `VideoConfig.telemetry_lag` |
| `--land-plan` | full-mission / sitl / replay | `RoutesConfig.land_plan`（与配置航点**二选一**，同时给会报错；replay 只规划不上传） |
| `--recon-upload {operator,auto}` | full-mission / sitl | `MissionConfig.recon_upload` |
| `--no-video` | full-mission | 本架次不接图传：跳过图传/感知，自检四项也相应关掉 |
| `--no-preflight` | full-mission | 把 `config.preflight` 各项全关掉（**关掉 ≠ 通过**：每项仍记 `ok=null` 事件） |
| `--require-airborne` / `--no-require-airborne` | full-mission / sitl | `MissionConfig.require_airborne`（地面演练才关） |
| `--target-offset N,E,D` | sitl | 合成目标相对盘旋点的 NED 偏移 |
| `--flight DIR` | replay / calibrate | 飞行目录（回放素材 / 标定素材） |
| `--calib FILE` | replay | `CameraConfig.calib_file`（**要用录该素材时同一份**；SITL 架次是 `.sitl-recon-tmp/camera_calib_sim.json`） |
| `--side-check TOL` | replay | 边长互校门限（**默认关**；回放优化时才给，如 0.25——诊断用，不剔点） |
| `--speed` | replay | 回放速度（0 = 全速） |
| `--strict` | replay / calibrate | 回放异常即失败 / 标定降级（外参或时间差没标出来）即返回 2 |
| `--lag S` | sitl-recon | 图传链路延时估计（秒；报告里还会给 lag 扫描与反向诊断） |
| `--recon-timeout S` | sitl-recon | RECON 段的上限（秒，状态机在这一段等侦查航线飞完） |
| `--video-wait S` | sitl-recon | 等第一帧画面的上限（秒；SITL 刚启动时可以给大一点） |
| `--work-dir DIR` | sitl-recon | 产物目录（模拟相机标定/SDP/检测明细/`report.json`） |
| `--impacts FILE` | fit-ballistics | 实测落点文件 |
| `--mass-kg` | fit-ballistics | `BallisticsConfig.mass_kg`（**实测值，不参与反演**） |
| `--seed N` | make-world | 天井位置与朝向的随机种子（默认 0 = 入库布局） |
| `--rounds {both,1,2}` | make-world | 生成哪几轮（第一轮图片靶标 / 第二轮数字靶标） |
| `--wind E,N,U` | make-world | 世界的恒定风（东,北,天 m/s；默认静风） |
| `--out-dir DIR` | make-world | 世界与资产写到哪个目录（默认 `sim/worlds/cuadc`） |
| `--well-arrows` | make-world | 画"天井箭头"（规则文本有、规则图 2/4 没有；默认不画） |
| `--sun AZ,EL` | make-world | 太阳位置（方位角,仰角 度；换阴影环境用，如 `90,12` = 东侧长阴影） |
| `--sun-random` | make-world | 太阳由 seed + 轮次随机（方位全向、仰角 15~80°；换 seed 换光照） |
| `--aerial-patches N` | make-world | 比赛区域外铺几块航拍干扰底图（0 = 不铺；默认 12） |
| `--size N` | make-backdrops / fetch-aerial | 底图贴图边长（像素，默认 512） |
| `--seed N` | make-backdrops | 底图随机种子（`ground_*` / 程序化 `aerial_*` 长什么样） |
| `--parts {all,ground,aerial}` | make-backdrops | 只写哪一类底图（默认 all） |
| `--tile-width-m M` | fetch-aerial | 每张真实航拍图覆盖的地面宽度（米，默认 160） |
| `--world-dir DIR` | preview-world | 预览哪个世界目录（默认 `sim/worlds/cuadc`） |
| `--round {1,2}` | preview-world | 预览哪一轮（默认 2） |
| `--out FILE` | preview-world | 输出 PNG 路径（默认 `sim/worlds/cuadc/preview.png`） |

参数不合法（未知子命令、未知选项、取值越界）一律由 argparse 以 **2** 退出。


### 6.3 正式任务的启动顺序（与自动测试档的区别）

```
INIT           等遥测 + NED 原点（不上传任何任务）
  ↓
PREFLIGHT      起飞前自检：载入 detector / ocr / camera → 视频自检
               （窗口 video_probe_s 内至少 video_min_frames 帧；有标定时核对分辨率）
  ↓            任一项失败 → ABORT("preflight_failed:<check>")；超 max_s → preflight_timeout
WAIT_AIRBORNE  什么都不下发，等"飞机真的在空中"
               （in_air 为主，取不到时 relative_altitude_m >= airborne_alt_m 兜底）
  ↓            超 airborne_timeout_s → ABORT（airborne_timeout）
RECON          正式任务：recon_upload="operator" —— **不上传**，等操作手在 QGC 启动侦查航线
               （本包只等它开始，然后监视进度）；自动测试才用 "auto" 由本包上传
  ↓
HOLD_PROCESS → OVERFLY（飞掠 + 投放）→ LAND → DONE
```

| 档位 | `MissionConfig.recon_upload` | 自检（`PreflightConfig`） | 谁上传侦查航线 | 入口 |
| --- | --- | --- | --- | --- |
| **正式任务** | `'operator'`（默认） | 全开（载入模型 + 视频自检） | **操作手**（QGC） | `examples/full_mission.py` |
| 自动测试 / SITL | `'auto'` | 全关或按需关（无相机、无模型、飞机不起飞） | 本包 | `examples/sitl_mission.py` |

⚠ SITL/地面演练还会关掉"等起飞"的检查：`examples/sitl_mission.py` 的 `REQUIRE_AIRBORNE = False`
（即 `MissionConfig.require_airborne=False`，见 §4.13）——飞机停在停机坪上也能把状态机整条跑完，
不必等到 `airborne_timeout_s`（默认 1800s）才失败。放行**留痕**：一条 `airborne_skipped`
事件 + 一条 WARNING；正式任务必须保持默认的 `True`。

⚠ 关掉的自检项**不算通过**：每项都会落一条 `preflight` 事件（`ok=null`），
正式任务起飞前请核对该事件；`preflight_skipped` 则表示根本没注入预检对象。

**SITL 用哪个世界**：`examples/sitl_mission.py` 的航线与合成目标已对准
`sim/worlds/cuadc/cuadc_recon_strike_r2.sdf`（CUADC 赛区：跑道 + A/B 两个 60x60m 目标区
分列起飞线两端、8 座天井在区内随机摆放，第二轮数字靶标；航线常量按 seed 0 手写，
`tests/test_world.py` 会核验它对准 A 区中位数天井）。起 SITL 前要把世界挂进 PX4 的 worlds 目录并把
`sim/worlds/cuadc` 加进 `GZ_SIM_RESOURCE_PATH`——两条命令的完整步骤、几何与规则出处见
[`simulation_world.md`](simulation_world.md)；一键版本是 `bash sim/run_sitl.sh r2`
（机型默认 `gz_rc_cessna_down_cam`：带 1280x720 下视相机，感知取图用的那个）。
**视频流是 PX4 官方机制**：`gz_bridge/server.config` 自动加载 `GstCameraSystem`，把世界里的
第一台相机编码成 RTP/H.264 推到 UDP 5600，QGC 设 “UDP h.264 / 5600” 就能看（详见世界文档 §5.1）。

### 6.4 改配置的三种方式

```python
# ① 改默认值（影响所有入口）：airdrop/config.py 对应字段
# ② 改文件内常量（只影响那一个入口）：示例/工具顶部的"配置（改这里）"区
# ③ 程序化派生（写脚本/测试时推荐）：
from dataclasses import replace
from airdrop import Config

base = Config()
config = Config(
    video=replace(base.video, telemetry_lag=0.18),
    drop=replace(base.drop, radius_m=1.5),
).validated()  # ⚠ 直接构造不会校验，必须调 validated()
```

### 6.5 测试怎么跑

```bash
./.venv/Scripts/python.exe -m pytest                     # 全部（附覆盖率，当前 ≈85%；默认已排除 realdata/sitl）
./.venv/Scripts/python.exe -m pytest -m "not realdata and not sitl and not stream"  # 再跳过要起 ffmpeg 的用例
./.venv/Scripts/python.exe -m pytest -m realdata         # GPU + 2024v2 实战素材（分钟级）
./.venv/Scripts/python.exe -m pytest -m sitl             # WSL 里已起 PX4 SITL + 图传（分钟级）
./.venv/Scripts/python.exe -m pytest -k mission -v       # 只跑某个主题
```

⚠ 命令行的 `-m` 是**覆盖** `addopts` 里的默认表达式，不是追加：写 `-m "not stream"` 会把 `realdata` 与 `sitl`
一起放进来（`sitl` 那条要 SITL 真在跑，否则只是白等一轮探活）。

| 测试文件 | 覆盖什么 | 需要硬件/GPU |
| --- | --- | --- |
| `tests/test_config.py` | `Config` 默认值与 `validated()` 取值域（非法值必须在启动前抛错） | 否 |
| `tests/test_telemetry.py` | broker 订阅推送、内插/外推、历史节流、遥测速率下发 | 否 |
| `tests/test_alignment.py` | 对齐器 lag 语义、等待上限、外推标记、超时策略 | 否 |
| `tests/test_buffer.py` | 环形缓冲容量/驱逐、复用缓冲区必须拷贝、`AlignmentWriter` 逐帧入库 | 否 |
| `tests/test_video.py` | ffmpeg 拉流：无效地址显式报错、sink 每帧不落、`stats.dropped` 语义 | 起 ffmpeg 子进程 |
| `tests/test_recorder.py` | 飞行目录七个文件齐全、帧零重编码、事件/检测/投放写入、幂等启停 | 否（用例自建工作区内临时目录） |
| `tests/test_replay.py` | 回放对齐结果与"直接从日志查询"的参考 broker 完全一致 | 否 |
| `tests/test_perception.py` | pipeline 逻辑（假 detector / 假 OCR 池）、去重、编号不被覆盖 | 否 |
| `tests/test_perception_realdata.py` | 真 YOLO + 真 OCR 在 2024v2 素材上读对编号（56/56/56） | ✅ GPU + 素材绝对路径 |
| `tests/test_georef.py` | 相机模型加载/回退、像素→NED、边长法交叉验证（用例自建工作区内临时目录） | 否 |
| `tests/test_calibrate.py` | 标定三步合成链路（含整个 `calibrate()` 的输出契约） | 否 |
| `tests/test_targeting.py` | DBSCAN 语义（eps 闭区间、样本权重是绝对权重）、众数、median/max | 否 |
| `tests/test_ballistics.py` | 无阻力/有阻力对照解析解、终端速度、风平移、判据触发/越过目标后的强制投放/锁存 | 否 |
| `tests/test_mission.py` | 状态机、航线规划、主循环（假控制器 + 假时钟，分钟级任务毫秒级跑完）；起飞前自检（真 `Preflight` 的四项检查/装配错误/视频帧数/分辨率核对）与等起飞 | 否 |
| `tests/test_plan.py` | QGC `.plan` 解析与复杂项展开、固定翼降落预检逐条判据、plan↔配置航点二选一 | 否 |
| `tests/test_e2e.py` | 回放驱动全链路：帧 → 感知 → 坐标 → 统计 → 航线（另有 `realdata` 变体） | 否（变体需 GPU） |
| `tests/test_fit.py` | 投放记录读写、反演真值回收、退化/不可辨识必拒、工具现场流程 | 否 |
| `tests/test_examples.py` | 示例与工具不脱节（导入即校验 + **导入不加载重库** + 真跑装配函数） | 否 |
| `tests/test_sitl_recon.py` | **SITL 侦查精度（可选，标 `sitl`；默认被 `-m "not realdata and not sitl"` 排除，显式 `-m sitl` 才跑）**：驱动 `tools/sitl_recon.py` 跑完整标准流程，断言状态机 `DONE`、三个编号全识别、任务选中 56、误差与留一验证都 < 2 m；探活用**裸 UDP 听 14540 心跳**（不建 MAVSDK 会话），没在跑就 skip | 否（要 WSL 里已起 PX4 SITL） |
| `tests/test_cli.py` | 集中式入口：子命令注册表（parser + handler）、`--help`/`check-docs` **不加载重库**（干净子进程实测）、参数不合法退出码 2、选项覆盖真的进了 `build_config()` | 否 |
| `tests/test_handbook.py` | **本手册 §3 的 11 段示例逐条真跑**（文档里的代码必须能运行） | 否 |
| `tests/test_world.py` | CUADC 赛区世界：**生成结果 == 入库文件**、目标区在起飞线两端（±200, 0）、天井区内随机摆放且间距 > 20m、五边形环壁（厚 5mm、高 400mm）、**规则里只是示意的东西不许出现**（4m/6m 打击圈、天井箭头）、同区天井同色、贴地标线不共面重叠（防 z-fighting）、两轮靶标摆位（第二轮中位数落在第 3 座）、演练航线对准中位数天井、目标区底色（常见路面色 + 少量纹理）、周边航拍底图（不压跑道/目标区、互不重叠）、底图生成器（可复现/离线校验）、俯视预览图（尺寸/可复现/参数校验）、资产逐级存在、数字板必须是**白底黑字** | 否 |

### 6.6 一次真实任务的推荐顺序

1. **标定**：`examples/calibration_capture.py` 录素材 → `tools/calibrate.py` 出 `camera_calib.json`
   → 把 `telemetry_lag` 回填 `VideoConfig`（一步同时解决相机外参与时间差）。
2. **配置**：`RECON_ROUTE`/`BACKUP_POINT` + 降落段——**要么**用操作手在 QGC 里画好的
   `routes/land.plan`（推荐，降落剖面由 QGC 保证），**要么**写 `LANDING_ROUTE` 航点
   （WGS84，高度是**相对起飞点**，且必须满足 §7 第 32 条的降落几何）；再配 `OVERFLY_HEADING_DEG`；
   用 `Config().validated()` 过一遍。
3. **拆桨/演练**：`DRY_RUN=True` 跑 `examples.full_mission.py`，或先在 SITL 跑 `examples/sitl_mission.py`。
4. **投放试验**：飞几次投放 → 量落点填 `impacts.jsonl` → `tools/fit_ballistics.py` →
   只把 `reliable=True` 的那组参数回填 `BallisticsConfig`。
5. **装弹实飞**：`DRY_RUN=False`，确认 PX4 侧 gripper 输出与 `abort_action` 符合空域要求。

## 7. 代码审查速查

### 7.1 不变量与硬规则（违反即 bug，逐条可查）

| # | 规则 | 代码位置 | 违反的后果 |
| --- | --- | --- | --- |
| 1 | `mavsdk` 没有公开 `close()`，会话结束必须显式释放，**且释放要有超时兜底**（守护线程 + `RELEASE_TIMEOUT_S`） | `telemetry/mavsdk_thread.py` 的 `stop()` / `_release_drone()` | mavsdk_server 子进程（固定 gRPC 端口 50051）变僵尸，新会话连上它并级联断连；⚠ 没有飞控时这条释放路径会**永久阻塞**（gRPC poller 报 `Event loop is closed`），没有兜底就会把 `stop()` 的调用方一起拖死 |
| 2 | 历史按时间查询用 `bisect(key=attrgetter("timestamp"))` 直接探 deque | `telemetry/broker.py` | 每次查询重建整张时间表（曾 22µs → 0.42µs） |
| 3 | 留存（`add_sink` 逐帧）与实时（`read()`/`latest()`）是两条路 | `video/source.py`、`video/buffer.py` | 用 `read()` 送入推理＝把丢帧引回来 |
| 4 | 跨帧持有画面必须 copy | `video/source.py` 的 `VideoFrame.copy()`、`video/buffer.py` | ffmpeg 后端复用管道缓冲，历史帧全变成"最新那一张" |
| 5 | 取遥测用 `frame.capture_timestamp`（= 收到时间 − `telemetry_lag`） | `video/source.py`、`video/align.py` | 目标坐标顺航迹偏一个链路延时（10 m/s → 1.5 m） |
| 6 | 快照是"最新可用值合并"，字段间无严格时间同步 | `telemetry/models.py`、`broker.py` | 误以为同一瞬间 → 解算误差归因错 |
| 7 | 去畸变只做一次（检测器 remap 过，georef 就不能再纠正） | `perception/detector.py` → `Detection.extra["undistorted"]` → `mission/targets.py` | 双重纠正，坐标系统性偏 |
| 8 | `Detection.code` 只由 `_emit(set_code=True)` 覆盖 | `perception/pipeline.py` | `cls12` 的类别编号被 OCR 路径清成 None |
| 9 | 缺失的姿态/位置**不猜**（计入 `attitude_missing`/`no_fix` 后跳过） | `mission/targets.py` | 用半个姿态解算出貌似合理的错坐标 |
| 10 | `ReleaseDecision.delay_s` 必须跟着决策走 | `ballistics/release.py`、`mission/runner.py` | 投放记录写错前推量，反演把延迟误差算到 Cd 头上 |
| 11 | 一次投放即锁存（`_released`） | `ballistics/release.py` | 重复投弹 |
| 12 | 判据失败**不编落点**（`ok=False` + `reason`） | `ballistics/model.py` | 用无意义预测触发投放 |
| 13 | 风取不到返回 `None`（≠ 零风），降级只记一次日志 | `ballistics/model.py`、`release.py` | 20Hz 每拍刷日志，或把"没有风估计"当"确实无风" |
| 14 | 质量是**称出来的输入**，不进反演；识别量是 `κ = Cd·A/m` | `ballistics/fit.py`、`docs/ballistics_fit.md` | 拟合出一组"看起来合理"的 Cd/质量分解 |
| 15 | `Cd` 与迎风面积只能固定一个（只以乘积可辨识） | `ballistics/fit.py` 的 `_DEGENERATE_PAIR` | 无穷多解 |
| 16 | 条件数超限 / 欠定 / 常数偏差 → 拒绝或单列，不硬给结论 | `ballistics/fit.py`、`tools/fit_ballistics.py` | 用户照抄不可信的参数 |
| 17 | 投放记录写盘失败**不改变任务状态**（弹已经出去了） | `mission/runner.py` 的 `_record_drop` | 因为磁盘满把任务判失败 |
| 18 | 命令失败 → `ABORT`；查询失败 → 只记一次日志继续（由超时兜底） | `mission/runner.py` 的 `_command`/`_query`/`_fail` | 20 Hz 反复记录同一条日志；或该失败时不失败 |
| 19 | 任务进度是**飞控侧状态**，"已飞完"的陈旧读数不算数（先见 False 再认 True） | `mission/runner.py` 的 `MissionMonitor` | 刚起飞就以为飞完了 |
| 20 | 降落项是**一项** `NAV_LAND`（`MissionItem.land`），且紧前一项必须高于落点 | `mission/planner.py`、`mission/items.py` | 拆成两项或被 PX4 判"进场点不在落点之上" ⇒ 整条任务被拒 |
| 21 | 未指定 = `NaN`（`UNSET`），**0 是有意义的值** | `mission/items.py` | 用 0 冒充"不指定"，飞控生成多余行为（接受半径 0 = 用 `NAV_ACC_RAD`） |
| 22 | 航点高度是**相对起飞点**；备用点换算 NED 时海拔 = 原点海拔 + 相对高度 | `mission/planner.py` | 几百米高度差 → 厘米级水平偏差 |
| 23 | 链路看门狗 `telemetry_stale_s`（默认 5s）与对齐的 1s 分开 | `mission/runner.py` 的 `_link_ok` | 单帧同步不上把整条任务判失败 |
| 24 | `ABORT` 只下**一条**安全动作 | `mission/runner.py` 的 `_do_abort_action` | 与飞控 failsafe 冲突 |
| 25 | 状态在**某一拍结束时**进入 | `mission/runner.py`、`mission/states.py` | 测试断言写成"进了状态就已经调用过" |
| 26 | `DroneController` 不在导入期读 `config` | `telemetry/controller.py`（装配走 `from_config`） | `config → video.source → telemetry` 成环 |
| 27 | 事件写入失败绝不影响控制流 | `mission/states.py` 的 `emit_event` | recorder 关闭后抛 RuntimeError 使状态机崩溃 |
| 28 | 权重必须 `best2.pt`；`cls12` 无 12 类权重（编号恒 1） | `config.py`、`docs/perception_ocr.md` | 换了废弃权重 → 零检出 |
| 29 | 转正形态门限 60°±15° 是**量出来的** | `perception/cropproc.py` 的 `HOUSE_APEX_ANGLE_DEG/TOL` | 凭主观判断调门限 → 误转正（曾把 56 读成 95） |
| 30 | 方向判别器只是**提示器**，不能当裁判 | `perception/cropproc.py`、`docs/perception_ocr.md` | 它会把目视正立的图判成倒置（帧 1160 实测 P(正立)=0.186） |
| 31 | 任务项一律走 **`mission_raw`**，不用 MAVSDK 的 `vehicle_action` 翻译 | `mission/items.py`、`telemetry/controller.py` 的 `to_raw_item`/`_upload` | `vehicle_action=LAND` 被拆成"同坐标航点 + `NAV_LAND`"⇒ PX4 固定翼拒**整条任务**，而 `start_mission()` 仍回成功 |
| 32 | 降落段必须过 `check_fixed_wing_landing`：紧前一项**严格高于**落点，且下滑角 ≤ `tan(FW_LND_ANG+0.1°)` | `mission/plan_file.py`、`mission/planner.py` | 上传后被飞控整条拒掉（`No valid mission available, loitering`），飞机原地盘旋 |
| 33 | 上传启动后**必须确认**飞控真进了 `MISSION`（`mission_start_timeout_s`） | `mission/runner.py` 的 `_confirm_started`、`telemetry/controller.py` 的 `in_mission_mode` | "命令回成功"当成"在飞"⇒ 空等到状态超时（实测 15 分钟） |
| 34 | 完成判定读快照 `mission_current == mission_total`（`mission_raw` 进度流） | `telemetry/controller.py` 的 `mission_finished`、`mavsdk_thread.py` 的 `_stream_mission_progress` | 用 MAVSDK `mission.is_mission_finished()`：raw 上传下**永远回 False**，状态机卡在 RECON |
| 35 | 上传前**先 `set_current_mission_item(0)`** 再启动 | `telemetry/controller.py` 的 `start_mission` | 同 CRC 的任务再启动时飞控不清"已飞完"锁存 ⇒ 停在 `HOLD` |
| 36 | 每条腿的航线来源**二选一**（waypoint 配置 / QGC `.plan`），同时给两个直接报错 | `config.py` 的 `validated`、`mission/planner.py` | 飞的可能不是你以为的那条航线 |
| 37 | `.plan` 复杂项按 QGC 的展开逻辑在本地展开；不支持的复杂项**显式报错** | `mission/plan_file.py` | 猜着展开 ⇒ 上传一条 QGC 里根本没画过的航线 |
| 38 | 同一时刻只允许一个 `mavsdk_server`；调试要接**已有** server | mavsdk 库（固定 gRPC 50051） | 第二个 `System()` 争抢端口，两个客户端互相冲突 |
| 39 | 正式流程 `INIT → PREFLIGHT → WAIT_AIRBORNE → RECON`：先自检（载入模型 + 视频自检），再等"在空中"，最后才进侦查；等待起飞的检查有**临时放行开关** `MissionConfig.require_airborne`（默认 `True`），置 `False`（**只给地面演练/离线测试**）时该检查立即放行，且**必须留痕**——记 `airborne_skipped` 事件（带 `reason`）+ WARNING 日志，绝不静默通过 | `mission/runner.py` 的 `_tick_preflight`/`_tick_airborne`、`config.py` 的 `MissionConfig.require_airborne`、`airdrop/preflight.py` | 在停机坪上就进侦查：PX4 会在地面"追"第一个航点，或因任务不可行直接盘旋；演练则相反——飞不飞都空等到 `airborne_timeout_s`（1800s）才失败 |
| 40 | 侦查航线**默认由操作手在 QGC 上传并启动**（`MissionConfig.recon_upload='operator'`）；`'auto'` 只用于自动测试 | `config.py` 的 `RECON_UPLOAD_MODES`、`runner._begin_recon` | 自动上传会覆盖操作手刚画好的航线 |
| 41 | 自检**关掉 ≠ 通过**：关掉的项记 `preflight` 事件（`ok=null`）；**开着却没注入回调 = 装配错误**，直接判失败 | `airdrop/preflight.py` 的 `PreflightCheck.skipped` / `PreflightError` | 把"漏了装配"当成"检查通过"，正式任务带着没自检的系统起飞 |
| 42 | OCR 请求 / OCR 结果 / 检测结果三条队列都**无界、不丢弃**；积压只告警（`ocr_queue_size` 与结果队列 2000 条阈值），绝不因为"队列满"丢数据 | `perception/ocr_worker.py` 的 `submit`/`_warn_if_backlogged`、`perception/pipeline.py` 的 `_emit`/`_collect_remaining` | 丢掉一个送检请求或一条结果 ⇒ 可能漏掉只清晰一瞬的目标编号（编号是坐标解算的输入） |
| 43 | 弹道密度以**真实海拔**为基准：`ground_altitude_m` 来自 GPS 原点/地面点（反演取记录 `origin.alt_m − ground_z`）；关闭 ISA 也在投放海拔算一次常密度 | `ballistics/model.py` 的 `predict_impact`、`mission/runner.py` 的 `_ground_altitude`、`ballistics/drops.py` 的 `predict_record_impact` | 把地面当海平面：1500 m 站点高估密度 ~16% ⇒ 高估阻力、预测落点偏近；反演参数迁移到别的海拔也会错 |

### 7.2 失败语义一览（"到底怎么算失败"）

| 调用 | 成功 | 失败表现 |
| --- | --- | --- |
| `DroneController.upload_mission/start_mission/hold/rtl` | 返回 `None` / 条数 | **抛** `ControllerError`（未连接、被拒、超时）；上传后**回读不一致**也抛 |
| `DroneController.gripper_release()` | `True`（指令已发出） | `False` = "读到了、但确实没投"（未启用/被拒）；异常仍抛 `ControllerError` |
| `DroneController.mission_finished()` / `in_mission_mode()` | `True`/`False` | 抛 `ControllerError`（进度/模式流还没数据；调用方按"查询失败只记一次"处理） |
| `DroneController.request_origin()` | `NedOrigin` | `None` = 原点**尚未就绪**（不是错误） |
| `load_plan(path)` | `QgcPlan` | 抛 `PlanError`（读不了 / 不是 QGC plan / `SimpleItem` 之外的复杂项 / 参数不全） |
| `build_drop_mission(...)` | `DropMissionPlan` | 抛 `PlanningError`（缺 NED 原点 / 降落段为空 / 降落几何不合格 / 目标与备用点都没有） |
| `MissionRunner` 的启动确认 | 记 `mission_confirmed` | `ABORT('mission_not_started')`（模式迟迟不进 `MISSION`）；模式流不可用时只告警并退化 |
| `BallisticsModel.predict_impact()` | `Impact(ok=True, ned=…)` | `ok=False` + `reason`（`below_ground`/`timeout`），`ned=None` |
| `ReleaseJudge.update()` | `ReleaseDecision`（`should_release` 真/假） | 状态不足时 `reason="unknown_position"`；预测不出时 `no_prediction:*` |
| `fit_ballistics()` | `FitResult(ok=True, …)` | `ok=False` + `reason`（无样本/退化/欠定/不可辨识/初值越界） |
| `targeting.analyze()` | `TargetingResult`（`ok=False` 表示没选出唯一结果） | 不抛异常（空输入 → 空结果） |
| `pixel_to_ned()` | `GroundIntersection(ok=True)` | `ok=False` + `reason`（视线朝上/与地面夹角过小），`ned=None` |
| `FrameTelemetryAligner.align()` | `AlignedSample`（带 `extrapolated`/`offset` 标记） | `None` = 该帧丢弃（超 `max_wait` / 超 `max_extrapolation`），计入 `AlignStats` |
| `MavsdkThread.connect()` | `System` | 抛异常（超时/连不上）；`connected` 属性可用于轮询 |
| `load_camera_model()` | `CameraModel`（`is_calibrated()` 区分来源） | **不抛**：退回默认模型 + warning |
| `Hm30VideoSource.start()` / `read()` | `self` / `VideoFrame` | 地址错 → `stats.last_error` + error 级日志；`read()` 超时返回 `None` |

### 7.3 协议与注入点（离线可测性的来源）

| 协议 / 注入参数 | 定义处 | 测试替身 |
| --- | --- | --- |
| `MissionController`（Protocol） | `telemetry/controller.py` | `tests/test_mission.py::FakeController`、`TimedController` |
| `ReleaseJudgeLike`（Protocol） | `mission/runner.py` | `FakeJudge`（按脚本给决策） |
| `target_result` / `target_busy`（回调） | `MissionRunner.__init__` | lambda 返回构造好的 `TargetingResult` |
| `DetectionReader`-类协议 `PerceptionConfigLike` | `perception/pipeline.py` | `PerceptionWorker(detector=…, pool=…)` 直接注入假对象 |
| `clock` / `sleep` | `MissionRunner`、`DroneController` | `FakeClock`（分钟级任务毫秒级跑完） |
| `on_event`（`(kind, data)`） | 几乎所有模块 | 收集到 list 里断言 |
| `on_drop`（`(DropRecord)`） | `MissionRunner` | 收集到 list 里断言 |
| `broker` / `buffer`（构造注入） | `FlightRecorder.start()`、`ReplayVideoSource` | 真 broker/缓冲 + 手写素材 |

### 7.4 常见扩展点：改哪几处

| 想做的事 | 要动的地方 |
| --- | --- |
| 换检测模型 | `PerceptionConfig.model_path/device/imgsz/conf_threshold`（权重放 `models/`，跑 `tools/fetch_models.py`） |
| 新增一种目标形态（非五边形） | `perception/cropproc.py` 的掩码与形态判据 + `docs/perception_ocr.md` 的实测流程（**先量分布再定门限**） |
| 加一个任务状态 | `mission/states.py` 的 `MissionState` + `TRANSITIONS` + `runner.py` 的 `update()` 分派与 `_tick_*` + `tests/test_mission.py` 的状态机用例 |
| 换投放机构（非 MAVSDK gripper） | 实现 `MissionController.gripper_release()`（或包一层，参考 `DryRunController`） |
| 换弹道模型（加升力/风场） | `ballistics/model.py` 的 `_acceleration`/`predict_impact`（保持 `Impact` 契约），反演侧 `ballistics/fit.py` 自动跟着走 |
| 新增一个可拟合参数 | `ballistics/fit.py` 的 `FIT_PARAMETERS` + `bounds_for`/`initial_for`/`with_parameters` + `tools/fit_ballistics.py` 的开关与报告（**先想清楚它跟已有参数是否退化**） |
| 换坐标解算（比如带高程图） | `georef/project.py`（`pixel_to_ned` 的 `ground_z` 来源）与 `GroundConfig` |
| 新增一个离线工具 | 放 `tools/`，写成**纯库模块**：常量 = 默认值、`build_config(**覆盖)`、`main(**kwargs)` 返回退出码（**不要自己 import argparse**），再到 `airdrop/run.py` 的 `SUBCOMMANDS` 注册一个子命令（`--help` 不许加载重库，重依赖放函数体内）；`tests/test_cli.py` 会校验注册表与帮助信息 |
| 新增一个示例 | 放 `examples/`，同样写成纯库模块：`build_config(**覆盖)` 暴露参数（文件内常量 = 默认值）、`main(**kwargs)` 返回退出码；`tests/test_examples.py` 会自动校验"不脱节 + 导入不加载重库" |

### 7.5 审查清单（照抄即可用）

- [ ] 新增/修改的公开 API 是否进了 `__all__` 与顶层 `airdrop/__init__.py`？跑 `tools/check_docs.py` 看文档是否跟着更新。
- [ ] 有没有在业务代码里硬编码数值（应进 `config.py`）？
- [ ] 新增参数有没有进 `Config.validated()` 的取值域检查？
- [ ] 失败路径是否**显式**（抛错 / `ok=False` / `None` 三选一，并在 docstring 写明）？
- [ ] 是否碰了 §7.1 里 43 条不变量中的任何一条？改了哪条就要同步 `AGENTS.md` 与对应专题笔记。
- [ ] 碰了任务项 / 航线来源 / 启动与完成判定（§7 第 31~35 条）时，除了离线用例，**跑一次 SITL 演练**（`examples/sitl_mission.py`）——这四条都是"离线测不出来、真飞控才暴露"的。
- [ ] 回归用例加了吗（`tests/` 是回归落点）？跑过 `pytest -m "not realdata and not sitl and not stream"` 吗？
- [ ] 三份文档是否同步：`README.md`（怎么用）、`AGENTS.md`（约定与易错点）、本手册（目录/参数/产物）？

---

## 8. 常见问题与排障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 图传一帧都没有，`stats.last_error` 有 ffmpeg 输出 | 地址/网络不对（HM30 网段 `192.168.144.0/24`，相机 `192.168.144.25`） | `ffmpeg -rtsp_transport udp -i rtsp://192.168.144.25:8554/main.264 -frames:v 1 -f null -` 手工核对；地址**不探测**，写错就报错 |
| 断了流仍在读旧画面 / `read()` 阻塞不返回 | ffmpeg 后端靠 `-timeout` 判死链路；cv2 后端 `read()` 30s 不可中断 | 用 `Hm30VideoSource`（可杀子进程）；`VideoConfig.read_timeout` 调小；详见 `docs/video_hm30_ffmpeg.md` |
| 新会话连不上飞控 / 端口 50051 被占 | 旧 mavsdk_server 僵尸进程 | 必须走 `MavsdkThread.stop()`（内部 `_release_drone()`）；必要时手动结束残留进程 |
| ONNX/TORCH 报 CUDA 不可用，静默退回 CPU | 进程里没有先 `import torch`（cuDNN/cuBLAS DLL 不在搜索路径） | 本项目 det/rec 走 TORCH 天然满足；`OcrEngine._report_cls_device()` 会告警——看日志 |
| OCR 读出 `95` 而目视是 `56` | 转正差 180°（形态门限被绕过） | 已修（形态判据 + `house_ok` 优先）；别改 60°±15°，要改先重新量分布（`docs/perception_ocr.md`） |
| 单帧编号错（如 `01`） | OCR 单帧误读 | 由 `targeting` 的**类内众数**兜底；多帧观测足够时自动纠正 |
| `drops.jsonl` 是空的 | 本次没有实际投放（判据没触发/未接判据 `no_judge`/投放被拒） | 看 `events.jsonl` 的 `drop_skipped`/`error`/`abort_action` |
| 反演报"参数不可辨识（条件数 …）" | 同高同速投放 → Cd 与释放延迟分不开 | 拉开各次投放的速度/高度；或先只拟合 Cd；有风反而更容易分离（`docs/ballistics_fit.md`） |
| 反演报"参数退化" | 同时勾了 Cd 与迎风面积（只以乘积可辨识） | 固定一个 |
| 反演 `reliable=False` | 自由度不足 / 顶到边界 / 常数偏差大 | 看 `warnings`；若提示常数偏差，先核对目标点与测量点坐标，**别调参** |
| 任务进 `ABORT` 后飞机没有动作 | `abort_action="none"`，或链路断了指令发不出去 | 看 `events.jsonl` 的 `abort_action`；链路断时交给飞控 failsafe |
| 刚起飞就以为任务飞完 | 任务进度是飞控侧状态，新任务刚上传时仍可能是上一条的"已飞完" | 已由 `MissionMonitor`（先见 False 再认 True）挡住；若自写循环要照做 |
| 任务"上传成功"飞机却原地盘旋 | 降落剖面被 PX4 可行性检查拒了整条任务（`mission_result.valid=0` ⇒ `No valid mission available, loitering`），而 `start_mission()` 仍回成功 | 看 ulog 的 `mission_feasibility_checker` 消息；本地先跑 `check_fixed_wing_landing`；降落段优先用 QGC `.plan`（§7 第 31/32 条） |
| 状态机停在 `RECON`，进度显示已到末项 | 完成判定用了 MAVSDK `mission.is_mission_finished()`——**raw 上传下它永远回 False** | 用快照的 `mission_current`/`mission_total`（`current == total` 即飞完，§7 第 34 条） |
| 启动后飞机停在 `HOLD` 不进任务 | 新任务与上一条**内容相同**（同一 CRC）时，飞控不清"已飞完"锁存 | `start_mission()` 已先 `set_current_mission_item(0)`；用 `in_mission_mode()` 确认（§7 第 33/35 条） |
| QGC 里"任务没在进行" | QGC 的活动任务视图按**它自己上传/下载**的任务同步，raw 上传的它不认 | 以 `events.jsonl` 的 `state`/`mission_confirmed` 与遥测 `mission_current` 为准；需要 QGC 显示就先在 QGC 里 download 一次 |
| 第二个脚本连不上飞控 / 争抢 50051 | 同一时刻只允许一个 `mavsdk_server` | 调试脚本用 `System(mavsdk_server_address="localhost", port=50051)` 接已有 server（§5.6） |
| `mavsdk_server --version` 长时间不返回 | 那个二进制没有 `--version`，它会去起服务并阻塞 | 版本看日志首行 `mavsdk_server: MAVSDK version: vX.Y.Z`（与 Python 包版本一致） |
| 目标坐标整体偏一段距离 | 没标定（默认外参）、或 `telemetry_lag` 是旧值 | 跑标定并把 `telemetry_lag` 回填；核对 `camera_calib.json` 的 `R_bc`/`t_bc` |
| `models/best2.pt` 找不到 | 没取权重 | `python -m airdrop.run fetch-models`（`--source-dir` 可覆盖文件内常量） |
| `examples.full_mission` 起不来 | 缺 `camera_calib.json`/航线为空/RTSP 地址不对 | 看 `flight.log`；`Config().validated()` 会提前挡掉取值域错误 |
| 状态机停在 `INIT` | 没等到遥测位置或 NED 原点（`INIT` 本身不上传任何任务） | 看 `flight.log`；`init_max_s` 超时会 `ABORT(init_no_telemetry` / `init_no_origin)` |
| 状态机停在 `PREFLIGHT` | 模型载入无响应/抛异常，或视频自检没等到足够的帧 | 看 `preflight` 事件与 `flight.log`：任一项失败 → `ABORT(preflight_failed:<check>)`；超 `preflight.max_s` → `preflight_timeout`；没有相机就把 `PreflightConfig.check_video` 关掉 |
| 状态机停在 `WAIT_AIRBORNE` | 飞控没报 `in_air`，相对高度也没超过 `airborne_alt_m` | 这是**正常**的等起飞；`airborne_timeout_s`（默认 1800s）超时会 `ABORT(airborne_timeout)`；核对 `in_air` 可选流是否就绪 |
| `preflight` 事件里某项 `ok=null` | 该项在 `PreflightConfig` 里被关掉了——**"没把关"不等于"通过"** | 正式任务前打开它并注入对应 `model_loaders`；若是 `preflight_skipped`，说明整个自检对象都没注入（离线/演练档） |
| 侦查段一直不开始（`RECON` 里空等） | `recon_upload='operator'` 时本包**不上传**，要操作手在 QGC 上传并启动 | 让操作手在 QGC 里上传并 start；或自动测试档设 `recon_upload='auto'` |
| `Config(...)` 构造后参数没生效 | 忘了 `.validated()` / 派生时漏了某一层 | 用 `dataclasses.replace` 逐层派生，最后 `.validated()` |

---

## 9. 已知边界与维护约定

### 9.1 已知边界与未验证项（别当成已完成）

| 项 | 现状 |
| --- | --- |
| 弹道参数 | 质量 0.365 kg 是**称重占位值**（要实测填入）；`drag_coefficient=0.6`、`cross_area_m2=0.004` 是 350ml 水瓶估计；**反演工具就绪但本地没有真实投放数据** |
| 飞掠段长/高度/航向 | `leg_length_m=200`、`altitude_m=20` 待实验验证 |
| 相机标定 | 只在合成链路上验证过；真实素材复测与 `telemetry_lag` 实测值待一次真实采集 |
| SITL 演练 | **已在本地 WSL + PX4 SITL 固定翼（`gz_rc_cessna`）上真跑通过**：`RECON → HOLD_PROCESS → OVERFLY → LAND → DONE`，2 次上传 / 1 次投放 / 0 错误，投放触发时预测落点误差 1.35 m。⚠ 演练**不验感知**（SITL 没相机，目标坐标由 `TARGET_OFFSET_NED` 合成），且 SITL 里必须先手动 **arm 并起飞**（`WAIT_AIRBORNE` 要等 `in_air`/相对高度 ≥ `airborne_alt_m` 才上传侦查航线；上面那次记录是加这道检查之前跑的，重跑按新顺序） |
| `.plan` 航线 | 解析与复杂项展开在 `tests/test_plan.py`（现场造的样例）+ 本地 `routes/land.plan` 上验证过；真实任务航线由运营方在 QGC 里提供，**只支持 `fwLandingPattern`**（VTOL 降落、测绘/结构航线显式报错） |
| 生成航点的接受半径 | `DEFAULT_WAYPOINT_ACCEPTANCE_M = 3.0` 对固定翼偏紧——SITL 里出现过飞机绕着末航点转、迟迟不"到点"（进度停在 `current=2/total=3`）；QGC `.plan` 的航点自带 `param2`，不受这条影响。真机前建议按机型确认 `NAV_ACC_RAD` 与此值 |
| `cls12` 模式 | 无 12 类权重（`best2.pt` 单类）→ 编号恒为 1，代码路径可跑但无实际意义 |
| 方向分类交叉验证 | 判别器会误报（帧 1160 实测），只能当提示器；门限 0.9 恰好挡住误报 |
| PX4 侧配置 | gripper 输出、起飞项航点动作、任务结束后是否 RTL 都不在本包范围内 |
| 仿真世界 | `sim/worlds/cuadc/` 的两个赛区世界已按规则（2026 版第 19~25 页）搭好并做离屏渲染核对（几何/尺寸/贴图方向，见 [`simulation_world.md`](simulation_world.md)）；天井位置与朝向按规则随机（`--seed` 复现，入库 = seed 0）；**未验证**：把 Gazebo 相机接进感知闭环、真机与世界的差异 |
| 工程设施 | 无 CI（pytest/ruff/pyright 只在本地跑）；覆盖率 ≈85% |
| 声明但尚未接入的参数 | 见 §4.18 的表（`save_crops`/`crop_dir`/`max_crops`/`lag_warn_frames`/相机 `undistort`/`flat`+经纬度/记录 `video`/`event_eval_hz`） |

### 9.2 数据与结果的可复现性

- **权重不进 git**：拿权重后配 `config_snapshot.json` 才知道当时用的是什么；`models/ppocr/*.txt` 字典入库，
  正是为了让 rec 的字符集固定。
- **一次飞行的全部输入都在飞行目录里**：`config_snapshot.json` + `telemetry.jsonl` + `frames/`，
  配合 `ReplayVideoSource` 可离线重跑同一条链路；改动算法后**先跑 `examples/replay_flight.py` 对比**
  再考虑实飞。
- **投放试验的结论是可追溯的**：`ballistics_fit.json` 里同时留着原始 `record` 与 `impact_ned`、
  以及用了哪次实测（`impact_source`）。

### 9.3 本文档的维护

本文档的**签名与默认值来自机器自省**，不靠手抄：

```bash
# 生成/刷新接口清单（写入 docs/api_reference.md，供人工比对）
./.venv/Scripts/python.exe -m tools.dump_api

# 核验本文档与代码是否一致（参数齐全 / `airdrop.*` 引用可解析 / 产物文件名齐全 / 无编造参数名）
./.venv/Scripts/python.exe -m tools.check_docs
```

改动代码后：① 跑 `tools.check_docs`；② 参数变了同步 §4；③ 新增产物/字段同步 §5；
④ 不变量变了同步 §7 与 `AGENTS.md`；⑤ 边界变了同步 §9.1 与 `README.md` 的"已知边界与风险"。

