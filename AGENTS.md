# AGENTS.md — airdrop 项目工作手册

> 面向在本仓库工作的 AI 智能体。动手前请先通读本文档；内容与代码冲突时以代码为准。

## 项目概述

无人机空投（AirDrop）控制系统，Python 包名 `airdrop`，当前版本 0.1。项目面向无人机竞赛的
"侦察与打击"任务（察打一体），设计约束为**不使用机载计算机**：感知、坐标解算与弹道预测
全部在地面计算机完成。已实现：

- 遥测：MAVSDK 工作线程 + 遥测代理 + 任务级控制器；
- 视频：RTSP 拉流、帧-遥测对齐、环形缓冲；
- 感知：YOLO 检测 + 五边形转正 + OCR 读编号；
- 坐标处理：像素 → NED → WGS84（georef）；
- 目标统计（targeting）；
- 弹道与投放（ballistics），含投放记录与弹道参数反演；
- 任务编排（mission）：状态机 + 航线拼接 + 坐标解算 + 主循环；
- QGC `.plan` 航线导入（固定翼降落航线复杂项展开 + 降落几何预检）；
- 起飞前自检（preflight：载入 detector/ocr/camera → 视频自检，各项可单独关闭）；
- 飞行记录与回放；三步标定流水线；端到端离线链路（`tests/test_e2e.py`）。

主要依赖：mavsdk、torch+cu130、ultralytics、opencv、rapidocr、scipy、pyproj、scikit-learn、onnxruntime-gpu。

## 文档地图

| 文档 | 内容 |
| --- | --- |
| `AGENTS.md`（本文件） | 工作手册：环境 / 目录结构 / 架构要点 / 关键约定 / 验证方式 / 模块约定补充 |
| [`docs/handbook.md`](docs/handbook.md) | 项目手册（面向用户与代码审查者）：目录总览、数据流与依赖、各模块调用方法与最小示例、参数总表（config 全字段）、输入输出字段、使用流程、审查速查、排障表 |
| [`docs/calibration_opencv.md`](docs/calibration_opencv.md) | 三步标定流水线（内参 → 时间差 → 手眼外参）及 OpenCV 5.0 的 API 约束：三类分因、`calibrateHandEye` 状态与替代方案、手眼方程与 PnP 要求、杆臂 `t_bc` 判据、不应采用的做法 |
| [`docs/video_rtsp.md`](docs/video_rtsp.md) | RTSP 视频链路：ffmpeg 后端要求、缓冲与丢帧语义、启动阶段帧丢失范围（`≈probesize`）、地址与分辨率约定 |
| [`docs/perception_ocr.md`](docs/perception_ocr.md) | 检测/OCR 流水线细节：五边形转正与顶角形态门限、OcrEngine/Cls 与权重命名、onnxruntime-gpu 的 CUDA 前提与速度表、转正形态与交叉验证要求 |
| [`docs/simulation_world.md`](docs/simulation_world.md) | CUADC 赛区 Gazebo 世界：规则条款 → 几何映射、天井随机摆放与朝向、五边形环壁与靶标资产、SITL 启动命令、`make-world` 重新生成与换场地 |
| [`docs/ballistics_fit.md`](docs/ballistics_fit.md) | 投放记录 + 弹道参数反演：数据结构、可辨识性判据（质量称重 / 识别量 κ）、常数偏差、σ 与留一验证、工具用法 |

专项文档与本文档同等权威；两者冲突时，以代码与专项文档中的实测记录为准。

**修改代码后运行 `python -m airdrop.run check-docs`**（等价于 `python -m tools.check_docs`）：
该命令核验 `docs/handbook.md` 与代码的一致性（参数齐全 / `airdrop.*` 引用可解析 /
产物文件名齐全 / 无编造参数名）。`docs/handbook.md` §3 的示例由 `tests/test_handbook.py` 逐条执行。

对外说明的分工：README 只说明使用方法；实现细节、参数依据与约束保留在本文档与 `docs/` 专项文档中。

## 环境

- **Python ≥ 3.14**；项目虚拟环境 `.venv`（uv 创建，3.14.6，**不包含 pip**）。
- 运行 Python：`./.venv/Scripts/python.exe`（Windows）。
- 安装依赖：`uv add` / `uv add --dev`；一次性工具：`uv tool run`。
  若当前环境不满足需求或添加依赖可以更好地完成任务，应主动向用户提出；
  **安装任何工具或依赖前必须先征得用户同意**。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `airdrop/config.py` | 全部参数集中处（frozen dataclass + 取值域校验）。命令行解析不在本文件：集中于唯一运行模块 `airdrop/run.py`（argparse 子命令）：`python -m airdrop.run <子命令> --help`（子命令与选项表见 `SUBCOMMANDS`） |
| `airdrop/telemetry/` | `models.py`（数据模型）、`broker.py`（遥测代理）、`mavsdk_thread.py`（MAVSDK 工作线程）、`controller.py`（任务级控制：`mission_raw` 上传 + 回读校验 / 启动（先复位）/ 任务模式确认 / hold/rtl、gripper、NED 原点） |
| `airdrop/video/` | `source.py`（RTSP 拉流）、`align.py`（帧-遥测时间对齐）、`buffer.py`（对齐结果环形缓冲） |
| `airdrop/record/` | `recorder.py`（FlightRecorder：飞行目录五类记录文件 + `drops.jsonl` 写入磁盘）、`replay.py`（ReplayVideoSource + 遥测回填） |
| `airdrop/perception/` | `detector.py`（YOLO 封装）、`cropproc.py`（五边形 → 转正 → RapidOCR）、`number.py`（编号纠错）、`ocr_worker.py`（OCR 独立进程）、`pipeline.py`（主循环）、`models.py`（Detection/PixelBox） |
| `airdrop/targeting/` | `models.py`（TargetPoint / Cluster / TargetingResult）、`cluster.py`（DBSCAN 聚类 + 编号众数 + median/max 选唯一结果） |
| `airdrop/ballistics/` | `model.py`（二次阻力 + RK4 定步长 → 落点/飞行时间，含 ISA 密度与风）、`release.py`（ReleaseJudge：预测触发 + 强制投放 + 一次性锁存）、`drops.py`（DropRecord 投放瞬间状态 + 实测落点读写/配对）、`fit.py`（`fit_ballistics`：最小二乘反演 + 可辨识性诊断） |
| `airdrop/mission/` | `items.py`（`MissionItem`：MAVLink 级任务项，command/frame/params/位置 + `MAV_CMD_*` 常量）、`plan_file.py`（QGC `.plan` 解析：`SimpleItem` + `fwLandingPattern` 复杂项展开 + 固定翼降落预检）、`states.py`（状态 + 合法转移表 + 历史）、`planner.py`（侦察段 / 飞掠 entry-exit / 与降落段拼接，纯函数；每条腿可来自配置航点或 `.plan`）、`targets.py`（TargetTracker：检测结果 → georef → 目标点 → 统计结果）、`runner.py`（MissionRunner 主循环：驱动控制器、监视遥测、推进状态机） |
| `routes/` | 在 QGC 中绘制的 `.plan` 航线（随仓库入库）：`land.plan` 为飞掠段之后的返航 + 降落段（含固定翼降落航线复杂项）；飞掠段由本包插入其前 |
| `airdrop/preflight.py` | 起飞前自检（正式流程第一步）：载入 detector/ocr/camera → 视频自检（窗口内帧数 + 分辨率与标定一致）；`Preflight`/`PreflightCheck`/`PreflightError`/`PreflightLike`。**"检查关闭"≠"检查通过"**：关闭的项记 `preflight` 事件（`ok=null`）；**开启但未注入载入回调为装配错误，直接失败** |
| `models/` | 本地模型权重（**不进 git**，用 `python -m airdrop.run fetch-models` 复制）；`models/ppocr/*.txt` 字典入库以固定字符集 |
| `sim/` | 仿真模块（世界 + 机型 + airframe + 启动脚本，仓库为唯一出处）：`sim/worlds/cuadc/`（生成产物，请勿手工修改：两个 `.sdf` + 网格/贴图）、`sim/vehicles/rc_cessna_down_cam/`（带下视相机的 Gazebo 机型：merge 基础 `rc_cessna` + 720p 相机）、`sim/airframes/4007_gz_rc_cessna_down_cam`（PX4 airframe）、`sim/patches/`（可选 PX4 补丁）、`sim/install_px4.sh`（软链进 PX4，幂等）、`sim/run_sitl.sh`（一键启动 SITL）；世界生成器 `tools/make_world.py`，布局/启动/装机说明见 [`docs/simulation_world.md`](docs/simulation_world.md) 与 [`sim/README.md`](sim/README.md) |
| `tools/` | `fetch_models.py`（复制外部权重到 `models/`）、`calibrate.py`（三步相机标定，见 [`docs/calibration_opencv.md`](docs/calibration_opencv.md)）、`fit_ballistics.py`（投放试验反演，见 [`docs/ballistics_fit.md`](docs/ballistics_fit.md)）、`make_world.py`（生成 CUADC 赛区世界，见 [`docs/simulation_world.md`](docs/simulation_world.md)）、`make_backdrops.py`（生成目标区地面纹理与程序化航拍底图）、`fetch_aerial.py`（获取真实航拍底图：USGS NAIP，公有领域，需联网）、`preview_world.py`（渲染带标注的俯视预览图）、`dump_api.py`（自省导出接口清单到 `docs/api_reference.md`）、`check_docs.py`（核验 `docs/handbook.md` 与代码一致）。全部为纯库模块：文件内常量即默认值，暴露 `build_config(**覆盖)` / `main(**kwargs)`，**不自行 import argparse**；命令行集中于 `airdrop/run.py`，例如 `python -m airdrop.run calibrate --flight flights/20260913-185512 --strict` |
| `examples/` | `full_mission.py`（完整任务：遥测/视频/感知/判据/状态机/记录）、`replay_flight.py`（回放全链路离线迭代）、`calibration_capture.py`（标定素材采集）、`sitl_mission.py`（SITL 演练：合成目标 + DryRunController）、`basic_usage.py` / `hm30_video.py` / `video_telemetry_sync.py`（基础演示）。全部为纯库模块：文件内常量即默认值，暴露 `build_config(**覆盖)` / `main(**kwargs)`，重型依赖只在函数体内导入（`import examples.x` 不加载 cv2/mavsdk/torch）；命令行集中于 `airdrop/run.py`，例如 `python -m airdrop.run full-mission --land-plan routes/land.plan --no-preflight` |
| `tests/` | pytest 套件（全部离线）：`conftest.py`（公共 fixture：本地 H.264 测试流、`live_source` 工厂、`broker`）+ `test_config.py` / `test_telemetry.py` / `test_alignment.py` / `test_buffer.py` / `test_video.py` / `test_recorder.py` / `test_replay.py` / `test_perception.py` / `test_perception_realdata.py`（标 `realdata`，默认跳过）/ `test_georef.py` / `test_calibrate.py` / `test_targeting.py` / `test_ballistics.py` / `test_mission.py` / `test_e2e.py`（回放驱动全链路）/ `test_fit.py`（投放记录 + 弹道参数反演）/ `test_examples.py`（示例不脱节 + 导入不加载重型依赖）/ `test_cli.py`（`airdrop/run.py`：子命令注册表、`--help` 不加载重型依赖、参数校验、选项覆盖进 `build_config`）/ `test_handbook.py`（`docs/handbook.md` §3 示例逐条执行）/ `test_plan.py`（QGC `.plan` 解析 + 复杂项展开 + 固定翼降落预检）/ `test_world.py`（CUADC 赛区世界：生成结果与入库文件一致、目标区在起飞线两端、天井区内随机摆放且间距 > 20m、五边形环壁、同区同色、贴地标线不共面重叠、目标区底色与周边航拍底图均在比赛区域外、底图生成器可复现、两轮靶标摆位、资产与贴图朝向） |

## 架构要点

- **MavsdkThread**：专用后台线程运行独立 asyncio 循环；supervisor 循环负责连接与重连；
  **6 条核心遥测流**（position / home / position_velocity_ned / attitude_euler /
  attitude_quaternion / gps_global_origin）经 `asyncio.TaskGroup` 并行采集，
  **任一核心流结束即视为断连**，交由 supervisor 按配置重连；指令经
  `run_coroutine_threadsafe` 串行执行。默认连接 `udpin://0.0.0.0:14540`。
  - **风估计为"可选流"（第 7 条）**：`drone.telemetry.wind()` 走 `_optional_stream`，
    结束或报错只记日志、不影响会话。风仅影响弹道精度（缺风时按零风降级并记录一次日志），
    且飞控未必支持或开启风估计；不应让该增强流触发整条遥测链路重连。
    **不得将其改为 `_guarded`。**
  - **另有两条可选流**：`mission_raw.mission_progress()`
    → 快照 `mission_current`/`mission_total`（完成判定的唯一来源），`telemetry.flight_mode()`
    → 快照 `flight_mode`（启动确认用）。二者不可用时相关功能降级：完成判定退回状态超时、
    启动确认跳过（仅告警），不影响会话判断。
  - 快照中的风字段为 `wind_north/east/down_m_s`（MAVSDK 侧为 `wind_x/y/z_ned_m_s`，
    在 `update_wind` 中映射）；与其他数值字段一样参与 `get_snapshot_at` 的线性插值。
- **TelemetryBroker**：`RLock` + `Condition` 保证线程安全；最新快照采用浅拷贝（字段均为标量）；
  历史 `deque(maxlen=1200)` 按 `history_interval=0.1s` 节流入库（约 2 分钟），
  最新快照与订阅推送不节流；`get_snapshot_at(ts, mode)` 支持 `interpolate`
  （范围内插值、范围外外推）与 `nearest`。
- **插值规则**：数值字段线性插值；欧拉角走最短弧（处理 ±180° 环绕）；四元数 slerp
  （含 q/-q 符号翻转，保持单位模长）。
- **FlightRecorder**（`record/recorder.py`）：将一次飞行记录为 `flights/<时间戳>/` 下的记录文件——
  `flight.log`（挂接在 root logger 上，业务模块无需感知）、`telemetry.jsonl`（默认 10Hz 节流）、
  `detections.jsonl`、`events.jsonl`、`frames/%06d.jpg` + `frames_index.jsonl`、
  `config_snapshot.json`，以及 `drops.jsonl`（每次实际投放一条 `DropRecord`；
  接法为 `MissionRunner(on_drop=recorder.drops.append)`，需用 lambda 延迟获取
  `recorder.drops`，写入器在 `start()` 之后才存在）。帧零重编码：直接写入缓冲中的
  jpeg 字节（`read_jpeg_bytes`）。录制器只写入有效快照（`TelemetrySnapshot.is_valid()`）——
  初始空快照会成为历史区间最左端的全空端点，导致最早几帧插值到空值。
- **ReplayVideoSource / TelemetryPacer**（`record/replay.py`）：按原时间轴重放飞行目录，
  离线迭代识别与坐标解算时与实飞使用同一代码路径。
  - `ReplayVideoSource` 与 `Hm30VideoSource` 公开接口一致（`add_sink`/`read`/`latest`/
    `stats`/`iter_frames`/`stop`），可直接替换。帧写入 `timestamp = capture_timestamp + lag`，
    因此 `frame.capture_timestamp` 与录制时逐位一致；`speed=0` 全速（批量回归）、`1.0` 原速。
    sink 路径逐帧不丢；`read()` 仍是允许跳帧的实时路径，跳帧计入 `stats.dropped`。
    `strict=True`（默认）时缺帧或 sink 抛异常都会使回放以 `state="error"` 结束。
  - `load_broker_from_log()` 返回尚未灌入的 broker + pacer；由回放源在投递每帧之前
    按帧时刻推进（`publish_until`），使该帧拍摄时刻必然落在历史区间内、插值命中。
    不应一次性全量灌入——否则"遥测追不上画面"这类行为无法被测试覆盖。
  - `TelemetryPacer.finish(末帧时刻)` 在播放前将覆盖范围延伸至素材末尾：录制器按固定频率
    写入遥测，日志最后一条通常早于末帧数十毫秒；不延伸会使末帧等待 1.0s 上限后被判为
    "链路停顿"而丢弃。该方法先灌完待灌的真实快照再延伸——顺序颠倒会打乱 broker 历史。
- **FrameTelemetryAligner**（`video/align.py`）：将帧的收到时间减去链路固定延时
  （`VideoConfig.telemetry_lag`，默认 0.15s）得到拍摄时刻，并以该时刻向 broker 查询遥测
  （默认 `interpolate`）；如实标记 `extrapolated`/`offset`（查询时刻落在历史范围外即为外推），
  可设置 `max_extrapolation` 作为硬保护（超限返回 None）。
  lag 优先级：对齐器显式 `lag=` > 帧自带 `frame.lag`；`frame.capture_timestamp`
  即"收到时间 − lag"。
- **AlignmentBuffer**（`video/buffer.py`）：将对齐结果（画面 + 拍摄时刻遥测）存入线程安全
  环形缓冲，单写者、任意多读者（各自持有游标并用 `wait_new` 跟随）。默认容量
  `capacity_for(30, 180)=5400` 帧；默认 `storage="jpeg"`（720p 约 0.1~0.2 MB/帧，
  满载约 0.6~1 GiB），`storage="raw"` 无损但 5400 帧约 13.9 GiB。溢出时从最旧一端驱逐；
  `max_bytes` 可设置字节上限。读取接口：`latest/at/iter_between/wait_new`。
- **Hm30VideoSource.add_sink / AlignmentWriter**：sink 在采集线程中逐帧调用，
  这是"读到的每一帧都被留存"的唯一保证点；`AlignmentWriter(buffer, aligner)` 为标准接法
  （对齐后写入缓冲，内部复制图像）。实时读取 `read()`/`latest()` 为另一条路径，允许丢帧——
  **推理与写入磁盘的输入应来自缓冲，不得用 `read()` 循环送入**。
- **Detector**（`perception/detector.py`）：YOLO 封装。`device` 必须显式指定（不依赖自动检测）；
  去畸变使用预计算 remap（比逐帧 `undistort` 快 3~5 倍），映射表锁定在标定分辨率上——
  尺寸不符的帧跳过并告警，不得用错误内参处理画面。`ultralytics` 惰性导入。
- **OpenCvPostProcess**（`perception/cropproc.py`）：裁剪图后处理，移植自旧版实现。
  去噪 → 等比放大到短边 300px → 颜色掩码饱和度逐级回退（blue 100→60→40→20，
  red 100→80→60→40→20）→ 凸包 + `approxPolyDP` 扫 `epsilon=3..40` 提取五边形 →
  转正（平行边法为主、最小内角法兜底）→ 彩图整图 OCR（v6 det 对灰度图检不出文本框）→
  单数字框按行聚类拼两位 → 置信度加权（两位 ×1.15、单位 ×0.6）→ `correct_ocr_number` 纠错。
  **顶角形态判据**（等边三角形内角恒 60°，`HOUSE_APEX_ANGLE_DEG=60` / `TOL=15`）、
  量内角与转正的要求、门限调整方法见 [`docs/perception_ocr.md`](docs/perception_ocr.md)；
  调整转正或形态门限前必须先阅读该文档。
- **OcrEngine**（`cropproc.py`）：RapidOCR 薄封装，det/rec 使用 TORCH 引擎 + 显式本地 `.pth` 路径；
  方向分类（Cls）已接入，默认 `cls_engine="onnx"`，`cls_autorotate` 默认开启
  （按判定将倒置文本行转正后再识别）；置信度过滤使用 `Global.text_score`。
  分类权重文件名必须保持 RapidOCR 原样（TORCH 版为 `ch_ptocr_...`）；
  `onnxruntime-gpu` 使用 CUDA 前需先 `import torch`——详见
  [`docs/perception_ocr.md`](docs/perception_ocr.md)。
- **PerceptionWorker**（`perception/pipeline.py`）：经 `AlignmentBuffer.wait_new` 逐帧取图
  （一帧不丢）→ 检测 → `ocr` 模式裁剪送检 / `cls12` 模式类别直出 → `Detection` 入结果队列。
  OCR 在独立进程池中运行（不阻塞逐帧检测）；默认不启用跨帧去重，仅当显式给出
  `ocr_dedupe_s`/`ocr_dedupe_px` 时按"拍摄时刻 + 中心位移"窗口去重。`_emit(set_code=True)`
  才覆盖编号——`cls12` 的编号来自 YOLO 类别，不得被 OCR 路径清空。
  **三条队列均不丢弃**（请求 / OCR 结果 / 检测结果均无界）：目标可能仅在一瞬清晰，
  丢失任一送检请求都可能丢失目标；积压超过 `ocr_queue_size`（OCR）或 2000 条（结果）时
  仅告警。**不因"队列满"丢弃数据**——积压长期不降时应调整 `ocr_workers` 或降低检测帧率。
- **DroneController**（`telemetry/controller.py`）：任务级控制，将 MAVSDK 的
  `mission_raw`（上传/下载/复位当前项/进度）、`mission`（仅用于启动）、`gripper`、`action`
  插件封装为"上传任务（含回读校验）/ 启动（先复位到第 0 项）/ 查询是否处于任务模式 /
  hold / rtl / 投放 / 查询原点 / 查询任务是否完成"，全部经 `MavsdkThread.submit`
  投递到 MAVSDK 线程执行，指令间以 `RLock` 串行。**失败语义统一**：命令失败抛 `ControllerError`；
  `None`/`False` 仅表示"已读取但确实尚未就绪"（原点未就绪、任务未完成）；而"快照中尚无
  飞行模式/任务进度"视为查询失败（抛 `ControllerError`，调用方按查询失败处理）。
  该模块不在导入期读取 config——`config → video.source → telemetry` 导入链会形成环
  （装配走 `from_config`）；对 `mission.items` 仅在类型标注下导入，运行时需要
  `command_name` 时在函数体内惰性导入。
- **mission**（`mission/`）：
  - `states.py`：`INIT→PREFLIGHT→WAIT_AIRBORNE→RECON→HOLD_PROCESS→OVERFLY→LAND→DONE`，
    任意环节异常 → `ABORT`；`INIT` 仅等待遥测 + NED 原点（不上传任何任务）；
    `PREFLIGHT` 执行一次自检（载入模型 + 视频自检，见 `airdrop/preflight.py`）；
    `WAIT_AIRBORNE` 等待 `in_air`（取不到时以
    `relative_altitude_m >= MissionConfig.airborne_alt_m` 兜底）；`RECON` 默认等待操作手
    在 QGC 上传并启动侦察航线（`MissionConfig.recon_upload="operator"`；`"auto"` 仅用于自动测试）；
    `TRANSITIONS` 是唯一的合法边表，非法转移抛 `InvalidTransition` 且状态不变，
    每次成功转移记录一条 `state` 事件（含 from/to/reason/timestamp）。
  - `items.py`：`MissionItem` 是任务的唯一表示（MAVLink 级：command/frame/params/位置），
    构造器包括 `waypoint()/takeoff()/land()/from_waypoint()`。任务项一律经 `mission_raw` 上传，
    不使用 MAVSDK 的 `vehicle_action` 翻译层（该层会将 `LAND` 一项拆成两项，被 PX4 整条拒绝，
    见"关键约定"第 10 条）。
  - `plan_file.py`（纯 JSON，离线）：`load_plan()` 解析 QGC `.plan`（`SimpleItem` 逐字读取、
    `fwLandingPattern` 复杂项按 QGC `LandingComplexItem::appendMissionItems` 展开），
    `check_fixed_wing_landing()` 按 PX4 可行性判据做本地预检。
  - `planner.py`（纯函数）：侦察段 → 任务项（首项带起飞项）；飞掠段以目标为中心、
    沿配置航向前后各半段长生成 `[entry, exit]`（方向须与判据的"越过目标"一致，符号不得反向）；
    飞掠段 + 降落段拼接为一条任务上传；无目标时使用备用点。
    **每条腿的航线来源二选一**：配置航点，或操作手的 QGC `.plan`（`recon_plan`/`land_plan`；
    降落段使用 plan 时，飞掠段插在其前，且仍须通过降落预检）。
  - `runner.py`：`MissionRunner.run()` 为普通循环（不自行创建线程），`update()` 支持单拍驱动；
    控制器与投放判据均按协议注入，离线测试使用假对象。`clock`/`sleep` 同样注入——测试中
    分钟级任务可在毫秒级确定性完成。上传启动后经 `_confirm_started()` 确认进入 `MISSION`
    （`mission_start_timeout_s` 超时 → `ABORT('mission_not_started')`），确认成功记录
    `mission_confirmed`。
    - **`preflight` 为注入项**（`PreflightLike | None`，见 `airdrop/preflight.py`）：为 `None` 时
      `_tick_preflight` 记录一条 `preflight_skipped`（并输出 warning）后放行——仅供离线测试/演练；
      正式入口（`examples/full_mission.py`）必须注入真实 `Preflight`。任一项失败或抛
      `PreflightError` → `ABORT("preflight_failed:<check>")`；整体超 `PreflightConfig.max_s`
      → `ABORT("preflight_timeout")`。
    - **`PREFLIGHT` / `WAIT_AIRBORNE` 为"瞬时状态"**（`_INSTANT_STATES`）：条件当场满足时
      在同一拍内连续跳转，但每一跳均正常写入历史与 `state` 事件。
    - `_begin_recon` 分两条路径：`recon_upload="auto"` 由本包上传并启动；`"operator"`（默认）
      不上传，记录一条 `recon_waiting_operator`，并将 `_start_confirmed` 置 False 等待操作手
      在 QGC 启动（`recon_max_s` 为该等待的上限）。相关事件：`airborne`（带 `source`）、
      `preflight_skipped`、`recon_waiting_operator`。
- **TargetTracker / PerceptionTargetSource**（`mission/targets.py`）：将 `Detection`
  （像素 + 编号 + 拍摄时刻遥测）经 georef 转为 `TargetPoint`，`result()` 即
  `analyze(points, config.targeting)`——实飞与回放共用同一路径。
  `PerceptionTargetSource(worker, tracker)` 将其包装为 `MissionRunner` 所需的
  `target_result` / `target_busy` 回调（`result()` 会同时抽干结果队列）。
  - **去畸变只执行一次**：检测器已 remap 时（`DetectorConfig.camera_matrix` 非空），
    像素位于纠正后的图中，georef 不得再次纠正。`Detector.detect` 会将
    `extra["undistorted"]` 写入 `Detection`，`TargetTracker(undistort=None)` 默认据此自动判断。
  - **边长交叉验证默认仅作诊断**：超门限记录 `side_check` 事件与计数，不剔除数据点
    （误判一个真实观测的代价高于保留一个野点；DBSCAN 本身依赖多帧观测筛选）。
    需要严格剔除时设置 `reject_on_side_mismatch=True`。
  - 缺姿态/缺位置的点不推测：分别计入 `attitude_missing` / `no_fix` 后跳过。

## 关键约定（务必遵守）

1. **mavsdk 没有公开 `close()`**：会话结束必须显式调用 `_release_drone()`
   （guarded 调用 `_stop_mavsdk_server`，幂等，与 `System.__del__` 路径一致）。
   不得仅将引用置 None 等待 GC——mavsdk_server 子进程占用固定 gRPC 端口 50051，
   旧进程未退出会成为僵尸 server，新会话可能连接其上并在其被回收时级联断连。
   `_run_session` 使用 try/finally 覆盖所有退出路径（含 connect 超时）。
2. **历史按时间查询**：使用 `bisect(key=operator.attrgetter("timestamp"))` 直接探测 deque，
   禁止重建整张时间列表（会导致单次查询慢两个数量级）；外推分支不做任何二分。
3. **代码风格**：注释/docstring 使用中文；类型标注使用现代写法（`X | None`、内置泛型，
   不使用 `typing.Dict`/`Optional`）；快照传递使用浅拷贝（字段均为标量）。
   规范由三条命令保证：`./.venv/Scripts/ruff.exe check .`、
   `./.venv/Scripts/ruff.exe format --check .`、`./.venv/Scripts/pyright.exe`
   均须零告警（详见"验证方式"）；提交前执行。
4. **git 提交**：使用 conventional 前缀 + 中文摘要，如 `fix(broker): ...`、
   `refactor(mavsdk_thread): ...`；一个逻辑改动一个提交。
5. **快照语义**：快照是"最新可用值的合并"，字段间无严格时间同步。帧-遥测的时间轴对齐
   已由 `airdrop.video.align` 解决（按拍摄时刻查询历史快照），但这不代表快照内部各字段
   来自同一瞬间。
6. **视频链路使用 ffmpeg 子进程读取 RTSP 流**：断流由 `-timeout`（微秒）判定，
   子进程可被父进程终止，拉流线程不会因管道读取阻塞。缓冲与丢帧语义、启动阶段帧丢失
   范围（`≈probesize`）见 [`docs/video_rtsp.md`](docs/video_rtsp.md)。
7. **"逐帧留存"与"实时读取"是两条路径，不得混用**（硬性要求：目标可能仅出现一瞬，
   漏一帧即可能漏掉目标；允许处理延时、不允许丢帧）：
   - **逐帧留存**：`Hm30VideoSource.add_sink(callback)` 注册的回调在采集线程中逐帧调用，
     一帧不落。标准接法为 `source.add_sink(AlignmentWriter(buffer, aligner))`，
     每一帧及其拍摄时刻遥测都会进入环形缓冲。sink 必须足够快（`put` 仅执行一次 JPEG 编码），
     其抛出的异常仅记日志、不中断拉流。
   - **实时读取**：`read()` / `latest()` 只保证"当前这一帧"；慢消费者会丢弃中间帧并计入
     `stats.dropped`，仅影响该实时路径，缓冲中的历史帧不受影响。
   - 因此**推理与写入磁盘的输入应来自 `AlignmentBuffer`（`iter_between` / `wait_new`），
     不得用 `read()` 循环送入**。`AlignmentWriter.written` / `skipped` 可用于核对：
     `skipped` 仅因"该时刻取不到遥测"增加，不因消费速度增加。
8. **帧-遥测对齐须扣除链路延时**：按时间查询遥测时使用 `frame.capture_timestamp`
   （= `frame.timestamp - telemetry_lag`），不得直接使用 `frame.timestamp`——后者是收到帧的
   时间，比画面真实拍摄时刻晚一整个链路延时（10 m/s 平飞时约 1.5 m 偏差）。
   `telemetry_lag` 为链路属性，配置在 `VideoConfig` 上并随每一帧下发，默认 0.15s
   （开发机在 ffmpeg 后端实测值）；更换相机/链路或标定出新值后修改此处即可。
   外推结果须依据 `AlignedSample.extrapolated` / `offset` 判断可用性（必要时设置
   `max_extrapolation`）。
9. **不得将帧的 ndarray 作为稳定数据持有**：
   - ffmpeg 后端的 `frame.image` 是复用管道缓冲区的视图，下一帧读入会覆盖同一内存。
     跨帧保留画面必须 copy（`AlignmentBuffer` 内部已处理；自行保存时同样需要
     `.copy()` 或 `frame.copy()`），否则历史帧最终都会变为同一张最新画面。
   - 回看历史统一使用 `AlignmentBuffer.iter_between()`（分批解码、迭代期间不持锁、快照式）。
     模块刻意不提供"一次性返回列表"的接口——5400 帧 720p 全部解码约 13.9 GiB，
     需要列表时应自行 `list(...)`，并承担相应内存开销。
   - `raw` 模式下读取的 image 为只读视图（零拷贝），需要原地绘制时应先 `record.copy()`。
10. **任务项一律经 `mission_raw` 上传，不得使用 MAVSDK 的 `vehicle_action`**：
    `mission` 插件会将 `vehicle_action=LAND` 的一项拆为两项（同坐标 `NAV_WAYPOINT` + `NAV_LAND`），
    拆出的项与落点同高同坐标，会被 PX4 固定翼可行性检查拒绝
    （`mission_feasibility_checker`: "the approach waypoint must be above the landing point" +
    `navigator`: "No valid mission available, loitering"），而 `start_mission()` 仍返回成功。
    raw 通道原样投递，不存在该翻译层；不得改回 `mission` 插件。
11. **固定翼降落两条硬判据**（PX4 `MissionFeasibility/FeasibilityChecker.cpp`）：
    紧前一项必须严格高于落点（`< FLT_EPSILON` 即拒绝）；下滑斜率（tan）
    `(前项高 − 落点高)/水平距离 ≤ tan(FW_LND_ANG+0.1°)`（出厂默认 8° ⇒ 上限 0.142 ≈ 8.1°；
    文档中的 0.219/0.067 等数字均为斜率，不是角度）；进场项只能是 `NAV_WAYPOINT` 或
    `NAV_LOITER_TO_ALT`，且 `LOITER_TO_ALT` 时落点须在盘旋圈外（水平距离按
    `sqrt(圆心距²−半径²)` 修正，与 PX4 一致）；距离计算使用与 PX4
    `get_distance_to_next_waypoint` 相同的球面 haversine（半径 6371000，`_EARTH_RADIUS_M`），
    不得替换为椭球/Geod——预检必须与飞控使用同一把"尺子"。
    本包通过 `check_fixed_wing_landing()` 在规划阶段预检，不合格直接抛 `PlanningError`。
12. **`start_mission()` 返回成功 ≠ 进入任务模式**：飞控可能拒绝模式切换却仍返回 ACK；
    相同 CRC 的任务再次启动时，飞控不会清除"已飞完"锁存，会停在 `HOLD`。
    因此 `DroneController.start_mission()` 必须先复位到第 0 项再启动，状态机再用
    `in_mission_mode()` 正向确认（`MissionConfig.mission_start_timeout_s` 超时 →
    `ABORT('mission_not_started')`）。
13. **`mission.is_mission_finished()` 在 raw 上传下始终返回 `False`**：该插件只识别自身
    上传的任务（`last_upload`）。完成判定改读 `mission_raw.mission_progress()` 的
    `current == total`（MAVSDK 语义：`current` 为 0 基下标，等于 `total` 即完成），
    由可选流写入快照 `mission_current`/`mission_total`；模式确认同理读取快照 `flight_mode`。
14. **运行环境两条**：(a) 同一时刻只能存在一个 `mavsdk_server`（固定占用 gRPC 50051），
    调试第二个脚本时应使用 `System(mavsdk_server_address="localhost", port=50051)`
    连接已有 server，不得再启动一个；(b) `mavsdk_server --version` 会阻塞（进程转入服务模式），
    版本以日志首行 `mavsdk_server: MAVSDK version: vX.Y.Z` 为准（与 Python 包版本一致）。
15. **QGC `.plan` 的两条约定**：(a) `fwLandingPattern` 复杂项在本地按 QGC
    `LandingComplexItem::appendMissionItems` 展开（`DO_LAND_START` → 可选 `DO_CHANGE_SPEED` /
    停止拍照录像 → 进场项 → `NAV_LAND`），PX4 的 `DO_LAND_START` 为
    `specifiesCoordinate=false` ⇒ 不带坐标、`frame=MAV_FRAME_MISSION`；
    其他复杂项（VTOL 降落、测绘/结构）显式报错，不得推测展开。
    (b) **每条腿的航线来源只能有一个**（配置航点或 `.plan`），同时提供时
    `Config.validated()` 直接报错——静默采用其一比报错更危险。
16. **正式任务的启动顺序为"自检 → 等起飞 → 侦察"，且侦察航线默认由人工上传**：
    `INIT` 等待遥测/原点 → `PREFLIGHT` 载入模型（detector/ocr/camera）+ 视频自检
    （`PreflightConfig`，各项可关闭）→ `WAIT_AIRBORNE` 等待 `in_air`
    （`MissionConfig.airborne_timeout_s` 兜底，取不到 `in_air` 时以
    `relative_altitude_m >= airborne_alt_m` 判断）→ `RECON`。
    `MissionConfig.recon_upload="operator"`（默认）表示侦察航线由操作手在 QGC 上传并启动，
    本包不上传、只等待其开始并监视进度；`"auto"` 由本包上传，仅用于自动测试
    （否则会覆盖操作手已绘制的航线）。
    三条要求：**检查关闭 ≠ 检查通过**（关闭的项记 `preflight` 事件、`ok=null`）；
    **开启但未注入载入回调为装配错误**（直接 `ABORT("preflight_failed:<check>")`，
    不得静默通过）；**不得在未起飞时进入侦察**——PX4 会在地面追踪第一个航点，
    或因任务不可行直接盘旋。
    `MissionConfig.require_airborne`（默认 `True`）置 `False` 时 `WAIT_AIRBORNE` 立即放行，
    并记录 `airborne_skipped` 事件（带 `reason`）+ WARNING 日志——仅用于地面演练/离线测试
    （`examples/sitl_mission.py` 的 `REQUIRE_AIRBORNE = False`），正式任务保持 `True`；
    放行不会静默，日志中可明确识别。
17. **入口与依赖的两条硬约定（`airdrop/run.py` + 惰性导出）**：
    (a) **命令行只在 `airdrop/run.py` 一处**（argparse 子命令，注册表 `SUBCOMMANDS`）：
    `examples/*.py` / `tools/*.py` 一律为纯库模块（常量 = 默认值、`build_config(**覆盖)`、
    `main(**kwargs)`），不得自行 import argparse；新增入口在注册表添加一条，
    `tools/check_docs.py` 会核验手册是否覆盖每个子命令。
    (b) **重型依赖（torch / cv2 / mavsdk / ultralytics / rapidocr / onnxruntime）只能在函数体内导入**：
    `airdrop/__init__.py` 与 7 个子包（video / telemetry / mission / georef / record /
    perception / ballistics）均为 PEP 562 惰性导出（`airdrop/_lazy.py`，精确的
    `_EXPORTS` 名字→叶子模块映射），因此 `import airdrop`、各子命令的 `--help` 与
    `python -m airdrop.run check-docs` 都不会加载这些库（`tests/test_cli.py` 在干净子进程中
    验证）。`__all__` 的内容与顺序不得改动——`tools/dump_api.py` / `check_docs.py` 依赖它。
    `check-docs` 的"引用可解析"步骤需要真实 import（如 `airdrop.video.buffer`），
    因此运行在**子进程**中，不得改回本进程 import。
18. **行尾统一 LF（`.gitattributes`）**：仓库使用 `* text=auto eol=lf`，索引与工作区均按 LF
    检出与比较；新增或改写文件不得引入 CRLF。自查：
    `git ls-files --eol | awk '$2=="w/crlf"'` 应为空。

## 验证方式

- **测试（pytest，全部离线，无需飞控/视频硬件）**：
  `./.venv/Scripts/python.exe -m pytest`（附覆盖率）
  `./.venv/Scripts/python.exe -m pytest -m "not stream"`（跳过需要启动 ffmpeg 的用例）
  `./.venv/Scripts/python.exe -m pytest -k alignment -v`（按主题运行）
  配置位于 `pyproject.toml` 的 `[tool.pytest.ini_options]`：`--strict-markers`、180s 全局
  超时、默认 `-m "not realdata"`；标记包括 `stream`（启动 ffmpeg / 占用 UDP 51234）、
  `network`（等待网络错误路径）、`realdata`（GPU + 真实素材，分钟级，需显式
  `-m realdata`）；公共 fixture 在 `tests/conftest.py`（本地 H.264 测试流 `sender`、
  `live_source` 工厂、`broker`/`received`）。测试依赖在 `dev` 组。
  修复缺陷时应在对应模块添加回归用例——`tests/` 即为回归测试落点。
  - `test_perception.py` 全部离线：重型依赖（YOLO/RapidOCR/torch）均不加载，
    pipeline 的 `detector`/`pool` 为构造注入的假对象——这正是这两个参数存在的理由。
  - `test_perception_realdata.py`（标 `realdata`）访问 GPU 与真实素材，验收标准为
    "真实实战视频的目标段能读出正确编号（56/56/56）"；视频路径由环境变量
    `AIRDROP_REALDATA_VIDEO` 指定，未设置或文件不存在时跳过。
  - `test_telemetry.py` 覆盖遥测订阅推送、插值/外推与遥测速率下发。
  - `test_alignment.py` 直接向 `broker._history` 注入确定性时间戳（不依赖 sleep）。
  - `test_buffer.py` 用"同一 bytearray 反复 reshape 成 view"复现 ffmpeg 后端的复用缓冲行为，
    验证"入缓冲必须拷贝"；另有 `AlignmentWriter` 的逐帧入库用例。
  - `test_recorder.py` / `test_replay.py` 的素材为手写数据（不依赖真实录制）：按写入磁盘的
    格式直接写 `frames_index.jsonl` + jpeg + `telemetry.jsonl`，存放于工作区下的临时目录。
    `test_replay.py` 的验收标准为"回放对齐结果与直接从日志查询的参考 broker 完全一致"——
    不以"与实飞 broker 一致"为标准（recorder 按 10Hz 节流写盘，更快的原始流已被有损降采样）。
    `test_calibrate.py` 的 `calibrate()` 输出契约用例同样自建临时飞行目录
    （工作区内的 `.calibrate-test-tmp/`，`workdir` fixture 用完即删）。
  - `test_mission.py`：假控制器 + 假时钟。`FakeController` 按 `MissionController` 协议实现
    （`fail_on` 注入命令失败、`sticky_finished` 模拟"新任务仍报上一次已完成"的陈旧读数、
    `mission_progress` 提供快照中的任务进度），`FakeBroker` 提供可控时间戳的快照，
    `TimedController` 按假时钟自动推进任务进度——分钟级任务可在毫秒级确定性完成
    （`run()` 用例断言 180s 假时间走完 DONE）。`DroneController` 一组使用"假线程 + 假 drone"
    离线验证 MAVSDK 调用装配（`mission_raw` 上传的 raw 项内容、`current` 仅第 0 项置 1、
    `autocontinue` 传 int、上传后回读校验、gripper 实例号、异常翻译）。
    注意：**状态在某一拍结束时进入**，因此"进入新状态后运行一拍"的断言需要再调用一次
    `update()`（`_drive` 与 `_drive_until` 的分工即为此）。另有一组覆盖起飞前自检与等起飞
    （`PREFLIGHT` 载入模型/视频自检、`preflight_failed:<check>`/`preflight_timeout`/
    `preflight_skipped`、`WAIT_AIRBORNE` 的 `in_air` 与高度兜底、`airborne_timeout`、
    `recon_upload` 的 operator/auto 两条路径）。
  - `test_plan.py`：QGC `.plan` 的纯 JSON 解析（`SimpleItem` / `fwLandingPattern` 复杂项展开）、
    `check_fixed_wing_landing` 的各条判据（同高必拒、下滑角过陡必拒、盘旋圈、进场项类型、
    混合高度基准）、plan 与配置航点二选一。离线，无需飞控。
  - `test_e2e.py`：回放驱动的全链路。手写合成飞行目录（帧索引 + jpeg + 遥测：
    位置/四元数/原点）→ `ReplayVideoSource` → 对齐 → 缓冲 → `PerceptionWorker`
    （注入脚本化假检测器）→ `TargetTracker`（真实 georef）→ `targeting` → `MissionRunner`
    → 飞掠航线。坐标可写死的原因：飞机水平朝下、目标像素取主点——主点视线即光轴，
    水平姿态下正对地面，目标必然位于飞机正下方 `(north, east, 0)`。
    另有 `-m realdata` 变体：真视频 + 真 YOLO + 真 OCR，位姿为合成数据（该素材无遥测），
    验证"真实像素可一路走到坐标"而不验证精度。
  - `test_examples.py`：示例导入即校验（API 未脱节、不自行 import argparse、均具备
    `main()` / `build_config()`、导入期不加载重型依赖），并实际运行若干示例的装配函数
    （`build_config()` 默认值与覆盖、SITL 的合成目标）。
  - `test_cli.py`（集中式入口）：子命令注册表（每个子命令具备 parser + handler + 默认值出处）、
    `--help`（顶层与全部子命令）与 `check-docs` 在干净子进程中运行且不加载
    torch/cv2/mavsdk/ultralytics/rapidocr/onnxruntime、`import airdrop` 同样轻量、
    惰性导出后跨子包名称仍可获取、未知子命令/非法参数退出码为 2、
    以及"选项 → 关键字 → `build_config()`"覆盖链（full-mission / sitl / replay /
    calibrate / fit-ballistics 五条）。
  - `test_video.py` 会启动 ffmpeg 子进程并占用本地 UDP 端口 51234。其中"地址无效应显式报错"
    一组连接 `rtsp://127.0.0.1:1/...`，日志中出现 error 级"视频不可用"属预期输出；
    "逐帧经过 sink"一组故意慢速消费，`stats.dropped > 0` 属预期（sink 侧仍不丢帧）。
    断言应在 `source.stop()` 之后读取 `stats`，否则会与在途帧存在偏差。
- **语法/编译检查**：
  `./.venv/Scripts/python.exe -m compileall -q airdrop tests tools examples`。
  注意不要使用 `py_compile airdrop/*.py`：PowerShell 不会为原生命令展开通配符，
  py_compile 会将 `airdrop/*.py` 当作字面文件名并报 `[Errno 22] Invalid argument`。
- **代码规范（ruff + pyright，均在 dev 依赖组，配置在 `pyproject.toml`）**：
  以下三条命令必须零告警，修改代码后先运行它们，再运行 pytest：
  `./.venv/Scripts/ruff.exe check .`、
  `./.venv/Scripts/ruff.exe format --check .`、
  `./.venv/Scripts/pyright.exe`。
  - ruff 采用务实规则集（E4/E7/E9/F/W/I/UP/B/SIM/C4/RUF/BLE/SLF/PL/PT），
    未使用 `select = ["ALL"]`——本项目注释/文档为中文，RUF001/002/003（易混 Unicode 字符）
    会对全角标点产生大量误报。这三条规则保留启用，仅将 12 个刻意使用的排版字符加入
    `allowed-confusables`（全角括号/逗号/冒号/分号/问号、`×`、`−`、`–`、`α`、`ρ`、`σ`）。
  - 关闭的规则均附有理由（`UP037` 会误删中文 docstring 中的强调标记、`RUF046` 的
    `int(round(...))` 对 numpy 标量是必要收窄、`PLR2004`/`PLC0415` 与"物理字面量 +
    重型依赖函数体内导入"的约定冲突）。`E501` 亦为刻意关闭：`ruff format` 会自动压缩
    可拆分的长行，而格式化器对不可拆分的长行（长 URL、单条长字符串、行尾 `noqa`）本就不限长。
    `line-length = 100` 用于指导格式化器，并非要求每行不超过 100 列。
  - 全库已按 `[tool.ruff.format]` 格式化；`format --check` 应保持全绿。格式化只允许改动排版：
    验收标准为格式化前后 AST（去除位置信息）完全一致，不应仅依赖测试通过。
  - `tests/**` 与 `tools|examples/**` 配置了 per-file ignores（私有成员访问、复合断言、
    打表输出、参数数量等），见 `[tool.ruff.lint.per-file-ignores]`。
  - pyright 使用 `basic` 模式；`reportUnsupportedDunderAll` 已关闭——惰性导出（PEP 562）的
    `__all__` 无法静态解析，一致性由 `tests/test_cli.py` 实际运行验证。
  - 排查原则：ruff/pyright 报告的问题应先作为真实缺陷排查，不得直接添加 `noqa`/`ignore`；
    确需抑制时须附理由：`# noqa: RULE - 原因`（库内绝大多数为 `BLE001`，
    即有意捕获所有异常）。
- **入口与帮助信息（`airdrop/run.py`）**：
  `./.venv/Scripts/python.exe -m airdrop.run --help` 与每个子命令的 `--help` 均须
  **exit 0**，且运行后 `sys.modules` 中不得包含 torch / cv2 / mavsdk / ultralytics /
  rapidocr / onnxruntime；`python -m airdrop.run check-docs` 同理。
  `tests/test_cli.py` 在干净子进程中逐条验证（含"未知子命令/非法参数 = 退出码 2"）。

## 模块约定补充

### 目标统计（targeting）

`TargetPoint`（单次观测）→ DBSCAN 聚类 → 类标签取类内 `code` 众数 → 跨类按标签
`median`（下中位）/`max` 选取唯一结果，坐标取类内均值。约束：

1. eps 边界为闭区间（恰好相距 eps 的两点相连）；启用 `sample_weight` 时权重为绝对权重，
   直接参与核心点判定 `Σw ≥ min_samples`，因此置信度加权必须先按均值归一到 1；
2. 输入先按（拍摄时刻，帧号）排序——DBSCAN 标签号依赖到达顺序，排序保证结果可复现；
3. 坐标非有限（inf/nan）的点单独剔除并计数；
4. 仅带编号的类参与 median/max；`require_label=False` 仅使无编号类保留在结果列表中；
5. 同一标签存在多个类时，按"成员数 → 置信度 → 坐标"排序以保证结果唯一。

离线用例：`tests/test_targeting.py`。

### 弹道与投放（ballistics）

`BallisticsModel`（二次阻力 + RK4 定步长 5ms → 落点/飞行时间；密度按真实海拔；
风作为空气速度参与阻力）；`ReleaseJudge`（每拍预测落点，误差 ≤ 半径即投放；
越过目标后强制投放；一次性锁存）。约束：

- 质量、`Cd`、迎风面积默认值为占位，须按实测/试验结果回填（`BallisticsConfig`）；
- 密度基准为地面平面海拔（`predict_impact(ground_altitude_m=...)`；记录缺原点时按 0 处理
  并在反演报告中告警）；`air_density_isa` 关闭时在投放海拔取一次常密度，开启时逐级求 ISA
  （耗时增加 26~30%）；
- 落点取穿越瞬间的线性插值（直接取第一步会引入最多一步的水平误差）；
- 失败必须显式：起点在地面以下 → `below_ground`；积分超过 120s 未落地 → `timeout`；
  两者均返回 `ok=False`、`ned=None`，不得编造落点；
- `wind_source="telemetry"` 取不到风时返回 None（与"风为 0"区分）；判据按零风降级并
  仅记录一次日志；
- 摘要按 5Hz 写入事件日志（`SUMMARY_HZ`）；触发瞬间写入完整预测；`on_event` 签名
  为 `(kind, data)`；事件写盘失败不影响判据；
- `delay_s` 的状态前推为一阶近似（位置 += 速度 × delay），不含加速度二次项；
- 投放记录与反演约束见 [`docs/ballistics_fit.md`](docs/ballistics_fit.md)。

离线用例：`tests/test_ballistics.py`、`tests/test_fit.py`。

### 任务编排（mission）

- `MissionItem.param1~param4` 以 `NaN`（`UNSET`）表示"不指定"，0 为有意义的值；
- 降落项为单项 `NAV_LAND`；任务项一律经 `mission_raw` 上传；上传前执行降落几何预检；
- 任务进度为飞控侧状态；`MissionMonitor` 要求先观测到一次未完成再认可完成（防止陈旧读数）；
- 命令类失败 → `ABORT`（原因写入事件日志）；查询类失败 → 记录一次日志与事件并继续，
  由状态超时兜底；链路看门狗阈值 `telemetry_stale_s`（默认 5s）；
- `ABORT` 仅下发一条安全动作（`MissionConfig.abort_action`：`hold` 默认 / `rtl` / `none`）；
- 相对高度口径：`Waypoint.alt_m` 与 `MissionItem.alt_m` 为相对起飞点的高度；
  备用点换算为 NED 时海拔应按"原点海拔 + 相对高度"提供；
- 状态在某一拍结束时进入：进入新状态后需再运行一拍才会执行该状态的判据与查询，
  测试断言须考虑该语义；
- 正式任务启动顺序为"自检 → 等起飞 → 侦察"，侦察航线默认由操作手在 QGC 上传并启动。

离线用例：`tests/test_mission.py`。

### 外部素材与权重

- `realdata` 用例需要外部视频，由环境变量 `AIRDROP_REALDATA_VIDEO` 指定；仓库不携带素材；
- `fetch-models` 需显式指定权重来源（`--source-dir` / `--source-ocr-dir`，或环境变量
  `AIRDROP_SOURCE_DIR` / `AIRDROP_SOURCE_OCR_DIR`）；仓库不携带权重；未指定时输出用法
  并以退出码 1 结束。
