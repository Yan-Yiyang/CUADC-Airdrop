# AGENTS.md — airdrop 项目智能体指南

> 给在本仓库工作的 AI 智能体：先读完这份再动手。事实均已核实，遇冲突以代码为准。

## 项目概述

CUADC固定翼无人机侦查与打击控制项目（固定翼无人机"先侦查后空投"控制系统），Python 包名 `airdrop`，**当前版本 0.1**。

> ⚠ **验证状态：只做过仿真验证**（SITL + 离线测试），**没有真机飞行数据**。真机测试需
> 谨慎：先拆桨/系留、逐步放开，并随时能切回手动接管（见 `README.md` 的"安全提示"）。
已实现：遥测（MAVSDK 线程 + 代理 + 任务级控制器）、图传（ffmpeg 拉流/对齐/缓冲）、
感知（YOLO + OCR + 转正）、坐标（georef）、目标统计（targeting）、弹道与投放（ballistics）、
任务编排（mission：状态机 + 航线 + 坐标解算 + 主循环）、QGC `.plan` 航线导入（含固定翼降落航线复杂项展开 + 降落几何预检）、起飞前自检（preflight：载入模型 detector/ocr/camera → 视频自检；每项可单独关）、记录与回放、三步标定流水线、
投放记录与弹道参数反演（投放瞬间状态 → 实测落点 → 最小二乘 + 可辨识性诊断），
外加端到端离线链路（`tests/test_e2e.py`）。
剩余工作都是**外部依赖**的验证，不是代码：真实素材标定复测、真实投放试验（工具已就绪、
需要你的数据）、SITL 演练（已在 WSL + PX4 SITL 固定翼上真跑通）。边界与风险见
`README.md` 的"已知边界与风险"。
主要依赖：mavsdk、torch、ultralytics、opencv、rapidocr、scipy、pyproj、scikit-learn、onnxruntime-gpu。

## 文档地图

| 文档 | 讲什么 |
| --- | --- |
| `AGENTS.md`（本文件） | 项目工作手册：环境 / 目录结构 / 架构要点 / 关键约定与易错点 / 验证方式 |
| [`docs/handbook.md`](docs/handbook.md) | **项目手册（给用户与代码审查者）**：目录总览、数据流与依赖、各模块调用方法与最小示例、**参数总表（config 全字段）**、输入输出与字段含义、使用流程、审查速查（43 条不变量）、排障表、已知边界 |
| [`docs/calibration_opencv.md`](docs/calibration_opencv.md) | 三步标定流水线（内参 → 时间差 → 手眼外参）+ OpenCV 5.0 问题与易错点全记录：三类分因、`calibrateHandEye` 缺失与 5.1 时间线、手眼方程与 PnP 三个易错点、杆臂 `t_bc` 判据、"不要再试"清单 |
| [`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md) | 图传链路：为什么只有 ffmpeg 一个后端（含 cv2 逐项排除表）、16 位条码丢帧实验、启动盲区 `≈probesize`、超时参数、HM30 网段与相机地址 |
| [`docs/perception_ocr.md`](docs/perception_ocr.md) | 检测/OCR 流水线细节：五边形转正与顶角形态门限、OcrEngine/Cls 与权重命名、onnxruntime-gpu 的 CUDA 前提与速度表、转正 180° 复盘、交叉验证"两向对比" |
| [`docs/simulation_world.md`](docs/simulation_world.md) | CUADC 赛区 Gazebo 世界：规则条款 → 几何映射（起降区/目标区/天井，2026 规则第 19~25 页）、天井在区内随机摆放与随机朝向、五边形环壁与靶标资产、**起 SITL 的两条命令**、`make-world` 重生成与换场地、已知边界 |
| [`docs/ballistics_fit.md`](docs/ballistics_fit.md) | 投放记录 + 弹道参数反演：数据结构、可辨识性判据（质量称重 / 识别量 κ）、常数偏差、σ 与留一验证、工具用法 |

**专题笔记与 `AGENTS.md` 同等权威**；两者冲突时，以**代码与专题笔记里的实测记录**为准
（笔记是本文件逐字搬走的原始结论，本文件只留摘要与指针）。
**改代码后跑一遍 `python -m airdrop.run check-docs`**（等价写法 `python -m tools.check_docs`）：它核验 `docs/handbook.md` 与代码是否一致
（参数齐全 / `airdrop.*` 引用可解析 / 产物文件名齐全 / 无编造参数名）；
`docs/handbook.md` §3 的示例由 `tests/test_handbook.py` 逐条真跑（11 条）。

## 环境

- **Python ≥ 3.14**；项目虚拟环境 `.venv`（uv 创建，3.14.6，**里面没有 pip**）。
- 运行 Python：`./.venv/Scripts/python.exe`（Windows）。
- 装依赖：`uv add` / `uv add --dev`；一次性工具：`uv tool run`。
  当前环境不满足，或添加依赖可以更好完成任务时要**主动向用户提出**，但**安装任何工具/依赖前必须先征得用户同意**（用户明确反对擅自安装）。
- 依赖索引走清华镜像（见 `pyproject.toml` / `uv.lock`）。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `airdrop/config.py` | 全部参数集中处（frozen dataclass + 取值域校验）。**命令行解析不在 config 里**：集中在唯一的运行模块 `airdrop/run.py`（argparse 子命令，**已落地**）：`python -m airdrop.run <子命令> --help`（子命令与选项表见 `SUBCOMMANDS`） |
| `airdrop/telemetry/` | `models.py`（数据模型）、`broker.py`（遥测代理）、`mavsdk_thread.py`（MAVSDK 工作线程）、`controller.py`（任务级控制：`mission_raw` 上传 + 回读校验 / 启动（先复位）/ 任务模式确认 / hold/rtl、gripper、NED 原点） |
| `airdrop/video/` | `source.py`（HM30 ffmpeg 拉流）、`align.py`（帧-遥测时间对齐）、`buffer.py`（对齐结果环形缓冲） |
| `airdrop/record/` | `recorder.py`（FlightRecorder：飞行目录"五个记录文件 + 投放记录 `drops.jsonl`"写入磁盘）、`replay.py`（ReplayVideoSource + 遥测回填，见下） |
| `airdrop/perception/` | `detector.py`（YOLO 封装）、`cropproc.py`（五边形→转正→RapidOCR）、`number.py`（编号纠错）、`ocr_worker.py`（OCR 独立进程）、`pipeline.py`（主循环）、`models.py`（Detection/PixelBox） |
| `airdrop/targeting/` | `models.py`（TargetPoint / Cluster / TargetingResult）、`cluster.py`（DBSCAN 聚类 + 编号众数 + median/max 选唯一结果） |
| `airdrop/ballistics/` | `model.py`（二次阻力 + RK4 定步长 → 落点/飞行时间，含 ISA 密度与风）、`release.py`（ReleaseJudge：预测触发 + 强制投放 + 一次性锁存）、`drops.py`（DropRecord 投放瞬间状态 + 实测落点读写/配对）、`fit.py`（`fit_ballistics`：最小二乘反演 + 可辨识性诊断） |
| `airdrop/mission/` | `items.py`（`MissionItem`：**MAVLink 级任务项**，command/frame/params/位置 + `MAV_CMD_*` 常量）、`plan_file.py`（QGC `.plan` 解析：`SimpleItem` + `fwLandingPattern` 复杂项展开 + 固定翼降落预检）、`states.py`（状态 + 合法转移表 + 历史）、`planner.py`（侦查段 / 飞掠 entry-exit / 与降落段拼接，纯函数；每条腿可来自配置航点或 `.plan`）、`targets.py`（TargetTracker：检测结果 → georef → 目标点 → 统计结果）、`runner.py`（MissionRunner 主循环：驱动控制器、监视遥测、推进状态机） |
| `routes/` | 操作手在 QGC 里画好的 `.plan` 航线（**入库**）：`land.plan` = 飞掠段之后的那一段（返航 + 降落，含固定翼降落航线复杂项）；飞掠段由项目插在它前面 |
| `airdrop/preflight.py` | 起飞前自检（正式流程的第一步）：载入 detector/ocr/camera → 视频自检（窗口内帧数 + 分辨率与标定一致）；`Preflight`/`PreflightCheck`/`PreflightError`/`PreflightLike`。**"检查关掉"≠"检查通过"**：关掉的项记 `preflight` 事件（`ok=null`）；**开着却没注入载入回调 = 装配错误，直接失败** |
| `models/` | 本地模型权重（**不进 git**，用 `python -m airdrop.run fetch-models` 取）；`models/ppocr/*.txt` 字典入库以固定字符集 |
| `sim/` | **仿真模块**（世界 + 机型 + airframe + 启动脚本，仓库是唯一出处）：`sim/worlds/cuadc/`（生成产物，别手改：两个 `.sdf` + 网格/贴图）、`sim/vehicles/rc_cessna_down_cam/`（带下视相机的 Gazebo 机型：merge 基础 `rc_cessna` + 720p 相机）、`sim/airframes/4007_gz_rc_cessna_down_cam`（PX4 airframe）、`sim/patches/`（可选 PX4 补丁）、`sim/install_px4.sh`（软链进 PX4，幂等）、`sim/run_sitl.sh`（一键起 SITL）；世界生成器 `tools/make_world.py`，布局/启动/装机说明见 [`docs/simulation_world.md`](docs/simulation_world.md) 与 [`sim/README.md`](sim/README.md) |
| `tools/` | `fetch_models.py`（把外部权重复制进 `models/`）、`calibrate.py`（三步相机标定：内参 → 画面/遥测时间差 → 手眼外参，见 [`docs/calibration_opencv.md`](docs/calibration_opencv.md)）、`fit_ballistics.py`（投放试验反演弹道参数，见 [`docs/ballistics_fit.md`](docs/ballistics_fit.md)）、`make_world.py`（生成 CUADC 赛区 Gazebo 世界，见 [`docs/simulation_world.md`](docs/simulation_world.md)）、`sitl_recon.py`（SITL 目标侦查精度自动测试：真飞 + 真感知，量"解算坐标 vs 天井中心"误差，见 [`sim/README.md`](sim/README.md)）、`make_backdrops.py`（生成目标区地面纹理与程序化航拍底图）、`fetch_aerial.py`（抓真实航拍底图：USGS NAIP，公有领域，需联网）、`preview_world.py`（把世界渲染成带标注的俯视预览图）、`dump_api.py`（自省导出接口清单到 `docs/api_reference.md`）、`check_docs.py`（核验 `docs/handbook.md` 与代码一致）。全是**纯库模块**：文件内的常量就是**默认值**，暴露 `build_config(**覆盖)` / `main(**kwargs)`，**自己不 import argparse**；命令行集中在 `airdrop/run.py`（**已落地**），例如 `python -m airdrop.run calibrate --flight flights/<架次> --strict` |
| `examples/` | `full_mission.py`（完整任务：遥测/图传/感知/判据/状态机/记录）、`replay_flight.py`（回放全链路离线迭代）、`calibration_capture.py`（标定素材采集）、`sitl_mission.py`（SITL 演练：合成目标 + DryRunController）、`basic_usage.py` / `hm30_video.py` / `video_telemetry_sync.py`（基础演示）。全是**纯库模块**：文件内常量 = 默认值，暴露 `build_config(**覆盖)` / `main(**kwargs)`，重依赖只在函数体内导入（`import examples.x` 不加载 cv2/mavsdk/torch）；命令行集中在 `airdrop/run.py`（**已落地**），例如 `python -m airdrop.run full-mission --land-plan routes/land.plan --no-preflight` |
| `tests/` | pytest 套件（全部离线）：`conftest.py`（公共 fixture：本地 H.264 测试流、`live_source` 工厂、`broker`）+ `test_config.py` / `test_telemetry.py` / `test_alignment.py` / `test_buffer.py` / `test_video.py` / `test_recorder.py` / `test_replay.py` / `test_perception.py` / `test_perception_realdata.py`（标 `realdata`，默认跳过）/ `test_georef.py` / `test_calibrate.py` / `test_targeting.py` / `test_ballistics.py` / `test_mission.py` / `test_e2e.py`（回放驱动全链路）/ `test_fit.py`（投放记录 + 弹道参数反演）/ `test_examples.py`（示例不脱节 + 导入不加载重库）/ `test_cli.py`（`airdrop/run.py`：子命令注册表、`--help` 不加载重库、参数校验、选项覆盖进 `build_config`）/ `test_handbook.py`（`docs/handbook.md` §3 的示例逐条真跑）/ `test_plan.py`（QGC `.plan` 解析 + 复杂项展开 + 固定翼降落预检）/ `test_world.py`（CUADC 赛区世界：生成结果 == 入库文件、目标区在起飞线两端、天井区内随机摆放且间距 > 20m、五边形环壁、同区同色、贴地标线不共面重叠、目标区底色与周边航拍底图（都必须在比赛区域外）、底图生成器可复现、两轮靶标摆位、资产与贴图朝向） |
| `PX4-Autopilot-1.17.0/`、`px4.tar.gz` | PX4 固件源码，**仅供参考，禁止当项目代码修改** |
| `qgroundcontrol-master/`、`qgc.tar.gz` | QGC 地面站源码，同上 |

## 架构要点

- **MavsdkThread**：专用后台线程跑独立 asyncio 循环；supervisor 循环负责连接/重连；
  **6 条核心遥测流**（position / home / position_velocity_ned / attitude_euler /
  attitude_quaternion / gps_global_origin）用 `asyncio.TaskGroup` 并行采集，
  **任一核心流结束即视为断连**，交给 supervisor 按配置重连；指令经
  `run_coroutine_threadsafe` 串行执行。默认连接 `udpin://0.0.0.0:14540`。
  - ⚠ **风估计是"可选流"（第 7 条）**：`drone.telemetry.wind()` 走 `_optional_stream`——
    它结束或报错**只记日志、不影响会话**。理由：风只影响弹道精度（没有它按零风降级并
    记一次日志），而飞控未必支持/未必开启风估计；让一条增强流把整条遥测链路拖进重连
    循环不划算。**别把它改成 `_guarded`**。
  - **另外两条可选流**（同样走 `_optional_stream`，2026-09 加）：`mission_raw.mission_progress()`
    → 快照 `mission_current`/`mission_total`（完成判定的唯一来源），`telemetry.flight_mode()`
    → 快照 `flight_mode`（启动确认用）。它们不可用时各自"相关功能降级"：完成判定退回状态超时、
    启动确认跳过（只告警），**不会**把会话判断。
  - 快照里的风字段是 `wind_north/east/down_m_s`（MAVSDK 那边叫 `wind_x/y/z_ned_m_s`，
    在 `update_wind` 里对好名）；它们与其它数值字段一样参与 `get_snapshot_at` 的线性内插
    （插值是按 dataclass 字段通用做的）。
- **TelemetryBroker**：`RLock`+`Condition` 线程安全；最新快照浅拷贝（字段全标量，注释有说明）；
  历史 `deque(maxlen=1200)` 按 `history_interval=0.1s` 节流入库（≈2 分钟），最新快照与订阅推送不节流；
  `get_snapshot_at(ts, mode)` 支持 `interpolate`（范围内内插、范围外外推）与 `nearest`。
- **插值规则**：数值字段线性；欧拉角走最短弧（解决 ±180° 环绕）；四元数 slerp（含 q/-q 符号翻转、保持单位模长）。
- **FlightRecorder**（`record/recorder.py`）：把一次飞行录成 `flights/<时间戳>/` 下的"五个记录文件"——
  `flight.log`（挂在 root logger 上，业务模块无感）、`telemetry.jsonl`（默认 10Hz 节流）、
  `detections.jsonl`、`events.jsonl`、`frames/%06d.jpg` + `frames_index.jsonl`，外加
  `config_snapshot.json`；**外加第 7 个文件 `drops.jsonl`**（每次实际投放一条
  `DropRecord`，接法是 `MissionRunner(on_drop=recorder.drops.append)`——用 lambda 延迟取
  `recorder.drops`，写入器要 `start()` 之后才存在）。帧**零重编码**：直接把缓冲里的
  jpeg 字节写入磁盘（`read_jpeg_bytes`）。
  注意它**只写有效快照**（`TelemetrySnapshot.is_valid()`）——初始空快照入库会成为历史
  区间最左端的"全 None"端点，让最先几帧内插到 None。
- **ReplayVideoSource / TelemetryPacer**（`record/replay.py`）：把飞行目录**按原时间轴**重放，
  离线迭代识别与坐标解算时与实飞走同一条代码路径。
  - `ReplayVideoSource` 与 `Hm30VideoSource` **同一套公开接口**（`add_sink`/`read`/`latest`/
    `stats`/`iter_frames`/`stop`），换进去即可。帧写 `timestamp = capture_timestamp + lag`，
    于是 `frame.capture_timestamp` 与录制时逐位相同；`speed=0` 全速（批量回归）、`1.0` 原速。
    sink 路径一帧不落；`read()` 仍是允许跳帧的实时路径，跳帧计入 `stats.dropped`。
    `strict=True`（默认）下缺帧或 sink 抛异常都让回放以 `state="error"` 结束。
  - `load_broker_from_log()` 返回**尚未灌入**的 broker + pacer；由回放源在投递每帧之前
    按帧时刻推进（`publish_until`），使该帧拍摄时刻必然落在历史区间内、内插命中。
    一次性全灌也能查，但"遥测追不上"这类行为就永远测不到了，所以不应那样使用。
  - `TelemetryPacer.finish(末帧时刻)` 在播放前把覆盖补到素材末尾：录制器按固定频率写遥测，
    日志最后一条通常早于末帧几十毫秒，不补的话末帧会空等 1.0s 上限再被当"链路停顿"丢弃
    （回放里的假警报）。它**先灌完待灌的真实快照再延伸**——顺序反了会把 broker 历史写乱序。
- **FrameTelemetryAligner**（`video/align.py`）：把帧的"收到时间"减去链路固定延时
  （`VideoConfig.telemetry_lag`，默认 0.15s）得到拍摄时刻，用该时刻向 broker 查询
  遥测（默认 `interpolate`）；如实标记 `extrapolated`/`offset`（查询时刻落在历史
  范围外即为外推），可设 `max_extrapolation` 做硬保护（超限返回 None）。
  lag 优先级：对齐器显式 `lag=` > 帧自带 `frame.lag`；`frame.capture_timestamp`
  就是"收到时间 - lag"。
- **AlignmentBuffer**（`video/buffer.py`）：把对齐结果（画面 + 拍摄时刻遥测）放进线程安全
  环形缓冲，写者一个、读者任意多个（各自持游标用 `wait_new` 跟随）。容量默认
  `capacity_for(30, 180)=5400` 帧；**默认 `storage="jpeg"`**（720p 约 0.1~0.2 MB/帧，
  满载 ≈0.6~1 GiB），`storage="raw"` 无损但 5400 帧 ≈13.9 GiB。满了从**最旧**一端
  驱逐；`max_bytes` 可加字节上限。读接口：`latest/at/iter_between/wait_new`。
- **Hm30VideoSource.add_sink / AlignmentWriter**：sink 在采集线程里**逐帧**被调用，
  这是"读到的每一帧都留存"的唯一保证点；`AlignmentWriter(buffer, aligner)` 是标准接法
  （对齐后写进缓冲，内部会拷贝图像）。实时读取 `read()`/`latest()` 是另一条路，允许丢
  中间帧——**推理/写入磁盘的输入要来自缓冲，不要用 `read()` 循环送入**。
- **Detector**（`perception/detector.py`）：YOLO 封装。`device` **显式指定**（自动检测
  曾误选 CPU）；去畸变用**预计算 remap**（比逐帧 `undistort` 快 3~5 倍），且映射表锁定在
  标定分辨率上——换尺寸的帧直接跳过并告警，绝不用错误内参扭坏画面。`ultralytics` 惰性导入。
- **OpenCvPostProcess**（`perception/cropproc.py`）：裁剪图后处理，移植自项目早期流水线。
  去噪 → 等比放大到短边 300px → 颜色掩码饱和度逐级回退（blue 100→60→40→20，
  red 100→80→60→40→20）→ 凸包 + `approxPolyDP` 扫 `epsilon=3..40` 出五边形 →
  转正（平行边法为主、最小内角法兜底）→ **彩图**整图 OCR（v6 det 对灰度图检不出文本框）
  → 单数字框按行聚类拼两位 → 置信度加权（两位 ×1.15、单位 ×0.6）→ `correct_ocr_number` 纠错。
  ⚠ **顶角形态判据（等边三角形内角恒 60°，`HOUSE_APEX_ANGLE_DEG=60` / `TOL=15`）、"两个会量错顶角的易错点"、
  几何转正曾差 180° 的完整复盘与 A/B、以及转正/形态门限怎么调，都在
  [`docs/perception_ocr.md`](docs/perception_ocr.md)**——动转正或形态门限前先读它，不要凭主观判断。
- **OcrEngine**（`cropproc.py`）：RapidOCR 薄封装，**det/rec 走 TORCH 引擎 + 显式本地 `.pth` 路径**；
  方向分类（Cls）已接，默认 `cls_engine="onnx"`（实测最快，四配置速度/精度表见笔记），
  `cls_autorotate` 默认开（按判定把倒置文本行转正再识别）；置信度过滤走 `Global.text_score`。
  ⚠ **权重命名（必须叫 RapidOCR 那个错拼的 `ch_ptocr_...`）、`default_models.yaml` 里没有 `torch:` 段、
  `onnxruntime-gpu` 的 CUDA 前提（进程里先 `import torch`）都在
  [`docs/perception_ocr.md`](docs/perception_ocr.md)**。
- **PerceptionWorker**（`perception/pipeline.py`）：`AlignmentBuffer.wait_new` 逐帧取图
  （**一帧不落**）→ 检测 → `ocr` 模式裁剪送检 / `cls12` 模式类别直出 → `Detection` 入结果队列。
  OCR 在**独立进程**池里跑（绝不阻塞逐帧检测）；默认**不启用**跨帧去重，只有显式给
  `ocr_dedupe_s`/`ocr_dedupe_px` 时才按"拍摄时刻 + 中心位移"窗内去重。`_emit(set_code=True)`
  才覆盖编号——`cls12` 的编号来自 YOLO 类别，不能被 OCR 路径清掉。
  ⚠ **三条队列都不丢弃**（请求 / OCR 结果 / 检测结果都无界）：目标可能只清晰一瞬，
  丢一个请求就可能丢目标；积压超过 `ocr_queue_size`（OCR）或 2000 条（结果）时只告警，
  **绝不因为"队列满"丢数据**——积压长期不降就调 `ocr_workers` 或降检测帧率。
- **DroneController**（`telemetry/controller.py`）：任务级控制，把 MAVSDK 的
  **`mission_raw`**（上传/下载/复位当前项/进度）/`mission`（只用来启动）/`gripper`/`action`
  插件包成"上传任务（含回读校验）/ 启动（先复位到第 0 项）/ 查是否真在任务模式 / hold / rtl /
  投放 / 查原点 / 查任务是否飞完"，全部经 `MavsdkThread.submit` 投到 MAVSDK 线程执行，
  指令间用一把 `RLock` 串行。**失败语义统一**：命令没做成抛 `ControllerError`；
  `None`/`False` 只表示"读到了、但确实还没有"（原点未就绪、任务未完成）；而"快照里还没有
  飞行模式/任务进度"算**查询失败**（抛 `ControllerError`，调用方按查询失败处理）。
  ⚠ 它**不在导入期读 config**——`config → video.source → telemetry` 这条导入链会让它成环
  （装配走 `from_config`）；对 `mission.items` 也只在类型标注下导入，运行时要 `command_name`
  就在函数体内惰性导入。
- **mission**（`mission/`）：
  - `states.py`：`INIT→PREFLIGHT→WAIT_AIRBORNE→RECON→HOLD_PROCESS→OVERFLY→LAND→DONE`，任意环节异常 → `ABORT`；
    `INIT` 只等遥测 + NED 原点（**不上传任何任务**）；`PREFLIGHT` 跑一次自检
    （载入模型 + 视频自检，见 `airdrop/preflight.py`）；`WAIT_AIRBORNE` 等 `in_air`
    （取不到时用 `relative_altitude_m >= MissionConfig.airborne_alt_m` 兜底）；`RECON` 默认
    **等操作手在 QGC 上传并启动侦查航线**（`MissionConfig.recon_upload="operator"`；`"auto"` 只用于自动测试）；
    `TRANSITIONS` 是**唯一**的合法边表，非法转移抛 `InvalidTransition` 且**状态不变**，
    每次成功转移记一条 `state` 事件（含 from/to/reason/timestamp）。
  - `items.py`：`MissionItem` = **任务的唯一表示**（MAVLink 级：command/frame/params/位置），
    构造器 `waypoint()/takeoff()/land()/from_waypoint()`。任务项**一律走 `mission_raw` 上传**，
    不过 MAVSDK 的 `vehicle_action` 翻译层（那层会把 `LAND` 一项拆成两项、被 PX4 整条拒掉，见"易错点"第 10 条）。
  - `plan_file.py`（纯 JSON，离线）：`load_plan()` 解析 QGC `.plan`（`SimpleItem` 逐字读、
    `fwLandingPattern` 复杂项照 QGC `LandingComplexItem::appendMissionItems` 展开），
    `check_fixed_wing_landing()` 照 PX4 的可行性判据做**本地预检**。
  - `planner.py`（纯函数）：侦查段 → 任务项（首项带起飞项）；飞掠段以目标为中心、
    沿配置航向前后各半段长生成 `[entry, exit]`（**方向与判据的"越过目标"一致，符号不能反**）；
    飞掠段 + 降落段**拼成一条任务**上传（Q13）；无目标时用备用点（Q11）。
    **每条腿的航线来源二选一**：配置航点，或操作手的 QGC `.plan`（`recon_plan`/`land_plan`；
    降落段用 plan 时飞掠段插在它前面，且仍要过降落预检）。
  - `runner.py`：`MissionRunner.run()` 是普通循环（**不自己起线程**），`update()` 可单拍驱动；
    控制器与投放判据都按协议注入，离线用假对象。`clock`/`sleep` 也是注入的——测试里
    分钟级任务压成毫秒级确定性跑完。上传启动后要 `_confirm_started()` 确认真进了 `MISSION`
    （`mission_start_timeout_s` 超时 → `ABORT('mission_not_started')`），确认成功记 `mission_confirmed`。
    - **`preflight` 也是注入的**（`PreflightLike | None`，见 `airdrop/preflight.py`）：`None` 时
      `_tick_preflight` 记一条 `preflight_skipped`（并 warning）后放过——**离线测试/演练专用，
      正式入口（`examples/full_mission.py`）必须注入真的 `Preflight`**；任一项失败/抛
      `PreflightError` → `ABORT("preflight_failed:<check>")`，整体超 `PreflightConfig.max_s`
      → `ABORT("preflight_timeout")`。
    - **`PREFLIGHT` / `WAIT_AIRBORNE` 是"瞬时状态"**（`_INSTANT_STATES`）：条件当场满足时
      **同一拍里连跳**（离线用例一拍就能从 `INIT` 进 `RECON`），但每一跳都照常写历史与
      `state` 事件——不应把"一拍跳三下"理解为都被跳过。
    - `_begin_recon` 分两条路：`recon_upload="auto"` 自己上传并启动；`"operator"`（默认）
      **不上传**，记一条 `recon_waiting_operator`，并把 `_start_confirmed` 置 False 等操作手
      在 QGC 启动（`recon_max_s` 是那道等待的上限）。相关新事件：`airborne`（带 `source`）、
      `preflight_skipped`、`recon_waiting_operator`。
- **TargetTracker / PerceptionTargetSource**（`mission/targets.py`）：把
  `Detection`（像素 + 编号 + **拍摄时刻**遥测）经 georef 变成 `TargetPoint`，
  `result()` 就是 `analyze(points, config.targeting)`——**实飞与回放共用同一条路**。
  `PerceptionTargetSource(worker, tracker)` 再把它包成 `MissionRunner` 要的
  `target_result` / `target_busy` 两个回调（`result()` 会顺便抽干结果队列）。
  - **去畸变只做一次**：检测器若已 remap（`DetectorConfig.camera_matrix` 非空），
    像素就在纠正后的图上，georef 不能再纠正。`Detector.detect` 会把
    `extra["undistorted"]` 写进 `Detection`，`TargetTracker(undistort=None)` 默认据此自动判断。
  - **边长交叉验证默认关掉**（`side_check_tolerance=0`）：它是**回放优化**用的诊断
    （用已知 1 m 边长独立估深度与求交互校；标定/几何有问题时它会大面积不通过），
    正式流程里大倾角帧天然超门限（实测 35 个观测里 15 次），只会刷日志。要打开就
    显式给门限——回放入口 `--side-check 0.25`；打开后仍只记 `side_check` 事件与计数、
    **不剔点**（误判一个真实观测比多一个野点的代价大；DBSCAN 本来就靠"看到几十次"
    筛野点），要真的剔点再加 `reject_on_side_mismatch=True`。
  - 缺姿态/缺位置的点**不猜**：分别计入 `attitude_missing` / `no_fix` 后跳过。

## 关键约定与易错点（务必遵守）

1. **mavsdk 没有公开 `close()`**：会话结束必须显式 `_release_drone()`（guarded 调用
   `_stop_mavsdk_server`，幂等，这是 `System.__del__` 的同款路径）。**绝不能**只把引用置 None
   等 GC——mavsdk_server 子进程的 gRPC 端口固定 50051，旧进程不死，新会话会连上僵尸 server
   并在其被回收时级联断连。`_run_session` 用 try/finally 保证覆盖所有退出路径（含 connect 超时）。
   ⚠ **释放必须有超时兜底**：没有飞控/SITL、`connect()` 超时之后，这条释放路径会**永久阻塞**
   （gRPC poller 线程报 `Event loop is closed`，随后 `System.__del__` / `_stop_mavsdk_server`
   里的调用不再返回），没有兜底就会把 `stop()` 的调用方一起拖死。`_release_drone` 因此把释放放进
   **守护线程**、只等 `RELEASE_TIMEOUT_S`（5s）就放手。**推论：探活"有没有飞控"不要建 MAVSDK
   会话**（`tests/test_sitl_recon.py` 用裸 UDP 听 14540 心跳）。
2. **历史按时间查询**：用 `bisect(key=operator.attrgetter("timestamp"))` 直接探测 deque，
   禁止重建整张时间列表（曾 22µs/次 → 现 0.42µs/次）；外推分支不做任何二分。
3. **代码风格**：注释/docstring 用中文；类型标注用现代写法（`X | None`、内置泛型、
   不要 `typing.Dict`/`Optional`）；快照传递用浅拷贝（字段全标量）。
   ⚠ **规范不是"口头约定"，是三条命令**：`./.venv/Scripts/ruff.exe check .`、
   `./.venv/Scripts/ruff.exe format --check .` 与 `./.venv/Scripts/pyright.exe`
   都必须零告警（详见下面"验证方式"里的代码规范一节）；提交前跑一遍，别等 review。
4. **git 提交**：conventional 前缀 + 中文摘要，如 `fix(broker): ...`、`refactor(mavsdk_thread): ...`；
   一个逻辑改动一个提交。
5. **快照语义**：是"最新可用值合并"，字段间无严格时间同步。帧-遥测的**时间轴**对齐
   已由 `airdrop.video.align` 解决（按拍摄时刻查历史快照），但这不等于快照内部各字段
   来自同一瞬间——两者不应混为一谈。
6. **HM30 图传只有 ffmpeg 一个后端，不要再引入 `cv2.VideoCapture`**（本地 OpenCV 5.0 实测）：
   cv2 路线断流后仍读得出 **17 帧 / 567ms**（后端内部排队深度，无开关可改），本项目 ffmpeg
   子进程是 **0 帧**；且 cv2 的 `read()` 卡住只能**杀进程**、断流超时要靠 ffmpeg 的 `-timeout <微秒>`。
   ⚠ **"OpenCV 5.0 移除了 `OPENCV_FFMPEG_CAPTURE_OPTIONS`"是错的**（它在插件 DLL 里）。
   完整逐项排除表、16 位条码丢帧实验、启动盲区 `≈probesize`、`CAP_PROP_READ_TIMEOUT_MSEC` 的
   params 形式、HM30 网段与相机地址 →
   [`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md)。
7. **"留存"与"实时看"是两条路，不要混用**（本项目硬要求：目标可能只出现一瞬，漏一帧
   就可能漏掉目标；"先侦查后空投"允许处理延时、不允许丢帧）：
   - **留存**：`Hm30VideoSource.add_sink(callback)` 注册的回调在**采集线程里逐帧**调用，
     一帧不落。标准接法是 `source.add_sink(AlignmentWriter(buffer, aligner))`，
     于是每一帧连同它拍摄时刻的遥测都会进环形缓冲。sink 必须快（`put` 只做一次 JPEG
     编码，几毫秒），它抛异常只记日志、不打断拉流。
   - **实时看**：`read()` / `latest()` 只保证"当前这一帧"。慢消费者会丢掉中间帧并计入
     `stats.dropped`——这只影响这条实时路径，缓冲里的历史一帧都不会少。
   - 因此：**推理/写入磁盘的输入要来自 `AlignmentBuffer`（`iter_between` / `wait_new`），
     不要用 `read()` 循环送入**——那等于把丢帧重新引回来。`AlignmentWriter.written` /
     `skipped` 用来核对：`skipped` 只会因"该时刻取不到遥测"增加，不会因消费慢增加。
8. **帧-遥测对齐要扣链路延时**：按时间取遥测时用 `frame.capture_timestamp`
   （= `frame.timestamp - telemetry_lag`），**不要**直接用 `frame.timestamp`——后者
   是"收到帧的时间"，比画面真实拍摄时刻晚一整个链路延时（飞机以 10 m/s 平飞时就是
   1.5 米偏差）。`telemetry_lag` 是链路属性、写在 `VideoConfig` 上并随每一帧下发，
   默认 0.15s（ffmpeg 后端实测值）；换相机/链路或标定出新值后改这一处即可。
   外推结果要按 `AlignedSample.extrapolated` / `offset` 判断是否可用（必要时设
   `max_extrapolation`）。
9. **别把"帧的 ndarray"当稳定数据持有**：
   - ffmpeg 后端的 `frame.image` 是**复用管道缓冲区的视图**，下一帧读入会覆盖同一块
     内存。要跨帧保留画面必须 copy（`AlignmentBuffer` 内部已处理；你自己保存也要
     `.copy()` 或 `frame.copy()`），否则历史帧最后全变成同一张最新画。
   - 回看历史一律走 `AlignmentBuffer.iter_between()`（分批解码、迭代期间不持锁、
     快照式）。模块刻意**不提供**"一把返回列表"的接口——5400 帧 720p 全解出来是
     13.9 GiB，想要列表就自己 `list(...)`，那份内存账由调用方认。
   - `raw` 模式读到的 image 是**只读视图**（零拷贝），需要就地画框先 `record.copy()`。
10. **任务项一律走 `mission_raw`，不应使用 MAVSDK 的 `vehicle_action`**：`mission` 插件会把
    `vehicle_action=LAND` 的**一项拆成两项**（同坐标 `NAV_WAYPOINT` + `NAV_LAND`）。PX4 固定翼
    的可行性检查要求 `NAV_LAND` 的**紧前一项严格高于落点**，被拆出来的那项与落点同高同坐标 ⇒
    **整条任务被拒**（`[mission_feasibility_checker] Mission rejected: the approach waypoint must
    be above the landing point.` + `[navigator] No valid mission available, loitering`），
    而 `start_mission()` **仍然回成功**——2026-09 的 SITL 演练里飞机就这么原地盘旋了 15 分钟。
    实测对照（同一飞控、同一会话）：`WP 50m + LAND alt=0` ⇒ 飞控存下 `WP 50m`、**`WP 0m`（复制品）**、
    `NAV_LAND 0m`；把 LAND 抬到 15m ⇒ 复制品跟着变 `WP 15m`，`NAV_LAND` 仍是 0m（于是又被
    "下滑角过陡"拒）。raw 通道原样投递，没有这层翻译；**别为了"方便"改回 `mission` 插件**。
11. **固定翼降落两条硬判据**（PX4 `MissionFeasibility/FeasibilityChecker.cpp`）：紧前一项必须
    **严格高于**落点（`< FLT_EPSILON` 就拒）；下滑**斜率（tan）**`(前项高 − 落点高)/水平距离 ≤
    tan(FW_LND_ANG+0.1°)`（出厂默认 8° ⇒ 上限 0.142 ≈ 8.1°；⚠ 下面的 0.219/0.067 都是斜率，
    不是角度），进场项只能是 `NAV_WAYPOINT` 或 `NAV_LOITER_TO_ALT`，且 `LOITER_TO_ALT` 时
    落点要在盘旋圈外（水平距离按 `sqrt(圆心距²−半径²)` 修正，与 PX4 一致）；距离用与 PX4
    `get_distance_to_next_waypoint` 同款的**球面 haversine**（半径 6371000，`_EARTH_RADIUS_M`），
    **别换成椭球/Geod**——预检要与飞控同一把尺子。实测：40m 高 /
    183m 远（tan 0.219 ≈ 12.4°）被判"下滑角过陡"；20m / 300m（tan 0.067 ≈ 3.8°）通过。
    本包用 `check_fixed_wing_landing()` 在**规划阶段**预检，不合格直接 `PlanningError`
    （别等飞控拒了再查）。
12. **`start_mission()` 回成功 ≠ 进了任务模式**：飞控可能拒绝模式切换却仍然回 ACK。另外
    **同一 CRC 的任务再启动时，飞控不清"已飞完"锁存**，会停在 `HOLD`。实测对照：HOLD 下直接
    `start_mission` ✗（`No valid mission available, loitering`）／先 `set_current_mission_item(0)`
    再启动 ✓。所以 `DroneController.start_mission()` **先复位再启动**，状态机再用 `in_mission_mode()`
    正向确认（`MissionConfig.mission_start_timeout_s` 超时 → `ABORT('mission_not_started')`）。
13. **`mission.is_mission_finished()` 在 raw 上传下永远回 `False`**：那个插件只认它自己上传过的
    任务（`last_upload`）。完成判定改读 `mission_raw.mission_progress()` 的 **`current == total`**
    （MAVSDK 文档语义：`current` 是 0 基下标，等于 `total` 即飞完），由可选流写进快照
    `mission_current`/`mission_total`；模式确认同理读快照 `flight_mode`。
14. **本地环境两条**：(a) 同一时刻只能有一个 `mavsdk_server`（固定占 gRPC 50051），调试要有第二个
    脚本就用 `System(mavsdk_server_address="localhost", port=50051)` 接**已有** server，别再起一个；
    (b) `mavsdk_server --version` **会阻塞**（它去起服务了），版本看日志首行
    `mavsdk_server: MAVSDK version: vX.Y.Z`（与 Python 包版本一致，本地 3.17.2）。
15. **QGC `.plan` 的两条约定**：(a) `fwLandingPattern` 复杂项在本地按 QGC
    `LandingComplexItem::appendMissionItems` 展开（`DO_LAND_START` → 可选 `DO_CHANGE_SPEED` /
    停止拍照录像 → 进场项 → `NAV_LAND`），PX4 的 `DO_LAND_START` 是 `specifiesCoordinate=false`
    ⇒ **不带坐标**、`frame=MAV_FRAME_MISSION`；其它复杂项（VTOL 降落、测绘/结构）显式报错，不猜着展开。
    (b) **每条腿的航线来源只能一个**（配置航点 或 `.plan`），两个都给时 `Config.validated()`
    直接报错——静默让其中一个优先比报错危险得多。
16. **正式任务的启动顺序是"自检 → 等起飞 → 侦查"，且侦查航线默认人工上传**：
    `INIT` 等遥测/原点 → `PREFLIGHT` 载入模型（detector/ocr/camera）+ 视频自检
    （`PreflightConfig`，各项可关；测试/SITL 常常没有相机或模型）→ `WAIT_AIRBORNE` 等
    `in_air`（`MissionConfig.airborne_timeout_s` 兜底，取不到 `in_air` 时用
    `relative_altitude_m >= airborne_alt_m`）→ `RECON`。
    `MissionConfig.recon_upload="operator"`（默认）表示**侦查航线由操作手在 QGC 上传并启动**，
    本包**不上传**、只等它开始再监视进度；`"auto"` 才由本包上传（**只用于自动测试**——否则会
    覆盖操作手刚画好的航线）。
    ⚠ 三条不能忘：**检查被关掉 ≠ 检查通过**（关掉的项记 `preflight` 事件、`ok=null`）；
    **开着却没注入载入回调 = 装配错误**（直接 `ABORT("preflight_failed:<check>")`，绝不静默通过）；
    在停机坪上就进侦查是旧行为留下的易错点——PX4 会在地面"追"第一个航点，或因任务不可行直接盘旋。
    **等待起飞的检查有临时放行开关**：`MissionConfig.require_airborne`（默认 `True`）置 `False` 时
    `WAIT_AIRBORNE` 立即放行，并记 `airborne_skipped` 事件（带 `reason`）+ WARNING 日志——
    **只给地面演练/离线测试用**（`examples/sitl_mission.py` 的 `REQUIRE_AIRBORNE = False`），
    正式任务保持 `True`；放行**不静默**，落地复盘时一眼能看出这次没有等起飞。
17. **入口与依赖的两条硬约定（`airdrop/run.py` + 惰性导出）**：
    (a) **命令行只在 `airdrop/run.py` 一处**（argparse 子命令，注册表 `SUBCOMMANDS`）：
    `examples/*.py` / `tools/*.py` 一律是纯库模块（常量 = 默认值、`build_config(**覆盖)`、
    `main(**kwargs)`），**谁都不许自己 import argparse**；新增入口就在注册表里加一条，
    `tools/check_docs.py` 会核验手册是否覆盖了每个子命令。
    (b) **重依赖（torch / cv2 / mavsdk / ultralytics / rapidocr / onnxruntime）只能在函数体内导入**：
    `airdrop/__init__.py` 与 7 个子包（video / telemetry / mission / georef / record /
    perception / ballistics）都是 PEP 562 惰性导出（`airdrop/_lazy.py`，精确的
    `_EXPORTS` 名字→叶子模块表），于是 `import airdrop`、每个子命令的 `--help` 与
    `python -m airdrop.run check-docs` 都**不加载**这些库（`tests/test_cli.py` 用干净子进程
    钉住）。`__all__` 的内容与顺序不许动——`tools/dump_api.py` / `check_docs.py` 依赖它。
    `check-docs` 的"引用可解析"那一步要真 import（`airdrop.video.buffer` 这类），因此它跑在
    **子进程**里，别改回本进程 import。
18. **行尾统一 LF（`.gitattributes`）**：仓库用 `* text=auto eol=lf`，索引与工作区都按 LF
    检出/比较；新增或改写文件别引入 CRLF——部分工具（如 OpenCode 的 Git 层）会硬编码
    `-c core.autocrlf=false`，CRLF 工作区会被显示成"整文件删除 + 插入"
    （`anomalyco/opencode#27276`）。本仓库已配 `core.autocrlf=false`（local）与
    VSCode `files.eol="\n"`；自查：`git ls-files --eol | awk '$2=="w/crlf"'` 应为空。

## 验证方式

- **测试（pytest，全部离线，无需飞控/图传硬件）**：
  `./.venv/Scripts/python.exe -m pytest`（附覆盖率；**默认就排除 `realdata` 与 `sitl`**）
  `./.venv/Scripts/python.exe -m pytest -m "not realdata and not sitl and not stream"`（再去掉要起 ffmpeg 的用例）
  `./.venv/Scripts/python.exe -m pytest -k alignment -v`（只跑某个主题）
  ⚠ **命令行的 `-m` 是"覆盖"而不是"追加"**：只写 `-m "not stream"` 会把 `realdata` 与 `sitl`
  重新放进来（后者要 SITL 真在跑，否则只是白等一轮探活）。
  配置在 `pyproject.toml` 的 `[tool.pytest.ini_options]`：`--strict-markers`、180s 全局
  超时兜底、默认 `-m "not realdata and not sitl"`；标记有 `stream`（起 ffmpeg / 占 UDP 51234）、
  `network`（等网络错误路径）、`realdata`（**GPU + 真实航拍素材，分钟级**，要跑显式
  `-m realdata`）、`sitl`（要 WSL 里已起 PX4 SITL 与图传，分钟级，显式 `-m sitl`）；公共 fixture 在
  `tests/conftest.py`（本地 H.264 测试流 `sender`、`live_source` 工厂、`broker`/`received`）。
  测试依赖在 `dev` 组。修 bug 时把回归用例加进对应模块——`tests/` 就是回归测试落点。
  - `test_perception.py` **全部离线**：重依赖（YOLO/RapidOCR/torch）都不加载，pipeline 的
    `detector`/`pool` 都是构造注入的假对象——这正是那两个参数存在的理由。
  - `test_perception_realdata.py`（标 `realdata`）才碰 GPU 与真实素材，验收口径是
    "真实航拍视频的目标段能读出正确编号（**56/56/56**）"。⚠ 第 3 帧原记的是 `95`，
    那是转正 180° 缺陷造成的翻转误读（物理目标是 56），已随缺陷修复一并改正——
    详见 [`docs/perception_ocr.md`](docs/perception_ocr.md)。
  - `test_telemetry.py` 覆盖遥测订阅推送、内插/外推与遥测速率下发。
  - `test_alignment.py` 直接往 `broker._history` 注入确定性时间戳（不靠 sleep 凑时间）。
  - `test_buffer.py` 用"同一个 bytearray 反复 reshape 成 view"复现 ffmpeg 后端的复用
    缓冲行为，盯的是"入缓冲必须拷贝"这条易错点；另有 `AlignmentWriter` 的逐帧入库用例。
  - `test_recorder.py` / `test_replay.py` 的素材都是**手写**的（不依赖真实录制）：按写入磁盘的
    格式直接写 `frames_index.jsonl` + jpeg + `telemetry.jsonl`，放在工作区下的临时目录里
    （不用 `tmp_path`，见下）。`test_replay.py` 的验收口径是"回放对齐结果与直接从日志
    查询的参考 broker **完全一致**"——不能拿"与实飞 broker 一致"当标准，因为 recorder
    按 10Hz 节流写盘，更快的原流本来就被有损降采样了。
    `test_calibrate.py` 的 `calibrate()` 输出契约用例同样自建临时飞行目录，
    放在工作区内的 `.calibrate-test-tmp/`（`workdir` fixture 用完即删；这两个根目录已进
    `.gitignore`，免得用例中途崩掉留下未跟踪文件）。
  - `test_mission.py`：**假控制器 + 假时钟**。`FakeController` 按
    `MissionController` 协议实现（`fail_on` 注入命令失败、`sticky_finished` 模拟"新任务
    仍报上一次已飞完"的陈旧读数、`mission_progress` 提供快照里的任务进度），`FakeBroker`
    提供可控时间戳的快照，`TimedController` 按假时钟自动推进任务进度——于是**分钟级任务在
    毫秒级内确定性跑完**（`run()` 用例断言 180s 假时间走完 DONE）。`DroneController` 那一组用
    "假线程 + 假 drone"离线验证 MAVSDK 调用的装配（`mission_raw` 上传的 raw 项内容、
    `current` 只给第 0 项置 1、`autocontinue` 传 int、上传后回读校验、gripper 实例号、异常翻译）。
    注意：**状态是在某一拍结束时进入的**，所以"进入新状态后跑一拍"的断言要自己再调一次
    `update()`（`_drive` 与 `_drive_until` 的分工就是这个）。另有一组专门覆盖**起飞前自检与等起飞**
    （`PREFLIGHT` 载入模型/视频自检、`preflight_failed:<check>`/`preflight_timeout`/`preflight_skipped`、
    `WAIT_AIRBORNE` 的 `in_air` 与高度兜底、`airborne_timeout`、`recon_upload` 的 operator/auto 两条路）。
  - `test_plan.py`：QGC `.plan` 的纯 JSON 解析（`SimpleItem` / `fwLandingPattern` 复杂项展开）、
    `check_fixed_wing_landing` 的每条判据（同高必拒、下滑角过陡必拒、盘旋圈、进场项类型、
    混合高度基准）、plan↔配置航点二选一。**离线、不需要飞控**。
  - `test_e2e.py` = **回放驱动的全链路**：手写一个合成飞行目录（帧索引 + jpeg +
    遥测：位置/四元数/原点）→ `ReplayVideoSource` → 对齐 → 缓冲 → `PerceptionWorker`
    （注入脚本化假检测器）→ `TargetTracker`（**真 georef**）→ `targeting` → `MissionRunner`
    → 飞掠航线。坐标能写死是因为让飞机**水平朝下看**且目标像素取**主点**：
    主点视线就是光轴，水平姿态下正对地面 ⇒ 目标必然在飞机正下方 `(north, east, 0)`。
    另有一条 `-m realdata`：**真视频 + 真 YOLO + 真 OCR**，位姿是合成的（那段素材没有遥测），
    验"真实像素能一路走到坐标"而不验精度。⚠ 那条用例把 OCR 放在**进程内**
    （`RealYoloOcrDetector`）；进程池那条路由 `test_perception.py` 的假池用例覆盖。
  - `test_examples.py`：示例导入即校验（API 没脱节、不自己 import argparse、都有
    `main()` / `build_config()`、**导入期不加载重库**），并**真跑**几个示例的装配函数
    （`build_config()` 的默认值与覆盖、SITL 的合成目标）——这几处不碰硬件，
    导入检查抓不到 `replace()` 写错这类问题。
  - `test_cli.py`（集中式入口）：子命令注册表（每个子命令都有 parser + handler + 默认值出处）、
    `--help`（顶层与全部子命令）与 `check-docs` 在**干净子进程**里跑完不许加载
    torch/cv2/mavsdk/ultralytics/rapidocr/onnxruntime、`import airdrop` 同样轻、
    惰性导出后跨子包名字仍取得到、未知子命令/非法参数退出码 2、
    以及"选项 → 关键字 → `build_config()`"真的覆盖得动（full-mission / sitl / replay /
    calibrate / fit-ballistics 五条）。
  - `test_video.py` 会起 ffmpeg 子进程并占用本地 UDP 端口 51234。其中"地址无效要显式
    报错"这组故意去连 `rtsp://127.0.0.1:1/...`，日志里出现 error 级"图传不可用"是预期输出；
    "每帧都经过 sink"那组故意慢消费，`stats.dropped > 0` 是预期（sink 侧仍一帧不少）。
    注意断言要在 `source.stop()` **之后**取 `stats`，否则会与在途帧差几帧。
- **语法/编译检查**：
  `./.venv/Scripts/python.exe -m compileall -q airdrop tests tools examples`
  （⚠ 别写 `py_compile airdrop/*.py`：PowerShell 不会给原生命令展开通配符，
  py_compile 会把 `airdrop/*.py` 当字面文件名，报 `[Errno 22] Invalid argument`——
  而且它是**非零退出**，别被后面的命令掩盖掉。）
- **代码规范（ruff + pyright，都在 dev 依赖里，配置在 `pyproject.toml`）**：
  三条命令**都必须零告警**，改完代码先跑它们，再跑 pytest：
  `./.venv/Scripts/ruff.exe check .`、
  `./.venv/Scripts/ruff.exe format --check .`、
  `./.venv/Scripts/pyright.exe`。
  - ruff 是**务实档**（E4/E7/E9/F/W/I/UP/B/SIM/C4/RUF/BLE/SLF/PL/PT），
    不是 `select = ["ALL"]`——本项目注释/文档是中文，ALL 里 RUF001/002/003
    （易混 Unicode 字符）对全角标点一条就能报 7k+ 次。那三条**保留启用**，
    只把 12 个刻意的排版字符列进 `allowed-confusables`（全角括号/逗号/冒号/分号/问号、
    `×`、`−`、`–`、`α`、`ρ`、`σ`），所以真正的形近字符误用照样能抓到。
  - 关掉的规则都写了理由（`UP037` 会连带删掉中文 docstring 里的 `**强调**`、
    `RUF046` 的 `int(round(...))` 在 numpy 标量上是必要的收窄、`PLR2004`/`PLC0415`
    与本项目"物理字面量 + 重依赖函数体内导入"的约定冲突）。**`E501` 也是刻意关的**：
    `ruff format` 在时会自动压掉它，而格式化器对"拆不开的行"（长 URL、单条长字符串、
    行尾 `noqa`）本就放弃限长——显式打开只会稳定报出 10 条它自己不愿拆的行（实测
    101~108 列）。`line-length = 100` 是给格式化器当**指导**用的，不是"每行都要 ≤100"。
  - 全库已格式化过一遍（2026-09，67/85 个文件），`format --check` 现在应是
    `85 files already formatted`。⚠ 格式化**只许动排版**，验收判据不是"看着像"而是
    **"格式化前后 AST（剥掉位置信息）完全一致"**——本次实测 77/77 个文件等价，
    唯一的内容变化是一个以引号开头的 docstring 按 PEP 257 补了首空格。
    以后要再动格式，照这条自己写个 AST 比对脚本验一遍（别只依赖测试通过）。
  - `tests/**` 与 `tools|examples/**` 有 per-file ignores（私有成员、复合断言、
    打表输出、参数多等），见 `[tool.ruff.lint.per-file-ignores]`。
  - pyright 用 `basic`（实测与 `standard` 报错数完全相同），
    `reportUnsupportedDunderAll` 关掉——惰性导出（PEP 562）的 `__all__` 静态解析不了，
    一个 `airdrop/__init__.py` 就是 175 条噪音；一致性由 `tests/test_cli.py` 真跑钉住。
  - ⚠ 排查原则：**这两条报出来的东西先当真 bug 看，别顺手加 `noqa`/`ignore`**。
    本次落地就靠它们抓到两个真缺陷：`fit.py` 给 `FitResult` 传了不存在的
    `iterations=`（字段叫 `nfev`，那条"解处仍有样本预测不出来"的分支一走到就
    `TypeError`）、`test_mission.py` 里把说明写成了 `f(), "…"` 这样的表达式语句
    （等于那条断言根本不存在）。确实要抑制就带理由写 `# noqa: RULE - 原因`
    （全库现有 28 处，绝大多数是 `BLE001`，即"这里就是要兜住所有异常"）。
- **入口与帮助信息（`airdrop/run.py`）**：
  `./.venv/Scripts/python.exe -m airdrop.run --help` 与**每个子命令**的 `--help` 都必须
  **exit 0**，且跑完 `sys.modules` 里**不许有** torch / cv2 / mavsdk / ultralytics /
  rapidocr / onnxruntime；`python -m airdrop.run check-docs` 同理。
  `tests/test_cli.py` 在干净子进程里逐条实测（含"未知子命令/非法参数 = 退出码 2"）。
