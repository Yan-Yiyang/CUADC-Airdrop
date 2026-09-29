# AirDrop —— 固定翼无人机察打一体控制系统

**版本 0.1**。完整任务链路已实现并有离线测试覆盖。

## 设计背景

本项目面向无人机竞赛的"侦察与打击"任务，设计约束是**不使用机载计算机**：飞行器上仅
搭载相机、视频下传链路、飞控与投放机构；目标检测、编号识别、像素→坐标解算、目标统计
与弹道预测全部在地面计算机完成。该约束带来两条实现要求：

- **视频逐帧留存**：目标可见时间可能极短，丢失任一帧都可能丢失目标；允许处理延时，
  不允许采集丢帧（见"关键约定"）。
- **帧-遥测按拍摄时刻对齐**：解算依赖画面与遥测的严格时间对应，须扣除链路固定延时后
  再查询遥测（见"关键约定"）。

任务按察打一体流程执行：

```text
起飞 → 预设侦察航线（约 1 分钟）→ 盘旋保持
     → 感知（YOLO 检测 + OCR 读编号）→ 像素→NED 坐标解算 → 目标统计（聚类并选取唯一结果）
     ├─ 有结果 → 生成飞掠航点（航向/高度/段长按配置）+ 降落航线，合并上传
     └─ 无结果 → 以备用点为目标生成相同航点
     → 飞掠段：弹道模型实时预测落点，距目标 ≤R 时触发挂载投放；越过目标未投则强制投放
     → 沿降落航线返航降落
全程：飞行数据（日志/遥测/检测/事件/投放记录/视频帧）写入磁盘，飞行后可回放复现
投放后：实测落点 → 反演弹道参数，供后续架次使用
```

| 模块 | 功能 | 离线用例 |
| --- | --- | --- |
| `airdrop.telemetry` | MAVSDK 线程 + 遥测代理（快照/历史/插值）+ 任务级控制器 | `test_telemetry.py` |
| `airdrop.video` | RTSP 拉流（逐帧留存）、帧-遥测对齐、环形缓冲 | `test_video/alignment/buffer.py` |
| `airdrop.perception` | YOLO 检测 + 五边形转正 + RapidOCR 读编号（或 12 类直出） | `test_perception.py` |
| `airdrop.georef` | 像素 → NED（视线与地面求交）→ WGS84（pyproj） | `test_georef.py` |
| `airdrop.targeting` | DBSCAN 聚类 + 类内编号众数 + 跨类 median/max 选唯一 | `test_targeting.py` |
| `airdrop.ballistics` | 二次阻力 RK4 落点预测 + 投放判据（越过目标后强制投放 + 锁存）+ 投放记录与参数反演 | `test_ballistics.py`、`test_fit.py` |
| `airdrop.mission` | 状态机 + 飞掠/降落航线拼接 + 目标解算 + 主循环 | `test_mission.py`、`test_e2e.py` |
| `airdrop.record` | 飞行记录（五类记录文件 + 投放记录）+ 按原时间轴回放 | `test_recorder/replay.py` |
| `tools.calibrate` | 三步标定：内参 → 画面/遥测时间差 → 手眼外参 | `test_calibrate.py` |
| `tools.fit_ballistics` | 投放试验反演弹道参数（最小二乘 + 可辨识性诊断） | `test_fit.py` |

> 使用说明与模块示例见 [`docs/handbook.md`](docs/handbook.md)（目录总览、各模块调用方法、
> 参数总表、输入输出字段、使用流程、代码审查速查、排障表）。
> 实现细节、参数依据与约束见 [`AGENTS.md`](AGENTS.md) 与 [`docs/`](docs/) 下的专题文档。
> README 只说明使用方法。

## 目录结构

```text
airdrop/
  run.py               # 唯一的命令行入口（argparse 子命令 → 关键字覆盖到各入口的 build_config）
  _lazy.py             # PEP 562 惰性导出：import airdrop / --help 不加载 cv2、mavsdk、torch
  config.py            # 全部参数的唯一集中处（frozen dataclass + 取值域校验，参数本身不带命令行）
  telemetry/           # models / broker / mavsdk_thread / controller（mission+gripper+原点）
  video/               # source（RTSP 拉流）/ align（扣链路延时）/ buffer（环形缓冲）
  perception/          # detector（YOLO）/ cropproc（五边形→转正→OCR）/ number / ocr_worker / pipeline
  georef/              # camera（K/dist/R_bc/t_bc）/ project（视线交地平面）/ geo（NED↔WGS84）
  targeting/           # models / cluster（DBSCAN + 众数 + median/max）
  ballistics/          # model（RK4 二次阻力）/ release（判据）/ drops（投放记录）/ fit（参数反演）
  mission/             # states（状态机）/ planner（航线）/ targets（坐标解算）/ runner（主循环）
  record/              # recorder（五类记录文件 + drops.jsonl）/ replay（回放源 + 遥测回填）
tools/                 # 纯库模块：fetch_models / calibrate / fit_ballistics / make_world / dump_api / check_docs
examples/              # 纯库模块：full_mission / replay_flight / calibration_capture / sitl_mission / 其余演示
sim/                   # 仿真模块：赛区世界（生成产物）+ 带下视相机的机型 + airframe + install/run 脚本
tests/                 # 全部离线（GPU 与真实素材的用例标 realdata，默认跳过）
```

## 安装

```bash
uv sync                 # 依赖见 pyproject.toml
python -m airdrop.run fetch-models --source-dir <权重目录> [--source-ocr-dir <OCR 权重目录>]
                        # 把 YOLO / PP-OCR 权重复制进 models/（不进 git；只复制、不联网）
```

## 入口

命令行入口只有 `airdrop/run.py`（argparse 子命令）：`examples/*.py` 与 `tools/*.py`
均为纯库模块——文件顶部的常量即默认值，`build_config(**覆盖)` / `main(**kwargs)`
按需传值；命令行选项按关键字覆盖默认值。
`--help` 与 `check-docs` 不加载 torch / cv2 / mavsdk 等重型依赖（惰性导出 + 函数体内导入）。

```bash
python -m airdrop.run --help                 # 列出全部子命令
python -m airdrop.run <子命令> --help         # 该子命令的选项与默认值来源

# 1) 离线迭代识别与坐标：用回放素材反复运行，无需硬件
python -m airdrop.run replay --flight flights/20260913-185512 --speed 2

# 2) 完整任务（需要飞控 + 视频链路 + 标定文件）
python -m airdrop.run full-mission --land-plan routes/land.plan       # --dry-run 为不装弹演练
python -m airdrop.run full-mission --no-video --no-preflight          # 无视频/免自检的演练配置

# 3) SITL 演练：只走控制链路（合成目标 + 投放只记日志，默认不等起飞）
python -m airdrop.run sitl --target-offset 0,-20,0      # 航线已对准 sim/worlds/cuadc 的赛区

# 4) 标定采集：录制标准飞行目录供 calibrate 使用
python -m airdrop.run calibration-capture
python -m airdrop.run calibrate --flight flights/<目录> --strict

# 5) 投放试验反演：测量实际落点后估计弹道参数（只读数据，不接硬件）
python -m airdrop.run fit-ballistics --impacts flights/<目录>/impacts.jsonl --mass-kg 0.365

# 6) 重建 CUADC 赛区 Gazebo 世界（换天井布局/朝向/轮次/风/光照与底图；SITL 启动见 docs/simulation_world.md）
python -m airdrop.run make-world                       # 入库的两个世界（r1 图片靶 / r2 数字靶）
python -m airdrop.run make-world --rounds 2 --seed 3 --wind 5,2,0
```

其余演示：`basic`（遥测/指令）、`hm30-video`（拉流统计）、`video-sync`（对齐 + 入缓冲 + 读者线程）。

## 配置

全部参数集中在 `airdrop/config.py`，装配成 `Config` 向下传递；非法取值在
`Config().validated()` 阶段报错（而不是在飞行途中才暴露）。关键配置组：

| 配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `telemetry` | `system_address`、各流速率 | 位置/姿态速率决定帧-遥测插值精度上限 |
| `video` | `url`、`telemetry_lag` | lag 为链路属性，标定后回填 |
| `perception` | `mode`（`ocr`/`cls12`）、`model_path`、`device` | 检测权重使用 `best2.pt` |
| `camera` | `calib_file` | `tools/calibrate.py` 产出的 `camera_calib.json` |
| `ground` | 地面点 GPS | `ground_z` = 原点海拔 − 地面点海拔（平地假设） |
| `targeting` | `eps_m`、`min_samples`、`selection_rule` | `median`/`max` 起飞前二选一 |
| `ballistics`/`drop` | 质量/Cd/面积、`radius_m`、`force_after_pass`、`delay_s` | 质量按实测称重填入；Cd 由投放试验反演；`delay_s` 也可反演后回填 |
| `overfly`/`gripper`/`routes` | 航向/高度/段长、gripper 实例、三条航线 | 高度为相对起飞点的高度 |
| `mission` | 各状态超时、`abort_action`、起飞/降落项开关 | 见"状态机" |

## 关键约定

1. **"逐帧留存"与"实时读取"是两条路径**：`add_sink()` 注册的回调在采集线程中**逐帧**调用，
   不会丢帧；`read()`/`latest()` 是允许丢帧的实时路径。**推理与写入磁盘的数据应来自
   `AlignmentBuffer`**，不得用 `read()` 循环送入。
2. **帧-遥测对齐须扣除链路延时**：按时间取遥测时使用 `frame.capture_timestamp`
   （= 收到时间 − `telemetry_lag`）；直接使用 `frame.timestamp` 会滞后整个链路延时
   （10 m/s 平飞时约 1.5 m 偏差）。
3. **不得跨帧持有帧的 ndarray**：ffmpeg 后端的 `frame.image` 是复用缓冲区的视图，跨帧保留
   必须 `copy()`；回看历史使用 `buffer.iter_between()`/`wait_new()`。
4. **像素与内参必须对应同一张图**：检测器已做去畸变时（`DetectorConfig.camera_matrix` 非空），
   georef 不得再次纠正——`TargetTracker` 默认按检测结果中的 `undistorted` 标记自动判断。
5. **失败必须显式**：解算失败不编造坐标、投放判据给不出落点则不投放、任务异常进入 `ABORT`，
   不得把失败当成功继续执行；日志与事件中记录原因。

## 状态机

`INIT → RECON → HOLD_PROCESS → OVERFLY → LAND → DONE`，任意环节异常 → `ABORT`。

- `RECON`：上传侦察航线（首航点带起飞项）并启动，按飞控侧任务进度
  （`mission_raw` 的 `current == total`）判定完成；新任务刚上传时的陈旧完成标记不会被
  `MissionMonitor` 采信（要求先出现一次未完成再认可完成）。
- `HOLD_PROCESS`：`hold()` 盘旋，等待目标统计结果；处理完成且无结果则提前结束，
  上限 `hold_process_max_s`（默认 10s），超时取当前已有结果。
- `OVERFLY`：结果坐标（无则备用点）→ `[entry, exit]` 飞掠航点 + 降落航线合并为一条任务上传，
  激活投放判据（每拍预测落点，≤`radius_m` 即投；越过目标后强制投放；一次投放即锁存）。
- `LAND`：沿降落航线返航，任务飞完 → `DONE`。
- `ABORT`：只下发一条安全动作（`abort_action`：`hold` 默认 / `rtl` / `none`）——
  链路断开时该指令也无法发出，此种情况交由飞控自身的 failsafe 处理。

完整链路接法见 `examples/full_mission.py`；不装弹演练见 `examples/sitl_mission.py`
（`DryRunController` 拦截投放指令，其余全部真实执行）。

## 标定（三步，离线）

棋盘格固定、手持飞机在其上方平移 + 旋转，录制标准飞行目录
（`examples/calibration_capture.py`），然后执行：

```bash
python -m airdrop.run calibrate --flight flights/<目录>   # 常量仍是默认值（--help 中可见出处）
```

1. **内参**：多视图棋盘格 → `calibrateCamera`（`K`、畸变 + 重投影 RMS）；
2. **画面-遥测时间差**：棋盘格 PnP 角速度 × 飞控角速度互相关 → `telemetry_lag`（回填配置）；
3. **手眼外参**：`AX=XB` 求 `R_bc`；`t_bc` 以尺量为准，标定估计值写入报告供对比校验
   （室内手持没有 GPS，平移不可信）。

## 记录与回放

一次飞行对应一个 `flights/<时间戳>/` 目录：`flight.log`、`telemetry.jsonl`、
`detections.jsonl`、`drops.jsonl`、`events.jsonl`、`frames/*.jpg` + `frames_index.jsonl`、
`config_snapshot.json`。帧零重编码（直接写入缓冲中的 jpeg），磁盘占用约 4.5MB/s @720p30。

回放按原时间轴重放帧与遥测，对齐/解算/统计均无需改动：
`ReplayVideoSource` 与实时视频源接口一致，遥测由 `TelemetryPacer`
按帧时刻推进（`load_broker_from_log`）。算法迭代使用 `examples/replay_flight.py`。

## 投放记录与弹道参数反演

质量用台秤/天平实测（填 `MEASURED_MASS_KG`），不参与反演：弹道方程中 `Cd`/`m`/`A`
只以 **κ = Cd·A/m** 的形式出现，仅凭落点数据无法分离三者。
数据可辨识量只有 κ 一个数（工具会打印它，以及按实测质量换算的 `Cd·A`）。

每次投放的瞬间状态当场写入磁盘——位置、速度、姿态（欧拉角与四元数）、风、
离地高度、NED 原点、目标点，以及判据当时的前推位置与预测落点：

```jsonc
// flights/<时间戳>/drops.jsonl（一行一次投放；数字不四舍五入）
{"index": 1, "timestamp": 1756... , "position_ned": [120.0, -35.0, -20.0],
 "velocity_ned": [0.0, 18.0, 0.0], "attitude_deg": {"roll": 4.0, "pitch": -3.0, "yaw": 90.0},
 "wind_ned": [2.0, -1.0, 0.0], "ground_z": 0.0, "delay_s": 0.08,
 "origin": {"lon_deg": 8.0, "lat_deg": 47.0, "alt_m": 500.0},
 "predicted_impact_ned": [...], "predicted_error_m": 0.45, "ballistics": {...}}
```

落地后测量实际落点，写入同一目录的 `impacts.jsonl`（经纬度或 NED 均可；
首次运行工具会生成待填模板），然后执行：

```bash
python -m airdrop.run fit-ballistics --impacts <落点文件> --mass-kg <称重值>
```

工具会打印逐次投放的"判据预测 / 反演后预测 / 实测"对照表（含沿航迹、垂直航迹分解），
执行最小二乘反演，并给出识别量 κ = Cd·A/m、不确定度 σ、参数相关性、条件数、
留一交叉验证，最后输出可直接回填配置的 `BallisticsConfig(...)` 与 `ballistics_fit.json` 报告。

> 引用结论前先阅读结论行：Cd 与迎风面积只以 `Cd·A` 的乘积出现（同时拟合必然退化）、
> 释放延迟与阻力沿航迹方向高度相关（同高同速投放无法分离）；常数偏移（测量点/目标点
> 坐标错误）无法被任何参数吸收。以上情形工具会拒绝给出结论并说明原因——详见
> [`docs/ballistics_fit.md`](docs/ballistics_fit.md)。
> 质量填错多少，Cd 就随之错多少（κ 不变）——换配重（形状不变 ⇒ `Cd·A` 不变）时按
> `κ' = Cd·A/m'` 重算即可，不必重做试验。

## 视频链路（RTSP）

视频链路为 **RTSP 拉流**，地址由 `config.video.url` 指定。实现要求：

- 使用 ffmpeg 子进程读取，通过 `-timeout`（微秒）实现断流判定；子进程可被父进程直接终止，
  拉流线程不会因管道读取而阻塞。
- **不自动探测地址与分辨率**：地址错误时将 ffmpeg 的输出原样上报（`stats.last_error`），
  由使用者核对；输出尺寸由 `VideoConfig.width/height` 显式给出（默认 720p）。
- 逐帧留存与实时读取的语义差异见"关键约定"第 1 条。

核对地址可用 ffmpeg 手动拉取一帧：

```bash
ffmpeg -rtsp_transport udp -i <RTSP 地址> -frames:v 1 -f null -
```

## 测试与代码规范

```bash
./.venv/Scripts/python.exe -m pytest                 # 全部（附覆盖率）
./.venv/Scripts/python.exe -m pytest -m "not stream" # 跳过需要启动 ffmpeg 的用例
./.venv/Scripts/python.exe -m pytest -m realdata     # GPU + 真实素材（AIRDROP_REALDATA_VIDEO，分钟级）
./.venv/Scripts/python.exe -m pytest -k mission -v   # 只跑某个主题

./.venv/Scripts/ruff.exe check .                     # 代码规范（配置在 pyproject.toml）
./.venv/Scripts/ruff.exe check . --fix               # 自动修可修项
./.venv/Scripts/ruff.exe format .                    # 统一格式
./.venv/Scripts/pyright.exe                          # 类型检查（同上）
```

- 测试套件全部离线：不需要飞控/视频链路；`stream` 标记的用例会启动本地 ffmpeg 并占用 UDP 51234。
- 端到端测试在 `tests/test_e2e.py`：回放素材 → 感知 → 坐标 → 统计 → 航线；其中
  `-m realdata` 用例使用真实视频 + 真实 YOLO + 真实 OCR（视频路径由环境变量
  `AIRDROP_REALDATA_VIDEO` 指定；位姿为合成数据，见用例说明）。
- 投放反演测试在 `tests/test_fit.py`：用已知真值的弹道模型生成"实测落点"再反演，
  并覆盖必须拒绝的退化情形、测量文件的各种写法与工具的完整流程。
- **`ruff check`、`ruff format --check`、`pyright` 三者均须零告警**（均在 `dev` 依赖组）。
  ruff 规则集按本项目定制（中文注释中的全角标点经 `allowed-confusables` 精确放行，
  未关闭 `RUF001/002/003`）；pyright 使用 `basic` 模式。
  全库已按 `[tool.ruff.format]` 格式化，改动后执行一次 `ruff format .` 即可。

## 安全提示

- 任何会驱动飞机的操作（arm / takeoff / mission upload / 投放）**只应在拆桨或 SITL 环境里试验**；
  真机演练请使用 `DRY_RUN = True`（投放指令只记日志）并确认 `abort_action` 符合空域要求。
- 投放判据是**建议**而非保证：它按弹道与风预测落点，参数未标定过则只是"按配置飞行"。

## 许可与参赛声明

本项目由**太原理工大学航模协会**开发（作者：**燕羿洋**），以 **Apache License 2.0**
授权（见 [`LICENSE`](LICENSE)），并附加一条**参赛声明要求**（见 [`NOTICE`](NOTICE)）：

> 以本项目（或其衍生作品）参加竞赛、提交作品或公开发布成果时，必须注明使用了本项目
> "AirDrop" 并给出其来源（例如：太原理工大学航模协会、燕羿洋 AirDrop 项目 + 本仓库地址）。

常规的复制、修改、分发按 Apache-2.0 处理（保留版权与许可声明、注明改动）；附加条款是
版权持有者提出的额外要求，使用与参赛前请先阅读 `NOTICE`。
