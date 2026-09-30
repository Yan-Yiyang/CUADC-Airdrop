# CUADC固定翼无人机侦查与打击控制项目

**版本 3.0.0**。一条完整链路已经跑通并有离线测试覆盖：

```
起飞 → 预设侦查航线（约 1 分钟）→ 盘旋 hold
     → 感知（YOLO 检测 + OCR 读编号）→ 像素→NED 坐标解算 → 目标统计（聚类 + 选唯一）
     ├─ 有结果 → 生成飞掠航点（航向/高度/段长按配置）+ 降落航线，合并上传
     └─ 无结果 → 以备用点为目标生成同样航点
     → 飞掠段：弹道实时预测落点，距目标 ≤R 触发 gripper 投放；飞过目标未投则强制投放
     → 沿降落航线返航降落
全程：飞行目录（文本/遥测/检测/事件/投放/视频帧）写入磁盘，飞后可回放迭代
投放后：量出实际落点 → 反演弹道参数（质量/阻力/延迟），下一架次用标定过的模型
```

| 模块 | 干什么 | 离线用例 |
| --- | --- | --- |
| `airdrop.telemetry` | MAVSDK 线程 + 遥测代理（快照/历史/内插）+ 任务级控制器 | `test_telemetry.py` |
| `airdrop.video` | HM30 ffmpeg 拉流（一帧不丢）、帧-遥测对齐、环形缓冲 | `test_video/alignment/buffer.py` |
| `airdrop.perception` | YOLO 检测 + 五边形转正 + RapidOCR 读编号（或 12 类直出） | `test_perception.py` |
| `airdrop.georef` | 像素 → NED（视线与地面求交）→ WGS84（pyproj） | `test_georef.py` |
| `airdrop.targeting` | DBSCAN 聚类 + 类内编号众数 + 跨类 median/max 选唯一 | `test_targeting.py` |
| `airdrop.ballistics` | 二次阻力 RK4 落点预测 + 投放判据（越过目标后强制投放判据 `force_after_pass` + 锁存）+ **投放记录与参数反演** | `test_ballistics.py`、`test_fit.py` |
| `airdrop.mission` | 状态机 + 飞掠/降落航线拼接 + 候选类→坐标 + 主循环 | `test_mission.py`、`test_e2e.py` |
| `airdrop.record` | 飞行记录（五个记录文件 + 投放记录）+ 按原时间轴回放 | `test_recorder/replay.py` |
| `tools.calibrate` | 三步标定：内参 → 画面/遥测时间差 → 手眼外参 | `test_calibrate.py` |
| `tools.fit_ballistics` | 投放试验反演弹道参数（最小二乘 + 可辨识性诊断） | `test_fit.py` |

> **第一次上手先看 [`docs/handbook.md`](docs/handbook.md)**（项目手册：目录总览、各模块调用方法与
> 最小示例、参数总表、输入输出字段、使用流程、代码审查速查、排障表）。
> 设计决策与"遇到过的问题/实测结论"在 **[`AGENTS.md`](AGENTS.md)**（本仓库的工作手册），
> 长专题（标定/OpenCV、图传/ffmpeg、感知/OCR、投放反演）在 **[`docs/`](docs/)** 下的专题笔记，
> 两者**同等权威**。README 只说明使用方法。

## 目录结构

```text
airdrop/
  run.py               # **唯一的命令行入口**（argparse 子命令 → 关键字覆盖到各入口的 build_config）
  _lazy.py             # PEP 562 惰性导出：import airdrop / --help 不加载 cv2、mavsdk、torch
  config.py            # 全部参数的唯一集中处（frozen dataclass + 取值域校验，参数本身不带命令行）
  telemetry/           # models / broker / mavsdk_thread / controller（mission+gripper+原点）
  video/               # source（ffmpeg 拉流）/ align（扣链路延时）/ buffer（环形缓冲）
  perception/          # detector（YOLO）/ cropproc（五边形→转正→OCR）/ number / ocr_worker / pipeline
  georef/              # camera（K/dist/R_bc/t_bc）/ project（视线交地平面）/ geo（NED↔WGS84）
  targeting/           # models / cluster（DBSCAN + 众数 + median/max）
  ballistics/          # model（RK4 二次阻力）/ release（判据）/ drops（投放记录）/ fit（参数反演）
  mission/             # states（状态机）/ planner（航线）/ targets（坐标解算）/ runner（主循环）
  record/              # recorder（五个记录文件 + drops.jsonl）/ replay（回放源 + 遥测回填）
tools/                 # 纯库模块：fetch_models / calibrate / fit_ballistics / make_world / dump_api / check_docs
examples/              # 纯库模块：full_mission / replay_flight / calibration_capture / sitl_mission / 其余演示
sim/                   # 仿真模块：赛区世界（生成产物）+ 带下视相机的机型 + airframe + install/run 脚本
tests/                 # 全部离线（GPU 与真实素材的用例标 realdata，默认跳过）
```

## 安装

```bash
uv sync                 # 依赖见 pyproject.toml（索引走清华镜像）
python -m airdrop.run fetch-models    # 把 YOLO / PP-OCR 权重取进 models/（不进 git）
```

## 几个入口

**只有 `airdrop/run.py` 一个命令行入口**（argparse 子命令）：`examples/*.py` 与 `tools/*.py`
都是纯库模块——文件顶部的常量就是默认值，`build_config(**覆盖)` / `main(**kwargs)`
按需传值；选项到配置走关键字覆盖，默认值直接取自模块常量。
`--help` 与 `check-docs` **不加载** torch / cv2 / mavsdk 这些重依赖（惰性导出 + 函数体内导入）。

```bash
python -m airdrop.run --help                 # 列出全部子命令
python -m airdrop.run <子命令> --help         # 该子命令的选项与默认值来源

# 1) 离线迭代识别与坐标：一份回放素材反复跑，不碰硬件
python -m airdrop.run replay --flight flights/<架次> --speed 2

# 2) 完整任务（需要飞控 + 图传 + 标定文件）
python -m airdrop.run full-mission --land-plan routes/land.plan       # --dry-run 不装弹演练
python -m airdrop.run full-mission --no-video --no-preflight          # 无图传/免自检的演练档

# 3) SITL 演练：只走控制链路（合成目标 + 投放只记日志，默认不等起飞）
python -m airdrop.run sitl --target-offset 0,-20,0      # 航线已对准 sim/worlds/cuadc 的赛区

# 4) 标定采集：录一个标准飞行目录给 calibrate 用
python -m airdrop.run calibration-capture
python -m airdrop.run calibrate --flight flights/<那份目录> --strict

# 5) 投放试验反演：量完实际落点后估计弹道参数（只读数据，不接硬件）
python -m airdrop.run fit-ballistics --impacts flights/<那份目录>/impacts.jsonl --mass-kg 0.365

# 6) 重建 CUADC 赛区的 Gazebo 世界（换天井布局与朝向/换轮次/加风/换光照与底图；SITL 启动见 docs/simulation_world.md）
python -m airdrop.run make-world                       # 入库的两个世界（r1 图片靶 / r2 数字靶）
python -m airdrop.run make-world --rounds 2 --seed 3 --wind 5,2,0
```

其余演示：`basic`（遥测/指令）、`hm30-video`（拉流与统计）、`video-sync`（对齐 + 入缓冲 + 读者线程）。

## 配置

**所有参数都在 `airdrop/config.py`**，装配成一个 `Config` 传下去；非法取值在
`Config().validated()` 就报错（而不是在飞行途中才发现）。关键几组：

| 配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `telemetry` | `system_address`、各流速率 | 位置/姿态速率决定帧-遥测内插精度上限 |
| `video` | `url`、`telemetry_lag` | lag 是**链路属性**，标定后回填 |
| `perception` | `mode`（`ocr`/`cls12`）、`model_path`、`device` | 权重必须用 `best2.pt`（另两个是废弃权重） |
| `camera` | `calib_file` | `tools/calibrate.py` 产出的 `camera_calib.json` |
| `ground` | 地面点 GPS | `ground_z` = 原点海拔 − 地面点海拔（平地假设） |
| `targeting` | `eps_m`、`min_samples`、`selection_rule` | `median`/`max` **起飞前二选一** |
| `ballistics`/`drop` | 质量/Cd/面积、`radius_m`、`force_after_pass`、`delay_s` | **质量按实测称重填入**；Cd 用投放试验反演（`tools/fit_ballistics.py`）；`delay_s` 也可反演后回填 |
| `overfly`/`gripper`/`routes` | 航向/高度/段长、gripper 实例、三条航线 | 高度是**相对起飞点**的高度 |
| `mission` | 各状态超时、`abort_action`、起飞/降落项开关 | 见下"状态机" |

## 关键约定（写代码前先看这几条）

1. **"留存"与"实时看"是两条路**：`add_sink()` 注册的回调在采集线程里**逐帧**调用
   （一帧不落），`read()`/`latest()` 是允许丢帧的实时路径。**推理与写入磁盘的输入要来自
   `AlignmentBuffer`**，不要用 `read()` 循环送入数据。
2. **帧-遥测对齐要扣链路延时**：用 `frame.capture_timestamp`（= 收到时间 − `telemetry_lag`）
   取遥测；直接用 `frame.timestamp` 会晚一整个链路延时（10m/s 平飞 ≈1.5m 偏差）。
3. **别持有帧的 ndarray**：ffmpeg 后端的 `frame.image` 是复用缓冲区的视图，跨帧保留
   必须 `copy()`；回看历史走 `buffer.iter_between()`/`wait_new()`。
4. **像素与内参必须同一张图**：检测器已经去畸变时（`DetectorConfig.camera_matrix` 非空），
   georef **不能**再纠正一次——`TargetTracker` 默认按检测结果里的 `undistorted` 标记自动判断。
5. **失败要显式**：解算不出来不编造坐标、投放判据给不出落点则不投放、任务异常进 `ABORT`
   而不是把失败当成成功继续执行。日志/事件里都有原因。

## 状态机

`INIT → RECON → HOLD_PROCESS → OVERFLY → LAND → DONE`，任意环节异常 → `ABORT`。

- `RECON`：上传侦查航线（首航点带**起飞项**）并启动，按 `is_mission_finished()` 判定完成；
  新任务刚上传时的**陈旧 True** 不会被 `MissionMonitor` 采信（要求先见到一次 `False`）。
- `HOLD_PROCESS`：`hold()` 盘旋，等 targeting 出结果；处理完且无结果就提前结束，
  上限 `hold_process_max_s`（默认 10s），超时取当前已有结果。
- `OVERFLY`：结果坐标（无则备用点）→ `[entry, exit]` 飞掠航点 + 降落航线**拼成一条任务**上传，
  激活投放判据（每拍预测落点，≤`radius_m` 即投；越过目标后强制投放；**一次投放即锁存**）。
- `LAND`：沿降落航线返航，任务飞完 → `DONE`。
- `ABORT`：只下**一条**安全动作（`abort_action`：`hold` 默认 / `rtl` / `none`）——
  链路断了这条指令也发不出去，那种情况交给飞控自己的 failsafe。

完整链路的接法见 `examples/full_mission.py`；不装弹演练见 `examples/sitl_mission.py`
（`DryRunController` 拦下投放指令，其余全部真实执行）。

## 标定（三步，离线）

棋盘格**固定**、手持飞机在其上方**平移 + 旋转**，录一个标准飞行目录
（`examples/calibration_capture.py`），然后：

```bash
python -m airdrop.run calibrate --flight flights/<那份目录>   # 常量仍是默认值（--help 里能看到出处）
```

1. **内参**：多视图棋盘格 → `calibrateCamera`（`K`、畸变 + 重投影 RMS）；
2. **画面-遥测时间差**：棋盘格 PnP 角速度 × 飞控角速度互相关 → `telemetry_lag`（回填配置）；
3. **手眼外参**：`AX=XB` 求 `R_bc`；**`t_bc` 以尺量为准**，标定估计值写进报告供对比校验
   （室内手持没有 GPS，平移不可信）。

## 记录与回放

一次飞行 = 一个 `flights/<时间戳>/` 目录：`flight.log`、`telemetry.jsonl`、
`detections.jsonl`、`drops.jsonl`、`events.jsonl`、`frames/*.jpg` + `frames_index.jsonl`、
`config_snapshot.json`。帧**零重编码**（直接写缓冲里的 jpeg），磁盘 ≈4.5MB/s @720p30。

回放按**原时间轴**重放帧与遥测，于是对齐/解算/统计**均无需改动**：
`ReplayVideoSource` 与 `Hm30VideoSource` 是同一套接口，遥测由 `TelemetryPacer`
按帧时刻推进（`load_broker_from_log`）。算法迭代使用 `examples/replay_flight.py`。

## 投放记录与弹道参数反演

**质量用台秤/天平实测**（填 `MEASURED_MASS_KG`）——它不参与反演：反演对质量的分辨力远不如
秤，而且弹道方程里 `Cd`/`m`/`A` 只以 **κ = Cd·A/m** 的形式出现，单靠落点数据分不开这三者。
**数据真正识别到的就是 κ 一个数**（工具会打印它，以及按实测质量换算出的 `Cd·A`）。

每次投放的**瞬间状态**都当场写入磁盘——位置、速度、**姿态**（欧拉角与四元数各一份）、风、
离地高度、NED 原点、目标点，外加判据当时的前推位置与预测落点：

```jsonc
// flights/<时间戳>/drops.jsonl（一行一次投放；数字不四舍五入）
{"index": 1, "timestamp": 1756... , "position_ned": [120.0, -35.0, -20.0],
 "velocity_ned": [0.0, 18.0, 0.0], "attitude_deg": {"roll": 4.0, "pitch": -3.0, "yaw": 90.0},
 "wind_ned": [2.0, -1.0, 0.0], "ground_z": 0.0, "delay_s": 0.08,
 "origin": {"lon_deg": 8.0, "lat_deg": 47.0, "alt_m": 500.0},
 "predicted_impact_ned": [...], "predicted_error_m": 0.45, "ballistics": {...}}
```

落地后量出实际落点，写进**同一个目录**的 `impacts.jsonl`（经纬度或 NED 都行；
第一次跑工具会生成待填模板），然后：

```bash
python -m airdrop.run fit-ballistics --impacts <落点文件> --mass-kg <称重值>
```

工具会打印逐次投放的"判据预测 / 反演后预测 / 实测"对照表（含沿航迹、垂直航迹分解），
做最小二乘反演，并给出**识别量 κ = Cd·A/m、不确定度 σ、参数相关性、条件数、留一交叉验证**，
最后输出可直接粘回配置的 `BallisticsConfig(...)` 与 `ballistics_fit.json` 报告。

> ⚠ **先看结论行再引用数字**：Cd 与迎风面积只以 `Cd·A` 的乘积出现（同时拟合必然退化）、
> 释放延迟与阻力沿航迹方向高度相关（同高同速投放分不开）；常数偏移（测量点/目标点坐标错）
> 任何参数都吸收不了。这三种情况工具会**拒绝给结论**并说明原因——详见
> [`docs/ballistics_fit.md`](docs/ballistics_fit.md)（P12 专题笔记）。
> 另外：**质量填错多少，Cd 就跟着错多少**（κ 不变）——换配重（形状不变 ⇒ `Cd·A` 不变）时按
> `κ' = Cd·A/m'` 重算即可，不必重做试验。

## HM30 图传

HM30 地面端把机载以太网 `192.168.144.0/24` **透明桥接**到 LAN/内置 WiFi：网线插地面端
LAN 口、电脑网卡配 `192.168.144.20/24`、能 ping 通 `192.168.144.25` 即可；
SIYI 相机默认 `rtsp://192.168.144.25:8554/main.264`。地址**不猜也不探测**，写错就把
ffmpeg 的错误原样抛出来（`stats.last_error`）；手动核对一条命令即可：

```bash
ffmpeg -rtsp_transport udp -i rtsp://192.168.144.25:8554/main.264 -frames:v 1 -f null -
```

图传**只有 ffmpeg 一个后端**，不要再引入 `cv2.VideoCapture`。真实理由是：(1) 管道反压让
接收侧保持浅流水（实测 0 帧积压，而 cv2 后端有 17 帧 ≈0.57s 的内部排队）；(2) `read()`
是 C 层不可中断调用，卡住只能杀进程；(3) 断流超时靠 ffmpeg 的 `-timeout`（微秒）。
⚠ 早期文档里"OpenCV 5.0 移除了 `OPENCV_FFMPEG_CAPTURE_OPTIONS`"的说法**是错的**
（那个字符串在插件 DLL 里，grep 错了文件）——完整复盘与逐项实测表见
[`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md)。

## 测试与代码规范

```bash
./.venv/Scripts/python.exe -m pytest                 # 全部（附覆盖率，当前 ≈85%；默认已排除 realdata/sitl）
./.venv/Scripts/python.exe -m pytest -m "not realdata and not sitl and not stream"  # 再跳过要起 ffmpeg 的用例
./.venv/Scripts/python.exe -m pytest -m realdata     # GPU + 旧版 实战素材（分钟级）
./.venv/Scripts/python.exe -m pytest -m sitl         # WSL 里已起 PX4 SITL + 图传（分钟级）
./.venv/Scripts/python.exe -m pytest -k mission -v   # 只跑某个主题

./.venv/Scripts/ruff.exe check .                     # 代码规范（配置在 pyproject.toml）
./.venv/Scripts/ruff.exe check . --fix               # 自动修可修项
./.venv/Scripts/ruff.exe format .                    # 统一格式
./.venv/Scripts/pyright.exe                          # 类型检查（同上）
```

- 套件**全部离线**：不需要飞控/图传；`stream` 标记的用例会起本地 ffmpeg 并占 UDP 51234。
- 端到端在 `tests/test_e2e.py`：回放素材 → 感知 → 坐标 → 统计 → 航线，其中
  `-m realdata` 那条用**真视频 + 真 YOLO + 真 OCR**（位姿是合成的，见用例 docstring）。
- 投放反演在 `tests/test_fit.py`：用**已知真值**的弹道模型造出"实测落点"再反演，
  并钉住三种必须拒绝的退化情形、测量文件的各种写法、以及工具的现场流程。
- **`ruff check`、`ruff format --check`、`pyright` 三者都保持零告警**（都在 `dev` 依赖组里）。
  ruff 选的是贴合本项目的规则集（中文注释里的全角标点走 `allowed-confusables` 精确放行，
  不是把 `RUF001/002/003` 关掉）；pyright 用 `basic`，`airdrop/` 全绿。
  全库已按 `[tool.ruff.format]` 格式化过一遍，改完代码跑一次 `ruff format .` 即可。
- ⚠ 受限沙箱里 pytest 的 `tmp_path` 不可用（`PermissionError`），12 个 `test_recorder.py`
  + 8 个 `test_georef.py` 会挂在 fixture 上——那是沙箱限制、不是代码问题
  （`tests/test_fit.py` 因此自带工作区内的 `workdir` fixture）。

## 已知边界与风险（不应视为已完成）

## 安全提示

- 任何会驱动飞机的操作（arm / takeoff / mission upload / 投放）**只在拆桨或 SITL 环境里试**；
  真机演练请用 `DRY_RUN = True`（投放指令只记日志）并确认 `abort_action` 符合空域要求。
- 投放判据是**建议**而不是保证：它按弹道与风预测落点，参数没标定过就只是"按配置在飞"。
