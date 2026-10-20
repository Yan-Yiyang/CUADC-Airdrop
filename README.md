# CUADC固定翼无人机侦查与打击控制项目

**版本 3.0.0**。一条完整链路已经跑通并有离线测试覆盖：

```
起飞 → 预设侦查航线（约 1 分钟）→ 盘旋 hold
     → 感知（YOLO 检测 + OCR 读编号）→ 像素→NED 坐标解算 → 目标统计（聚类 + 选唯一）
     ├─ 有结果 → 生成飞掠航点（航向/高度/段长按配置）+ 降落航线，合并上传
     └─ 无结果 → 以备用点为目标生成同样航点
     → 飞掠段：预测沿向过零且横向合格时优先投；过零后进入 5m 目标圈可近靶投放，沿向超过 5m 可兜底
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
| `airdrop.ballistics` | 二次阻力 RK4 落点预测 + 沿向过零正常投放（横向 `radius_m` 容限）+ 过零后近靶/距离兜底 + 锁存 + **投放记录与参数反演** | `test_ballistics.py`、`test_fit.py` |
| `airdrop.mission` | 状态机 + 飞掠/降落航线拼接 + 候选类→坐标 + 主循环 | `test_mission.py`、`test_e2e.py` |
| `airdrop.record` | 飞行记录（五个记录文件 + 投放记录）+ 按原时间轴回放 | `test_recorder/replay.py` |
| `airdrop_calibration.calibrate` | 三步标定：内参 → 画面/遥测时间差 → 手眼外参 | `test_calibrate.py` |
| `tools.fit_ballistics` | 投放试验反演弹道参数（最小二乘 + 可辨识性诊断） | `test_fit.py` |

> **第一次上手先看 [`docs/handbook.md`](docs/handbook.md)**（项目手册：目录总览、各模块调用方法与
> 最小示例、参数总表、输入输出字段、使用流程、代码审查速查、排障表）。
> AI 智能体的执行约束与按需阅读索引见 **[`AGENTS.md`](AGENTS.md)**；用户使用说明见项目手册与专题文档。
> 长专题（标定/OpenCV、图传/ffmpeg、感知/OCR、投放反演）在 **[`docs/`](docs/)** 下的专题笔记，
> 按任务需要查阅。README 说明项目使用方法。

## 目录结构

```text
airdrop/
  run.py               # **主仓命令行入口**（argparse 子命令 → 关键字覆盖到各入口的 build_config）
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
tools/                 # 主仓工具：fetch_models / fit_ballistics / dump_api / check_docs
airdrop/runtime/       # 同一任务入口、参数化阶段、组件装配及 real/SITL 适配
vendor/                # 固定版本的 airdrop-sim 与 airdrop-calibration 子模块
tests/                 # 全部离线（GPU 与真实素材的用例标 realdata，默认跳过）
```

## 安装

可选工具已拆成两个独立仓库，初始化、独立运行和本地镜像方式见 [工具仓库说明](docs/tool_repositories.md)。主任务运行与帮助无需工具子模块。

```bash
uv sync                 # 依赖见 pyproject.toml（索引走清华镜像）
python -m airdrop.run fetch-models    # 把 YOLO / PP-OCR 权重取进 models/（不进 git）
```

## 任务运行

任务只有真实运行与 SITL 两种环境，使用同一个入口和状态机。接收视频、识别、保存记录、侦查后的投放阶段、真实挂架动作及自动降落分别控制。参数语义与组合约束见 [运行说明](docs/runtime.md)。

```bash
python -m airdrop.run mission --help
python -m airdrop.run mission --mode real
python -m airdrop.run mission --mode sitl
# 只采集，不控制飞机、不载入识别模型
python -m airdrop.run mission --mode real --no-mission-control --capture --no-recognize --record
# 侦查与识别完成后盘旋结束，不规划投放阶段
python -m airdrop.run mission --mode real --no-drop-stage
# 运行飞掠投放判据，飞出出口后盘旋，不上传降落段
python -m airdrop.run mission --mode real --no-auto-land
```

真实环境默认由操作手上传并启动侦查任务；`--auto-recon` 才由程序上传。真实环境不会自动解锁。`--real-release` 显式启用真实挂架动作；默认模拟该动作，仍运行弹道判据并记录结果。SITL 不允许真实挂架动作。回放是离线分析工具，不构成第三种任务环境：

```bash
python -m airdrop.run replay --flight flights/<架次>
python -m airdrop.run calibrate --flight flights/<架次> --strict
python -m airdrop.run fit-ballistics --help
```

正式任务先按场地设置侦查航线、备用点、飞掠参数和降落航线，并使用对应相机标定。只采集模式不要求识别模型或相机标定；启用识别时必须接收视频。SIH/QGC 已验证只读任务同步、任务替换、断连重连、进度完成及仿真落地；Gazebo 相机到被动采集、录制和合成输出已在受控配置下联通，完整感知/投放闭环和真实设备仍待验收。SITL 的世界、相机和真值来自可选仿真工具包，使用方法见 [仿真说明](docs/simulation_world.md)。

可选的 QGC 合成视频由 `mission` / `replay` 的 `--qgc-video` 开启；`qgc-video-test --seconds 10` 可在不连接飞控、不加载模型时测试接收。默认 UDP 端口为 5602，画面、封装与资源隔离见 [QGC 视频说明](docs/qgc_video.md)。

## 配置

**所有参数都在 `airdrop/config.py`**，装配成一个 `Config` 传下去；非法取值在
`Config().validated()` 就报错（而不是在飞行途中才发现）。关键几组：

| 配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `telemetry` | `system_address`、各流速率 | 位置/姿态速率决定帧-遥测内插精度上限 |
| `video` | `url`、`telemetry_lag` | lag 是**链路属性**，标定后回填 |
| `perception` | `mode`（`ocr`/`cls12`）、`model_path`、`device` | 权重必须用 `best2.pt`（另两个是废弃权重） |
| `camera` | `calib_file` | `vendor/airdrop-calibration/airdrop_calibration/calibrate.py` 产出的 `camera_calib.json` |
| `ground` | 地面点 GPS | `ground_z` = 原点海拔 − 地面点海拔（平地假设） |
| `targeting` | `eps_m`、`min_samples`、`selection_rule` | `median`/`max` **起飞前二选一** |
| `ballistics`/`drop` | 质量/Cd/面积、沿航向过零、横向 `radius_m`、过零近靶 `post_cross_radius_m`、距离兜底 `force_after_pass`/`fallback_after_m`、`delay_s` | **质量按实测称重填入**；Cd 用投放试验反演（`tools/fit_ballistics.py`）；`delay_s` 也可反演后回填 |
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

`INIT → PREFLIGHT → WAIT_AIRBORNE → RECON → HOLD_PROCESS → OVERFLY → LAND → DONE`，
关闭投放阶段可从 `HOLD_PROCESS` 盘旋结束；关闭自动降落可从 `OVERFLY` 飞完出口、盘旋结束。任意环节异常 → `ABORT`。

- `INIT`：等遥测位置 + NED 原点（**不上传任何任务**）；`init_max_s` 超时 → `ABORT`。
- `PREFLIGHT`：起飞前自检（载入 detector/ocr/camera + 视频自检）；任一项失败 → `ABORT`。
- `WAIT_AIRBORNE`：什么都不下发，等飞机真的在空中（`in_air` 为主，取不到时用相对高度
  ≥ `airborne_alt_m` 兜底）；`airborne_timeout_s` 超时 → `ABORT`。
- `RECON`：正式档（`recon_upload="operator"`）**不上传**，等操作手在 QGC 启动侦查航线并监视
  进度；`"auto"` 档才由本包上传并启动（SITL 适配器先完成仿真起飞，再运行共同的侦查上传流程）。完成判定读
  飞控侧 `mission_raw` 进度（`current == total`）——新任务刚上传时的**陈旧 True** 不会被
  `MissionMonitor` 采信（要求先见到一次 `False`）。
- `HOLD_PROCESS`：`hold()` 盘旋，等 targeting 出结果；处理完且无结果就提前结束，
  上限 `hold_process_max_s`（默认 10s），超时取当前已有结果。
- `OVERFLY`：结果坐标（无则备用点）→ `[entry, exit]` 飞掠航点 + 降落航线**拼成一条任务**上传，
  激活投放判据（连续预测沿航向过零且横向误差≤`radius_m` 优先投；本轮过零后，当前预测落点进入 `post_cross_radius_m` 水平半径可近靶投放，或沿向误差>`fallback_after_m` 时距离兜底；**一次投放即锁存**）。
- `LAND`：沿降落航线返航，任务飞完 → `DONE`。
- `ABORT`：只下**一条**安全动作（`abort_action`：`hold` 默认 / `rtl` / `none`）——
  链路断了这条指令也发不出去，那种情况交给飞控自己的 failsafe。

完整链路的接法见 `mission --mode real`；不装弹演练见 `mission --mode sitl`
（`DryRunController` 拦下投放指令，其余全部真实执行）。

## 标定（三步，离线）

棋盘格**固定**、手持飞机在其上方**平移 + 旋转**，录一个标准飞行目录
（`mission --mode real --no-mission-control --no-recognize --record`），然后：

```bash
python -m airdrop.run calibrate --flight flights/<那份目录>   # 常量仍是默认值（--help 里能看到出处）
```

1. **内参**：多视图棋盘格 → `calibrateCamera`（`K`、畸变 + 重投影 RMS）；
2. **画面-遥测时间差**：棋盘格 PnP 角速度 × 飞控角速度互相关 → `telemetry_lag`（回填配置）；
3. **手眼外参**：`AX=XB` 求 `R_bc`；**`t_bc` 以尺量为准**，标定估计值写进报告供对比校验
   （室内手持没有 GPS，平移不可信）。

> 目前仅有**合成链路**验证；真实素材复测与 `lag_s` 现场值尚未验证。

## 记录与回放

一次飞行 = 一个 `flights/<时间戳>/` 目录：`flight.log`、`telemetry.jsonl`、
`detections.jsonl`、`drops.jsonl`、`events.jsonl`、`frames/*.jpg` + `frames_index.jsonl`、
`config_snapshot.json`。帧**零重编码**（直接写缓冲里的 jpeg）。

回放按**原时间轴**重放帧与遥测，于是对齐/解算/统计**均无需改动**：
`ReplayVideoSource` 与 `Hm30VideoSource` 是同一套接口，遥测由 `TelemetryPacer`
按帧时刻推进（`load_broker_from_log`）。算法迭代使用 `python -m airdrop.run replay`。

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

图传**只有 ffmpeg 一个后端**，不要再引入 `cv2.VideoCapture`：采集由独立进程负责，断流超时
通过 ffmpeg 的 `-timeout`（微秒）控制。完整链路说明见 [`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md)。

## 测试与代码规范

```bash
./.venv/Scripts/python.exe -m pytest --test-tier fast       # 默认：不启动 ffmpeg/WSL/GPU/飞控
./.venv/Scripts/python.exe -m pytest --test-tier offline    # 加入本地文件与跨仓集成
./.venv/Scripts/python.exe -m pytest --test-tier platform   # 加入 ffmpeg/UDP 与 WSL
./.venv/Scripts/python.exe -m pytest --test-tier all        # 加入 GPU 真实素材与 SITL
./.venv/Scripts/python.exe -m pytest --test-tier all -m realdata  # -m 是 tier 上的额外筛选
./.venv/Scripts/python.exe -m pytest --test-tier all --cov=airdrop --cov-branch --cov-report=term-missing:skip-covered
./.venv/Scripts/python.exe -m pytest -k mission -v   # 只跑某个主题

./.venv/Scripts/ruff.exe check .                     # 代码规范（配置在 pyproject.toml）
./.venv/Scripts/ruff.exe check . --fix               # 自动修可修项
./.venv/Scripts/ruff.exe format .                    # 统一格式
./.venv/Scripts/pyright.exe                          # 类型检查（同上）
```

## 已知边界与风险（不应视为已完成）

- **弹道参数**：**质量按称重实测填入**（默认值仅为占位）；
  `drag_coefficient` 仍是占位值（0.6），反演工具与流程已就绪（`tools/fit_ballistics.py`），
  但工具尚未通过真实投放数据验证——首次采集后按工具输出判断（可能因为"分不开"而要求
  你改变投放剖面）。飞掠段长/航向同样**待实验验证**。
- **标定的真实素材复测未做**；`t_bc` 以尺量为准，`telemetry_lag` 的实测值需一次真实采集。
- SITL 演练使用合成目标，不能验证真实感知或实机投放精度；完整任务在侦查前要求确认飞机已离地。
- **CUADC 赛区世界已入库**（`vendor/airdrop-sim/airdrop_sim/assets/worlds/cuadc/`：起降区 + A/B 两个 60×60m 目标区（分列起飞线两端，各距起降区约 200m）+ 8 座五边形天井（**位置与朝向按规则随机**，`--seed` 复现，默认 seed 0 = 入库布局；规则里只是示意的 4m/6m 打击圈与"天井箭头"默认不画；太阳位置可换 `--sun 方位,仰角`，或 `--sun-random` 由 seed 随机；目标区底色按种子取常见路面色、比赛区域外随机铺航拍干扰底图 `--aerial-patches`，都可以重新生成以模拟不同光照/背景），r1 图片靶标 / r2 数字靶标；`python -m airdrop.run make-world` 重生成）。世界的几何与贴图方向做过离屏渲染核对，演练航线（`mission --mode sitl`，对准入库布局的中位数天井）与降落航线（`routes/land.plan`）已对准它；SITL 默认机型是带下视相机的 `gz_rc_cessna_down_cam`。**未验证**：把 Gazebo 相机接进感知闭环、世界与真机场地的差异。启动步骤见 `docs/simulation_world.md`。
- PX4 侧需要自己配置：gripper 输出、起飞项的航点动作、任务结束后的行为。
- `cls12` 的 12 类权重目前不存在（`best2.pt` 是单类），该模式当前只能跑出编号 1。
- **无 CI**：pytest / ruff / pyright 都只在本地跑。

## 安全提示

- 任何会驱动飞机的操作（arm / takeoff / mission upload / 投放）**只在拆桨或 SITL 环境里试**；
  真机演练请用 `DRY_RUN = True`（投放指令只记日志）并确认 `abort_action` 符合空域要求。
- 投放判据是**建议**而不是保证：它按弹道与风预测落点，参数没标定过就只是"按配置在飞"。
