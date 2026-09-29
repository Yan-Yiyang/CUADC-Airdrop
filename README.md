# CUADC固定翼无人机侦查与打击控制项目

**版本 0.1**。完整任务链路已实现并有离线测试覆盖（不需要飞控、图传或 GPU 也能跑通测试）。

> ⚠ **验证状态：目前只做过仿真验证**（PX4 SITL + Gazebo 赛区世界，完整流程"起飞 → 侦查 →
> 盘旋出目标 → 飞掠投放 → 降落"已跑通；算法链路有离线测试覆盖）。**尚未做真机飞行验证**：
> 弹道参数是占位值、相机标定与链路延时随设备变化，都要按你自己的设备和场地重新标定。
> 真机测试请务必谨慎——先拆桨/系留，再逐步放开，全程有人掌握接管手段（见"安全提示"
> 与"已知边界与风险"）。

> **授权**：Apache License 2.0，见 [`LICENSE`](LICENSE)。
> **署名**：参加竞赛、提交作品或公开发布相关成果时，请按 [`NOTICE`](NOTICE) 注明使用了本项目。

## 设计背景

本项目面向无人机竞赛的"侦察与打击"任务，设计约束是**不使用机载计算机**：飞行器上仅
搭载相机、视频下传链路、飞控与投放机构；目标检测、编号识别、像素→坐标解算、目标统计
与弹道预测全部在地面计算机完成。该约束带来两条实现要求：

- **视频逐帧留存**：目标可见时间可能极短，丢失任一帧都可能丢失目标；允许处理延时，
  不允许采集丢帧（见"关键约定"）。
- **帧-遥测按拍摄时刻对齐**：解算依赖画面与遥测的严格时间对应，须扣除链路固定延时后
  再查询遥测（见"关键约定"）。

任务按下列流程执行：

```text
起飞 → 侦查航线（约 1 分钟）→ 盘旋保持
     → 感知（YOLO 检测 + OCR 读编号）→ 像素→NED 坐标解算 → 目标统计（聚类 + 选唯一）
     ├─ 有结果 → 生成飞掠航点（航向/高度/段长按配置）+ 降落航线，合并上传
     └─ 无结果 → 以备用点为目标生成同样航点
     → 飞掠段：弹道实时预测落点，距目标 ≤R 触发投放；飞过目标未投则强制投放
     → 沿降落航线返航降落
全程：飞行目录（日志/遥测/检测/事件/投放/视频帧）写入磁盘，飞后可回放迭代
投放后：量出实际落点 → 反演弹道参数（阻力系数/释放延迟），下一架次用标定过的模型
```

状态机：`PREFLIGHT → WAIT_AIRBORNE → RECON → HOLD_PROCESS → OVERFLY → LAND → DONE`；
任意环节异常 → `ABORT`，只下**一条**安全动作（`abort_action`：`hold` / `rtl` / `none`）。

| 模块 | 干什么 | 离线用例 |
| --- | --- | --- |
| `airdrop.telemetry` | MAVSDK 线程 + 遥测代理（快照/历史/内插）+ 任务级控制器 | `test_telemetry.py` |
| `airdrop.video` | ffmpeg 拉流（一帧不丢）、帧-遥测对齐、环形缓冲 | `test_video/alignment/buffer.py` |
| `airdrop.perception` | YOLO 检测 + 五边形转正 + RapidOCR 读编号（或 12 类直出） | `test_perception.py` |
| `airdrop.georef` | 像素 → NED（视线与地面求交）→ WGS84 | `test_georef.py` |
| `airdrop.targeting` | DBSCAN 聚类 + 类内编号众数 + 跨类 median/max 选唯一 | `test_targeting.py` |
| `airdrop.ballistics` | 二次阻力 RK4 落点预测 + 投放判据（含越过目标后的强制投放与锁存）+ 投放记录与参数反演 | `test_ballistics.py`、`test_fit.py` |
| `airdrop.mission` | 起飞前自检 + 状态机 + 飞掠/降落航线拼接 + 候选类→坐标 + 主循环 | `test_mission.py`、`test_e2e.py` |
| `airdrop.record` | 飞行记录（五个记录文件 + 投放记录）+ 按原时间轴回放 | `test_recorder/replay.py` |
| `tools.calibrate` | 三步标定：内参 → 画面/遥测时间差 → 手眼外参 | `test_calibrate.py` |
| `tools.sitl_recon` | SITL 侦查精度测试：真感知 + 真坐标，量"解算坐标 vs 目标真值" | `test_sitl_recon.py`（可选） |
| `tools.fit_ballistics` | 投放试验反演弹道参数（最小二乘 + 可辨识性诊断） | `test_fit.py` |

> **第一次上手先看 [`docs/handbook.md`](docs/handbook.md)**（项目手册：目录总览、各模块调用方法与
> 最小示例、参数总表、输入输出字段、使用流程、排障表）。长专题在 [`docs/`](docs/) 下：
> 标定（OpenCV 实战记录）、图传（ffmpeg 选型与实测）、感知/OCR、仿真世界、投放反演。
> 开发约定与实测结论汇总在 [`AGENTS.md`](AGENTS.md)。

## 目录结构

```text
airdrop/
  run.py               # 唯一的命令行入口（argparse 子命令 → 关键字覆盖到各入口的 build_config）
  _lazy.py             # PEP 562 惰性导出：import airdrop / --help 不加载 cv2、mavsdk、torch
  config.py            # 全部参数的唯一集中处（frozen dataclass + 取值域校验，参数本身不带命令行）
  telemetry/           # models / broker / mavsdk_thread / controller（mission + gripper + 原点）
  video/               # source（ffmpeg 拉流）/ align（扣链路延时）/ buffer（环形缓冲）
  perception/          # detector（YOLO）/ cropproc（五边形→转正→OCR）/ number / ocr_worker / pipeline
  georef/              # camera（K/dist/R_bc/t_bc）/ project（视线交地平面）/ geo（NED↔WGS84）
  targeting/           # models / cluster（DBSCAN + 众数 + median/max）
  ballistics/          # model（RK4 二次阻力）/ release（判据）/ drops（投放记录）/ fit（参数反演）
  mission/             # states（状态机）/ planner（航线）/ targets（坐标解算）/ runner（主循环）
  record/              # recorder（五个记录文件 + drops.jsonl）/ replay（回放源 + 遥测回填）
  preflight.py         # 起飞前自检（载入模型/OCR/相机 + 视频自检）
tools/                 # 纯库模块：fetch_models / calibrate / fit_ballistics / sitl_recon / make_world / dump_api / check_docs
examples/              # 纯库模块：full_mission / replay_flight / calibration_capture / sitl_mission / 其余演示
sim/                   # 仿真模块：赛区世界（生成产物）+ 带下视相机的机型 + airframe + install/run 脚本
routes/                # 操作手在 QGC 里画好的航线（.plan）：降落段 land.plan
tests/                 # 全部离线（需要 GPU 或真实素材的用例标 realdata，默认跳过）
```

## 安装

需要 **Python ≥ 3.14**（依赖见 [`pyproject.toml`](pyproject.toml)）+ 一块支持 CUDA 的 NVIDIA
显卡（检测与 OCR 默认走 GPU；CPU 也能跑，只是慢）。

```bash
uv sync                 # 建虚拟环境并装依赖
python -m airdrop.run fetch-models --source-dir <YOLO 权重目录> [--source-ocr-dir <OCR 权重目录>]
                        # 把 YOLO / PP-OCR 权重复制进 models/（权重不进 git；只复制、不联网）
```

权重文件名与目录结构见 `tools/fetch_models.py` 的说明；OCR 的字典文件（`models/ppocr/*.txt`）
已随仓库提供。

## 快速开始

**命令行入口只有 `airdrop/run.py`**（argparse 子命令）：`examples/*.py` 与 `tools/*.py` 都是
纯库模块——文件顶部的常量就是默认值，`build_config(**覆盖)` / `main(**kwargs)` 按需传值，
命令行选项按关键字覆盖。`--help` 不加载 torch / cv2 / mavsdk 这些重依赖。

```bash
python -m airdrop.run --help                 # 全部子命令
python -m airdrop.run <子命令> --help         # 该子命令的选项与默认值出处

# 1) 离线迭代识别与坐标：拿一份飞行记录反复跑，不碰硬件
python -m airdrop.run replay --flight flights/<架次> --speed 2
#    飞行记录裁剪过（缺帧）时要 --no-strict；要用录素材时同一份标定就带 --calib

# 2) 完整任务（需要飞控 + 图传 + 标定文件）
python -m airdrop.run full-mission --land-plan routes/land.plan   # --dry-run 不装弹演练

# 3) SITL 演练：只走控制链路（合成目标，投放只记日志）
python -m airdrop.run sitl --target-offset 0,-20,0

# 4) SITL 侦查精度测试：真感知 + 真坐标，量"解算坐标 vs 目标真值"（验收 < 2 m）
HEADLESS=1 bash sim/run_sitl.sh r2                 # 无头起仿真（见下"注意"）
python -m airdrop.run sitl-recon

# 5) 标定采集与三步标定（离线）
python -m airdrop.run calibration-capture
python -m airdrop.run calibrate --flight flights/<架次> --strict

# 6) 投放试验反演弹道参数（量完实际落点后，只读数据、不接硬件）
python -m airdrop.run fit-ballistics --impacts <落点文件> --mass-kg <称重值>

# 7) 重建 CUADC 赛区的 Gazebo 世界（换天井布局/轮次/风/光照）
python -m airdrop.run make-world
python -m airdrop.run make-world --rounds 2 --seed 3 --wind 5,2,0
```

其余演示：`basic`（遥测与指令）、`hm30-video`（拉流与统计）、`video-sync`（对齐 + 入缓冲）。

## 配置

**所有参数都在 `airdrop/config.py`**，装配成一个 `Config` 传下去；非法取值在
`Config().validated()` 就报错（而不是在飞行途中才发现）。关键几组：

| 配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `telemetry` | `system_address`、各流速率 | 位置/姿态速率决定帧-遥测内插精度上限 |
| `video` | `url`、`telemetry_lag` | lag 是**链路属性**（真机 HM30 ≈0.15 s、SITL ≈0.5 s，各自标定后回填） |
| `perception` | `mode`（`ocr`/`cls12`）、`model_path`、`device` | 权重用 `best2.pt`；`cls12` 需要 12 类权重（当前不提供） |
| `camera` | `calib_file` | `tools/calibrate.py` 产出的 `camera_calib.json` |
| `ground` | 地面点 GPS | `ground_z` = 原点海拔 − 地面点海拔（平地假设） |
| `targeting` | `eps_m`、`min_samples`、`selection_rule` | `median`/`max` **起飞前二选一** |
| `ballistics`/`drop` | 质量/Cd/面积、`radius_m`、`force_after_pass`、`delay_s` | **质量按实测称重填入**；Cd 与 `delay_s` 用投放试验反演后回填 |
| `overfly`/`gripper`/`routes` | 航向/高度/段长、gripper 实例、航线 | 航线高度是**相对起飞点**的高度 |
| `mission` | 各状态超时、`abort_action`、起飞/降落项开关 | 见"任务流程" |

## 关键约定

1. **"留存"与"实时看"是两条路**：`add_sink()` 注册的回调在采集线程里**逐帧**调用（一帧不落），
   `read()`/`latest()` 是允许丢帧的实时路径。**推理与写入磁盘的输入要来自 `AlignmentBuffer`**，
   不要用 `read()` 循环送数据——那等于把丢帧重新引回来。
2. **帧-遥测对齐要扣链路延时**：用 `frame.capture_timestamp`（= 收到时间 − `telemetry_lag`）取
   遥测；直接用 `frame.timestamp` 会晚一整个链路延时（10 m/s 平飞 ≈1.5 m 偏差）。
3. **别持有帧的 ndarray**：ffmpeg 后端的 `frame.image` 是复用缓冲区的视图，跨帧保留必须
   `copy()`；回看历史走 `buffer.iter_between()` / `wait_new()`。
4. **像素与内参必须同一张图**：检测器已经去畸变时，坐标解算**不能**再纠正一次——程序按
   检测结果里的标记自动判断。
5. **失败要显式**：解算不出来不编造坐标、投放判据给不出落点则不投放、任务异常进 `ABORT`
   而不是当成功继续跑；日志与事件里都写了原因。

## 记录与回放

一次飞行 = 一个 `flights/<架次>/` 目录：`flight.log`、`telemetry.jsonl`、`detections.jsonl`、
`drops.jsonl`、`events.jsonl`、`frames/*.jpg` + `frames_index.jsonl`、`config_snapshot.json`。
帧**零重编码**（直接写缓冲里的 jpeg），磁盘约 4.5 MB/s @720p30。

回放按**原时间轴**重放帧与遥测，于是对齐/解算/统计均无需改动：`ReplayVideoSource` 与实飞
拉流源是同一套接口，遥测由 `TelemetryPacer` 按帧时刻推进。算法迭代用
`python -m airdrop.run replay`（见"快速开始"）。

## 投放记录与弹道参数反演

**质量用台秤实测**——它不参与反演：弹道方程里 `Cd`/`m`/`A` 只以 **κ = Cd·A/m** 的形式出现，
单靠落点数据分不开这三者（工具会打印 κ，以及按实测质量换算出的 `Cd·A`）。

每次投放的**瞬间状态**都当场写入 `drops.jsonl`：位置、速度、姿态（欧拉角与四元数各一份）、
风、离地高度、NED 原点、目标点，外加判据当时的前推位置与预测落点。落地后量出实际落点，
写进**同一个目录**的 `impacts.jsonl`（第一次跑工具会生成待填模板），然后：

```bash
python -m airdrop.run fit-ballistics --impacts <落点文件> --mass-kg <称重值>
```

工具打印逐次投放的"判据预测 / 反演后预测 / 实测"对照表，做最小二乘反演，并给出识别量
κ、不确定度 σ、参数相关性、条件数、留一交叉验证，最后输出可直接粘回配置的
`BallisticsConfig(...)` 与 `ballistics_fit.json` 报告。

> ⚠ **先看结论行再引用数字**：Cd 与迎风面积只以乘积出现（同时拟合必然退化）、释放延迟与
> 阻力沿航迹方向高度相关（同高同速投放分不开）、常数偏移（坐标测量错）任何参数都吸收不了
> ——这三种情况工具会**拒绝给结论**并说明原因。详见 [`docs/ballistics_fit.md`](docs/ballistics_fit.md)。
> 另注：**质量填错多少，Cd 就跟着错多少**（κ 不变）——换配重时按 `κ' = Cd·A/m'` 重算即可。

## HM30 图传

HM30 地面端把机载以太网 `192.168.144.0/24` **透明桥接**到 LAN/内置 WiFi：网线插地面端
LAN 口、电脑网卡配 `192.168.144.20/24`、能 ping 通 `192.168.144.25` 即可；SIYI 相机默认
`rtsp://192.168.144.25:8554/main.264`。地址**不猜也不探测**，写错就把 ffmpeg 的错误原样抛出来
（`stats.last_error`）；手动核对一条命令即可：

```bash
ffmpeg -rtsp_transport udp -i rtsp://192.168.144.25:8554/main.264 -frames:v 1 -f null -
```

图传**只有 ffmpeg 一个后端**（管道反压让接收侧保持浅流水、`read()` 卡住时不会变成只能杀进程、
断流超时靠 ffmpeg 的 `-timeout`）。选型过程与实测表见 [`docs/video_hm30_ffmpeg.md`](docs/video_hm30_ffmpeg.md)。

## 测试与代码规范

```bash
./.venv/Scripts/python.exe -m pytest                 # 全部（附覆盖率；默认已排除 realdata/sitl）
./.venv/Scripts/python.exe -m pytest -m "not realdata and not sitl and not stream"  # 再跳过要起 ffmpeg 的用例
./.venv/Scripts/python.exe -m pytest -m realdata     # GPU 与真实素材（分钟级）
./.venv/Scripts/python.exe -m pytest -m sitl         # WSL 里已起 PX4 SITL 与图传（分钟级）
./.venv/Scripts/python.exe -m pytest -k mission -v   # 只跑某个主题

./.venv/Scripts/ruff.exe check .                     # 代码规范（配置在 pyproject.toml）
./.venv/Scripts/ruff.exe format .                    # 统一格式
./.venv/Scripts/pyright.exe                          # 类型检查
```

⚠ 命令行的 `-m` 是**覆盖** `pyproject.toml` 里的默认表达式，而不是追加：只写 `-m "not stream"` 会把
`realdata` 与 `sitl` 一起放进来（后者要 SITL 真在跑，否则只是白等一轮探活）。

- 套件**全部离线**：不需要飞控与图传；`stream` 标记的用例会起本地 ffmpeg 并占用 UDP 51234，
  `realdata` 标记的用例需要 GPU 与真实素材（默认跳过），`sitl` 标记的用例要 WSL 里已起 PX4 SITL
  与图传（同样默认跳过，`-m sitl` 才跑）。
- 端到端在 `tests/test_e2e.py`：回放素材 → 感知 → 坐标 → 统计 → 航线。
- **`ruff check`、`ruff format --check`、`pyright` 三者都保持零告警**（都在 `dev` 依赖组里）。

## 已知边界与风险（不应视为已完成）

- **验证范围：只做过仿真验证**（SITL + 离线测试），**没有真机飞行数据**。下面这些必须由
  你的实测补齐；在补齐之前，任何真机动作都应当作"未验证"对待。
- **弹道参数**：质量必须按称重实测填入（仓库里的默认值只是跑通链路的占位值），
  `drag_coefficient` 同样是占位值——反演工具与流程已就绪，需要你自己的投放试验数据；
  飞掠段长/航向也**需按你的空域实验确定**。
- **标定要用你自己的设备做**：内参、画面/遥测时间差（`telemetry_lag`）、手眼外参都随设备与
  安装变化；三步流程与常见坑见 [`docs/calibration_opencv.md`](docs/calibration_opencv.md)。
- **PX4 侧需要自己配置**：gripper 输出、起飞项的航点动作、任务结束后的行为。
- **`cls12` 的 12 类权重目前不提供**（`best2.pt` 是单类），该模式当前只能跑出编号 1。
- **无 CI**：测试、ruff、pyright 都在本地跑。
- **SITL 注意**：仿真要与感知共用一台机器时**务必用无头模式**（`HEADLESS=1`，`gz sim -g` 会和
  检测抢 GPU，把仿真拖到 ~0.5x 实时、感知落后到看不到目标）；SITL 没有遥控，测试里要临时把
  `NAV_RCL_ACT` 置 0，否则 RC-loss 失控保护会把飞机拉回场；SITL 机型还要关掉地形相对降落
  （`FW_LND_USETER=0`，仿真没有测距传感器）。这几条在 `sim/README.md` 里有原因说明。

## 安全提示

- **本项目只做过仿真验证，没有真机飞行数据**：任何会驱动飞机的操作（arm / takeoff /
  mission upload / 投放）**先只在拆桨或 SITL 环境里试**；装机上电后第一次真机测试要系留/
  拆桨逐步放开，并确保随时能切回手动接管。
- 真机演练请用 `DRY_RUN`（投放指令只记日志）并确认 `abort_action` 符合空域要求。
- 投放判据是**建议**而不是保证：它按弹道与风预测落点，参数没标定过就只是"按配置在飞"。
