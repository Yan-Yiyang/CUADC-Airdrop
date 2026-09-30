# sim/ — Gazebo / PX4 SITL 仿真模块

把"赛区世界 + 带下视相机的机型 + airframe + 启动脚本"收在一个地方，**仓库是唯一出处**：
PX4 源码树里不再存任何手工副本（全部靠 `install_px4.sh` 软链过去）。

```
sim/
├── README.md                     # 本文件
├── install_px4.sh                # 把机型/世界/airframe 挂进 PX4（幂等；--uninstall 摘掉）
├── run_sitl.sh                   # 一键：install + 起 SITL
├── airframes/
│   └── 4007_gz_rc_cessna_down_cam   # PX4 airframe（固定翼 + 下视相机机型）
├── vehicles/
│   └── rc_cessna_down_cam/          # Gazebo 机型：merge 基础 rc_cessna + 720p 相机
│       ├── model.sdf / model.config
│       └── camera_720p/{model.sdf, model.config}
├── patches/
│   └── gst_camera_nvenc_probe.patch # 可选：NX/WSL 下 NVENC 可用性探测（见下）
└── worlds/
    └── cuadc/                       # 赛区世界（生成产物，别手改）
        ├── cuadc_recon_strike_r1.sdf  # 第一轮：图片靶标
        ├── cuadc_recon_strike_r2.sdf  # 第二轮：数字靶标
        └── materials/                 # 网格 + 贴图（靶纸/目标区底色/航拍底图）
```

## 起 SITL（WSL / Linux）

```bash
# 一键（先装机型/世界/airframe，再起）
bash sim/run_sitl.sh r2                       # 第二轮世界 + 带下视相机机型（默认）
HEADLESS=1 bash sim/run_sitl.sh r2            # 只起 server，无 GUI
bash sim/run_sitl.sh r2 gz_rc_cessna          # 换无相机机身（用 PX4 自带模型）
PX4_DIR=/path/to/PX4-Autopilot bash sim/run_sitl.sh r2    # PX4 不在默认位置时

停 SITL：`pkill -f 'bin/px4'; pkill -x ruby`。

⚠ `gz sim` 的**进程名是 `ruby`**（只有命令行里才带 "gz sim"）：`pkill -f 'gz sim'`
会把"命令行里含这串字样"的自己也算进去，容易先杀掉发起命令的 shell（实测踩过：
以为关了、其实 server 还在以 600%+ CPU 跑）。按 `-x ruby` 杀最稳，
或用 `ps -eo pid,pcpu,args --sort=-pcpu` 确认没有 `gz sim ...` 再收工。
```

只装不改：

```bash
bash sim/install_px4.sh            # 幂等：软链机型/世界/airframe，并登记 airframe
bash sim/install_px4.sh --uninstall   # 摘掉（含还原 airframes/CMakeLists.txt）
```

手动起（等价于 `run_sitl.sh` 做的事）：

```bash
bash sim/install_px4.sh
export GZ_SIM_RESOURCE_PATH="$PWD/sim/worlds/cuadc:$GZ_SIM_RESOURCE_PATH"
cd ~/PX4-Autopilot
PX4_GZ_WORLD=cuadc_recon_strike_r2 make px4_sitl gz_rc_cessna_down_cam
```

## 目标侦查精度自动测试（`sitl-recon`）

一条命令量"**解算出的目标坐标离天井中心多远**"（验收口径 < 2 m）：

```bash
# WSL 里先起 SITL（带下视相机机型 + r2 世界）
bash sim/run_sitl.sh r2
# Windows 侧（仓库 venv）
./.venv/Scripts/python.exe -m airdrop.run sitl-recon

# 更稳的顺序是"先起脚本、再起 SITL"：脚本先把 14540 绑住，SITL 一启动就连上
# （PX4 的 API/offboard 实例以 -o 14540 推心跳，长时间没有接收方就停发，mavsdk_server
#  会永远停在 `Waiting to discover system`；脚本自己也会重连）
```

严格**任务流程**，全程没有 offboard 设定点：本脚本只扮演操作手——上传「只含起飞项」的任务 →
**先 `arm` 再 `MISSION_START`**（⚠ 实测 `MISSION_START` 里的 `arm(mission_start)` 只切模式、不解锁：
ulog 里 `nav_state=3` / `arming_state=1`，飞机不动、QGC 上看不到任何动静）→ 等起飞 →
上传并启动侦查航线，其余交给 `MissionRunner`（监视遥测与画面 → 空中出目标 → 飞掠投放 → 降落）。
视频走 Gazebo 自带的 `GstCameraSystem`（RTP/H.264 → `127.0.0.1:5600`），Windows 侧用
ffmpeg 直接收（mirrored 网络共享端口；**不必用 HM30，也不用 QGC 转发**）。

产物：终端报告 + `.sitl-recon-tmp/report.json`
（任务结果与分井误差、延时敏感性——同一批检测按不同 `telemetry_lag` 重算、
反向诊断——同一条扫掠线两个方向的偏差 ⇒ 链路延时估计）+ `flights/<ts>/` 记录。

⚠ **链路延时是这里的头号精度项**：SITL 实测 ≈0.5 s（Gazebo 渲染 + GStreamer/x264 编码缓冲
+ RTP/UDP + ffmpeg 解码），**与真机 HM30 的 ≈0.15 s 完全不同**。用错延时的代价是沿航迹
的系统偏差（实测：0.08 s → 3.5 m；0.49~0.52 s → 0.05~1.0 m）。所以报告里会给
"**自标定**"：用已知的世界真值把本架次的链路延时解出来，并**留一个天井只做验证**；
相机外参则严格按模型给（`sim/vehicles/rc_cessna_down_cam`）——外参若错，修正延时后
仍会剩下恒定的像素偏差（实测修正后残差 (+9, +1) px ≈ 噪声）。

### ⚠ 侦查航线必须按固定翼转弯几何生成

`tools/sitl_recon.py` 的 `RECON_LEGS_ENU` 只是**折线骨架**，真正下发的航点由
`round_corners()` 把每个尖角切成**半径已知的圆弧**（`turn_radius_m() = v²/(g·tanφ)`，
按实测沿迹速度 15 m/s、最大倾角 45° ⇒ **R ≈ 22.9 m**），相邻航点航向差因此被钉在 45° 以内。

两条硬约束，都是踩过的坑：

* **绝不能出现原地 180° 掉头**。固定翼做不到，而且这**不是"航线好不好看"的问题**：
  180° 掉头的横向挪移恒等于 `2R`（两段反向圆弧的横移之和 `R(1-cosα)+R(1+cosα)` 与两段
  圆心角无关），所以"同一条线来回飞"只有在两条线相距 ≥2R 时才飞得出来。2026-09 那版两条
  平行线只隔 **19 m**、掉头环半径只有 9.5 m，20 m/s 下要 77° 倾角——PX4 只能"尽量绕"，
  实际航迹与航线对不上，而坐标解算完全依赖飞机此刻在哪。
* **代价：靶区南北只有 19 m**（12@N-14、56@N-10.8、94@N+6.9），15 m 高度的视场半幅
  `15·tan(0.87)=17.8 m`。两条线按 2R≈46 m 拉开后**第二条线一个靶标都看不到**——
  "同线反向"与"转得出来"在这个布局下互斥。所以现在是：**单条 N=0 的线一次压过三个靶标**
  （偏南 14 / 10.8、偏北 6.9 m，全在视场内）+ **一条西南向对角线反向掠过**同样三个靶标。
  全程转角 ≤45°，没有任何 180°。

⚠ 连带后果：反向观测来自**不同切向**，所以报告里 `reverse.implied_lag_s` **混了切向变化、
只能当量级参考**，不是标定值（`tools/sitl_recon.py` 的 `_reverse_diagnostic` docstring
写明了）。要一个干净的同线反向数，得给侦查段加 `DO_CHANGE_SPEED` 把速度压到 ~12 m/s
（半径才缩得下 19 m 的靶区间距）——本包目前没有这个能力。

回归口径在 `tests/test_sitl_recon.py`（纯离线、默认跑）：最大转角 ≤45°、圆弧拟合半径 ≈ 声明值、
三个靶标都在视场半幅内。

### ⚠ 起飞后 10~30 s，PX4 报的 `yaw` 会偏 25~35°（落点误差的头号来源）

这是 2026-09 用**6 个架次 / 114 个带编号观测**逐条查出来的，结论与本项目代码无关：

* **现象**：起飞后 0~10 s 航向偏差中位 ≈0°；**10~20 s 掉到 −12~−26°；20~30 s 到 −26~−35°**
  （`|偏差|>10°` 占 82~100%）；30~40 s 回到 ±6° 以内，之后一直正常。6 个架次全部复现。
* **证据链**：位置差分与 `velocity_ned`（**两条互相独立**的流）都说航迹角是 +90°（正东），
  而 `attitude` 里的 `yaw` 说 112~118°；按 `yaw` 把 NED 速度换到机体系，**侧滑角 22~34°**
  （`v_body` 右分量 −5.4~−9.0 m/s）。滚转 <1.5° 的定翼平飞扛不住 30° 稳态侧滑——垂尾的
  恢复力矩会把它推回去，所以是**姿态估计错了**，不是飞机真的 crab 着飞。
* **旁证：图像站在航迹角这边。** 相机刚性挂在机体上，本来就该跟姿态；把 yaw 当自由参数
  做重投影拟合，"图像要求的航向"与**航迹角**的差中位 −0.85°、标准差 10.6°，而与**记录的
  yaw** 的差有 42% 超过 10°。
* **后果**：`pixel_to_ned` 用的是机体姿态（这是对的，不该为了分数改成速度矢量），于是这批
  观测的落点误差被放大。同一批点把 yaw 换成航迹角重算：

  | yaw 取法 | 中位 | 均方根 | p90 | 最大 | <2 m |
  | --- | --- | --- | --- | --- | --- |
  | 记录 yaw（管线用的） | 1.31 m | 3.72 m | 6.28 m | 13.16 m | 66.7% |
  | 速度矢量航迹角 | **0.82 m** | **1.09 m** | 1.72 m | 3.15 m | **94.7%** |

  按 `|航迹角 − yaw|` 分组更直白：<5° 的 47 个观测重投影残差中位 32 px；**>15° 的 41 个中位
  153 px**——就这 41 个把全局中位数从 24 px 拖到 46 px。**本项目的坐标解算本身精度约 1 m。**
* **顺带排除掉的**（都不是原因，别再查）：Gazebo 相机**不加畸变**（`<camera>` 要显式写
  `<distortion>` 段才生效，`camera_720p/model.sdf` 里没有 ⇒ 理想针孔，与 `dist_coeffs` 全零
  一致）；`r_bc` 安装角无偏（自由拟合三个小角度只把残差 92→83 px，解释不掉）；帧-遥测
  **插值距离过长**也不行——插值跨度中位 0.125 s、最大 0.376 s，管线自带的那份对齐快照与从
  `telemetry.jsonl` 插值出来的给出**完全相同**的重投影残差（49.1 px），两者位置差中位仅
  0.28 m。
* **所以工具里有对应诊断**：`tools/sitl_recon.py` 的 `heading_consistency`（航迹角 vs yaw）
  与 `observation_errors`（**全量**与**航向一致子集**两个口径）。阈值 10°，**只报告不剔点**
  ——剔了就看不到"精度是被姿态估计吃掉的"这个事实，而那正是这个工具要回答的问题。
* **还没定性的**：PX4 侧为什么偏（EKF2 航向源 / 磁模型 / 起飞后 yaw 对齐）。已确认我们的机型
  是 `<include>model://rc_cessna</include>` + 相机，**继承 PX4 原版的 `magnetometer_sensor`**
  （base_link，100 Hz），所以不是"没罗盘"；airframe 里没设任何 mag/ekf/yaw 参数，走默认；
  世界的 `<magnetic_field>6e-06 2.3e-05 -4.05e-05</magnetic_field>`（T）量级 47 µT、倾角
  59.6° 都合理，但**磁偏角 75.4° 对这个经纬度是不合理的**（当地实际 ~2~3°）——恒定偏差不解释
  这条瞬态，但值得下次连同 EKF2 参数一起看。

与环境有关的三条：

* 测试会临时把 `NAV_DLL_ACT` 置 0（SITL 里没有操作手盯数据链，免得 failsafe 中途抢
  控制），收尾还原成机型标准值 2；
* **`MIS_TKO_LAND_REQ` 已在仿真机型里永久置 0**：侦查段的任务只有"起飞 + 扫掠"——
  真实流程里投弹航线要等空中出结果才能生成，降落段没法预先和侦查段拼成一条任务——而
  `rc.fw_defaults` 对固定翼默认 2（"带了起飞就必须带降落"），不关掉的话 PX4 的
  `mission_feasibility_checker` 直接拒任务
  （`Mission rejected: Landing waypoint/pattern required.`），飞机连解锁都不会发生。
  置 0 = 起飞项/降落项都不再必需，只影响这台仿真机型；
  ⚠ 顺带一条：**已解锁状态下切模式还要 `canRun(AUTO_MISSION)` 通过**——任务无效时同样切不过去
  （控制台原话 `Switching to Mission is currently not available`），所以"任务被拒"与"切不进任务模式"
  是同一件事的两面；
* **`FW_LND_USETER=0` 也在仿真机型里永久置 0**：SITL 机型没有测距传感器，而 FW 自动
  降落默认要用地形估计（1），拿不到估计时会按 `FW_LND_ABORT`（默认 3 含地形位）
  在进入降落段 10 s 后 abort 降落、在落点上方 30 m（`MIS_LND_ABRT_ALT`）无限盘旋——任务永远不结束
  （实测架次 <架次>：`Holding at 30 m above landing waypoint.`，
  最后靠状态机 `land_timeout` 收场）。置 0 = 不要求地形估计，flare/下滑用航点高度；
* 有 QGC 连着更好（真机流程本来就有地面站；它能满足预检的数据链检查）；
* **跑精度测试一律用无头模式**（`HEADLESS=1 bash sim/run_sitl.sh r2`）：带 GUI 时
  `gz sim -g` 会和感知抢同一块 GPU，实测把仿真拖到 ~0.5x 实时（ulog 里"侦查飞完"
  =53.6 模拟秒、遥测同一事件=101.6 实时秒）⇒ 检测只有 8~22 fps、感知落后 ~1900 帧，
  `HOLD_PROCESS` 看不到目标（架次 <架次> 实测）。`px4-rc.gzsim` 里
  `if [ -z "${HEADLESS}" ]` 才起 GUI，所以置 1 即可；
* 测试期间会临时置 0 的飞控参数有两条（跑完都还原）：`NAV_DLL_ACT`（无数据链 failsafe）
  与 `NAV_RCL_ACT`——SITL **没有遥控**（ulog `manual_control_signal_lost` 恒 true），
  默认的 RC-loss 动作（Return）会在飞几十秒后把飞机拉回场（架次 <架次> 实测）。
* **"任务上传成功、飞机却原地盘旋"这类问题的第一手证据在 PX4 的 ulog 里**：
  `~/PX4-Autopilot/build/px4_sitl_default/rootfs/log/<日期>/*.ulg`，用 pyulog 读
  `logged_messages`（看 `mission_feasibility_checker` / `navigator` 的原话）与
  `mission_result` 数据集（`valid` / `mission_id` / `seq_total` / `finished`）——
  2026-09 那两次"整条任务被拒"就是从这里定性的。

## 装进 PX4 的是什么

| 本仓库 | PX4 里的落点 | 方式 |
| --- | --- | --- |
| `sim/vehicles/rc_cessna_down_cam/` | `Tools/simulation/gz/models/rc_cessna_down_cam` | 软链（Gazebo 按 `model://` 找） |
| `sim/worlds/cuadc/cuadc_recon_strike_r*.sdf` | `Tools/simulation/gz/worlds/` | 软链（`px4-rc.gzsim` 只认这个目录） |
| `sim/airframes/4007_gz_rc_cessna_down_cam` | `ROMFS/px4fmu_common/init.d-posix/airframes/` | 软链 |
| （同上） | `.../airframes/CMakeLists.txt` 里登记一行 | ⚠ **sed 插入**（PX4 不 glob；先备份 `.bak`） |

那行登记是唯一会动 PX4 已跟踪文件的地方：PX4 要求 airframe 显式列进
`px4_add_romfs_files`。`install_px4.sh` 先备份成 `CMakeLists.txt.bak`，`--uninstall`
会用它还原。PX4 自己的 `git status` 会因此多出 1 个 modified + 几条 untracked
软链——属预期，别往 PX4 仓库提交。

## 机型与相机

* `rc_cessna_down_cam` = 合并 PX4 模型库里的 `rc_cessna`（**基础模型仍来自 PX4 的
  gz 模型库**，本仓库只存这一层变体）+ 一个 720p 相机的合并子模型；
* 相机：1280×720 @ 30Hz，`clip far = 3000`，水平视场 1.74 rad，固定**下视**安装
  （`CameraJoint` 的 pitch 90°）；
* 相机话题（r2 世界实测）：
  `/world/cuadc_recon_strike_r2/model/rc_cessna_down_cam_0/link/camera_link/sensor/imager/image`；
* 地面抓帧会因贴地穿模看到背景色，**空中取图正常**（详见
  [`docs/simulation_world.md`](../docs/simulation_world.md) §5）。

## patches/（可选）

`gst_camera_nvenc_probe.patch` 是给 PX4 的 `GstCameraSystem.cpp` 加一段
"NVENC 到底能不能用"的探测：某些驱动栈（例如 WSL）里 `nvh264enc` 能创建成功、
一跑就报 `Selected preset not supported`，不打补丁的话插件会选到永远出不了帧的编码器。

```bash
cd ~/PX4-Autopilot
git apply <本仓库>/sim/patches/gst_camera_nvenc_probe.patch
make px4_sitl        # 需要重新编译（改了 C++ 源码）
```

不打补丁也能跑：视频流插件会退回软编（只是 CPU 占用高一点）。补丁与仓库版本
之间可能有偏移，`git apply --3way` 或按文件顶部注释手工合。

## 世界怎么重建 / 预览

```bash
python -m airdrop.run make-world        # 重新生成世界（改生成器后必跑；产物落在 sim/worlds/cuadc）
python -m airdrop.run preview-world     # 俯视预览图（带中文标注）→ sim/worlds/cuadc/preview.png
python -m airdrop.run fetch-aerial      # 换一批真实航拍底图（联网）
```

世界是**生成产物**：`.sdf` / 网格 / 材质都别手改（`tests/test_world.py` 会核验
"生成结果 == 入库文件"）。布局、规则依据与已知边界见
[`docs/simulation_world.md`](../docs/simulation_world.md)。
