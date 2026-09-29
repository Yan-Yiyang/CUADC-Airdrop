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
# （PX4 的 API/offboard 链路长时间没有接收方时会停发；脚本自己也会重连）
```

严格**任务流程**，全程没有 offboard 设定点：上传「起飞 + 扫掠」任务 → `MISSION_START`
（PX4 自己解锁、切任务模式、执行首项起飞）→ 本包只监视遥测与画面 → 飞完 `hold`。
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

与环境有关的三条：

* 测试会临时把 `NAV_DLL_ACT` 置 0（SITL 里没有操作手盯数据链，免得 failsafe 中途抢
  控制），收尾还原成机型标准值 2；
* **`MIS_TKO_LAND_REQ` 已在仿真机型里永久置 0**：侦查段的任务只有"起飞 + 扫掠"——
  真实流程里投弹航线要等空中出结果才能生成，降落段没法预先和侦查段拼成一条任务——而
  `rc.fw_defaults` 对固定翼默认 2（"带了起飞就必须带降落"），不关掉的话 PX4 的
  `mission_feasibility_checker` 直接拒任务
  （`Mission rejected: Landing waypoint/pattern required.`），飞机连解锁都不会发生。
  置 0 = 起飞项/降落项都不再必需，只影响这台仿真机型；
* **`FW_LND_USETER=0` 也在仿真机型里永久置 0**：SITL 机型没有测距传感器，而 FW 自动
  降落默认要用地形估计（1），拿不到估计时会按 `FW_LND_ABORT`（默认 3 含地形位）
  在进入降落段 10 s 后 abort 降落、在落点上方 30 m 无限盘旋——任务永远不结束
  （实测架次 某架次：`Holding at 30 m above landing waypoint.`，
  最后靠状态机 `land_timeout` 收场）。置 0 = 不要求地形估计，flare/下滑用航点高度；
* 有 QGC 连着更好（真机流程本来就有地面站；它能满足预检的数据链检查）；
* **跑精度测试一律用无头模式**（`HEADLESS=1 bash sim/run_sitl.sh r2`）：带 GUI 时
  `gz sim -g` 会和感知抢同一块 GPU，实测把仿真拖到 ~0.5x 实时（ulog 里"侦查飞完"
  =53.6 模拟秒、遥测同一事件=101.6 实时秒）⇒ 检测只有 8~22 fps、感知落后 ~1900 帧，
  `HOLD_PROCESS` 看不到目标（架次 某架次 实测）。`px4-rc.gzsim` 里
  `if [ -z "${HEADLESS}" ]` 才起 GUI，所以置 1 即可；
* 测试期间会临时置 0 的飞控参数有两条（跑完都还原）：`NAV_DLL_ACT`（无数据链 failsafe）
  与 `NAV_RCL_ACT`——SITL **没有遥控**（ulog `manual_control_signal_lost` 恒 true），
  默认的 RC-loss 动作（Return）会在飞几十秒后把飞机拉回场（架次 某架次 实测）。

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
