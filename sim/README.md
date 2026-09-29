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
"生成结果 == 入库文件"）。布局与规则依据见
[`docs/simulation_world.md`](../docs/simulation_world.md)。
