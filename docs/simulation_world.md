# CUADC 赛区仿真世界

`sim/worlds/cuadc/` 是 PX4 SITL 使用的比赛场景世界，按《2026 中国大学生飞行器设计创新大赛
竞赛规则》（CUADC）"固定翼无人机侦察与打击"赛项场地构建，供 SITL 演练
（`examples/sitl_mission.py`）、弹道/投放试验与感知离线迭代使用。
世界为生成产物：修改请改生成器 `tools/make_world.py` 并重新生成。

```text
sim/                                 # 仿真模块（世界 + 机型 + airframe + 脚本，见 sim/README.md）
├── install_px4.sh                   # 将机型/世界/airframe 软链进 PX4（幂等；--uninstall 移除）
├── run_sitl.sh                      # 一键：install + 启动 SITL
├── airframes/4007_gz_rc_cessna_down_cam   # PX4 airframe
├── vehicles/rc_cessna_down_cam/     # Gazebo 机型（带下视相机；merge 基础 rc_cessna）
├── patches/                         # 可选 PX4 补丁（NVENC 探测）
└── worlds/cuadc/
    ├── cuadc_recon_strike_r1.sdf    # 第一轮：天井放置图片靶标
    ├── cuadc_recon_strike_r2.sdf    # 第二轮：天井放置数字靶标（两位数）
    └── materials/
        ├── meshes/                  # 五边形靶板/环壁 + 各类贴图平面（生成器生成的文本资产）
        └── textures/                # 12 张图片靶 + 10 张数字板 + 6 张目标区地面 + 6 张航拍底图（入库 PNG）
```

## 1. 规则依据（条款 → 世界实现）

| 规则 | 原文要点 | 世界实现 |
| --- | --- | --- |
| 3.2 | 起降区：约 50×50m 的跑道区域 | 原点处 200×30m 跑道 + 50×50m 起降区标记。跑道加长为工程选择：固定翼 SITL 需要滑跑距离，规则中"约 50×50m"为整块起降地块 |
| 3.3 | 目标区约 60×60m、距起降区约 200m、分 A/B 两区、四角插旗 | 按规则图 1 分列起飞线两端：A 区（蓝）中心 (+200, 0)、B 区（红）中心 (-200, 0)，各距原点 200m；各自 60×60m 边界线与四角旗 |
| 3.3 | 每区 4 座天井、间距 > 20m、高 400mm；A 区底面蓝、B 区底面红 | 位置与朝向均随机（由种子决定；全部位于区内、两两间距 > 20m）；天井为五边形环壁（内轮廓 = 靶板轮廓，壁厚 5mm、高 400mm）+ 同形底面与靶板（A 蓝 / B 红，同区同色） |
| 3.3.1 | 任务一：每区 3 座天井放置 600×600mm 图片靶标，"目标价值" 1~12 | `r1` 世界；图片靶标来自规则附件中的官方剪影（见 §4） |
| 3.3.2 | 任务二：每区 3 座天井各放置 2 块 600×300mm 数字板，字高约 400mm，白底黑字 | `r2` 世界；每座天井两块 0.3×0.6m 数字板拼成两位数 |
| 3.3.3 | 以天井中心为圆心，r=4m 精确打击区、r=6m 有效打击区 | 不绘制：该内容为规则图 5 中的示意，场地上没有这两个圈 |
| 3.3.1/3.3.2 | 靶标方向、"天井箭头"方向均随机 | 整座天井（壁、底面、靶板）按随机朝向摆放，靶标方向随之随机；"天井箭头"默认不绘制（规则图 2/图 4 中没有该箭头，见 §3），需要时用 `make-world --well-arrows` 生成 |

尺寸口径：五边形 = 1m 方形 + 等边三角顶角（顶角 60°，与感知模块的
`HOUSE_APEX_ANGLE_DEG` 同形态），靶板厚 20mm、天井底面加强板厚 60mm、
环壁厚 5mm、壁高 400mm。靶板与底面同形，置于壁内并四边贴壁。

## 2. 坐标与几何

世界使用 ENU 坐标：`+x` = 东、`+y` = 北、`+z` = 天；原点 = 起降区中心 = 飞机出生点，
飞机出生朝向 +x（沿跑道）。

| 要素 | 位置 / 尺寸 |
| --- | --- |
| 跑道 | 以原点为中心，沿 x 轴 200m × 30m；中线每 20m 一段、两端各一组阈值条 |
| 起降区标记 | 原点 50×50m |
| 操纵区 | (0, -34)，20×16m（位于起降区标记南侧，与之不重叠） |
| 目标区 A | 中心 (+200, 0)，60×60m（跑道东端外约 100m） |
| 目标区 B | 中心 (-200, 0)，60×60m（跑道西端外约 100m） |
| 天井（每区 4 座） | 区内随机摆放（A1…A4 / B1…B4 仅为编号，无固定方位），两两间距 > 20m；默认 seed 0 的坐标见 §3 |

贴地标线分层摆放（跑道 0.01 / 起降区标记 0.03 / 操纵区 0.05 / 跑道中线与阈值条 0.07，
单位米）：任意两层不得共面重叠（共面重叠会产生 z-fighting），
`tests/test_world.py` 包含对应的回归检查。

`spherical_coordinates` 沿用 PX4 默认世界原点（47.397971, 8.546164，苏黎世），
因此 GPS 原点、示例航线与 `routes/land.plan` 无需另行配置；**更换场地**时需修改三处：
世界的 `spherical_coordinates` + `routes/land.plan` 的经纬度 + `examples/sitl_mission.py`
的航点常量（见 `routes/README.md`）。

### 2.1 光照：太阳位置可配置

世界包含一盏平行光（太阳），其传播方向由"方位角 + 仰角"计算，阴影随之变化：

- 默认参数 `SUN_AZEL_DEG = (218, 55)`（南偏西 55° 高，短阴影）；
- 命令行：`python -m airdrop.run make-world --sun 90,12`。方位角从北（+y）起顺时针
  （0=北、90=东、180=南、270=西），仰角自地平线起算，必须位于 (0, 90]
  （位于地平线下时直接报错并返回 1，不生成无效世界）；
- 典型配置：`--sun 90,12`（东侧低角度侧光，长阴影）、`--sun 270,15`（西侧长阴影）、
  `--sun 0,80`（近顶光，几乎无阴影）。修改后需**重启 SITL** 生效（世界文件在启动时读取）；
- **随机太阳**：`--sun-random` 由 `seed + 轮次` 决定太阳位置（方位 0~360°、仰角 15~80°，
  相同输入可复现）。批量生成不同阴影环境时更换 `--seed` 即可；同时指定 `--sun` 时以随机为准。
  默认关闭（入库世界使用 `SUN_AZEL_DEG` 固定值）；
- 阴影在两个位置开启：世界 `<scene><shadows>true</shadows>` 与光源
  `<cast_shadows>true</cast_shadows>`；背景与环境光配置在 `<scene>` 中。

### 2.2 目标区底色与场外航拍底图

规则仅规定天井底面颜色（A 蓝 / B 红），场地面层与场地外观均为随机项。本世界将其实现为
可复现的随机，用于检验检测/识别的抗扰能力：

- **目标区底色**：60×60m 外扩 6m 的一块"常见路面色 + 少量纹理"贴图，从 6 种
  （沥青 / 旧水泥 / 混凝土 / 土面 / 砖面 / 砂石）中由 `seed` 为 A、B 两区各抽取一种；
  底色块略大于区界，白色边界线覆盖在其上。`make-world` 的输出会打印当前种子对应的选择
  （"底色 ground_xxx/ground_yyy"）。
- **比赛区域外**：随机铺设 12 块（`--aerial-patches N`，0 = 不铺设）航拍干扰底图
  （农田/道路/房屋/路面标线等）。位置、尺寸（60~130m）、朝向均由 `seed` 决定，
  硬性约束为：整块不得覆盖跑道 / 起降区标记 / 操纵区 / 两个目标区的底色块（可紧贴边界），
  且块与块之间不重叠；每块 z 坐标依次抬高 3mm，避免共面重叠。用于在下视画面中提供
  "不应被识别为目标"的背景。
- 两类底图均可由 `python -m airdrop.run make-backdrops` 离线生成（可复现，`--seed` 更换批次）；
  `python -m airdrop.run fetch-aerial` 可将航拍底图替换为真实影像（USGS/NAIP，公有领域，
  需联网）。更换底图后需重新 `make-world` 并重启 SITL 生效。

## 3. 两个世界与靶标排布

| 天井 | 轮次 | A 区内容 | B 区内容 |
| --- | --- | --- | --- |
| 第 1~2 座 | `r1` | 机枪兵(1)、坦克(7) | 多旋翼(3)、直升机(8) |
| 第 3 座 | `r1` / `r2` | 轰炸机(12) / 两位数 **56** | 侦察机(11) / 两位数 **61** |
| 第 4 座 | 空 | 空 | 空 |

`r2` 中 A 区三个数为 94 / 12 / 56（中位数 56，位于 A3），B 区为 38 / 70 / 61
（中位数 61，位于 B3），对应规则任务二"打击中位数靶标"；`examples/sitl_mission.py`
的演练航线终点与合成目标对准 A3（56 号）。

**天井位置与朝向均随机**（规则 3.3），布局由 `--seed` 决定：默认 seed 0 为入库布局，
更换种子会同时更换位置与朝向。摆放有两条硬性约束：两两间距 > 20m（规则要求）；
中心距区边界至少 8m（使规则 3.3.3 的 r=6m 有效打击圈完整落在目标区内，圈外保留 2m 余量）。
入库布局（seed 0，ENU 米 / 度）：

| 天井 | A 区（蓝）坐标 | 朝向 | B 区（红）坐标 | 朝向 |
| --- | --- | --- | --- | --- |
| 1 | (213.11, 6.91) | 339.5° | (-199.49, 20.44) | 253.6° |
| 2 | (178.01, -13.99) | 109.0° | (-178.18, 13.77) | 336.8° |
| 3 | (200.30, -10.80) | 146.9° | (-191.93, -15.22) | 186.2° |
| 4（空） | (180.89, 15.83) | 291.6° | (-221.78, 4.20) | 250.9° |

演练航线（`examples/sitl_mission.py`）中的经纬度常量按上表 A3 = (200.30, -10.80) 编写；
`tests/test_world.py` 会将其换算回 NED 并与入库世界比对。

两个轮次入库时使用同一套天井布局（同一 seed）。如需模拟"两轮布局不同"，可分别以不同
`--seed` 生成（会覆盖入库的世界文件）。

关于"天井箭头"：规则正文 3.3.1/3.3.2 规定"靶标方向、天井箭头方向均随机"、数字靶方向
须与箭头一致，但规则图 2 / 图 4 中未绘制该箭头。本世界默认不绘制，需要时使用
`python -m airdrop.run make-world --well-arrows` 开启；"靶标方向随机"由天井朝向本身实现。

## 4. 资产（网格与贴图）

文本网格由生成器写出：

- `materials/meshes/pentagon_plate.obj`：五边形靶板（1m 方形 + 60° 顶角，厚 20mm）。
  同一网格也用作天井底面（z 方向放大为 60mm 加强板）。
- `materials/meshes/pentagon_well_wall.obj`：天井五边形环壁（内轮廓 = 靶板轮廓，
  外轮廓外扩 5mm；高 400mm；含顶/底环与内外侧面及法线）。
- `materials/meshes/sheet_*.obj` + `.mtl`：靶纸平面（1m² 单位面）。UV 约定：图像上方
  → +x、图像左侧 → +y，即"数字正读方向 = 顶角方向"（已通过离屏渲染核对：
  在贴图四边绘制红/蓝条，红条位于 +x 侧、蓝条位于 +y 侧）。

贴图 PNG 为入库的二进制资产：

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `pic_01…pic_12_*.png`（600×600） | 官方 12 张图片靶标（机枪兵/火箭兵/多旋翼/固定翼/卡车/防空炮/坦克/直升机/战斗机/运输机/侦察机/轰炸机） | 规则附件《固定翼无人机侦察与打击靶标》PDF 中抽取的剪影，裁剪至包围盒并等比居中放入 60×60 靶纸 |
| `digit_0…digit_9.png`（300×600） | 数字板，白底黑字、字高 400/600 | 脚本以加粗无衬线字体渲染（黑像素占比 21%~36%） |
| `ground_asphalt…ground_gravel.png`（512×512，6 张） | 目标区底色：6 种常见路面/地面色，纯色 + 少量同色系纹理 | `python -m airdrop.run make-backdrops`（程序化，可复现；`--seed` 更换批次） |
| `aerial_1…aerial_6.png`（512×512，6 张） | 比赛区域外航拍干扰底图：农田/商区/住宅/工业/林地/湿地 | 默认同上（程序化）；也可经 `python -m airdrop.run fetch-aerial` 替换为真实影像（USDA NAIP via USGS National Map，公有领域；160m/张、约 0.3m/像素）。再分发时请保留来源说明 |

图片靶素材仅用于仿真迭代，再分发前请自行确认素材授权。
`ground_*` / `aerial_*` 两组文件名由两个命令写入：`make-backdrops`（离线、程序化）与
`fetch-aerial`（联网、真实影像）；后运行者生效。若需保留真实影像不被程序化覆盖，
使用 `make-backdrops --parts ground`。

## 5. 启动 SITL（WSL / Linux）

世界文件必须位于 PX4 的 worlds 目录（`px4-rc.gzsim` 按
`${PX4_GZ_WORLDS}/${PX4_GZ_WORLD}.sdf` 查找文件）；世界的资产（网格/贴图）通过
`GZ_SIM_RESOURCE_PATH` 定位本仓库目录。两者均需配置：

```bash
# 一键命令（WSL 环境；先自动安装机型/世界/airframe，再启动）：
bash sim/run_sitl.sh r2                 # 第二轮世界 + 带下视相机机型（默认）
HEADLESS=1 bash sim/run_sitl.sh r2      # 无 GUI
PX4_DIR=/path/to/PX4-Autopilot bash sim/run_sitl.sh r2   # PX4 不在默认位置

# 等价手动步骤：
bash sim/install_px4.sh                 # 幂等：软链机型/世界/airframe 进 PX4
export GZ_SIM_RESOURCE_PATH="$PWD/sim/worlds/cuadc:$GZ_SIM_RESOURCE_PATH"
cd "$HOME/PX4-Autopilot"
PX4_GZ_WORLD=cuadc_recon_strike_r2 make px4_sitl gz_rc_cessna_down_cam
```

机型与 airframe 均在本仓库中（`sim/vehicles/rc_cessna_down_cam` +
`sim/airframes/4007_gz_rc_cessna_down_cam`），由 `sim/install_px4.sh` 软链进 PX4。
装机细节、可选补丁与还原方法见 [`sim/README.md`](../sim/README.md)。

默认机型为 **`gz_rc_cessna_down_cam`**：机身下挂 1280×720@30Hz、水平视场 1.74 rad 的
相机（merge PX4 模型库中的基础 `rc_cessna` + 本仓库的 `camera_720p` 子模型）。
相机话题由 Gazebo 按作用域命名（r2 世界）：

```text
/world/cuadc_recon_strike_r2/model/rc_cessna_down_cam_0/link/camera_link/sensor/imager/image
/world/cuadc_recon_strike_r2/model/rc_cessna_down_cam_0/link/camera_link/sensor/imager/camera_info
```

注意：在地面上抓帧会看到天空。相机朝向正确（相机链接世界姿态为 +90° 俯仰），
原因是机体停在地面时相机贴地穿模，越过单面 plane 地面看到背景色；在空中取图正常。
如需在地面验证画面，应先起飞或临时将机体抬升数米。

验收：GUI 中应能看到跑道、两个目标区与 8 座天井；`gz topic -l | grep '/world/cuadc'`
应包含 `/world/cuadc_recon_strike_r2/clock`。无头环境使用 `HEADLESS=1`（PX4 仅启动 server）。

### 5.1 视频流：PX4 自带的 GstCameraSystem

PX4 的 `gz_bridge/server.config` 会为每个世界自动加载 GStreamer 相机系统插件
（`src/modules/simulation/gz_plugins/gstreamer/`）。依据官方源码与插件 README：

- 插件自动发现世界中的第一台相机（本项目为 `rc_cessna_down_cam` 的下视相机），
  以 GStreamer 编码为 **RTP/H.264 推送至 `127.0.0.1:5600`**（默认值）；
- **QGC 配置**：Application Settings → Video → Video Source =
  `UDP h.264 Video Stream`、UDP Port = `5600`（官方默认接收端配置）。
  不应选择 TCP 类视频源（`TCP-MPEG2 Video Stream`）并填写 `0.0.0.0:5600`：
  该组合会导致 QGC 启动时异常退出；如已发生，将 ini 文件（Windows：
  `%APPDATA%\QGroundControl\QGroundControl.ini`）中的 `videoSource` 改回
  `UDP h.264 Video Stream`、`udpUrl=0.0.0.0:5600` 并清空 `tcpUrl` 即可恢复；
- **在 Windows 侧接收**：启动仿真前设置 `PX4_VIDEO_HOST_IP=<主机局域网 IP>`
  （插件源码中该环境变量优先于默认 `127.0.0.1`）：
  `PX4_VIDEO_HOST_IP=<主机局域网 IP> bash sim/run_sitl.sh r2`；
- 两条硬性约束：**同一时刻只能有一个 QGC 监听 5600**（mirrored 模式下 WSL 与 Windows
  共享端口空间）；**Windows 侧需放行入站 UDP 5600**（防火墙默认拦截 QGC 入站流量）：
  `New-NetFirewallRule -DisplayName "QGC video UDP 5600" -Direction Inbound -Protocol UDP -LocalPort 5600 -Action Allow`（管理员权限）。
- 不使用 QGC 时，可用官方查看命令（插件 README）：
  `gst-launch-1.0 udpsrc port=5600 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink`

SDF 中还可配置 `udpHost` / `udpPort` / `rtmpLocation` / `useCuda`（插件参数）；
本项目不修改 PX4 源码，更换目的地通过环境变量实现。官方参考：PX4 文档
"Gazebo Simulation → Video Streaming" 与插件 README
（`src/modules/simulation/gz_plugins/gstreamer/README.md`）。

与项目其他约定一致：同一时刻只允许一个 `mavsdk_server`（固定占用 gRPC 50051）。
Windows 侧运行演练使用 `python -m airdrop.run sitl`，航线已对准本赛区；
飞行前需通过 QGC 或遥控器手动起飞。

## 6. 重建 / 更换场景

```bash
# 入库的两个世界（默认 seed 0、静风）
python -m airdrop.run make-world
# 仅生成第二轮、更换天井位置与朝向、设置 5 m/s 东 + 2 m/s 北 恒定风
python -m airdrop.run make-world --rounds 2 --seed 3 --wind 5,2,0
# 生成到其他目录（贴图会一并复制，可直接运行 SITL）
python -m airdrop.run make-world --out-dir /tmp/scenario_7
# 更换阴影环境：东侧低角度侧光（方位角,仰角；详见 §2.1）
python -m airdrop.run make-world --sun 90,12
# 需要"天井箭头"的场景（规则文本有、规则图未绘制，默认不生成）
python -m airdrop.run make-world --well-arrows
# 不铺设场外航拍底图 / 铺设 12 块
python -m airdrop.run make-world --aerial-patches 0
python -m airdrop.run make-world --aerial-patches 12
# 重新生成底图：程序化（离线）/ 真实航拍（联网，USGS NAIP，公有领域）
python -m airdrop.run make-backdrops
python -m airdrop.run fetch-aerial
# 仅重新生成目标区地面纹理，保留真实航拍图
python -m airdrop.run make-backdrops --parts ground
# 查看当前布局（带标注的俯视预览图；只读世界文件）
python -m airdrop.run preview-world
```

生成器仅写文本（SDF/网格/材质）；写盘前会校验 XML 合法性、世界引用的网格与材质、
材质引用的贴图是否齐全，缺失时报告并返回 1。`tests/test_world.py` 校验
"生成结果与入库文件一致"，因此**修改生成器后需重新运行 `make-world` 并提交生成结果**。
