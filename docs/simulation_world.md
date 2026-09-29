# CUADC 赛区 Gazebo 世界模型（固定翼无人机侦察与打击）

`sim/worlds/cuadc/` 是 PX4 SITL 用的**比赛场景世界**：按《2026 中国大学生飞行器设计创新大赛
竞赛规则》（CUADC，`docs/simulation_world.md` 末节有出处链接）里"固定翼无人机侦察与打击"
赛项的场地搭出来，供本项目的 SITL 演练（`examples/sitl_mission.py`）、弹道/投放试验与
感知离线迭代使用。世界是**生成产物**：改动请改生成器 `tools/make_world.py` 再重跑。

```text
sim/                                 # 仿真模块（世界 + 机型 + airframe + 脚本，见 sim/README.md）
├── install_px4.sh                   # 把机型/世界/airframe 软链进 PX4（幂等；--uninstall 摘掉）
├── run_sitl.sh                      # 一键：install + 起 SITL
├── airframes/4007_gz_rc_cessna_down_cam   # PX4 airframe
├── vehicles/rc_cessna_down_cam/     # Gazebo 机型（带下视相机；merge 基础 rc_cessna）
├── patches/                         # 可选 PX4 补丁（WSL 下 NVENC 探测）
└── worlds/cuadc/
    ├── cuadc_recon_strike_r1.sdf    # 第一轮：天井里放图片靶标
    ├── cuadc_recon_strike_r2.sdf    # 第二轮：天井里放数字靶标（两位数）
    └── materials/
        ├── meshes/                  # 五边形靶板/环壁 + 各类贴图平面（生成器写的文本资产）
        └── textures/                # 12 张图片靶 + 10 张数字板 + 6 张目标区地面 + 6 张航拍底图（入库的 PNG）
```

## 1. 规则依据（条款 → 世界里的实现）

| 规则 | 原文要点 | 世界里的实现 |
| --- | --- | --- |
| 3.2 | 起降区：约 50×50m 的跑道区域 | 原点处 200×30m 跑道 + 50×50m 起降区标记。**跑道加长是工程选择**：固定翼 SITL 需要滑跑距离，规则那块"约 50×50m"是整块起降地块 |
| 3.3 | 目标区约 60×60m、距起降区约 200m、分 A/B 两区、四角插旗 | 按**规则图 1**分列起飞线两端：A 区（蓝）中心 (+200, 0)、B 区（红）中心 (-200, 0)，各距原点 200m；各自 60×60m 边界线与四角旗 |
| 3.3 | 每区 4 座天井、间距 >20m、高 400mm；A 区底面蓝、B 区底面红 | **位置与朝向都随机**（种子决定；全部落在区内、两两间距 > 20m）；天井 = **五边形环壁**（内轮廓 = 靶板轮廓，壁厚 5mm、高 400mm）+ 同形底面与靶板（A 蓝 / B 红，同区一色） |
| 3.3.1 | 任务一：每区 3 座天井放 600×600mm 图片靶标，"目标价值" 1~12 | `r1` 世界；图片靶标来自规则附件里的官方剪影（见 §4） |
| 3.3.2 | 任务二：每区 3 座天井各放 2 块 600×300mm 数字板，字高约 400mm，白底黑字 | `r2` 世界；每座天井两块 0.3×0.6m 数字板拼成两位数 |
| 3.3.3 | 以天井中心为圆心，r=4m 精确打击区、r=6m 有效打击区 | **不画**：这只是规则图 5 里"帮助理解"的示意，场地上没有这两个圈 |
| 3.3.1/3.3.2 | 靶标方向、"天井箭头"方向均随机 | 整座天井（壁、底面、靶板）按随机朝向摆放，靶标方向随之随机；"天井箭头"默认**不画**（规则图 2/图 4 里没有这个箭头，见 §3），要画用 `make-world --well-arrows` |

> 尺寸口径：五边形 = 1m 方形 + 等边三角顶角（顶角 60°，与包内感知的
> `HOUSE_APEX_ANGLE_DEG` 同一形态），靶板厚 20mm、天井底面加强板厚 60mm、
> 环壁厚 5mm、壁高 400mm。靶板与底面同形，放进壁内四边贴壁。

## 2. 坐标与几何

世界用 ENU：`+x` = 东、`+y` = 北、`+z` = 天；**原点 = 起降区中心 = 飞机出生点**，
飞机出生朝向 +x（沿跑道）。

| 要素 | 位置 / 尺寸 |
| --- | --- |
| 跑道 | 原点为中心，沿 x 轴 200m × 30m；中线每 20m 一段、两端各一组阈值条 |
| 起降区标记 | 原点 50×50m |
| 操纵区 | (0, -34)，20×16m（在起降区标记南侧，与它**不重叠**——见下面标高那条） |
| 目标区 A | 中心 (+200, 0)，60×60m（跑道东端外 ≈100m） |
| 目标区 B | 中心 (-200, 0)，60×60m（跑道西端外 ≈100m） |
| 天井（每区 4 座） | 区内**随机**摆放（A1…A4 / B1…B4 只是编号，没有固定方位），两两间距 > 20m；默认 seed 0 的坐标见 §3 |

贴地标线分层摆放（跑道 0.01 / 起降区标记 0.03 / 操纵区 0.05 / 跑道中线与阈值条 0.07，
米）：**任意两层不许共面又重叠**——曾经操纵区与起降区标记都是 z=0.02 且有一块重叠，
渲染出来就是贴图打架（z-fighting），`tests/test_world.py` 现在有回归用例钉住这条。

`spherical_coordinates` 沿用 PX4 默认世界的原点（47.397971, 8.546164，苏黎世），
所以 GPS 原点、示例航线与 `routes/land.plan` 不用另配；**换场地**时改三处：
世界的 `spherical_coordinates` + `routes/land.plan` 的经纬度 + `examples/sitl_mission.py`
的航点常量（见 `routes/README.md`）。

### 2.1 光照：太阳位置可变（阴影环境）

世界带一盏平行光（太阳），它的**传播方向**由"方位角 + 仰角"算出来，阴影随之变化：

* 参数是 `SUN_AZEL_DEG = (218, 55)`（默认：南偏西 55° 高，短阴影）；
* 命令行：`python -m airdrop.run make-world --sun 90,12`——方位角从北（+y）起**顺时针**
  （0=北、90=东、180=南、270=西），仰角从地平线起算、必须落在 (0, 90]（太阳在地平线下
  直接报错返回 1，不生成黑世界）；
* 典型用法：`--sun 90,12`（东侧低角度侧光 → 长阴影）、`--sun 270,15`（西侧长阴影）、
  `--sun 0,80`（近顶光、几乎无影）。生成后要**重启 SITL** 才生效（世界文件是启动时读的）；
* **随机太阳**：`--sun-random` 让太阳由 `seed + 轮次` 决定（方位全向 0~360°、仰角 15~80°，
  同一对输入永远复现）——批量造"不同阴影环境"的场地时换 `--seed` 就行；同时给了 `--sun`
  时以随机为准。默认关（入库世界用 `SUN_AZEL_DEG` 的固定值）；
* 阴影本身开在两处：世界 `<scene><shadows>true</shadows>` 与光源
  `<cast_shadows>true</cast_shadows>`；背景/环境光在 `<scene>` 里。

### 2.2 目标区底色与比赛区域外的航拍底图（抗扰变量）

规则只规定了天井底面颜色（A 蓝 / B 红），**场地面层**和场地外长什么样都是随机项；
这里做成可复现的随机，用来锻炼检测/识别的抗扰能力：

* **目标区底色**：60×60m 外扩 6m 的一块**常见路面色 + 少量纹理**贴图，从 6 种
  （沥青 / 旧水泥 / 混凝土 / 土面 / 砖面 / 砂石）里由 `seed` 给 A、B 两区各抽一种；
  底色块比区界略大，白色边界线压在它上面。当前种子抽到哪种，`make-world` 的打印里
  会写（"底色 ground_xxx/ground_yyy"）。
* **比赛区域外**：随机铺 12 块（`--aerial-patches N`，0 = 不铺）**航拍干扰底图**——
  农田/道路/房屋/路面标线这类画面。位置、大小（60~130m）、朝向都由 `seed` 随机，
  硬约束只有两条：**整块不许压到跑道 / 起降区标记 / 操纵区 / 两个目标区的底色块**
  （可以紧贴着它们铺，实测最近的一块离边界只有 0.3m），以及块与块之间不重叠；
  每块 z 依次抬高 3mm 兜底（万一叠了也不会共面打架）。
  下视画面里就是"不该被当成目标"的背景。
* 两类底图都由 `python -m airdrop.run make-backdrops` 生成（离线、可复现，`--seed` 换一批）；
  `python -m airdrop.run fetch-aerial` 可把航拍底图换成**真实影像**（USGS/NAIP，公有领域，
  需联网；来源说明见 §4）。换完底图要重新 `make-world` + 重启 SITL 才生效。

## 3. 两个世界与靶标排布

| 天井 | 轮次 | A 区内容 | B 区内容 |
| --- | --- | --- | --- |
| 第 1~2 座 | `r1` | 机枪兵(1)、坦克(7) | 多旋翼(3)、直升机(8) |
| 第 3 座 | `r1` / `r2` | 轰炸机(12) / 两位数 **56** | 侦察机(11) / 两位数 **61** |
| 第 4 座 | 都空 | 空 | 空 |

`r2` 里 A 区三个数是 94 / 12 / 56（中位数 56，落在 A3），B 区是 38 / 70 / 61（中位数 61，
落在 B3）——正好对上规则任务二的"打中位数靶标"；`examples/sitl_mission.py` 的演练航线
终点与合成目标就是照着 A3（56 号）来的。

**天井的位置与朝向都是随机的**（规则 3.3），布局由 `--seed` 决定：默认 seed 0 = 入库布局，
换种子会同时换一批位置与朝向（用来验证识别/转正对随机摆放的鲁棒性）。
摆放有两条硬约束：两两间距 > 20m（规则），**中心离区边界至少 8m**——后者是为了让
规则 3.3.3 的 r=6m 有效打击圈整圈落在目标区内（圈外还留 2m 余量）。
入库布局（seed 0，ENU 米 / 度）：

| 天井 | A 区（蓝）坐标 | 朝向 | B 区（红）坐标 | 朝向 |
| --- | --- | --- | --- | --- |
| 1 | (213.11, 6.91) | 339.5° | (-199.49, 20.44) | 253.6° |
| 2 | (178.01, -13.99) | 109.0° | (-178.18, 13.77) | 336.8° |
| 3 | (200.30, -10.80) | 146.9° | (-191.93, -15.22) | 186.2° |
| 4（空） | (180.89, 15.83) | 291.6° | (-221.78, 4.20) | 250.9° |

演练航线（`examples/sitl_mission.py`）里的经纬度常量就是照上表 A3 = (200.30, -10.80)
写的：`tests/test_world.py` 会把它换算回 NED 与入库世界对一遍，世界改布局就会红。

两个轮次入库时用的是**同一套天井布局**（同一个 seed；现实中每轮由组委会重新公布场地，
所以模拟"两轮布局不同"时给 `--seed` 分别生成即可——注意会覆盖入库的世界文件）。

关于"天井箭头"：规则正文 3.3.1/3.3.2 写了"靶标方向、天井箭头方向均随机"、数字靶方向
要与箭头一致，但**规则图 2 / 图 4 里并没有画这个箭头**；本世界默认不画（需要的场合用
`python -m airdrop.run make-world --well-arrows` 打开）——"靶标方向随机"这件事由天井
朝向本身承担。

## 4. 资产（网格与贴图）

文本网格由生成器写出（可逐字比对）：

* `materials/meshes/pentagon_plate.obj`：五边形靶板（1m 方形 + 60° 顶角，厚 20mm）。
  同一网格也被当作天井底面（z 方向放大成 60mm 的加强板）。
* `materials/meshes/pentagon_well_wall.obj`：天井五边形环壁（内轮廓 = 靶板轮廓，
  外轮廓外扩 5mm；高 400mm；顶/底环 + 内外侧面，含法线）。
* `materials/meshes/sheet_*.obj` + `.mtl`：靶纸平面（1m² 单位面）。**UV 约定：图像上方
  → +x、图像左侧 → +y**——即"数字正读方向 = 顶角方向 = 箭头方向"（用离屏渲染核对过：
  在贴图四边画红/蓝条，红条出现在 +x 侧、蓝条出现在 +y 侧）。

贴图 PNG 是入库的二进制资产：

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `pic_01…pic_12_*.png`（600×600） | 官方 12 张图片靶标（机枪兵/火箭兵/多旋翼/固定翼/卡车/防空炮/坦克/直升机/战斗机/运输机/侦察机/轰炸机） | 规则附件《固定翼无人机侦察与打击靶标》PDF 里抽出的剪影，裁到剪影包围盒、等比居中放进 60×60 靶纸 |
| `digit_0…digit_9.png`（300×600） | 数字板，白底黑字、字高 400/600 | 一次性脚本用加粗无衬线字体渲染（黑像素占比 21%~36%，即白底黑字） |
| `ground_asphalt…ground_gravel.png`（512×512，6 张） | 目标区"底色"：6 种常见路面/地面色（沥青/旧水泥/混凝土/土面/砖面/砂石），**纯色 + 少量同色系纹理**（污渍/颗粒/淡裂纹） | `python -m airdrop.run make-backdrops`（程序化，可复现；`--seed` 换一批） |
| `aerial_1…aerial_6.png`（512×512，6 张） | 比赛区域外的"随机航拍干扰底图"：农田/商区/住宅/工业/林地/湿地 | 默认同上（程序化）；也可 `python -m airdrop.run fetch-aerial` 换成真实航拍影像——**USDA NAIP via USGS National Map，公有领域**（160m/张、约 0.3m/像素），入库请保留这句来源说明 |

图片靶素材只用于本地仿真迭代；要对外再分发请自行确认附件素材的授权。
`ground_*` / `aerial_*` 两张底图集合是**同一组文件名**由两个命令写：`make-backdrops`
（离线、程序化）与 `fetch-aerial`（联网、真实影像）；谁后跑谁生效，想让程序化生成
也覆盖不到真实航拍图就用 `make-backdrops --parts ground`。

## 5. 起 SITL（WSL / Linux）

世界文件必须出现在 PX4 的 worlds 目录里（`px4-rc.gzsim` 按
`${PX4_GZ_WORLDS}/${PX4_GZ_WORLD}.sdf` 取文件），而世界的资产（网格/贴图）靠
`GZ_SIM_RESOURCE_PATH` 找到我们自己的目录——两者都要有：

```bash
# 一条命令（WSL 里，先自动装机型/世界/airframe，再起）：
bash sim/run_sitl.sh r2                 # 第二轮世界 + 带下视相机机型（默认）
HEADLESS=1 bash sim/run_sitl.sh r2      # 无 GUI
PX4_DIR=/path/to/PX4-Autopilot bash sim/run_sitl.sh r2   # PX4 不在默认位置

# 等价的手动步骤：
bash sim/install_px4.sh                 # 幂等：软链机型/世界/airframe 进 PX4
export GZ_SIM_RESOURCE_PATH="$PWD/sim/worlds/cuadc:$GZ_SIM_RESOURCE_PATH"
cd "$HOME/PX4-Autopilot"
PX4_GZ_WORLD=cuadc_recon_strike_r2 make px4_sitl gz_rc_cessna_down_cam
```

机型与 airframe **都在本仓库**里（`sim/vehicles/rc_cessna_down_cam` +
`sim/airframes/4007_gz_rc_cessna_down_cam`），由 `sim/install_px4.sh` 软链进 PX4——
PX4 树里不再存手工副本；装机细节、可选补丁与还原办法见 [`sim/README.md`](../sim/README.md)。

机型默认用 **`gz_rc_cessna_down_cam`**：机身下挂 1280×720@30Hz、水平视场 1.74 rad 的
相机（merge PX4 模型库里的基础 `rc_cessna` + 本仓库的 `camera_720p` 子模型）。
相机话题由 Gazebo 按作用域命名，本地实测（r2 世界）：

```text
/world/cuadc_recon_strike_r2/model/rc_cessna_down_cam_0/link/camera_link/sensor/imager/image
/world/cuadc_recon_strike_r2/model/rc_cessna_down_cam_0/link/camera_link/sensor/imager/camera_info
```

⚠ **实测记录（2026-09）**：该话题在出图（1280×720 原始 RGB，可用 `gz.msgs.Image`
订阅）。**在地面上抓帧会看到天空——那不是相机朝向错**（相机链接的世界姿态是 +90° 俯仰，
朝向已核对正确；同世界另加独立相机也能看到跑道/天井），而是机体停在地面时相机贴地
**穿模**，越过了地面平面（世界是单面 plane）看到了地面以下的背景色/天空。**在空中取图
正常**；要在地面上验证画面，先把飞机起飞（或临时把机体抬高几米）。

验收：GUI 里能看到跑道、两个目标区与 8 座天井；`gz topic -l | grep '/world/cuadc'`
能看到 `/world/cuadc_recon_strike_r2/clock`。无头环境加 `HEADLESS=1`（PX4 只起 server）。

### 5.1 视频流：PX4 自带的 GstCameraSystem（官方机制，无需自己搭）

PX4 的 `gz_bridge/server.config` 给**每个世界**自动加载一个 GStreamer 相机系统插件
（`src/modules/simulation/gz_plugins/gstreamer/`；本地实测 gz server 进程里已加载
`libGstCameraSystem.so`）。按官方源码与插件 README：

* 插件**自动发现世界里的第一台相机**（本项目 = `rc_cessna_down_cam` 的下视相机），
  用 GStreamer 编码成 **RTP/H.264 推到 `127.0.0.1:5600`**（默认值）；
* **QGC 侧**：Application Settings → Video → Video Source = `UDP h.264 Video Stream`、
  UDP Port = `5600` 即可看画面（WSL 里的 QGC 开箱即用——它是官方的默认接收端配置）。
  ⚠ 别把 Video Source 选成 TCP 类的源（`TCP-MPEG2 Video Stream`）再填 `0.0.0.0:5600`：
  本地实测 QGC 会**启动即闪退**；真闪退了就改回 ini（Windows：
  `%APPDATA%\QGroundControl\QGroundControl.ini`）里的 `videoSource=UDP h.264 Video Stream`
  + `udpUrl=0.0.0.0:5600` 即可恢复（TCP 那条 `tcpUrl` 删掉，它的默认值是空）；
* **想让 Windows 的 QGC 收**：起仿真前设 `PX4_VIDEO_HOST_IP=<主机的局域网 IP>`（例如
  192.168.1.18；插件源码里这个环境变量优先于默认的 `127.0.0.1`）——
  `PX4_VIDEO_HOST_IP=192.168.1.18 bash sim/run_sitl.sh r2`；
* 两条硬约束：**同一时刻只能有一个 QGC 监听 5600**（mirrored 模式下 WSL 与 Windows 共享
  端口空间，WSL 的 QGC 开着时 Windows 的收不到）；**Windows 侧要放行入站 UDP 5600**
  （防火墙默认拦 QGC 的入站）——
  `New-NetFirewallRule -DisplayName "QGC video UDP 5600" -Direction Inbound -Protocol UDP -LocalPort 5600 -Action Allow`（管理员）。
* 不用 QGC 时的官方查看命令（插件 README）：
  `gst-launch-1.0 udpsrc port=5600 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink`

SDF 里还可以给 `udpHost` / `udpPort` / `rtmpLocation` / `useCuda`（插件参数）；本项目
不改 PX4 源码，所以换目的地用环境变量那条。官方出处：PX4 文档 “Gazebo Simulation →
Video Streaming”（UDP 5600 / RTP）与插件自带的 README
（`src/modules/simulation/gz_plugins/gstreamer/README.md`）。

⚠ 与本项目其它约定一致：同一时刻只允许一个 `mavsdk_server`（固定占 gRPC 50051）；
Windows 侧跑演练用 `python -m airdrop.run sitl`（`examples/sitl_mission.py` 的航线
已对准本赛区），先用 QGC 或遥控手动起飞。

**自动侦查精度测试**：`python -m airdrop.run sitl-recon`（`tools/sitl_recon.py`）——
自动上传「起飞 + 扫掠」任务并**先解锁、再 `MISSION_START`**，只用 Gazebo 自带的
RTP/H.264 视频流（`127.0.0.1:5600`，不必用 HM30），跑完给"解算坐标 vs 天井中心"的误差
报告。装机细节、参数约定（`MIS_TKO_LAND_REQ` / `NAV_DLL_ACT`）与三条实测教训见
[`../sim/README.md`](../sim/README.md)。

## 6. 重建 / 换场景

```bash
# 入库的两个世界（默认 seed 0、静风）
python -m airdrop.run make-world
# 只生成第二轮、换一批天井位置与朝向、给 5 m/s 东 + 2 m/s 北 的恒定风
python -m airdrop.run make-world --rounds 2 --seed 3 --wind 5,2,0
# 生成到别的目录（贴图会一起复制过去，够跑 SITL）
python -m airdrop.run make-world --out-dir /tmp/scenario_7
# 换一套阴影环境：东侧低角度侧光（方位角,仰角；详见 §2.1）
python -m airdrop.run make-world --sun 90,12
# 需要"天井箭头"的场合（规则文本有、规则图里没有，默认不画）
python -m airdrop.run make-world --well-arrows
# 不铺周边航拍底图 / 铺 12 块
python -m airdrop.run make-world --aerial-patches 0
python -m airdrop.run make-world --aerial-patches 12
# 重新生成底图：程序化（离线）/ 真实航拍（联网，USGS NAIP，公有领域）
python -m airdrop.run make-backdrops
python -m airdrop.run fetch-aerial
# 只想重生目标区地面纹理、别动抓来的真实航拍图
python -m airdrop.run make-backdrops --parts ground
# 快速看一眼当前布局（俯视预览图，带中文标注；只读世界文件，不改任何东西）
python -m airdrop.run preview-world
```

生成器只写文本（SDF/网格/材质）；落盘前会自检 XML 合法、世界引用的网格与材质、
材质引用的贴图都齐全，缺什么就报什么并返回 1。`tests/test_world.py` 钉住
"生成结果 == 入库文件"，所以**改了生成器就要重跑 `make-world` 并提交生成结果**。

## 7. 已知边界（别当成已验证）

* 不是实物复刻：天井壁厚 5mm、底面 60mm、跑道长度是按"够 SITL 用 + 规则文本"取的工程值；
  规则里没写的尺寸（天井内净空、板厚）以实现口径为准。
* 不计分、不判命中：模拟弹没有实体模型，规则 3.3.3 的"精确/有效打击区"（r=4m/6m）
  只是**规则图 5 里的示意**，场地上没有这两个圈（见 §1 表）；
  投放精度由包内弹道与判据评估（SITL 演练用 `DryRunController`，只记日志）。
* 风默认静风。要考风：`python -m airdrop.run make-world --wind 5,2,0` 重新生成（改的是世界
  `<wind>` 元素），再重启 SITL——生成器里没有留运行时改风的开关（这个版本的 Gazebo
  server 只有 `<wind>` 静态配置，没有对应的 wind 主题）。
* 光照（太阳位置）同理是**静态配置**：换阴影环境要 `--sun` 重新生成 + 重启 SITL，
  没有运行时改太阳的主题（见 §2.1）。
* 天井的位置与朝向是随机的（规则 3.3），本世界做成"由 `--seed` 复现"：**入库布局是 seed 0**。
  换种子会换一批天井位置——`examples/sitl_mission.py` 的航线常量是按 seed 0 写的
  （`tests/test_world.py` 会核验"合成目标落在入库世界的 A3 上"），换种子后要同步改航线。
* "天井箭头"默认不画：规则正文 3.3.1/3.3.2 提到它（方向随机、数字靶要与它对齐），
  但规则图 2/图 4 里没有画，所以默认按"图里没有"来（`--well-arrows` 可以打开）。
* 目标区底色与比赛区域外的航拍底图是**抗扰用的随机项**，不在规则里：入库的是
  程序化生成的底图（离线可复现）；换成 `fetch-aerial` 抓的真实影像（USGS/NAIP，
  公有领域）后请按 §4 保留来源说明。真实影像只是"背景素材"，不参与任何几何/规则约束。
* 感知只在离线回放（2024v2 素材）与真实相机链路上验证过；**把 Gazebo 相机接进
  `airdrop.perception` 还没做**——世界这边已就位：默认机型 `gz_rc_cessna_down_cam`
  在出图（1280×720@30Hz，朝向正确；贴地时相机穿模看到背景色，空中正常），
  且 **PX4 自带的 GstCameraSystem 会把第一台相机自动推到 UDP 5600 的 RTP/H.264**
  （QGC 直接能看，见 §5.1）——离线链路将来可以从这条流取图（例如 ffmpeg 读
  `udp://127.0.0.1:5600`，未验证）。
* 要飞无相机的机身就传第二个参数：`bash sim/run_sitl.sh r2 gz_rc_cessna`。
