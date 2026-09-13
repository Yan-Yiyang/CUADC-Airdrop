# AirDrop Python 控制程序

当前阶段实现 MAVSDK 的三个基础能力：

1. **独立线程运行 MAVSDK**：MAVSDK 的 asyncio/gRPC 协程全部运行在专用后台线程的独立事件循环中，避免污染主线程或其他协程。
2. **实时遥测 + 跨进程桥接**：
    - *进程内访问*：同一进程内通过线程安全的 `TelemetryBroker` 读取最新快照或订阅推送。
    - *跨进程访问*：通过 `ExternalBridge` 把快照共享给其他进程（`multiprocessing.Manager` dict），并让其他进程通过队列向 MAVSDK 线程发指令。
    - *时间估计*：根据已有离散遥测快照计算任意时间戳的估计值；历史范围内使用**内插**，超出历史范围使用**外推**。
3. **跨线程/跨进程指令**：任何线程/进程都可以发送 `Command`，由 MAVSDK 线程串行执行。
4. **思翼 HM30 图传拉流**：采集线程读到的**每一帧都交给 sink**（一帧不丢，供侦查与事后分析），实时读取（`read()` / `latest()`）另外只看当前一帧；自带断流重连，不需要额外依赖。
5. **画面与遥测的时间对齐**：拿到一帧后扣掉链路固定延时（默认 150ms）得到真实拍摄时刻，再按那一刻取遥测快照，消除"用当前姿态解释 150ms 前画面"的系统性偏差。
6. **对齐结果缓冲**：拉流源读到的**每一帧**都经 sink 对齐后写进线程安全环形缓冲（默认 30 fps × 3 分钟 = 5400 帧），供其他模块按自己的节奏回看/异步消费——消费端再慢也不会丢帧。

## 目录结构

```text
airdrop/
    __init__.py
    models.py           # TelemetrySnapshot / Command / CommandResult
    broker.py           # 进程内：线程安全遥测仓库与订阅
    mavsdk_thread.py    # MAVSDK 专用线程 + 遥测采集 + 指令分发
    external_bridge.py  # 跨进程：快照共享 + 指令队列
    video.py            # 思翼 HM30 图传拉流（ffmpeg 子进程）
    alignment.py        # 帧 ↔ 遥测时间对齐（收到时间 - 链路延时 → 取遥测）
    buffer.py           # 对齐结果环形缓冲（画面 + 遥测，供其他模块回看）
examples/
    basic_usage.py      # 线程内读取/订阅 + 指令示例
    external_process.py # 子进程读取共享快照 + 发指令示例
    hm30_video.py       # HM30 图传：拉流 / 统计 / 预览 / 存盘
    video_telemetry_sync.py  # 帧-遥测对齐 + 入缓冲 + 读者线程示例
tests/
    conftest.py         # 公共 fixture：本地 H.264 测试流、broker、拉流源工厂
    test_telemetry.py   # MAVSDK 基础设施（快照/插值/线程/跨进程桥接）
    test_alignment.py   # 帧-遥测时间对齐（注入确定性历史时间戳）
    test_buffer.py      # 对齐结果缓冲 + AlignmentWriter 接线
    test_video.py       # 图传拉流（本地 UDP H.264，无需硬件；标记 stream）
```

## 安装

项目使用 `uv` 管理依赖：

```bash
uv sync
```

## 快速开始

示例不带命令行参数：地址、分辨率、开关等都在各自文件顶部的"配置"区，改完直接运行。
默认连接 SITL 或真机（UDP 入站 `udpin://0.0.0.0:14540`，兼容旧写法 `udp://:14540`）：

```bash
python -m examples.basic_usage        # 遥测订阅/轮询 +（可选）arm/takeoff/land 演示
python -m examples.external_process   # 子进程读取共享快照并下发指令
python -m examples.hm30_video         # HM30 图传拉流与链路统计
python -m examples.video_telemetry_sync  # 帧-遥测对齐 + 环形缓冲
```

`examples/basic_usage.py` 里的 `DEMO_COMMANDS` 打开后会依次发 arm → takeoff → land，
**只在安全环境（桨已拆除 / SITL）里用**。

## 核心用法

```python
from airdrop import Command, MavsdkThread, TelemetryBroker

broker = TelemetryBroker()
mavsdk = MavsdkThread(broker=broker, system_address="udpin://0.0.0.0:14540")
mavsdk.connect()

# 同进程其他线程读取
snapshot = broker.get_snapshot()
print(snapshot.as_dict())

# 同进程订阅推送
def on_snapshot(snap):
    print(snap.yaw_deg)

broker.subscribe(on_snapshot)

# 任何线程向 MAVSDK 线程发送指令
result = mavsdk.send_command(Command("arm"))
print(result.success, result.error)
```

## 连接流程与选项

`MavsdkThread.connect()` 遵循 MAVSDK 官方 Python 用法，按顺序执行：

1. `System.connect(system_address=...)` 建立 gRPC 连接；
2. 等待 `core.connection_state().is_connected` 确认飞控上线；
3. 可选等待 `telemetry.health()`（`is_global_position_ok` / `is_home_position_ok`）就绪；
4. 查询 NED 原点（`GPS_GLOBAL_ORIGIN`）并写入快照；
5. 启动各遥测流与 NED 原点后台刷新任务。

任一遥测流异常/结束时，会按配置自动重连。常用参数：

```python
mavsdk = MavsdkThread(
    broker=broker,
    system_address="udpin://0.0.0.0:14540",
    connect_timeout=30.0,        # System.connect 与等待连接状态的超时
    wait_for_health=False,       # 连接后是否等待 health 就绪（默认关）
    health_timeout=30.0,
    require_health_for_actions=False,  # arm/takeoff 前是否强制等 health
    origin_refresh_interval=5.0, # NED 原点刷新周期（秒）
    reconnect=True,              # 遥测流断开后自动重连
    reconnect_delay=5.0,
)
mavsdk.connect()
print(mavsdk.connected)          # 是否已连接并就绪
```

## 按时间戳查询

代理会保存最近一段历史快照（默认 `TelemetryBroker(history_maxlen=1200, history_interval=0.1)`，即按 10 Hz 采样、约覆盖 2 分钟；`history_interval=0` 表示每次遥测更新都写入历史）。历史写入是节流的，但最新快照与订阅推送始终实时。其他模块给出一个时间戳时，可以获取“那一刻”的状态：

```python
# 默认：历史范围内线性内插，超出范围线性外推
snap = broker.get_snapshot_at(1710000000.123)
print(snap.as_dict())

# 也可以只取时间上最近的一条历史快照
snap = broker.get_snapshot_at(1710000000.123, mode="nearest")
```

跨进程也可以通过指令队列查询：

```python
from airdrop import Command

result = mavsdk.send_command(Command(
    "get_snapshot_at",
    {"timestamp": 1710000000.123, "mode": "interpolate"},
))
print(result.success, result.data["snapshot"])
```

注意：时间戳单位是秒，并与 `TelemetrySnapshot.timestamp` 使用同一时间轴（默认是 `time.time()` 本地墙钟时间）。

`mode` 只接受 `"interpolate"`（默认）或 `"nearest"`，其余取值会抛出 `ValueError`，
错误信息会列出可选值。该校验在 `broker` 层与 `get_snapshot_at` 指令入口两处生效，
因此跨进程发指令时非法 `mode` 也会得到明确提示。

跨进程：

```python
from airdrop import ExternalBridge

bridge = ExternalBridge(mavsdk, broker=broker, publish_hz=20.0)
bridge.start()

# 把 bridge.snapshot / bridge.command_queue / bridge.result_queue
# 作为参数传给 multiprocessing.Process 即可。
```

## 遥测快照字段

`TelemetrySnapshot` 包含：

- **世界坐标**：`latitude_deg` / `longitude_deg` / `absolute_altitude_m` / `relative_altitude_m`
- **HOME 点 GPS 坐标**：`home_latitude_deg` / `home_longitude_deg` / `home_absolute_altitude_m` / `home_relative_altitude_m`
- **本地 NED 原点 GPS 坐标（GPS_GLOBAL_ORIGIN）**：`origin_latitude_deg` / `origin_longitude_deg` / `origin_altitude_m`
- **本地坐标（NED）**：`north_m` / `east_m` / `down_m`
- **速度（NED）**：`vx_m_s` / `vy_m_s` / `vz_m_s`
- **姿态（欧拉角）**：`roll_deg` / `pitch_deg` / `yaw_deg`
- **姿态（四元数）**：`quaternion_w` / `quaternion_x` / `quaternion_y` / `quaternion_z`

## PX4 NED 原点

注意：**HOME 点不一定等于本地 NED 坐标系原点**。

- PX4 的本地 NED 原点由 `vehicle_local_position.ref_lat / ref_lon / ref_alt` 表示。
- MAVSDK 中对应的是 `telemetry.get_gps_global_origin()`，返回 `GpsGlobalOrigin`。
- `telemetry.home()` 返回的是 HOME 点，二者通常接近，但可能在飞控初始化后、飞机移动或用户手动改 HOME 后不一致。

本项目已自动获取 NED 原点并写入 `origin_*` 字段，同时保留 `home_*` 字段：

```python
snap = broker.get_snapshot()
print("HOME:", snap.home_latitude_deg, snap.home_longitude_deg)
print("NED 原点:", snap.origin_latitude_deg, snap.origin_longitude_deg, snap.origin_altitude_m)
```

## 支持指令

| 指令名 | 参数 |
| --- | --- |
| `arm` | - |
| `arm_force` | - |
| `disarm` | - |
| `takeoff` | - |
| `land` | - |
| `hold` | - |
| `kill` | - |
| `rtl` / `return_to_launch` | - |
| `set_takeoff_altitude` | `altitude_m` |
| `set_return_to_launch_altitude` | `relative_altitude_m` |
| `set_current_speed` | `speed_m_s` |
| `get_snapshot_at` | `timestamp`（秒）、`mode`（`interpolate`/`nearest`） |
| `get_gps_global_origin` | - |

## 思翼 HM30 图传

HM30 地面端不"推"视频给上位机，而是把机载以太网 `192.168.144.0/24` **透明桥接**到
LAN 口 / 内置 WiFi / USB WiFi 上（这三者同时承载 UDP 遥测）。官方 FAQ 的说法是：
任何输出 RTSP、能接以太网的 IP 相机都可以配 HM30。所以"拉流"就是打开一条 RTSP。

**接线**

1. 网线插 HM30 **地面端** LAN 口；
2. 电脑网卡配静态地址 `192.168.144.20` / 掩码 `255.255.255.0`；
3. 确认能 ping 通机载相机 `192.168.144.25`。默认 RTSP
   `rtsp://192.168.144.25:8554/main.264` 是 SIYI 云台/相机系列的约定地址；若用
   HDMI-Ethernet 输入转换模块或第三方 IP 相机则不同。

**地址不猜、不探测**：地址写错时不会静默重试，而是把 ffmpeg 的输出原样作为错误抛出来
（错误日志 + `stats.last_error`）。想手动确认地址，直接用 ffmpeg 命令行试一次最清楚：

```bash
ffmpeg -rtsp_transport udp -i rtsp://192.168.144.25:8554/main.264 -frames:v 1 -f null -

python -m examples.hm30_video     # 拉流并打印帧率 / 丢帧 / 重连 / 延迟
```

**代码用法**

```python
from airdrop import Hm30VideoSource, VideoConfig

# 尺寸在配置里给（默认 720p）；本模块不探测源分辨率
with Hm30VideoSource(VideoConfig(width=1280, height=720)) as source:
    while (frame := source.read(timeout=1.0)) is not None:
        results = model(frame.image)   # YOLO / OCR 直接吃 BGR ndarray
```

**每帧都不丢：sink**

`read()` / `latest()` 是**实时**路径——只保证你看到的是当前一帧，慢消费者会丢掉中间帧
（计入 `stats.dropped`）。但本项目的要求是"**读到的每一帧都要留下来**"：目标出现的
时间可能极短，漏一帧就可能漏掉目标；而"先侦查后空投"允许一定的处理延时，不允许丢帧。

所以"留存"走另一条路：采集线程读出的**每一帧都会交给注册的 sink**，一帧不落。

```python
from airdrop import AlignmentBuffer, AlignmentWriter, FrameTelemetryAligner, capacity_for

buffer = AlignmentBuffer(capacity_for(30, 180))          # 30 fps × 3 分钟 = 5400 帧
writer = AlignmentWriter(buffer, FrameTelemetryAligner(broker, lag=0.15))

with Hm30VideoSource(VideoConfig(width=1280, height=720)) as source:
    source.add_sink(writer)          # 从这一帧起，读到的每一帧都进缓冲
    ...
```

sink 在采集线程里同步执行，所以必须快（缓冲写入只做一次 JPEG 编码，几毫秒）；
它抛异常只记日志，不会打断拉流。`AlignmentWriter.written` / `skipped` 分开计数：
只有"对齐取不到遥测"才会跳过画面，**不会因为消费端慢而丢帧**。

`source.stats` 可读到 `fps` / `dropped` / `reconnects` / `last_error`，断流后自动按
指数退避重连（`dropped` 只统计实时路径没跟上的帧，与缓冲无关）。

**为什么只有 ffmpeg 一个后端**

ffmpeg 后端还完全掌握低延迟 flag，且子进程能被父进程直接杀掉——断流判定与 `stop()`
都不会被卡住。细节见 `AGENTS.md` 的"关键约定与坑"。

## 画面与遥测的时间对齐

图传链路从"相机曝光"到"本进程读出一帧"之间有约 **150ms** 的固定延时（H.264 编码 +
无线传输 + 解码 + 管道缓冲）。直接用收到帧的时刻去查遥测，等于拿"现在的姿态"解释
"150ms 前的画面"——飞机以 10 m/s 平飞时，这就是 1.5 米的位置偏差，而且它会随速度
变化伪装成"算法不准"。

所以每拿到一帧，先把它**收到的时间减掉这段延时**，得到画面真正的拍摄时刻，再按
那一刻向 `TelemetryBroker` 查询遥测（默认在历史之间线性内插）：

```python
from airdrop import FrameTelemetryAligner, Hm30VideoSource, TelemetryBroker, VideoConfig

broker = TelemetryBroker()
aligner = FrameTelemetryAligner(broker)          # 默认扣 150ms

with Hm30VideoSource(VideoConfig()) as source:
    while (frame := source.read(timeout=1.0)) is not None:
        sample = aligner.align(frame)            # 帧 + 它拍摄时刻的遥测
        if sample is None:                       # 遥测历史为空，或超出外推保护
            continue
        if sample.extrapolated:                  # 那一刻的遥测是外推来的，慎用
            continue
        aim_at(sample.image, sample.snapshot.latitude_deg, sample.snapshot.yaw_deg)
```

延时是**链路属性**而不是消费者的选择，所以它定义在 `VideoConfig.telemetry_lag` 上、
由拉流源写进每一帧（`frame.capture_timestamp`）；对齐器默认直接采用帧上的值，也可以
用 `FrameTelemetryAligner(broker, lag=...)` 覆盖。等价的手写形式：

```python
snapshot = broker.get_snapshot_at(frame.capture_timestamp)   # = frame.timestamp - 0.15
```

几点注意：

- **延时是链路属性**：默认值 `DEFAULT_TELEMETRY_LAG = 0.15` 是 ffmpeg 后端的实测值。
  换相机、换传输方式、或标定出新值之后，只改 `VideoConfig.telemetry_lag` 一处，之后
  每一帧都会带上新值。
- **外推有风险**：查询时刻落在历史范围之外时，代理仍会线性外推并返回结果。
  `AlignedSample.extrapolated` / `offset` 如实标记了这一点；需要硬保证时给对齐器设
  `max_extrapolation`，超限的帧直接返回 `None`。
- **150ms 是经验值，不是精确测量**：标定方式是对着秒表拍摄，比较画面读数与本地时钟。
  链路抖动（毫秒级）要靠相机自己的 RTCP 时间戳解决，不在本模块范围内。
- **延时补偿 ≠ 实时性**：前者修的是"这帧是什么时候拍的"，后者说的是"你手里的画面
  有多新"。两者都要：延时补偿保证回看时遥测对得上画面，实时路径保证当前画面不积压。

运行示例（地址、延时、缓冲容量都在文件顶部配置区）：

```bash
python -m examples.video_telemetry_sync
```

## 对齐结果的缓冲

`AlignmentBuffer` 是本项目的**留存层**。为什么必须留存：目标出现的时间可能极短，
漏一帧就可能漏掉目标，而"先侦查后空投"的模式可以接受处理延时、不能接受丢帧。

写者是采集线程里的 `AlignmentWriter`（挂成拉流源的 sink），它把每一帧连同**拍摄
时刻**的遥测写进缓冲；读者（推理、落盘、回放、复盘）任意多个，各自按自己的节奏
取走。消费端再慢也不会让缓冲少一帧，最多是它自己的游标落后——这一点和实时路径
（`read()` / `latest()`，只看当前帧）互不影响。

```python
from airdrop import (
    AlignmentBuffer, AlignmentWriter, FrameTelemetryAligner, capacity_for,
)

buffer = AlignmentBuffer(capacity_for(30, 180))   # 30 fps × 3 分钟 = 5400 帧
writer = AlignmentWriter(buffer, FrameTelemetryAligner(broker, lag=0.15))

# 写者：挂在采集线程上，逐帧对齐并入缓冲（缓冲内部会拷贝，不存复用视图）
source.add_sink(writer)

# 读者（其他模块）：自己记游标，往前跟
index = 0
while running:
    record = buffer.wait_new(after_index=index, timeout=1.0)
    if record is None:
        continue
    index = record.index
    inference(record.image)               # 画面（BGR ndarray）
    print(record.snapshot.yaw_deg)        # 那一刻的遥测
    print(record.capture_timestamp, record.age)
```

`writer.written` / `writer.skipped` 用来核对"有没有丢"：`skipped` 只会因为"该时刻取不到
遥测"（历史为空或超过 `max_extrapolation`）而增加，**不会因为消费慢而增加**。

**容量与内存（先算账再选参数）**

默认容量 5400 帧（30 fps × 3 分钟）。未压缩 720p 单帧是 2.64 MiB，5400 帧 ≈
**13.9 GiB**——普通机器放不下内存，所以默认 `storage="jpeg"`（质量 80，约
0.1~0.2 MB/帧，满载 0.6~1 GiB）。要像素级无损就用 `storage="raw"`，但请同时设
`max_bytes` 兜底。入库第一帧时会打印一次按实测帧大小算出的满载估算。

| 存储 | 单帧（720p） | 5400 帧 | 精度 |
| --- | --- | --- | --- |
| `jpeg`（默认，q=80） | 约 0.1~0.2 MB | 约 0.6~1 GiB | 有损（看画面/找目标足够） |
| `raw` | 2.64 MiB | ≈13.9 GiB | 无损 |

**几条必须知道的语义**

- **入缓冲会拷贝**：ffmpeg 后端的帧是复用管道缓冲区的视图，下一帧到达时同一块内存
  会被覆盖。缓冲在 `jpeg` 模式下编码成独立字节串、`raw` 模式下显式 `copy()`；你自己
  保存 `frame.image` 时同样要 `.copy()`（或 `frame.copy()`）。
- **满时从最旧一端驱逐**，不是丢最新——丢最新等于让所有读者一起挨饿。`stats.evicted`
  记录累计驱逐数，`indices()` 给出当前保留的序号范围。
- **序号单调递增且不回退**，可以安全地当读者游标；若游标落后到记录已被驱逐，会直接
  返回最新一条（跳帧，但不会卡死）。
- **回看历史一律用 `iter_between()`**：它分批解码，内存占用与 `batch` 同阶、迭代期间
  不持锁，并且是快照式迭代（只覆盖开始那一刻已入库的记录）。要想一次拿全（5400 帧
  720p 就是 13.9 GiB）就自己 `list(...)`，那份内存账由调用方认。
- `raw` 模式读到的 `image` 是**只读视图**（零拷贝）；`jpeg` 模式是新解码的数组。
  需要就地画框请先 `record.copy()`。

## 测试

基于 **pytest**，全部离线（不需要真实飞控或图传硬件）：

```bash
./.venv/Scripts/python.exe -m pytest                    # 全部，附覆盖率报告
./.venv/Scripts/python.exe -m pytest -m "not stream"    # 跳过要起 ffmpeg 的用例
./.venv/Scripts/python.exe -m pytest -k alignment -v    # 只跑对齐相关
./.venv/Scripts/python.exe -m pytest --no-cov           # 不统计覆盖率
```

`tests/conftest.py` 提供公共 fixture：本地 H.264 测试流发送端（`sender`，可反复启停）、
"起流并等到首帧"的拉流源工厂（`live_source`，可在 `start()` 之前挂 sink）、
已就绪的 broker（`broker`）与订阅记录（`received`）。

覆盖内容：

## 安全提示

- 真机测试前务必在仿真环境验证。
- `arm` / `takeoff` / `kill` 等指令有安全风险，请自行增加确认/急停逻辑。
- 当前遥测快照是“最新可用值合并”，各 MAVSDK 数据流本身没有全局时间戳，因此不保证多个字段来自同一时刻；如需严格时间同步请扩展为按 `timestamp_us` 对齐。
