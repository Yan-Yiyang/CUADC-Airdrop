# AirDrop Python 控制程序

当前阶段实现 MAVSDK 的三个基础能力：

1. **独立线程运行 MAVSDK**：MAVSDK 的 asyncio/gRPC 协程全部运行在专用后台线程的独立事件循环中，避免污染主线程或其他协程。
2. **实时遥测 + 跨进程桥接**：
    - *进程内访问*：同一进程内通过线程安全的 `TelemetryBroker` 读取最新快照或订阅推送。
    - *跨进程访问*：通过 `ExternalBridge` 把快照共享给其他进程（`multiprocessing.Manager` dict），并让其他进程通过队列向 MAVSDK 线程发指令。
    - *时间估计*：根据已有离散遥测快照计算任意时间戳的估计值；历史范围内使用**内插**，超出历史范围使用**外推**。
3. **跨线程/跨进程指令**：任何线程/进程都可以发送 `Command`，由 MAVSDK 线程串行执行。
4. **思翼 HM30 图传拉流**：按"最新一帧"语义拉取 RTSP 画面，供后续视觉引导（YOLO + OCR）使用；自带断流重连，不需要额外依赖。
5. **画面与遥测的时间对齐**：拿到一帧后扣掉链路固定延时（默认 150ms）得到真实拍摄时刻，再按那一刻取遥测快照，消除"用当前姿态解释 150ms 前画面"的系统性偏差。
6. **对齐结果缓冲**：把（画面 + 拍摄时刻遥测）写进线程安全环形缓冲（默认 30 fps × 3 分钟 = 5400 帧），供其他模块按自己的节奏回看/异步消费。

## 目录结构

```text
airdrop/
    __init__.py
    models.py           # TelemetrySnapshot / Command / CommandResult
    broker.py           # 进程内：线程安全遥测仓库与订阅
    mavsdk_thread.py    # MAVSDK 专用线程 + 遥测采集 + 指令分发
    external_bridge.py  # 跨进程：快照共享 + 指令队列
    video.py            # 思翼 HM30 图传拉流（ffmpeg / OpenCV 两条后端）
    alignment.py        # 帧 ↔ 遥测时间对齐（收到时间 - 链路延时 → 取遥测）
    buffer.py           # 对齐结果环形缓冲（画面 + 遥测，供其他模块回看）
examples/
    basic_usage.py      # 线程内读取/订阅 + 指令示例
    external_process.py # 子进程读取共享快照 + 发指令示例
    hm30_video.py       # HM30 图传：探流 / 预览 / 统计 / 存盘
    video_telemetry_sync.py  # 帧-遥测对齐 + 入缓冲 + 读者线程示例
tests/
    run_all.py          # 跑全部离线测试
    test_telemetry.py   # MAVSDK 基础设施（8 组）
    test_alignment.py   # 帧-遥测时间对齐（8 组，注入确定性历史）
    test_buffer.py      # 对齐结果缓冲（9 组）
    test_video.py       # 图传拉流（8 组，本地 UDP H.264，无需硬件）
```

## 安装

项目使用 `uv` 管理依赖：

```bash
uv sync
```

## 快速开始

连接 SITL 或真机（默认 UDP 入站 `udpin://0.0.0.0:14540`，兼容旧写法 `udp://:14540`）：

```bash
# 仅查看遥测
python -m examples.basic_usage --address udpin://0.0.0.0:14540

# 演示 arm -> takeoff -> land（仅在安全环境使用）
python -m examples.basic_usage --address udpin://0.0.0.0:14540 --demo-commands

# 演示跨进程读取共享快照和发送指令
python -m examples.external_process --address udpin://0.0.0.0:14540
```

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
3. 先探流确认地址——默认 `rtsp://192.168.144.25:8554/main.264` 是 SIYI 云台/相机
   系列的约定地址，若用 HDMI-Ethernet 输入转换模块或第三方 IP 相机则不同。

```bash
# 地址不确定时先探流（可重复传 --url 试多个候选）
python -m examples.hm30_video --probe
python -m examples.hm30_video --probe --url rtsp://192.168.144.25:8554/main.264 \
                               --url rtsp://192.168.144.25:8554/sub.264

# 拉流并打印帧率 / 丢帧 / 重连 / 画面延迟
python -m examples.hm30_video

# 开窗口预览（按 q 退出）
python -m examples.hm30_video --preview

# 缩放到 960x540（管道带宽降到 1/4），并用 NVDEC 硬解
python -m examples.hm30_video --width 960 --height 540 --decoder h264_cuvid
```

**代码用法**

```python
from airdrop import Hm30VideoSource, VideoConfig

with Hm30VideoSource(VideoConfig(width=960, height=540)) as source:
    while (frame := source.read(timeout=1.0)) is not None:
        results = model(frame.image)   # YOLO / OCR 直接吃 BGR ndarray
```

语义与 `TelemetryBroker` 一致：后台线程**只保留最新一帧**，慢消费者丢旧帧而不是
积压（积压等于给无人机看历史照片）。`source.stats` 可读到 `fps` / `dropped` /
`reconnects` / `last_error`，断流后自动按指数退避重连。

**为什么默认用 ffmpeg 子进程而不是 `cv2.VideoCapture`**

| 后端 | 断流后积压 | 画面滞后 |
| --- | --- | --- |
| `ffmpeg`（默认） | 0 帧 | ≈0 ms |
| `opencv` | 17 帧 | ≈567 ms |

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

- **换后端要同步改延时**：`ffmpeg` 后端约 0.15s，`opencv` 后端约 0.57s（见上表实测）。
  用 `--backend opencv` 做对照时必须同时把 `telemetry_lag` 调大，否则对齐会系统地偏向未来。
- **外推有风险**：查询时刻落在历史范围之外时，代理仍会线性外推并返回结果。
  `AlignedSample.extrapolated` / `offset` 如实标记了这一点；需要硬保证时给对齐器设
  `max_extrapolation`，超限的帧直接返回 `None`。
- **150ms 是经验值，不是精确测量**：标定方式是对着秒表拍摄，比较画面读数与本地时钟。
  链路抖动（毫秒级）要靠相机自己的 RTCP 时间戳解决，不在本模块范围内。
- **延时补偿 ≠ 最新一帧**：前者修的是"这帧是什么时候拍的"，后者保证你看到的是当前
  画面而不是积压帧。两者都要，视觉引导才准。

运行示例：

```bash
python -m examples.video_telemetry_sync --address udpin://0.0.0.0:14540
python -m examples.video_telemetry_sync --lag 0.32 --duration 10
python -m examples.video_telemetry_sync --backend opencv --lag 0.57
```

## 对齐结果的缓冲

下游有两类相反的消费方式：实时引导只要"当前画面"（所以 `Hm30VideoSource` 只保留
最新一帧，慢消费者丢帧），而异步推理、落盘、回放、复盘需要**回看**刚过去的一段时间。
`AlignmentBuffer` 就是这两者的接缝：写者（拿帧的线程）把对齐结果压进去，任意多个
读者按自己的节奏取走，互不影响。

```python
from airdrop import AlignmentBuffer, FrameTelemetryAligner, capacity_for

buffer = AlignmentBuffer(capacity_for(30, 180))   # 30 fps × 3 分钟 = 5400 帧
aligner = FrameTelemetryAligner(broker)

# 写者（采集线程）：对齐后直接入缓冲，缓冲内部会拷贝，不存复用视图
sample = aligner.align(source.read(timeout=1.0))
if sample is not None:
    buffer.put(sample)

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
- **`between()` 会把命中帧全部解码进内存**（5400 帧 720p 就是 13.9 GiB）。批量消费用
  `iter_between()` —— 它分批解码，内存占用与 `batch` 同阶，且迭代期间不持锁。
- `raw` 模式读到的 `image` 是**只读视图**（零拷贝）；`jpeg` 模式是新解码的数组。
  需要就地画框请先 `record.copy()`。

## 测试

全部离线，不需要真实飞控或图传硬件：

```bash
# 跑全部
./.venv/Scripts/python.exe tests/run_all.py

# 单独跑某一模块
./.venv/Scripts/python.exe tests/test_telemetry.py
./.venv/Scripts/python.exe tests/test_alignment.py
./.venv/Scripts/python.exe tests/test_buffer.py
./.venv/Scripts/python.exe tests/test_video.py
```

测试是"纯 assert + 直接运行"的脚本，不依赖 pytest（项目尚未引入测试框架）。

## 安全提示

- 真机测试前务必在仿真环境验证。
- `arm` / `takeoff` / `kill` 等指令有安全风险，请自行增加确认/急停逻辑。
- 当前遥测快照是“最新可用值合并”，各 MAVSDK 数据流本身没有全局时间戳，因此不保证多个字段来自同一时刻；如需严格时间同步请扩展为按 `timestamp_us` 对齐。
