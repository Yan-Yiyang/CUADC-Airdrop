# AirDrop Python 控制程序

当前阶段实现 MAVSDK 的三个基础能力：

1. **独立线程运行 MAVSDK**：MAVSDK 的 asyncio/gRPC 协程全部运行在专用后台线程的独立事件循环中，避免污染主线程或其他协程。
2. **实时遥测 + 跨进程桥接**：
    - *进程内访问*：同一进程内通过线程安全的 `TelemetryBroker` 读取最新快照或订阅推送。
    - *跨进程访问*：通过 `ExternalBridge` 把快照共享给其他进程（`multiprocessing.Manager` dict），并让其他进程通过队列向 MAVSDK 线程发指令。
    - *时间估计*：根据已有离散遥测快照计算任意时间戳的估计值；历史范围内使用**内插**，超出历史范围使用**外推**。
3. **跨线程/跨进程指令**：任何线程/进程都可以发送 `Command`，由 MAVSDK 线程串行执行。

## 目录结构

```text
airdrop/
    __init__.py
    models.py           # TelemetrySnapshot / Command / CommandResult
    broker.py           # 进程内：线程安全遥测仓库与订阅
    mavsdk_thread.py    # MAVSDK 专用线程 + 遥测采集 + 指令分发
    external_bridge.py  # 跨进程：快照共享 + 指令队列
examples/
    basic_usage.py      # 线程内读取/订阅 + 指令示例
    external_process.py # 子进程读取共享快照 + 发指令示例
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

## 安全提示

- 真机测试前务必在仿真环境验证。
- `arm` / `takeoff` / `kill` 等指令有安全风险，请自行增加确认/急停逻辑。
- 当前遥测快照是“最新可用值合并”，各 MAVSDK 数据流本身没有全局时间戳，因此不保证多个字段来自同一时刻；如需严格时间同步请扩展为按 `timestamp_us` 对齐。
