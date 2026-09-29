# 视频链路（RTSP）

本文档说明视频拉流、帧-遥测对齐与帧缓冲的实现要求、参数约定与边界行为。
相关代码：`airdrop/video/source.py`、`airdrop/video/align.py`、`airdrop/video/buffer.py`。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `source.py` | RTSP 拉流（ffmpeg 子进程），逐帧产出 BGR 图像 |
| `align.py` | 帧-遥测时间对齐：按拍摄时刻查询遥测快照 |
| `buffer.py` | 对齐结果环形缓冲：单写者、多读者 |

## 拉流要求

- 使用 **ffmpeg 子进程**读取 RTSP 流，通过 `-timeout`（微秒）实现断流判定；
  子进程可被父进程直接终止，拉流线程不会因管道读取而阻塞。
- 输出尺寸由 `VideoConfig.width/height` 显式指定（默认 720p）。
- **不自动探测流地址与分辨率**：地址配置错误时，将 ffmpeg 的输出原样上报
  （一帧未收到时以 error 级别记录，见 `Hm30VideoSource._fail`），由使用者核对。
  自动探测或地址轮询会把配置错误转化为难以定位的静默行为。
- `VideoConfig.url` 指定 RTSP 地址；`telemetry_lag` 为链路固定延时（见下）。

## 帧与遥测的时间对齐

- 每帧记录两个时间：收到时间与拍摄时间。拍摄时间 = 收到时间 − 链路延时。
- 链路延时 `VideoConfig.telemetry_lag`（默认 0.15s）覆盖 H.264 编码、无线传输、解码与
  管道缓冲的总和，属链路属性，标定后回填。
- 按拍摄时间查询遥测快照；查询时刻落在历史范围之外时标记为外推
  （`AlignedSample.extrapolated`），可设置 `max_extrapolation` 作为硬保护（超限返回空）。
- 帧自带 `frame.lag`；优先使用对齐器显式传入的 `lag=`。

## 逐帧留存与实时读取

视频源提供两条互不干扰的读取路径：

| 路径 | 接口 | 语义 |
| --- | --- | --- |
| 逐帧留存 | `add_sink(callback)` | 回调在采集线程中逐帧调用，一帧不丢；标准接法为 `AlignmentWriter(buffer, aligner)` |
| 实时读取 | `read()` / `latest()` | 仅保证"当前这一帧"；慢消费者会丢弃中间帧并计入 `stats.dropped` |

**推理与写入磁盘的输入应来自 `AlignmentBuffer`（`iter_between` / `wait_new`），
不得使用 `read()` 循环送入**。`AlignmentWriter.written` / `skipped` 可用于核对：
`skipped` 仅因"该时刻取不到遥测"增加，不因消费速度增加。

## 缓冲语义

`AlignmentBuffer` 为线程安全环形缓冲（单写者、任意多读者，读者各自持有游标）：

- 默认容量 `capacity_for(30, 180) = 5400` 帧；默认 `storage="jpeg"`
  （720p 约 0.1~0.2 MB/帧，满载约 0.6~1 GiB）；`storage="raw"` 无损（5400 帧约 13.9 GiB）。
- 溢出时从最旧一端驱逐；`max_bytes` 可设置字节上限。
- 读取接口：`latest` / `at` / `iter_between` / `wait_new`。`iter_between` 分批解码，
  迭代期间不持锁。
- **不得跨帧持有帧的 ndarray**：ffmpeg 后端的 `frame.image` 是复用缓冲区的视图，
  下一帧读入会覆盖同一内存；跨帧保留必须 `copy()`。

## 边界行为

- **消费者跟得上时不丢帧**：带帧号合成流实测 432/450 帧逐号连续（缺口仅在首、尾：
  首为启动阶段，尾为发送端退出时的少量帧）。
- **消费者跟不上时会整段丢帧**：此类丢帧发生在 UDP 套接字层、位于管道上游，
  不计入 `stats.dropped`。`stats.dropped == 0` 不能作为"网络未丢帧"的依据。
- **启动阶段帧丢失**：连接建立后，前 `≈ probesize / 每帧字节数` 帧不会进入 sink
  （ffmpeg 探测格式期间读走的字节对不可 seek 的 UDP 流是丢弃的）。默认
  `-probesize 500000` 对应 720p 约 0.5~2s；要求"连接即逐帧不丢"时应调小该值
  （个别流可能因此探测失败——此时按设计显式报错）。
- 对 UDP 输入，`-rw_timeout` 无效，断流判定应使用 `-timeout`；缺少该参数时管道读取
  会永久阻塞。

## 地址核对

配置 RTSP 地址后，可用 ffmpeg 手动拉取一帧验证连通性：

```bash
ffmpeg -rtsp_transport udp -i <RTSP 地址> -frames:v 1 -f null -
```
