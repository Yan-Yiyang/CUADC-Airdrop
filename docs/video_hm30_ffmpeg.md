# HM30 图传：ffmpeg 后端约束

本文档说明：图传后端的选型依据，以及积压、丢帧和超时相关限制、
为什么不能退回 `cv2.VideoCapture`。改 `airdrop/video/source.py` 或排查"画面卡住/断流"时请先阅读本文档。

---

6. **HM30 图传只有 ffmpeg 一个后端，不要再引入 `cv2.VideoCapture`**：
   - OpenCV 5.0 把 FFmpeg 后端从 `cv2.pyd` 拆成了
     **插件 DLL**（`cv2/opencv_videoio_ffmpeg500_64.dll`；`cv2.pyd` 里只剩
     `OPENCV_VIDEOIO_PLUGIN_*`、`CAP_FFMPEG` 和 `OPENCV_FFMPEG_DLL_DIR`），当时是
      **grep 错了文件**。插件里这些都健在：`OPENCV_FFMPEG_CAPTURE_OPTIONS` /
     `_WRITER_OPTIONS` / `_DEBUG` / `_LOGLEVEL` / `_READ_ATTEMPTS` / `_THREADS` /
     `_IS_THREAD_SAFE`，以及 `rtsp_transport`、`rw_timeout`、`nobuffer`、`low_delay`、
     `max_delay`、`video_codec` 等选项键和日志串 "using capture options from environment"。
     **推论：5.0 之后"在 `cv2.pyd` 里搜不到某字符串"不再能证明它被移除**——
     视频 I/O 的东西要去插件 DLL 里搜。
   - 旧的本地对比样本观察到，受测 OpenCV FFmpeg 后端存在无法通过公开缓冲选项消除的内部排队。
     这只描述样本所用构建与链路，不构成跨平台的积压、延迟或丢帧保证。`read()` 可能阻塞在
     C 层调用；本项目采用 ffmpeg 子进程，使停止时能终止该进程，也避免为逐帧 sink 再解码一次。
   - UDP 上游可能丢失数据，且不一定反映在 `stats.dropped` 中。不可 seek 的 UDP 流在格式探测期间
     消耗的数据也可能不会进入输出；影响取决于 `probesize`、帧大小与流特征。旧的本地样本不作为
     通用验收证据；减小探测数据量也可能使个别流无法完成格式探测。
   - `CAP_PROP_READ_TIMEOUT_MSEC` 必须走 params 形式
     （`VideoCapture(url, api, params)` / `open(url, api, params)`）传入；用 `set()` 设
     不进去，超时行为会退回后端默认值，可能使断流后的读取继续阻塞。
   - ffmpeg 的 UDP 断流处理使用协议层 `-timeout <微秒>`；在本项目验证过的 UDP 路径里，
     `-rw_timeout` 不能替代它。具体超时受协议和后端影响；缺少有效超时控制时，管道读可能
     长时间阻塞，`stop()` 需依赖终止子进程收尾。
   - **本地文件类输入不能加 `-timeout`**：`Hm30VideoSource` 按 URL 里有没有 `://` 决定加不加
     ——SITL 演练走的是本地 **SDP 文件**，加上去 ffmpeg 直接
     `Option timeout not found` 打不开输入（见 `airdrop/video/source.py` 的建命令处注释）。
   - 子进程能被父进程直接杀掉（OpenCV 的 `read()` 是 C 层调用、外部无法中断），
     这是 `stop()` 永远不会被阻塞的管道读卡住的根本原因。
   - **不探测流地址、也不探测分辨率**：地址错就把 ffmpeg 的输出原样当错误抛出
     （一帧都没收到时用 error 级别，见 `Hm30VideoSource._fail`），由使用者核对；
     输出尺寸由 `VideoConfig.width/height` 显式给出（默认 720p）。"自动探测/多地址轮询"
     只会把明显的配置错误变成难以定位的静默行为，别再往回加。
   - HM30 与 SIYI 相机地址属于部署网络配置，应按现场网络设置 `VideoConfig.url`。
