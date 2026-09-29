# HM30 图传：只有 ffmpeg 一个后端（含逐项实测表）

本文档说明：图传选型的依据与全部实测——何时会积压/丢帧、超时参数的取值方式、
为什么不能退回 `cv2.VideoCapture`。改 `airdrop/video/source.py` 或排查"画面卡住/断流"时请先阅读本文档。
**内容逐字搬自 `AGENTS.md`"关键约定与易错点"第 6 条，与 AGENTS.md 同等权威。**
（下面正文保留原来的编号 `6.`——它对应 AGENTS.md 里那条约定的序号。）

---

6. **HM30 图传只有 ffmpeg 一个后端，不要再引入 `cv2.VideoCapture`**（本地 OpenCV 5.0 实测）：
   - ⚠ **更正（2026-09 复查）**：此前记的"OpenCV 5.0 已移除 `OPENCV_FFMPEG_CAPTURE_OPTIONS`
     （该字符串在 `cv2.pyd` 里都不存在）"**是错的**。5.0 把 FFmpeg 后端从 `cv2.pyd` 拆成了
     **插件 DLL**（`cv2/opencv_videoio_ffmpeg500_64.dll`；`cv2.pyd` 里只剩
     `OPENCV_VIDEOIO_PLUGIN_*`、`CAP_FFMPEG` 和 `OPENCV_FFMPEG_DLL_DIR`），当时是
      **grep 错了文件**。插件里这些都健在：`OPENCV_FFMPEG_CAPTURE_OPTIONS` /
     `_WRITER_OPTIONS` / `_DEBUG` / `_LOGLEVEL` / `_READ_ATTEMPTS` / `_THREADS` /
     `_IS_THREAD_SAFE`，以及 `rtsp_transport`、`rw_timeout`、`nobuffer`、`low_delay`、
     `max_delay`、`video_codec` 等选项键和日志串 "using capture options from environment"。
     **推论：5.0 之后"在 `cv2.pyd` 里搜不到某字符串"不再能证明它被移除**——
     视频 I/O 的东西要去插件 DLL 里搜。
   - **2026-09 重测（本地同一口径：切断发送端后还能读出多少帧）**：cv2 路线
     **17 帧 / 567ms**（当时的数字逐帧复现），本项目的 ffmpeg 子进程路线 **0 帧**。
     但当时给出的解释（"环境变量被移除、nobuffer/low_delay 传不进去"）**是错的**，
     而且当时的写法本身也是错的——格式是 ``键;值|键;值``，``"nobuffer;low_delay"``
     等于"选项 nobuffer = 值 low_delay"。逐项排除结果：

     | cv2 配置 | 断流积压 | 链路断开后 `read()` 阻塞 |
     | --- | --- | --- |
     | 默认 | 17 帧 / 567ms | **30.0 s** |
     | `OPENCV_FFMPEG_CAPTURE_OPTIONS=nobuffer;low_delay`（语法错） | 17 帧 | 30.1 s |
     | 正确语法 `fflags;nobuffer\|flags;low_delay` | 17 帧 | 30.1 s |
     | 再加 `analyzeduration;0\|probesize;500000`（项目那两条原样） | 17 帧 | 30.1 s |
     | `CAP_PROP_BUFFERSIZE=1` | 17 帧 | 30.1 s |
     | `buffer_size;8192`（缩小 UDP 接收缓冲） | 17 帧 | 30.1 s |
     | `threads;1`、`max_delay;0`、三者叠加 | 17 帧 | 30.1 s |
     | **params 形式传 `CAP_PROP_READ_TIMEOUT_MSEC=3000`** | 17 帧 | **3.06 s** ✓ |

     环境变量**确实生效**（DEBUG 日志实测打出 `using capture options from environment: …`），
     所以这 17 帧**不是**解复用/探测/UDP 套接字/解码线程的缓冲，而是 OpenCV FFmpeg
     后端的**内部排队深度**，没有任何用户可达的开关能改。
     **结论不变，理由要换**：选 ffmpeg 子进程的真实理由是——(1) 管道反压让接收侧保持
     浅流水（实测 0 帧积压）；(2) `read()` 是 C 层不可中断调用，卡住只能**杀进程**；
     (3) 断流超时要靠 ffmpeg 的 `-timeout`（微秒），而不是 cv2 的 params 形式；
     (4) 不必为"一帧不落"的写入磁盘，在 Python 侧再解一遍码。
     这两条实测结论也写进了 `airdrop/video/source.py` 的模块 docstring（那里原本同样记着
     错误解释，已改）。
   - **"0 帧积压"是否以丢包为代价：2026-09 用带 16 位条码的合成流验证过（发送端 numpy 逐帧
     画 8 个 4 级灰块 = 2bit/块，接收端解码帧号，缺号即丢帧）**：
     * **正常（消费者跟得上）：不丢** —— 低码率素材 282 帧连续无缺口、高码率噪声素材
       432 帧无缺口，缺口只出现在**首**（见下）和**尾**（发送端退出时的 3 帧）。
       所以"0 积压"是真的浅流水，不是靠丢帧换的。
     * **消费者跟不上（sink 睡 150ms ≈ 6.7fps）：会丢，而且是整段丢** —— 450 帧只交付
       76 帧，从 id 241 起整段消失。结合"断流积压 0 帧"可判定这些帧是**丢**不是**等**
       （根本没有队列可以等）。**而 `stats.dropped` 仍然是 0**——这类丢帧发生在 UDP
       套接字层、在管道上游，现有计数器**看不见**。UDP + 无缓冲下这是物理必然，
       cv2 的深缓冲也作用有限（只有 17 帧 ≈ 0.57s 余量）。
     * **新发现的"启动盲区"（需要特别注意的一条）**：连接建立后头 `≈ probesize / 每帧字节数`
       帧**永远进不了 sink**——ffmpeg 在探测格式期间读走的字节对不可 seek 的 UDP 流是
       丢弃的。实测 `-probesize 500000`：低码率（~3KB/帧）丢 **165 帧 ≈ 5.5s**；
       高码率噪声（~35KB/帧）丢 **15 帧 ≈ 0.5s**——两组都严格 ≈500KB ✓。
       真机 720p 大概对应 0.5~2s。任务开始前的那一段通常没有影响，但如果要求
       "接上就一帧不落"，就需要调小 `-probesize`（代价：个别流会探测失败——按项目
       原则它会**显式报错**，不会静默）。
   - `CAP_PROP_READ_TIMEOUT_MSEC` 必须走 params 形式
     （`VideoCapture(url, api, params)` / `open(url, api, params)`）传入；用 `set()` 设
     不进去，会退回 30 秒默认超时——链路一断拉流线程就卡死半分钟。
   - ffmpeg 断流判定靠 `-timeout <微秒>`（协议层 socket 超时，实测 2.7s 生效）；
     **`-rw_timeout` 对 UDP 输入无效**（12s 都不退出）。缺了 `-timeout`，管道读会
     永久阻塞，`stop()` 也退不掉。
   - 子进程能被父进程直接杀掉（OpenCV 的 `read()` 是 C 层调用、外部无法中断），
     这是 `stop()` 永远不会被阻塞的管道读卡住的根本原因。
   - **不探测流地址、也不探测分辨率**：地址错就把 ffmpeg 的输出原样当错误抛出
     （一帧都没收到时用 error 级别，见 `Hm30VideoSource._fail`），由使用者核对；
     输出尺寸由 `VideoConfig.width/height` 显式给出（默认 720p）。"自动探测/多地址轮询"
     只会把明显的配置错误变成难以定位的静默行为，别再往回加。
   - HM30 是透明以太网桥（`192.168.144.0/24`），SIYI 相机默认
     `rtsp://192.168.144.25:8554/main.264`；手动核对地址用 ffmpeg 命令行拉一帧即可。
