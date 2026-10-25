# 相机标定

离线三步标定位于 [airdrop-calibration](../vendor/airdrop-calibration/README.md)，算法与 OpenCV 说明见工具仓的 [专题](../vendor/airdrop-calibration/docs/calibration_opencv.md)。

标定采集使用 `python -m airdrop.run mission --mode real --no-mission-control --no-recognize --record --calibration-capture`，录制标准飞行目录。此处 `real` 指真机遥测/视频输入，不表示起飞；在地面手持飞机绕多轴拍摄固定标靶即可采集内外参与视频延迟。关闭任务控制和识别。主旋转与延迟不依赖位置；存在可用的位置源时标时，独立工具会额外做平移杆臂对表。工具读取目录并输出 `camera_calib.json`，生产相机读取器消费该文件。字段、时间轴和坐标约定见 [数据契约](tool_contracts.md)，初始化与独立运行见 [工具仓库](tool_repositories.md)。

离线分析默认覆盖完整采集过程，不再截取前 200 帧或前 60 个有效内参视图。工具逐帧解码并缓存原精度角点与时间记录，所有有效角点共同求解内参；相位和位姿分析按需重读当前帧，不保留整段原始图像。检测分辨率、亚像素精度和相位判据保持不变。全量采集的完整性验收还会单独逐帧核验图像。

```powershell
.\.venv\Scripts\python.exe -m airdrop.run calibrate --flight "flights/本次目录" --out camera_calib.json --pattern 9,6 --square 0.025 --strict
```

`--max-frames` 仅用于显式调试前缀，省略即分析全程。`meta.frame_analysis` 记录索引帧数、实际分析帧数、接收时间范围、有效内参视图数和位姿数，便于核对覆盖范围。角点、位姿和遥测等紧凑数据仍随样本量增长，像素图像只保留当前处理所需的少数帧。

## 延迟参照与已确认缺陷

把异步位置、姿态按合并快照的接收/组装时间当作同时刻，是已确认的对齐 bug。正式配置默认启用 `TelemetryConfig.source_time_alignment`；关闭它只用于复现旧缺陷，不能作为正确的标定或定位路径。

在线和标定统一使用四元数的原始 `quaternion_received_at` 建立接收时刻到飞控源时标的局部映射，位置与四元数按各自源时标内插。标定第 2 步按源时标去重四元数，用其原始接收时刻计算姿态序列；第 3 步只用同一映射插值四元数旋转，不依赖位置。合并快照的 `timestamp` 只表示记录组装时刻，不能代替四元数接收参照。

`telemetry_lag = d_video − d_quaternion_reference` 是相对于该参照的有效对齐量。标定用帧原始 `received_timestamp` 测绝对量，手眼标定和在线处理都只在 `received_timestamp − telemetry_lag` 查询一次；录制时的 `capture_timestamp` 只用于“录制基准 + 残差 ≈ 绝对量”互核。它不等于独立测得的视频单向传输延迟，也不证明精确曝光同步。

标定文件的 `meta.time_offset.telemetry_reference` 为 `quaternion_received_at`，`pose_alignment` 为 `source_timestamp_us`。缺少源时标、跨重启或四元数断流的记录明确拒绝，重新采集连续片段；原接收快照路径测得的旧数值不能直接沿用。

地面标定帧独立按收到时刻记录，帧索引 `telemetry_aligned=false` 表示没有同帧遥测；四元数由独立全量遥测日志提供。主手眼旋转外参与延迟只使用姿态，因此缺失或异常的位置时间轴不会改变它们。默认会额外尝试用位置源时标对表平移杆臂，`--no-estimate-translation` 可关闭；此诊断固定主流程已求得的旋转，仅输出估计值、尺量值、逐轴差与差值距离供人工判断。标定文件顶层 `t_bc` 始终是尺量值。位置缺失/异常或平移方程不可解时，报告会说明跳过原因。若主手眼求解失败，报告 `R_bc=null`，运行包会拒绝将该文件识别为已标定相机。

实飞识别装配在创建视频源和记录器之前，将该文件的 `telemetry_lag` 写入本次 `VideoConfig`，并校验上述参照；参照未知的文件要求重新标定。无延迟估计时保留显式视频配置。SITL 使用本次 manifest/`--lag`，回放使用录制帧的原时间轴，不被新相机文件覆盖。

专用 `mission --calibration-capture` profile 将记录器设为逐更新记录，并将四元数请求提高到至少 100 Hz、保持欧拉角 30 Hz；位置请求频率保持配置值。PX4 会按链路预算用 `rate_mult` 降低实际频率；这些请求不会修改飞控链路参数。报告应核对实际四元数源样本数量、频率和间隔。源时间戳由 `time_boot_ms * 1000` 得到，精度为 1 ms 的发布时标，不代表传感器曝光或采样的精确时刻。

全量记录的完整性只覆盖本进程已接纳的回调；帧入库前编码或复制失败也会计入 `recorder_stopped.frame_encode_errors`，非零时拒绝完整采集。操作时还要检查 `mavsdk_server` 是否报告 `User callback queue overflown`。`quaternion_received_at` 是 Python 收到更新时打戳，包含 SDK/gRPC 排队。高频采集的延迟估计不能保证在正式任务负载与速率下原样成立；须以正式任务速率及视频、显示和识别负载独立复验，并报告请求/实际频率与接收抖动。飞行中的弹道识别只需可靠记录投放事件附近的状态，不应因此提高全航次遥测记录频率。
