# 相机标定

离线三步标定位于 [airdrop-calibration](../vendor/airdrop-calibration/README.md)，算法与 OpenCV 说明见工具仓的 [专题](../vendor/airdrop-calibration/docs/calibration_opencv.md)。

主仓保留 `python -m airdrop.run mission --mode real --no-mission-control --no-recognize --record`，录制标准飞行目录。独立工具读取目录并输出 `camera_calib.json`，生产相机读取器消费该文件。字段、时间轴和坐标约定见 [数据契约](tool_contracts.md)，初始化与独立运行见 [工具仓库](tool_repositories.md)。

## 延迟参照与已确认缺陷

把异步位置、姿态按合并快照的接收/组装时间当作同时刻，是已确认的对齐 bug。正式配置默认启用 `TelemetryConfig.source_time_alignment`；关闭它只用于复现旧缺陷，不能作为正确的标定或定位路径。

在线和标定统一使用四元数的原始 `quaternion_received_at` 建立接收时刻到飞控源时标的局部映射，位置与四元数按各自源时标内插。标定第 2 步按源时标去重四元数，用其原始接收时刻计算姿态序列；第 3 步在同一映射下配对位置与姿态。合并快照的 `timestamp` 只表示记录组装时刻，不能代替四元数接收参照。

`telemetry_lag = d_video − d_quaternion_reference` 是相对于该参照的有效对齐量。标定用帧原始 `received_timestamp` 测绝对量，手眼标定和在线处理都只在 `received_timestamp − telemetry_lag` 查询一次；录制时的 `capture_timestamp` 只用于“录制基准 + 残差 ≈ 绝对量”互核。它不等于独立测得的视频单向传输延迟，也不证明精确曝光同步。

标定文件的 `meta.time_offset.telemetry_reference` 为 `quaternion_received_at`，`pose_alignment` 为 `source_timestamp_us`。缺少源时标、跨重启或四元数断流的记录明确拒绝，重新采集连续片段；原接收快照路径测得的旧数值不能直接沿用。

实飞识别装配在创建视频源和记录器之前，将该文件的 `telemetry_lag` 写入本次 `VideoConfig`，并校验上述参照；参照未知的文件要求重新标定。无延迟估计时保留显式视频配置。SITL 使用本次 manifest/`--lag`，回放使用录制帧的原时间轴，不被新相机文件覆盖。

记录器的 `RecordConfig.telemetry_hz` 默认约 10 Hz，会少于在线原始姿态样本；同一种延迟定义不代表逐包重现。标定采集应提高该频率、核对实际保存的独立位置/四元数源样本覆盖和间隔，单独报告采样分辨率；仅提高配置值也不保证完整保留原始流。变更接收链路、遥测参照或传感器配置后，重新标定并用独立素材验收。
