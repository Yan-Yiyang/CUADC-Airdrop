# 标定与仿真工具的数据契约

标定计算位于独立包 `airdrop_calibration`，仿真配置和生成位于独立包 `airdrop_sim`。
两者通过文件与运行包交换数据，不导入 `airdrop`；
`airdrop/runtime/` 中的 SITL 适配 是主仓的集成协调器，允许使用运行包并消费工具产物。
主仓通过 `vendor/airdrop-sim` 和 `vendor/airdrop-calibration` 的 submodule 固定提交。
初始化、独立运行和升级见 [工具仓库](tool_repositories.md)。

## 飞行目录 → 标定

工具从飞行目录读取 `frames_index.jsonl`、`frames/` 和 `telemetry.jsonl`。
字段名兼容主仓录制器；未知遥测字段忽略，缺测量保持 `None`，不以零补齐。

| 输入 | 字段 | 单位和约定 |
| --- | --- | --- |
| 帧索引 | `filename` | 相对 `frames/` 的路径；拒绝越界路径和越界符号链接 |
| 帧索引 | `received_timestamp` | 收到画面的原始时刻，秒；标定的主要时间轴 |
| 帧索引 | `capture_timestamp`、`lag` | 录制时推定的拍摄时刻与当时配置延时，秒；供残差互核 |
| 遥测 | `timestamp` | 本地墙钟的快照组装时刻，秒；只用于整理记录，不作为姿态采样时刻 |
| 遥测 | `north_m`、`east_m`、`down_m` | 局部 NED 位置，米；缺少任一位置分量时不构造完整机体位姿 |
| 遥测 | `quaternion_w/x/y/z` | 机体 FRD → 局部 NED，顺序为 wxyz；缺失姿态不补零 |
| 遥测 | `quaternion_timestamp_us`、`position_timestamp_us` | 各字段自己的飞控启动时标，微秒；用于字段间内插 |
| 遥测 | `quaternion_received_at`、`position_velocity_received_at` | 字段原始接收时刻，秒；四元数接收时刻是在线与标定统一的参照 |

遥测中的坏 JSON、非对象、缺时间戳及非有限测量行会告警并跳过；缺少日志文件明确失败。
帧索引缺少 `received_timestamp` 会失败，不能悄悄改用推定拍摄时刻。坏图像连同它自己的时间记录
一起跳过。读取与报告由工具自己的 DTO 和模块负责，不依赖运行包的快照或 `FlightLog` 类。

标定生产入口按四元数源时标去重，以 `quaternion_received_at` 测量延迟；手眼位姿按位置与四元数各自源时标查询。合并快照接收时刻混合字段属于已确认 bug，不能作为标定参照。缺少原始时标或跨重启、断流的记录须重新采集连续片段。

## 标定或模型参数 → 相机文件

`camera_calib.json` 保留运行包已支持的顶层字段：

| 字段 | 约定 |
| --- | --- |
| `camera_matrix` | 3×3 内参矩阵，像素；`image_size` 为 `[width, height]` |
| `dist_coeffs` | OpenCV 顺序的无量纲畸变系数；仿真相机为零 |
| `R_bc` | OpenCV optical（右、下、前）→ 机体 FRD（前、右、下） |
| `t_bc` | 光心在机体系中的位置，米 |
| `telemetry_lag` | 帧接收时刻相对四元数接收参照的有效对齐量，秒；无有效估计时为 null |

标定报告在 `meta` 中记录版本、来源、单位、坐标约定及内参/时间差/手眼质量信息。
`t_bc` 的权威值仍为尺量值，手眼估计杆臂用于对表。来源标注必须区分实际标定结果与
`CameraConfig` 推导的模拟参数；模型推导值不代表实测相机或链路已经标定。
运行包读取既有顶层字段，新增元数据不改变旧文件的读兼容。

## 仿真配置 → SDF、相机文件与 manifest

`CameraConfig` 是尺寸、水平视场、安装角、杆臂、裁剪距离和帧率的参数来源。
内参使用方形像素：`fx = fy = width / (2 tan(hfov / 2))`。
`VehicleConfig` 记录基础模型 URI 和 `base_link` 的父坐标偏移。
相机 include 的平移在合并模型坐标下表示，不能直接把 FRD 杆臂写入 SDF。
默认基础机型的 `base_link` 高度偏移为 0.246 m；它是配置假设，换基础模型时必须重新核对。

Gazebo link 为前、左、上；图像 optical 为右、下、前。安装姿态、轴转换共同推导 `R_bc`，
视觉外壳和惯性中心的偏移不当作光心。已有 SDF 的 pitch `1.5707` 是 π/2 的近似；新生成器使用
配置的完整精度，默认机型以关键 XML 结构、三轴和杆臂作兼容检查，角容差为 1e-4 rad。
默认世界与网格仍逐字匹配入库产物，随机数调用顺序和既有 seed/layout 保持兼容。

每个世界输出对应的 `.manifest.json`；机型输出 `simulation.manifest.json`。
schema v1 包含工具版本、seed、轮次、ENU/WGS84 原点、全部天井真值、相机配置、机型配置及相对产物路径。
世界真值由生成 SDF 的同一布局函数导出，报告不再维护另一张手写坐标表。
清单只列本次实际生成的产物；世界生成不等于同时生成了基础飞机模型。
读取器验证版本、字段类型、有限数值、坐标约定、目标编号和配置一致性；产物路径相对于 manifest 所在目录。

```bash
python -m airdrop.run make-world --rounds 2 --seed 0 --out-dir output/world
python -m airdrop.run make-vehicle --out-dir output/vehicle
python -m airdrop.run make-vehicle --width 800 --height 600 --hfov 1.2 --out-dir output/custom
```

各工具仓的 CLI 分别位于 `airdrop_sim/cli.py` 与 `airdrop_calibration/cli.py`；主仓旧命令只转发，工作流模块不创建解析器。库调用可以直接使用
`WorldConfig.world_texts()`、`VehicleConfig.write(output_dir)`、`CameraConfig.write_calibration(path)`。
默认随包资产按模块位置定位；显式相对输入/输出路径相对于调用方当前目录。
这些库入口不要求安装运行包，但仍需要各自用到的第三方库与源资产。

## 主仓演练消费

`mission --mode sitl --manifest PATH` 固定本次相机配置、原点与真值，随后报告和延时扫描使用同一份场景，
不修改其它演练的全局参数。省略 manifest 时仍使用 r2/seed 0。省略 `--lag` 时使用相机配置中的延时；
显式 `--lag` 覆盖本次延时。图片靶标 manifest 不用于数字识别精度报告。

内置侦查航线仍是固定翼 r2/seed 0 的路径。改变布局、视场或安装位置后先核验其覆盖；
内置航线无法覆盖时会在创建运行产物和连接飞控前失败，需通过 Python 调用显式提供匹配的 `route`。
非默认 WGS84 原点还必须显式提供匹配的 `land_plan`。相机或世界生成并不自动规划飞行航线。
离线契约测试验证格式和几何，不替代 Gazebo、PX4、QGC 或真实设备验收。
