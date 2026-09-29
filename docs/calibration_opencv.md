# 相机标定

本文档说明三步标定流程（内参 → 画面/遥测时间差 → 手眼外参）的采集要求、验证基线与
OpenCV 5.0 相关的 API 约束。相关代码：`tools/calibrate.py`。
修改标定逻辑或 `georef` 外参前请先阅读本文档。

## 采集要求

- 棋盘格固定不动，手持飞机在其上方平移 + 旋转（"板子不动、飞机在动"）。
- 使用 `examples/calibration_capture.py` 录制标准飞行目录，再执行
  `python -m airdrop.run calibrate --flight <飞行目录>`。

## 流水线

1. **内参**：多视图棋盘格 → `calibrateCamera`（输出 `K`、畸变系数与重投影 RMS）；
2. **画面-遥测时间差**：棋盘格 PnP 角速度 × 飞控角速度互相关 → `telemetry_lag`（回填配置）；
3. **手眼外参**：求解 `A X = X B` 得到 `R_bc`；`t_bc` 以尺量为准，标定估计值写入报告
   供对比校验（室内手持没有 GPS，平移分量不可信）。

`tests/test_calibrate.py` 包含完整的 `calibrate()` 输出契约用例：脚本生成标定飞行目录
（渲染棋盘格帧 + 与真值外参自洽的遥测，遥测首末各多写一条以覆盖帧区间——`telemetry_pose_at`
拒绝外推），并断言输出 JSON 中 `t_bc` 恒为尺量值、估计值与差值写入 `meta.lever_arm`。
合成链路的验证基线：

| 项 | 实测 |
| --- | --- |
| 内参重投影 RMS | 0.055 px（fx 901.1 / 真值 900） |
| 画面/遥测时间差 | Δt ≈ 0 |
| 运动对数量 | 430 |
| 旋转残差 | 0.215° |
| 杆臂估计 | `[0.0584, -0.0315, 0.0836]`（真值 `[0.06, -0.03, 0.09]`，差 3 mm） |

## OpenCV 5.0 相关约束

### 问题分类

排查 OpenCV 5.0 相关问题时先分类，不要一概归因于绑定标记：

- **(a) 绑定标记丢失**——唯一症状为"`cv2` 中没有该属性"。5.0 之后属于此类的是手眼两个函数；
- **(b) 5.0 的行为/位置变更**——函数存在，但类型、模块或后端拆分发生变化：
  FFmpeg 后端拆分至插件 DLL（见 [`video_rtsp.md`](video_rtsp.md)）、
  `VideoCapture::get()` 不支持时返回 −1、DNN 更换引擎；
- **(c) 既有约定**——`calibrateCamera` 只接受 float32、`moments` 不接受 float64 等，
  并非 5.0 引入的变化。类别 (b)(c) 均不会表现为"属性消失"。

### `cv2.calibrateHandEye` 不可用（替代方案已内置）

OpenCV 5.0 将 `calib3d` 拆分为 `geometry`/`calib`/`stereo`/`ptcloud`，函数迁移至
`modules/calib`，C++ 声明与签名未变，但绑定标记为 `CV_EXPORTS` 而非 `CV_EXPORTS_W`，
因此 Python 绑定未生成（`cv2/__init__.pyi` 中存在 `CALIB_HAND_EYE_*` 等常量，唯独没有函数）。

- **`opencv-contrib-python` 无法解决**（contrib 使用同一份主仓源码），且会扰动
  torch/ultralytics，不应安装。
- 上游已修复但尚未进入发行版；**升级后用一行即可判断**：
  `hasattr(cv2, "calibrateHandEye")`。5.1 的 wheel 发布后，可用官方函数交叉验证自研解法
  （同一 `A X = X B` 约定；`calibrateRobotWorldHandEye` 需要 `(3,1)` 列向量）。
- 在官方函数可用之前，第三步使用内置的 `solve_hand_eye`。

## 手眼方程

记棋盘格固定位姿 `C = T_world_board`、`G = T_world_body`（来自遥测）、
`P = T_cam_board`（来自 PnP）、待求 `X = T_body_cam`。由 `G·X·P = C` 得
`A_ij = G_j⁻¹G_i`、`B_ij = P_jP_i⁻¹`，方程为 **`A X = X B`**（与 `cv::calibrateHandEye`
约定一致）。**`A` 必须来自独立的遥测位姿**。

两个关键数值结论：

1. **构造方向**：`G = P⁻¹X⁻¹` 时 `max|AX−XB| = 6.7e-16`，解到机器精度；
   若写成 `G = P⁻¹X`，`max|AX−XB| = 0.95`（两种构造互为反向方程，容易混淆）。
2. **零空间符号**：`vec(R)` 与 `vec(−R)` 表示同一旋转，但 `det(−R) = −1`。
   若不按行列式整体取反而直接做"翻转最小奇异值"式正交化，会得到 `R·diag(−1,−1,1)`
   （精确相差 180°，且取决于 SVD 符号而时对时错）。

## PnP 侧要求

以下三点均会导致"重投影残差很小但位姿相差 180°"的结果：

- `render_board` 的 `to_board` 必须减去一个 `tile`：棋盘格第一个内角点位于模板像素
  `(tile, tile)`，不是 `(0,0)`。缺少该偏移会使渲染棋盘格整体错一格，检测角点落在
  "板子真实点 +1 格"处（正面视图误差 248 px、姿态精确相差 180°，而 PnP 残差仍仅
  0.19 px）。使用 `pattern_phase_ok` 校验（正确渲染为 1.000）。
- **棋盘格标签二义性单帧无解**：绕板面法向旋转 180° 后图像完全一致，两种"0 号角点"
  分配均可拟合至亚像素，方格相位均为 1.000。两支位姿仅相差一个常量右乘，
  全局使用同一支时 `AX=XB` 的解完全相同，混用才会失败。
  `solve_board_poses` 依据相邻帧连续性（与上一帧夹角最小）定支，不应直接使用 `solvePnP`。
- `_base_look_down_pose` 的图像 `v` 轴必须与板子 `+y` 同向，否则合成图为镜像、
  角点标签整体反序（4 个视图全部相差 180°）。`look_at_board` 的扰动必须以旋转矩阵
  右乘施加，不能加在旋转向量上（`|rvec|` 接近 π 时向量相加会剧烈跳变）。

## 类型与数据约定

- **`calibrateCamera` 只接受 float32**（4.x 亦然）：objectPoints/imagePoints 任一为
  float64 都会抛 `objectPoints should contain vector of vectors of points of type Point3f`。
  `detect_corners` 的 SB 路径返回 float64，`run_intrinsics` 必须显式转换。
  `solvePnP` 与 `moments` 也属于"只接受特定类型"的既有约定。
- **配对不得按索引**：遥测 10 Hz、画面 30 Hz，应按帧时刻重采样
  （`align_body_poses`：位置线性插值、姿态 slerp，仅在范围内插值）。
- **`lag_s` 符号**：`estimate_lag_by_correlation` 返回的 `lag > 0` 表示画面滞后，
  即该帧内容对应的真实时刻比 `capture_timestamp` 早 `lag` 秒 ⇒ 查询遥测应使用
  `capture_timestamp − lag_s`。

## 杆臂 `t_bc`

- **以尺量为准，标定估计值仅用于对比校验**：平移方程
  `(R_a − I)t_x = R_x t_b − t_a` 依赖遥测位置 `t_a`，室内手持场景没有 GPS，位置不可信。
  标定文件中的 `t_bc` 恒为文件顶部的尺量常量 `MEASURED_T_BC`；估计值
  （`estimate_body_translation=True`，默认开启）连同与尺量值的差值写入
  `meta.extrinsics.lever_arm` 供核对。
- **判据基于残差而非数值大小**：`t_a` 无论真伪都能解出 `t_x`，假 `t_a` 的表现是方程
  自身不自洽。`t_bc_reliable` 要求平移方程残差 `residual_translation_m` ≤
  `max(0.02 m, 0.05 × 画面平移跨度)`（`lever_arm_tolerance`）。合成对照：位置正确时
  残差 `4e-14`；位置置零（无 GPS 的典型情形）时残差 1.81 m、估计偏差 0.125 m，
  而旋转解不受影响。诊断量 `body_span_m`（遥测位置跨度）与 `camera_span_m`
  （PnP 位置跨度）可用于定位原因。
- `estimate_body_translation=False` 时 `t_bc=None`、`t_bc_reliable=False`。
- **`lever_arm_report` 的 `measured` 默认值必须为 `None`**，不能使用
  `= MEASURED_T_BC`：后者是定义时绑定，用户修改文件顶部常量后 payload 与报告会不一致。

## 激励与渲染要求

- **激励**：至少需要 2 个不平行的旋转轴（与 OpenCV 文档一致），阈值
  `MIN_AXIS_SPREAD_DEG = 15°`；所有运动对绕同一根轴时解不唯一，`run_extrinsics`
  会显式报错。采不到旋转时应重新采集，不应调整阈值。
- **棋盘格渲染**：必须使用"深底 + 亮格"标准极性，且模板四周不留边距——留边距会使
  背景与边格同色、棋盘边界不可辨，`findChessboardCorners` 直接检不出
  （而 `checkChessboard` 仍返回 True）。
- `pattern_phase_ok` 仅用于发现"整体错格"这类重投影误差不敏感的几何错误，
  不能消除标签二义性。
- **`cv2.Rodrigues` 的约定**：其输入为世界→相机（相机轴作为行向量）；
  相对于手推旋转矩阵，建议使用"构造已知投影 + `solvePnP` 反解"。

## 不应采用的实现

- 依赖方格相位区分正序/反序标签；
- 将 `A` 构造为"棋盘格相对机体"的相对运动（等同于用 `X` 自身构造 `A`，`A ≡ B`，
  真值不在解空间中）；
- 使用 C 序 reshape 拆分零空间向量（会得到另一个形似旋转矩阵的错误解）。
