# 标定与 OpenCV 5.0 易错点（`tools/calibrate.py`）

本文档说明：三步标定流水线（内参 → 画面/遥测时间差 → 手眼外参）的实测结论，以及围绕
OpenCV 5.0 的全部易错点与判据。修改标定逻辑、改 `georef` 外参、或怀疑"某个 cv2 函数没了"时请先阅读本文档。

---

- **`tools/calibrate.py`（P6 下半）= 三步标定流水线，已落地**（内参 → 画面/遥测时间差 →
  手眼外参）。代码与合成数据的回归契约见 `tests/test_calibrate.py`；它覆盖整个
  `calibrate()` 输出，包括 `t_bc` 使用尺量值、估计值与差值写入 `meta.lever_arm`。
  这些合成回归不代表真实素材标定精度；真实素材复测仍未完成。
  - **OpenCV 相关问题的三类分因（不要一概归因于 `CV_EXPORTS_W`）**：
    **(a) 绑定标记丢失**——症状是"`cv2` 里没有这个属性"；
    **(b) 5.0 的行为/位置变更**——函数在，但类型、模块或后端拆分变了：FFmpeg 后端拆成
    插件 DLL（见 [`video_hm30_ffmpeg.md`](video_hm30_ffmpeg.md)）、`VideoCapture::get()` 不支持时返回 −1、DNN 换引擎；
    **(c) 一直沿用的既有约定、被我们误记成"5.0 变了"**——如 `calibrateCamera` 只收 float32、
    `moments` 不收 float64。**(b)(c) 两类都不会表现为"属性消失"，别混为一谈。**
  - **`cv2.calibrateHandEye` 在 Python 侧不可用，但原因不是"被移除"**：OpenCV 5.0 把
    `calib3d` 拆成 `geometry`/`calib`/`stereo`/`ptcloud`，函数搬进 `modules/calib`，
    **C++ 声明与签名一字未改**，但 OpenCV 5.0.0 的绑定标记从 `CV_EXPORTS_W` 变成了
    `CV_EXPORTS`，于是 Python
    侧没生成。旁证：`cv2/__init__.pyi` 里 `CALIB_HAND_EYE_*` /
    `HandEyeCalibrationMethod` / `RobotWorldHandEyeCalibrationMethod` 都在，唯独没有函数。
    `opencv-contrib-python` **解决不了**（contrib 用同一份主仓源码，一样没有 `_W`），
    装了还会扰动 torch/ultralytics——**不要安装**。第三步是自己实现的 `solve_hand_eye`。
    - OpenCV 的 issue 记录了该绑定问题，修复已合并；
      报告 [issue #29565](https://github.com/opencv/opencv/issues/29565)
      "calibrateHandEye is missing in python package for version 5.0.0.93"
      （标为 `bug` 与 `category: python bindings`，milestone 5.1，已关闭）；修复
      [PR #29584](https://github.com/opencv/opencv/pull/29584) 将函数的绑定标记改回
      `CV_EXPORTS_W` 并补 Python 回归测试，4.x 的测试回移见
      [PR #29693](https://github.com/opencv/opencv/pull/29693)。可用性取决于安装的 OpenCV wheel；使用时应以当前
      环境中的 Python API 为准。自研 `solve_hand_eye` 使用同一 `A X = X B` 约定。
  - **手眼的方程与来源（唯一正确写法）**：棋盘格固定在 `C = T_world_board`，
    `G = T_world_body`（**遥测**）、`P = T_cam_board`（**PnP**）、`X = T_body_cam` 待求。
    由 `G·X·P = C` 得 `A_ij = G_j⁻¹G_i`、`B_ij = P_jP_i⁻¹`，方程 **`A X = X B`**
    （与 `cv::calibrateHandEye` 同一约定）。**`A` 必须来自独立的遥测位姿**。
  - **零空间符号**：`vec(R)` 与 `vec(−R)` 表示同一旋转，但 `det(−R) = −1`。
    正交化前必须按行列式统一符号，否则可能得到差 180° 的旋转；对应处理由
    `solve_hand_eye` 的回归用例覆盖。
  - **PnP 侧三个易错点（都会给出"重投影残差很小但位姿差 180°"的结果）**：
    - `render_board` 的 `to_board` 必须减一个 `tile`：棋盘格第一个**内角点**在模板像素
      `(tile, tile)` 处，不是 `(0,0)`。少了这个偏移，渲染出的棋盘格整体错**一整格**，
      检测角点落在"板子真实点 +1 格"处，可能造成位姿错误而重投影残差仍然较小；
      用 `pattern_phase_ok` 检查棋盘格相位。
    - **棋盘格的标签二义性无解**：绕板面法向转 180° 后图像**完全一样**，两种"哪个角点是
      0 号"的分配都能拟合到亚像素，方格相位也无法区分——单帧图像里没有
      任何信息能定支。好在两支位姿只差一个**常量右乘**，`B'_ij = P'_jP'_i⁻¹ = B_ij`，
      **只要全局同一支，`AX=XB` 的解完全相同；混用才会失败**（残差几十度）。
      `solve_board_poses` 用**相邻帧连续性**（与上一帧夹角最小）定支，不要退回直接使用 `solvePnP`。
    - `_base_look_down_pose` 的图像 `v` 轴必须与板子 `+y` **同向**：反过来合成图成了镜像，
      角点标签整体反序。另外 `look_at_board` 的扰动必须
      **旋转矩阵右乘**，不能加在旋转向量上（`|rvec|` 接近 π 时"向量相加"会剧烈跳变）。
  - **`calibrateCamera` 只收 float32——这不是 5.0 的变化**：实测 objectPoints/imagePoints
    任一为 float64 都抛 `objectPoints should contain vector of vectors of points of type
    Point3f`；查 4.x 源码（PR #13803 的 diff，`modules/calib3d/src/calibration.cpp`）里
    写的是同样的 `checkVector(3, CV_32F)`，所以 4.x 也收不了 float64。
    真正的问题出在**我们自己**的代码：`detect_corners` 的 SB 路径返回 float64，
    `run_intrinsics` 必须显式转 float32，否则内参这步直接失败
    （`solvePnP` 则 float32/float64 都收；`moments` 同属此类既有约定，只收 int32/int64/float32）。
  - **杆臂 `t_bc` 以尺量为准，标定估计值只用于对比校验**：
    `(R_a − I)t_x = R_x t_b − t_a` 需要 `t_a`（遥测位置），室内手持没有 GPS，不可信。
    所以标定文件里的 `t_bc` **恒**为文件顶部的尺量常量 `MEASURED_T_BC`（georef 读的就是
    它），估计值（`estimate_body_translation=True`，默认开）连同"与尺量值差多少"一起写进
    `meta.extrinsics.lever_arm` 供人核对——两者应当同量级、逐轴接近。
    - **判据有两条**（`t_bc_reliable`）：
      1. **残差自洽**：平移方程残差 `residual_translation_m` ≤
         `max(0.02 m, 0.05 × 画面平移跨度)`（`lever_arm_tolerance`）。`t_a` 是真是假**都**
         解得出一个 `t_x`（最小二乘照样给数），假 `t_a` 常常表现为**方程自身不自洽**。
         ⚠ **但残差不是充分判据**：残差是右端项在 `A = stack(R_a − I)` 列空间**正交补**上的
         投影，而"遥测位置恒零/几乎不动"带来的扰动 `Δt_a` 可能**整体落在列空间里**。
         因此，即使残差很小，也不能单独据此认定遥测位置可信。
      2. **量纲合格**（`telemetry_span_ok`）：遥测位置跨度 ≥ `BODY_SPAN_FLOOR_M = 0.01 m`，
         且 `body_span_m / camera_span_m ∈ [0.1, 10]`（两边本该同量级；画面跨度为 0 =
         没有平移可比对，也判不合格）。回归用例
         `test_extrinsics_flags_zeroed_telemetry_even_though_residual_is_clean` 与
         `test_extrinsics_flags_lever_arm_when_telemetry_barely_moves` 钉住。
         它是**勤勉性检查、不是证明**：只能抓"位置恒零/几乎不动"与"两边差一个数量级"，
         验证不了"位置偏了一半"这种中间态；采不到可信位置就重采，或者只用手眼旋转解。
      位置跨度检查只约束平移结果，不用于判断旋转解。
      ⚠ 画面跨度在"真值位姿"下可能**恰好为 0**（相机原地转）——gate 会判不合格，这是刻意的保守。
    - **平移分支使用堆叠最小二乘**：`(R_a − I)t_x = R_x t_b − t_a` 按
      `(3n×3)` 堆叠后交给 `lstsq`；逐行相加会使各对运动等权，不等价于堆叠后求解。
      回归用例 `test_translation_branch_is_stacked_least_squares` 覆盖该实现约定。
    - `estimate_body_translation=False` 时 `t_bc=None`、`t_bc_reliable=False`（老行为保留）。
    - **`lever_arm_report` 的 `measured` 默认值必须是 `None` 而不是 `= MEASURED_T_BC`**：
      后者是**定义时绑定**，用户改了文件顶部常量后，payload 里的 `t_bc`（运行时读取）会变、
      报告里的 `authoritative` 却是旧值，自相矛盾。
  - **配对不能按索引**：遥测 10 Hz、画面 30 Hz，"第 i 条遥测配第 i 帧"是错的。
    用 `align_body_poses` 按帧时刻重采样（位置线性内插、姿态 slerp，范围内才内插）。
    `lag_s` 的符号：`estimate_lag_by_correlation` 的 `lag > 0` 表示"画面滞后"，
    即该帧内容对应的真实时刻比帧的时刻 **早** `lag` 秒 ⇒ 查遥测用 `时刻 − lag_s`
    （`test_align_body_poses_applies_lag_with_the_right_sign`
    把它钉住了：`lag·ω` 是固定偏差，符号错了外参就带恒定姿态偏）。
  - **时间差必须用"收到帧的时刻"（`received_timestamp`）测，不能用 `capture_timestamp`**：
    索引里两条时间戳都在，`capture_timestamp = received_timestamp − 录制时的 telemetry_lag`
    是录制时**按配置假设**出来的拍摄时刻。拿它去互相关，测出来的是"该假设的残差"——
    **改动录制时的配置会平移这个残差**。所以第 2/3 步统一走 received 轴，标出的
    `telemetry_lag` = **绝对对齐量** `d_video − d_tel`（遥测自身的滞后必须一起抵掉：
    运行时按 `拍摄时刻 = 收到时刻 − telemetry_lag` 取遥测），可直接回填 `VideoConfig`。
    `meta.time_offset` 里另外给 `lag_baseline_s`（录制时的 lag）、
    `lag_residual_on_capture_axis_s`（capture 轴残差）与 `cross_check_ok`
    （`基准 + 残差 ≈ 绝对量`；只告警不改结果）。
    老素材缺 `received_timestamp` 时 `load_flight_stamps` **显式报错**，不静默退回 capture 轴。
    回归用例：`test_lag_measurement_is_absolute_on_received_axis`、
    `test_flight_index_time_bases_and_missing_received`。
  - **激励要求**：至少要 2 个**不平行**的旋转轴（OpenCV 文档同款要求），
    实测阈值 `MIN_AXIS_SPREAD_DEG = 15°`；所有运动对绕同一根轴时解不唯一，
    `run_extrinsics` 会显式报错而不是给个解。采不到旋转就重采，别调阈值。
  - `pattern_phase_ok` 的用途只有一个：发现"整体错格"这类**重投影误差不敏感**的几何错误；
    它**不能**消解标签二义性（见上）。
  - **仍未做**：真实素材上的复测（目前只有合成链路验证）；`lag_s` 的实测值需要真实标定采集。
  - **棋盘格渲染**：必须"深底 + 亮格"标准极性，且**模板四周不留边距**——留边距会让
    背景与边格同色、棋盘边界不可辨，`findChessboardCorners` 直接检不出
    （而 `checkChessboard` 仍返回 True，很容易误判成"几何没问题"）。
  - **`cv2.Rodrigues` 的约定遇到过两次问题**：它收的是世界→相机（相机轴作**行**）；
    与其手推旋转矩阵，不如"构造已知投影 + `solvePnP` 反解"更可靠。
