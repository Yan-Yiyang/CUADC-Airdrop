# 感知与 OCR

本文档说明目标检测与编号识别流水线的处理步骤、参数要求与验证方法。
相关代码：`airdrop/perception/detector.py`、`airdrop/perception/cropproc.py`、
`airdrop/perception/number.py`、`airdrop/perception/pipeline.py`。

## 处理流水线

1. **检测**：YOLO 检出目标框（单类 `target`），按需去畸变（预计算 remap）；
2. **裁剪与后处理**（`OpenCvPostProcess`）：
   去噪 → 等比放大到短边 300px → 颜色掩码饱和度逐级回退
   （blue 100→60→40→20，red 100→80→60→40→20）→ 凸包 + `approxPolyDP`
   扫 `epsilon=3..40` 提取五边形 → 转正（平行边法为主、最小内角法兜底）→
   彩图整图 OCR（v6 det 对灰度图检不出文本框）→ 单数字框按行聚类拼两位 →
   置信度加权（两位 ×1.15、单位 ×0.6）→ `correct_ocr_number` 纠错；
3. **编号聚合**：跨帧统计由 `airdrop/targeting` 承担（见"跨帧聚合"）。

## 目标形态与顶角几何

目标为"正方形 + 架在正方形上边的等边三角形"。等边三角形内角恒为 60°，
在正确的绕行顺序（`apex, tl, bl, br, tr`）下五边形内角为
60°（顶角）/ 90°(bl) / 90°(br) / 150°(tl) / 150°(tr)，合计 540°，最小内角即顶角。
合成用例 `test_pentagon_interior_angles` 固定该组数值。

量取内角的两条要求（否则会得到错误角度）：

1. 顶点顺序必须为 `apex, tl, bl, br, tr`；写成 `apex, tl, tr, br, bl` 会构成自交多边形，
   顶角的邻居取错，量出的角度不属于该顶点；
2. `_find_min_angle` 必须使用角所在顶点的真实前后邻居（`_angle(prev, vertex, next)`），
   不能将 `pts[i+1]` 当作角所在顶点。

## 转正形态门限

仅验证"存在一对平行边"不足以判定转正角度：饱和度偏严时掩码可能只圈到编号底板
（近方形），`approxPolyDP` 仍会拟合出五边形与平行边，从而产生任意角度的"转正"，
并可能把 56 读成 95 这类翻转误读（OCR 分数无法区分）。

实现要求（`cropproc.py`）：

1. `_is_house_pentagon()` 以顶角恒为 60° 作为形态判据
   （`HOUSE_APEX_ANGLE_DEG=60` / `TOL=15`）；
2. 形态不成立时不提前返回，继续尝试更宽松的饱和度级别；
3. `CropResult.house_ok` 记录形态是否成立；`_best_candidate` 优先选择形态成立的结果，
   同形态再比较置信度。

**顶角实测分布**：真目标 46.4/57.0/61.6/70.9/73.6°，假目标
82.5/90.4/91.8/131.1/145.4/166.6°。60±15 位于两者之间，余量较小（近侧 1.4°）；
目标边长仅 15~20 px，量角存在噪声。调整该门限前应先重新测量分布。

代价：形态不成立的级别不再提前返回，最坏情况下单帧需完成 4 个饱和度级别
（约 4 次 OCR）。OCR 在独立进程池中运行，该开销可接受。

## 模型与运行要求

- **分类权重文件名必须保持 RapidOCR 原样**：TORCH 的 `arch_config.yaml` 中 cls 架构键名
  为 `ch_ptocr_mobile_v2.0_cls_mobile`（`pt`），TORCH 引擎按文件名 stem 查找架构
  （`networks/main.py` 的 `_load_arch_config`），文件名与键名不一致会导致加载失败。
  两个 cls 权重由 `tools/fetch_models.py` 复制到 `models/ppocr/`，不得重命名。
- **`onnxruntime-gpu` 使用 CUDA 前必须在进程内先 `import torch`**：torch/lib 自带
  cuDNN/cuBLAS，加载后相关 DLL 才进入进程搜索路径；未导入 torch 时，多种 provider
  指定方式都会静默退回 CPU。本项目的 det/rec 使用 TORCH 引擎，天然满足该条件；
  `OcrEngine._report_cls_device()` 会核对首个 provider 并告警。
- 检测权重使用 `best2.pt`（单类 `target`），仓库不携带权重，通过
  `python -m airdrop.run fetch-models` 复制；`cls12` 模式需要 12 类权重（当前不提供，
  该模式编号恒为 1）。

### 方向分类引擎速度对比

同一权重、6 张一批（正立/倒置各一张）：

| 引擎 | ms/批 | ms/张 | 判定 |
| --- | --- | --- | --- |
| ONNX + CUDA | **5.74** | **0.96** | 与其余三者逐位相同（0.9801 / 0.9799） |
| ONNX + CPU | 7.70 | 1.28 | 同上 |
| TORCH + CUDA | 13.68 | 2.28 | 同上 |
| TORCH + CPU | 24.85 | 4.14 | 同上 |

四者精度一致，因此默认选择最快的 ONNX + CUDA（`OcrEngineConfig.cls_engine="onnx"`）。
更换机器或 CUDA 版本后应重新测量；`onnxruntime-gpu` 与 CUDA/cuDNN 版本必须匹配。

## 交叉验证方法

几何转正与学习式方向判别可互相校验，但须遵守以下要求：

- **使用两向对比，不使用固定置信度门限**：真实素材上判别置信度较低（约 0.19~0.86，
  干净合成图约 0.98）。做法为：同一数字条分别询问"正立"与"转 180° 后正立"，
  取正立概率较高的一向。
- 按单个数字框逐个询问噪声较大，应将所有数字框并成一条文本行再询问。
- `TextClassifier` 的分数为 softmax 概率（两类之和恒为 1.0000），
  `label=="180"` 时正立概率为 `1 - score`。
- **判别器可能误报，只能作为提示，不能作为判定依据**。交叉验证用例仅对已人工核对的
  样本做硬断言，其余结果打印供复核。
- 真实素材用例（`tests/test_perception_realdata.py`）不写磁盘，视频路径由环境变量
  `AIRDROP_REALDATA_VIDEO` 指定。

## 跨帧聚合

单帧 OCR 可能给出低置信度错读，最终把关点是 `airdrop/targeting` 的类内编号众数
（`cluster.py::_mode_code`）：平票取较小值，纯计数不加权——单个高置信度错读不应
压过多数投票。
