# 感知/OCR：转正形态判据、Cls 与 onnxruntime（实测）

本文档说明：检测 → 裁剪 → 五边形转正 → OCR 的流水线细节与实测结论，含顶角形态门限的来龙
去脉、RapidOCR 的 `Cls` 与权重命名、onnxruntime-gpu 的 CUDA 前提与速度表、几何转正 180°
缺陷复盘。修改 `perception/cropproc.py`、换 OCR 权重/引擎、或复测真实素材准确率时请先阅读本文档。
**本文件是该项目对应模块的专题笔记（流水线细节与实测记录）。**

---

## 流水线：OpenCvPostProcess 与 OcrEngine（自 架构要点）

- **OpenCvPostProcess**（`perception/cropproc.py`）：裁剪图后处理，移植自 2024v2。
  去噪 → 等比放大到短边 300px → 颜色掩码饱和度逐级回退（blue 100→60→40→20，
  red 100→80→60→40→20）→ 凸包 + `approxPolyDP` 扫 `epsilon=3..40` 出五边形 →
  转正（平行边法为主、最小内角法兜底）→ **彩图**整图 OCR（v6 det 对灰度图检不出文本框）
  → 单数字框按行聚类拼两位 → 置信度加权（两位 ×1.15、单位 ×0.6）→ `correct_ocr_number` 纠错。
  **顶角几何**：目标是"正方形 + 架在**正方形上边**上的等边三角形"。等边三角形
  内角恒为 **60°**，在正确的绕行顺序（`apex, tl, bl, br, tr`）下五边形内角是
  60°（顶角）/ 90°(bl) / 90°(br) / 150°(tl) / 150°(tr)，合计 540°，
  最小内角就是顶角，兜底路径的 40~130° 门限对标准形态是**通过**的。
  合成用例 `test_pentagon_interior_angles` 把 60/90/90/150/150 钉死。
  **两个遇到过的问题（都会让顶角量错）**：
  1. 顶点顺序写成 `apex, tl, tr, br, bl` 是**自交**多边形，顶角的"邻居"变成两个
     不同侧的角，量出 15°，据此误判成"门限对标准形态失效"；
  2. ``_find_min_angle`` 原来用 ``_angle(pts[i], pts[i+1], pts[i+2])`` 并把
     ``pts[i+1]`` 当角所在顶点——那不是"顶点+邻居"，同样会量出不属于该顶点的角。
     现已改成用 ``pts[i]`` 的**真实前后邻居**。
- **OcrEngine**（`cropproc.py`）：RapidOCR 薄封装，**det/rec 走 TORCH 引擎 + 显式本地 `.pth` 路径**。
  `default_models.yaml` 里没有 `torch:` 段，TORCH 只能靠 `model_path` 指定本地权重。
  **方向分类（Cls）现已接入**，默认 `cls_engine="onnx"`（实测最快，见本文件末节"实测结论"里的速度表），
  另有 `"torch"` 与 `"off"`；`cls_autorotate` 决定它是否参与识别（默认开，
  即 PP-OCR 流水线会按判定把倒置文本行转正再识别）。对外只多一个
  `classify_orientation(image) -> OcrOrientation`（给几何转正做交叉验证用）。
  置信度过滤走 `Global.text_score`（3.9.2 的 `__call__` 只收图像，2024v2 那些
  `allowlist`/`low_text`/`text_threshold` 参数**已不存在**）。

## 实测结论

- **RapidOCR 的 `Cls`（方向分类）可用，旧记录"TORCH 装不了"是错的**（2026-09 实测）：
  TORCH 的 `inference_engine/pytorch/networks/arch_config.yaml` 里**有** cls 架构
  （`model_type: cls` / MobileNetV3 / `ClsHead`），只是键名被 RapidOCR 自己拼错了——
  `ch_ptocr_mobile_v2.0_cls_mobile`（`pt`）而正规模型名是 `ch_ppocr_...`（`pp`）。
  TORCH 引擎**按文件名 stem 查架构**（`networks/main.py` 的 `_load_arch_config`），
  所以权重必须命名成那个错拼的名字才加载得上：
  实测改名成 `ch_ppocr_...pth` → `ValueError: architecture ... is not in arch_config.yaml`；
  叫 `ch_ptocr_...pth` → 正常，极性也对（正立 `'0'`@0.980、转 180° `'180'`@0.980）。
  `tools/fetch_models.py` 已把两个 cls 权重都取进 `models/ppocr/`（**名字不许规范化**）。
- **`onnxruntime-gpu` 已装并复测（2026-09）—— CUDA EP 能用了，但有个隐藏前提**：
  1.30.0 的 `get_available_providers()` 给出 `[Tensorrt, CUDA, CPU]`，会话也能真的拿到
  `CUDAExecutionProvider`（profiler 的 node 级 `args.provider` 实测 **179 个节点全在 CUDA**）。
  **但前提是进程里先 `import torch`**：不 import torch 时三个 provider 传法（直接传字符串、
  `(名, options)` 元组、rapidocr 同款两项列表）**全部静默退回 `['CPUExecutionProvider']`**，
  不抛异常、不打日志——torch 一 import 立刻变 `['CUDAExecutionProvider', 'CPUExecutionProvider']`。
  机制：torch/lib 自带 cuDNN/cuBLAS，加载后这些 DLL 才在进程搜索路径上。
  本项目 det/rec 走 TORCH 引擎、天然满足，但**别把它当成理所当然**：
  `OcrEngine._report_cls_device()` 会显式核对首个 provider 并告警（项目原则：不许静默）。
  **四个配置的实测速度与精度**（同一权重、6 张一批、正立/倒置各一张）：

  | 引擎 | ms/批 | ms/张 | 判定 |
  | --- | --- | --- | --- |
  | ONNX + CUDA | **5.74** | **0.96** | 与其余三者**逐位相同**（0.9801 / 0.9799） |
  | ONNX + CPU | 7.70 | 1.28 | 同上 |
  | TORCH + CUDA | 13.68 | 2.28 | 同上 |
  | TORCH + CPU | 24.85 | 4.14 | 同上 |

  精度相同 → **按"取较快者"选 ONNX + CUDA**（`OcrEngineConfig.cls_engine="onnx"` 默认值）。
  ⚠ 换机器/换 CUDA 版本要重测：`onnxruntime-gpu` 与 CUDA/cuDNN 版本必须匹配。
- **几何转正曾差 180°，已修（根因：平行边法缺一道形态门限）**：
  * **症状**：帧 1840 的物理目标是 **56**，转正后 OCR 读成 **95**——"56" 绕图心转 180°
    恰好读成 "95"，两者 OCR 置信度也几乎相同（1.15 vs 1.13），**靠 OCR 分数分辨不出来**。
    证据：同段 1815/1835/1845 都读 56 且原始裁剪姿态一致；把 1840 的原始裁剪放大看就是 56；
    方向判别给倒立图 P(180)=0.7538。
  * **根因**：`_find_apex_by_parallel` 只验证"存在一对平行边"，**不验证这是不是房子**。
    饱和度偏严时掩码圈到的可能只是**编号底板**（近方形），`approxPolyDP` 照样凑出 5 个
    顶点、也照样有一对平行边 → 算出**任意角度**的"转正"；而 `_geometry_ocr` 在第一个读出
    数字的级别就 `return`，于是这个假房子的 "95" 被优先返回，真房子（s_min=60）的 "56" 没有机会被识别。
  * **修法**（`cropproc.py`）：① `_is_house_pentagon()`——用**顶角恒为 60°**（等边三角形内角）
    当形态判据，`HOUSE_APEX_ANGLE_DEG=60` / `TOL=15`；② 形态**不成立就不提前返回**，
    继续往下试更宽松的饱和度；③ `CropResult.house_ok` 记录形态是否成立，`_best_candidate`
    **形态成立优先**、同形态才比置信度。
  * **实测顶角分布**：真房子 46.4/57.0/61.6/70.9/73.6°，假房子 82.5/90.4/91.8/131.1/145.4/166.6°
    ——60±15 卡在中间但**余量不大**（近侧 1.4°）。目标只有 15~20 px 边长，量角本身有噪声；
    要调这个门限，先按上面的办法重新量分布，不要凭主观判断。
  * **修复前后 A/B**（900~1900 每 20 帧，8 个编号）：**2 帧变好、6 帧不变**——
    1840 的 95→56、1180 的 99→56（都目视核对过转正图：房子正立、"56" 正读），
    其余 54/59/56/56/66/56 一字未变，**没有回退**。
  * ⚠ **验收口径随之改了**：`expected` 里 `1840: 95` → **56**（95 是那个翻转误读，别改回去）。
  * **代价**：形态不成立的级别不再提前返回，最坏一帧走完 4 个饱和度级别（≈4 次 OCR）；
    OCR 本来就在独立进程池里跑，可以接受。
- **交叉验证要用"两向对比"，不能用置信度门限**：真实低分素材上判定的置信度很低
  （实测 0.19~0.86，而干净合成图 0.98），`cls_thresh=0.9` 会**漏判**——帧 1840 的倒立条
  只有 0.7538，卡在 0.9 以下。做法：同一块数字条，分别问"它正立吗"与"把它转 180°
  后正立吗"，取正立概率大的一向。另外**按单个数字框逐个问噪声很大**（一位数字的朝向
  信息本来就少），要把所有数字框并成一条文本行再问。
  `TextClassifier` 的 score 是 **softmax 概率**（实测两类之和恒为 `1.0000`），
  所以 `label=="180"` 时正立概率是 `1 - score`。
  ⚠ **判别器会误报，只能作为提示、不能作为判定依据**：帧 1160 的转正图**目视明确正立**、
  "56" 正读，它却给出 P(正立)=0.186（即倒置 0.81）。所以交叉验证用例只对**已目视核对过**
  的帧硬断言，其余全部打印供人复核。值得注意的是 **0.9 门限在"动作"路径上恰好能滤除这类
  误报**（0.81 < 0.9，生产里的自动转正不会因它动图）——门限太严会漏判、太松会误动，
  这里恰好合适。
- **`test_perception_realdata.py` 不接 `tmp_path`**（它本身不需要临时目录；去掉参数后 `-m realdata` 能直接跑）。
- **`cls12` 与 OCR 的"跨帧投票"由 targeting 承担，已落地**：单帧 OCR 仍可能给出像 `01`
  这样的低置信度错读（合成素材上观察到过一次，真实素材 4/4 正确）。把关点就是
  `airdrop/targeting` 的**类内编号众数**（`cluster.py::_mode_code`）：平票取较小值、
  **纯计数不加权**（一个高置信度的错读不该压过多数票）。已完成，21 个用例。
