# 感知/OCR：转正形态判据、Cls 与 onnxruntime

本文档说明：检测 → 裁剪 → 五边形转正 → OCR 的流水线细节与实测结论，含顶角形态门限的来龙
去脉、RapidOCR 的 `Cls` 与权重命名、onnxruntime-gpu 的 CUDA 前提、几何转正缺陷与已知边界。
修改 `perception/cropproc.py`、换 OCR 权重/引擎、或复测真实素材准确率时请先阅读本文档。

---

## 流水线：OpenCvPostProcess 与 OcrEngine

- **OpenCvPostProcess**（`perception/cropproc.py`）：裁剪图后处理。
  去噪 → 等比放大到短边 300px → 颜色掩码饱和度逐级回退（blue 100→60→40→20，
  red 100→80→60→40→20）→ 凸包 + `approxPolyDP` 扫 `epsilon=3..40` 出五边形 →
  转正（平行边法为主、最小内角法兜底）→ **彩图**整图 OCR（v6 det 对灰度图检不出文本框）
  → 单数字框按行聚类拼两位 → 置信度加权（两位 ×1.15、单位 ×0.6）→ `correct_ocr_number` 纠错。
  **顶角几何**：目标是"正方形 + 架在**正方形上边**上的等边三角形"。等边三角形
  内角恒为 **60°**，在正确的绕行顺序（`apex, tl, bl, br, tr`）下五边形内角是
  60°（顶角）/ 90°(bl) / 90°(br) / 150°(tl) / 150°(tr)，合计 540°，
  最小内角就是顶角，兜底路径的 40–130° 门限对标准形态是**通过**的。
  合成用例 `test_pentagon_interior_angles` 把 60/90/90/150/150 钉死。
  **两个遇到过的问题（都会让顶角量错）**：
  1. 顶点顺序写成 `apex, tl, tr, br, bl` 是**自交**多边形，顶角的"邻居"变成两个
     不同侧的角，量出 15°，据此误判成"门限对标准形态失效"；
  2. ``_find_min_angle`` 原来用 ``_angle(pts[i], pts[i+1], pts[i+2])`` 并把
     ``pts[i+1]`` 当角所在顶点——那不是"顶点+邻居"，同样会量出不属于该顶点的角。
     现已改成用 ``pts[i]`` 的**真实前后邻居**。
- **OcrEngine**（`cropproc.py`）：RapidOCR 薄封装，**det/rec 走 TORCH 引擎 + 显式本地 `.pth` 路径**。
  `default_models.yaml` 里没有 `torch:` 段，TORCH 只能靠 `model_path` 指定本地权重。
  **方向分类（Cls）已接入，但只作检查/调试（交叉验证与排故），默认不参与识别**：正式链路的转向由
  几何转正保证（平行边法 + 顶角 `60°±15°` 门限），所以 `cls_autorotate` 默认 `False`，省掉逐帧
  分类计算；要它按判定把倒置文本行转正再识别就显式设 `True`。
  `cls_engine` 决定用哪个后端（`"onnx"` 默认，另有 `"torch"`/`"off"`），**只在启用它时才生效**；
  `"off"` 连会话都不建，但 `classify_orientation()` 会明确拒绝（不静默给值）。对外只多一个
  `classify_orientation(image) -> OcrOrientation`（**诊断入口**，给几何转正做交叉验证用）。
  置信度过滤走 `Global.text_score`（3.9.2 的 `__call__` 只收图像，旧调用中的
  `allowlist`/`low_text`/`text_threshold` 参数**已不存在**）。

## 行为说明与边界

- **RapidOCR 的 `Cls`（方向分类）使用约定**：
  TORCH 的 `inference_engine/pytorch/networks/arch_config.yaml` 里**有** cls 架构
  （`model_type: cls` / MobileNetV3 / `ClsHead`），只是键名被 RapidOCR 自己拼错了——
  `ch_ptocr_mobile_v2.0_cls_mobile`（`pt`）而正规模型名是 `ch_ppocr_...`（`pp`）。
  TORCH 引擎**按文件名 stem 查架构**（`networks/main.py` 的 `_load_arch_config`），
  所以权重文件名须使用该键名。`tools/fetch_models.py` 提供的 cls 权重文件名应保持不变。
- **`onnxruntime-gpu` 的 CUDA 前提**：在该流程中，创建会话前先导入 `torch`，便于加载 CUDA
  运行库；是否可用以实际 provider 为准。`onnxruntime-gpu` 与 CUDA/cuDNN 版本必须匹配。
  项目的 `OcrEngine._report_cls_device()` 会检查首个 provider 并告警，不能把静默退回 CPU
  当作 CUDA 正常工作。
  本项目 det/rec 走 TORCH 引擎、天然满足，但**别把它当成理所当然**：
  `cls_engine="onnx"` 是配置默认值；不同运行环境需核对实际 provider。
- **方向分类默认关（`cls_autorotate=False`），只作检查/调试**：正式链路的方向由几何转正保证，
  Cls 用于交叉验证与排故，不参与默认识别。
  * **边界**：阈值筛选不能保证方向判定正确；如果几何转正方向错误，OCR 仍可能读错编号。
    `classify_orientation()` 是诊断接口。
  * **旋钮**：`cls_autorotate=True` 恢复自动转正；`cls_engine="off"` 连会话都不建（进一步省
    onnxruntime/GPU 开销），代价是方向 API 不可用。
- **几何转正曾差 180°，已修（根因：平行边法缺一道形态门限）**：
  以下复盘来自旧样本，不构成通用识别准确率验收。
  * **根因**：`_find_apex_by_parallel` 只验证"存在一对平行边"，**不验证这是不是房子**。
    饱和度偏严时掩码圈到的可能只是**编号底板**（近方形），`approxPolyDP` 照样凑出 5 个
    顶点、也照样有一对平行边 → 可能算出错误的"转正"；而 `_geometry_ocr` 在第一个读出
    数字的级别就 `return`，后续候选可能因此没有机会被识别。
  * **修法**（`cropproc.py`）：① `_is_house_pentagon()`——用**顶角恒为 60°**（等边三角形内角）
    当形态判据，`HOUSE_APEX_ANGLE_DEG=60` / `TOL=15`；② 形态**不成立就不提前返回**，
    继续往下试更宽松的饱和度；③ `CropResult.house_ok` 记录形态是否成立，`_best_candidate`
    **形态成立优先**、同形态才比置信度。
  * ⚠ **门限范围**：`HOUSE_APEX_ANGLE_DEG=60`、容差 `15°` 源于特定历史样本测量，不能视为
    对所有素材的通用验收结果；样本识别精度仍需独立验证。
  * **行为边界**：形态不成立的级别不提前返回，继续尝试后续饱和度级别；具体耗时依输入与运行环境而异。
- **方向交叉验证要用"两向对比"，不能用置信度门限**：同一块数字条，分别比较原图与旋转
  180° 后的分类结果；该判别只作提示，不作为生产识别的判定依据。另外**按单个数字框逐个问噪声很大**（一位数字的朝向
  信息本来就少），要把所有数字框并成一条文本行再问。
  `TextClassifier` 的 score 是 **softmax 概率**，
  所以 `label=="180"` 时正立概率是 `1 - score`。
  ⚠ `cls_thresh=0.9` 是分类分数阈值，判别器仍可能误报，只能作为提示、不能作为判定依据。
- **`cls12` 与 OCR 的"跨帧投票"由 targeting 承担**：单帧 OCR 可能误读；把关点是
  `airdrop/targeting` 的**类内编号众数**（`cluster.py::_mode_code`）：平票取较小值、
  **纯计数不加权**（置信度不替代多数票）。
