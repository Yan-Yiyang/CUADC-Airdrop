# 感知/OCR：转正形态判据、Cls 与 onnxruntime（实测）

本文档说明：检测 → 裁剪 → 五边形转正 → OCR 的流水线细节与实测结论，含顶角形态门限的来龙
去脉、RapidOCR 的 `Cls` 与权重命名、onnxruntime-gpu 的 CUDA 前提与速度表、几何转正 180°
缺陷复盘。修改 `perception/cropproc.py`、换 OCR 权重/引擎、或复测真实素材准确率时请先阅读本文档。
**内容逐字搬自 `AGENTS.md`（"架构要点"两条 + "工程缺口"尾部），与 AGENTS.md 同等权威。**

---

## 流水线：OpenCvPostProcess 与 OcrEngine（自 架构要点）

- **OpenCvPostProcess**（`perception/cropproc.py`）：裁剪图后处理，移植自 旧版。
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
  分类计算（实测依据见下一条"方向分类默认关"）；要它按判定把倒置文本行转正再识别就显式设 `True`。
  `cls_engine` 决定用哪个后端（`"onnx"` 默认、实测最快，另有 `"torch"`/`"off"`），**只在启用它时才生效**；
  `"off"` 连会话都不建，但 `classify_orientation()` 会明确拒绝（不静默给值）。对外只多一个
  `classify_orientation(image) -> OcrOrientation`（**诊断入口**，给几何转正做交叉验证用）。
  置信度过滤走 `Global.text_score`（3.9.2 的 `__call__` 只收图像，旧版 那些
  `allowlist`/`low_text`/`text_threshold` 参数**已不存在**）。

## 实测结论（自 工程缺口 尾部）

  | 引擎 | ms/批 | ms/张 | 判定 |
  | --- | --- | --- | --- |
  | ONNX + CUDA | **5.74** | **0.96** | 与其余三者**逐位相同**（0.9801 / 0.9799） |
  | ONNX + CPU | 7.70 | 1.28 | 同上 |
  | TORCH + CUDA | 13.68 | 2.28 | 同上 |
  | TORCH + CPU | 24.85 | 4.14 | 同上 |
