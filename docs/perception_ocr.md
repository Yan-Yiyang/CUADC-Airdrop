# 感知/OCR：转正形态判据与 Cls 方向诊断

本文档说明：检测 → 裁剪 → 五边形转正 → OCR 的流水线细节，含顶角形态门限、RapidOCR 的 `Cls` 与 PyTorch 权重命名、几何转正缺陷与已知边界。
修改 `perception/cropproc.py`、换 OCR 权重/引擎、或复测真实素材准确率时请先阅读本文档。

---

## 流水线：OpenCvPostProcess 与 OcrEngine

- **OpenCvPostProcess**（`perception/cropproc.py`）：裁剪图后处理。
  去噪 → 等比放大到短边 300px → 颜色掩码饱和度逐级回退（blue 100→60→40→20，
  red 100→80→60→40→20）→ 凸包 + `approxPolyDP` 扫 `epsilon=3..40` 出五边形 →
  转正（平行边法为主、最小内角法兜底）→ **彩图**整图 OCR（v6 det 对灰度图检不出文本框）
  → 单数字框按行聚类拼两位 → `correct_ocr_number` 纠错，至少识别出两位才接受 → 按编号数值加权（10–99 ×1.15、00–09 ×0.6）。
  **顶角几何**：目标是"正方形 + 架在**正方形上边**上的等边三角形"。等边三角形
  内角恒为 **60°**，在正确的绕行顺序（`apex, tl, bl, br, tr`）下五边形内角是
  60°（顶角）/ 90°(bl) / 90°(br) / 150°(tl) / 150°(tr)，合计 540°，
  最小内角就是顶角，兜底路径的 40–130° 门限对标准形态是**通过**的。
  合成用例 `test_pentagon_interior_angles` 验证标准五边形的 60°/90°/90°/150°/150° 内角关系。
  **顶点绕行与内角计算约束**：
  1. 轮廓顶点须按非自交顺序排列（`apex, tl, bl, br, tr`），避免自交多边形导致邻边错配；
  2. `_find_min_angle` 计算顶点 `pts[i]` 的内角时，必须使用其真实前后相邻顶点 `pts[i-1]` 与 `pts[i+1]`。
- **OcrEngine**（`cropproc.py`）：按权重后缀支持 **TensorRT FP16 离线引擎（默认，`models/trt/PP-OCRv6_{det,rec}_medium_fp16.engine`，经 `trt_ocr.py` 复用 RapidOCR 前后处理而不加载 `RapidOCR` 主入口或 `onnxruntime`）** 与 **RapidOCR TORCH 备用后端（`--backend pytorch`，显式本地 `.pth` 路径，不加载 ONNX 模型）**。
  TensorRT 引擎文件或伴随清单缺失、SHA256 不一致或运行环境不匹配时直接报错退出，提示运行 `./.venv/Scripts/python.exe -m airdrop.run build-trt --target all` 或显式传 `--backend pytorch`，绝不静默回退。
  在 PyTorch 备用后端下，`default_models.yaml` 里没有 `torch:` 段，TORCH 只能靠 `model_path` 指定本地权重。
  **方向分类（Cls）已接入，但只作检查/调试（交叉验证与排故），默认不参与识别**：正式链路的转向由
  几何转正保证（平行边法 + 顶角 `60°±15°` 门限），所以 `cls_autorotate` 默认 `False`，省掉逐帧
  分类计算；要它按判定把倒置文本行转正再识别就显式设 `True`。
  `cls_engine` 决定是否启用方向分类（`"off"` 默认，仅 PyTorch `.pth` 后端诊断时可设为 `"torch"`，TensorRT 后端强制要求 `"off"`）；
  `"off"` 下 `classify_orientation()` 会明确拒绝（不静默给值）。对外只多一个
  `classify_orientation(image) -> OcrOrientation`（**诊断入口**，给几何转正做交叉验证用）。
  通过 `det_max_side_len=960` 约束狭长裁切图在 `limit_type="min"`（736）下的等比放大上限
  （超限时缩放并在短边补黑边至 736，检测框自动映射回原图坐标）。
  置信度过滤走 `Global.text_score`（3.9.2 的 `__call__` 只收图像，旧调用中的
  `allowlist`/`low_text`/`text_threshold` 参数**已不存在**）。

## 行为说明与边界

- **RapidOCR 的 `Cls`（方向分类）使用约定**：
  TORCH 的 `inference_engine/pytorch/networks/arch_config.yaml` 里**有** cls 架构
  （`model_type: cls` / MobileNetV3 / `ClsHead`），只是键名被 RapidOCR 自己拼错了——
  `ch_ptocr_mobile_v2.0_cls_mobile`（`pt`）而正规模型名是 `ch_ppocr_...`（`pp`）。
  TORCH 引擎**按文件名 stem 查架构**（`networks/main.py` 的 `_load_arch_config`），
  所以权重文件名须使用该键名。`tools/fetch_models.py` 提供的 cls 权重文件名应保持不变。
- **方向分类默认关（`cls_engine="off"`、`cls_autorotate=False`），只作检查/调试**：正式链路的方向由几何转正保证，
  Cls 用于交叉验证与排故，不参与默认识别。
  * **边界**：阈值筛选不能保证方向判定正确；如果几何转正方向错误，OCR 仍可能读错编号。
    `classify_orientation()` 是诊断接口。
  * **旋钮**：诊断交叉验证显式设 `cls_engine="torch"`；`cls_autorotate=True` 恢复自动转正。
- **几何转正的顶角形态门限（60°±15°）**：
  * **设计意图**：`_find_apex_by_parallel` 仅验证轮廓中存在一对平行边，无法单独区分完整靶板与近方形编号底板。当高饱和度掩码仅提取到底板轮廓时，`approxPolyDP` 仍可能拟合出含一对平行边的五边形并给出错误转正角；若在首个读出数字的档位即提前返回，将遮蔽后续宽松饱和度档位上的真实靶板候选。
  * **实现机制**（`cropproc.py`）：① `_is_house_pentagon()` 以等边三角形顶角恒为 60° 作为形态判据（`HOUSE_APEX_ANGLE_DEG=60`、`TOL=15`）；② 形态不成立的档位不提前返回，继续尝试后续更宽松的饱和度档位；③ `CropResult.house_ok` 记录形态是否成立，`_best_candidate` 优先选择形态成立的候选，同形态下再比较置信度。
  * **边界**：`HOUSE_APEX_ANGLE_DEG=60` 与 `15°` 容差属于经验形态门限，不构成对全部场景的通用识别精度保证；形态不成立时会继续尝试后续饱和度档位，实际耗时随输入图像与运行环境变化。
- **方向交叉验证要用"两向对比"，不能用置信度门限**：同一块数字条，分别比较原图与旋转
  180° 后的分类结果；该判别只作提示，不作为生产识别的判定依据。另外**按单个数字框逐个问噪声很大**（一位数字的朝向
  信息本来就少），要把所有数字框并成一条文本行再问。
  `TextClassifier` 的 score 是 **softmax 概率**，
  所以 `label=="180"` 时正立概率是 `1 - score`。
  ⚠ `cls_thresh=0.9` 是分类分数阈值，判别器仍可能误报，只能作为提示、不能作为判定依据。
- **`cls12` 与 OCR 的"跨帧投票"由 targeting 承担**：单帧 OCR 可能误读；把关点是
  `airdrop/targeting` 的**类内编号众数**（`cluster.py::_mode_code`）：平票取较小值、
  **纯计数不加权**（置信度不替代多数票）。
# 飞行记录裁切图

启用飞行记录后，感知工作器在 OCR 去重之前为每个非空目标裁切图保存一张 JPG，质量为 95；`cls12` 同样保存。文件名由全局帧序号、帧内检测序号和处理阶段组成：`_box` 表示原始检测框裁切，`_rectified` 表示几何转正成功但 OCR 未成功，`_ocr` 表示 OCR 成功。异步识别完成后原子重命名最终文件，并追加索引事件；处理中的转正候选仅作为临时文件，结束后只保留一张 JPG。`crops/index.jsonl` 记录采集时间、检测框、实际外扩裁切像素范围、检测模式、置信度及去畸变状态。目录在记录器创建本次飞行目录后注入，不启用飞行记录时不产生裁切文件。

OCR 输入以 PNG 无损写入飞行记录的 `ocr_spool/`，请求元数据与业务检测一并保存；mp 请求队列只传有界文件引用。启动或恢复时流式扫描目录一次，清理已有确认标记的残留并恢复最大编号、积压数和磁盘 SQLite 顺序索引；运行期间只查询下一个编号并从索引移除已派发请求，编号空洞不会导致逐号探测，内存不保存积压列表。sidecar 与 ack 标记仍是恢复依据，索引可在重启时重建。OCR 子进程可将转正候选写到临时 JPG，父进程关联结果后再决定最终阶段文件名。请求文件只在成功关联结果后删除；异常结束时保留未完成文件供检查和重启恢复。磁盘满、图像编码或索引失败会使感知排空失败并写入错误状态。
