"""P5 真实素材端到端验证（转成用例）：视频帧 → YOLO → 裁剪 → OCR。

素材是外部文件，用环境变量 ``AIRDROP_REALDATA_VIDEO`` 指定（不设或文件不存在
时用例跳过）；仓库不携带视频。默认帧段取自一段实战视频的目标段。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

#: 真实素材视频的环境变量名
REALDATA_VIDEO_ENV = "AIRDROP_REALDATA_VIDEO"


def _realdata_video() -> Path | None:
    """从环境变量读视频路径；没设或文件不存在时返回 None（用例跳过）。"""
    raw = os.environ.get(REALDATA_VIDEO_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


VIDEO = _realdata_video()
WEIGHTS = Path("models/best2.pt")
MODELS_DIR = Path("models/ppocr")

pytestmark = pytest.mark.realdata


def _read_frame(capture: cv2.VideoCapture, index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return frame if ok else None


@pytest.mark.skipif(VIDEO is None, reason=f"未用 {REALDATA_VIDEO_ENV} 指定真实素材视频")
@pytest.mark.skipif(not WEIGHTS.is_file(), reason="缺少 YOLO 权重 models/best2.pt")
@pytest.mark.skipif(
    not (MODELS_DIR / "PP-OCRv6_det_medium.pth").is_file(), reason="缺少 PP-OCR 权重"
)
def test_real_video_reads_target_codes() -> None:
    """实战视频的目标段：检出的真目标必须读出正确编号（56/56/56）。

    刻意不要 ``tmp_path`` 参数：受限沙箱里该 fixture 建目录会
    ``PermissionError: [WinError 5]``（``--basetemp`` 也绕不开，pytest 收尾时
    自己也要枚举它）。本用例本来就不写入磁盘，去掉参数才能在沙箱里跑。
    """
    pytest.importorskip("ultralytics")
    pytest.importorskip("rapidocr")
    if os.environ.get("AIRDROP_SKIP_GPU"):
        pytest.skip("设了 AIRDROP_SKIP_GPU，跳过需要 GPU 的真实素材用例")

    from airdrop.perception.cropproc import OcrEngineConfig, OpenCvPostProcess
    from airdrop.perception.detector import Detector, DetectorConfig

    detector = Detector(DetectorConfig(model_path=str(WEIGHTS), device="0"))
    detector.load()
    post = OpenCvPostProcess(color="blue", ocr_conf_threshold=0.6, engine_config=OcrEngineConfig())

    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    results: dict[int, list[int | None]] = {}
    try:
        for index in (924, 1535, 1840):
            frame = _read_frame(capture, index)
            assert frame is not None, f"读不到第 {index} 帧"
            batch = detector.detect(frame, frame_index=index, mode="ocr")
            assert batch.detections, f"第 {index} 帧没有检出目标"
            codes = []
            for detection in batch.detections:
                crop = detector.crop(batch, detection)
                codes.append(post.recognize(crop).number)
            results[index] = codes
    finally:
        capture.release()

    # 每帧最高置信度的那个框应当读出编号；三帧都是 56。
    # ⚠ 1840 必须读出 56，不能读成 95："56" 绕图心转 180° 恰好形似 "95"，两者 OCR
    # 分数也几乎相同，只能靠"顶角≈60°"的形态门限保证转正方向正确。别把 95 改回来。
    expected = {924: 56, 1535: 56, 1840: 56}
    for index, want in expected.items():
        codes = [c for c in results[index] if c is not None]
        assert want in codes, f"第 {index} 帧没读出 {want}（实际 {results[index]}）"


def test_weights_reject_junk_crop() -> None:
    """纯色/无目标裁剪图必须读不出编号（不能瞎猜一个）。"""
    pytest.importorskip("rapidocr")
    if os.environ.get("AIRDROP_SKIP_GPU"):
        pytest.skip("设了 AIRDROP_SKIP_GPU")
    import numpy as np

    from airdrop.perception.cropproc import OcrEngineConfig, OpenCvPostProcess

    post = OpenCvPostProcess(color="blue", engine_config=OcrEngineConfig())
    blank = np.full((64, 64, 3), 120, np.uint8)
    result = post.recognize(blank)
    assert result.number is None
    assert result.stage in ("no_pentagon", "none", "empty")


# ----------------------------------------------------------------------
# 转正交叉验证：几何（形状）vs 方向判别（学出来的笔画方向）
# ----------------------------------------------------------------------
def _upright_probability(engine, strip) -> float:
    """这张数字条正立的概率。

    ``TextClassifier`` 的 score 是 softmax 概率（实测两类之和恒为 1.0000），
    所以 ``label=="180"`` 时正立概率是 ``1 - score``。
    """
    orientation = engine.classify_orientation(strip)
    return 1.0 - orientation.score if orientation.flipped else orientation.score


def _number_strip(engine, rectified):
    """转正图里所有数字框的并集（当一条文本行用）。

    按单个数字框逐个问判别器噪声很大（一位数字的朝向信息本来就少），
    并成一条更接近这个模型的训练分布。
    """
    import numpy as np

    boxes = []
    for text in engine.read_text(rectified):
        if not text.text.strip().isdigit():
            continue
        boxes.append(np.asarray(text.box, dtype=np.float64).reshape(-1, 2))
    if not boxes:
        return None
    stacked = np.vstack(boxes)
    height, width = rectified.shape[:2]
    x0 = max(int(stacked[:, 0].min()) - 3, 0)
    x1 = min(int(stacked[:, 0].max()) + 3, width)
    y0 = max(int(stacked[:, 1].min()) - 3, 0)
    y1 = min(int(stacked[:, 1].max()) + 3, height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return rectified[y0:y1, x0:x1]


@pytest.mark.skipif(VIDEO is None, reason=f"未用 {REALDATA_VIDEO_ENV} 指定真实素材视频")
@pytest.mark.skipif(not WEIGHTS.is_file(), reason="缺少 YOLO 权重 models/best2.pt")
@pytest.mark.skipif(
    not (MODELS_DIR / "ch_ppocr_mobile_v2.0_cls_mobile.onnx").is_file(),
    reason="缺少 Cls 权重（跑 python -m tools.fetch_models）",
)
def test_rectification_cross_checked_by_orientation_cls() -> None:
    """几何转正 vs 学习式方向判别：两条信息完全不同的判据互相比对。

    * 几何转正只用形状（正方形 + 上边一个等边三角形）决定转向；
    * 方向判别（``ch_ppocr_mobile_v2.0_cls``）用的是学出来的笔画方向。

    裁决用两向对比而不是门限：同一块数字条，分别问"它正立吗"与"把它转 180°
    后正立吗"，取正立概率更大的一向。真实低分素材上判别置信度很低（0.19~0.86，
    干净合成图是 0.98），``cls_thresh=0.9`` 那种门限会漏判——帧 1840 的倒立条
    只有 0.7538，两向对比才判得对。

    这条用例当初就是这样抓到"帧 1840 转正差 180°"的（那时标 xfail）；转正补上
    "顶角 ≈60°"的形态门限后修好，标记随之摘掉。

    ⚠ 为什么不硬断言"全局零冲突"：判别器在真实小目标上会误报——帧 1160
    的转正图目视明确正立、"56" 正读，它却给出 P(正立)=0.186（= 倒置 0.81）。
    所以这里只对已经目视核对过的三帧硬断言，其余全部打印出来供人复核；
    把它当"提示器"而不是"裁判"。顺带一提，0.9 门限在动作路径上正好挡住了
    这种误报：0.81 < 0.9，生产里的自动转正不会因它而动图。
    """
    pytest.importorskip("ultralytics")
    pytest.importorskip("rapidocr")
    if os.environ.get("AIRDROP_SKIP_GPU"):
        pytest.skip("设了 AIRDROP_SKIP_GPU，跳过需要 GPU 的真实素材用例")

    from airdrop.perception.cropproc import OcrEngine, OcrEngineConfig, OpenCvPostProcess
    from airdrop.perception.detector import Detector, DetectorConfig

    engine = OcrEngine(OcrEngineConfig())
    engine.ensure()  # cls_providers() 只在加载后才有内容
    if not engine.cls_providers():
        pytest.skip("方向分类未就绪")
    detector = Detector(DetectorConfig(model_path=str(WEIGHTS), device="0"))
    detector.load()
    post = OpenCvPostProcess(color="blue", ocr_conf_threshold=0.6, engine=engine)

    # 已经目视核对过的帧（原始裁剪 + 转正图都看过）：它们必须判为 0。
    # 1840 是修复前的翻转帧，1180 修复前读成 99（翻转后的 56 被读花）。
    verified = {924, 1180, 1535, 1840}

    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    # 顺序读、隔一段处理一张：反复 seek 在长 GOP 上很慢。
    # 采样集里必须含 1840 与 1180——翻转就发生在那里，隔 30 帧取样会正好漏掉。
    sample = set(range(900, 1901, 20)) | verified
    rows: list[str] = []
    checked = 0
    verified_conflicts: list[int] = []
    # 先 seek 到采样起点，之后顺序读——index 与真实帧号才对得上
    capture.set(cv2.CAP_PROP_POS_FRAMES, 900)
    index = 900
    try:
        while index <= 1900:
            ok, frame = capture.read()
            if not ok:
                break
            if index in sample:
                batch = detector.detect(frame, frame_index=index, mode="ocr")
                for detection in batch.detections:
                    result = post.recognize(detector.crop(batch, detection))
                    if result.number is None or result.rectified is None:
                        continue
                    strip = _number_strip(engine, result.rectified)
                    if strip is None:
                        continue
                    forward = _upright_probability(engine, strip)
                    backward = _upright_probability(engine, cv2.rotate(strip, cv2.ROTATE_180))
                    verdict = "0" if forward >= backward else "180"
                    checked += 1
                    if verdict == "180" and index in verified:
                        verified_conflicts.append(index)
                    rows.append(
                        f"  帧 {index}: 编号 {result.number}"
                        f"（{result.stage}{'' if result.house_ok else '/形态不成立'}）"
                        f" P(正立)={forward:.3f} vs 转180° {backward:.3f} → {verdict}"
                    )
            index += 1
    finally:
        capture.release()

    report = "\n".join(rows) if rows else "  （一张都没查成）"
    flipped = [row for row in rows if row.rstrip().endswith("→ 180")]
    print(
        f"\n转正交叉验证：查了 {checked} 个编号，判为 180° 的有 {len(flipped)} 个"
        f"（其中已目视核对的帧 {sorted(verified_conflicts)}）\n{report}"
    )
    assert checked > 0, f"没有可交叉验证的样本，检查素材是否可用\n{report}"
    assert not verified_conflicts, (
        f"已目视核对过的帧 {sorted(verified_conflicts)} 被判为 180°——"
        f"几何转正又有转倒的疑点\n{report}"
    )
