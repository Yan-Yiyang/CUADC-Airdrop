"""perception 离线单测：编号纠错、几何转正、去重、pipeline 装配。

**全部离线**：不加载 YOLO/RapidOCR 权重，不碰 GPU。重依赖一律用假对象替掉——
pipeline 的 detector / pool 都是构造注入的，这里正是为了可测性才那样设计的。

真实素材 + GPU 的端到端在 ``tests/test_perception_realdata.py``（标记 ``realdata``）。
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from airdrop.config import Config
from airdrop.perception import (
    CropResult,
    Detection,
    DetectorConfig,
    OcrEngine,
    OcrEngineConfig,
    OcrOrientation,
    OcrResult,
    OcrText,
    OpenCvPostProcess,
    PerceptionConfigLike,
    PerceptionWorker,
    PixelBox,
    correct_ocr_number,
)
from airdrop.perception.detector import Detector, FrameSizeMismatchError
from airdrop.telemetry.models import TelemetrySnapshot
from airdrop.video.buffer import AlignedSample, AlignmentBuffer
from airdrop.video.source import VideoFrame


# ----------------------------------------------------------------------
# 编号纠错
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("42", 42),
        ("7", 7),
        ("07", 7),
        ("00", 0),
        ("99", 99),
        ("10", 10),
        # 字符映射：o→0 l→1 s→5 b→6 g/q→9 z→2
        ("lO", 10),
        ("oS", 5),
        ("bg", 69),
        ("Z1", 21),
        # 首尾 '1' 是五边形边缘噪声（实测 156/561/1561 都是 56）
        ("156", 56),
        ("561", 56),
        ("1561", 56),
        ("1 5 6", 56),
        # 超过两位仍无法去掉首尾 1 时取末两位
        ("1234", 34),
        # 混入非数字
        ("4a2", 42),
        # 含易混字母：abc42xyz → b 映射成 6、z 映射成 2 → "6c42a2xy2"
        # → 数字 "6422" → 取末两位 22
        ("abc42xyz", 22),
    ],
)
def test_correct_ocr_number(raw: str, want: int) -> None:
    assert correct_ocr_number(raw) == want


@pytest.mark.parametrize("raw", [None, "", "   ", "!!!", "-"])
def test_correct_ocr_number_rejects_garbage(raw) -> None:
    assert correct_ocr_number(raw) is None


def test_correct_ocr_number_maps_confusable_letters_in_words() -> None:
    """纯字母串里若含易混字符，仍会被映射成数字——这是旧版实现的既定行为。

    ``'abc'`` 里的 ``b`` 按表映射成 6，所以结果是 6 而不是"拒绝"。这条**故意
    保留**：那张映射表是实测调出来的，改它就得重新验证所有历史识别结果。
    真正的把关在别处——OCR 置信度阈值 + 像素边长区间 + 后续的跨帧投票。
    """
    assert correct_ocr_number("abc") == 6  # b→6
    assert correct_ocr_number("oo") == 0  # o→0
    assert correct_ocr_number("sl") == 51  # s→5, l→1


def test_two_digit_results_are_not_truncated() -> None:
    """两位结果里的 1 是真实数字，不能被"去首尾 1"砍掉。"""
    assert correct_ocr_number("15") == 15
    assert correct_ocr_number("51") == 51
    assert correct_ocr_number("11") == 11


# ----------------------------------------------------------------------
# PixelBox
# ----------------------------------------------------------------------
def test_pixel_box_basics() -> None:
    box = PixelBox(10, 20, 40, 60)
    assert box.width == 30 and box.height == 40
    assert box.center == (25.0, 40.0)
    assert box.area == 1200.0
    assert box.as_dict() == {"x1": 10, "y1": 20, "x2": 40, "y2": 60}


def test_pixel_box_expand_and_clip() -> None:
    box = PixelBox(10, 20, 40, 60)
    expanded = box.expanded(0.2)  # 外扩 20%
    assert expanded.x1 == pytest.approx(7.0)
    assert expanded.x2 == pytest.approx(43.0)
    clipped = PixelBox(-5, -5, 200, 300).clipped(100, 100)
    assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0.0, 0.0, 99.0, 99.0)


def test_pixel_box_expanded_never_inverts() -> None:
    box = PixelBox(0, 0, 0, 0)
    assert box.expanded(0.2).area == 0.0


# ----------------------------------------------------------------------
# 几何：五边形顶点与转正
# ----------------------------------------------------------------------
def _pentagon_approx(angle_deg: float = 0.0) -> np.ndarray:
    """造一个目标五边形 approx（正方形 + 架在正方形上边的等边三角形）。

    正方形 (70,90)-(130,150)；三角形底边 = 正方形上边 (tl—tr，长 60)，
    顶角在上边中点正上方 ``90 − 60·√3/2``。三条边都是 60，**顶角内角就是 60°**。

    ⚠ **顶点顺序必须是简单多边形的绕行**：``apex, tl, bl, br, tr``。
    写成 ``apex, tl, tr, br, bl`` 会得到**自交**图形，顶角的两个"邻居"取错，
    量出的角度不属于顶角。正确顺序下的内角是
    60（顶角）/150（tl）/90（bl）/90（br）/150（tr），合计 540 = (5−2)·180。
    """
    side = 60.0
    half = side / 2.0
    tri = side * 0.8660254037844386  # 等边三角形的高 = 边长·√3/2
    x0, y0 = 70.0, 90.0
    cx = x0 + half
    apex = [cx, y0 - tri]
    tl = [x0, y0]
    tr = [x0 + side, y0]
    br = [x0 + side, y0 + side]
    bl = [x0, y0 + side]
    # 绕行顺序：顶角 → 左上 → 左下 → 右下 → 右上（简单多边形）
    pts = np.array([apex, tl, bl, br, tr], dtype=np.float64)
    if angle_deg:
        theta = np.radians(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        center = np.array([cx, y0 + half])
        pts = (pts - center) @ rot.T + center
    return pts.reshape(-1, 1, 2)


def test_find_apex_by_parallel_finds_triangle_tip() -> None:
    approx = _pentagon_approx()
    apex, pair, diff = OpenCvPostProcess._find_apex_by_parallel(approx)
    assert apex is not None, "正立五边形应当能找出平行边对"
    assert diff == pytest.approx(0.0, abs=1e-6), "正方形的上下/左右边应当平行"
    # 顶点是三角形尖，应当落在正方形上方（y 更小）
    assert apex[1] < 120.0


def test_find_apex_by_parallel_rejects_non_pentagon() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64).reshape(-1, 1, 2)
    apex, pair, diff = OpenCvPostProcess._find_apex_by_parallel(square)
    assert apex is None and pair is None


def _square_with_noise_corner() -> np.ndarray:
    """近方形五边形：正方形边上插一个顶点（饱和度过严时掩码只圈住编号底板）。

    这个构造样例模拟近方形掩码被 ``approxPolyDP`` 误拟合成五边形的情况；
    旧判据只看平行边时会接受该候选。
    """
    return np.array([[0, 0], [30, 3], [60, 0], [60, 60], [0, 60]], dtype=np.float64).reshape(
        -1, 1, 2
    )


def test_is_house_pentagon_accepts_the_target_shape() -> None:
    """标准"房子"任意旋转都算数——顶角恒为 60°，与朝向无关。"""
    for angle in (0.0, 12.0, -25.0, 40.0, 180.0):
        approx = _pentagon_approx(angle_deg=angle)
        apex, pair, _ = OpenCvPostProcess._find_apex_by_parallel(approx)
        assert apex is not None
        assert OpenCvPostProcess._is_house_pentagon(approx, apex), f"angle={angle}"


def test_is_house_pentagon_rejects_square_like_pentagon() -> None:
    """近方形必须判负：它有平行边、能凑出 5 个顶点，但顶角不是 60°。

    这一条是**真实缺陷**的回归——缺了这道门限时，这种假房子会给出任意角度的
    "转正"，而 OCR 仍可能读出高置信度的错误编号。
    """
    approx = _square_with_noise_corner()
    apex, pair, _ = OpenCvPostProcess._find_apex_by_parallel(approx)
    assert apex is not None, "前提：这种形状确实能找出一对平行边"
    assert not OpenCvPostProcess._is_house_pentagon(approx, apex)


def test_best_candidate_prefers_house_shape_over_confidence() -> None:
    """形态成立的候选优先于置信度更高的假房子——两种先后顺序都要成立。"""
    post = OpenCvPostProcess(color="blue")
    probe = np.zeros((10, 10, 3), np.uint8)
    empty = CropResult(None, -1.0, None, "", 0.0, stage="none")
    fake = [(5.0, 5.0, "95", 1.0)]  # 假房子读出来的（置信度更高）
    real = [(5.0, 5.0, "56", 0.9)]

    picked = post._best_candidate(
        fake, 10.0, probe, empty, saturation_level=100, stage="parallel", house_ok=False
    )
    picked = post._best_candidate(
        real, 10.0, probe, picked, saturation_level=60, stage="parallel", house_ok=True
    )
    assert picked.number == 56 and picked.house_ok

    picked = post._best_candidate(
        real, 10.0, probe, empty, saturation_level=60, stage="parallel", house_ok=True
    )
    picked = post._best_candidate(
        fake, 10.0, probe, picked, saturation_level=100, stage="parallel", house_ok=False
    )
    assert picked.number == 56, "形态成立的结果不能被后来的假房子挤掉"


def test_best_candidate_still_compares_confidence_without_house() -> None:
    """都不成立时仍按置信度挑（老行为不能丢：真实素材里常常一个房子都找不到）。"""
    post = OpenCvPostProcess(color="blue")
    probe = np.zeros((10, 10, 3), np.uint8)
    empty = CropResult(None, -1.0, None, "", 0.0, stage="none")
    low = post._best_candidate(
        [(5.0, 5.0, "12", 0.7)],
        10.0,
        probe,
        empty,
        saturation_level=100,
        stage="parallel",
        house_ok=False,
    )
    high = post._best_candidate(
        [(5.0, 5.0, "56", 0.95)],
        10.0,
        probe,
        low,
        saturation_level=40,
        stage="min_angle",
        house_ok=False,
    )
    assert high.number == 56 and high.saturation_level == 40


def test_parallel_rectify_angle_makes_parallel_edges_vertical() -> None:
    """转正角必须把平行边对（房子的上/下边）转到竖直。"""
    for angle in (0.0, 12.0, -25.0, 40.0):
        approx = _pentagon_approx(angle_deg=angle)
        apex, pair, _ = OpenCvPostProcess._find_apex_by_parallel(approx)
        assert apex is not None, f"angle={angle}: 应能找出平行边对"
        center = OpenCvPostProcess._center_of_polygon(approx)
        row = OpenCvPostProcess._parallel_rectify_angle(approx, pair, apex, center)
        pts = approx.reshape(-1, 2)
        n = len(pts)
        e1 = OpenCvPostProcess._edge_angle_deg(pts[pair[0]], pts[(pair[0] + 1) % n])
        # 旋转后（α_after = α - row）边方向应当竖直
        after = (e1 - row) % 180.0
        assert abs(after - 90.0) < 1e-6, f"angle={angle}: 转正后平行边方向 {after} 不是竖直"


def test_find_apex_by_parallel_reports_the_pair() -> None:
    """平行边对必须是正方形的**两条对边**。

    在 ``apex, tl, bl, br, tr`` 这个绕行顺序下，边 1（tl→bl）与边 3（br→tr）是
    正方形的左右两边——它们互相平行、方向角都是 90°，是轮廓里唯一的平行边对。
    """
    approx = _pentagon_approx()
    apex, pair, diff = OpenCvPostProcess._find_apex_by_parallel(approx)
    assert apex is not None
    assert pair == (1, 3), f"平行边对应当是正方形的两条对边(1 与 3)，实际 {pair}"
    assert diff == pytest.approx(0.0, abs=1e-6)
    # 顶角就是 apex 顶点本身
    assert np.allclose(apex, [100.0, 90.0 - 60.0 * 0.8660254037844386], atol=1e-6)


# 关于"转正后顶点朝哪边"：不在合成数据上断言。目标形态（三角形朝向相对正方形
# 的方位、轮廓绕行方向）决定了转正后的朝向，而这是靠真实实战素材标定的；
# 真实素材的端到端用例（tests/test_perception_realdata.py：读出 56/56/56）
# 才是这条逻辑的真正验收——合成数据的朝向假设不足以验收这条逻辑。


def test_find_min_angle_locates_triangle_tip() -> None:
    """最小内角顶点 = 三角形顶角，且那个角就是 60°。

    等边三角形的内角恒为 60°。正方形上边中点架等边三角形时，五边形内角是
    60°（顶角）、120°（两个上角）、90°（两个下角）——最小仍是顶角，
    所以"用最小内角定位顶角"这条判据是成立的。
    """
    approx = _pentagon_approx()
    point, min_deg = OpenCvPostProcess._find_min_angle(approx)
    assert min_deg == pytest.approx(60.0, abs=0.5)
    assert np.allclose(point, [100.0, 90.0 - 60.0 * 0.8660254037844386], atol=1e-6)


def test_pentagon_interior_angles() -> None:
    """把五边形内角钉死：60/120/120/90/90——顶点摆错时这里会立刻报出来。"""
    pts = _pentagon_approx().reshape(-1, 2)
    labels = ["apex", "tl", "bl", "br", "tr"]
    got = {}
    for index, name in enumerate(labels):
        previous = pts[(index - 1) % 5]
        following = pts[(index + 1) % 5]
        got[name] = OpenCvPostProcess._angle(previous, pts[index], following)
    assert got["apex"] == pytest.approx(60.0, abs=1e-6)
    assert got["tl"] == pytest.approx(150.0, abs=1e-6)
    assert got["tr"] == pytest.approx(150.0, abs=1e-6)
    assert got["br"] == pytest.approx(90.0, abs=1e-6)
    assert got["bl"] == pytest.approx(90.0, abs=1e-6)
    assert sum(got.values()) == pytest.approx(540.0, abs=1e-6)  # (5-2)·180


def test_find_min_angle_accepts_standard_pentagon() -> None:
    """兜底路径的 40~130° 门限对标准形态是**通过**的（顶角 60° 落在区间内）。"""
    approx = _pentagon_approx()
    _, min_deg = OpenCvPostProcess._find_min_angle(approx)
    assert 40.0 < min_deg < 130.0


def test_find_min_angle_gate_rejects_bent_pentagon() -> None:
    """门限要能挡掉不合法（自交/折线）的轮廓形状。

    这里直接构造一个自交顺序（``apex, tl, tr, br, bl``）——顶角的两个邻居被错认成
    两个不同侧的角，量出 15°，落在 40~130° 之外。门限拦下它是**正确**行为。
    """
    side = 60.0
    x0, y0 = 70.0, 90.0
    tri = side * 0.8660254037844386
    self_intersecting = np.array(
        [
            [x0 + side / 2, y0 - tri],
            [x0, y0],
            [x0 + side, y0],
            [x0 + side, y0 + side],
            [x0, y0 + side],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    _, min_deg = OpenCvPostProcess._find_min_angle(self_intersecting)
    assert not (40.0 < min_deg < 130.0), f"自交轮廓不该通过门限，量得 {min_deg}"


def test_angle_helpers() -> None:
    assert OpenCvPostProcess._angle_diff_deg(10.0, 170.0) == pytest.approx(20.0)
    assert OpenCvPostProcess._angle_diff_deg(0.0, 90.0) == pytest.approx(90.0)
    assert OpenCvPostProcess._angle_norm_deg(370.0) == pytest.approx(10.0)
    assert OpenCvPostProcess._angle_norm_deg(-190.0) == pytest.approx(170.0)


def test_rotate_img_keeps_target_visible() -> None:
    image = np.zeros((80, 60, 3), np.uint8)
    image[10:30, 20:40] = 255
    rotated = OpenCvPostProcess._rotate_img(image, 30.0)
    assert rotated.shape[0] >= 80 and rotated.shape[1] >= 60
    assert rotated.max() > 0, "旋转后内容不能被裁掉"


# ----------------------------------------------------------------------
# 候选合并
# ----------------------------------------------------------------------
def _text(
    text: str, cx: float, cy: float, w: float = 20.0, h: float = 30.0, score: float = 0.9
) -> OcrText:
    box = np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float64,
    )
    return OcrText(box=box, text=text, score=score)


def test_collect_candidates_merges_adjacent_single_digits() -> None:
    """同一行的两个单数字框要拼成两位数。"""
    texts = [_text("5", 100, 100), _text("6", 122, 102)]
    candidates = OpenCvPostProcess._collect_candidates(texts)
    joined = [c[2] for c in candidates if len(c[2]) == 2]
    assert "56" in joined, f"未拼出 56: {candidates}"


def test_collect_candidates_keeps_far_digits_separate() -> None:
    """相距很远的单数字不能拼在一起（那是两个不同目标）。"""
    texts = [_text("5", 100, 100), _text("6", 300, 100)]
    candidates = OpenCvPostProcess._collect_candidates(texts)
    assert not [c for c in candidates if len(c[2]) == 2]


def test_collect_candidates_keeps_multi_char_text() -> None:
    texts = [_text("42", 100, 100)]
    candidates = OpenCvPostProcess._collect_candidates(texts)
    assert any(c[2] == "42" for c in candidates)


# ----------------------------------------------------------------------
# 引擎：惰性加载与失败降级
# ----------------------------------------------------------------------
def test_ocr_engine_reports_missing_models() -> None:
    engine = OcrEngine(OcrEngineConfig(model_dir="models/does-not-exist"))
    with pytest.raises(FileNotFoundError):
        engine.ensure()
    assert engine.failed, "失败后应记住，不再反复重试"
    assert engine.read_text(np.zeros((8, 8, 3), np.uint8)) == []


def test_postprocess_unavailable_engine_returns_empty_result() -> None:
    """引擎不可用时 recognize 必须返回"没读出"，而不是抛异常。"""
    post = OpenCvPostProcess(color="blue", engine_config=OcrEngineConfig(model_dir="models/nope"))
    result = post.recognize(np.full((64, 64, 3), 100, np.uint8))
    assert result.number is None
    assert result.raw_text == ""
    assert not result.ok


def test_recognize_reports_no_pentagon_when_no_level_finds_one() -> None:
    """一个五边形都没圈出来时报 ``stage="no_pentagon"``（而不是含糊的 "none"）。

    ``recognize()`` 曾经先跑一遍 ``_detect_pentagon`` 只取布尔值、把算出的多边形
    丢掉，紧接着 ``_geometry_ocr`` 又从同一档饱和度重算整条链。那次多余的一遍
    现在由 ``_geometry_ocr`` 自己用 ``no_pentagon`` 表达——"从没找到"与
    "找到了但没读出编号"（``none``）是两回事，排故时要能分开。
    """
    post = OpenCvPostProcess(engine=OcrEngine(OcrEngineConfig(model_dir="models/nope")))
    result = post.recognize(np.full((80, 120, 3), 127, np.uint8))

    assert result.number is None
    assert result.stage == "no_pentagon", f"期望 no_pentagon，实际 {result.stage!r}"


def test_postprocess_rejects_unknown_color() -> None:
    with pytest.raises(ValueError, match="blue/red"):
        OpenCvPostProcess(color="green")


def test_postprocess_empty_image_is_safe() -> None:
    post = OpenCvPostProcess(color="blue")
    result = post.recognize(np.zeros((0, 0, 3), np.uint8))
    assert result.number is None and result.stage == "empty"


def test_engine_config_paths() -> None:
    config = OcrEngineConfig(model_dir="m", det_model="d.pth", rec_model="r.pth", rec_keys="k.txt")
    assert config.det_path() == Path("m/d.pth")
    assert config.rec_path() == Path("m/r.pth")
    assert config.keys_path() == Path("m/k.txt")


# ----------------------------------------------------------------------
# 方向分类（Cls）：接线与方向 API（不加载任何权重）
# ----------------------------------------------------------------------
class _FakeEngineType:
    """假的 ``rapidocr.EngineType``——只为在不 import rapidocr 的前提下验参数。"""

    TORCH = "engine:torch"
    ONNXRUNTIME = "engine:onnxruntime"


def _cls_params(config: OcrEngineConfig) -> dict:
    return OcrEngine(config)._cls_params(_FakeEngineType)


def test_cls_engine_selects_engine_and_model() -> None:
    """``cls_engine`` 决定引擎与权重文件；``cls_autorotate`` 决定它是否参与识别。

    ⚠ 这里**显式**写 ``cls_autorotate=True``：本用例测的是"引擎选择"，不该依赖默认值
    （默认是 False，见下一条用例）。
    """
    onnx = _cls_params(OcrEngineConfig(cls_engine="onnx", cls_use_cuda=True, cls_autorotate=True))
    assert onnx["Cls.engine_type"] == "engine:onnxruntime"
    assert onnx["EngineConfig.onnxruntime.use_cuda"] is True
    assert onnx["Global.use_cls"] is True
    assert onnx["Cls.model_path"].endswith("ch_ppocr_mobile_v2.0_cls_mobile.onnx")

    torch = _cls_params(OcrEngineConfig(cls_engine="torch", cls_use_cuda=False))
    assert torch["Cls.engine_type"] == "engine:torch"
    assert torch["EngineConfig.torch.use_cuda"] is False
    # ⚠ 权重文件名里的 ptocr 是 RapidOCR 的拼写错误，与 arch_config.yaml 的键一致，
    #   改名会直接加载失败（见 cropproc 模块 docstring）
    assert torch["Cls.model_path"].endswith("ch_ptocr_mobile_v2.0_cls_mobile.pth")

    off = _cls_params(OcrEngineConfig(cls_engine="off"))
    assert off["Global.use_cls"] is False
    assert "Cls.engine_type" not in off, "off 时不该选引擎（仍会给本地路径避免联网下载）"


def test_cls_is_diagnostic_only_by_default() -> None:
    """方向分类默认**不参与识别**：正式流程靠几何转正，cls 只作诊断/交叉验证。

    方向分类默认关闭，不参与正式链路；仅在任务数据验证后，才应启用自动转正。
    引擎与方向 API 都保留，复盘时仍可
    ``classify_orientation()`` 核对方向（真要连会话都不建就设 ``cls_engine="off"``）。
    """
    default = OcrEngineConfig()
    assert default.cls_autorotate is False
    assert default.cls_enabled() is True, "默认只是不参与识别，不是禁用方向 API"
    params = _cls_params(default)
    assert params["Global.use_cls"] is False
    assert params["Cls.engine_type"] == "engine:onnxruntime", "引擎仍在，供诊断用"


def test_cls_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="cls_engine"):
        _cls_params(OcrEngineConfig(cls_engine="openvino"))


def test_cls_config_paths_follow_engine() -> None:
    config = OcrEngineConfig(model_dir="m", cls_engine="torch")
    assert (
        config.cls_path()
        == config.cls_torch_path()
        == Path("m/ch_ptocr_mobile_v2.0_cls_mobile.pth")
    )
    config = OcrEngineConfig(model_dir="m", cls_engine="onnx")
    assert (
        config.cls_path()
        == config.cls_onnx_path()
        == Path("m/ch_ppocr_mobile_v2.0_cls_mobile.onnx")
    )
    assert config.cls_enabled() and not OcrEngineConfig(cls_engine="off").cls_enabled()


def test_cls_weight_is_required_when_enabled() -> None:
    """启用方向分类时缺权重必须显式报错，且错误里点出是哪个文件。"""
    engine = OcrEngine(OcrEngineConfig(model_dir="models/does-not-exist"))
    with pytest.raises(FileNotFoundError, match="cls_mobile"):
        engine.ensure()


def test_classify_orientation_refuses_when_disabled() -> None:
    """``cls_engine="off"`` 时方向 API 要明确拒绝，而不是给个默认值。"""
    engine = OcrEngine(OcrEngineConfig(cls_engine="off"))
    with pytest.raises(RuntimeError, match="未启用方向分类"):
        engine.classify_orientation(np.zeros((48, 192, 3), np.uint8))


def test_cls_providers_is_empty_before_loading() -> None:
    """``cls_providers()`` 只是读取：没加载时返回空，别把它当"未启用"。"""
    assert OcrEngine(OcrEngineConfig()).cls_providers() == ()


def test_ocr_warmup_runs_det_rec_and_enabled_cls_even_without_text_boxes() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class Engine:
        def __call__(self, image):
            raise AssertionError("预热不得依赖整图流水线是否检出文本框")

        def text_det(self, image):
            calls.append(("det", image.shape))
            return []

        def recognize_txt(self, images):
            calls.append(("rec", images[0].shape))
            return []

        def text_cls(self, images):
            calls.append(("cls", images[0].shape))
            return []

    config = OcrEngineConfig(cls_engine="onnx", cls_autorotate=False)
    ocr = OcrEngine(config)
    ocr._engine = Engine()

    assert ocr.warmup()
    assert calls == [
        ("det", (320, 320, 3)),
        ("rec", (48, 320, 3)),
        ("cls", (48, 320, 3)),
    ]
    assert config.cls_engine == "onnx" and config.cls_autorotate is False


@pytest.mark.parametrize("failed_stage", ["det", "rec", "cls"])
def test_ocr_warmup_fails_when_any_model_branch_raises(failed_stage: str) -> None:
    class Engine:
        def text_det(self, image):
            if failed_stage == "det":
                raise RuntimeError("det failed")
            return []

        def recognize_txt(self, images):
            if failed_stage == "rec":
                raise RuntimeError("rec failed")
            return []

        def text_cls(self, images):
            if failed_stage == "cls":
                raise RuntimeError("cls failed")
            return []

    ocr = OcrEngine(OcrEngineConfig(cls_engine="onnx"))
    ocr._engine = Engine()

    assert not ocr.warmup()


def test_ocr_warmup_skips_disabled_cls() -> None:
    calls: list[str] = []

    class Engine:
        def text_det(self, image):
            calls.append("det")
            return []

        def recognize_txt(self, images):
            calls.append("rec")
            return []

        def text_cls(self, images):
            calls.append("cls")
            return []

    ocr = OcrEngine(OcrEngineConfig(cls_engine="off"))
    ocr._engine = Engine()

    assert ocr.warmup()
    assert calls == ["det", "rec"]


def test_ocr_orientation_flipped_flag() -> None:
    assert OcrOrientation(label="0", score=0.98).flipped is False
    assert OcrOrientation(label="180", score=0.98).flipped is True


# ----------------------------------------------------------------------
# Detector（不加载权重）
# ----------------------------------------------------------------------
def test_detector_missing_weights_raises() -> None:
    detector = Detector(DetectorConfig(model_path="models/does-not-exist.pt"))
    with pytest.raises(FileNotFoundError, match="权重不存在"):
        detector.load()


def test_detector_load_is_serialized_and_publishes_names_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    class Model:
        def __init__(self) -> None:
            self.names = {0: "目标", 1: "背景"}

    def yolo(path: str) -> Model:
        nonlocal calls
        assert path.endswith("weights.pt")
        calls += 1
        entered.set()
        assert release.wait(5)
        return Model()

    module = cast(Any, ModuleType("ultralytics"))
    module.YOLO = yolo
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test")
    detector = Detector(DetectorConfig(model_path=str(weights)))
    second_lock_attempted = threading.Event()
    results: list[Detector] = []
    errors: list[BaseException] = []

    class ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "second-loader":
                second_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_: object) -> None:
            self._lock.release()

    detector._load_lock = cast(Any, ObservedLock())

    def load_detector(*, second: bool = False) -> None:
        try:
            results.append(detector.load())
        except BaseException as exc:  # noqa: BLE001 - 在线程边界收集测试失败
            errors.append(exc)

    first = threading.Thread(target=load_detector)
    second = threading.Thread(target=load_detector, name="second-loader")
    first.start()
    assert entered.wait(2)
    second.start()
    assert second_lock_attempted.wait(2)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert results == [detector, detector]
    assert calls == 1
    assert detector.loaded
    assert detector.names == {0: "目标", 1: "背景"}
    names = detector.names
    names.clear()
    assert detector.class_count == 2


def test_detector_load_failure_does_not_publish_partial_state_and_can_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    class InvalidModel:
        def __init__(self) -> None:
            self.names = {"bad-index": "目标"}

    class Model:
        def __init__(self) -> None:
            self.names = {0: "目标"}

    def yolo(path: str) -> InvalidModel | Model:
        nonlocal calls
        del path
        calls += 1
        return InvalidModel() if calls == 1 else Model()

    module = cast(Any, ModuleType("ultralytics"))
    module.YOLO = yolo
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test")
    detector = Detector(DetectorConfig(model_path=str(weights)))

    with pytest.raises(ValueError):
        detector.load()
    assert not detector.loaded
    assert detector.names == {}
    assert detector.class_count == 0

    assert detector.load() is detector
    assert detector.loaded
    assert detector.names == {0: "目标"}
    assert calls == 2


def test_detector_undistort_is_skipped_without_camera_matrix() -> None:
    detector = Detector(DetectorConfig(undistort=True, camera_matrix=None))
    image = np.zeros((48, 64, 3), np.uint8)
    out, done = detector.undistort(image)
    assert not done and out is image


def test_detector_prepares_remap_for_matrix() -> None:

    detector = Detector(
        DetectorConfig(
            camera_matrix=np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]]),
            dist_coeffs=np.zeros(5),
        )
    )
    image = np.zeros((480, 640, 3), np.uint8)
    out, done = detector.undistort(image)
    assert done, "给了内参就应当真的做去畸变"
    assert out.shape == image.shape
    # 幂等：第二次不再重算
    assert detector._remap is not None
    first = detector._remap[0]
    detector.undistort(image)
    assert detector._remap[0] is first


def test_detector_undistort_skips_on_size_mismatch() -> None:
    """帧尺寸与映射表不符时跳过（标定内参只对单一分辨率有效）。"""
    detector = Detector(
        DetectorConfig(
            camera_matrix=np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]]),
            dist_coeffs=np.zeros(5),
        )
    )
    assert detector.prepare_undistort(640, 480)
    image = np.zeros((720, 1280, 3), np.uint8)
    out, done = detector.undistort(image)
    assert not done and out is image, "尺寸不符时必须跳过，不能硬套内参"
    # 映射表保持在原尺寸上，没有被悄悄替换
    assert detector._remap_shape == (640, 480)


def test_detector_refuses_to_build_remap_at_a_non_calibration_size() -> None:
    """给了 ``calib_size``：非标定分辨率**连表都不建**（而不是按首帧尺寸默默建）。

    没有这道守卫时，"首帧分辨率就与标定不符"会静默按错误尺寸建表——内参是按
    标定分辨率算的，硬套会把画面扭坏，而 :class:`TargetTracker` 会拿这些像素
    去解算坐标。
    """
    detector = Detector(
        DetectorConfig(
            camera_matrix=np.array([[600.0, 0, 640], [0, 600, 360], [0, 0, 1]]),
            dist_coeffs=np.zeros(5),
            calib_size=(1280, 720),
        )
    )
    assert not detector.prepare_undistort(640, 360), "非标定分辨率不该建表"
    assert detector._remap is None

    image = np.zeros((360, 640, 3), np.uint8)
    out, done = detector.undistort(image)
    assert not done and out is image

    # 标定分辨率本身照常可用
    assert detector.prepare_undistort(1280, 720)
    assert detector._remap_shape == (1280, 720)


def test_pipeline_config_carries_camera_intrinsics() -> None:
    """``to_pipeline_config(camera)`` 必须把内参/畸变/标定尺寸带进检测器。

    曾经这三个字段不存在，装配处也没人传 ⇒ 整条"预计算 remap"快路径是死代码，
    每帧都退回逐点 undistortPoints。现在断言 Detection.extra["undistorted"]
    会被置位（georef 据此不再逐点纠正）。
    """
    from airdrop.georef import CameraModel

    camera = CameraModel(
        camera_matrix=[[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[[0.0, 0.0, 0.0, 0.0, 0.0]],
        image_size=(1280, 720),
    )
    cfg = Config().perception.to_pipeline_config(camera)
    assert cfg.camera_matrix is not None
    assert cfg.calib_size == (1280, 720)

    # 不给相机时保持 None（退回逐帧 undistortPoints，功能仍正确）
    plain = Config().perception.to_pipeline_config()
    assert plain.camera_matrix is None and plain.calib_size is None


def test_detector_undistort_force_recomputes() -> None:
    detector = Detector(
        DetectorConfig(
            camera_matrix=np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]]),
            dist_coeffs=np.zeros(5),
        )
    )
    detector.prepare_undistort(640, 480)
    assert detector.prepare_undistort(1280, 720, force=True)
    assert detector._remap_shape == (1280, 720)


def test_detector_crop_uses_expanded_box() -> None:
    from airdrop.perception.detector import DetectionBatch

    detector = Detector(DetectorConfig(crop_expand_ratio=0.2))
    image = np.zeros((100, 100, 3), np.uint8)
    image[30:60, 30:60] = 255
    detection = Detection(
        frame_index=1,
        capture_timestamp=1.0,
        pixel=(45.0, 45.0),
        box=PixelBox(40, 40, 50, 50),
        confidence=0.9,
        telemetry=TelemetrySnapshot(),
    )
    batch = DetectionBatch(image=image, detections=(detection,))
    crop = detector.crop(batch, detection)
    # 外扩 20% → 每边多 1px → 12x12
    assert crop.shape[:2] == (12, 12)
    assert crop.max() == 255


def test_detector_crop_clips_at_border() -> None:
    from airdrop.perception.detector import DetectionBatch

    detector = Detector(DetectorConfig())
    image = np.zeros((50, 50, 3), np.uint8)
    detection = Detection(
        frame_index=1,
        capture_timestamp=1.0,
        pixel=(0.0, 0.0),
        box=PixelBox(0, 0, 4, 4),
        confidence=0.9,
        telemetry=TelemetrySnapshot(),
    )
    batch = DetectionBatch(image=image, detections=(detection,))
    crop = detector.crop(batch, detection)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_detector_replace_config_resets_model() -> None:
    detector = Detector(DetectorConfig())
    detector._model = object()  # 假装已加载
    detector.replace_config(conf_threshold=0.5)
    assert detector._model is None
    assert detector.config.conf_threshold == 0.5


# ----------------------------------------------------------------------
# PerceptionConfigLike
# ----------------------------------------------------------------------
def test_pipeline_config_validates_mode_and_color() -> None:
    with pytest.raises(ValueError, match="ocr/cls12"):
        PerceptionConfigLike(mode="nope")
    with pytest.raises(ValueError, match="blue/red"):
        PerceptionConfigLike(target_color="green")
    with pytest.raises(ValueError, match="去重"):
        PerceptionConfigLike(ocr_dedupe_s=-1.0)


def test_config_bridges_to_perception() -> None:
    from airdrop import Config

    config = Config().validated()
    pipeline_config = config.perception.to_pipeline_config()
    assert pipeline_config.mode == config.perception.mode
    assert pipeline_config.model_path == config.perception.model_path
    assert pipeline_config.engine.model_dir == config.perception.models_dir
    assert pipeline_config.engine.use_cuda == config.perception.ocr_use_cuda


# ----------------------------------------------------------------------
# Pipeline（假 detector + 假 pool）
# ----------------------------------------------------------------------
class _FakeDetector:
    """按脚本产出检测框的假 detector（不加载权重、不跑 GPU）。"""

    def __init__(self, detections_per_frame: int = 1) -> None:
        self.detections_per_frame = detections_per_frame
        self.calls: list[int] = []

    def load(self) -> "_FakeDetector":
        return self

    def detect(
        self,
        image,
        *,
        frame_index=0,
        capture_timestamp=0.0,
        telemetry=None,
        mode="",
        undistort=None,
    ):
        from airdrop.perception.detector import DetectionBatch

        self.calls.append(frame_index)
        snapshot = telemetry if telemetry is not None else TelemetrySnapshot()
        detections = tuple(
            Detection(
                frame_index=frame_index,
                capture_timestamp=capture_timestamp,
                pixel=(100.0 + 200.0 * i, 200.0),
                box=PixelBox(90.0 + 200.0 * i, 190.0, 110.0 + 200.0 * i, 210.0),
                confidence=0.9 - 0.1 * i,
                telemetry=snapshot,
                # cls12：类别直出编号（类别 0 → 1 起编号）
                code=(0 + 1) if mode == "cls12" else None,
                mode=mode,
            )
            for i in range(self.detections_per_frame)
        )
        return DetectionBatch(image=image, detections=detections)

    def crop(self, batch, detection):
        return np.full((32, 32, 3), 200, np.uint8)


class _FakePool:
    """同步返回结果的假 OCR 池（在 submit 时就生成结果，便于确定性断言）。"""

    def __init__(self, numbers: Sequence[int | None] | None = None, side_px: float = 120.0) -> None:
        self.numbers = list(numbers or [])
        self.side_px = side_px
        self.submitted: list[tuple[int, float, float]] = []
        self._pending: list[OcrResult] = []
        self._started = False
        self._next = 0
        self._dropped = 0

    def start(self):
        self._started = True
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._started = False

    @property
    def alive(self) -> int:
        return 1 if self._started else 0

    def check_alive(self) -> int:
        return self.alive

    @property
    def stats(self) -> dict[str, int]:
        return {
            "submitted": len(self.submitted),
            "dropped": self._dropped,
            "results": self._next,
            "workers": 1,
            "pending": 0,
        }

    def submit(self, crop, *, frame_index: int, capture_timestamp: float, payload=None) -> int:
        self._next += 1
        number = self.numbers.pop(0) if self.numbers else None
        self.submitted.append((frame_index, capture_timestamp, self._next))
        self._pending.append(
            OcrResult(
                request_id=self._next,
                frame_index=frame_index,
                capture_timestamp=capture_timestamp,
                number=number,
                raw_text="" if number is None else f"{number:02d}",
                confidence=0.95 if number is not None else 0.0,
                side_px=self.side_px,
                stage="parallel",
                payload=payload,
            )
        )
        return self._next

    def poll(self, timeout: float = 0.0):
        if not self._pending:
            return None
        return self._pending.pop(0)


def _buffer_with(frames: int = 3, *, storage: str = "jpeg") -> AlignmentBuffer:
    buffer = AlignmentBuffer(capacity=64, storage=storage)
    for index in range(1, frames + 1):
        capture = 1000.0 + index * 0.1
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=index,
                    image=np.full((64, 64, 3), 50 + index, np.uint8),
                    timestamp=capture + 0.15,
                    lag=0.15,
                ),
                snapshot=TelemetrySnapshot(timestamp=capture, north_m=float(index)),
                timestamp=capture,
                lag=0.15,
                mode="interpolate",
            )
        )
    return buffer


class _UnboundedWaitCondition(threading.Condition):
    """测试用等待器：忽略期限，以通知作为唯一唤醒来源。"""

    def __init__(self, lock: threading.RLock, entered: threading.Event) -> None:
        super().__init__(lock)
        self.entered = entered

    def wait(self, timeout: float | None = None) -> bool:
        self.entered.set()
        return super().wait(None)


def _make_unbounded_wait(buffer: AlignmentBuffer) -> threading.Event:
    entered = threading.Event()
    buffer._condition = _UnboundedWaitCondition(buffer._lock, entered)
    return entered


def test_drain_waits_for_inflight_frame_and_late_ocr() -> None:
    """游标已推进但检测未结束，以及尾部 OCR 未返回，都不能算排空。"""
    entered = threading.Event()
    release = threading.Event()
    ocr_ready = threading.Event()

    class Detector(_FakeDetector):
        def detect(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().detect(*args, **kwargs)

    class Pool(_FakePool):
        def poll(self, timeout=0.0):
            return super().poll(timeout) if ocr_ready.is_set() else None

    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr"),
        buffer=_buffer_with(1),
        detector=Detector(),
        pool=Pool(numbers=[42]),
    )
    worker.start()
    try:
        assert entered.wait(5)
        assert worker.lag_frames == 0
        assert not worker.drain(timeout=0.01)
        release.set()
        assert not worker.drain(timeout=0.01)
        ocr_ready.set()
        assert worker.drain(timeout=5)
        assert worker.stats.frames == worker.stats.submitted == worker.stats.results == 1
        assert [item.code for item in worker.drain_results()] == [42]
    finally:
        release.set()
        ocr_ready.set()
        worker.stop()


def test_stop_wakes_worker_waiting_on_empty_buffer() -> None:
    buffer = _buffer_with(0)
    entered = _make_unbounded_wait(buffer)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=_FakeDetector(),
    ).start()
    assert entered.wait(2)

    worker.stop(timeout=2)

    assert not worker.running
    assert worker._thread is None
    assert worker._wait_cancel.is_set()


def test_finish_input_cancels_old_buffer_wait_then_stop_wakes_snapshot() -> None:
    original = _buffer_with(0)
    entered = _make_unbounded_wait(original)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=original,
        detector=_FakeDetector(),
    ).start()
    assert entered.wait(2)
    old_cancel = worker._wait_cancel

    assert worker.finish_input() == 0
    frozen = worker._buffer
    assert frozen is not original
    assert old_cancel.is_set()
    worker.stop(timeout=2)

    assert not worker.running
    assert worker._thread is None
    assert worker._buffer is frozen


def test_stop_restart_and_repeated_stop_use_fresh_wait_generation() -> None:
    buffer = _buffer_with(0)
    entered = _make_unbounded_wait(buffer)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=_FakeDetector(),
    ).start()
    assert entered.wait(2)
    first_cancel = worker._wait_cancel
    worker.stop(timeout=2)
    worker.stop(timeout=2)
    assert first_cancel.is_set()

    worker.start()
    second_cancel = worker._wait_cancel
    assert second_cancel is not first_cancel
    assert not second_cancel.is_set()
    worker.stop(timeout=2)

    assert first_cancel.is_set() and second_cancel.is_set()
    assert not worker.running


def test_stop_timeout_retains_live_thread_and_pool_until_frame_finishes() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Detector(_FakeDetector):
        def detect(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().detect(*args, **kwargs)

    class Pool(_FakePool):
        def __init__(self) -> None:
            super().__init__(numbers=[42])
            self.stop_calls = 0

        def stop(self, timeout: float = 5.0) -> None:
            self.stop_calls += 1
            super().stop(timeout)

    pool = Pool()
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr"),
        buffer=_buffer_with(1),
        detector=Detector(),
        pool=pool,
    )
    worker._owns_pool = True
    worker.start()
    assert entered.wait(2)
    thread = worker._thread

    worker.stop(timeout=0.01)

    assert worker.running and worker._thread is thread
    assert pool._started and pool.stop_calls == 0
    release.set()
    worker.stop(timeout=5)

    assert not worker.running and worker._thread is None
    assert not pool._started and pool.stop_calls == 1


def test_decoded_frame_returned_during_stop_is_not_accepted(monkeypatch) -> None:
    buffer = _buffer_with(1)
    decoded = threading.Event()
    release = threading.Event()
    original_wait_new = buffer.wait_new

    def wait_then_pause(*args, **kwargs):
        frame = original_wait_new(*args, **kwargs)
        if frame is not None:
            decoded.set()
            assert release.wait(5)
        return frame

    monkeypatch.setattr(buffer, "wait_new", wait_then_pause)
    detector = _FakeDetector()
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=detector,
    ).start()
    assert decoded.wait(2)

    worker.stop(timeout=0.01)
    assert worker.running
    release.set()
    worker.stop(timeout=5)

    assert detector.calls == []
    assert worker.stats.frames == 0
    assert not worker.running


def test_drain_empty_and_failed_worker() -> None:
    """空输入可排空；未启动或处理失败不能假报成功。"""
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(0),
        detector=_FakeDetector(),
    )
    assert not worker.drain(timeout=0)
    with worker:
        assert worker.drain(timeout=5)

    class BrokenDetector(_FakeDetector):
        def detect(self, *args, **kwargs):
            raise RuntimeError("检测失败")

    with PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(2),
        detector=BrokenDetector(),
    ) as failed:
        assert not failed.drain(timeout=5)
        assert "检测失败" in (failed.stats.last_error or "")


def test_recon_cutoff_drains_only_accepted_frames_and_late_ocr() -> None:
    """侦查截止时仍有在途检测；之后录像缓冲继续追加，不能扩大 OCR 任务。"""
    entered, release, ocr_ready = threading.Event(), threading.Event(), threading.Event()

    class Detector(_FakeDetector):
        def detect(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().detect(*args, **kwargs)

    class Pool(_FakePool):
        def poll(self, timeout=0.0):
            return super().poll(timeout) if ocr_ready.is_set() else None

    original = _buffer_with(3)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr"),
        buffer=original,
        detector=Detector(),
        pool=Pool(numbers=[42] * 3),
    )
    worker.start()
    try:
        assert entered.wait(5)
        assert worker.processing
        assert worker.finish_input() == 3
        assert worker.finish_input() == 3
        for i in range(4, 70):
            original.put(
                AlignedSample(
                    frame=VideoFrame(
                        index=i, image=np.zeros((64, 64, 3), np.uint8), timestamp=1000 + i
                    ),
                    snapshot=TelemetrySnapshot(timestamp=1000 + i),
                    timestamp=1000 + i,
                    lag=0,
                    mode="interpolate",
                ),
                materialize=False,
            )
        assert original.latest_index() == 69
        release.set()
        assert not worker.drain(timeout=0.05)
        ocr_ready.set()
        assert worker.drain(timeout=5)
        assert worker.stats.frames == worker.stats.results == 3
        assert [item.frame_index for item in worker.drain_results()] == [1, 2, 3]
        assert not worker.processing
    finally:
        release.set()
        ocr_ready.set()
        worker.stop()


def test_size_mismatch_stops_worker_without_processing_later_frames() -> None:
    """尺寸错误必须终止整次处理，不能被普通逐帧异常处理吞掉。"""

    class WrongSize(_FakeDetector):
        def detect(self, *args, **kwargs):
            self.calls.append(kwargs["frame_index"])
            raise FrameSizeMismatchError("帧尺寸 64x64 与标定 128x128 不符，停止运行")

    detector = WrongSize()
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(3),
        detector=detector,
    )
    worker.start()
    try:
        assert not worker.drain(timeout=5)
        with pytest.raises(FrameSizeMismatchError, match="停止运行"):
            worker.raise_if_failed()
        assert detector.calls == [1]
        assert worker.stats.frames == 0
    finally:
        worker.stop()
    with pytest.raises(FrameSizeMismatchError):
        worker.start()


@pytest.mark.parametrize("mode", ["ocr", "cls12"])
@pytest.mark.parametrize(
    ("missing", "strict", "expected", "fatal"),
    [
        (False, True, 0, False),
        (True, True, 1, False),
        (True, False, 0, False),
        (False, True, 1, True),
    ],
)
def test_replay_entry_drains_tail_and_preserves_failure(
    tmp_path,
    monkeypatch,
    caplog,
    mode,
    missing,
    strict,
    expected,
    fatal,
) -> None:
    """真实回放入口在播放完成后才放行检测，确定性复现尾帧积压。"""
    import cv2

    import airdrop
    from airdrop.perception import pipeline as perception_pipeline
    from airdrop.perception.detector import Detector as RealDetector
    from airdrop.record.replay import ReplayVideoSource
    from airdrop.runtime import replay as replay_flight

    release = threading.Event()
    frames = tmp_path / "frames"
    frames.mkdir()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    assert ok
    rows = []
    snapshots = []
    for index in range(1, 6):
        stamp = 1000 + index * 0.1
        name = f"{index:06d}.jpg"
        if not (missing and index == 5):
            (frames / name).write_bytes(jpeg.tobytes())
        rows.append(
            json.dumps(
                dict(
                    index=index,
                    filename=name,
                    capture_timestamp=stamp,
                    received_timestamp=stamp + 0.15,
                    lag=0.15,
                    extrapolated=False,
                    offset=0.0,
                    bytes=len(jpeg),
                )
            )
        )
        snapshots.append(
            json.dumps(
                TelemetrySnapshot(
                    timestamp=stamp,
                    north_m=10,
                    east_m=20,
                    down_m=-50,
                    roll_deg=0,
                    pitch_deg=0,
                    yaw_deg=0,
                ).as_dict()
            )
        )
    (tmp_path / "frames_index.jsonl").write_text("\n".join(rows), encoding="utf-8")
    (tmp_path / "telemetry.jsonl").write_text("\n".join(snapshots), encoding="utf-8")

    class Source(ReplayVideoSource):
        def stop(self, timeout=5.0):
            super().stop(timeout)
            release.set()

    class Detector(_FakeDetector):
        def detect(self, *args, **kwargs):
            if fatal:
                return RealDetector(DetectorConfig(calib_size=(128, 128))).detect(args[0])
            assert release.wait(10)
            return super().detect(*args, **kwargs)

    accepted = 4 if missing else 5

    class Pool(_FakePool):
        def poll(self, timeout=0.0):
            # 全部送检后才一起返回，避免同步假池掩盖尾部 OCR 丢失。
            return super().poll(timeout) if len(self.submitted) == accepted else None

    workers = []

    def create_worker(config, *, buffer):
        worker = PerceptionWorker(
            config,
            buffer=buffer,
            detector=Detector(),
            pool=Pool(numbers=[42] * accepted) if mode == "ocr" else None,
        )
        workers.append(worker)
        return worker

    monkeypatch.setattr(airdrop, "ReplayVideoSource", Source)
    monkeypatch.setattr(perception_pipeline, "PerceptionWorker", create_worker)
    caplog.set_level(logging.INFO)
    assert (
        replay_flight.main(
            flight_dir=tmp_path,
            perception_mode=mode,
            strict=strict,
            plan_drop_mission=False,
            speed=0.1 if fatal else 0,
        )
        == expected
    )
    worker = workers[0]
    assert not worker.running
    if fatal:
        assert worker.stats.frames == 0
        assert "帧尺寸 64x64" in caplog.text
        assert "回放终止" in caplog.text
        return
    assert worker.stats.frames == accepted
    assert worker.queued_results == 0
    if mode == "ocr":
        assert worker.stats.submitted == worker.stats.results == accepted
    assert f"回放结束：状态={'error' if missing and strict else 'finished'}" in caplog.text


def test_pipeline_ocr_mode_emits_detections_with_codes() -> None:
    buffer = _buffer_with(3)
    pool = _FakePool(numbers=[42, 56, 95])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while len(results) < 3 and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()

    assert len(results) == 3, f"应产出 3 条结果，实际 {len(results)}"
    assert [d.code for d in results] == [42, 56, 95]
    assert [d.frame_index for d in results] == [1, 2, 3]
    assert all(d.mode == "ocr" for d in results)
    assert all(d.side_px == 120.0 for d in results)
    assert all(d.raw_text for d in results)
    stats = worker.stats
    assert stats.frames == 3 and stats.detections == 3
    assert stats.submitted == 3 and stats.results == 3
    assert stats.with_code == 3


def test_pipeline_keeps_results_queued_until_drained() -> None:
    """结果队列**无界、不丢弃**：没人抽就一直在队列里（``queued_results`` 如实反映）。

    这条是"实飞里 HOLD 之后才产出的结果不能丢"的钉子：消费者（``pump``）随时来抽，
    之前攒下的结果都还在；收尾补抽之后队列必须归零。
    """
    buffer = _buffer_with(3)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=_FakePool(numbers=[42, 56, 95]),
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        while worker.queued_results < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert worker.queued_results == 3, "处理完就该在队列里等着被抽，不能丢"
        assert worker.stats.results == 3
    finally:
        worker.stop()

    drained = worker.drain_results()
    assert [d.code for d in drained] == [42, 56, 95]
    assert worker.queued_results == 0


def test_pipeline_cls12_mode_emits_without_ocr() -> None:
    buffer = _buffer_with(2)
    pool = _FakePool()
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while len(results) < 2 and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()

    assert len(results) == 2
    # 单类权重 → 类别 0 → code = 0+1 = 1
    assert [d.code for d in results] == [1, 1]
    assert pool.submitted == [], "cls12 不该走 OCR"


def test_pipeline_surfaces_dead_ocr_workers(caplog: pytest.LogCaptureFixture) -> None:
    """OCR worker 全死时要把故障喊出来，而不是"永远零产出"。

    worker 死亡的路径很现实（CUDA OOM、RapidOCR/torch 原生崩溃、结果队列写入
    自身抛错——那行不在 try 内）。掉了之后它名下的请求永远不返回，pending 单调
    上升；而积压告警只打**一条**就置位不再打 ⇒ 没有这条 ERROR，操作手从日志里
    根本看不出已经死了。
    """
    from airdrop.perception.ocr_worker import OcrWorkerPool

    pool = OcrWorkerPool(workers=2, queue_size=1)
    pool._started = True
    procs: list[Any] = [_FakeProcess(alive=True), _FakeProcess(alive=True)]
    pool._processes = procs
    assert pool.alive == 2
    assert pool.stats["alive"] == 2

    with caplog.at_level(logging.ERROR):
        pool.check_alive()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], "都活着不该报错"

    procs[1] = _FakeProcess(alive=False)
    with caplog.at_level(logging.ERROR):
        assert pool.check_alive() == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "worker 掉了必须打 ERROR"
    assert "1/2" in errors[0]

    # 同一数值不重复打，避免刷屏
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        pool.check_alive()
        pool.check_alive()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], "同一数值应已节流"

    # 恶化到全死是**新信息**，必须再报一次（这是最该被看见的情形）
    procs[0] = _FakeProcess(alive=False)
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert pool.check_alive() == 0
    assert any("0/2" in r.getMessage() for r in caplog.records)


class _FakeProcess:
    """只提供 ``is_alive()`` 的进程替身（OcrWorkerPool 的存活巡检只用到它）。"""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def test_prepared_cache_reuses_work_that_does_not_depend_on_s_min() -> None:
    """与 ``s_min`` 无关的预处理（模糊 + BGR→HSV）在同一张图上只做一次。

    饱和度回退最多试 4 档，而 GaussianBlur 与 cvtColor(BGR2HSV) 的结果**逐档完全
    相同**（只有 inRange 的阈值随 s_min 变）。原来逐档各算一遍：实测 300px 裁剪图
    上 cvtColor 约 0.14ms、GaussianBlur 约 0.03ms，4 档白扔约 0.7ms/图。叠加此前
    去掉的那遍重复检测，recognize() 中位耗时从 9.2ms 降到 5.3ms。
    """
    post = OpenCvPostProcess(engine=OcrEngine(OcrEngineConfig(model_dir="models/nope")))
    picture = np.zeros((64, 64, 3), np.uint8)

    first_blur, first_hsv = post._prepared(picture)
    second_blur, second_hsv = post._prepared(picture)
    assert first_blur is second_blur, "同一张图不该重算模糊"
    assert first_hsv is second_hsv, "同一张图不该重算 BGR→HSV"

    post.image = np.zeros((64, 64, 3), np.uint8)  # 新数组 ⇒ 缓存必须失效
    other_blur, _ = post._prepared(post.image)
    assert other_blur is not first_blur, "换图后必须重算"

    # 逐档取出的掩码仍要各不相同（说明只缓存了公共部分，没有把 s_min 也冻住）
    # BGR(160,120,120) 是 HSV S=64 的淡蓝：s_min=100 抓不到，s_min=60 能抓到。
    # 注意 _extract_blue 返回的是 BGR2GRAY **亮度**，所以用"抓得到/抓不到"比较，
    # 不去断言具体像素值。
    post.image = np.zeros((64, 64, 3), np.uint8)
    post.image[20:40, 20:40] = (160, 120, 120)
    blurred, hsv = post._prepared(post.image)
    assert post._extract_blue(blurred, hsv, 100)[30, 30] == 0, "S=64 的淡蓝不该被 s_min=100 抓到"
    assert post._extract_blue(blurred, hsv, 60)[30, 30] > 0, "S=64 的淡蓝应被 s_min=60 抓到"
    # 两档共用同一份 hsv（缓存生效），但阈值各自生效
    assert post._prepared(post.image)[0] is blurred


def test_extract_red_level_finds_a_red_object() -> None:
    """红通道提取：纯红目标要被完整取出来（掩码用按位或，不用 uint8 相加）。

    改成 ``bitwise_or`` 是防御性的：两段 hue 区间（0-10 与 160-179）互斥，所以
    同一像素最多命中一段，``m1 + m2`` 与 ``m1 | m2`` 当前等价——但 255+255 会溢出
    成 254，改动 hue 边界就会变成真 bug，而按位或"非零即命中"的语义也更正确。
    """
    post = OpenCvPostProcess(color="red", engine=OcrEngine(OcrEngineConfig(model_dir="nope")))
    image = np.zeros((40, 40, 3), np.uint8)
    image[10:30, 10:30] = (0, 0, 255)  # BGR 纯红
    blurred, hsv = post._prepared(image)
    gray = post._extract_red_level(blurred, hsv, 20)

    # 返回的是 BGR2GRAY 亮度（纯红约 29），所以比"比背景亮"，不比绝对值
    assert gray[20, 20] > gray[0, 0], "目标中心应比背景亮"
    assert gray[0, 0] == 0, "背景不该被取到"
    # 蓝通道的同一块不该被当成红色目标
    blue = np.zeros((40, 40, 3), np.uint8)
    blue[10:30, 10:30] = (255, 0, 0)
    b_blur, b_hsv = post._prepared(blue)
    assert post._extract_red_level(b_blur, b_hsv, 20)[20, 20] == 0


def test_pipeline_sends_every_frame_to_ocr_by_default() -> None:
    """默认配置不启用去重，确保每个可见帧都送 OCR。"""
    buffer = _buffer_with(5)
    pool = _FakePool(numbers=[42] * 5)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr"),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while (
            worker.stats.frames < 5 or len(pool.submitted) < 5 or len(results) < 1
        ) and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()
    assert worker.stats.frames == 5, "必须确认完整 5 帧输入都已处理"
    # 默认关闭去重，5 帧里同一位置的目标仍逐帧送检
    assert len(pool.submitted) == 5
    stats = worker.stats
    assert stats.deduped == 0


def test_pipeline_dedupe_when_enabled() -> None:
    """显式打开去重时，相近重复目标在窗口内只保留首个 OCR 请求。"""
    buffer = _buffer_with(5)
    pool = _FakePool(numbers=[42] * 5)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=10.0, ocr_dedupe_px=30.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while len(results) < 1 and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()
    assert len(pool.submitted) == 1
    stats = worker.stats
    assert stats.deduped == 4


def test_pipeline_dedupe_allows_distinct_targets() -> None:
    """同帧多目标互不误伤：位置不同就都送检。"""
    buffer = _buffer_with(1)
    pool = _FakePool(numbers=[42, 56])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=10.0, ocr_dedupe_px=30.0),
        buffer=buffer,
        detector=_FakeDetector(detections_per_frame=2),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        while len(pool.submitted) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert len(pool.submitted) == 2, "两个不同位置的目标应当分别送检"


def test_pipeline_dedupe_works_with_only_the_pixel_threshold() -> None:
    """只设 ``ocr_dedupe_px``（时间窗为 0）也必须真的去重。

    ``_dedupe_enabled`` 是"任一非零即开"，但 ``_expire`` 曾对
    ``ocr_dedupe_s <= 0`` 无条件 ``clear()``，于是刚记下的位置下一拍就没了，
    纯像素去重**静默完全不生效**、且没有任何日志。这里只给 px，钉住该行为。
    """
    buffer = _buffer_with(5)
    pool = _FakePool(numbers=[42] * 5)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=30.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        while worker.stats.deduped < 4 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert len(pool.submitted) == 1, "只设 ocr_dedupe_px 时仍应只送检首个目标"
    assert worker.stats.deduped == 4


def test_pipeline_pixel_only_dedupe_cache_is_bounded() -> None:
    """纯像素去重没有时间窗可淘汰，位置缓存必须有条数上限兜住内存。"""
    from airdrop.perception import pipeline as pipeline_mod

    config = PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=30.0)
    worker = PerceptionWorker(
        config,
        buffer=_buffer_with(1),
        detector=_FakeDetector(),
        pool=_FakePool(numbers=[42]),
    )
    detection = Detection(
        frame_index=0,
        capture_timestamp=100.0,
        pixel=(100.0, 100.0),
        box=PixelBox(95, 95, 105, 105),
        confidence=0.9,
        telemetry=TelemetrySnapshot(),
    )
    for _ in range(pipeline_mod._DEDUPE_MAX_RECENT * 3):
        worker._remember(detection)
    assert len(worker._recent) == pipeline_mod._DEDUPE_MAX_RECENT


def test_pipeline_recent_cache_stays_empty_when_dedupe_disabled() -> None:
    """去重关闭时 **不** 能往位置缓存里记东西。

    ``_expire`` 的唯一调用点在 ``_should_skip`` 里，而去重关闭时 ``_should_skip``
    直接返回——记进去的条目没有任何淘汰路径，会随每帧检测无界增长。
    """
    buffer = _buffer_with(5)
    pool = _FakePool(numbers=[42] * 5)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    for index in range(1, 6):
        frame = buffer.at(index)
        assert frame is not None
        worker._process_frame(frame)
    assert worker._recent == []
    assert len(pool.submitted) == 5, "关闭去重时每帧都要送检"


def test_pipeline_result_queue_never_drops() -> None:
    """检测结果队列无界：消费者暂时不抽也不会丢（超过旧的 2000 上限）。"""
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=AlignmentBuffer(capacity=8),
        detector=_FakeDetector(),
        pool=None,
    )
    detection = Detection(
        frame_index=1,
        capture_timestamp=1000.0,
        pixel=(10.0, 10.0),
        box=PixelBox(0.0, 0.0, 20.0, 20.0),
        confidence=0.9,
        telemetry=TelemetrySnapshot(),
    )
    total = 2101
    for frame_index in range(1, total + 1):
        worker._emit(replace(detection, frame_index=frame_index), code=1, set_code=True)
    drained = worker.drain_results()
    assert len(drained) == total
    assert drained[0].frame_index == 1
    assert drained[-1].frame_index == total


def test_pipeline_drops_invalid_side_length() -> None:
    """像素边长超出有效区间 → 编号作废（但仍产出该检测，供解算参考）。"""
    buffer = _buffer_with(1)
    pool = _FakePool(numbers=[42], side_px=5.0)  # 5px < min_side_px=10
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", min_side_px=10.0, max_side_px=400.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while not results and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()
    assert results and results[0].code is None, "无效边长不该产出编号"
    assert worker.stats.invalid_side == 1


def test_pipeline_emits_detection_without_code_on_ocr_failure() -> None:
    """OCR 没读出编号时仍产出检测（画面里确实有目标），只是 code 为空。"""
    buffer = _buffer_with(1)
    pool = _FakePool(numbers=[None])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while not results and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()
    assert len(results) == 1
    assert results[0].code is None
    assert not results[0].has_code


def test_pipeline_result_callback_is_called() -> None:
    buffer = _buffer_with(1)
    pool = _FakePool(numbers=[7])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    seen: list[int | None] = []
    worker.set_result_callback(lambda d: seen.append(d.code))
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert seen == [7]


def test_pipeline_result_callback_exception_does_not_kill_loop() -> None:
    buffer = _buffer_with(2)
    pool = _FakePool(numbers=[1, 2])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )

    def bad_callback(detection) -> None:
        raise RuntimeError("回调炸了")

    worker.set_result_callback(bad_callback)
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        results: list[Detection] = []
        while len(results) < 2 and time.monotonic() < deadline:
            results.extend(worker.drain_results())
            time.sleep(0.02)
    finally:
        worker.stop()
    assert len(results) == 2, "回调异常不该打断主循环"


def test_pipeline_stop_is_idempotent() -> None:
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(1),
        detector=_FakeDetector(),
        pool=_FakePool(),
    )
    worker.start()
    worker.stop()
    worker.stop()
    assert not worker.running


def test_pipeline_start_is_idempotent() -> None:
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(1),
        detector=_FakeDetector(),
        pool=_FakePool(),
    )
    worker.start()
    thread = worker._thread
    worker.start()
    try:
        assert worker._thread is thread
    finally:
        worker.stop()


@pytest.mark.integration
def test_pipeline_iter_results_ends_after_stop() -> None:
    buffer = _buffer_with(2)
    pool = _FakePool(numbers=[11, 22])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=pool,
    )
    drained = threading.Event()
    completed: list[object] = []

    def on_result(_detection) -> None:
        completed.append(_detection)
        if len(completed) == 2:
            drained.set()

    worker.set_result_callback(on_result)
    worker.start()
    try:
        assert drained.wait(timeout=2.0), "两条推理结果未完成"
    finally:
        worker.stop()
    codes = [d.code for d in worker.iter_results(timeout=0.2)]
    assert sorted(c for c in codes if c is not None) == [11, 22]


def test_pipeline_lag_frames_tracks_newest() -> None:
    buffer = _buffer_with(5)
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=_FakeDetector(),
        pool=_FakePool(),
        start_index=0,
    )
    assert worker.lag_frames == 5, "还没开始处理时落后全部帧"


def test_detection_as_dict_includes_telemetry() -> None:
    detection = Detection(
        frame_index=7,
        capture_timestamp=1000.0,
        pixel=(10.0, 20.0),
        box=PixelBox(5, 15, 15, 25),
        confidence=0.8,
        telemetry=TelemetrySnapshot(
            timestamp=999.0,
            north_m=1.0,
            east_m=2.0,
            down_m=-20.0,
            yaw_deg=90.0,
            attitude_timestamp_us=100,
            quaternion_timestamp_us=200,
            position_timestamp_us=150,
            alignment_source_timestamp_us=175,
        ),
        code=42,
        side_px=120.0,
        raw_text="42",
        ocr_confidence=0.95,
        mode="ocr",
    )
    payload = detection.as_dict()
    assert payload["code"] == 42
    assert payload["frame_index"] == 7
    assert payload["pixel"] == [10.0, 20.0]
    assert payload["box"]["x1"] == 5
    assert payload["telemetry"]["north_m"] == 1.0
    assert payload["telemetry"]["yaw_deg"] == 90.0
    # 保留独立时标供复算，但仍不复制任务进度等无关状态。
    assert payload["telemetry"]["attitude_timestamp_us"] == 100
    assert payload["telemetry"]["quaternion_timestamp_us"] == 200
    assert payload["telemetry"]["position_timestamp_us"] == 150
    assert payload["telemetry"]["alignment_source_timestamp_us"] == 175
    assert "mission_current" not in payload["telemetry"]


# ----------------------------------------------------------------------
# OcrWorkerPool：不丢弃 + 积压告警
# ----------------------------------------------------------------------
def _close_pool_queues(pool) -> None:
    """关掉测试里手工打开的进程池队列，别让 feeder 线程挂到进程退出。"""
    for bounded in (pool._request_q, pool._result_q, pool._status_q):
        with contextlib.suppress(Exception):
            bounded.close()
            bounded.join_thread()


class _StartupQueue(queue.Queue):
    """初始化协议测试用队列，保留关闭接口。"""

    def close(self):
        pass

    def join_thread(self):
        pass


class _StartupContext:
    """只替换进程启动边界，父进程的等待、错误和回滚逻辑照常执行。"""

    Queue = _StartupQueue

    def __init__(self, actions):
        self.actions = actions
        self.processes = []
        self.spawned = threading.Event()

    def Process(self, *, target, args, name, daemon):
        context = self
        index = len(self.processes)

        class Process:
            alive = False
            terminated = False

            def __init__(self):
                self.name = name

            def start(self):
                action = context.actions[index]
                if action == "spawn_error":
                    raise OSError("无法创建进程")
                self.alive = action != "exit"
                status = args[-1]
                if action == "ready":
                    status.put((index, None))
                elif action == "error":
                    status.put((index, "权重缺失"))
                elif action == "duplicate":
                    status.put((0, None))
                if index == len(context.actions) - 1:
                    context.spawned.set()

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def join(self, timeout):
                self.alive = False

        process = Process()
        self.processes.append(process)
        return process


def _startup_pool(actions):
    from airdrop.perception.ocr_worker import OcrWorkerPool

    pool: Any = OcrWorkerPool(workers=len(actions))
    _close_pool_queues(pool)
    context = _StartupContext(actions)
    pool._context = context
    pool._request_q = context.Queue()
    pool._result_q = context.Queue()
    pool._status_q = context.Queue()
    return pool, context


def test_ocr_start_requires_every_worker_ready() -> None:
    """第一个就绪不能代表整池就绪，等待阶段不接受送检。"""
    pool, context = _startup_pool(["ready", "silent"])
    finished = threading.Event()
    errors = []

    def start():
        try:
            pool.start(timeout=5)
        except Exception as exc:  # noqa: BLE001 - 将测试线程异常交回主线程断言
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=start)
    thread.start()
    try:
        assert context.spawned.wait(2)
        assert not finished.is_set()
        assert not pool.started
        assert pool.submit(np.zeros((2, 2, 3)), frame_index=1, capture_timestamp=0) is None
        pool._status_q.put((1, None))
        assert finished.wait(2)
        assert not errors
        assert pool.started and pool.alive == 2
        assert pool.poll() is None, "初始化消息不能混入识别结果"
    finally:
        pool._status_q.put((1, None))
        thread.join(5)
        pool.stop()


@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        ("error", RuntimeError, "权重缺失"),
        ("exit", RuntimeError, "退出"),
        ("silent", TimeoutError, "超时"),
        ("duplicate", TimeoutError, "超时"),
        ("spawn_error", OSError, "创建进程"),
    ],
)
def test_ocr_start_failure_rolls_back_all_children(action, error, message) -> None:
    pool, context = _startup_pool(["ready", action])
    with pytest.raises(error, match=message):
        pool.start(timeout=0.2)
    assert not pool.started
    assert pool.alive == 0
    assert not pool._processes
    assert context.processes[0].terminated


@pytest.mark.parametrize("outcome", [True, False, "raise"])
def test_ocr_child_reports_warmup_outcome(monkeypatch, outcome) -> None:
    """子进程入口必须检查 warmup 的布尔结果，失败不消费识别请求。"""
    from airdrop.perception import ocr_worker

    class Post:
        color = "blue"

        def __init__(self, **kwargs):
            pass

        def warmup(self):
            if outcome == "raise":
                raise RuntimeError("初始化异常")
            return outcome

    monkeypatch.setattr(ocr_worker, "OpenCvPostProcess", Post)
    requests, results, status = queue.Queue(), queue.Queue(), queue.Queue()
    requests.put(None)
    ocr_worker.ocr_worker_main(requests, results, {}, 3, status)
    worker_id, error = status.get_nowait()
    assert worker_id == 3
    assert (error is None) == (outcome is True)
    assert requests.empty() == (outcome is True)
    assert results.empty()


@pytest.mark.integration
def test_ocr_spawn_reports_initialization_error() -> None:
    """真实 Windows spawn 验证错误可跨进程返回，不加载模型或 GPU。"""
    from airdrop.perception.ocr_worker import OcrWorkerPool

    pool = OcrWorkerPool(workers=1, config={"engine": {"invalid_option": True}})
    try:
        with pytest.raises(RuntimeError, match="初始化"):
            pool.start(timeout=20)
        assert not pool.started and pool.alive == 0
    finally:
        pool.stop()
        _close_pool_queues(pool)


def test_ocr_initialization_failure_cannot_pass_preflight() -> None:
    """真实自检经真实感知启动收到错误，不能把 Process.start 当作成功。"""
    from airdrop.config import PreflightConfig
    from airdrop.preflight import Preflight, PreflightError

    pool, _ = _startup_pool(["error"])
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr"),
        buffer=_buffer_with(0),
        pool=pool,
        detector=_FakeDetector(),
    )
    preflight = Preflight(
        PreflightConfig(load_detector=False, load_camera=False, check_video=False),
        model_loaders={"ocr": worker.start},
    )
    with pytest.raises(PreflightError, match="权重缺失"):
        preflight.run()
    assert not worker.running and not pool.started


def test_ocr_restart_uses_fresh_queues() -> None:
    """失败启动和正常关闭都不能把旧就绪消息或退出哨兵留给下一轮。"""
    pool, _ = _startup_pool(["error"])
    with pytest.raises(RuntimeError):
        pool.start()
    for _ in range(2):
        previous = pool._request_q
        pool._context = _StartupContext(["ready"])
        pool.start()
        assert pool.started
        assert pool._request_q is not previous
        assert pool._request_q.empty()
        pool.stop()


def test_ocr_pool_submit_never_drops() -> None:
    """queue_size 只是告警阈值：请求队列无界，submit 永远回任务号。"""
    from airdrop.perception.ocr_worker import OcrWorkerPool

    pool = OcrWorkerPool(workers=1, queue_size=2)
    # 不拉起真的 OCR 进程（那会加载 torch/权重）：只验证队列策略
    pool._started = True
    try:
        ids = [
            pool.submit(
                np.zeros((2, 2, 3), np.uint8), frame_index=index, capture_timestamp=float(index)
            )
            for index in range(5)
        ]
        assert ids == [1, 2, 3, 4, 5]
        assert pool.stats["submitted"] == 5
        assert pool.stats["dropped"] == 0, "不丢弃策略下 dropped 必须恒为 0"
    finally:
        _close_pool_queues(pool)


def test_ocr_pool_warns_when_backlog_exceeds_threshold(caplog) -> None:
    """积压超过 queue_size 时告警（只跨阈值一次）；回落后再次涨上来会重新告警。"""
    from airdrop.perception.ocr_worker import OcrWorkerPool

    logger_name = "airdrop.perception.ocr_worker"
    pool = OcrWorkerPool(workers=1, queue_size=4)
    pool._started = True
    try:
        with caplog.at_level(logging.WARNING, logger=logger_name):
            for index in range(5):  # 第 4 个请求时越过阈值
                pool.submit(
                    np.zeros((2, 2, 3), np.uint8),
                    frame_index=index,
                    capture_timestamp=float(index),
                )
        warnings = [record for record in caplog.records if "积压" in record.message]
        assert len(warnings) == 1, caplog.text

        # 造 4 条结果取走：未完成数回到 1（< 阈值一半），告警状态复位。
        # ⚠ mp.Queue 的 put_nowait 只是写进父进程的 feeder 缓冲，紧接着的 get_nowait
        # 可能读不到——这里用带超时的 get，别让用例依赖调度时序。
        for request_id in range(1, 5):
            pool._result_q.put_nowait(
                OcrResult(
                    request_id=request_id,
                    frame_index=request_id,
                    capture_timestamp=float(request_id),
                    number=None,
                )
            )
        for _ in range(4):
            assert pool.poll(timeout=1.0) is not None

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=logger_name):
            for index in range(5, 9):  # 再次积压到阈值
                pool.submit(
                    np.zeros((2, 2, 3), np.uint8),
                    frame_index=index,
                    capture_timestamp=float(index),
                )
        assert [record for record in caplog.records if "积压" in record.message], caplog.text
    finally:
        _close_pool_queues(pool)
