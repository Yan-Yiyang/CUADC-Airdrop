"""QGC 固定尺寸画布合成。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from airdrop.qgcvideo.models import DisplaySnapshot

CANVAS_SIZE = (1760, 990)
VIDEO_RECT = (0, 130, 1280, 720)
PANEL_RECT = (1280, 130, 480, 720)
STATUS_RECT = (0, 0, 1760, 130)
FOOTER_RECT = (0, 850, 1760, 140)
BACKGROUND_COLOR = (31, 25, 19)
SURFACE_COLOR = (39, 31, 24)
PANEL_COLOR = (36, 29, 23)


def _rounded_box(
    image: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int], radius: int = 6
) -> None:
    """低成本绘制圆角面板，避免 UI 与图像标记使用相同的视觉形态。"""
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for x, y in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


_STAGES = (
    ("INIT", "初始化"),
    ("PREFLIGHT", "准备检查"),
    ("WAIT_AIRBORNE", "等待起飞"),
    ("RECON", "侦查"),
    ("HOLD_PROCESS", "盘旋处理"),
    ("OVERFLY", "投放航线"),
    ("LAND", "降落"),
    ("DONE", "完成"),
    ("ABORT", "中止"),
)
_FLOW = (
    ("PREFLIGHT", "自检"),
    ("WAIT_AIRBORNE", "等待起飞"),
    ("RECON", "侦查"),
    ("HOLD_PROCESS", "盘旋处理"),
    ("OVERFLY", "飞掠投放"),
    ("LAND", "返航降落"),
    ("DONE", "完成"),
)
_CODE_COLORS = (
    (55, 220, 100),
    (220, 75, 220),
    (40, 170, 255),
    (255, 210, 40),
    (235, 105, 45),
    (190, 110, 240),
    (70, 220, 235),
    (160, 210, 100),
)


def _code_color(code: int | None) -> tuple[int, int, int]:
    if code is None:
        return (175, 180, 185)
    return _CODE_COLORS[code % len(_CODE_COLORS)]


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "未知"
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def _format_clock(timestamp: float | None) -> str:
    if timestamp is None or not math.isfinite(timestamp):
        return "未知"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def _fit_text_anchor(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    origin: tuple[int, int],
    clip: tuple[int, int, int, int],
) -> tuple[ImageFont.FreeTypeFont, tuple[int, int], tuple[int, int, int, int]] | None:
    """按真实字形边界缩字并移动基线，使文字和底色留在视口内。"""
    left, top, right, bottom = clip
    if right - left < 24 or bottom - top < 16:
        return None
    size = font.size
    bounds = draw.textbbox(origin, text, font=font, anchor="ls")
    while (
        bounds[2] - bounds[0] > right - left - 16 or bounds[3] - bounds[1] > bottom - top - 9
    ) and size > 8:
        size -= 1
        font = _font(size)
        bounds = draw.textbbox(origin, text, font=font, anchor="ls")
    if bounds[2] - bounds[0] > right - left - 16 or bounds[3] - bounds[1] > bottom - top - 9:
        return None
    shift_x = round(max(left + 8 - bounds[0], min(0, right - 8 - bounds[2])))
    shift_y = round(max(top + 4 - bounds[1], min(0, bottom - 5 - bounds[3])))
    anchor = (origin[0] + shift_x, origin[1] + shift_y)
    final_left, final_top, final_right, final_bottom = draw.textbbox(
        anchor, text, font=font, anchor="ls"
    )
    final_bounds = (
        round(final_left),
        round(final_top),
        round(final_right),
        round(final_bottom),
    )
    return font, anchor, final_bounds


@lru_cache(maxsize=12)
def _font(size: int) -> ImageFont.FreeTypeFont:
    paths = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("找不到可用的中文字体（微软雅黑、黑体或 Noto Sans CJK）")


class FrameComposer:
    """把只读显示快照渲染为 1760×990 BGR 画布。"""

    def __init__(self) -> None:
        self._publisher_stats: Any | None = None
        self._text_commands: list[
            tuple[str, tuple[int, int], int, tuple[int, int, int], tuple[int, int, int, int] | None]
        ] = []
        self._video_label_bounds: list[tuple[int, int, int, int]] = []

    def set_publisher_stats(self, stats: Any) -> None:
        """接收发布线程的只读统计快照。"""
        self._publisher_stats = stats

    def compose(self, snapshot: DisplaySnapshot) -> np.ndarray:
        """渲染单张画布，所有绘制都在新数组上完成。"""
        self._text_commands = []
        self._video_label_bounds = []
        canvas = np.full((990, 1760, 3), BACKGROUND_COLOR, dtype=np.uint8)
        canvas[:130] = SURFACE_COLOR
        canvas[130:850, 1280:] = PANEL_COLOR
        canvas[850:] = SURFACE_COLOR
        cv2.line(canvas, (1280, 130), (1280, 849), (60, 49, 38), 1)
        cv2.line(canvas, (0, 850), (1759, 850), (60, 49, 38), 1)
        self._render_video(canvas, snapshot)
        self._render_status(canvas, snapshot)
        self._render_panel(canvas, snapshot)
        self._render_footer(canvas, snapshot)
        self._draw_text_commands(canvas)
        return canvas

    def _render_video(self, canvas: np.ndarray, snapshot: DisplaySnapshot) -> None:  # noqa: PLR0912
        current = getattr(snapshot, "frame", None)
        image = None if current is None else getattr(current, "image", None)
        if image is None:
            self._text(canvas, "等待视频画面", (28, 505), 0.9, (190, 195, 200))
            return
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            self._text(canvas, "视频帧无效", (28, 505), 0.9, (40, 80, 240))
            return
        h, w = image.shape[:2]
        scale = min(1280 / w, 720 / h)
        out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
        ox, oy = (1280 - out_w) // 2, (720 - out_h) // 2
        resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        canvas[130 + oy : 130 + oy + out_h, ox : ox + out_w] = resized

        mode = str(getattr(snapshot, "mode", ""))
        if mode in {"recon", "hold_follow"}:
            for detection in getattr(snapshot, "detections", ()):
                box = getattr(detection, "box", None)
                if box is None or len(box) != 4:
                    continue
                x1, y1, x2, y2 = box
                p1 = (ox + round(float(x1) * scale), 130 + oy + round(float(y1) * scale))
                p2 = (ox + round(float(x2) * scale), 130 + oy + round(float(y2) * scale))
                color = _code_color(getattr(detection, "code", None))
                cv2.rectangle(canvas, p1, p2, color, 2)
                code = getattr(detection, "code", None)
                label = f"{code:02d} · {detection.confidence:.0%}" if code is not None else "待识别"
                self._text(
                    canvas,
                    label,
                    (p1[0], max(157, p1[1] - 9)),
                    0.43,
                    color,
                    2,
                    clip=(0, 130, 1280, 850),
                )
        if mode in {"recon", "hold_follow", "hold_live"}:
            self._render_observations(canvas, snapshot, (scale, out_w, out_h, ox, oy))
        if mode == "drop":
            if getattr(snapshot, "flypast_completed", False):
                self._text(
                    canvas,
                    "已飞掠目标航段",
                    (24, 166),
                    0.48,
                    (55, 220, 100),
                    clip=(ox, 130 + oy, ox + out_w, 130 + oy + out_h),
                )
            else:
                self._render_projected(canvas, snapshot, (scale, out_w, out_h, ox, oy))
        index = getattr(current, "frame_index", None)
        if index is not None:
            delay = getattr(snapshot, "capture_delay_s", None)
            if getattr(snapshot, "replay_mode", False):
                delay_text = "回放原时间轴"
            else:
                delay_text = "未知" if delay is None else f"{delay * 1000:.0f} 毫秒"
            self._text(
                canvas,
                (
                    f"帧 #{index} | 离线回放（原时间轴）"
                    if getattr(snapshot, "replay_mode", False)
                    else f"帧 #{index}（{self._mode_name(mode)}）| 采集延迟 {delay_text}"
                ),
                (24, 832),
                0.40,
                (225, 225, 225),
                clip=(0, 130, 1280, 850),
            )
        is_demo = current is not None and any(
            item.mode == "demo-cuadc" for item in current.detections
        )
        if is_demo:
            self._text(
                canvas,
                "合成演示 · 任务二 / A区 · 15m 下视",
                (24, 166),
                0.43,
                (235, 226, 212),
                clip=(0, 130, 1280, 850),
            )
            pose = current.pose if current is not None else None
            matrix = current.camera_matrix if current is not None else None
            if pose is not None and pose.position_ned is not None and matrix is not None:
                altitude = abs(pose.position_ned[2])
                if altitude > 0:
                    length = min(260, round(matrix[0][0] * 5 / altitude))
                    cv2.line(canvas, (1244 - length, 824), (1244, 824), (230, 230, 230), 2)
                    for x in (1244 - length, 1244):
                        cv2.line(canvas, (x, 819), (x, 829), (230, 230, 230), 2)
                    self._text(
                        canvas,
                        "5 米",
                        (1244 - length, 812),
                        0.38,
                        (230, 230, 230),
                        clip=(0, 130, 1280, 850),
                    )
        if mode == "drop":
            self._text(
                canvas,
                "准星：选定目标    圆环：预计落点",
                (24, 794),
                0.40,
                (235, 226, 212),
                clip=(0, 130, 1280, 850),
            )

    def _render_observations(
        self,
        canvas: np.ndarray,
        snapshot: DisplaySnapshot,
        viewport: tuple[float, int, int, int, int],
    ) -> None:
        scale, out_w, out_h, ox, oy = viewport
        observations: Sequence[Any] = getattr(snapshot, "projected_observations", ())
        codes = tuple(getattr(snapshot, "observations", ()))
        roi = canvas[130 + oy : 130 + oy + out_h, ox : ox + out_w]
        overlay = roi.copy()
        for observed, point in zip(codes, observations, strict=False):
            if not getattr(point, "valid", False) or not getattr(point, "in_bounds", False):
                continue
            pixel = getattr(point, "pixel", None)
            if pixel is None:
                continue
            x = ox + round(float(pixel[0]) * scale)
            y = 130 + oy + round(float(pixel[1]) * scale)
            if not (ox <= x < ox + out_w and 130 + oy <= y < 130 + oy + out_h):
                continue
            cv2.circle(
                overlay,
                (x - ox, y - 130 - oy),
                2,
                _code_color(getattr(observed, "code", None)),
                -1,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, dst=roi)

    def _render_projected(
        self,
        canvas: np.ndarray,
        snapshot: DisplaySnapshot,
        viewport: tuple[float, int, int, int, int],
    ) -> None:
        scale, out_w, out_h, ox, oy = viewport
        clip = (ox, 130 + oy, ox + out_w, 130 + oy + out_h)
        roi = canvas[clip[1] : clip[3], clip[0] : clip[2]]
        for name, color, label in (
            ("target_projected", (55, 220, 100), "目标"),
            ("impact_projected", (35, 80, 245), "落点"),
        ):
            if getattr(snapshot, "flypast_completed", False):
                continue
            projected = getattr(snapshot, name, None)
            if projected is None:
                continue
            if not getattr(projected, "valid", False):
                reason = str(getattr(projected, "reason", "")).lower()
                if any(token in reason for token in ("behind", "rear", "后方")):
                    geometry_note = (
                        "目标当前不在相机前方" if label == "目标" else "预测落点当前不在相机前方"
                    )
                    self._text(canvas, geometry_note, (18, 162), 0.43, color, clip=clip)
                continue
            pixel = getattr(projected, "pixel", None)
            if pixel is None:
                continue
            if not getattr(projected, "in_bounds", True):
                self._edge_arrow(
                    canvas,
                    snapshot,
                    (projected, color, label),
                    viewport,
                )
                continue
            px = ox + round(float(pixel[0]) * scale)
            py = 130 + oy + round(float(pixel[1]) * scale)
            if name == "target_projected":
                cx, cy = px - ox, py - 130 - oy
                cv2.circle(roi, (cx, cy), 25, color, 1, cv2.LINE_AA)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(
                        roi,
                        (cx + dx * 18, cy + dy * 18),
                        (cx + dx * 33, cy + dy * 33),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.circle(roi, (px - ox, py - 130 - oy), 15, color, 2, cv2.LINE_AA)
                cv2.circle(roi, (px - ox, py - 130 - oy), 4, color, -1, cv2.LINE_AA)
            self._projected_label(
                canvas, snapshot, projected, label, (px + 25, py - 16), color, clip
            )

    def _edge_arrow(
        self,
        canvas: np.ndarray,
        snapshot: DisplaySnapshot,
        overlay: tuple[Any, tuple[int, int, int], str],
        geometry: tuple[float, int, int, int, int],
    ) -> None:
        projected, color, label = overlay
        scale, out_w, out_h, ox, oy = geometry
        pixel = projected.pixel
        center_x, center_y = ox + out_w / 2, oy + out_h / 2
        dx, dy = (
            ox + float(pixel[0]) * scale - center_x,
            oy + float(pixel[1]) * scale - center_y,
        )
        magnitude = math.hypot(dx, dy)
        if not math.isfinite(magnitude) or magnitude < 1e-6:
            return
        bounds = (
            float(ox + 16),
            float(ox + out_w - 16),
            float(oy + 16),
            float(oy + out_h - 16),
        )
        factors = (
            (bounds[1] - center_x) / dx
            if dx > 0
            else (bounds[0] - center_x) / dx
            if dx < 0
            else math.inf,
            (bounds[3] - center_y) / dy
            if dy > 0
            else (bounds[2] - center_y) / dy
            if dy < 0
            else math.inf,
        )
        factor = min(value for value in factors if value >= 0)
        if not math.isfinite(factor):
            return
        tip = (round(center_x + dx * factor), 130 + round(center_y + dy * factor))
        unit_x, unit_y = dx / magnitude, dy / magnitude
        tail = (round(tip[0] - unit_x * 48), round(tip[1] - unit_y * 48))
        clip = (ox, 130 + oy, ox + out_w, 130 + oy + out_h)
        roi = canvas[clip[1] : clip[3], clip[0] : clip[2]]
        cv2.arrowedLine(
            roi,
            (tail[0] - ox, tail[1] - 130 - oy),
            (tip[0] - ox, tip[1] - 130 - oy),
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.35,
        )
        self._projected_label(
            canvas, snapshot, projected, label, (tip[0] - 8, tip[1] - 10), color, clip
        )

    def _projected_label(  # noqa: PLR0917
        self,
        canvas: np.ndarray,
        snapshot: DisplaySnapshot,
        projected: Any,
        label: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        clip: tuple[int, int, int, int],
    ) -> None:
        ned = getattr(projected, "ned", None)
        frame = getattr(snapshot, "frame", None)
        pose = None if frame is None else getattr(frame, "pose", None)
        position = None if pose is None else getattr(pose, "position_ned", None)
        if ned is None or position is None:
            distance = "水平距离未知" if label == "目标" else "距该落点未知"
        else:
            horizontal = math.hypot(float(ned[0]) - position[0], float(ned[1]) - position[1])
            distance = (
                f"水平距离 {horizontal:.0f} 米"
                if label == "目标"
                else f"距该落点 {horizontal:.0f} 米"
            )
        if label == "落点":
            live = getattr(snapshot, "live_prediction", None)
            prefix = "实时预测落点" if live is not None and live.valid else "预测落点"
        else:
            prefix = "目标"
        text = f"{prefix} · {distance}"
        draw = ImageDraw.Draw(Image.new("RGB", CANVAS_SIZE))
        x, y = origin
        if label == "落点":
            candidates = ((x, y + 72), (x, y), (x, y - 72), (x - 160, y), (x + 160, y))
        else:
            candidates = ((x, y), (x, y - 72), (x, y + 72), (x - 160, y), (x + 160, y))
        font = _font(14)
        for candidate in candidates:
            fitted = _fit_text_anchor(draw, text, font, candidate, clip)
            if fitted is None:
                continue
            _, _, bounds = fitted
            padded = (bounds[0] - 7, bounds[1] - 4, bounds[2] + 7, bounds[3] + 4)
            overlaps = any(
                padded[0] < occupied[2]
                and padded[2] > occupied[0]
                and padded[1] < occupied[3]
                and padded[3] > occupied[1]
                for occupied in self._video_label_bounds
            )
            if overlaps:
                continue
            self._video_label_bounds.append(padded)
            self._text(canvas, text, candidate, 0.40, color, clip=clip)
            return
        self._text(canvas, text, origin, 0.40, color, clip=clip)

    def _render_status(self, canvas: np.ndarray, snapshot: DisplaySnapshot) -> None:
        stage = str(getattr(snapshot, "stage", None) or "")
        progress = getattr(snapshot, "progress", None)
        replay = bool(getattr(snapshot, "replay_mode", False))
        self._text(canvas, "任务状态", (24, 23), 0.40, (177, 158, 138))
        order = {name: index for index, (name, _) in enumerate(_FLOW)}
        active_index = order.get(stage)
        completed = set(snapshot.completed_stages) if not replay else set()
        if stage == "DONE":
            active_index = len(_FLOW) - 1
        if stage == "ABORT" or replay:
            active_index = None
        x = 24
        for index, (name, label) in enumerate(_FLOW):
            width = 158
            left, top, right, bottom = x, 34, x + width, 77
            if name in completed and index != active_index:
                fill, text_color = (54, 65, 28), (182, 216, 123)
            elif index == active_index:
                fill, text_color = (46, 179, 239), (57, 38, 19)
            else:
                fill, text_color = (58, 45, 34), (155, 136, 119)
            if stage == "ABORT" and index == active_index:
                fill = (45, 55, 190)
            _rounded_box(canvas, (left, top, right, bottom), fill)
            finished = name in completed and index != active_index
            if finished:
                cv2.polylines(
                    canvas,
                    [np.array([(left + 12, 55), (left + 16, 59), (left + 23, 49)])],
                    False,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )
            self._text(canvas, label, (left + (31 if finished else 15), 63), 0.47, text_color, 2)
            if index < len(_FLOW) - 1:
                cv2.line(canvas, (right + 3, 55), (right + 13, 55), (95, 110, 125), 2)
            x = right + 15

        mode = str(getattr(snapshot, "mode", ""))
        lag = int(getattr(progress, "lag_frames", 0))
        queued = int(getattr(progress, "queued_results", 0)) + int(
            getattr(progress, "ocr_pending", 0)
        )
        processing = (
            "持续侦查"
            if stage == "RECON"
            else "处理中"
            if getattr(progress, "processing", False) or lag or queued
            else "已排空"
        )
        processed_frames = getattr(progress, "processed_frames", None)
        processed_text = "未知" if processed_frames is None else str(processed_frames)
        preflight = stage in {"INIT", "PREFLIGHT", "WAIT_AIRBORNE"} or stage not in (
            order.keys() | {"DONE", "ABORT"}
        )
        summary = (
            "离线回放（原时间轴）"
            if replay
            else f"任务已中止 | {self._mode_name(mode)}"
            if stage == "ABORT"
            else "任务已完成"
            if stage == "DONE"
            else "起飞前：仅实时预览 / 自检，侦查尚未开始"
            if preflight
            else (
                f"{self._mode_name(mode)}：{processing} | 积压 {lag} 帧 | "
                f"排队 {queued} | 已处理 {processed_text} 帧"
            )
        )
        self._text(canvas, summary, (24, 109), 0.42, (211, 194, 174))
        self._render_stage_card(canvas, snapshot, stage)

    def _render_stage_card(self, canvas: np.ndarray, snapshot: DisplaySnapshot, stage: str) -> None:
        progress = getattr(snapshot, "progress", None)
        replay = bool(getattr(snapshot, "replay_mode", False))
        aborted = stage == "ABORT"
        accent = (89, 113, 245) if aborted else (213, 177, 115)
        _rounded_box(canvas, (1296, 14, 1744, 115), (51, 39, 28), 8)
        cv2.line(canvas, (1296, 29), (1296, 101), accent, 3, cv2.LINE_AA)
        title = "离线回放" if replay else "任务中止" if aborted else self._stage_name(stage)
        self._text(canvas, title, (1314, 39), 0.46, accent, 2)
        mission_current = getattr(progress, "mission_current", None)
        mission_total = getattr(progress, "mission_total", None)
        elapsed = getattr(progress, "stage_elapsed_s", None)
        if replay:
            route = "不显示历史任务阶段"
            detail = "回放原时间轴"
        elif aborted:
            route = "当前阶段已中止"
            detail = f"已运行 {_format_duration(elapsed)}"
        elif stage == "RECON":
            route = (
                "侦查航线 未知"
                if mission_current is None or mission_total is None
                else f"侦查航线 {mission_current}/{mission_total}"
            )
            detail = f"已用时 {_format_duration(elapsed)}"
        elif stage in {"INIT", "PREFLIGHT", "WAIT_AIRBORNE"} or stage not in dict(_STAGES):
            route = "侦查尚未开始"
            detail = "仅实时预览 / 自检"
        elif stage == "HOLD_PROCESS":
            remaining = getattr(progress, "hold_remaining_s", None)
            eta = getattr(progress, "eta_s", None)
            detail = f"距超时 {remaining:.1f} 秒" if remaining is not None else "距超时 未知"
            route = f"预计处理 {eta:.1f} 秒" if eta is not None else "预计处理 未知"
        elif stage in {"OVERFLY", "LAND", "DONE", "ABORT"}:
            status = self._release_text(snapshot)
            stamp = getattr(snapshot, "release_timestamp", None)
            if stamp is None:
                stamp = getattr(snapshot, "release_command_timestamp", None)
            route = status
            moment = "投放时刻" if getattr(snapshot, "released", False) else "指令时刻"
            detail = f"{moment} {_format_clock(stamp)}"
        else:
            route = f"阶段用时 {_format_duration(elapsed)}"
            detail = self._mode_name(str(getattr(snapshot, "mode", "")))
        self._text(canvas, route, (1314, 72), 0.57, (239, 232, 222))
        self._text(canvas, detail, (1314, 100), 0.41, (184, 165, 145))

    @staticmethod
    def _release_text(snapshot: DisplaySnapshot) -> str:
        if getattr(snapshot, "released", False):
            if getattr(snapshot, "release_simulated", False):
                return "模拟投放已记录"
            return "投放已记录"
        status = str(getattr(snapshot, "release_command_status", "not_sent"))
        text = {
            "pending": "正在发送指令",
            "sent": "投放指令已发出",
            "failed": "投放指令失败",
            "unknown": "投放结果未知",
        }.get(status, "尚未发送投放指令")
        if getattr(snapshot, "release_simulated", False) and status in {
            "pending",
            "sent",
            "failed",
            "unknown",
        }:
            return f"{text}（模拟）"
        return text

    def _render_panel(self, canvas: np.ndarray, snapshot: DisplaySnapshot) -> None:
        self._text(canvas, "目标观测", (1298, 162), 0.65, (239, 232, 222), 2)
        self._text(canvas, "最近三个编号 · 原帧截图", (1298, 187), 0.39, (177, 158, 138))
        entries = tuple(getattr(snapshot, "panel", ()))
        entry_by_code = {getattr(entry, "code", None): entry for entry in entries}
        recent_codes = tuple(getattr(snapshot, "recent_codes", ()))
        panel = tuple(entry_by_code[code] for code in recent_codes if code in entry_by_code)[:3]
        observed_at = getattr(snapshot, "observed_at", None)
        card_left = 1296
        card_width = 148
        for index in range(3):
            left = card_left + index * 151
            right = left + card_width
            top = 194
            bottom = 336
            _rounded_box(canvas, (left, top, right, bottom), (54, 42, 31))
            entry = panel[index] if index < len(panel) else None
            color = _code_color(None if entry is None else getattr(entry, "code", None))
            cv2.line(canvas, (left + 10, bottom), (right - 10, bottom), color, 2)
            if entry is None:
                self._text(canvas, "暂无目标", (left + 8, top + 30), 0.36, (160, 170, 180))
                continue
            code = getattr(entry, "code", "?")
            self._text(canvas, f"{code:02d}", (left + 10, top + 25), 0.57, color, 2)
            thumbnail = self._thumbnail(snapshot, entry)
            if thumbnail is not None:
                height, width = thumbnail.shape[:2]
                ratio = min(128 / width, 80 / height)
                tw, th = max(1, round(width * ratio)), max(1, round(height * ratio))
                resized = cv2.resize(thumbnail, (tw, th), interpolation=cv2.INTER_AREA)
                tx, ty = left + 10 + (128 - tw) // 2, top + 32 + (80 - th) // 2
                canvas[ty : ty + th, tx : tx + tw] = resized
            else:
                self._text(canvas, "暂无截图", (left + 29, top + 76), 0.34, (155, 165, 175))
            count = getattr(entry, "count", 0)
            age = self._thumbnail_age(snapshot, entry, observed_at)
            self._text(
                canvas, f"{age}  ·  {count}次", (left + 9, bottom - 8), 0.35, (189, 171, 151)
            )

        self._text(canvas, "观测统计", (1298, 376), 0.54, (239, 232, 222), 2)
        self._text(canvas, "按观测次数排序", (1602, 376), 0.37, (177, 158, 138))
        cv2.line(canvas, (1297, 390), (1748, 390), (65, 51, 39), 1)
        self._text(
            canvas, "编号 / 次数                 观测均值坐标", (1300, 414), 0.36, (177, 158, 138)
        )
        for index, entry in enumerate(entries[:5]):
            code = getattr(entry, "code", "?")
            count = getattr(entry, "count", 0)
            lla = getattr(entry, "position_lla", None)
            ned = getattr(entry, "position_ned", None)
            color = _code_color(code if isinstance(code, int) else None)
            y = 446 + index * 74
            self._text(
                canvas,
                f"{code:02d}  {count} 次",
                (1300, y),
                0.42,
                color,
                2,
            )
            if ned is not None:
                self._text(
                    canvas,
                    f"NED  N {ned[0]:.1f}  E {ned[1]:.1f}  D {ned[2]:.1f} 米",
                    (1415, y),
                    0.37,
                    (210, 220, 230),
                )
            else:
                self._text(canvas, "NED 未知", (1415, y), 0.31, (150, 160, 170))
            if lla is not None:
                self._text(
                    canvas,
                    f"经 {lla[0]:.6f}°  纬 {lla[1]:.6f}°",
                    (1415, y + 29),
                    0.36,
                    (184, 165, 145),
                )
            else:
                self._text(canvas, "WGS84 未知", (1415, y + 29), 0.30, (150, 160, 170))
            cv2.line(canvas, (1297, y + 42), (1748, y + 42), (55, 44, 33), 1)
        error = getattr(snapshot, "geometry_error", None)
        if error:
            self._text(canvas, "空间投影不可用", (1300, 826), 0.34, (65, 170, 250))

    @staticmethod
    def _thumbnail_age(snapshot: DisplaySnapshot, entry: Any, observed_at: float | None) -> str:
        capture_timestamp = getattr(entry, "thumbnail_capture_timestamp", None)
        if capture_timestamp is not None and observed_at is not None:
            return f"{max(0.0, observed_at - float(capture_timestamp)):.1f}秒前"
        frame_index = getattr(entry, "thumbnail_frame_index", None)
        if frame_index is None or observed_at is None:
            return "时间未知"
        for frame in getattr(snapshot, "history_frames", ()):
            if getattr(frame, "frame_index", None) == frame_index:
                age = max(0.0, observed_at - float(frame.capture_timestamp))
                return f"{age:.1f}秒前"
        return "无截图"

    def _render_footer(self, canvas: np.ndarray, snapshot: DisplaySnapshot) -> None:  # noqa: PLR0915
        for x, title in ((24, "处理进度"), (600, "发布与链路"), (1185, "航线与投放")):
            self._text(canvas, title, (x, 878), 0.50, (213, 177, 115), 2)
        cv2.line(canvas, (580, 866), (580, 980), (55, 75, 95), 1)
        cv2.line(canvas, (1165, 866), (1165, 980), (55, 75, 95), 1)
        progress = getattr(snapshot, "progress", None)
        captured = getattr(progress, "processed_frames", None)
        pending = getattr(progress, "lag_frames", 0)
        completed = getattr(progress, "ocr_completed", None)
        ocr_pending = getattr(progress, "ocr_pending", 0)
        queued = getattr(progress, "queued_results", 0)
        proc_fps = getattr(progress, "processing_fps", None)
        ocr_rate = getattr(progress, "ocr_rate_per_s", None)
        self._text(
            canvas,
            f"处理 {captured if captured is not None else '未知'} 帧 · 积压 {pending} 帧 · 队列 {queued}",
            (20, 906),
            0.35,
            (220, 225, 230),
        )
        self._text(
            canvas,
            f"检测 {proc_fps:.1f} 帧/秒" if proc_fps is not None else "检测速率 未知",
            (20, 936),
            0.35,
            (190, 200, 210),
        )
        ocr_text = (
            f"OCR 已完成 {completed if completed is not None else '未知'} · 待处理 {ocr_pending}"
        )
        if ocr_rate is not None:
            ocr_text += f" · {ocr_rate:.1f} 次/秒"
        self._text(canvas, ocr_text, (20, 966), 0.33, (190, 200, 210))

        stats = self._publisher_stats
        encoded = 0 if stats is None else stats.encoded
        dropped = 0 if stats is None else stats.dropped
        restarts = 0 if stats is None else stats.restarts
        latency = None if stats is None else stats.last_latency_s
        latency_text = "未知" if latency is None else f"{latency * 1000:.0f} 毫秒"
        self._text(
            canvas,
            "静态预览 · 发布器未启动"
            if stats is None
            else f"发布 {encoded} 帧 · 丢弃 {dropped} · 重启 {restarts}",
            (600, 906),
            0.36,
            (220, 225, 230),
        )
        latency_label = "提交至编码" if getattr(snapshot, "replay_mode", False) else "采集至编码"
        self._text(
            canvas,
            f"{latency_label} {latency_text}",
            (600, 936),
            0.35,
            (190, 200, 210),
        )
        telemetry_age = getattr(progress, "telemetry_age_s", None)
        telemetry_text = (
            "遥测时延 回放不适用"
            if getattr(snapshot, "replay_mode", False)
            else "遥测新鲜度 未知"
            if telemetry_age is None
            else f"遥测新鲜度 {telemetry_age:.2f} 秒"
        )
        self._text(canvas, telemetry_text, (600, 966), 0.35, (190, 200, 210))

        mission_current = getattr(progress, "mission_current", None)
        mission_total = getattr(progress, "mission_total", None)
        stage = str(getattr(snapshot, "stage", "") or "")
        route_label = "侦查航线" if stage == "RECON" else "当前航线"
        route = (
            f"{route_label} 未知"
            if mission_current is None or mission_total is None
            else f"{route_label} {mission_current}/{mission_total}"
        )
        stage_elapsed = getattr(progress, "stage_elapsed_s", None)
        if stage == "RECON":
            if mission_total is not None and mission_total > 0 and mission_current is not None:
                route += " · 进度完成" if mission_current >= mission_total else " · 执行中"
            route += f" · 已用时 {_format_duration(stage_elapsed)}"
        elif stage == "HOLD_PROCESS":
            remaining = getattr(progress, "hold_remaining_s", None)
            route = f"距超时 {remaining:.1f} 秒" if remaining is not None else "距超时 未知"
            eta = getattr(progress, "eta_s", None)
            route += f" · 预计处理 {eta:.1f} 秒" if eta is not None else " · 预计处理 未知"
        if getattr(snapshot, "replay_mode", False):
            route = "历史任务阶段不可用"
            release_text = "回放不包含投放状态"
        elif stage == "ABORT":
            route = "任务已中止"
            release_text = self._release_text(snapshot)
        else:
            release_text = self._release_text(snapshot)
        self._text(canvas, route, (1185, 906), 0.36, (220, 225, 230))
        self._text(canvas, release_text, (1185, 936), 0.36, (220, 225, 230))
        live = getattr(snapshot, "live_prediction", None)
        live_error = None if live is None or not live.valid else live.error_m
        live_flight = None if live is None or not live.valid else live.flight_time_s
        observed_at = getattr(snapshot, "observed_at", None)
        received_at = None if live is None else live.position_velocity_received_at
        state_age = (
            None
            if observed_at is None or received_at is None
            else float(observed_at) - float(received_at)
        )
        live_text = (
            f"实时预测误差 {live_error:.1f} 米 · 预计滞空 {live_flight:.1f} 秒"
            if live_error is not None and live_flight is not None
            else f"实时无效 · {('未启动' if live is None else live.reason)}"
        )
        live_text += " · 状态龄 "
        live_text += "未知" if state_age is None else f"{state_age:.2f} 秒"
        if live is not None and live.valid and not live.wind_available:
            live_text += " · 缺风按零风"
        release_error = getattr(snapshot, "release_error_m", None)
        release_time = getattr(snapshot, "release_flight_time_s", None)
        release_stamp = getattr(snapshot, "release_prediction_timestamp", None)
        release_text = (
            f"投放时模型：误差 {release_error:.1f} 米 · 时刻 {_format_clock(release_stamp)}"
            if release_error is not None and release_time is not None
            else "投放时模型预测 · 未知（无匹配的成功投放预测）"
        )
        if release_error is not None and release_time is not None:
            release_text += f" · 滞空 {release_time:.1f} 秒"
        self._text(
            canvas,
            live_text,
            (1185, 953),
            0.29,
            (190, 200, 210),
        )
        self._text(canvas, release_text, (1185, 978), 0.29, (190, 200, 210))

    @staticmethod
    def _thumbnail(snapshot: DisplaySnapshot, entry: Any) -> np.ndarray | None:
        cached = getattr(entry, "thumbnail", None)
        if isinstance(cached, np.ndarray):
            return cached
        frame_index = getattr(entry, "thumbnail_frame_index", None)
        if frame_index is None:
            return None
        for frame in getattr(snapshot, "history_frames", ()):
            if getattr(frame, "frame_index", None) != frame_index:
                continue
            image = getattr(frame, "image", None)
            if image is None:
                return None
            for detection in getattr(frame, "detections", ()):
                if getattr(detection, "code", None) != getattr(entry, "code", None):
                    continue
                box = getattr(entry, "thumbnail_box", None)
                if box is None:
                    continue
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                height, width = image.shape[:2]
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 > x1 and y2 > y1:
                    return image[y1:y2, x1:x2].copy()
        return None

    def _text(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
        *,
        clip: tuple[int, int, int, int] | None = None,
    ) -> None:
        del image
        self._text_commands.append((text, origin, max(12, round(scale * 34)), color, clip))

    def _draw_text_commands(self, canvas: np.ndarray) -> None:
        if not self._text_commands:
            return
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for text, origin, size, color, clip in self._text_commands:
            font = _font(size)
            fill = (color[2], color[1], color[0], 255)
            stroke = (17, 14, 12, 255)
            if clip is None:
                draw.text(
                    origin,
                    text,
                    font=font,
                    anchor="ls",
                    fill=fill[:3],
                    stroke_width=0,
                )
                continue
            left, top, right, bottom = clip
            if right <= left or bottom <= top:
                continue
            draw = ImageDraw.Draw(image)
            fitted = _fit_text_anchor(draw, text, font, origin, clip)
            if fitted is None:
                continue
            font, anchor, bounds = fitted
            draw.rounded_rectangle(
                (bounds[0] - 7, bounds[1] - 4, bounds[2] + 7, bounds[3] + 4),
                radius=4,
                fill=(12, 20, 28),
            )
            draw.text(
                anchor,
                text,
                font=font,
                anchor="ls",
                fill=fill,
                stroke_width=1,
                stroke_fill=stroke,
            )
        canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        self._text_commands.clear()

    @staticmethod
    def _mode_name(mode: str) -> str:
        return {
            "recon": "侦查",
            "standby": "起飞前预览",
            "hold_follow": "盘旋跟随处理画面",
            "hold_live": "盘旋实时画面",
            "drop": "投放实时画面",
            "done": "任务完成",
            "aborted": "任务中止",
        }.get(mode, mode or "等待")

    @staticmethod
    def _stage_name(stage: str) -> str:
        return dict(_STAGES).get(stage, "等待画面")

    @staticmethod
    def _label_name(label: str) -> str:
        return {"Target": "目标", "Impact": "预测落点"}.get(label, label)

    @staticmethod
    def _event_name(kind: str) -> str:
        names = {
            "drop_check": "投放判据满足",
            "release": "投放记录",
            "drop_command_pending": "正在发送投放指令",
            "drop_command_sent": "投放指令已发出",
            "drop_command_failed": "投放指令失败",
            "drop_command_unknown": "投放结果未知",
            "drop_released": "投放已记录",
        }
        return names.get(kind, kind)
