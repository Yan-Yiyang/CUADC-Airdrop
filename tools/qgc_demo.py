"""任务二 A 区的确定性相机示意；几何与入库 seed=0 场景一致。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from airdrop.georef.camera import CameraModel
from airdrop.georef.project import ned_to_pixel

WIDTH, HEIGHT = 1280, 720
ALTITUDE_M = 15.0
HFOV_RAD = 1.74
ORIGIN_LLA = (8.546164, 47.397971, 488.0)


@dataclass(frozen=True, slots=True)
class DemoWell:
    """天井中心采用 NED 的北、东顺序，朝向采用世界 ENU 偏航。"""

    code: int | None
    north_m: float
    east_m: float
    yaw_deg: float

    @property
    def ned(self) -> tuple[float, float, float]:
        return self.north_m, self.east_m, 0.0

    @property
    def number(self) -> int:
        if self.code is None:
            raise ValueError("空天井没有数字靶标")
        return self.code


WELLS = (
    DemoWell(94, 6.91, 213.11, 339.5),
    DemoWell(12, -13.99, 178.01, 109.0),
    DemoWell(56, -10.80, 200.30, 146.9),
    DemoWell(None, 15.83, 180.89, 291.6),
)
SELECTED = WELLS[2]


@lru_cache(maxsize=1)
def camera() -> CameraModel:
    """复用项目相机尺寸、视场与默认下视外参，不依赖工具子模块。"""
    focal = WIDTH / (2 * math.tan(HFOV_RAD / 2))
    return CameraModel(
        np.array([[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]]),
        image_size=(WIDTH, HEIGHT),
    )


def project(point: tuple[float, float, float], north: float, east: float) -> tuple[float, float]:
    """与正式叠加共用投影，防止预览准星与实体靶标错位。"""
    result = ned_to_pixel(
        point,
        camera=camera(),
        position_ned=(north, east, -ALTITUDE_M),
        image_size=(WIDTH, HEIGHT),
        euler_deg=(0.0, 0.0, 0.0),
        distort=False,
    )
    if result.pixel is None:
        raise ValueError("合成场景中的地面点不可投影")
    return result.pixel


@lru_cache(maxsize=4)
def _well_sprite(code: int | None) -> np.ndarray:
    """一米方形加等边三角顶、蓝底天井，中央两块白底黑字数字板。"""
    from airdrop.qgcvideo.render import _font

    sprite = np.zeros((520, 320, 4), np.uint8)
    # 每米 200 像素；局部 +E 指向屋顶，+N 指向靶标左侧。
    points = np.array([(260, 400), (260, 200), (160, 27), (60, 200), (60, 400)])
    cv2.fillConvexPoly(sprite, points, (172, 75, 32, 255), cv2.LINE_AA)
    cv2.polylines(sprite, [points], True, (190, 192, 190, 255), 3, cv2.LINE_AA)
    if code is not None:
        rgba = Image.fromarray(cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA))
        draw = ImageDraw.Draw(rgba)
        for offset, digit in enumerate(f"{code:02d}"):
            left = 100 + offset * 60
            draw.rectangle((left, 240, left + 59, 360), fill=(244, 244, 239, 255))
            mask = Image.new("L", (160, 180))
            ink = ImageDraw.Draw(mask)
            font = _font(140)
            bounds = ink.textbbox((0, 0), digit, font=font, stroke_width=2)
            ink.text((-bounds[0], -bounds[1]), digit, font=font, fill=255, stroke_width=2)
            glyph = mask.crop((0, 0, bounds[2] - bounds[0], bounds[3] - bounds[1]))
            glyph = glyph.resize((43, 80), Image.Resampling.LANCZOS)
            rgba.paste((17, 18, 20, 255), (left + 8, 260), glyph)
        sprite = cv2.cvtColor(np.asarray(rgba), cv2.COLOR_RGBA2BGRA)
    return sprite


def _world_points(well: DemoWell, local: np.ndarray) -> np.ndarray:
    angle = math.radians(well.yaw_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return local @ rotation.T + (well.east_m, well.north_m)


def well_box(well: DemoWell, north: float, east: float) -> tuple[float, float, float, float]:
    """框住实际五边形轮廓，留少量边缘而不放大实体。"""
    outline = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5 + math.sqrt(3) / 2, 0], [0.5, 0.5], [-0.5, 0.5]]
    )
    world = _world_points(well, outline)
    pixels = np.array([project((n, e, 0.0), north, east) for e, n in world])
    low, high = pixels.min(axis=0) - 5, pixels.max(axis=0) + 5
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


@lru_cache(maxsize=4)
def render_scene(north: float, east: float) -> np.ndarray:
    """15m 下视纹理与靶标按米制投影；不把计分示意圈画在地面上。"""
    focal = float(camera().camera_matrix[0, 0])
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    world_e = east + (xx - WIDTH / 2) * ALTITUDE_M / focal
    world_n = north - (yy - HEIGHT / 2) * ALTITUDE_M / focal
    hash_grid = np.floor(world_e * 33) * 12.9898 + np.floor(world_n * 33) * 78.233
    grain = (np.remainder(np.sin(hash_grid) * 43758.5453, 1) - 0.5) * 3
    stain = np.sin(world_e * 0.61 + world_n * 0.27) * np.cos(world_n * 0.73) * 3
    in_zone = (abs(world_e - 200) <= 33) & (abs(world_n) <= 33)
    concrete = np.stack((143 + grain + stain, 149 + grain + stain, 147 + grain + stain), axis=-1)
    grass = np.stack((77 + grain, 99 + grain + stain, 82 + grain), axis=-1)
    image = np.where(in_zone[..., None], concrete, grass).clip(0, 255).astype(np.uint8)
    joints = (np.remainder(world_e + 1.1, 5.5) < 0.025) | (np.remainder(world_n, 5.5) < 0.025)
    image[in_zone & joints] = (116, 122, 121)
    boundary = ((abs(abs(world_e - 200) - 30) < 0.065) & (abs(world_n) <= 30)) | (
        (abs(abs(world_n) - 30) < 0.065) & (abs(world_e - 200) <= 30)
    )
    image[boundary] = (223, 226, 222)
    # 透视映射同时转动底板和数字纸，保持随机朝向与真实图像裁剪口径。
    source = np.array([[0, 0], [319, 0], [319, 519], [0, 519]], np.float32)
    local = np.array([[1.5, 0.8], [1.5, -0.795], [-1.095, -0.795], [-1.095, 0.8]])
    for well in WELLS:
        world = _world_points(well, local)
        destination = np.array([project((n, e, 0.0), north, east) for e, n in world], np.float32)
        matrix = cv2.getPerspectiveTransform(source, destination)
        layer = cv2.warpPerspective(_well_sprite(well.code), matrix, (WIDTH, HEIGHT))
        alpha = layer[:, :, 3:4].astype(np.float32) / 255
        shadow = cv2.warpAffine(
            layer[:, :, 3], np.array([[1, 0, 9], [0, 1, 6]], np.float32), (WIDTH, HEIGHT)
        )
        image = (image * (1 - shadow[..., None] / 255 * 0.16)).astype(np.uint8)
        image = (image * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
    return np.frombuffer(image.tobytes(), np.uint8).reshape((HEIGHT, WIDTH, 3))
