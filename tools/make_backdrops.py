"""生成场景"底图"贴图（PNG）：目标区地面纹理 + 比赛区域外的随机航拍干扰底图。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run make-backdrops
    ./.venv/Scripts/python.exe -m airdrop.run make-backdrops --seed 3 --size 640

产出两类贴图（名字分别取自 ``make_world.ZONE_GROUND_COLORS`` 与
``make_world.AERIAL_TILES``，改名字要两边一起改）：

* ``ground_*.png``：目标区"底色 + 少量对应地面纹理"（常见路面色：沥青/水泥/土面/
  砖面/砂石）。世界里由 ``seed`` 给 A/B 两区各抽一种，模拟不同场地；
* ``aerial_*.png``：农田/道路/房屋/路面标线这类"航拍底图"（**不写数字/文字**，免得
  跟真靶标的数字板混淆）。世界生成器会把它们随机
  铺在**比赛区域之外**（``--aerial-patches``，默认 6 块），给下视画面造干扰背景，
  用来锻炼检测与识别的抗扰能力（白底靶标之外的东西不该被当成目标）。

贴图由 ``seed`` 决定、可复现。只依赖 Pillow，且**在函数体内导入**：
``import tools.make_backdrops`` 不加载 PIL。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from tools import make_world

if TYPE_CHECKING:  # 只为类型标注；运行时不导入 PIL（惰性导入的约定）
    from PIL import Image, ImageDraw

# ----------------------------------------------------------------------
# 默认值（命令行选项的默认值直接取自这里）
# ----------------------------------------------------------------------
OUT_DIR = make_world.OUT_DIR
SEED = 0
SIZE_PX = 512
#: 生成哪一类底图：all（两类都生成）/ ground（只目标区地面纹理）/ aerial（只航拍干扰底图）
PARTS = "all"

#: 航拍底图的地表底色（农田/荒地/裸土）——从这里起步，再叠地块/道路/建筑
TERRAIN_COLORS: tuple[tuple[int, int, int], ...] = (
    (92, 106, 66),
    (104, 116, 72),
    (126, 118, 88),
    (138, 128, 100),
    (86, 92, 70),
)
#: 地块边界（田埂/沟渠）、植被、道路、屋顶、标线的颜色
FIELD_LINE_COLOR = (60, 66, 46)
VEGETATION_COLORS: tuple[tuple[int, int, int], ...] = (
    (58, 82, 48),
    (46, 70, 42),
    (70, 92, 54),
)
ROAD_COLOR = (74, 74, 76)
ROAD_LINE_COLOR = (208, 206, 190)
ROOF_COLORS: tuple[tuple[int, int, int], ...] = (
    (152, 152, 154),
    (128, 128, 132),
    (172, 160, 140),
    (112, 112, 118),
)
SHADOW_COLOR = (52, 52, 56)
MARKING_COLOR = (222, 220, 206)


@dataclass(frozen=True, slots=True)
class MakeBackdropsConfig:
    """底图生成器配置（字段 = 命令行能覆盖的东西）。"""

    out_dir: Path = make_world.OUT_DIR
    seed: int = SEED
    size: int = SIZE_PX
    parts: str = PARTS


def build_config(**overrides) -> MakeBackdropsConfig:
    """按关键字覆盖派生一份配置（``dataclasses.replace``）。"""
    if "out_dir" in overrides:
        overrides["out_dir"] = Path(overrides["out_dir"])
    return replace(MakeBackdropsConfig(), **overrides)


# ----------------------------------------------------------------------
# 公共小工具
# ----------------------------------------------------------------------
def _shift(color: tuple[int, int, int], rng: Random, spread: int = 18) -> tuple[int, int, int]:
    """颜色整体加一点随机偏移（地块/草地看起来才不会像纯色块）。"""
    channels = [max(0, min(255, channel + rng.randint(-spread, spread))) for channel in color]
    return channels[0], channels[1], channels[2]


def _direction(angle_deg: float, length: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return math.cos(radians) * length, math.sin(radians) * length


def _rgb01(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(channel * 255)) for channel in color)  # type: ignore[return-value]


def _scatter_noise(image: Image.Image, rng: Random, dots: int) -> None:
    """低对比度的细颗粒（航拍和路面的"少量纹理"都靠它）。"""
    from PIL import ImageDraw as PilImageDraw

    draw = PilImageDraw.Draw(image, "RGBA")
    for _ in range(dots):
        x = rng.uniform(0.0, image.width)
        y = rng.uniform(0.0, image.height)
        shade = rng.choice(((0, 0, 0, 26), (255, 255, 255, 22)))
        draw.point([x, y], fill=shade)
        if rng.random() < 0.3:
            draw.rectangle([x, y, x + rng.randint(1, 3), y + rng.randint(1, 3)], fill=shade)


# ----------------------------------------------------------------------
# 目标区地面纹理：纯色底 + 少量对应纹理（污渍/补丁/裂纹/颗粒）
# ----------------------------------------------------------------------
def draw_ground_texture(variant: str, *, seed: int = SEED, size: int = SIZE_PX) -> Image.Image:
    """画某种目标区底色的地面纹理（同 variant/seed/size 永远画出同一张）。

    底色取 ``make_world.ZONE_GROUND_COLORS[variant]``；纹理只是同色系的少量变化
    （大块污渍 + 细颗粒 + 几条裂纹/接缝），别做成花哨图案——规则里"底色"是纯色。
    """
    from PIL import Image as PilImage
    from PIL import ImageDraw as PilImageDraw
    from PIL import ImageFilter

    if variant not in make_world.ZONE_GROUND_COLORS:
        raise ValueError(f"未知的目标区底色：{variant!r}")
    rng = Random(f"ground-texture:{seed}:{variant}")
    base = _rgb01(make_world.ZONE_GROUND_COLORS[variant])
    image = PilImage.new("RGB", (size, size), base)
    draw = PilImageDraw.Draw(image, "RGBA")
    # 大块污渍/补丁：亮度只动一点点，远看还是"一块纯色"
    for _ in range(rng.randint(10, 18)):
        radius = rng.uniform(size * 0.05, size * 0.22)
        x = rng.uniform(0.0, size)
        y = rng.uniform(0.0, size)
        delta = rng.randint(-14, 14)
        tint = tuple(max(0, min(255, channel + delta)) for channel in base)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(tint[0], tint[1], tint[2], rng.randint(70, 150)),
        )
    # 裂纹/接缝：几条很淡的细线（只有沥青/水泥那几种才画）
    if variant in ("ground_asphalt", "ground_concrete", "ground_concrete_old"):
        for _ in range(rng.randint(2, 5)):
            x = rng.uniform(0.0, size)
            y = rng.uniform(0.0, size)
            points = [(x, y)]
            for _ in range(rng.randint(3, 6)):
                x += rng.uniform(-size * 0.25, size * 0.25)
                y += rng.uniform(-size * 0.25, size * 0.25)
                points.append((x, y))
            draw.line(points, fill=(30, 30, 32, 60), width=2)
    _scatter_noise(image, rng, dots=2200)
    return image.filter(ImageFilter.GaussianBlur(0.5))


# ----------------------------------------------------------------------
# 航拍干扰底图
# ----------------------------------------------------------------------
def _draw_fields(draw: ImageDraw.ImageDraw, rng: Random, size: int) -> None:
    """把画面切成带抖动的方格地块（田埂用深色描边）。"""
    divisions = rng.choice((3, 4))
    step = size / divisions
    jitter = step * 0.18
    points: list[list[tuple[float, float]]] = []
    for row in range(divisions + 1):
        line: list[tuple[float, float]] = []
        for column in range(divisions + 1):
            x = column * step
            y = row * step
            if 0 < column < divisions:
                x += rng.uniform(-jitter, jitter)
            if 0 < row < divisions:
                y += rng.uniform(-jitter, jitter)
            line.append((x, y))
        points.append(line)
    for row in range(divisions):
        for column in range(divisions):
            quad = (
                points[row][column],
                points[row][column + 1],
                points[row + 1][column + 1],
                points[row + 1][column],
            )
            color = _shift(rng.choice(TERRAIN_COLORS), rng, spread=14)
            draw.polygon(quad, fill=color, outline=FIELD_LINE_COLOR)


def _draw_vegetation(draw: ImageDraw.ImageDraw, rng: Random, size: int) -> None:
    for _ in range(rng.randint(10, 22)):
        radius = rng.uniform(8.0, 34.0)
        x = rng.uniform(0.0, size)
        y = rng.uniform(0.0, size)
        color = (*rng.choice(VEGETATION_COLORS), rng.randint(120, 210))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def _draw_roads(draw: ImageDraw.ImageDraw, rng: Random, size: int) -> None:
    for _ in range(rng.randint(1, 2)):
        angle = rng.uniform(0.0, 180.0)
        width = rng.uniform(9.0, 18.0)
        offset = rng.uniform(-size * 0.35, size * 0.35)
        dx, dy = _direction(angle, size * 1.2)
        px, py = _direction(angle + 90.0, offset)
        center_x, center_y = size / 2 + px, size / 2 + py
        draw.line(
            [center_x - dx, center_y - dy, center_x + dx, center_y + dy],
            fill=ROAD_COLOR,
            width=int(width),
        )
        if rng.random() < 0.5:  # 一半的路画中间虚线
            dashes = 14
            for index in range(dashes):
                t = index / dashes
                start = (center_x - dx + 2 * dx * t, center_y - dy + 2 * dy * t)
                end = (start[0] + dx / dashes * 0.5, start[1] + dy / dashes * 0.5)
                draw.line([start, end], fill=ROAD_LINE_COLOR, width=2)


def _draw_buildings(draw: ImageDraw.ImageDraw, rng: Random, size: int) -> None:
    for _ in range(rng.randint(4, 10)):
        width = rng.uniform(12.0, 46.0)
        height = rng.uniform(12.0, 40.0)
        x = rng.uniform(0.0, size - width)
        y = rng.uniform(0.0, size - height)
        draw.rectangle([x + 3, y + 3, x + width + 3, y + height + 3], fill=SHADOW_COLOR)
        draw.rectangle(
            [x, y, x + width, y + height],
            fill=rng.choice(ROOF_COLORS),
            outline=(36, 36, 40),
        )


def _draw_markings(draw: ImageDraw.ImageDraw, rng: Random, size: int) -> None:
    """路面标线：几排短白线（停车位/车道那种），**不写任何数字或文字**。

    ⚠ 贴图里不许出现可读的数字/文字：那会和真靶标（数字板）混淆。
    """
    for _ in range(rng.randint(1, 2)):
        base_x = rng.uniform(0.0, size * 0.7)
        base_y = rng.uniform(0.0, size * 0.7)
        for index in range(rng.randint(4, 8)):
            x = base_x + index * 14.0
            draw.line([x, base_y, x, base_y + 30.0], fill=MARKING_COLOR, width=2)


def draw_aerial_tile(index: int, *, seed: int = SEED, size: int = SIZE_PX) -> Image.Image:
    """画第 ``index`` 张航拍干扰底图（同 index/seed/size 永远画出同一张；不写数字/文字）。"""
    from PIL import Image as PilImage
    from PIL import ImageDraw as PilImageDraw
    from PIL import ImageFilter

    rng = Random(f"aerial-tile:{seed}:{index}")
    image = PilImage.new("RGB", (size, size), _shift(rng.choice(TERRAIN_COLORS), rng, spread=8))
    draw = PilImageDraw.Draw(image, "RGBA")
    _draw_fields(draw, rng, size)
    _draw_vegetation(draw, rng, size)
    _draw_roads(draw, rng, size)
    _draw_buildings(draw, rng, size)
    _draw_markings(draw, rng, size)
    _scatter_noise(image, rng, dots=1600)
    return image.filter(ImageFilter.GaussianBlur(0.6))


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
def main(
    *,
    out_dir: Path | str = make_world.OUT_DIR,
    seed: int = SEED,
    size: int = SIZE_PX,
    parts: str = PARTS,
) -> int:
    """把底图写进 ``<out_dir>/materials/textures/``；返回进程退出码。

    ``parts``：``all`` 两类都写；``ground`` 只写目标区地面纹理（**不会覆盖**
    ``fetch-aerial`` 抓来的真实航拍底图）；``aerial`` 只写航拍干扰底图。
    """
    if parts not in ("all", "ground", "aerial"):
        print(f"FAIL: parts 只能是 all / ground / aerial，收到 {parts!r}")
        return 1
    if size < 128:
        print(f"FAIL: 贴图尺寸太小（要 >= 128 像素），收到 {size!r}")
        return 1
    target = Path(out_dir) / make_world.TEXTURE_DIR
    target.mkdir(parents=True, exist_ok=True)
    want_ground = parts in ("all", "ground")
    want_aerial = parts in ("all", "aerial")
    expected: set[str] = set()
    stale_files: list[Path] = []
    if want_ground:
        expected |= {f"{name}.png" for name in make_world.ZONE_GROUND_COLORS}
        stale_files += sorted(target.glob("ground_*.png"))
    if want_aerial:
        expected |= {f"{name}.png" for name in make_world.AERIAL_TILES}
        stale_files += sorted(target.glob("aerial_*.png"))
    for stale in stale_files:
        if stale.name not in expected:
            stale.unlink()
            print(f"[OK  ] 清掉多余的旧贴图 {stale.name}")
    if want_ground:
        for variant in make_world.ZONE_GROUND_COLORS:
            path = target / f"{variant}.png"
            draw_ground_texture(variant, seed=seed, size=size).save(path)
            print(f"[OK  ] {path}（目标区地面纹理，种子 {seed}）")
    if want_aerial:
        for index, name in enumerate(make_world.AERIAL_TILES, start=1):
            path = target / f"{name}.png"
            draw_aerial_tile(index, seed=seed, size=size).save(path)
            print(f"[OK  ] {path}（航拍干扰底图，种子 {seed}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
