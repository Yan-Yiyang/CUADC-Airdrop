"""把赛区世界渲染成一张**俯视预览图**（PNG，带中文标注），供人眼核对布局。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run preview-world
    ./.venv/Scripts/python.exe -m airdrop.run preview-world --round 1 --out /tmp/r1.png

画的内容（**只读世界文件**，不重新生成）：草地底色、跑道 / 起降区标记 / 操纵区、
两个目标区的底色块（贴真实贴图）、白色区界、天井（A 蓝 / B 红，中位数那座标出它的
两位数）、比赛区域外的航拍干扰底图（按 pose 旋转、角上露草地——和 Gazebo 里一致）
并标注编号。

几点约定：

* 预览图是**给人看的辅助图**，不参与世界生成；世界文件才是唯一事实。
* 文字用系统中文字体渲染（Windows 的 msyh.ttc / simhei.ttf，Linux 的 Noto Sans CJK），
  都没有时退回 Pillow 内置字体并打警告（那时中文可能显示成方框，不影响世界本身）。
* 预览图上的编号/标签只存在于这张 PNG 里，**不会**出现在世界 SDF 或任何贴图里。

只依赖标准库 + Pillow（Pillow 在函数体内导入）。
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

from tools import make_world

#: 世界目录与轮次（预览哪个世界）
WORLD_DIR = make_world.OUT_DIR
ROUND = 2
#: 输出文件（默认写在世界目录里；已在 .gitignore 里，属于临时产物）
OUT_PATH = make_world.OUT_DIR / "preview.png"
#: 目标画布宽度（像素）——会自动换算 px/m，保证整张图大约这么宽
TARGET_WIDTH_PX = 1500
#: 中文字体候选（按顺序取第一个存在的）
CJK_FONTS: tuple[Path, ...] = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)


@dataclass(frozen=True, slots=True)
class PreviewWorldConfig:
    """预览图配置（字段 = 命令行能覆盖的东西）。"""

    world_dir: Path = WORLD_DIR
    round_no: int = ROUND
    out_path: Path = OUT_PATH


def build_config(**overrides) -> PreviewWorldConfig:
    """按关键字覆盖派生一份配置（``dataclasses.replace``）。"""
    if "world_dir" in overrides:
        overrides["world_dir"] = Path(overrides["world_dir"])
    if "out_path" in overrides:
        overrides["out_path"] = Path(overrides["out_path"])
    return replace(PreviewWorldConfig(), **overrides)


# ----------------------------------------------------------------------
# 字体与画字
# ----------------------------------------------------------------------
def _load_font(size: int):
    """取一个能画中文的字体；找不到就退回 Pillow 内置字体（调用方据此打警告）。"""
    from PIL import ImageFont

    for path in CJK_FONTS:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size), path
            except OSError:  # pragma: no cover - 字体损坏时才走到
                continue
    return ImageFont.load_default(size=size), None


def _text(draw, position, text, font, fill=(255, 255, 255), halo=(0, 0, 0)) -> None:
    """带描边的文字（压在照片上也读得清）。"""
    x, y = position
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=halo)
    draw.text((x, y), text, font=font, fill=fill)


# ----------------------------------------------------------------------
# 读世界文件
# ----------------------------------------------------------------------
def _pose_of(element: ET.Element) -> tuple[float, ...]:
    pose = element.find("pose")
    if pose is None or not pose.text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(float(value) for value in pose.text.split())


def _diffuse_of(visual: ET.Element) -> tuple[float, float, float] | None:
    text = visual.findtext("material/diffuse")
    if text is None:
        return None
    values = [float(value) for value in text.split()]
    return (values[0], values[1], values[2]) if len(values) >= 3 else None


def _mesh_scale(visual: ET.Element) -> tuple[float, float]:
    text = visual.findtext("geometry/mesh/scale") or "1 1 1"
    values = [float(value) for value in text.split()]
    if len(values) >= 2:
        return values[0], values[1]
    return (values[0] if values else 1.0), 1.0


def _mesh_uri(visual: ET.Element) -> str | None:
    return visual.findtext("geometry/mesh/uri")


def _read_world(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"{path} 里没有 <world>")
    return world


def _ground_color(world: ET.Element) -> tuple[int, int, int]:
    for model in world.findall("model"):
        if model.get("name") != "ground_plane":
            continue
        for visual in model.iter("visual"):
            diffuse = _diffuse_of(visual)
            if diffuse:
                return tuple(round(channel * 255) for channel in diffuse)  # type: ignore[return-value]
    return (89, 140, 64)


def _model(world: ET.Element, name: str) -> ET.Element | None:
    return next((model for model in world.findall("model") if model.get("name") == name), None)


# ----------------------------------------------------------------------
# 画各个图层
# ----------------------------------------------------------------------
def _paste_ground_textures(image, markings, world_dir: Path, to_px, scale: float) -> None:
    """目标区底色块：贴真实贴图（世界里的 mesh visual）。"""
    from PIL import Image

    for visual in markings.iter("visual"):
        uri = _mesh_uri(visual)
        if not uri:
            continue
        variant = Path(uri).stem.removeprefix("sheet_")
        texture_path = Path(world_dir) / make_world.TEXTURE_DIR / f"{variant}.png"
        if not texture_path.is_file():
            continue
        size_x, size_y = _mesh_scale(visual)
        pixel_x = max(2, int(size_x * scale))
        pixel_y = max(2, int(size_y * scale))
        ground = Image.open(texture_path).convert("RGB").resize((pixel_x, pixel_y))
        pose = _pose_of(visual)
        x, y = to_px(pose[0], pose[1])
        image.paste(ground, (int(x - pixel_x / 2), int(y - pixel_y / 2)))


def _paste_aerial_patches(image, patches, world_dir: Path, to_px, scale: float) -> None:
    """航拍干扰底图：旋转方形面片（角上露草地，和 Gazebo 一致）。"""
    from PIL import Image

    for east, north, yaw, size, uri in patches:
        name = Path(uri).stem.removeprefix("sheet_")
        texture_path = Path(world_dir) / make_world.TEXTURE_DIR / f"{name}.png"
        if not texture_path.is_file():
            continue
        tile = Image.open(texture_path).convert("RGBA")
        pixel = max(2, int(size * scale))
        tile = tile.resize((pixel, pixel))
        tile = tile.rotate(
            yaw, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
        )
        x, y = to_px(east, north)
        image.paste(tile, (int(x - tile.width / 2), int(y - tile.height / 2)), tile)


def _draw_boxes(draw, markings, to_px) -> None:
    """跑道 / 起降区标记 / 操纵区 / 区界：世界里的 box visual 直接画成矩形。"""
    for visual in markings.iter("visual"):
        if _mesh_uri(visual):
            continue
        size_text = visual.findtext("geometry/box/size")
        if size_text is None:
            continue
        size_x, size_y, _size_z = (float(value) for value in size_text.split())
        pose = _pose_of(visual)
        diffuse = _diffuse_of(visual)
        color = tuple(round(channel * 255) for channel in diffuse) if diffuse else (200, 200, 200)
        x0, y0 = to_px(pose[0] - size_x / 2, pose[1] + size_y / 2)
        x1, y1 = to_px(pose[0] + size_x / 2, pose[1] - size_y / 2)
        draw.rectangle([x0, y0, x1, y1], fill=color)


def _draw_wells(draw, wells, to_px, scale: float) -> None:
    """天井：小圆点 + 编号；数字靶标拼出的两位数标在编号后面。"""
    small, _path = _load_font(13)
    for model in wells:
        name = model.get("name") or ""
        pose = _pose_of(model)
        x, y = to_px(pose[0], pose[1])
        radius = max(3.0, 0.8 * scale)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(90, 150, 240) if name.startswith("well_a") else (230, 90, 90),
        )
        label = name.removeprefix("well_")
        digits = ""
        for visual in model.iter("visual"):
            uri = _mesh_uri(visual) or ""
            if "sheet_digit_" in uri:
                digits += Path(uri).stem.rsplit("_", 1)[1]
        if len(digits) == 2:
            label = f"{label}={digits}"
        # 四座天井的标签上下错开，免得挤在一起
        row = int(name[-1]) if name[-1].isdigit() else 1
        offset_y = -radius - 8 if row % 2 else radius + 2
        _text(draw, (x + radius + 2, y + offset_y), label, small)


def _draw_patch_numbers(draw, patches, to_px) -> None:
    number_font, _path = _load_font(18)
    for index, (east, north, _yaw, _size, _uri) in enumerate(patches, start=1):
        x, y = to_px(east, north)
        _text(draw, (x - 8, y - 10), str(index), number_font, fill=(255, 230, 120))


def _draw_titles(draw, world_file: Path, height: int) -> Path | None:
    title_font, cjk_path = _load_font(22)
    legend_font, _legend_path = _load_font(15)
    _text(draw, (10, 8), f"CUADC 赛区俯视预览（{world_file.name}）", title_font)
    legend_lines = (
        "蓝点 = A 区天井（两位数来自数字靶标）；红点 = B 区天井",
        "黄字编号 = 比赛区域外的航拍干扰底图（角上露草地 = 世界地面）",
        "中间 = 跑道 / 起降区标记 / 操纵区；两侧 = 目标区底色块（真实贴图）",
    )
    for row, line in enumerate(legend_lines):
        _text(draw, (10, height - 66 + row * 20), line, legend_font)
    return cjk_path


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------
def _features(world: ET.Element, world_dir: Path) -> dict:
    """从世界文件里抽出要画的东西（标线模型 / 天井 / 航拍块）。"""
    markings = _model(world, "cuadc_field_markings")
    wells = [
        model for model in world.findall("model") if (model.get("name") or "").startswith("well_")
    ]
    patches: list[tuple[float, float, float, float, str]] = []
    surroundings = _model(world, "cuadc_surroundings")
    if surroundings is not None:
        for visual in surroundings.iter("visual"):
            pose = _pose_of(visual)
            size = max(_mesh_scale(visual))
            patches.append((pose[0], pose[1], math.degrees(pose[5]), size, _mesh_uri(visual) or ""))
    return {"markings": markings, "wells": wells, "patches": patches}


def _extents(
    patches: list[tuple[float, float, float, float, str]],
) -> tuple[float, float, float, float]:
    east_values = [-make_world.RUNWAY_LENGTH_M / 2, make_world.RUNWAY_LENGTH_M / 2]
    north_values = [-make_world.RUNWAY_WIDTH_M / 2, make_world.RUNWAY_WIDTH_M / 2]
    for zone_east, zone_north in make_world.TARGET_ZONE_CENTERS.values():
        half = (make_world.TARGET_ZONE_SIDE_M + make_world.ZONE_GROUND_MARGIN_M) / 2
        east_values += [zone_east - half, zone_east + half]
        north_values += [zone_north - half, zone_north + half]
    for east, north, _yaw, size, _uri in patches:
        radius = size * math.sqrt(2.0) / 2.0
        east_values += [east - radius, east + radius]
        north_values += [north - radius, north + radius]
    return (
        min(east_values) - 10.0,
        max(east_values) + 10.0,
        min(north_values) - 10.0,
        max(north_values) + 10.0,
    )


def render(world_dir: Path, round_no: int, out_path: Path) -> int:
    from PIL import Image, ImageDraw

    world_file = Path(world_dir) / f"{make_world.WORLD_PREFIX}_r{round_no}.sdf"
    if not world_file.is_file():
        print(f"FAIL: 找不到世界文件 {world_file}")
        return 1
    try:
        world = _read_world(world_file)
    except (ET.ParseError, ValueError) as exc:
        print(f"FAIL: 读世界失败：{exc}")
        return 1

    features = _features(world, Path(world_dir))
    patches = features["patches"]
    east_min, east_max, north_min, north_max = _extents(patches)
    scale = min(3.0, max(0.6, TARGET_WIDTH_PX / max(1.0, east_max - east_min)))
    width = int((east_max - east_min) * scale)
    height = int((north_max - north_min) * scale)

    def to_px(east: float, north: float) -> tuple[float, float]:
        return ((east - east_min) * scale, (north_max - north) * scale)

    image = Image.new("RGB", (width, height), _ground_color(world))
    draw = ImageDraw.Draw(image)
    if features["markings"] is not None:
        _paste_ground_textures(image, features["markings"], world_dir, to_px, scale)
    _paste_aerial_patches(image, patches, world_dir, to_px, scale)
    if features["markings"] is not None:
        _draw_boxes(draw, features["markings"], to_px)
    _draw_wells(draw, features["wells"], to_px, scale)
    _draw_patch_numbers(draw, patches, to_px)
    cjk_path = _draw_titles(draw, world_file, height)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    if cjk_path is None:
        print("WARN: 没找到中文字体，预览里的中文可能显示成方框（世界本身不受影响）")
    print(f"[OK  ] {out_path}（{width}x{height}，{len(patches)} 块航拍底图）")
    return 0


def main(
    *,
    world_dir: Path | str = WORLD_DIR,
    round_no: int = ROUND,
    out_path: Path | str = OUT_PATH,
) -> int:
    """渲染俯视预览图；返回进程退出码（0 成功 / 1 世界文件读不了）。"""
    if round_no not in (1, 2):
        print(f"FAIL: 轮次只能是 1 或 2，收到 {round_no!r}")
        return 1
    return render(Path(world_dir), round_no, Path(out_path))


if __name__ == "__main__":
    sys.exit(main())
