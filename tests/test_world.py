"""CUADC 赛区世界（``sim/worlds/cuadc/``）的离线校验：生成结果、几何、资产。

不启动 Gazebo（离线、秒级）：真跑 SITL 的步骤见 ``docs/simulation_world.md``。
这里盯的是五件容易悄悄坏掉的事：

1. **生成结果 == 入库文件**：世界与网格都是 ``tools/make_world.py`` 的产物，
   改了生成器却不重跑（或手改了生成物）要立刻失败；
2. **几何符合规则**：A/B 两个 60x60m 目标区在起飞线两端（±200, 0）、各距起降区约
   200m；8 座天井在区内随机摆放但**两两间距 > 20m**；五边形环壁（厚 5mm、高 400mm）；
3. **规则里只是示意的东西不许出现在场地上**：3.3.3 的 r=4m/6m 打击圈（规则图 5）、
   规则文本提到但图里没有的"天井箭头"（默认不画，``well_arrows=True`` 才画）；
4. **靶标摆对**：第一轮每区 3 张图片靶、第二轮每区 3 个两位数（中位数落在第 3 座）；
   同一区的天井必须**同一种颜色**（A 蓝 / B 红，规则 3.3）；
5. **资产齐全且不是坏的那种**：世界引用的网格 → 材质 → 贴图逐级存在；
   数字板必须是**白底黑字**。

另有一条防 z-fighting 的回归：贴地标线不许有"共面且重叠"的两层
（共面重叠会渲染成贴图打架）。
"""

from __future__ import annotations

import math
import shutil
import struct
import urllib.parse
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from tools import make_world

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_DIR = REPO_ROOT / "sim" / "worlds" / "cuadc"
WORK_ROOT = REPO_ROOT / ".world-test-tmp"
SIM_DIR = REPO_ROOT / "sim"

ROUNDS = (1, 2)
WELL_NAMES = ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4")


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉（不用 ``tmp_path``：见 AGENTS）。"""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / "scenario"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------
# 生成结果 == 入库文件
# ----------------------------------------------------------------------
@pytest.mark.parametrize("round_no", ROUNDS)
def test_generated_world_matches_committed_file(round_no: int) -> None:
    committed = (WORLD_DIR / f"cuadc_recon_strike_r{round_no}.sdf").read_text(encoding="utf-8")
    assert make_world.build_world(round_no) == committed, (
        "入库的世界与生成器不一致——改完生成器请重跑 python -m airdrop.run make-world"
    )


def test_generated_meshes_match_committed_files() -> None:
    for relative, text in make_world.build_meshes().items():
        committed = (WORLD_DIR / relative).read_text(encoding="utf-8")
        assert text == committed, f"{relative} 与生成器不一致"


def test_committed_worlds_are_valid_xml() -> None:
    for round_no in ROUNDS:
        path = WORLD_DIR / f"cuadc_recon_strike_r{round_no}.sdf"
        ET.parse(path)


# ----------------------------------------------------------------------
# 几何：目标区、天井、打击圈
# ----------------------------------------------------------------------
def _well_model(round_no: int, name: str) -> ET.Element:
    root = ET.parse(WORLD_DIR / f"cuadc_recon_strike_r{round_no}.sdf").getroot()
    world = root.find("world")
    assert world is not None
    for model in world.findall("model"):
        if model.get("name") == f"well_{name}":
            return model
    raise AssertionError(f"世界 r{round_no} 里没有 well_{name}")


def _pose_of(model: ET.Element) -> tuple[float, ...]:
    pose = model.find("pose")
    assert pose is not None and pose.text
    return tuple(float(value) for value in pose.text.split())


@pytest.mark.parametrize("round_no", ROUNDS)
def test_wells_sit_inside_the_zones_with_rule_spacing(round_no: int) -> None:
    """规则 3.3：A/B 各 4 座天井、区内随机摆放、两两间距 > 20m；目标区距起降区约 200m。"""
    wells = make_world.wells_for_round(round_no)
    for zone, (center_east, center_north) in make_world.TARGET_ZONE_CENTERS.items():
        positions = [(well.east_m, well.north_m) for well in wells if well.zone == zone]
        assert len(positions) == make_world.WELLS_PER_ZONE, zone
        half = make_world.TARGET_ZONE_SIDE_M / 2.0
        for east, north in positions:
            assert abs(east - center_east) <= half and abs(north - center_north) <= half, (
                f"{zone} 区的天井跑到区外了：E={east}, N={north}"
            )
        spacing = min(
            math.dist(first, second)
            for index, first in enumerate(positions)
            for second in positions[index + 1 :]
        )
        assert spacing > make_world.WELL_MIN_SPACING_M, (
            f"{zone} 区天井最小间距只有 {spacing:.2f}m（规则要求 > 20m）"
        )
        # 中心离区边界至少 8m：r=6m 的有效打击圈要整圈落在区内（规则 3.3.3）
        assert make_world.WELL_EDGE_MARGIN_M >= make_world.EFFECTIVE_STRIKE_RADIUS_M
        for east, north in positions:
            margin = half - max(abs(east - center_east), abs(north - center_north))
            assert margin >= make_world.EFFECTIVE_STRIKE_RADIUS_M, (
                f"{zone} 区的天井离边界只有 {margin:.2f}m，打击圈会出区"
            )
        distance = math.hypot(center_east, center_north)
        assert distance == pytest.approx(200.0, abs=1.0), "目标区中心到起降区约 200m"
    # 每个区四座天井都必须在这个世界里，且是静态模型
    for name in WELL_NAMES:
        model = _well_model(round_no, name)
        pose = _pose_of(model)
        assert model.find("static") is not None, "天井必须是静态模型"
        center_east, center_north = make_world.TARGET_ZONE_CENTERS[name[0].upper()]
        assert abs(pose[0] - center_east) <= 30.0 and abs(pose[1] - center_north) <= 30.0


@pytest.mark.parametrize("seed", range(0, 8))
def test_well_placement_random_but_valid_for_other_seeds(seed: int) -> None:
    """换种子换一批位置，但"区内 + 间距 > 20m + 打击圈不出区"这几条约束不许破。"""
    wells = make_world.wells_for_round(2, seed=seed)
    for zone in ("A", "B"):
        positions = [(well.east_m, well.north_m) for well in wells if well.zone == zone]
        assert len(positions) == make_world.WELLS_PER_ZONE
        center_east, center_north = make_world.TARGET_ZONE_CENTERS[zone]
        half = make_world.TARGET_ZONE_SIDE_M / 2.0
        for east, north in positions:
            assert abs(east - center_east) <= half and abs(north - center_north) <= half
            margin = half - max(abs(east - center_east), abs(north - center_north))
            assert margin >= make_world.EFFECTIVE_STRIKE_RADIUS_M, (
                f"{zone} 区天井离边界只有 {margin:.2f}m（有效打击圈会出区）"
            )
        spacing = min(
            math.dist(first, second)
            for index, first in enumerate(positions)
            for second in positions[index + 1 :]
        )
        assert spacing > make_world.WELL_MIN_SPACING_M


@pytest.mark.parametrize("round_no", ROUNDS)
def test_rule_diagrams_are_not_physical_markings(round_no: int) -> None:
    """规则里只做"帮助理解"的东西不许出现在场地上：打击圈（图 5）、天井箭头（图 2/4）。"""
    root = ET.parse(WORLD_DIR / f"cuadc_recon_strike_r{round_no}.sdf").getroot()
    world = root.find("world")
    assert world is not None
    names = [visual.get("name", "") for visual in world.iter("visual")]
    assert not [name for name in names if "_effective_" in name or "_precise_" in name], (
        "3.3.3 的 r=4m/6m 打击圈只是规则图 5 的示意，场地上不该有"
    )
    assert not [name for name in names if name.startswith("arrow_")], (
        "默认不画天井箭头（需要时用 make-world --well-arrows）"
    )
    markings = next(
        (model for model in world.findall("model") if model.get("name") == "cuadc_field_markings"),
        None,
    )
    assert markings is not None, "缺少场区标线模型"


def test_well_arrows_can_be_switched_on() -> None:
    """``well_arrows=True`` 时才画天井箭头（每座 1 根箭杆 + 2 个箭头）。"""
    text = make_world.build_world(2, well_arrows=True)
    assert text.count('name="arrow_shaft"') == 8
    assert text.count('name="arrow_head_') == 16
    assert "arrow_" not in make_world.build_world(2)


def test_zone_colors_are_uniform_within_a_zone() -> None:
    """规则 3.3：A 区底面蓝、B 区底面红——同一区的四座天井必须是同一种颜色。"""
    root = ET.parse(WORLD_DIR / "cuadc_recon_strike_r2.sdf").getroot()
    world = root.find("world")
    assert world is not None
    for zone, expected in (("A", (0.12, 0.25, 0.7)), ("B", (0.7, 0.12, 0.12))):
        colors: list[tuple[float, float, float]] = []
        for model in world.findall("model"):
            if not model.get("name", "").startswith(f"well_{zone.lower()}"):
                continue
            for visual in model.iter("visual"):
                name = visual.get("name")
                if name not in ("floor", "board"):
                    continue
                diffuse = visual.findtext("material/diffuse")
                assert diffuse is not None
                red, green, blue = (float(value) for value in diffuse.split()[:3])
                colors.append((red, green, blue))
        # 四座底 + 三座靶板 = 7 个面，全是这个区的颜色
        assert len(colors) == 7, colors
        assert set(colors) == {expected}, f"{zone} 区颜色不统一：{set(colors)}"


@pytest.mark.parametrize("round_no", ROUNDS)
def test_ground_markings_never_z_fight(round_no: int) -> None:
    """贴地标线不许"共面且重叠"：两层盒子高度差 < 2mm 又投影重叠 = 贴图打架。"""
    root = ET.parse(WORLD_DIR / f"cuadc_recon_strike_r{round_no}.sdf").getroot()
    world = root.find("world")
    assert world is not None
    boxes: list[tuple[str, float, float, float, float, float, float]] = []
    for visual in world.iter("visual"):
        box = visual.find("geometry/box")
        pose = visual.find("pose")
        if box is None or pose is None:
            continue
        size_text = box.findtext("size")
        assert size_text is not None and pose.text is not None
        _x, _y, z, _r, _p, _yaw = (float(value) for value in pose.text.split())
        size_x, size_y, size_z = (float(value) for value in size_text.split())
        if z > 1.0:  # 旗面之类挂在高处的盒子不参与"贴地"判断
            continue
        _east, _north = float(pose.text.split()[0]), float(pose.text.split()[1])
        boxes.append(
            (
                visual.get("name", ""),
                _east - size_x / 2.0,
                _east + size_x / 2.0,
                _north - size_y / 2.0,
                _north + size_y / 2.0,
                z - size_z / 2.0,
                z + size_z / 2.0,
            )
        )
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            if abs(first[5] - second[5]) >= 0.002:  # 高度差够大 → 不会打架
                continue
            overlap_x = min(first[2], second[2]) - max(first[1], second[1])
            overlap_y = min(first[4], second[4]) - max(first[3], second[3])
            assert overlap_x <= 0 or overlap_y <= 0, (
                f"{first[0]} 与 {second[0]} 共面（z≈{first[5]:.3f}）且投影重叠，会 z-fighting"
            )


def test_well_wall_is_a_five_millimetre_pentagon_ring() -> None:
    """竖壁：五边形环（内轮廓 = 靶板轮廓）、厚 5mm、高 400mm。"""
    vertices: list[tuple[float, float, float]] = []
    for line in (WORLD_DIR / make_world.WELL_RING_MESH).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
    assert {z for _x, _y, z in vertices} == {0.0, make_world.WELL_HEIGHT_M}
    bottom = [(x, y) for x, y, z in vertices if z == 0.0]
    assert len(bottom) == 10, "环壁底面应有 10 个顶点（外 5 + 内 5）"
    outline = [(round(x, 3), round(y, 3)) for x, y in make_world._pentagon_outline()]
    assert [(round(x, 3), round(y, 3)) for x, y in bottom[5:]] == outline, (
        "环壁内轮廓必须与靶板轮廓一致（否则靶板放不进壁内）"
    )
    # 底边是水平边：内外两点的 y 差就是壁厚
    assert abs(bottom[5][1] - bottom[0][1]) == pytest.approx(make_world.WELL_WALL_M)


# ----------------------------------------------------------------------
# 靶标摆位（第一轮图片靶 / 第二轮数字靶）
# ----------------------------------------------------------------------
def _sheet_uris(round_no: int, name: str) -> list[str]:
    model = _well_model(round_no, name)
    plate = next(
        (child for child in model.findall("model") if child.get("name") == "plate"),
        None,
    )
    if plate is None:
        return []
    return [
        uri
        for mesh in plate.iter("mesh")
        if (uri := mesh.findtext("uri") or "") and "sheet_" in uri
    ]


@pytest.mark.parametrize("round_no", ROUNDS)
def test_each_zone_has_three_targets_and_one_empty_well(round_no: int) -> None:
    for zone in ("a", "b"):
        contents = [_sheet_uris(round_no, f"{zone}{index}") for index in range(1, 5)]
        assert contents[3] == [], f"{zone}4 必须是留空的天井（规则：3 座放靶标）"
        assert all(len(sheets) == 1 for sheets in contents[:3]) or all(
            len(sheets) == 2 for sheets in contents[:3]
        )


def test_round_one_uses_official_picture_sheets() -> None:
    expected = {
        "a1": "pic_01_machine_gunner",
        "a2": "pic_07_tank",
        "a3": "pic_12_bomber",
        "b1": "pic_03_multirotor",
        "b2": "pic_08_helicopter",
        "b3": "pic_11_recon_plane",
    }
    for name, variant in expected.items():
        uris = _sheet_uris(1, name)
        assert len(uris) == 1 and f"sheet_{variant}.obj" in uris[0], name


def test_round_two_digits_and_median_well() -> None:
    """第二轮：每区 3 个两位数，中位数那座就是要打击的天井（规则任务二）。"""
    numbers = {"a": [94, 12, 56], "b": [38, 70, 61]}
    for zone, expected in numbers.items():
        values: list[int] = []
        for index in range(1, 4):
            uris = _sheet_uris(2, f"{zone}{index}")
            assert len(uris) == 2, "数字靶是两块 0.3x0.6m 板"
            digits = [int(Path(uri).stem.rsplit("_", 1)[1]) for uri in uris]
            values.append(digits[0] * 10 + digits[1])
        assert values == expected, f"{zone} 区的数字排布变了"
        median = sorted(expected)[1]
        assert values[2] == median, "中位数必须落在第 3 座天井（演练航线就是照它定的）"


@pytest.mark.parametrize("round_no", ROUNDS)
def test_sheet_sizes_and_uv_convention(round_no: int) -> None:
    """靶纸尺寸：图片靶 0.6x0.6m；数字板两块 0.6x0.3m；UV 约定写在网格注释里。"""
    for name in ("a1", "a2", "a3", "b1", "b2", "b3"):
        model = _well_model(round_no, name)
        plate = next(child for child in model.findall("model") if child.get("name") == "plate")
        for visual in plate.iter("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or "sheet_" not in (mesh.findtext("uri") or ""):
                continue
            scale = [float(value) for value in (mesh.findtext("scale") or "1 1 1").split()]
            assert scale[0] == pytest.approx(make_world.SHEET_SIDE_M)
            if round_no == 2:
                assert scale[1] == pytest.approx(make_world.DIGIT_SHEET_WIDTH_M)
            else:
                assert scale[1] == pytest.approx(make_world.SHEET_SIDE_M)
    obj_text = (WORLD_DIR / "materials/meshes/sheet_digit_5.obj").read_text(encoding="utf-8")
    assert "图像上方(v=1) → +x" in obj_text and "图像左侧(u=0) → +y" in obj_text


# ----------------------------------------------------------------------
# 资产：引用逐级存在 + 贴图内容正确（白底黑字）
# ----------------------------------------------------------------------
@pytest.mark.parametrize("round_no", ROUNDS)
def test_referenced_assets_exist(round_no: int) -> None:
    text = (WORLD_DIR / f"cuadc_recon_strike_r{round_no}.sdf").read_text(encoding="utf-8")
    problems = make_world.asset_problems(WORLD_DIR, {Path("world.sdf"): text})
    assert problems == [], problems


def _png_rows(path: Path) -> tuple[int, int, int, list[bytearray]]:
    """极简 PNG 解码（8 位 RGB/RGBA、非隔行）：返回 (宽, 高, 通道数, 每行像素)。"""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    width = height = color_type = 0
    payload = bytearray()
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", chunk[:10])
            assert depth == 8 and color_type in (2, 6), "测试只认 8 位 RGB/RGBA"
        elif kind == b"IDAT":
            payload += chunk
        elif kind == b"IEND":
            break
    assert width and height, f"{path} 里没有 IHDR"
    channels = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(payload))
    stride = width * channels
    previous = bytearray(stride)
    rows: list[bytearray] = []
    offset = 0
    for _row in range(height):
        filter_type = raw[offset]
        line = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += 1 + stride
        for index in range(stride):
            left = line[index - channels] if index >= channels else 0
            up = previous[index]
            corner = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                line[index] = (line[index] + left) & 0xFF
            elif filter_type == 2:
                line[index] = (line[index] + up) & 0xFF
            elif filter_type == 3:
                line[index] = (line[index] + (left + up) // 2) & 0xFF
            elif filter_type == 4:
                estimate = left + up - corner
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - corner))
                predictor = (left, up, corner)[distances.index(min(distances))]
                line[index] = (line[index] + predictor) & 0xFF
            elif filter_type != 0:
                raise AssertionError(f"未知 PNG 过滤器 {filter_type}")
        rows.append(line)
        previous = line
    return width, height, channels, rows


def _png_size_and_luminance(path: Path) -> tuple[int, int, float, float]:
    """极简 PNG 解码（8 位 RGB/RGBA、非隔行），返回 (宽, 高, 亮像素占比, 暗像素占比)。"""
    width, height, channels, rows = _png_rows(path)
    bright = dark = 0
    for line in rows:
        for index in range(0, len(line), channels):
            luminance = 0.299 * line[index] + 0.587 * line[index + 1] + 0.114 * line[index + 2]
            if luminance > 200:
                bright += 1
            elif luminance < 60:
                dark += 1
    total = width * height
    return width, height, bright / total, dark / total


def _png_mean_and_spread(path: Path) -> tuple[tuple[float, float, float], float]:
    """(平均 RGB, 平均绝对偏差)——判"底色是不是那一路面色、有没有少量纹理"。"""
    width, height, channels, rows = _png_rows(path)
    sums = [0, 0, 0]
    for line in rows:
        for index in range(0, len(line), channels):
            for channel in range(3):
                sums[channel] += line[index + channel]
    total = width * height
    mean = (sums[0] / total, sums[1] / total, sums[2] / total)
    deviation = 0.0
    for line in rows:
        for index in range(0, len(line), channels):
            for channel in range(3):
                deviation += abs(line[index + channel] - mean[channel])
    return mean, deviation / (total * 3)


@pytest.mark.parametrize("digit", range(10))
def test_digit_sheets_are_black_on_white(digit: int) -> None:
    """规则 3.3.2：白底、加粗黑体黑色。"""
    width, height, bright, dark = _png_size_and_luminance(
        WORLD_DIR / f"materials/textures/digit_{digit}.png"
    )
    assert (width, height) == (300, 600)
    assert bright > 0.5, f"digit_{digit} 的底色不是白的（亮像素只有 {bright:.1%}）"
    assert 0.05 < dark < 0.5, f"digit_{digit} 的字形占比异常（暗像素 {dark:.1%}）"


@pytest.mark.parametrize("index", range(1, 13))
def test_picture_sheets_are_black_silhouettes_on_white(index: int) -> None:
    name = next(
        variant for variant in make_world.PICTURE_TARGETS if variant.startswith(f"pic_{index:02d}")
    )
    width, height, bright, dark = _png_size_and_luminance(
        WORLD_DIR / f"materials/textures/{name}.png"
    )
    assert (width, height) == (600, 600)
    assert bright > 0.5, f"{name} 的底色不是白的（亮像素只有 {bright:.1%}）"
    assert dark > 0.05, f"{name} 里看不到剪影"


def test_asset_problems_reports_missing_texture(workdir: Path) -> None:
    """生成器落盘前的自检要真的能抓到"引用了不存在的贴图"。"""
    world_text = make_world.build_world(2)
    (workdir / "materials" / "meshes").mkdir(parents=True)
    for relative, text in make_world.build_meshes().items():
        (workdir / relative).write_text(text, encoding="utf-8", newline="\n")
    problems = make_world.asset_problems(workdir, {Path("world.sdf"): world_text})
    assert any("贴图不存在" in problem for problem in problems), problems


# ----------------------------------------------------------------------
# 目标区底色 + 比赛区域外的航拍干扰底图
# ----------------------------------------------------------------------
def test_zone_ground_uses_pavement_palette_and_varies_with_seed() -> None:
    """目标区底色 = 常见路面色（纯色 + 少量纹理），seed 给 A/B 各抽一种。"""
    for zone in ("A", "B"):
        assert make_world.zone_ground_variant(0, zone) in make_world.ZONE_GROUND_COLORS
    root = ET.parse(WORLD_DIR / "cuadc_recon_strike_r2.sdf").getroot()
    world = root.find("world")
    assert world is not None
    markings = next(
        model for model in world.findall("model") if model.get("name") == "cuadc_field_markings"
    )
    grounds = {
        visual.get("name", ""): visual
        for visual in markings.iter("visual")
        if visual.get("name", "").endswith("_ground")
    }
    assert set(grounds) == {"zone_a_ground", "zone_b_ground"}
    for zone in ("a", "b"):
        visual = grounds[f"zone_{zone}_ground"]
        variant = make_world.zone_ground_variant(0, zone.upper())
        assert visual.findtext("geometry/mesh/uri") == f"materials/meshes/sheet_{variant}.obj"
        scale = [float(value) for value in (visual.findtext("geometry/mesh/scale") or "").split()]
        side = make_world.TARGET_ZONE_SIDE_M + make_world.ZONE_GROUND_MARGIN_M
        assert scale[0] == pytest.approx(side) and scale[1] == pytest.approx(side)
    combinations = {
        tuple(make_world.zone_ground_variant(seed, zone) for zone in ("A", "B"))
        for seed in range(8)
    }
    assert len(combinations) >= 4, "换 seed 应当换出不同的底色组合"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_aerial_patches_stay_outside_the_competition_area(seed: int) -> None:
    """比赛区域外才铺航拍底图：整块在外、互相不压、尺寸/贴图合法、z 逐块抬高。"""
    patches = make_world.aerial_patch_layout(seed)
    assert len(patches) == make_world.AERIAL_PATCHES
    radii = [patch.size_m * math.sqrt(2.0) / 2.0 for patch in patches]
    boxes = make_world.keepout_boxes()
    for index, (patch, radius) in enumerate(zip(patches, radii, strict=True)):
        for box in boxes:
            assert not make_world._circle_hits_box(patch.east_m, patch.north_m, radius, box), (
                f"第 {index + 1} 块压到比赛区（跑道/操纵区/目标区）了"
            )
        assert patch.size_m >= make_world.AERIAL_PATCH_SIZE_MIN_M
        assert patch.size_m <= make_world.AERIAL_PATCH_SIZE_MAX_M
        assert patch.tile in make_world.AERIAL_TILES
        if index:
            assert patch.z_m - patches[index - 1].z_m >= 0.002, "相邻底图 z 太近会共面打架"
        for other, other_radius in zip(patches[index + 1 :], radii[index + 1 :], strict=True):
            distance = math.hypot(patch.east_m - other.east_m, patch.north_m - other.north_m)
            assert distance >= radius + other_radius - 1e-9, "两块航拍底图叠在一起了"


def test_world_contains_the_surroundings_and_count_is_configurable() -> None:
    text = make_world.build_world(2)
    assert '<model name="cuadc_surroundings">' in text
    assert text.count('<visual name="aerial_') == make_world.AERIAL_PATCHES
    for patch in make_world.aerial_patch_layout(0):
        assert f"sheet_{patch.tile}.obj" in text
    assert "cuadc_surroundings" not in make_world.build_world(2, aerial_patches=0), (
        "0 块就不该有这个模型"
    )
    assert make_world.build_world(2, aerial_patches=3).count('<visual name="aerial_') == 3


def test_ground_textures_are_solid_pavement_colours_with_a_hint_of_texture() -> None:
    """目标区底图：主体是那一路面色（别跑偏），但必须有少量纹理（不是死平的一块色）。"""
    for variant, colour in make_world.ZONE_GROUND_COLORS.items():
        mean, spread = _png_mean_and_spread(WORLD_DIR / f"materials/textures/{variant}.png")
        for actual, expected in zip(mean, (channel * 255 for channel in colour), strict=True):
            assert abs(actual - expected) < 24, (variant, mean)
        assert spread > 1.0, f"{variant} 太平了（没有少量纹理）"


def test_aerial_tiles_are_photo_like() -> None:
    """航拍底图必须是照片那样的（细节丰富、不是纯色块，也不是接口返的占位图）。"""
    for tile in make_world.AERIAL_TILES:
        mean, spread = _png_mean_and_spread(WORLD_DIR / f"materials/textures/{tile}.png")
        assert spread > 6.0, f"{tile} 细节太少（像纯色块）"
        assert all(10.0 < channel < 245.0 for channel in mean), f"{tile} 平均色异常：{mean}"


# ----------------------------------------------------------------------
# 底图生成器（程序化 make-backdrops / 真实影像 fetch-aerial）
# ----------------------------------------------------------------------
def _image_bytes(image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_make_backdrops_writes_both_kinds_and_respects_parts(workdir: Path) -> None:
    from tools import make_backdrops

    assert make_backdrops.main(out_dir=workdir, size=160) == 0
    textures = workdir / make_world.TEXTURE_DIR
    assert {path.name for path in textures.glob("ground_*.png")} == {
        f"{name}.png" for name in make_world.ZONE_GROUND_COLORS
    }
    assert {path.name for path in textures.glob("aerial_*.png")} == {
        f"{name}.png" for name in make_world.AERIAL_TILES
    }
    # parts="ground" 不许碰航拍底图（真实航拍是 fetch-aerial 抓来的，别被覆盖）
    aerial = textures / f"{make_world.AERIAL_TILES[0]}.png"
    aerial.write_bytes(b"sentinel")
    assert make_backdrops.main(out_dir=workdir, size=160, parts="ground") == 0
    assert aerial.read_bytes() == b"sentinel"
    # 参数不合法：不落盘、返回 1
    assert make_backdrops.main(out_dir=workdir, parts="bogus") == 1
    assert make_backdrops.main(out_dir=workdir, size=64) == 1


def test_backdrop_drawing_is_reproducible() -> None:
    """同 seed 画出来逐字节一样，换 seed 换一张。"""
    from tools import make_backdrops

    variant = next(iter(make_world.ZONE_GROUND_COLORS))
    assert _image_bytes(make_backdrops.draw_ground_texture(variant, size=128)) == _image_bytes(
        make_backdrops.draw_ground_texture(variant, size=128)
    )
    assert _image_bytes(make_backdrops.draw_ground_texture(variant, seed=5, size=128)) != (
        _image_bytes(make_backdrops.draw_ground_texture(variant, size=128))
    )
    assert _image_bytes(make_backdrops.draw_aerial_tile(1, size=128)) != _image_bytes(
        make_backdrops.draw_aerial_tile(2, size=128)
    )


def test_preview_world_renders_a_readable_png(workdir: Path) -> None:
    """俯视预览图：能渲染、尺寸合理、同输入逐字节一致（图上的标注只在 PNG 里）。"""
    from tools import preview_world

    out = workdir / "preview.png"
    assert preview_world.main(out_path=out) == 0
    assert out.is_file() and out.stat().st_size > 10_000
    width, height, _channels, _rows = _png_rows(out)
    assert width >= 800 and height >= 400, (width, height)
    # 同输入可复现（预览是纯函数：世界文件 + 贴图）
    again = workdir / "again.png"
    assert preview_world.main(out_path=again) == 0
    assert again.read_bytes() == out.read_bytes()
    # 世界文件不在 / 轮次非法 → 返回 1
    assert preview_world.main(world_dir=workdir / "missing", out_path=out) == 1
    assert preview_world.main(round_no=7, out_path=out) == 1


def test_fetch_aerial_builds_a_sane_request_and_validates_input(workdir: Path) -> None:
    """抓取脚本：URL 里 bbox 的跨度要和 --tile-width-m 对得上；离线/非法参数不落盘。"""
    from tools import fetch_aerial

    url = fetch_aerial.export_image_url(-93.5, 42.0, width_m=160.0, size=512)
    assert url.startswith(fetch_aerial.SERVICE_URL)
    query = dict(pair.split("=", 1) for pair in url.split("?", 1)[1].split("&"))
    west, south, east, north = (
        float(value) for value in urllib.parse.unquote(query["bbox"]).split(",")
    )
    assert (north - south) * 111320.0 == pytest.approx(160.0, rel=0.01)
    assert (east - west) * 111320.0 * math.cos(math.radians(42.0)) == pytest.approx(160.0, rel=0.02)
    assert query["size"] == "512%2C512"
    # 参数不合法 → 返回 1（还没到联网那一步）
    assert fetch_aerial.main(out_dir=workdir, size=64) == 1
    assert fetch_aerial.main(out_dir=workdir, tile_width_m=0.0) == 1


# ----------------------------------------------------------------------
# 仿真模块（sim/）：机型 + airframe + 装机/启动脚本
# ----------------------------------------------------------------------
def test_sim_module_keeps_the_down_camera_vehicle_in_repo() -> None:
    """带下视相机的机型与 airframe 都必须在仓库里（PX4 树只放软链，不存副本）。"""
    model = SIM_DIR / "vehicles" / "rc_cessna_down_cam"
    model_sdf = (model / "model.sdf").read_text(encoding="utf-8")
    assert "model://rc_cessna<" in model_sdf, "机型要 merge PX4 模型库里的基础 rc_cessna"
    assert "model://rc_cessna_down_cam/camera_720p<" in model_sdf
    assert "1.5707" in model_sdf, "相机要 90° 俯仰（下视）"
    camera_sdf = (model / "camera_720p" / "model.sdf").read_text(encoding="utf-8")
    assert "<width>1280</width>" in camera_sdf and "<height>720</height>" in camera_sdf
    assert '<sensor name="imager" type="camera">' in camera_sdf
    assert "<far>3000</far>" in camera_sdf
    for name in ("model.config", "camera_720p/model.config"):
        assert (model / name).is_file(), f"缺 {name}"

    airframe = (SIM_DIR / "airframes" / "4007_gz_rc_cessna_down_cam").read_text(encoding="utf-8")
    assert "PX4_SIM_MODEL=${PX4_SIM_MODEL:=rc_cessna_down_cam}" in airframe
    assert "param set-default SIM_GZ_EN 1" in airframe
    assert "param set-default MIS_TAKEOFF_ALT" in airframe


def test_sim_scripts_install_from_the_repo() -> None:
    """装机/启动脚本必须把 **仓库里** 的机型/世界/airframe 软链进 PX4。"""
    install = (SIM_DIR / "install_px4.sh").read_text(encoding="utf-8")
    run = (SIM_DIR / "run_sitl.sh").read_text(encoding="utf-8")
    assert "vehicles/rc_cessna_down_cam" in install
    assert "airframes/" in install and "4007_gz_rc_cessna_down_cam" in install
    assert "CMakeLists.txt" in install, "要登记 airframe（PX4 不 glob）"
    assert "GZ_SIM_RESOURCE_PATH" in run and "install_px4.sh" in run
    assert "PX4_GZ_WORLD" in run
    assert (SIM_DIR / "patches" / "gst_camera_nvenc_probe.patch").is_file()
    assert (SIM_DIR / "README.md").is_file()


# ----------------------------------------------------------------------
# 生成配置：out-dir / seed / wind
# ----------------------------------------------------------------------
def test_main_writes_a_complete_scenario(workdir: Path) -> None:
    assert make_world.main(out_dir=workdir, rounds="2") == 0
    assert (workdir / "cuadc_recon_strike_r2.sdf").exists()
    assert not (workdir / "cuadc_recon_strike_r1.sdf").exists(), "rounds=2 只写第二轮的"
    textures = list((workdir / "materials" / "textures").glob("*.png"))
    assert len(textures) == len(make_world.texture_variants()), "贴图要一起复制过去"
    assert make_world.main(out_dir=workdir / "bad", rounds="3") == 1, "非法轮次要返回 1"


def test_seed_changes_well_positions_and_yaw() -> None:
    """天井的位置与朝向都是随机的（规则 3.3）：换种子换一批，同种子可复现。"""

    def poses(text: str) -> dict[str, tuple[float, ...]]:
        root = ET.fromstring(text)
        result: dict[str, tuple[float, ...]] = {}
        world = root.find("world")
        assert world is not None
        for model in world.findall("model"):
            if (model.get("name") or "").startswith("well_"):
                result[model.get("name", "")] = _pose_of(model)
        return result

    base = poses(make_world.build_world(2, seed=0))
    assert base == poses(make_world.build_world(2, seed=0)), "同一种子必须完全一致"
    other = poses(make_world.build_world(2, seed=5))
    assert base.keys() == other.keys()
    assert [base[name][:2] for name in base] != [other[name][:2] for name in other], (
        "换种子要换来一批位置（规则里位置是随机的）"
    )
    assert [base[name][5] for name in base] != [other[name][5] for name in other], (
        "换种子要换来一批朝向（规则里朝向是随机的）"
    )


def test_sitl_recon_route_aims_at_the_median_well() -> None:
    """演练航线不许与世界脱节：盘旋点 + 目标偏移 = 入库世界里 A 区的中位数天井。"""
    from airdrop.georef.geo import wgs84_to_ned
    from examples import sitl_mission

    reference = (make_world.WORLD_LON_DEG, make_world.WORLD_LAT_DEG, 0.0)

    def enu(waypoint) -> tuple[float, float]:
        north, east, _down = wgs84_to_ned(waypoint.lon, waypoint.lat, 0.0, reference)
        return east, north

    hover_east, hover_north = enu(sitl_mission.RECON_ROUTE[-1])
    offset_north, offset_east, _offset_down = sitl_mission.TARGET_OFFSET_NED
    target_east = hover_east + offset_east
    target_north = hover_north + offset_north
    median = next(well for well in make_world.wells_for_round(2) if well.name == "a3")
    assert math.hypot(target_east - median.east_m, target_north - median.north_m) < 0.5, (
        "演练的合成目标没落在 A 区中位数天井上（世界换布局后要同步 examples/sitl_mission.py）"
    )
    backup_east, backup_north = enu(sitl_mission.BACKUP_POINT)
    center_east, center_north = make_world.TARGET_ZONE_CENTERS["A"]
    assert math.hypot(backup_east - center_east, backup_north - center_north) < 0.5, (
        "备用点应当就是 A 区中心"
    )


def test_wind_option_writes_the_wind_element() -> None:
    text = make_world.build_world(1, wind_enu_m_s=(5.0, 2.0, 0.0))
    assert "<linear_velocity>5 2 0</linear_velocity>" in text
    assert "<linear_velocity>0 0 0</linear_velocity>" in make_world.build_world(1)
    with pytest.raises(ValueError):
        make_world.world_texts(rounds="7")


# ----------------------------------------------------------------------
# 光照：太阳位置可变（用来模拟不同的阴影环境）
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("azimuth", "elevation", "expected"),
    [
        (0.0, 90.0, (0.0, 0.0, -1.0)),  # 正上方：光垂直向下
        (90.0, 45.0, (-math.sqrt(0.5), 0.0, -math.sqrt(0.5))),  # 东侧 45°：光往西打
        (270.0, 45.0, (math.sqrt(0.5), 0.0, -math.sqrt(0.5))),  # 西侧 45°：光往东打
        (180.0, 30.0, (0.0, math.cos(math.radians(30.0)), -0.5)),  # 南侧 30°：光往北打
    ],
)
def test_sun_direction_matches_the_handy_cases(
    azimuth: float, elevation: float, expected: tuple[float, float, float]
) -> None:
    """方位角从北（+y）起顺时针、仰角从地平线起算；得到的是**光传播方向**。"""
    direction = make_world.sun_direction(azimuth, elevation)
    for actual, want in zip(direction, expected, strict=True):
        assert actual == pytest.approx(want, abs=1e-9), (azimuth, elevation)


def test_sun_direction_is_a_unit_vector_from_the_sun() -> None:
    for azimuth in (0.0, 45.0, 120.0, 218.0, 300.0):
        for elevation in (5.0, 30.0, 55.0, 89.0):
            east, north, up = make_world.sun_direction(azimuth, elevation)
            length = math.sqrt(east * east + north * north + up * up)
            assert length == pytest.approx(1.0, abs=1e-9)
            assert up < 0.0, "太阳在头顶上，光必须朝下"
            # 水平分量指着太阳的对面：从传播方向反推出来的方位角 = 输入方位角
            recovered = math.degrees(math.atan2(-east, -north)) % 360.0
            assert recovered == pytest.approx(azimuth % 360.0, abs=1e-6)


@pytest.mark.parametrize("elevation", [0.0, -10.0, 120.0])
def test_sun_below_the_horizon_is_rejected(elevation: float) -> None:
    with pytest.raises(ValueError, match="太阳仰角"):
        make_world.sun_direction(218.0, elevation)


def test_random_sun_is_reproducible_and_bounded() -> None:
    """``--sun-random``：由 (seed, 轮次) 复现；方位全向、仰角在设定区间内。"""
    drawn = {
        (seed, round_no): make_world.sun_azel_for(seed, round_no)
        for seed in range(8)
        for round_no in (1, 2)
    }
    for (seed, round_no), (azimuth, elevation) in drawn.items():
        assert make_world.sun_azel_for(seed, round_no) == (azimuth, elevation), "必须可复现"
        assert 0.0 <= azimuth < 360.0, (seed, round_no, azimuth)
        assert make_world.SUN_ELEVATION_MIN_DEG <= elevation <= make_world.SUN_ELEVATION_MAX_DEG
    # 换 seed 要真换出不同的太阳（不是形同虚设的"随机"）
    azimuths = {azimuth for azimuth, _elevation in drawn.values()}
    assert len(azimuths) >= len(drawn) - 1, "不同 seed/轮次基本应当给出不同的太阳"


def test_sun_random_switch_writes_the_drawn_direction() -> None:
    """打开随机太阳后，世界里的 <direction> 就是抽出来的那个太阳（--sun 的固定值让位）。"""
    azimuth, elevation = make_world.sun_azel_for(0, 2)
    expected = " ".join(
        make_world._fmt(component) for component in make_world.sun_direction(azimuth, elevation)
    )
    text = make_world.build_world(2, sun_random=True)
    assert f"<direction>{expected}</direction>" in text
    # --sun-random 与 --sun 同时给：随机优先
    assert make_world.build_world(2, sun_azel_deg=(90.0, 12.0), sun_random=True) == text
    # 关着的时候仍然是固定太阳
    assert make_world.build_world(2) != text


@pytest.mark.parametrize("round_no", ROUNDS)
def test_world_sun_direction_is_configurable(round_no: int) -> None:
    """换 ``--sun`` 就换阴影环境：世界里的 <direction> 跟着走。"""

    def direction_text(azel: tuple[float, float]) -> str:
        return " ".join(make_world._fmt(component) for component in make_world.sun_direction(*azel))

    text = make_world.build_world(round_no)
    assert f"<direction>{direction_text(make_world.SUN_AZEL_DEG)}</direction>" in text, (
        "入库世界的太阳方向与 SUN_AZEL_DEG 不一致（改完生成器要重跑 make-world）"
    )
    other = make_world.build_world(round_no, sun_azel_deg=(90.0, 12.0))
    assert f"<direction>{direction_text((90.0, 12.0))}</direction>" in other
    assert direction_text((90.0, 12.0)) != direction_text(make_world.SUN_AZEL_DEG)


def test_main_rejects_a_sun_below_the_horizon(workdir: Path, capsys: pytest.CaptureFixture) -> None:
    assert make_world.main(out_dir=workdir, sun_azel_deg=(90.0, 0.0)) == 1
    assert "太阳仰角" in capsys.readouterr().out
    assert not (workdir / "cuadc_recon_strike_r1.sdf").exists(), "参数不合法就不该落盘"
