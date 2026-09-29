"""生成 CUADC「固定翼无人机侦察与打击」赛区的 Gazebo 世界（SDF）。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run make-world
    ./.venv/Scripts/python.exe -m airdrop.run make-world --rounds 2 --seed 3
    ./.venv/Scripts/python.exe -m airdrop.run make-world --out-dir sim/worlds/cuadc

本文件是纯库模块：顶部常量 = 默认值，暴露 ``build_config(**覆盖)`` 与 ``main(**kwargs)``；
命令行只在 airdrop/run.py。只依赖标准库（不 import PIL/cv2 等重家伙）。

生成什么
--------
按 2026 竞赛规则（固定翼无人机侦察与打击，出处见 ``docs/simulation_world.md``）生成两个世界与全部文本资产：

* ``cuadc_recon_strike_r1.sdf``：第一轮，天井里放**图片靶标**（官方 12 张里的 6 张）；
* ``cuadc_recon_strike_r2.sdf``：第二轮，天井里放**数字靶标**（两块数字板组成两位数）；
* ``materials/meshes/``：五边形靶板网格 + 每张贴图对应的靶纸平面与 MTL。

贴图 PNG 本身是入库的二进制资产（来源与生成方式见 ``docs/simulation_world.md``）；
生成器只写文本，并用 ``_asset_problems()`` 把「引用得到、文件不在」这类错误挡在落盘之前。

设计口径（规则文本 → 世界）
----------------------------
* 起降区：原点处 200m x 30m 跑道（规则 3.2 的"约 50 x 50m 跑道区域"是整块起降地块；
  固定翼 SITL 需要更长的滑跑距离，50 x 50m 的起降区标记画在原点），沿 x 轴、起飞方向 +x；
* 目标区：A / B 两个 60 x 60m 地块（规则 3.3），按**规则图 1**分列起飞线两端、
  各距起降区约 200m：A 区中心 (-200, 0)、B 区中心 (+200, 0)，四角插旗划定范围；
  A 区天井底面蓝、B 区底面红（规则 3.3）；
* 每区 4 座天井、高 400mm、两两间距 > 20m：**位置与朝向都由 ``seed`` 随机决定**
  （规则里两者都是随机的），默认 seed 0 = 入库布局；天井中心离区边界至少 8m，
  保证规则 3.3.3 的 r=6m 有效打击圈整圈落在目标区内；
* 每座天井 = 五边形环壁（厚 5mm、高 400mm，内轮廓 = 靶板轮廓）+ 同形的蓝/红底面。
  靶板与底面同形、颜色随所属区（A 蓝 / B 红），正好放进壁内、四边贴壁；
  每区第 4 座天井留空（规则 3.3.1/3.3.2"3 个天井放靶标"）；
* 不画打击圈：规则 3.3.3 的 r=4m / r=6m 圆只是**规则图 5 里的示意**，场地上没有；
* 不画天井箭头：规则文本 3.3.1/3.3.2 提到"天井箭头方向随机"，但规则图 2/图 4 里
  没有画；默认不画（见 ``--well-arrows`` 开关），"靶标方向随机"由天井朝向体现；
* 目标区底色：**常见路面色 + 少量对应地面纹理**（沥青/水泥/土面/砖面/砂石），
  由 ``seed`` 给 A/B 两区各抽一种（``zone_ground_variant``）；纹理由
  ``make-backdrops`` 生成（程序化），也可以用 ``fetch-aerial`` 换成真实航拍贴图
  （见下一条的兄弟项：那是"铺在比赛区域外"的干扰底图，两者不是一回事）；
* 比赛区域外：随机铺若干块**航拍干扰底图**（``--aerial-patches``，默认 12 块，
  取自 ``AERIAL_TILES``）——农田/道路/房屋/路面标线，用来锻炼识别抗扰；
  硬约束只有"不许压到跑道/起降区标记/操纵区/目标区底色块"和"块与块不重叠"，
  所以贴图可以紧贴着场地铺；
* 光照：世界带一盏平行光（太阳），位置 = ``(方位角, 仰角)``（默认 218°, 55°），
  阴影开着；换 ``--sun`` 可以模拟不同的阴影环境（侧光长阴影 / 近顶光短阴影），
  或者用 ``--sun-random`` 让太阳由 ``seed`` 随机（方位全向、仰角 15°~80°）——
  批量生成不同光照的场地时换 ``--seed`` 就够了。

坐标口径
--------
世界用 ENU：``+x`` = 东、``+y`` = 北、``+z`` = 天；原点 = 起降区中心（飞机出生点）。
``spherical_coordinates`` 沿用 PX4 默认世界的原点（苏黎世），这样跑 SITL 时 GPS 原点
与 ``examples/sitl_mission.py`` / ``routes/land.plan`` 的经纬度不用另配。
"""

from __future__ import annotations

import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

# ----------------------------------------------------------------------
# 默认值（命令行选项的默认值直接取自这里）
# ----------------------------------------------------------------------
OUT_DIR = Path("sim/worlds/cuadc")
SEED = 0
ROUNDS = "both"

WORLD_PREFIX = "cuadc_recon_strike"

#: 世界原点（GPS 参考点）：与 PX4 默认世界一致，SITL 开箱可用
WORLD_LAT_DEG = 47.397971057728974
WORLD_LON_DEG = 8.546163739800146

#: 跑道（起降区地块）：沿 x 轴、以原点为中心
RUNWAY_LENGTH_M = 200.0
RUNWAY_WIDTH_M = 30.0
#: 规则里的"起降区"标记（约 50 x 50m），画在原点
LANDING_ZONE_M = 50.0
#: 操纵区（规则 3.1 的比赛区之一，紧贴起飞线外侧）：跑道南侧、起降区标记之外
CONTROL_ZONE_LENGTH_M = 20.0
CONTROL_ZONE_WIDTH_M = 16.0
CONTROL_ZONE_EAST_M = 0.0
CONTROL_ZONE_NORTH_M = -34.0

#: 贴地标线的标高（米）：层与层之间留 20mm。⚠ **绝不让两层共面重叠**——
#: 共面又重叠会渲染成贴图打架（z-fighting）。
RUNWAY_Z_M = 0.01
LANDING_ZONE_Z_M = 0.03
CONTROL_ZONE_Z_M = 0.05
RUNWAY_MARK_Z_M = 0.07
ZONE_LINE_Z_M = 0.03

#: 目标区：A / B 两块 60 x 60m，分列起飞线两端、各距起降区约 200m（规则 3.3、图 1）。
#: A 区（蓝）在 +x（跑道起飞方向那头）、B 区（红）在 -x：规则图里 A/B 就在起飞线两端，
#: 哪端配哪区是场地布置的随机项之一，这里取"起飞方向先到 A 区"。
TARGET_ZONE_SIDE_M = 60.0
TARGET_ZONE_CENTERS: dict[str, tuple[float, float]] = {  # 区 → (east_m, north_m)
    "A": (200.0, 0.0),
    "B": (-200.0, 0.0),
}
#: 每个目标区的天井数量（规则 3.3）
WELLS_PER_ZONE = 4
#: 留空的天井下标（每区 4 座里第 4 座空着）
EMPTY_WELL_INDEX = 3
#: 天井在区内随机摆放的约束：两两间距 > 20m（规则）留 1m 余量；
#: 中心离区边界至少 8m —— 规则 3.3.3 的 r=6m 有效打击圈要整圈落在目标区内（再留 2m 余量）
WELL_MIN_SPACING_M = 20.0
WELL_SPACING_MARGIN_M = 1.0
#: 规则 3.3.3 的有效打击区半径（只用于推算天井离边界要多远，不画到地上）
EFFECTIVE_STRIKE_RADIUS_M = 6.0
WELL_EDGE_MARGIN_M = 8.0
#: 拒绝采样次数（取不到就退回抖动的 2x2 网格，见 _sample_well_positions）
WELL_PLACEMENT_ATTEMPTS = 500

#: 天井（"天井" = 五边形竖壁 + 整块红/蓝底面，壁厚 5mm、高 400mm）
#: ⚠ 竖壁形状与靶板同形（1m 方形 + 60° 顶角的"房子"五边形）：
#: 底面（蓝/红）就是五边形本身，竖壁紧贴它外侧，靶板正好放进壁内、四边贴壁。
WELL_WALL_M = 0.005
WELL_HEIGHT_M = 0.4
WELL_FLOOR_M = 0.06
#: 五边形靶板：1m 方形 + 等边三角顶角（高 sqrt(3)/2），厚 20mm
PLATE_SIDE_M = 1.0
PLATE_THICKNESS_M = 0.02
#: 天井底面 = 靶板形状的加强板（厚 60mm = 靶板网格 z 方向放大 3 倍）
WELL_FLOOR_SCALE_Z = WELL_FLOOR_M / PLATE_THICKNESS_M
#: 靶纸：图片靶 0.6 x 0.6m；数字靶两块 0.3 x 0.6m（字高 0.4m）
SHEET_SIDE_M = 0.6
DIGIT_SHEET_WIDTH_M = 0.3
#: 旗杆（四角插旗）
FLAG_POLE_HEIGHT_M = 3.0
FLAG_POLE_RADIUS_M = 0.03

#: 目标区"底色"：**常见路面色 + 少量对应地面纹理**（纹理由 make-backdrops 生成）。
#: 由 seed 给 A/B 两区各抽一种，模拟不同场地。名字 → 基色（r, g, b；0~1）——
#: 纹理就是在这个基色上加少量污渍/颗粒/裂纹，远看仍是一块纯色。
ZONE_GROUND_COLORS: dict[str, tuple[float, float, float]] = {
    "ground_asphalt": (0.22, 0.22, 0.24),  # 沥青（深灰）
    "ground_concrete_old": (0.45, 0.44, 0.42),  # 旧水泥
    "ground_concrete": (0.62, 0.60, 0.56),  # 混凝土
    "ground_soil": (0.55, 0.42, 0.26),  # 土面
    "ground_brick": (0.52, 0.33, 0.24),  # 砖红
    "ground_gravel": (0.68, 0.62, 0.48),  # 砂石
}
#: 底色块比 60x60m 略大一点：白色边界线要压在底色上
ZONE_GROUND_MARGIN_M = 6.0
ZONE_GROUND_Z_M = 0.006

#: 比赛区域外的"随机航拍底图"（干扰背景，锻炼识别抗扰）：贴图名由 make-aerial 生成
AERIAL_TILES: tuple[str, ...] = (
    "aerial_1",
    "aerial_2",
    "aerial_3",
    "aerial_4",
    "aerial_5",
    "aerial_6",
)
#: 默认铺几块、每块多大、铺在哪儿（比赛区域外的环带）
AERIAL_PATCHES = 12
AERIAL_PATCH_SIZE_MIN_M = 60.0
AERIAL_PATCH_SIZE_MAX_M = 130.0
AERIAL_PATCH_Z_M = 0.01
#: 每块依次抬高一点：万一两块叠在一起也不会共面（z-fighting）
AERIAL_PATCH_Z_STEP_M = 0.003
AERIAL_PLACEMENT_ATTEMPTS = 800
#: 航拍底图的撒放范围（世界地面 1400x1400m 之内）；可以贴着比赛区铺
SURROUNDINGS_BAND_EAST_M = 400.0
SURROUNDINGS_BAND_NORTH_M = 260.0
#: 兜底环（撒不到位置时按这个半径均匀摆一圈；要大于比赛区所有占地的外接圆）
SURROUNDINGS_FALLBACK_RADIUS_M = 260.0

#: 世界里的恒定风（东, 北, 天；米/秒）。默认静风：弹道试验/演练需要可复现的初值；
#: 要考风的影响就把这里（或 --wind）设成比如 (5, 2, 0) 再重跑生成器（改的是世界文件里
#: 的 <wind> 元素；这个版本的 Gazebo server 没有对应的运行时主题）。
WIND_ENU_M_S: tuple[float, float, float] = (0.0, 0.0, 0.0)

#: 太阳位置（方位角, 仰角；度）——光照方向就是平行光的传播方向，阴影随它变。
#: 方位角从北（+y）起顺时针量：0=北、90=东、180=南、270=西；仰角从地平线起算（0~90）。
#: 默认 (218, 55) 是接近正午偏西南的高照度（短阴影）；想模拟不同的阴影环境就改这里
#: 或命令行 --sun，例如 --sun 90,12（东侧低角度侧光、长阴影）、--sun 0,80（高照度近顶光）。
SUN_AZEL_DEG: tuple[float, float] = (218.0, 55.0)

#: 太阳随机模式：由 seed + 轮次决定太阳位置（``--sun-random``），用来批量生成
#: "不同阴影环境"的场地。方位角全向 0~360；仰角限制在下面这个区间——
#: 太低（<15°）阴影长得夸张且景物发暗，太高（>80°）几乎没有影子。
SUN_RANDOM = False
SUN_ELEVATION_MIN_DEG = 15.0
SUN_ELEVATION_MAX_DEG = 80.0

#: 是否在天井里画"天井箭头"：规则文本 3.3.1/3.3.2 提到"天井箭头方向随机"、
#: 数字靶方向要与箭头一致，但规则图 2 / 图 4 里并没有画箭头，所以**默认不画**
#: （``--well-arrows`` 可以打开）。"靶标方向随机"这件事由天井朝向本身体现。
WELL_ARROWS = False

#: 图片靶标（官方 12 张；序号 = 目标价值 1..12，名字用英文便于做文件名）
PICTURE_TARGETS: tuple[str, ...] = (
    "pic_01_machine_gunner",
    "pic_02_rocket_soldier",
    "pic_03_multirotor",
    "pic_04_fixed_wing",
    "pic_05_truck",
    "pic_06_aa_gun",
    "pic_07_tank",
    "pic_08_helicopter",
    "pic_09_fighter",
    "pic_10_transport",
    "pic_11_recon_plane",
    "pic_12_bomber",
)
#: 第一轮每个区放哪三张（按该区第 1~3 座天井的顺序）
ROUND1_PICTURES: dict[str, tuple[str, ...]] = {
    "A": ("pic_01_machine_gunner", "pic_07_tank", "pic_12_bomber"),
    "B": ("pic_03_multirotor", "pic_08_helicopter", "pic_11_recon_plane"),
}
#: 第二轮每个区的两位数字（按该区第 1~3 座天井的顺序）
#: ⚠ 第 3 座必须是三个数的**中位数**——演练航线（examples/sitl_mission.py）就是照
#: "中位数天井"定的目标点，tests/test_world.py 钉住这条不变量。
ROUND2_DIGITS: dict[str, tuple[tuple[int, int], ...]] = {
    "A": ((9, 4), (1, 2), (5, 6)),
    "B": ((3, 8), (7, 0), (6, 1)),
}

#: 资产目录（相对输出目录）
TEXTURE_DIR = Path("materials/textures")
MESH_DIR = Path("materials/meshes")
PLATE_MESH = MESH_DIR / "pentagon_plate.obj"
#: 天井五边形竖壁（环状网格；外轮廓 = 底面外扩 5mm，高 400mm）
WELL_RING_MESH = MESH_DIR / "pentagon_well_wall.obj"


@dataclass(frozen=True, slots=True)
class MakeWorldConfig:
    """生成器配置（字段 = 命令行能覆盖的东西）。"""

    out_dir: Path = OUT_DIR
    seed: int = SEED
    rounds: str = ROUNDS
    wind_enu_m_s: tuple[float, float, float] = WIND_ENU_M_S
    well_arrows: bool = WELL_ARROWS
    sun_azel_deg: tuple[float, float] = SUN_AZEL_DEG
    sun_random: bool = SUN_RANDOM
    aerial_patches: int = AERIAL_PATCHES


def build_config(**overrides) -> MakeWorldConfig:
    """按关键字覆盖派生一份配置（``dataclasses.replace``）。"""
    if "out_dir" in overrides:
        overrides["out_dir"] = Path(overrides["out_dir"])
    return replace(MakeWorldConfig(), **overrides)


def zone_ground_variant(seed: int, zone: str) -> str:
    """目标区底色：从常见路面颜色里随机抽一种（同 seed/区永远一样）。

    返回 ``ZONE_GROUND_COLORS`` 里的名字；对应的地面纹理由 ``make-backdrops`` 生成
    （纯色底 + 少量同色系纹理），世界按这个名字引用 ``sheet_<名字>.obj``。
    """
    rng = Random(f"ground:{seed}:{zone}")
    names = tuple(ZONE_GROUND_COLORS)
    return names[rng.randrange(len(names))]


@dataclass(frozen=True, slots=True)
class AerialPatch:
    """比赛区域外的一块航拍底图（贴地贴图块；纯干扰背景，与竞赛规则无关）。"""

    tile: str
    east_m: float
    north_m: float
    size_m: float
    yaw_deg: float
    z_m: float


def keepout_boxes() -> tuple[tuple[float, float, float, float], ...]:
    """航拍底图**不许压到**的地方（比赛区的实际占地，不含外扩余量）。

    跑道（含起降区标记）、操纵区、两个目标区的底色块——贴图块可以紧贴着它们铺，
    只要不覆盖。每项 ``(east_min, east_max, north_min, north_max)``。
    """
    half_side = (TARGET_ZONE_SIDE_M + ZONE_GROUND_MARGIN_M) / 2.0
    boxes = [
        (
            -RUNWAY_LENGTH_M / 2.0,
            RUNWAY_LENGTH_M / 2.0,
            -RUNWAY_WIDTH_M / 2.0,
            RUNWAY_WIDTH_M / 2.0,
        ),
        (
            CONTROL_ZONE_EAST_M - CONTROL_ZONE_LENGTH_M / 2.0,
            CONTROL_ZONE_EAST_M + CONTROL_ZONE_LENGTH_M / 2.0,
            CONTROL_ZONE_NORTH_M - CONTROL_ZONE_WIDTH_M / 2.0,
            CONTROL_ZONE_NORTH_M + CONTROL_ZONE_WIDTH_M / 2.0,
        ),
    ]
    for center_east, center_north in TARGET_ZONE_CENTERS.values():
        boxes.append(
            (
                center_east - half_side,
                center_east + half_side,
                center_north - half_side,
                center_north + half_side,
            )
        )
    return tuple(boxes)


def _circle_hits_box(
    east: float, north: float, radius: float, box: tuple[float, float, float, float]
) -> bool:
    """圆（贴图块的外接圆）与矩形有没有交叠。"""
    east_min, east_max, north_min, north_max = box
    closest_east = min(max(east, east_min), east_max)
    closest_north = min(max(north, north_min), north_max)
    return math.hypot(east - closest_east, north - closest_north) < radius


def _sample_surroundings_center(
    rng: Random,
    radius: float,
    index: int,
    count: int,
    placed: tuple[tuple[float, float, float], ...] = (),
) -> tuple[float, float]:
    """在比赛区之外撒一个中心点（``radius`` = 外接圆，判"不压跑道/目标区/别人"）。

    ``placed`` 是已经摆好的 ``(east, north, 外接圆半径)``——两块底图互相不压，
    免得一堆航拍块叠成一团；能不能贴到跑道边上见 :func:`keepout_boxes`。
    """
    boxes = keepout_boxes()
    for _ in range(AERIAL_PLACEMENT_ATTEMPTS):
        east = rng.uniform(-SURROUNDINGS_BAND_EAST_M, SURROUNDINGS_BAND_EAST_M)
        north = rng.uniform(-SURROUNDINGS_BAND_NORTH_M, SURROUNDINGS_BAND_NORTH_M)
        if any(_circle_hits_box(east, north, radius, box) for box in boxes):
            continue
        if any(
            math.hypot(east - other_east, north - other_north) < radius + other_radius
            for other_east, other_north, other_radius in placed
        ):
            continue
        return east, north
    # 兜底：沿环带一圈圈往外扫，直到找到不压比赛区、也不压别人的位置
    # （正常路径几乎不会走到这里；真找不到就返回最外圈的点，每块 z 不同，叠了也不打架）
    fallback = (0.0, 0.0)
    for extra in (0.0, 60.0, 140.0, 260.0, 420.0):
        ring = SURROUNDINGS_FALLBACK_RADIUS_M + radius + 40.0 + extra
        for step in range(24):
            angle = math.tau * (index + step) / 24.0
            fallback = (ring * math.cos(angle), ring * math.sin(angle))
            if any(_circle_hits_box(*fallback, radius, box) for box in boxes):
                continue
            if any(
                math.hypot(fallback[0] - other_east, fallback[1] - other_north)
                < radius + other_radius
                for other_east, other_north, other_radius in placed
            ):
                continue
            return fallback
    return fallback


def aerial_patch_layout(seed: int, count: int = AERIAL_PATCHES) -> tuple[AerialPatch, ...]:
    """比赛区域外随机铺 ``count`` 块航拍底图（换 seed 换一批；``count=0`` 就是不铺）。"""
    if count <= 0:
        return ()
    rng = Random(f"aerial:{seed}")
    patches: list[AerialPatch] = []
    for index in range(count):
        size = rng.uniform(AERIAL_PATCH_SIZE_MIN_M, AERIAL_PATCH_SIZE_MAX_M)
        tile = AERIAL_TILES[rng.randrange(len(AERIAL_TILES))]
        yaw = rng.uniform(0.0, 360.0)
        radius = size * math.sqrt(2.0) / 2.0
        placed = tuple(
            (patch.east_m, patch.north_m, patch.size_m * math.sqrt(2.0) / 2.0) for patch in patches
        )
        east, north = _sample_surroundings_center(rng, radius, index, count, placed)
        patches.append(
            AerialPatch(
                tile=tile,
                east_m=east,
                north_m=north,
                size_m=size,
                yaw_deg=yaw,
                z_m=AERIAL_PATCH_Z_M + AERIAL_PATCH_Z_STEP_M * index,
            )
        )
    return tuple(patches)


def _surroundings_model(patches: tuple[AerialPatch, ...]) -> str:
    """比赛区域外的随机航拍底图（给下视画面造农田/道路/房屋/路面标线这类干扰背景）。"""
    parts = [
        '    <model name="cuadc_surroundings">',
        "      <static>true</static>",
        '      <link name="link">',
    ]
    for index, patch in enumerate(patches, start=1):
        parts.append(
            f'        <visual name="aerial_{index}">\n'
            f"          {_pose(patch.east_m, patch.north_m, patch.z_m, yaw_deg=patch.yaw_deg)}\n"
            "          <geometry>\n"
            "            <mesh>\n"
            f"              <uri>{(MESH_DIR / f'sheet_{patch.tile}.obj').as_posix()}</uri>\n"
            f"              <scale>{_fmt(patch.size_m)} {_fmt(patch.size_m)} 1</scale>\n"
            "            </mesh>\n"
            "          </geometry>\n"
            "        </visual>"
        )
    parts.extend(["      </link>", "    </model>"])
    return "\n".join(parts)


# ----------------------------------------------------------------------
# 场景描述（纯数据）
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Well:
    """一座天井：位置、朝向与内容。"""

    name: str  # 例如 "a1"
    zone: str  # "A" / "B"
    east_m: float
    north_m: float
    yaw_deg: float
    #: 图片靶：``("pic", "pic_07_tank")``；数字靶：``("digits", (5, 6))``；空：None
    content: tuple[str, object] | None


def _sample_well_positions(rng: Random) -> tuple[tuple[float, float], ...]:
    """在一个目标区里随机摆 ``WELLS_PER_ZONE`` 座天井（区内、两两间距达标）。

    规则 3.3 只规定"区内 4 座、间距大于 20m"，位置本身是随机的：这里按种子做
    拒绝采样；采不到就退回"抖动的 2x2 网格"（一定满足约束）。
    返回相对区中心的（东, 北）偏移。
    """
    half = TARGET_ZONE_SIDE_M / 2.0 - WELL_EDGE_MARGIN_M
    minimum = WELL_MIN_SPACING_M + WELL_SPACING_MARGIN_M
    for _ in range(WELL_PLACEMENT_ATTEMPTS):
        points = tuple(
            (rng.uniform(-half, half), rng.uniform(-half, half)) for _ in range(WELLS_PER_ZONE)
        )
        if all(
            math.dist(first, second) >= minimum
            for index, first in enumerate(points)
            for second in points[index + 1 :]
        ):
            return points
    # 兜底：2x2 网格 + 抖动，且抖动量按"间距仍达标 + 不出区"取
    grid = 13.0
    jitter = min((2.0 * grid - minimum) / 2.0, half - grid)
    return tuple(
        (
            sign_east * grid + rng.uniform(-jitter, jitter),
            sign_north * grid + rng.uniform(-jitter, jitter),
        )
        for sign_east, sign_north in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    )


def wells_for_round(round_no: int, *, seed: int = SEED) -> tuple[Well, ...]:
    """按轮次与种子排出 8 座天井：**位置与朝向都是随机的**（规则 3.3），由种子复现。"""
    if round_no not in (1, 2):
        raise ValueError(f"轮次只能是 1 或 2，收到 {round_no!r}")
    rng = Random(seed)
    pools: dict[str, tuple[object, ...]] = (
        {
            zone: tuple(source[zone])
            for zone, source in (("A", ROUND1_PICTURES), ("B", ROUND1_PICTURES))
        }
        if round_no == 1
        else {zone: tuple(ROUND2_DIGITS[zone]) for zone in ("A", "B")}
    )
    result: list[Well] = []
    for zone in ("A", "B"):
        center_east, center_north = TARGET_ZONE_CENTERS[zone]
        pool = list(pools[zone])
        offsets = _sample_well_positions(rng)
        for index, (offset_east, offset_north) in enumerate(offsets):
            if index >= EMPTY_WELL_INDEX:
                content: tuple[str, object] | None = None
            elif round_no == 1:
                content = ("pic", str(pool[index]))
            else:
                content = ("digits", pool[index])
            result.append(
                Well(
                    name=f"{zone.lower()}{index + 1}",
                    zone=zone,
                    east_m=center_east + offset_east,
                    north_m=center_north + offset_north,
                    yaw_deg=rng.uniform(0.0, 360.0),
                    content=content,
                )
            )
    return tuple(result)


def build_world(
    round_no: int,
    *,
    seed: int = SEED,
    wind_enu_m_s: tuple[float, float, float] = WIND_ENU_M_S,
    well_arrows: bool = WELL_ARROWS,
    sun_azel_deg: tuple[float, float] = SUN_AZEL_DEG,
    sun_random: bool = SUN_RANDOM,
    aerial_patches: int = AERIAL_PATCHES,
) -> str:
    """生成一个轮次的世界 SDF 文本。"""
    wells = wells_for_round(round_no, seed=seed)
    if sun_random:
        # 随机模式优先：给了 --sun 又开 --sun-random 时以随机为准（避免"半随机"的歧义）
        sun_azel_deg = sun_azel_for(seed, round_no)
    patches = aerial_patch_layout(seed, aerial_patches)
    wind = " ".join(_fmt(component) for component in wind_enu_m_s)
    world_name = f"{WORLD_PREFIX}_r{round_no}"
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sdf version="1.9">',
        f'  <world name="{world_name}">',
        "    <!-- CUADC 固定翼无人机侦察与打击赛区（2026 规则）；由 tools/make_world.py 生成 -->",
        '    <physics type="ode">',
        "      <max_step_size>0.004</max_step_size>",
        "      <real_time_factor>1.0</real_time_factor>",
        "      <real_time_update_rate>250</real_time_update_rate>",
        "    </physics>",
        "    <gravity>0 0 -9.8</gravity>",
        "    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>",
        '    <atmosphere type="adiabatic"/>',
        "    <scene>",
        "      <sky>",
        "        <clouds>",
        "          <speed>12</speed>",
        "        </clouds>",
        "      </sky>",
        "      <grid>false</grid>",
        "      <ambient>0.45 0.45 0.45 0.6</ambient>",
        "      <background>0.35 0.45 0.6 0.8</background>",
        "      <shadows>true</shadows>",
        "    </scene>",
        "    <!-- 风：默认静风；改这里（或重跑 make-world 时用 wind 选项）就给弹道加风 -->",
        "    <wind>",
        f"      <linear_velocity>{wind}</linear_velocity>",
        "    </wind>",
        "",
        _ground_model(),
        "",
        _field_markings_model(seed),
        "",
    ]
    if patches:
        lines.append(_surroundings_model(patches))
        lines.append("")
    for well in wells:
        lines.append(_well_model(well, arrows=well_arrows))
        lines.append("")
    lines.extend(
        [
            _light(sun_azel_deg),
            "",
            "    <spherical_coordinates>",
            "      <surface_model>EARTH_WGS84</surface_model>",
            "      <world_frame_orientation>ENU</world_frame_orientation>",
            f"      <latitude_deg>{WORLD_LAT_DEG}</latitude_deg>",
            f"      <longitude_deg>{WORLD_LON_DEG}</longitude_deg>",
            "      <elevation>0</elevation>",
            "    </spherical_coordinates>",
            "  </world>",
            "</sdf>",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    """统一的小数格式（生成结果可逐字比对：固定三位、去掉尾零与 -0）。"""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _pose(x: float, y: float, z: float = 0.0, *, yaw_deg: float = 0.0) -> str:
    return f"<pose>{_fmt(x)} {_fmt(y)} {_fmt(z)} 0 0 {_fmt(math.radians(yaw_deg))}</pose>"


def _color(r: float, g: float, b: float) -> str:
    return (
        "          <material>\n"
        f"            <diffuse>{r} {g} {b} 1</diffuse>\n"
        "            <specular>0.05 0.05 0.05 1</specular>\n"
        "            <emissive>0 0 0 1</emissive>\n"
        "          </material>"
    )


def _ground_model() -> str:
    return """    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1 1</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.6</mu>
                <mu2>0.6</mu2>
              </ode>
            </friction>
            <bounce/>
            <contact>
              <collide_bitmask>65535</collide_bitmask>
              <ode>
                <min_depth>0.005</min_depth>
                <kp>1e8</kp>
              </ode>
            </contact>
          </surface>
        </collision>
        <visual name="visual">
          <cast_shadows>false</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1400 1400</size>
            </plane>
          </geometry>
          <material>
            <diffuse>0.35 0.55 0.25 1</diffuse>
            <specular>0 0 0 1</specular>
            <emissive>0 0 0 1</emissive>
          </material>
        </visual>
      </link>
    </model>"""


def _flat_visual(
    name: str,
    size: tuple[float, float],
    pose: tuple[float, float, float],
    color: tuple[float, float, float],
) -> str:
    """一块贴地薄片（只有视觉：飞机滑跑时不会被小台阶绊到）。"""
    x, y, z = pose
    return (
        f'        <visual name="{name}">\n'
        f"          {_pose(x, y, z)}\n"
        "          <geometry>\n"
        "            <box>\n"
        f"              <size>{_fmt(size[0])} {_fmt(size[1])} 0.004</size>\n"
        "            </box>\n"
        "          </geometry>\n"
        f"{_color(*color)}\n"
        "        </visual>"
    )


def _field_markings_model(seed: int = SEED) -> str:
    """跑道、起降区标记、操纵区、目标区底色/边界与四角旗（同一个静态模型）。

    ⚠ 不画打击圈：规则 3.3.3 的 r=4m / r=6m 只是规则图 5 里的示意，场地上没有。
    ⚠ 每层标高见 ``*_Z_M`` 常量——两层共面重叠会 z-fighting。
    目标区"底色"是"常见路面色 + 少量对应地面纹理"的贴图块（由 seed 给 A/B 各抽一种），
    用来模拟不同场地面层；纹理由 ``make-backdrops`` 生成。
    """
    parts = [
        '    <model name="cuadc_field_markings">',
        "      <static>true</static>",
        '      <link name="link">',
        '        <visual name="runway">',
        f"          {_pose(0.0, 0.0, RUNWAY_Z_M)}",
        "          <geometry>",
        "            <box>",
        f"              <size>{_fmt(RUNWAY_LENGTH_M)} {_fmt(RUNWAY_WIDTH_M)} 0.004</size>",
        "            </box>",
        "          </geometry>",
        _color(0.62, 0.58, 0.5),
        "        </visual>",
        _flat_visual(
            "landing_zone",
            (LANDING_ZONE_M, LANDING_ZONE_M),
            (0.0, 0.0, LANDING_ZONE_Z_M),
            (0.72, 0.7, 0.64),
        ),
    ]
    for index in range(-4, 5):
        parts.append(
            _flat_visual(
                f"centerline_{index + 5}",
                (10.0, 0.3),
                (index * 20.0, 0.0, RUNWAY_MARK_Z_M),
                (0.92, 0.92, 0.92),
            )
        )
    for edge_name, edge in (("w", -1.0), ("e", 1.0)):
        east = edge * (RUNWAY_LENGTH_M / 2.0 - 10.0)
        for side_name, side in (("s", -1.0), ("n", 1.0)):
            parts.append(
                _flat_visual(
                    f"threshold_{edge_name}_{side_name}",
                    (2.0, 6.0),
                    (east, side * 8.0, RUNWAY_MARK_Z_M),
                    (0.92, 0.92, 0.92),
                )
            )
    parts.append(
        _flat_visual(
            "control_zone",
            (CONTROL_ZONE_LENGTH_M, CONTROL_ZONE_WIDTH_M),
            (CONTROL_ZONE_EAST_M, CONTROL_ZONE_NORTH_M, CONTROL_ZONE_Z_M),
            (0.55, 0.12, 0.12),
        )
    )
    half = TARGET_ZONE_SIDE_M / 2.0
    for zone, (center_east, center_north) in TARGET_ZONE_CENTERS.items():
        tag = zone.lower()
        # 底色：60x60m 的贴图块（纯色 + 少量地面纹理），比区界略大一点
        variant = zone_ground_variant(seed, zone)
        side = TARGET_ZONE_SIDE_M + ZONE_GROUND_MARGIN_M
        parts.append(
            f'        <visual name="zone_{tag}_ground">\n'
            f"          {_pose(center_east, center_north, ZONE_GROUND_Z_M)}\n"
            "          <geometry>\n"
            "            <mesh>\n"
            f"              <uri>{(MESH_DIR / f'sheet_{variant}.obj').as_posix()}</uri>\n"
            f"              <scale>{_fmt(side)} {_fmt(side)} 1</scale>\n"
            "            </mesh>\n"
            "          </geometry>\n"
            "        </visual>"
        )
        # 东西两条线锯短半个线宽：不然四个角上两条线会共面重叠（z-fighting）
        for side, offset in (("w", -half), ("e", half)):
            parts.append(
                _flat_visual(
                    f"zone_{tag}_line_{side}",
                    (0.25, TARGET_ZONE_SIDE_M - 0.5),
                    (center_east + offset, center_north, ZONE_LINE_Z_M),
                    (0.95, 0.95, 0.95),
                )
            )
        for side, offset in (("s", -half), ("n", half)):
            parts.append(
                _flat_visual(
                    f"zone_{tag}_line_{side}",
                    (TARGET_ZONE_SIDE_M, 0.25),
                    (center_east, center_north + offset, ZONE_LINE_Z_M),
                    (0.95, 0.95, 0.95),
                )
            )
        for corner, (offset_east, offset_north) in (
            ("sw", (-half, -half)),
            ("se", (half, -half)),
            ("nw", (-half, half)),
            ("ne", (half, half)),
        ):
            east = center_east + offset_east
            north = center_north + offset_north
            parts.append(
                f'        <visual name="flag_{tag}_{corner}_pole">\n'
                f"          <pose>{_fmt(east)} {_fmt(north)} "
                f"{_fmt(FLAG_POLE_HEIGHT_M / 2)} 0 0 0</pose>\n"
                "          <geometry>\n"
                "            <cylinder>\n"
                f"              <radius>{_fmt(FLAG_POLE_RADIUS_M)}</radius>\n"
                f"              <length>{_fmt(FLAG_POLE_HEIGHT_M)}</length>\n"
                "            </cylinder>\n"
                "          </geometry>\n"
                f"{_color(0.9, 0.9, 0.9)}\n"
                "        </visual>"
            )
            parts.append(
                f'        <visual name="flag_{tag}_{corner}_pennant">\n'
                f"          <pose>{_fmt(east + 0.45)} {_fmt(north)} "
                f"{_fmt(FLAG_POLE_HEIGHT_M - 0.3)} 0 0 0</pose>\n"
                "          <geometry>\n"
                "            <box>\n"
                "              <size>0.9 0.03 0.6</size>\n"
                "            </box>\n"
                "          </geometry>\n"
                f"{_color(0.8, 0.1, 0.1)}\n"
                "        </visual>"
            )
            parts.append(
                f'        <collision name="flag_{tag}_{corner}_collision">\n'
                f"          <pose>{_fmt(east)} {_fmt(north)} "
                f"{_fmt(FLAG_POLE_HEIGHT_M / 2)} 0 0 0</pose>\n"
                "          <geometry>\n"
                "            <cylinder>\n"
                f"              <radius>{_fmt(FLAG_POLE_RADIUS_M)}</radius>\n"
                f"              <length>{_fmt(FLAG_POLE_HEIGHT_M)}</length>\n"
                "            </cylinder>\n"
                "          </geometry>\n"
                "        </collision>"
            )
    parts.extend(["      </link>", "    </model>"])
    return "\n".join(parts)


def _zone_color(zone: str) -> tuple[float, float, float]:
    """分区色：A 区蓝、B 区红（规则 3.3：天井底面颜色就是分区的标志）。

    底板与靶板都用它——同一个区里的四座天井从上看必须是**一种颜色**。
    """
    return (0.12, 0.25, 0.7) if zone == "A" else (0.7, 0.12, 0.12)


def _well_model(well: Well, *, arrows: bool = WELL_ARROWS) -> str:
    """一座天井：五边形竖壁（厚 5mm）+ 同形的蓝/红底面 + 靶板 + 靶纸。

    底面就是"房子"五边形本身（A 区蓝、B 区红，规则 3.3），竖壁紧贴它外侧；
    靶板与底面同形同色，放进壁内正好四边贴壁。整个天井按 ``yaw_deg`` 旋转
    （规则里天井位置与方向都是随机的）。
    ``arrows=True`` 时才画"天井箭头"（规则文本提到、规则图里没有，默认不画）。
    """
    plate_top = WELL_FLOOR_M + PLATE_THICKNESS_M
    floor_color = _zone_color(well.zone)
    parts = [
        f'    <model name="well_{well.name}">',
        "      <static>true</static>",
        f"      {_pose(well.east_m, well.north_m, 0.0, yaw_deg=well.yaw_deg)}",
        '      <link name="well_link">',
        # 底面：靶板形状的加强板（网格 z 放大到 60mm）
        '        <visual name="floor">',
        "          <geometry>",
        "            <mesh>",
        f"              <uri>{PLATE_MESH.as_posix()}</uri>",
        f"              <scale>1 1 {_fmt(WELL_FLOOR_SCALE_Z)}</scale>",
        "            </mesh>",
        "          </geometry>",
        _color(*floor_color),
        "        </visual>",
        '        <collision name="floor_collision">',
        "          <geometry>",
        "            <mesh>",
        f"              <uri>{PLATE_MESH.as_posix()}</uri>",
        f"              <scale>1 1 {_fmt(WELL_FLOOR_SCALE_Z)}</scale>",
        "            </mesh>",
        "          </geometry>",
        "        </collision>",
        # 竖壁：五边形环（外轮廓 = 底面外扩 5mm），高 400mm
        '        <visual name="wall">',
        "          <geometry>",
        "            <mesh>",
        f"              <uri>{WELL_RING_MESH.as_posix()}</uri>",
        "            </mesh>",
        "          </geometry>",
        _color(0.78, 0.78, 0.74),
        "        </visual>",
        '        <collision name="wall_collision">',
        "          <geometry>",
        "            <mesh>",
        f"              <uri>{WELL_RING_MESH.as_posix()}</uri>",
        "            </mesh>",
        "          </geometry>",
        "        </collision>",
    ]
    if arrows:
        parts.extend(_arrow_visuals(plate_top))
    parts.append("      </link>")
    if well.content is not None:
        parts.append(_plate_model(well))
    parts.append("    </model>")
    return "\n".join(parts)


def _arrow_visuals(plate_top: float) -> list[str]:
    """ "天井箭头"：白色箭杆 + 两支箭头，画在靶板尾部、指向天井局部 +x。

    ⚠ 默认**不画**（``WELL_ARROWS = False``）：规则文本 3.3.1/3.3.2 提到"天井箭头
    方向随机"、数字靶方向要与箭头一致，但规则图 2 / 图 4 里并没有这个箭头。
    """
    arrow_z = plate_top + 0.002
    parts = [
        '        <visual name="arrow_shaft">',
        f"          <pose>-0.37 0 {_fmt(arrow_z)} 0 0 0</pose>",
        "          <geometry>",
        "            <box>",
        "              <size>0.22 0.05 0.002</size>",
        "            </box>",
        "          </geometry>",
        _color(0.95, 0.95, 0.95),
        "        </visual>",
    ]
    tip_x = -0.24
    for side, sign in (("ccw", 1.0), ("cw", -1.0)):
        stroke = 0.14
        angle = 40.0 * sign
        center_x = tip_x - 0.5 * stroke * math.cos(math.radians(40.0))
        center_y = sign * 0.5 * stroke * math.sin(math.radians(40.0))
        parts.extend(
            [
                f'        <visual name="arrow_head_{side}">',
                f"          <pose>{_fmt(center_x)} {_fmt(center_y)} {_fmt(arrow_z)} 0 0 "
                f"{_fmt(math.radians(180.0 - angle))}</pose>",
                "          <geometry>",
                "            <box>",
                f"              <size>{_fmt(stroke)} 0.04 0.002</size>",
                "            </box>",
                "          </geometry>",
                _color(0.95, 0.95, 0.95),
                "        </visual>",
            ]
        )
    return parts


def _plate_model(well: Well) -> str:
    """天井里的靶板（嵌套模型）：分区色五边形板（A 蓝 / B 红）+ 靶纸。"""
    if well.content is None:  # pragma: no cover - 调用方已判空
        raise ValueError("空天井没有靶板")
    kind, payload = well.content
    sheet_z = PLATE_THICKNESS_M + 0.001
    parts = [
        '      <model name="plate">',
        "        <static>true</static>",
        f"        <pose>0 0 {_fmt(WELL_FLOOR_M)} 0 0 0</pose>",
        '        <link name="plate_link">',
        '        <visual name="board">',
        "          <geometry>",
        "            <mesh>",
        f"              <uri>{PLATE_MESH.as_posix()}</uri>",
        "            </mesh>",
        "          </geometry>",
        _color(*_zone_color(well.zone)),
        "        </visual>",
    ]
    sheets: list[tuple[str, object, float, float]] = []
    if kind == "pic":
        sheets.append((f"sheet_{payload}", payload, SHEET_SIDE_M, SHEET_SIDE_M))
    else:
        tens, ones = payload  # type: ignore[misc]
        sheets.append(("sheet_0", f"digit_{tens}", SHEET_SIDE_M, DIGIT_SHEET_WIDTH_M))
        sheets.append(("sheet_1", f"digit_{ones}", SHEET_SIDE_M, DIGIT_SHEET_WIDTH_M))
    for index, (name, variant, size_x, size_y) in enumerate(sheets):
        offset_y = DIGIT_SHEET_WIDTH_M / 2.0 if index == 0 else -DIGIT_SHEET_WIDTH_M / 2.0
        offset_y = 0.0 if kind == "pic" else offset_y
        parts.extend(
            [
                f'        <visual name="{name}">',
                f"          <pose>0 {_fmt(offset_y)} {_fmt(sheet_z)} 0 0 0</pose>",
                "          <geometry>",
                "            <mesh>",
                f"              <uri>{(MESH_DIR / f'sheet_{variant}.obj').as_posix()}</uri>",
                f"              <scale>{_fmt(size_x)} {_fmt(size_y)} 1</scale>",
                "            </mesh>",
                "          </geometry>",
                "        </visual>",
            ]
        )
    parts.extend(["        </link>", "      </model>"])
    return "\n".join(parts)


def sun_direction(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """太阳位置 → 平行光的**传播方向**（ENU 分量，单位向量：从太阳指向场景）。

    方位角从北（+y）起顺时针量（0=北、90=东、180=南、270=西），仰角从地平线起算。
    仰角必须在 (0, 90] 内——太阳在地平线下就没有"从上方来的平行光"可言了，
    直接报错而不是生成一个黑世界。
    """
    if not 0.0 < elevation_deg <= 90.0:
        raise ValueError(f"太阳仰角必须在 (0, 90] 度内，收到 {elevation_deg!r}")
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    sun_east = math.cos(elevation) * math.sin(azimuth)
    sun_north = math.cos(elevation) * math.cos(azimuth)
    sun_up = math.sin(elevation)
    return (-sun_east, -sun_north, -sun_up)


def sun_azel_for(seed: int, round_no: int) -> tuple[float, float]:
    """随机太阳（``--sun-random``）：方位角全向、仰角在 ``SUN_ELEVATION_*`` 区间内。

    由 ``seed`` + 轮次决定（同一对输入永远得到同一个太阳）——批量生成不同阴影环境时
    换 ``--seed`` 就能换光照；`Random("sun:<seed>:<轮次>")` 的字符串种子与平台无关。
    """
    rng = Random(f"sun:{seed}:{round_no}")
    return (
        rng.uniform(0.0, 360.0),
        rng.uniform(SUN_ELEVATION_MIN_DEG, SUN_ELEVATION_MAX_DEG),
    )


def _light(sun_azel_deg: tuple[float, float] = SUN_AZEL_DEG) -> str:
    """太阳（平行光）：位置由 ``(方位角, 仰角)`` 决定，阴影随它变。

    阴影开在 ``<cast_shadows>`` 与世界 ``<scene><shadows>`` 两处；
    想模拟不同的阴影环境（侧光长阴影 / 近顶光短阴影）就换 ``--sun``。
    """
    east, north, up = sun_direction(*sun_azel_deg)
    direction = f"{_fmt(east)} {_fmt(north)} {_fmt(up)}"
    return f"""    <light name="sun" type="directional">
      <pose>0 0 500 0 -0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>{direction}</direction>
      <diffuse>0.95 0.95 0.9 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <attenuation>
        <range>2000</range>
        <linear>0</linear>
        <constant>1</constant>
        <quadratic>0</quadratic>
      </attenuation>
    </light>"""


# ----------------------------------------------------------------------
# 网格与材质（文本资产）
# ----------------------------------------------------------------------
def texture_variants() -> tuple[str, ...]:
    """世界里可能用到的全部贴图变体（图片靶 12 张 + 数字板 10 张 + 底图若干）。"""
    return (
        *PICTURE_TARGETS,
        *(f"digit_{digit}" for digit in range(10)),
        *ZONE_GROUND_COLORS,
        *AERIAL_TILES,
    )


def plate_obj() -> str:
    """红色五边形靶板网格（1m 方形 + 等边三角顶角，厚 20mm）。

    局部坐标：方形部分的中心在原点，顶角指向 +x；底面 z=0、顶面 z=20mm。
    顶点 1~5 = 底面、6~10 = 顶面（同序号上下对应）。
    同一份网格也被当作天井底面（z 方向放大成 60mm 厚的加强板）。
    """
    outline = _pentagon_outline()
    lines = [
        "# CUADC 五边形靶板（生成器：tools/make_world.py plate_obj()）",
        "# 局部坐标：方形部分中心在原点、顶角指向 +x；厚 20mm（底面 z=0）",
        "o pentagon_plate",
    ]
    lines += [f"v {_fmt(x)} {_fmt(y)} 0" for x, y in outline]
    lines += [f"v {_fmt(x)} {_fmt(y)} {_fmt(PLATE_THICKNESS_M)}" for x, y in outline]
    lines.append("vn 0 0 -1")
    lines.append("vn 0 0 1")
    for index in range(5):
        x0, y0 = outline[index]
        x1, y1 = outline[(index + 1) % 5]
        nx, ny = (y1 - y0), -(x1 - x0)
        length = math.hypot(nx, ny) or 1.0
        lines.append(f"vn {_fmt(nx / length)} {_fmt(ny / length)} 0")
    lines.append("f 5//1 4//1 3//1 2//1 1//1")
    lines.append("f 6//2 7//2 8//2 9//2 10//2")
    for index in range(5):
        nxt = (index + 1) % 5
        normal = 3 + index
        lines.append(
            f"f {index + 1}//{normal} {nxt + 1}//{normal} {nxt + 6}//{normal} {index + 6}//{normal}"
        )
    return "\n".join(lines) + "\n"


def _pentagon_outline() -> tuple[tuple[float, float], ...]:
    """底板的五边形轮廓（逆时针）：1m 方形 + 等边三角顶角，顶角指向 +x。"""
    return (
        (-PLATE_SIDE_M / 2.0, -PLATE_SIDE_M / 2.0),
        (PLATE_SIDE_M / 2.0, -PLATE_SIDE_M / 2.0),
        (PLATE_SIDE_M / 2.0 + math.sqrt(3.0) / 2.0, 0.0),
        (PLATE_SIDE_M / 2.0, PLATE_SIDE_M / 2.0),
        (-PLATE_SIDE_M / 2.0, PLATE_SIDE_M / 2.0),
    )


def _inward_normal(
    a: tuple[float, float], b: tuple[float, float], distance: float
) -> tuple[float, float]:
    """逆时针多边形里边 (a→b) 的左法线（指向内部），长度 = distance。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    return (-dy / length * distance, dx / length * distance)


def _line_intersection(
    p: tuple[float, float],
    direction_p: tuple[float, float],
    q: tuple[float, float],
    direction_q: tuple[float, float],
) -> tuple[float, float]:
    """两条直线 ``p + t·direction_p`` 与 ``q + s·direction_q`` 的交点。"""
    determinant = direction_p[0] * direction_q[1] - direction_p[1] * direction_q[0]
    if abs(determinant) < 1e-12:
        raise ValueError("两条边平行，无法求偏移后的顶点")
    t = ((q[0] - p[0]) * direction_q[1] - (q[1] - p[1]) * direction_q[0]) / determinant
    return (p[0] + t * direction_p[0], p[1] + t * direction_p[1])


def _offset_polygon(
    outline: tuple[tuple[float, float], ...], distance: float
) -> tuple[tuple[float, float], ...]:
    """凸多边形沿边法线偏移（正 = 向内）：相邻两条偏移直线求交得到新顶点。"""
    count = len(outline)
    result: list[tuple[float, float]] = []
    for index in range(count):
        previous = outline[(index - 1) % count]
        current = outline[index]
        following = outline[(index + 1) % count]
        first = _inward_normal(previous, current, distance)
        second = _inward_normal(current, following, distance)
        result.append(
            _line_intersection(
                (previous[0] + first[0], previous[1] + first[1]),
                (current[0] - previous[0], current[1] - previous[1]),
                (current[0] + second[0], current[1] + second[1]),
                (following[0] - current[0], following[1] - current[1]),
            )
        )
    return tuple(result)


def _face_normal(
    vertices: list[tuple[float, float, float]], face: tuple[int, ...]
) -> tuple[float, float, float]:
    """按面的绕序算法线（右手定则）。"""
    p0, p1, p2 = vertices[face[0] - 1], vertices[face[1] - 1], vertices[face[2] - 1]
    u = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    length = math.sqrt(sum(component * component for component in normal)) or 1.0
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def well_wall_obj() -> str:
    """天井的五边形竖壁（环状网格，厚 ``WELL_WALL_M``、高 ``WELL_HEIGHT_M``）。

    内轮廓 = 底板轮廓（底面就贴在壁上，靶板正好放进壁内）；外轮廓 = 内轮廓外扩。
    局部坐标与底板一致：方形部分中心在原点、顶角指向 +x。
    """
    inner = _pentagon_outline()
    outer = _offset_polygon(inner, -WELL_WALL_M)
    bottom, top = 0.0, WELL_HEIGHT_M
    vertices: list[tuple[float, float, float]] = []
    for ring in (outer, inner):  # 1~5 外轮廓 / 6~10 内轮廓（先底后顶各一轮）
        vertices += [(x, y, bottom) for x, y in ring]
        vertices += [(x, y, top) for x, y in ring]
    faces: list[tuple[int, ...]] = []
    for index in range(len(inner)):
        nxt = (index + 1) % len(inner)
        ob, ot = index + 1, index + 6  # 外轮廓底/顶
        nb, nt = index + 11, index + 16  # 内轮廓底/顶
        faces.append((ob, nxt + 1, nxt + 6, ot))  # 外侧面（法线朝外）
        faces.append((nb, nt, nxt + 16, nxt + 11))  # 内侧面（法线朝井内）
        faces.append((ot, nxt + 6, nxt + 16, nt))  # 顶环
        faces.append((ob, nb, nxt + 11, nxt + 1))  # 底环
    lines = [
        "# CUADC 天井五边形竖壁（生成器：tools/make_world.py well_wall_obj()）",
        "# 内轮廓 = 靶板轮廓；外轮廓 = 内轮廓外扩 5mm；高 400mm；方形部分中心在原点、顶角指向 +x",
        "o pentagon_well_wall",
    ]
    lines += [f"v {_fmt(x)} {_fmt(y)} {_fmt(z)}" for x, y, z in vertices]
    for normal_index, face in enumerate(faces, start=1):
        nx, ny, nz = _face_normal(vertices, face)
        lines.append(f"vn {_fmt(nx)} {_fmt(ny)} {_fmt(nz)}")
        lines.append("f " + " ".join(f"{vertex}//{normal_index}" for vertex in face))
    return "\n".join(lines) + "\n"


def sheet_obj(variant: str) -> str:
    """靶纸平面（1m x 1m 单位面）。UV 约定：图像上方 → +x、图像左侧 → +y。"""
    return f"""# CUADC 靶纸平面（生成器：tools/make_world.py sheet_obj()）
# UV 约定：图像上方(v=1) → +x；图像左侧(u=0) → +y（已用离屏渲染核对，见 docs/simulation_world.md）
mtllib sheet_{variant}.mtl
o sheet_{variant}
v -0.5 -0.5 0.0
v 0.5 -0.5 0.0
v 0.5 0.5 0.0
v -0.5 0.5 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vt 0.0 0.0
vn 0.0 0.0 1.0
usemtl {variant}
f 1/1/1 2/2/1 3/3/1 4/4/1
"""


def sheet_mtl(variant: str) -> str:
    """靶纸材质：``map_Kd`` 指向贴图（相对 meshes/ 目录）。"""
    return f"""newmtl {variant}
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
Ns 10.0
d 1.0
illum 1
map_Kd ../textures/{variant}.png
"""


def build_meshes() -> dict[str, str]:
    """全部文本资产：路径（相对输出目录）→ 内容。"""
    files: dict[str, str] = {
        PLATE_MESH.as_posix(): plate_obj(),
        WELL_RING_MESH.as_posix(): well_wall_obj(),
    }
    for variant in texture_variants():
        files[f"{(MESH_DIR / f'sheet_{variant}.obj').as_posix()}"] = sheet_obj(variant)
        files[f"{(MESH_DIR / f'sheet_{variant}.mtl').as_posix()}"] = sheet_mtl(variant)
    return files


# ----------------------------------------------------------------------
# 落盘与校验
# ----------------------------------------------------------------------
def referenced_assets(world_text: str) -> list[str]:
    """从世界 SDF 里抽出引用的资产路径（``<uri>`` / ``<albedo_map>``，去重、保持顺序）。

    ⚠ 这些 URI 是**元素文本**（``<uri>materials/...</uri>``），不是属性值——
    解析必须按元素文本取，否则一条都抓不到、自检形同虚设（回归用例见 tests/test_world.py）。
    """
    root = ET.fromstring(world_text)
    found: list[str] = []
    for element in root.iter():
        if element.tag not in ("uri", "albedo_map") or not element.text:
            continue
        value = element.text.strip()
        if value.startswith("materials/") and value not in found:
            found.append(value)
    return found


def _mtl_textures(mtl_path: Path) -> list[Path]:
    """MTL 里 ``map_Kd`` 指向的贴图（相对 MTL 所在目录）。"""
    textures: list[Path] = []
    for line in mtl_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("map_Kd "):
            textures.append(mtl_path.parent / line.split(" ", 1)[1].strip())
    return textures


def asset_problems(out_dir: Path, world_texts: dict[Path, str]) -> list[str]:
    """核验「世界引用的网格/材质 + 材质引用的贴图」都在（缺什么就报什么）。"""
    problems: list[str] = []
    checked: set[str] = set()
    for world_text in world_texts.values():
        for reference in referenced_assets(world_text):
            if reference in checked:
                continue
            checked.add(reference)
            path = out_dir / reference
            if not path.exists():
                problems.append(f"缺少资产：{reference}")
            elif path.suffix == ".obj":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.startswith("mtllib "):
                        continue
                    mtl = path.parent / line.split(" ", 1)[1].strip()
                    if not mtl.exists():
                        problems.append(f"{reference} 引用的材质不存在：{mtl}")
                        continue
                    for texture in _mtl_textures(mtl):
                        if not texture.exists():
                            problems.append(f"{mtl.name} 引用的贴图不存在：{texture}")
    return problems


def copy_textures(out_dir: Path) -> list[str]:
    """把入库的贴图复制进目标目录（``--out-dir`` 指到别处时世界才是完整可跑的）。"""
    source = OUT_DIR / TEXTURE_DIR
    target = out_dir / TEXTURE_DIR
    if target.resolve() == source.resolve():
        return []
    if not source.is_dir():
        return [f"找不到贴图源目录：{source}（入库资产被移走了？）"]
    target.mkdir(parents=True, exist_ok=True)
    for png in sorted(source.glob("*.png")):
        shutil.copyfile(png, target / png.name)
    return []


def world_texts(
    *,
    seed: int = SEED,
    rounds: str = ROUNDS,
    wind_enu_m_s: tuple[float, float, float] = WIND_ENU_M_S,
    well_arrows: bool = WELL_ARROWS,
    sun_azel_deg: tuple[float, float] = SUN_AZEL_DEG,
    sun_random: bool = SUN_RANDOM,
    aerial_patches: int = AERIAL_PATCHES,
) -> dict[Path, str]:
    """按轮次选择要生成的世界：文件名 → SDF 文本。"""
    if rounds not in ("both", "1", "2"):
        raise ValueError(f"rounds 只能是 both / 1 / 2，收到 {rounds!r}")
    selected = (1, 2) if rounds == "both" else (int(rounds),)
    return {
        Path(f"{WORLD_PREFIX}_r{round_no}.sdf"): build_world(
            round_no,
            seed=seed,
            wind_enu_m_s=wind_enu_m_s,
            well_arrows=well_arrows,
            sun_azel_deg=sun_azel_deg,
            sun_random=sun_random,
            aerial_patches=aerial_patches,
        )
        for round_no in selected
    }


def main(
    *,
    out_dir: Path | str = OUT_DIR,
    seed: int = SEED,
    rounds: str = ROUNDS,
    wind_enu_m_s: tuple[float, float, float] = WIND_ENU_M_S,
    well_arrows: bool = WELL_ARROWS,
    sun_azel_deg: tuple[float, float] = SUN_AZEL_DEG,
    sun_random: bool = SUN_RANDOM,
    aerial_patches: int = AERIAL_PATCHES,
) -> int:
    """生成世界与文本资产；返回进程退出码（0 成功 / 1 参数非法或缺资产）。"""
    out = Path(out_dir)
    try:
        worlds = world_texts(
            seed=seed,
            rounds=rounds,
            wind_enu_m_s=wind_enu_m_s,
            well_arrows=well_arrows,
            sun_azel_deg=sun_azel_deg,
            sun_random=sun_random,
            aerial_patches=aerial_patches,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    # 先自检（XML 合法），再落盘——别把坏世界写进仓库
    for name, text in worlds.items():
        try:
            ET.fromstring(text)
        except ET.ParseError as exc:
            print(f"FAIL: {name} 不是合法 XML：{exc}")
            return 1

    meshes = build_meshes()
    (out / MESH_DIR).mkdir(parents=True, exist_ok=True)
    for relative, text in meshes.items():
        (out / relative).write_text(text, encoding="utf-8", newline="\n")
    problems = copy_textures(out) + asset_problems(out, worlds)
    if problems:
        print("FAIL: 资产不齐（缺贴图就补 sim/worlds/cuadc/materials/textures/）")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    for name, text in worlds.items():
        round_no = int(name.stem.rsplit("r", 1)[1])
        wells = wells_for_round(round_no, seed=seed)
        targets = sum(1 for well in wells if well.content is not None)
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        drawn = sun_azel_for(seed, round_no) if sun_random else sun_azel_deg
        tag = "太阳随机 " if sun_random else "太阳 "
        ground = "/".join(zone_ground_variant(seed, zone) for zone in ("A", "B"))
        print(
            f"[OK  ] {out / name}：8 座天井 / {targets} 座有靶标"
            f"（种子 {seed}，风 {wind_enu_m_s}，{tag}{drawn[0]:g}°/{drawn[1]:g}°，"
            f"底色 {ground}，周边航拍 {aerial_patches} 块）"
        )
    print(f"[OK  ] 文本资产 {len(meshes)} 个（靶板网格 + 靶纸网格/材质）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
