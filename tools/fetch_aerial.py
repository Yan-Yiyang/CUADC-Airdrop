"""抓取**真实航拍底图**（可选：替换 ``make-backdrops`` 的程序化底图）。

来源：USGS 的 NAIP 影像服务（美国农业部 NAIP 航拍影像，**公有领域**，可自由再分发；
入库时请保留来源说明，见 ``docs/simulation_world.md`` §4）。分辨率约 0.3~1m/像素，
跟本项目下视相机（50~100m 高度）看到的尺度接近。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run fetch-aerial
    ./.venv/Scripts/python.exe -m airdrop.run fetch-aerial --size 640 --tile-width-m 200

每张底图 = 在服务上按经纬度取一块正方形影像，裁成 ``aerial_1..N.png``
（名字来自 ``make_world.AERIAL_TILES``）写进 ``<out_dir>/materials/textures/``；
之后 ``make-world`` 就会把这些真实影像铺在比赛区域外。
⚠ **需要联网**；离线（或不想要真实影像）请用 ``make-backdrops`` 的程序化底图——
两者写的是同一组文件名，谁后跑谁生效。抓下来的 PNG 是否入库、是否对外分发，
由使用者决定（公有领域，但请保留署名）。

只依赖标准库 + Pillow（Pillow 在函数体内导入）。
"""

from __future__ import annotations

import math
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from tools import make_world

if TYPE_CHECKING:  # 只为类型标注；运行时不导入 PIL（惰性导入的约定）
    from PIL import Image

#: USGS 影像服务（NAIP，公有领域；无需 API key）
SERVICE_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
)
#: 抓取点：不同地貌（农田/郊区/工业/荒漠/林地/湿地），经纬度写死 → 每次抓同一批
SPOTS: tuple[tuple[str, float, float], ...] = (
    ("iowa_farmland", -93.50000, 42.00000),
    ("texas_suburb", -97.70000, 30.30000),
    ("illinois_industrial", -87.90000, 41.80000),
    ("arizona_desert", -112.00000, 33.40000),
    ("tennessee_woodland", -86.50000, 35.90000),
    ("louisiana_wetland", -90.10000, 29.60000),
)

OUT_DIR = make_world.OUT_DIR
SIZE_PX = 512
#: 每张底图覆盖的地面宽度（米）；160m/512px ≈ 0.31 m/像素
TILE_WIDTH_M = 160.0
TIMEOUT_S = 60.0
#: 来源署名（写进汇总文档；抓到的东西是公有领域，但署名是好习惯）
ATTRIBUTION = "USDA NAIP imagery via USGS National Map (public domain)"


@dataclass(frozen=True, slots=True)
class FetchAerialConfig:
    """航拍底图抓取配置（字段 = 命令行能覆盖的东西）。"""

    out_dir: Path = make_world.OUT_DIR
    size: int = SIZE_PX
    tile_width_m: float = TILE_WIDTH_M


def build_config(**overrides) -> FetchAerialConfig:
    """按关键字覆盖派生一份配置（``dataclasses.replace``）。"""
    if "out_dir" in overrides:
        overrides["out_dir"] = Path(overrides["out_dir"])
    return replace(FetchAerialConfig(), **overrides)


def export_image_url(lon: float, lat: float, *, width_m: float, size: int) -> str:
    """按中心经纬度与地面宽度拼出 exportImage 请求（等距近似：1° 纬度 ≈ 111320m）。"""
    half_north = width_m / 2.0
    half_east = half_north / max(0.2, abs(math.cos(math.radians(lat))))
    west = lon - half_east / 111320.0
    east = lon + half_east / 111320.0
    south = lat - half_north / 111320.0
    north = lat + half_north / 111320.0
    query = urllib.parse.urlencode(
        {
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{size},{size}",
            "format": "png",
            "f": "image",
        }
    )
    return f"{SERVICE_URL}?{query}"


def fetch_tile(url: str, *, timeout_s: float = TIMEOUT_S) -> Image.Image:
    """抓一张影像并转成 RGB（PNG 由服务端直接返回）。"""
    from io import BytesIO

    from PIL import Image as PilImage

    request = urllib.request.Request(url, headers={"User-Agent": "airdrop-cuadc-world/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read()
    if not payload.startswith(b"\x89PNG"):
        raise ValueError(f"服务没有返回 PNG（前 8 字节：{payload[:8]!r}）")
    return PilImage.open(BytesIO(payload)).convert("RGB")


def main(
    *,
    out_dir: Path | str = make_world.OUT_DIR,
    size: int = SIZE_PX,
    tile_width_m: float = TILE_WIDTH_M,
) -> int:
    """抓 ``len(AERIAL_TILES)`` 张底图写进 ``<out_dir>/materials/textures/``。"""
    if size < 128:
        print(f"FAIL: 贴图尺寸太小（要 >= 128 像素），收到 {size!r}")
        return 1
    if tile_width_m <= 0:
        print(f"FAIL: 地面宽度必须为正数，收到 {tile_width_m!r}")
        return 1
    target = Path(out_dir) / make_world.TEXTURE_DIR
    target.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(make_world.AERIAL_TILES):
        spot, lon, lat = SPOTS[index % len(SPOTS)]
        url = export_image_url(lon, lat, width_m=tile_width_m, size=size)
        try:
            image = fetch_tile(url)
        except (OSError, ValueError) as exc:  # 抓取失败要给出人话（含服务端 4xx/5xx）
            print(f"FAIL: 抓 {name}（{spot}）失败：{exc}")
            print("      （离线环境请用 python -m airdrop.run make-backdrops）")
            return 1
        path = target / f"{name}.png"
        image.save(path)
        print(f"[OK  ] {path}（{spot} @ {lon},{lat}，{tile_width_m:g}m/{size}px）")
    print(f"[OK  ] 来源：{ATTRIBUTION}")
    print(
        "       ⚠ 入库/对外分发由你决定；保留来源说明（docs/simulation_world.md §4）。"
        "想换回程序化底图：python -m airdrop.run make-backdrops"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
