"""主仓演练对仿真产物的适配；工具包不依赖此集成层。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.optional_tools import load_tool_module

load_tool_module("airdrop_sim")

from airdrop_sim.camera import CameraConfig  # noqa: E402
from airdrop_sim.manifest import read_manifest  # noqa: E402
from airdrop_sim.world import WorldConfig, WorldTarget, world_truth  # noqa: E402


@dataclass(frozen=True, slots=True)
class ReconScene:
    """固定一次演练的相机与真值，避免不同演练共享可变全局配置。"""

    camera: CameraConfig
    origin_lla: tuple[float, float, float]
    wells_enu: dict[int, tuple[float, float]]
    world_name: str
    seed: int
    round_no: int

    @property
    def backup_point_enu(self) -> tuple[float, float]:
        """用同一批真值的形心作为备用点。"""
        count = len(self.wells_enu)
        return (
            sum(point[0] for point in self.wells_enu.values()) / count,
            sum(point[1] for point in self.wells_enu.values()) / count,
        )

    @property
    def holdout_code(self) -> int:
        """中位数编号只作延时留出验证。"""
        codes = sorted(self.wells_enu)
        return codes[len(codes) // 2]


def _numbered_targets(truth: tuple[WorldTarget, ...]) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for target in truth:
        if target.zone != "A" or target.code is None:
            continue
        if target.code in points:
            raise ValueError(f"侦查真值存在重复编号 {target.code}，无法唯一对应落点")
        points[target.code] = (target.east_m, target.north_m)
    if not points:
        raise ValueError("侦查演练需要 A 区数字靶标真值")
    return points


def load_recon_scene(manifest_path: Path | str | None = None) -> ReconScene:
    """缺省复现 r2/seed 0；显式 manifest 从文件读取本次真值和相机配置。"""
    if manifest_path is None:
        world = WorldConfig(rounds="2")
        return ReconScene(
            camera=CameraConfig(),
            origin_lla=(world.origin_lon_deg, world.origin_lat_deg, world.origin_alt_m),
            wells_enu=_numbered_targets(world_truth(2, seed=world.seed)),
            world_name="cuadc_recon_strike_r2",
            seed=world.seed,
            round_no=2,
        )
    manifest = read_manifest(Path(manifest_path))
    if manifest.round_no != 2:
        raise ValueError("侦查演练当前只支持第二轮数字靶标，不能用图片靶标报告编号误差")
    return ReconScene(
        camera=CameraConfig(**manifest.camera),
        origin_lla=(
            float(manifest.origin["longitude_deg"]),
            float(manifest.origin["latitude_deg"]),
            float(manifest.origin["elevation_m"]),
        ),
        wells_enu=_numbered_targets(manifest.truth),
        world_name=f"cuadc_recon_strike_r{manifest.round_no}",
        seed=manifest.seed,
        round_no=manifest.round_no,
    )
