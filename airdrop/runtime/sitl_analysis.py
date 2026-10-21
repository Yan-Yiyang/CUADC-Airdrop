"""SITL 结果分析与精度报告。"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from airdrop import Config, TelemetryBroker

from .sitl_geometry import (
    _DEFAULT_SCENE,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    LAG_SWEEP_S,
    TELEMETRY_LAG_S,
)
from .sitl_scene import ReconScene

LOGGER = logging.getLogger("sitl_recon")


def calibrate_lag(
    sweep_rows: list[dict[str, Any]], *, holdout_code: int = 56
) -> tuple[float | None, dict[str, Any]]:
    """从延时扫描结果里**自标定链路延时**：用除 ``holdout_code`` 外的天井误差最小者。

    留一个天井不参与标定、只用它验收——避免"用同一批数据既标定又出成绩"。
    返回 ``(lag_s, 说明)``；扫描结果不足时返回 ``(None, {})``。
    """
    candidates = [
        row
        for row in sweep_rows
        if row.get("selected_error_m") is not None and len(row.get("code_errors_m", {})) >= 2
    ]
    if not candidates:
        return None, {}

    def score(row: dict[str, Any]) -> float:
        errors = [value for key, value in row["code_errors_m"].items() if int(key) != holdout_code]
        return sum(errors) / len(errors) if errors else float("inf")

    best = min(candidates, key=score)
    holdout = best["code_errors_m"].get(str(holdout_code))
    return best["lag_s"], {
        "holdout_code": holdout_code,
        "holdout_error_m": holdout,
        "calibration_score_m": round(score(best), 3),
        "note": "用其余天井标定链路延时，留出天井只做验证",
    }


# ----------------------------------------------------------------------
# 结果分析
# ----------------------------------------------------------------------
def _well_truth_ned(
    code: int, origin_lla: tuple[float, float, float], *, scene: ReconScene | None = None
) -> tuple[float, float]:
    """天井中心在**本地 NED** 里的真值（原点 = EKF 原点）。"""
    from airdrop import ned_to_wgs84, wgs84_to_ned

    context = scene or _DEFAULT_SCENE
    east_m, north_m = context.wells_enu[code]
    lon, lat, _ = ned_to_wgs84((north_m, east_m, 0.0), ref=context.origin_lla)
    north, east, _ = wgs84_to_ned(lon, lat, context.origin_lla[2], ref=origin_lla)
    return north, east


def _cluster_error(cluster, truth: tuple[float, float]) -> float:
    return math.hypot(cluster.north_m - truth[0], cluster.east_m - truth[1])


def _select_by_code(clusters, code: int):
    same = [cluster for cluster in clusters if cluster.code == code]
    if not same:
        return None
    same.sort(key=lambda item: (-item.count, -item.confidence))
    return same[0]


def lag_sweep(
    detections: list[dict[str, Any]],
    *,
    broker: TelemetryBroker,
    calib_file: str,
    sweeps: tuple[float, ...] = LAG_SWEEP_S,
    base_lag: float = TELEMETRY_LAG_S,
    scene: ReconScene | None = None,
) -> list[dict[str, Any]]:
    """同一批检测按不同链路延时重算坐标（纯离线：只用像素 + 拍摄时刻 + 遥测）。

    像素→NED 用的是与实飞完全相同的 :func:`~airdrop.georef.pixel_to_ned`；
    遥测取"收到时刻 − 延时"，从飞行记录重建的历史里查（``interpolate``）。
    """
    from airdrop import TargetPoint, analyze, load_camera_model, pixel_to_ned

    camera = load_camera_model(calib_file, fallback_size=(CAMERA_WIDTH, CAMERA_HEIGHT))
    base_config = Config()
    targeting = replace(base_config.targeting, selection_rule="median")
    results: list[dict[str, Any]] = []
    for lag in sweeps:
        points = []
        for row in detections:
            stamp = float(row["capture_timestamp"]) + base_lag - lag
            snapshot = broker.get_snapshot_at(stamp, "interpolate")
            if snapshot is None or snapshot.north_m is None or snapshot.roll_deg is None:
                continue
            fix = pixel_to_ned(
                (row["pixel"][0], row["pixel"][1]),
                camera=camera,
                ground_z=0.0,
                position_ned=(snapshot.north_m, snapshot.east_m, snapshot.down_m or 0.0),
                euler_deg=(snapshot.roll_deg, snapshot.pitch_deg, snapshot.yaw_deg),
            )
            if not fix.ok or fix.ned is None:
                continue
            points.append(
                TargetPoint(
                    north_m=fix.ned[0],
                    east_m=fix.ned[1],
                    down_m=fix.ned[2],
                    capture_timestamp=stamp,
                    frame_index=int(row["frame_index"]),
                    code=row["code"],
                    confidence=float(row["ocr_confidence"] or row["confidence"]),
                )
            )
        result = analyze(points, targeting)
        origin = broker.get_snapshot()
        origin_lla = (
            origin.origin_longitude_deg,
            origin.origin_latitude_deg,
            origin.origin_altitude_m,
        )
        codes: dict[int, float] = {}
        for code in (scene or _DEFAULT_SCENE).wells_enu:
            cluster = _select_by_code(result.clusters, code)
            if cluster is None:
                continue
            codes[code] = _cluster_error(cluster, _well_truth_ned(code, origin_lla, scene=scene))
        selected = result.selected
        results.append(
            {
                "lag_s": lag,
                "points": len(points),
                "clusters": len(result.clusters),
                "selected_code": None if selected is None else selected.code,
                "selected_error_m": (
                    None
                    if selected is None
                    else round(
                        _cluster_error(
                            selected, _well_truth_ned(selected.code, origin_lla, scene=scene)
                        ),
                        3,
                    )
                ),
                "code_errors_m": {str(code): round(value, 3) for code, value in codes.items()},
            }
        )
    return results


def _load_detail(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _roll_stats(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """扫掠段（有检测的那些帧）的横滚统计：转弯倾斜角有没有污染成像。

    明细来自记录器的 ``detections.jsonl``（``Detection.as_dict()`` 口径），
    横滚在嵌套的 ``telemetry`` 里。
    """
    rolls = [
        abs(float(row["telemetry"]["roll_deg"]))
        for row in detections
        if row.get("telemetry", {}).get("roll_deg") is not None
    ]
    if not rolls:
        return {"n": 0}
    return {
        "n": len(rolls),
        "mean_abs_deg": round(sum(rolls) / len(rolls), 2),
        "max_abs_deg": round(max(rolls), 2),
    }


#: 航向一致性容差（度）：``|航迹角 − 记录的 yaw|`` 超过它就认为**姿态估计不可信**。
#:
#: 航迹角与记录航向不一致可提示姿态估计异常，也可能来自转弯或侧滑。
#: 该指标只用于诊断并写入报告，不据此替换姿态或剔除观测；相机固定在机体上，
#: 像素到地面的投影仍须使用同一时刻的机体姿态。
HEADING_TOL_DEG = 10.0

#: 算航迹角所需的最小水平速度（m/s）：低于它 ``atan2(vy, vx)`` 的方向没有意义
#:（停机坪上速度≈0，方向纯噪声）。接近静止时，速度方向主要由定位噪声决定。
MIN_COURSE_SPEED_M_S = 3.0


def _course_deg(snap: dict[str, Any] | None) -> float | None:
    """NED 速度矢量的航迹角（度，归一到 ``[0, 360)``，正东 = 90）。

    用的是快照里**独立于姿态**的一条流（``velocity_ned``），所以它能当"航向"的独立参照。
    位置差分也能算，但那条路要相邻两条位置、且 36~40% 的行位置是保持不变的（源更新率低于
    记录率），差分噪声大得多；速度矢量每条都有。
    """
    if not snap:
        return None
    vx, vy = snap.get("vx_m_s"), snap.get("vy_m_s")
    if vx is None or vy is None:
        return None
    if math.hypot(vx, vy) < MIN_COURSE_SPEED_M_S:
        return None
    return math.degrees(math.atan2(vy, vx)) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """两个角度之差，折叠到 ``[0, 180]``（不分正负）。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _heading_consistency(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """遥测 ``yaw`` 与速度矢量航迹角的差（数据质量诊断，见 :data:`HEADING_TOL_DEG`）。

    ⚠ ``yaw_deg`` 嵌在 ``telemetry`` 子字典里（``Detection.as_dict()`` 口径，与
    :func:`_roll_stats` 同源）；速度在 ``vx_m_s``/``vy_m_s``，同样嵌着。
    """
    devs: list[float] = []
    for row in detections:
        snap = row.get("telemetry") or {}
        course = _course_deg(snap)
        yaw = snap.get("yaw_deg")
        if course is None or yaw is None:
            continue
        devs.append(_angle_diff_deg(course, float(yaw)))
    if not devs:
        return {"n": 0, "tolerance_deg": HEADING_TOL_DEG}
    devs.sort()
    over = [value for value in devs if value > HEADING_TOL_DEG]
    n = len(devs)
    p90 = devs[min(n - 1, int(0.9 * n))]
    return {
        "n": n,
        "tolerance_deg": HEADING_TOL_DEG,
        "median_deg": round(devs[n // 2], 2),
        "p90_deg": round(p90, 2),
        "max_deg": round(devs[-1], 2),
        "over_tolerance": len(over),
        "over_fraction": round(len(over) / n, 3),
        "note": (
            "航迹角取自速度矢量（独立于姿态的一条流），与记录的 yaw 比。"
            "偏差大说明**姿态估计不可信**，而解算用的是姿态——"
            "这批观测的落点误差会被放大，不该算在本项目坐标解算头上。"
            "本工具不剔点，两个口径的精度都报（见 observation_errors）。"
        ),
    }


def _per_observation_errors(
    detections: list[dict[str, Any]],
    *,
    origin_lla: tuple[float, float, float],
    calib_file: str,
    scene: ReconScene | None = None,
) -> list[dict[str, Any]]:
    """逐观测的落点误差 + 该观测的航向偏差（米 / 度）。

    与 :func:`lag_sweep` 同一条解算路径（同一个 :func:`~airdrop.georef.pixel_to_ned`、
    同一份标定），只是**直接用检测自带的那份遥测**——省掉一次延时扫描，且和管线实际
    用的输入完全一致。``ground_z`` 与 :func:`lag_sweep` 一样取 0（靶区地面在世界里 z=0）。
    """
    from airdrop import load_camera_model, pixel_to_ned

    camera = load_camera_model(calib_file, fallback_size=(CAMERA_WIDTH, CAMERA_HEIGHT))
    rows: list[dict[str, Any]] = []
    for row in detections:
        code = row.get("code")
        if code is None or code not in (scene or _DEFAULT_SCENE).wells_enu:
            continue
        snap = row.get("telemetry") or {}
        if snap.get("north_m") is None or snap.get("roll_deg") is None:
            continue
        fix = pixel_to_ned(
            (row["pixel"][0], row["pixel"][1]),
            camera=camera,
            ground_z=0.0,
            position_ned=(snap["north_m"], snap["east_m"], snap["down_m"] or 0.0),
            euler_deg=(snap["roll_deg"], snap["pitch_deg"], snap["yaw_deg"]),
        )
        if not fix.ok or fix.ned is None:
            continue
        truth = _well_truth_ned(code, origin_lla, scene=scene)
        course = _course_deg(snap)
        rows.append(
            {
                "code": code,
                "frame_index": int(row["frame_index"]),
                "error_m": round(math.hypot(fix.ned[0] - truth[0], fix.ned[1] - truth[1]), 3),
                "heading_dev_deg": (
                    None if course is None else round(_angle_diff_deg(course, snap["yaw_deg"]), 2)
                ),
            }
        )
    return rows


def _error_stats(values: list[float]) -> dict[str, Any]:
    """一组误差（米）的汇总：中位 / 均方根 / p90 / 最大 / <2 m 的比例。"""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    mean_square = sum(value * value for value in ordered) / n
    return {
        "n": n,
        "median_m": round(ordered[n // 2], 3),
        "rms_m": round(math.sqrt(mean_square), 3),
        "p90_m": round(ordered[min(n - 1, int(0.9 * n))], 3),
        "max_m": round(ordered[-1], 3),
        "under_2m_fraction": round(sum(1 for v in ordered if v < 2.0) / n, 3),
    }


#: 反向诊断的**同轴容差**（度）：两遍的平均航迹角都要落在正东/正西 ±这个范围内。
#: 15° 覆盖了转弯中短暂的偏航，又足以把"西南向对角线"（差 40~50°）挡在外面。
REVERSE_AXIS_TOL_DEG = 15.0


def _circular_mean_deg(values: list[float]) -> float:
    """角度的圆周均值，归一到 ``[0, 360)``。

    直接取算术平均会在 ±180° 附近翻错（179° 与 181° 会算出 0°）。
    ⚠ 末尾那个 ``if`` 不是冗余：``% 360.0`` 遇到**恰好** 0 的输入会因浮点误差返回
    360.0（实测 ``[350°, 10°]`` → 360.0），报告里显示"航迹角 360°"很难看。
    """
    if not values:
        raise ValueError("角度列表为空")
    radians = [math.radians(value) for value in values]
    result = (
        math.degrees(
            math.atan2(sum(math.sin(r) for r in radians), sum(math.cos(r) for r in radians))
        )
        % 360.0
    )
    return 0.0 if result >= 360.0 else result


def _axis_check_note(bearings: dict[str, float]) -> str | None:
    """两遍观测是否沿**同一条轴反向**通过；不是就返回一句原因（不是 ``None``）。

    判据是**两组航迹角之差接近 180°**（反向），而不是"各自都接近正东 90°"。
    ⚠ 后者会把真正合格的**正东 90° / 正西 270°** 判成不合格——正西离正东正好 180°，
    正是这个校验最该放行的那一组。（第一版就是这么写的，用例当场抓住。）
    """
    if len(bearings) < 2:  # 两组各一个方向，少于 2 组就没法比
        return "缺少两个方向的航迹角，无法做反向诊断"
    east, west = bearings.get("+east"), bearings.get("-east")
    if east is None or west is None:
        return "缺少 +east 或 -east 的航迹角，无法做反向诊断"

    # 角差折叠到 0~180，再看离"正好反向"（180°）有多远
    separation = abs((east - west + 180.0) % 360.0 - 180.0)
    off_opposite = abs(separation - 180.0)
    if off_opposite > REVERSE_AXIS_TOL_DEG:
        return (
            f"两遍不在同一条轴上反向通过（航迹角 +east {east:.0f}° / -east {west:.0f}°，"
            f"相差 {separation:.0f}°，离正好反向差 {off_opposite:.0f}° > 容差 "
            f"{REVERSE_AXIS_TOL_DEG:.0f}°）：Δeast 里混的是切向差而不是链路延时，"
            "不给延时估计"
        )
    return None


def _reverse_diagnostic(
    detections: list[dict[str, Any]],
    broker: TelemetryBroker,
    *,
    calib_file: str,
    code: int,
    lag_s: float,
    base_lag: float = TELEMETRY_LAG_S,
) -> dict[str, Any]:
    """同一座天井在两个方向的观测偏差 → 估计链路延时（偏差 = 2·v·Δt）。

    用检测时飞机的**速度方向**把观测分组：东向那遍、东向为负那遍，两者之差除以
    2·v 就是延时估计。只做**诊断**，不拿它改结果。

    ⚠ **同轴校验是这个数的成立前提**：``Δeast/(2v)`` 只有在两遍**沿同一条轴反向**通过时
    才成立——坐标差里除了延时没有别的东西。只按东向速度的符号分组**不够**：侦查航线的
    反向观测来自一条**西南向对角线**（见 :data:`RECON_LEGS_ENU`——固定翼不能在原地掉头，
    同线来回飞与"转得出来"在这个靶区布局下互斥），它与主扫掠线差 40~50°，照样会被分到
    "-east" 组里，于是 Δeast 主要反映**切向差**而不是延时。

    若忽略这一前提，航迹几何差异可能被误认为时间延迟，甚至导致估计符号相反。因此用
    :func:`_axis_check_note` 显式校验两组的平均航迹角：都落在正东/正西 ±
    :data:`REVERSE_AXIS_TOL_DEG` 才给数，否则只报"两遍不同轴"和两个航迹角。
    要一个稳定的同轴反向数，得给侦查段加 ``DO_CHANGE_SPEED`` 把速度压到 ~12 m/s
    （半径才缩得下 19 m 的靶区间距），本包目前没有这个能力。
    """
    # ⚠ yaw_deg 嵌在 telemetry 子字典里（Detection.as_dict() 口径，同文件的
    # _roll_stats 就是这么读的）。曾经在顶层取 row.get("yaw_deg")，恒为 None ⇒
    # rows 恒空 ⇒ 这个反向诊断恒返回 {}，报告里"链路延时自标定"一栏永远空着，
    # 且没有任何异常或告警。
    rows = [
        row
        for row in detections
        if row.get("code") == code and row.get("telemetry", {}).get("yaw_deg") is not None
    ]
    if not rows:
        return {}
    groups: dict[str, list[tuple[float, float]]] = {}
    bearings: dict[str, list[float]] = {}
    speeds: list[float] = []
    for row in rows:
        stamp = float(row["capture_timestamp"]) + base_lag - lag_s
        snapshot = broker.get_snapshot_at(stamp, "interpolate")
        if snapshot is None or snapshot.north_m is None or snapshot.east_m is None:
            continue
        # ⚠ 用**速度方向**分类，不要用偏航角：东西向航线时 yaw≈±90°，cos(yaw)≈0 会退化
        if snapshot.vy_m_s is None:
            continue
        direction = "+east" if snapshot.vy_m_s >= 0 else "-east"
        if snapshot.vx_m_s is not None:
            speeds.append(math.hypot(snapshot.vx_m_s, snapshot.vy_m_s))
            # 航迹角（0=北、90=东），用来判断两遍是不是真的**同一条轴**
            bearings.setdefault(direction, []).append(
                math.degrees(math.atan2(snapshot.vy_m_s, snapshot.vx_m_s)) % 360.0
            )
        groups.setdefault(direction, []).append((snapshot.north_m, snapshot.east_m))
    means: dict[str, tuple[float, float]] = {}
    for direction, points in groups.items():
        if not points:
            continue
        means[direction] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    if "+east" not in means or "-east" not in means:
        return {"directions": sorted(means), "note": "只有一个方向的观测，无法做反向诊断"}

    # ⚠⚠ **同轴校验**：Δeast/(2v) 这个式子只有在两遍**沿同一条轴反向**通过时才成立。
    # 只按 vy 的符号分组是不够的——西南向对角线（vx<0, vy<0）和东向扫掠线（vx≈0, vy>0）
    # 会被分到两个组里，但它们相差几十度，Δeast 主要反映的是**切向差**而不是延时。
    # 不同航迹轴会把空间几何差误认为延迟，甚至使估计符号相反。
    # 所以这里显式校验两组的平均航迹角：必须都接近正东/正西（±TOL），且彼此相反。
    mean_bearing = {direction: _circular_mean_deg(values) for direction, values in bearings.items()}
    axis_note = _axis_check_note(mean_bearing)
    if axis_note is not None:
        return {
            "directions": sorted(means),
            "bearings_deg": {k: round(v, 1) for k, v in sorted(mean_bearing.items())},
            "note": axis_note,
        }
    delta_east = means["+east"][1] - means["-east"][1]
    speed = sum(speeds) / len(speeds) if speeds else float("nan")
    return {
        "directions": sorted(means),
        "bearings_deg": {k: round(v, 1) for k, v in sorted(mean_bearing.items())},
        "delta_east_m": round(delta_east, 3),
        "speed_m_s": round(speed, 2) if not math.isnan(speed) else None,
        "implied_lag_s": round(delta_east / (2.0 * speed), 3)
        if speed and not math.isnan(speed)
        else None,
        "note": (
            "implied_lag = Δeast / (2·v)：两向观测的坐标差里，延时贡献为 2·v·Δt。"
            "两遍已校验**同轴反向**（航迹角 "
            f"{mean_bearing['+east']:.0f}°/{mean_bearing['-east']:.0f}°，"
            f"容差 ±{REVERSE_AXIS_TOL_DEG:.0f}°），所以这个值可用；"
            "但它只反映**这一座井**这一次架次，别当标定值——"
            "标定请看 lag_calibration（留一验证）。"
        ),
    }


@dataclass(slots=True)
class PerceptionSummary:
    """从真实工作线程快照生成的报告数据。"""

    frames: int
    detections: int
    read_codes: int
    last_error: str | None
    ocr_submitted: int
    ocr_results: int
    ocr_with_code: int
    ocr_deduped: int
    ocr_queued: int

    @classmethod
    def from_worker(cls, worker: Any) -> PerceptionSummary:
        """读取收尾时工作线程的完整统计，避免依赖 target source 的旧接口。"""
        stats = worker.stats
        return cls(
            frames=int(stats.frames),
            detections=int(stats.detections),
            read_codes=int(stats.with_code),
            last_error=stats.last_error,
            ocr_submitted=int(stats.submitted),
            ocr_results=int(stats.results),
            ocr_with_code=int(stats.with_code),
            ocr_deduped=int(stats.deduped),
            ocr_queued=int(worker.queued_results),
        )

    def snapshot(self) -> dict[str, int]:
        """返回 OCR 队列与结果计数。"""
        return {
            "ocr_submitted": self.ocr_submitted,
            "ocr_results": self.ocr_results,
            "ocr_with_code": self.ocr_with_code,
            "ocr_deduped": self.ocr_deduped,
            "ocr_queued": self.ocr_queued,
        }


@dataclass(slots=True)
class _RunOutcome:
    """一次测试跑完后的汇总：报告阶段只依赖这些（都在 finally 里取好）。"""

    work: Path
    sdp: Path
    calib: Path
    lag_s: float
    flown_s: float | None
    result: Any
    stats: dict[str, int]
    perception: PerceptionSummary
    video_stats: Any
    writer_stats: tuple[int, int]
    detail: list[dict[str, Any]]
    broker: TelemetryBroker
    flight_dir: Path | None
    runner_stats: dict[str, Any] | None = None
    perf: list[dict[str, Any]] | None = None
    #: EKF 原点 ``(lon, lat, alt)``——从 :attr:`MissionRunner.origin` 取（见 :func:`_origin_lla`）
    origin: tuple[float, float, float] | None = None
    #: 目标跟踪器（报告要拿它的**点集**按更大的聚类半径重聚，见 :func:`_analyze_with_retry`）
    tracker: Any = None
    #: 装配好的配置（重聚要用它的 ``targeting``）
    config: Any = None
    scene: ReconScene | None = None


def _analysis_broker(outcome: _RunOutcome) -> TelemetryBroker:
    """报告分析用的 broker：**优先从飞行记录重建**，读不出来才退回活的 broker。

    ⚠ 不能直接用活的 broker 取原点：会话结束的 ``_run_session`` finally 会调
    :meth:`~airdrop.telemetry.broker.TelemetryBroker.reset`，快照被换成**全新的空
    快照**、原点字段全成 ``None``；再把它当 ``ref`` 传进 ``wgs84_to_ned`` 就是
    ``TypeError: float() argument must be ... not 'NoneType'``——任务好好跑完 DONE，
    报告却在最后一步崩掉、``report.json`` 从未写出来（2026-09-30 实测，退出码 1）。

    同一份 :func:`lag_sweep` **本来就是从记录重建** broker 的，所以这里用同一个来源
    拿原点，两处口径一致；顺带让报告变成"飞行目录 + 检测明细"的函数——录完就能
    反复重算，不依赖进程还活着。分辨率与活 broker 相同（记录器 10Hz、活历史 0.1s
    节流）。没有飞行记录（``--no-record``）时退回活的 broker。
    """
    if outcome.flight_dir is not None:
        from airdrop import load_broker_from_log

        try:
            broker, pacer = load_broker_from_log(outcome.flight_dir, history_maxlen=0)
        except Exception:
            # 读不出来就退回活 broker，不让报告崩在这里
            LOGGER.warning(
                "从 %s 重建遥测失败，退回活的 broker 取原点", outcome.flight_dir, exc_info=True
            )
        else:
            # 全量回填：离线分析要的是"任意历史时刻都查得到"，不是回放那套按帧推进
            # （publish_until 传到 +inf 就等于全部灌入；finished_to 是属性不是方法）
            pacer.publish_until(float("inf"))
            return broker
    return outcome.broker


def _origin_lla(outcome: _RunOutcome, broker: TelemetryBroker) -> tuple[float, float, float]:
    """报告要用的 EKF 原点 ``(lon, lat, alt)``。

    三级取法，任何一级拿到就返回：① 从记录重建的 broker 快照（与 lag 扫描同源）；
    ② :attr:`MissionRunner.origin`——INIT 期间从控制器取到的权威副本，存在 runner
    自己的字段上，不受 broker 重置影响；③ 都没有就**显式报错**，不让 ``None`` 一路
    传到 georef 深处变成看不懂的 ``TypeError``。
    """
    snapshot = broker.get_snapshot()
    lon = snapshot.origin_longitude_deg
    lat = snapshot.origin_latitude_deg
    alt = snapshot.origin_altitude_m
    if lon is not None and lat is not None and alt is not None:
        return (lon, lat, alt)
    if outcome.origin is not None:
        return outcome.origin
    raise RuntimeError(
        "拿不到 EKF 原点，无法把天井真值换算到本地 NED：记录里没有原点遥测，runner 也没记到 origin"
    )


def _performance_summary(samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    """把监控样本压成一份汇总（平均吞吐 / 峰值积压 / 峰值内存）。"""
    if not samples:
        return {}
    first, last = samples[0], samples[-1]
    span = max(last["t_s"] - first["t_s"], 1e-6)
    frames = last["frames"] - first["frames"]
    submitted = last["submitted"] - first["submitted"]
    results = last["results"] - first["results"]
    deduped = last["deduped"] - first["deduped"]

    def peak(key: str) -> float | None:
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        return None if not values else max(values)

    return {
        "samples": len(samples),
        "span_s": round(span, 1),
        "detection_fps_avg": round(frames / span, 1),
        "ocr_in_per_s_avg": round(submitted / span, 1),
        "ocr_out_per_s_avg": round(results / span, 1),
        "dedupe_per_s_avg": round(deduped / span, 1),
        "ocr_requests": submitted,
        "ocr_deduped": deduped,
        "pending_max": peak("pending"),
        "queued_max": peak("queued"),
        "points_max": peak("points"),
        "lag_frames_max": peak("lag_frames"),
        "rss_mb_max": peak("rss_mb"),
        "children_mb_max": peak("children_mb"),
    }


def _state_durations(
    transitions: list[dict[str, Any]], ended_at: float | None
) -> list[dict[str, Any]]:
    """由状态转移时间戳算每段停留时长（最后一个状态算到 ``ended_at``）。"""
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(transitions):
        start = float(record["timestamp"])
        if index + 1 < len(transitions):
            end = float(transitions[index + 1]["timestamp"])
        else:
            end = ended_at if ended_at is not None else start
        rows.append({"state": str(record["to_state"]), "seconds": round(max(end - start, 0.0), 1)})
    return rows


#: 聚类半径不够时**重试**用的半径（米）。默认 1.5 装不下时放宽到 3.0。
#: 3.0 相对 20 m 的靶标间距仍有 6.7 倍余量，不会把相邻靶标并成一个。
RETRY_EPS_M = 3.0


def _analyze_with_retry(outcome: _RunOutcome) -> tuple[Any, float]:
    """按配置的半径聚类；**带编号的类不足靶标数**就用 :data:`RETRY_EPS_M` 再聚一次。

    为什么要重试：同一目标的重复观测散布会随像素在画幅中的位置、观测角度和航迹变化。
    当初始半径下带编号的聚类数量不足时，再以较大的备选半径聚类一次；重试是兜底，
    用于目标布局或观测几何变化导致少量观测未能聚合的情况。

    ⚠ 重试**必须把实际用的半径记进报告**（``report["eps_m"]``）：否则"三个靶标误差都
    很小"会让人以为 1.5 m 就够了，而其实是放宽到 3.0 才聚出来的。
    """
    from dataclasses import replace as _replace

    from airdrop import analyze

    points = getattr(outcome.tracker, "points", None)
    if points is None:
        return outcome.result, float(outcome.config.targeting.eps_m)
    base = outcome.config.targeting
    first = analyze(tuple(points), base)
    labeled = sum(1 for cluster in first.clusters if cluster.code is not None)
    if labeled >= len((getattr(outcome, "scene", None) or _DEFAULT_SCENE).wells_enu):
        return first, float(base.eps_m)
    widened = analyze(tuple(points), _replace(base, eps_m=RETRY_EPS_M))
    labeled_wide = sum(1 for cluster in widened.clusters if cluster.code is not None)
    if labeled_wide <= labeled:
        return first, float(base.eps_m)  # 放宽也没多聚出来，用原结果（如实）
    return widened, RETRY_EPS_M


def _observation_errors(
    detections: list[dict[str, Any]],
    *,
    origin_lla: tuple[float, float, float],
    calib_file: str,
    scene: ReconScene | None = None,
) -> dict[str, Any]:
    """逐观测落点误差的两个口径（全量 / 航向一致子集）。

    口径与为什么这么做见 :data:`HEADING_TOL_DEG`：偏差大的那批观测**不剔点**，两个口径
    都报出来——剔了就看不到"精度是被姿态估计吃掉的"这个事实，而那正是本工具要回答的问题。
    诊断失败不抛，返回 ``{"failed": ...}`` 让报告继续。
    """
    try:
        per_obs = _per_observation_errors(
            detections, origin_lla=origin_lla, calib_file=calib_file, scene=scene
        )
    except Exception as exc:  # noqa: BLE001 - 诊断失败不影响主结果
        LOGGER.warning("逐观测误差分析失败：%s", exc)
        return {"failed": str(exc)}
    consistent = [
        row["error_m"] for row in per_obs if (row["heading_dev_deg"] or 0.0) <= HEADING_TOL_DEG
    ]
    return {
        "tolerance_deg": HEADING_TOL_DEG,
        "all": _error_stats([row["error_m"] for row in per_obs]),
        "heading_consistent": _error_stats(consistent),
        "worst": sorted(per_obs, key=lambda row: -row["error_m"])[:8],
    }


def _report_heading_and_errors(
    detections: list[dict[str, Any]],
    *,
    origin_lla: tuple[float, float, float],
    calib_file: str,
    scene: ReconScene | None = None,
) -> dict[str, Any]:
    """航向一致性 + 逐观测误差：算完打印、超阈告警，返回两块直接并进报告。

    合成一个函数是为了不把 :func:`_report` 的分支数顶破上限（它本来就有 15 个）。
    """
    observation_errors = _observation_errors(
        detections, origin_lla=origin_lla, calib_file=calib_file, scene=scene
    )
    heading = _heading_consistency(detections)
    if heading.get("n"):
        print(
            "航向一致性  : |航迹角−yaw| 中位 {:.1f}°/p90 {:.1f}°/最大 {:.1f}°，超 ±{:.0f}° 的 {}/{}（{:.0f}%）".format(
                heading["median_deg"],
                heading["p90_deg"],
                heading["max_deg"],
                HEADING_TOL_DEG,
                heading["over_tolerance"],
                heading["n"],
                heading["over_fraction"] * 100.0,
            )
        )
    if heading.get("over_tolerance"):
        LOGGER.warning(
            "有 %d/%d 个观测的航向偏差超过 ±%.0f°（中位 %.1f°、最大 %.1f°）："
            "落点误差是被姿态估计放大的，别当成本项目坐标解算的精度——"
            "看报告 observation_errors 的两个口径。",
            heading["over_tolerance"],
            heading["n"],
            HEADING_TOL_DEG,
            heading["median_deg"],
            heading["max_deg"],
        )
    all_stats = observation_errors.get("all") or {}
    ok_stats = observation_errors.get("heading_consistent") or {}
    if all_stats.get("n"):
        print(
            "逐观测误差  : 全量 中位 {:.2f}/均方根 {:.2f}/最大 {:.2f} m（<2m {:.0f}%）；"
            "航向一致子集 中位 {:.2f}/均方根 {:.2f}/最大 {:.2f} m（<2m {:.0f}%，n={}）".format(
                all_stats["median_m"],
                all_stats["rms_m"],
                all_stats["max_m"],
                all_stats["under_2m_fraction"] * 100.0,
                ok_stats.get("median_m", float("nan")),
                ok_stats.get("rms_m", float("nan")),
                ok_stats.get("max_m", float("nan")),
                (ok_stats.get("under_2m_fraction") or 0.0) * 100.0,
                ok_stats.get("n", 0),
            )
        )
    return {"observation_errors": observation_errors, "heading_consistency": heading}
