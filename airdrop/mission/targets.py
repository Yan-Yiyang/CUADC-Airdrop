"""感知结果 → 目标点 → 统计结果：全链路里"坐标解算"这一段（P10）。

数据流
------
``Detection``（像素 + 编号 + 拍摄时刻的遥测） → :func:`~airdrop.georef.pixel_to_ned`
→ :class:`~airdrop.targeting.TargetPoint` → :func:`~airdrop.targeting.analyze`
→ 唯一结果坐标 → :class:`~airdrop.mission.MissionRunner` 的飞掠航线。

为什么单独一个模块
------------------
georef 是纯几何、targeting 是纯统计，中间那段"用哪一帧的位姿、地面高程怎么算、
解算不出来的点怎么办"没有归属：塞进 perception 会让视频模块依赖坐标解算，塞进
georef 又会让纯几何层依赖目标语义。所以放在这里，并且实飞与回放共用——回放的
遥测来自日志、走的是同一条路，换参数重跑就能直接对比（计划第 7 章）。

两个必须记住的约定
------------------
1. 像素与内参必须是同一张图上的。检测器若已经做过去畸变（``DetectorConfig.
   camera_matrix`` 非空时逐帧 remap），像素就在纠正后的图上，这里再纠正一次
   就等于纠正两遍。默认 ``undistort=None`` 表示自动判断：读 ``Detection.extra``
   里检测器留下的 ``undistorted`` 标记（:meth:`~airdrop.perception.Detector.detect`
   会写），检测器纠正过就不再纠正。
2. 边长交叉验证**默认关掉**（``side_check_tolerance=0``）。它是"用已知边长（1m）独立
   估深度、再与地面求交互校"的*诊断*，只在**回放优化**时选择性打开（例如定位标定/几何
   问题：标定不对时它会大面积不通过）；正式流程里大倾角帧天然会超门限（实测架次
   某架次：35 个观测里 15 次），它既不影响解算也不该刷日志。打开后仍只记
   事件与计数、**不剔点**——误判一个真实观测的代价比多一个野点大，而 targeting 的
   DBSCAN 本来就靠"看到几十次"筛野点；要真的按门限剔点再加
   ``reject_on_side_mismatch=True``。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..georef import CameraModel, cross_check_by_side, pixel_to_ned
from ..perception import Detection, PerceptionWorker
from ..targeting import TargetingResult, TargetPoint, analyze

LOGGER = logging.getLogger(__name__)

__all__ = ["PerceptionTargetSource", "TargetTracker"]

#: 边长交叉验证的默认门限（**0 = 关掉**，见模块 docstring 第 2 条）。
#: 它是**回放优化**用的诊断：要打开就显式给门限（如 0.25），例如回放入口的
#: ``--side-check 0.25``；正式流程保持关掉（大倾角帧必超门限，只会刷日志）。
DEFAULT_SIDE_TOLERANCE = 0.0


@dataclass(slots=True)
class TargetTracker:
    """把 ``Detection`` 流攒成目标点，并随时给出统计结果。

    用法::

        tracker = TargetTracker(config, camera=camera)      # 实飞与回放同一份代码
        tracker.extend(worker.drain_results())              # 每拍抽干结果队列
        result = tracker.result()                           # == analyze(points, cfg.targeting)

    ``on_event`` 的签名与项目里其它模块一致（``(kind, data)``，接 recorder 就是
    ``lambda kind, data: recorder.events.emit(kind, **data)``），事件写盘失败不影响解算。
    """

    config: Config
    camera: CameraModel
    #: ``None`` = 按检测器的 ``undistorted`` 标记自动判断（见模块 docstring）
    undistort: bool | None = None
    side_check_tolerance: float = DEFAULT_SIDE_TOLERANCE
    reject_on_side_mismatch: bool = False
    on_event: Callable[[str, dict[str, Any]], None] | None = None

    _points: list[TargetPoint] = field(default_factory=list, init=False)
    _no_fix: int = field(default=0, init=False)  # 遥测/几何不足以解算
    _side_mismatch: int = field(default=0, init=False)  # 边长法与求交差异超门限
    _undistort_warned: bool = field(default=False, init=False)
    _ground_warned: bool = field(default=False, init=False)
    _attitude_missing: int = field(default=0, init=False)

    # ------------------------------------------------------------------
    # 攒点
    # ------------------------------------------------------------------
    def add(self, detection: Detection) -> TargetPoint | None:
        """一条检测 → 一个目标点；解算不出来时返回 None（并计数）。"""
        snapshot = detection.telemetry
        position = _position_of(snapshot)
        if position is None:
            self._no_fix += 1
            return None
        attitude = _attitude_of(snapshot)
        if attitude is None:
            self._attitude_missing += 1
            if self._attitude_missing == 1:
                LOGGER.warning(
                    "帧 #%d 的遥测里既没有四元数也没有欧拉角，无法解算坐标",
                    detection.frame_index,
                )
            return None

        ground_z = self.ground_z(snapshot)
        fix = pixel_to_ned(
            detection.pixel,
            camera=self.camera,
            ground_z=ground_z,
            position_ned=position,
            undistort=self._should_undistort(detection),
            **attitude,
        )
        if not fix.ok or fix.ned is None:
            self._no_fix += 1
            LOGGER.debug(
                "帧 #%d 的像素 %s 解算失败：%s", detection.frame_index, detection.pixel, fix.reason
            )
            return None

        if detection.side_px > 0 and self.side_check_tolerance > 0:
            side_ok = self._side_check_ok(detection, fix.depth_m, attitude)
            # 边长交叉验证默认只当诊断（见模块 docstring）：不剔点，只计数/记事件；
            # reject_on_side_mismatch=True 时才真的丢弃这次观测。
            if not side_ok and self.reject_on_side_mismatch:
                return None

        point = TargetPoint(
            north_m=fix.ned[0],
            east_m=fix.ned[1],
            down_m=fix.ned[2],
            capture_timestamp=detection.capture_timestamp,
            frame_index=detection.frame_index,
            code=detection.code,
            confidence=detection.confidence,
        )
        self._points.append(point)
        return point

    def extend(self, detections: Iterable[Detection]) -> int:
        """批量攒点（``worker.drain_results()`` 的返回值直接传入）；返回成功条数。"""
        added = 0
        for detection in detections:
            if self.add(detection) is not None:
                added += 1
        return added

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------
    def result(self) -> TargetingResult:
        """当前统计结果（``analyze(points, config.targeting)``）——实飞与回放同一条路。"""
        return analyze(tuple(self._points), self.config.targeting)

    @property
    def points(self) -> tuple[TargetPoint, ...]:
        return tuple(self._points)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "points": len(self._points),
            "labeled": sum(1 for point in self._points if point.labeled),
            "no_fix": self._no_fix,
            "attitude_missing": self._attitude_missing,
            "side_mismatch": self._side_mismatch,
        }

    def reset(self) -> None:
        """清空（换架次/换目标时用）。"""
        self._points.clear()
        self._no_fix = 0
        self._side_mismatch = 0
        self._attitude_missing = 0

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def ground_z(self, snapshot: Any) -> float:
        """地面在 NED 下的 z；地面点参数不全时按 0（原点高度面）并只警告一次。"""
        ground = self.config.ground.ground_z(snapshot.origin_altitude_m)
        if ground is None:
            if not self._ground_warned:
                self._ground_warned = True
                LOGGER.warning(
                    "地面点参数不全（GroundConfig），坐标解算按 ground_z=0（原点高度面）——"
                    "目标点会带上原点与地面点的高差"
                )
            return 0.0
        return float(ground)

    def _should_undistort(self, detection: Detection) -> bool:
        if self.undistort is not None:
            return self.undistort
        marked = detection.extra.get("undistorted")
        if marked is None:
            if not self._undistort_warned:
                self._undistort_warned = True
                LOGGER.warning(
                    "检测结果里没有去畸变标记（Detection.extra['undistorted']），"
                    "按“检测器已去畸变”处理——若检测器没去畸变，坐标会带畸变误差"
                )
            return False
        return not bool(marked)

    def _side_check_ok(
        self, detection: Detection, depth_m: float, attitude: dict[str, Any]
    ) -> bool:
        """边长法独立估深度，与地面求交互校；超门限记事件（默认不剔点）。"""
        check = cross_check_by_side(
            side_px=detection.side_px,
            side_m=self.config.perception.target_side_length_m,
            depth_by_intersection_m=depth_m,
            camera=self.camera,
            tolerance=self.side_check_tolerance,
            **attitude,
        )
        if check.ok:
            return True
        self._side_mismatch += 1
        self._emit(
            "side_check",
            {
                "frame_index": detection.frame_index,
                "capture_timestamp": detection.capture_timestamp,
                "code": detection.code,
                "side_px": round(float(detection.side_px), 2),
                "depth_by_side_m": round(float(check.depth_by_side_m), 2),
                "depth_by_intersection_m": round(float(check.depth_by_intersection_m), 2),
                "relative_error": round(float(check.relative_error), 4)
                if math.isfinite(float(check.relative_error))
                else None,
                "reason": check.reason,
                "rejected": bool(self.reject_on_side_mismatch),
            },
        )
        LOGGER.info(
            "帧 #%d 边长法互校不通过：%s（累计 %d 次）",
            detection.frame_index,
            check.reason,
            self._side_mismatch,
        )
        return False

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, data)
        except Exception:
            LOGGER.exception("写入解算事件失败（解算继续）")


@dataclass(slots=True)
class PerceptionTargetSource:
    """把"感知工作线程 + 目标跟踪"包成 :class:`MissionRunner` 要的两个回调。

    ``MissionRunner(target_result=source.result, target_busy=source.busy)`` 即可：
    实飞时 ``worker`` 吃的是图传缓冲，回放时吃的是回放缓冲，其余代码一模一样。

    ``on_detection``：每条 :class:`~airdrop.perception.models.Detection` 在**进 tracker
    之前**会被回调一次（异常只记日志、不影响解算）——生产链路用它把检出写进飞行记录的
    ``detections.jsonl``（``FlightRecorder.detections.append``），否则那个文件永远是空的。
    """

    worker: PerceptionWorker
    tracker: TargetTracker
    on_detection: Callable[[Detection], None] | None = None

    def pump(self) -> int:
        """抽干结果队列并解算（每拍调一次，或由 :meth:`result` 顺便调）。"""
        detections = self.worker.drain_results()
        if self.on_detection is not None:
            for detection in detections:
                try:
                    self.on_detection(detection)
                except Exception:
                    LOGGER.exception("检出记录回调失败（不影响解算）")
        return self.tracker.extend(detections)

    def result(self) -> TargetingResult:
        self.pump()
        return self.tracker.result()

    def busy(self) -> bool:
        """感知/解算还有活没干完吗。

        判据：缓冲里还有帧没处理（``lag_frames``）或送出去的 OCR 还没回来
        （``submitted > results``——请求/结果都不丢弃，这个差值就是真正的未完成数）。
        刻意不包含"结果还在队列里没抽干"——那由 :meth:`pump` 负责，
        而 :meth:`result` 每次都会先抽一遍。
        """
        stats = self.worker.stats
        return self.worker.lag_frames > 0 or stats.submitted > stats.results


def _position_of(snapshot: Any) -> tuple[float, float, float] | None:
    values = (snapshot.north_m, snapshot.east_m, snapshot.down_m)
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _attitude_of(snapshot: Any) -> dict[str, Any] | None:
    """优先四元数（大机动下欧拉角有万向节问题，计划 4.4 明确要求）。"""
    quaternion = (
        snapshot.quaternion_w,
        snapshot.quaternion_x,
        snapshot.quaternion_y,
        snapshot.quaternion_z,
    )
    if all(value is not None for value in quaternion):
        return {"quaternion": tuple(float(value) for value in quaternion)}
    euler = (snapshot.roll_deg, snapshot.pitch_deg, snapshot.yaw_deg)
    if all(value is not None for value in euler):
        return {"euler_deg": tuple(float(value) for value in euler)}
    return None
