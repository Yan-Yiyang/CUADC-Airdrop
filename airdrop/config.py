'AirDrop 3.0 全局配置——全部参数的唯一集中处。\n\n约定\n----\n- 全部为 frozen dataclass（可用 ``dataclasses.replace`` 派生变体）；\n- **不使用命令行参数**：要改参数就改这里，或在构造 :class:`Config` 时显式传；\n- 每个字段注明单位与来源（实测值 / 标定产出 / 占位待微调）；\n- :class:`Config` 是装配点，各模块从它取自己的配置片段，模块内部不散落魔法数字；\n- ``Config.validated()`` 做取值域检查（异常在启动前暴露，而不是飞到一半才发现）。'

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .video.source import HM30_DEFAULT_RTSP, VideoConfig

__all__ = [
    "AlignConfig",
    "BallisticsConfig",
    "CameraConfig",
    "Config",
    "DropConfig",
    "GripperConfig",
    "GroundConfig",
    "OverflyConfig",
    "PerceptionConfig",
    "RecordConfig",
    "RoutesConfig",
    "TargetingConfig",
    "TelemetryConfig",
    "Waypoint",
]

# ---------------------------------------------------------------- 取值域
PERCEPTION_MODES = frozenset({"ocr", "cls12"})
TARGET_COLORS = frozenset({"blue", "red"})
SELECTION_RULES = frozenset({"median", "max"})
ALIGN_TIMEOUT_ACTIONS = frozenset({"drop"})
WIND_SOURCES = frozenset({"telemetry", "zero"})


# ---------------------------------------------------------------- 遥测与控制
@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """MAVSDK 连接、遥测订阅速率与历史缓冲。

    速率字段是**请求给飞控的上发频率**。它们直接决定帧-遥测内插的上限精度：
    固定翼 15~20 m/s 平飞时，位置流 1Hz 的内插误差是米级，10Hz 降到分米级；
    姿态流变化更快，20m 高度下 1° 姿态误差 ≈0.35m 地面误差，所以要求 ≥30Hz。
    实际速率在 P2 阶段经 ``set_rate_*`` 下发（当前实现尚未设置）。
    """

    system_address: str = "udpin://0.0.0.0:14540"
    connect_timeout: float = 30.0
    wait_for_health: bool = False
    health_timeout: float = 30.0
    require_health_for_actions: bool = False
    origin_refresh_interval: float = 5.0
    reconnect: bool = True
    reconnect_delay: float = 5.0

    # 遥测订阅速率（Hz）
    position_rate_hz: float = 10.0
    position_velocity_ned_rate_hz: float = 10.0
    attitude_rate_hz: float = 30.0

    # 历史缓冲：按固定间隔入库（0 = 每次更新都入库）
    history_interval: float = 0.1
    history_maxlen: int = 1200


# ---------------------------------------------------------------- 视频接收
@dataclass(frozen=True, slots=True)
class AlignConfig:
    """帧-遥测对齐。

    ``max_wait``：拍摄时刻晚于最新遥测时最多等待多久（秒）——等到了就内插，
    超时按 ``on_timeout`` 处理（"drop" = 丢弃该帧）。
    """

    mode: str = "interpolate"
    max_wait: float = 1.0
    on_timeout: str = "drop"
    max_extrapolation: float | None = None


# ---------------------------------------------------------------- 视频处理
@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    '视频处理（YOLO + RapidOCR）。**起飞前按任务选模式**。\n\n    - ``mode="ocr"``：YOLO 检出目标框 → 裁剪 → RapidOCR 读印刷数字（编码 00–99）；\n    - ``mode="cls12"``：YOLO 12 类直出编号（1–12），无 OCR 环节。'

    mode: str = "ocr"
    weights_ocr: str = "weights/ocr_best.pt"
    weights_cls12: str = "weights/cls12_best.pt"
    device: str = "0"
    conf_threshold: float = 0.25

    # OCR 目标形态（旧版 实战规格：五边形=正方形+等边三角形，边长 1m）
    target_color: str = "blue"          # blue / red
    target_side_length_m: float = 1.0
    min_side_px: float = 10.0
    max_side_px: float = 400.0
    ocr_conf_threshold: float = 0.6
    ocr_workers: int = 2
    ocr_queue_size: int = 500
    ocr_dedupe_s: float = 0.8           # 同一目标送检最小时间间隔（0 = 不去重）
    ocr_dedupe_px: float = 30.0         # 同一目标送检最小中心位移（px）

    # 排故素材
    save_crops: bool = False
    crop_dir: str = "log/target"
    max_crops: int = 5000

    # 处理落后监控（全帧处理，不抽帧）
    lag_warn_frames: int = 900


# ---------------------------------------------------------------- 坐标处理
@dataclass(frozen=True, slots=True)
class CameraConfig:
    """相机模型（内参 + 相机→机身外参），由标定流程产出后加载。"""

    calib_file: str = "camera_calib.json"
    undistort: bool = True


@dataclass(frozen=True, slots=True)
class GroundConfig:
    """地面假设。

    平地假设下目标地面高程由**地面点 GPS 与 NED 原点的高差**换算：
    ``ground_z = 原点海拔 - 地面点海拔``（NED 的 z 向下为正）。
    """

    flat: bool = True
    ground_point_lat: float | None = None
    ground_point_lon: float | None = None
    ground_point_alt: float | None = None


# ---------------------------------------------------------------- 目标统计
@dataclass(frozen=True, slots=True)
class TargetingConfig:
    """聚类与结果筛选（scikit-learn DBSCAN）。

    ``selection_rule``：跨聚类标签取中位数（``"median"``）或最大值（``"max"``）对应的类，
    **起飞前按任务选定**，代码不做隐式回退。结果坐标 = 类内成员坐标均值。
    """

    eps_m: float = 0.75                 # 聚类半径，预期目标间距下 0.5~1.0
    min_samples: int = 2                # 噪声点（label=-1）剔除
    selection_rule: str = "median"
    weight_by_confidence: bool = False  # 开时按置信度加权（聚类与均值同时加权）
    require_label: bool = True          # 只考虑有有效编码的类


# ---------------------------------------------------------------- 弹道与投放
@dataclass(frozen=True, slots=True)
class BallisticsConfig:
    """二次阻力弹道（RK 数值积分）。参数为 350ml 水瓶占位值，**需投放试验微调**。"""

    mass_kg: float = 0.365
    drag_coefficient: float = 0.6
    cross_area_m2: float = 0.004
    gravity: float = 9.80665
    rk_dt: float = 0.005
    air_density_isa: bool = True        # 空气密度按 ISA 标准大气随高度变化
    wind_source: str = "telemetry"      # telemetry（PX4 风估计）/ zero


@dataclass(frozen=True, slots=True)
class DropConfig:
    """投放判据：预测落点距目标 ≤ radius 即投；飞过点后保底投放。"""

    radius_m: float = 2.0
    delay_s: float = 0.0                # 指令→离机延迟（状态前推补偿）
    force_after_pass: bool = True
    evaluation_hz: float = 20.0


@dataclass(frozen=True, slots=True)
class OverflyConfig:
    """飞掠航线几何（按空域配置；段长需实验验证）。"""

    heading_deg: float = 0.0
    altitude_m: float = 20.0
    leg_length_m: float = 200.0


@dataclass(frozen=True, slots=True)
class GripperConfig:
    """MAVSDK gripper 投放（PX4 侧需预先配置 gripper 输出）。"""

    enabled: bool = True
    release_settle_s: float = 0.5


# ---------------------------------------------------------------- 航线
@dataclass(frozen=True, slots=True)
class Waypoint:
    """任务航点（WGS84）。``acceptance_radius_m`` 为 0 表示用飞控默认。"""

    lat: float
    lon: float
    alt_m: float
    acceptance_radius_m: float = 0.0


@dataclass(frozen=True, slots=True)
class RoutesConfig:
    """预设航线：侦查航线（含起飞项）、备用投放点、降落航线。"""

    recon_route: tuple[Waypoint, ...] = ()
    backup_point: Waypoint | None = None
    landing_route: tuple[Waypoint, ...] = ()


# ---------------------------------------------------------------- 记录与回放
@dataclass(frozen=True, slots=True)
class RecordConfig:
    """飞行记录五件套（文本日志/遥测/检测/事件/视频帧）与回放。"""

    dir: str = "flights"
    video: bool = True                  # 落盘缓冲内的 jpeg（零重编码）
    telemetry_hz: float = 10.0
    event_eval_hz: float = 5.0          # 释放评估摘要频率


# ---------------------------------------------------------------- 顶层装配
@dataclass(frozen=True, slots=True)
class Config:
    """整个系统的配置装配点。"""

    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    video: VideoConfig = field(
        default_factory=lambda: VideoConfig(url=HM30_DEFAULT_RTSP)
    )
    align: AlignConfig = field(default_factory=AlignConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    targeting: TargetingConfig = field(default_factory=TargetingConfig)
    ballistics: BallisticsConfig = field(default_factory=BallisticsConfig)
    drop: DropConfig = field(default_factory=DropConfig)
    overfly: OverflyConfig = field(default_factory=OverflyConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    routes: RoutesConfig = field(default_factory=RoutesConfig)
    record: RecordConfig = field(default_factory=RecordConfig)

    def validated(self) -> "Config":
        """取值域检查：不合法就抛错，别等飞到一半才发现。"""
        if self.perception.mode not in PERCEPTION_MODES:
            raise ValueError(
                f"perception.mode 只支持 {sorted(PERCEPTION_MODES)}: "
                f"{self.perception.mode!r}"
            )
        if self.perception.target_color not in TARGET_COLORS:
            raise ValueError(
                f"perception.target_color 只支持 {sorted(TARGET_COLORS)}: "
                f"{self.perception.target_color!r}"
            )
        if self.targeting.selection_rule not in SELECTION_RULES:
            raise ValueError(
                f"targeting.selection_rule 只支持 {sorted(SELECTION_RULES)}: "
                f"{self.targeting.selection_rule!r}"
            )
        if self.align.on_timeout not in ALIGN_TIMEOUT_ACTIONS:
            raise ValueError(
                f"align.on_timeout 只支持 {sorted(ALIGN_TIMEOUT_ACTIONS)}: "
                f"{self.align.on_timeout!r}"
            )
        if self.ballistics.wind_source not in WIND_SOURCES:
            raise ValueError(
                f"ballistics.wind_source 只支持 {sorted(WIND_SOURCES)}: "
                f"{self.ballistics.wind_source!r}"
            )
        if self.perception.ocr_workers < 1:
            raise ValueError(f"perception.ocr_workers 至少为 1: {self.perception.ocr_workers}")
        if self.targeting.eps_m <= 0:
            raise ValueError(f"targeting.eps_m 必须为正: {self.targeting.eps_m}")
        if self.drop.radius_m <= 0:
            raise ValueError(f"drop.radius_m 必须为正: {self.drop.radius_m}")
        self.video.validated()
        return self

    def replace(self, **changes) -> "Config":
        """派生新配置（浅层替换顶层片段）。"""
        return replace(self, **changes)
