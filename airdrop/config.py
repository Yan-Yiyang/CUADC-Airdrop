"""CUADC固定翼无人机侦查与打击控制项目：全局配置——全部参数的唯一集中处。

约定
----
- 全部为 frozen dataclass（可用 ``dataclasses.replace`` 派生变体）；
- **不使用命令行参数**：要改参数就改这里，或在构造 :class:`Config` 时显式传；
- 每个字段注明单位与来源（实测值 / 标定产出 / 占位待微调）；
- :class:`Config` 是装配点，各模块从它取自己的配置片段，模块内部不散落魔法数字；
- ``Config.validated()`` 做取值域检查（异常在启动前暴露，而不是飞到一半才发现）。

字段来源与现状
--------------
- ``video.telemetry_lag``：ffmpeg 后端实测 ≈0.15s；**标定流程步骤二会给出实测值并覆盖**；
- ``ballistics``：350ml 矿泉水瓶的占位参数，**需投放试验微调**；
- ``overfly.leg_length_m``：保证平飞的段长，**需实验验证**；
- ``perception.mode`` / ``targeting.selection_rule``：**起飞前按任务选定**。
"""

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
    "MissionConfig",
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
ABORT_ACTIONS = frozenset({"hold", "rtl", "none"})
#: 侦查航线由谁上传：operator = 操作手在 QGC 上传并启动（正式任务）；auto = 本包上传（自动测试）
RECON_UPLOAD_MODES = frozenset({"operator", "auto"})


# ---------------------------------------------------------------- 遥测与控制
@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """MAVSDK 连接、遥测订阅速率与历史缓冲。

    速率字段是**请求给飞控的上发频率**。它们直接决定帧-遥测内插的上限精度：
    固定翼 15~20 m/s 平飞时，位置流 1Hz 的内插误差是米级，10Hz 降到分米级；
    姿态流变化更快，20m 高度下 1° 姿态误差 ≈0.35m 地面误差，所以要求 ≥30Hz。
    速率由 :class:`~airdrop.telemetry.MavsdkThread` 在连接就绪后经 ``set_rate_*``
    统一下发（见其 ``_apply_telemetry_rates``）；单条速率失败只记日志，不中断连接。
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
    """视频处理（YOLO + RapidOCR）。**起飞前按任务选模式**。

    - ``mode="ocr"``：YOLO 检出目标框 → 裁剪 → RapidOCR 读印刷数字（编码 00–99）；
    - ``mode="cls12"``：YOLO 12 类直出编号（1–12），无 OCR 环节。

    三条要求：

    1. **``mode="cls12"`` 需要 12 类检测权重**（仓库目前只提供单类 ``target``
       权重；12 类权重就位前该模式编号恒为 1）。
    2. **检测权重用 ``best2.pt``**（单类 ``target``；仓库不带权重，
       用 :mod:`tools.fetch_models` 复制）。
    3. **``device`` 必须显式给定**：不依赖 ultralytics 的自动检测
       （当前实测会选中 ``cuda:0``，显式指定零成本地固定设备）。
    """

    mode: str = "ocr"
    # YOLO 检测权重（单类 target）
    model_path: str = "models/best2.pt"
    device: str = "0"
    conf_threshold: float = 0.25
    imgsz: int = 1280  # best2.pt 训练分辨率 640；实测 640/1280 检出数一致

    # OCR 引擎（TORCH，本地权重；P5 复测：medium ~97ms/帧 @RTX3060Laptop）
    models_dir: str = "models/ppocr"
    det_model: str = "PP-OCRv6_det_medium.pth"
    rec_model: str = "PP-OCRv6_rec_medium.pth"
    rec_keys: str = "ppocrv6_dict.txt"
    ocr_use_cuda: bool = True

    # OCR 目标形态（实战规格：五边形=正方形+等边三角形，边长 1m）
    target_color: str = "blue"  # blue / red
    target_side_length_m: float = 1.0
    min_side_px: float = 10.0
    max_side_px: float = 400.0
    ocr_conf_threshold: float = 0.6
    ocr_workers: int = 2
    #: OCR 积压告警阈值（请求/结果队列都无界、不丢弃；见 airdrop.perception.ocr_worker）
    ocr_queue_size: int = 500
    # 保留旧字段以兼容配置文件；当前 OCR 逐帧送检，这两个字段不再参与运行时判定。
    ocr_dedupe_s: float = 0.0
    ocr_dedupe_px: float = 0.0

    # 排故素材
    save_crops: bool = False
    crop_dir: str = "log/target"
    max_crops: int = 5000

    # 处理落后监控（全帧处理，不抽帧）
    lag_warn_frames: int = 900

    def ocr_engine_config(self):
        """装配 OCR 引擎参数（返回 :class:`~airdrop.perception.OcrEngineConfig`）。"""
        from .perception import OcrEngineConfig

        return OcrEngineConfig(
            model_dir=self.models_dir,
            det_model=self.det_model,
            rec_model=self.rec_model,
            rec_keys=self.rec_keys,
            use_cuda=self.ocr_use_cuda,
        )

    def to_pipeline_config(self):
        """装配 :class:`~airdrop.perception.PerceptionWorker` 需要的参数切片。"""
        from .perception import PerceptionConfigLike

        return PerceptionConfigLike(
            mode=self.mode,
            model_path=self.model_path,
            device=self.device,
            conf_threshold=self.conf_threshold,
            imgsz=self.imgsz,
            target_color=self.target_color,
            ocr_conf_threshold=self.ocr_conf_threshold,
            ocr_workers=self.ocr_workers,
            ocr_queue_size=self.ocr_queue_size,
            ocr_dedupe_s=self.ocr_dedupe_s,
            ocr_dedupe_px=self.ocr_dedupe_px,
            min_side_px=self.min_side_px,
            max_side_px=self.max_side_px,
            engine=self.ocr_engine_config(),
        )


# ---------------------------------------------------------------- 坐标处理
@dataclass(frozen=True, slots=True)
class CameraConfig:
    """相机模型（内参 + 相机→机身外参），由标定流程产出后加载。"""

    calib_file: str = "camera_calib.json"
    undistort: bool = True
    #: 标定文件缺失时的回退图像尺寸（用于粗估内参；正式任务必须先标定）
    fallback_width: int = 1280
    fallback_height: int = 720

    def load_model(self):
        """按 :attr:`calib_file` 装配 :class:`~airdrop.georef.CameraModel`。"""
        from .georef import load_camera_model

        return load_camera_model(
            self.calib_file,
            fallback_size=(self.fallback_width, self.fallback_height),
        )


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

    def ground_z(self, origin_alt_m: float | None) -> float | None:
        """地面在 NED 下的 z（向下为正）；缺参数时返回 None。

        ``ground_z = 原点海拔 − 地面点海拔``：三者同高时 ground_z=0，
        地面比原点高时 ground_z 为负（NED 里向上是负）。参数不全**不猜**，
        由调用方决定是否记事件日志。
        """
        if origin_alt_m is None or self.ground_point_alt is None:
            return None
        return float(origin_alt_m) - float(self.ground_point_alt)


# ---------------------------------------------------------------- 目标统计
@dataclass(frozen=True, slots=True)
class TargetingConfig:
    """聚类与结果筛选（scikit-learn DBSCAN）。

    ``selection_rule``：跨聚类标签取中位数（``"median"``，在**去重后**的编号上取，下中位）
    或最大值（``"max"``）对应的类，
    **起飞前按任务选定**，代码不做隐式回退。结果坐标 = 类内成员坐标均值。
    """

    eps_m: float = 0.75  # 聚类半径，预期目标间距下 0.5~1.0
    min_samples: int = 2  # 噪声点（label=-1）剔除
    selection_rule: str = "median"
    weight_by_confidence: bool = False  # 开时按置信度加权（聚类与均值同时加权）
    require_label: bool = True  # 只考虑有有效编码的类


# ---------------------------------------------------------------- 弹道与投放
@dataclass(frozen=True, slots=True)
class BallisticsConfig:
    """二次阻力弹道（RK 数值积分）。

    ⚠ **质量是称出来的**（台秤/天平实测，别当待估参数）；``drag_coefficient`` 与
    ``cross_area_m2`` 只以 ``Cd·A`` 的乘积影响弹道，单靠落点数据分不开——固定面积、
    **靠投放试验反演 Cd**（`tools/fit_ballistics.py`，见 :mod:`airdrop.ballistics.fit`）。

    空气密度以**真实海拔**为基准（飞行取 GPS 原点/地面点海拔，反演取投放记录里的
    原点，见 ``BallisticsModel.predict_impact`` 的 ``ground_altitude_m``）：
    ``air_density_isa=False``（默认）在投放海拔算一次常密度；``True`` 则逐级随海拔变化。
    """

    mass_kg: float = 0.365  # **实测称重**（占位值仅用于跑通链路）
    drag_coefficient: float = 0.6  # 投放试验反演（占位值）
    cross_area_m2: float = 0.004  # 几何量（卡尺/投影面积）
    gravity: float = 9.80665
    rk_dt: float = 0.005
    # False（默认）= 在投放（初始）海拔上算一次常密度、全弹道共用；
    # True = 逐级按"地面海拔 + 离地高度"求 ISA 密度（贵 26~30%，高海拔/大落差时更准）
    air_density_isa: bool = False
    wind_source: str = "telemetry"  # telemetry（PX4 风估计）/ zero


@dataclass(frozen=True, slots=True)
class DropConfig:
    """投放判据：预测落点距目标 ≤ radius 即投；飞过点后保底投放。"""

    radius_m: float = 2.0
    delay_s: float = 0.0  # 指令→离机延迟（状态前推补偿）
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
    """MAVSDK gripper 投放（PX4 侧需预先配置 gripper 输出）。

    ``instance`` 是 gripper 实例号（MAVSDK ``gripper.release(instance)``）；
    ``release_settle_s`` 是投放指令之后给伺服/挂架留的动作时间——这段时间里
    紧接着的下一条指令容易被飞控以 BUSY 拒绝，等一等更稳。
    """

    enabled: bool = True
    instance: int = 0
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
    """预设航线：侦查段与降落段，**每条腿二选一**（配置航点 或 QGC ``.plan``）。

    * ``recon_route`` / ``recon_plan``：侦查段。plan 模式下是操作手在 QGroundControl
      里画好、另存为 ``.plan`` 的完整航线，本包**原样使用**（不自动补起飞项）；
    * ``landing_route`` / ``land_plan``：飞掠段**之后**要飞的那一段（返航 + 降落）。
      plan 模式下飞掠段插在它前面，合成一条任务上传——"必须让飞控认可的降落剖面"
      因此由画航线的人（QGC）负责；
    * ``backup_point``：没有目标时的备用投放点（Q11）；
    * ``fw_land_angle_deg``：飞控参数 ``FW_LND_ANG``（默认 8.0）。降落预检用它算
      允许的最大下滑角（:func:`airdrop.mission.plan_file.check_fixed_wing_landing`），
      飞机改过这个参数就要同步，否则预检会与飞控的判断不一致。

    一条腿同时给两个来源会在 :meth:`Config.validated` 里报错：让其中一个**静默**优先
    比报错危险得多（飞的可能不是你以为的那条航线）。
    """

    recon_route: tuple[Waypoint, ...] = ()
    backup_point: Waypoint | None = None
    landing_route: tuple[Waypoint, ...] = ()
    #: QGC ``.plan`` 路径（相对仓库根或绝对路径）；空串 = 不用 plan
    recon_plan: str = ""
    land_plan: str = ""
    fw_land_angle_deg: float = 8.0


# ---------------------------------------------------------------- 任务编排
@dataclass(frozen=True, slots=True)
class MissionConfig:
    """任务状态机（计划 4.7）。

    时长类参数都是**上限兜底**：正常流程靠事件推进（mission 完成、出结果），
    只有卡住时才由这些上限把状态机推进到 `ABORT`——所以宁可给宽一点，
    也不要让它去替飞控做决定。

    ``abort_action``：进入 `ABORT` 时下的安全动作——``"hold"``（盘旋，默认，
    把飞机留在空中交给操作手）、``"rtl"``（返航）、``"none"``（什么都不下，
    用于演练/仿真）。链路本身断了时这条指令也发不出去，那种情况交给飞控自己的
    failsafe。

    ``takeoff_first`` / ``land_last``：侦查航线首个航点带**起飞项**（固定翼起飞
    由它承担，计划 4.1）、降落航线末航点带**降落项**。

    ``require_airborne``：``WAIT_AIRBORNE`` 那道门的逃生开关。默认为 ``True``
    （正式任务：必须等飞机真的在空中）；``False`` 时该状态**立即放行**，并记一条
    ``airborne_skipped`` 事件 + WARNING 日志——**只给地面演练/离线测试用**。
    """

    tick_hz: float = 20.0
    init_max_s: float = 30.0  # 等遥测 + NED 原点
    recon_max_s: float = 600.0  # 侦查航线（约 1 分钟）的兜底上限
    hold_process_max_s: float = 10.0  # HOLD_PROCESS 等结果的上限（计划 4.7）
    #: 至少等这么久再接受"没有结果"的判断——否则缓冲里还没被消费的帧会被当成"没看到目标"
    hold_process_min_s: float = 2.0
    land_max_s: float = 900.0  # 飞掠段 + 降落航线的兜底上限
    #: 上传并启动后，等多久确认"飞控真的进了任务模式"（见 MissionRunner 的启动确认）。
    #: 飞控可能拒绝模式切换却仍然回 ACK：不确认的话飞机一直盘旋，要等到状态超时才失败。
    mission_start_timeout_s: float = 20.0
    #: ``WAIT_AIRBORNE`` 等"飞机在空中"的上限与判据：
    #: 以遥测 ``in_air`` 为主，取不到时用 ``relative_altitude_m >= airborne_alt_m`` 兜底
    airborne_timeout_s: float = 1800.0
    airborne_alt_m: float = 5.0
    #: ``False`` = 不等飞机起飞就放行 ``WAIT_AIRBORNE``（**地面演练/测试开关**，正式任务必须为 True）
    require_airborne: bool = True
    #: 侦查航线谁上传：``"operator"``（默认，正式任务：操作手在 QGC 上传并启动）/ ``"auto"``
    recon_upload: str = "operator"
    #: 遥测超过这么久没更新就按链路故障处理（对齐侧 1s 就丢帧了，这里留两次余量）
    telemetry_stale_s: float = 5.0
    abort_action: str = "hold"
    takeoff_first: bool = True
    land_last: bool = True


# ---------------------------------------------------------------- 起飞前自检
@dataclass(frozen=True, slots=True)
class PreflightConfig:
    """起飞前自检（正式任务流程的第一步：**载入模型 → 视频自检**）。

    每一项都能单独关掉——测试/SITL 里常常没有相机、没有模型：

    * ``load_detector`` / ``load_ocr`` / ``load_camera``：分别载入 YOLO 权重、
      OCR 权重与字典、相机标定；
    * ``check_video``：视频自检（``video_probe_s`` 窗口内至少 ``video_min_frames`` 帧；
      已载入标定时还要核对画面分辨率与标定一致）；
    * ``max_s``：整个预检的总上限。

    ⚠ **关掉 ≠ 通过**：被关掉的检查会记一条 ``preflight`` 事件（``ok=None``），
    正式任务起飞前核一眼就知道哪几项没有把关。
    """

    load_detector: bool = True
    load_ocr: bool = True
    load_camera: bool = True
    check_video: bool = True
    #: 视频探测窗口（秒）与窗口内要求的最少帧数
    video_probe_s: float = 5.0
    video_min_frames: int = 10
    #: 预检总上限（超过就算失败，避免卡在载入上不动）
    max_s: float = 120.0


# ---------------------------------------------------------------- 记录与回放
@dataclass(frozen=True, slots=True)
class RecordConfig:
    """飞行记录（文本日志/遥测/检测/事件/投放/视频帧）与回放。"""

    dir: str = "flights"
    video: bool = True  # 落盘缓冲内的 jpeg（零重编码）
    telemetry_hz: float = 10.0
    event_eval_hz: float = 5.0  # 释放评估摘要频率


# ---------------------------------------------------------------- 顶层装配
@dataclass(frozen=True, slots=True)
class Config:
    """整个系统的配置装配点。"""

    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    video: VideoConfig = field(default_factory=lambda: VideoConfig(url=HM30_DEFAULT_RTSP))
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
    mission: MissionConfig = field(default_factory=MissionConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    record: RecordConfig = field(default_factory=RecordConfig)

    def validated(self) -> Config:  # noqa: PLR0912 - 逐片段的取值域检查，串行铺开最直观
        """取值域检查：不合法就抛错，别等飞到一半才发现。"""
        if self.perception.mode not in PERCEPTION_MODES:
            raise ValueError(
                f"perception.mode 只支持 {sorted(PERCEPTION_MODES)}: {self.perception.mode!r}"
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
        if self.perception.ocr_queue_size < 1:
            raise ValueError(
                f"perception.ocr_queue_size 至少为 1: {self.perception.ocr_queue_size}"
            )
        if self.perception.imgsz < 32:
            raise ValueError(f"perception.imgsz 太小: {self.perception.imgsz}")
        if self.perception.min_side_px >= self.perception.max_side_px:
            raise ValueError(
                "perception 像素边长区间无效: "
                f"{self.perception.min_side_px} >= {self.perception.max_side_px}"
            )
        if self.perception.ocr_dedupe_s < 0 or self.perception.ocr_dedupe_px < 0:
            raise ValueError("perception 去重阈值不能为负")
        if self.targeting.eps_m <= 0:
            raise ValueError(f"targeting.eps_m 必须为正: {self.targeting.eps_m}")
        if self.drop.radius_m <= 0:
            raise ValueError(f"drop.radius_m 必须为正: {self.drop.radius_m}")
        if self.mission.abort_action not in ABORT_ACTIONS:
            raise ValueError(
                f"mission.abort_action 只支持 {sorted(ABORT_ACTIONS)}: "
                f"{self.mission.abort_action!r}"
            )
        if self.mission.tick_hz <= 0:
            raise ValueError(f"mission.tick_hz 必须为正: {self.mission.tick_hz}")
        for name in (
            "init_max_s",
            "recon_max_s",
            "land_max_s",
            "mission_start_timeout_s",
            "airborne_timeout_s",
            "telemetry_stale_s",
        ):
            value = getattr(self.mission, name)
            if value <= 0:
                raise ValueError(f"mission.{name} 必须为正: {value}")
        if self.mission.hold_process_max_s < 0 or self.mission.hold_process_min_s < 0:
            raise ValueError("mission 的 HOLD_PROCESS 时长不能为负")
        if self.mission.hold_process_min_s > self.mission.hold_process_max_s:
            raise ValueError(
                "mission.hold_process_min_s 不能大于 hold_process_max_s: "
                f"{self.mission.hold_process_min_s} > {self.mission.hold_process_max_s}"
            )
        if self.gripper.instance < 0:
            raise ValueError(f"gripper.instance 不能为负: {self.gripper.instance}")
        if self.gripper.release_settle_s < 0:
            raise ValueError(f"gripper.release_settle_s 不能为负: {self.gripper.release_settle_s}")
        if self.routes.recon_route and self.routes.recon_plan:
            raise ValueError(
                "侦查段只能二选一：routes.recon_route（配置航点）与 "
                f"routes.recon_plan（{self.routes.recon_plan}）同时给了"
            )
        if self.routes.landing_route and self.routes.land_plan:
            raise ValueError(
                "降落段只能二选一：routes.landing_route（配置航点）与 "
                f"routes.land_plan（{self.routes.land_plan}）同时给了"
            )
        if not 0.0 < self.routes.fw_land_angle_deg <= 45.0:
            raise ValueError(
                f"routes.fw_land_angle_deg 应在 (0, 45] 度：{self.routes.fw_land_angle_deg}"
            )
        if self.mission.recon_upload not in RECON_UPLOAD_MODES:
            raise ValueError(
                f"mission.recon_upload 只支持 {sorted(RECON_UPLOAD_MODES)}: "
                f"{self.mission.recon_upload!r}"
            )
        if self.mission.airborne_alt_m < 0:
            raise ValueError(f"mission.airborne_alt_m 不能为负：{self.mission.airborne_alt_m}")
        if self.preflight.video_probe_s <= 0 or self.preflight.max_s <= 0:
            raise ValueError(
                "preflight.video_probe_s / max_s 必须为正："
                f"{self.preflight.video_probe_s} / {self.preflight.max_s}"
            )
        if self.preflight.video_min_frames < 1:
            raise ValueError(
                f"preflight.video_min_frames 至少为 1：{self.preflight.video_min_frames}"
            )
        self.video.validated()
        return self

    def replace(self, **changes) -> "Config":
        """派生新配置（浅层替换顶层片段）。"""
        return replace(self, **changes)
