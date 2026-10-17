"""SITL 飞行几何与场景任务配置。"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from airdrop import Config, Waypoint

from .sitl_scene import ReconScene, load_recon_scene
from .tool_loader import load_tool_module

LOGGER = logging.getLogger("sitl_recon")

SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"

#: 世界的 ENU 原点（``vendor/airdrop-sim/airdrop_sim/assets/worlds/cuadc/*.sdf`` 的 ``spherical_coordinates``）
_DEFAULT_SCENE = load_recon_scene()
WORLD_ORIGIN_LLA = _DEFAULT_SCENE.origin_lla

#: r2 世界（seed 0）A 区三座天井的真值：编号 → 中心 (east_m, north_m)（ENU 米）。
#: 与 ``vendor/airdrop-sim/airdrop_sim/make_world.py`` 的布局一致（``vendor/airdrop-sim/tests/test_world.py`` 核验过这一批坐标），
#: 误差口径 = "解算坐标到天井中心"。
WELLS_A_ENU = _DEFAULT_SCENE.wells_enu

#: 起飞爬升点高度（任务首项 = 起飞项，固定翼由它承担起飞）
TAKEOFF_ALT_M = 30.0
#: 侦查高度（相对起飞点）。15 m 下手掌大小的数字牌约 14 px（hfov 1.74 rad），
#: 是"能读清编号"与"不撞天井壁"之间的折中。⚠ 别为了"第二条线也能看到井"把它
#: 抬到 20 m 以上：视场半幅是 20·tan(0.87)=20.2 m，而数字牌会缩到 ~10 px，OCR 就不可靠了。
RECON_ALT_M = 15.0

# ----------------------------------------------------------------------
# 侦查航线的**固定翼几何**
# ----------------------------------------------------------------------
#: 侦查段的沿迹速度（m/s），**转弯半径的输入**。固定翼过弯靠倾角扛离心力，
#: ``R = v² / (g·tanφ)``——速度一高半径就跟着涨，转弯就做不出来。
#: 2026-09-30 实测沿迹速度 14.84 m/s（反向诊断里的 ``speed_m_s``），取整留余量。
RECON_SPEED_M_S = 15.0
#: 相机模型（与 ``vendor/airdrop-sim/airdrop_sim/assets/vehicles/rc_cessna_down_cam`` 一致）
CAMERA_WIDTH = _DEFAULT_SCENE.camera.width_px
CAMERA_HEIGHT = _DEFAULT_SCENE.camera.height_px
CAMERA_HFOV_RAD = _DEFAULT_SCENE.camera.hfov_rad
#: 允许的最大倾角（度）。rc_cessna / PX4 固定翼默认上限 60°，这里留余量。
TURN_MAX_BANK_DEG = 45.0
#: **观测段**的倾角预算（度）——不是转弯允许的倾角，是"算覆盖还够不够"用的那个。
#: 相机是机体固连下视（``CameraJoint`` pitch 90°），横滚 φ 时光轴偏离天底，地面覆盖
#: 点横向挪 ``h·tanφ``：15 m 高时 5°→1.3 m、10°→2.7 m、**15°→4.0 m**、27°→7.6 m。
#: 坐标解算本身用姿态四元数把这点补偿掉了（所以误差不受影响），但**覆盖**会白掉一块：
#: 靶标被推到画幅边缘甚至出框，就根本检不出来。实测带编号检出的 |roll| 最大 16.85°
#: （旧航线 49.29°，它原地掉头压得更狠），所以按 15° 预算、并留一点余量。
OBS_ROLL_BUDGET_DEG = 15.0


def effective_half_width_m(
    altitude_m: float = RECON_ALT_M,
    hfov_rad: float = CAMERA_HFOV_RAD,
    roll_budget_deg: float = OBS_ROLL_BUDGET_DEG,
) -> float:
    """扣掉倾角偏移之后，**真正还能用来对准靶标**的半幅（米）。

    静态半幅是 ``h·tan(hfov/2)``；横滚 φ 时覆盖中心挪 ``h·tanφ``，所以要减掉它。
    这两个方向**大致正交**（横滚偏移是横滚方向、与视线垂直），所以按"两个方向的可用
    半径取小"来扣：``h·tan(hfov/2) − h·tan(roll)``。
    """
    return altitude_m * (math.tan(hfov_rad / 2.0) - math.tan(math.radians(roll_budget_deg)))


def turn_radius_m(speed_m_s: float = RECON_SPEED_M_S, bank_deg: float = TURN_MAX_BANK_DEG) -> float:
    """固定翼协调转弯的半径（米）：``R = v² / (g·tanφ)``。

    ⚠ 这条式子是**航线设计的硬边界**，不是装饰：一条腿两端如果要求飞机转过
    ``Δψ``，而两侧直线段又没有 ``R`` 这么长，飞控就只能在航点附近"尽量绕"——
    转小了就切弯（航线与实际航迹不符，坐标解算直接偏），转大了就冲过头。
    """
    bank = math.radians(float(bank_deg))
    if not 0.0 < bank < math.pi / 2:
        raise ValueError(f"倾角必须在 (0°, 90°) 之间：{bank_deg}")
    return float(speed_m_s) ** 2 / (9.80665 * math.tan(bank))


#: 由上面两项推出的最小转弯半径：15 m/s、45° 倾角 ⇒ R ≈ 22.9 m。
TURN_RADIUS_M = turn_radius_m()

#: 扫掠线（ENU 米）。**单线一次覆盖三个井**，再接一条西南向对角线反向掠过。
#:
#: ⚠ 为什么不再用"两条平行线来回飞"（2026-09 那版）：固定翼的 180° 掉头**必然**要把
#: 飞机横向挪开 ``2R``（两段反向圆弧的横移之和恒为 2R，与两段圆心角无关），所以
#: "同一条线来回飞"只有在两条线相距 ≥2R 时才飞得出来，挪不够就是**原地掉头**——
#: 那正是要消除的东西。
#:
#: 而本赛区的 A 区三个井南北只跨 19 m（12@N-14、56@N-10.8、94@N+6.9），
#: 15 m 高度的视场半幅只有 15·tan(0.87)=17.8 m。两条线一旦按 ``2R≈46 m`` 拉开，
#: 第二条线**一个井都看不到**——"同线反向"与"转得出来"在这个布局下是互斥的。
#:
#: 所以改成：① 一条**N=RECON_LINE_NORTH_M** 的扫掠线一次压过三个井；② 返程用一条
#: **西南向对角线**顺次掠过同样三个井，提供**反向**（东向速度为负）的观测。全程转角
#: ≤ 45°，没有任何 180°。
#:
#: 扫掠线取在三个井的**形心** N≈-6（不是随便取 0）：南北跨度 19 m，线放中间才能让
#: **最大**偏差最小——形心处最大偏差 12.9 m（井 94），放到 N=0 则是 14.0 m（井 12）。
#: 覆盖判据要用**扣掉倾角**的半幅 :func:`effective_half_width_m`（15° 预算 ⇒ 13.8 m），
#: 12.9 m 刚好在里面。⚠ 只用静态半幅 17.8 m 去算就会**高估余量**——倾角把覆盖中心挪
#: ``h·tanφ``，靶标被推到画幅边缘就检不出来（坐标解算有姿态补偿，覆盖没有）。
#:
#: 代价要说清：对角线与扫掠线的**切向不同**，所以反向诊断里的 ``Δeast`` 不再是纯粹的
#: 链路延时（它混了切向变化），只能当量级参考——这正是 2026-09 那份报告里
#: ``reverse`` 为空的原因；宁可给一个带说明的估算值，也不给一个飞不出来的"干净"值。
RECON_LEGS_ENU: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((105.0, -6.0), (150.0, -6.0)),  # 起飞后下降到侦查高度，进入扫掠线西端
    ((150.0, -6.0), (245.0, -6.0)),  # 扫掠线东向：12 / 56 / 94 一次全压过
    ((245.0, -6.0), (285.0, -6.0)),  # 东端外拉，给转弯留出半径
    ((285.0, -6.0), (312.0, 21.0)),  # 45° 东北
    ((312.0, 21.0), (312.0, 52.0)),  # 转正北
    ((312.0, 52.0), (250.0, 26.0)),  # 西南（长腿，顺势离开赛区）
    ((250.0, 26.0), (150.0, -31.0)),  # 西南对角线：反向掠过 94 → 56 → 12
)
#: 无目标时的**备用点**（ENU 米）——三个靶标真值的形心。
#:
#: ⚠ 不配它就没有"无目标"这条分支：:func:`~airdrop.mission.planner.build_drop_mission`
#: 在"既没有目标坐标、也没有备用点"时抛 :class:`PlanningError`，状态机直接
#: ``ABORT(plan_drop_failed)``，整架次 0 上传 0 投放。实测踩过（架次
#: 20260930-145929）：SITL 里感知比图传慢（缓冲落后 ~2000 帧 ≈ 70 s），编号在
#: ``HOLD_PROCESS`` 60 s 窗口的最后几秒才聚类出来，状态机已经放弃——而**事后离线分析
#: 这架次的精度是合格的**（12 → 0.364 m、94 → 1.268 m，都 < 2 m）。也就是说整轮测试
#: 被一个**时序竞争**作废了。配了备用点就会走 Q11 那条设计好的分支（照常飞掠投放，
#: 报告里 ``source="backup"``），时序 unlucky 也不会把整轮作废。
BACKUP_POINT_ENU = _DEFAULT_SCENE.backup_point_enu

#: 降落段：操作手在 QGC 画好、入库的 ``routes/land.plan``（已对准本赛区跑道）。
LAND_PLAN = str(Path(__file__).resolve().parents[2] / "routes" / "land.plan")
MANIFEST_PATH = None

#: RECON 段的上限（秒）：状态机在这一段等"侦查航线飞完"（生产默认 600 s）。
RECON_TIMEOUT_S = 600.0
#: 起飞前自检的上限（秒）：SITL 里 OCR 进程池第一次加载要 15~20 s，给宽一点。
PREFLIGHT_MAX_S = 180.0
#: HOLD_PROCESS 的等待上限（秒）。
#:
#: ⚠ **必须覆盖感知积压**，否则会在"其实已经读出编号"的时候放弃。实测（架次
#: 20260930-145929）：检测 ~17.6 fps 而相机 30 fps，缓冲**只涨不落**，
#: ``lag_frames_max = 2188`` 帧 ≈ **73 s**；而 60 s 的窗口在编号聚类出来的前几秒就到期
#: → ``ABORT(plan_drop_failed)``，整架次 0 上传 0 投放。**事后离线分析这架次精度是
#: 合格的**（12 → 0.364 m、94 → 1.268 m），所以那不是精度问题、纯粹是时序竞争。
#:
#: 积压消不掉、只能等：``imgsz`` 不能降（15 m 高数字牌只有 ~14 px，降到 640 就 7 px，
#: OCR 直接读不出来），OCR 侧也不是瓶颈（``ocr_in_per_s ≈ 0.1``，2 个 worker 闲着），
#: 而队列**不许丢帧**。所以这里给 150 s：
#: 73 s 积压 + 侦查段本身 ~65 s + 聚类余量。
HOLD_WAIT_S = 150.0
#: 跨帧去重的窗口（秒 / 像素）——**OCR 吞吐阀**。
#: 30 fps × 最多 3 个目标时，OCR 的原始需求可达 30~90 次/s；2 个 worker 的实测服务
#: 能力 ≈8.5 次/s，不去重时积压无界增长（请求不丢，但内存与结果延时都涨）。
#: 目标像素漂移实测 ≈500 px/s（15 m 高、~20 m/s 平飞）：250 px 窗口 ≈ 每 0.5 s
#: 一次请求/目标 ⇒ 3 个目标 ≈6 次/s，落在服务能力内；被挡下的送检计入
#: ``PerceptionStats.deduped``，所以"不开去重的原始需求"在报告里仍可还原。
OCR_DEDUPE_S = 1.5
OCR_DEDUPE_PX = 250.0

#: 图传链路延时（秒）**默认值 = SITL 实测**（2026-09，本机 Gazebo→GStreamer→RTP→ffmpeg：
#: ≈0.49 s；x264 编码未开零延迟时会缓冲几十帧）。⚠ 这是**链路属性**，与真机 HM30 的
#: ≈0.15 s 完全不同（`VideoConfig.telemetry_lag`）——报告里会给"自标定 + 延时扫描"
#: 作证据：自标定用已知的世界真值把本架次的链路延时解出来（留一个天井只做验证）。
TELEMETRY_LAG_S = _DEFAULT_SCENE.camera.telemetry_lag_s
LAG_OVERRIDE_S = None
#: 相机系 → 机体系（OpenCV 相机系 → NED 机体系）：绕机体 z +90°
CAMERA_R_BC = tuple(tuple(row) for row in _DEFAULT_SCENE.camera.r_bc())
#: 相机光心在机体系的位置（杆臂，米）：base_link (-0.08, 0, -0.08) 的 NED 写法
CAMERA_T_BC = _DEFAULT_SCENE.camera.t_bc()

#: 视频流（RTP/H.264）的接收端口与 SDP 文件名
STREAM_PORT = 5600
STREAM_SDP_NAME = "stream.sdp"

#: 产物目录（标定文件、SDP、检测明细、报告；不进 git）
WORK_DIR = ".sitl-recon-tmp"
#: 是否写飞行记录（flights/<时间戳>/）
RECORD = True
#: 等第一帧画面的上限（秒）
VIDEO_WAIT_S = 25.0
#: 遥测/原点就绪的上限（秒）
READY_TIMEOUT_S = 60.0
#: lag 扫描的候选值（秒）：覆盖 SITL 实测的 ≈0.49 s 与真机量级
LAG_SWEEP_S = tuple(round(0.02 * index, 2) for index in range(0, 31))


def _enu_to_waypoint(
    east_m: float, north_m: float, alt_m: float, *, scene: ReconScene | None = None
):
    """ENU（相对世界原点）→ :class:`Waypoint`（经纬度 + 相对高度）。"""
    from airdrop import ned_to_wgs84

    lon, lat, _ = ned_to_wgs84((north_m, east_m, 0.0), ref=(scene or _DEFAULT_SCENE).origin_lla)
    return Waypoint(lat=lat, lon=lon, alt_m=alt_m)


def _unit(step: tuple[float, float]) -> tuple[float, float]:
    """ENU 平面上的单位向量。"""
    norm = math.hypot(step[0], step[1])
    if norm <= 0.0:
        raise ValueError(f"零长向量：{step}")
    return (step[0] / norm, step[1] / norm)


def round_corners(
    points: Sequence[tuple[float, float]], radius_m: float, *, step_deg: float = 45.0
) -> tuple[tuple[float, float], ...]:
    """把折线的每个尖角替换成**半径已知的圆弧**（沿弧每 ``step_deg`` 出一个航点）。

    为什么要有这一步：折线只给"从哪到哪"，转角由飞控自己想办法——转小了切弯、
    转大了冲过头，航线与实际航迹对不上，而坐标解算完全依赖"飞机此刻在哪"。
    所以这里**把转弯显式画出来**：每个角切成半径 ``radius_m`` 的弧，于是

    * 相邻航点的航向差被钉在 ``step_deg`` 以内（不会突然要转 90° 以上）；
    * 需要的半径就是算出来的 :func:`turn_radius_m`，与速度/倾角自洽。

    切线长 ``t = R·tan(Δψ/2)`` 要能塞进两侧直线段，所以会按 0.45 倍边长夹紧——
    宁可弧变小（转得更急但仍是弧），也不要切到相邻航点之外去。
    """
    if radius_m <= 0.0:
        raise ValueError(f"转弯半径必须为正：{radius_m}")
    if not 0.0 < step_deg <= 90.0:
        raise ValueError(f"步进角必须在 (0°, 90°]：{step_deg}")
    if len(points) < 2:
        return tuple(points)

    out: list[tuple[float, float]] = [tuple(points[0])]  # type: ignore[list-item]
    for index in range(1, len(points) - 1):
        corner = points[index]
        before = points[index - 1]
        after = points[index + 1]
        v_in = _unit((corner[0] - before[0], corner[1] - before[1]))
        v_out = _unit((after[0] - corner[0], after[1] - corner[1]))
        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
        dot = max(-1.0, min(1.0, v_in[0] * v_out[0] + v_in[1] * v_out[1]))
        turn_deg = math.degrees(math.acos(dot))
        if turn_deg < 1e-6 or abs(cross) < 1e-12:
            out.append(corner)  # 共线：不切
            continue
        sign = 1.0 if cross > 0.0 else -1.0
        tangent = min(
            radius_m * math.tan(math.radians(turn_deg) / 2.0),
            0.45 * math.dist(before, corner),
            0.45 * math.dist(corner, after),
        )
        if tangent < 1e-6:  # 切线长到没有意义就保留尖角
            out.append(corner)
            continue
        entry = (corner[0] - v_in[0] * tangent, corner[1] - v_in[1] * tangent)
        # 圆心在入口的法线方向上（法线按转向取哪一侧）
        normal = (-v_in[1] * sign, v_in[0] * sign)
        center = (entry[0] + normal[0] * radius_m, entry[1] + normal[1] * radius_m)
        start = math.atan2(entry[1] - center[1], entry[0] - center[0])
        sweep = math.radians(turn_deg) * sign
        steps = max(1, math.ceil(turn_deg / step_deg))
        # k=0 是**入口**点：必须输出，否则上一个顶点会直接连到第一个弧采样点，
        # 等于把这个弯又切掉一块（切掉的量正好是切线长 t）。
        for k in range(0, steps + 1):
            angle = start + sweep * (k / steps)
            out.append(
                (center[0] + radius_m * math.cos(angle), center[1] + radius_m * math.sin(angle))
            )
    out.append(tuple(points[-1]))  # type: ignore[arg-type]
    return tuple(out)


def recon_route(*, scene: ReconScene | None = None) -> tuple[Waypoint, ...]:
    """扫掠线 → 航点：先摊平折线，再把每个尖角切成半径已知的圆弧。

    首点取 :data:`TAKEOFF_ALT_M`（起飞爬升），其余取 :data:`RECON_ALT_M`。
    """
    flat: list[tuple[float, float]] = []
    for start, end in RECON_LEGS_ENU:
        for point in (start, end):
            if not flat or flat[-1] != point:
                flat.append(point)
    dense = round_corners(flat, TURN_RADIUS_M)
    return tuple(
        _enu_to_waypoint(east, north, TAKEOFF_ALT_M if index == 0 else RECON_ALT_M, scene=scene)
        for index, (east, north) in enumerate(dense)
    )


def _validate_builtin_recon(scene: ReconScene) -> None:
    """固定航线只用于已覆盖的下视场景，其它场景须显式给航线。"""
    angles = scene.camera.mount_rpy_rad
    expected = _DEFAULT_SCENE.camera.mount_rpy_rad
    if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(angles, expected, strict=True)):
        raise ValueError("非默认相机安装姿态需要显式提供经过覆盖核验的 route")
    camera_height = RECON_ALT_M - scene.camera.t_bc()[2]
    if not scene.camera.near_m < camera_height < scene.camera.far_m:
        raise ValueError("相机光心高度不在可用裁剪范围内，需要核验 route 与安装位置")
    margin = effective_half_width_m(altitude_m=camera_height, hfov_rad=scene.camera.hfov_rad)
    margin -= math.hypot(*scene.camera.mount_translation_frd_m[:2])

    def distance(point: tuple[float, float], leg) -> float:
        (ax, ay), (bx, by) = leg
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        weight = (
            0.0
            if length_sq == 0
            else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq))
        )
        return math.hypot(point[0] - ax - weight * dx, point[1] - ay - weight * dy)

    missed = [
        code
        for code, point in scene.wells_enu.items()
        if min(distance(point, leg) for leg in RECON_LEGS_ENU) > margin
    ]
    if missed:
        raise ValueError(f"场景靶标 {missed} 超出内置侦查航线覆盖，请显式提供 route")


def build_config(
    *,
    system_address: str = SYSTEM_ADDRESS,
    calib_file: str = str(Path(WORK_DIR) / "camera_calib_sim.json"),
    route: tuple[Waypoint, ...] | None = None,
    recon_plan: str = "",
    backup_point: Waypoint | None = None,
    land_plan: str = LAND_PLAN,
    recon_timeout_s: float = RECON_TIMEOUT_S,
    scene: ReconScene | None = None,
) -> Config:
    """装配测试配置：真航线（侦查 + 降落）、真感知参数、模拟相机标定、标准任务流程。"""
    context = scene or _DEFAULT_SCENE
    if land_plan == LAND_PLAN and context.origin_lla != WORLD_ORIGIN_LLA:
        raise ValueError("非默认世界原点必须显式提供与该场景匹配的 land_plan")
    if route is None and not recon_plan:
        _validate_builtin_recon(context)
    base = Config()
    return Config(
        telemetry=replace(base.telemetry, system_address=system_address),
        routes=replace(
            base.routes,
            recon_route=(
                () if recon_plan else route if route is not None else recon_route(scene=scene)
            ),
            recon_plan=recon_plan,
            land_plan=land_plan,
            # ⚠ 必须配：没有它就没有"无目标"这条分支，感知晚一步就整架次 ABORT
            # （见 BACKUP_POINT_ENU 的说明）。
            backup_point=backup_point
            or _enu_to_waypoint(
                *(scene or _DEFAULT_SCENE).backup_point_enu, alt_m=0.0, scene=scene
            ),
        ),
        camera=replace(base.camera, calib_file=calib_file),
        # 模拟相机没有畸变：不让检测器做去畸变（像素与内参保持同一张图）。
        # 去重窗口是**吞吐阀**（口径见 OCR_DEDUPE_* 的说明）：不挡的话 OCR 必被淹没
        perception=replace(
            base.perception,
            mode="ocr",
            target_color="blue",
            ocr_dedupe_s=OCR_DEDUPE_S,
            ocr_dedupe_px=OCR_DEDUPE_PX,
        ),
        # 任务规则：三个编号里打中位数（r2 的 A 区 = 94/12/56 → 56）
        targeting=replace(base.targeting, selection_rule="median"),
        # **标准流程**：侦查航线由"操作手"（本脚本自己扮演）上传并启动；
        # 状态机只监视（recon_upload="operator"），等起飞后才进 RECON。
        # HOLD 等待放宽（HOLD_WAIT_S）：SITL 感知可能比图传慢，别让它"到得太晚"
        mission=replace(
            base.mission,
            recon_upload="operator",
            require_airborne=True,
            recon_max_s=recon_timeout_s,
            hold_process_max_s=HOLD_WAIT_S,
        ),
        # SITL 里 OCR 进程池首次加载慢（15~20 s），自检上限放宽
        preflight=replace(
            base.preflight,
            max_s=PREFLIGHT_MAX_S,
        ),
        # 平地：地面高程 = 原点高程（世界 elevation=0，飞机出生点就在地面）
        ground=replace(base.ground, ground_point_alt=0.0),
    ).validated()


def write_sim_calib(
    path: Path | str, *, lag_s: float | None = None, camera: Any | None = None
) -> Path:
    """从生成 SDF 的同一相机配置导出模型推导值。"""
    model = camera or _DEFAULT_SCENE.camera
    if lag_s is not None:
        model = replace(model, telemetry_lag_s=float(lag_s))
    return model.write_calibration(path)


def write_stream_sdp(path: Path | str, *, port: int = STREAM_PORT) -> Path:
    """为 Gazebo RTP 流写入接收端 SDP。"""
    module = load_tool_module("airdrop_sim.stream")
    return module.write_stream_sdp(path, port=port)
