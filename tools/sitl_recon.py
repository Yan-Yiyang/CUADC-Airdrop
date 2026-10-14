"""SITL 目标侦查精度自动测试：真世界 + 真相机 + 真感知 + 真坐标解算。

它回答一个问题：**飞一遍侦查航线，本包解算出来的目标坐标离天井真值中心有多远**。

用法
----
1) WSL/Linux 里起 SITL（带下视相机的机型 + CUADC r2 世界）::

       HEADLESS=1 bash sim/run_sitl.sh r2      # 自动化一律用无头模式

   ⚠ **`HEADLESS=1` 不是可选项**：带 GUI 时 Gazebo 会和感知抢同一块 GPU，实测把仿真
   拖到 ~0.5x 实时（ulog 里"侦查飞完"=53.6 模拟秒 vs 遥测 101.6 实时秒）⇒ 感知落后
   ~1900 帧、`HOLD_PROCESS` 看不到目标（架次 20260929-200316：ABORT）；
   `px4-rc.gzsim` 里 `if [ -z "${HEADLESS}" ]` 才起 `gz sim -g`，所以置 1 即可。

   Gazebo 自带的 GstCameraSystem 插件会把世界里的相机编成 RTP/H.264 推到
   ``127.0.0.1:5600``；WSL 是 mirrored 网络，Windows 侧直接用 ffmpeg 收同一条流，
   本脚本就吃这条流（不依赖 HM30 图传）。

2) Windows 侧（仓库 venv）::

       ./.venv/Scripts/python.exe -m airdrop.run sitl-recon

它做什么（**完全走标准流程**，全脚本没有 offboard 设定点、没有自造状态机）
----------------------------------------------------------------------
本脚本只扮演**操作手**，其余全部交给生产组件：

1. 连上 SITL → 等遥测/原点 → 起视频流 → 起 ``PerceptionWorker``（YOLO 逐帧 +
   **OCR 进程池**）与 ``FlightRecorder``（五个记录文件 + ``drops.jsonl`` 全写，
   检出经 ``PerceptionTargetSource.on_detection`` 进 ``detections.jsonl``）；
2. **操作手起跑**：上传"只含 ``NAV_TAKEOFF``"的任务 → ``arm`` → ``MISSION_START``
   → 等在空中（真机里这一步是手飞/RC 起飞，SITL 里用起飞任务等价替代）；
3. **操作手上传并启动侦查航线**（已空中 ⇒ 不带起飞项）；
4. 之后交给 ``MissionRunner`` 状态机 + 真 ``Preflight`` + ``DryRunController``：
   ``PREFLIGHT``（载入 detector/ocr/camera + 视频自检）→ ``RECON`` 监视 →
   ``HOLD_PROCESS`` 出目标 → ``OVERFLY`` 飞掠 + 投放（干跑）→ ``LAND`` 降落 → ``DONE``
   （飞掠航线的规划、固定翼降落预检、投放判据、一次性锁存全部走生产代码）；
5. 收尾出报告：拿解算结果与世界的真值（天井中心，见 :data:`WELLS_A_ENU`）比误差，
   并给出链路延时**自标定**与 lag 扫描。

SITL 里临时改两条飞控参数（跑完在 ``finally`` 里还原成机型标准值，不动真机配置）：

* ``NAV_DLL_ACT=0``——SITL 没有操作手盯数据链，别让"数据链丢失"的 failsafe 中途抢控制；
* ``NAV_RCL_ACT=0``——SITL 没有遥控（ulog 的 ``manual_control_signal_lost`` 恒 true），
  默认动作是 Return，实测会在盘旋阶段触发 RC-loss 失控保护（架次 20260929-200316：
  "Failsafe activated: entering Hold for 5 seconds" → RTL）。

另一条 ``MIS_TKO_LAND_REQ`` 已经在**仿真机型里永久置 0**（见
``sim/airframes/4007_gz_rc_cessna_down_cam``）：真实流程的投弹航线要等空中出结果才能
生成、降落段没法预先和侦查段拼成一条任务，而 ``rc.fw_defaults`` 默认要求"带起飞就
必须带降落"（否则 ``mission_feasibility_checker`` 直接拒任务、连解锁都不会发生）。

精度预算里的三个刻意安排
------------------------
* **相机是机体固连、不垂直向下**：内参/外参按 ``sim/vehicles/rc_cessna_down_cam``
  的模型写死（hfov 1.74 rad、绕机体 z +90°、杆臂 ``(-0.08, 0, +0.08)``）——
  姿态一变视线就偏，这正是要测的东西，不做"目标在正下方"的近似；
* **转弯会倾斜、也照样测**：扫掠段实测横滚常年在 5~15°（真实飞行也会有大倾角，
  这正好压测坐标转换模块）；报告里单独统计扫掠时刻的 |roll|；
* **侦查航线是按固定翼转弯几何生成的，不是随手画的折线**：``RECON_LEGS_ENU`` 给的是
  折线骨架，:func:`round_corners` 把每个尖角切成半径 :func:`turn_radius_m` 算出来的圆弧
  （15 m/s、45° 倾角 ⇒ R≈22.9 m），于是相邻航点的航向差被钉在 45° 以内。
  ⚠ **绝不能出现原地 180° 掉头**：固定翼做不到，而且两条平行线的间距一旦小于 2R
  就等于逼它原地翻面（2026-09 那版两条线只隔 19 m、掉头环半径 9.5 m，20 m/s 下要 77° 倾角）。
  代价：两条线按 2R≈46 m 拉开后就**一个靶标都看不到**（靶区南北只跨 19 m、视场半幅
  17.8 m），所以改成"单条 N=0 的线一次压过三个靶标 + 一条西南对角线反向掠过"；
  连带的后果是反向诊断的 ``implied_lag`` 混了切向变化，只能量级参考（见
  :func:`_reverse_diagnostic`）。
* **图传链路延时是头号精度项**：``TELEMETRY_LAG_S`` 默认取 **SITL 实测 ≈0.49 s**
  （Gazebo 渲染 + GStreamer/x264 编码缓冲 + RTP/UDP + ffmpeg 解码；真机 HM30 ≈0.15 s，
  两者必须各自标定）。用错的代价是沿航迹的系统偏差：实测 0.08 s → 3.5 m、
  0.52 s → 0.3 m。报告里给出**自标定**（用已知世界真值解出本架次延时，留一个天井
  只做验证）与全量 lag 扫描——外参不变时若修正延时后仍剩恒定像素偏差，才是外参问题。

本文件是纯库模块：``build_config(**覆盖)`` / ``main(**kwargs)`` 的默认值就是文件
顶部常量；命令行由 ``airdrop/run.py`` 解析，重依赖都在函数体内导入。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from airdrop import Config, TelemetryBroker, Waypoint

LOGGER = logging.getLogger("sitl_recon")

#: MAVSDK 连 SITL 的地址（与其它 SITL 入口一致）
SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"

#: 世界的 ENU 原点（``sim/worlds/cuadc/*.sdf`` 的 ``spherical_coordinates``）
WORLD_ORIGIN_LLA = (8.546163739800146, 47.397971057728974, 0.0)

#: r2 世界（seed 0）A 区三座天井的真值：编号 → 中心 (east_m, north_m)（ENU 米）。
#: 与 ``tools/make_world.py`` 的布局一致（``tests/test_world.py`` 核验过这一批坐标），
#: 误差口径 = "解算坐标到天井中心"。
WELLS_A_ENU: dict[int, tuple[float, float]] = {
    94: (213.11, 6.91),
    12: (178.01, -13.99),
    56: (200.30, -10.80),
}

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
#: 相机模型（与 ``sim/vehicles/rc_cessna_down_cam`` 一致）
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV_RAD = 1.74
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
BACKUP_POINT_ENU = (197.14, -5.96)

#: 降落段：操作手在 QGC 画好、入库的 ``routes/land.plan``（已对准本赛区跑道）。
LAND_PLAN = "routes/land.plan"

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
TELEMETRY_LAG_S = 0.49
#: 相机系 → 机体系（OpenCV 相机系 → NED 机体系）：绕机体 z +90°
CAMERA_R_BC = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
#: 相机光心在机体系的位置（杆臂，米）：base_link (-0.08, 0, -0.08) 的 NED 写法
CAMERA_T_BC = (-0.08, 0.0, 0.08)

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


def _enu_to_waypoint(east_m: float, north_m: float, alt_m: float):
    """ENU（相对世界原点）→ :class:`Waypoint`（经纬度 + 相对高度）。"""
    from airdrop import ned_to_wgs84

    lon, lat, _ = ned_to_wgs84((north_m, east_m, 0.0), ref=WORLD_ORIGIN_LLA)
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


def recon_route() -> tuple[Waypoint, ...]:
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
        _enu_to_waypoint(east, north, TAKEOFF_ALT_M if index == 0 else RECON_ALT_M)
        for index, (east, north) in enumerate(dense)
    )


def build_config(
    *,
    system_address: str = SYSTEM_ADDRESS,
    calib_file: str = str(Path(WORK_DIR) / "camera_calib_sim.json"),
    route: tuple[Waypoint, ...] | None = None,
    land_plan: str = LAND_PLAN,
    recon_timeout_s: float = RECON_TIMEOUT_S,
) -> Config:
    """装配测试配置：真航线（侦查 + 降落）、真感知参数、模拟相机标定、标准任务流程。"""
    base = Config()
    return Config(
        telemetry=replace(base.telemetry, system_address=system_address),
        routes=replace(
            base.routes,
            recon_route=route if route is not None else recon_route(),
            land_plan=land_plan,
            # ⚠ 必须配：没有它就没有"无目标"这条分支，感知晚一步就整架次 ABORT
            # （见 BACKUP_POINT_ENU 的说明）。
            backup_point=_enu_to_waypoint(*BACKUP_POINT_ENU, alt_m=0.0),
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


def _write_param(thread: Any, name: str, value: int) -> None:
    """在 MAVSDK 线程上写一条飞控参数（同步等待结果）。"""

    async def _write() -> None:
        await thread.require_drone().param.set_param_int(name, value)

    thread.submit(_write()).result(timeout=15.0)


#: 数据链丢失时飞控的动作（airframe 的 ``NAV_DLL_ACT``）：测试里临时置 0，
#: 收尾还原成机型标准值 2（Return）。真机保持机型/参数的配置，不在测试里动。
NAV_DLL_ACT_DEFAULT = 2


def read_param(thread: Any, name: str) -> int:
    """读一条飞控参数（在 MAVSDK 线程上执行）。"""

    async def _read() -> int:
        return await thread.require_drone().param.get_param_int(name)

    return thread.submit(_read()).result(timeout=15.0)


def zero_test_params(
    thread: Any, names: tuple[str, ...] = ("NAV_DLL_ACT", "NAV_RCL_ACT")
) -> dict[str, int]:
    """把 SITL 测试要临时关掉的飞控参数置 0，返回原值（``main`` 收尾时还原）。

    * ``NAV_DLL_ACT``：SITL 里没有操作手盯数据链，置 0 免得"数据链丢失"的 failsafe
      在飞行中途抢控制；
    * ``NAV_RCL_ACT``：SITL 里**没有遥控**（ulog 的 ``manual_control_signal_lost``
      恒 true），默认动作是 Return——实测架次 20260929-200316 在盘旋时被 RC-loss
      失控保护拉成 RTL（"Failsafe activated: entering Hold for 5 seconds" → RTL）。

    ⚠ "起飞项/降落项是否必需"那条检查（``MIS_TKO_LAND_REQ``）**不在这里改**：
    它已经在仿真机型里永久置 0（``sim/airframes/4007_gz_rc_cessna_down_cam``，附了原因）
    ——投弹航线要空中出结果才能生成、降落段没法预先和侦查段拼成一条任务。
    """
    originals: dict[str, int] = {}
    for name in names:
        originals[name] = read_param(thread, name)
        _write_param(thread, name, 0)
    return originals


def restore_params(thread: Any, values: dict[str, int]) -> None:
    """还原测试期间改过的飞控参数（尽力而为：失败只告警）。"""
    for name, value in values.items():
        # 当年被测试改过的参数恢复成机型标准值（持久化在 SITL 的参数库里）
        target = NAV_DLL_ACT_DEFAULT if name == "NAV_DLL_ACT" else value
        try:
            _write_param(thread, name, target)
            LOGGER.info("飞控参数 %s 已还原为 %s", name, target)
        except Exception:  # noqa: BLE001 - 还原失败不影响报告
            LOGGER.warning("还原飞控参数 %s 失败（不影响报告）", name)


def start_status_text_logger(thread: Any) -> None:
    """把 PX4 的原生消息（STATUSTEXT）原样打进日志：出问题时第一手证据就在这儿。

    只**监听**，不下任何指令。
    """

    async def _start(_=None) -> None:
        drone = thread.require_drone()

        async def _loop() -> None:
            async for message in drone.telemetry.status_text():
                LOGGER.info("PX4: %s", message.text)

        # 存住引用：任务要一直活到线程退出（RUF006）
        _tasks.append(asyncio.create_task(_loop()))

    thread.submit(_start()).result(timeout=10.0)


#: 后台监听任务（只用它拿住强引用）
_tasks: list[Any] = []


def takeoff_items(config: Config) -> tuple[Any, ...]:
    """只含起飞项的任务：位置取侦查航线首点、高度取 :data:`TAKEOFF_ALT_M`。"""
    from airdrop import MissionItem, command_name  # noqa: F401 - command_name 由调用方用

    first = config.routes.recon_route[0]
    return (MissionItem.takeoff(float(first.lat), float(first.lon), TAKEOFF_ALT_M),)


def operator_takeoff(
    config: Config, controller: Any, thread: Any, broker: TelemetryBroker
) -> int | None:
    """**操作手起跑**（自动化的等价物）：上传"只含起飞项"的任务 → arm → 启动 → 等在空中。

    真实流程里这一步由操作手完成（手飞/RC，或在 QGC 里点起飞）；SITL 里没有遥控，
    用一条只含 ``NAV_TAKEOFF`` 的任务代替——固定翼的起飞由飞控按任务首项执行，
    与"侦查航线首项带起飞"用的是同一套机制（``MissionItem.takeoff``）。起飞完成后
    飞控转 AUTO_LOITER 在起点附近盘旋，等下一步"操作手上传侦查航线"。

    两条硬要求（都是实测踩出来的）：
    * **先解锁、再启动**：``MISSION_START`` 里的 ``arm(mission_start)`` 实测只切模式、
      不解锁（ulog: nav_state=3 / arming_state=1），飞机不动；
    * **任务先过可行性检查**：装了起飞项就必须满足 ``MIS_TKO_LAND_REQ``
      （仿真机型里已永久置 0，见 ``sim/airframes/4007_gz_rc_cessna_down_cam``）。

    返回 ``None`` 表示已在空中，否则返回退出码（4 没解锁 / 5 没进任务模式 / 6 没起飞）。
    """

    async def _armed() -> bool:
        async for value in thread.require_drone().telemetry.armed():
            return bool(value)
        return False

    from airdrop import command_name

    disarm = thread.send_command({"name": "disarm"}, timeout=15.0)
    LOGGER.info("起跑前 disarm：%s", "OK" if disarm.success else f"跳过（{disarm.error}）")
    start_status_text_logger(thread)

    items = takeoff_items(config)
    uploaded = controller.upload_mission(items)
    LOGGER.info("起飞任务已上传：%d 项（首项 %s）", uploaded, command_name(items[0].command))

    arm = thread.send_command({"name": "arm"}, timeout=30.0)
    if not arm.success:
        LOGGER.error("解锁指令被拒：%s（预检没过？看 QGC / PX4 的提示）", arm.error)
        return 4
    if not _wait_for(lambda: bool(thread.submit(_armed()).result(timeout=5.0)), 20.0, "解锁生效"):
        LOGGER.error("解锁指令被接受但飞机没有真的解锁：预检未过（QGC 会写原因）")
        return 4
    LOGGER.info("已解锁")

    controller.start_mission()
    if not _wait_for(controller.in_mission_mode, 20.0, "进入任务模式"):
        LOGGER.error("启动后 20s 没进任务模式（飞控拒绝任务？）")
        return 5
    if not _wait_for(
        lambda: (
            bool(broker.get_snapshot().in_air)
            or (broker.get_snapshot().relative_altitude_m or 0.0) >= 5.0
        ),
        60.0,
        "起飞",
    ):
        LOGGER.error("60s 内没有起飞（自动起飞失败？）")
        return 6
    LOGGER.info("已起飞（操作手起跑步骤完成）")
    return None


def operator_start_recon(config: Config, controller: Any) -> int:
    """**操作手角色**：上传并启动侦查航线（已在上空中 ⇒ 不带起飞项）。"""
    from dataclasses import replace as _replace

    from airdrop import build_recon_mission, command_name

    airborne = _replace(config, mission=_replace(config.mission, takeoff_first=False))
    items = build_recon_mission(airborne)
    uploaded = controller.upload_mission(items)
    LOGGER.info("侦查航线已上传：%d 项（首项 %s）", uploaded, command_name(items[0].command))
    controller.start_mission()
    LOGGER.info("侦查航线已启动（操作手步骤完成）")
    return uploaded


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


def write_sim_calib(path: Path | str, *, lag_s: float = TELEMETRY_LAG_S) -> Path:
    """写一份**模拟相机**的标定文件（内参/外参由模型推导，不是标定出来的）。

    ``fx = (w/2)/tan(hfov/2)``（Gazebo 相机的水平视场定义），畸变为零；
    ``R_bc``/``t_bc`` 见文件顶部常量。``telemetry_lag`` 原样写进去，便于核对。
    """
    fx = (CAMERA_WIDTH / 2.0) / math.tan(CAMERA_HFOV_RAD / 2.0)
    payload = {
        "camera_matrix": [
            [fx, 0.0, CAMERA_WIDTH / 2.0],
            [0.0, fx, CAMERA_HEIGHT / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R_bc": [list(row) for row in CAMERA_R_BC],
        "t_bc": list(CAMERA_T_BC),
        "image_size": [CAMERA_WIDTH, CAMERA_HEIGHT],
        "telemetry_lag": float(lag_s),
        "meta": {"source": "sim/vehicles/rc_cessna_down_cam（由模型推导，非标定）"},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_stream_sdp(path: Path | str, *, port: int = STREAM_PORT) -> Path:
    """写接收 RTP/H.264 的 SDP（ffmpeg 靠它认动态 payload type 96）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "v=0",
                "o=- 0 0 IN IP4 127.0.0.1",
                "s=CUADC SITL down camera",
                "c=IN IP4 127.0.0.1",
                "t=0 0",
                f"m=video {port} RTP/AVP 96",
                "a=rtpmap:96 H264/90000",
                "a=recvonly",
                "",
            ]
        ),
        encoding="ascii",
    )
    return target


def _wait_for(predicate, timeout_s: float, what: str, *, interval_s: float = 0.2) -> bool:
    """轮询等条件成立；超时返回 False（不抛，交给调用方决定）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            LOGGER.debug("%s：查询失败，继续等", what, exc_info=True)
        time.sleep(interval_s)
    return False


@dataclass(slots=True)
class PerceptionSetup:
    """**标准感知链路**：`PerceptionWorker`（YOLO 逐帧 + OCR 进程池）+ `TargetTracker`。

    与生产入口（`examples/full_mission.py`）同一套组件；检出经
    `PerceptionTargetSource.on_detection` 写进飞行记录的 `detections.jsonl`。

    ⚠ 为什么不用"进程内 OCR"（2026-09 实测，架次 20260929-183059）：
    * YOLO（imgsz=1280）≈29 ms/帧 ≈ 35 fps ✓——检测本身够实时；
    * OCR 单张裁剪：中位 156 ms、均值 274 ms、p95 927 ms ✗（远超项目里记的 ~100 ms）；
    * 进程内串行时，扫掠段一帧常有 1~3 个目标 ⇒ 单帧 0.2~2.8 s ⇒ 整条链路 ~3~8 fps ✗
      ⇒ 30 fps 的画面必然积压（强制收尾也追不完）；
    * 进程池（2 worker）≈8.5 crop/s 且**不阻塞检测线程** ⇒ 检测回到 ~32 fps ✓。
    ⚠ 代价：每个 OCR worker 都是独立进程（torch + 权重）≈1.2~1.5 GB 内存 / ~1 GB 显存，
      所以 worker 数保持项目默认的 2（实测 4 个会把 16 GB 内存与 6 GB 显存一起打满）。
    """

    worker: Any
    tracker: Any
    source: Any

    @classmethod
    def build(cls, config: Config, buffer: Any, *, on_detection: Any = None) -> "PerceptionSetup":
        """按配置装配（``on_detection`` 一般接 ``recorder.detections.append``）。"""
        from airdrop import PerceptionTargetSource, PerceptionWorker, TargetTracker

        # 同一个 camera 实例给检测器与 TargetTracker：解析两次不仅浪费，还可能出现
        # "预检核对的模型与解算用的不是同一个"（full_mission 注释里明确要避免）
        camera = config.camera.load_model()
        worker = PerceptionWorker(config.perception.to_pipeline_config(camera), buffer=buffer)
        tracker = TargetTracker(config, camera=camera)
        source = PerceptionTargetSource(worker=worker, tracker=tracker, on_detection=on_detection)
        return cls(worker=worker, tracker=tracker, source=source)

    # ------------------------------------------------------------------
    # 报告用的统计（口径来自 PerceptionWorker.stats）
    # ------------------------------------------------------------------
    @property
    def frames(self) -> int:
        return int(self.worker.stats.frames)

    @property
    def detections(self) -> int:
        return int(self.worker.stats.detections)

    @property
    def read_codes(self) -> int:
        return int(self.worker.stats.with_code)

    @property
    def last_error(self) -> str | None:
        return self.worker.stats.last_error

    def snapshot(self) -> dict[str, Any]:
        """收尾口径的明细（报告用）：OCR 请求/结果/队列/去重挡下。

        与 :attr:`frames` 等属性一样读 ``PerceptionWorker.stats``；``ocr_queued``
        是**还没被消费者抽走**的条数——收尾补抽之后应当归零（队列无界、不丢弃）。
        """
        stats = self.worker.stats
        return {
            "ocr_submitted": stats.submitted,
            "ocr_results": stats.results,
            "ocr_with_code": stats.with_code,
            "ocr_deduped": stats.deduped,
            "ocr_queued": self.worker.queued_results,
        }

    def warmup(self) -> None:
        """起飞前把 YOLO 权重与 OCR 进程池热起来（与生产入口一致）。"""
        self.worker.detector.warmup(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.worker.start()

    def stop(self, timeout: float = 10.0) -> None:
        self.worker.stop(timeout=timeout)

    def drain(self, timeout: float = 120.0) -> int:
        """等感知追完缓冲：返回仍有未处理的帧数（0 = 全追完）。

        "未处理"取"缓冲里还没检测的帧"与"OCR 送出去还没回来的请求"两者较大值——
        与 `PerceptionTargetSource.busy` 的口径一致。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stats = self.worker.stats
            if stats.lag_frames <= 0 and stats.submitted <= stats.results:
                return 0
            time.sleep(0.2)
        stats = self.worker.stats
        return int(max(stats.lag_frames, stats.submitted - stats.results, 0))


class PerformanceMonitor:
    """每 5 s 采一次性能样本（感知吞吐 / 积压 / 内存），日志一行、报告全量。

    三个"积压"口径要分开看：

    * ``lag_frames``——缓冲里**还没检测**的帧（检测线程追不上图传）；
    * ``pending``——送出去**还没回来**的 OCR 裁剪图（OCR 追不上检测）；
    * ``deduped``——跨帧去重**挡下**的送检（用来还原"不开去重的原始需求"）。

    内存两项靠 psutil（ultralytics 的传递依赖）；取不到就只少这两列，不影响别的。
    """

    def __init__(
        self,
        perception: PerceptionSetup,
        source: Any,
        writer: Any,
        *,
        interval_s: float = 5.0,
    ) -> None:
        self._perception = perception
        self._source = source
        self._writer = writer
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.monotonic()
        self._previous: dict[str, float] = {}
        self.samples: list[dict[str, Any]] = []
        self._process: Any = None
        try:  # psutil 是可选依赖：拿不到就降级（少内存两列）
            import psutil  # 可选依赖：不在导入期拉起它

            self._process = psutil.Process(os.getpid())
        except Exception:  # noqa: BLE001 - 观测不能影响主流程
            self._process = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="perf-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0 * self._interval_s)
            self._thread = None
        return self.samples

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._sample()
            except Exception:  # 观测绝不把任务带崩
                LOGGER.exception("性能采样失败")

    def _sample(self) -> None:
        stats = self._perception.worker.stats
        tracker_stats = self._perception.tracker.stats
        video = self._source.stats
        written, _skipped = self._writer.stats()
        now = time.monotonic()
        counters = {
            "frames": float(stats.frames),
            "submitted": float(stats.submitted),
            "results": float(stats.results),
            "deduped": float(stats.deduped),
            "video_frames": float(video.frames),
        }

        def rate(key: str) -> float:
            last = self._previous.get(key)
            last_at = self._previous.get("_at")
            if last is None or last_at is None or now <= last_at:
                return 0.0
            return (counters[key] - last) / (now - last_at)

        sample: dict[str, Any] = {
            "t_s": round(now - self._started_at, 1),
            "frames": stats.frames,
            "detections": stats.detections,
            "with_code": stats.with_code,
            "submitted": stats.submitted,
            "results": stats.results,
            "deduped": stats.deduped,
            "pending": max(stats.submitted - stats.results, 0),
            "queued": self._perception.worker.queued_results,
            "points": tracker_stats.get("points", 0),
            "labeled": tracker_stats.get("labeled", 0),
            "lag_frames": stats.lag_frames,
            "ocr_dropped": stats.ocr_dropped,
            "video_frames": video.frames,
            "buffer_written": written,
            "detection_fps": round(rate("frames"), 1),
            "video_fps": round(rate("video_frames"), 1),
            "ocr_in_per_s": round(rate("submitted"), 1),
            "ocr_out_per_s": round(rate("results"), 1),
            "dedupe_per_s": round(rate("deduped"), 1),
            "rss_mb": None,
            "children_mb": None,
        }
        if self._process is not None:
            try:
                sample["rss_mb"] = round(self._process.memory_info().rss / 2**20)
                children = self._process.children(recursive=True)
                sample["children_mb"] = round(
                    sum(child.memory_info().rss for child in children) / 2**20
                )
            except Exception:  # noqa: BLE001 - 子进程可能刚好退出
                pass
        self._previous = {**counters, "_at": now}
        self.samples.append(sample)
        LOGGER.info(
            "性能 %5.0fs: 检测 %.1f fps（累计 %d 帧 → 检出 %d，读号 %d）|"
            "OCR 入 %.1f / 出 %.1f 次/s（积压 %d，去重挡下 %.1f 次/s）|"
            "缓冲落后 %s 帧|目标点 %d（带编号 %d，队列 %d）|内存 %s + 子进程 %s MB",
            sample["t_s"],
            sample["detection_fps"],
            sample["frames"],
            sample["detections"],
            sample["with_code"],
            sample["ocr_in_per_s"],
            sample["ocr_out_per_s"],
            sample["pending"],
            sample["dedupe_per_s"],
            sample["lag_frames"],
            sample["points"],
            sample["labeled"],
            sample["queued"],
            sample["rss_mb"],
            sample["children_mb"],
        )


# ----------------------------------------------------------------------
# 结果分析
# ----------------------------------------------------------------------
def _well_truth_ned(code: int, origin_lla: tuple[float, float, float]) -> tuple[float, float]:
    '运行接口与行为以当前实现及测试为准。'
    from airdrop import ned_to_wgs84, wgs84_to_ned

    east_m, north_m = WELLS_A_ENU[code]
    lon, lat, _ = ned_to_wgs84((north_m, east_m, 0.0), ref=WORLD_ORIGIN_LLA)
    north, east, _ = wgs84_to_ned(lon, lat, WORLD_ORIGIN_LLA[2], ref=origin_lla)
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
        for code in WELLS_A_ENU:
            cluster = _select_by_code(result.clusters, code)
            if cluster is None:
                continue
            codes[code] = _cluster_error(cluster, _well_truth_ned(code, origin_lla))
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
                        _cluster_error(selected, _well_truth_ned(selected.code, origin_lla)), 3
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
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
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
#: 为什么要这条诊断（2026-09 用 6 个架次 114 个带编号观测实测，不是推断）：
#: 起飞后 10~30 s，PX4 报的 ``yaw`` 与 NED 速度矢量的航迹角差 **−26 ~ −35°**（6 个架次
#: 全部复现，|偏差|>10° 占 82~100%），40 s 后回到 ±6° 以内。定翼平飞（滚转 < 1.5°）扛不住
#: 30° 稳态侧滑——垂尾的恢复力矩会把它推回去，所以是**姿态估计错了**，不是飞机真的 crab 着飞。
#:
#: 后果很直接：``pixel_to_ned`` 用的是机体姿态（相机刚性挂在机体上，本来就该用姿态而不是
#: 速度矢量），于是这批观测的落点误差被放大到 3~13 m。同一批点把 yaw 换成航迹角重算：
#: 地面误差中位 **1.31 → 0.82 m**、均方根 **3.72 → 1.09 m**、最大 **13.16 → 3.15 m**、
#: <2 m 的比例 **66.7% → 94.7%**。
#:
#: ⚠ 所以这条诊断**只报告、不剔点**：剔了就看不到"精度是被姿态估计吃掉的"这个事实，
#: 而这正是本工具要回答的问题。10° 既能抓住那 20 s 的瞬态，又不会被转弯中的正常侧滑误伤
#: （正常段的偏差中位数只有 ±3~6°）。
HEADING_TOL_DEG = 10.0

#: 算航迹角所需的最小水平速度（m/s）：低于它 ``atan2(vy, vx)`` 的方向没有意义
#:（停机坪上速度≈0，方向纯噪声）。实测各架次的定位噪声在 0.1 m/s 量级，3 m/s 足够远。
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
        if code is None or code not in WELLS_A_ENU:
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
        truth = _well_truth_ned(code, origin_lla)
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

    实测踩过（架次 20260930-151637）：就这样算出了 ``implied_lag = -0.349 s``——比真实
    延时（~0.36 s）大一个数量级且**符号相反**，纯误导。所以现在用
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
    # 实测（架次 20260930-151637）：就这样算出了 implied_lag = **-0.349 s**——比真实延时
    # 大一个数量级且**符号相反**，纯误导。
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
class _RunOutcome:
    """一次测试跑完后的汇总：报告阶段只依赖这些（都在 finally 里取好）。"""

    work: Path
    sdp: Path
    calib: Path
    lag_s: float
    flown_s: float | None
    result: Any
    stats: dict[str, int]
    perception: "PerceptionSetup"
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
        "拿不到 EKF 原点，无法把天井真值换算到本机 NED：记录里没有原点遥测，runner 也没记到 origin"
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

    为什么要重试：同一目标的重复观测散布强烈依赖**像素在画幅里的离心距**。
    实测（架次 20260930-151637，同一批点只改 ``eps``）：

    ==========  ==========================================================
    ``eps``     带编号的类 / 误差
    ==========  ==========================================================
    0.75 m      2 个（94、12）——56 的 3 个点散在 20 m 上，聚不起来
    **1.5 m**   **3 个**：94 → 0.32 m、56 → 1.22 m、12 → 0.75 m
    3.0 m       3 个：94 → 0.32 m、56 → 0.23 m、12 → 0.75 m
    ==========  ==========================================================

    1.5 m 已能覆盖 r2 这三个靶标，所以它是默认；重试只是兜底——世界布局或观测几何
    变了导致某个靶标差一点聚不起来时，不必手工重跑整架次。

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
    if labeled >= len(WELLS_A_ENU):
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
) -> dict[str, Any]:
    """逐观测落点误差的两个口径（全量 / 航向一致子集）。

    口径与为什么这么做见 :data:`HEADING_TOL_DEG`：偏差大的那批观测**不剔点**，两个口径
    都报出来——剔了就看不到"精度是被姿态估计吃掉的"这个事实，而那正是本工具要回答的问题。
    诊断失败不抛，返回 ``{"failed": ...}`` 让报告继续。
    """
    try:
        per_obs = _per_observation_errors(detections, origin_lla=origin_lla, calib_file=calib_file)
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
) -> dict[str, Any]:
    """航向一致性 + 逐观测误差：算完打印、超阈告警，返回两块直接并进报告。

    合成一个函数是为了不把 :func:`_report` 的分支数顶破上限（它本来就有 15 个）。
    """
    observation_errors = _observation_errors(
        detections, origin_lla=origin_lla, calib_file=calib_file
    )
    heading = _heading_consistency(detections)
    if heading.get("n"):
        print(
            "航向一致性  : |航迹角−yaw| 中位 %.1f°/p90 %.1f°/最大 %.1f°，超 ±%.0f° 的 %d/%d（%.0f%%）"
            % (
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
            "逐观测误差  : 全量 中位 %.2f/均方根 %.2f/最大 %.2f m（<2m %.0f%%）；"
            "航向一致子集 中位 %.2f/均方根 %.2f/最大 %.2f m（<2m %.0f%%，n=%d）"
            % (
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


def _report(outcome: _RunOutcome) -> dict[str, Any]:
    """出报告：目标误差（任务结果 + 分井）、延时敏感性、反向诊断；并落 report.json。"""
    origin_lla = _origin_lla(outcome, _analysis_broker(outcome))
    result = outcome.result
    stats = outcome.stats
    video_stats = outcome.video_stats
    writer_stats = outcome.writer_stats
    perception = outcome.perception
    report: dict[str, Any] = {
        "world": "sim/worlds/cuadc/cuadc_recon_strike_r2.sdf（A 区真值 94/12/56）",
        "lag_s": outcome.lag_s,
        "flight_s": None if outcome.flown_s is None else round(outcome.flown_s, 1),
        "video": {
            "url": str(outcome.sdp),
            "frames": video_stats.frames,
            "dropped": video_stats.dropped,
            "reconnects": video_stats.reconnects,
            "written": writer_stats[0],
            "skipped": writer_stats[1],
        },
        "perception": {
            "frames": perception.frames,
            "detections": perception.detections,
            "read_codes": perception.read_codes,
            "last_error": perception.last_error,
            **perception.snapshot(),
            **stats,
        },
        "performance": _performance_summary(outcome.perf),
        "roll": _roll_stats(outcome.detail),
        "truth_ned": {
            str(code): [round(value, 2) for value in _well_truth_ned(code, origin_lla)]
            for code in WELLS_A_ENU
        },
        "code_errors_m": {},
    }
    # 聚类半径不够就放宽一档再聚一次（口径见 _analyze_with_retry 的 docstring）。
    result, eps_used = _analyze_with_retry(outcome)
    report["eps_m"] = eps_used
    report["clusters"] = [
        {
            "code": cluster.code,
            "count": cluster.count,
            "north_m": round(cluster.north_m, 2),
            "east_m": round(cluster.east_m, 2),
            "confidence": round(cluster.confidence, 3),
        }
        for cluster in result.clusters
    ]
    selected = result.selected
    report["selected"] = (
        None
        if selected is None
        else {
            "code": selected.code,
            "north_m": round(selected.north_m, 2),
            "east_m": round(selected.east_m, 2),
            "error_m": round(
                _cluster_error(selected, _well_truth_ned(selected.code, origin_lla)), 3
            ),
        }
    )
    for code in WELLS_A_ENU:
        cluster = _select_by_code(result.clusters, code)
        if cluster is not None:
            report["code_errors_m"][str(code)] = round(
                _cluster_error(cluster, _well_truth_ned(code, origin_lla)), 3
            )

    print("\n===== SITL 侦查精度测试 =====")
    print("世界        :", report["world"])
    print(
        "视频        : 帧 %s（丢 %s）→ 入缓冲 %s（跳过 %s）"
        % (video_stats.frames, video_stats.dropped, writer_stats[0], writer_stats[1])
    )
    print(
        "感知        : 处理 %d 帧 → 检出 %d → 读号 %d（no_fix %s / side_mismatch %s）"
        % (
            perception.frames,
            perception.detections,
            perception.read_codes,
            stats.get("no_fix"),
            stats.get("side_mismatch"),
        )
    )
    runner_stats = outcome.runner_stats or {}
    report["mission"] = {
        "state": runner_stats.get("state"),
        "uploads": runner_stats.get("uploads"),
        "releases": runner_stats.get("releases"),
        "aborts": runner_stats.get("aborts"),
        "errors": runner_stats.get("errors"),
        "path": runner_stats.get("history", []),
        "state_seconds": _state_durations(
            runner_stats.get("transitions") or [], runner_stats.get("ended_at")
        ),
    }
    print(
        "任务流程    : %s（上传 %s / 投放 %s / ABORT %s / 错误 %s）；轨迹 %s"
        % (
            runner_stats.get("state"),
            runner_stats.get("uploads"),
            runner_stats.get("releases"),
            runner_stats.get("aborts"),
            runner_stats.get("errors"),
            " → ".join(runner_stats.get("history", [])),
        )
    )
    if report["mission"]["state_seconds"]:
        print(
            "状态时长    : %s"
            % " → ".join(
                "%s %.1fs" % (row["state"], row["seconds"])
                for row in report["mission"]["state_seconds"]
            )
        )
    performance = report["performance"]
    if performance:
        print(
            "性能        : 检测 %s fps|OCR 入 %s / 出 %s 次/s"
            "（请求 %s、去重挡下 %s；峰值积压 %s）|缓冲峰值落后 %s 帧"
            "|内存峰值 %s + 子进程 %s MB"
            % (
                performance.get("detection_fps_avg"),
                performance.get("ocr_in_per_s_avg"),
                performance.get("ocr_out_per_s_avg"),
                performance.get("ocr_requests"),
                performance.get("ocr_deduped"),
                performance.get("pending_max"),
                performance.get("lag_frames_max"),
                performance.get("rss_mb_max"),
                performance.get("children_mb_max"),
            )
        )
    print(
        "扫掠姿态    : |roll| 均值 %s°/最大 %s°（大倾角照测）"
        % (report["roll"].get("mean_abs_deg"), report["roll"].get("max_abs_deg"))
    )

    # 逐观测误差 + 航向一致性：把"精度被姿态估计吃掉"这件事摆在明面上（口径见 HEADING_TOL_DEG）
    report.update(
        _report_heading_and_errors(
            outcome.detail, origin_lla=origin_lla, calib_file=str(outcome.calib)
        )
    )

    print(
        "目标点      : 类 %d 个 + 噪声 %d（共 %s 点，带编号 %s）"
        % (len(result.clusters), len(result.noise), stats.get("points"), stats.get("labeled"))
    )
    for cluster in result.clusters:
        code = cluster.code
        truth = _well_truth_ned(code, origin_lla) if code in WELLS_A_ENU else None
        error = None if truth is None else _cluster_error(cluster, truth)
        print(
            "  编号 %s @ NED (%8.2f, %8.2f)  %5d 点  置信 %.2f  真值 (%s)  误差 %s"
            % (
                code,
                cluster.north_m,
                cluster.east_m,
                cluster.count,
                cluster.confidence,
                "%.1f, %.1f" % truth if truth else "—",
                "%.2f m" % error if error is not None else "—",
            )
        )
    if result.selected is not None:
        error = _cluster_error(result.selected, _well_truth_ned(result.selected.code, origin_lla))
        print(
            "任务结果    : 编号 %s @ (%.2f, %.2f)，误差 %.2f m（要求 < 2 m）"
            % (result.selected.code, result.selected.north_m, result.selected.east_m, error)
        )
    else:
        print("任务结果    : 无（没有带编号的候选类）")

    # 离线：延时扫描 + 反向诊断（用重建的遥测历史，不依赖进程内 broker 的窗口）
    report["lag_sweep"] = []
    try:
        from airdrop import load_broker_from_log

        if outcome.flight_dir is not None:
            sweep_broker, pacer = load_broker_from_log(outcome.flight_dir)
            pacer.publish_until(float("inf"))
            report["lag_sweep"] = lag_sweep(
                outcome.detail,
                broker=sweep_broker,
                calib_file=str(outcome.calib),
                base_lag=outcome.lag_s,
            )
            report["reverse"] = _reverse_diagnostic(
                outcome.detail,
                sweep_broker,
                calib_file=str(outcome.calib),
                code=56,
                lag_s=outcome.lag_s,
                base_lag=outcome.lag_s,
            )
    except Exception as exc:  # noqa: BLE001 - 离线诊断失败不影响主结果
        LOGGER.warning("离线延时分析失败：%s", exc)
    if report["lag_sweep"]:
        print("延时敏感性  :（同一批检测按不同 telemetry_lag 重算，看任务结果/分井误差）")
        for row in report["lag_sweep"]:
            print(
                "  lag %.2f s → 结果 %s，误差 %s m；分井 %s"
                % (
                    row["lag_s"],
                    row["selected_code"],
                    row["selected_error_m"],
                    row["code_errors_m"],
                )
            )
    calibrated_lag, calibration = calibrate_lag(report["lag_sweep"])
    report["lag_calibrated_s"] = calibrated_lag
    report["lag_calibration"] = calibration
    if calibrated_lag is not None:
        row = next((item for item in report["lag_sweep"] if item["lag_s"] == calibrated_lag), None)
        print(
            "链路延时自标定: %.2f s（标定用其余天井均方误差 %s m；留出的 %s 号只做验证，误差 %s m）"
            % (
                calibrated_lag,
                calibration["calibration_score_m"],
                calibration["holdout_code"],
                calibration["holdout_error_m"],
            )
        )
        if row is not None:
            print(
                "  校准后的任务结果误差: %s m（编号 %s，真值 (%s)）"
                % (
                    row["selected_error_m"],
                    row["selected_code"],
                    report["truth_ned"].get(str(row["selected_code"])),
                )
            )
    if report.get("reverse"):
        print("反向诊断    :", json.dumps(report["reverse"], ensure_ascii=False))

    (outcome.work / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("报告        : %s" % (outcome.work / "report.json"))
    print("====================\n")
    return report


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(
    *,
    work_dir: str = WORK_DIR,
    lag_s: float = TELEMETRY_LAG_S,
    record: bool = RECORD,
    recon_timeout_s: float = RECON_TIMEOUT_S,
    video_wait_s: float = VIDEO_WAIT_S,
    **config_overrides,
) -> int:
    """跑一次 SITL 侦查精度测试（**完全走标准流程**）。

    ``config_overrides`` 原样交给 :func:`build_config`。流程与生产入口
    （``examples/full_mission.py``）同一套：``MissionRunner`` 状态机 + 真 ``Preflight``
    + ``PerceptionWorker``（OCR 进程池）+ ``DryRunController``（SITL 没有 gripper）+
    ``FlightRecorder``（五个记录文件 + ``drops.jsonl`` 全写）。本脚本只扮演**操作手**：
    上传/启动"起飞任务"与"侦查航线"两条任务，其余交给状态机（RECON 监视 →
    HOLD_PROCESS 出目标 → OVERFLY 飞掠 + 投放 → LAND 降落 → DONE）。
    """
    from airdrop import (
        AlignmentBuffer,
        AlignmentWriter,
        BallisticsModel,
        DroneController,
        DryRunController,
        FlightRecorder,
        FrameTelemetryAligner,
        Hm30VideoSource,
        MavsdkThread,
        MissionRunner,
        MissionState,
        Preflight,
        ReleaseJudge,
        capacity_for,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    sdp = write_stream_sdp(work / STREAM_SDP_NAME)
    calib = write_sim_calib(work / "camera_calib_sim.json", lag_s=lag_s)
    config = build_config(
        calib_file=str(calib), recon_timeout_s=recon_timeout_s, **config_overrides
    )

    # ---- 与生产入口逐件对齐：遥测 → 记录 → 控制器 → 图传 → 感知 → 判据 → 状态机 ----
    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)
    recorder = FlightRecorder(config) if record else None
    on_event = (lambda kind, data: recorder.events.emit(kind, **data)) if recorder else None
    on_drop = (
        (lambda record_: recorder.drops.append(record_))  # noqa: PLW0108 - 延迟取 recorder
        if recorder
        else None
    )
    controller = DryRunController(
        DroneController.from_config(config, thread, on_event=on_event), on_event=on_event
    )

    buffer = AlignmentBuffer(capacity_for(30.0, 240.0), storage="jpeg")
    source = Hm30VideoSource(
        replace(
            config.video,
            url=str(sdp),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            telemetry_lag=lag_s,
            # SDP 里写了 udp/rtp，ffmpeg 需要放开这两条协议才能收
            extra_input_args=("-protocol_whitelist", "file,udp,rtp"),
        )
    )
    writer = AlignmentWriter(
        buffer,
        FrameTelemetryAligner(
            broker,
            max_wait=config.align.max_wait,
            mode=config.align.mode,
            on_timeout=config.align.on_timeout,
            max_extrapolation=config.align.max_extrapolation,
        ),
    )
    source.add_sink(writer)

    perception = PerceptionSetup.build(
        config,
        buffer,
        # 检出写进飞行记录（detections.jsonl）——与生产入口一致
        on_detection=(
            (lambda detection: recorder.detections.append(detection))  # noqa: PLW0108 - 同上
            if recorder
            else None
        ),
    )
    judge = ReleaseJudge(
        config.drop,
        BallisticsModel(config.ballistics),
        overfly_heading_deg=config.overfly.heading_deg,
        on_event=on_event,
    )
    camera = config.camera.load_model()
    preflight = Preflight(
        config.preflight,
        model_loaders={
            "detector": perception.worker.detector.load,
            "ocr": perception.worker.start,  # 起 OCR 进程池（SITL 里加载约 15~20 s）
            "camera": lambda: f"{config.camera.calib_file}（模拟相机：内参/外参按模型推导）",
        },
        video_source=source,  # 真图传源：读 stats.frames 计数、latest() 取最新帧
        camera_model=camera,  # 核对画面分辨率是否与模型一致
        on_event=on_event,
    )
    runner = MissionRunner(
        config,
        controller,
        broker,
        target_result=perception.source.result,
        target_busy=perception.source.busy,
        health_check=perception.worker.raise_if_failed,
        finish_recon=perception.worker.finish_input,
        release_judge=judge,
        on_event=on_event,
        on_drop=on_drop,
        preflight=preflight,
    )
    original_params: dict[str, int] = {}
    monitor: PerformanceMonitor | None = None
    exit_code = 0

    # ---- 操作手步骤：地面状态 → 起飞 → 上传并启动侦查航线 → 交给状态机 ----
    try:
        thread.connect()
        LOGGER.info("已连接 SITL：%s", config.telemetry.system_address)

        def ready() -> bool:
            snapshot = broker.get_snapshot()
            return (
                snapshot.is_valid()
                and snapshot.north_m is not None
                and snapshot.origin_latitude_deg is not None
                and snapshot.quaternion_w is not None
            )

        if not _wait_for(ready, READY_TIMEOUT_S, "遥测/原点就绪"):
            snapshot = broker.get_snapshot()
            LOGGER.error(
                "等不到有效遥测（位置/姿态/原点）——SITL 起来了吗？当前快照：north=%s roll=%s qw=%s",
                snapshot.north_m,
                snapshot.roll_deg,
                snapshot.quaternion_w,
            )
            return 2

        # 测试期间把数据链 failsafe 关掉（SITL 没有操作手盯链路），收尾还原
        original_params.update(zero_test_params(thread))
        LOGGER.info("SITL 测试参数已置 0（原值 %s；跑完还原）", original_params)
        if read_param(thread, "MIS_TKO_LAND_REQ") != 0:
            LOGGER.warning(
                "MIS_TKO_LAND_REQ != 0：只带起飞的侦查任务会被 PX4 拒掉"
                "（查 sim/install_px4.sh 是否装好机型）"
            )
        if read_param(thread, "FW_LND_USETER") != 0:
            LOGGER.warning(
                "FW_LND_USETER != 0：SITL 机型没有测距传感器，降落会在进入降落段 10 s 后"
                " abort、在落点上方 30 m 无限盘旋（查 sim/airframes/4007_gz_rc_cessna_down_cam）"
            )

        perception.worker.start()
        source.start()
        if not source.wait_ready(video_wait_s):
            LOGGER.error(
                "等不到第一帧画面（%.0fs）：检查 SITL 是否在推 127.0.0.1:%d（%s）",
                video_wait_s,
                STREAM_PORT,
                source.stats.last_error,
            )
            return 3
        LOGGER.info("视频流已就绪：%s", source.stats)
        if recorder is not None:
            recorder.start(broker=broker, buffer=buffer)
            LOGGER.info("飞行记录：%s", recorder.flight_dir)

        monitor = PerformanceMonitor(perception, source, writer)
        monitor.start()

        exit_code = operator_takeoff(config, controller, thread, broker) or 0
        if exit_code:
            return exit_code
        operator_start_recon(config, controller)

        state = runner.run()
        LOGGER.info(
            "任务结束：%s（上传 %s 次、投放 %s 次、错误 %s）",
            state,
            runner.stats.uploads,
            runner.stats.releases,
            runner.stats.errors,
        )
        if state is not MissionState.DONE:
            exit_code = 7
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("操作手中断")
        runner.stop("operator_interrupt")
    except Exception:
        LOGGER.exception("测试异常终止")
        exit_code = 1
        runner.stop("exception")
    finally:
        source.stop()
        remaining = perception.drain(120.0)
        if remaining:
            LOGGER.warning("收尾时还有 %d 帧/请求没处理完（结果只按已处理的算）", remaining)
        perception.stop()
        perf = [] if monitor is None else monitor.stop()
        # 收尾补抽：HOLD_PROCESS 之后感知还可能产出结果（队列无界、不丢弃），
        # 这里把剩下的抽干并记账，报告里能看出"到底有没有漏"。
        pumped = perception.source.pump()
        result = perception.tracker.result()
        stats = perception.tracker.stats
        worker_stats = perception.worker.stats
        LOGGER.info(
            "收尾补抽 %d 条结果（队列剩 %d；submitted=%d results=%d 读号=%d 去重挡下=%d）",
            pumped,
            perception.worker.queued_results,
            worker_stats.submitted,
            worker_stats.results,
            worker_stats.with_code,
            worker_stats.deduped,
        )
        video_stats = source.stats
        writer_stats = writer.stats()
        restore_params(thread, original_params)
        if recorder is not None:
            recorder.stop()
        # 记录器停了就固定 flight_dir（start() 之前它还不存在，所以只能在这儿取）
        detections_path = (
            None if recorder is None else Path(recorder.flight_dir) / "detections.jsonl"
        )
        # 报告必须在**会话还活着的时候**出：lag 扫描要按检测的拍摄时刻查 broker 的
        # 遥测历史（get_snapshot_at），原点也要取快照字段——而紧接着的 thread.stop()
        # 会让 _run_session 的 finally 调 broker.reset()，把历史和原点**一起清空**。
        # 后果实测过（2026-09-30）：任务好好跑完 DONE，报告这一步却先因 origin=None
        # 报 TypeError 崩掉；就算原点修好，lag 扫描也会因为历史被清空而每个 lag 都
        # 返回 0 点，整节精度分析静默作废。
        try:
            _report(
                _RunOutcome(
                    work=work,
                    sdp=sdp,
                    calib=calib,
                    lag_s=lag_s,
                    flown_s=None,
                    result=result,
                    stats=stats,
                    perception=perception,
                    video_stats=video_stats,
                    writer_stats=writer_stats,
                    detail=_load_detail(detections_path) if detections_path else [],
                    broker=broker,
                    flight_dir=None if recorder is None else Path(recorder.flight_dir),
                    runner_stats={
                        "state": str(runner.state),
                        "uploads": runner.stats.uploads,
                        "releases": runner.stats.releases,
                        "aborts": runner.stats.aborts,
                        "errors": runner.stats.errors,
                        "history": [str(record.to_state) for record in runner.history],
                        "transitions": [record.as_dict() for record in runner.history],
                        "ended_at": time.time(),
                    },
                    perf=perf,
                    # 原点取 runner 的权威副本（见 _origin_lla）：broker 快照在
                    # thread.stop() 之后已被 reset() 清空，那时再取就只剩 None 了。
                    origin=(
                        None
                        if runner.origin is None
                        else (
                            float(runner.origin.lon_deg),
                            float(runner.origin.lat_deg),
                            float(runner.origin.alt_m),
                        )
                    ),
                    tracker=perception.tracker,
                    config=config,
                )
            )
        except Exception:
            # 报告失败不该连会话释放一起拖掉：记异常 + 非零退出，thread.stop() 照走
            LOGGER.exception(
                "出报告失败（任务结果仍已落在 events.jsonl 与飞行目录里，可事后重放分析）"
            )
            exit_code = exit_code or 1
        thread.stop()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
