"""标定采集示例：录一个标准飞行目录给 ``tools/calibrate.py`` 用。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run calibration-capture
    ./.venv/Scripts/python.exe -m airdrop.run calibration-capture --rtsp-url rtsp://... --telemetry-lag 0.18

本文件是纯库模块：顶部常量是默认值，build_config(**覆盖) / main(**kwargs) 按需传值；
命令行由 airdrop/run.py 解析，重依赖都在函数体内导入（--help 不加载它们）。

采集方法（"板子不动、飞机在动"）
--------------------------------
1. 棋盘格固定放在地面/桌面，板面平整，方格边长用尺量准（``BOARD_SQUARE_M``
   只是提醒——真正填进 ``tools/calibrate.py`` 的是你量出来的那个值）；
2. 飞机通电（飞控输出姿态）、相机正常拉流；
3. 手持飞机在棋盘格上方缓慢平移 + 旋转，让棋盘格全程可见，并且要有充分的
   旋转激励（至少绕两个不平行的轴转）——时间差互相关与手眼标定都靠它；
4. 录完按 Ctrl-C 结束，目录会打印出来，把它作为标定工具的输入
   （``python -m airdrop.run calibrate --flight <目录>``）。

建议：标定时把姿态流速率调高（``config.telemetry.attitude_rate_hz``，本示例用 50Hz），
姿态内插误差直接进外参。录出来的就是标准飞行目录（五个记录文件），标定脚本读它即可。
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from airdrop import Config

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
SYSTEM_ADDRESS: str | None = None  # None = 用 TelemetryConfig 的默认地址
RTSP_URL = None  # None = 使用 VideoConfig 的默认地址
TELEMETRY_LAG_S = 0.15  # 本次采集不要用旧标定值去改它：时间差正是要标的东西
ATTITUDE_RATE_HZ = 50.0  # 标定建议 ≥50Hz（姿态内差直接进外参）
POSITION_RATE_HZ = 20.0
BUFFER_SECONDS = 300.0  # 缓冲够长，免得采到一半被驱逐
BOARD_SQUARE_M = 0.025  # 只作提醒：真正用到的值填在 tools/calibrate.py 里
MAX_SECONDS = 180.0  # 采集上限（秒），到点自动停

LOGGER = logging.getLogger("calibration_capture")


def build_config(
    *,
    system_address: str | None = SYSTEM_ADDRESS,
    rtsp_url: str | None = RTSP_URL,
    telemetry_lag_s: float = TELEMETRY_LAG_S,
    attitude_rate_hz: float = ATTITUDE_RATE_HZ,
    position_rate_hz: float = POSITION_RATE_HZ,
) -> Config:
    """标定采集用的配置（姿态/位置速率拉高；其余用 :class:`Config` 默认值）。"""
    base = Config()
    telemetry = replace(
        base.telemetry,
        attitude_rate_hz=attitude_rate_hz,
        position_rate_hz=position_rate_hz,
        position_velocity_ned_rate_hz=position_rate_hz,
    )
    if system_address is not None:
        telemetry = replace(telemetry, system_address=system_address)
    video = base.video if rtsp_url is None else replace(base.video, url=rtsp_url)
    return Config(
        telemetry=telemetry,
        video=replace(video, telemetry_lag=telemetry_lag_s),
    ).validated()


def main(
    *,
    buffer_seconds: float = BUFFER_SECONDS,
    max_seconds: float = MAX_SECONDS,
    **config_overrides,
) -> int:
    """录一段标定素材；``config_overrides`` 原样交给 :func:`build_config`。"""
    # 重依赖在使用时才导入：`python -m airdrop.run calibration-capture --help` 不加载它们
    from airdrop import (
        AlignmentBuffer,
        AlignmentWriter,
        FlightRecorder,
        FrameTelemetryAligner,
        Hm30VideoSource,
        MavsdkThread,
        TelemetryBroker,
        capacity_for,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.info(
        "采集开始：棋盘格固定、手持飞机在其上方平移+旋转（方格边长 %.3fm，"
        "填进 tools/calibrate.py 的 BOARD_SQUARE_M）；最多采 %.0fs，Ctrl-C 提前结束",
        BOARD_SQUARE_M,
        max_seconds,
    )
    config = build_config(**config_overrides)

    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)
    source = Hm30VideoSource(config.video)
    aligner = FrameTelemetryAligner(broker, max_wait=config.align.max_wait, mode=config.align.mode)
    buffer = AlignmentBuffer(capacity_for(30.0, buffer_seconds), storage="jpeg")
    writer = AlignmentWriter(buffer, aligner)
    recorder = FlightRecorder(config)

    exit_code = 0
    try:
        thread.connect()
        recorder.start(broker=broker, buffer=buffer)
        source.add_sink(writer)
        source.start()
        if not source.wait_ready(timeout=30.0):
            LOGGER.error("视频没有出帧：%s", source.stats.last_error or "未知原因")
            return 1
        deadline = time.monotonic() + max_seconds
        while source.running and time.monotonic() < deadline:
            time.sleep(0.2)
        LOGGER.info("采集结束（%.0fs）", max_seconds)
    except KeyboardInterrupt:
        LOGGER.info("用户中断，正常收尾")
    finally:
        source.stop()
        recorder.stop()
        thread.stop()

    written, skipped = writer.stats()
    stats = source.stats
    LOGGER.info(
        "投递 %d 帧、跳过 %d 帧；对齐入库 %d 帧（取不到遥测跳过 %d 帧）",
        stats.frames,
        stats.skipped,
        written,
        skipped,
    )
    LOGGER.info(
        "素材目录：%s\n把它作为标定输入：\n"
        "    ./.venv/Scripts/python.exe -m airdrop.run calibrate --flight %s",
        recorder.flight_dir,
        recorder.flight_dir,
    )
    if stats.frames == 0:
        LOGGER.error("一帧都没收到，检查视频地址与相机（见 README 的视频链路一节）")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
