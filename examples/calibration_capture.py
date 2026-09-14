"""标定采集示例：录一个标准飞行目录给 ``tools/calibrate.py`` 用。

用法::

    python -m examples.calibration_capture

**改本文件顶部的常量，不带命令行参数**（项目约定）。

采集方法（"板子不动、飞机在动"）
--------------------------------
1. 棋盘格**固定**放在地面/桌面，板面平整，方格边长用尺量准（``BOARD_SQUARE_M``
   只是提醒——真正填进 ``tools/calibrate.py`` 的是你量出来的那个值）；
2. 飞机通电（飞控输出姿态）、相机正常拉流；
3. **手持飞机在棋盘格上方缓慢平移 + 旋转**，让棋盘格全程可见，并且要有充分的
   **旋转激励**（至少绕两个不平行的轴转）——时间差互相关与手眼标定都靠它；
4. 录完按 Ctrl-C 结束，目录会打印出来，把它填进 ``tools/calibrate.py`` 的
   ``FLIGHT_DIR``。

建议：标定时把姿态流速率调高（``config.telemetry.attitude_rate_hz``，本示例用 50Hz），
姿态内插误差直接进外参。录出来的就是标准飞行目录（五件套），标定脚本读它即可。
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace

from airdrop import (
    AlignmentBuffer,
    AlignmentWriter,
    Config,
    FlightRecorder,
    FrameTelemetryAligner,
    Hm30VideoSource,
    MavsdkThread,
    TelemetryBroker,
    capacity_for,
)

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
RTSP_URL = None                 # None = 用 VideoConfig 默认的 HM30 地址
TELEMETRY_LAG_S = 0.15          # 本次采集**不要**用旧标定值去改它：时间差正是要标的东西
ATTITUDE_RATE_HZ = 50.0         # 标定建议 ≥50Hz（姿态内差直接进外参）
POSITION_RATE_HZ = 20.0
BUFFER_SECONDS = 300.0          # 缓冲够长，免得采到一半被驱逐
BOARD_SQUARE_M = 0.025          # 只作提醒：真正用到的值填在 tools/calibrate.py 里
MAX_SECONDS = 180.0             # 采集上限（秒），到点自动停

LOGGER = logging.getLogger("calibration_capture")


def build_config() -> Config:
    base = Config()
    telemetry = replace(
        base.telemetry,
        attitude_rate_hz=ATTITUDE_RATE_HZ,
        position_rate_hz=POSITION_RATE_HZ,
        position_velocity_ned_rate_hz=POSITION_RATE_HZ,
    )
    video = (
        base.video
        if RTSP_URL is None
        else replace(base.video, url=RTSP_URL)
    )
    return Config(
        telemetry=telemetry,
        video=replace(video, telemetry_lag=TELEMETRY_LAG_S),
    ).validated()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.info(
        "采集开始：棋盘格固定、手持飞机在其上方平移+旋转（方格边长 %.3fm，"
        "填进 tools/calibrate.py 的 BOARD_SQUARE_M）；最多采 %.0fs，Ctrl-C 提前结束",
        BOARD_SQUARE_M,
        MAX_SECONDS,
    )
    config = build_config()

    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)
    source = Hm30VideoSource(config.video)
    aligner = FrameTelemetryAligner(broker, max_wait=config.align.max_wait, mode=config.align.mode)
    buffer = AlignmentBuffer(capacity_for(config.video.fps or 30.0, BUFFER_SECONDS), storage="jpeg")
    writer = AlignmentWriter(buffer, aligner)
    recorder = FlightRecorder(config)

    exit_code = 0
    try:
        thread.connect()
        recorder.start(broker=broker, buffer=buffer)
        source.add_sink(writer)
        source.start()
        if not source.wait_ready(timeout=30.0):
            LOGGER.error("图传没有出帧：%s", source.stats.last_error or "未知原因")
            return 1
        deadline = time.monotonic() + MAX_SECONDS
        while source.running and time.monotonic() < deadline:
            time.sleep(0.2)
        LOGGER.info("采集结束（%.0fs）", MAX_SECONDS)
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
        "素材目录：%s\n把它填进 tools/calibrate.py 的 FLIGHT_DIR，然后："
        "\n    ./.venv/Scripts/python.exe -m tools.calibrate",
        recorder.flight_dir,
    )
    if stats.frames == 0:
        LOGGER.error("一帧都没收到，检查图传地址与相机（见 README 的 HM30 一节）")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
