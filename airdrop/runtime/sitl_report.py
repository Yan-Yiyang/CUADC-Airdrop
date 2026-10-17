"""SITL JSON 精度报告生成。"""

from __future__ import annotations

import json
import logging
from typing import Any

from .sitl_analysis import (
    _analysis_broker,
    _analyze_with_retry,
    _cluster_error,
    _origin_lla,
    _performance_summary,
    _report_heading_and_errors,
    _reverse_diagnostic,
    _roll_stats,
    _RunOutcome,
    _select_by_code,
    _state_durations,
    _well_truth_ned,
    calibrate_lag,
    lag_sweep,
)
from .sitl_geometry import _DEFAULT_SCENE

LOGGER = logging.getLogger("sitl_recon")


def _report(outcome: _RunOutcome) -> dict[str, Any]:  # noqa: PLR0915
    """出报告：目标误差（任务结果 + 分井）、延时敏感性、反向诊断；并落 report.json。"""
    origin_lla = _origin_lla(outcome, _analysis_broker(outcome))
    result = outcome.result
    stats = outcome.stats
    video_stats = outcome.video_stats
    writer_stats = outcome.writer_stats
    perception = outcome.perception
    scene = getattr(outcome, "scene", None) or _DEFAULT_SCENE
    report: dict[str, Any] = {
        "world": scene.world_name,
        "simulation": {"seed": scene.seed, "round": scene.round_no, "origin_lla": scene.origin_lla},
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
            str(code): [round(value, 2) for value in _well_truth_ned(code, origin_lla, scene=scene)]
            for code in scene.wells_enu
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
                _cluster_error(selected, _well_truth_ned(selected.code, origin_lla, scene=scene)), 3
            ),
        }
    )
    for code in scene.wells_enu:
        cluster = _select_by_code(result.clusters, code)
        if cluster is not None:
            report["code_errors_m"][str(code)] = round(
                _cluster_error(cluster, _well_truth_ned(code, origin_lla, scene=scene)), 3
            )

    print("\n===== SITL 侦查精度测试 =====")
    print("世界        :", report["world"])
    print(
        f"视频        : 帧 {video_stats.frames}（丢 {video_stats.dropped}）→ 入缓冲 {writer_stats[0]}（跳过 {writer_stats[1]}）"
    )
    print(
        "感知        : 处理 {} 帧 → 检出 {} → 读号 {}（no_fix {} / side_mismatch {}）".format(
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
        "任务流程    : {}（上传 {} / 投放 {} / ABORT {} / 错误 {}）；轨迹 {}".format(
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
            "状态时长    : {}".format(
                " → ".join(
                    "{} {:.1f}s".format(row["state"], row["seconds"])
                    for row in report["mission"]["state_seconds"]
                )
            )
        )
    performance = report["performance"]
    if performance:
        print(
            "性能        : 检测 {} fps|OCR 入 {} / 出 {} 次/s"
            "（请求 {}、去重挡下 {}；峰值积压 {}）|缓冲峰值落后 {} 帧"
            "|内存峰值 {} + 子进程 {} MB".format(
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
        "扫掠姿态    : |roll| 均值 {}°/最大 {}°（大倾角照测）".format(
            report["roll"].get("mean_abs_deg"), report["roll"].get("max_abs_deg")
        )
    )

    # 逐观测误差 + 航向一致性：把"精度被姿态估计吃掉"这件事摆在明面上（口径见 HEADING_TOL_DEG）
    report.update(
        _report_heading_and_errors(
            outcome.detail, origin_lla=origin_lla, calib_file=str(outcome.calib), scene=scene
        )
    )

    print(
        "目标点      : 类 {} 个 + 噪声 {}（共 {} 点，带编号 {}）".format(
            len(result.clusters), len(result.noise), stats.get("points"), stats.get("labeled")
        )
    )
    for cluster in result.clusters:
        code = cluster.code
        truth = _well_truth_ned(code, origin_lla, scene=scene) if code in scene.wells_enu else None
        error = None if truth is None else _cluster_error(cluster, truth)
        print(
            "  编号 {} @ NED ({:8.2f}, {:8.2f})  {:5d} 点  置信 {:.2f}  真值 ({})  误差 {}".format(
                code,
                cluster.north_m,
                cluster.east_m,
                cluster.count,
                cluster.confidence,
                "{:.1f}, {:.1f}".format(*truth) if truth else "—",
                f"{error:.2f} m" if error is not None else "—",
            )
        )
    if result.selected is not None:
        error = _cluster_error(
            result.selected, _well_truth_ned(result.selected.code, origin_lla, scene=scene)
        )
        print(
            f"任务结果    : 编号 {result.selected.code} @ ({result.selected.north_m:.2f}, {result.selected.east_m:.2f})，误差 {error:.2f} m（要求 < 2 m）"
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
                scene=scene,
            )
            report["reverse"] = _reverse_diagnostic(
                outcome.detail,
                sweep_broker,
                calib_file=str(outcome.calib),
                code=scene.holdout_code,
                lag_s=outcome.lag_s,
                base_lag=outcome.lag_s,
            )
    except Exception as exc:  # noqa: BLE001 - 离线诊断失败不影响主结果
        LOGGER.warning("离线延时分析失败：%s", exc)
    if report["lag_sweep"]:
        print("延时敏感性  :（同一批检测按不同 telemetry_lag 重算，看任务结果/分井误差）")
        for row in report["lag_sweep"]:
            print(
                "  lag {:.2f} s → 结果 {}，误差 {} m；分井 {}".format(
                    row["lag_s"],
                    row["selected_code"],
                    row["selected_error_m"],
                    row["code_errors_m"],
                )
            )
    calibrated_lag, calibration = calibrate_lag(
        report["lag_sweep"], holdout_code=scene.holdout_code
    )
    report["lag_calibrated_s"] = calibrated_lag
    report["lag_calibration"] = calibration
    if calibrated_lag is not None:
        row = next((item for item in report["lag_sweep"] if item["lag_s"] == calibrated_lag), None)
        print(
            "链路延时自标定: {:.2f} s（标定用其余天井均方误差 {} m；留出的 {} 号只做验证，误差 {} m）".format(
                calibrated_lag,
                calibration["calibration_score_m"],
                calibration["holdout_code"],
                calibration["holdout_error_m"],
            )
        )
        if row is not None:
            print(
                "  校准后的任务结果误差: {} m（编号 {}，真值 ({})）".format(
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
