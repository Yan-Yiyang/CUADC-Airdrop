"""SITL 侦查精度测试（**可选**：要 WSL 里的 SITL 已经在跑，默认跳过）。

    ./.venv/Scripts/python.exe -m pytest tests/test_sitl_recon.py -m sitl -v

前置（本测试只**驱动**，不负责启动飞控/世界/图传）::

    bash sim/run_sitl.sh r2          # WSL 里起 PX4 SITL + Gazebo + 图传

断言（都取自 ``.sitl-recon-tmp/report.json``）：

* 状态机走到 ``DONE``——标准流程：起飞 → 侦查监视 → 空中出目标 → 飞掠投放 → 降落；
* 三个编号 94/12/56 全部识别，任务结果按"中位数"选中 **56**；
* 最终误差 < 2 m（原始口径与"自标定延时"口径各查一遍，留一验证的结果也要 < 2 m）。

SITL 没在跑就 ``skip``（不报失败）——避免"忘了起仿真"被当成代码坏了。
全流程分钟级，所以这条用例把超时放宽到 30 min，**而不是关掉兜底**（``timeout(0)``）：
探活那一步曾经因为 ``MavsdkThread.stop()`` 永久阻塞而永远回不来，把整轮全量测试挂死。
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / ".sitl-recon-tmp" / "report.json"

pytestmark = pytest.mark.sitl


def _sitl_is_up(timeout: float = 8.0) -> bool:
    """SITL 在跑吗：直接听 14540 上有没有数据，**不建 MAVSDK 会话**。

    刻意不用 ``MavsdkThread.connect()`` 探活：那要拉起 mavsdk_server 与 gRPC 通道，而
    "连不上再收尾"这条路径实测会卡死整个进程（gRPC poller 线程报
    ``Event loop is closed``，见 AGENTS.md 易错点 1）——探活只需要知道
    "14540 上有没有人在发心跳"。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", 14540))
    except OSError:
        # 端口被占：已经有别的会话/地面站在收，或者本来就没有 SITL——都不足以判定"可以跑"
        return False
    try:
        sock.settimeout(timeout)
        try:
            sock.recv(2048)
        except TimeoutError:
            return False
        return True
    finally:
        sock.close()


@pytest.mark.timeout(1800)  # 分钟级全流程：放宽到 30 min，但**不**像 timeout(0) 那样彻底关掉兜底
def test_sitl_recon_accuracy() -> None:
    if not _sitl_is_up():
        pytest.skip("SITL 没在跑（先 bash sim/run_sitl.sh r2）")

    from tools import sitl_recon

    exit_code = sitl_recon.main()
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    runner = report.get("runner_stats") or {}
    assert str(runner.get("state", "")).endswith("DONE"), f"任务没走完：{runner}"
    codes = {int(cluster["code"]) for cluster in report["clusters"] if cluster["code"] is not None}
    assert codes >= {94, 12, 56}, f"三个靶标没全识别：{codes}"
    selected = report["selected"]
    assert selected is not None and selected["code"] == 56, f"任务结果选错：{selected}"
    assert selected["error_m"] < 2.0, f"误差超 2 m：{selected}"
    calibration = report.get("lag_calibration") or {}
    holdout = calibration.get("holdout_error_m")
    assert holdout is None or holdout < 2.0, f"留一验证超 2 m：{calibration}"
    assert exit_code in (0, 7), f"退出码 {exit_code}"
