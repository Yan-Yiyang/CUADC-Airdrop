"""运行 tests/ 下的全部离线测试。

    ./.venv/Scripts/python.exe tests/run_all.py

不需要真实飞控或图传硬件；全部通过时退出码为 0。
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._harness import use_utf8_console  # noqa: E402

# 顺序有意义：MAVSDK 基础设施先跑（快、依赖少），图传在后（要起 ffmpeg）
MODULES = ("tests.test_telemetry", "tests.test_video")


def main() -> int:
    use_utf8_console()
    failed: list[str] = []
    for name in MODULES:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        try:
            module = importlib.import_module(name)
            code = module.main()
        except Exception:  # noqa: BLE001 - 测试失败要继续跑完剩下的模块
            traceback.print_exc()
            code = 1
        if code != 0:
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"测试套件失败：{', '.join(failed)}")
        return 1
    print(f"测试套件全部通过（{len(MODULES)} 个模块）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
