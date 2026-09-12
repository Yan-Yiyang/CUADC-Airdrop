"""测试公共辅助：路径引导与控制台编码。

只做每个测试文件都需要的两件事：让 ``import airdrop`` 在任意工作目录下
都能找到包，以及让中文输出在 Windows 控制台不乱码。

测试风格遵循仓库既有约定：纯 assert + ``print("[ok] ...")``，不引入
任何测试框架（见 AGENTS.md 的"工程缺口"）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 仓库根目录加入 sys.path，这样从任意工作目录运行都能 import airdrop
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def use_utf8_console() -> None:
    """把标准输出切到 UTF-8，避免 Windows 控制台按 GBK 解码中文时报错/乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 输出被重定向等情况，忽略即可
            pass
