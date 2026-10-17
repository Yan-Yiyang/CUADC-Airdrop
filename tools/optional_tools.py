"""兼容父仓工具 CLI 的可选子模块加载入口。"""

from airdrop.runtime.tool_loader import PACKAGES, load_tool_module

__all__ = ["PACKAGES", "load_tool_module"]
