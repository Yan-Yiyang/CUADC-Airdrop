"""仅在调用工具或 SITL 集成时加载可选子模块，不安装依赖。"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PACKAGES = {
    "airdrop_sim": "airdrop-sim",
    "airdrop_calibration": "airdrop-calibration",
}


def load_tool_module(name: str) -> ModuleType:
    """优先使用已安装包，否则加载当前父仓固定的子模块。"""
    package = name.split(".", 1)[0]
    repository = PACKAGES[package]
    if package not in sys.modules and importlib.util.find_spec(package) is None:
        location = Path(__file__).resolve().parents[1] / "vendor" / repository / package
        init = location / "__init__.py"
        if not init.is_file():
            raise ModuleNotFoundError(
                f"缺少 {repository}。请初始化子模块：git submodule update --init "
                f"vendor/{repository}；也可使用已安装的独立工具包。不会自动安装依赖。",
                name=package,
            )
        spec = importlib.util.spec_from_file_location(
            package, init, submodule_search_locations=[str(location)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(package, None)
            raise
    return importlib.import_module(name)
