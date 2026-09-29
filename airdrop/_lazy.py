"""惰性导出的公共实现（PEP 562 模块级 ``__getattr__``）。

为什么需要它
------------
每个子包的 ``__init__`` 都把公开名字从叶子模块汇到包命名空间，于是"只想要一个
``Waypoint``"也会把 cv2 / mavsdk / torch 这些重依赖拉进进程——CLI（``airdrop/run.py``）
光是打一份 ``--help`` 就要等一整套 GPU/飞控栈加载完。

本模块给出统一做法：包的 ``__init__`` 只登记叶子模块名，属性在第一次访问时
才真正 import，解析结果随即写回包的命名空间（之后走正常属性查找，不再进 ``__getattr__``）。

用法::

    from . import _lazy

    _LEAVES = ("source", "align", "buffer")     # 顺序即查找顺序
    __getattr__ = _lazy.lazy_exports(__name__, _LEAVES, subpackages=())

约定
----
* 找不到就抛 ``AttributeError``（与普通模块一致，不做静默兜底）；
* 名字按 ``_LEAVES`` 的顺序在 ``vars(子模块)`` 里查——轻依赖的叶子要排在前面，
  否则"取个常量"也会顺带加载重库；
* 双下划线名字（``__wrapped__``、``__bases__`` 这类运行时/工具链探测）直接拒绝，
  免得一次 ``getattr`` 把全部叶子模块都 import 一遍。
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["lazy_dir", "lazy_exports"]


def _cache(package: str, name: str, value: Any) -> None:
    """把解析结果写回包的命名空间（下次访问不再进 ``__getattr__``）。"""
    module = sys.modules.get(package)
    if module is not None:
        setattr(module, name, value)


def lazy_exports(
    package: str,
    modules: Mapping[str, str] | tuple[str, ...],
    subpackages: tuple[str, ...] = (),
) -> Callable[[str], Any]:
    """为包生成 ``__getattr__``：属性第一次被访问时才 import 对应的子模块。

    参数
    ----
    package:
        包名（``__name__``），用于相对导入与回写缓存。
    modules:
        两种写法：

        * 精确映射 ``{"Config": "config", ...}``——按名字只 import 提供它的那个
          叶子模块（``airdrop/__init__.py`` 用这种：取 ``MissionState`` 不该顺带加载
          排在它前面的 cv2 / mavsdk 模块）；
        * 有序元组 ``("source", "align", "buffer")``——依次查 ``vars(子模块)``，
          第一个含该名字的模块胜出（子包用这种，叶子数量少、顺序一眼能看懂）。
    subpackages:
        以子包本身为属性的名字（``airdrop.video`` 这类）；命中就直接返回子包模块。
    """
    exact = modules if isinstance(modules, Mapping) else None
    ordered = tuple(modules) if exact is None else ()

    def __getattr__(name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"模块 {package!r} 没有属性 {name!r}")
        if name in subpackages:
            module = importlib.import_module(f".{name}", package)
            _cache(package, name, module)
            return module
        if exact is not None:
            module_name = exact.get(name)
            if module_name is None:
                raise AttributeError(f"模块 {package!r} 没有属性 {name!r}（不在惰性导出表里）")
            module = importlib.import_module(f".{module_name}", package)
            value = vars(module)[name]
        else:
            for module_name in ordered:
                module = importlib.import_module(f".{module_name}", package)
                values = vars(module)
                if name in values:
                    value = values[name]
                    break
            else:
                raise AttributeError(
                    f"模块 {package!r} 没有属性 {name!r}（已按顺序查过 {len(ordered)} 个子模块）"
                )
        _cache(package, name, value)
        return value

    return __getattr__


def lazy_dir(package: str) -> Callable[[], list[str]]:
    """为包生成 ``__dir__``：``dir(pkg)`` 里既有已解析的名字，也有 ``__all__``。"""

    def __dir__() -> list[str]:
        module = sys.modules.get(package)
        current = set(vars(module)) if module is not None else set()
        current |= set(getattr(module, "__all__", ()) or ())
        return sorted(current)

    return __dir__
