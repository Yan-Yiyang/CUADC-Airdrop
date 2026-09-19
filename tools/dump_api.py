"""自省导出接口清单：把全部公开签名与默认值写成一份 Markdown。

用法（命令行解析在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run dump-api
    ./.venv/Scripts/python.exe -m airdrop.run dump-api --out docs/api_reference.md

为什么要有它
------------
docs/handbook.md（项目手册）里所有签名与默认值都声称"来自机器自省、不是手抄"。
这个工具就是那句话的出处：它遍历 airdrop 各子包的 __all__，把"定义在本模块里"的
类/函数/常量的真实签名、dataclass 字段默认值、方法列表导出成 docs/api_reference.md，
供人工逐行比对，或给审查者当一份"这个包到底对外承诺了什么"的清单。

⚠ 生成物是派生产物：它随代码变，不随文档变。默认写进 docs/ 但不进 git
（.gitignore 已忽略），要核对时现跑一次即可；真正需要长期维护的是
docs/handbook.md，一致性由 tools/check_docs.py 把关。
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
OUTPUT_PATH = Path("docs/api_reference.md")

#: 要导出的模块（顺序即输出顺序；airdrop 顶层单独处理）
MODULES: tuple[str, ...] = (
    "airdrop.config",
    "airdrop.telemetry.models",
    "airdrop.telemetry.broker",
    "airdrop.telemetry.mavsdk_thread",
    "airdrop.telemetry.controller",
    "airdrop.video.source",
    "airdrop.video.align",
    "airdrop.video.buffer",
    "airdrop.record.recorder",
    "airdrop.record.replay",
    "airdrop.perception.detector",
    "airdrop.perception.cropproc",
    "airdrop.perception.number",
    "airdrop.perception.ocr_worker",
    "airdrop.perception.pipeline",
    "airdrop.perception.models",
    "airdrop.georef.camera",
    "airdrop.georef.project",
    "airdrop.georef.geo",
    "airdrop.targeting.models",
    "airdrop.targeting.cluster",
    "airdrop.ballistics.model",
    "airdrop.ballistics.release",
    "airdrop.ballistics.drops",
    "airdrop.ballistics.fit",
    "airdrop.mission.states",
    "airdrop.mission.planner",
    "airdrop.mission.targets",
    "airdrop.mission.runner",
)

#: 另外附一份"模块级常量"清单（产物文件名、取值域等）
CONSTANT_MODULES: tuple[str, ...] = (
    "airdrop.record.recorder",
    "airdrop.record.replay",
    "airdrop.ballistics.drops",
    "airdrop.ballistics.release",
    "airdrop.ballistics.model",
    "airdrop.mission.planner",
    "airdrop.mission.targets",
    "airdrop.mission.states",
    "airdrop.video.source",
    "airdrop.video.buffer",
    "airdrop.perception.cropproc",
    "airdrop.config",
)


def _summary(obj: Any) -> str:
    """docstring 第一行（没有就是空串）。"""
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0] if doc else ""


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except TypeError, ValueError:
        return "(?)"


def _default_of(field: dataclasses.Field) -> str:
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return f"factory={field.default_factory()!r}"  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - 工厂函数可能需要参数，取不到就标 <?>
            return "factory=<?>"
    return "<required>"


def _describe(name: str, obj: Any) -> list[str]:
    """一个公开名字 → Markdown 行。"""
    out: list[str] = []
    if inspect.isclass(obj):
        bases = ", ".join(base.__name__ for base in obj.__bases__ if base is not object)
        out.append(f"- **class `{name}`**（{bases or 'object'}）— {_summary(obj)}")
        if dataclasses.is_dataclass(obj):
            out.append("  - 字段：")
            for field in dataclasses.fields(obj):
                out.append(f"    - `{field.name}: {field.type} = {_default_of(field)}`")
        for member_name, member in vars(obj).items():
            if member_name.startswith("_") or inspect.isclass(member):
                continue
            if isinstance(member, property):
                out.append(f"  - `{member_name}` [property] — {_summary(member.fget)}")
            elif callable(member):
                out.append(f"  - `{member_name}{_signature(member)}` — {_summary(member)}")
    elif inspect.isfunction(obj):
        out.append(f"- **function `{name}{_signature(obj)}`** — {_summary(obj)}")
    else:
        out.append(f"- **constant `{name}` = `{obj!r}`**")
    return out


def exported_names(module: Any, module_name: str) -> list[str]:
    """模块的公开名字，且只保留定义在本模块里的（避免把 import 进来的重复列一遍）。"""
    names = getattr(module, "__all__", None) or [n for n in vars(module) if not n.startswith("_")]
    kept: list[str] = []
    for name in names:
        obj = getattr(module, name, None)
        if obj is None:
            continue
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and (
            getattr(obj, "__module__", module_name) != module_name
        ):
            continue
        kept.append(name)
    return sorted(kept)


def build_report() -> str:
    """生成整份接口清单（Markdown 文本）。"""
    lines: list[str] = [
        "# `airdrop` 接口清单（自省生成）",
        "",
        "> ⚠ **本文件由 `tools/dump_api.py` 生成，不要手改**——改了下次生成就没了。",
        "> 这里只列**签名与默认值**；字段语义、失败语义与用法见 `docs/handbook.md`。",
        "> 生成命令：`./.venv/Scripts/python.exe -m tools.dump_api`",
        "",
    ]
    top = importlib.import_module("airdrop")
    lines += [
        "## `airdrop`（顶层包）",
        "",
        f"- `__version__` = `{top.__version__!r}`",
        f"- `__all__` 共 {len(top.__all__)} 个名字："
        + "、".join(f"`{name}`" for name in top.__all__),
        "",
    ]
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        names = exported_names(module, module_name)
        lines.append(f"## `{module_name}`（{len(names)} 个公开名字）")
        lines.append("")
        if _summary(module):
            lines += [f"模块摘要：{_summary(module)}", ""]
        for name in names:
            lines += _describe(name, getattr(module, name))
        lines.append("")

    lines += ["## 模块级常量", ""]
    for module_name in CONSTANT_MODULES:
        module = importlib.import_module(module_name)
        lines.append(f"### `{module_name}`")
        for name in sorted(n for n in vars(module) if n.isupper()):
            lines.append(f"- `{name}` = `{getattr(module, name)!r}`")
        lines.append("")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class DumpConfig:
    """导出接口清单用的输出路径（默认值 = 本文件顶部的常量）。"""

    output_path: Path = OUTPUT_PATH


def build_config(**overrides) -> DumpConfig:
    """按关键字覆盖派生一份导出配置（``dataclasses.replace``）。"""
    return replace(DumpConfig(), **overrides)


def main(**overrides) -> int:
    """写出接口清单；关键字与 :func:`build_config` 一致。返回进程退出码。"""
    settings = build_config(**overrides)
    report = build_report()
    output_path = Path(settings.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(
        f"已写出 {output_path}：{len(report.splitlines())} 行 / {len(report.encode('utf-8'))} 字节"
    )
    print("提示：该文件是派生产物（不进 git）。核对文档一致性请跑 python -m airdrop.run check-docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
