"""核验 docs/handbook.md 与代码是否一致（文档不许漂）。

用法（命令行解析在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run check-docs
    ./.venv/Scripts/python.exe -m tools.check_docs          # 等价写法

检查七件事
----------
1. 参数齐全：airdrop.config 里每个 dataclass 的每个字段都出现在手册里；
2. 公开 API 齐全：airdrop.__all__ 里的类/函数名都出现在手册里；
3. 引用可解析：手册里每个 `` airdrop.x.y `` 引用都能真正 import + getattr 到
   （这条专抓"文档里写了不存在的 API"这类幻觉）；
4. 产物文件名齐全：飞行目录七个文件、标定产物、反演输入输出等文件名都在手册里；
5. 入口与测试覆盖：airdrop.run 的每个子命令与 tests/*.py 都被手册覆盖；
6. 章节结构：目录 / 模块 / 参数 / 输出 / 流程 / 审查 六个主题都有；
7. 反查编造：形似 config 字段（*_m / *_s / *_hz …）但 config 里没有的名字。

⚠ 检查 3 在子进程里做：要核验 airdrop.video.buffer 这类引用就得真 import 它们
（会加载 cv2 / mavsdk），而本工具本身是被 airdrop.run check-docs 调起来的——
"帮助/工具入口不加载重库"这条约束因此要求把重活挪出本进程。语义与直接 import 完全一致。

退出码：0 = 全过；1 = 有缺口（缺口逐条打印，方便直接补文档）。
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import unquote, urlsplit

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
DOC_PATH = Path("docs/handbook.md")

#: 必须在手册里出现的产物/输入文件名
REQUIRED_FILES: tuple[str, ...] = (
    "flight.log",
    "telemetry.jsonl",
    "detections.jsonl",
    "events.jsonl",
    "drops.jsonl",
    "frames_index.jsonl",
    "config_snapshot.json",
    "camera_calib.json",
    "ballistics_fit.json",
    "impacts.jsonl",
    "impacts.csv",
    "frames/",
)

#: 必须在手册里出现的入口与测试文件
#: （子命令由 airdrop.run.SUBCOMMANDS 动态给出，见 main 的第 5 项检查）
REQUIRED_ENTRIES: tuple[str, ...] = (
    "tests/test_config.py",
    "tests/test_telemetry.py",
    "tests/test_alignment.py",
    "tests/test_buffer.py",
    "tests/test_video.py",
    "tests/test_recorder.py",
    "tests/test_replay.py",
    "tests/test_perception.py",
    "tests/test_perception_realdata.py",
    "tests/test_georef.py",
    "tests/test_targeting.py",
    "tests/test_ballistics.py",
    "tests/test_mission.py",
    "tests/test_e2e.py",
    "tests/test_fit.py",
    "tests/test_examples.py",
    "tests/test_runtime_factory.py",
    "tests/test_runtime_session.py",
    "tests/test_runtime_sitl.py",
    "tests/test_mission_transfer.py",
    "tests/test_qgcvideo_geometry.py",
    "tests/test_qgcvideo_observers.py",
    "tests/test_qgcvideo_state.py",
    "tests/test_qgcvideo_render.py",
    "tests/test_qgcvideo_publisher.py",
    "tests/test_qgcvideo_stream.py",
    "tests/test_qgcvideo_config.py",
    "tests/test_runtime_qgcvideo.py",
    "tests/test_handbook.py",
    "tests/test_plan.py",
    "tests/test_cli.py",
    "tests/test_tool_repositories.py",
    "tests/test_diagnostics.py",
    "tests/test_calibration_contract.py",
    "tests/test_sim_integration.py",
)

#: 手册必须覆盖的主题关键词
REQUIRED_TOPICS: tuple[str, ...] = ("目录", "模块", "参数", "输出", "流程", "审查")

_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\s*\)")
_REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
_LOCAL_PATH_PATTERNS = (
    re.compile(
        r"(?i)(?<![\w])[a-z]:[\\/](?!\.\.\.(?:[\\/]|$|[^\w]))"
        r"(?!(?:Windows|Program Files(?: \(x86\))?|ProgramData|Python\d*)(?:[\\/]|$))"
        r"[^\\/\s`<>\"']+(?:[\\/][^\\/\s`<>\"']+)*"
    ),
    re.compile(
        r"(?i)(?<![\w])/(?:home/[^/\s`<>\"']+|Users/[^/\s`<>\"']+|"
        r"mnt/[a-z]/[^/\s`<>\"']+|workspace/[^/\s`<>\"']+|"
        r"Volumes/[^/\s`<>\"']+)(?:/[^/\s`<>\"']+)*"
    ),
)
_LEGACY_NAME_RE = re.compile(r"(?i)2024v2")
_LEGACY_ENTRY_RE = re.compile(
    r"(?i)(?<![\w.])airdrop\.(?:alignment|buffer)(?!\w)|"
    r"(?<![\w./\\])(?:examples[/\\]sitl_mission\.py|tools[/\\]sitl_recon\.py|"
    r"drone[/\\](?:coordinates|target|cvproc)\.py)(?![\w.])"
)


def _git_index(root: Path) -> tuple[set[str], set[str]]:
    """读取仓库索引中的文件路径与 gitlink 路径。"""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    files: set[str] = set()
    gitlinks: set[str] = set()
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        path = raw_path.decode("utf-8", "surrogateescape").replace("\\", "/")
        files.add(path)
        if mode == b"160000":
            gitlinks.add(path)
    return files, gitlinks


def _tracked_documents(indexes: dict[Path, set[str]]) -> list[tuple[Path, str]]:
    """返回各仓库索引中跟踪的 Markdown 文档。"""
    documents: list[tuple[Path, str]] = []
    for repo, paths in indexes.items():
        for relative in sorted(paths):
            name = Path(relative).name.lower()
            if relative.lower().endswith(".md") or name in {"readme", "agents"}:
                path = repo / relative
                if path.is_file():
                    documents.append((path, path.read_text(encoding="utf-8", errors="replace")))
    return documents


def _markdown_body(text: str) -> str:
    """剔除围栏代码，避免把示例语法当作公开链接或路径声明。"""
    return re.sub(r"(?ms)^\s*(```|~~~).*?^\s*\1\s*$", "", text)


def _document_link_issues(
    indexes: dict[Path, set[str]], gitlinks: dict[Path, set[str]]
) -> list[str]:
    """检查跟踪文档链接是否指向 Git 索引中的发布内容。"""
    roots = sorted(indexes, key=lambda path: len(path.parts), reverse=True)
    issues: list[str] = []
    for document, raw_text in _tracked_documents(indexes):
        text = _markdown_body(raw_text)
        targets = [match.group(1).strip("<>") for match in _MARKDOWN_LINK_RE.finditer(text)]
        targets.extend(match.group(1).strip("<>") for match in _REFERENCE_LINK_RE.finditer(text))
        for target in targets:
            parsed = urlsplit(target)
            windows_absolute = re.match(r"^[A-Za-z]:[\\/]", target) is not None
            if (parsed.scheme and not windows_absolute) or parsed.netloc or not parsed.path:
                continue
            decoded = unquote(target if windows_absolute else parsed.path).replace("\\", "/")
            if decoded.startswith("/") or re.match(r"^[A-Za-z]:/", decoded):
                resolved = Path(decoded)
            else:
                resolved = (document.parent / decoded).resolve()
            repo = next(
                (root for root in roots if resolved == root or root in resolved.parents), None
            )
            if repo is None:
                issues.append(f"{document}: 链接目标不属于已索引仓库：{target}")
                continue
            relative = resolved.relative_to(repo).as_posix()
            indexed = indexes[repo]
            if (
                relative in indexed
                or relative == "."
                or any(path.startswith(f"{relative.rstrip('/')}/") for path in indexed)
            ):
                continue
            # 主仓内的 gitlink 是可发布入口；子仓内目标必须由子仓自己的索引证明。
            if relative in gitlinks.get(repo, set()):
                continue
            issues.append(f"{document}: 链接目标未被 Git 跟踪：{target}")
    return issues


def _python_public_text(path: Path) -> list[str]:
    """仅提取 Python 注释和 AST docstring，不扫描普通运行时字符串。"""
    source = path.read_text(encoding="utf-8", errors="replace")
    snippets = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return snippets
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    snippets.extend(doc for node in nodes if (doc := ast.get_docstring(node, clean=False)))
    return snippets


def _privacy_issues(indexes: dict[Path, set[str]]) -> list[str]:
    """检查文档及 Python 注释/docstring 中的本地路径和过期入口。"""
    issues: list[str] = []
    for path, text in _tracked_documents(indexes):
        if any(pattern.search(text) for pattern in _LOCAL_PATH_PATTERNS):
            issues.append(f"{path}: 包含本地绝对路径")
        if _LEGACY_NAME_RE.search(text):
            issues.append(f"{path}: 包含过期 2024v2 命名")
        if _LEGACY_ENTRY_RE.search(text):
            issues.append(f"{path}: 包含旧包/入口")
    for repo, paths in indexes.items():
        for relative in sorted(paths):
            if not relative.lower().endswith(".py"):
                continue
            path = repo / relative
            if not path.is_file():
                continue
            for snippet in _python_public_text(path):
                if any(pattern.search(snippet) for pattern in _LOCAL_PATH_PATTERNS):
                    issues.append(f"{path}: Python 注释/docstring 包含本地绝对路径")
                if _LEGACY_NAME_RE.search(snippet):
                    issues.append(f"{path}: Python 注释/docstring 包含过期 2024v2 命名")
                if _LEGACY_ENTRY_RE.search(snippet):
                    issues.append(f"{path}: Python 注释/docstring 包含旧包/入口")
    return issues


def _repository_hygiene(root: Path) -> tuple[list[str], list[str]]:
    """收集主仓及当前已初始化子仓的跟踪索引并执行公开文档检查。"""
    root = root.resolve()
    files, submodules = _git_index(root)
    indexes = {root: files}
    gitlinks = {root: submodules}
    errors: list[str] = []
    for relative in sorted(submodules):
        child = root / relative
        if not child.is_dir() or not (child / ".git").exists():
            continue
        try:
            child_files, child_links = _git_index(child)
        except RuntimeError as exc:
            errors.append(f"子仓索引无法读取 {relative}: {exc}")
            continue
        indexes[child.resolve()] = child_files
        gitlinks[child.resolve()] = child_links
    return _document_link_issues(indexes, gitlinks), [*_privacy_issues(indexes), *errors]


def _config_dataclasses() -> dict[str, type]:
    config = importlib.import_module("airdrop.config")
    return {
        name: getattr(config, name)
        for name in dir(config)
        if inspect.isclass(getattr(config, name))
        and dataclasses.is_dataclass(getattr(config, name))
        and getattr(config, name).__module__ == "airdrop.config"
    }


#: 在子进程里核验引用的探针（stdin 收 JSON 列表，stdout 回不可解析的那些）
_RESOLVE_PROBE = r"""
import importlib, json, sys

bad = []
for reference in json.loads(sys.stdin.read()):
    parts = reference.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for attr in parts[cut:]:
                target = getattr(target, attr)
        except AttributeError:
            bad.append(reference)
        break
    else:
        bad.append(reference)
print(json.dumps(bad))
"""


def _resolve_all(references: list[str]) -> tuple[list[str], str]:
    """批量核验引用；返回 ``(不可解析的引用, 错误说明)``。

    在子进程里 import：核验 ``airdrop.video.buffer`` 这类引用必然加载 cv2 / mavsdk，
    而本工具会被 ``airdrop.run check-docs`` 调起来——那条路径要求"CLI 进程不加载重库"。
    子进程继承工作目录（``python -c`` 会把 cwd 放进 ``sys.path``），语义与直接 import 一致。
    """
    probe = subprocess.run(
        [sys.executable, "-c", _RESOLVE_PROBE],
        input=json.dumps(references),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,  # 探针的失败信息由 stdout 回传，这里不看退出码
    )
    if probe.returncode != 0:
        return [], f"引用核验子进程退出码 {probe.returncode}：{probe.stderr.strip()[:200]}"
    try:
        return json.loads(probe.stdout.strip().splitlines()[-1]), ""
    except (ValueError, IndexError) as exc:  # pragma: no cover - 子进程被环境搞坏
        return [], f"引用核验子进程输出无法解析（{exc}）：{probe.stdout.strip()[:200]}"


@dataclass(frozen=True, slots=True)
class CheckDocsConfig:
    """核验用到的路径（默认值 = 本文件顶部的常量）。"""

    doc_path: Path = DOC_PATH


def build_config(**overrides) -> CheckDocsConfig:
    """按关键字覆盖派生一份核验配置（``dataclasses.replace``）。"""
    return replace(CheckDocsConfig(), **overrides)


def main(**overrides) -> int:
    """跑一遍全部检查；关键字与 :func:`build_config` 一致。返回进程退出码。"""
    settings = build_config(**overrides)
    doc_path = Path(settings.doc_path)
    # 控制台编码（Windows 上常是 GBK）装不下的字符不要抛出异常：打印非 ASCII 字符
    # 抛 UnicodeEncodeError 会把退出码变成 1——检查明明全过，却看起来像失败。
    # 结论行用 ASCII，同时给 stdout 加 replace 兜底。
    with contextlib.suppress(AttributeError, ValueError):
        # 非标准 stdout（已被重定向/包装）时放弃，不影响检查结果。
        # `reconfigure` 只在 TextIOWrapper 上有，类型桩里的 `sys.stdout` 是 TextIO，
        # 所以这里显式忽略"没有该属性"。
        sys.stdout.reconfigure(errors="replace")  # pyright: ignore[reportAttributeAccessIssue]
    if not doc_path.exists():
        print(f"找不到手册：{doc_path}", file=sys.stderr)
        return 1
    # ⚠ 必须读 doc_path，不能读模块常量 DOC_PATH：那会让 doc_path 变成"半个选项"——
    # 存在性检查用传入的路径、内容却用默认手册，最坏情况是"核验通过了但核验的不是
    # 你以为的那本"。main() 的 docstring 承诺关键字与 build_config 一致。
    text = doc_path.read_text(encoding="utf-8")
    failures: list[str] = []

    def report(title: str, missing: list[str]) -> None:
        print(f"[{'OK  ' if not missing else 'FAIL'}] {title}：缺 {len(missing)}")
        for item in missing[:30]:
            print(f"        - {item}")
        if missing:
            failures.append(title)

    # 1) config 全字段
    missing: list[str] = []
    field_count = 0
    tables = _config_dataclasses()
    for class_name, cls in tables.items():
        for field in dataclasses.fields(cls):
            field_count += 1
            if f"`{field.name}`" not in text and f" {field.name} " not in text:
                missing.append(f"{class_name}.{field.name}")
    report(
        f"config 字段齐全（{len(tables)} 个 dataclass / {field_count} 个字段）",
        sorted(set(missing)),
    )

    # 2) 顶层公开 API
    top = importlib.import_module("airdrop")
    missing = [
        f"airdrop.{name}"
        for name in top.__all__
        if not name.startswith("__") and not name.isupper() and f"`{name}`" not in text
    ]
    report(f"顶层公开 API 齐全（{len(top.__all__)} 个名字）", sorted(missing))

    # 3) airdrop.* 引用可解析（在子进程里真 import，避免本进程拉起 cv2/mavsdk）
    refs = sorted(set(re.findall(r"`(airdrop\.[A-Za-z_][\w.]*)", text)))
    bad, probe_error = _resolve_all(refs)
    if probe_error:
        print(f"[FAIL] 引用核验没能跑起来：{probe_error}")
        failures.append("引用核验子进程")
    report(f"文档里的 airdrop.* 引用可解析（共 {len(refs)} 个）", bad)

    # 4) 产物文件名
    report("产物/输入文件名齐全", [name for name in REQUIRED_FILES if name not in text])

    # 5) 入口与测试文件：子命令取自 airdrop.run 的注册表（加子命令就必须写进手册）
    # 只核验当前仓库实际存在的用例，兼容可选工具的独立组织。
    entry_missing = [name for name in REQUIRED_ENTRIES if name not in text and Path(name).exists()]
    report("测试文件被文档覆盖", entry_missing)
    from airdrop.run import SUBCOMMANDS  # 惰性导入：只有本检查需要它

    commands = sorted(SUBCOMMANDS)
    report(
        f"CLI 子命令被文档覆盖（共 {len(commands)} 个）",
        [name for name in commands if f"airdrop.run {name}" not in text],
    )

    # 6) 章节关键词
    report("必备章节关键词", [topic for topic in REQUIRED_TOPICS if topic not in text])

    # 7) 反查编造的参数名
    known = {f.name for cls in tables.values() for f in dataclasses.fields(cls)}
    suffixes = ("_m", "_s", "_hz", "_deg", "_px", "_m2", "_kg", "_m_s", "_percent", "_ms")
    suspects: list[str] = []
    for line in text.splitlines():
        if "config" not in line.lower():
            continue
        for token in re.findall(r"`([a-z][a-z0-9_]{3,})`", line):
            if token.endswith(suffixes) and token not in known:
                suspects.append(f"{token}  ← {line.strip()[:90]}")
    report("可疑参数名（形似 config 字段但 config 里没有）", sorted(set(suspects)))

    # 8) 发布文档链接必须落在仓库索引中，公开文本不得泄露私人路径/旧入口。
    try:
        link_issues, privacy_issues = _repository_hygiene(Path(__file__).resolve().parents[1])
    except RuntimeError as exc:
        link_issues, privacy_issues = [], [f"Git 索引读取失败：{exc}"]
    report("跟踪文档内部链接指向已跟踪发布文件", link_issues)
    report("公开文档与 Python 注释/docstring 无私人路径或旧命名", privacy_issues)

    print(
        f"\n手册：{doc_path} — {len(text.encode('utf-8'))} 字节 / {len(text.splitlines())} 行 / "
        f"{len(re.findall(r'^#{1,6} ', text, flags=re.M))} 个标题"
    )
    if failures:
        print(f"结论：有 {len(failures)} 项未通过（FAIL） -> {failures}")
        return 1
    print("结论：全部通过（OK）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
