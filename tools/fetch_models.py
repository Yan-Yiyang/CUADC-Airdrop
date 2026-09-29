"""把外部模型权重取到工程内的 ``models/`` 目录。

为什么需要这个脚本
------------------
权重体积大（YOLO 6MB + PP-OCR 约 170MB），不进 git；但起飞前必须把模型
版本固定下来（否则某次飞行前模型悄悄换了，识别率变化无法归因），所以约定：
权重放在 ``models/`` 下，由本脚本从指定来源复制过来，路径写进 ``Config``。

用法（命令行解析在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run fetch-models --source-dir <YOLO 权重目录>
    ./.venv/Scripts/python.exe -m airdrop.run fetch-models \
        --source-dir <YOLO 权重目录> --source-ocr-dir <OCR 权重目录>

来源目录也可以先放进环境变量（命令行选项优先）::

    AIRDROP_SOURCE_DIR=<YOLO 权重目录>
    AIRDROP_SOURCE_OCR_DIR=<OCR 权重目录>

不单独指定 OCR 来源时按 YOLO 来源目录处理（两组权重常常放在一起）。

本文件是纯库模块：顶部常量是默认值，:func:`build_config` / :func:`main` 按需覆盖。

需要哪些模型文件
----------------
* YOLO 检测：``best2.pt`` → ``models/best2.pt``（单类 ``target``）
* PP-OCRv6（TORCH）：``*.pth`` 与 ``*dict.txt`` → ``models/ppocr/``
* 方向分类（Cls）：``ch_ppocr_mobile_v2.0_cls_mobile.onnx`` 与
  ``ch_ptocr_mobile_v2.0_cls_mobile.pth`` → ``models/ppocr/``（文件名保持原样）

权重来源需自备（仓库不带权重）。本脚本只做复制，不联网。
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path


# ----------------------------------------------------------------------
# 配置（改这里；也可由命令行选项 / 环境变量覆盖）
# ----------------------------------------------------------------------
def _env_path(name: str) -> Path | None:
    """从环境变量取一个可选目录；没设或为空时返回 None。"""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


#: YOLO 权重来源目录（环境变量 ``AIRDROP_SOURCE_DIR``；命令行 ``--source-dir`` 优先）
SOURCE_DIR = _env_path("AIRDROP_SOURCE_DIR")
#: OCR 权重来源目录（环境变量 ``AIRDROP_SOURCE_OCR_DIR``）；
#: rapidocr 的 TORCH 权重通常在它的 site-packages/rapidocr/models 下
SOURCE_OCR_DIR = _env_path("AIRDROP_SOURCE_OCR_DIR")
TARGET_DIR = Path("models")
TARGET_OCR_DIR = TARGET_DIR / "ppocr"

YOLO_FILES = ("best2.pt",)
OCR_FILES = (
    "PP-OCRv6_det_medium.pth",
    "PP-OCRv6_rec_medium.pth",
    "PP-OCRv6_det_small.pth",
    "PP-OCRv6_rec_small.pth",
    "PP-OCRv6_det_tiny.pth",
    "PP-OCRv6_rec_tiny.pth",
    "ppocrv6_dict.txt",
    "ppocrv6_tiny_dict.txt",
    # 方向分类（Cls）两个权重都要（ONNX 是默认引擎，TORCH 备选）；
    # 文件名保持 RapidOCR 原命名，不要改动（TORCH 引擎按文件名 stem 查架构）。
    "ch_ptocr_mobile_v2.0_cls_mobile.pth",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
)


def copy_missing(source: Path, target: Path, names: tuple[str, ...]) -> int:
    copied = 0
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = source / name
        dst = target / name
        if not src.is_file():
            print(f"  [跳过] 源缺失: {src}")
            continue
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            print(f"  [已有] {dst}（{dst.stat().st_size / 1e6:.1f} MB）")
            continue
        shutil.copy2(src, dst)
        print(f"  [复制] {src.name} -> {dst}（{dst.stat().st_size / 1e6:.1f} MB）")
        copied += 1
    return copied


@dataclass(frozen=True, slots=True)
class FetchConfig:
    """复制权重用到的路径（默认值 = 本文件顶部的常量）。

    ``source_dir`` 为 None 表示"还没指定来源"，:func:`main` 会打印用法并以
    非零退出；``source_ocr_dir`` 不指定时按 ``source_dir`` 处理。
    """

    source_dir: Path | None = SOURCE_DIR
    source_ocr_dir: Path | None = SOURCE_OCR_DIR
    target_dir: Path = TARGET_DIR
    yolo_files: tuple[str, ...] = YOLO_FILES
    ocr_files: tuple[str, ...] = OCR_FILES


def build_config(**overrides) -> FetchConfig:
    """按关键字覆盖派生一份取权重配置（``dataclasses.replace``）。"""
    return replace(FetchConfig(), **overrides)


def main(**overrides) -> int:
    """复制缺失的权重；关键字与 :func:`build_config` 一致。返回进程退出码。"""
    settings = build_config(**overrides)
    source_dir = settings.source_dir
    if source_dir is None:
        print(
            "未指定权重来源：用 --source-dir <目录> 指定，或先设环境变量 AIRDROP_SOURCE_DIR。",
            file=sys.stderr,
        )
        return 1
    source_ocr_dir = settings.source_ocr_dir or source_dir
    if settings.source_ocr_dir is None:
        print(f"（未单独指定 OCR 来源，按 {source_ocr_dir} 处理）")
    print(f"YOLO 权重源: {source_dir}")
    print(f"OCR  权重源: {source_ocr_dir}")
    print(f"目标目录   : {settings.target_dir.resolve()}")
    print()
    total = copy_missing(Path(source_dir), Path(settings.target_dir), tuple(settings.yolo_files))
    total += copy_missing(
        Path(source_ocr_dir),
        Path(settings.target_dir) / "ppocr",
        tuple(settings.ocr_files),
    )
    print(f"\n共复制 {total} 个文件")
    if total == 0:
        print("（都已在位，无需复制）")
    print("\n记得在 Config 里核对路径：")
    print("  perception.model_path = models/best2.pt")
    print("  perception.models_dir = models/ppocr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
