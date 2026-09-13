"""把外部模型权重取到工程内的 ``models/`` 目录。

为什么需要这个脚本
------------------
权重体积大（YOLO 6MB + PP-OCR 约 170MB），**不进 git**；但起飞前必须把模型
版本固定下来（否则某次飞行前模型悄悄换了，识别率变化无法归因），所以约定：
权重放在 ``models/`` 下，由本脚本从指定来源复制过来，路径写进 ``Config``。

用法（无命令行参数，改下面常量）::

    python -m tools.fetch_models

来源
----
* YOLO：``SOURCE_DIR/best2.pt`` → ``models/best2.pt``
  （实测可用的是 best2.pt；best.pt / best1.pt 对现目标形态零检出）
* PP-OCRv6：``SOURCE_DIR`` 里的 ``*.pth`` 与 ``*dict.txt`` → ``models/ppocr/``
* 方向分类（Cls）：同一目录下的 ``ch_ptocr_mobile_v2.0_cls_mobile.pth``（TORCH 版）
  与 ``ch_ppocr_mobile_v2.0_cls_mobile.onnx``（ONNX 版）→ ``models/ppocr/``；
  **文件名不能规范化**，理由见下面 ``OCR_FILES`` 的注释。

 ``SOURCE_DIR`` 默认指向 旧版 的安装位置（那里已下载好 TORCH 权重）。
换机器时改成你自己的素材路径即可。**本脚本只做复制，不联网**——要重新下载
TORCH 版 PP-OCR 权重，用 ``rapidocr`` 自己的下载器或 modelscope 取，
下完放到 ``SOURCE_DIR`` 再跑本脚本。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
SOURCE_DIR = Path(r"input/flight.dat")
# rapidocr 的 TORCH 权重通常在它的 site-packages/rapidocr/models 下
SOURCE_OCR_DIR = Path(
    r"input/flight.dat"
)
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
    # 方向分类（Cls）。两个名字**都不能改**：
    # * ``.onnx`` 用 RapidOCR 的正规模型名；
    # * ``.pth`` 必须叫 ``ch_ptocr_...``——RapidOCR 的 ``arch_config.yaml`` 里那条 cls
    #   架构被拼错了（``ptocr`` 而非 ``ppocr``），而 TORCH 引擎是**按文件名 stem 查架构**
    #   （``networks/main.py`` 的 ``_load_arch_config``），名字对不上直接
    #   ``ValueError: architecture ... is not in arch_config.yaml``。
    #   实测：同一个权重改名成 ``ch_ppocr_...pth`` 就加载不了，叫 ``ch_ptocr_...pth`` 才行。
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


def main() -> int:
    print(f"YOLO 权重源: {SOURCE_DIR}")
    print(f"OCR  权重源: {SOURCE_OCR_DIR}")
    print(f"目标目录   : {TARGET_DIR.resolve()}")
    print()
    total = copy_missing(SOURCE_DIR, TARGET_DIR, YOLO_FILES)
    total += copy_missing(SOURCE_OCR_DIR, TARGET_OCR_DIR, OCR_FILES)
    print(f"\n共复制 {total} 个文件")
    if total == 0:
        print("（都已在位，无需复制）")
    print("\n记得在 Config 里核对路径：")
    print("  perception.model_path = models/best2.pt")
    print("  perception.models_dir = models/ppocr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
