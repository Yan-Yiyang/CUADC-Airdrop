"""OCR 编号纠错：把识别出的字符串收敛到 00–99。

目标编号的先验很强——它只会是 00~99 两位数字。这条先验足以纠正 OCR 的
大部分错误，而且不依赖任何模型：印刷数字被认成形状相近的字母（``O``/``l``/
``S``/``B``/``g``）是最常见的失败模式，直接映射回数字即可。

移植自旧版实现 ``drone/target.py`` 的 ``correct_ocr_number``（久经实测），
逻辑与阈值一字不改——这条规则是调出来的，没有理由重调。
"""

from __future__ import annotations

__all__ = ["OCR_CHAR_MAP", "correct_ocr_number"]

# 易混淆字符映射（OCR 纠错）——与旧版实现完全一致，不要随手加条目：
# 每多一条映射就多一类"把真数字改错"的风险，这张表是实测调出来的。
OCR_CHAR_MAP = {
    "o": "0",
    "O": "0",
    "Q": "0",
    "l": "1",
    "I": "1",
    "i": "1",
    "|": "1",
    "J": "1",
    "z": "2",
    "Z": "2",
    "s": "5",
    "S": "5",
    "b": "6",
    "g": "9",
    "q": "9",
}


def correct_ocr_number(raw: str | None) -> int | None:
    """把 OCR 原始串纠错成 00–99 的整数；无法判定返回 None。

    规则（顺序固定）：

    1. 按 :data:`OCR_CHAR_MAP` 做字符映射（``o``→0、``l``→1、``s``→5 …）；
    2. 过滤掉所有非数字字符；
    3. 去掉首尾的 ``'1'``（五边形边缘/装饰线常被认成 ``'1'``——实测
       ``'156'``/``'561'``/``'1561'`` 都是 ``56``）；
    4. 仍然超过两位 → 取末两位；只剩一位 → 前置补 0（``'7'``→``07``）。

    第 3 步是"去首尾 1"而不是"直接取末两位"：``'15'`` 这种两位结果里的 ``1``
    是真实数字，不能被砍掉；只有超过两位时首尾的 ``1`` 才判定为噪声。
    """
    if not raw:
        return None
    mapped = "".join(OCR_CHAR_MAP.get(char, char) for char in raw.strip())
    digits = "".join(char for char in mapped if char.isdigit())
    if not digits:
        return None
    if len(digits) > 2:
        while len(digits) > 2 and digits[0] == "1":
            digits = digits[1:]
        while len(digits) > 2 and digits[-1] == "1":
            digits = digits[:-1]
    if len(digits) > 2:
        digits = digits[-2:]
    elif len(digits) == 1:
        digits = "0" + digits
    return int(digits)
