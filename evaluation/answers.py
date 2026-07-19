from __future__ import annotations

import re
from typing import Any


def _balanced_braced_value(text: str, open_brace: int) -> str | None:
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index].strip()
    return None


def extract_answer(text: str) -> str | None:
    """Extract the final boxed value, including nested LaTeX braces."""
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for position in reversed(positions):
        open_brace = text.find("{", position)
        value = _balanced_braced_value(text, open_brace)
        if value is not None:
            return value

    final_match = re.search(
        r"(?:final answer|answer is|答案(?:是|为))\s*[:：]?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        return final_match.group(1).strip().rstrip("。.")
    return None


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\\(?:text|mathrm|textbf|textit)\{([^{}]*)\}", r"\1", text)
    replacements = {
        "−": "-",
        "–": "-",
        "\\left": "",
        "\\right": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "\\,": "",
        "\\!": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.strip("$ ").rstrip(".。")
    return re.sub(r"\s+", "", text)


def is_correct(prediction: str | None, gold: Any) -> bool:
    if prediction is None:
        return False
    try:
        from math_verify import parse, verify

        gold_parsed = parse(str(gold))
        pred_parsed = parse(prediction)
        if gold_parsed and pred_parsed and bool(verify(gold_parsed, pred_parsed)):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    return normalize_answer(prediction) == normalize_answer(gold)

