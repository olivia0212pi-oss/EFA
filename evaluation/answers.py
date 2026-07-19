from __future__ import annotations

import re
from typing import Any


def _balanced_braced_span(text: str, open_brace: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index].strip(), index + 1
    return None


# Text separating two \boxed{} values that we treat as one chained multi-part
# answer (e.g. "... are \(\boxed{-2}\) and \(\boxed{1}\).") rather than the
# model re-boxing an unrelated later answer. Inline-math delimiters around
# each \boxed{} (\(...\), $...$) are stripped before matching.
_CHAIN_GAP = re.compile(r"^[\s,;:.、和及]{0,20}(?:and|or)?[\s,;:.、和及]{0,20}$", re.IGNORECASE)
_INLINE_MATH_DELIM = re.compile(r"\\\(|\\\)|\$")


def _is_chain_gap(gap: str) -> bool:
    if len(gap) > 30:
        return False
    return bool(_CHAIN_GAP.match(_INLINE_MATH_DELIM.sub("", gap)))


def extract_answer(text: str) -> str | None:
    """Extract the final boxed value(s), including nested LaTeX braces.

    Multiple questions in MATH-500 ask for several values (e.g. "enter all
    such integers, separated by commas"), and models often answer with a
    chain of separate \\boxed{} at the very end instead of one combined
    box. Trailing boxed values joined only by short connector text (",",
    "and", ...) are chained together; earlier, unrelated \\boxed{} (e.g.
    an abandoned intermediate guess) are still ignored as before.
    """
    matches: list[tuple[str, int, int]] = []  # (value, start, end)
    for position in [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]:
        open_brace = text.find("{", position)
        span = _balanced_braced_span(text, open_brace)
        if span is not None:
            value, end = span
            matches.append((value, position, end))

    if matches:
        chain = [matches[-1][0]]
        for i in range(len(matches) - 1, 0, -1):
            gap = text[matches[i - 1][2] : matches[i][1]]
            if _is_chain_gap(gap):
                chain.append(matches[i - 1][0])
            else:
                break
        chain.reverse()
        return ", ".join(chain)

    final_match = re.search(
        r"(?:final answer|answer is|答案(?:是|为))\s*[:：]?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        return final_match.group(1).strip().rstrip("。.")
    return None


def _split_top_level(value: str) -> list[str]:
    """Split on commas that are not nested inside brackets/braces/parens."""
    parts = []
    depth = 0
    current: list[str] = []
    for char in value:
        if char in "{[(":
            depth += 1
            current.append(char)
        elif char in "}])":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def _bare_fraction_to_latex(match: re.Match[str]) -> str:
    num_sign, num, den_sign, den = match.groups()
    negative = (num_sign == "-") != (den_sign == "-")
    return f"{'-' if negative else ''}\\frac{{{num}}}{{{den}}}"


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\\(?:text|mathrm|textbf|textit)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"(-?)(\d+)/(-?)(\d+)", _bare_fraction_to_latex, text)
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

    gold_parts = _split_top_level(str(gold))
    if len(gold_parts) > 1:
        pred_parts = _split_top_level(prediction)
        gold_set = {normalize_answer(part) for part in gold_parts}
        pred_set = {normalize_answer(part) for part in pred_parts}
        if gold_set == pred_set:
            return True

    try:
        from math_verify import parse, verify

        gold_parsed = parse(str(gold))
        pred_parsed = parse(prediction)
        if gold_parsed and pred_parsed and bool(verify(gold_parsed, pred_parsed)):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    return normalize_answer(prediction) == normalize_answer(gold)

