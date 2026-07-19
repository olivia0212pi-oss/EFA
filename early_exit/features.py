from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FEATURE_NAMES = [
    "avg_logprob",
    "min_logprob",
    "answer_same_count",
    "answer_changed",
    "reasoning_length",
    "deer_confidence",
    "entropy",
    "margin",
]

STATE_CORRECT_STABLE = "correct_stable"
STATE_CORRECT_UNSTABLE = "correct_unstable"
STATE_INCORRECT_RECOVERABLE = "incorrect_recoverable"
STATE_INCORRECT_TERMINAL = "incorrect_terminal"


def count_consecutive_same(values: Sequence[str | None]) -> int:
    if not values:
        return 0
    count = 1
    for value in reversed(values[:-1]):
        if value != values[-1]:
            break
        count += 1
    return count


def deer_confidence(trial_answer_logprobs: Sequence[float]) -> float:
    """Geometric-mean per-token probability of a forced trial-answer completion.

    Named after the DEER early-exit family, which reads model confidence off
    a forced "give the answer now" continuation rather than the ongoing
    reasoning tokens.
    """
    finite = np.asarray(trial_answer_logprobs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    return float(np.exp(finite.mean()))


def entropy_and_margin(top_logprobs: Sequence[float]) -> tuple[float, float]:
    """Entropy and top1-top2 margin from a token's top-k logprobs (descending)."""
    finite = [lp for lp in top_logprobs if np.isfinite(lp)]
    if not finite:
        return 0.0, 0.0
    probs = np.exp(np.asarray(finite, dtype=float))
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    margin = float(finite[0] - finite[1]) if len(finite) > 1 else float(-finite[0])
    return entropy, margin


def classify_state(current_correct: bool, final_correct: bool) -> str:
    """Four-way state used to reason about stop safety at a checkpoint.

    correct_stable: right now and stays right -> ideal stop point.
    correct_unstable: right now but the full trace later goes wrong -> the
        dangerous false-positive-early-exit case.
    incorrect_recoverable: wrong now but the full trace fixes itself -> must
        not stop here.
    incorrect_terminal: wrong now and stays wrong -> not a stop candidate.
    """
    if current_correct and final_correct:
        return STATE_CORRECT_STABLE
    if current_correct and not final_correct:
        return STATE_CORRECT_UNSTABLE
    if not current_correct and final_correct:
        return STATE_INCORRECT_RECOVERABLE
    return STATE_INCORRECT_TERMINAL


def make_features(
    logprobs_recent: Sequence[float],
    answer_history: Sequence[str | None],
    reasoning_length: int,
    deer_confidence_value: float = 0.0,
    entropy: float = 0.0,
    margin: float = 0.0,
) -> dict[str, float]:
    finite = np.asarray(logprobs_recent, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "avg_logprob": float(finite.mean()) if len(finite) else 0.0,
        "min_logprob": float(finite.min()) if len(finite) else 0.0,
        "answer_same_count": float(count_consecutive_same(answer_history)),
        "answer_changed": float(
            len(answer_history) >= 2 and answer_history[-1] != answer_history[-2]
        ),
        "reasoning_length": float(reasoning_length),
        "deer_confidence": float(deer_confidence_value),
        "entropy": float(entropy),
        "margin": float(margin),
    }


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]
