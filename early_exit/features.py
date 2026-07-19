from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FEATURE_NAMES = [
    "avg_logprob",
    "min_logprob",
    "answer_same_count",
    "answer_changed",
    "reasoning_length",
]


def count_consecutive_same(values: Sequence[str | None]) -> int:
    if not values:
        return 0
    count = 1
    for value in reversed(values[:-1]):
        if value != values[-1]:
            break
        count += 1
    return count


def make_features(
    logprobs_recent: Sequence[float],
    answer_history: Sequence[str | None],
    reasoning_length: int,
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
    }


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]

