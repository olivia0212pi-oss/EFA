from __future__ import annotations

import math
from collections.abc import Sequence


def should_stop_mcmc(
    loglik_history: Sequence[float], min_steps: int = 2, epsilon: float = 0.01
) -> bool:
    if len(loglik_history) < min_steps + 1:
        return False
    recent = loglik_history[-(min_steps + 1) :]
    return all(
        abs(right - left) < epsilon for left, right in zip(recent, recent[1:], strict=False)
    )


def criticality_score(logprobs: Sequence[float], position: int) -> float:
    return -float(logprobs[position])


def acceptance_probability(
    current_base_loglik: float,
    proposal_base_loglik: float,
    current_proposal_loglik: float,
    proposal_proposal_loglik: float,
    beta: float,
) -> float:
    """Metropolis-Hastings acceptance for target p(x)^beta and proposal q(x)."""
    log_ratio = beta * (proposal_base_loglik - current_base_loglik)
    log_ratio += current_proposal_loglik - proposal_proposal_loglik
    return min(1.0, math.exp(min(0.0, log_ratio)))
