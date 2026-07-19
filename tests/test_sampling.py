import math

import pytest

from sampling.utils import acceptance_probability, criticality_score, should_stop_mcmc


def test_temperature_one_reduces_to_power_acceptance() -> None:
    probability = acceptance_probability(
        current_base_loglik=-10,
        proposal_base_loglik=-11,
        current_proposal_loglik=-10,
        proposal_proposal_loglik=-11,
        beta=2,
    )
    assert probability == pytest.approx(math.exp(-1))


def test_adaptive_stop_requires_repeated_small_gains() -> None:
    assert should_stop_mcmc([-10.0, -9.995, -9.991], min_steps=2, epsilon=0.01)
    assert not should_stop_mcmc([-10.0, -9.5, -9.49], min_steps=2, epsilon=0.01)


def test_criticality_is_negative_logprob() -> None:
    assert criticality_score([-0.1, -2.0], 1) == 2.0

