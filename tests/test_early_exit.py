import pytest

from early_exit.features import count_consecutive_same, make_features
from early_exit.oracle import oracle_for_record


def test_counts_stable_answers() -> None:
    assert count_consecutive_same(["1", "2", "2", "2"]) == 3
    assert count_consecutive_same([]) == 0


def test_features_use_total_reasoning_length() -> None:
    features = make_features([-0.2, -0.4], ["17", "17"], 512)
    assert features["avg_logprob"] == pytest.approx(-0.3)
    assert features["answer_same_count"] == 2.0
    assert features["answer_changed"] == 0.0
    assert features["reasoning_length"] == 512.0


def test_oracle_uses_first_permanently_correct_checkpoint() -> None:
    record = {
        "sample_id": "x",
        "full_total_tokens": 1024,
        "checkpoints": [
            {"token": 256, "safe_to_stop": False},
            {"token": 512, "safe_to_stop": True},
            {"token": 768, "safe_to_stop": True},
        ],
    }
    result = oracle_for_record(record)
    assert result["oracle_stop_token"] == 512
    assert result["saved_fraction"] == 0.5
