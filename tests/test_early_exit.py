import pytest

from early_exit.build_checkpoints import _checkpoint_positions, _probe_prompt, probe_to_checkpoint
from early_exit.features import (
    classify_state,
    count_consecutive_same,
    deer_confidence,
    entropy_and_margin,
    make_features,
)
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
    assert features["deer_confidence"] == 0.0
    assert features["entropy"] == 0.0
    assert features["margin"] == 0.0


def test_deer_confidence_is_geometric_mean_probability() -> None:
    import math

    assert deer_confidence([math.log(0.5), math.log(0.5)]) == pytest.approx(0.5)
    assert deer_confidence([]) == 0.0
    assert deer_confidence([float("nan"), 0.0]) == pytest.approx(1.0)


def test_entropy_and_margin_from_topk_logprobs() -> None:
    import math

    # A near-certain top token: low entropy, large margin.
    entropy, margin = entropy_and_margin([math.log(0.98), math.log(0.01), math.log(0.01)])
    assert entropy < 0.2
    assert margin > 3.0

    # A uniform 2-way tie: high entropy, ~zero margin.
    entropy, margin = entropy_and_margin([math.log(0.5), math.log(0.5)])
    assert entropy == pytest.approx(math.log(2), abs=1e-3)
    assert margin == pytest.approx(0.0, abs=1e-6)

    assert entropy_and_margin([]) == (0.0, 0.0)


def test_classify_state_four_way() -> None:
    assert classify_state(True, True) == "correct_stable"
    assert classify_state(True, False) == "correct_unstable"
    assert classify_state(False, True) == "incorrect_recoverable"
    assert classify_state(False, False) == "incorrect_terminal"


def test_checkpoint_positions_always_includes_total() -> None:
    assert _checkpoint_positions(1000, 256) == [256, 512, 768, 1000]
    assert _checkpoint_positions(512, 256) == [256, 512]
    assert _checkpoint_positions(0, 256) == []


def test_probe_prompt_is_raw_continuation_not_a_new_turn() -> None:
    prompt = _probe_prompt("BASE>", "some reasoning so far")
    assert prompt.startswith("BASE>some reasoning so far")
    assert prompt.rstrip().endswith("\\boxed{")


def test_probe_to_checkpoint_builds_full_schema() -> None:
    import math

    checkpoint = probe_to_checkpoint(
        position=256,
        trial_completion="17}. Done.",
        trial_answer_logprobs=[math.log(0.9), math.log(0.95)],
        first_token_topk_logprobs=[math.log(0.9), math.log(0.05)],
        ground_truth="17",
        recent_reasoning_logprobs=[-0.1, -0.2],
        answer_history=["17"],
    )
    assert checkpoint["token"] == 256
    assert checkpoint["trial_answer"] == "17"
    assert checkpoint["trial_text"] == "\\boxed{17}. Done."
    assert checkpoint["current_correct"] is True
    expected_confidence = math.exp((math.log(0.9) + math.log(0.95)) / 2)
    assert checkpoint["deer_confidence"] == pytest.approx(expected_confidence)
    assert checkpoint["entropy"] > 0
    assert checkpoint["margin"] > 0
    assert checkpoint["features"]["deer_confidence"] == checkpoint["deer_confidence"]
    # persistent_correct/final_correct/state are filled in by _analyze_record's
    # record-level pass, not by this per-probe helper.
    assert "persistent_correct" not in checkpoint
    assert "final_correct" not in checkpoint


def test_oracle_uses_first_permanently_correct_checkpoint() -> None:
    record = {
        "sample_id": "x",
        "full_total_tokens": 1024,
        "checkpoints": [
            {"token": 256, "persistent_correct": False},
            {"token": 512, "persistent_correct": True},
            {"token": 768, "persistent_correct": True},
        ],
    }
    result = oracle_for_record(record)
    assert result["oracle_stop_token"] == 512
    assert result["saved_fraction"] == 0.5
