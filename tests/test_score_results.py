from evaluation.score_results import score_record


def test_rescoring_reextracts_instead_of_trusting_stale_cached_answer() -> None:
    record = {
        "question": "Find all integers n such that f(n) = n.",
        "ground_truth": "1,-2",
        "reasoning_text": (
            r"The solutions are \( n = -2 \) and \( n = 1 \). "
            r"Thus, the integers are \boxed{-2} and \boxed{1}."
        ),
        # Stale value as if cached by an older extract_answer implementation
        # that only kept the last \boxed{}.
        "final_answer": "1",
        "total_tokens": 100,
    }
    scored = score_record(record)
    assert scored["final_answer"] == "-2, 1"
    assert scored["correct"] is True
