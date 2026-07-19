from evaluation.answers import extract_answer, is_correct, normalize_answer


def test_extracts_last_box_with_nested_latex() -> None:
    text = r"First guess: \boxed{2}. Finally: \boxed{\frac{1}{2}}"
    assert extract_answer(text) == r"\frac{1}{2}"


def test_extracts_plain_final_answer() -> None:
    assert extract_answer("Reasoning\nFinal answer: 42.") == "42"


def test_normalizes_common_latex_variants() -> None:
    assert normalize_answer(r"$\dfrac{1}{ 2 }$") == normalize_answer(r"\frac{1}{2}")


def test_exact_fallback_correctness() -> None:
    assert is_correct("180", "180")
    assert not is_correct(None, "180")


def test_strips_text_wrapper_in_ground_truth() -> None:
    assert is_correct("Evelyn", r"\text{Evelyn}")
    assert normalize_answer(r"\text{Evelyn}") == normalize_answer("Evelyn")

