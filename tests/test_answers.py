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


def test_chains_trailing_boxed_values_for_multi_part_answers() -> None:
    text = (
        r"The solutions are \( n = -2 \) and \( n = 1 \). "
        r"Thus, the integers are \boxed{-2} and \boxed{1}."
    )
    assert extract_answer(text) == "-2, 1"
    assert is_correct(extract_answer(text), "1,-2")


def test_chains_boxed_values_wrapped_in_inline_math_delimiters() -> None:
    # Real model output wraps each \boxed{} in \( \), not just plain text.
    text = (
        r"The solutions are \( n = -2 \) and \( n = 1 \). Both satisfy \( f(n) = n \). "
        r"Thus, the integers \( n \) such that \( f(n) = n \) are "
        r"\(\boxed{-2}\) and \(\boxed{1}\)."
    )
    assert extract_answer(text) == "-2, 1"
    assert is_correct(extract_answer(text), "1,-2")


def test_does_not_chain_unrelated_earlier_guess() -> None:
    text = r"First guess: \boxed{2}. Finally: \boxed{\frac{1}{2}}"
    assert extract_answer(text) == r"\frac{1}{2}"


def test_single_value_answer_unaffected_by_multi_value_path() -> None:
    assert is_correct("42", "42")
    assert not is_correct("41", "42")

