from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Show your reasoning, then put only "
    "the final answer in \\boxed{}."
)


def make_math_prompt(problem: str) -> str:
    return f"Solve the following problem step by step.\nProblem: {problem}"


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{prompt}\n"


def chosen_logprobs(token_ids: list[int], logprobs: list[dict[int, Any]] | None) -> list[float]:
    if not logprobs:
        return []
    values: list[float] = []
    for token_id, candidates in zip(token_ids, logprobs, strict=False):
        chosen = candidates.get(token_id)
        if chosen is None and candidates:
            chosen = next(iter(candidates.values()))
        values.append(float(chosen.logprob) if chosen is not None else float("nan"))
    return values

