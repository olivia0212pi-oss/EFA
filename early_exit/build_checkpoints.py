from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.config import load_config
from common.io import read_jsonl, write_jsonl
from early_exit.features import make_features
from evaluation.answers import extract_answer, is_correct
from generation.utils import apply_chat_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peek at intermediate reasoning checkpoints.")
    parser.add_argument("input", help="JSONL produced by generation.generate_dataset.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="results/checkpoints.jsonl")
    return parser.parse_args()


def _checkpoint_positions(total: int, interval: int) -> list[int]:
    positions = list(range(interval, total + 1, interval))
    if total and (not positions or positions[-1] != total):
        positions.append(total)
    return positions


def _peek_prompt(question: str, reasoning_prefix: str) -> str:
    return (
        "Review the partial solution below and give the best answer available now. "
        "Return only the final answer in \\boxed{}.\n\n"
        f"Problem:\n{question}\n\nPartial solution:\n{reasoning_prefix}\n"
    )


def _analyze_record(
    record: dict[str, Any], llm: Any, params: Any, interval: int
) -> dict[str, Any]:
    tokenizer = llm.get_tokenizer()
    token_ids = record.get("token_ids") or tokenizer.encode(
        record["reasoning_text"], add_special_tokens=False
    )
    logprobs = record.get("token_logprobs", [])
    answers: list[str | None] = []
    checkpoints: list[dict[str, Any]] = []

    for position in _checkpoint_positions(len(token_ids), interval):
        prefix = tokenizer.decode(token_ids[:position], skip_special_tokens=True)
        prompt = apply_chat_template(tokenizer, _peek_prompt(record["question"], prefix))
        peek = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0].text
        answer = extract_answer(peek)
        answers.append(answer)
        recent = logprobs[max(0, position - interval) : position]
        checkpoints.append(
            {
                "token": position,
                "answer": answer,
                "peek_text": peek,
                "correct": is_correct(answer, record["ground_truth"]),
                "features": make_features(recent, answers, position),
            }
        )

    future_all_correct = True
    for checkpoint in reversed(checkpoints):
        future_all_correct = future_all_correct and bool(checkpoint["correct"])
        checkpoint["safe_to_stop"] = future_all_correct

    full_final_answer = extract_answer(record["reasoning_text"])
    return {
        "schema_version": 1,
        "sample_id": record["sample_id"],
        "question": record["question"],
        "ground_truth": record["ground_truth"],
        "full_total_tokens": record["total_tokens"],
        "full_final_answer": full_final_answer,
        "full_correct": is_correct(full_final_answer, record["ground_truth"]),
        "checkpoints": checkpoints,
        "metadata": record.get("metadata", {}),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = list(read_jsonl(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    from vllm import LLM, SamplingParams

    model_config = config["model"]
    llm = LLM(
        model=model_config["generation"],
        dtype=model_config["dtype"],
        gpu_memory_utilization=float(model_config["gpu_memory_utilization"]),
        max_model_len=int(model_config["max_model_len"]),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        seed=int(config["seed"]),
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=int(config["checkpoint"]["peek_max_tokens"]),
        seed=int(config["seed"]),
    )
    interval = int(config["checkpoint"]["interval_tokens"])

    analyzed = []
    for index, record in enumerate(records, start=1):
        analyzed.append(_analyze_record(record, llm, params, interval))
        print(f"[{index}/{len(records)}] checkpoints={len(analyzed[-1]['checkpoints'])}")
    write_jsonl(Path(args.output), analyzed)
    print(f"Wrote checkpoint data to {args.output}")


if __name__ == "__main__":
    main()

