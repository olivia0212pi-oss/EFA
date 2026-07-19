from __future__ import annotations

import argparse
from typing import Any

from common.config import load_config
from common.io import append_jsonl, existing_sample_ids, read_jsonl
from early_exit.features import classify_state, deer_confidence, entropy_and_margin, make_features
from evaluation.answers import extract_answer, is_correct
from generation.utils import apply_chat_template, chosen_logprobs, make_math_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peek at intermediate reasoning checkpoints.")
    parser.add_argument("input", help="JSONL produced by generation.generate_dataset.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="results/checkpoints.jsonl")
    parser.add_argument(
        "--resume", action="store_true", help="Skip sample IDs already present in the output file."
    )
    return parser.parse_args()


def _checkpoint_positions(total: int, interval: int) -> list[int]:
    positions = list(range(interval, total + 1, interval))
    if total and (not positions or positions[-1] != total):
        positions.append(total)
    return positions


def _probe_prompt(base_prompt: str, reasoning_prefix: str) -> str:
    """A raw continuation of the same prompt/reasoning stream, forced to answer now.

    Unlike an instructional "review this and answer" prompt, this keeps the
    probe on the exact token distribution the model would have seen mid
    generation, so the trial answer and its logprobs reflect what the model
    actually believes at this point, not a differently-framed re-ask.
    """
    return f"{base_prompt}{reasoning_prefix}\n\nTherefore, the final answer is \\boxed{{"


def probe_to_checkpoint(
    position: int,
    trial_completion: str,
    trial_answer_logprobs: list[float],
    first_token_topk_logprobs: list[float],
    ground_truth: Any,
    recent_reasoning_logprobs: list[float],
    answer_history: list[str | None],
) -> dict[str, Any]:
    """Pure post-processing of one probe's raw generation into a checkpoint record.

    Kept free of vLLM types so it is unit-testable without a GPU: callers do
    the vLLM-specific extraction (chosen_logprobs, output.logprobs[0]) and
    hand in plain floats/strings.
    """
    trial_text = "\\boxed{" + trial_completion
    trial_answer = extract_answer(trial_text)
    entropy, margin = entropy_and_margin(first_token_topk_logprobs)
    confidence = deer_confidence(trial_answer_logprobs)
    current_correct = is_correct(trial_answer, ground_truth)
    features = make_features(
        recent_reasoning_logprobs,
        answer_history,
        position,
        deer_confidence_value=confidence,
        entropy=entropy,
        margin=margin,
    )
    return {
        "token": position,
        "trial_answer": trial_answer,
        "trial_text": trial_text,
        "trial_answer_logprobs": trial_answer_logprobs,
        "current_correct": current_correct,
        "deer_confidence": confidence,
        "top_logprobs": first_token_topk_logprobs,
        "entropy": entropy,
        "margin": margin,
        "features": features,
    }


def _analyze_record(
    record: dict[str, Any], llm: Any, tokenizer: Any, params: Any, interval: int
) -> dict[str, Any]:
    token_ids = record.get("token_ids") or tokenizer.encode(
        record["reasoning_text"], add_special_tokens=False
    )
    logprobs = record.get("token_logprobs", [])
    base_prompt = apply_chat_template(tokenizer, make_math_prompt(record["question"]))

    answer_history: list[str | None] = []
    checkpoints: list[dict[str, Any]] = []
    for position in _checkpoint_positions(len(token_ids), interval):
        prefix = tokenizer.decode(token_ids[:position], skip_special_tokens=True)
        prompt = _probe_prompt(base_prompt, prefix)
        output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        trial_answer_logprobs = chosen_logprobs(output.token_ids, output.logprobs)
        first_token_topk = (
            sorted((float(c.logprob) for c in output.logprobs[0].values()), reverse=True)
            if output.logprobs
            else []
        )
        recent = logprobs[max(0, position - interval) : position]
        checkpoint = probe_to_checkpoint(
            position,
            output.text,
            trial_answer_logprobs,
            first_token_topk,
            record["ground_truth"],
            recent,
            answer_history,
        )
        answer_history.append(checkpoint["trial_answer"])
        checkpoints.append(checkpoint)

    persistent_correct = True
    for checkpoint in reversed(checkpoints):
        persistent_correct = persistent_correct and bool(checkpoint["current_correct"])
        checkpoint["persistent_correct"] = persistent_correct

    full_final_answer = extract_answer(record["reasoning_text"])
    final_correct = is_correct(full_final_answer, record["ground_truth"])
    for checkpoint in checkpoints:
        checkpoint["final_correct"] = final_correct
        checkpoint["state"] = classify_state(checkpoint["current_correct"], final_correct)

    return {
        "schema_version": 2,
        "sample_id": record["sample_id"],
        "question": record["question"],
        "ground_truth": record["ground_truth"],
        "full_total_tokens": record["total_tokens"],
        "full_final_answer": full_final_answer,
        "full_correct": final_correct,
        "checkpoints": checkpoints,
        "metadata": record.get("metadata", {}),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = list(read_jsonl(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    complete_records = []
    skipped_ids = []
    for record in records:
        if extract_answer(record["reasoning_text"]) is None:
            skipped_ids.append(record["sample_id"])
            continue
        complete_records.append(record)
    if skipped_ids:
        print(
            f"Skipping {len(skipped_ids)} truncated records with no final \\boxed{{}} answer "
            f"(not usable for persistent-correctness/oracle analysis): {skipped_ids}"
        )
    records = complete_records
    if not records:
        raise SystemExit("No complete (untruncated) records left to build checkpoints from.")

    completed = existing_sample_ids(args.output) if args.resume else set()
    pending = [r for r in records if str(r["sample_id"]) not in completed]

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
    tokenizer = llm.get_tokenizer()
    checkpoint_config = config["checkpoint"]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=int(checkpoint_config["peek_max_tokens"]),
        logprobs=int(checkpoint_config.get("probe_top_k", 5)),
        seed=int(config["seed"]),
    )
    interval = int(checkpoint_config["interval_tokens"])

    written = 0
    for index, record in enumerate(pending, start=1):
        analyzed = _analyze_record(record, llm, tokenizer, params, interval)
        append_jsonl(args.output, analyzed)
        written += 1
        print(
            f"[{index}/{len(pending)}] sample_id={record['sample_id']} "
            f"checkpoints={len(analyzed['checkpoints'])}"
        )
    print(f"Wrote {written} records to {args.output} ({len(completed)} skipped as already done)")


if __name__ == "__main__":
    main()
