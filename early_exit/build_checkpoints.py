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


def _required_max_model_len(
    total_token_counts: list[int], peek_max_tokens: int, current_max_model_len: int
) -> int:
    """Grow max_model_len so the longest record's end-of-trace probe fits.

    The longest probe prompt is base_prompt + the longest record's full
    reasoning prefix + the forced "\\boxed{" suffix, generating up to
    peek_max_tokens more. Without headroom for this, a probe near the end
    of a long trace gets silently truncated to 1-2 tokens by vLLM's context
    limit, producing a spuriously "confident" answer instead of a real one.
    """
    max_total_tokens = max(total_token_counts, default=0)
    required_len = max_total_tokens + peek_max_tokens + 1024
    return max(required_len, current_max_model_len)


def _probe_prompt(base_prompt: str, reasoning_prefix: str) -> str:
    """A raw continuation of the same prompt/reasoning stream, forced to answer now.

    Unlike an instructional "review this and answer" prompt, this keeps the
    probe on the exact token distribution the model would have seen mid
    generation, so the trial answer and its logprobs reflect what the model
    actually believes at this point, not a differently-framed re-ask.
    """
    return f"{base_prompt}{reasoning_prefix}\n\nTherefore, the final answer is \\boxed{{"


def _first_boxed_close_offset(completion: str) -> int | None:
    """Character offset within `completion` where the forced \\boxed{ closes.

    The probe prompt always ends in "...\\boxed{", so the completion is
    everything generated after that opening brace; this walks brace depth
    from 1 to find where it closes, independent of extract_answer's
    multi-box chaining (built for the full final trace, not a short forced
    single-box probe). Returns None if it never closes within the probe's
    token budget.
    """
    depth = 1
    for index, char in enumerate(completion):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _answer_span_token_count(token_texts: list[str], close_offset: int) -> int:
    """How many leading tokens (by decoded text length) cover the answer
    content through and including the character at close_offset.
    """
    cumulative = 0
    for count, text in enumerate(token_texts, start=1):
        cumulative += len(text)
        if cumulative > close_offset:
            return count
    return len(token_texts)


def probe_to_checkpoint(
    position: int,
    trial_completion: str,
    completion_token_texts: list[str],
    trial_answer_logprobs: list[float],
    first_token_topk_logprobs: list[float],
    ground_truth: Any,
    recent_reasoning_logprobs: list[float],
    answer_history: list[str | None],
) -> dict[str, Any]:
    """Pure post-processing of one probe's raw generation into a checkpoint record.

    Kept free of vLLM types so it is unit-testable without a GPU: callers do
    the vLLM-specific extraction (chosen_logprobs, output.logprobs[0],
    per-token decoded text) and hand in plain floats/strings.
    """
    trial_text = "\\boxed{" + trial_completion
    close_offset = _first_boxed_close_offset(trial_completion)
    probe_answer_complete = close_offset is not None

    if probe_answer_complete:
        # The answer is read from the canonical first \boxed{...} span only
        # (trial_completion[:close_offset]), never from extract_answer on the
        # full trial_text: that function chains/prefers *later* \boxed{}
        # occurrences for the real final trace, which would let a second,
        # possibly-conflicting box the model writes after closing the first
        # one silently override the label this checkpoint is actually
        # scored and probed on.
        trial_answer = trial_completion[:close_offset].strip()
        span_tokens = _answer_span_token_count(completion_token_texts, close_offset)
        # Only the tokens that actually form the boxed answer count toward
        # confidence -- anything the model kept writing after closing the
        # box (a re-check, a second "Final Answer" restatement, ...) must
        # not dilute or inflate it.
        answer_logprobs = trial_answer_logprobs[:span_tokens]
        confidence: float | None = deer_confidence(answer_logprobs)
        current_correct = is_correct(trial_answer, ground_truth)
    else:
        # The box never closed within the probe's token budget. Force
        # trial_answer to None here -- not extract_answer's plain-text
        # fallback on unfinished text -- so a stray answer-shaped phrase in
        # mid-sentence reasoning can never feed answer_history/features and
        # fake a stable streak, and this checkpoint never looks like a safe
        # stop.
        trial_answer = None
        answer_logprobs = []
        confidence = None
        current_correct = False

    entropy, margin = entropy_and_margin(first_token_topk_logprobs)
    # answer_history is the history *before* this checkpoint; make_features'
    # answer_same_count/answer_changed describe stability up to and including
    # the current checkpoint, so the current trial_answer must be included.
    features = make_features(
        recent_reasoning_logprobs,
        [*answer_history, trial_answer],
        position,
        deer_confidence_value=confidence or 0.0,
        entropy=entropy,
        margin=margin,
        probe_complete=1.0 if probe_answer_complete else 0.0,
    )
    return {
        "token": position,
        "trial_answer": trial_answer,
        "trial_text": trial_text,
        "probe_answer_complete": probe_answer_complete,
        "trial_answer_logprobs": answer_logprobs,
        "current_correct": current_correct,
        "deer_confidence": confidence,
        "top_logprobs": first_token_topk_logprobs,
        "entropy": entropy,
        "margin": margin,
        "features": features,
    }


def _cumulative_token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    """Per-token decoded text pieces via cumulative-decode diffs.

    Decoding each token id in isolation can misplace merge/whitespace
    artifacts for some tokenizers; diffing successive cumulative decodes
    keeps each piece aligned with where it actually lands in the full text.
    """
    texts = []
    previous = ""
    for i in range(1, len(token_ids) + 1):
        current = tokenizer.decode(token_ids[:i], skip_special_tokens=True)
        texts.append(current[len(previous) :])
        previous = current
    return texts


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
        completion_token_texts = _cumulative_token_texts(tokenizer, list(output.token_ids))
        first_token_topk = (
            sorted((float(c.logprob) for c in output.logprobs[0].values()), reverse=True)
            if output.logprobs
            else []
        )
        recent = logprobs[max(0, position - interval) : position]
        checkpoint = probe_to_checkpoint(
            position,
            output.text,
            completion_token_texts,
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
        checkpoint["state"] = classify_state(
            checkpoint["current_correct"], checkpoint["persistent_correct"], final_correct
        )

    return {
        "schema_version": 3,
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
    checkpoint_config = config["checkpoint"]
    required_len = _required_max_model_len(
        [int(r["total_tokens"]) for r in records],
        int(checkpoint_config["peek_max_tokens"]),
        int(model_config["max_model_len"]),
    )
    if required_len > int(model_config["max_model_len"]):
        print(
            f"Bumping max_model_len {model_config['max_model_len']} -> {required_len} "
            "so the longest probe prompt fits."
        )
        model_config["max_model_len"] = required_len
    llm = LLM(
        model=model_config["generation"],
        dtype=model_config["dtype"],
        gpu_memory_utilization=float(model_config["gpu_memory_utilization"]),
        max_model_len=int(model_config["max_model_len"]),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        seed=int(config["seed"]),
    )
    tokenizer = llm.get_tokenizer()
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
