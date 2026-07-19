from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import load_config
from common.io import append_jsonl, existing_sample_ids, read_jsonl
from common.reproducibility import set_seed
from evaluation.answers import extract_answer
from generation.utils import apply_chat_template, chosen_logprobs, make_math_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MATH responses and token logprobs.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--num-samples", type=int, help="Override data.num_samples.")
    parser.add_argument("--output", help="Output JSONL path.")
    parser.add_argument(
        "--resume", action="store_true", help="Skip sample IDs already present in the output file."
    )
    parser.add_argument(
        "--retry-from",
        help=(
            "Instead of pulling from the HF dataset, regenerate only the truncated "
            "(no final \\boxed{} answer) records from this JSONL, reusing their "
            "question/ground_truth. Never writes back to this file."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override generation.max_tokens (e.g. for retrying truncations).",
    )
    return parser.parse_args()


def _build_record(
    item: dict[str, Any],
    index: int,
    output: Any,
    runtime_seconds: float,
    config: dict[str, Any],
    retry_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_name = config["model"]["generation"]
    text = output.text
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset": config["data"]["dataset"],
        "split": config["data"]["split"],
        "seed": config["seed"],
        "sampling": config["generation"],
    }
    if retry_source is not None:
        metadata["retried_from"] = retry_source
    return {
        "schema_version": 1,
        "sample_id": str(item.get("id", index)),
        "question": item["problem"],
        "ground_truth": item["answer"],
        "reasoning_text": text,
        "final_answer": extract_answer(text),
        "token_ids": list(output.token_ids),
        "token_logprobs": chosen_logprobs(output.token_ids, output.logprobs),
        "total_tokens": len(output.token_ids),
        "runtime_seconds": round(runtime_seconds, 3),
        "metadata": metadata,
    }


def _truncated_items(path: str) -> list[dict[str, Any]]:
    items = []
    for record in read_jsonl(path):
        if extract_answer(record["reasoning_text"]) is not None:
            continue
        items.append(
            {
                "id": record["sample_id"],
                "problem": record["question"],
                "answer": record["ground_truth"],
                "_previous_total_tokens": record["total_tokens"],
            }
        )
    return items


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.num_samples is not None:
        config["data"]["num_samples"] = args.num_samples
    if args.max_tokens is not None:
        config["generation"]["max_tokens"] = args.max_tokens
    set_seed(int(config["seed"]))

    from vllm import LLM, SamplingParams

    if args.retry_from:
        source_path = Path(args.retry_from)
        default_name = f"{source_path.stem}_retry.jsonl"
        output_path = Path(args.output or source_path.parent / default_name)
        if output_path == source_path:
            raise SystemExit(
                "--output must differ from --retry-from; refusing to overwrite the source."
            )
        dataset = _truncated_items(args.retry_from)
        if not dataset:
            raise SystemExit(f"No truncated (unanswered) records found in {args.retry_from}.")
    else:
        from datasets import load_dataset

        count = int(config["data"]["num_samples"])
        default_name = "math500_smoke.jsonl" if count <= 5 else f"math500_{count}.jsonl"
        output_path = Path(args.output or Path(config["output"]["results_dir"]) / default_name)
        hf_dataset = load_dataset(config["data"]["dataset"], split=config["data"]["split"])
        dataset = hf_dataset.select(range(min(count, len(hf_dataset))))

    if output_path.exists() and not args.resume:
        raise SystemExit(f"{output_path} already exists. Pass --resume or choose another --output.")
    completed = existing_sample_ids(output_path) if args.resume else set()

    model_config = config["model"]
    generation_config = config["generation"]
    required_len = int(generation_config["max_tokens"]) + 1024
    if required_len > int(model_config["max_model_len"]):
        print(
            f"Bumping max_model_len {model_config['max_model_len']} -> {required_len} "
            "so the prompt plus --max-tokens fits."
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
        temperature=float(generation_config["temperature"]),
        top_p=float(generation_config["top_p"]),
        max_tokens=int(generation_config["max_tokens"]),
        logprobs=int(generation_config["logprobs"]),
        seed=int(config["seed"]),
    )

    written = 0
    for index, item in enumerate(dataset):
        sample_id = str(item.get("id", index))
        if sample_id in completed:
            continue
        prompt = apply_chat_template(tokenizer, make_math_prompt(item["problem"]))
        started = time.perf_counter()
        output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        runtime = time.perf_counter() - started
        retry_source = (
            {"source": args.retry_from, "previous_total_tokens": item["_previous_total_tokens"]}
            if args.retry_from
            else None
        )
        record = _build_record(item, index, output, runtime, config, retry_source)
        append_jsonl(output_path, record)
        written += 1
        print(
            f"[{index + 1}/{len(dataset)}] id={sample_id} "
            f"tokens={record['total_tokens']} time={runtime:.2f}s"
        )
    print(f"Wrote {written} records to {output_path}")


if __name__ == "__main__":
    main()
