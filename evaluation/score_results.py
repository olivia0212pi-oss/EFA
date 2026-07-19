from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.io import read_jsonl, write_jsonl
from evaluation.answers import extract_answer, is_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a generated JSONL file.")
    parser.add_argument("input", help="Generated JSONL file.")
    parser.add_argument("--output", help="Optional scored JSONL output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = list(read_jsonl(input_path))
    if not records:
        raise SystemExit(f"No records found in {input_path}")

    scored = []
    for record in records:
        prediction = record.get("final_answer") or extract_answer(record["reasoning_text"])
        enriched = dict(record)
        enriched["final_answer"] = prediction
        enriched["correct"] = is_correct(prediction, record["ground_truth"])
        scored.append(enriched)

    correct = sum(bool(record["correct"]) for record in scored)
    total_tokens = sum(int(record.get("total_tokens", 0)) for record in scored)
    total_runtime = sum(float(record.get("runtime_seconds", 0.0)) for record in scored)
    summary = {
        "samples": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored),
        "average_tokens": total_tokens / len(scored),
        "average_runtime_seconds": total_runtime / len(scored),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        write_jsonl(args.output, scored)
        print(f"Wrote scored records to {args.output}")


if __name__ == "__main__":
    main()

