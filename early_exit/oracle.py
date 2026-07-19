from __future__ import annotations

import argparse
import json

from common.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the checkpoint early-exit oracle.")
    parser.add_argument("input", help="Checkpoint JSONL file.")
    return parser.parse_args()


def oracle_for_record(record: dict) -> dict:
    full_tokens = int(record["full_total_tokens"])
    oracle = next(
        (checkpoint for checkpoint in record["checkpoints"] if checkpoint["safe_to_stop"]),
        None,
    )
    stop_token = int(oracle["token"]) if oracle else full_tokens
    saved = max(0, full_tokens - stop_token)
    return {
        "sample_id": record["sample_id"],
        "full_tokens": full_tokens,
        "oracle_stop_token": stop_token,
        "saved_tokens": saved,
        "saved_fraction": saved / full_tokens if full_tokens else 0.0,
        "oracle_found": oracle is not None,
    }


def main() -> None:
    args = parse_args()
    rows = [oracle_for_record(record) for record in read_jsonl(args.input)]
    if not rows:
        raise SystemExit(f"No records found in {args.input}")
    total_full = sum(row["full_tokens"] for row in rows)
    total_saved = sum(row["saved_tokens"] for row in rows)
    summary = {
        "samples": len(rows),
        "samples_with_oracle_stop": sum(row["oracle_found"] for row in rows),
        "total_full_tokens": total_full,
        "total_saved_tokens": total_saved,
        "micro_saved_fraction": total_saved / total_full if total_full else 0.0,
        "mean_saved_fraction": sum(row["saved_fraction"] for row in rows) / len(rows),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

