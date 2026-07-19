from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.io import read_jsonl
from early_exit.features import FEATURE_NAMES, feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and calibrate the early-exit classifier.")
    parser.add_argument("input", help="Checkpoint JSONL file.")
    parser.add_argument("--output", default="checkpoints/early_exit.joblib")
    parser.add_argument("--max-stop-error", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _choose_threshold(y: np.ndarray, probability: np.ndarray, max_error: float) -> float:
    for threshold in np.linspace(0.5, 0.999, 500):
        selected = probability >= threshold
        if selected.any() and float(1.0 - y[selected].mean()) <= max_error:
            return float(threshold)
    return 1.0


def main() -> None:
    args = parse_args()
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for record in read_jsonl(args.input):
        for checkpoint in record["checkpoints"]:
            rows.append(feature_vector(checkpoint["features"]))
            labels.append(int(checkpoint["safe_to_stop"]))
            groups.append(str(record["sample_id"]))
    if len(set(groups)) < 3 or len(set(labels)) < 2:
        raise SystemExit("Need at least 3 problems and both labels before training.")

    x = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, calibration_idx = next(splitter.split(x, y, groups=groups))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
    )
    model.fit(x[train_idx], y[train_idx])
    probability = model.predict_proba(x[calibration_idx])[:, 1]
    threshold = _choose_threshold(y[calibration_idx], probability, args.max_stop_error)

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_names": FEATURE_NAMES,
        "max_stop_error": args.max_stop_error,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    selected = probability >= threshold
    summary = {
        "train_checkpoints": len(train_idx),
        "calibration_checkpoints": len(calibration_idx),
        "threshold": threshold,
        "calibration_stops": int(selected.sum()),
        "calibration_stop_error": (
            float(1.0 - y[calibration_idx][selected].mean()) if selected.any() else None
        ),
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote classifier to {output}")


if __name__ == "__main__":
    main()

