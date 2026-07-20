"""Day 4.5 protocol repair: grouped outer 5-fold, out-of-fold-only reporting,
Wilson CIs on stop error, a hard probe_answer_complete gate, a single-point
vs history-only vs combined feature ablation, and baselines on the same
folds. Pure CPU (sklearn), no GPU/vLLM involved.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHECKPOINTS_PATH = "results/checkpoints_math500_93_v3.jsonl"
SEED = 42
N_OUTER_FOLDS = 5
TARGET_ERROR = 0.05

SINGLE_POINT_FEATURES = ["avg_logprob", "min_logprob", "deer_confidence", "entropy", "margin"]
HISTORY_FEATURES = ["answer_same_count", "answer_changed", "reasoning_length"]
COMBINED_FEATURES = SINGLE_POINT_FEATURES + HISTORY_FEATURES


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def simulate_sequential_policy(
    records: list[dict[str, Any]], score_fn: Callable[[dict], float], threshold: float
) -> list[dict[str, Any]]:
    """Walk each record's checkpoints in order; stop at the first one whose
    score clears the threshold, but ONLY if that probe actually closed its
    \\boxed{} (probe_answer_complete). Never crosses -> full trace."""
    results = []
    for r in records:
        stopped = None
        for cp in r["checkpoints"]:
            if not cp["probe_answer_complete"]:
                continue
            if score_fn(cp) >= threshold:
                stopped = cp
                break
        if stopped is not None:
            results.append(
                {
                    "sample_id": r["sample_id"],
                    "stopped_early": True,
                    "stop_token": stopped["token"],
                    "correct": stopped["current_correct"],
                    "full_tokens": r["full_total_tokens"],
                }
            )
        else:
            results.append(
                {
                    "sample_id": r["sample_id"],
                    "stopped_early": False,
                    "stop_token": r["full_total_tokens"],
                    "correct": r["full_correct"],
                    "full_tokens": r["full_total_tokens"],
                }
            )
    return results


def calibrate_threshold(
    records: list[dict[str, Any]],
    score_fn: Callable[[dict], float],
    candidate_thresholds: list[float],
    target_error: float = TARGET_ERROR,
) -> float:
    """Pick the threshold (from candidates, ascending savings-permissiveness)
    that keeps the SEQUENTIAL stop-error rate on `records` at or below
    target_error, preferring the most permissive (lowest) such threshold.
    Falls back to the strictest candidate (never stop) if none qualifies."""
    best = max(candidate_thresholds)
    for t in sorted(candidate_thresholds):
        sim = simulate_sequential_policy(records, score_fn, t)
        stopped = [s for s in sim if s["stopped_early"]]
        errors = sum(1 for s in stopped if not s["correct"])
        error_rate = errors / len(stopped) if stopped else 0.0
        if error_rate <= target_error:
            best = t
            break
    return best


def feature_vector_for(cp: dict[str, Any], names: list[str]) -> list[float]:
    return [cp["features"][name] for name in names]


def make_classifier_score_fn(model, names: list[str]) -> Callable[[dict], float]:
    def score(cp: dict[str, Any]) -> float:
        return float(model.predict_proba([feature_vector_for(cp, names)])[0][1])

    return score


def run_nested_cv_for_feature_set(
    records: list[dict[str, Any]], groups: list[str], feature_names: list[str]
) -> list[dict[str, Any]]:
    """Grouped outer 5-fold CV. Inside each outer-train fold, a further
    grouped train/calibration split fits the model and calibrates the
    threshold against the real sequential policy. All reported rows come
    only from each fold's held-out outer-test records."""
    unique_groups = sorted(set(groups), key=int)
    group_to_records = {g: [r for r in records if r["sample_id"] == g] for g in unique_groups}

    outer = GroupKFold(n_splits=N_OUTER_FOLDS)
    dummy_x = np.zeros((len(unique_groups), 1))
    dummy_y = np.zeros(len(unique_groups))

    oof_results: list[dict[str, Any]] = []
    for outer_train_idx, outer_test_idx in outer.split(dummy_x, dummy_y, groups=unique_groups):
        outer_train_ids = [unique_groups[i] for i in outer_train_idx]
        outer_test_ids = [unique_groups[i] for i in outer_test_idx]
        outer_train_records = [r for g in outer_train_ids for r in group_to_records[g]]
        outer_test_records = [r for g in outer_test_ids for r in group_to_records[g]]

        # Inner split of the outer-train fold into fit/calibration.
        rows, labels, inner_groups = [], [], []
        for r in outer_train_records:
            for cp in r["checkpoints"]:
                rows.append(feature_vector_for(cp, feature_names))
                labels.append(int(cp["persistent_correct"]))
                inner_groups.append(r["sample_id"])
        x = np.asarray(rows, dtype=float)
        y = np.asarray(labels, dtype=int)
        if len(set(labels)) < 2 or len(set(inner_groups)) < 3:
            continue
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        fit_idx, calib_idx = next(splitter.split(x, y, groups=inner_groups))
        calib_ids = {inner_groups[i] for i in calib_idx}

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        )
        model.fit(x[fit_idx], y[fit_idx])
        score_fn = make_classifier_score_fn(model, feature_names)

        calib_records = [r for r in outer_train_records if r["sample_id"] in calib_ids]
        candidate_thresholds = sorted(set(np.linspace(0.5, 0.999, 200)) | {1.0})
        threshold = calibrate_threshold(calib_records, score_fn, candidate_thresholds)

        oof_results.extend(simulate_sequential_policy(outer_test_records, score_fn, threshold))

    return oof_results


def run_nested_cv_for_rule(
    records: list[dict[str, Any]],
    groups: list[str],
    score_fn: Callable[[dict], float],
    candidate_thresholds: list[float],
) -> list[dict[str, Any]]:
    """Same nested outer-fold discipline as the classifier, but for a fixed
    (non-learned) scoring rule: only the threshold is calibrated per fold on
    that fold's outer-train data, never touching the outer-test fold."""
    unique_groups = sorted(set(groups), key=int)
    group_to_records = {g: [r for r in records if r["sample_id"] == g] for g in unique_groups}
    outer = GroupKFold(n_splits=N_OUTER_FOLDS)
    dummy_x = np.zeros((len(unique_groups), 1))
    dummy_y = np.zeros(len(unique_groups))

    oof_results: list[dict[str, Any]] = []
    for outer_train_idx, outer_test_idx in outer.split(dummy_x, dummy_y, groups=unique_groups):
        outer_train_ids = [unique_groups[i] for i in outer_train_idx]
        outer_test_ids = [unique_groups[i] for i in outer_test_idx]
        outer_train_records = [r for g in outer_train_ids for r in group_to_records[g]]
        outer_test_records = [r for g in outer_test_ids for r in group_to_records[g]]
        threshold = calibrate_threshold(outer_train_records, score_fn, candidate_thresholds)
        oof_results.extend(simulate_sequential_policy(outer_test_records, score_fn, threshold))
    return oof_results


def summarize(label: str, oof: list[dict[str, Any]], records_by_id: dict[str, Any]) -> None:
    n = len(oof)
    n_correct = sum(1 for s in oof if s["correct"])
    n_stopped = sum(1 for s in oof if s["stopped_early"])
    n_wrong_stopped = sum(1 for s in oof if s["stopped_early"] and not s["correct"])
    total_full = sum(s["full_tokens"] for s in oof)
    total_saved = sum(max(0, s["full_tokens"] - s["stop_token"]) for s in oof)
    saved_fracs = [
        max(0, s["full_tokens"] - s["stop_token"]) / s["full_tokens"] if s["full_tokens"] else 0.0
        for s in oof
    ]
    lo, hi = wilson_ci(n_wrong_stopped, n_stopped) if n_stopped else (0.0, 0.0)

    print(f"--- {label} (out-of-fold N={n}) ---")
    print(f"accuracy: {n_correct}/{n} = {n_correct / n:.4f}")
    print(f"stopped early: {n_stopped}/{n} = {n_stopped / n:.4f}")
    if n_stopped:
        print(
            f"stop error: {n_wrong_stopped}/{n_stopped} = {n_wrong_stopped / n_stopped:.4f} "
            f"(95% Wilson CI: [{lo:.4f}, {hi:.4f}])"
        )
    else:
        print("stop error: n/a (never stopped)")
    print(f"micro_saved_fraction: {total_saved / total_full:.4f}")
    print(f"mean_saved_fraction: {statistics.mean(saved_fracs):.4f}")
    print()


def main() -> None:
    records = load_records(CHECKPOINTS_PATH)
    groups = [r["sample_id"] for r in records]
    records_by_id = {r["sample_id"]: r for r in records}

    print("=" * 78)
    print(f"N records: {len(records)}  outer folds: {N_OUTER_FOLDS} (grouped by sample_id)")
    print()

    print("=" * 78)
    print("FEATURE-SET ABLATION (classifier, nested grouped 5-fold, OOF only)")
    print()
    for label, feats in [
        ("single_point", SINGLE_POINT_FEATURES),
        ("history_only", HISTORY_FEATURES),
        ("single_plus_history (full)", COMBINED_FEATURES),
    ]:
        oof = run_nested_cv_for_feature_set(records, groups, feats)
        summarize(f"classifier[{label}]", oof, records_by_id)

    print("=" * 78)
    print("BASELINES (same nested grouped 5-fold discipline)")
    print()

    def confidence_score(cp):
        c = cp["deer_confidence"]
        return c if c is not None else -1.0

    oof = run_nested_cv_for_rule(
        records, groups, confidence_score, sorted(set(np.linspace(0.5, 0.999, 200)) | {1.0})
    )
    summarize("baseline[confidence_only]", oof, records_by_id)

    def entropy_score(cp):
        # Lower entropy = more confident; invert so "higher score = stop-worthy".
        return -cp["entropy"]

    entropy_candidates = sorted({-cp["entropy"] for r in records for cp in r["checkpoints"]})
    oof = run_nested_cv_for_rule(records, groups, entropy_score, entropy_candidates)
    summarize("baseline[entropy_only]", oof, records_by_id)

    def stable3_score(cp):
        return 1.0 if cp["features"]["answer_same_count"] >= 3 else 0.0

    oof = run_nested_cv_for_rule(records, groups, stable3_score, [0.5])
    summarize("baseline[answer_stable_3x]", oof, records_by_id)

    def token_score(cp):
        return float(cp["token"])

    token_candidates = sorted({float(cp["token"]) for r in records for cp in r["checkpoints"]})
    oof = run_nested_cv_for_rule(records, groups, token_score, token_candidates)
    summarize("baseline[fixed_token]", oof, records_by_id)

    # Reference rows that don't fit the "calibrated OOF" framework.
    print("=" * 78)
    print("REFERENCE ROWS (not nested-CV, for scale only)")
    never_stop = [
        {
            "sample_id": r["sample_id"],
            "stopped_early": False,
            "stop_token": r["full_total_tokens"],
            "correct": r["full_correct"],
            "full_tokens": r["full_total_tokens"],
        }
        for r in records
    ]
    summarize("no_early_exit (full trace baseline)", never_stop, records_by_id)


if __name__ == "__main__":
    main()
