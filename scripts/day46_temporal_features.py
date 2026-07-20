"""Day 4.6: does genuine confidence *temporal dynamics* help early-exit
decisions, beyond a single-point snapshot?

Motivation: day45_rigorous_eval.py's "history_only"/"combined" feature sets
were answer_same_count/answer_changed/reasoning_length -- answer-stability
proxies, not actual confidence trajectory. That ablation could only answer
"does answer stability help", not "does the trajectory of confidence over
time carry independent signal". This script builds real causal temporal
features (confidence delta, short-window slope/std, max historical drop,
entropy slope, cumulative answer-change count, a confidence/answer-change
conflict signal) and compares three nested feature sets under the exact
same nested-CV/first-crossing-stop protocol as day45:

  M0 = confidence alone
  M1 = pointwise snapshot (confidence, entropy, margin, avg/min logprob,
       reasoning_length) -- renamed/clarified version of day45's
       "single_point"
  M2 = M1 + genuine temporal-dynamics features

All temporal features at checkpoint i are computed only from
checkpoints[0..i] of that record (no lookahead). Comparisons are made at
matched risk (target stop-error swept, not a single point), plus a paired
bootstrap on the M2-M1 delta at the original 5% target so a 0.7-point
difference can be judged against its own uncertainty instead of taken at
face value.

Reads results/checkpoints_math500_93_v3.jsonl (does not modify it). Pure
CPU / sklearn. No vLLM, no SSH, no AutoDL, no GPU involved anywhere in
this file.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHECKPOINTS_PATH = "results/checkpoints_math500_93_v3.jsonl"
SEED = 42
N_OUTER_FOLDS = 5
N_BOOTSTRAP = 2000
PRIMARY_TARGET_ERROR = 0.05
RISK_SWEEP = [0.03, 0.05, 0.10, 0.15]

POINTWISE_FEATURE_NAMES = [
    "deer_confidence",
    "entropy",
    "margin",
    "avg_logprob",
    "min_logprob",
    "reasoning_length",
    "probe_complete",
]
TEMPORAL_FEATURE_NAMES = [
    "confidence_delta",
    "confidence_slope3",
    "confidence_std3",
    "confidence_max_drop",
    "entropy_slope3",
    "answer_streak",
    "answer_change_count",
    "confidence_answer_conflict",
]
M0_FEATURES = ["deer_confidence"]
M1_FEATURES = POINTWISE_FEATURE_NAMES
M2_FEATURES = POINTWISE_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES


def load_records(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _streak(answer_hist: list[str | None]) -> int:
    if not answer_hist or answer_hist[-1] is None:
        return 0
    count = 1
    for v in reversed(answer_hist[:-1]):
        if v != answer_hist[-1]:
            break
        count += 1
    return count


def causal_temporal_features(checkpoints: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Per-checkpoint temporal features using only checkpoints[0..i]."""
    conf_hist: list[float] = []
    ent_hist: list[float] = []
    answer_hist: list[str | None] = []
    change_count = 0
    out = []
    for cp in checkpoints:
        # Use the already-safe features dict (0.0 for incomplete probes,
        # with probe_complete=0.0 marking that explicitly), not the raw
        # top-level deer_confidence which is None for incomplete probes.
        conf = float(cp["features"]["deer_confidence"])
        ent = float(cp["features"]["entropy"])
        ans = cp["trial_answer"]

        prev_answer = answer_hist[-1] if answer_hist else None
        changed_now = bool(answer_hist) and (ans is None or ans != prev_answer)
        if changed_now:
            change_count += 1

        conf_hist.append(conf)
        ent_hist.append(ent)
        answer_hist.append(ans)

        window_conf = conf_hist[-3:]
        window_ent = ent_hist[-3:]
        confidence_delta = conf_hist[-1] - conf_hist[-2] if len(conf_hist) >= 2 else 0.0
        confidence_max_drop = max(
            (conf_hist[j - 1] - conf_hist[j] for j in range(1, len(conf_hist))), default=0.0
        )
        out.append(
            {
                "confidence_delta": confidence_delta,
                "confidence_slope3": _slope(window_conf),
                "confidence_std3": float(np.std(window_conf)) if len(window_conf) >= 2 else 0.0,
                "confidence_max_drop": max(confidence_max_drop, 0.0),
                "entropy_slope3": _slope(window_ent),
                "answer_streak": float(_streak(answer_hist)),
                "answer_change_count": float(change_count),
                "confidence_answer_conflict": conf if changed_now else 0.0,
            }
        )
    return out


def augment_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach features_v2 (pointwise + temporal, merged) to every checkpoint."""
    augmented = []
    for r in records:
        temporal = causal_temporal_features(r["checkpoints"])
        new_checkpoints = []
        for cp, temp in zip(r["checkpoints"], temporal):
            # cp["features"] already has safe (0.0-substituted) values for
            # incomplete probes plus probe_complete; don't overwrite with the
            # raw (possibly None) top-level fields.
            merged = dict(cp["features"])
            merged.update(temp)
            new_cp = dict(cp)
            new_cp["features_v2"] = merged
            new_checkpoints.append(new_cp)
        new_r = dict(r)
        new_r["checkpoints"] = new_checkpoints
        augmented.append(new_r)
    return augmented


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def feature_vector_for(cp: dict[str, Any], names: list[str]) -> list[float]:
    return [cp["features_v2"][name] for name in names]


def probe_token_cost(cp: dict[str, Any]) -> int:
    return len(cp.get("trial_answer_logprobs") or [])


def simulate_sequential_policy(
    records: list[dict[str, Any]], score_fn: Callable[[dict], float], threshold: float
) -> list[dict[str, Any]]:
    """Walk checkpoints in order; stop at the first complete-probe crossing.
    Also tracks probe_tokens_spent = sum of tokens actually generated by
    every probe attempted up to (and including) the stopping checkpoint, so
    "net of probing overhead" savings can be reported alongside the naive
    number."""
    results = []
    for r in records:
        stopped = None
        probe_cost = 0
        for cp in r["checkpoints"]:
            probe_cost += probe_token_cost(cp)
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
                    "probe_tokens_spent": probe_cost,
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
                    "probe_tokens_spent": probe_cost,
                }
            )
    return results


def calibrate_threshold(
    records: list[dict[str, Any]],
    score_fn: Callable[[dict], float],
    candidate_thresholds: list[float],
    target_error: float,
) -> float:
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


def make_classifier_score_fn(model, names: list[str]) -> Callable[[dict], float]:
    def score(cp: dict[str, Any]) -> float:
        return float(model.predict_proba([feature_vector_for(cp, names)])[0][1])

    return score


def run_nested_cv(
    records: list[dict[str, Any]], groups: list[str], feature_names: list[str], target_error: float
) -> list[dict[str, Any]]:
    """Grouped outer 5-fold CV, inner fit/calibration split, OOF-only report.
    Identical protocol to day45_rigorous_eval.py's run_nested_cv_for_feature_set."""
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
        fit_ids = {inner_groups[i] for i in fit_idx}
        calib_ids = {inner_groups[i] for i in calib_idx}

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        )
        model.fit(x[fit_idx], y[fit_idx])
        score_fn = make_classifier_score_fn(model, feature_names)

        calib_records = [r for r in outer_train_records if r["sample_id"] in calib_ids]
        candidate_thresholds = sorted(set(np.linspace(0.5, 0.999, 200)) | {1.0})
        threshold = calibrate_threshold(calib_records, score_fn, candidate_thresholds, target_error)

        oof_results.extend(simulate_sequential_policy(outer_test_records, score_fn, threshold))

    return oof_results


def metrics(oof: list[dict[str, Any]]) -> dict[str, float]:
    n = len(oof)
    n_correct = sum(1 for s in oof if s["correct"])
    n_stopped = sum(1 for s in oof if s["stopped_early"])
    n_wrong_stopped = sum(1 for s in oof if s["stopped_early"] and not s["correct"])
    total_full = sum(s["full_tokens"] for s in oof)
    total_saved_naive = sum(max(0, s["full_tokens"] - s["stop_token"]) for s in oof)
    total_probe_cost = sum(s["probe_tokens_spent"] for s in oof)
    total_saved_net = total_saved_naive - total_probe_cost
    saved_fracs = [
        max(0, s["full_tokens"] - s["stop_token"]) / s["full_tokens"] if s["full_tokens"] else 0.0
        for s in oof
    ]
    lo, hi = wilson_ci(n_wrong_stopped, n_stopped) if n_stopped else (0.0, 0.0)
    return {
        "n": n,
        "accuracy": n_correct / n if n else 0.0,
        "stop_rate": n_stopped / n if n else 0.0,
        "stop_error": n_wrong_stopped / n_stopped if n_stopped else float("nan"),
        "stop_error_ci_lo": lo,
        "stop_error_ci_hi": hi,
        "micro_saved_naive": total_saved_naive / total_full if total_full else 0.0,
        "micro_saved_net_of_probe_cost": total_saved_net / total_full if total_full else 0.0,
        "mean_saved_fraction": statistics.mean(saved_fracs) if saved_fracs else 0.0,
    }


def print_row(label: str, m: dict[str, float]) -> None:
    err = "n/a" if math.isnan(m["stop_error"]) else f"{m['stop_error']:.4f}"
    ci = "" if math.isnan(m["stop_error"]) else f" CI[{m['stop_error_ci_lo']:.4f},{m['stop_error_ci_hi']:.4f}]"
    print(
        f"  {label:34s} acc={m['accuracy']:.4f} stop_rate={m['stop_rate']:.4f} "
        f"stop_error={err}{ci} saved_naive={m['micro_saved_naive']:.4f} "
        f"saved_net_of_probes={m['micro_saved_net_of_probe_cost']:.4f}"
    )


def paired_bootstrap_delta(
    oof_a: list[dict[str, Any]], oof_b: list[dict[str, Any]], n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, tuple[float, float, float]]:
    """B minus A, paired by sample_id (both ran under the same grouped folds,
    so every sample_id has exactly one OOF outcome in each list)."""
    by_id_a = {s["sample_id"]: s for s in oof_a}
    by_id_b = {s["sample_id"]: s for s in oof_b}
    ids = sorted(set(by_id_a) & set(by_id_b), key=int)
    rng = random.Random(SEED)

    def scalar_metrics(subset_ids: list[str]) -> dict[str, float]:
        a = [by_id_a[i] for i in subset_ids]
        b = [by_id_b[i] for i in subset_ids]
        return {
            "saved_fraction": statistics.mean(
                [
                    max(0, s["full_tokens"] - s["stop_token"]) / s["full_tokens"] if s["full_tokens"] else 0.0
                    for s in b
                ]
            )
            - statistics.mean(
                [
                    max(0, s["full_tokens"] - s["stop_token"]) / s["full_tokens"] if s["full_tokens"] else 0.0
                    for s in a
                ]
            ),
            "accuracy": statistics.mean([1.0 if s["correct"] else 0.0 for s in b])
            - statistics.mean([1.0 if s["correct"] else 0.0 for s in a]),
        }

    point = scalar_metrics(ids)
    boot: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_bootstrap):
        sample_ids = [rng.choice(ids) for _ in ids]
        m = scalar_metrics(sample_ids)
        for k, v in m.items():
            boot[k].append(v)

    out = {}
    for k, v in point.items():
        lo = float(np.percentile(boot[k], 2.5))
        hi = float(np.percentile(boot[k], 97.5))
        out[k] = (v, lo, hi)
    return out


def main() -> None:
    records = augment_records(load_records(CHECKPOINTS_PATH))
    groups = [r["sample_id"] for r in records]

    print("=" * 78)
    print(f"Day 4.6: causal temporal-dynamics features, N={len(records)}, {N_OUTER_FOLDS}-fold grouped CV")
    print("=" * 78)
    print()
    print("M0 =", M0_FEATURES)
    print("M1 =", M1_FEATURES)
    print("M2 = M1 +", TEMPORAL_FEATURE_NAMES)
    print()

    print("-" * 78)
    print(f"PRIMARY COMPARISON at target_error={PRIMARY_TARGET_ERROR} (matches day45 protocol)")
    print("-" * 78)
    oof = {}
    for label, feats in [("M0[confidence-only]", M0_FEATURES), ("M1[pointwise]", M1_FEATURES), ("M2[pointwise+temporal]", M2_FEATURES)]:
        oof[label] = run_nested_cv(records, groups, feats, PRIMARY_TARGET_ERROR)
        print_row(label, metrics(oof[label]))
    print()

    print("-" * 78)
    print("MATCHED-RISK SWEEP (same target_error across M0/M1/M2, not a single point)")
    print("-" * 78)
    for target in RISK_SWEEP:
        print(f"target_error={target}:")
        for label, feats in [("M0", M0_FEATURES), ("M1", M1_FEATURES), ("M2", M2_FEATURES)]:
            oof_t = run_nested_cv(records, groups, feats, target)
            print_row(f"  {label}", metrics(oof_t))
    print()

    print("-" * 78)
    print(f"PAIRED BOOTSTRAP, M2 - M1, target_error={PRIMARY_TARGET_ERROR}, {N_BOOTSTRAP} resamples, paired by sample_id")
    print("-" * 78)
    deltas = paired_bootstrap_delta(oof["M1[pointwise]"], oof["M2[pointwise+temporal]"])
    for name, (point, lo, hi) in deltas.items():
        print(f"  {name} delta (M2-M1): {point:+.4f}  95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    print()

    print("=" * 78)
    print("LIMITATIONS (must be read alongside the numbers above)")
    print("=" * 78)
    print(
        "1. Easy-case bias: this N=93 excludes the 7 records that didn't finish even at\n"
        "   8192 tokens (see log.md Day 3). Those excluded records are exactly the\n"
        "   longest/hardest reasoning traces in the sample, so this N=93 is biased\n"
        "   toward easier problems than the full MATH-500-derived 100-question set.\n"
        "2. Probing overhead: 'saved_naive' matches the framing used for the Day 3\n"
        "   oracle number and day45's original report -- it only subtracts the tokens\n"
        "   after the stop point, ignoring the cost of the probes themselves.\n"
        "   'saved_net_of_probe_cost' subtracts the actual tokens generated by every\n"
        "   probe attempted up to and including the stopping checkpoint (or all\n"
        "   checkpoints, for records that never stop). It still does not include any\n"
        "   KV-cache/prefix-recomputation cost of re-running the growing prefix at\n"
        "   each checkpoint, so even the 'net' number remains optimistic relative to a\n"
        "   real online deployment.\n"
        "3. No stop-error guarantee is claimed anywhere in this report: only the\n"
        "   observed out-of-fold rate and its 95% Wilson CI. At this sample size the\n"
        "   CI is wide enough that 'meets the 5% target' cannot be asserted from a\n"
        "   point estimate alone."
    )


if __name__ == "__main__":
    main()
