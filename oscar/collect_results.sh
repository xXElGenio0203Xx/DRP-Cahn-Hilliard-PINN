#!/bin/bash
# =============================================================================
# collect_results.sh — Aggregate seeded experiment summaries recursively
# =============================================================================
#
# Usage:
#   bash oscar/collect_results.sh
#   bash oscar/collect_results.sh results
#
# Output:
#   results_summary_per_seed.csv
#   results_summary_by_experiment.csv
# =============================================================================

set -euo pipefail

RESULT_ROOT="${1:-results}"

python3 - "$RESULT_ROOT" <<'PY'
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

result_root = sys.argv[1]
repo_root = os.getcwd()
abs_root = os.path.join(repo_root, result_root)

if not os.path.isdir(abs_root):
    raise SystemExit(f"Results directory not found: {abs_root}")

rows = []
for dirpath, _, filenames in os.walk(abs_root):
    if "summary.json" not in filenames:
        continue
    summary_path = os.path.join(dirpath, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    seed = None
    base = os.path.basename(dirpath)
    if base.startswith("seed_"):
        try:
            seed = int(base.split("_", 1)[1])
        except ValueError:
            seed = base

    rows.append({
        "run": os.path.relpath(dirpath, abs_root),
        "experiment": summary.get("experiment", os.path.basename(os.path.dirname(dirpath))),
        "seed": seed,
        "optimizer": summary.get("optimizer"),
        "adam_epochs": summary.get("adam_epochs"),
        "bfgs_iters": summary.get("bfgs_iters"),
        "adam_time_s": summary.get("adam_time_s"),
        "bfgs_time_s": summary.get("bfgs_time_s"),
        "total_time_s": summary.get("total_time_s"),
        "final_adam_loss": summary.get("final_adam_loss"),
        "final_bfgs_loss": summary.get("final_bfgs_loss"),
        "l2_rel_error_overall": summary.get("l2_rel_error_overall"),
        "iters_to_l2_rel_threshold": summary.get("iters_to_l2_rel_threshold"),
        "threshold_cross_phase": summary.get("threshold_cross_phase"),
        "mass_drift": summary.get("mass_drift"),
        "n_params": summary.get("n_params"),
    })

rows.sort(key=lambda row: (row["experiment"] or "", row["seed"] or -1, row["run"]))

per_seed_path = os.path.join(abs_root, "results_summary_per_seed.csv")
with open(per_seed_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
        "run", "experiment", "seed", "optimizer", "adam_epochs", "bfgs_iters",
        "adam_time_s", "bfgs_time_s", "total_time_s", "final_adam_loss",
        "final_bfgs_loss", "l2_rel_error_overall", "iters_to_l2_rel_threshold",
        "threshold_cross_phase", "mass_drift", "n_params",
    ])
    writer.writeheader()
    writer.writerows(rows)

grouped = defaultdict(list)
for row in rows:
    grouped[row["experiment"]].append(row)

def mean_or_none(values):
    return sum(values) / len(values) if values else None

def std_or_none(values):
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)

agg_rows = []
for experiment, items in sorted(grouped.items()):
    l2_values = [
        row["l2_rel_error_overall"] for row in items
        if isinstance(row["l2_rel_error_overall"], (int, float)) and math.isfinite(row["l2_rel_error_overall"])
    ]
    threshold_values = [
        row["iters_to_l2_rel_threshold"] for row in items
        if isinstance(row["iters_to_l2_rel_threshold"], (int, float))
    ]
    agg_rows.append({
        "experiment": experiment,
        "n_runs": len(items),
        "optimizer": items[0].get("optimizer"),
        "l2_rel_error_mean": mean_or_none(l2_values),
        "l2_rel_error_std": std_or_none(l2_values),
        "iters_to_l2_rel_threshold_mean": mean_or_none(threshold_values),
        "iters_to_l2_rel_threshold_std": std_or_none(threshold_values),
        "best_l2_rel_error": min(l2_values) if l2_values else None,
        "best_run": min(items, key=lambda row: row["l2_rel_error_overall"] if isinstance(row["l2_rel_error_overall"], (int, float)) else float("inf"))["run"],
    })

agg_path = os.path.join(abs_root, "results_summary_by_experiment.csv")
with open(agg_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows else [
        "experiment", "n_runs", "optimizer", "l2_rel_error_mean", "l2_rel_error_std",
        "iters_to_l2_rel_threshold_mean", "iters_to_l2_rel_threshold_std",
        "best_l2_rel_error", "best_run",
    ])
    writer.writeheader()
    writer.writerows(agg_rows)

print(f"Wrote {per_seed_path}")
print(f"Wrote {agg_path}")
PY
