#!/usr/bin/env python3
"""
Compute relative L2 errors for all experiment runs against a reference solution.

Usage:
    python compute_l2_errors.py
    python compute_l2_errors.py --ref reference_solution_t10_dt0p01.npz --pred results
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml

from evaluation_utils import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_N_EVAL_TIMES,
    discover_run_dirs,
    evaluate_net_against_reference,
    first_threshold_crossing,
    make_legacy_l2_error_map,
)


class InputNormalization(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        lo = torch.tensor(lo, dtype=torch.get_default_dtype())
        hi = torch.tensor(hi, dtype=torch.get_default_dtype())
        self.register_buffer("shift", 0.5 * (lo + hi))
        self.register_buffer("scale", 2.0 / (hi - lo))

    def forward(self, x):
        return (x - self.shift) * self.scale


class OutputScaling(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x * self.s


class CahnHilliardNet(nn.Module):
    ACTIVATIONS = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(self, layer_dims, activation_name="tanh",
                 output_scale=1.0, normalize=False, input_bounds=None):
        super().__init__()
        layers = []
        if normalize and input_bounds is not None:
            lo, hi = input_bounds
            layers.append(InputNormalization(lo, hi))
        act_cls = self.ACTIVATIONS.get(activation_name, nn.Tanh)
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(act_cls())
        if output_scale != 1.0:
            layers.append(OutputScaling(output_scale))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_pinn(run_dir, device="cpu"):
    """Load a trained PINN from an experiment directory."""
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    model_path = os.path.join(run_dir, "model.pt")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dtype_str = cfg.get("dtype", "float64")
    torch_dtype = torch.float64 if dtype_str == "float64" else torch.float32
    torch.set_default_dtype(torch_dtype)

    net_cfg = cfg["network"]
    layer_dims = tuple(net_cfg["layer_dims"])
    activation_name = net_cfg.get("activation", "tanh")
    output_scaling = net_cfg.get("output_scaling", 1.0)
    normalize_inputs = cfg["domain"].get("normalize_inputs", True)

    input_bounds = (
        [cfg["domain"]["t_min"], cfg["domain"]["x_min"], cfg["domain"]["y_min"]],
        [cfg["domain"]["t_max"], cfg["domain"]["x_max"], cfg["domain"]["y_max"]],
    )

    net = CahnHilliardNet(
        layer_dims,
        activation_name=activation_name,
        output_scale=output_scaling,
        normalize=normalize_inputs,
        input_bounds=input_bounds,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    return net, cfg


def resolve_reference_path(run_dir, cfg, cli_ref):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if cli_ref:
        return cli_ref
    cfg_ref = cfg.get("logging", {}).get("reference_solution")
    if cfg_ref:
        if os.path.isabs(cfg_ref):
            return cfg_ref
        return os.path.join(repo_root, cfg_ref)
    return os.path.join(repo_root, "reference_solution_t10_dt0p01.npz")


def summarize_run(run_dir, summary, threshold, adam_phase_label, optimizer_label):
    metrics_path = os.path.join(run_dir, "metrics.npz")
    if not os.path.isfile(metrics_path):
        return summary

    metrics = None
    try:
        metrics = dict(**np.load(metrics_path))
    except Exception:
        return summary

    l2_rel_curve = metrics.get("l2_rel_curve")
    l2_rel_iters = metrics.get("l2_rel_iters")
    if l2_rel_curve is None or l2_rel_iters is None or len(l2_rel_curve) == 0:
        return summary

    l2_rel_curve = [float(v) for v in l2_rel_curve.tolist()]
    l2_rel_iters = [int(v) for v in l2_rel_iters.tolist()]
    summary["l2_rel_curve"] = l2_rel_curve
    summary["l2_threshold"] = threshold
    summary["iters_to_l2_rel_threshold"] = first_threshold_crossing(
        l2_rel_iters, l2_rel_curve, threshold
    )
    if summary["iters_to_l2_rel_threshold"] is None:
        summary["threshold_cross_phase"] = None
    elif summary["iters_to_l2_rel_threshold"] <= summary.get("adam_epochs", 0):
        summary["threshold_cross_phase"] = adam_phase_label
    else:
        summary["threshold_cross_phase"] = optimizer_label
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute relative L2 errors vs reference")
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Optional override for the reference solution path.",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="results",
        help="Root directory containing experiment output subdirectories.",
    )
    parser.add_argument(
        "--n-eval-times",
        type=int,
        default=DEFAULT_N_EVAL_TIMES,
        help="Number of uniformly spaced times to evaluate.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Batch size used when evaluating the network on the reference grid.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Relative L2 threshold used for the iteration-to-threshold summary.",
    )
    args = parser.parse_args()

    run_dirs = discover_run_dirs(args.pred)
    if not run_dirs:
        print(f"No experiment runs found under {args.pred}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} runs under {args.pred}\n")

    header = (
        f"{'Run':45s} {'Optimizer':15s} {'L2 overall':>12s} "
        f"{'L2 start':>12s} {'L2 mid':>12s} {'L2 end':>12s} "
        f"{'Thresh iters':>12s} {'Loss':>12s}"
    )
    print(header)
    print("-" * len(header))

    results = []
    by_experiment = defaultdict(list)

    for run_dir in run_dirs:
        run_label = os.path.relpath(run_dir, args.pred)
        try:
            net, cfg = load_pinn(run_dir, device=args.device)
            ref_path = resolve_reference_path(run_dir, cfg, args.ref)
            if not os.path.isfile(ref_path):
                raise FileNotFoundError(f"reference solution not found: {ref_path}")

            evaluation = evaluate_net_against_reference(
                net,
                ref_path=ref_path,
                device=args.device,
                batch_size=args.eval_batch_size,
                n_eval_times=args.n_eval_times,
            )

            summary_path = os.path.join(run_dir, "summary.json")
            if os.path.isfile(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
            else:
                summary = {}

            summary["l2_rel_error_overall"] = float(evaluation["error_overall"])
            summary["l2_rel_error_per_t"] = [
                float(val) for val in evaluation["errors_per_t"]
            ]
            summary["l2_eval_times"] = [
                float(val) for val in evaluation["t_eval"]
            ]
            summary["l2_error_overall"] = round(float(evaluation["error_overall"]), 8)
            summary["l2_error_per_t"] = make_legacy_l2_error_map(
                evaluation["t_eval"], evaluation["errors_per_t"]
            )

            optimizer = summary.get("optimizer", "?")
            adam_phase_label = cfg.get("training", {}).get("adam", {}).get(
                "optimizer", "adam"
            )
            final_loss = summary.get("final_bfgs_loss")
            if final_loss is None:
                final_loss = summary.get("final_adam_loss")
            summary = summarize_run(
                run_dir,
                summary,
                args.threshold,
                adam_phase_label,
                optimizer,
            )
            threshold_iters = summary.get("iters_to_l2_rel_threshold")

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            errors_per_t = evaluation["errors_per_t"]
            idx_start = 0
            idx_mid = len(errors_per_t) // 2
            idx_end = len(errors_per_t) - 1

            print(
                f"{run_label:45s} {optimizer:15s} "
                f"{evaluation['error_overall']:12.4e} "
                f"{errors_per_t[idx_start]:12.4e} "
                f"{errors_per_t[idx_mid]:12.4e} "
                f"{errors_per_t[idx_end]:12.4e} "
                f"{str(threshold_iters):>12s} "
                f"{(final_loss or 0.0):12.4e}"
            )

            row = {
                "run": run_label,
                "run_dir": run_dir,
                "experiment": summary.get("experiment", cfg["experiment"]["name"]),
                "optimizer": optimizer,
                "l2_rel_error_overall": float(evaluation["error_overall"]),
                "l2_rel_error_per_t": [float(v) for v in errors_per_t],
                "l2_eval_times": [float(v) for v in evaluation["t_eval"]],
                "iters_to_l2_rel_threshold": threshold_iters,
                "final_loss": final_loss,
            }
            results.append(row)
            by_experiment[row["experiment"]].append(row)
        except Exception as exc:
            print(f"{run_label:45s} ERROR: {exc}")

    out_path = os.path.join(args.pred, "l2_errors.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    by_experiment_rows = []
    for experiment, rows in sorted(by_experiment.items()):
        l2_values = [row["l2_rel_error_overall"] for row in rows]
        threshold_values = [
            row["iters_to_l2_rel_threshold"]
            for row in rows
            if row["iters_to_l2_rel_threshold"] is not None
        ]
        by_experiment_rows.append({
            "experiment": experiment,
            "n_runs": len(rows),
            "l2_rel_error_mean": float(sum(l2_values) / len(l2_values)),
            "l2_rel_error_std": (
                float(np.std(l2_values, ddof=0))
                if len(l2_values) > 1 else 0.0
            ),
            "iters_to_l2_rel_threshold_mean": (
                float(sum(threshold_values) / len(threshold_values))
                if threshold_values else None
            ),
            "iters_to_l2_rel_threshold_std": (
                float(np.std(threshold_values, ddof=0))
                if len(threshold_values) > 1 else 0.0
            ) if threshold_values else None,
        })

    out_by_experiment = os.path.join(args.pred, "l2_errors_by_experiment.json")
    with open(out_by_experiment, "w") as f:
        json.dump(by_experiment_rows, f, indent=2)

    print(f"\nPer-run results saved to {out_path}")
    print(f"Experiment summary saved to {out_by_experiment}")

    if results:
        best = min(results, key=lambda row: row["l2_rel_error_overall"])
        print(
            f"\nBest overall L2: {best['run']} -> "
            f"{best['l2_rel_error_overall']:.4e}"
        )


if __name__ == "__main__":
    main()
