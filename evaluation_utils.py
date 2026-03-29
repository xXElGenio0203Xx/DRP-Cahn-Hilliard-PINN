#!/usr/bin/env python3
"""Shared utilities for reference evaluation and run discovery."""

import os

import numpy as np
import torch


DEFAULT_N_EVAL_TIMES = 21
DEFAULT_EVAL_BATCH_SIZE = 16384


def load_reference_solution(ref_path, n_eval_times=DEFAULT_N_EVAL_TIMES):
    """Load the reference solution and select uniformly spaced evaluation times."""
    ref = np.load(ref_path)
    t_ref = ref["t"]
    xs_ref = ref["x"]
    ys_ref = ref["y"]
    u_ref_full = ref["U"]

    if n_eval_times is None or n_eval_times >= len(t_ref):
        eval_indices = np.arange(len(t_ref), dtype=int)
    else:
        eval_indices = np.linspace(0, len(t_ref) - 1, n_eval_times, dtype=int)

    t_eval = t_ref[eval_indices]
    u_ref = u_ref_full[eval_indices]
    return {
        "t_ref": t_ref,
        "x_ref": xs_ref,
        "y_ref": ys_ref,
        "u_ref_full": u_ref_full,
        "eval_indices": eval_indices,
        "t_eval": t_eval,
        "u_ref": u_ref,
    }


def evaluate_pinn_on_grid(
    net,
    t_vals,
    xs,
    ys,
    device="cpu",
    batch_size=DEFAULT_EVAL_BATCH_SIZE,
    predict_fn=None,
):
    """Evaluate a network on every (t, x, y) point of a structured grid."""
    nx, ny = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    n_spatial = nx * ny
    u = np.zeros((len(t_vals), ny, nx))

    net.eval()
    for i, t_val in enumerate(t_vals):
        tt = np.full(n_spatial, t_val)
        coords = np.column_stack([tt, xx.ravel(), yy.ravel()])

        u_all = []
        for start in range(0, n_spatial, batch_size):
            end = min(start + batch_size, n_spatial)
            x_batch = torch.tensor(
                coords[start:end],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            with torch.no_grad():
                if predict_fn is None:
                    pred_batch = net(x_batch)
                else:
                    pred_batch = predict_fn(net, x_batch)
            if pred_batch.ndim == 2:
                pred_batch = pred_batch[:, 0]
            u_all.append(pred_batch.detach().cpu().numpy())

        u[i] = np.concatenate(u_all).reshape(ny, nx)

    return u


def compute_l2_errors(u_pinn, u_ref):
    """Compute relative L2 error at each time and overall."""
    n_t = u_ref.shape[0]
    errors_per_t = np.zeros(n_t)

    for i in range(n_t):
        diff = u_pinn[i] - u_ref[i]
        errors_per_t[i] = np.linalg.norm(diff) / (np.linalg.norm(u_ref[i]) + 1e-30)

    diff_all = u_pinn - u_ref
    error_overall = np.linalg.norm(diff_all) / (np.linalg.norm(u_ref) + 1e-30)
    return errors_per_t, float(error_overall)


def evaluate_net_against_reference(
    net,
    ref_path=None,
    *,
    ref_data=None,
    device="cpu",
    batch_size=DEFAULT_EVAL_BATCH_SIZE,
    n_eval_times=DEFAULT_N_EVAL_TIMES,
    predict_fn=None,
):
    """Evaluate a network against a saved reference solution."""
    if ref_data is None:
        if ref_path is None:
            raise ValueError("Either ref_path or ref_data must be provided.")
        ref = load_reference_solution(ref_path, n_eval_times=n_eval_times)
    else:
        ref = ref_data
    u_pinn = evaluate_pinn_on_grid(
        net,
        ref["t_eval"],
        ref["x_ref"],
        ref["y_ref"],
        device=device,
        batch_size=batch_size,
        predict_fn=predict_fn,
    )
    errors_per_t, error_overall = compute_l2_errors(u_pinn, ref["u_ref"])
    return {
        "t_eval": ref["t_eval"],
        "errors_per_t": errors_per_t,
        "error_overall": error_overall,
        "u_pinn": u_pinn,
        "u_ref": ref["u_ref"],
    }


def make_legacy_l2_error_map(t_eval, errors_per_t):
    """Compatibility map matching the older summary.json structure."""
    return {
        f"t_{float(t_val):.1f}": round(float(err), 8)
        for t_val, err in zip(t_eval, errors_per_t)
    }


def first_threshold_crossing(iterations, values, threshold):
    """Return the first iteration at which a curve crosses a threshold."""
    for iteration, value in zip(iterations, values):
        if value <= threshold:
            return int(iteration)
    return None


def discover_run_dirs(root_dir):
    """Recursively find experiment output directories containing a trained model."""
    run_dirs = []
    if not os.path.isdir(root_dir):
        return run_dirs

    for dirpath, _, filenames in os.walk(root_dir):
        if "model.pt" in filenames and "config_used.yaml" in filenames:
            run_dirs.append(dirpath)

    run_dirs.sort()
    return run_dirs
