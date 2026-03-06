#!/usr/bin/env python3
"""
Compute relative ℓ₂ errors for all PINN experiments against the spectral
reference solution.

Usage:
    python compute_l2_errors.py                        # defaults
    python compute_l2_errors.py --ref reference_solution.npz --pred predictions/

Reads each experiment's model.pt + config_used.yaml, reconstructs the PINN,
evaluates it on the reference grid, and computes:
    relative ℓ₂ error = ‖U_PINN − U_ref‖₂ / ‖U_ref‖₂

Results are appended to each experiment's summary.json and printed as a table.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

import torch
import torch.nn as nn


# ============================================================================
# Network definition (must match run_experiment.py exactly)
# ============================================================================

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


def variance_scaling_init(linear, scale=1.0):
    fi, fo = linear.in_features, linear.out_features
    limit = math.sqrt(3.0 * scale / (0.5 * (fi + fo)))
    with torch.no_grad():
        linear.weight.uniform_(-limit, limit)
        if linear.bias is not None:
            linear.bias.zero_()


class CahnHilliardNet(nn.Module):
    ACTIVATIONS = {
        "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU,
        "silu": nn.SiLU, "sigmoid": nn.Sigmoid,
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
        # Apply variance scaling init to last linear layer
        for m in reversed(list(self.net.modules())):
            if isinstance(m, nn.Linear):
                sc = 1.0 / (output_scale ** 2) if output_scale != 0 else 1.0
                variance_scaling_init(m, scale=sc)
                break

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Reconstruct and evaluate PINN
# ============================================================================

def load_pinn(exp_dir, device="cpu"):
    """Load a trained PINN from an experiment directory."""
    cfg_path = os.path.join(exp_dir, "config_used.yaml")
    model_path = os.path.join(exp_dir, "model.pt")

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


def evaluate_pinn_on_grid(net, t_vals, xs, ys, device="cpu", batch_size=16384):
    """
    Evaluate the PINN at every (t, x, y) grid point.

    Parameters
    ----------
    net : CahnHilliardNet
    t_vals : (n_t,) array of time values
    xs, ys : (nx,), (ny,) coordinate arrays
    device : torch device

    Returns
    -------
    U : (n_t, ny, nx) array of PINN predictions
    """
    nx, ny = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    n_spatial = nx * ny
    U = np.zeros((len(t_vals), ny, nx))

    for i, t_val in enumerate(t_vals):
        tt = np.full(n_spatial, t_val)
        coords = np.column_stack([tt, xx.ravel(), yy.ravel()])

        # Evaluate in batches to avoid OOM
        u_all = []
        for start in range(0, n_spatial, batch_size):
            end = min(start + batch_size, n_spatial)
            X = torch.tensor(
                coords[start:end],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            with torch.no_grad():
                u_batch = net(X)[:, 0].cpu().numpy()
            u_all.append(u_batch)

        U[i] = np.concatenate(u_all).reshape(ny, nx)

    return U


# ============================================================================
# Error computation
# ============================================================================

def compute_l2_errors(U_pinn, U_ref, t_vals):
    """
    Compute relative ℓ₂ error at each time and overall.

    Returns
    -------
    errors_per_t : (n_t,) array — relative ℓ₂ error at each saved time
    error_overall : float — relative ℓ₂ error across all time and space
    """
    n_t = len(t_vals)
    errors_per_t = np.zeros(n_t)

    for i in range(n_t):
        diff = U_pinn[i] - U_ref[i]
        errors_per_t[i] = np.linalg.norm(diff) / (np.linalg.norm(U_ref[i]) + 1e-30)

    # Overall error (Frobenius norm across all time × space)
    diff_all = U_pinn - U_ref
    error_overall = np.linalg.norm(diff_all) / (np.linalg.norm(U_ref) + 1e-30)

    return errors_per_t, error_overall


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute ℓ₂ errors vs reference")
    parser.add_argument("--ref", type=str, default="reference_solution.npz",
                        help="Path to reference solution .npz")
    parser.add_argument("--pred", type=str, default="predictions",
                        help="Directory containing experiment subdirectories")
    parser.add_argument("--n-eval-times", type=int, default=21,
                        help="Number of uniformly spaced times to evaluate")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load reference
    ref = np.load(args.ref)
    t_ref = ref["t"]       # (101,)
    xs_ref = ref["x"]      # (128,)
    ys_ref = ref["y"]      # (128,)
    U_ref_full = ref["U"]  # (101, 128, 128)

    # Select subset of times for evaluation
    eval_indices = np.linspace(0, len(t_ref) - 1, args.n_eval_times, dtype=int)
    t_eval = t_ref[eval_indices]
    U_ref = U_ref_full[eval_indices]

    print(f"Reference: {args.ref}  ({len(t_ref)} snapshots, using {len(t_eval)} for eval)")
    print(f"Grid: {len(xs_ref)}×{len(ys_ref)},  t ∈ [{t_eval[0]:.1f}, {t_eval[-1]:.1f}]")
    print()

    # Find experiments
    exp_dirs = sorted([
        d for d in os.listdir(args.pred)
        if os.path.isdir(os.path.join(args.pred, d))
        and os.path.isfile(os.path.join(args.pred, d, "model.pt"))
    ])

    if not exp_dirs:
        print(f"No experiments found in {args.pred}")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} experiments.\n")

    # Results table
    results = []
    header = f"{'Experiment':30s} {'Optimizer':15s} {'ℓ₂ rel err':>14s} {'ℓ₂(t=0)':>12s} {'ℓ₂(t=10)':>12s} {'ℓ₂(t=20)':>12s} {'Loss':>12s}"
    print(header)
    print("-" * len(header))

    for exp_name in exp_dirs:
        exp_path = os.path.join(args.pred, exp_name)
        try:
            net, cfg = load_pinn(exp_path, device=args.device)
            U_pinn = evaluate_pinn_on_grid(net, t_eval, xs_ref, ys_ref,
                                           device=args.device)
            errors_per_t, error_overall = compute_l2_errors(U_pinn, U_ref, t_eval)

            # Get the loss from summary
            summary_path = os.path.join(exp_path, "summary.json")
            with open(summary_path, "r") as f:
                summary = json.load(f)
            loss = summary.get("final_bfgs_loss") or summary.get("final_adam_loss") or 0
            optimizer = summary.get("optimizer", "?")

            # Find indices closest to t=0, t=10, t=20
            idx_0 = np.argmin(np.abs(t_eval - 0.0))
            idx_10 = np.argmin(np.abs(t_eval - 10.0))
            idx_20 = np.argmin(np.abs(t_eval - 20.0))

            print(f"{exp_name:30s} {optimizer:15s} {error_overall:>14.4e} "
                  f"{errors_per_t[idx_0]:>12.4e} "
                  f"{errors_per_t[idx_10]:>12.4e} "
                  f"{errors_per_t[idx_20]:>12.4e} "
                  f"{loss:>12.4e}")

            # Update summary.json with ℓ₂ errors
            summary["l2_rel_error_overall"] = float(error_overall)
            summary["l2_rel_error_t0"] = float(errors_per_t[idx_0])
            summary["l2_rel_error_t10"] = float(errors_per_t[idx_10])
            summary["l2_rel_error_t20"] = float(errors_per_t[idx_20])
            summary["l2_rel_error_per_t"] = errors_per_t.tolist()
            summary["l2_eval_times"] = t_eval.tolist()

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            results.append({
                "experiment": exp_name,
                "optimizer": optimizer,
                "l2_overall": error_overall,
                "l2_t0": errors_per_t[idx_0],
                "l2_t10": errors_per_t[idx_10],
                "l2_t20": errors_per_t[idx_20],
                "loss": loss,
            })

        except Exception as e:
            print(f"{exp_name:30s} ERROR: {e}")

    # Save combined results
    out_path = os.path.join(args.pred, "l2_errors.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary statistics
    print("\n=== Best results ===")
    if results:
        best = min(results, key=lambda r: r["l2_overall"])
        print(f"  Best overall ℓ₂: {best['experiment']}  →  {best['l2_overall']:.4e}")
        best_t0 = min(results, key=lambda r: r["l2_t0"])
        print(f"  Best ℓ₂(t=0):    {best_t0['experiment']}  →  {best_t0['l2_t0']:.4e}")
        best_t20 = min(results, key=lambda r: r["l2_t20"])
        print(f"  Best ℓ₂(t=20):   {best_t20['experiment']}  →  {best_t20['l2_t20']:.4e}")


if __name__ == "__main__":
    main()
