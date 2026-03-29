#!/usr/bin/env python3
"""
Cahn-Hilliard PINN — Headless Training Script
==============================================
Converted from Cahn_Hilliard_Pytorch.ipynb for batch execution on OSCAR.

Usage:
    python -u run_experiment.py configs/v2_core/A5_ssbroyden2.yaml
    python -u run_experiment.py configs/v2_core/A5_ssbroyden2.yaml --seed 1
    python -u run_experiment.py configs/v2_fh_ablation/H2_rad_k1_2p0.yaml --seed 3

The first positional argument is the path to a YAML configuration file.
Use --seed N to override the random seed from the config; results are
then placed in results_dir/seed_N/ to keep multi-seed runs separate.

All results (model weights, loss curves, plots, metrics) are saved to
the directory specified by logging.results_dir in the config (optionally
suffixed with /seed_N).

Use `python -u` (unbuffered) so SLURM captures output in real time.
"""

import os
import sys
import math
import json
import yaml
import argparse
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for OSCAR (no display)
import matplotlib.pyplot as plt

from evaluation_utils import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_N_EVAL_TIMES,
    evaluate_net_against_reference,
    first_threshold_crossing,
    load_reference_solution,
    make_legacy_l2_error_map,
)

# ============================================================================
# 0. PARSE CLI ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description="Cahn-Hilliard PINN — Batch Training on OSCAR"
)
parser.add_argument(
    "config", help="Path to YAML configuration file"
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Override the random seed from config. Results are placed in "
         "results_dir/seed_N/ to keep multi-seed runs separate."
)
args = parser.parse_args()

CONFIG_PATH = args.config
if not os.path.isfile(CONFIG_PATH):
    print(f"ERROR: Config file not found: {CONFIG_PATH}")
    sys.exit(1)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

print(f"=" * 70)
print(f"Cahn-Hilliard PINN — Batch Training")
print(f"Config: {CONFIG_PATH}")
if args.seed is not None:
    print(f"Seed override: {args.seed}")
print(f"=" * 70)

# ============================================================================
# 1. LOAD CONFIGURATION
# ============================================================================

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

D  = cfg["equation"]["D"]
a2 = cfg["equation"]["a2"]
a4 = cfg["equation"]["a4"]

x_min = cfg["domain"]["x_min"]
x_max = cfg["domain"]["x_max"]
y_min = cfg["domain"]["y_min"]
y_max = cfg["domain"]["y_max"]
t_min = cfg["domain"]["t_min"]
t_max = cfg["domain"]["t_max"]
Lx = x_max - x_min
Ly = y_max - y_min
normalize_inputs = cfg["domain"].get("normalize_inputs", True)

ic_cfg        = cfg["initial_condition"]
ic_type       = ic_cfg["type"]
ic_seed       = ic_cfg.get("seed", 42)
ic_grid_nx    = ic_cfg.get("grid_nx", 128)
ic_grid_ny    = ic_cfg.get("grid_ny", 128)
ic_smooth_sig = ic_cfg.get("smoothing_sigma", 2.0)

bc_cfg         = cfg["boundary_conditions"]
bc_type        = bc_cfg["type"]
bc_enforcement = bc_cfg.get("enforcement", "soft")
bc_deriv_order = bc_cfg.get("derivative_order", 0)

net_cfg        = cfg["network"]
layer_dims     = tuple(net_cfg["layer_dims"])
activation_name = net_cfg.get("activation", "tanh")
output_scaling = net_cfg.get("output_scaling", 1.0)

adam_cfg = cfg["training"]["adam"]
bfgs_cfg = cfg["training"]["bfgs"]

Nepochs_ADAM   = adam_cfg["epochs"]
adam_optimizer = adam_cfg.get("optimizer", "adam").lower()  # "adam" or "radam"
adam_lr        = adam_cfg["lr"]
adam_betas     = tuple(adam_cfg["betas"])
adam_eps        = adam_cfg["eps"]
lr_decay_rate  = adam_cfg.get("lr_decay_rate", 0.98)
lr_decay_steps = adam_cfg.get("lr_decay_steps", 1000)

Nbfgs          = bfgs_cfg["epochs"]
Nchange        = bfgs_cfg["batch_size"]
bfgs_method    = bfgs_cfg["method"]
bfgs_variant   = bfgs_cfg.get("variant", "none")
bfgs_init_scale = bfgs_cfg.get("initial_scale", False)
bfgs_power     = bfgs_cfg.get("power", 1.0)
bfgs_maxcor    = bfgs_cfg.get("maxcor", 20)

samp_cfg   = cfg["sampling"]
Nint       = samp_cfg["n_interior"]
N0         = samp_cfg["n_initial"]
Nb         = samp_cfg["n_boundary"]
Nresample  = samp_cfg["resample_every"]
rad_cfg    = samp_cfg.get("rad", {})
rad_on     = rad_cfg.get("enabled", True)
k1         = rad_cfg.get("k1", 1.0)
k2         = rad_cfg.get("k2", 1.0)
Nsampling  = rad_cfg.get("n_candidates", 50000)
rad_args   = (k1, k2)

lw         = cfg["loss_weights"]
lam_pde    = lw.get("pde", 1.0)
lam_ic     = lw.get("ic", 5.0)
lam_bc     = lw.get("bc", 5.0)

log_cfg    = cfg.get("logging", {})
Nprint     = log_cfg.get("print_every", 100)
RESULTS_DIR = log_cfg.get("results_dir", "results_cahn_hilliard")
EVAL_EVERY = log_cfg.get("eval_every", 500)
REF_FILE   = log_cfg.get("reference_solution", None)
EVAL_BATCH_SIZE = log_cfg.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE)
L2_THRESHOLD = float(log_cfg.get("l2_threshold", 0.5))
if REF_FILE and not os.path.isabs(REF_FILE):
    REF_FILE = os.path.join(PROJECT_ROOT, REF_FILE)

# --- DRP MODIFICATION: --seed CLI override -----------------------------------
# When --seed is provided, override the config's seed and place results in
# a per-seed subdirectory (results_dir/seed_N/) for multi-seed experiments.
if args.seed is not None:
    SEED = args.seed
    RESULTS_DIR = os.path.join(RESULTS_DIR, f"seed_{args.seed}")
else:
    SEED = cfg.get("seed", 2)
# --- DRP MODIFICATION END ----------------------------------------------------
dtype_str  = cfg.get("dtype", "float64")

torch_dtype = torch.float64 if dtype_str == "float64" else torch.float32
torch.set_default_dtype(torch_dtype)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Config : {cfg['experiment']['name']}")
print(f"  PDE  : dU/dt = {D}*lap(lap U + {a2}*U + {a4}*U^3)")
print(f"  Domain: x=[{x_min},{x_max}], y=[{y_min},{y_max}], t=[{t_min},{t_max}]")
print(f"  Net  : {layer_dims}  activation={activation_name}")
if Nbfgs > 0:
    print(f"  Train: Adam({Nepochs_ADAM}) -> {bfgs_variant if bfgs_method == 'BFGS' else bfgs_method}({Nbfgs})")
else:
    print(f"  Train: Adam-only({Nepochs_ADAM})")
print(f"  Samp : Nint={Nint}, N0={N0}, Nb={Nb}, RAD={'on' if rad_on else 'off'}")
print(f"  Device: {DEVICE}")
print()

# ============================================================================
# 2. NEURAL NETWORK
# ============================================================================

ACTIVATIONS = {
    "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU,
    "silu": nn.SiLU, "sigmoid": nn.Sigmoid,
}


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
    def __init__(self, layer_dims, activation_name="tanh",
                 output_scale=1.0, normalize=False, input_bounds=None):
        super().__init__()
        layers = []
        if normalize and input_bounds is not None:
            lo, hi = input_bounds
            layers.append(InputNormalization(lo, hi))
        act_cls = ACTIVATIONS.get(activation_name, nn.Tanh)
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(act_cls())
        if output_scale != 1.0:
            layers.append(OutputScaling(output_scale))
        self.net = nn.Sequential(*layers)
        for m in reversed(list(self.net.modules())):
            if isinstance(m, nn.Linear):
                sc = 1.0 / (output_scale ** 2) if output_scale != 0 else 1.0
                variance_scaling_init(m, scale=sc)
                break

    def forward(self, x):
        return self.net(x)


input_bounds = ([t_min, x_min, y_min], [t_max, x_max, y_max])
net = CahnHilliardNet(
    layer_dims,
    activation_name=activation_name,
    output_scale=output_scaling,
    normalize=normalize_inputs,
    input_bounds=input_bounds,
).to(DEVICE)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable parameters : {n_params}")

# ============================================================================
# 3. INITIAL CONDITION & SAMPLING
# ============================================================================

def build_ic_interpolator(ic_cfg, x_min, x_max, y_min, y_max):
    rng   = np.random.RandomState(ic_cfg.get("seed", 42))
    nx    = ic_cfg.get("grid_nx", 128)
    ny    = ic_cfg.get("grid_ny", 128)
    sigma = ic_cfg.get("smoothing_sigma", 2.0)
    low   = ic_cfg.get("low", -1.0)
    high  = ic_cfg.get("high", 1.0)
    field = rng.uniform(low, high, size=(ny, nx))
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma, mode="wrap")
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    xs = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)
    ys = np.linspace(y_min + 0.5 * dy, y_max - 0.5 * dy, ny)
    xs_ext = np.concatenate([[x_min], xs, [x_max]])
    ys_ext = np.concatenate([[y_min], ys, [y_max]])
    field_ext = np.pad(field, ((1, 1), (1, 1)), mode="wrap")
    interp = RegularGridInterpolator(
        (ys_ext, xs_ext), field_ext,
        method="cubic", bounds_error=False, fill_value=None,
    )
    return interp, field


ic_interp, ic_field = build_ic_interpolator(ic_cfg, x_min, x_max, y_min, y_max)


def eval_ic(X_t0):
    x_np = X_t0[:, 1].detach().cpu().numpy()
    y_np = X_t0[:, 2].detach().cpu().numpy()
    vals = ic_interp(np.column_stack([y_np, x_np]))
    return torch.tensor(vals, dtype=torch.get_default_dtype(),
                        device=DEVICE).reshape(-1, 1)


def sample_interior(n):
    t = t_min + (t_max - t_min) * np.random.rand(n)
    x = x_min + Lx * np.random.rand(n)
    y = y_min + Ly * np.random.rand(n)
    return torch.tensor(np.column_stack([t, x, y]),
                        dtype=torch.get_default_dtype(), device=DEVICE)


def sample_initial(n):
    t = np.full(n, t_min)
    x = x_min + Lx * np.random.rand(n)
    y = y_min + Ly * np.random.rand(n)
    return torch.tensor(np.column_stack([t, x, y]),
                        dtype=torch.get_default_dtype(), device=DEVICE)


def sample_boundary_periodic(n):
    n2 = max(n // 2, 1)
    t_x = t_min + (t_max - t_min) * np.random.rand(n2)
    y_x = y_min + Ly * np.random.rand(n2)
    X_xlo = torch.tensor(np.column_stack([t_x, np.full(n2, x_min), y_x]),
                          dtype=torch.get_default_dtype(), device=DEVICE)
    X_xhi = torch.tensor(np.column_stack([t_x, np.full(n2, x_max), y_x]),
                          dtype=torch.get_default_dtype(), device=DEVICE)
    t_y = t_min + (t_max - t_min) * np.random.rand(n2)
    x_y = x_min + Lx * np.random.rand(n2)
    X_ylo = torch.tensor(np.column_stack([t_y, x_y, np.full(n2, y_min)]),
                          dtype=torch.get_default_dtype(), device=DEVICE)
    X_yhi = torch.tensor(np.column_stack([t_y, x_y, np.full(n2, y_max)]),
                          dtype=torch.get_default_dtype(), device=DEVICE)
    return X_xlo, X_xhi, X_ylo, X_yhi


# ============================================================================
# 4. PDE RESIDUAL
# ============================================================================

def forward_pass(net, X):
    return net(X)[:, 0:1]


def compute_pde_residual(net, X):
    X = X.clone().detach().requires_grad_(True)
    U = forward_pass(net, X)
    ones = torch.ones_like(U)
    ones1 = torch.ones_like(U[:, 0])

    dU  = torch.autograd.grad(U, X, grad_outputs=ones,
                              create_graph=True, retain_graph=True)[0]
    U_t, U_x, U_y = dU[:, 0], dU[:, 1], dU[:, 2]

    dUx = torch.autograd.grad(U_x, X, grad_outputs=ones1,
                              create_graph=True, retain_graph=True)[0]
    U_xx = dUx[:, 1]

    dUy = torch.autograd.grad(U_y, X, grad_outputs=ones1,
                              create_graph=True, retain_graph=True)[0]
    U_yy = dUy[:, 2]

    dUxx = torch.autograd.grad(U_xx, X, grad_outputs=ones1,
                               create_graph=True, retain_graph=True)[0]
    U_xxx, U_xxy = dUxx[:, 1], dUxx[:, 2]

    dUyy = torch.autograd.grad(U_yy, X, grad_outputs=ones1,
                               create_graph=True, retain_graph=True)[0]
    U_yyy = dUyy[:, 2]

    dUxxx = torch.autograd.grad(U_xxx, X, grad_outputs=ones1,
                                create_graph=True, retain_graph=True)[0]
    U_xxxx = dUxxx[:, 1]

    dUxxy = torch.autograd.grad(U_xxy, X, grad_outputs=ones1,
                                create_graph=True, retain_graph=True)[0]
    U_xxyy = dUxxy[:, 2]

    dUyyy = torch.autograd.grad(U_yyy, X, grad_outputs=ones1,
                                create_graph=True, retain_graph=True)[0]
    U_yyyy = dUyyy[:, 2]

    biharmonic = U_xxxx + 2.0 * U_xxyy + U_yyyy
    laplacian  = U_xx + U_yy
    Uf = U[:, 0]
    lap_U3 = 6.0 * Uf * (U_x ** 2 + U_y ** 2) + 3.0 * Uf ** 2 * laplacian
    rhs = D * (biharmonic + a2 * laplacian + a4 * lap_U3)
    residual = U_t - rhs
    return U, residual


def compute_bc_derivatives(net, X):
    X = X.clone().detach().requires_grad_(True)
    U = forward_pass(net, X)
    ones = torch.ones_like(U)
    dU = torch.autograd.grad(U, X, grad_outputs=ones,
                             create_graph=True, retain_graph=True)[0]
    return U, dU[:, 1], dU[:, 2]


# ============================================================================
# 5. LOSS & TRAINING UTILITIES
# ============================================================================

mse = nn.MSELoss()


def compute_loss_components(net, X_int, X_ic, bc_data):
    X_xlo, X_xhi, X_ylo, X_yhi = bc_data
    _, residual = compute_pde_residual(net, X_int)
    L_pde = mse(residual, torch.zeros_like(residual))

    U_ic_pred = forward_pass(net, X_ic)
    U_ic_true = eval_ic(X_ic)
    L_ic = mse(U_ic_pred, U_ic_true)

    if bc_deriv_order >= 1:
        U_lo_x, Ux_lo, _ = compute_bc_derivatives(net, X_xlo)
        U_hi_x, Ux_hi, _ = compute_bc_derivatives(net, X_xhi)
        U_lo_y, _, Uy_lo  = compute_bc_derivatives(net, X_ylo)
        U_hi_y, _, Uy_hi  = compute_bc_derivatives(net, X_yhi)
        L_bc = (mse(U_lo_x, U_hi_x) + mse(U_lo_y, U_hi_y)
                + mse(Ux_lo, Ux_hi) + mse(Uy_lo, Uy_hi))
    else:
        U_lo_x = forward_pass(net, X_xlo)
        U_hi_x = forward_pass(net, X_xhi)
        U_lo_y = forward_pass(net, X_ylo)
        U_hi_y = forward_pass(net, X_yhi)
        L_bc = mse(U_lo_x, U_hi_x) + mse(U_lo_y, U_hi_y)

    total = lam_pde * L_pde + lam_ic * L_ic + lam_bc * L_bc
    return {"total": total, "pde": L_pde, "ic": L_ic, "bc": L_bc}


def detach_loss_components(loss_components):
    return {
        name: float(value.detach().cpu())
        for name, value in loss_components.items()
    }


def compute_loss(net, X_int, X_ic, bc_data):
    loss_components = compute_loss_components(net, X_int, X_ic, bc_data)
    return loss_components["total"], float(loss_components["pde"].detach())


def evaluate_loss_components(net, X_int, X_ic, bc_data):
    return detach_loss_components(compute_loss_components(net, X_int, X_ic, bc_data))


def adam_step(net, optimizer, X_int, X_ic, bc_data):
    for p in net.parameters():
        p.grad = None
    loss_components = compute_loss_components(net, X_int, X_ic, bc_data)
    loss_components["total"].backward()
    optimizer.step()
    return detach_loss_components(loss_components)


def set_model_weights(net, weights):
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    w = torch.as_tensor(weights, dtype=dtype, device=device)
    with torch.no_grad():
        vector_to_parameters(w, net.parameters())


def scipy_loss_and_grad(weights, net, X_int, X_ic, bc_data, power=1.0):
    set_model_weights(net, weights)
    loss_components = compute_loss_components(net, X_int, X_ic, bc_data)
    loss_val = loss_components["total"]
    loss_eff = loss_val if power == 1.0 else loss_val ** (1.0 / power)
    params = list(net.parameters())
    grads = torch.autograd.grad(loss_eff, params,
                                create_graph=False, retain_graph=False,
                                allow_unused=False)
    g_flat = torch.cat([g.reshape(-1) for g in grads]).detach().cpu().numpy()
    return float(loss_eff.detach().cpu()), g_flat


def adaptive_rad(net, n_int, rad_args, n_cand=50000):
    X_cand = sample_interior(n_cand)
    k1, k2 = rad_args
    Y = torch.abs(compute_pde_residual(net, X_cand)[1]).detach().reshape(-1)
    w = Y ** k1
    p = (w / w.mean() + k2)
    p = (p / p.sum()).clamp_min(1e-12)
    ids = torch.multinomial(p, num_samples=n_int, replacement=False)
    return X_cand[ids]


reference_eval_data = None
if REF_FILE:
    if os.path.isfile(REF_FILE):
        reference_eval_data = load_reference_solution(
            REF_FILE, n_eval_times=DEFAULT_N_EVAL_TIMES
        )
        print(
            f"Reference eval: {REF_FILE} "
            f"({len(reference_eval_data['t_eval'])} times)"
        )
    else:
        print(
            f"WARNING: reference_solution '{REF_FILE}' not found at startup; "
            "relative L2 checkpoints will be skipped."
        )

l2_rel_iters = []
l2_rel_curve = []
last_l2_eval = None


def record_l2_checkpoint(total_iteration):
    global last_l2_eval
    if reference_eval_data is None:
        return None
    was_training = net.training
    last_l2_eval = evaluate_net_against_reference(
        net,
        ref_data=reference_eval_data,
        device=DEVICE,
        batch_size=EVAL_BATCH_SIZE,
        predict_fn=lambda model, x: forward_pass(model, x),
    )
    if was_training:
        net.train()
    l2_rel_iters.append(int(total_iteration))
    l2_rel_curve.append(float(last_l2_eval["error_overall"]))
    return last_l2_eval


# ============================================================================
# 6. PHASE 1 — ADAM
# ============================================================================

X_int = sample_interior(Nint)
X_ic  = sample_initial(N0)
bc_data = sample_boundary_periodic(Nb)

adam_total_loss = []
adam_pde_loss = []
adam_ic_loss = []
adam_bc_loss = []
adam_t0 = perf_counter()

if Nepochs_ADAM > 0:
    _optim_cls = {"adam": torch.optim.Adam, "radam": torch.optim.RAdam}
    if adam_optimizer not in _optim_cls:
        raise ValueError(f"Unknown adam optimizer '{adam_optimizer}'; use 'adam' or 'radam'")
    optimizer = _optim_cls[adam_optimizer](
        net.parameters(), lr=adam_lr, betas=adam_betas, eps=adam_eps
    )
    print(f"Phase 1 optimizer: {adam_optimizer.upper()}  lr={adam_lr}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_decay_rate ** (step / lr_decay_steps)
    )

    for epoch in range(Nepochs_ADAM):
        if (epoch + 1) % Nresample == 0:
            if rad_on:
                X_int = adaptive_rad(net, Nint, rad_args, Nsampling)
            else:
                X_int = sample_interior(Nint)
            X_ic    = sample_initial(N0)
            bc_data = sample_boundary_periodic(Nb)

        loss_components = adam_step(net, optimizer, X_int, X_ic, bc_data)
        scheduler.step()
        adam_total_loss.append(loss_components["total"])
        adam_pde_loss.append(loss_components["pde"])
        adam_ic_loss.append(loss_components["ic"])
        adam_bc_loss.append(loss_components["bc"])

        if (epoch + 1) % EVAL_EVERY == 0:
            record_l2_checkpoint(epoch + 1)

        if (epoch + 1) % Nprint == 0:
            print(f"Adam  epoch {epoch+1:5d}/{Nepochs_ADAM}  "
                  f"loss={loss_components['total']:.4e}  "
                  f"pde={loss_components['pde']:.4e}  "
                  f"ic={loss_components['ic']:.4e}  "
                  f"bc={loss_components['bc']:.4e}")

adam_time = perf_counter() - adam_t0
if adam_total_loss:
    print(f"\nAdam phase complete: {adam_time:.1f}s  "
          f"final loss={adam_total_loss[-1]:.4e}")
else:
    print("Adam phase skipped (0 epochs).")

# ============================================================================
# 7. PHASE 2 — BFGS / L-BFGS-B
# ============================================================================

bfgs_time = 0.0
bfgs_checkpoint_iters = []
bfgs_total_loss = []
bfgs_pde_loss = []
bfgs_ic_loss = []
bfgs_bc_loss = []

if Nbfgs > 0:
    initial_weights = parameters_to_vector(
        [p.detach() for p in net.parameters()]
    ).cpu().numpy()

    use_dense_hessian = (bfgs_method == "BFGS")
    if use_dense_hessian:
        H0 = np.eye(initial_weights.size, dtype=np.float64)

    state = {"cont": 0}

    initial_scale = bfgs_init_scale
    power = bfgs_power

    def bfgs_callback(*, intermediate_result):
        state["cont"] += 1
        cont = state["cont"]
        should_eval = (cont % EVAL_EVERY == 0)
        should_log = (cont % Nprint == 0) or (cont == Nbfgs)

        if not should_eval and not should_log:
            return

        if hasattr(intermediate_result, "x"):
            set_model_weights(net, intermediate_result.x)

        if should_eval:
            record_l2_checkpoint(Nepochs_ADAM + cont)

        if not should_log:
            return

        loss_components = evaluate_loss_components(net, X_int, X_ic, bc_data)
        bfgs_checkpoint_iters.append(cont)
        bfgs_total_loss.append(loss_components["total"])
        bfgs_pde_loss.append(loss_components["pde"])
        bfgs_ic_loss.append(loss_components["ic"])
        bfgs_bc_loss.append(loss_components["bc"])
        optimizer_label = bfgs_variant if use_dense_hessian else bfgs_method
        print(
            f"{optimizer_label}  iter {cont:5d}/{Nbfgs}  "
            f"loss={loss_components['total']:.4e}  "
            f"pde={loss_components['pde']:.4e}  "
            f"ic={loss_components['ic']:.4e}  "
            f"bc={loss_components['bc']:.4e}"
        )

    def build_bfgs_options():
        opts = {"maxiter": Nchange, "gtol": 0}
        if bfgs_method == "BFGS":
            opts["hess_inv0"]     = H0
            opts["method_bfgs"]   = bfgs_variant
            opts["initial_scale"] = initial_scale
        elif bfgs_method == "L-BFGS-B":
            opts["maxcor"] = bfgs_maxcor
            opts["ftol"]   = 0
        return opts

    bfgs_t0 = perf_counter()

    while state["cont"] < Nbfgs:
        result = minimize(
            scipy_loss_and_grad,
            initial_weights,
            args=(net, X_int, X_ic, bc_data, power),
            method=bfgs_method,
            jac=True,
            options=build_bfgs_options(),
            tol=0,
            callback=bfgs_callback,
        )

        initial_weights = result.x

        if use_dense_hessian:
            H0 = result.hess_inv
            H0 = 0.5 * (H0 + H0.T)
            try:
                cholesky(H0)
            except LinAlgError:
                print("  [!] H lost positive-definiteness — resetting to I")
                H0 = np.eye(len(initial_weights), dtype=np.float64)

        if rad_on:
            X_int = adaptive_rad(net, Nint, rad_args, Nsampling)
        else:
            X_int = sample_interior(Nint)
        X_ic    = sample_initial(N0)
        bc_data = sample_boundary_periodic(Nb)
        initial_scale = False

    set_model_weights(net, initial_weights)
    bfgs_time = perf_counter() - bfgs_t0
    optimizer_label = bfgs_variant if use_dense_hessian else bfgs_method
    print(f"\n{optimizer_label} phase complete: {bfgs_time:.1f}s")
    print(f"Total training time: {adam_time + bfgs_time:.1f}s")
else:
    print(f"\nAdam-only mode: no BFGS phase.  Total time: {adam_time:.1f}s")

total_training_iters = Nepochs_ADAM + Nbfgs
if reference_eval_data is not None:
    if not l2_rel_iters or l2_rel_iters[-1] != total_training_iters:
        record_l2_checkpoint(total_training_iters)

adam_losses = np.array(adam_total_loss, dtype=np.float64)
bfgs_losses = np.array(bfgs_total_loss, dtype=np.float64)
bfgs_iters = np.array(bfgs_checkpoint_iters, dtype=np.int64)
adam_pde_loss = np.array(adam_pde_loss, dtype=np.float64)
adam_ic_loss = np.array(adam_ic_loss, dtype=np.float64)
adam_bc_loss = np.array(adam_bc_loss, dtype=np.float64)
bfgs_pde_loss = np.array(bfgs_pde_loss, dtype=np.float64)
bfgs_ic_loss = np.array(bfgs_ic_loss, dtype=np.float64)
bfgs_bc_loss = np.array(bfgs_bc_loss, dtype=np.float64)
l2_rel_iters = np.array(l2_rel_iters, dtype=np.int64)
l2_rel_curve = np.array(l2_rel_curve, dtype=np.float64)

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

optimizer_label = (bfgs_variant if bfgs_method == "BFGS" else bfgs_method) if Nbfgs > 0 else "none"

# Model weights
torch.save(net.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))
print(f"\nModel saved to {RESULTS_DIR}/model.pt")

# Metrics
np.savez_compressed(
    os.path.join(RESULTS_DIR, "metrics.npz"),
    adam_losses=adam_losses,
    bfgs_losses=bfgs_losses,
    adam_total_loss=adam_losses,
    adam_pde_loss=adam_pde_loss,
    adam_ic_loss=adam_ic_loss,
    adam_bc_loss=adam_bc_loss,
    bfgs_total_loss=bfgs_losses,
    bfgs_pde_loss=bfgs_pde_loss,
    bfgs_ic_loss=bfgs_ic_loss,
    bfgs_bc_loss=bfgs_bc_loss,
    bfgs_iters=bfgs_iters,
    l2_rel_iters=l2_rel_iters,
    l2_rel_curve=l2_rel_curve,
    adam_time=adam_time,
    bfgs_time=bfgs_time,
    total_time=adam_time + bfgs_time,
    n_params=n_params,
)
print(f"Metrics saved to {RESULTS_DIR}/metrics.npz")

# Config snapshot
with open(os.path.join(RESULTS_DIR, "config_used.yaml"), "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

# JSON summary
final_l2_overall = float(last_l2_eval["error_overall"]) if last_l2_eval else None
final_l2_per_t = (
    [float(val) for val in last_l2_eval["errors_per_t"]]
    if last_l2_eval else None
)
final_l2_times = (
    [float(val) for val in last_l2_eval["t_eval"]]
    if last_l2_eval else None
)
iters_to_l2_rel_threshold = (
    first_threshold_crossing(l2_rel_iters, l2_rel_curve, L2_THRESHOLD)
    if l2_rel_curve.size > 0 else None
)
threshold_cross_phase = None
if iters_to_l2_rel_threshold is not None:
    if iters_to_l2_rel_threshold <= Nepochs_ADAM or Nbfgs == 0:
        threshold_cross_phase = adam_optimizer
    else:
        threshold_cross_phase = optimizer_label

summary = {
    "experiment": cfg["experiment"]["name"],
    "optimizer": optimizer_label,
    "adam_epochs": Nepochs_ADAM,
    "bfgs_iters": Nbfgs,
    "adam_time_s": round(adam_time, 2),
    "bfgs_time_s": round(bfgs_time, 2),
    "total_time_s": round(adam_time + bfgs_time, 2),
    "final_adam_loss": float(adam_losses[-1]) if adam_losses.size > 0 else None,
    "final_bfgs_loss": float(bfgs_losses[-1]) if bfgs_losses.size > 0 else None,
    "n_params": n_params,
    "l2_rel_error_overall": final_l2_overall,
    "l2_rel_error_per_t": final_l2_per_t,
    "l2_eval_times": final_l2_times,
    "l2_threshold": L2_THRESHOLD,
    "iters_to_l2_rel_threshold": iters_to_l2_rel_threshold,
    "threshold_cross_phase": threshold_cross_phase,
    "l2_rel_curve": [float(val) for val in l2_rel_curve],
}
if final_l2_overall is not None:
    summary["l2_error_overall"] = round(final_l2_overall, 8)
if final_l2_times is not None and final_l2_per_t is not None:
    summary["l2_error_per_t"] = make_legacy_l2_error_map(
        final_l2_times, final_l2_per_t
    )

# ============================================================================
# 9. PLOTS (saved to disk, no display)
# ============================================================================

def build_phase_curve(adam_values, bfgs_values):
    x_parts = []
    y_parts = []
    if adam_values.size > 0:
        x_parts.append(np.arange(1, len(adam_values) + 1, dtype=np.int64))
        y_parts.append(adam_values)
    if bfgs_values.size > 0:
        x_parts.append(Nepochs_ADAM + bfgs_iters)
        y_parts.append(bfgs_values)
    if not x_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    return np.concatenate(x_parts), np.concatenate(y_parts)


def safe_plot_values(values):
    return np.maximum(values, 1e-30)


# Total loss curve
fig, ax = plt.subplots(figsize=(10, 4))
if adam_losses.size > 0:
    ax.semilogy(
        np.arange(1, len(adam_losses) + 1),
        safe_plot_values(adam_losses),
        label=adam_optimizer.upper(),
        alpha=0.85,
    )
if bfgs_losses.size > 0:
    ax.semilogy(
        Nepochs_ADAM + bfgs_iters,
        safe_plot_values(bfgs_losses),
        label=optimizer_label,
        alpha=0.85,
    )
if Nbfgs > 0 and Nepochs_ADAM > 0:
    ax.axvline(Nepochs_ADAM, color="gray", ls="--", lw=0.8, label="Adam→BFGS")
ax.set_xlabel("Iteration")
ax.set_ylabel("Total Loss")
ax.set_title(f"Cahn-Hilliard PINN — {cfg['experiment']['name']}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "loss_total.png"), dpi=150)
fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=150)
plt.close(fig)

# Component loss curves
pde_x, pde_y = build_phase_curve(adam_pde_loss, bfgs_pde_loss)
ic_x, ic_y = build_phase_curve(adam_ic_loss, bfgs_ic_loss)
bc_x, bc_y = build_phase_curve(adam_bc_loss, bfgs_bc_loss)

fig, ax = plt.subplots(figsize=(10, 4))
if pde_y.size > 0:
    ax.semilogy(pde_x, safe_plot_values(pde_y), label="Residual / PDE", alpha=0.85)
if bc_y.size > 0:
    ax.semilogy(bc_x, safe_plot_values(bc_y), label="Boundary", alpha=0.85)
if ic_y.size > 0:
    ax.semilogy(ic_x, safe_plot_values(ic_y), label="Initial Condition", alpha=0.85)
if Nbfgs > 0 and Nepochs_ADAM > 0:
    ax.axvline(Nepochs_ADAM, color="gray", ls="--", lw=0.8, label="Adam→BFGS")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title(f"Loss Components — {cfg['experiment']['name']}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "loss_components.png"), dpi=150)
plt.close(fig)

component_values = []
for arr in (pde_y, ic_y, bc_y):
    if arr.size > 0:
        component_values.extend(arr[arr > 0].tolist())
if component_values:
    component_spread = max(component_values) / max(min(component_values), 1e-30)
    if component_spread > 100.0:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        component_specs = [
            ("Residual / PDE", pde_x, pde_y),
            ("Boundary", bc_x, bc_y),
            ("Initial Condition", ic_x, ic_y),
        ]
        for ax, (label, x_vals, y_vals) in zip(axes, component_specs):
            if y_vals.size > 0:
                ax.semilogy(x_vals, safe_plot_values(y_vals), label=label, alpha=0.85)
            if Nbfgs > 0 and Nepochs_ADAM > 0:
                ax.axvline(Nepochs_ADAM, color="gray", ls="--", lw=0.8)
            ax.set_ylabel("Loss")
            ax.legend(loc="best")
        axes[-1].set_xlabel("Iteration")
        fig.suptitle(f"Split Loss Components — {cfg['experiment']['name']}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "loss_components_split.png"), dpi=150)
        plt.close(fig)

if l2_rel_curve.size > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(l2_rel_iters, safe_plot_values(l2_rel_curve), color="black", alpha=0.85)
    ax.axhline(L2_THRESHOLD, color="tab:red", ls="--", lw=1.0,
               label=f"L2 threshold = {L2_THRESHOLD:.2f}")
    if iters_to_l2_rel_threshold is not None:
        ax.axvline(iters_to_l2_rel_threshold, color="tab:green", ls=":",
                   lw=1.0, label=f"first hit @ {iters_to_l2_rel_threshold}")
    ax.set_xlabel("Total Iteration")
    ax.set_ylabel("Relative L2")
    ax.set_title(f"Relative L2 vs Iteration — {cfg['experiment']['name']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "l2_curve.png"), dpi=150)
    plt.close(fig)

# Solution snapshots
n_snap = 5
snap_times = np.linspace(t_min, t_max, n_snap)
nx_plot, ny_plot = 64, 64
xs_plot = np.linspace(x_min, x_max, nx_plot)
ys_plot = np.linspace(y_min, y_max, ny_plot)
xx, yy = np.meshgrid(xs_plot, ys_plot)

fig, axes = plt.subplots(1, n_snap, figsize=(4 * n_snap, 3.5))
net.eval()
for i, t_val in enumerate(snap_times):
    tt = np.full_like(xx, t_val)
    X_plot = torch.tensor(
        np.column_stack([tt.ravel(), xx.ravel(), yy.ravel()]),
        dtype=torch.get_default_dtype(), device=DEVICE,
    )
    with torch.no_grad():
        U_plot = forward_pass(net, X_plot).cpu().numpy().reshape(ny_plot, nx_plot)
    im = axes[i].pcolormesh(xs_plot, ys_plot, U_plot, cmap="RdBu_r", shading="auto")
    axes[i].set_title(f"t = {t_val:.1f}")
    axes[i].set_aspect("equal")
    fig.colorbar(im, ax=axes[i], fraction=0.046)
fig.suptitle("Predicted U(t, x, y)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "snapshots.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# IC comparison
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
X_ic_plot = torch.tensor(
    np.column_stack([np.full(nx_plot * ny_plot, t_min), xx.ravel(), yy.ravel()]),
    dtype=torch.get_default_dtype(), device=DEVICE,
)
U_ic_true_plot = eval_ic(X_ic_plot).cpu().numpy().reshape(ny_plot, nx_plot)
im0 = axes[0].pcolormesh(xs_plot, ys_plot, U_ic_true_plot, cmap="RdBu_r", shading="auto")
axes[0].set_title("True IC  U₀(x,y)")
axes[0].set_aspect("equal")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

net.eval()
with torch.no_grad():
    U_ic_pred_plot = forward_pass(net, X_ic_plot).cpu().numpy().reshape(ny_plot, nx_plot)
im1 = axes[1].pcolormesh(xs_plot, ys_plot, U_ic_pred_plot, cmap="RdBu_r", shading="auto")
axes[1].set_title("PINN at t = 0")
axes[1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1], fraction=0.046)
fig.suptitle("Initial Condition Comparison", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "ic_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Mass conservation diagnostic
# CH with periodic BCs conserves spatial average: d/dt ∫_Ω U dx dy = 0
n_mass_times = 21
mass_times = np.linspace(t_min, t_max, n_mass_times)
mean_U = np.zeros(n_mass_times)

net.eval()
with torch.no_grad():
    for i, t_val in enumerate(mass_times):
        tt = np.full(nx_plot * ny_plot, t_val)
        X_mass = torch.tensor(
            np.column_stack([tt, xx.ravel(), yy.ravel()]),
            dtype=torch.get_default_dtype(), device=DEVICE,
        )
        U_mass = forward_pass(net, X_mass).cpu().numpy().ravel()
        mean_U[i] = U_mass.mean()
net.train()

mass_drift = mean_U.max() - mean_U.min()
print(f"\nMass conservation diagnostic (<U> over domain at {n_mass_times} times):")
print(f"  <U>(t={t_min}) = {mean_U[0]:+.6f}")
print(f"  <U>(t={t_max}) = {mean_U[-1]:+.6f}")
print(f"  max drift       = {mass_drift:.2e}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(mass_times, mean_U, "o-", markersize=4, linewidth=1.5)
ax.axhline(mean_U[0], color="gray", ls="--", lw=0.8, label=f"<U>(t0) = {mean_U[0]:.4f}")
ax.set_xlabel("t")
ax.set_ylabel("<U>(t)")
ax.set_title(f"Mass Conservation — drift = {mass_drift:.2e}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "mass_conservation.png"), dpi=150)
plt.close(fig)

# Add mass data to summary
summary["mass_mean_t0"] = round(float(mean_U[0]), 6)
summary["mass_mean_tmax"] = round(float(mean_U[-1]), 6)
summary["mass_drift"] = round(float(mass_drift), 8)

# ============================================================================
# 10. COMPARISON SNAPSHOTS  (spectral reference vs PINN)
# ============================================================================
if REF_FILE and os.path.isfile(REF_FILE):
    print(f"\nLoading reference solution from {REF_FILE} ...")
    ref_data = np.load(REF_FILE)
    t_ref = ref_data["t"]        # (n_save,)
    x_ref = ref_data["x"]        # (nx_ref,)
    y_ref = ref_data["y"]        # (ny_ref,)
    U_ref = ref_data["U"]        # (n_save, ny_ref, nx_ref)

    # Use the same snap_times and plot grid as the PINN snapshot section
    # snap_times, xs_plot, ys_plot, xx, yy, nx_plot, ny_plot already defined

    # For each snapshot time, find the nearest reference time index
    ref_snap_idx = [int(np.argmin(np.abs(t_ref - st))) for st in snap_times]

    # Interpolate reference to the PINN evaluation grid (xs_plot, ys_plot)
    from scipy.interpolate import RegularGridInterpolator
    ref_on_grid = np.zeros((n_snap, ny_plot, nx_plot))
    for i, ri in enumerate(ref_snap_idx):
        interp = RegularGridInterpolator(
            (y_ref, x_ref), U_ref[ri], method="linear",
            bounds_error=False, fill_value=None,
        )
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        ref_on_grid[i] = interp(pts).reshape(ny_plot, nx_plot)

    # PINN predictions (reuse from snapshot section or recompute)
    pinn_snaps = np.zeros((n_snap, ny_plot, nx_plot))
    net.eval()
    with torch.no_grad():
        for i, t_val in enumerate(snap_times):
            tt = np.full_like(xx, t_val)
            X_cmp = torch.tensor(
                np.column_stack([tt.ravel(), xx.ravel(), yy.ravel()]),
                dtype=torch.get_default_dtype(), device=DEVICE,
            )
            pinn_snaps[i] = forward_pass(net, X_cmp).cpu().numpy().reshape(ny_plot, nx_plot)
    net.train()

    # Absolute error
    abs_err = np.abs(ref_on_grid - pinn_snaps)

    # Compute per-snapshot relative L2 error
    l2_per_t = []
    for i in range(n_snap):
        denom = np.linalg.norm(ref_on_grid[i])
        if denom > 1e-15:
            l2_per_t.append(float(np.linalg.norm(abs_err[i]) / denom))
        else:
            l2_per_t.append(float("nan"))
    l2_overall = float(
        np.linalg.norm(ref_on_grid - pinn_snaps)
        / max(np.linalg.norm(ref_on_grid), 1e-15)
    )

    print(f"  Per-snapshot relative L2 errors:")
    for i, (tv, e) in enumerate(zip(snap_times, l2_per_t)):
        print(f"    t={tv:.1f}  L2_rel={e:.4e}")
    print(f"  Overall relative L2 = {l2_overall:.4e}")

    # 3-row figure: reference / PINN / absolute error
    vmin = min(ref_on_grid.min(), pinn_snaps.min())
    vmax = max(ref_on_grid.max(), pinn_snaps.max())

    fig, axes = plt.subplots(3, n_snap, figsize=(4 * n_snap, 10))
    row_labels = ["Spectral Reference", "PINN Prediction", "Absolute Error"]
    for i in range(n_snap):
        # Row 0: reference
        im0 = axes[0, i].pcolormesh(xs_plot, ys_plot, ref_on_grid[i],
                                     cmap="RdBu_r", shading="auto",
                                     vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"t = {snap_times[i]:.1f}")
        axes[0, i].set_aspect("equal")

        # Row 1: PINN
        im1 = axes[1, i].pcolormesh(xs_plot, ys_plot, pinn_snaps[i],
                                     cmap="RdBu_r", shading="auto",
                                     vmin=vmin, vmax=vmax)
        axes[1, i].set_aspect("equal")

        # Row 2: error
        im2 = axes[2, i].pcolormesh(xs_plot, ys_plot, abs_err[i],
                                     cmap="hot", shading="auto")
        axes[2, i].set_aspect("equal")
        axes[2, i].set_xlabel(f"L2_rel={l2_per_t[i]:.2e}")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)

    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
    fig.colorbar(im2, ax=axes[2, :].tolist(), fraction=0.02, pad=0.02)

    fig.suptitle(
        f"Comparison — {cfg['experiment']['name']}  "
        f"(overall L2_rel = {l2_overall:.2e})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "comparison_snapshots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison_snapshots.png")
elif REF_FILE:
    print(f"\nWARNING: reference_solution '{REF_FILE}' not found — skipping comparison.")

with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {RESULTS_DIR}/summary.json")

print(f"\nPlots saved to {RESULTS_DIR}/")

# Final summary
print(f"\n{'=' * 70}")
print(f"EXPERIMENT COMPLETE: {cfg['experiment']['name']}")
print(f"  Adam: {Nepochs_ADAM} epochs, {adam_time:.1f}s")
if Nbfgs > 0:
    print(f"  BFGS: {Nbfgs} iters ({optimizer_label}), {bfgs_time:.1f}s")
print(f"  Total: {adam_time + bfgs_time:.1f}s")
if adam_losses.size > 0 and Nbfgs == 0:
    print(f"  Final loss: {adam_losses[-1]:.4e}")
elif bfgs_losses.size > 0:
    print(f"  Final loss: {bfgs_losses[-1]:.4e}")
if final_l2_overall is not None:
    print(f"  Final relative L2: {final_l2_overall:.4e}")
if iters_to_l2_rel_threshold is not None:
    print(f"  Iterations to L2<={L2_THRESHOLD:.2f}: {iters_to_l2_rel_threshold}")
print(f"  Results: {RESULTS_DIR}/")
print(f"{'=' * 70}")
