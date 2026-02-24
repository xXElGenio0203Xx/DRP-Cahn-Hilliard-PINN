#!/usr/bin/env python3
"""
Cahn-Hilliard PINN — Headless Training Script
==============================================
Converted from Cahn_Hilliard_Pytorch.ipynb for batch execution on OSCAR.

Usage:
    python -u run_experiment.py configs/cahn_hilliard_canonical.yaml
    python -u run_experiment.py configs/adam_only.yaml

The single CLI argument is the path to a YAML configuration file.
All results (model weights, loss curves, plots, metrics) are saved to
the directory specified by logging.results_dir in the config.

Use `python -u` (unbuffered) so SLURM captures output in real time.
"""

import os
import sys
import math
import json
import yaml
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

# ============================================================================
# 0. PARSE CLI ARGUMENT
# ============================================================================

if len(sys.argv) < 2:
    print("Usage: python -u run_experiment.py <config.yaml>")
    sys.exit(1)

CONFIG_PATH = sys.argv[1]
if not os.path.isfile(CONFIG_PATH):
    print(f"ERROR: Config file not found: {CONFIG_PATH}")
    sys.exit(1)

print(f"=" * 70)
print(f"Cahn-Hilliard PINN — Batch Training")
print(f"Config: {CONFIG_PATH}")
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

SEED       = cfg.get("seed", 2)
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


def compute_loss(net, X_int, X_ic, bc_data):
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
    return total, float(L_pde.detach())


def adam_step(net, optimizer, X_int, X_ic, bc_data):
    for p in net.parameters():
        p.grad = None
    loss_val, pde_val = compute_loss(net, X_int, X_ic, bc_data)
    loss_val.backward()
    optimizer.step()
    return float(loss_val.detach()), pde_val


def scipy_loss_and_grad(weights, net, X_int, X_ic, bc_data, power=1.0):
    device = next(net.parameters()).device
    dtype  = next(net.parameters()).dtype
    w = torch.as_tensor(weights, dtype=dtype, device=device)
    with torch.no_grad():
        vector_to_parameters(w, net.parameters())
    loss_val, _ = compute_loss(net, X_int, X_ic, bc_data)
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


# ============================================================================
# 6. PHASE 1 — ADAM
# ============================================================================

X_int = sample_interior(Nint)
X_ic  = sample_initial(N0)
bc_data = sample_boundary_periodic(Nb)

adam_losses = []
adam_t0 = perf_counter()

if Nepochs_ADAM > 0:
    optimizer = torch.optim.Adam(
        net.parameters(), lr=adam_lr, betas=adam_betas, eps=adam_eps
    )
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

        loss_v, pde_v = adam_step(net, optimizer, X_int, X_ic, bc_data)
        scheduler.step()
        adam_losses.append(loss_v)

        if (epoch + 1) % Nprint == 0:
            print(f"Adam  epoch {epoch+1:5d}/{Nepochs_ADAM}  "
                  f"loss={loss_v:.4e}  pde={pde_v:.4e}")

adam_time = perf_counter() - adam_t0
if adam_losses:
    print(f"\nAdam phase complete: {adam_time:.1f}s  "
          f"final loss={adam_losses[-1]:.4e}")
else:
    print("Adam phase skipped (0 epochs).")

# ============================================================================
# 7. PHASE 2 — BFGS / L-BFGS-B
# ============================================================================

bfgs_time = 0.0
n_ckpts = 0
bfgs_losses = np.array([])
bfgs_pde = np.array([])

if Nbfgs > 0:
    initial_weights = parameters_to_vector(
        [p.detach() for p in net.parameters()]
    ).cpu().numpy()

    use_dense_hessian = (bfgs_method == "BFGS")
    if use_dense_hessian:
        H0 = np.eye(initial_weights.size, dtype=np.float64)

    cont = 0
    n_ckpts = Nbfgs // Nprint
    bfgs_losses = np.zeros(n_ckpts)
    bfgs_pde    = np.zeros(n_ckpts)

    initial_scale = bfgs_init_scale
    power = bfgs_power

    def bfgs_callback(*, intermediate_result):
        global cont
        cont += 1
        if cont % Nprint != 0:
            return
        idx = cont // Nprint - 1
        if idx < 0 or idx >= n_ckpts:
            return
        loss_val = float(intermediate_result.fun)
        if power != 1.0:
            loss_val = loss_val ** power
        bfgs_losses[idx] = loss_val
        optimizer_label = bfgs_variant if use_dense_hessian else bfgs_method
        print(f"{optimizer_label}  iter {cont:5d}/{Nbfgs}  loss={loss_val:.4e}")

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

    while cont < Nbfgs:
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

    bfgs_time = perf_counter() - bfgs_t0
    optimizer_label = bfgs_variant if use_dense_hessian else bfgs_method
    print(f"\n{optimizer_label} phase complete: {bfgs_time:.1f}s")
    print(f"Total training time: {adam_time + bfgs_time:.1f}s")
else:
    print(f"\nAdam-only mode: no BFGS phase.  Total time: {adam_time:.1f}s")

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
    adam_losses=np.array(adam_losses),
    bfgs_losses=bfgs_losses if Nbfgs > 0 else np.array([]),
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
summary = {
    "experiment": cfg["experiment"]["name"],
    "optimizer": optimizer_label,
    "adam_epochs": Nepochs_ADAM,
    "bfgs_iters": Nbfgs,
    "adam_time_s": round(adam_time, 2),
    "bfgs_time_s": round(bfgs_time, 2),
    "total_time_s": round(adam_time + bfgs_time, 2),
    "final_adam_loss": adam_losses[-1] if adam_losses else None,
    "final_bfgs_loss": float(bfgs_losses[bfgs_losses > 0][-1]) if Nbfgs > 0 and (bfgs_losses > 0).any() else None,
    "n_params": n_params,
}
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {RESULTS_DIR}/summary.json")

# ============================================================================
# 9. PLOTS (saved to disk, no display)
# ============================================================================

# Loss curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(range(1, len(adam_losses) + 1), adam_losses, label="Adam", alpha=0.8)
if Nbfgs > 0:
    bfgs_epochs = Nepochs_ADAM + np.arange(1, n_ckpts + 1) * Nprint
    valid = bfgs_losses > 0
    if valid.any():
        ax.semilogy(bfgs_epochs[valid], bfgs_losses[valid],
                     label=optimizer_label, alpha=0.8)
    ax.axvline(Nepochs_ADAM, color="gray", ls="--", lw=0.8, label="Adam→BFGS")
ax.set_xlabel("Iteration")
ax.set_ylabel("Total Loss")
ax.set_title(f"Cahn-Hilliard PINN — {cfg['experiment']['name']}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=150)
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
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nPlots saved to {RESULTS_DIR}/")

# Final summary
print(f"\n{'=' * 70}")
print(f"EXPERIMENT COMPLETE: {cfg['experiment']['name']}")
print(f"  Adam: {Nepochs_ADAM} epochs, {adam_time:.1f}s")
if Nbfgs > 0:
    print(f"  BFGS: {Nbfgs} iters ({optimizer_label}), {bfgs_time:.1f}s")
print(f"  Total: {adam_time + bfgs_time:.1f}s")
if adam_losses and Nbfgs == 0:
    print(f"  Final loss: {adam_losses[-1]:.4e}")
elif Nbfgs > 0 and (bfgs_losses > 0).any():
    print(f"  Final loss: {bfgs_losses[bfgs_losses > 0][-1]:.4e}")
print(f"  Results: {RESULTS_DIR}/")
print(f"{'=' * 70}")
