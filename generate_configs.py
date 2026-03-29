#!/usr/bin/env python3
"""
generate_configs.py — Generate all v2 experiment configs
=========================================================

Creates split config subdirectories under configs/ with 27 paper-aligned
YAML files for the Cahn-Hilliard PINN experiment suite. Each config is
fully self-contained (no inheritance) and includes a detailed description
of what it tests, how it differs from the canonical, and what paper
section it parallels.

Run once:
    python generate_configs.py

The suite is structured in 8 groups (A–H), each testing one ablation axis:
    A. Optimizer variant comparison     (6 configs) — core paper result
    B. Adam warmup duration             (3 configs) — preconditioning study
    C. Network architecture             (2 configs) — capacity study
    D. Adam-only baseline               (1 config)  — first-order floor
    E. RAdam warmup (DRP extension)     (2 configs) — not in original paper
    F. Resampling frequency & points    (6 configs) — collocation study
    G. Hessian scaling & power transform(3 configs) — conditioning study
    H. RAD parameter sweep              (4 configs) — adaptive sampling study

Total: 27 configs × 3 seeds = 81 SLURM jobs.
"""

import os
import yaml
from copy import deepcopy

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
CORE_OUT_DIR = os.path.join(CONFIG_ROOT, "v2_core")
FH_OUT_DIR = os.path.join(CONFIG_ROOT, "v2_fh_ablation")
os.makedirs(CORE_OUT_DIR, exist_ok=True)
os.makedirs(FH_OUT_DIR, exist_ok=True)

FH_ABLATION_NAMES = {
    "F4_points_5k",
    "F5_adam_resample_200",
    "F6_adam_resample_1000",
    "H1_rad_k1_0p5",
    "H2_rad_k1_2p0",
    "H3_rad_k2_0p5",
    "H4_rad_k2_2p0",
}


# ============================================================================
# CANONICAL TEMPLATE — every config starts from this and overrides
# ============================================================================
#
# Design choices (matching the paper's methodology):
#   - 1,000 Adam warmup  (paper Burgers Cases 1-2: 1,000 Adam)
#   - 50,000 BFGS iters  (paper Burgers Case 1: 50,000 BFGS)
#   - Nchange = 500       (paper standard)
#   - 10K interior / 1K IC / 500 BC  (paper Burgers: 8K/500/500)
#   - RAD enabled           (paper standard)
#   - λ_pde=1, λ_ic=5, λ_bc=5  (soft periodic BCs need higher weight)
#   - float64               (paper: essential for quasi-Newton)
#   - print_every=500       (50K iters → 100 checkpoints)

CANONICAL = {
    "experiment": {
        "name": "A5_ssbroyden2",
        "description": (
            "CANONICAL configuration. SSBroyden2 (self-scaled Broyden, BFGS "
            "direction) with 1,000 Adam warmup followed by 50,000 BFGS "
            "iterations. This is the reference config against which all "
            "ablations are compared. Matches the paper's Burgers Case 1 "
            "budget (1K Adam + 50K BFGS). "
            "Paper ref: Kiyani et al., CMAME 446:118308, 2025, Table 2."
        ),
    },
    "equation": {"D": 1.0, "a2": 1.0, "a4": 1.0},
    "domain": {
        "x_min": 0.0, "x_max": 128.0,
        "y_min": 0.0, "y_max": 128.0,
        "t_min": 0.0, "t_max": 10.0,
        "normalize_inputs": True,
    },
    "initial_condition": {
        "type": "uniform_random",
        "low": -1.0, "high": 1.0,
        "grid_nx": 128, "grid_ny": 128,
        "smoothing_sigma": 2.0,
        "seed": 42,
    },
    "boundary_conditions": {
        "type": "periodic",
        "enforcement": "soft",
        "derivative_order": 1,
    },
    "network": {
        "layer_dims": [3, 20, 20, 20, 20, 1],
        "activation": "tanh",
        "output_scaling": 1.0,
        "final_layer_init": "variance_scaling",
    },
    "training": {
        "adam": {
            "epochs": 1000,
            "lr": 0.001,
            "betas": [0.99, 0.999],
            "eps": 1.0e-20,
            "lr_decay_rate": 0.98,
            "lr_decay_steps": 1000,
        },
        "bfgs": {
            "epochs": 50000,
            "batch_size": 500,
            "method": "BFGS",
            "variant": "SSBroyden2",
            "initial_scale": False,
            "power": 1.0,
        },
    },
    "sampling": {
        "n_interior": 10000,
        "n_initial": 1000,
        "n_boundary": 500,
        "resample_every": 500,
        "rad": {
            "enabled": True,
            "k1": 1.0,
            "k2": 1.0,
            "n_candidates": 50000,
        },
    },
    "loss_weights": {"pde": 1.0, "ic": 5.0, "bc": 5.0},
    "logging": {
        "print_every": 500,
        "eval_every": 500,
        "reference_solution": "reference_solution_t10_dt0p01.npz",
        "results_dir": "results/A5_ssbroyden2",
    },
    "seed": 2,
    "dtype": "float64",
}


def make(name, description, overrides):
    """Create a config from canonical + nested key-path overrides."""
    cfg = deepcopy(CANONICAL)
    cfg["experiment"]["name"] = name
    cfg["experiment"]["description"] = description
    cfg["logging"]["results_dir"] = f"results/{name}"

    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    return cfg


# ============================================================================
# GROUP A: Optimizer Variant Comparison (the core paper result)
# ============================================================================
# Paper: Table 2 (Burgers), Table 7 (Allen-Cahn), Figs 3-5, 12-13.
# This is THE central comparison — all 6 quasi-Newton variants at identical
# settings. SSBroyden consistently wins by 2-3 orders of magnitude.

EXPERIMENTS = {}

EXPERIMENTS["A1_bfgs"] = make(
    "A1_bfgs",
    "Standard BFGS (textbook formula, no self-scaling). "
    "Baseline quasi-Newton method. The paper reports BFGS consistently "
    "underperforms SSBroyden by 2-3 orders of magnitude on Burgers "
    "(Table 2: BFGS ~1e-5 vs SSBroyden ~1e-8). "
    "Differs from canonical: variant=BFGS instead of SSBroyden2.",
    {"training.bfgs.variant": "BFGS"},
)

EXPERIMENTS["A2_ssbfgs_ol"] = make(
    "A2_ssbfgs_ol",
    "Self-Scaled BFGS with Oren-Luenberger scaling (tau_k = 1/b_k). "
    "The simplest self-scaling strategy — scales the Hessian by 1/b_k "
    "every iteration. Paper: intermediate performance between BFGS and "
    "SSBroyden on most problems. "
    "Differs from canonical: variant=SSBFGS_OL instead of SSBroyden2.",
    {"training.bfgs.variant": "SSBFGS_OL"},
)

EXPERIMENTS["A3_ssbfgs_ab"] = make(
    "A3_ssbfgs_ab",
    "Self-Scaled BFGS with Al-Baali scaling (tau_k = min(1, 1/b_k)). "
    "More conservative than OL — clips scaling to prevent over-scaling "
    "when b_k < 1. Paper: generally better than OL. "
    "Differs from canonical: variant=SSBFGS_AB instead of SSBroyden2.",
    {"training.bfgs.variant": "SSBFGS_AB"},
)

EXPERIMENTS["A4_ssbroyden1"] = make(
    "A4_ssbroyden1",
    "Self-Scaled Broyden class 1 (DFP direction, phi -> 1). "
    "Interpolates between DFP and BFGS updates with optimal theta "
    "selection. NOTE: uses DRP-patched version with NaN/Inf guards "
    "and BFGS fallback (original code had numerical instability). "
    "Paper: SSBroyden1 and SSBroyden2 perform similarly on most PDEs. "
    "Differs from canonical: variant=SSBroyden1 instead of SSBroyden2.",
    {"training.bfgs.variant": "SSBroyden1"},
)

EXPERIMENTS["A5_ssbroyden2"] = make(
    "A5_ssbroyden2",
    CANONICAL["experiment"]["description"],
    {},  # no overrides — this IS the canonical
)

EXPERIMENTS["A6_lbfgs"] = make(
    "A6_lbfgs",
    "Limited-memory BFGS (L-BFGS-B) with 20 correction pairs. "
    "Uses O(m*N) memory instead of O(N^2), enabling much larger "
    "networks but at the cost of less accurate Hessian approximation. "
    "Paper: L-BFGS consistently worst among quasi-Newton methods "
    "(Burgers Table 2: ~1e-3 vs SSBroyden ~1e-8). "
    "Differs from canonical: method=L-BFGS-B, variant=none, maxcor=20.",
    {
        "training.bfgs.method": "L-BFGS-B",
        "training.bfgs.variant": "none",
    },
)


# ============================================================================
# GROUP B: Adam Warmup Duration
# ============================================================================
# Paper: Burgers Case 3 (0 warmup), Cases 1-2 (1000 warmup), Allen-Cahn
# (5000 warmup). Tests how much first-order preconditioning the quasi-Newton
# phase needs. Our v1 suite found a U-shaped relationship.

EXPERIMENTS["B1_warmup_0"] = make(
    "B1_warmup_0",
    "Zero Adam warmup — SSBroyden2 from random initialization (50K BFGS). "
    "Tests whether quasi-Newton can converge from scratch without "
    "first-order preconditioning. Paper Burgers Case 3: 0 warmup + 30K "
    "BFGS still reaches ~1e-7 with SSBroyden. Our v1 suite found "
    "warmup_0 had the best mass conservation but worst L2 error (1.66). "
    "Differs from canonical: 0 Adam epochs (BFGS budget stays at 50K).",
    {"training.adam.epochs": 0},
)

EXPERIMENTS["B2_warmup_5000"] = make(
    "B2_warmup_5000",
    "Long warmup: 5,000 Adam epochs before 50,000 SSBroyden2. "
    "Matches Allen-Cahn warmup duration from the paper. Tests whether "
    "extended first-order training gives BFGS a significantly better "
    "starting point for the stiff CH biharmonic operator. "
    "Differs from canonical: 5000 Adam epochs instead of 1000.",
    {"training.adam.epochs": 5000},
)

EXPERIMENTS["B3_warmup_10000"] = make(
    "B3_warmup_10000",
    "Very long warmup: 10,000 Adam epochs before 50,000 SSBroyden2. "
    "Extreme warmup test — at 10K Adam iters the network should be "
    "well into the loss basin before quasi-Newton refinement. Tests "
    "diminishing returns of extended warmup. "
    "Differs from canonical: 10000 Adam epochs instead of 1000.",
    {"training.adam.epochs": 10000},
)


# ============================================================================
# GROUP C: Network Architecture
# ============================================================================
# Paper: Burgers Case 4 (8x20 = 3,021 params), GL Cases 1-3 (5x30, 8x30).
# Our v1 suite had O(1) L2 errors with 1,361 params — strongly suggesting
# insufficient capacity. The biharmonic operator and 128x128 domain likely
# need more parameters.

EXPERIMENTS["C1_deep_8x20"] = make(
    "C1_deep_8x20",
    "Deeper network: 8 hidden layers x 20 neurons = 3,041 parameters. "
    "Directly parallels Burgers Case 4 (8x20 = 3,021 params). Tests "
    "whether depth (more compositional power) helps capture CH's complex "
    "phase-separation dynamics. The biharmonic operator may benefit from "
    "deeper function composition. Hessian: ~74 MB (fits in 4GB memory). "
    "Differs from canonical: [3,20,20,20,20,20,20,20,20,1] instead of "
    "[3,20,20,20,20,1].",
    {"network.layer_dims": [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]},
)

EXPERIMENTS["C2_wide_5x30"] = make(
    "C2_wide_5x30",
    "Wider network: 5 hidden layers x 30 neurons = 3,871 parameters. "
    "Parallels GL Case 1 (5x30 = 3,184 params with 2 outputs) and KS "
    "(5x30 = 4,411 params with Fourier encoding). Tests whether width "
    "(more expressiveness per layer) is more effective than depth for CH. "
    "Hessian: ~120 MB (fits in 4GB memory). "
    "Differs from canonical: [3,30,30,30,30,30,1] instead of "
    "[3,20,20,20,20,1].",
    {"network.layer_dims": [3, 30, 30, 30, 30, 30, 1]},
)


# ============================================================================
# GROUP D: Adam-Only Baseline
# ============================================================================
# Paper: compares Adam baselines against quasi-Newton across all PDEs.
# Establishes the floor that quasi-Newton methods must beat.

EXPERIMENTS["D1_adam_only"] = make(
    "D1_adam_only",
    "Adam-only baseline — 51,000 Adam epochs, no quasi-Newton phase. "
    "Same total iteration budget as canonical (1K Adam + 50K BFGS = 51K). "
    "The paper consistently shows Adam alone stalls at much higher loss "
    "than quasi-Newton methods. This establishes the performance floor. "
    "Differs from canonical: 51000 Adam, 0 BFGS epochs.",
    {
        "training.adam.epochs": 51000,
        "training.bfgs.epochs": 0,
        "training.bfgs.variant": "none",
    },
)


# ============================================================================
# GROUP E: RAdam Warmup (DRP extension — not in original paper)
# ============================================================================
# RAdam (Liu et al., ICLR 2020) dynamically rectifies the adaptive learning
# rate variance, acting as built-in warmup. Tests whether this improves the
# quality of the starting point for BFGS.

EXPERIMENTS["E1_radam_1000"] = make(
    "E1_radam_1000",
    "RAdam warmup (1,000 epochs) instead of Adam. "
    "Rectified Adam (Liu et al., ICLR 2020) dynamically turns off "
    "momentum correction when the variance estimate is unreliable, "
    "acting as a built-in warmup mechanism. Eliminates sensitivity to "
    "early learning rate. This is a DRP extension — not in the paper. "
    "Direct comparison with A5_ssbroyden2 (Adam 1K). "
    "Differs from canonical: optimizer='radam' instead of 'adam'.",
    {"training.adam.optimizer": "radam"},
)

EXPERIMENTS["E2_radam_5000"] = make(
    "E2_radam_5000",
    "RAdam warmup (5,000 epochs) — longer RAdam preconditioning. "
    "Combines RAdam's built-in variance rectification with extended "
    "warmup (5K epochs). Direct comparison with B2_warmup_5000 (Adam 5K). "
    "This is a DRP extension — not in the paper. "
    "Differs from canonical: optimizer='radam', 5000 epochs.",
    {
        "training.adam.optimizer": "radam",
        "training.adam.epochs": 5000,
    },
)


# ============================================================================
# GROUP F: Resampling Frequency & Collocation Points
# ============================================================================
# Tests sensitivity to Nchange (how often collocation points are refreshed
# during BFGS) and total collocation count. Our v1 suite found Nchange=200
# was the BEST performer, suggesting more frequent resampling prevents
# overfitting to a fixed point set.

EXPERIMENTS["F1_nchange_200"] = make(
    "F1_nchange_200",
    "Frequent BFGS resampling: Nchange=200. Collocation points are "
    "refreshed every 200 BFGS iterations (250 minimize() calls over 50K). "
    "Our v1 suite found Nchange=200 was the BEST performer "
    "(L2=0.854, lowest overall). More frequent resampling prevents the "
    "quasi-Newton optimizer from overfitting to a fixed point set. "
    "Differs from canonical: batch_size=200 instead of 500.",
    {"training.bfgs.batch_size": 200},
)

EXPERIMENTS["F2_nchange_1000"] = make(
    "F2_nchange_1000",
    "Infrequent BFGS resampling: Nchange=1000. Collocation points "
    "refreshed every 1000 BFGS iters (50 minimize() calls over 50K). "
    "Tests whether longer optimization on a fixed point set helps "
    "convergence or causes overfitting. v1 result: among the worst. "
    "Differs from canonical: batch_size=1000 instead of 500.",
    {"training.bfgs.batch_size": 1000},
)

EXPERIMENTS["F3_points_20k"] = make(
    "F3_points_20k",
    "Double collocation points: 20,000 interior (vs 10,000 canonical), "
    "2,000 IC points (vs 1,000). The CH domain is 128x128x20 = 327,680 "
    "spatio-temporal volume, so 10K points may under-resolve the PDE "
    "residual, especially the biharmonic nabla^4 term which requires "
    "4th-order spatial derivatives. "
    "Differs from canonical: n_interior=20000, n_initial=2000.",
    {
        "sampling.n_interior": 20000,
        "sampling.n_initial": 2000,
    },
)

EXPERIMENTS["F4_points_5k"] = make(
    "F4_points_5k",
    "Half collocation points: 5,000 interior (vs 10,000 canonical), "
    "500 IC points (vs 1,000). Tests whether the canonical point count is "
    "already above the useful resolution level for the t<=10 horizon. "
    "Differs from canonical: n_interior=5000, n_initial=500.",
    {
        "sampling.n_interior": 5000,
        "sampling.n_initial": 500,
    },
)

EXPERIMENTS["F5_adam_resample_200"] = make(
    "F5_adam_resample_200",
    "Frequent Adam resampling: refresh collocation points every 200 Adam "
    "epochs instead of 500. Tests whether earlier first-order exposure to "
    "fresh RAD points improves the warm-start before quasi-Newton. "
    "Differs from canonical: resample_every=200.",
    {"sampling.resample_every": 200},
)

EXPERIMENTS["F6_adam_resample_1000"] = make(
    "F6_adam_resample_1000",
    "Infrequent Adam resampling: refresh collocation points every 1,000 "
    "Adam epochs instead of 500. Tests whether a longer-lived Adam point "
    "set gives a cleaner warm-start or encourages overfitting. "
    "Differs from canonical: resample_every=1000.",
    {"sampling.resample_every": 1000},
)


# ============================================================================
# GROUP G: Hessian Scaling & Power Transform
# ============================================================================
# Paper: tests initial Hessian scaling (Nocedal-Wright H0 scaling).
# Power transform: minimizes L^(1/p) instead of L, flattening the loss
# landscape to improve BFGS conditioning near sharp minima.

EXPERIMENTS["G1_hessian_scaled"] = make(
    "G1_hessian_scaled",
    "Nocedal-Wright initial Hessian scaling enabled. On the first BFGS "
    "iteration (when H_0 = I), scales H_0 by tau_k = rho_k * (y^T y) "
    "so the initial step length is appropriate. After the first step, "
    "scaling is disabled (handled by the variant's self-scaling). "
    "Differs from canonical: initial_scale=true instead of false.",
    {"training.bfgs.initial_scale": True},
)

EXPERIMENTS["G2_power_2"] = make(
    "G2_power_2",
    "Power transform p=2: minimizes sqrt(L) instead of L. "
    "Flattens the loss landscape, making the Hessian better conditioned "
    "near sharp minima. Gradient is scaled by 1/(2*sqrt(L)). "
    "Effective loss for BFGS = L^(1/2); reported loss = (effective)^2. "
    "Differs from canonical: power=2.0 instead of 1.0.",
    {"training.bfgs.power": 2.0},
)

EXPERIMENTS["G3_power_4"] = make(
    "G3_power_4",
    "Power transform p=4: minimizes L^(1/4). "
    "Stronger flattening than p=2. Our v1 suite found p=4 was the second "
    "best in L2 error (0.945). Gradient is scaled by (1/4)*L^(-3/4). "
    "Effective loss for BFGS = L^(1/4); reported loss = (effective)^4. "
    "Differs from canonical: power=4.0 instead of 1.0.",
    {"training.bfgs.power": 4.0},
)


# ============================================================================
# GROUP H: RAD Parameter Sweep
# ============================================================================
# RAD samples PDE points with probability proportional to |residual|^k1
# plus a baseline k2 term. These configs perturb each parameter one at a time.

EXPERIMENTS["H1_rad_k1_0p5"] = make(
    "H1_rad_k1_0p5",
    "RAD exponent sweep: k1=0.5 instead of 1.0. Softer residual weighting "
    "makes RAD less concentrated on sharp residual peaks. "
    "Differs from canonical: sampling.rad.k1=0.5.",
    {"sampling.rad.k1": 0.5},
)

EXPERIMENTS["H2_rad_k1_2p0"] = make(
    "H2_rad_k1_2p0",
    "RAD exponent sweep: k1=2.0 instead of 1.0. Stronger residual weighting "
    "pushes sampling more aggressively toward discontinuities and sharp "
    "interfaces. Differs from canonical: sampling.rad.k1=2.0.",
    {"sampling.rad.k1": 2.0},
)

EXPERIMENTS["H3_rad_k2_0p5"] = make(
    "H3_rad_k2_0p5",
    "RAD baseline sweep: k2=0.5 instead of 1.0. Lower baseline probability "
    "makes the residual-driven term dominate more strongly. "
    "Differs from canonical: sampling.rad.k2=0.5.",
    {"sampling.rad.k2": 0.5},
)

EXPERIMENTS["H4_rad_k2_2p0"] = make(
    "H4_rad_k2_2p0",
    "RAD baseline sweep: k2=2.0 instead of 1.0. Higher baseline probability "
    "keeps RAD closer to uniform sampling while still biasing toward high "
    "residual regions. Differs from canonical: sampling.rad.k2=2.0.",
    {"sampling.rad.k2": 2.0},
)


# ============================================================================
# WRITE ALL CONFIGS
# ============================================================================

def write_config(name, cfg):
    """Write a single config file with a descriptive header comment."""
    out_dir = FH_OUT_DIR if name in FH_ABLATION_NAMES else CORE_OUT_DIR
    path = os.path.join(out_dir, f"{name}.yaml")

    # Wrap description into comment lines
    desc = cfg["experiment"]["description"]
    desc_lines = []
    line = ""
    for word in desc.split():
        if len(line) + len(word) + 1 > 70:
            desc_lines.append(f"#   {line}")
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        desc_lines.append(f"#   {line}")

    header = (
        f"# {'=' * 72}\n"
        f"# {name}\n"
        f"# {'=' * 72}\n"
        + "\n".join(desc_lines) + "\n"
        f"# {'=' * 72}\n"
    )

    with open(path, "w") as f:
        f.write(header + "\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                  width=80, allow_unicode=True)

    return path


if __name__ == "__main__":
    print(f"Generating {len(EXPERIMENTS)} configs in {CONFIG_ROOT}/\n")

    for name in sorted(EXPERIMENTS):
        path = write_config(name, EXPERIMENTS[name])
        print(f"  {path}")

    print(f"\nDone. {len(EXPERIMENTS)} configs written.\n")

    # Summary table
    print(f"{'Name':25s} {'Adam':>6s} {'BFGS':>6s} "
          f"{'Variant':15s} {'Network':30s} {'Nchg':>5s} {'Extras'}")
    print("-" * 110)
    for name in sorted(EXPERIMENTS):
        cfg = EXPERIMENTS[name]
        adam_ep = cfg["training"]["adam"]["epochs"]
        bfgs_ep = cfg["training"]["bfgs"]["epochs"]
        variant = cfg["training"]["bfgs"]["variant"]
        net = str(cfg["network"]["layer_dims"])
        nchange = cfg["training"]["bfgs"]["batch_size"]
        extras = []
        if cfg["training"]["adam"].get("optimizer", "adam") == "radam":
            extras.append("RAdam")
        if cfg["training"]["bfgs"]["initial_scale"]:
            extras.append("H0-scale")
        if cfg["training"]["bfgs"]["power"] != 1.0:
            extras.append(f"p={cfg['training']['bfgs']['power']}")
        if cfg["sampling"]["n_interior"] != 10000:
            extras.append(f"Nint={cfg['sampling']['n_interior']}")
        if cfg["sampling"]["resample_every"] != 500:
            extras.append(f"resamp={cfg['sampling']['resample_every']}")
        if cfg["sampling"]["rad"]["k1"] != 1.0:
            extras.append(f"k1={cfg['sampling']['rad']['k1']}")
        if cfg["sampling"]["rad"]["k2"] != 1.0:
            extras.append(f"k2={cfg['sampling']['rad']['k2']}")
        print(f"{name:25s} {adam_ep:6d} {bfgs_ep:6d} "
              f"{variant:15s} {net:30s} {nchange:5d} "
              f"{', '.join(extras)}")
