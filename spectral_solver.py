#!/usr/bin/env python3
"""
Pseudo-Spectral Reference Solver for the Cahn-Hilliard Equation
================================================================

Solves:
    ∂U/∂t = D · ∇²(∇²U + a₂·U + a₄·U³)

on a periodic domain (default 128×128) using FFT + semi-implicit time-stepping.

The initial condition and domain are matched **exactly** to the PINN training
config (cahn_hilliard_canonical.yaml):
  - Domain: [0, 128]² with Δx = Δy = 1
  - IC: U₀ ~ Uniform(-1, 1) on 128×128 grid, seed=42, σ=2.0 Gaussian smoothing
  - Time: t ∈ [0, 20]
  - D = a₂ = a₄ = 1

Time integration: semi-implicit Euler (linear part implicit, nonlinear explicit),
with small dt for accuracy.  Optionally ETDRK4 for higher-order accuracy.

Outputs:
    reference_solution.npz  — U(t, x, y) at saved time-points, plus grid arrays
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================================
# PDE parameters (match canonical config)
# ============================================================================
D  = 1.0
A2 = 1.0
A4 = 1.0

# Domain
NX, NY = 128, 128
LX, LY = 128.0, 128.0
DX, DY = LX / NX, LY / NY

T_MIN, T_MAX = 0.0, 20.0


# ============================================================================
# Initial Condition — must match run_experiment.py / build_ic_interpolator()
# ============================================================================
def generate_ic(seed=42, nx=128, ny=128, low=-1.0, high=1.0, sigma=2.0):
    """Generate the same IC used by the PINN training script."""
    rng = np.random.RandomState(seed)
    field = rng.uniform(low, high, size=(ny, nx))
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma, mode="wrap")
    return field


# ============================================================================
# Wavenumber arrays
# ============================================================================
def build_wavenumbers(nx, ny, lx, ly):
    """FFT wavenumbers on a periodic [0, Lx] × [0, Ly] domain."""
    kx = 2.0 * np.pi / lx * np.fft.fftfreq(nx, d=1.0 / nx)
    ky = 2.0 * np.pi / ly * np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    K4 = K2**2
    return KX, KY, K2, K4


# ============================================================================
# Semi-Implicit Euler (1st-order, unconditionally stable for linear part)
# ============================================================================
def step_semi_implicit(U_hat, K2, K4, dt):
    """
    One semi-implicit Euler step.

    PDE in Fourier space:
        dÛ/dt = -(D·K⁴ + D·a₂·K²)·Û - D·a₄·K²·FT(U³)

    Linear operator:  L = -(D·K⁴ + D·a₂·K²)   [treated implicitly]
    Nonlinear:        N = -D·a₄·K²·FT(U³)       [treated explicitly]

    Update:  Û^{n+1} = (Û^n + dt·N^n) / (1 - dt·L)
                      = (Û^n - dt·D·a₄·K²·FT(U³ⁿ)) / (1 + dt·D·(K⁴ + a₂·K²))
    """
    U = np.real(np.fft.ifft2(U_hat))
    U3_hat = np.fft.fft2(U**3)

    # Numerator: Û^n + dt * (nonlinear explicit term)
    numer = U_hat - dt * D * A4 * K2 * U3_hat
    # Denominator: 1 - dt * L = 1 + dt * D * (K⁴ + a₂*K²)
    denom = 1.0 + dt * D * (K4 + A2 * K2)

    return numer / denom


# ============================================================================
# ETDRK4 (4th-order exponential time-differencing Runge-Kutta)
# ============================================================================
def build_etdrk4_coefficients(L, dt, M=32):
    """
    Precompute ETDRK4 coefficients using the contour-integral trick
    of Kassam & Trefethen (2005) for numerical stability.

    L : array of linear operator eigenvalues (shape = grid shape)
    dt: time step
    M : number of contour quadrature points
    """
    # Contour points on a circle
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # (M,)

    # L has shape (NY, NX); expand for broadcasting
    LR = dt * L[..., np.newaxis] + r[np.newaxis, np.newaxis, :]  # (NY, NX, M)

    E  = np.exp(dt * L)      # exp(L·dt)
    E2 = np.exp(dt * L / 2)  # exp(L·dt/2)

    f1 = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f2 = dt * np.real(np.mean(
        (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3,
        axis=-1,
    ))
    f3 = dt * np.real(np.mean(
        (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3,
        axis=-1,
    ))
    f4 = dt * np.real(np.mean(
        (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3,
        axis=-1,
    ))

    return E, E2, f1, f2, f3, f4


def step_etdrk4(U_hat, K2, E, E2, f1, f2, f3, f4):
    """
    One ETDRK4 step.

    The nonlinear operator N(U) = -D * A4 * K² * FT(U³).
    """
    def NL(v_hat):
        v = np.real(np.fft.ifft2(v_hat))
        return -D * A4 * K2 * np.fft.fft2(v**3)

    Nu = NL(U_hat)
    a_hat = E2 * U_hat + f1 * Nu
    Na = NL(a_hat)
    b_hat = E2 * U_hat + f1 * Na
    Nb = NL(b_hat)
    c_hat = E2 * a_hat + f1 * (2 * Nb - Nu)
    Nc = NL(c_hat)

    return E * U_hat + f2 * Nu + 2 * f3 * (Na + Nb) + f4 * Nc


# ============================================================================
# Main solver
# ============================================================================
def solve_cahn_hilliard(
    dt=0.005,
    method="etdrk4",
    n_save=101,
    seed=42,
    sigma=2.0,
    verbose=True,
    nx=None,
    ny=None,
    lx=None,
    ly=None,
):
    """
    Solve Cahn-Hilliard and return solution snapshots.

    Parameters
    ----------
    dt : float
        Time step.
    method : str
        'semi_implicit' or 'etdrk4'.
    n_save : int
        Number of time-points to save (uniformly in [T_MIN, T_MAX]).
    seed : int
        IC random seed (must match PINN config).
    sigma : float
        IC Gaussian smoothing σ.
    verbose : bool
        Print progress.
    nx, ny : int or None
        Grid resolution (defaults to module-level NX, NY).
    lx, ly : float or None
        Domain size (defaults to module-level LX, LY).

    Returns
    -------
    t_save : (n_save,) array of saved times
    xs, ys : (nx,), (ny,) grid-centre coordinate arrays
    U_save : (n_save, ny, nx) array of solution snapshots
    """
    nx = nx or NX
    ny = ny or NY
    lx = lx or LX
    ly = ly or LY
    dx = lx / nx
    dy = ly / ny

    # Grid coordinates (cell-centred, matching PINN's interpolator)
    xs = np.linspace(0.5 * dx, lx - 0.5 * dx, nx)
    ys = np.linspace(0.5 * dy, ly - 0.5 * dy, ny)

    # Initial condition
    U = generate_ic(seed=seed, nx=nx, ny=ny, sigma=sigma)
    U_hat = np.fft.fft2(U)

    # Wavenumbers
    _, _, K2, K4 = build_wavenumbers(nx, ny, lx, ly)

    # Linear operator eigenvalues:  L(k) = -D*(K⁴ + a₂*K²)
    L = -D * (K4 + A2 * K2)

    # ETDRK4 setup
    if method == "etdrk4":
        E, E2, f1, f2, f3, f4 = build_etdrk4_coefficients(L, dt)

    # Time-stepping
    n_steps = int(np.ceil((T_MAX - T_MIN) / dt))
    t_save = np.linspace(T_MIN, T_MAX, n_save)
    U_save = np.zeros((n_save, ny, nx))
    U_save[0] = U.copy()
    save_idx = 1

    t = T_MIN
    for step in range(1, n_steps + 1):
        if method == "semi_implicit":
            U_hat = step_semi_implicit(U_hat, K2, K4, dt)
        elif method == "etdrk4":
            U_hat = step_etdrk4(U_hat, K2, E, E2, f1, f2, f3, f4)
        else:
            raise ValueError(f"Unknown method: {method}")

        t = T_MIN + step * dt

        # Save snapshot if we've passed the next save time
        if save_idx < n_save and t >= t_save[save_idx] - 1e-12:
            U_save[save_idx] = np.real(np.fft.ifft2(U_hat))
            save_idx += 1
            if verbose and save_idx % 10 == 0:
                print(f"  saved snapshot {save_idx}/{n_save}  t={t:.4f}  "
                      f"<U>={U_save[save_idx-1].mean():.6f}  "
                      f"max|U|={np.abs(U_save[save_idx-1]).max():.4f}")

    if verbose:
        print(f"Solver complete: {n_steps} steps, {save_idx} snapshots saved.")
        print(f"  <U>(t=0)  = {U_save[0].mean():.8f}")
        print(f"  <U>(t=20) = {U_save[-1].mean():.8f}")
        print(f"  Mass drift = {U_save[-1].mean() - U_save[0].mean():.2e}")

    return t_save, xs, ys, U_save


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate reference solution for Cahn-Hilliard equation"
    )
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Time step (default: 0.005)")
    parser.add_argument("--method", choices=["semi_implicit", "etdrk4"],
                        default="etdrk4", help="Time integration method")
    parser.add_argument("--n-save", type=int, default=101,
                        help="Number of snapshots to save")
    parser.add_argument("--seed", type=int, default=42,
                        help="IC random seed")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="IC Gaussian smoothing σ")
    parser.add_argument("--nx", type=int, default=NX,
                        help=f"Grid points in x (default: {NX})")
    parser.add_argument("--ny", type=int, default=NY,
                        help=f"Grid points in y (default: {NY})")
    parser.add_argument("--lx", type=float, default=LX,
                        help=f"Domain length in x (default: {LX})")
    parser.add_argument("--ly", type=float, default=LY,
                        help=f"Domain length in y (default: {LY})")
    parser.add_argument("--output", type=str, default="reference_solution.npz",
                        help="Output file path")
    args = parser.parse_args()

    print("=" * 60)
    print("Cahn-Hilliard Spectral Solver")
    print(f"  Method: {args.method}   dt={args.dt}")
    print(f"  Domain: {args.nx}×{args.ny}, L={args.lx}×{args.ly}")
    print(f"  Time:   [{T_MIN}, {T_MAX}]")
    print(f"  IC:     seed={args.seed}, σ={args.sigma}")
    print(f"  Saving {args.n_save} snapshots → {args.output}")
    print("=" * 60)

    t_save, xs, ys, U_save = solve_cahn_hilliard(
        dt=args.dt,
        method=args.method,
        n_save=args.n_save,
        seed=args.seed,
        sigma=args.sigma,
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
    )

    np.savez_compressed(
        args.output,
        t=t_save,
        x=xs,
        y=ys,
        U=U_save,
        dt_solver=args.dt,
        method=args.method,
    )
    fsize = os.path.getsize(args.output) / 1024**2
    print(f"\nSaved: {args.output}  ({fsize:.1f} MB)")
    print(f"  U shape: {U_save.shape}  (n_save, NY, NX)")


if __name__ == "__main__":
    main()
