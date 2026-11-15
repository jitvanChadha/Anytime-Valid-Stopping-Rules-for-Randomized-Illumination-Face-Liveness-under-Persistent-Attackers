#!/usr/bin/env python3
"""
sim_wp3.py -- Implementation of Work Package 3 (WP3) Experiments:
  E4' : Live-offset spread (fixed-lambda vs plug-in lambda) with theoretical floor comparison
  E9' : Heavy tails and quantized frame-grid noise using the WP2 Bernstein e-process
  E5'/E6' : Overlay of closed-form Lundberg bounds on wrong-floor and wrong-sigma sweeps
  E11 : Intermittent floor violations (palette guessing q in {0.01..0.5}, cheap-tint D in [0, 5])
  E12 : Wrong-sigma sweep sigma'/sigma in [0.8, 1.5]

Ground rules:
  - Fixed seeds (default 20261011)
  - N >= 500,000 per cell
  - 95% Clopper-Pearson confidence intervals on every rate
  - Rule-of-three upper bound for zero-event cells (3/N)
  - Exports clean CSVs for automated plotting in make_figures.py
"""
import argparse, math, os, time
import numpy as np
import pandas as pd
from scipy import stats

def cp_ci(k, n, level=0.95):
    """Clopper-Pearson exact 95% CI with rule-of-three for k=0."""
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2.0, k, n - k + 1))
    hi = (3.0 / n) if k == 0 else (1.0 if k == n else float(stats.beta.ppf(1.0 - a / 2.0, k + 1, n - k)))
    return lo, hi

# ------------------------------------------------------------------ E-process engines
def run_sequential(increments, alpha, beta):
    """General sequential engine for cumulative log-evidence L_t = sum increments."""
    L = np.cumsum(increments, axis=1)
    up, lo = math.log(1.0 / alpha), math.log(beta)
    hu, hl = L >= up, L <= lo
    never = 10 ** 9
    t_up = np.where(hu.any(1), hu.argmax(1), never)
    t_lo = np.where(hl.any(1), hl.argmax(1), never)
    accept = t_up < t_lo
    rounds = np.minimum(np.minimum(t_up, t_lo), L.shape[1] - 1) + 1
    return accept, rounds

# ------------------------------------------------------------------ E4': Live Offset (Fixed vs Plug-in)
def run_e4_prime(n, seed, alpha=1e-3, beta=1e-2, delta=50.0, sigma=25.0, T=20):
    rng = np.random.default_rng([seed, 4])
    s2 = sigma ** 2
    tau_vals = [0.0, 5.0, 10.0, 15.0, 20.0]
    rows = []

    for tau in tau_vals:
        c_live = rng.normal(0.0, tau, (n, 1)) if tau > 0 else np.zeros((n, 1))
        eps = rng.normal(0.0, sigma, (n, T))
        Y = c_live + eps

        # 1. Fixed lambda = delta / sigma^2
        inc_fixed = delta * (delta / 2.0 - Y) / s2
        acc_fixed, r_fixed = run_sequential(inc_fixed, alpha, beta)
        bpcer_fixed = float((~acc_fixed).mean())
        k_fixed = int((~acc_fixed).sum())
        lo_f, hi_f = cp_ci(k_fixed, n)

        # 2. Plug-in lambda: lambda_t = (delta - mu_hat_t)_+ / sigma^2
        # mu_hat_1 = 0, mu_hat_t = running mean of Y_1..Y_{t-1}
        inc_plugin = np.empty_like(Y)
        # t=1: prior mu_hat = 0 => lambda_1 = delta / sigma^2
        lam_1 = delta / s2
        inc_plugin[:, 0] = lam_1 * (delta - Y[:, 0]) - 0.5 * (lam_1 ** 2) * s2
        cum_Y = Y[:, 0].copy()
        for t in range(1, T):
            mu_hat = cum_Y / t
            lam_t = np.clip((delta - mu_hat) / s2, 0.0, 2.0 * delta / s2)
            inc_plugin[:, t] = lam_t * (delta - Y[:, t]) - 0.5 * (lam_t ** 2) * s2
            cum_Y += Y[:, t]

        acc_plugin, r_plugin = run_sequential(inc_plugin, alpha, beta)
        bpcer_plugin = float((~acc_plugin).mean())
        k_plugin = int((~acc_plugin).sum())
        lo_p, hi_p = cp_ci(k_plugin, n)

        # Theoretical floors from WP2(iii)
        floor_fixed = stats.norm.sf(delta / (2.0 * tau)) if tau > 0 else 0.0
        floor_plugin = stats.norm.sf(delta / tau) if tau > 0 else 0.0

        rows.append({
            "tau_live": tau,
            "test": "E-floor (fixed lambda)",
            "BPCER": bpcer_fixed,
            "ci_lo": lo_f,
            "ci_hi": hi_f,
            "mean_rounds": float(r_fixed.mean()),
            "theoretical_floor": floor_fixed
        })
        rows.append({
            "tau_live": tau,
            "test": "E-floor (plug-in lambda)",
            "BPCER": bpcer_plugin,
            "ci_lo": lo_p,
            "ci_hi": hi_p,
            "mean_rounds": float(r_plugin.mean()),
            "theoretical_floor": floor_plugin
        })

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ E9': Heavy Tails & Quantized Noise (Bernstein E-Process)
def run_e9_prime(n, seed, alpha=1e-3, beta=1e-2, delta=50.0, sigma=25.0, T=20):
    rng = np.random.default_rng([seed, 9])
    c_bound = delta
    lam = delta / (sigma ** 2)
    psi_B = (sigma ** 2 / c_bound ** 2) * (math.exp(lam * c_bound) - 1.0 - lam * c_bound)

    noise_models = [
        ("Gaussian", lambda sz: rng.normal(0, sigma, sz)),
        ("Student-t (df=5)", lambda sz: rng.standard_t(5, sz) * (sigma * math.sqrt(3.0 / 5.0))),
        ("Student-t (df=3)", lambda sz: rng.standard_t(3, sz) * (sigma * math.sqrt(1.0 / 3.0))),
        ("Laplace", lambda sz: rng.laplace(0, sigma / math.sqrt(2.0), sz)),
        ("Contaminated 10%", lambda sz: np.where(rng.random(sz) < 0.1, rng.normal(0, 3*sigma/math.sqrt(1.8), sz), rng.normal(0, sigma/math.sqrt(1.8), sz))),
        ("Quantized (16.7ms frame)", lambda sz: (np.round(rng.normal(0, sigma, sz) / 16.6667) * 16.6667))
    ]

    rows = []
    for name, noise_fn in noise_models:
        eps_att = noise_fn((n, T))
        Y_att = delta + eps_att

        # Method 1: Clamped Gaussian LR
        Y_clamped = np.maximum(0.0, Y_att)
        inc_clamped = delta * (delta / 2.0 - Y_clamped) / (sigma ** 2)
        acc_cg, r_cg = run_sequential(inc_clamped, alpha, beta)
        k_cg = int(acc_cg.sum())
        lo_cg, hi_cg = cp_ci(k_cg, n)

        # Method 2: Non-parametric Bernstein E-Process (WP2 Theorem 5.2)
        X_b = np.minimum(c_bound, delta - np.maximum(0.0, Y_att))
        inc_bern = lam * X_b - psi_B
        acc_b, r_b = run_sequential(inc_bern, alpha, beta)
        k_b = int(acc_b.sum())
        lo_b, hi_b = cp_ci(k_b, n)

        rows.append({
            "noise_model": name,
            "test": "Clamped Gaussian e-process",
            "APCER": float(acc_cg.mean()),
            "ci_lo": lo_cg,
            "ci_hi": hi_cg,
            "APCER_over_alpha": float(acc_cg.mean() / alpha),
            "mean_rounds": float(r_cg.mean())
        })
        rows.append({
            "noise_model": name,
            "test": "Bernstein e-process (Theorem 5.2)",
            "APCER": float(acc_b.mean()),
            "ci_lo": lo_b,
            "ci_hi": hi_b,
            "APCER_over_alpha": float(acc_b.mean() / alpha),
            "mean_rounds": float(r_b.mean())
        })

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ E5'/E6': Misspecification Sweeps with Lundberg Bounds
def run_e5_prime(n, seed, alpha=1e-3, beta=1e-2, delta_design=50.0, sigma=25.0, T=20):
    rng = np.random.default_rng([seed, 5])
    d_primes = [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 65.0]
    rows = []

    for dp in d_primes:
        eps = rng.normal(0, sigma, (n, T))
        Y = dp + eps
        inc = delta_design * (delta_design / 2.0 - Y) / (sigma ** 2)
        acc, r = run_sequential(inc, alpha, beta)
        k = int(acc.sum())
        lo, hi = cp_ci(k, n)

        # Closed-form Lundberg root and upper bound (WP2 Theorem 6.1)
        if dp > delta_design / 2.0:
            theta_star = 2.0 * dp / delta_design - 1.0
            lundberg_bound = float(alpha ** theta_star)
        else:
            theta_star = 0.0
            lundberg_bound = 1.0

        rows.append({
            "true_delta_prime": dp,
            "APCER": float(acc.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "APCER_over_alpha": float(acc.mean() / alpha),
            "mean_rounds": float(r.mean()),
            "lundberg_root": theta_star,
            "lundberg_bound_over_alpha": lundberg_bound / alpha
        })

    return pd.DataFrame(rows)

def run_e12(n, seed, alpha=1e-3, beta=1e-2, delta=50.0, sigma_design=25.0, T=20):
    rng = np.random.default_rng([seed, 12])
    factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    rows = []

    for f in factors:
        sigma_prime = sigma_design * f
        eps = rng.normal(0, sigma_prime, (n, T))
        Y = delta + eps
        inc = delta * (delta / 2.0 - Y) / (sigma_design ** 2)
        acc, r = run_sequential(inc, alpha, beta)
        k = int(acc.sum())
        lo, hi = cp_ci(k, n)

        # Lundberg bound (WP2 Corollary 6.2)
        theta_star = (sigma_design / sigma_prime) ** 2
        lundberg_bound = float(alpha ** theta_star)

        rows.append({
            "sigma_ratio": f,
            "true_sigma_prime": sigma_prime,
            "APCER": float(acc.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "APCER_over_alpha": float(acc.mean() / alpha),
            "mean_rounds": float(r.mean()),
            "lundberg_root": theta_star,
            "lundberg_bound_over_alpha": lundberg_bound / alpha
        })

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ E11: Intermittent Floor Violations & Cheap-Tint
def run_e11(n, seed, alpha=1e-3, beta=1e-2, delta=50.0, sigma=25.0, T=20):
    rng = np.random.default_rng([seed, 11])
    rows = []

    # Part A: Intermittent floor violation (palette guessing prob q)
    q_vals = [0.01, 0.05, 0.10, 0.25, 0.50]
    for q in q_vals:
        # With prob q, attacker correctly guesses color in round t and pre-renders (D=0), else D=delta
        is_guess = rng.random((n, T)) < q
        D_mat = np.where(is_guess, 0.0, delta)
        eps = rng.normal(0, sigma, (n, T))
        Y = D_mat + eps
        inc = delta * (delta / 2.0 - Y) / (sigma ** 2)
        acc, r = run_sequential(inc, alpha, beta)
        k = int(acc.sum())
        lo, hi = cp_ci(k, n)
        rows.append({
            "attack_type": f"Palette-Guessing (q={q})",
            "parameter": q,
            "APCER": float(acc.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "APCER_over_alpha": float(acc.mean() / alpha),
            "mean_rounds": float(r.mean())
        })

    # Part B: Cheap-tint attacker (screen-buffer linear tint, D in [0, 5] ms)
    for d_tint in [0.0, 1.0, 2.0, 5.0]:
        eps = rng.normal(0, sigma, (n, T))
        Y = d_tint + eps
        inc = delta * (delta / 2.0 - Y) / (sigma ** 2)
        acc, r = run_sequential(inc, alpha, beta)
        k = int(acc.sum())
        lo, hi = cp_ci(k, n)
        rows.append({
            "attack_type": f"Cheap-Tint (D={d_tint}ms)",
            "parameter": d_tint,
            "APCER": float(acc.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "APCER_over_alpha": float(acc.mean() / alpha),
            "mean_rounds": float(r.mean())
        })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="WP3 Experiments runner.")
    parser.add_argument("--n", type=int, default=500_000, help="Sessions per cell")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Starting WP3 simulation battery: N={args.n}, seed={args.seed}")

    t0 = time.time()
    print("Running E4' (Live Offset fixed vs plug-in)...", flush=True)
    df_e4 = run_e4_prime(args.n, args.seed)
    df_e4.to_csv(os.path.join(args.out, "e4_prime_live_offset.csv"), index=False)
    print(f"E4' done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("Running E9' (Heavy Tails & Bernstein E-Process)...", flush=True)
    df_e9 = run_e9_prime(args.n, args.seed)
    df_e9.to_csv(os.path.join(args.out, "e9_prime_heavy_tails.csv"), index=False)
    print(f"E9' done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("Running E5' (Wrong-Floor Lundberg Overlay)...", flush=True)
    df_e5 = run_e5_prime(args.n, args.seed)
    df_e5.to_csv(os.path.join(args.out, "e5_prime_wrong_floor.csv"), index=False)
    print(f"E5' done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("Running E12 (Wrong-Sigma Lundberg Overlay)...", flush=True)
    df_e12 = run_e12(args.n, args.seed)
    df_e12.to_csv(os.path.join(args.out, "e12_wrong_sigma.csv"), index=False)
    print(f"E12 done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("Running E11 (Intermittent Violations & Cheap-Tint)...", flush=True)
    df_e11 = run_e11(args.n, args.seed)
    df_e11.to_csv(os.path.join(args.out, "e11_intermittent_tint.csv"), index=False)
    print(f"E11 done ({time.time()-t0:.1f}s)")

    print("\nAll WP3 experiments completed successfully.")

if __name__ == "__main__":
    main()
