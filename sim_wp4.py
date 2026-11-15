#!/usr/bin/env python3
"""
sim_wp4.py -- Implementation of Work Package 4 (WP4) Efficiency Analysis.
Evaluates:
  - Theoretical asymptotic ratio R_inf = 2 ln(1/alpha) / (z_alpha + z_beta)^2
  - Siegmund boundary overshoot correction
  - Finite-sample Monte Carlo simulated rounds vs Neyman-Pearson M*
  - Reports in dimensionless rounds and frames only.
"""
import argparse, math, os
import numpy as np
import pandas as pd
from scipy import stats

def run_efficiency_grid(n=200_000, seed=20261011, sigma=25.0):
    rng = np.random.default_rng(seed)
    alphas = [1e-2, 5e-3, 1e-3, 1e-4]
    betas = [1e-1, 5e-2, 1e-2, 1e-3]
    delta_vals = [25.0, 30.0, 40.0, 50.0, 60.0, 80.0]

    rows = []

    for a in alphas:
        za = stats.norm.isf(a)
        for b in betas:
            zb = stats.norm.isf(b)
            r_inf = 2.0 * math.log(1.0 / a) / ((za + zb) ** 2)

            for d in delta_vals:
                # Neyman-Pearson fixed sample size M*
                m_fixed = max(1, math.ceil(((za + zb) * sigma / d) ** 2))

                # Siegmund overshoot correction: rho_+(theta) ~ 0.583 for small theta
                theta = d / (2.0 * sigma)
                # approx rho_+ = 0.583 + 0.25 * theta
                mean_overshoot = (d / sigma) * (0.583 + 0.25 * theta)
                drift_live = (d ** 2) / (2.0 * sigma ** 2)
                theory_asn = (math.log(1.0 / a) + mean_overshoot) / drift_live

                # Simulation of sequential e-process under live user (D = 0)
                # Horizon T set to 2 * m_fixed + 20 to avoid premature truncation
                T_sim = max(40, 2 * m_fixed + 20)
                eps = rng.normal(0, sigma, (n, T_sim))
                # Live subject D = 0 => Y = eps
                inc = d * (d / 2.0 - eps) / (sigma ** 2)
                L = np.cumsum(inc, axis=1)
                up, lo = math.log(1.0 / a), math.log(b)
                hu, hl = L >= up, L <= lo
                never = 10 ** 9
                t_up = np.where(hu.any(1), hu.argmax(1), never)
                t_lo = np.where(hl.any(1), hl.argmax(1), never)
                r_sim = np.minimum(np.minimum(t_up, t_lo), T_sim - 1) + 1
                asn_sim = float(r_sim.mean())

                ratio_sim = asn_sim / m_fixed
                ratio_siegmund = theory_asn / m_fixed

                rows.append({
                    "alpha": a,
                    "beta": b,
                    "delta": d,
                    "fixed_M_star": m_fixed,
                    "sim_ASN_rounds": asn_sim,
                    "siegmund_ASN_rounds": theory_asn,
                    "ratio_sim": ratio_sim,
                    "ratio_siegmund": ratio_siegmund,
                    "ratio_asymptotic": r_inf
                })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--out", default=".")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    print(f"Running WP4 Efficiency Analysis (N={args.n})...", flush=True)
    df = run_efficiency_grid(args.n, args.seed)
    p = os.path.join(args.out, "efficiency_analysis.csv")
    df.to_csv(p, index=False)
    print("Efficiency analysis complete. Written to", p)

    # Print summary slice at benchmark (alpha=1e-3, beta=1e-2)
    bench = df[(df.alpha == 1e-3) & (df.beta == 1e-2)]
    print("\nBenchmark slice (alpha=1e-3, beta=1e-2, sigma=25):")
    print(bench[["delta", "fixed_M_star", "sim_ASN_rounds", "ratio_sim", "ratio_siegmund", "ratio_asymptotic"]].to_string(index=False))

if __name__ == "__main__":
    main()
