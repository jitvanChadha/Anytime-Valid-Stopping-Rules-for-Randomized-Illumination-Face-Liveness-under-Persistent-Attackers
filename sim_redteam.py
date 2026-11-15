#!/usr/bin/env python3
"""
sim_redteam.py -- experiments added after the 2026-11-15 red-team pass.

R1  Bernstein e-process with a usable betting weight.
    The earlier E9' run used lambda = delta/sigma^2 in the Bernstein form. With that weight the
    per-round compensator psi_B exceeds the largest possible gain, so every session (attacker AND
    live) is rejected in round 1: "0 false accepts" was vacuous (BPCER = 100%).
    Here lambda_B = ln(1 + delta^2/sigma_bar^2) / delta, the maximiser of lambda*delta - psi_B(lambda),
    and we report APCER under six noise models AND live BPCER / rounds.
R2  Numerical check of the corrected proof of Theorem 5.2 (two-point bound monotone in the mean).
R3  Tail of the stopping time (p50/p95/p99 rounds) for live users, E-floor vs fixed M.
R4  Plug-in betting at the calibrated sigma (live-offset sweep).

Usage: python sim_redteam.py [--n 500000] [--out redteam_out]
"""
import argparse, math, os
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261011
ALPHA, BETA, DELTA = 1e-3, 1e-2, 50.0


def cp_ci(k, n, level=0.95):
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2.0, k, n - k + 1))
    hi = (3.0 / n) if k == 0 else (1.0 if k == n else float(stats.beta.ppf(1.0 - a / 2.0, k + 1, n - k)))
    return lo, hi


def run_sequential(inc, alpha=ALPHA, beta=BETA):
    L = np.cumsum(inc, axis=1)
    up, lo = math.log(1.0 / alpha), math.log(beta)
    hu, hl = L >= up, L <= lo
    never = 10 ** 9
    t_up = np.where(hu.any(1), hu.argmax(1), never)
    t_lo = np.where(hl.any(1), hl.argmax(1), never)
    accept = t_up < t_lo
    rounds = np.minimum(np.minimum(t_up, t_lo), L.shape[1] - 1) + 1
    return accept, rounds


def noise_models(rng, sigma):
    return [
        ("Gaussian", lambda sz: rng.normal(0, sigma, sz)),
        ("Student-t (df=5)", lambda sz: rng.standard_t(5, sz) * (sigma * math.sqrt(3.0 / 5.0))),
        ("Student-t (df=3)", lambda sz: rng.standard_t(3, sz) * (sigma * math.sqrt(1.0 / 3.0))),
        ("Laplace", lambda sz: rng.laplace(0, sigma / math.sqrt(2.0), sz)),
        ("Contaminated 10%", lambda sz: np.where(rng.random(sz) < 0.1,
                                                 rng.normal(0, 3 * sigma / math.sqrt(1.8), sz),
                                                 rng.normal(0, sigma / math.sqrt(1.8), sz))),
        ("Quantized (33.3ms frame)", lambda sz: np.round(rng.normal(0, sigma, sz) / 33.3333) * 33.3333),
    ]


def bern_inc(Y, lam, sigma_bar, c=DELTA):
    psi = (sigma_bar ** 2 / c ** 2) * (math.exp(lam * c) - 1.0 - lam * c)
    X = DELTA - np.maximum(0.0, Y)          # X <= delta = c
    return lam * X - psi


def r1_bernstein(n, out):
    rows = []
    for sigma, T in [(25.0, 20), (25.0, 40), (11.756, 20)]:
        rng = np.random.default_rng([SEED, 901, int(sigma * 1000), T])
        lam_b = math.log(1.0 + DELTA ** 2 / sigma ** 2) / DELTA
        lam_g = DELTA / sigma ** 2
        for name, fn in noise_models(rng, sigma):
            Ya = DELTA + fn((n, T))                    # floor attacker
            Yl = fn((n, T))                            # live, no offset
            for test, inc_a, inc_l in [
                ("Bernstein (lambda_B)", bern_inc(Ya, lam_b, sigma), bern_inc(Yl, lam_b, sigma)),
                ("Bernstein (lambda=delta/sigma^2, old E9')", bern_inc(Ya, lam_g, sigma), bern_inc(Yl, lam_g, sigma)),
                ("Clamped Gaussian", DELTA * (DELTA / 2 - np.maximum(0, Ya)) / sigma ** 2,
                                     DELTA * (DELTA / 2 - np.maximum(0, Yl)) / sigma ** 2),
            ]:
                acc_a, r_a = run_sequential(inc_a)
                acc_l, r_l = run_sequential(inc_l)
                k = int(acc_a.sum()); lo, hi = cp_ci(k, n)
                kb = int((~acc_l).sum()); blo, bhi = cp_ci(kb, n)
                rows.append(dict(sigma=sigma, T_max=T, noise=name, test=test,
                                 APCER=k / n, APCER_over_alpha=k / n / ALPHA, apcer_ci_lo=lo, apcer_ci_hi=hi,
                                 BPCER=kb / n, bpcer_ci_lo=blo, bpcer_ci_hi=bhi,
                                 live_mean_rounds=float(r_l.mean()), live_p95_rounds=int(np.percentile(r_l, 95))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "r1_bernstein.csv"), index=False)
    return df


def r2_twopoint_check():
    """Hoeffding two-point bound H(m) for X<=c, E X = m <= 0, Var <= v; check H(m) <= H(0) <= Bennett."""
    worst = -1e9
    rows = []
    for c, sig in [(50.0, 25.0), (50.0, 11.756), (50.0, 50.0), (25.0, 12.0), (80.0, 25.0)]:
        v = sig ** 2
        for lam in np.linspace(1e-4, min(2 * c / v, 6.0 / c), 60):
            ms = -np.linspace(0, 20 * c, 4001)
            b = c - ms
            H = np.exp(lam * ms) * (v / (b ** 2 + v) * np.exp(lam * b) + b ** 2 / (b ** 2 + v) * np.exp(-lam * v / b))
            bennett = math.exp(v / c ** 2 * (math.exp(lam * c) - 1 - lam * c))
            mono = float(np.max(np.diff(H)))            # H decreasing as m decreases -> diffs <= 0
            gap = float(H[0] - bennett)
            worst = max(worst, mono, gap)
            rows.append(dict(c=c, sigma=sig, lam=lam, max_increase_as_m_decreases=mono, H0_minus_bennett=gap))
    return pd.DataFrame(rows), worst


def r3_tails(n, out):
    rows = []
    for sigma in [25.0, 11.756]:
        for tau in [0.0, 10.0]:
            rng = np.random.default_rng([SEED, 903, int(sigma * 1000), int(tau)])
            T = 20
            Y = rng.normal(0, tau, (n, 1)) + rng.normal(0, sigma, (n, T)) if tau > 0 else rng.normal(0, sigma, (n, T))
            acc, r = run_sequential(DELTA * (DELTA / 2 - Y) / sigma ** 2)
            z = stats.norm.isf(ALPHA) + stats.norm.isf(BETA)
            M = max(1, math.ceil((z * sigma / DELTA) ** 2))
            rows.append(dict(sigma=sigma, tau_live=tau, M_fixed=M, mean_rounds=float(r.mean()),
                             p50=int(np.percentile(r, 50)), p95=int(np.percentile(r, 95)), p99=int(np.percentile(r, 99)),
                             frac_exceeding_M=float((r > M).mean()), BPCER=float((~acc).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "r3_round_tails.csv"), index=False)
    return df


def r4_plugin_calibrated(n, out):
    rows = []
    sigma, T = 11.756, 20
    s2 = sigma ** 2
    for tau in [0.0, 5.0, 10.0, 15.0, 20.0]:
        rng = np.random.default_rng([SEED, 904, int(tau)])
        c = rng.normal(0, tau, (n, 1)) if tau > 0 else np.zeros((n, 1))
        for pop, base in [("live", c), ("floor", DELTA)]:
            Y = base + rng.normal(0, sigma, (n, T))
            inc_f = DELTA * (DELTA / 2 - Y) / s2
            inc_p = np.empty_like(Y)
            lam1 = DELTA / s2
            inc_p[:, 0] = lam1 * (DELTA - Y[:, 0]) - 0.5 * lam1 ** 2 * s2
            cum = Y[:, 0].copy()
            for t in range(1, T):
                lam = np.clip((DELTA - cum / t) / s2, 0.0, 2 * DELTA / s2)
                inc_p[:, t] = lam * (DELTA - Y[:, t]) - 0.5 * lam ** 2 * s2
                cum += Y[:, t]
            for test, inc in [("fixed lambda", inc_f), ("plug-in lambda", inc_p)]:
                acc, r = run_sequential(inc)
                rate = float(acc.mean())
                rows.append(dict(tau_live=tau, pop=pop, test=test,
                                 BPCER=(1 - rate) if pop == "live" else np.nan,
                                 APCER_over_alpha=(rate / ALPHA) if pop == "floor" else np.nan,
                                 mean_rounds=float(r.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "r4_plugin_calibrated.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--out", default="redteam_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4g}".format)
    d1 = r1_bernstein(a.n, a.out) if not os.environ.get("SKIP_R1") else pd.read_csv(os.path.join(a.out, "r1_bernstein.csv"))
    print("R1 Bernstein\n", d1[["sigma", "T_max", "noise", "test", "APCER_over_alpha", "BPCER", "live_mean_rounds", "live_p95_rounds"]].to_string(index=False))
    d2, worst = r2_twopoint_check()
    d2.to_csv(os.path.join(a.out, "r2_twopoint_check.csv"), index=False)
    print(f"\nR2 two-point check: worst violation = {worst:.3e} (must be <= 0 up to rounding)")
    print("\nR3 tails\n", r3_tails(a.n, a.out).to_string(index=False))
    print("\nR4 plug-in at sigma_hat\n", r4_plugin_calibrated(a.n, a.out).to_string(index=False))


if __name__ == "__main__":
    main()
