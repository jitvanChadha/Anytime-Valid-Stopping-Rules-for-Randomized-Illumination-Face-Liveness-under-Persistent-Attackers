#!/usr/bin/env python3
"""
sim_extensions.py -- Implementation of experiments E7, E8, E9, E10.

E7: Optional continuation (peeking on borderline sessions inflates fixed-M APCER; E-floor stays valid).
E8: AR(1) and adaptive/reactive attackers above delta (E-floor APCER <= alpha in every cell).
E9: Heavy-tailed measurement lag noise (Student-t, Laplace, contaminated Gaussian).
E10: High frame rate timing (60 fps, reduced sigma, sub-50 ms verification).
"""
import argparse, math, os, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

@dataclass(frozen=True)
class Cfg:
    sigma: float = 25.0
    delta: float = 50.0
    lazy: float = 150.0
    pi_opt: float = 0.10
    tau_live: float = 10.0
    T: int = 20
    alpha: float = 1e-3
    beta: float = 1e-2

def fixed_M(alpha, sigma, delta, beta_target=1e-2):
    z = stats.norm.isf(alpha) + stats.norm.isf(beta_target)
    return max(1, math.ceil((z * sigma / delta) ** 2))

def sequential_efloor(Y, delta, sigma, alpha, beta):
    s2 = sigma ** 2
    inc = delta * (delta / 2.0 - Y) / s2
    L = np.cumsum(inc, axis=1)
    up, lo = math.log(1.0 / alpha), math.log(beta)
    hu, hl = L >= up, L <= lo
    never = 10 ** 9
    t_up = np.where(hu.any(1), hu.argmax(1), never)
    t_lo = np.where(hl.any(1), hl.argmax(1), never)
    accept = t_up < t_lo
    rounds = np.minimum(np.minimum(t_up, t_lo), L.shape[1] - 1) + 1
    return accept, rounds

def sequential_sprt_iid(Y, delta, lazy, pi_opt, sigma, alpha, beta):
    s2 = 2.0 * sigma ** 2
    a = -(Y - delta) ** 2 / s2 + math.log(pi_opt)
    b = -(Y - lazy) ** 2 / s2 + math.log(1.0 - pi_opt)
    LL = -(Y ** 2) / s2 - np.logaddexp(a, b)
    L = np.cumsum(LL, axis=1)
    up, lo = math.log(1.0 / alpha), math.log(beta)
    hu, hl = L >= up, L <= lo
    never = 10 ** 9
    t_up = np.where(hu.any(1), hu.argmax(1), never)
    t_lo = np.where(hl.any(1), hl.argmax(1), never)
    accept = t_up < t_lo
    rounds = np.minimum(np.minimum(t_up, t_lo), L.shape[1] - 1) + 1
    return accept, rounds

# ------------------------------------------------------------------ E7: Optional Continuation
def run_e7(n, seed, cfg):
    rng = np.random.default_rng([seed, 7])
    sigma, delta, alpha, beta = cfg.sigma, cfg.delta, cfg.alpha, cfg.beta
    M1 = fixed_M(alpha, sigma, delta, beta)  # 8
    M2 = 2 * M1                              # 16
    z_alpha = stats.norm.isf(alpha)
    c1 = delta - z_alpha * sigma / math.sqrt(M1)
    c2 = delta - z_alpha * sigma / math.sqrt(M2)

    # Floor attacker D = delta
    eps = rng.normal(0.0, sigma, (n, M2))
    Y = delta + eps

    ybar1 = Y[:, :M1].mean(axis=1)
    ybar2 = Y.mean(axis=1)

    acc_f_std = ybar1 <= c1
    acc_e_floor, r_e_floor = sequential_efloor(Y, delta, sigma, alpha, beta)

    rows = []
    # Standard benchmark (no peeking)
    rows.append({
        "peeking_window_se": 0.0,
        "test": "F-floor (fixed M=8)",
        "APCER": float(acc_f_std.mean()),
        "APCER_over_alpha": float(acc_f_std.mean() / alpha),
        "mean_rounds": float(M1)
    })
    rows.append({
        "peeking_window_se": 0.0,
        "test": "E-floor (sequential)",
        "APCER": float(acc_e_floor.mean()),
        "APCER_over_alpha": float(acc_e_floor.mean() / alpha),
        "mean_rounds": float(r_e_floor.mean())
    })

    # Optional continuation sweeps
    se1 = sigma / math.sqrt(M1)
    for k_se in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        w = k_se * se1
        is_borderline = (ybar1 > c1) & (ybar1 <= c1 + w)
        acc_peek = acc_f_std | (is_borderline & (ybar2 <= c2))
        avg_rounds = float(np.where(is_borderline, M2, M1).mean())
        rows.append({
            "peeking_window_se": k_se,
            "test": f"F-floor-peek (w={k_se}*se)",
            "APCER": float(acc_peek.mean()),
            "APCER_over_alpha": float(acc_peek.mean() / alpha),
            "mean_rounds": avg_rounds
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ E8: AR(1) & Adaptive Attacker
def run_e8(n, seed, cfg):
    rng = np.random.default_rng([seed, 8])
    sigma, delta, alpha, beta, T = cfg.sigma, cfg.delta, cfg.alpha, cfg.beta, cfg.T
    M = fixed_M(alpha, sigma, delta, beta)
    c_floor = delta - stats.norm.isf(alpha) * sigma / math.sqrt(M)

    rows = []

    def eval_delays(D_matrix, scenario_name, phi_val=0.0):
        eps = rng.normal(0.0, sigma, (n, T))
        Y = D_matrix + eps
        acc_efloor, r_efloor = sequential_efloor(Y, delta, sigma, alpha, beta)
        ybar = Y[:, :M].mean(axis=1)
        acc_ffloor = ybar <= c_floor
        rows.append({
            "scenario": scenario_name,
            "phi": phi_val,
            "test": "E-floor",
            "APCER": float(acc_efloor.mean()),
            "APCER_over_alpha": float(acc_efloor.mean() / alpha),
            "mean_rounds": float(r_efloor.mean())
        })
        rows.append({
            "scenario": scenario_name,
            "phi": phi_val,
            "test": "F-floor",
            "APCER": float(acc_ffloor.mean()),
            "APCER_over_alpha": float(acc_ffloor.mean() / alpha),
            "mean_rounds": float(M)
        })

    # 1. Baseline floor attacker
    D_base = np.full((n, T), delta)
    eval_delays(D_base, "Floor attacker (D=delta)", 0.0)

    # 2. AR(1) process with excess latency
    sigma_ar = 10.0
    for phi in [0.0, 0.5, 0.8, 0.95]:
        for delta_excess in [0.0, 5.0, 15.0]:
            X = np.empty((n, T))
            X[:, 0] = rng.normal(0.0, sigma_ar, n)
            for t in range(1, T):
                innov = rng.normal(0.0, sigma_ar, n)
                X[:, t] = phi * X[:, t - 1] + math.sqrt(1.0 - phi ** 2) * innov
            D_ar = np.maximum(delta, delta + delta_excess + X)
            eval_delays(D_ar, f"AR(1) excess={delta_excess}ms", phi)

    # 3. Reactive / Adaptive Attacker
    D_adapt = np.full((n, T), delta)
    for t in range(1, T):
        # Attacker tries to react: if previous observed lag was high, push to delta; else add slack
        prev_lag = D_adapt[:, t - 1] + rng.normal(0.0, sigma, n)
        slack = np.where(prev_lag < delta, 0.0, 10.0)
        D_adapt[:, t] = delta + slack
    eval_delays(D_adapt, "Reactive/Adaptive feedback", 0.0)

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ E9: Heavy-Tailed Lag Noise
def run_e9(n, seed, cfg):
    rng = np.random.default_rng([seed, 9])
    sigma, delta, alpha, beta, T = cfg.sigma, cfg.delta, cfg.alpha, cfg.beta, cfg.T
    M = fixed_M(alpha, sigma, delta, beta)
    c_floor = delta - stats.norm.isf(alpha) * sigma / math.sqrt(M)

    noise_types = [
        ("Gaussian (baseline)", lambda size: rng.normal(0.0, sigma, size)),
        ("Student-t (df=10)", lambda size: rng.standard_t(10, size) * (sigma * math.sqrt(8.0 / 10.0))),
        ("Student-t (df=5)", lambda size: rng.standard_t(5, size) * (sigma * math.sqrt(3.0 / 5.0))),
        ("Student-t (df=3)", lambda size: rng.standard_t(3, size) * (sigma * math.sqrt(1.0 / 3.0))),
        ("Laplace", lambda size: rng.laplace(0.0, sigma / math.sqrt(2.0), size)),
        ("Contaminated Gaussian (10% outliers at 3x)", lambda size: (
            np.where(rng.random(size) < 0.10, rng.normal(0.0, 3.0, size), rng.normal(0.0, 1.0, size))
            * (sigma / math.sqrt(0.9 + 0.1 * 9.0))
        ))
    ]

    rows = []
    for name, noise_gen in noise_types:
        # 1. Floor Attacker (evaluate APCER)
        eps_att = noise_gen((n, T))
        Y_att = delta + eps_att
        acc_efloor, r_efloor = sequential_efloor(Y_att, delta, sigma, alpha, beta)
        # Physically clamped e-process (Y >= 0 constraint)
        acc_efloor_clamped, r_efloor_clamped = sequential_efloor(np.maximum(0.0, Y_att), delta, sigma, alpha, beta)
        ybar_att = Y_att[:, :M].mean(axis=1)
        acc_ffloor = ybar_att <= c_floor

        rows.append({
            "noise_model": name,
            "population": "floor_attacker",
            "test": "E-floor (raw)",
            "error_rate": float(acc_efloor.mean()),
            "metric": "APCER",
            "over_alpha": float(acc_efloor.mean() / alpha),
            "mean_rounds": float(r_efloor.mean())
        })
        rows.append({
            "noise_model": name,
            "population": "floor_attacker",
            "test": "E-floor (clamped Y>=0)",
            "error_rate": float(acc_efloor_clamped.mean()),
            "metric": "APCER",
            "over_alpha": float(acc_efloor_clamped.mean() / alpha),
            "mean_rounds": float(r_efloor_clamped.mean())
        })
        rows.append({
            "noise_model": name,
            "population": "floor_attacker",
            "test": "F-floor",
            "error_rate": float(acc_ffloor.mean()),
            "metric": "APCER",
            "over_alpha": float(acc_ffloor.mean() / alpha),
            "mean_rounds": float(M)
        })

        # 2. Live Users (evaluate BPCER)
        eps_live = noise_gen((n, T))
        c_live = rng.normal(0.0, cfg.tau_live, (n, 1))
        Y_live = c_live + eps_live
        acc_live_e, r_live_e = sequential_efloor(Y_live, delta, sigma, alpha, beta)
        ybar_live = Y_live[:, :M].mean(axis=1)
        acc_live_f = ybar_live <= c_floor

        rows.append({
            "noise_model": name,
            "population": "live_users",
            "test": "E-floor",
            "error_rate": float(1.0 - acc_live_e.mean()),
            "metric": "BPCER",
            "over_alpha": np.nan,
            "mean_rounds": float(r_live_e.mean())
        })
        rows.append({
            "noise_model": name,
            "population": "live_users",
            "test": "F-floor",
            "error_rate": float(1.0 - acc_live_f.mean()),
            "metric": "BPCER",
            "over_alpha": np.nan,
            "mean_rounds": float(M)
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ E10: High Frame-Rate (60 fps)
def run_e10(n, seed, cfg):
    rng = np.random.default_rng([seed, 10])
    alpha, beta, T = cfg.alpha, cfg.beta, 40
    frame_ms = 1000.0 / 60.0  # 16.67 ms per frame

    rows = []
    # Test combinations of reduced sigma and various delta floors
    for sigma_fps in [12.0, 15.0, 20.0]:
        for delta_fps in [25.0, 35.0, 50.0]:
            M = fixed_M(alpha, sigma_fps, delta_fps, beta)
            c_floor = delta_fps - stats.norm.isf(alpha) * sigma_fps / math.sqrt(M)

            # 1. Floor attacker
            eps_att = rng.normal(0.0, sigma_fps, (n, T))
            Y_att = delta_fps + eps_att
            acc_e_att, r_e_att = sequential_efloor(Y_att, delta_fps, sigma_fps, alpha, beta)
            ybar_att = Y_att[:, :M].mean(axis=1)
            acc_f_att = ybar_att <= c_floor

            # 2. Live users (tau_live = 5 ms at 60 fps calibrated webcam)
            tau_live_fps = 5.0
            c_live = rng.normal(0.0, tau_live_fps, (n, 1))
            eps_live = rng.normal(0.0, sigma_fps, (n, T))
            Y_live = c_live + eps_live
            acc_e_live, r_e_live = sequential_efloor(Y_live, delta_fps, sigma_fps, alpha, beta)
            ybar_live = Y_live[:, :M].mean(axis=1)
            acc_f_live = ybar_live <= c_floor

            asn_e = float(r_e_live.mean())
            wallclock_e_ms = asn_e * frame_ms
            wallclock_f_ms = M * frame_ms

            rows.append({
                "sigma": sigma_fps,
                "delta": delta_fps,
                "delta_over_sigma": delta_fps / sigma_fps,
                "test": "E-floor",
                "M_or_ASN": asn_e,
                "wallclock_ms": wallclock_e_ms,
                "APCER_over_alpha": float(acc_e_att.mean() / alpha),
                "BPCER": float(1.0 - acc_e_live.mean())
            })
            rows.append({
                "sigma": sigma_fps,
                "delta": delta_fps,
                "delta_over_sigma": delta_fps / sigma_fps,
                "test": "F-floor",
                "M_or_ASN": float(M),
                "wallclock_ms": wallclock_f_ms,
                "APCER_over_alpha": float(acc_f_att.mean() / alpha),
                "BPCER": float(1.0 - acc_f_live.mean())
            })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Simulate E7-E10 extensions.")
    parser.add_argument("--n", type=int, default=1_000_000, help="Sessions per cell")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--stages", default="7,8,9,10")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stages = {int(x) for x in args.stages.split(",") if x}
    cfg = Cfg()

    print(f"Running sim_extensions with n={args.n}, seed={args.seed}, stages={stages}")

    if 7 in stages:
        t0 = time.time()
        print("Running E7 (Optional Continuation / Peeking)...", flush=True)
        df7 = run_e7(args.n, args.seed, cfg)
        p7 = os.path.join(args.out, "e7_peeking.csv")
        df7.to_csv(p7, index=False)
        print(f"E7 done ({time.time()-t0:.1f}s):\n", df7.to_string(index=False), "\n")

    if 8 in stages:
        t0 = time.time()
        print("Running E8 (AR(1) and Adaptive Attacker)...", flush=True)
        df8 = run_e8(args.n, args.seed, cfg)
        p8 = os.path.join(args.out, "e8_adaptive_attacker.csv")
        df8.to_csv(p8, index=False)
        print(f"E8 done ({time.time()-t0:.1f}s):\n", df8.head(10).to_string(index=False), "\n")

    if 9 in stages:
        t0 = time.time()
        print("Running E9 (Heavy-Tailed Lag Noise)...", flush=True)
        df9 = run_e9(args.n, args.seed, cfg)
        p9 = os.path.join(args.out, "e9_heavy_tails.csv")
        df9.to_csv(p9, index=False)
        print(f"E9 done ({time.time()-t0:.1f}s):\n", df9.to_string(index=False), "\n")

    if 10 in stages:
        t0 = time.time()
        print("Running E10 (High Frame-Rate 60 fps)...", flush=True)
        df10 = run_e10(args.n, args.seed, cfg)
        p10 = os.path.join(args.out, "e10_60fps.csv")
        df10.to_csv(p10, index=False)
        print(f"E10 done ({time.time()-t0:.1f}s):\n", df10.to_string(index=False), "\n")

if __name__ == "__main__":
    main()
