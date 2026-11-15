#!/usr/bin/env python3
"""
Anytime-valid stopping for randomized-illumination liveness -- falsification simulation.

Question: do tests calibrated to an i.i.d. attacker keep their nominal APCER when
attacker latency is persistent across rounds, and does a floor-calibrated e-process
keep it at modest extra cost in rounds?

Model (per round t of a randomized flash challenge)
  Y_t = measured response lag (ms) = D_t + eps_t,  eps_t ~ N(0, sigma^2)
  live      : D_t = c_s            (session calibration offset c_s ~ N(0, tau_live^2))
  attacker  : D_t >= delta         (hardware latency floor; the composite null)
  optimised attackers sit at delta, unoptimised ones at `lazy`; share pi_opt optimised.
  rho in [0,1] = persistence of the attacker type across rounds (0 = i.i.d., 1 = fixed per session).

Tests (all accept = "live")
  F-iid    : mean lag over M rounds, threshold calibrated exactly for the i.i.d. mixture attacker
  F-floor  : mean lag over M rounds, threshold calibrated for the floor attacker (D_t = delta)
  SPRT-iid : Wald SPRT, live vs the i.i.d. mixture marginal  (valid only for i.i.d. rounds)
  E-floor  : likelihood-ratio e-process, live vs the floor attacker N(delta, sigma^2);
             a test supermartingale for EVERY attacker with D_t >= delta, however adaptive/correlated
Both sequential tests accept at LR >= 1/alpha, reject at LR <= beta, and reject at T_max.

Usage: python3 sim.py [--n 2000000] [--out DIR] [--sigma 25] [--delta 50] [--tau-live 10] [--stages 1,2,3,4,5]
Tip: use a fresh --out per parameter set; each stage writes its own CSV there.
"""
import argparse, math, os, time
import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from scipy import stats, optimize

@dataclass(frozen=True)
class Cfg:
    sigma: float = 25.0     # per-round lag measurement noise (ms)
    delta: float = 50.0     # attacker latency floor (ms)
    lazy: float = 150.0     # latency of unoptimised attackers (ms)
    pi_opt: float = 0.10    # share of optimised attackers
    tau_live: float = 10.0  # spread of live-user session offsets (ms)
    T: int = 20             # max rounds per session
    beta: float = 1e-2      # lower (reject) boundary of the sequential tests

# ------------------------------------------------------------------ data generation
def sim_lags(rng, n, cfg, pop, rho=1.0):
    T, s = cfg.T, cfg.sigma
    eps = rng.normal(0.0, s, (n, T))
    if pop == "live":
        return rng.normal(0.0, cfg.tau_live, (n, 1)) + eps
    if pop == "floor":                       # best-case attacker, worst case for the e-process
        return cfg.delta + eps
    if pop == "lazy":
        return cfg.lazy + eps
    if pop == "mix":                         # optimised/lazy mixture with persistence rho
        typ = np.empty((n, T), dtype=bool)
        typ[:, 0] = rng.random(n) < cfg.pi_opt
        for t in range(1, T):
            keep = rng.random(n) < rho
            typ[:, t] = np.where(keep, typ[:, t - 1], rng.random(n) < cfg.pi_opt)
        return np.where(typ, cfg.delta, cfg.lazy) + eps
    raise ValueError(pop)

# ------------------------------------------------------------------ per-round log-likelihood ratios (live : attacker)
def ll_floor(y, cfg):
    return cfg.delta * (cfg.delta / 2.0 - y) / cfg.sigma ** 2

def ll_iid_mix(y, cfg):
    s2 = 2.0 * cfg.sigma ** 2
    a = -(y - cfg.delta) ** 2 / s2 + math.log(cfg.pi_opt)
    b = -(y - cfg.lazy) ** 2 / s2 + math.log(1.0 - cfg.pi_opt)
    return -(y ** 2) / s2 - np.logaddexp(a, b)

# ------------------------------------------------------------------ decision rules
def sequential(LL, alpha, beta):
    L = np.cumsum(LL, axis=1)
    up, lo = math.log(1.0 / alpha), math.log(beta)
    hu, hl = L >= up, L <= lo
    never = 10 ** 9
    t_up = np.where(hu.any(1), hu.argmax(1), never)
    t_lo = np.where(hl.any(1), hl.argmax(1), never)
    accept = t_up < t_lo
    rounds = np.minimum(np.minimum(t_up, t_lo), L.shape[1] - 1) + 1
    return accept, rounds

def fixed_threshold_iid(M, alpha, cfg):
    """Exact alpha-quantile of the M-round mean lag when every round draws its type i.i.d."""
    k = np.arange(M + 1)
    w = stats.binom.pmf(k, M, 1.0 - cfg.pi_opt)             # k = number of lazy rounds
    m = (k * cfg.lazy + (M - k) * cfg.delta) / M
    f = lambda c: float(np.sum(w * stats.norm.cdf((c - m) * math.sqrt(M) / cfg.sigma))) - alpha
    return optimize.brentq(f, cfg.delta - 8 * cfg.sigma, cfg.lazy)

def fixed_threshold_floor(M, alpha, cfg):
    return cfg.delta - stats.norm.isf(alpha) * cfg.sigma / math.sqrt(M)

def fixed_M(alpha, cfg, beta_target=1e-2):
    z = stats.norm.isf(alpha) + stats.norm.isf(beta_target)
    return max(1, math.ceil((z * cfg.sigma / cfg.delta) ** 2))

def decide(Y, cfg, alpha, M):
    out = {}
    ybar = Y[:, :M].mean(1)
    out["F-iid"] = (ybar <= fixed_threshold_iid(M, alpha, cfg), np.full(len(Y), M))
    out["F-floor"] = (ybar <= fixed_threshold_floor(M, alpha, cfg), np.full(len(Y), M))
    out["SPRT-iid"] = sequential(ll_iid_mix(Y, cfg), alpha, cfg.beta)
    out["E-floor"] = sequential(ll_floor(Y, cfg), alpha, cfg.beta)
    return out

# ------------------------------------------------------------------ evaluation harness
def cp_ci(k, n, level=0.95):
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi

def evaluate(cfg, pops, alphas, n, rng, rho=1.0, chunk=100_000):
    acc = {}
    Ms = {a: fixed_M(a, cfg) for a in alphas}
    for pop in pops:
        for done in range(0, n, chunk):
            m = min(chunk, n - done)
            Y = sim_lags(rng, m, cfg, pop, rho)
            for a in alphas:
                for test, (ok, r) in decide(Y, cfg, a, Ms[a]).items():
                    key = (pop, a, test)
                    rec = acc.setdefault(key, {"k": 0, "n": 0, "hist": np.zeros(cfg.T + 1, dtype=np.int64)})
                    rec["k"] += int(ok.sum()); rec["n"] += m
                    rec["hist"] += np.bincount(r, minlength=cfg.T + 1)[: cfg.T + 1]
    rows = []
    for (pop, a, test), rec in acc.items():
        h, nn = rec["hist"], rec["n"]
        asn = float((np.arange(cfg.T + 1) * h).sum() / nn)
        p95 = int(np.searchsorted(np.cumsum(h) / nn, 0.95))
        lo, hi = cp_ci(rec["k"], nn)
        rows.append(dict(pop=pop, alpha=a, test=test, M_fixed=Ms[a], n=nn, accept_rate=rec["k"] / nn,
                         ci_lo=lo, ci_hi=hi, asn=asn, p95_rounds=p95))
    return pd.DataFrame(rows)

def summarise(df):
    """Turn accept rates into APCER / BPCER and an inflation factor."""
    d = df.copy()
    is_live = d["pop"] == "live"
    d["APCER"] = np.where(is_live, np.nan, d.accept_rate)
    d["BPCER"] = np.where(is_live, 1 - d.accept_rate, np.nan)
    d["APCER_over_alpha"] = d.APCER / d.alpha
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--out", default="/mnt/user-data/outputs")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--sigma", type=float, default=25.0, help="per-round lag noise sd (ms), e.g. from fraud1_jitter.py")
    ap.add_argument("--delta", type=float, default=50.0, help="assumed attacker latency floor (ms)")
    ap.add_argument("--tau-live", type=float, default=10.0, help="live-user session offset sd (ms)")
    ap.add_argument("--stages", default="1,2,3,4,5",
                    help="experiments to (re)run; the others are read back from their CSV")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stages = {int(x) for x in args.stages.split(",") if x}
    cfg, half, t0 = Cfg(sigma=args.sigma, delta=args.delta, tau_live=args.tau_live), max(args.n // 2, 200_000), time.time()

    def stage(k, name, build):
        """Each experiment has its own RNG stream, so stages can be re-run independently."""
        f = f"{args.out}/{name}.csv"
        if k in stages:
            df = build(np.random.default_rng([args.seed, k]))
            df.to_csv(f, index=False)
            print(f"[E{k} done {time.time()-t0:.0f}s]", flush=True)
            return df
        return pd.read_csv(f) if os.path.exists(f) else None

    def e1(rng):  # headline: persistent vs i.i.d. optimised attackers
        a = summarise(evaluate(cfg, ["live", "floor", "lazy", "mix"], [1e-2, 1e-3], args.n, rng, rho=1.0))
        a.loc[a["pop"] == "mix", "pop"] = "mix_persistent(rho=1)"
        b = summarise(evaluate(cfg, ["mix"], [1e-2, 1e-3], args.n, rng, rho=0.0))
        b["pop"] = "mix_iid(rho=0)"
        return pd.concat([a, b], ignore_index=True)

    def e2(rng):  # persistence sweep
        rows = []
        for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]:
            d = summarise(evaluate(cfg, ["mix"], [1e-3], half, rng, rho=rho)); d["rho"] = rho; rows.append(d)
        return pd.concat(rows, ignore_index=True)

    def e3(rng):  # latency-floor design curve (clean theory check: no live offset spread, long horizon)
        rows = []
        for delta in [25.0, 30.0, 40.0, 50.0, 60.0, 80.0]:
            c = replace(cfg, delta=delta, T=60, tau_live=0.0)
            d = summarise(evaluate(c, ["live", "floor"], [1e-3], half, rng)); d["delta"] = delta
            d["theory_rounds"] = 2 * c.sigma ** 2 * math.log(1 / 1e-3) / delta ** 2
            rows.append(d)
        return pd.concat(rows, ignore_index=True)

    def e4(rng):  # live calibration-offset sensitivity
        rows = []
        for tau in [0.0, 5.0, 10.0, 15.0]:
            d = summarise(evaluate(replace(cfg, tau_live=tau), ["live"], [1e-3], half, rng)); d["tau_live"] = tau; rows.append(d)
        return pd.concat(rows, ignore_index=True)

    def e5(rng):  # what if the assumed floor is wrong? constant-latency attacker with D below/above delta
        rows = []
        for D in [0.0, 20.0, 35.0, 45.0, 50.0, 55.0, 65.0, 80.0, 100.0]:
            d = summarise(evaluate(replace(cfg, lazy=D), ["lazy"], [1e-3], max(args.n // 4, 200_000), rng))
            d = d[d.test.isin(["E-floor", "F-floor"])].copy(); d["attacker_latency"] = D
            rows.append(d)
        return pd.concat(rows, ignore_index=True)

    r1 = stage(1, "e1_headline", e1)
    r2 = stage(2, "e2_persistence", e2)
    r3 = stage(3, "e3_delta_sweep", e3)
    r4 = stage(4, "e4_live_offset", e4)
    r5 = stage(5, "e5_wrong_floor", e5)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4g}".format)

    def show(title, df, cols, sort):
        if df is not None:
            print("\n" + title)
            print(df[cols].sort_values(sort).to_string(index=False))

    c = ["pop", "alpha", "test", "M_fixed", "APCER", "BPCER", "APCER_over_alpha", "ci_lo", "ci_hi", "asn", "p95_rounds"]
    show(f"E1 headline (sigma={args.sigma}, delta={args.delta}, lazy=150, pi_opt=0.1, tau_live={args.tau_live}, T=20)", r1, c, ["alpha", "pop", "test"])
    show("E2 persistence sweep (alpha=1e-3)", r2, ["rho", "test", "APCER", "APCER_over_alpha", "asn"], ["test", "rho"])
    show("E3 floor sweep (alpha=1e-3, tau_live=0, T=60)", r3, ["delta", "pop", "test", "M_fixed", "APCER", "BPCER", "asn", "p95_rounds", "theory_rounds"], ["pop", "test", "delta"])
    show("E4 live offset sensitivity (alpha=1e-3, T=20)", r4, ["tau_live", "test", "BPCER", "asn", "p95_rounds"], ["test", "tau_live"])
    show("E5 wrong-floor stress test (design delta=50, alpha=1e-3, constant attacker latency D)", r5, ["attacker_latency", "test", "APCER", "APCER_over_alpha", "asn"], ["test", "attacker_latency"])

if __name__ == "__main__":
    main()
