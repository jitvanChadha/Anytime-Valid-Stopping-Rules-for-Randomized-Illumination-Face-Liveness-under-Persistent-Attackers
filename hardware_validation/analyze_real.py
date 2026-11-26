#!/usr/bin/env python3
"""
analyze_real.py -- hardware-validation analysis for the WcAISR-2027 revision.

Input: lags.csv from extract_lags.py (one row per recorded round), optional attack_*.csv
processing logs from attack_vcam.py, and an RTT trace for the network-delay emulation.

  python analyze_real.py lags.csv --out ../revision_out/real --rtt ../revision_out/rtt_wifi_1.1.1.1.txt
         --attack-logs "sessions/attack_*.csv"
  python analyze_real.py --selftest           # synthetic data only, checks the code path

Sections of the output (each written as CSV/JSON into --out):
  A  baseline and noise    device baseline b (A2), sigma_hat, tau_live per condition; assumption checks
                           (Shapiro-Wilk, excess kurtosis, lag-1 autocorrelation)
  B  latency floor         per-round attack lag above baseline: quantiles, bootstrap lower bound -> delta_hat
  C  recorded sessions     every decision rule applied to each recorded session (first T rounds):
                           APCER per attack type and BPCER for live, Clopper-Pearson CIs
  D  trace-driven          sessions resampled from the recorded ones (session identity kept, so the
                           offset persists), N per cell, including a worst-case attacker shifted to the floor
  E  network delay         RTT-trace jitter added to every round (server-side timing)
  F  overhead and latency  us per e-process update; decision time in rounds and seconds;
                           per-frame attack-pipeline cost from the logs
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim  # noqa: E402  (decision rules shared with the simulation study)

ALPHA, BETA, T = 1e-3, 1e-2, 20

# --------------------------------------------------------------------------- helpers
def cp(k, n):
    lo, hi = sim.cp_ci(int(k), int(n))
    return float(lo), float(hi)

def run_seq(inc):
    return sim.sequential(inc, ALPHA, BETA)

def plugin_inc(Y, delta, sig):
    s2 = sig ** 2
    inc = np.empty_like(Y)
    lam = delta / s2
    inc[:, 0] = lam * (delta - Y[:, 0]) - 0.5 * lam ** 2 * s2
    cum = Y[:, 0].copy()
    for t in range(1, Y.shape[1]):
        lam = np.clip((delta - cum / t) / s2, 0.0, 2 * delta / s2)
        inc[:, t] = lam * (delta - Y[:, t]) - 0.5 * lam ** 2 * s2
        cum += Y[:, t]
    return inc

def bern_inc(Y, delta, sig):
    lam = math.log(1 + delta ** 2 / sig ** 2) / delta
    psi = (sig ** 2 / delta ** 2) * (math.exp(lam * delta) - 1 - lam * delta)
    return lam * (delta - np.maximum(0.0, Y)) - psi

def rules(Y, delta, sig, colour_ok=None):
    """All decision rules on an (n, T) array of baseline-corrected lags. Returns {rule: (accept, rounds)}."""
    cfg = sim.Cfg(sigma=sig, delta=delta, lazy=3 * delta, pi_opt=0.1, T=Y.shape[1], beta=BETA)
    M = sim.fixed_M(ALPHA, cfg)
    n = len(Y)
    out = {}
    out["Single-round timing [5]"] = (Y[:, 0] <= sim.fixed_threshold_floor(1, ALPHA, cfg), np.ones(n, int))
    out.update(sim.decide(Y, cfg, ALPHA, M))                   # F-iid [6], F-floor, SPRT-iid [7], E-floor
    out["E-floor plug-in"] = run_seq(plugin_inc(Y, delta, sig))
    out["E-Bernstein"] = run_seq(bern_inc(Y, delta, sig))
    if colour_ok is not None:
        out["Colour check [5],[18]"] = (colour_ok, np.full(n, Y.shape[1]))
        acc, r = out["E-floor"]
        out["E-floor + colour check"] = (acc & colour_ok, r)
    return out, M

# --------------------------------------------------------------------------- A. baseline and noise
def noise_stats(df):
    """sigma = pooled within-session sd; tau = between-session sd of means, corrected for sigma."""
    g = df.groupby("session_id")["Y"]
    resid = (df["Y"] - g.transform("mean")).to_numpy()
    ns = g.size()
    dof = int((ns - 1).sum())
    sig = math.sqrt(float((resid ** 2).sum()) / dof) if dof > 0 else float("nan")
    means = g.mean()
    var_tau = float(means.var(ddof=1) - np.mean(sig ** 2 / ns)) if len(means) > 1 else float("nan")
    tau = math.sqrt(max(var_tau, 0.0)) if np.isfinite(var_tau) else float("nan")
    return sig, tau, resid

def boot_sigma(df, B=1000, seed=7):
    ids = df.session_id.unique()
    rng = np.random.default_rng(seed)
    parts = {s: d for s, d in df.groupby("session_id")}
    est = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids))
        d = pd.concat([parts[s].assign(session_id=f"{s}#{i}") for i, s in enumerate(pick)])
        est.append(noise_stats(d)[0])
    return tuple(np.percentile(est, [2.5, 97.5]))

def assumption_checks(resid, df):
    r = resid[np.isfinite(resid)]
    sub = np.random.default_rng(3).choice(r, min(len(r), 4999), replace=False) if len(r) > 3 else r
    sw = stats.shapiro(sub) if len(sub) >= 3 else (np.nan, np.nan)
    ac = []
    for _, d in df.sort_values("round").groupby("session_id"):
        y = d["Y"].to_numpy() - d["Y"].mean()
        if len(y) > 3 and y.var() > 0:
            ac.append(np.sum(y[1:] * y[:-1]) / np.sum(y * y))
    ac = np.array(ac)
    n_r = len(df) / max(df.session_id.nunique(), 1)
    return dict(n_residuals=int(len(r)), shapiro_W=float(sw[0]), shapiro_p=float(sw[1]),
                excess_kurtosis=float(stats.kurtosis(r)), skewness=float(stats.skew(r)),
                lag1_autocorr_mean=float(ac.mean()) if len(ac) else np.nan,
                lag1_autocorr_ci=[float(ac.mean() - 1.96 * ac.std(ddof=1) / math.sqrt(len(ac))),
                                  float(ac.mean() + 1.96 * ac.std(ddof=1) / math.sqrt(len(ac)))] if len(ac) > 2 else None,
                lag1_autocorr_null_mean=float(-1 / (n_r - 1)) if n_r > 1 else np.nan,
                max_abs_resid_over_sigma=float(np.max(np.abs(r)) / np.std(r)) if len(r) else np.nan)

# --------------------------------------------------------------------------- B. latency floor
def floor_estimate(Y_attack, B=2000, seed=11, q=0.01):
    rng = np.random.default_rng(seed)
    Y_attack = np.asarray(Y_attack, float)
    qs = [np.quantile(rng.choice(Y_attack, len(Y_attack)), q) for _ in range(B)]
    return dict(n_rounds=int(len(Y_attack)), median=float(np.median(Y_attack)), mean=float(np.mean(Y_attack)),
                sd=float(np.std(Y_attack, ddof=1)), q01=float(np.quantile(Y_attack, 0.01)),
                q05=float(np.quantile(Y_attack, 0.05)), min=float(np.min(Y_attack)),
                q01_lower95=float(np.percentile(qs, 5)))

# --------------------------------------------------------------------------- D. trace-driven resampling
def resample_sessions(df, n, rng, shift=0.0):
    """Pick a recorded session, then T rounds from it with replacement: the session offset persists."""
    groups = [d["Y"].to_numpy() for _, d in df.groupby("session_id")]
    pick = rng.integers(0, len(groups), n)
    Y = np.empty((n, T))
    for k, g in enumerate(groups):
        idx = np.where(pick == k)[0]
        if len(idx):
            Y[idx] = g[rng.integers(0, len(g), (len(idx), T))]
    return Y + shift

def colour_classifier(train):
    """Nearest-centroid colour classifier on the unit direction of the ROI colour step."""
    X = train[["dir_b", "dir_g", "dir_r"]].to_numpy(float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cents = {c: X[train.color_idx.to_numpy() == c].mean(0) for c in sorted(train.color_idx.unique())}
    cents = {c: v / np.linalg.norm(v) for c, v in cents.items()}
    def predict(d):
        x = d[["dir_b", "dir_g", "dir_r"]].to_numpy(float)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        keys = list(cents)
        S = x @ np.stack([cents[k] for k in keys]).T
        return np.array(keys)[S.argmax(1)]
    return predict

def session_matrix(df, n_rounds=None):
    """(sessions, T) matrix of the first T QC-passed rounds; sessions with fewer rounds are dropped."""
    n_rounds = n_rounds or T
    rows, ids, colour = [], [], []
    for sid, d in df.sort_values("round").groupby("session_id"):
        y = d["Y"].to_numpy()[:n_rounds]
        if len(y) == n_rounds:
            rows.append(y); ids.append(sid)
            colour.append(bool(d["colour_match"].to_numpy()[:n_rounds].mean() >= 0.9))
    return np.array(rows), ids, np.array(colour, bool)

def evaluate_block(Ysets, delta, sig, colour=None):
    out = []
    for label, Y in Ysets.items():
        if len(Y) == 0:
            continue
        res, M = rules(Y, delta, sig, None if colour is None else colour.get(label))
        for rule, (acc, r) in res.items():
            k, n = int(acc.sum()), len(acc)
            lo, hi = cp(k, n)
            out.append(dict(population=label, rule=rule, n=n, accept=k, accept_rate=k / n, ci_lo=lo, ci_hi=hi,
                            mean_rounds=float(np.mean(r)), p95_rounds=float(np.percentile(r, 95)), M_fixed=M))
    return pd.DataFrame(out)

# --------------------------------------------------------------------------- F. overhead
def overhead_us(reps=200_000):
    delta, sig = 50.0, 12.0
    lam, c = delta / sig ** 2, delta ** 2 / (2 * sig ** 2)
    ys = np.random.default_rng(0).normal(0, sig, reps).tolist()
    logE, lo, up = 0.0, math.log(BETA), math.log(1 / ALPHA)
    t0 = time.perf_counter()
    for y in ys:
        logE += lam * (delta / 2 - y)          # one round of E-floor in log form
        if logE >= up or logE <= lo:
            logE = 0.0
    return (time.perf_counter() - t0) / reps * 1e6, c

# --------------------------------------------------------------------------- main
def analyse(lags, out, rtt=None, attack_logs=None, n_trace=200_000, seed=20261125, round_ms=None, skip_rounds=2, delta_override=None):
    """skip_rounds: the first rounds of every session are warm-up (camera exposure and the subject settle);
    fixed at 2 on 25 Nov 2026 after the first 10 live sessions, before any attack session was recorded,
    and applied identically to every label. Run with --skip-rounds 0 for the all-rounds sensitivity check."""
    global T
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = pd.read_csv(lags)
    T = int(df["round"].max()) - skip_rounds
    df = df[df["round"] > skip_rounds]
    df = df[df.qc_ok].copy()
    df["Y_raw"] = df["lag_on_ms"]
    summary = {}

    # A. device baseline (A2): median live lag in normal light on the PHYSICAL camera, per (fps, resolution).
    # Every session at that setting -- live in other conditions, and attacks arriving through a virtual
    # camera that claims to be this device -- is measured against it, as a verifier would.
    if "res" not in df:
        df["res"] = "na"
    calib = df[(df.label == "live") & (df.condition == "normal")]
    if calib.empty:
        calib = df[df.label == "live"]
    keys = ["fps", "res"]
    base = calib.groupby(keys)["Y_raw"].median().rename("baseline_ms").reset_index()
    df = df.merge(base, on=keys, how="left")
    fb = calib.groupby("fps")["Y_raw"].median()
    df["baseline_ms"] = df["baseline_ms"].fillna(df["fps"].map(fb))
    df = df[df["baseline_ms"].notna()].copy()
    df["Y"] = df["Y_raw"] - df["baseline_ms"]
    base.to_csv(os.path.join(out, "A_baseline.csv"), index=False)

    rows, checks = [], {}
    for (cond, cam, fps, res), d in df[df.label == "live"].groupby(["condition", "camera_name", "fps", "res"]):
        s, tau, resid = noise_stats(d)
        lo, hi = boot_sigma(d)
        rows.append(dict(condition=cond, camera=cam, fps=fps, res=res, sessions=d.session_id.nunique(), rounds=len(d),
                         sigma_ms=s, sigma_ci_lo=lo, sigma_ci_hi=hi, tau_live_ms=tau,
                         lag_on_median_raw_ms=float(d.Y_raw.median())))
        checks[f"{cond}|{cam}|{fps}|{res}"] = assumption_checks(resid, d)
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(out, "A_noise_by_condition.csv"), index=False)
    live = df[df.label == "live"]
    sig_all, tau_all, resid_all = noise_stats(live)
    sig_lo, sig_hi = boot_sigma(live)
    checks["ALL_LIVE"] = assumption_checks(resid_all, live)
    json.dump(checks, open(os.path.join(out, "A_assumption_checks.json"), "w"), indent=1)
    np.savetxt(os.path.join(out, "A_live_residuals.csv"), resid_all, delimiter=",", header="resid_ms", comments="")
    summary["sigma_hat_ms"], summary["sigma_ci"], summary["tau_live_ms"] = sig_all, [sig_lo, sig_hi], tau_all

    # B. latency floor per attack type
    B_rows = {}
    for lab, d in df[df.label != "live"].groupby("label"):
        B_rows[lab] = floor_estimate(d["Y"].to_numpy())
    json.dump(B_rows, open(os.path.join(out, "B_latency_floor.json"), "w"), indent=1)
    swap_key = "faceswap" if "faceswap" in B_rows else None
    delta_hat = B_rows[swap_key]["q01_lower95"] if swap_key else 50.0
    delta_design = float(delta_override) if delta_override else max(5.0, math.floor(delta_hat))
    sig_design = sig_hi                      # sigma_bar: upper confidence bound (Theorem 1 needs sigma_bar >= sigma)
    summary.update(delta_hat_ms=delta_hat, delta_design_ms=delta_design, sigma_design_ms=sig_design,
                   delta_over_sigma=delta_design / sig_design)

    # colour-consistency check: classifier trained on live sessions only
    pred = colour_classifier(live)
    df["colour_match"] = pred(df) == df["color_idx"].to_numpy()

    # C. recorded sessions
    Ysets, colour = {}, {}
    for lab, d in df.groupby("label"):
        Y, ids, col = session_matrix(d)
        Ysets[lab], colour[lab] = Y, col
    C = evaluate_block(Ysets, delta_design, sig_design, colour)
    C.to_csv(os.path.join(out, "C_recorded_sessions.csv"), index=False)

    # D. trace-driven: resample recorded sessions; add a worst-case attacker moved down to the floor
    Dsets = {lab: resample_sessions(d, n_trace, rng) for lab, d in df.groupby("label")}
    if swap_key:
        d = df[df.label == swap_key]
        # Worst case inside H_delta: move the recorded attack down until its fastest rounds (the same
        # conservative 1% quantile that defines delta_hat) sit at the design floor. Shifting the session
        # MEAN to delta instead would put many rounds below delta, i.e. an attacker outside H_delta.
        shift = delta_design - delta_hat
        Dsets["faceswap shifted to floor"] = resample_sessions(d, n_trace, rng, shift)
        summary["floor_shift_ms"] = shift
    # Trace-driven versions of the paper's synthetic attackers: the attacker latency is set by the model,
    # the measurement noise is drawn from the recorded within-session residuals of the live sessions
    # (real noise law, outliers included) instead of a Gaussian.
    lv = df[df.label == "live"]
    eps = (lv["Y"] - lv.groupby("session_id")["Y"].transform("mean")).to_numpy()
    E = rng.choice(eps, (n_trace, T))
    Dsets["floor attacker + recorded noise"] = delta_design + E
    typ = rng.random((n_trace, 1)) < 0.1                       # persistent: type fixed for the session
    Dsets["persistent mixture + recorded noise"] = np.where(typ, delta_design, 3 * delta_design) + rng.choice(eps, (n_trace, T))
    summary["noise_residual_sd_ms"] = float(eps.std())
    D = evaluate_block(Dsets, delta_design, sig_design)
    D.to_csv(os.path.join(out, "D_trace_driven.csv"), index=False)

    # E. network delay: server-side timing adds (RTT - median RTT) to each round
    if rtt and os.path.exists(rtt):
        r = pd.to_numeric(pd.Series(open(rtt, encoding="utf-8-sig").read().split()), errors="coerce").dropna().to_numpy(float)
        jit = r - np.median(r)
        E_rows = []
        for scale in [0.0, 1.0, 2.0, 4.0]:
            extra = lambda shape: scale * rng.choice(jit, shape)
            Lv = Dsets["live"] + extra(Dsets["live"].shape)
            # the verifier re-estimates sigma_bar from live sessions that include the network jitter
            sig_net = math.sqrt(sig_design ** 2 + (scale * jit.std()) ** 2)
            sets = {"live": Lv}
            for lab in Dsets:
                if lab != "live":
                    sets[lab] = Dsets[lab] + extra(Dsets[lab].shape)
            e = evaluate_block(sets, delta_design, sig_net)
            e.insert(0, "rtt_jitter_scale", scale); e.insert(1, "rtt_jitter_sd_ms", scale * jit.std())
            e.insert(2, "sigma_used_ms", sig_net)
            E_rows.append(e)
        pd.concat(E_rows).to_csv(os.path.join(out, "E_network_delay.csv"), index=False)
        summary["rtt_ms"] = dict(n=int(len(r)), median=float(np.median(r)), sd=float(r.std()),
                                 p99=float(np.quantile(r, 0.99)))

    # F. overhead and decision latency
    us, _ = overhead_us()
    summary["e_process_update_us"] = us
    if round_ms is None:
        round_ms = 600 + 800                   # flash on time + mean gap in flash_capture.py defaults
    summary["round_duration_ms"] = round_ms
    if attack_logs:
        F = []
        for p in sorted(glob.glob(attack_logs)):
            L = pd.read_csv(p)
            F.append(dict(log=os.path.basename(p), frames=len(L), proc_ms_median=float(L.proc_ms.median()),
                          proc_ms_p95=float(L.proc_ms.quantile(0.95)), face_rate=float(L.face.mean()),
                          fps_out=float(1e9 / np.median(np.diff(L.t_sent_ns)))))
        pd.DataFrame(F).to_csv(os.path.join(out, "F_attack_pipeline_cost.csv"), index=False)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("\nA. noise by condition\n", A.round(2).to_string(index=False))
    print("\nB. latency floor\n", pd.DataFrame(B_rows).T.round(1).to_string())
    print("\nC. recorded sessions\n", C.round(4).to_string(index=False))
    print("\nD. trace-driven\n", D.round(5).to_string(index=False))

# --------------------------------------------------------------------------- self-test (synthetic only)
def selftest():
    import tempfile
    rng = np.random.default_rng(1)
    dirs = np.abs(rng.normal(size=(8, 3))); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rows = []
    def add(label, cond, n_sess, base, off_sd, sd, extra=0.0):
        for s in range(n_sess):
            off = rng.normal(0, off_sd)
            for k in range(1, 31):
                c = int(rng.integers(0, 8))
                d = dirs[c] + 0.03 * rng.normal(size=3)
                rows.append(dict(session_id=f"{label}_{cond}_{s}", label=label, condition=cond, camera_name="cam",
                                 fps=30, round=k, color_idx=c, lag_on_ms=base + extra + off + rng.normal(0, sd),
                                 lag_off_ms=np.nan, amp=10, qc_ok=True, dir_b=d[0], dir_g=d[1], dir_r=d[2]))
    add("live", "normal", 20, 90, 4, 12)
    add("live", "dim", 10, 90, 4, 15)
    add("replay", "normal", 10, 90, 4, 12)
    add("faceswap", "normal", 10, 90, 6, 14, extra=120)
    tmp = tempfile.mkdtemp()
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "lags.csv"), index=False)
    analyse(os.path.join(tmp, "lags.csv"), os.path.join(tmp, "out"), n_trace=20_000)
    print("\nSELFTEST (synthetic data) finished:", tmp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lags", nargs="?")
    ap.add_argument("--out", default="real_out")
    ap.add_argument("--rtt")
    ap.add_argument("--attack-logs")
    ap.add_argument("--n-trace", type=int, default=200_000)
    ap.add_argument("--round-ms", type=float)
    ap.add_argument("--skip-rounds", type=int, default=2)
    ap.add_argument("--delta", type=float, default=None, help="design floor (ms); default = measured lower bound")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    analyse(a.lags, a.out, a.rtt, a.attack_logs, a.n_trace, round_ms=a.round_ms, skip_rounds=a.skip_rounds, delta_override=a.delta)

if __name__ == "__main__":
    main()
