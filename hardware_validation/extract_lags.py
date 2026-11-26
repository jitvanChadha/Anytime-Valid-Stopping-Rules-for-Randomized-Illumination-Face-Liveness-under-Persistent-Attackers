#!/usr/bin/env python3
"""
extract_lags.py -- turn captured sessions (flash_capture.py output) into per-round lags.

For each round the face-ROI colour is projected on the direction of its own colour step
(plateau minus pre-flash baseline). The onset lag is the half-height crossing time,
linearly interpolated between frame timestamps, minus the flash timestamp. The offset lag
is the same for the falling edge after the screen goes black.

  python extract_lags.py sessions/ --out lags.csv
Columns: session_id,label,condition,camera_name,fps,round,color_idx,lag_on_ms,lag_off_ms,
         amp,qc_ok,dir_b,dir_g,dir_r
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

def crossing(t, s, t_from, rising=True, level=0.5):
    """First interpolated crossing of `level` at or after t_from."""
    i0 = np.searchsorted(t, t_from)
    for i in range(max(i0, 1), len(t)):
        a, b = s[i - 1], s[i]
        if (rising and a < level <= b) or (not rising and a > level >= b):
            return t[i - 1] + (level - a) / (b - a) * (t[i] - t[i - 1])
    return np.nan

def session_lags(base, amp_min=1.5):
    meta = json.load(open(base + "_meta.json"))
    fr = pd.read_csv(base + "_frames.csv")
    rd = pd.read_csv(base + "_rounds.csv")
    t = (fr["t_ns"].to_numpy(np.int64) - fr["t_ns"].iloc[0]) / 1e6          # ms
    C = fr[["b", "g", "r"]].to_numpy(float)
    t0 = fr["t_ns"].iloc[0]
    rows = []
    for _, r in rd.iterrows():
        tf = (r.t_flash_ns - t0) / 1e6
        toff = (r.t_off_ns - t0) / 1e6
        on = toff - tf
        pre = (t >= tf - 250) & (t < tf)
        plat = (t >= tf + 0.6 * on) & (t < toff + 40)
        post = (t >= toff + 0.6 * on) & (t < toff + on)       # black again, after the fall
        ok = pre.sum() >= 3 and plat.sum() >= 2
        lag_on = lag_off = amp = np.nan
        d = np.full(3, np.nan)
        if ok:
            b0 = np.median(C[pre], 0)
            p = np.median(C[plat], 0)
            d = p - b0
            amp = float(np.linalg.norm(d))
            ok = amp >= amp_min
            if ok:
                s = (C - b0) @ d / amp ** 2
                lag_on = crossing(t, s, tf - 30, True) - tf
                if post.sum() >= 2:
                    lag_off = crossing(t, s, toff - 30, False) - toff
                ok = np.isfinite(lag_on) and -30 <= lag_on <= 0.6 * on
        rows.append(dict(session_id=meta["session_id"], label=meta["label"], condition=meta["condition"],
                         camera_name=meta["camera_name"], fps=int(round(meta["fps_measured"] / 5.0) * 5), res=f"{meta['width']}x{meta['height']}",
                         round=int(r["round"]), color_idx=int(r.color_idx), lag_on_ms=lag_on,
                         lag_off_ms=lag_off, amp=amp, qc_ok=bool(ok),
                         dir_b=d[0], dir_g=d[1], dir_r=d[2]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions")
    ap.add_argument("--out", default="lags.csv")
    ap.add_argument("--amp-min", type=float, default=1.5)
    a = ap.parse_args()
    rows = []
    for m in sorted(glob.glob(os.path.join(a.sessions, "*_meta.json"))):
        if json.load(open(m)).get("fps_switch"):
            print("skipped (frame rate changed mid-session):", os.path.basename(m)); continue
        rows += session_lags(m[: -len("_meta.json")], a.amp_min)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    g = df.groupby(["label", "condition", "camera_name", "fps", "res"])
    print(g.agg(sessions=("session_id", "nunique"), rounds=("round", "size"), qc=("qc_ok", "mean"),
                lag_med=("lag_on_ms", "median"), lag_sd=("lag_on_ms", "std"), amp=("amp", "median")).round(2))

if __name__ == "__main__":
    main()
