#!/usr/bin/env python3
"""
fraud1_jitter.py (v2) -- estimate per-round TIMING JITTER (sigma, ms) of a screen-flash
challenge-response from FRAUD1-style videos (live subject, screen shows N short coloured flashes).

v2 changes (after a real-data run rejected ~89% of FRAUD1 videos with the v1 detector):
  * detector "peaks" (default): per-flash half-prominence edges via scipy find_peaks/peak_widths.
    Each flash is thresholded against ITS OWN height, so weak flashes (blue light on skin, auto-gain)
    are no longer missed. "schmitt" = the v1 global-threshold detector, kept for A/B comparison.
  * ROI "auto" (default): the window whose pixels respond most to the flashes (works for off-centre faces).
  * signal = L2 norm of per-channel RGB deviation from baseline (colour-agnostic), not summed luminance.
  * --diagnose: shows WHY videos fail (pulse-count histogram, weakest/strongest flash ratio, ROI boxes).
  * --dump-signals DIR: per-video CSV (frame, signal, edges) for plotting.

What it can and cannot do
  FRAUD1 has no display timestamps, so absolute lag and the between-device offset spread (tau_live) are NOT
  identifiable. What IS identifiable is how irregular the measured flash onsets are within a video: for
  nominally periodic flashes, d2 = t[k+2] - 2 t[k+1] + t[k] has variance 6 sigma^2 for i.i.d. per-flash jitter.
  This is a JITTER proxy for sigma in sim.py, not a latency.

Usage
  python3 fraud1_jitter.py --list     DIR
  python3 fraud1_jitter.py --diagnose DIR --include 'live' [--roi auto|center|face|x0,y0,x1,y1]
  python3 fraud1_jitter.py DIR --include 'live' --out calib.json [--detector peaks|schmitt]
  python3 fraud1_jitter.py --selftest
"""
import argparse, json, math, os, re, sys, tempfile
import numpy as np
import cv2
from scipy import ndimage, signal

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv", ".divx", ".m4v")

# ---------------------------------------------------------------- video -> frames -> ROI -> signal
def load_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps >= 1 else 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 12:
        raise ValueError("too few frames")
    return frames, fps

def auto_roi(frames, frac=0.4):
    """Window (frac of width/height) whose pixels respond most strongly to the flashes."""
    small = np.stack([cv2.resize(f, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32).sum(2) for f in frames])
    sd = small.std(0)                                          # temporal std per pixel
    win = (max(3, int(48 * frac)), max(3, int(64 * frac)))
    score = ndimage.uniform_filter(sd, size=win, mode="constant")
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    H, W = frames[0].shape[:2]
    y0 = int(np.clip(iy - win[0] // 2, 0, 48 - win[0])); x0 = int(np.clip(ix - win[1] // 2, 0, 64 - win[1]))
    sx, sy = W / 64.0, H / 48.0
    return int(x0 * sx), int(y0 * sy), int((x0 + win[1]) * sx), int((y0 + win[0]) * sy)

def choose_roi(frames, mode, W, H):
    if mode == "center":
        return int(.25 * W), int(.20 * H), int(.75 * W), int(.80 * H)
    if mode == "auto":
        return auto_roi(frames)
    if mode == "face":
        idx = np.linspace(0, len(frames) - 1, min(15, len(frames))).astype(int)
        med = np.median(np.stack([frames[i] for i in idx]), axis=0).astype(np.uint8)
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = casc.detectMultiScale(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x + .2 * w), int(y + .2 * h), int(x + .8 * w), int(y + .8 * h)
        return auto_roi(frames)                                # fallback
    x0, y0, x1, y1 = [int(v) for v in mode.split(",")]
    return x0, y0, x1, y1

def save_roi_preview(frames, box, out_dir, name):
    """Median frame with the chosen ROI drawn on it. Contains faces: keep local, never publish (licence/privacy)."""
    os.makedirs(out_dir, exist_ok=True)
    idx = np.linspace(0, len(frames) - 1, min(15, len(frames))).astype(int)
    img = np.median(np.stack([frames[i] for i in idx]), axis=0).astype(np.uint8).copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    path = os.path.join(out_dir, "roi_" + os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(path, img)
    return path

def make_signal(frames, box, kind="l2"):
    x0, y0, x1, y1 = box
    sig = np.array([f[y0:y1, x0:x1].reshape(-1, 3).mean(0) for f in frames])           # n x 3 (BGR)
    dev = sig - np.percentile(sig, 20, axis=0)
    if kind == "sum":                                                                  # v1 behaviour
        return dev.sum(1)
    return np.sqrt((np.clip(dev, 0, None) ** 2).sum(1))                               # colour-agnostic

# ---------------------------------------------------------------- edge detectors (times in ms, sub-frame)
def edges_schmitt(s, fps, lo_pct=20, hi_pct=92, lo_thr=.35, hi_thr=.65, mid=.5):
    """v1: global threshold at half of a global amplitude. Misses flashes weaker than ~half the strongest."""
    lo, hi = np.percentile(s, lo_pct), np.percentile(s, hi_pct)
    if hi - lo < 1e-6:
        return [], [], []
    z, dt = (s - lo) / (hi - lo), 1000.0 / fps
    rises, falls = [], []
    state = "off_armed" if z[0] <= lo_thr else ("off_unarmed" if z[0] < mid else "on_unarmed")
    for i in range(1, len(z)):
        a, b = z[i - 1], z[i]
        if state == "off_unarmed" and b <= lo_thr:
            state = "off_armed"
        elif state == "off_armed" and a < mid <= b:
            rises.append((i - 1 + (mid - a) / (b - a)) * dt); state = "on_unarmed"
        elif state == "on_unarmed" and b >= hi_thr:
            state = "on_armed"
        elif state == "on_armed" and a >= mid > b:
            falls.append((i - 1 + (a - mid) / (a - b)) * dt); state = "off_unarmed"
    return rises, falls, []

def edges_peaks(s, fps, cycles=4, prom_frac=0.12, noise_k=6.0):
    """v2: each flash is a peak; its edges are the half-prominence crossings of THAT peak."""
    dt = 1000.0 / fps
    low = np.percentile(s, 20)
    A = np.percentile(s, 95) - low
    if A <= 1e-6:
        return [], [], []
    d = np.diff(s)
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2)
    prom = max(prom_frac * A, noise_k * noise)
    pk, pr = signal.find_peaks(s, prominence=prom, distance=2)
    if len(pk) == 0:
        return [], [], []
    p = pr["prominences"]
    order = np.argsort(p)[::-1]
    if len(pk) > cycles:                                       # keep the strongest `cycles` only if clearly separated
        if p[order[cycles]] < 0.5 * p[order[cycles - 1]]:
            keep = np.sort(order[:cycles])
            pk, p = pk[keep], p[keep]
            pr = {k: v[keep] for k, v in pr.items()}
    _, _, left, right = signal.peak_widths(s, pk, rel_height=0.5,
                                           prominence_data=(pr["prominences"], pr["left_bases"], pr["right_bases"]))
    return list(left * dt), list(right * dt), list(pr["prominences"] / A)

# ---------------------------------------------------------------- estimators
def second_diffs(t):
    t = np.asarray(t, float)
    return t[2:] - 2 * t[1:-1] + t[:-2]

def sigma_from_d2(d2_all):
    d2_all = np.asarray(d2_all, float)
    return math.sqrt(np.mean(d2_all ** 2) / 6.0) if len(d2_all) else float("nan")

def bootstrap_sigma(per_video_d2, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(per_video_d2)
    if n < 3:
        return float("nan"), float("nan")
    est = [sigma_from_d2(np.concatenate([per_video_d2[i] for i in rng.integers(0, n, n)])) for _ in range(B)]
    return tuple(np.percentile(est, [2.5, 97.5]))

def analyse(paths, roi="auto", cycles=4, detector="peaks", group_regex=None, dump_dir=None, roi_preview=None, root=None):
    rows, d2_all, rejected, diag, fps_seen = [], [], [], [], []
    iv_frames, d2_disc, previews = [], [], []
    nm = (lambda q: os.path.relpath(q, root)) if root else (lambda q: q)      # report names relative to the data dir (no usernames)
    for p in paths:
        try:
            frames, fps = load_frames(p)
            H, W = frames[0].shape[:2]
            box = choose_roi(frames, roi, W, H)
            s = make_signal(frames, box, "sum" if detector == "schmitt" else "l2")
            if roi_preview and len(previews) < 12:
                previews.append(save_roi_preview(frames, box, roi_preview, os.path.basename(p)))
        except Exception as e:
            rejected.append((nm(p), f"read error: {e}")); continue
        fps_seen.append(fps)
        if detector == "schmitt":
            r, f, prom = edges_schmitt(s, fps)
        else:
            r, f, prom = edges_peaks(s, fps, cycles)
        diag.append(dict(video=nm(p), n_rise=len(r), n_fall=len(f), box=box,
                         weak_ratio=(min(prom) / max(prom)) if len(prom) >= 2 else None))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            np.savetxt(os.path.join(dump_dir, os.path.basename(p) + ".csv"), np.c_[np.arange(len(s)), s],
                       delimiter=",", header="frame,signal", comments="")
        if len(r) != cycles or len(f) != cycles:
            rejected.append((nm(p), f"{len(r)} rises / {len(f)} falls (want {cycles})")); continue
        dd = np.concatenate([second_diffs(r), second_diffs(f)])
        d2_all.append(dd)
        fm = 1000.0 / fps
        iv_frames.append(np.concatenate([np.diff(r), np.diff(f)]) / fm)          # onset/offset intervals in frames
        for t in (r, f):                                                           # same schedule rounded to whole frames
            d2_disc.append(second_diffs(np.r_[0.0, np.cumsum(np.round(np.diff(t) / fm))] * fm))
        g = (re.search(group_regex, p).group(1) if group_regex and re.search(group_regex, p) else "all")
        rows.append(dict(video=nm(p), group=g, fps=fps, period_ms=(r[-1] - r[0]) / (cycles - 1),
                         width_ms=float(np.mean(np.array(f) - np.array(r))), box=str(box)))
    tot = len(rows) + len(rejected)
    res = dict(detector=detector, roi=roi, n_used=len(rows), n_rejected=len(rejected),
               fps_median=float(np.median(fps_seen)) if fps_seen else None,
               reject_rate=(len(rejected) / tot) if tot else None)
    res["warning"] = ("reject rate > 15%: videos with irregular pulses are dropped, so sigma is biased LOW "
                      "(treat as lower bound)") if tot and len(rejected) / tot > 0.15 else None
    if rows:
        res["frame_ms"] = 1000.0 / res["fps_median"]
        res["quantisation_floor_ms"] = 1000.0 / res["fps_median"] / math.sqrt(12)
        res["sigma_pooled_ms"] = sigma_from_d2(np.concatenate(d2_all))
        res["sigma_pooled_ci95"] = list(bootstrap_sigma(d2_all))
        res["period_ms_mean"] = float(np.mean([r["period_ms"] for r in rows]))
        if len(rows) > 1:
            res["period_ms_between_video_sd"] = float(np.std([r["period_ms"] for r in rows], ddof=1))
            res["width_ms_between_video_sd"] = float(np.std([r["width_ms"] for r in rows], ddof=1))
        iv = np.concatenate(iv_frames)
        R = float(abs(np.mean(np.exp(2j * np.pi * iv))))                              # 1 = intervals sit on a common fractional frame
        resid = float(np.sqrt(np.mean((iv - np.round(iv)) ** 2)))                    # rms distance of intervals from whole frames (uniform = 0.29)
        vals, cnts = np.unique(np.round(iv).astype(int), return_counts=True)
        sd = sigma_from_d2(np.concatenate(d2_disc))
        share = float((sd / res["sigma_pooled_ms"]) ** 2) if res["sigma_pooled_ms"] > 0 else None
        res["framelock"] = dict(R=R, resid_rms_frames=resid, interval_frames_hist={int(v): int(c) for v, c in zip(vals, cnts)},
                                sigma_whole_frame_part_ms=sd, whole_frame_share_of_variance=share,
                                reading=("FRAME-LOCKED: onset intervals sit on whole frames (rms distance <= 0.15 frame), so sigma_hat mostly reflects "
                                         "frame-count variability of the capture app, not display-to-camera timing noise" if resid <= 0.15 else
                                         "NOT frame-locked: intervals vary continuously (rms distance >= 0.22 frame), sigma_hat reflects timing noise" if resid >= 0.22 else
                                         "MIXED: partly frame-locked; report the interval histogram"))
        if previews:
            res["roi_previews"] = [os.path.basename(q) for q in previews]
        groups = sorted({r["group"] for r in rows})
        if len(groups) > 1:
            res["sigma_by_group_ms"] = {g: sigma_from_d2(np.concatenate([d for d, r in zip(d2_all, rows) if r["group"] == g]))
                                        for g in groups}
    return res, rows, rejected, diag

def print_diagnosis(diag, cycles):
    from collections import Counter
    c = Counter(d["n_rise"] for d in diag)
    print(f"\nflash-count histogram over {len(diag)} videos (want {cycles}):")
    for k in sorted(c):
        print(f"  {k:2d} detected: {c[k]:4d}  ({c[k] / len(diag):5.1%})")
    wr = [d["weak_ratio"] for d in diag if d["weak_ratio"] is not None]
    if wr:
        wr = np.array(wr)
        print(f"weakest/strongest flash ratio: median {np.median(wr):.2f}; {np.mean(wr < 0.5):.0%} of videos have a flash < 50% of the strongest")
        print("  -> if that share is large, a global threshold (schmitt) misses weak flashes; the peaks detector handles them")
    bx = np.array([d["box"] for d in diag])
    print(f"ROI box (x0,y0,x1,y1) median: {np.median(bx, axis=0).astype(int).tolist()}  (check it lies on the face)")

# ---------------------------------------------------------------- synthetic multi-colour videos (self-test)
COLOURS = {"R": (0.25, 0.30, 1.00), "G": (0.35, 1.00, 0.55), "B": (1.00, 0.45, 0.25),
          "C": (1.00, 0.80, 0.50), "M": (0.85, 0.45, 0.90), "Y": (0.30, 0.90, 0.95)}     # BGR gains on skin

def synth_video(path, sigma_ms, rng, fps=30.0, cycles=4, period_ms=203.3, width_ms=96.7, exposure_ms=20.0,
                size=(320, 240), pix_noise=2.0, amp=60.0, variable_amp=True, offcenter=True, bg_resp=0.15, frame_locked=False):
    T = 1000.0 / fps
    lead = rng.uniform(300, 600)
    names = rng.choice(list(COLOURS), cycles)
    pulses = []
    if frame_locked:                                           # app toggles every 5 or 6 camera frames; fixed display phase
        fr = rng.integers(10, 18) + np.r_[0, np.cumsum(rng.choice([5, 6], cycles - 1, p=[0.3, 0.7]))]
        phi = rng.uniform(0, T)
    for k in range(cycles):
        if frame_locked:
            r = fr[k] * T + phi + rng.normal(0, 1.0); f = (fr[k] + 3) * T + phi + rng.normal(0, 1.0)
        else:
            r = lead + k * period_ms + rng.normal(0, sigma_ms)
            f = lead + k * period_ms + width_ms + rng.normal(0, sigma_ms)
        a = amp * (rng.uniform(0.35, 1.0) if variable_amp else 1.0)
        pulses.append((r, f, a * np.array(COLOURS[names[k]])))
    n = int((max(f for _, f, _ in pulses) + 600) / T)
    W, H = size
    cx = rng.uniform(.3, .7) * W if offcenter else W / 2
    cy = rng.uniform(.35, .65) * H if offcenter else H / 2
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (90 + 40 * xx / W).astype(np.float32)
    face = ((xx - cx) ** 2 / (W * .20) ** 2 + (yy - cy) ** 2 / (H * .28) ** 2) < 1
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        t1 = i * T; t0 = t1 - exposure_ms
        add = np.zeros(3)
        for r, f, vec in pulses:
            add += vec * max(0.0, min(t1, f) - max(t0, r)) / exposure_ms
        img = np.stack([bg, bg, bg], -1)
        img += bg_resp * add * (~face)[..., None] * 0.5
        img[face] += add
        img += rng.normal(0, pix_noise, img.shape)
        vw.write(np.clip(img, 0, 255).astype(np.uint8))
    vw.release()

def selftest():
    rng = np.random.default_rng(11)
    print("self-test: synthetic 30 fps videos, 4 coloured flashes of UNEQUAL strength, face OFF-centre, known per-edge jitter\n")
    configs = [("schmitt", "center"), ("schmitt", "auto"), ("peaks", "center"), ("peaks", "auto")]
    print(f"{'true_sigma':>10} | " + " | ".join(f"{d}+{r}: est (rej%)".rjust(24) for d, r in configs))
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for sig in [5.0, 10.0, 15.0]:
            paths = []
            for j in range(40):
                p = os.path.join(d, f"s{int(sig)}_{j}.avi"); synth_video(p, sig, rng); paths.append(p)
            cells = []
            for det, roi in configs:
                res, _, _, _ = analyse(paths, roi, 4, det)
                est = res.get("sigma_pooled_ms", float("nan"))
                cells.append(f"{est:6.1f} ({res['reject_rate']:5.1%})".rjust(24))
                if (det, roi) == ("peaks", "auto"):
                    ok &= (res["reject_rate"] <= 0.15) and abs(est - sig) <= max(0.30 * sig, 2.0)
            print(f"{sig:10.1f} | " + " | ".join(cells))
    with tempfile.TemporaryDirectory() as d:
        pl, pt = [], []
        for j in range(30):
            a = os.path.join(d, f"fl{j}.avi"); synth_video(a, 0.0, rng, frame_locked=True); pl.append(a)
            b = os.path.join(d, f"tj{j}.avi"); synth_video(b, 10.0, rng); pt.append(b)
        fl = analyse(pl, "auto", 4, "peaks")[0]["framelock"]; tj = analyse(pt, "auto", 4, "peaks")[0]["framelock"]
        print(f"\nframe-lock check: frame-locked app -> rms-from-whole-frame={fl['resid_rms_frames']:.2f}, R={fl['R']:.2f}, hist {fl['interval_frames_hist']}")
        print(f"                  time-jittered    -> rms-from-whole-frame={tj['resid_rms_frames']:.2f}, R={tj['R']:.2f}, hist {tj['interval_frames_hist']}")
        ok &= fl["resid_rms_frames"] <= 0.15 and tj["resid_rms_frames"] >= 0.22
    print("\nPASS: peaks+auto keeps reject rate <= 15% and recovers sigma within 30%; frame-lock check separates the two schedules" if ok
          else "\nFAIL: peaks+auto did not recover sigma -- do not use")
    print("The schmitt columns show the v1 failure mode this version fixes (unequal flash strength).")
    print("Known property: the estimate includes measurement noise and reads ~10% high (n=150: 11.05 ms for a true 10 ms; floor 0.9 ms at true 0).")
    return 0 if ok else 1

# ---------------------------------------------------------------- CLI
def discover(root, include, exclude):
    out = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(VIDEO_EXT):
                p = os.path.join(dp, f)
                if include and not re.search(include, p):
                    continue
                if exclude and re.search(exclude, p):
                    continue
                out.append(p)
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--include"); ap.add_argument("--exclude")
    ap.add_argument("--roi", default="auto"); ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--detector", choices=["peaks", "schmitt"], default="peaks")
    ap.add_argument("--group-regex"); ap.add_argument("--dump-signals"); ap.add_argument("--out", default="calib.json")
    ap.add_argument("--roi-preview", help="dir for ROI preview JPEGs (first 12 videos); contains faces, keep local")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.dir:
        ap.error("give a directory, or --selftest")
    paths = discover(a.dir, a.include, a.exclude)
    if a.list:
        for p in paths:
            c = cv2.VideoCapture(p); print(f"{c.get(cv2.CAP_PROP_FRAME_COUNT):6.0f} frames  {c.get(cv2.CAP_PROP_FPS):5.1f} fps  {p}"); c.release()
        print(f"{len(paths)} videos"); return
    res, rows, rej, diag = analyse(paths, a.roi, a.cycles, a.detector, a.group_regex, a.dump_signals, a.roi_preview, root=a.dir)
    if a.diagnose:
        print_diagnosis(diag, a.cycles)
    json.dump(dict(summary=res, rejected=rej, args=vars(a)), open(a.out, "w"), indent=1, default=float)
    try:
        import pandas as pd; pd.DataFrame(rows).to_csv(a.out.replace(".json", "_videos.csv"), index=False)
    except Exception:
        pass
    print(json.dumps(res, indent=1, default=float))
    print(f"\nrejected {len(rej)} videos (edge-count QC); first few:", *[f"\n  {p}: {why}" for p, why in rej[:5]])
    if res.get("n_used"):
        print(f"\nNext: python3 sim.py --sigma {res['sigma_pooled_ms']:.1f}   # re-run E1-E6 with the measured jitter")
        print("Caveat: timing JITTER on a frame-locked schedule, not display->camera latency; tau_live is NOT identifiable from FRAUD1.")

if __name__ == "__main__":
    main()
