#!/usr/bin/env python3
"""
fraud1_revision.py -- FRAUD1 analyses added for the WcAISR-2027 revision (reviewer comments R3-R5).

1. Jitter sigma per camera and per object (live / paper face / white paper), with video-level
   bootstrap CIs, using the same v2 onset detector as fraud1_jitter.py.
2. A colour-reflection consistency check of the kind used by FRAUD1 [Smith et al.] and Face
   Flashing [Tang et al.]: each of the four flashes is classified by the chromaticity of the
   ROI colour change (nearest centroid, camera-specific, leave-one-subject-out for live videos),
   and a session is accepted only if all four colours match the challenge.
   Reported: FRR on live videos; acceptance of paper-face and white-paper presentations against
   their own challenge; acceptance of replayed live videos against a different challenge.
3. FRAUD1's own published classifier outcome per live video (from the *-Results.txt files).

  python testbed/fraud1_revision.py FRAUD1_Dataset --out revision_out
"""
import argparse, glob, json, math, os, re, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fraud1_jitter as fj  # noqa: E402

COLOURS = ["Black", "Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"]

def challenge(path):
    txt = open(os.path.splitext(path)[0] + ".txt").read()
    m = re.search(r"Challenge colours are:\s*(.+)", txt)
    return m.group(1).split()

def video_features(path):
    t0 = time.perf_counter()
    frames, fps = fj.load_frames(path)
    t_load = time.perf_counter() - t0
    H, W = frames[0].shape[:2]
    box = fj.choose_roi(frames, "auto", W, H)
    t1 = time.perf_counter()
    x0, y0, x1, y1 = box
    sig3 = np.array([f[y0:y1, x0:x1].reshape(-1, 3).mean(0) for f in frames])
    t_roi = (time.perf_counter() - t1) / len(frames)
    s = fj.make_signal(frames, box, "l2")
    r, f, _ = fj.edges_peaks(s, fps, 4)
    out = dict(fps=fps, n_frames=len(frames), roi_ms_per_frame=1000 * t_roi, rises=r, falls=f, d2=[], chroma=[])
    if len(r) == 4 and len(f) == 4:
        out["d2"] = list(np.concatenate([fj.second_diffs(r), fj.second_diffs(f)]))
        base = np.percentile(sig3, 20, axis=0)
        dt = 1000.0 / fps
        for a, b in zip(r, f):
            i0, i1 = int(math.ceil(a / dt)), int(math.floor(b / dt))
            seg = sig3[i0:max(i1, i0) + 1]
            d = np.clip(seg.mean(0) - base, 0, None)
            n = np.linalg.norm(d)
            out["chroma"].append(list(d / n) if n > 0 else [np.nan] * 3)
    return out

def collect(root, cache):
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    rows = []
    vids = sorted(glob.glob(os.path.join(root, "FRAUD1-*", "*", "**", "*.avi"), recursive=True))
    for i, p in enumerate(vids):
        rel = os.path.relpath(p, root)
        obj = rel.split(os.sep)[0].replace("FRAUD1-", "")
        cam = "Logitech" if "Logitech" in rel else "LifeCam"
        subj = re.search(r"(Subject\d+)", rel)
        try:
            ft = video_features(p)
        except Exception as e:
            ft = dict(error=str(e), d2=[], chroma=[])
        rows.append(dict(video=rel, object=obj, camera=cam, subject=subj.group(1) if subj else obj,
                         challenge=challenge(p), **ft))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(vids)} videos", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df

def boot_sigma(d2_list, B=2000, seed=1):
    d2_list = [np.asarray(d) for d in d2_list if len(d)]
    if len(d2_list) < 3:
        return float("nan"), float("nan"), float("nan"), len(d2_list)
    est = fj.sigma_from_d2(np.concatenate(d2_list))
    lo, hi = fj.bootstrap_sigma(d2_list, B, seed)
    return est, lo, hi, len(d2_list)

def centroids(df):
    """Mean unit chromaticity per (camera, colour), from the flashes of the given videos."""
    acc = {}
    for _, v in df.iterrows():
        for c, x in zip(v.challenge, v.chroma):
            if np.all(np.isfinite(x)):
                acc.setdefault((v.camera, c), []).append(x)
    return {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in acc.items()}

def predict(v, cent):
    out = []
    for x in v.chroma:
        cands = [(c, float(np.dot(x, m))) for (cam, c), m in cent.items() if cam == v.camera]
        out.append(max(cands, key=lambda z: z[1])[0] if np.all(np.isfinite(x)) and cands else None)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="revision_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = collect(a.root, os.path.join(a.out, "fraud1_features.pkl"))
    df["qc_ok"] = df["chroma"].apply(len) == 4
    res = {}

    # 1. sigma per object x camera
    sig = []
    for (obj, cam), g in df.groupby(["object", "camera"]):
        est, lo, hi, n = boot_sigma(g["d2"].tolist())
        sig.append(dict(object=obj, camera=cam, videos=len(g), used=n, reject_rate=1 - n / len(g),
                        sigma_ms=est, ci_lo=lo, ci_hi=hi))
    for obj, g in df.groupby("object"):
        est, lo, hi, n = boot_sigma(g["d2"].tolist())
        sig.append(dict(object=obj, camera="both", videos=len(g), used=n, reject_rate=1 - n / len(g),
                        sigma_ms=est, ci_lo=lo, ci_hi=hi))
    sig = pd.DataFrame(sig)
    sig.to_csv(os.path.join(a.out, "fraud1_sigma_by_camera.csv"), index=False)
    print(sig.round(2).to_string(index=False))

    # 2. colour-reflection check
    live = df[df.object == "Live_Persons"]
    preds = {}
    for subj in sorted(live.subject.unique()):            # leave one subject out
        cent = centroids(live[(live.subject != subj) & live.qc_ok])
        for i, v in live[live.subject == subj].iterrows():
            preds[i] = predict(v, cent) if v.qc_ok else None
    cent_all = centroids(live[live.qc_ok])
    for i, v in df[df.object != "Live_Persons"].iterrows():
        preds[i] = predict(v, cent_all) if v.qc_ok else None
    df["pred"] = pd.Series(preds)
    df["match"] = [p is not None and list(p) == list(c) for p, c in zip(df.pred, df.challenge)]
    flash_acc = np.mean([pc == cc for p, c in zip(live.index.map(df.pred), live.challenge) if p is not None
                         for pc, cc in zip(p, c)])

    def rate(k, n):
        lo, hi = fj_ci(k, n)
        return dict(k=int(k), n=int(n), rate=k / n if n else float("nan"), ci_lo=lo, ci_hi=hi)

    rows = []
    for (obj, cam), g in list(df.groupby(["object", "camera"])) + [((o, "both"), g) for o, g in df.groupby("object")]:
        acc = int(g.match.sum()); n = len(g)
        if obj == "Live_Persons":
            rows.append(dict(object=obj, camera=cam, metric="FRR (incl. QC failures)", **rate(n - acc, n)))
            q = g[g.qc_ok]
            rows.append(dict(object=obj, camera=cam, metric="FRR (QC-passed only)", **rate(len(q) - q.match.sum(), len(q))))
        else:
            rows.append(dict(object=obj, camera=cam, metric="accept rate vs own challenge", **rate(acc, n)))
    # replay: live response presented against every other live video's challenge, same camera
    for cam in ["Logitech", "LifeCam", "both"]:
        g = live if cam == "both" else live[live.camera == cam]
        P = [df.pred[i] for i in g.index]
        C = g.challenge.tolist()
        k = n = 0
        for i, p in enumerate(P):
            for j, c in enumerate(C):
                if i == j or (cam == "both" and g.camera.iloc[i] != g.camera.iloc[j]):
                    continue
                n += 1
                k += p is not None and list(p) == list(c)
        rows.append(dict(object="Live replayed", camera=cam, metric="FAR replay vs new challenge", **rate(k, n)))
    # FRAUD1's own published classifier on live videos
    for cam, fn in [("Logitech", "LogitechPro9000"), ("LifeCam", "MicrosoftLifeCam")]:
        txt = open(os.path.join(a.root, "FRAUD1-Live_Persons", fn, f"{fn}-Results.txt")).read()
        n = txt.count("Resp -"); m = txt.count(" - Matches")
        rows.append(dict(object="Live_Persons", camera=cam, metric="FRR FRAUD1 original SVM [18]", **rate(n - m, n)))
    rr = pd.DataFrame(rows)
    rr.to_csv(os.path.join(a.out, "fraud1_reflection_check.csv"), index=False)
    print(f"\nper-flash colour accuracy (live, LOSO): {flash_acc:.4f}")
    print(rr.round(4).to_string(index=False))

    res["flash_colour_accuracy_live_loso"] = float(flash_acc)
    res["roi_ms_per_frame_median"] = float(np.nanmedian(df["roi_ms_per_frame"]))
    json.dump(res, open(os.path.join(a.out, "fraud1_revision_summary.json"), "w"), indent=1)

def fj_ci(k, n, level=0.95):
    from scipy import stats
    if n == 0:
        return float("nan"), float("nan")
    al = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(al / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - al / 2, k + 1, n - k)
    return float(lo), float(hi)

if __name__ == "__main__":
    main()
