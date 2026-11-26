#!/usr/bin/env python3
"""Fig. for the revision: per-round flash-to-reflection lag on the hardware testbed, by path.
Live and the three injection paths, 30 fps, 640x480, normal light, rounds 3..20, QC-passed.
Lag is relative to the device baseline (median live lag, normal light, 30 fps)."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lags, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(lags)
d = d[d.qc_ok & (d["round"] > 2) & (d.condition == "normal") & (d.fps == 30) & (d.res == "640x480")]
base = d[d.label == "live"].lag_on_ms.median()
rows = [("tint", "Tint\n(no re-synthesis)", "#eda100"), ("live", "Live face", "#2a78d6"),
        ("passthrough", "Injected,\npassthrough", "#1baf7a"), ("faceswap", "Injected,\nface swap", "#eb6834")]
delta_hat = float(np.percentile(d[d.label == "faceswap"].lag_on_ms - base, 1))

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.05), dpi=300)
rng = np.random.default_rng(0)
for i, (lab, name, col) in enumerate(rows):
    y = d[d.label == lab].lag_on_ms.to_numpy() - base
    ax.scatter(y, i + rng.uniform(-0.18, 0.18, len(y)), s=2.2, color=col, alpha=0.55, linewidths=0, zorder=2)
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    ax.plot([q1, q3], [i + 0.3, i + 0.3], color="#333333", lw=1.6, solid_capstyle="round", zorder=3)
    ax.plot([med], [i + 0.3], marker="|", ms=6, mew=1.4, color="#111111", zorder=4)
    ax.text(q3 + 5, i + 0.3, f"{med:+.0f} ms", va="center", fontsize=6.3, color="#444444")
ax.axvline(0, color="#777777", lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.axvline(delta_hat, color="#777777", lw=0.6, ls=(0, (1, 1.5)), zorder=1)
ax.text(-3, 3.8, "live baseline", fontsize=5.8, color="#555555", ha="right", va="center")
ax.text(delta_hat + 3, 3.8, "face-swap 1st percentile", fontsize=5.8, color="#555555", ha="left", va="center")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows])
ax.set_ylim(-0.5, 4.15); ax.set_xlim(-110, 300)
ax.set_xlabel("Lag relative to the live baseline (ms)")
ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(length=2, width=0.5)
fig.tight_layout(pad=0.3)
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print("baseline", round(base, 1), "delta_hat(1st pct, faceswap)", round(delta_hat, 1), "->", out)
