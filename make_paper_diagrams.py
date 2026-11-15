#!/usr/bin/env python3
"""
make_paper_diagrams.py -- the four figures used in paper_ieee.tex, drawn for one IEEE column (3.5 in).

Design rules applied to every figure
  * one fixed colour, marker and line style per test, identical in all figures (colour is never the only cue);
  * legend in a single row above the axes, never on top of the data;
  * light horizontal grid only, no top/right spines, no minor-grid clutter;
  * Times-like 8 pt text so labels match the paper body; reference lines are labelled directly;
  * output: PaperDiagrams/<name>.png at 600 dpi and <name>.pdf (vector).

Data: e2_persistence.csv, e5_prime_wrong_floor.csv, e12_wrong_sigma.csv, runs/real_v2/e6_sigma_sweep.csv,
      runs/real_v2/e7_delta_phase.csv. Nothing is simulated here.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from scipy import integrate, stats

OUT = "PaperDiagrams"
os.makedirs(OUT, exist_ok=True)
COL = 3.5                                   # IEEE single-column width, inches

INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "lines.linewidth": 1.3, "lines.markersize": 4.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

# One identity per test, used everywhere (Okabe-Ito hues, plus distinct markers and dashes for print).
STYLE = {
    "F-iid":    dict(color="#D55E00", marker="o", linestyle="-",            label="F-iid"),
    "SPRT-iid": dict(color="#E69F00", marker="s", linestyle="-",            label="SPRT-iid"),
    "F-floor":  dict(color="#009E73", marker="^", linestyle=(0, (4, 2)),    label="F-floor"),
    "E-floor":  dict(color="#0072B2", marker="D", linestyle="-",            label="E-floor"),
}
ORDER = ["F-iid", "SPRT-iid", "F-floor", "E-floor"]
MK = dict(markeredgecolor="white", markeredgewidth=0.6)     # thin surface ring keeps overlapping markers apart


def decade(v, _):
    """Plain tick labels on log axes: 0.01, 0.1, 1, 10, 100, 1000."""
    return ("%g" % v) if v >= 1 else ("%g" % v)


def log_axis(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [10.0 ** k for k in range(int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi))) + 1)]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(decade))
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def nominal_line(ax, x_text, text=r"nominal level $\alpha$", va="bottom"):
    ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax.text(x_text, 1.0 * (1.25 if va == "bottom" else 0.8), text, fontsize=7, color=MUTED, ha="right", va=va)


def top_legend(fig, handles, labels, ncol, y=0.995):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
               handlelength=2.3, columnspacing=1.3, handletextpad=0.5, borderaxespad=0.0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, name + "." + ext))
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1: persistence
def fig_persistence():
    d = pd.read_csv("e2_persistence.csv")
    rhos = sorted(d.rho.unique())
    x = np.arange(len(rhos))                                   # evenly spaced: the values bunch up near 1
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for t in ORDER:
        y = d[d.test == t].set_index("rho").APCER_over_alpha.reindex(rhos).clip(lower=2e-4)
        ax.plot(x, y.values, **STYLE[t], **MK)
    log_axis(ax, 1e-4, 1e3)
    nominal_line(ax, x[-1] + 0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % r for r in rhos])
    ax.set_xlim(-0.35, len(rhos) - 0.65)
    ax.set_xlabel(r"Attacker persistence $\rho$ (values evenly spaced)")
    ax.set_ylabel(r"False-accept rate, APCER / $\alpha$")
    h, l = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.215, top=0.865)
    top_legend(fig, h, l, 4)
    save(fig, "fig1_persistence")


# ------------------------------------------------------------------ Fig. 2: misspecification
def fig_misspecification():
    a = pd.read_csv("e5_prime_wrong_floor.csv").sort_values("true_delta_prime")
    b = pd.read_csv("e12_wrong_sigma.csv").sort_values("sigma_ratio")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL, 2.05), sharey=True)
    sim = dict(color=STYLE["E-floor"]["color"], marker="D", linestyle="-", label="Simulated E-floor")
    bnd = dict(color=INK, linestyle=(0, (4, 2)), linewidth=1.0, label="Lundberg bound")

    ok = a.true_delta_prime > 25                                # the bound exists only above delta/2
    ax1.axvspan(-3, 25, color="#efefef", zorder=0, linewidth=0)
    ax1.text(11, 3e-3, "no bound\n" + r"($\delta' \leq \delta/2$)", fontsize=6.8, color=MUTED, ha="center", va="bottom", linespacing=1.15)
    ax1.plot(a.true_delta_prime, a.APCER_over_alpha, **sim, **MK)
    ax1.plot(a.true_delta_prime[ok], a.lundberg_bound_over_alpha[ok], **bnd)
    ax1.axvline(50, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
    ax1.text(51.5, 6e2, "design\n" + r"$\delta = 50$", fontsize=6.8, color=MUTED, ha="left", va="top", linespacing=1.15)
    ax1.set_xlim(-3, 68)
    ax1.set_xticks([0, 25, 50])
    ax1.set_xlabel(r"True attacker floor $\delta'$ (ms)")
    ax1.set_ylabel(r"APCER / $\alpha$")
    ax1.set_title("(a) Attacker faster than assumed", pad=4)

    ax2.plot(b.sigma_ratio, b.APCER_over_alpha, **sim, **MK)
    ax2.plot(b.sigma_ratio, b.lundberg_bound_over_alpha, **bnd)
    ax2.axvline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
    ax2.set_xlim(0.75, 1.55)
    ax2.set_xticks([0.8, 1.0, 1.2, 1.4])
    ax2.set_xlabel(r"True noise / assumed noise, $\sigma'/\sigma$")
    ax2.set_title("(b) Noise underestimated", pad=4)

    for ax in (ax1, ax2):
        log_axis(ax, 1e-3, 3e3)
        ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax2.text(1.54, 0.72, r"nominal $\alpha$", fontsize=7, color=MUTED, ha="right", va="top")
    h, l = ax1.get_legend_handles_labels()
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.21, top=0.80, wspace=0.10)
    top_legend(fig, h, l, 2)
    save(fig, "fig3_misspecification")


# ------------------------------------------------------------------ Fig. 3: noise sweep and SPRT drift
def sprt_drift(sig, delta=50.0, lazy=150.0, pi=0.10):
    def f(y):
        pd_ = stats.norm.pdf(y, delta, sig)
        pm = pi * pd_ + (1 - pi) * stats.norm.pdf(y, lazy, sig)
        return pd_ * np.log(pd_ / pm)
    kl_mix, _ = integrate.quad(f, delta - 8 * sig, delta + 8 * sig)
    return kl_mix - 0.5 * (delta / sig) ** 2


def fig_sigma():
    # The sigma sweep written by calibrated_run/sim.py (stage 6). It fixes delta = 50 ms and varies sigma, so it does
    # not depend on the calibrated sigma even though it lives in the calibrated run's folder.
    raw = pd.read_csv("runs/real_v2/e6_sigma_sweep.csv")
    d = raw.pivot_table(index="sigma", columns="test", values="APCER_over_alpha").reset_index().sort_values("sigma")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL, 2.9), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    for t in ORDER:
        ax1.plot(d.sigma, d[t].clip(lower=2e-3), **STYLE[t], **MK)
    log_axis(ax1, 1e-3, 1e3)
    nominal_line(ax1, 51.5)
    ax1.set_ylabel(r"APCER / $\alpha$")
    ax1.set_title("(a) False accepts under persistent attackers", pad=4)

    sig = np.linspace(10, 50, 161)
    dr = np.array([sprt_drift(s) for s in sig])
    c = STYLE["SPRT-iid"]["color"]
    ax2.axhline(0, color=INK, linewidth=0.7)
    ax2.fill_between(sig, 0, dr, where=dr > 0, color=c, alpha=0.22, linewidth=0)
    ax2.plot(sig, dr, color=c, linewidth=1.4)
    root = 24.17
    ax2.axvline(root, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
    ax2.set_ylim(-4.2, 1.5)
    ax2.set_yticks([-4, -3, -2, -1, 0, 1])
    ax2.grid(axis="y", color=GRID, linewidth=0.5); ax2.set_axisbelow(True)
    ax2.text(root + 0.8, -3.9, r"drift changes sign at $\sigma = 24.2$ ms" + "\n" + r"($\delta/\sigma = 2.07$)", fontsize=7, color=MUTED, ha="left", va="bottom", linespacing=1.2)
    ax2.text(39, 0.95, "positive drift: attack accepted", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax2.set_xlim(8.5, 51.5)
    ax2.set_xticks([10, 20, 30, 40, 50])
    ax2.set_xlabel(r"Per-round noise $\sigma$ (ms), with $\delta = 50$ ms")
    ax2.set_ylabel("SPRT-iid drift per round")
    ax2.set_title("(b) Drift of the SPRT-iid statistic under the floor attacker", pad=4)
    h, l = ax1.get_legend_handles_labels()
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.14, top=0.85, hspace=0.46)
    top_legend(fig, h, l, 4)
    save(fig, "fig4_sigma_sensitivity_drift")


# ------------------------------------------------------------------ Fig. 4: phase diagram at the calibrated jitter
def fig_phase():
    d = pd.read_csv("runs/real_v2/e7_delta_phase.csv")
    d = d[d["pop"] == "mix"]
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    for t in ORDER:
        s = d[d.test == t].sort_values("delta_over_sigma")
        ax.plot(s.delta_over_sigma, s.APCER_over_alpha.clip(lower=2e-3), **STYLE[t], **MK)
    log_axis(ax, 1e-3, 1e3)
    nominal_line(ax, 5.35)
    ax.axvline(2.0684, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
    ax.text(2.0684 + 0.07, 430, "SPRT-iid drift = 0", fontsize=7, color=MUTED, ha="left", va="center")
    ax.set_xlim(1.5, 5.35)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel(r"Floor-to-noise ratio $\delta/\hat\sigma$   ($\hat\sigma = 11.756$ ms)")
    ax.set_ylabel(r"False-accept rate, APCER / $\alpha$")
    h, l = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.21, top=0.87)
    top_legend(fig, h, l, 4)
    save(fig, "fig6_phase_diagram")


if __name__ == "__main__":
    fig_persistence()
    fig_misspecification()
    fig_sigma()
    fig_phase()
