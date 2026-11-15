#!/usr/bin/env python3
"""sim_guessing.py -- challenge-guessing attacker for every palette size in Table V of the paper.

Each round the attacker guesses the challenge with probability q (latency 0) and otherwise sits at the floor delta.
Same model as run_e11 in sim_wp3.py, extended to q = 1/16 and 1/8, which the original run did not cover.
Output: redteam_out/r5_guessing.csv
"""
import math, os
import numpy as np
import pandas as pd
from scipy import optimize, stats

SEED, N, T = 20261011, 500_000, 20
ALPHA, BETA, DELTA, SIGMA = 1e-3, 1e-2, 50.0, 25.0


def cp(k, n):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


rows = []
a = DELTA ** 2 / (2 * SIGMA ** 2)
for q in [0.01, 0.0625, 0.125, 0.25]:
    rng = np.random.default_rng([SEED, 905, int(q * 10000)])
    D = np.where(rng.random((N, T)) < q, 0.0, DELTA)
    Y = D + rng.normal(0, SIGMA, (N, T))
    L = np.cumsum(DELTA * (DELTA / 2 - Y) / SIGMA ** 2, axis=1)
    hu, hl = L >= math.log(1 / ALPHA), L <= math.log(BETA)
    t_up = np.where(hu.any(1), hu.argmax(1), 10 ** 9)
    t_lo = np.where(hl.any(1), hl.argmax(1), 10 ** 9)
    k = int((t_up < t_lo).sum())
    lo, hi = cp(k, N)
    theta = optimize.brentq(lambda th: (1 - q) * math.exp(a * th * (th - 1)) + q * math.exp(a * th * (th + 1)) - 1, 1e-8, 1 - 1e-8)
    rows.append(dict(q=q, K=round(1 / q), theta_star=theta, bound_over_alpha=ALPHA ** theta / ALPHA,
                     sim_over_alpha=k / N / ALPHA, ci_lo=lo / ALPHA, ci_hi=hi / ALPHA))
df = pd.DataFrame(rows)
os.makedirs("redteam_out", exist_ok=True)
df.to_csv("redteam_out/r5_guessing.csv", index=False)
print(df.round(3).to_string(index=False))
