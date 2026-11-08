"""Paired statistics for v1 rep-0 cost ratios and pass rates.

WHY: v1 has one rep per (harness, task) cell, so the only defensible
inference is paired across tasks. This script reports, for each harness vs
Local Operator: geometric-mean and median per-task cost ratio with task-level
bootstrap 95% CIs (10k resamples, fixed seed), ratio of means with CI, sign
test and Wilcoxon signed-rank on log ratios, exact McNemar on pass/fail
discordant pairs, Wilson CIs on pass rates, a bootstrap CI on cost per pass,
and a Friedman test across the four full-arm harnesses.
Constraints: CIs capture TASK sampling only; within-cell run-to-run variance
is unobservable with n=1 and is NOT in these intervals. Needs numpy+scipy
(optional reviews/requirements.txt) and generated/reviews/cold_alt.json from
cold.py. Read-only. Run from kit root after regenerate.py; not a causal or
current-performance study, and not a measured run-to-run uncertainty estimate.
"""

import json

import numpy as np
from scipy import stats

rng = np.random.default_rng(20261103)
R = [r for r in json.load(open("generated/results.json"))["rows"] if r["rep"] == 0]
cold = json.load(open("generated/reviews/cold_alt.json"))


def table(arm, base="local-operator", metric="warm", tier=None):
    cell = {}
    for r in R:
        if r["arm"] != arm or (tier and r["tier"] != tier):
            continue
        v = r["cost_usd"] if metric == "warm" else r["cold_cost"]
        cell[(r["harness"], r["task_id"][:8])] = (v, r["score"])
    return cell


B = 10000


def boot_geo(ratios):
    lr = np.log(ratios)
    n = len(lr)
    idx = rng.integers(0, n, (B, n))
    bs = np.exp(lr[idx].mean(1))
    med = np.median(ratios[idx], 1)
    return (
        np.exp(lr.mean()),
        np.percentile(bs, [2.5, 97.5]),
        np.median(ratios),
        np.percentile(med, [2.5, 97.5]),
    )


def boot_meanratio(a, b):
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    bs = b[idx].mean(1) / a[idx].mean(1)
    return b.mean() / a.mean(), np.percentile(bs, [2.5, 97.5])


out = []
for arm, metric, tier in [
    ("full", "warm", None),
    ("full", "cold", None),
    ("full", "warm", "short"),
    ("full", "warm", "long"),
    ("clean", "warm", None),
    ("clean", "cold", None),
]:
    cell = table(arm, metric=metric, tier=tier)
    tasks = sorted({t for h, t in cell if h == "local-operator"})
    print(f"\n== arm={arm} metric={metric} tier={tier or 'all'} n_tasks={len(tasks)}")
    for h in ["omp", "opencode", "claude-code", "codex"]:
        T = [t for t in tasks if (h, t) in cell]
        if len(T) < 4:
            continue
        a = np.array([cell[("local-operator", t)][0] for t in T])
        b = np.array([cell[(h, t)][0] for t in T])
        rat = b / a
        g, gci, md, mci = boot_geo(rat)
        mr, mrci = boot_meanratio(a, b)
        w = stats.wilcoxon(np.log(rat)) if len(T) >= 6 else None
        wins = int((rat > 1).sum())
        sign = stats.binomtest(wins, len(T)).pvalue
        # pass: McNemar exact on discordant pairs
        pa = np.array([cell[("local-operator", t)][1] >= 1 for t in T])
        pb = np.array([cell[(h, t)][1] >= 1 for t in T])
        n10 = int((pa & ~pb).sum())
        n01 = int((~pa & pb).sum())
        mc = stats.binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
        line = (
            f"{h:12} n={len(T):2} {h}/LOP geo-mean ratio {g:.2f} [{gci[0]:.2f},{gci[1]:.2f}]  "
            f"median {md:.2f} [{mci[0]:.2f},{mci[1]:.2f}]  "
            f"ratio-of-means {mr:.2f} [{mrci[0]:.2f},{mrci[1]:.2f}]  "
            f"LOP cheaper {wins}/{len(T)} sign p={sign:.2g}  "
            f"Wilcoxon p={w[1]:.2g}"
            if w
            else ""
        ) + f"  | pass LOP {pa.sum()} vs {pb.sum()} discordant {n10}/{n01} McNemar p={mc:.2f}"
        print(line)
# pass-rate CIs (Wilson) full arm
print("\n== pass rate Wilson 95% CI, full arm")
for h in ["local-operator", "omp", "opencode", "claude-code"]:
    s = [r["score"] >= 1 for r in R if r["arm"] == "full" and r["harness"] == h]
    k, n = sum(s), len(s)
    ci = stats.binomtest(k, n).proportion_ci(method="wilson")
    print(f"{h:15} {k}/{n} = {k/n:.0%} [{ci.low:.0%},{ci.high:.0%}]")
# cost per pass bootstrap
print("\n== cost per pass, bootstrap over tasks, full arm (ratio vs LOP)")
cell = table("full")
tasks = sorted({t for h, t in cell if h == "local-operator"})


def cpp(h, idx):
    c = np.array([cell[(h, tasks[i])][0] for i in idx])
    p = np.array([cell[(h, tasks[i])][1] >= 1 for i in idx])
    return c.sum() / max(p.sum(), 1)


for h in ["local-operator", "omp", "opencode", "claude-code"]:
    bs = []
    rs = []
    for _ in range(B):
        idx = rng.integers(0, 24, 24)
        x = cpp(h, idx)
        bs.append(x)
        rs.append(x / cpp("local-operator", idx))
    print(
        f"{h:15} cpp {cpp(h, range(24))*100:.1f}c "
        f"[{np.percentile(bs, 2.5)*100:.1f},{np.percentile(bs, 97.5)*100:.1f}]  "
        f"ratio vs LOP [{np.percentile(rs, 2.5):.2f},{np.percentile(rs, 97.5):.2f}]"
    )
# Friedman across 4 harnesses, warm
M = np.array(
    [[cell[(h, t)][0] for h in ["local-operator", "omp", "opencode", "claude-code"]] for t in tasks]
)
print("\nFriedman (4 harnesses, 24 tasks, warm):", stats.friedmanchisquare(*M.T))
print(
    "mean ranks (1=cheapest):",
    dict(zip(["lop", "omp", "oc", "cc"], stats.rankdata(M, axis=1).mean(0).round(2))),
)
