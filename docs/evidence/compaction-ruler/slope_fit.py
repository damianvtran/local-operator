"""Provider-vs-local token slope, fitted per epoch and per model.

The two rulers this PR stops mixing: the local cl100k estimator
(`compaction/tokens.py`) and a provider's reported `context_tokens`. Fits
`provider = a * local + b` over each inter-compaction epoch of the real
session, and separately over every sampled point grouped by the model whose
usage record it came from.

The epoch table is the identifiability check. An epoch that stays on ONE model
fits tightly (mean error in the hundreds); an epoch that switches models does
not (tens of thousands), because two different tokenizers are being fitted with
one line. So the slope is a per-model property and may only be quoted for
model-homogeneous stretches -- which is exactly why this PR does not bake a
slope constant into the code and scales proportionally instead.

Run with LO_REPO=<repo-root>; reads the transcript read-only.
"""

import collections
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("LO_REPO", os.path.expanduser("~/local-operator")))
from local_operator.compaction.tokens import estimate_messages_tokens  # noqa: E402
from local_operator.session.session import _default_convert_to_llm  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402

PATH = os.path.expanduser("~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl")
lines = [x for x in open(PATH) if x.strip()]
objs = [json.loads(x) for x in lines]
tmp = tempfile.mkdtemp()


def local_at(n):
    with open(os.path.join(tmp, "transcript.jsonl"), "w") as f:
        f.writelines(lines[:n])
    return estimate_messages_tokens(
        _default_convert_to_llm(list(Transcript(tmp).build_llm_history()))
    )


def usage(e):
    return e.get("payload", {}).get("usage") or {}


def ctx(e):
    return usage(e).get("context_tokens")


def model_of(e):
    u = usage(e)
    return f"{u.get('provider')}/{u.get('model_id')}"


def fit(pts):
    n = len(pts)
    sx = sum(a for a, _ in pts)
    sy = sum(b for _, b in pts)
    sxx = sum(a * a for a, _ in pts)
    sxy = sum(a * b for a, b in pts)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    inter = (sy - slope * sx) / n
    err = sum(abs(b - (slope * a + inter)) for a, b in pts) / n
    return slope, inter, err


comp = [i for i, e in enumerate(objs) if e.get("type") == "compaction"]
bounds = [(comp[k] + 1, comp[k + 1]) for k in range(len(comp) - 1)]
print(f"{'epoch':>12} {'n':>4} {'slope':>6} {'fiterr':>7}  models present")
for lo, hi in bounds:
    idxs = [j for j in range(lo, hi) if ctx(objs[j])]
    if len(idxs) < 10:
        continue
    step = max(1, len(idxs) // 14)
    sample = idxs[::step]
    keep = [(j, local_at(j), ctx(objs[j])) for j in sample]
    keep = [(j, a, b) for j, a, b in keep if a > 5000]
    if len(keep) < 5:
        continue
    slope, _i, err = fit([(a, b) for _j, a, b in keep])
    models = collections.Counter(model_of(objs[j]) for j, _a, _b in keep)
    flag = "HOMOGENEOUS" if len(models) == 1 else f"MIXED({len(models)})"
    shown = ", ".join(f"{m} x{c}" for m, c in models.most_common())
    print(f"{f'{lo}-{hi}':>12} {len(keep):>4} {slope:>6.3f} {err:>7,.0f}  {flag}: {shown}")

print("\n=== pooled per MODEL ===")
bymodel = collections.defaultdict(list)
for lo, hi in bounds:
    idxs = [j for j in range(lo, hi) if ctx(objs[j])]
    step = max(1, len(idxs) // 14)
    for j in idxs[::step]:
        local = local_at(j)
        if local > 5000:
            bymodel[model_of(objs[j])].append((local, ctx(objs[j])))
for model, pts in sorted(bymodel.items(), key=lambda kv: -len(kv[1])):
    if len(pts) < 5:
        continue
    slope, inter, err = fit(pts)
    ratios = sorted(b / a for a, b in pts)
    print(
        f"  {model:<34} n={len(pts):>3} slope={slope:>6.3f} "
        f"inter={inter:>9,.0f} err={err:>8,.0f} "
        f"ratio p50={ratios[len(ratios) // 2]:.2f}"
    )
