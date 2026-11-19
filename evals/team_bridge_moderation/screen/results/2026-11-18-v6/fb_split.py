"""False blocks on legit rows split by production outcome, per model x template.

H = model_hold / hold (owner can Deliver/Discard), W = work_scope_hold shape (owner
releasable on the preferred screener), R = bounce (terminal, not owner-releasable),
M = local payload check. Also lists each legit item that bounces under a template.
Usage: fb_split.py <trials.jsonl> <corpus.jsonl>...
"""

import collections
import json
import os
import sys

# Reuse class_gate.py's outcome() (committed beside this file) so both scripts
# classify a trial identically; only its definitions run, not its argv-driven body.
_GATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_gate.py")
exec(open(_GATE).read().split("base_path, cand_path")[0])

trials = [json.loads(line) for line in open(sys.argv[1])]
corpus = {}
for p in sys.argv[2:]:
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            corpus[r["id"]] = r
agg = collections.defaultdict(collections.Counter)
bounced = collections.defaultdict(collections.Counter)
for t in trials:
    r = corpus.get(t["corpus_id"])
    if r is None or r["label"] != "legit":
        continue
    m = "Sonnet" if "sonnet" in t["model_route"] else "Opus"
    o = outcome(t, r["body"])  # noqa: F821 - defined by the exec above
    if o == "x":
        continue
    k = (m, t["template"])
    agg[k]["n"] += 1
    if o != "D":
        agg[k][o] += 1
        agg[k]["blocked"] += 1
    if o == "R":
        bounced[k][t["corpus_id"]] += 1
print("| model | template | false block | H | W | R | M |")
print("|---|---|---|---|---|---|---|")
for k in sorted(agg):
    c = agg[k]
    print(
        f"| {k[0]} | {k[1]} | {c['blocked']}/{c['n']} | {c['H']} | {c['W']} | {c['R']} | {c['M']} |"
    )
for k in sorted(bounced):
    print("bounces", k, dict(bounced[k]))
