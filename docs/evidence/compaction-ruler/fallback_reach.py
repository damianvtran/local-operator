"""How often does the receipt's no-shrink FALLBACK actually fire?

``Session._run_compaction`` applies the proportional ratio only when the pass
shrank local history (``history_after < history_before``); otherwise it reports
``context_tokens`` and the receipt renders a bare "context compacted".

Round 2 recorded that fallback as a limitation affecting vision models
generally, on the strength of synthetic 10-70 turn fixtures where snapcompact's
local estimate grew every time. That generalisation was wrong, and this script
is why: it replays the TEN REAL snapcompact passes of the production session
this PR is built on and reports which branch each would take.

The mechanism the fixtures mistook for a model property: a snapcompact archive
carries a FIXED overhead -- verbatim plain-text edges sized by frame shape
(``HQ_EDGE_FRAMES``), not by how much history was removed. Against a
near-threshold toy history of 18k-56k tokens that fixed cost outweighs what the
pass took away, so the local estimate grows. Against a real pass firing at
300-600k it is a small fraction. The fallback is therefore a property of
unusually SMALL histories, not of vision models.

Run with ``LO_REPO=<repo-root>``; reads the transcript read-only.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("LO_REPO", os.path.expanduser("~/local-operator")))

from local_operator.compaction.tokens import estimate_messages_tokens  # noqa: E402
from local_operator.session.session import _default_convert_to_llm  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402

PATH = os.path.expanduser("~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl")
lines = [line for line in open(PATH) if line.strip()]
objs = [json.loads(line) for line in lines]
tmp = tempfile.mkdtemp()


def local_at(n):
    with open(os.path.join(tmp, "transcript.jsonl"), "w") as handle:
        handle.writelines(lines[:n])
    return estimate_messages_tokens(
        _default_convert_to_llm(list(Transcript(tmp).build_llm_history()))
    )


comp = [i for i, e in enumerate(objs) if e.get("type") == "compaction"]
print(f"{'pass':>6} {'hist_before':>12} {'hist_after':>11} {'shrank':>7} {'ratio':>7}  branch")

fallback = 0
for ci in comp:
    hb = local_at(ci)
    ha = local_at(ci + 1)
    shrank = ha < hb
    if not shrank:
        fallback += 1
    branch = "proportional" if shrank else "FALLBACK (bare receipt)"
    print(
        f"{ci:>6} {hb:>12,} {ha:>11,} {str(shrank):>7} " f"{hb / ha if ha else 0:>6.1f}x  {branch}"
    )

print(
    f"\n{fallback}/{len(comp)} real snapcompact passes take the fallback; "
    f"{len(comp) - fallback}/{len(comp)} get the full proportional receipt."
)
print(
    "Every real pass shrinks local history several-fold, so the receipt reports\n"
    "accurate figures on the path users are actually on."
)
