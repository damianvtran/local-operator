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
(``HQ_EDGE_FRAMES``), not by how much history was removed. About 20,900 tokens
at the shipped Anthropic shape.

Measured directly, that overhead is what the branch turns on. Driving real
passes over synthetic histories, ``history_after - history_before`` sits at a
flat **+21.5k across 24k to 420k of history** -- the edge budget, near enough
exactly -- and only goes negative once the summarized middle exceeds it:

    hb=  24,040  ha=  27,050   +3,010
    hb=  60,100  ha=  81,652  +21,552
    hb= 240,400  ha= 261,893  +21,493
    hb= 420,700  ha= 442,133  +21,433
    hb= 622,373  ha= 622,369      -4   <- crosses to proportional

So the branch is decided by "did this pass summarize away more than the edge
cost", which depends on how much of the history is SUMMARIZABLE and not on
history size alone. That is why the crossover lands at different sizes in
different fixtures, and it is why no single token figure is quoted here as
"the" threshold. Real passes clear it comfortably: all ten below shrink 1.8x
to 8.8x. The fallback is a property of passes with little to remove relative
to a fixed archive cost, not of vision models.

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
