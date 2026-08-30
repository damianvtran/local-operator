"""Retention effect of the floor cap, with genuine_user_ids computed the way
the SESSION computes it.

First attempt (retention_real.py) passed every user-role message in the
RENDERED history as genuine, which is wrong and measured the span far too
short. In production (session.py `_plan_compaction`) the set is built from
`self._context.messages` — the live transcript vocabulary, BEFORE
`_convert_to_llm` renders wake/hub/todo/incident CustomMessages into user-role
Messages. So the genuine set is a strict SUBSET of the rendered user messages,
the anchor sits further back, and the span is correspondingly larger.

Replicated here by taking the ids of user-role Messages from
`Transcript.build_llm_history()` (transcript vocabulary) and passing those to
task_boundary_floor over the rendered history — exactly the pairing session.py
uses.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("LO_REPO", os.path.expanduser("~/local-operator")))

from local_operator.compaction.api import CompactionSettings  # noqa: E402
from local_operator.compaction.cutpoint import task_boundary_floor  # noqa: E402
from local_operator.compaction.tokens import estimate_messages_tokens  # noqa: E402
from local_operator.harness.types import Message  # noqa: E402
from local_operator.session.session import (  # noqa: E402
    _TASK_FLOOR_KEEP_MULTIPLE,
    _default_convert_to_llm,
)
from local_operator.session.transcript import Transcript  # noqa: E402

PATH = os.path.expanduser("~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl")
lines = [line for line in open(PATH) if line.strip()]
objs = [json.loads(line) for line in lines]
tmp = tempfile.mkdtemp()

KEEP = CompactionSettings().keep_recent_tokens  # 20,000
OLD_CAP = 300_000  # max(keep, resolve_threshold_tokens(1M window) // 2)
NEW_CAP = KEEP * _TASK_FLOOR_KEEP_MULTIPLE  # 100,000


def at(n):
    with open(os.path.join(tmp, "transcript.jsonl"), "w") as handle:
        handle.writelines(lines[:n])
    raw = list(Transcript(tmp).build_llm_history())
    # The session's own discriminator: user-role entries in the TRANSCRIPT
    # vocabulary, before rendering turns injections into user messages.
    genuine = {m.id for m in raw if isinstance(m, Message) and m.role == "user"}
    return _default_convert_to_llm(raw), genuine


comp = [i for i, e in enumerate(objs) if e.get("type") == "compaction"]
print(
    f"keep_recent={KEEP:,}   old cap={OLD_CAP:,}   new cap={NEW_CAP:,} "
    f"(= {_TASK_FLOOR_KEEP_MULTIPLE} x keep_recent)"
)
print(
    f"{'pass':>6} {'hist_before':>11} {'task_span':>10} | {'keep_old':>8} {'keep_new':>8} "
    f"| {'retain_old':>10} {'retain_new':>10}"
)

rows = []
for ci in comp:
    before, genuine = at(ci)
    lb = estimate_messages_tokens(before)
    span = task_boundary_floor(before, genuine, cap=10**9)  # uncapped
    keep_old = max(KEEP, min(span, OLD_CAP))
    keep_new = max(KEEP, min(span, NEW_CAP))
    rows.append((ci, lb, span, keep_old, keep_new))
    print(
        f"{ci:>6} {lb:>11,} {span:>10,} | {keep_old:>8,} {keep_new:>8,} "
        f"| {100*keep_old/lb:>9.1f}% {100*keep_new/lb:>9.1f}%"
    )

changed = [r for r in rows if r[3] != r[4]]
print(f"\npasses the new cap CHANGES: {len(changed)}/{len(rows)}")
for ci, lb, span, ko, kn in changed:
    print(
        f"  pass {ci}: span {span:,} -> keep_recent {ko:,} becomes {kn:,} "
        f"({100*ko/lb:.1f}% -> {100*kn/lb:.1f}% of history kept verbatim)"
    )

spans = sorted(r[2] for r in rows)
print(f"\nmeasured task spans: min={spans[0]:,}  p50={spans[len(spans)//2]:,}  max={spans[-1]:,}")
for cap, name in ((80_000, "4x keep_recent"), (NEW_CAP, "5x keep_recent")):
    clipped = [s for s in spans if s > cap]
    print(f"  spans a {name} cap ({cap:,}) would clip: {len(clipped)}/{len(spans)} {clipped}")
