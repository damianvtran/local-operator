"""The preserve-window cap across the shipped registry's context windows.

Agent review round 1 (blocker-1) found that a cap expressed purely in
``keep_recent_tokens`` multiples is independent of the model's capacity, and so
LOOSENS the preserve window on every model below the crossover -- on a 32k or
64k model a flat 100,000 exceeds the entire context window, which is the
"never compact" state the cap exists to prevent.

This prints the three candidate rules against the real
``resolve_threshold_tokens``, plus the registry's own window distribution, so
the crossover and the size of the affected population are checkable rather
than asserted.

Reads no transcript; needs only the package importable.
"""

import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.environ.get("LO_REPO", os.path.expanduser("~/local-operator")))

from local_operator.compaction.api import (  # noqa: E402
    CompactionSettings,
    resolve_threshold_tokens,
)
from local_operator.compaction.thresholds import resolve_threshold_percent  # noqa: E402
from local_operator.session.session import _TASK_FLOOR_KEEP_MULTIPLE  # noqa: E402

SETTINGS = CompactionSettings()
KEEP = SETTINGS.keep_recent_tokens

WINDOWS = [32_768, 64_000, 128_000, 131_072, 200_000, 256_000, 400_000, 1_000_000]

print(f"keep_recent={KEEP:,}  task term = keep_recent x {_TASK_FLOOR_KEEP_MULTIPLE}")
print(
    f"{'window':>10} {'threshold':>10} {'shipped':>9} {'flat-only':>10} "
    f"{'this PR':>9}  effect vs shipped"
)
for window in WINDOWS:
    threshold = resolve_threshold_tokens(window, SETTINGS)
    shipped = max(KEEP, threshold // 2)
    flat = max(KEEP, KEEP * _TASK_FLOOR_KEEP_MULTIPLE)
    fixed = max(KEEP, min(KEEP * _TASK_FLOOR_KEEP_MULTIPLE, threshold // 2))
    if fixed == shipped:
        effect = "unchanged"
    elif fixed < shipped:
        effect = f"tighter by {shipped - fixed:,} (the win)"
    else:
        effect = "LOOSER"
    warn = "  <- flat cap EXCEEDS THE WINDOW" if flat > window else ""
    print(
        f"{window:>10,} {threshold:>10,} {shipped:>9,} {flat:>10,} " f"{fixed:>9,}  {effect}{warn}"
    )

# How much of the shipped registry sits below the crossover, measured from the
# source rather than quoted: the argument only matters if the affected
# population is large.
root = os.environ.get("LO_REPO", os.path.expanduser("~/local-operator"))
grep = subprocess.run(
    ["grep", "-rhoE", "context_window=[0-9_]+", os.path.join(root, "local_operator")],
    capture_output=True,
    text=True,
)
digits = [
    m.group(1).replace("_", "")
    for m in (re.match(r"context_window=([0-9_]+)", line) for line in grep.stdout.split())
    if m
]
values = sorted(int(d) for d in digits if d.isdigit() and int(d) > 1_000)
if values:
    # The crossover is where ``threshold // 2`` first reaches the task term,
    # i.e. threshold == 2 * task_cap. Expressed in WINDOW tokens, since that is
    # what the label says and what a reader compares a model against: the
    # threshold is ``threshold_percent`` of the window, so the window crossover
    # is the threshold crossover divided by that fraction (200,000 / 0.8 =
    # 250,000). Reporting the threshold figure under a "of window" label put
    # 200,000 next to a table whose own 200,000 row reads "unchanged"
    # (agent review round 2, nit-1).
    threshold_crossover = KEEP * _TASK_FLOOR_KEEP_MULTIPLE * 2
    crossover = round(threshold_crossover / resolve_threshold_percent(SETTINGS))
    below = [v for v in values if resolve_threshold_tokens(v, SETTINGS) // 2 < KEEP * 5]
    print(
        f"\nregistry context_window entries: {len(values)}  "
        f"median={statistics.median(values):,.0f}"
    )
    print(
        f"below the crossover (capacity term binds): {len(below)} "
        f"({100 * len(below) / len(values):.0f}%)  crossover ~= {crossover:,} tokens of window"
    )
