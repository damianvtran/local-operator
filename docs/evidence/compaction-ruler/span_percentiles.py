"""Pooled active-task span statistics behind _TASK_FLOOR_KEEP_MULTIPLE.

The multiple that caps the preserve window is chosen against every active-task
span measured on a real session: the seven recorded in
``compaction/cutpoint.py`` ``task_boundary_floor`` and the ten this PR measured
in ``retention_real2.py``. This prints the pooled distribution and what each
candidate multiple would clip, which is the whole basis for choosing 5.

Pure arithmetic over recorded measurements -- it reads no transcript, so it is
the one script here that runs anywhere.
"""

import math

#: The seven spans documented in ``cutpoint.task_boundary_floor``'s docstring.
CUTPOINT_SPANS = [300, 46_900, 48_800, 30_000, 19_800, 123_400, 49_100]

#: The ten measured by ``retention_real2.py`` on the 10-pass session, using the
#: session's own ``genuine_user_ids`` discriminator.
PR_SPANS = [469, 47_445, 53_732, 33_054, 22_561, 129_660, 49_779, 113_835, 10_879, 131_376]

KEEP_RECENT_DEFAULT = 20_000


def percentile(values, q):
    """Linear-interpolated percentile (the numpy default), so p50 is a median."""
    ordered = sorted(values)
    n = len(ordered)
    k = (n - 1) * q
    floor_index = math.floor(k)
    ceil_index = math.ceil(k)
    lower = ordered[floor_index]
    return lower + (k - floor_index) * (ordered[ceil_index] - lower)


pooled = CUTPOINT_SPANS + PR_SPANS
for name, spans in (
    ("cutpoint 7-pass", CUTPOINT_SPANS),
    ("this PR 10-pass", PR_SPANS),
    ("pooled", pooled),
):
    print(
        f"{name:<18} n={len(spans):>2} "
        f"p50={percentile(spans, 0.50):>9,.0f} "
        f"p75={percentile(spans, 0.75):>9,.0f} "
        f"p90={percentile(spans, 0.90):>9,.0f} "
        f"max={max(spans):>9,}"
    )

# The distribution is bimodal, which is what makes "between the clusters" the
# right way to choose the bound rather than a percentile.
lower = sorted(s for s in pooled if s < 80_000)
upper = sorted(s for s in pooled if s >= 80_000)
print(f"\nlower cluster: n={len(lower)} max={max(lower):,}")
print(f"upper cluster: n={len(upper)} min={min(upper):,}  {upper}")

print("\ncandidate multiples against the pooled spans:")
for multiple in (3, 4, 5, 6, 7):
    cap = KEEP_RECENT_DEFAULT * multiple
    clipped = sorted(s for s in pooled if s > cap)
    note = ""
    if clipped and min(clipped) < min(upper):
        note = "  <- clips an ORDINARY task"
    if cap > min(upper):
        note = "  <- lets an OUTLIER through"
    print(f"  {multiple}x -> cap {cap:>7,}  clips {len(clipped):>2}/{len(pooled)} {clipped}{note}")
