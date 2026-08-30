"""Do the three tokens_after CONSUMERS change their decision under the fix?

The hand-modelled controlflow.py approximated the advisor rule as
(threshold - after)/before; the shipped rule is
cleared_headroom(after, outcome.tokens_before) / before, i.e. the fraction of
the STARTING context the pass freed. This version calls the real
resolve_threshold_tokens / RECOVERY_BAND / cleared_headroom so the check is
against the code, not a paraphrase of it.

Consumers, per session.py:
  (1) _held_context_tokens  (session.py ~5473)  -- display carrier, no branch
  (2) RECOVERY_BAND auto-continue gate (~5482): after <= 0.8 * threshold
  (3) advisor reclaim kill switch (~5906): cleared_headroom(after, before)/before
                                            >= _ADVISOR_MIN_RECLAIM_FRACTION
"""

import os
import sys

sys.path.insert(0, os.environ.get("LO_REPO", os.path.expanduser("~/local-operator")))

from local_operator.compaction.api import (  # noqa: E402
    CompactionSettings,
    cleared_headroom,
    resolve_threshold_tokens,
)
from local_operator.compaction.thresholds import RECOVERY_BAND  # noqa: E402
from local_operator.session.session import _ADVISOR_MIN_RECLAIM_FRACTION  # noqa: E402

# ci, provB (=plan.context_tokens), REAL_after, lb (=plan.tokens_before), la
ROWS = [
    (1050, 440417, 117948, 246989, 69473),
    (2123, 596569, 150000, 344395, 94311),
    (2929, 359569, 85869, 284328, 71287),
    (3998, 511300, 157832, 471776, 98332),
    (4660, 600031, 157562, 344742, 99774),
    (5439, 600050, 161098, 338613, 99855),
    (6865, 590412, 87684, 533774, 77691),
    (7979, 600035, 130702, 322812, 78090),
    (8982, 586520, 127835, 341986, 78604),
    (9954, 546458, 311220, 318084, 190579),
]

WINDOW = 1_000_000
SETTINGS = CompactionSettings()
THRESHOLD = resolve_threshold_tokens(WINDOW, SETTINGS)


def shipped(pb, lb, la):
    return max(la, pb - max(0, lb - la))


def proposed(pb, lb, la):
    return max(la, round(pb * la / lb)) if lb > 0 else la


print(
    f"threshold={THRESHOLD:,}  RECOVERY_BAND={RECOVERY_BAND}  "
    f"min_reclaim={_ADVISOR_MIN_RECLAIM_FRACTION}"
)
print(
    f"{'pass':>6} {'old_after':>9} {'new_after':>9} | {'cont_old':>8} {'cont_new':>8} "
    f"| {'recl_old':>8} {'recl_new':>8} {'adv_old':>7} {'adv_new':>7}"
)

flips_cont = flips_adv = 0
for ci, pb, _real, lb, la in ROWS:
    old, new = shipped(pb, lb, la), proposed(pb, lb, la)
    # (2) auto-continue
    cont_old = old <= RECOVERY_BAND * THRESHOLD
    cont_new = new <= RECOVERY_BAND * THRESHOLD
    # (3) advisor kill switch, measured against the pass's own "before"
    before = pb
    r_old = cleared_headroom(old, before) / before
    r_new = cleared_headroom(new, before) / before
    adv_old = r_old >= _ADVISOR_MIN_RECLAIM_FRACTION
    adv_new = r_new >= _ADVISOR_MIN_RECLAIM_FRACTION
    flips_cont += cont_old != cont_new
    flips_adv += adv_old != adv_new
    print(
        f"{ci:>6} {old:>9,} {new:>9,} | {str(cont_old):>8} {str(cont_new):>8} "
        f"| {r_old:>8.2f} {r_new:>8.2f} {str(adv_old):>7} {str(adv_new):>7}"
    )

print("\n(1) _held_context_tokens: display carrier only, no branch on the value.")
print(f"(2) auto-continue decision flips: {flips_cont}/{len(ROWS)}")
print(f"(3) advisor kill-switch flips:    {flips_adv}/{len(ROWS)}")
print(
    "\nDirection note: the proposed after-figure is always <= the shipped one on"
    "\nthis data, so both gates can only move toward 'the pass helped' — neither"
    "\ncan newly disable the advisor nor newly suppress a continuation."
)
