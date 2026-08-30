"""Receipt accuracy per compaction pass: shipped formula vs proportional.

For each of the ten compaction passes in the real session, replays the
transcript up to and past the pass, estimates the history either side with the
LOCAL estimator, and compares both candidate after-figures against the ground
truth: the provider's OWN next-reported ``context_tokens`` after the pass.

  current  = max(la, provB - max(0, lb - la))     # subtracts a local saving
                                                  # from a provider total
  shipped  = the arithmetic in ``Session._run_compaction``, mirrored exactly

The ``shipped`` column models the WHOLE of that arithmetic, not just the
ratio, because the parts interact:

  * the frame correction is added AFTER the division, never folded into the
    numerator. It is a PROVIDER-scale addend and the ratio's soundness rests
    on numerator and denominator being on one LOCAL ruler; folding it in
    inflates the ratio and then multiplies the addend by the provider total a
    second time (agent review round 1, major-2). An earlier revision of THIS
    script did exactly that, and so measured a formula the codebase does not
    ship (round 3, minor-2);
  * the ratio applies only when ``la < lb``. Snapcompact can leave the local
    estimate larger than it found it, and a ratio above 1 multiplied against a
    provider total reports a compaction that grew the context (round 2,
    blocker-1);
  * the result is clamped to ``provB``, the upper bound the subtraction form
    had for free and a product does not.

Keep this function in step with ``Session._run_compaction``. A divergence here
does not break a test; it silently republishes an accuracy figure for a
formula nobody runs.

Run with ``LO_REPO=<repo-root>`` if the checkout is not at ~/local-operator.
The transcript is opened read-only and never written.
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

#: The local ruler prices every image at a flat estimate while the provider
#: bills an Anthropic 1932px frame at ~5,000 visual tokens; the after-figure
#: carries the same correction the shipped code applies.
IMAGE_TOKEN_ESTIMATE = 1200


def local_at(n):
    """Local estimate of the history as of the first ``n`` transcript records."""
    with open(os.path.join(tmp, "transcript.jsonl"), "w") as handle:
        handle.writelines(lines[:n])
    return estimate_messages_tokens(
        _default_convert_to_llm(list(Transcript(tmp).build_llm_history()))
    )


def ctx(entry):
    return (entry.get("payload", {}).get("usage") or {}).get("context_tokens")


comp = [i for i, e in enumerate(objs) if e.get("type") == "compaction"]
print(
    f"{'pass':>6} {'provB':>7} {'REAL_after':>10} | {'current':>8} {'err':>8} "
    f"| {'shipped':>8} {'err':>7}"
)

current_errors = []
shipped_errors = []
for ci in comp:
    # The provider figure the pass acted on, and the one it reported next.
    pb = next((ctx(objs[j]) for j in range(ci - 1, -1, -1) if ctx(objs[j])), None)
    pa = next((ctx(objs[j]) for j in range(ci + 1, len(objs)) if ctx(objs[j])), None)
    if not pb or not pa:
        continue
    lb = local_at(ci)
    la = local_at(ci + 1)
    archive = objs[ci]["payload"]["preserve_data"]["snapcompact"]
    frames = len(archive.get("frames") or [])
    per_frame = 5000 if "1932" in archive.get("shape_id", "") else 3293
    correction = frames * (per_frame - IMAGE_TOKEN_ESTIMATE)

    # The pre-PR form, which folds the correction into the after-figure and
    # subtracts a LOCAL saving from a PROVIDER total.
    current = max(la + correction, pb - max(0, lb - (la + correction)))

    # The shipped form: ratio on the UNCORRECTED local figures, correction
    # added afterwards, whole result clamped to the figure it reduces from.
    if lb > 0 and la < lb:
        shipped = max(la, round(pb * la / lb))
    else:
        shipped = pb
    shipped = min(pb, shipped + correction)

    current_errors.append(abs(current - pa))
    shipped_errors.append(abs(shipped - pa))
    print(
        f"{ci:>6} {pb:>7} {pa:>10} | {current:>8} {current - pa:>+8} "
        f"| {shipped:>8} {shipped - pa:>+7}"
    )

mae_current = sum(current_errors) / len(current_errors)
mae_shipped = sum(shipped_errors) / len(shipped_errors)
print(f"\nmean abs error  pre-PR={mae_current:>9,.0f}   shipped={mae_shipped:>9,.0f}")
