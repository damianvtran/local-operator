"""Receipt accuracy per compaction pass: shipped formula vs proportional.

For each of the ten compaction passes in the real session, replays the
transcript up to and past the pass, estimates the history either side with the
LOCAL estimator, and compares both candidate after-figures against the ground
truth: the provider's OWN next-reported ``context_tokens`` after the pass.

  current  = max(la, provB - max(0, lb - la))     # subtracts a local saving
                                                  # from a provider total
  proposed = max(la, round(provB * la / lb))      # scales the provider total
                                                  # by the local shrink ratio

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
    f"| {'proposed':>8} {'err':>7}"
)

current_errors = []
proposed_errors = []
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
    la += frames * (per_frame - IMAGE_TOKEN_ESTIMATE)

    current = max(la, pb - max(0, lb - la))
    proposed = max(la, round(pb * (la / lb)) if lb > 0 else la)
    current_errors.append(abs(current - pa))
    proposed_errors.append(abs(proposed - pa))
    print(
        f"{ci:>6} {pb:>7} {pa:>10} | {current:>8} {current - pa:>+8} "
        f"| {proposed:>8} {proposed - pa:>+7}"
    )

mae_current = sum(current_errors) / len(current_errors)
mae_proposed = sum(proposed_errors) / len(proposed_errors)
print(f"\nmean abs error  current={mae_current:>9,.0f}   proposed={mae_proposed:>9,.0f}")
