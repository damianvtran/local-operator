"""Score mapping: OSWorld's float evaluator -> a SCORED ScoreArtifact, or raise.

The contract is scored-or-raise (runner/episode.py:766-768). "Unscored" is a
HARNESS decision expressed via finalization intent, never an adapter return
value. OSWorld's own ``evaluate()`` does the opposite of what we need: it
swallows metric exceptions into ``0.0`` and logs a missing evaluator rather
than raising (desktop_env.py:594-700). Mapping "could not evaluate" to 0.0
would report a failure the agent did not commit — the exact score-deflation
error this module exists to prevent.

Binary completion and partial reward are distinct metrics (OSWorld 2.0 paper,
https://arxiv.org/html/2606.29537v1, sections 2.1.3 and 3.2). Only a full raw
score is binary completion: the pinned d578d2d4 upstream monitor/static/index.js
counts fullScoreTasks only when score === 1 (lines 265-270). Rounding partial
reward must never promote a near miss. The original evaluate() result is retained
as canonical JSON, including any safety/checkpoint/error fields, not reconstructed
from its rounded ppm.
A scalar result cannot supply checkpoint data the upstream evaluator discarded.
The evaluator's own scoring-path output -- its prints, its log records and the
state it fetched -- does not appear in that return value at all, so it travels in
the SAME detail artifact (see ``evaluator_diagnostics`` below) rather than in a
second artifact or a new protocol field.
Invalid or over-budget details fail closed rather than claiming full evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from local_operator.evaluation.evidence.models import EvidenceArtifactRef, ScoreArtifact

# Evaluator summaries are small. Bound traversal as well as encoded bytes before
# the parent's independent media/redaction checks (whose JSON ceiling is 32 MiB).
MAX_SCORE_DETAIL_BYTES = 1024 * 1024
MAX_SCORE_DETAIL_NODES = 100_000
MAX_SCORE_DETAIL_DEPTH = 64

# The two keys the retained-diagnostics detail is built from. Namespaced and
# fixed so a reader never has to guess which half is upstream's and which is
# ours, and so an evaluator returning a dict of its own can never collide with
# them: they are the WRAPPER's keys, and the evaluator's return value sits
# verbatim under ``evaluator_result``.
EVALUATOR_RESULT_KEY = "evaluator_result"
EVALUATOR_DIAGNOSTICS_KEY = "evaluator_diagnostics"

# What that key holds when a diagnostics block was produced but could not be
# attached within the limits above. One field, so it fits whatever the rejected
# block was, and a fixed value, so a reader can tell a refusal from a capture
# that genuinely retained nothing (an absent block).
DIAGNOSTICS_REFUSED: dict[str, str] = {
    "schema": "lop-evaluator-diagnostics-v1",
    "refused": "over_budget",
}


class ScoringUnavailable(RuntimeError):
    """The episode cannot be honestly scored. Raised, never returned as 0.0."""


class ScoringProtocolError(RuntimeError):
    """The evaluator score or its details violate the bounded JSON contract."""


def _detail_bytes(raw: Any) -> bytes:
    nodes = 0
    characters = 0

    def validate(value: Any, depth: int) -> None:
        nonlocal nodes, characters
        nodes += 1
        if nodes > MAX_SCORE_DETAIL_NODES or depth > MAX_SCORE_DETAIL_DEPTH:
            raise ScoringProtocolError("evaluator details exceed structural limits")
        if type(value) is str:
            characters += len(value)
            if characters > MAX_SCORE_DETAIL_BYTES:
                raise ScoringProtocolError("evaluator details exceed byte limit")
        elif type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ScoringProtocolError("evaluator details require JSON string keys")
                validate(key, depth + 1)
                validate(item, depth + 1)
        elif type(value) is list:
            for item in value:
                validate(item, depth + 1)
        elif value is not None and type(value) not in (bool, int, float):
            # No default=str or tuple/key coercions: those would misrepresent the
            # upstream result while presenting it as preserved JSON evidence.
            raise ScoringProtocolError("evaluator details contain a non-JSON value")

    validate(raw, 0)
    data = bytearray()
    encoder = json.JSONEncoder(
        allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    try:
        for chunk in encoder.iterencode(raw):
            encoded = chunk.encode("utf-8")
            if len(data) + len(encoded) > MAX_SCORE_DETAIL_BYTES:
                raise ScoringProtocolError("evaluator details exceed byte limit")
            data.extend(encoded)
    except (ValueError, OverflowError, UnicodeError, RecursionError) as error:
        # Encoder errors can quote raw values; only a fixed diagnostic crosses RPC.
        raise ScoringProtocolError("evaluator details are not finite UTF-8 JSON") from error
    return bytes(data)


def _detail_payload(raw: Any, diagnostics: Mapping[str, Any] | None) -> Any:
    """What is persisted for a score: the raw return, plus retained diagnostics.

    A capture that produced nothing (``None``) leaves ``raw`` untouched, which
    is what keeps an episode whose evaluator emitted nothing byte-identical to
    what this function staged before diagnostics existed.

    Diagnostics can never cost the score. The block is attached only if the
    wrapped payload still satisfies the same structural and byte limits every
    other detail obeys; a block that could not be attached is replaced by a
    one-field marker saying so, so that "no block" never has to mean both
    "nothing was emitted" and "something was refused". If even the marker does
    not fit -- unreachable, since the capture's own bounds cap the whole
    payload at well under a third of the byte limit -- the exact pre-change
    bytes are staged: an explanation must never cost the number it explains.
    """

    if not diagnostics:
        return raw
    wrapped = {EVALUATOR_RESULT_KEY: raw, EVALUATOR_DIAGNOSTICS_KEY: diagnostics}
    try:
        _detail_bytes(wrapped)
        return wrapped
    except ScoringProtocolError:
        pass
    refused = {EVALUATOR_RESULT_KEY: raw, EVALUATOR_DIAGNOSTICS_KEY: DIAGNOSTICS_REFUSED}
    try:
        _detail_bytes(refused)
        return refused
    except ScoringProtocolError:
        return raw


def score_to_artifact(
    raw: Any, *, artifact_root: Path, diagnostics: Mapping[str, Any] | None = None
) -> ScoreArtifact:
    """Map OSWorld's raw ``evaluate()`` return to a SCORED artifact, or raise.

    V2 may return a dict carrying ``{"score": float}`` (task_base.py:79-88)
    when a task overrides ``evaluate``; unwrap exactly that one key. Anything
    else that is not a real number in [0.0, 1.0] is a protocol violation, not
    a zero.

    ``diagnostics`` is the evaluator's own scoring-path output retained by
    ``lop_osworld_v2_adapter.diagnostics``. It changes nothing about which
    value is the score and nothing about the numeric result; it only widens
    what the detail artifact carries, so a reader can tell a genuinely unmet
    checkpoint from an evaluator that bailed out before checking one.
    """

    value = raw
    if isinstance(raw, dict):
        if "score" not in raw:
            raise ScoringProtocolError("evaluator returned a dict without a 'score' key")
        value = raw["score"]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringProtocolError("evaluator returned a non-numeric score")
    if isinstance(value, float) and not math.isfinite(value):
        raise ScoringProtocolError("evaluator returned NaN or infinite score")
    if not (0 <= value <= 1):
        raise ScoringProtocolError("evaluator score is outside [0.0, 1.0]")
    value = float(value)
    data = _detail_bytes(_detail_payload(raw, diagnostics))
    details = EvidenceArtifactRef(
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="application/json",
        byte_count=len(data),
    )
    # Like observations, these are worker-staged bytes, not trusted evidence.
    # The parent reopens by digest, verifies, and scans them before its receipt.
    try:
        fd = os.open(artifact_root / details.sha256, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass  # The parent's verifier rejects conflicting or non-regular entries.
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    return ScoreArtifact(
        status="scored",
        binary=1 if value == 1.0 else 0,
        partial_ppm=round(value * 1_000_000),
        details=details,
    )
