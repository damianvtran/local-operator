"""Building and appending ``supplement_v1`` journal rows (memo §2.4).

WHY A CUSTOM ENTRY AND NOT A MESSAGE ROW. ``Transcript.append_custom`` rows "never enter LLM
context": ``build_llm_history`` ignores them, so a supplement has zero prefix-cache effect and
zero compaction weight, and ``_render_for_compaction`` never sees one (so the
``cut_not_replayable`` hazard for non-persisted ids cannot arise). A ``CustomMessage`` row
would reach the model unless every renderer allow-list stayed shut -- one forgotten list away
from a leak. ``supplement_v1`` is therefore deliberately NOT in
``session._PERSISTABLE_CUSTOM_TYPES`` and never rendered by ``convert_to_llm``;
``tests/unit/supplements/test_context_parity.py`` pins that with a byte-for-byte request
comparison, which is the test that fails if a later change opens either door.

ACTIVITY CLOCK. The type is in ``transcript.BOOKKEEPING_CUSTOM_TYPES`` and rows are appended
with ``preserve_mtime=True``: a supplement lands AFTER the turn settled, and re-ranking a
session as freshly worked because a callout arrived would reorder the sidebar for nothing.

WHAT C1a WRITES. Files-only versions: ``components`` is always ``[]``. A files-only job has no
generator coming, so its row is TERMINAL (``done``) on the first write -- ``decided`` would be
read by a cold reader as "cancelled -- Retry" (the stale-row rule, ``contract.reader_disposition``)
under an answer that never had anything to retry. ``decided``/``queued`` appear only when a
generator is attached (lane C1b), through the same builders.

PATHS IN THE ROW ARE NEVER ABSOLUTE (memo S-R14): ``Candidate.path`` is cwd- or ``~/``-relative
already, and ``more[]`` carries the same form, capped at :data:`MORE_MAX`, behind the same
denylist because only candidates that survived the pre-filter can reach this module.
"""

import time
import uuid
from typing import Any, Final, Mapping

from local_operator.supplements.candidates import Candidate
from local_operator.supplements.contract import (
    MORE_MAX,
    SUPERSEDED_ERROR,
    SUPPLEMENT_CUSTOM_TYPE,
    SupplementDecision,
    SupplementDetails,
    SupplementFile,
)
from local_operator.supplements.decision import Decision

#: Bounds on strings that originate in the model or the filesystem, so one pathological row
#: cannot bloat the journal or a viewer's history page.
_ERROR_MAX_CHARS: Final = 300


def new_job_id() -> str:
    """12 lowercase hex characters, stable across the versions of one job (memo §2.4)."""
    return uuid.uuid4().hex[:12]


def file_entry(candidate: Candidate) -> SupplementFile:
    return SupplementFile(
        path=candidate.path,
        name=candidate.name,
        kind=candidate.kind,
        size_bytes=candidate.size_bytes,
        mtime=candidate.mtime,
        why=candidate.why,
    )


def decision_record(decision: Decision, *, skipped: str | None = None) -> SupplementDecision:
    return SupplementDecision(
        vendor=decision.vendor,
        files_p=dict(decision.files_p),
        graphics_p=decision.graphics_p,
        skipped=skipped,
    )


def build_details(
    *,
    anchor: str,
    job: str,
    version: int,
    state: str,
    decision: Decision,
    error: str = "",
    at: float | None = None,
) -> SupplementDetails:
    """One row's ``details``: the files half of the decision, ``components`` empty.

    ``files_more`` counts the files the decision QUALIFIED but the featured cap held back
    (never the ones it rejected), and ``more`` stores up to :data:`MORE_MAX` of their paths so
    a surface can disclose them ("N more"). Keys with nothing to say are omitted so a row stays
    byte-lean and readers keep tolerating rows from builds that wrote fewer keys.
    """
    details: dict[str, Any] = {
        "anchor": anchor,
        "job": job,
        "version": version,
        "state": state,
        "files": [file_entry(c) for c in decision.featured],
        "components": [],
        "decision": decision_record(decision),
        "at": time.time() if at is None else at,
    }
    if decision.more:
        details["files_more"] = len(decision.more)
        details["more"] = [c.path for c in decision.more[:MORE_MAX]]
    if error:
        details["error"] = error[:_ERROR_MAX_CHARS]
    return details  # type: ignore[return-value]


def next_version(
    details: Mapping[str, Any], *, state: str, error: str = ""
) -> SupplementDetails:
    """The follow-on row for the same ``anchor`` and ``job``: ``version + 1``, new state.

    Everything else is carried verbatim -- older versions stay in the journal as an audit trail
    and readers take the newest, so a terminal write must restate the files it still shows.
    """
    nxt: dict[str, Any] = dict(details)
    nxt["version"] = int(details["version"]) + 1
    nxt["state"] = state
    nxt["at"] = time.time()
    if error:
        nxt["error"] = error[:_ERROR_MAX_CHARS]
    return nxt  # type: ignore[return-value]


def superseded(details: Mapping[str, Any]) -> SupplementDetails:
    """The row that cuts a non-terminal job a newer user turn replaced (renders nothing)."""
    return next_version(details, state="cancelled", error=SUPERSEDED_ERROR)


async def append_row(transcript: Any, details: Mapping[str, Any]) -> None:
    """Journal one row. ``preserve_mtime=True``: see the module docstring."""
    await transcript.append_custom(SUPPLEMENT_CUSTOM_TYPE, dict(details), preserve_mtime=True)
