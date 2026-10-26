"""One-shot re-attribution of the Librarian's own edits, from her transcript.

WHY A BACKFILL EXISTS AT ALL. Until this change ``library``'s ``actor``
defaulted to ``agent`` and the Librarian's own calls never passed one, so four
days of her naming, describing and tagging were recorded as an ordinary agent's
work and ``organized_at`` was never stamped for any of it (on this install: 39
assets, 37 tag assignments, 0 organized). The store cannot be repaired by
inference from itself — an agent's row and hers are identical in it, and the
reason they are identical is the bug — so the evidence comes from the one place
that kept it: her own journal.

WHAT COUNTS AS PROOF, and why it is this narrow. A row moves only when a SINGLE
assistant ``tool_calls`` entry of her transcript contains BOTH the asset id and
the field's exact current value (JSON-escaped — the form the value takes inside
a serialized call). Requiring both is what makes the pass self-limiting:

- a field another agent overwrote afterwards no longer matches, so the row keeps
  its agent provenance: this pass cannot claim an ordinary agent's work;
- a value matches only while it still stands, which is exactly the condition
  under which the store's own precedence would have kept her copy anyway;
- so a rerun moves nothing new, whatever the guard says.

WHAT THIS PASS DOES NOT RE-ATTRIBUTE: ``asset_tags.source``. Her 37 pre-fix
tag assignments stay an agent's, and the label is DISPLAY PROVENANCE only —
``set_asset_tags`` replaces a set with no rank check, and nothing decides
anything from the column: the ``asset_tags()`` projection renders it, and no
module outside this file reads it. The bug this pass exists for is
``organized``, which the naming half restores in full (39 of 39 here).

Inferring tag provenance from the journal was attempted, four rounds of it, and
every rule that reached the 37 assignments also claimed an ordinary agent's work:
a substring match took an agent's tag called ``a`` — or ``library``, the tool's
own name — an id-only match reached 2 of 37, and reading the constants a cell
PASSES still pools a cell's values across every asset it names and cannot tell a
tag value from a result key (``d["tags"]``). Closing the last of those needs
data-flow analysis, which is out of proportion to a cosmetic label: a wrong
attribution is worse than an agent's.

WHAT IT DELIBERATELY DOES NOT DO.

- No schema bump. A migration cannot read a transcript, and a database whose
  ``schema_version`` is newer than the code makes a downgraded ``lop`` answer an
  EMPTY catalog (``metadata.py``: ``version > SCHEMA_VERSION``).
- No ``edits`` rewrite. That table is the audit log of what callers did, and
  this pass re-reads it rather than adding to it; ``organized_at`` therefore
  carries her ORIGINAL timestamp, not the clock of this pass.
- ``tags.created_by`` and ``conventions.created_by`` stay as they are (cosmetic
  attribution, listed as not addressed in the PR). ``confidence`` stays NULL:
  recovering it means parsing it out of code, which is fragile.

The pass runs once per install, gated by the ``state`` fact :data:`GUARD_KEY`,
which is written when it FINISHES — including when it finishes with nothing to
move — so every later boot costs one read.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from local_operator.library.metadata import Attribution, MetadataStore

logger = logging.getLogger(__name__)

#: The ``state`` fact that makes this pass one-shot. Versioned in its NAME so a
#: future pass over wider evidence is not skipped by this one's mark, and read
#: through the same small-durable-fact pattern ``primary_tag:`` uses.
GUARD_KEY = "backfill:librarian_organized:v1"

#: The fields a call in her transcript can prove she wrote.
_FIELDS = ("name", "description")


def _escaped(value: str) -> str:
    """``value`` as it appears inside a serialized tool call, without quotes.

    Quotes stay the caller's business: an argument reaches a journal either as
    part of a JSON object (plain quotes) or inside an eval cell's code string
    (escaped ones), so only the escaped CONTENT is form-independent.
    """
    return json.dumps(value)[1:-1]


def _calls(config: Path) -> list[dict[str, Any]]:
    """Every assistant tool call in her journal that mentions the Library.

    Her journal is resolved through ``session_id_of`` and the archive-aware
    reader (a compressed session must be read through its materialised copy,
    never by looking for ``transcript.jsonl`` by name). Malformed and blank rows
    are skipped, exactly as every other transcript reader does: a torn tail must
    cost this pass a row, not a boot.
    """
    from local_operator.librarian import state
    from local_operator.session import compression

    session_id = state.session_id_of(config)
    if not session_id:
        return []
    # ``<config>/sessions/<id>`` is the shape the session factory and the
    # bootstrap both build; there is no shared helper for it to call.
    #
    # ``read_only=True``: this pass only ever READS. Without it, an archive
    # whose cache copy cannot be made falls back to ``restore_for_write``,
    # which re-inflates her transcript in place without the writer lock — a
    # pure reader re-inflating the very journal the pass compressed (see
    # ``compression.transcript_for_read``).
    path = compression.transcript_for_read(config / "sessions" / session_id, read_only=True)
    if path is None:
        return []
    calls: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "library" not in line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, RecursionError):
                # ValueError: a torn tail. RecursionError: a pathological line
                # nested deeper than the decoder's stack — a 20,000-term
                # concatenation cell was measured raising it out of the parse of
                # its own code. Either way this row is evidence the pass cannot
                # read, and it costs the pass a row, never a boot.
                continue
            if not isinstance(row, dict) or row.get("type") != "message":
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict) or payload.get("role") != "assistant":
                continue
            for call in payload.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                if "library" in json.dumps(call):
                    calls.append(call)
    return calls


def _attributions(store: MetadataStore, calls: list[dict[str, Any]]) -> list[Attribution]:
    """The rows those calls prove are hers, with what changes on each."""
    #: The serialized call TEXT, which is what this pass matches against: a value
    #: she wrote is evidence whoever else quotes it.
    texts = [json.dumps(call) for call in calls]
    attributions: list[Attribution] = []
    for row in store.unorganized_agent_assets():
        asset_id = row["asset_id"]
        naming = [text for text in texts if _escaped(asset_id) in text]
        if not naming:
            continue
        fields = tuple(
            field
            for field in _FIELDS
            if row[f"{field}_source"] == "agent"
            and isinstance(row[field], str)
            and any(_escaped(row[field]) in text for text in naming)
        )
        if not fields:
            continue
        at = store.last_edit_at(asset_id, actor="agent", fields=_FIELDS)
        if at is None:
            # Her naming edit is what organized the asset; only if the audit log
            # holds no agent row for either field does any agent edit do, so the
            # stamp stays the moment she acted rather than a later favorite's.
            at = store.last_edit_at(asset_id, actor="agent")
        if at is None:
            # The value stands but the journal has no actor row for it, so there
            # is no honest timestamp to stamp. Leave the row an agent's.
            logger.debug("library backfill: no edit row for %s; leaving it alone", asset_id)
            continue
        attributions.append(Attribution(asset_id, fields, at))
    return attributions


def reattribute_organized(config_dir: str | Path) -> int:
    """Move the Librarian's own un-recorded work to her name. Returns rows moved.

    Called best-effort from :func:`local_operator.librarian.bootstrap.ensure_session`
    once per process. A missing transcript or an install that never had her is
    not an error: both mean there is nothing to re-read, and the guard is still
    written so the next boot answers from one read.
    """
    config = Path(config_dir)
    store = MetadataStore(config)
    if store.state_value(GUARD_KEY) is not None:
        return 0
    attributions = _attributions(store, _calls(config))
    return store.apply_attributions(
        attributions,
        guard_key=GUARD_KEY,
        # ``matched`` counts the rows the evidence selected, written in the same
        # transaction that moves them, so the fact and the store cannot disagree.
        guard_value=json.dumps(
            {"at": time.time(), "matched": len(attributions)}, separators=(",", ":")
        ),
    )
