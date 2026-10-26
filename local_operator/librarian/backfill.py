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

``op="tags"`` in a call that names the asset is the same proof for her tag
assignments — but the value that has to match is the TAG, and it is matched
STRUCTURALLY, against the values the call PASSES, never against the text it
happens to contain.

- A native ``library`` call holds its arguments as data: ``op`` and ``action``
  say it assigns tags, ``asset_id`` says which asset, ``tags``/``primary`` are
  the values it assigns.
- An eval cell is text, so it is parsed and its string constants are read —
  minus the call's own plumbing: the tool's own name in ``tool("library", …)``,
  and the values of keyword arguments named ``op``, ``action`` and ``i``, which
  say what a call is DOING rather than what it assigns. Dict KEYS are
  deliberately not excluded: her cells hold the tag name in one
  (``idm={"eng-3708": "f8957d3b32"}``).

Why not the text: a substring test claims an ordinary agent's tag for her
whenever the name occurs anywhere in the cell. Measured on the live install,
that moved agent tags named ``a``, ``op``, ``asset`` and ``library`` — the last
of them the tool's own name, the one before it part of an argument name — on
exactly the class of asset this pass exists for. Whole-string equality against
the values a call passes is what bounds it: a renamed tag stops matching, while
an id literal (``"f8957d3b32"``) still matches. Either form counts — name or id
— because her cells write the NAME and resolve the id at runtime
(``idm={t['name']:t['id'] for t in L(op="tags",action="list")['tags']}``), so an
id-only match reaches 2 of the 37 assignments on this install. A cell that does
not parse contributes NOTHING: the direction is to leave a row an agent's rather
than claim it.

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

import ast
import json
import logging
import re
import time
from dataclasses import dataclass
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

#: The tag op as it appears INSIDE an eval cell's code: ``op="tags"``,
#: ``op='tags'``, and the dict forms ``{"op": "tags"}`` / ``'op': 'tags'``. A
#: native call carries the same op as an argument instead, which
#: :func:`_is_tag_call` reads structurally. Matching one spelling would silently
#: under-recover on a journal written in another, and the miss is invisible: the
#: rows simply stay an agent's.
_TAG_OP_IN_CODE = re.compile(r"""\bop\b["']?\s*[=:]\s*["']tags["']""")

#: Keyword arguments that are a call's own plumbing rather than a tag: ``op`` and
#: ``action`` name the operation and ``i`` is the intent line, so their VALUES
#: must not be read as tag names — a tag that happens to be called ``tags`` is
#: not hers because a call says ``op="tags"``.
_TOOL_KEYWORDS = ("op", "action", "i")


@dataclass(frozen=True)
class _TagEvidence:
    """What ONE journal call shows about a tag assignment.

    ``assets`` are the asset ids the call's tags op addresses, ``values`` the
    strings it passes as tag values. Both are compared by WHOLE-STRING equality,
    which is the difference that matters: a substring test over the call's text
    claimed an ordinary agent's tag named ``a`` — or ``library``, the tool's own
    name — for her whenever the name occurred anywhere in the cell (QA round 3,
    cell (d)).
    """

    assets: frozenset[str]
    values: frozenset[str]


def _escaped(value: str) -> str:
    """``value`` as it appears inside a serialized tool call, without quotes.

    Quotes stay the caller's business: an argument reaches a journal either as
    part of a JSON object (plain quotes) or inside an eval cell's code string
    (escaped ones), so only the escaped CONTENT is form-independent. That is
    also why the tag op is read structurally rather than by one literal.
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
            except ValueError:
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


def _is_tag_call(call: dict[str, Any]) -> bool:
    """Whether a journal call is a ``tags`` op, in EITHER of the two forms.

    A native ``library`` call holds the op as an argument; an eval cell holds it
    as text inside its code. Missing the eval form would miss nearly every write
    she made, because that is the route her ``tool()`` lines take.
    """
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    if arguments.get("op") == "tags":
        return True
    code = arguments.get("code")
    return isinstance(code, str) and _TAG_OP_IN_CODE.search(code) is not None


def _native_tag_evidence(arguments: dict[str, Any]) -> _TagEvidence | None:
    """The tags a NATIVE ``library`` call assigns, or ``None``.

    A native call holds its arguments as data: the op and action say whether it
    assigns tags at all, ``asset_id`` says which asset, and ``tags``/``primary``
    are the values it assigns. A clear is not an assignment, and a call with no
    asset cannot address one.
    """
    if arguments.get("op") != "tags" or arguments.get("action") != "set":
        return None
    asset_id = arguments.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        return None
    named = arguments.get("tags")
    values = (
        {value for value in named if isinstance(value, str)} if isinstance(named, list) else set()
    )
    primary = arguments.get("primary")
    if isinstance(primary, str):
        values.add(primary)
    return _TagEvidence(frozenset({asset_id}), frozenset(values))


def _cell_tag_evidence(code: str) -> _TagEvidence | None:
    """The tag values an eval CELL passes, or ``None`` when it proves nothing.

    The cell is TEXT, so it is parsed and its string constants are read, minus
    the call's own plumbing (:data:`_TOOL_KEYWORDS` and the tool's own name). A
    name that merely occurs inside a sentence, or inside another name, is not
    evidence that she assigned it.

    ``None`` on a cell that does not parse: the pass cannot read it, and the safe
    direction is to leave its rows an agent's rather than claim them from text it
    only half understands.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        # ValueError: a NUL byte in the source, which ``ast.parse`` refuses.
        return None
    plumbing: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "tool" and node.args:
            # ``tool("library", …)`` — the first argument names the tool.
            plumbing.add(id(node.args[0]))
        for keyword in node.keywords:
            if keyword.arg in _TOOL_KEYWORDS:
                plumbing.add(id(keyword.value))
    values = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in plumbing
    }
    # An eval cell holds its asset ids among the same constants, and the per-row
    # test for "this call names that asset" is membership: one set serves both.
    return _TagEvidence(frozenset(values), frozenset(values))


def _tag_evidence(calls: list[dict[str, Any]]) -> list[_TagEvidence]:
    """What every one of her journal calls shows about tag assignments.

    The two shapes a journal holds are read structurally and separately, because
    they hold their arguments differently: a native call as a dict, an eval cell
    as the text of its code.
    """
    evidence: list[_TagEvidence] = []
    for call in calls:
        if not _is_tag_call(call):
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            continue
        code = arguments.get("code")
        if call.get("name") == "eval" and isinstance(code, str):
            found = _cell_tag_evidence(code)
        else:
            found = _native_tag_evidence(arguments)
        if found is not None:
            evidence.append(found)
    return evidence


def _attributions(store: MetadataStore, calls: list[dict[str, Any]]) -> list[Attribution]:
    """The rows those calls prove are hers, with what changes on each."""
    #: The serialized call TEXT, which is what the name and description halves
    #: match against: a value she wrote is evidence whoever else quotes it. The
    #: tag half reads the same calls STRUCTURALLY instead (see
    #: :func:`_tag_evidence`), because a tag name is short enough that a
    #: substring test claims names that are merely present in the text.
    texts = [json.dumps(call) for call in calls]
    assigned_calls = _tag_evidence(calls)
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
        # The calls that BOTH name this asset and pass a tag VALUE: the same call a
        # tag has to be passed in for its assignment to be hers. A set another
        # agent replaced afterwards passes neither its name nor its id, so those
        # rows keep their agent provenance.
        assigned = [call for call in assigned_calls if asset_id in call.assets]
        tag_ids = tuple(
            tag["id"]
            for tag in store.asset_tags(asset_id)
            if tag["source"] == "agent"
            and any(tag["name"] in call.values or tag["id"] in call.values for call in assigned)
        )
        attributions.append(Attribution(asset_id, fields, at, tag_ids=tag_ids))
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
