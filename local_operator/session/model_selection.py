"""Conversation model identity, shared by owners and runtime-less viewers.

The journal is authoritative, not config.yml and not the effective fallback
route. Version 2 rows own the initial selection as well as explicit switches.
Legacy writers only journalled switches; their later frontend checkpoints can
therefore be newer evidence of the primary they actually selected on resume.
Reading never creates a directory or migrates a transcript: only its leased
owner may append the upgraded selection when it admits real work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

# The backward row walker is IMPORTED rather than re-written here. Its job —
# splitting a journal into rows while reading toward the head, including a row
# larger than one chunk and a row whose bytes straddle a boundary — is subtle
# enough that transcript.py already shares ONE implementation between the page
# reader and the latest-custom reader for exactly that reason. A second copy in
# this module would be a second thing to keep in step with it, and the failure
# mode of getting it wrong is a silently mis-parsed row rather than an error.
from local_operator.session.transcript import _iter_complete_lines_backward

SELECTED_MODEL_CUSTOM_TYPE = "selected_model"
SELECTION_VERSION = 2


@dataclass(frozen=True)
class StoredModelSelection:
    provider: str
    model_id: str
    effort: str | None = None
    authoritative: bool = False
    recovered: bool = False
    # This is the conversation's birth selector, not the pair configured while
    # constructing a resumed owner. Existing /effort journalling depends on it.
    boot_selector: str | None = None

    @property
    def selector(self) -> str:
        return f"{self.provider}/{self.model_id}"


def _selection(selector: Any, effort: Any, *, authoritative: bool = False, boot: Any = None):
    if not isinstance(selector, str) or "/" not in selector:
        return None
    provider, model_id = selector.split("/", 1)
    if not provider or not model_id:
        return None
    from local_operator.providers.registry import (
        get_provider_definition,
        is_decision_only,
    )

    if get_provider_definition(provider) is None:
        return None
    # A provider that can serve no chat completion is not a selection this reader
    # may hand out, for the same reason the provider registry refuses one as a
    # hosting: TypeSafe's Jev rejects ``chat/completions`` on every host we reach
    # it through (``ProviderDefinition.decision_only``), so honouring the row would
    # resume a conversation onto a model that 400s every turn. Refused HERE, in the
    # one validator both owner-side readers go through —
    # ``session_factory.resolve_hosting_model_with_source`` and
    # ``Session._restore_selected_model`` — rather than only in the resolver, which
    # the second of those does not consult. A v2 row that reaches this therefore
    # reads as UNUSABLE (``recovered``), which is the existing "we cannot honour
    # what the journal says" outcome the callers already handle: the next turn
    # writes a selection that does describe a running session. The refusal is not
    # silent for a RESUME: ``refused_decision_only_selection`` below lets the
    # resolver name the provider and the remedy instead of falling back quietly.
    if is_decision_only(provider):
        return None
    return StoredModelSelection(
        provider,
        model_id,
        effort if isinstance(effort, str) and effort else None,
        authoritative,
        boot_selector=(
            boot if isinstance(boot, str) and "/" in boot and all(boot.split("/", 1)) else None
        ),
    )


def selection_from_payloads(payloads: Iterable[dict[str, Any]]) -> StoredModelSelection | None:
    """Resolve CUSTOM-entry payloads; both callers enforce the envelope first."""
    authoritative = None
    legacy = None
    unusable = False
    for payload in payloads:
        kind = payload.get("custom_type")
        details = payload.get("details")
        if not isinstance(details, dict):
            continue
        if kind == SELECTED_MODEL_CUSTOM_TYPE:
            version = details.get("version")
            selected = _selection(
                details.get("selector"),
                details.get("effort"),
                authoritative=version == SELECTION_VERSION,
                boot=details.get("boot"),
            )
            if selected is not None:
                if selected.authoritative:
                    authoritative = selected
                    unusable = False
                elif version is None:
                    legacy = selected
            elif version == SELECTION_VERSION or (version is None and authoritative is None):
                unusable = True
        elif kind == "frontend_state_checkpoint_v1":
            state = details.get("state")
            model = state.get("selected_model") if isinstance(state, dict) else None
            if isinstance(model, dict):
                selected = _selection(
                    f"{model.get('provider', '')}/{model.get('model_id', '')}",
                    model.get("reasoning_effort"),
                )
                if selected is not None:
                    # A checkpoint refreshes the observed primary, not its
                    # birth. Retain known provenance only for the SAME primary;
                    # an abandoned old switch cannot lend its birth to a new one.
                    if legacy is not None and legacy.selector == selected.selector:
                        selected = replace(selected, boot_selector=legacy.boot_selector)
                    legacy = selected
    selected = authoritative or legacy
    return replace(selected, recovered=unusable) if selected is not None else None


def _forward_payloads(directory: Path) -> Iterable[dict[str, Any]]:
    """Every custom payload in the journal, oldest first — the original scan.

    Kept because it is the answer for a journal that holds no valid
    ``version == 2`` row at all (the backward scan settles all but 103 of the
    operator's 5,527 transcripts; re-counted by the whole-store differential,
    which is also where this fold's equality with the scan is checked): those
    predate v2 writers, so their selection can come from a legacy
    row OR from a `frontend_state_checkpoint_v1` row that is NEWER than it, and
    only a fold over the whole file settles which. Costs exactly what it always
    cost, on exactly the population that has always paid it.

    The SAME rows are what :func:`refused_decision_only_selection` needs, which
    is why the two readers are stated together rather than each growing its own
    idea of which rows count — that reader walks them BACKWARD instead (it runs on
    exactly the journals whose fold here is expensive), and this is the shape it
    must stay in step with.
    """
    try:
        with (directory / "transcript.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                # Large transcripts mostly contain messages. Do not parse
                # their payloads just to recover a two-field selector.
                if '"custom_type"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if (
                    isinstance(row, dict)
                    and row.get("type") == "custom"
                    and isinstance(row.get("payload"), dict)
                ):
                    yield row["payload"]
    except (OSError, UnicodeError):
        return


def _settled_selection(directory: Path) -> StoredModelSelection | None:
    """The newest VALID ``version == 2`` row's selection, found from EOF.

    WHY BACKWARD, and why it is exact. :func:`selection_from_payloads` folds
    rows oldest-first, and the fold has one property that makes reading it from
    the end unnecessary: **a valid v2 row resets ``unusable`` and replaces
    ``authoritative``**, and the result is ``authoritative or legacy`` — so
    once ANY valid v2 row exists, the answer is the NEWEST one of them and
    ``legacy`` (every version-less row, and every
    ``frontend_state_checkpoint_v1`` row, which is what a legacy writer
    journalled its switches through) is discarded outright. Everything BELOW
    that row cannot change either field any more:

    - ``authoritative`` — only another valid v2 row above the settled one
      replaces it, and there is none by construction;
    - ``recovered`` — set only by a row at or above the settled one, because
      every valid v2 row clears it and only an INVALID v2 row, or a
      version-less row seen while no v2 row has been read, can raise it. The
      second case cannot arise above a settled v2 row.

    So the scan walks toward the head one chunk at a time, stops at the first
    valid v2 row, and reports ``recovered`` from the invalid v2 rows it passed
    on the way. Rows below the settled one are never read, which is the whole
    saving: the forward fold JSON-decodes every ``"custom_type"`` row in the
    file — 3,416 of 21,600 rows / 185 MB of the operator's 262 MB journal, of
    which 243 ``frontend_state_checkpoint_v1`` rows alone are 947 ms — to
    recover two fields it then throws away. Measured on that journal:
    1,244 ms -> 16.9 ms.

    ``None`` means NO valid v2 row was found, i.e. the case the fold alone can
    answer; the caller then runs it. That fallback is not a shortcut: it is the
    population (pre-v2 transcripts) whose answer genuinely depends on rows
    below the newest v2 one, and it keeps today's cost exactly.

    ONE NAMED DIVERGENCE, and it only reaches a journal no writer produces:
    rows are decoded with ``errors="replace"``, as the module's sibling
    backward readers (``read_transcript_page``, ``read_latest_custom_entry``)
    already do. :func:`_forward_payloads` decodes strictly, so a journal with
    an invalid byte raises out of ``read_text``/iteration and its fold ends at
    the corruption — silently answering from the prefix above it. This scan
    answers from the newest valid v2 row instead, in the same direction those
    two readers chose, and a malformed row is still skipped individually.
    Journals this build writes cannot contain such a byte: every append encodes
    the whole row before writing it.
    """
    path = directory / "transcript.jsonl"
    try:
        handle = path.open("rb")
    except OSError:
        return None
    with handle:
        handle.seek(0, os.SEEK_END)
        unusable = False
        for _chunk_start, lines in _iter_complete_lines_backward(handle, handle.tell()):
            for raw in lines:
                # The same cheap prefilter the forward scan uses, in bytes: a
                # row without the marker cannot be a selection row, and most
                # rows are messages that would otherwise be JSON-decoded.
                if b'"custom_type"' not in raw:
                    continue
                try:
                    row = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, TypeError):
                    continue
                if not isinstance(row, dict) or row.get("type") != "custom":
                    continue
                payload = row.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("custom_type") != SELECTED_MODEL_CUSTOM_TYPE:
                    continue
                details = payload.get("details")
                # ``details`` that is not a mapping is skipped by the fold too
                # (``selection_from_payloads``'s first test), and a version
                # that is neither 2 nor absent is neither authoritative nor
                # legacy there, so it cannot raise ``recovered`` either.
                if not isinstance(details, dict):
                    continue
                if details.get("version") != SELECTION_VERSION:
                    continue
                selected = _selection(
                    details.get("selector"),
                    details.get("effort"),
                    authoritative=True,
                    boot=details.get("boot"),
                )
                if selected is not None:
                    return replace(selected, recovered=unusable)
                # A version-2 row whose selector is unusable: the fold records
                # this as `recovered`, clears it again if a NEWER valid v2 row
                # is read, and keeps looking back otherwise — which is exactly
                # what this flag does on the way toward the head.
                unusable = True
    return None


def read_model_selection(directory: Path) -> StoredModelSelection | None:
    """Read only identity rows without loading message attachments or history.

    The settle scan first (see :func:`_settled_selection`), then today's fold
    over the whole journal when the scan found no valid ``version == 2`` row.
    The two paths are provably equal (see that function's note), and the second
    one is the unchanged original code path rather than a re-derivation of it.

    This is the SHARED resolver — the cold desktop open
    (``cold_model.resolve_conversation_model``), ``draft_birth_selection`` on
    the desktop's own read path, and the CLI's /resume all call it — so the
    signature, the return type and the answers are load-bearing for callers
    rather than free to change.
    """
    settled = _settled_selection(directory)
    if settled is not None:
        return settled
    return selection_from_payloads(_forward_payloads(directory))


def refused_decision_only_selection(directory: Path) -> str | None:
    """The provider id of a stored selection this build refuses to run as chat.

    WHY a second reader instead of a flag on the result: the row is refused by
    ``_selection`` (see the decision-only branch there), so it never reaches a
    caller as a ``StoredModelSelection`` — and the refusal still deserves to be
    EXPLAINED, because the alternative a resolver has without it is to fall
    through to the configured hosting and resume the conversation on a model its
    own journal never named. This answers the one question the resolver cannot ask
    of the row itself: "was there a stored identity, and did THIS build refuse it
    for being un-chattable?" The message it feeds names the provider and ``/model``.

    BACKWARD, like :func:`_settled_selection`, and for the same reason: this runs
    only on the journals whose forward fold is expensive (the resolver has already
    failed to find a usable row), so walking them forward again would re-introduce
    the very cost that scan exists to remove. It stops at the newest
    ``version == 2`` selector row, which is the row the fold would have settled on.

    Only a v2 row is inspected, and that is a claim about the population rather
    than a shortcut: a selector naming a decision-only provider can only have been
    WRITTEN by a build that already shipped that provider — every writer today
    journals v2 — while a hand-edited version-less row still cannot become a
    selection at all (``_selection`` refuses it too), so the only thing lost on
    that path is the named explanation, never the refusal itself.
    """
    from local_operator.providers.registry import is_decision_only

    path = directory / "transcript.jsonl"
    try:
        handle = path.open("rb")
    except OSError:
        return None
    with handle:
        handle.seek(0, os.SEEK_END)
        for _chunk_start, lines in _iter_complete_lines_backward(handle, handle.tell()):
            for raw in lines:
                if b'"custom_type"' not in raw:
                    continue
                try:
                    row = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, TypeError):
                    continue
                if not isinstance(row, dict) or row.get("type") != "custom":
                    continue
                payload = row.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("custom_type") != SELECTED_MODEL_CUSTOM_TYPE:
                    continue
                details = payload.get("details")
                if not isinstance(details, dict) or details.get("version") != SELECTION_VERSION:
                    continue
                selector = details.get("selector")
                if not isinstance(selector, str) or "/" not in selector:
                    # The newest v2 row is settled but unreadable, so the fold would
                    # have kept looking below it and any name here would be a guess.
                    return None
                provider = selector.split("/", 1)[0]
                return provider if provider and is_decision_only(provider) else None
    return None


def session_uses_test_hosting(directory: Path) -> bool:
    """Whether this session's journal says it is CURRENTLY on the TEST hosting.

    WHY A SECOND READER, given the process-wide kill switch in
    ``tui.notify.suppress_notifications_for_process``: the switch silences the
    process that RAN the mock, and a store outlives that process. A QA rig's
    scratch store keeps its mock conversations, and every other reader of that
    store — the machine-wide desktop feed, a bridge attached by the desktop
    app, an operator's TUI looking at a per-rig config dir — would still
    compose a banner whose body is a snippet of the session's last assistant
    line, which for a mock session is always "Hello from the mock provider!".
    That sentence on a lock screen is the reported symptom, and the mock exists
    only for tests, so a stored session that is on it must not be announced.

    THE NEWEST SELECTION WINS, which is the same rule every other reader of
    this journal follows (:func:`read_model_selection`): a conversation that
    switched onto the mock is a test surface from that point, and one that
    switched off it is a real session again — for the store's benefit, since
    the process that ran the mock has already silenced itself.

    CHEAP, by construction: only the newest ``version == 2`` row is read, via
    the same bounded backward scan :func:`_settled_selection` uses (16.9 ms on
    the operator's 262 MB journal, and the callers only ask about sessions they
    are about to banner anyway — the composer reads the same transcript's tail
    on the very next line).

    TOLERANT, AND FAILS TOWARD NOTIFYING. ``False`` for a missing, unreadable
    or selection-free journal, and for an unusable row. Two reasons that is the
    right direction: an unreadable transcript is not evidence of a test
    session, and silencing a REAL session's completion is a worse failure than
    bannering a test one — the operator's complaint is about noise, and a
    "fix" that also mutes real work would be its own bug report. A mock
    session written by any build in this release carries a v2 row (see
    :func:`Session._persist_selected_model`), so the version gate loses nothing
    in practice; a version-less legacy row is deliberately not enough to
    silence a session.
    """
    try:
        from local_operator.providers.registry import is_mock_provider

        settled = _settled_selection(directory)
    except Exception:  # noqa: BLE001 — a banner decision never fails on a store read
        return False
    return settled is not None and is_mock_provider(settled.provider)
