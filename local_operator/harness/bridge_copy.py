"""Keyed English copy for a Team Bridge (``bridge``) tool row.

WHY A SEPARATE MODULE. :func:`local_operator.harness.rows.bridge_row` decides
what a bridge call is about as plain tokens and must stay i18n-free (that module
is imported below every host and carries no catalogue dependency). The TERMINAL
tool card and the PHONE projection both need the same words for those tokens,
so the token → copy step lives here, once, and both import it. The desktop
renders the same tokens through its own ``t()`` keys; this module is the core
side of that contract.

THE COPY lives in ``wire.tool_rows.bridge`` — a ``wire.*`` namespace because two
core-owned surfaces (TUI and relay projection) paint it and the projection's
rows travel to the phone as already-rendered text. Every string here is a key;
labels, previews, codes and error text are DATA (member-written or machine) and
are interpolated, never translated.

Read the result as ``verb`` + ``summary`` + an optional ``outcome`` word: the
TUI paints the verb and summary on the row and the outcome in its status cap;
the phone joins all three into one compacted line.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from local_operator.i18n.keys import wire_tool_rows_bridge as _k
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import tr

_t = tr("wire.tool_rows.bridge")

#: ``action`` token → its (settled, running, failed) verb keys. The composite
#: actions spell a dash on the wire (``react-remove``) and an underscore in the
#: key, because catalogue keys are lower snake_case (docs/I18N.md).
_VERBS: dict[str, tuple[Callable[[], Msg], Callable[[], Msg], Callable[[], Msg]]] = {
    "channel_post": (
        _k.verb_settled_channel_post,
        _k.verb_running_channel_post,
        _k.verb_failed_channel_post,
    ),
    "dm": (_k.verb_settled_dm, _k.verb_running_dm, _k.verb_failed_dm),
    "send": (_k.verb_settled_send, _k.verb_running_send, _k.verb_failed_send),
    "reply": (_k.verb_settled_reply, _k.verb_running_reply, _k.verb_failed_reply),
    "channels": (_k.verb_settled_channels, _k.verb_running_channels, _k.verb_failed_channels),
    "channel_create": (
        _k.verb_settled_channel_create,
        _k.verb_running_channel_create,
        _k.verb_failed_channel_create,
    ),
    "channel_history": (
        _k.verb_settled_channel_history,
        _k.verb_running_channel_history,
        _k.verb_failed_channel_history,
    ),
    "channel_thread": (
        _k.verb_settled_channel_thread,
        _k.verb_running_channel_thread,
        _k.verb_failed_channel_thread,
    ),
    "react": (_k.verb_settled_react, _k.verb_running_react, _k.verb_failed_react),
    "react-remove": (
        _k.verb_settled_react_remove,
        _k.verb_running_react_remove,
        _k.verb_failed_react_remove,
    ),
    "directory": (_k.verb_settled_directory, _k.verb_running_directory, _k.verb_failed_directory),
    "shared_conversations": (
        _k.verb_settled_shared_conversations,
        _k.verb_running_shared_conversations,
        _k.verb_failed_shared_conversations,
    ),
    "shared_history": (
        _k.verb_settled_shared_history,
        _k.verb_running_shared_history,
        _k.verb_failed_shared_history,
    ),
    "take_up": (_k.verb_settled_take_up, _k.verb_running_take_up, _k.verb_failed_take_up),
    "ack-working": (
        _k.verb_settled_ack_working,
        _k.verb_running_ack_working,
        _k.verb_failed_ack_working,
    ),
    "ack-done": (_k.verb_settled_ack_done, _k.verb_running_ack_done, _k.verb_failed_ack_done),
}
#: ``ack`` with no state is refused by the executor (``ack needs message_id and
#: state``) — but it is still a documented op, so its refusal must not borrow the
#: unknown-op copy (``Used Team Bridge ack``; review round 1, R3). It reads as the
#: claim it most often is ("Couldn't claim a message"), with no new copy to audit.
_VERBS["ack"] = _VERBS["ack-working"]
_UNKNOWN_VERBS = (_k.verb_settled_unknown, _k.verb_running_unknown, _k.verb_failed_unknown)

#: Actions whose verb is a complete phrase: no target follows, only a count.
_OBJECTLESS = frozenset({"channels", "directory", "shared_conversations"})

#: Outcomes that turn the verb into its "Couldn't …" form. ``unconfirmed`` is
#: deliberately absent: the hub may well have taken the write, so "Couldn't post"
#: would invite a duplicate; it keeps the settled verb and says
#: "delivery unconfirmed" beside it.
_FAILED_OUTCOMES = frozenset({"refused", "failed"})


def _target_text(row: Mapping[str, Any], *, in_thread: bool) -> str:
    """The rendered target: ``#name``, a bare person/DM label, or a kind noun.

    ``in_thread`` appends the ``{label} thread`` suffix. Callers pass it only
    for a ``channel_post`` reply: ``channel_thread`` also carries a
    ``thread_root_id``, but its verb already says ``Read thread in``, and the
    suffix painted ``Read thread in #general thread``.
    """
    raw = row.get("target")
    target: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    kind = target.get("kind")
    label = target.get("label")
    label = label if isinstance(label, str) and label else None
    if kind == "channel":
        if label is None:
            text = _t(_k.target_conversation())
        else:
            # A label that already carries its sigil (a channel named "#ops")
            # must not paint "##ops".
            text = label if label.startswith("#") else f"#{label}"
        return _t(_k.target_thread(label=text)) if in_thread else text
    if kind == "dm":
        return label or _t(_k.target_dm())
    if kind == "person":
        return label or _t(_k.target_person())
    if kind == "message":
        return _t(_k.target_message())
    return ""


def _count_text(action: str, count: object) -> str:
    if not isinstance(count, int) or isinstance(count, bool):
        return ""
    if action in ("channel_history", "channel_thread", "shared_history"):
        return _t(_k.count_messages(count=count))
    if action == "directory":
        return _t(_k.count_members(count=count))
    if action in ("channels", "shared_conversations"):
        return _t(_k.count_conversations(count=count))
    return ""


def outcome_word(row: Mapping[str, Any], *, short: bool = False) -> str | None:
    """The status-cap word for a non-plain outcome, or None for a plain one.

    ``refused``/``failed`` carry the machine code after `` · `` (never
    translated: it is what an operator searches for).

    ``short`` is the narrow-row form (``not posted``, ``awaiting``,
    ``unconfirmed``): the TUI caps its status column at a third of the row, and
    below the width where the full word fits that third the short word keeps
    the distinction instead of eating the summary. ``refused``/``failed`` are
    already one word, so they have no short form; the full sentence always
    stays in the expanded card.
    """
    outcome = row.get("outcome")
    code = row.get("code") if isinstance(row.get("code"), str) else None
    if short and outcome == "inbox_only":
        return _t(_k.outcome_short_inbox_only())
    if short and outcome == "awaiting_acceptance":
        return _t(_k.outcome_short_awaiting_acceptance())
    if short and outcome == "unconfirmed":
        return _t(_k.outcome_short_unconfirmed())
    if outcome == "inbox_only":
        return _t(_k.outcome_inbox_only())
    if outcome == "refused":
        word = _t(_k.outcome_refused())
        return f"{word} · {code}" if code else word
    if outcome == "failed":
        word = _t(_k.outcome_failed())
        return f"{word} · {code}" if code else word
    if outcome == "unconfirmed":
        return _t(_k.outcome_unconfirmed())
    if outcome == "awaiting_acceptance":
        return _t(_k.outcome_awaiting_acceptance())
    return None


def render_bridge_row(
    row: Mapping[str, Any], *, running: bool, in_thread: bool = False, failed: bool = False
) -> tuple[str, str, str | None]:
    """``(verb, summary, outcome_word)`` for one :func:`bridge_row` token dict.

    ``in_thread`` is the call's ``thread_root_id`` presence: a reply in a
    channel thread reads ``#name thread``. It is a call argument, not a token,
    so the caller (who holds the args) passes it rather than the shared token
    contract growing a field only one phrase reads.

    ``failed`` is the HOST's settle verdict, for a call that errored without
    returning a details dict (an exception before ``execute_bridge`` answered):
    the tokens then carry no outcome, and the settled verb ("Posted in") beside
    a ✗ would claim a post that never happened.

    The summary is ``<target> · <preview>`` for writes, ``<target> <count>``
    for reads, the emoji for a reaction, and the raw op word for an op this
    build has no copy for (never a guess at what it did).
    """
    action = str(row.get("action") or "")
    outcome = row.get("outcome")
    in_thread = in_thread and action == "channel_post"
    keys = _VERBS.get(action)
    known = keys is not None
    settled, live, failed_verb = keys if keys is not None else _UNKNOWN_VERBS
    if running:
        verb = _t(live())
    elif outcome in _FAILED_OUTCOMES or (failed and outcome is None):
        verb = _t(failed_verb())
    else:
        verb = _t(settled())
    if not known:
        return verb, str(row.get("op") or ""), outcome_word(row) if not running else None
    parts: list[str] = []
    if action in _OBJECTLESS:
        head = _count_text(action, row.get("count"))
    else:
        head = _target_text(row, in_thread=in_thread)
        count = _count_text(action, row.get("count"))
        if count:
            head = f"{head} {count}" if head else count
    if head:
        parts.append(head)
    emoji = row.get("emoji")
    if action in ("react", "react-remove") and isinstance(emoji, str) and emoji:
        parts.append(emoji)
    preview = row.get("preview")
    if isinstance(preview, str) and preview:
        parts.append(preview)
    return verb, " · ".join(parts), outcome_word(row) if not running else None


def bridge_row_line(
    row: Mapping[str, Any], *, running: bool, in_thread: bool = False, failed: bool = False
) -> str:
    """The row as ONE line (verb, summary, outcome), for single-cell surfaces.

    The phone's summary cell has no status cap of its own, so the outcome rides
    the end of the line — after the preview, because the projection compacts
    from the right and the target is what must survive first.
    """
    verb, summary, outcome = render_bridge_row(
        row, running=running, in_thread=in_thread, failed=failed
    )
    line = f"{verb} {summary}".strip() if summary else verb
    return f"{line} · {outcome}" if outcome else line


#: Internal fields the expanded card shows LAST, under a "Details" heading. Order
#: is the order a debugging reader needs: where, which message, then the
#: delivery-stage receipts, then the raw text the model saw.
_INTERNAL_FIELDS = (
    "conversation_id",
    "message_id",
    "seq",
    "thread_root_id",
    "body_sha256",
    "destination_type",
    "receipt_stage",
    "execution_confirmed",
    "resolved_account_id",
    "account_id",
    "endpoint_id",
    "channel_id",
    "recipient_state",
    "code",
    "content",
)


#: An opaque Team Bridge id inside free error text (``conversation bcv_… is
#: archived``). Same prefixes as ``rows._BRIDGE_ID_PREFIXES``, matched
#: case-insensitively as the label filter there is.
_ID_TOKEN_RE = re.compile(r"\b(bcv|bcmsg|acc|ep)_[A-Za-z0-9_-]+", re.IGNORECASE)


def _scrub_ids(error: str, row: Mapping[str, Any]) -> str:
    """``error`` with its raw ids replaced by words a reader can use.

    The expanded sentence is the readable account; the ids belong under
    ``Details``, where the raw ``error`` is NOT shown either but the ``code``
    and the conversation/message ids are. A conversation id becomes the row's
    own target (``#ops`` or "a conversation"); any other id becomes ``…``.
    The desktop's twin makes the same substitution (UI PR #1529).
    """
    if not error:
        return error
    target = _target_text(row, in_thread=False) or _t(_k.target_conversation())

    def swap(match: re.Match[str]) -> str:
        return target if match.group(1).lower() == "bcv" else "…"

    return _ID_TOKEN_RE.sub(swap, error)


def _refusal_sentence(error: str, code: str, *, failed: bool) -> str:
    """The refused/failed sentence without empty slots (never ``Refused:  ()``).

    A refusal can arrive with no message, no code, or neither (an executor
    exception path, a hub that answers with a bare status), so each shape has
    its own key rather than an ICU select a translator would have to keep
    whole across four arms.
    """
    if failed:
        if error and code:
            return _t(_k.expanded_failed(error=error, code=code))
        if error:
            return _t(_k.expanded_failed_error(error=error))
        if code:
            return _t(_k.expanded_failed_code(code=code))
        return _t(_k.expanded_failed_bare())
    if error and code:
        return _t(_k.expanded_refused(error=error, code=code))
    if error:
        return _t(_k.expanded_refused_error(error=error))
    if code:
        return _t(_k.expanded_refused_code(code=code))
    return _t(_k.expanded_refused_bare())


def bridge_expanded(
    row: Mapping[str, Any],
    args: Mapping[str, Any],
    details: Mapping[str, Any] | None,
    *,
    failed: bool = False,
) -> dict[str, Any]:
    """The expanded card's parts: ``{headline, body, outcome, internals}``.

    ``body`` is the raw (markdown) text the call wrote, for the host to render;
    ``outcome`` the keyed sentence; ``internals`` ``[(field, value)]`` pairs,
    shown last behind a disclosure (desktop) or a section heading (TUI). Identifiers
    live ONLY in ``internals``: the readable parts never show one.
    """
    details = details if isinstance(details, Mapping) else {}
    action = str(row.get("action") or "")
    in_thread = bool(str(args.get("thread_root_id") or "").strip())
    # Always the SETTLED form: an expansion is drawn only for a settled card.
    verb, summary, _word = render_bridge_row(row, running=False, in_thread=in_thread, failed=failed)
    if action == "dm":
        # A DM's headline names the PERSON: the bare kind fallback painted
        # "Direct message to a DM" (relay round 1), so an unlabelled one reads
        # "Direct message to a teammate".
        raw = row.get("target")
        label = raw.get("label") if isinstance(raw, Mapping) else None
        target = label if isinstance(label, str) and label else _t(_k.target_person())
        headline = _t(_k.expanded_dm_headline(label=target))
    else:
        head = summary.split(" · ", 1)[0] if summary else ""
        headline = f"{verb} {head}".strip()
    body = ""
    for key in ("body", "text"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            body = value
            break
    if not body and isinstance(details.get("text"), str):
        body = details["text"]
    outcome = row.get("outcome")
    error = _scrub_ids(str(details.get("error") or ""), row)
    code = str(details.get("code") or "")
    sentence = ""
    if outcome == "inbox_only":
        sentence = _t(_k.expanded_inbox_only())
    elif outcome == "refused":
        sentence = _refusal_sentence(error, code, failed=False)
    elif outcome == "failed":
        sentence = _refusal_sentence(error, code, failed=True)
    elif outcome == "unconfirmed":
        sentence = (
            _t(_k.expanded_unconfirmed(error=error))
            if error
            else _t(_k.expanded_unconfirmed_bare())
        )
    elif outcome == "awaiting_acceptance":
        sentence = _t(
            _k.expanded_awaiting_acceptance(state=str(details.get("recipient_state") or ""))
        )
    elif details.get("ok") is True and details.get("state") == "reacted":
        # `react` answers `state: reacted|removed` and no seq, so the plain-post
        # arm below never fired and the card said nothing about whether the
        # reaction landed (review round 1, NIT-2).
        sentence = _t(_k.expanded_reacted())
    elif details.get("ok") is True and details.get("state") == "removed":
        sentence = _t(_k.expanded_reaction_removed())
    elif details.get("ok") is True and details.get("state") == "sent":
        seq = details.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            sentence = _t(_k.expanded_posted_at(seq=str(seq)))
        else:
            sentence = _t(_k.expanded_posted())
    internals: list[tuple[str, str]] = []
    values: dict[str, Any] = {}
    for field in _INTERNAL_FIELDS:
        value = details.get(field, args.get(field) if field not in ("code", "content") else None)
        if value is None or value == "":
            continue
        values[field] = value
    # One destination, one name (design round 1, D2): the call addresses a
    # `channel_id` and the receipt answers `conversation_id`; when they are the
    # same id, two key names told a reader diagnosing a misroute that the post
    # went to a conversation AND a channel, a distinction that does not exist.
    if values.get("channel_id") == values.get("conversation_id"):
        values.pop("channel_id", None)
    # A write receipt's `content` (`posted to bcv_…: message bcmsg_… at seq 42`)
    # restates the id, message id and seq already listed above it. A receipt
    # with no message id ("accepted, but the hub returned no message id"), a
    # pair-lane inbox note and a read's payload all say something the fields do
    # not, so they stay.
    if details.get("state") == "sent" and values.get("message_id"):
        values.pop("content", None)
    for field, value in values.items():
        # JSON spelling, not the Python repr: the desktop paints `false` for the
        # same field, and `False` read as a stray code token (design D5).
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        internals.append((field, text))
    return {"headline": headline, "body": body, "outcome": sentence, "internals": internals}


def details_label() -> str:
    """The section heading above :func:`bridge_expanded`'s internals."""
    return _t(_k.expanded_details())
