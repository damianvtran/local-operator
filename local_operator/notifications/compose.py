"""Render one notification's text from the facts, for every surface at once.

Pure composition, no delivery. The vocabulary (:data:`~local_operator.tui.
notify.CONTEXTS`, :data:`~local_operator.tui.notify.BODIES`, the budgets) is
imported from ``tui/notify.py`` rather than copied or relocated: the constants
already have ten import sites and a second declaration of the same wording is a
second thing to keep in step. Importing them here is safe in both directions —
``tui/notify.py`` pulls in no Textual, which is why the desktop bridge and the
detached runtime can both reach this module without dragging a terminal UI into
a server process.

WHY THE BACKEND COMPOSES AT ALL. Three surfaces show the same completion: the
TUI's own notifier, the detached runtime's OS fallback, and the desktop app.
Only the backend can read ``display.notification_session_name``, so only the
backend can decide whether a banner may carry the conversation's name or a line
of its content. A renderer composing its own strings would either re-derive a
privacy rule it cannot see, or ship the wording inside a signed binary where a
fix costs a full release. Both halves of the result travel (the rendered
strings AND the facts they came from) so a future surface can re-render — a
localisation, a narrower budget — without the backend guessing who is asking.

THE PRIVACY FLAG GATES BOTH THE NAME AND THE SNIPPET, on purpose. The flag
exists to keep model-written session text off a screen other people can see,
and a snippet is strictly more session-derived than a name: a name is a topic,
a snippet is content. A user who opted out of the name has necessarily opted
out of the snippet, so there is no second setting — one clear promise instead
of a flag that lets conversation content leak while the name is hidden.

See ``docs/design/descriptive-notifications.md`` §3 for the full contract and
the table this module implements literally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from local_operator.tui.notify import (
    APP_NAME,
    BACKGROUND_FALLBACK_TITLE,
    BACKGROUND_SNIPPET_MAX_CHARS,
    BODIES,
    BODY_COMPLETE,
    CONTEXT_COMPLETE,
    CONTEXTS,
    MAX_TITLE_CHARS,
    sanitize_text,
    session_names_in_notifications,
)

logger = logging.getLogger(__name__)

#: The kinds a composed notification may carry. Deliberately the
#: :data:`~local_operator.tui.notify.CONTEXTS` key set, so the vocabulary
#: cannot drift from the TUI's. Not every kind is reachable on every surface:
#: the desktop bridge narrows it further (``BRIDGE_NOTIFIABLE_KINDS``) because
#: ``ask``/``approval`` already reach that app as ``pending_gate``.
NotificationKind = Literal["complete", "error", "interrupted", "ask", "approval"]

#: Wire contract version for the ``notification`` frame's payload shape,
#: advertised as ``features.notification_contract`` in ``/v1/capabilities``.
#:
#: Bumped ONLY on a breaking change to field names or types; an additive field
#: does not bump it, because the renderer reads fields by name and ignores the
#: ones it does not know. If a field is ever removed or retyped, the old one is
#: kept beside the new one for one release — a renderer is a signed binary the
#: user updates on their own schedule, so the backend cannot assume the two
#: ship together.
NOTIFICATION_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ComposedNotification:
    """One notification's rendered text plus the facts it was rendered from.

    Both halves travel. The strings are what every surface shows, so wording
    parity across the TUI, the detached-runtime fallback and the desktop app is
    free; the facts are what lets a future consumer re-render without the
    backend having to guess which surface is asking.
    """

    kind: NotificationKind
    #: Banner title: the session's name, :data:`APP_NAME` when the privacy flag
    #: is off, or :data:`BACKGROUND_FALLBACK_TITLE` when a completion's session
    #: has no stored title.
    title: str
    #: Short state category from :data:`CONTEXTS` ("Complete", "Input
    #: required", "Needs attention", "Interrupted"). Rendered as the subtitle
    #: where a surface has one and folded into the body where it does not.
    status: str
    #: The content line. A last-assistant-line snippet for ``complete`` when
    #: the privacy flag allows and one exists; the house sentence otherwise.
    body: str
    #: True when ``body`` is model-written text rather than a house constant.
    #: Nothing reads it today; it exists so a surface that must not show
    #: conversation content (a shared screen, a recording) can degrade without
    #: re-deriving the privacy rule this module owns.
    body_is_snippet: bool
    #: Whether ``title`` is the conversation's real name. False when the
    #: privacy flag is off or the session is untitled.
    title_is_session_name: bool
    #: True when ``body`` is the PROVIDER's failure text rather than a house
    #: constant. Separate from :attr:`body_is_snippet` because the two are
    #: different kinds of untrusted text and a surface may reasonably treat
    #: them differently: a snippet is the model's own prose about the user's
    #: work, while this is an error envelope that may name a provider, a model
    #: or a quota. Never both true at once — a body is one thing or the other.
    #:
    #: Defaulted, and LAST, because it was added after the first draft of this
    #: dataclass: a caller constructing one positionally keeps working.
    body_is_failure: bool = False


def gate_body(kind: str, gate_title: str, gate_detail: str) -> str:
    """The body for a parked ``ask``/``approval``: the action, or the sentence.

    Extracted so the detached runtime's OS fallback
    (``runtime/serving.py::_announce_pending``) and any other gate surface
    cannot drift from each other's wording.

    NOT ``f"{title}: {detail}"``. A tool's ``describe_approval`` already leads
    with its own action word (``_describe_path_approval`` emits ``"write:
    /path"``) and the title IS the tool name, so the naive form rendered every
    such approval as ``"write: write: /path"`` — on the release's headline
    surface, every time (round 4, Q3). The prefix is applied only when the
    detail does not already carry it.

    An empty detail falls back to the house vocabulary rather than to the bare
    tool name: "write" alone says less than "Waiting for approval", and an
    ``ask`` with no text used to render as a single word with no hint that it
    was a question rather than an approval.
    """
    subject = (gate_detail or "").strip()
    if subject and gate_title and not subject.lower().startswith(gate_title.lower()):
        subject = f"{gate_title}: {subject}".strip().rstrip(":").strip()
    return subject or BODIES.get(kind, BODY_COMPLETE)


def compose(
    kind: NotificationKind,
    *,
    session_dir: Path | None,
    session_name: str = "",
    gate_title: str = "",
    gate_detail: str = "",
) -> ComposedNotification:
    """Render one notification. Pure apart from two best-effort disk reads.

    ``session_dir`` is read for the stored title and, for ``complete``, the
    last assistant line. ``session_name`` is the caller's already-resolved name
    (the TUI holds it live; the desktop bridge reads ``conversation_title`` off
    the frontend snapshot) and WINS over the stored title when non-empty,
    because a rename reaches the live state before it reaches the sidecar — a
    toast naming a session by the name it had ten seconds ago is worse than one
    naming it by none.

    ``gate_title``/``gate_detail`` are used only for ``ask``/``approval``; they
    carry the tool name and the action being authorised, in the shape
    ``_announce_pending`` already builds. Ignored for every other kind.

    NEVER RAISES. A notification is chrome and this runs on paths that must not
    fail for it: the desktop bridge's 1 s attention poll, and the TUI's turn-end
    handler. Every read is guarded and degrades to the house vocabulary, which
    asserts nothing about the conversation and is therefore always safe to show.

    The privacy flag is read PER CALL rather than cached, because ``/settings``
    writes it live: a value resolved once at construction keeps leaking the name
    for the rest of a session that had just turned it off.
    """
    names_ok = _names_allowed()
    status = CONTEXTS.get(kind, CONTEXT_COMPLETE)

    title, title_is_session_name = _compose_title(
        kind, names_ok=names_ok, session_dir=session_dir, session_name=session_name
    )
    body, body_is_snippet, body_is_failure = _compose_body(
        kind,
        names_ok=names_ok,
        session_dir=session_dir,
        gate_title=gate_title,
        gate_detail=gate_detail,
    )
    return ComposedNotification(
        kind=kind,
        title=title,
        status=status,
        body=body,
        body_is_snippet=body_is_snippet,
        body_is_failure=body_is_failure,
        title_is_session_name=title_is_session_name,
    )


def _names_allowed() -> bool:
    """The privacy flag, with a settings store that cannot be read treated as OFF.

    Failing CLOSED is the only defensible direction here: the flag's whole
    purpose is to keep model-written text off a screen other people can see, so
    an unreadable or corrupt settings file must not be the thing that puts a
    conversation's name on a lock screen. The cost of the wrong answer is
    asymmetric — a missed name is a duller banner, a leaked one is the failure
    the setting exists to prevent.
    """
    try:
        return session_names_in_notifications()
    except Exception:  # noqa: BLE001 — chrome; see the docstring for the direction
        logger.debug("notification privacy flag unavailable; assuming off", exc_info=True)
        return False


def _compose_title(
    kind: str, *, names_ok: bool, session_dir: Path | None, session_name: str
) -> tuple[str, bool]:
    """``(title, title_is_session_name)`` per the §3.2 table.

    The nameless case splits on kind, and the split is deliberate: "A session
    finished" is a true sentence for a completion and a false one for a parked
    question, so a gate with no resolvable name falls back to the brand — which
    is what the detached runtime's own gate fallback already does.

    :data:`MAX_TITLE_CHARS` is 80, and the state category never rides in the
    title: macOS clips a banner title at roughly 43 characters, so appending
    anything after the name is appending the part that gets cut.
    """
    if not names_ok:
        # The opt-out keeps the BRAND, which is exactly what a user who turned
        # this off is asking for: no statement about their sessions at all.
        return APP_NAME, False
    name = sanitize_text(session_name, MAX_TITLE_CHARS)
    if not name and session_dir is not None:
        name = sanitize_text(_stored_title(session_dir), MAX_TITLE_CHARS)
    if name:
        return name, True
    if kind in ("ask", "approval"):
        return APP_NAME, False
    return BACKGROUND_FALLBACK_TITLE, False


def _compose_body(
    kind: str,
    *,
    names_ok: bool,
    session_dir: Path | None,
    gate_title: str,
    gate_detail: str,
) -> tuple[str, bool, bool]:
    """``(body, body_is_snippet, body_is_failure)`` per the §3.2 table.

    THE SNIPPET IS FOR ``complete`` ONLY, and the restriction is a correctness
    rule rather than a stylistic one. An errored or interrupted session's last
    assistant line is whatever it happened to be saying before it stopped,
    which routinely reads as a success ("All 412 tests pass.") beside a state
    that says it failed. "Snippet iff complete" is decidable from the kind
    alone; any rule that asked whether a particular line reads honestly would
    depend on a judgement no code here can make, on a lock screen where prose
    is read once with nothing beside it to check it against.

    ``error`` GETS THE FAILURE TEXT INSTEAD, which is the exception that
    proves that rule rather than a hole in it. "Stopped with an error" names a
    state the user must act on while withholding the only fact that says WHICH
    action — top up a quota, fix a credential, or just retry (design round 1,
    D4). The text comes from the session's own ``session_incident`` record, so
    it describes the failure itself and cannot be mistaken for a claim about
    the work, which is precisely what made the last-assistant-line unsafe here.
    """
    if kind in ("ask", "approval"):
        return gate_body(kind, gate_title, gate_detail), False, False
    if kind == "error" and names_ok and session_dir is not None:
        # Same privacy gate as the snippet, and for a stronger reason: a
        # provider's error envelope can quote a prompt fragment, a file path or
        # an account identifier. A user who opted out of seeing their session
        # named has certainly opted out of that.
        failure = sanitize_text(_failure(session_dir), BACKGROUND_SNIPPET_MAX_CHARS)
        if failure:
            return failure, False, True
    if kind != "complete" or not names_ok or session_dir is None:
        return BODIES.get(kind, BODY_COMPLETE), False, False
    # `max_chars` is PASSED rather than applied afterwards, so the
    # word-boundary ellipsis `session_preview` computes lands against the
    # budget the banner actually has; trimming a 200-char preview down to 120
    # afterwards cuts mid-word after the ellipsis was already placed elsewhere.
    preview = _preview(session_dir)
    # `sanitize_text` re-applies the budget as its `limit`; deliberate
    # belt-and-braces, since stripping control characters can only shorten the
    # string and the two bounds therefore agree. The scrub itself is required:
    # this text reaches argv (`cmux notify`, `notify-send`) and an AppleScript
    # string literal.
    snippet = sanitize_text(preview, BACKGROUND_SNIPPET_MAX_CHARS)
    if snippet:
        return snippet, True, False
    return BODIES.get("complete", BODY_COMPLETE), False, False


def _stored_title(session_dir: Path) -> str:
    """The session's stored title, or ``""`` for anything that goes wrong.

    Imported inside the function because ``resume`` drags the session-directory
    machinery, and this module is imported by the detached runtime's gate path
    where that cost buys nothing until a gate actually parks.
    """
    try:
        from local_operator.resume import stored_session_title

        return stored_session_title(session_dir)
    except Exception:  # noqa: BLE001 — a title is chrome; delivery is not
        logger.debug("notification title unavailable for %s", session_dir, exc_info=True)
        return ""


def _failure(session_dir: Path) -> str:
    """The last failure's raw text, or ``""`` for anything that goes wrong.

    Guarded and imported lazily for the same reasons as :func:`_preview`, and
    it shares that function's bounded tail window, so adding it costs one more
    read of an already-warm 64 KiB region rather than a second scan strategy.
    """
    try:
        from local_operator.resume import session_failure_summary

        return session_failure_summary(session_dir, max_chars=BACKGROUND_SNIPPET_MAX_CHARS)
    except Exception:  # noqa: BLE001 — a body is chrome; delivery is not
        logger.debug("notification failure text unavailable for %s", session_dir, exc_info=True)
        return ""


def _preview(session_dir: Path) -> str:
    """The last assistant line, or ``""`` for anything that goes wrong.

    ``session_preview`` already returns ``""`` for a missing, unreadable or
    assistant-text-free transcript and swallows ``OSError`` itself; the broad
    guard is for everything above that contract — bytes that decode into
    something unexpected, a directory that vanished between the two calls.
    """
    try:
        from local_operator.resume import session_preview

        return session_preview(session_dir, max_chars=BACKGROUND_SNIPPET_MAX_CHARS)
    except Exception:  # noqa: BLE001 — a body is chrome; delivery is not
        logger.debug("notification preview unavailable for %s", session_dir, exc_info=True)
        return ""
