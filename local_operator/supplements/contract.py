"""The frozen turn-supplement contract (lane C0): vocabulary, shapes, capability strings.

WHY THIS MODULE EXISTS. The memo (``docs/design/turn-supplements.md`` §6) lets seven lanes
(engine, routes, TUI, relay web, UI, native, docs) run in parallel only because the wire
and the journal row are frozen first, AS CODE, with fixtures under
``tests/fixtures/supplements/``. Everything here is data and pure functions over data: no
I/O, no session, no behaviour. A lane that needs a new field changes THIS file and the
fixtures in the same PR, which is what makes drift visible.

LEAF ON PURPOSE. Standard library only. ``harness/types.py`` imports the state vocabulary
from here (the event class lives with the other events), so anything heavier would put
that import on every process start.

WHAT IS NOT HERE. The runner, the decision, the generator, the validator, the denylist and
every surface renderer are later lanes (C1/C2/T/R/U/N).
"""

# NO ``from __future__ import annotations`` here, deliberately: with stringified annotations
# ``TypedDict`` cannot see ``NotRequired[...]`` (``__required_keys__`` then lists EVERY key),
# and the required/optional split below is part of the frozen contract.
import math
import re
from typing import Any, Final, Iterable, Literal, Mapping, NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Journal vocabulary (memo §2.4)
# ---------------------------------------------------------------------------

#: ``Transcript.append_custom(custom_type, details)`` type of a supplement row. Versioned
#: in the name (the ``stt_transcript_v1`` precedent): a shape change is a NEW type, never
#: an edit of this one, because rows are durable and old builds read them.
#:
#: NOT yet a member of ``transcript.BOOKKEEPING_CUSTOM_TYPES`` and not yet written by
#: anything: both land with the persistence lane (C1, memo §3.1 row 10), which is the
#: first writer. C0 changes no behaviour.
SUPPLEMENT_CUSTOM_TYPE: Final = "supplement_v1"

#: What a row's ``state`` may hold, i.e. what is JOURNALED.
SupplementRowState = Literal["decided", "queued", "done", "failed", "cancelled", "skipped"]

#: What a LIVE ``supplement_progress`` event may hold: the row vocabulary plus the two
#: transient values. ``running``/``cancelling`` are never journaled (memo §2.4, following
#: image-gen's live-only progress precedent), so readers map them onto the same words.
SupplementState = Literal[
    "decided", "queued", "running", "cancelling", "done", "failed", "cancelled", "skipped"
]

JOURNALED_STATES: Final[frozenset[str]] = frozenset(
    {"decided", "queued", "done", "failed", "cancelled", "skipped"}
)
LIVE_ONLY_STATES: Final[frozenset[str]] = frozenset({"running", "cancelling"})
#: The generator's stage names a ``running`` event may carry in ``stage``.
SUPPLEMENT_STAGES: Final = ("deciding", "generating", "validating", "repairing")

#: ``error`` value of a row cut by a newer user turn. It renders NOTHING (memo §2.8):
#: no Retry under an answer the user has already moved past.
SUPERSEDED_ERROR: Final = "superseded"

#: Digest of a stored component (``AttachmentStore`` digest): 32 lowercase hex characters.
#: The path-parameter regex of both document routes and the key under which sync finds it.
DIGEST_PATTERN: Final = r"^[a-f0-9]{32}$"
_DIGEST_RE: Final = re.compile(DIGEST_PATTERN)

#: ``height_hint`` is clamped by the validator at WRITE time (memo §2.4, round-1 R8); the
#: constants live here so the writer and every reader's reserved box use one range.
HEIGHT_HINT_MIN: Final = 120
HEIGHT_HINT_MAX: Final = 480
#: ``more[]`` disclosure bound (the stored "N more" paths), same denylist as ``files``.
MORE_MAX: Final = 20
#: Components per job (memo §2.8 aggregate frame budget names the same number).
MAX_COMPONENTS: Final = 3
#: ``supplement_steer`` text bound, in characters (memo §2.7 ops table).
STEER_TEXT_MAX_CHARS: Final = 500


class SupplementFile(TypedDict):
    """A file callout. ``path`` is session-relative or ``~/``-relative, NEVER absolute, so
    the operator's directory layout does not replicate to peers (memo S-R14)."""

    path: str
    name: str
    kind: str
    size_bytes: int
    mtime: float
    why: str


class SupplementComponent(TypedDict):
    """One generated HTML component. ``attachment`` is the LITERAL key sync's byte regex
    ``"attachment":"<32hex>"`` scans for, so the blob travels with a moved conversation
    with no sync change (memo §2.4)."""

    attachment: str
    title: str
    source: str
    mime: str
    height_hint: int


class SupplementImage(TypedDict):
    attachment: str
    mime: str
    title: str


class SupplementDecision(TypedDict):
    """The pre-filter + decision record, kept on the row for the spam-rate measurement
    (memo §5.2). ``skipped`` names why a job was not run (``"prefilter"`` etc.) or is
    ``None``."""

    vendor: str
    files_p: dict[str, float]
    graphics_p: float
    skipped: str | None


class SupplementDetails(TypedDict):
    """``payload["details"]`` of a ``supplement_v1`` custom entry (memo §2.4).

    The anchor key is ``anchor`` -- the final assistant message id, which the TUI calls
    ``completion_anchor_id`` on its block. There is deliberately no second key by that
    name: one id, one spelling on the wire.

    REQUIRED are the keys every reader needs to place and classify a row: ``anchor``,
    ``job``, ``version``, ``state``, ``files``, ``components`` (empty lists allowed) and
    ``at``. Everything else is optional because the lifecycle writes it in stages
    (``decided`` has no model/tokens) and because readers must tolerate a row from a
    build that wrote fewer keys.

    Versions: one job id is stable across regeneration; a new version is a new row with
    the same ``anchor`` and ``job`` and ``version + 1``. Readers take the NEWEST version
    per anchor (a reader rule, not a collapse: older rows stay as an audit trail).
    """

    anchor: str
    job: str
    version: int
    state: SupplementRowState
    files: list[SupplementFile]
    components: list[SupplementComponent]
    at: float
    files_more: NotRequired[int]
    more: NotRequired[list[str]]
    images: NotRequired[list[SupplementImage]]
    decision: NotRequired[SupplementDecision]
    instruction: NotRequired[str]
    model: NotRequired[str]
    turns: NotRequired[int]
    tokens_in: NotRequired[int]
    tokens_out: NotRequired[int]
    cost_usd: NotRequired[float]
    error: NotRequired[str]
    #: The operator's "not useful" signal (``supplement_dismiss`` writes
    #: ``state=skipped, dismissed=true``); surfaces hide the row.
    dismissed: NotRequired[bool]


# ---------------------------------------------------------------------------
# The stale-row reader rule (memo §2.4 "frozen in C0", display copy §2.8)
# ---------------------------------------------------------------------------

#: What a surface paints for the NEWEST row of an anchor. One of:
#:
#: * ``preparing``       -- ``◌ Preparing highlights… · Adjust… · Cancel``
#: * ``block``           -- the settled block (files and/or components)
#: * ``cancelled_retry`` -- ``Highlights cancelled · Retry`` (neutral ink)
#: * ``failed_retry``    -- ``Couldn't prepare highlights · Retry``
#: * ``nothing``         -- no line at all
ReaderDisposition = Literal["preparing", "block", "cancelled_retry", "failed_retry", "nothing"]


def reader_disposition(row: Mapping[str, Any], *, job_live: bool) -> ReaderDisposition:
    """Classify the newest ``supplement_v1`` row for one anchor, as every surface must.

    ``job_live`` is whether THIS runtime's ``_supplement_task`` registry holds the job.
    A cold reader (history page, reopened app) or a reaper-cut job has no live job, and
    then a non-terminal row (``decided``/``queued``) renders as ``cancelled · Retry``
    instead of spinning forever -- the stale-row rule. C0 ships the fixture
    ``state=queued``, no live job, that every lane asserts against.

    Precedence is the memo's: a superseded or dismissed row renders nothing whatever its
    state; then the committed ``state`` decides (a job-level ``failed`` is never replaced
    by the frame-level fallback line, which is a surface concern and never rewrites a
    row).
    """
    state = row.get("state")
    if row.get("error") == SUPERSEDED_ERROR or row.get("dismissed") or state == "skipped":
        return "nothing"
    if state == "done":
        # A finish with nothing to show renders nothing: no header, no reserved frame.
        return (
            "block"
            if (row.get("files") or row.get("components") or row.get("images"))
            else "nothing"
        )
    if state == "failed":
        return "failed_retry"
    if state == "cancelled":
        return "cancelled_retry"
    if state in ("decided", "queued", "running", "cancelling"):
        return "preparing" if job_live else "cancelled_retry"
    return "nothing"


def newest_per_anchor(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Newest-version-wins per anchor. Ties (a re-read of the same version) keep the later
    row in journal order, which is the one the writer appended last."""
    newest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        anchor = row.get("anchor")
        if not isinstance(anchor, str):
            continue
        held = newest.get(anchor)
        if held is None or int(row.get("version", 0)) >= int(held.get("version", 0)):
            newest[anchor] = row
    return newest


def is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


# ---------------------------------------------------------------------------
# Capabilities and the negotiation gate (memo §2.7)
# ---------------------------------------------------------------------------

#: A viewer that renders supplements. The ATTACH gate has two halves, ANDed: the runtime's
#: owner-record capability list (assembled beside ``display-history-audit-v1``) and the
#: viewer's own boolean on its auth frame (``auth["supplements"] = True``). The desktop
#: live path negotiates by the events route's query param ``supplements=1`` instead (the
#: ``frontend_replace``/``entry_ts`` shape); relay and attach clients read the owner's
#: list. See :func:`negotiated`.
SUPPLEMENTS_CAPABILITY: Final = "supplements-v1"
#: The auth-frame key a viewer declares. A name on ``network.dial.AUTH_FIELDS`` so a
#: relayed dial forwards it (the entry-times declaration once shipped inert for want of it).
SUPPLEMENTS_AUTH_FIELD: Final = "supplements"
#: ``GET /v1/capabilities`` ``features`` key and version. The STATIC HTTP flag; it is not
#: the attach gate (memo round-1 R5).
SUPPLEMENTS_FEATURE_KEY: Final = "supplements"
SUPPLEMENTS_FEATURE_VERSION: Final = 1
#: The runtime read op a handle must implement for the owner to advertise the capability
#: (``supplements_for(anchors[]) -> {anchor: newest row}``). An owner whose handle lacks it
#: must NOT advertise: advertising what the handle cannot honour is worse than omitting it.
SUPPLEMENTS_READ_OP: Final = "supplements_for"

#: Control ops (additive, capability-probed exactly like ``cancel``). ``steer`` and
#: ``restart`` spend money, so none of these is in ``_SYNC_PRIORITY_OPS``.
SUPPLEMENT_CONTROL_OPS: Final = (
    "supplement_cancel",
    "supplement_steer",
    "supplement_restart",
    "supplement_dismiss",
)
#: Every op name the contract reserves (control ops + the lazy read).
SUPPLEMENT_OPS: Final = (*SUPPLEMENT_CONTROL_OPS, SUPPLEMENTS_READ_OP)


def negotiated(owner_capabilities: Iterable[str], viewer_declared: bool) -> bool:
    """The two-half attach gate: the owner advertises AND the viewer declared.

    Live ``supplement_progress`` events and the history projection of ``supplement`` rows
    are sent only when this is True, so an older viewer never sees an unknown kind (old
    native builds would paint it visibly). The runtime ALWAYS journals rows regardless.
    """
    return bool(viewer_declared) and SUPPLEMENTS_CAPABILITY in set(owner_capabilities)


# ---------------------------------------------------------------------------
# The iframe message wire (memo §2.6 theme, §4.1 isolation, §4.2 liveness)
# ---------------------------------------------------------------------------

#: Host -> frame discriminator (``lo``) and frame -> host discriminator.
HOST_MESSAGE_TAG: Final = "supplement-host"
FRAME_MESSAGE_TAG: Final = "supplement"
FRAME_PROTOCOL_VERSION: Final = 1
#: The ONLY frame -> host shapes a host accepts.
FRAME_MESSAGE_TYPES: Final = ("ready", "resize", "error", "pong")
#: Those that can move host state and so MUST echo the per-frame nonce. ``ready`` precedes
#: the first theme push (which is what delivers the nonce) and moves nothing.
NONCE_REQUIRED_TYPES: Final = ("resize", "error", "pong")
#: The watchdog bound: a frame that never posts ``ready`` or fails to answer a ping in this
#: long is unmounted into the FRAME-level fallback line (never a rewrite of the row).
WATCHDOG_S: Final = 5.0


class ThemeMessage(TypedDict):
    """Host -> frame. ``vars`` holds RESOLVED values (never a palette id) for names
    matching ``--lo-*``/``--font-*``; the nonce is minted per frame by the host and sent
    in its FIRST theme push. The frame binds the first nonce and ignores later ones."""

    lo: Literal["supplement-host"]
    t: Literal["theme"]
    mode: Literal["light", "dark"]
    vars: dict[str, str]
    nonce: str


class PingMessage(TypedDict):
    lo: Literal["supplement-host"]
    t: Literal["ping"]


class FrameMessage(TypedDict):
    """Frame -> host. ``n`` is the nonce echo: required on resize/error/pong, absent on
    ready. ``h`` (px) only on resize, ``msg`` (<= 300 chars) only on error."""

    lo: Literal["supplement"]
    v: Literal[1]
    t: Literal["ready", "resize", "error", "pong"]
    n: NotRequired[str]
    h: NotRequired[float]
    msg: NotRequired[str]


def accept_frame_message(message: object, nonce: str | None) -> FrameMessage | None:
    """The SHAPE half of the host's acceptance rule, as a reference for every host.

    Returns the message when it is one of the four accepted shapes and, for a
    state-moving type, carries ``nonce`` (``nonce`` is None before the host has minted one:
    then only ``ready`` can pass). Everything else -- wrong tag or version, an unknown
    ``t``, a missing or stale nonce, a non-finite or negative ``h`` -- is ``None``.

    NOT the whole check. ``event.source === frame.contentWindow``, ``event.origin ===
    "null"``, the navigation counter and the one-message-per-animation-frame coalescing
    are host-side (memo §4.1/§4.2) and live in each surface's own code.
    """
    if not isinstance(message, Mapping):
        return None
    if message.get("lo") != FRAME_MESSAGE_TAG or message.get("v") != FRAME_PROTOCOL_VERSION:
        return None
    kind = message.get("t")
    if kind not in FRAME_MESSAGE_TYPES:
        return None
    if kind in NONCE_REQUIRED_TYPES and (not nonce or message.get("n") != nonce):
        return None
    if kind == "resize":
        height = message.get("h")
        if (
            isinstance(height, bool)
            or not isinstance(height, (int, float))
            or not math.isfinite(height)
            or height < 0
        ):
            return None
    return message  # type: ignore[return-value]
