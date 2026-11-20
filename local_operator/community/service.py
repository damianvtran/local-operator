"""Member operations shaped for the terminal surfaces: the TUI page and ``lop community``.

WHY A SECOND LAYER OVER THE CLIENT. ``client.CommunityClient`` speaks the wire
(typed models in, typed errors out). Both terminal surfaces need the SAME
answers on top of it — "which community is mine", "find #help by name",
"what happened to my post, in the sender-only vocabulary of TUI frame T5", and
the CLI's documented exit codes — and two copies of that mapping would drift
(the TUI saying "rate limited" where the script got exit 3). One mapping lives
here, as plain data; each surface only paints it.

SENDER-ONLY OUTCOMES (RFC §5.5, frame T5). A refusal's category and
``MOD-XXXX-XXXX`` reference, a rate-limit scope and countdown, "moderation
unavailable" — these are shown only to the sender. :class:`PostOutcome` carries
exactly those fields and never the message text: refused text is never stored
(RFC §6.5), so the outcome cannot be the place it leaks to.

EXIT CODES (``tui-frames.md`` "Commands for scripts"): 0 posted, 3 refused
``{category, reference}``, 4 rate limited ``{scope, retry_after_s}``, 5
moderation unavailable, 6 offline, 7 restricted, 8 guidelines acceptance needed
``{version}``; ``cancel``: 0 cancelled, 9 already posted; 10 refused because
the command ran inside an agent's shell (``cli.AGENT_SHELL_VERBS``). 1 is every
other failure, 2 is argparse's usage error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Literal

from local_operator.community import errors as e

#: How many directory pages (25 rows each) a name lookup may read; see
#: ``agent_ops.CHANNEL_LOOKUP_MAX_PAGES`` for the bound's reasoning.
LOOKUP_MAX_PAGES = 4

OutcomeKind = Literal[
    "posted",
    "checking",
    "refused",
    "rate_limited",
    "moderation_unavailable",
    "offline",
    "restricted",
    "guidelines",
    "cancelled",
    "too_late",
    "signin",
    "error",
]

EXIT_CODES: dict[str, int] = {
    "posted": 0,
    "checking": 0,
    "cancelled": 0,
    "refused": 3,
    "rate_limited": 4,
    "moderation_unavailable": 5,
    "offline": 6,
    "restricted": 7,
    "guidelines": 8,
    "too_late": 9,
    # Not a hub outcome: ``lop community`` refuses its acting/reading verbs in
    # an agent's shell before any request (``cli._agent_shell_refusal``).
    "agent_shell": 10,
    "signin": 1,
    "error": 1,
}

#: The four refusal labels' wire categories, in the order the report form and
#: the CLI ``--category`` list them (RFC §6.2).
REFUSAL_CATEGORIES: tuple[str, ...] = (
    "unsafe_or_illegal",
    "harassment_or_abuse",
    "spam_or_promotion",
    "private_information",
)


@dataclass(frozen=True)
class PostOutcome:
    """What happened to one submission, in the sender-only vocabulary."""

    kind: OutcomeKind
    submission_id: str | None = None
    message_id: str | None = None
    category: str | None = None
    reference: str | None = None
    scope: str | None = None
    retry_after_s: int | None = None
    version: str | None = None
    code: str | None = None

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.kind]

    def as_json(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def display_safe(value: Any) -> Any:
    """A copy of ``value`` with every string run through ``strip_control_sequences``.

    WHY (review R1-3). Post text, display names, channel names and purposes,
    guidelines, refusal references and error codes come from the community
    server, and much of it is written by strangers. Painted raw, an OSC 52
    sequence writes the owner's clipboard, ``ESC[2J`` clears their screen and
    an OSC 0 retitles the window — ``rich`` strips only BEL/BS/VT/FF/CR, not
    ESC. Both terminal surfaces read through this ONE function at their ingress
    (the page's backend reads, the CLI's text printer) instead of guarding
    each paint site, so a field added to a model later is covered without
    anyone remembering to. Newlines and tabs survive (post bodies are
    multi-line); printable text, including emoji and RTL, is unchanged.

    Walks pydantic models, dataclasses, lists, tuples and dicts; returns other
    values as they are. Enum members are left alone (their values are ours).
    The wire objects are never mutated — the client's own copies stay exact,
    which is what ``--json`` and the request path must see.
    """

    from enum import Enum

    from pydantic import BaseModel

    from local_operator.ansi import strip_control_sequences

    if isinstance(value, Enum):
        return value
    if isinstance(value, str):
        return strip_control_sequences(value)
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={name: display_safe(getattr(value, name)) for name in type(value).model_fields}
        )
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value, **{f.name: display_safe(getattr(value, f.name)) for f in fields(value) if f.init}
        )
    if isinstance(value, list):
        return [display_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(display_safe(item) for item in value)
    if isinstance(value, dict):
        return {key: display_safe(item) for key, item in value.items()}
    return value


def error_word(exc: BaseException) -> str:
    """The short code a surface paints for a failure, display-safe (a hub code is server text)."""

    return display_safe(str(getattr(exc, "code", None) or type(exc).__name__))


def outcome_from_submission(outcome: Any) -> PostOutcome:
    """Map a ``SubmissionOutcome`` (submit, status or cancel read)."""

    state = getattr(outcome, "state", None)
    key = getattr(outcome, "idempotency_key", None)
    if state == "published":
        return PostOutcome("posted", submission_id=key, message_id=outcome.message_id)
    if state == "refused":
        refusal = getattr(outcome, "refusal", None)
        return PostOutcome(
            "refused",
            submission_id=key,
            category=getattr(refusal, "category", None),
            reference=getattr(refusal, "reference", None),
        )
    if state == "cancelled":
        return PostOutcome("cancelled", submission_id=key)
    if state == "checking":
        return PostOutcome("checking", submission_id=key)
    return PostOutcome("error", submission_id=key, code=str(state))


def outcome_from_error(exc: BaseException, *, submission_id: str | None = None) -> PostOutcome:
    """Map a raised ``CommunityError`` (or anything else) to one outcome."""

    if isinstance(exc, e.CommunityRefused):
        return PostOutcome(
            "refused", submission_id=submission_id, category=exc.category, reference=exc.reference
        )
    if isinstance(exc, e.CommunityRateLimited):
        return PostOutcome(
            "rate_limited",
            submission_id=submission_id,
            scope=exc.scope,
            retry_after_s=exc.retry_after,
        )
    if isinstance(exc, e.CommunityModerationUnavailable):
        return PostOutcome(
            "moderation_unavailable", submission_id=submission_id, retry_after_s=exc.retry_after
        )
    if isinstance(exc, e.CommunityGuidelinesChanged):
        return PostOutcome(
            "guidelines", submission_id=submission_id, version=exc.guidelines_version
        )
    if isinstance(exc, (e.CommunityUnreachable, e.CommunityUnavailable)):
        return PostOutcome("offline", submission_id=submission_id, code=exc.code)
    if isinstance(exc, e.CommunityForbidden):
        return PostOutcome("restricted", submission_id=submission_id, code=exc.code)
    if isinstance(exc, e.CommunitySignInRequired):
        return PostOutcome("signin", submission_id=submission_id, code=exc.code)
    code = getattr(exc, "code", None) if isinstance(exc, e.CommunityError) else type(exc).__name__
    return PostOutcome("error", submission_id=submission_id, code=str(code))


def channel_word(value: str) -> str:
    """``#help`` / ``help`` / `` #Help `` → ``help`` (names are lower-case slugs)."""

    return value.strip().lstrip("#").strip().lower()


ClientFactory = Callable[..., Any]


def make_client(config_dir: Any = None, factory: ClientFactory | None = None) -> Any:
    if factory is not None:
        return factory(config_dir)
    from local_operator.community.client import CommunityClient

    return CommunityClient(config_dir)


async def joined_community(client: Any, community_id: str | None = None) -> Any | None:
    """The community this surface talks to: the named one, else the first joined.

    Local Operator Community is the first (and today only) community; the
    platform may host more (Amendment A), so the id is taken when given.
    """

    listing = await client.list_communities()
    for community in listing.joined:
        if community_id is None or community.id == community_id:
            return community
    return None


async def any_community(client: Any, community_id: str | None = None) -> tuple[Any | None, bool]:
    """``(community, joined)``: a joined one first, else a featured one to join."""

    listing = await client.list_communities()
    for group, joined in ((listing.joined, True), (listing.featured, False)):
        for community in group:
            if community_id is None or community.id == community_id:
                return community, joined
    return None, False


async def list_channels(
    client: Any, community_id: str, *, filter: str | None = None, q: str | None = None
) -> list[Any]:
    """Every channel the directory shows under ``filter``, up to the lookup bound."""

    rows: list[Any] = []
    cursor: str | None = None
    for _ in range(LOOKUP_MAX_PAGES):
        page = await client.channels(community_id, q=q, filter=filter, cursor=cursor)
        rows.extend(page.items)
        cursor = page.next_cursor
        if not cursor:
            break
    return rows


async def find_channel(client: Any, community_id: str, value: str) -> Any | None:
    """Resolve ``#name`` (or an id) among the channels the member has joined."""

    word = channel_word(value)
    if not word:
        return None
    for channel in await list_channels(client, community_id, filter="joined", q=word):
        if channel.name.lower() == word or channel.id == value.strip():
            return channel
    return None


__all__ = [
    "EXIT_CODES",
    "PostOutcome",
    "REFUSAL_CATEGORIES",
    "any_community",
    "channel_word",
    "find_channel",
    "joined_community",
    "list_channels",
    "make_client",
    "outcome_from_error",
    "outcome_from_submission",
]
