"""State for the ``/community`` page and the ``/orgs`` switcher, without Textual.

WHY A MODEL MODULE. The page has two pieces of logic that must be right in
every state and are miserable to test through pixels: the composer's
sender-only outcome row (TUI frame T5: Checking… with ``esc`` cancel, refused,
three rate-limit scopes with countdowns, moderation unavailable, posted, and
"cancel was too late"), and the org switcher's default selection (RFC v2.4
Amendment A). Both live here as plain data the widget paints, the
``notifications_model`` / ``notifications_view`` split.

THE COMPOSER RULES (T5, T10):

* ``↵`` clears the composer at once; the submission runs in the background
  and its outcome lands on the status row, which only the sender sees.
* ``esc`` while a post is checking CANCELS it (best effort) and does not leave
  the composer.
* A refused or cancelled post's text comes back into the composer if the
  composer is still empty; otherwise it is set aside for ``f2 → restore``.
  Drafts and set-aside text live in memory only — refused text is never
  stored (RFC §6.5; TUI decision 3).
* A rate-limited, offline or restricted composer keeps the draft and refuses
  ``↵`` until the countdown ends (or the state changes); the countdown is
  derived from a deadline, so a repaint once a second updates only its text.
* A leading ``/`` is a local command and is never posted; ``//`` posts a
  literal slash (TUI decision 2).

ORG SWITCHER (Amendment A): private orgs and joined communities are two
separate groups, never merged; "All organizations" merges private orgs only;
the default selection is a pending private invitation, else the default
(home) private org, else the last selection — never a community while a
private org exists.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from local_operator.community.service import PostOutcome

#: The status-row kinds the composer paints (T5/T10 frame names).
StatusKind = Literal[
    "idle",
    "checking",
    "posted",
    "cancelled",
    "too_late",
    "refused",
    "rate_limited",
    "moderation_unavailable",
    "offline",
    "restricted",
    "read_only",
    "locked",
    "guidelines",
    "error",
    "report_sent",
]

#: Kinds whose countdown (or condition) keeps ``↵`` off with the draft kept.
_BLOCKING: frozenset[str] = frozenset({"rate_limited", "restricted", "read_only", "locked"})


@dataclass
class Composer:
    """One channel composer's draft and its sender-only status row."""

    draft: str = ""
    status: StatusKind = "idle"
    outcome: PostOutcome | None = None
    #: The text of the post in flight (memory only; cleared on any outcome).
    in_flight: str | None = None
    in_flight_key: str | None = None
    #: A refused/cancelled post's text when the composer already held a new
    #: draft: offered back through ``f2 → restore``.
    set_aside: str | None = None
    #: Monotonic deadline for a countdown (rate limit); ``None`` = no countdown.
    until: float | None = None
    #: Wall-clock ``HH:MM`` the last outcome landed (the "Posted 10:11" stamp);
    #: set by the painter, which owns the clock the user reads.
    at: str = ""
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    # -- gestures ---------------------------------------------------------

    def seconds_left(self) -> int:
        if self.until is None:
            return 0
        return max(0, int(self.until - self.clock() + 0.999))

    def can_send(self) -> bool:
        if self.status == "checking":
            return False
        if self.status in ("read_only", "locked", "restricted"):
            return False
        if self.status == "rate_limited":
            return self.seconds_left() == 0
        return True

    def postable_text(self, text: str) -> str | None:
        """The text a submit would post, or ``None`` for a local command/empty."""

        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("//"):
            return text.lstrip()[1:]
        if stripped.startswith("/"):
            return None
        return text

    def begin(self, text: str, key: str) -> None:
        """``↵`` accepted: the composer clears now; the post is checking."""

        self.in_flight = text
        self.in_flight_key = key
        self.draft = ""
        self.status = "checking"
        self.outcome = None
        self.until = None

    def settle(self, outcome: PostOutcome) -> None:
        """The submission's outcome arrived (or the cancel's answer did)."""

        text = self.in_flight
        self.in_flight = None
        self.in_flight_key = None
        self.outcome = outcome
        kind = outcome.kind
        if kind == "posted":
            self.status = "posted"
            return
        if kind == "too_late":
            self.status = "too_late"
            return
        if kind == "checking":
            # The hub still holds it; keep the row honest and the text safe.
            self.status = "checking"
            self.in_flight = text
            self.in_flight_key = outcome.submission_id
            return
        if kind == "signin":
            kind = "error"
        self.status = kind  # type: ignore[assignment]
        if kind == "rate_limited":
            self.until = self.clock() + float(outcome.retry_after_s or 1)
        if text is not None:
            self._restore(text)

    def _restore(self, text: str) -> None:
        # T5: back into the composer when it is empty, else set aside.
        if self.draft.strip():
            self.set_aside = text
        else:
            self.draft = text

    def restore_set_aside(self) -> bool:
        if self.set_aside is None or self.draft.strip():
            return False
        self.draft, self.set_aside = self.set_aside, None
        return True

    def tick(self) -> bool:
        """Once a second: True when a countdown just ended (repaint the row)."""

        if self.status == "rate_limited" and self.until is not None and self.seconds_left() == 0:
            self.until = None
            self.status = "idle"
            return True
        return False


@dataclass(frozen=True)
class ChannelView:
    """What the message pane shows for one channel (T1 "States")."""

    state: Literal["loading", "empty", "populated", "error", "degraded", "offline"]
    messages: tuple[Any, ...] = ()
    error_status: int | None = None


def pane_state(
    messages: Sequence[Any] | None, *, failure: Any = None, degraded: bool = False
) -> ChannelView:
    from local_operator.community import errors as e

    if failure is not None:
        if isinstance(failure, (e.CommunityUnreachable,)):
            return ChannelView("offline", tuple(messages or ()))
        return ChannelView("error", (), getattr(failure, "http_status", None))
    if messages is None:
        return ChannelView("loading")
    if not messages:
        return ChannelView("empty")
    return ChannelView("degraded" if degraded else "populated", tuple(messages))


def composer_for_channel(composer: Composer, channel: Any) -> Composer:
    """T10: a locked or read-only channel's composer says why, draft kept."""

    if getattr(channel, "locked", False):
        composer.status = "locked"
    elif channel is not None and is_read_only(channel):
        composer.status = "read_only"
    elif composer.status in ("locked", "read_only"):
        composer.status = "idle"
    return composer


#: The initial channels in rail order (RFC §5.3, frame T1).
INITIAL_CHANNELS: tuple[str, ...] = (
    "welcome-and-rules",
    "general",
    "help",
    "feedback",
    "show-and-tell",
)

#: Members read #welcome-and-rules; moderators and staff post (T10). The
#: directory's ``Channel`` model carries no per-viewer post permission, so the
#: name is the signal until the contract adds one; a member's post there is
#: refused by the hub (403) either way.
READ_ONLY_CHANNELS: frozenset[str] = frozenset({"welcome-and-rules"})


def is_read_only(channel: Any) -> bool:
    return getattr(channel, "name", "") in READ_ONLY_CHANNELS


def rail_order(channels: Sequence[Any]) -> list[Any]:
    """Initial channels first, in their canonical order; then the rest by name."""

    rank = {name: index for index, name in enumerate(INITIAL_CHANNELS)}
    return sorted(
        channels,
        key=lambda c: (0, rank[c.name]) if c.name in rank else (1, c.name.lower()),
    )


# -- the org switcher (Amendment A) -----------------------------------------------


@dataclass(frozen=True)
class OrgEntry:
    """One switcher row: a private Team org, "All organizations", or a community."""

    kind: Literal["private", "all", "community"]
    key: str
    name: str
    invited: bool = False
    home: bool = False
    unread: int = 0


def org_entries(
    memberships: Sequence[dict[str, Any]], communities: Sequence[Any]
) -> tuple[list[OrgEntry], list[OrgEntry]]:
    """``(organizations, communities)`` — two groups, never merged.

    ``memberships`` are ``GET /me/memberships`` rows (``tenant_id``,
    ``tenant_name``, ``status`` active|invited, ``is_home``); anything else is
    not a private org here. "All organizations" (private only) is offered only
    when there is more than one private org to merge.
    """

    orgs: list[OrgEntry] = []
    for row in memberships:
        status = row.get("status")
        tenant = str(row.get("tenant_id") or "")
        if not tenant or status not in ("active", "invited"):
            continue
        orgs.append(
            OrgEntry(
                "private",
                tenant,
                str(row.get("tenant_name") or tenant),
                invited=status == "invited",
                home=bool(row.get("is_home")) and status == "active",
            )
        )
    # An invitation is not membership (R1-8): "All organizations" merges orgs
    # the person belongs to, so it needs two ACTIVE ones.
    if sum(1 for entry in orgs if not entry.invited) > 1:
        orgs = [OrgEntry("all", "all", ""), *orgs]
    joined = [OrgEntry("community", c.id, c.name) for c in communities]
    return orgs, joined


def default_selection(
    orgs: Sequence[OrgEntry], communities: Sequence[OrgEntry], last: str | None
) -> OrgEntry | None:
    """Pending private invitation > default private org > last selection (Amendment A)."""

    private = [o for o in orgs if o.kind == "private"]
    for entry in private:
        if entry.invited:
            return entry
    for entry in private:
        if entry.home:
            return entry
    every = [*orgs, *communities]
    if last:
        for entry in every:
            if entry.key == last and (entry.kind != "community" or not private):
                return entry
    if len(private) == 1:
        return private[0]
    if private:
        return next(o for o in orgs if o.kind == "all")
    return communities[0] if communities else None


def with_unread(entry: OrgEntry, unread: int) -> OrgEntry:
    return replace(entry, unread=max(0, int(unread)))


__all__ = [
    "ChannelView",
    "Composer",
    "INITIAL_CHANNELS",
    "OrgEntry",
    "READ_ONLY_CHANNELS",
    "composer_for_channel",
    "default_selection",
    "is_read_only",
    "org_entries",
    "pane_state",
    "rail_order",
    "with_unread",
]
