"""Team Bridge taint latch and the egress tier decision (finding S-38, slice 1).

WHY THIS EXISTS
---------------
Team Bridge delivers text written by people outside this machine (another
tenant's member, a guest in a shared room) into a session's context: as a
``team_bridge_message`` card, or as the result of any ``bridge`` tool call.
Screening (``team_bridge/client.py``) is a probabilistic first layer. When an
injected instruction gets through it, the cheapest exfiltration route was the
read-tier web tools: ``web_fetch``, ``read https://…``, ``web_search`` and
``web_read`` never reached the approval gate, so in ``ask`` mode a model could
put a secret into a URL with nobody asked. Design memo:
``cos-read-tier-egress-design-2026-11-11.md`` (three security rounds; round 3
terminal), operator decision Option C.

STATUS, IN THE ONLY WORDS ALLOWED: closed for ask mode; accepted risk in
auto/yolo. A session running ``tool_approval_mode: auto`` (machine default) or
``--yolo``, or one whose host declared an unattended ``--tools`` inventory
naming the tool, still runs these calls unattended. That is the operator's
recorded decision (ask a-432c), not an oversight; do not describe this module
as closing S-38 outright.

WHAT IT DOES
------------
* **The latch** (:class:`BridgeTaintLatch`). A session latches the first time
  any bridge content reaches it, and never unlatches: only ``/new`` or a fresh
  session starts clean. Taint has no epoch shorter than the session, because
  the hostile text survives the turn, compaction, pruning, the assistant's
  restatement of it and anything fetched while tainted (memo §3.1, §3.5). The
  state is persisted as one host bookkeeping row
  (:data:`BRIDGE_TAINT_LATCH_TYPE`) in the append-only transcript, so it rides
  resume, fork and ``/move`` with no new sidecar. Custom bookkeeping entries
  never render to the model (``Transcript.append_custom``). A full re-scan of
  the in-memory journal is the backstop for a failed row write or a
  transcript that predates this module.
* **The seam** (:func:`escalate_for_bridge_taint`). Called from the loop's
  ONE tier decision (``harness/loop.py`` ``_runner_result``). It may only
  raise a tier, or refuse; it never lowers one.
* **The card rules** (:func:`requires_person`, :func:`session_grant_covers`).
  Every card raised in a latched session is Allow/Deny only, no mode change
  answers a parked one, and a session-scoped auto granted BEFORE the latch no
  longer covers calls after it (memo §4.2, §4.2.1).

FAIL-CLOSED INVARIANT (memo §3.6). Nothing on this path catches an exception
and carries on as "clean". An error deriving taint means tainted; an error
classifying a call means ``write``; an unparseable target means egress. The
codebase's habitual ``except Exception: logger.debug(...)`` wrapper is
forbidden here (a guard test greps for it).
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: The host bookkeeping row that records the latch. Written once per session,
#: at the first source (or at a latched parent's child spawn). A custom ENTRY
#: (``Transcript.append_custom``), so it never enters LLM context.
BRIDGE_TAINT_LATCH_TYPE = "bridge_taint_latch"

#: The card type a screened inbox delivery is appended as
#: (``harness/message_types.py``). Restated here, not imported, because this
#: leaf module must not import the team-bridge package (the loop imports us).
TEAM_BRIDGE_MESSAGE_TYPE = "team_bridge_message"

#: Every result of this tool is a source, whatever the op, refusals included
#: (S-45): read ops return screened bodies, write ops echo resolved names and
#: thread state. Over-approximating costs nothing.
BRIDGE_TOOL_NAME = "bridge"

Tier = Literal["read", "write", "exec"]

#: Search providers whose wire shape is a plain index lookup (verified in
#: security round 2: DDG HTML POST, Brave ``/res/v1/web/search``, SerpAPI
#: ``engine=google``). An ALLOWLIST, so a provider added later starts on the
#: safe side. SearXNG joins only when its endpoint is loopback/private
#: (:func:`index_only_provider`). Every other id -- tavily, deepseek,
#: perplexity, exa, parallel -- is an answer engine or live crawler, which can
#: itself dereference an attacker URL (S-43).
INDEX_ONLY_PROVIDERS: frozenset[str] = frozenset({"brave", "serpapi", "duckduckgo"})

#: Per-tool op allowlists for the durable cross-session writers (S-44). Under
#: taint every op NOT listed escalates to ``write``: a planted instruction in a
#: project description or an agent profile is read later by a clean session.
#: An allowlist so a new op starts gated. ``team`` is deliberately absent: its
#: account-scope writes already pass ``_authorize_account_profile_write``,
#: which is stricter (memo §2.2) -- adding it here would be a second gate.
DURABLE_WRITER_READ_OPS: Mapping[str, frozenset[str]] = {
    "project": frozenset({"list", "show"}),
    "library": frozenset({"stats", "list", "get", "conventions"}),
    "agent": frozenset({"list", "show", "search", "org_list"}),
    "tags": frozenset({"list"}),
    "skill": frozenset({"list", "show"}),
}

#: The refusal for a search an approval could not make meaningful (S-53):
#: approval never widens the provider set, so a forced answer-engine provider
#: or an empty filtered chain is refused with no card. Model-facing text.
SEARCH_PROVIDER_REFUSAL = (  # i18n: ignore model-facing tool refusal
    "answer-engine and live-crawl search providers are unavailable while this "
    "conversation holds Team Bridge content; configure an index-only provider "
    "(brave, serpapi, duckduckgo, or a private searxng) or start a new session"
)

#: Internal URL schemes ``read`` resolves locally (``tools/builtin.py``). Any
#: other ``scheme://`` target is treated as egress: fail closed.
_LOCAL_READ_SCHEMES = frozenset(
    {"skill", "spill", "tool", "guide", "mcp", "scratchpad", "file", "artifact", "library"}
)

#: A host-shaped token: two or more dot-separated labels ending in an
#: alphabetic TLD-like label. Deliberately broad -- ``Next.js`` and
#: ``README.md`` match, which is a counted false positive (memo §2.1), because
#: a DNS-exfil host (``<canary>.atk.test``) must never slip through.
_HOST_TOKEN = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.[A-Za-z][A-Za-z0-9-]*")


# -- sources -------------------------------------------------------------------


def message_is_taint_source(message: Any) -> bool:
    """Whether one live message carries Team Bridge content into context."""
    custom_type = getattr(message, "custom_type", None)
    if custom_type == TEAM_BRIDGE_MESSAGE_TYPE:
        return True
    return (
        getattr(message, "role", None) == "tool"
        and getattr(message, "tool_name", None) == BRIDGE_TOOL_NAME
    )


def entry_is_taint_source(entry_type: str, payload: Mapping[str, Any]) -> bool:
    """Whether one durable transcript row is a source, or the latch row itself."""
    if entry_type == "custom":
        return payload.get("custom_type") == BRIDGE_TAINT_LATCH_TYPE
    if entry_type != "message":
        return False
    if payload.get("custom_type") == TEAM_BRIDGE_MESSAGE_TYPE:
        return True
    return payload.get("role") == "tool" and payload.get("tool_name") == BRIDGE_TOOL_NAME


class BridgeTaintLatch:
    """The session's monotonic Team Bridge taint state.

    ``entries`` returns the transcript's durable rows as ``(type, payload)``
    pairs for the backstop scan; ``write_row`` persists the latch row (it may
    raise -- the latch is set in memory regardless and a warning is logged).
    """

    def __init__(
        self,
        *,
        inherited: bool = False,
        entries: Callable[[], Iterable[tuple[str, Mapping[str, Any]]]] | None = None,
    ) -> None:
        self.inherited = bool(inherited)
        self._latched = self.inherited
        self._entries = entries
        #: Whether the durable journal is known to carry the latch row.
        self.row_present = False
        #: Whether the backstop scan has completed once WITHOUT error. A
        #: scan that raised is never cached as clean (§3.6): it is retried.
        self._scanned = False
        #: Called once, with no argument, the moment the latch first sets in
        #: this process (the latch-time notice and grant suspension hang here).
        self.on_latched: list[Callable[[], None]] = []

    @property
    def latched(self) -> bool:
        """The cached state only (no scan). Use :meth:`is_latched` to decide."""
        return self._latched

    def mark(self) -> bool:
        """Set the latch. Returns True when this call is what set it."""
        if self._latched:
            return False
        self._latched = True
        for callback in list(self.on_latched):
            # A callback failure must not unset or hide the latch; it is
            # reported loudly, never at debug.
            try:
                callback()
            except Exception:  # noqa: BLE001 - the latch stays set; logged at warning
                logger.warning("bridge taint latch callback failed", exc_info=True)
        return True

    def observe(self, messages: Iterable[Any]) -> bool:
        """Latch on any source among live messages. Returns the latched state."""
        if self._latched:
            return True
        for message in messages:
            if message_is_taint_source(message):
                self.mark()
                return True
        return False

    def observe_entries(self, rows: Iterable[tuple[str, Mapping[str, Any]]]) -> bool:
        """Latch on any source among durable rows (commit observer / scan)."""
        hit = False
        for entry_type, payload in rows:
            if entry_type == "custom" and payload.get("custom_type") == BRIDGE_TAINT_LATCH_TYPE:
                self.row_present = True
                hit = True
                break
            if entry_is_taint_source(entry_type, payload):
                # Stop at the first source (R6): the latch is monotonic, so the
                # rest of the journal cannot change the answer. Whether the row
                # already exists is checked separately, in O(1), by the writer.
                hit = True
                break
        if hit:
            self.mark()
        return self._latched

    def is_latched(self, live_messages: Iterable[Any] = ()) -> bool:
        """Decide taint for one call. Never fails open.

        Order: the cached latch; the live context the model actually saw (a
        bridge result from earlier in THIS turn may not be durable yet); then,
        once per session object, a full scan of the durable journal from row 0
        (not the bounded provenance scan, which stops at 400 rows -- T13).
        """
        if self._latched:
            return True
        try:
            if self.observe(live_messages):
                return True
            if not self._scanned and self._entries is not None:
                self.observe_entries(self._entries())
                self._scanned = True
            elif self._entries is None:
                self._scanned = True
        except Exception:  # noqa: BLE001 - §3.6: an error deriving taint means tainted
            logger.warning(
                "bridge taint derivation failed; treating this call as tainted",
                exc_info=True,
            )
            return True
        return self._latched

    def latch_row_details(self) -> dict[str, Any]:
        return {"version": 1, "inherited": self.inherited}


def latch_of(tool_context: Any) -> BridgeTaintLatch | None:
    return getattr(tool_context, "bridge_taint", None) if tool_context is not None else None


def context_is_latched(tool_context: Any, live_messages: Iterable[Any] = ()) -> bool:
    """Whether the session behind ``tool_context`` is latched. Fail-closed.

    A context with no latch object at all is a host that does not track taint
    (a bare test host, an SDK embedder): it is judged by the live messages
    alone, which is what that host gave the model.
    """
    latch = latch_of(tool_context)
    try:
        if latch is None:
            inherited = bool(getattr(tool_context, "inherited_bridge_taint", False))
            return inherited or any(message_is_taint_source(m) for m in live_messages)
        return latch.is_latched(live_messages)
    except Exception:  # noqa: BLE001 - §3.6: an error deriving taint means tainted
        logger.warning("bridge taint lookup failed; treating as tainted", exc_info=True)
        return True


# -- the classifier --------------------------------------------------------------


def has_host_token(text: Any) -> bool:
    """Whether search text carries a URL or a host-shaped token (fail-closed)."""
    if not isinstance(text, str):
        return True
    if "://" in text:
        return True
    return _HOST_TOKEN.search(text) is not None


def read_target_is_egress(path: Any) -> bool:
    """Whether a ``read`` target leaves the machine. Undecidable means yes."""
    if not isinstance(path, str):
        return True
    stripped = path.strip()
    if "://" not in stripped:
        return False
    scheme = stripped.split("://", 1)[0].strip().lower()
    if scheme in _LOCAL_READ_SCHEMES:
        return False
    return True


def private_endpoint(url: Any) -> bool:
    """Whether a SearXNG endpoint is loopback/private by its LITERAL address.

    No DNS lookup: a hostname is not resolved here (a resolution in a policy
    predicate is slow and rebinding-prone), so only an IP literal or
    ``localhost`` qualifies. Anything else is public -- fail closed.
    """
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def index_only_provider(provider: str, *, searxng_endpoint: Any = None) -> bool:
    if provider in INDEX_ONLY_PROVIDERS:
        return True
    return provider == "searxng" and private_endpoint(searxng_endpoint)


def filter_index_only(chain: Sequence[str], *, searxng_endpoint: Any = None) -> list[str]:
    """Tighten-only: remove non-index-only providers, never add or reorder."""
    return [p for p in chain if index_only_provider(p, searxng_endpoint=searxng_endpoint)]


@dataclass(frozen=True)
class EgressDecision:
    """What the seam decided for one call.

    ``taint_raised`` is True only when THIS module raised the tier.
    ``refusal`` is a model-facing reason to refuse the call outright, with no
    card (S-58): an approval could not make it meaningful.
    """

    tier: Tier
    taint_raised: bool = False
    bridge_latched: bool = False
    refusal: str | None = None


_TIER_RANK = {"read": 0, "write": 1, "exec": 2}


def _raise_to_write(tier: str) -> Tier:
    if _TIER_RANK.get(tier, 2) >= _TIER_RANK["write"]:
        return tier if tier in _TIER_RANK else "exec"  # type: ignore[return-value]
    return "write"


#: Resolves the effective search chain for a tool context, and the SearXNG
#: endpoint, as ``(chain, searxng_endpoint)``. Installed by the web_search
#: package at import (``web_search/tool.py``) so this leaf does not import it.
SearchChainResolver = Callable[[Any], tuple[list[str], Any]]
_search_chain_resolver: SearchChainResolver | None = None


def install_search_chain_resolver(resolver: SearchChainResolver) -> None:
    global _search_chain_resolver
    _search_chain_resolver = resolver


def _classify(
    tool_name: str, tier: Tier, arguments: Mapping[str, Any], tool_context: Any
) -> tuple[Tier, str | None]:
    """The tainted-session tier for one call. May raise; the caller fails closed."""
    if tool_name == "web_fetch":
        return "write", None
    if tool_name == "read":
        if read_target_is_egress(arguments.get("path")):
            return "write", None
        return tier, None
    if tool_name in ("web_search", "web_read"):
        resolver = _search_chain_resolver
        if resolver is None:
            raise RuntimeError("search chain resolver not installed")
        chain, endpoint = resolver(tool_context)
        filtered = filter_index_only(chain, searxng_endpoint=endpoint)
        forced = arguments.get("provider")
        needs_search = tool_name == "web_search" or bool(arguments.get("search"))
        if needs_search and not filtered:
            return tier, SEARCH_PROVIDER_REFUSAL
        if forced is not None and forced not in filtered:
            return tier, SEARCH_PROVIDER_REFUSAL
        if tool_name == "web_read":
            # Its answering step is an auxiliary model call over the pages,
            # and ``urls`` fetches arbitrary hosts; neither is provably
            # index-only, so it always asks while latched (memo §2.1).
            return "write", None
        if has_host_token(arguments.get("query")):
            return "write", None
        return tier, None
    read_ops = DURABLE_WRITER_READ_OPS.get(tool_name)
    if read_ops is not None:
        if arguments.get("op") not in read_ops:
            return _raise_to_write(tier), None
        return tier, None
    return tier, None


def escalate_for_bridge_taint(
    tool_name: str,
    tier: Tier,
    arguments: Any,
    tool_context: Any,
    live_messages: Iterable[Any] = (),
) -> EgressDecision:
    """The seam: raise (or refuse) a tier for a call in a latched session.

    Pure apart from the latch lookup. Never lowers ``tier``. Any exception in
    the classifier escalates to ``write`` (§3.6).
    """
    latched = context_is_latched(tool_context, live_messages)
    if not latched:
        return EgressDecision(tier=tier)
    if tier != "read":
        # Already gated; only the card policy (bridge_latched) changes.
        return EgressDecision(tier=tier, bridge_latched=True)
    try:
        args = arguments if isinstance(arguments, Mapping) else {}
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments are not a mapping")
        raised, refusal = _classify(tool_name, tier, args, tool_context)
    except Exception:  # noqa: BLE001 - §3.6: an error classifying a call means write
        logger.warning(
            "bridge egress classification failed for %s; escalating to write",
            tool_name,
            exc_info=True,
        )
        return EgressDecision(tier="write", taint_raised=True, bridge_latched=True)
    if refusal is not None:
        return EgressDecision(tier=tier, bridge_latched=True, refusal=refusal)
    return EgressDecision(tier=raised, taint_raised=raised != tier, bridge_latched=True)


# -- the card and grant rules ------------------------------------------------------


def requires_person(title: str, *, bridge_latched: bool) -> bool:
    """Whether a pending card must be answered by a person, never a mode.

    The ONE predicate every card-side discharge site reads (R8): an account
    profile save, or any card raised while the session is latched.
    """
    from local_operator.harness.approval import never_auto_approved

    return never_auto_approved(title) or bool(bridge_latched)


GrantScope = Literal["session", "global"]


@dataclass(frozen=True)
class ApprovalGrant:
    """Which auto a gate holder is running under, and when it was granted.

    ``granted_while_latched`` is True only for a session grant made while the
    session was ALREADY latched (a deliberate post-latch ``/approvals auto``).
    Absent/unknown decodes to False and an undetermined scope to ``session``:
    both fail closed (§4.2.1).
    """

    auto: bool = False
    scope: GrantScope = "session"
    granted_while_latched: bool = False


NO_GRANT = ApprovalGrant()


def session_grant_covers(grant: ApprovalGrant | None, *, title: str, bridge_latched: bool) -> bool:
    """Whether an auto grant answers a NEW call without a card (§4.2.1).

    A session grant made before the latch is void once the session latches;
    the call falls through to the card (not to the global default beneath).
    Global auto is left exactly as today: that is Option C's accepted risk.
    """
    from local_operator.harness.approval import never_auto_approved

    if grant is None or not grant.auto:
        return False
    if never_auto_approved(title):
        return False
    if bridge_latched and grant.scope != "global" and not grant.granted_while_latched:
        return False
    return True


# -- copy ---------------------------------------------------------------------------
#
# Resolved per call (never materialised at import) so a live language move
# follows. Lazily imported: this leaf is imported by the loop, and the i18n
# package must not become an import-time dependency of the tier seam.


def _copy(name: str) -> str:
    from local_operator.i18n.keys import wire_bridge_taint as keys
    from local_operator.i18n.text import tr

    return tr("wire.bridge_taint")(getattr(keys, name)())


def bridge_card_notice() -> str:
    """The line every approval card carries while the session is latched (S-46)."""
    return _copy("card_notice")


def grant_suspended_notice() -> str:
    """The one-time notice when a pre-latch session auto stops covering calls."""
    return _copy("grant_suspended")


def ack_refused_notice() -> str:
    """The receipt detail when ``approvals auto`` lacks ``bridge_ack`` (§4.2.2)."""
    return _copy("ack_refused")


# -- the full target on a latched card (S-46) ------------------------------------

#: Tools whose card must show the egress target IN FULL while latched: the
#: egress-capable read tools, and the two schedulers whose single approval
#: arms unattended future calls (``monitor``/``wake`` creation).
FULL_TARGET_TOOLS = frozenset(
    {"web_fetch", "read", "web_search", "web_read", "monitor", "wake", "browser"}
)

#: Render bound for the full-target line, not a policy threshold: RFC 9110
#: §4.1 recommends senders and recipients support URIs of at least 8000
#: octets, so 8192 shows any URL a server would plausibly accept. Longer text
#: is still clipped with the sanitizer's marker, never silently.
FULL_TARGET_LIMIT = 8192


def full_target_line(tool_name: str, arguments: Any) -> str:
    """The untruncated call arguments for a latched card, or ``""``.

    The tool's own one-line summary is clipped at 500 characters for an
    ordinary card; a hostile URL hides its payload past that clip. Serialised
    canonically and sanitised to one line (no control sequences, no forged
    second prompt), exactly as the ordinary summary is.
    """
    import json

    from local_operator.ansi import sanitize_prompt_line

    if tool_name not in FULL_TARGET_TOOLS:
        return ""
    try:
        rendered = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(arguments)
    return sanitize_prompt_line(rendered, limit=FULL_TARGET_LIMIT)
