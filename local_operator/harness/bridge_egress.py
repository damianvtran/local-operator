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
put a secret into a URL with nobody asked. The rules, threat model and
accepted risks are written down in ``docs/security/team-bridge-egress.md``.

STATUS, IN THE ONLY WORDS ALLOWED: closed for ask mode; accepted risk in
auto/yolo. A session running ``tool_approval_mode: auto`` (machine default) or
``--yolo``, or one whose host declared an unattended ``--tools`` inventory
naming the tool, still runs these calls unattended. That is a recorded
product decision, not an oversight; do not describe this module as closing
S-38 outright.

WHAT IT DOES
------------
* **The latch** (:class:`BridgeTaintLatch`). A session latches the first time
  any bridge content reaches it, and never unlatches: only ``/new`` or a fresh
  session starts clean. Taint has no epoch shorter than the session, because
  the hostile text survives the turn, compaction, pruning, the assistant's
  restatement of it and anything fetched while tainted. The
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
  longer covers calls after it.
* **Propagation between sessions.** The latch follows bridge text wherever it
  is carried, in both directions:

  - PUSH: a ``hub`` send/steer/ask from a latched session stamps the child
    message (:data:`BRIDGE_TAINT_DETAILS_KEY`), and a live parent/child pair is
    linked so whichever latches first latches the other (:func:`link_latches`)
    -- a child's report always flows to its parent, and a parent can write
    anything into a running child.
  - PULL: any tool result that carries ANOTHER session's transcript content
    (``sessions`` peek in every mode, a mesh peek or mesh reply, ``read``/
    ``grep`` of a file under another session's directory) is a source when
    that session is latched -- or when its state cannot be determined, which
    is every mesh peer today, because no peer reports it (fail closed).
    The stamp is :data:`BRIDGE_TAINT_DETAILS_KEY` in the result's
    ``details``; it feeds the same latch, it is not a second gate.

FAIL-CLOSED INVARIANT. Nothing on this path catches an exception
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

#: The ``details`` key that marks a message (a tool result, or a host card
#: such as a ``hub_message``) as carrying another latched session's content.
#: Only ever written as ``True``: a clean result carries no key, so untainted
#: rows stay byte-identical.
BRIDGE_TAINT_DETAILS_KEY = "bridge_taint"

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
#: which is stricter (team-bridge-egress.md: "Seam") -- adding it here would be a second gate.
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
    "conversation holds a message from a teammate's agent (Agent messages); "
    "configure an index-only provider "
    "(brave, serpapi, duckduckgo, or a private searxng) or start a new session"
)

#: Internal URL schemes ``read`` resolves locally (``tools/builtin.py``). Any
#: other ``scheme://`` target is treated as egress: fail closed.
_LOCAL_READ_SCHEMES = frozenset(
    {"skill", "spill", "tool", "guide", "mcp", "scratchpad", "file", "artifact", "library"}
)

#: A host-shaped token, three alternatives:
#:
#: * two or more dot-separated labels ending in a letter-led TLD-like label,
#:   Unicode labels included (``\w`` is Unicode-aware on ``str`` patterns, so an
#:   IDNA host typed in its native script matches as well as its ``xn--`` form);
#: * an IPv4 literal (``93.184.216.34``), whose last label is all digits;
#: * an IPv6-looking run of at least two colons (``2001:db8::1``).
#:
#: Deliberately broad -- ``Next.js``, ``README.md`` and a ``10:30:00`` clock
#: time match, which are counted false positives, because a DNS-exfil host
#: (``<canary>.atk.test``) must never slip through.
_HOST_TOKEN = re.compile(
    r"[\w-]+(?:\.[\w-]+)*\.[^\W\d_][\w-]*"
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"
    r"|[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}"
)


# -- sources -------------------------------------------------------------------


def details_stamped(details: Any) -> bool:
    """Whether a ``details`` mapping carries the cross-session taint stamp."""
    return isinstance(details, Mapping) and details.get(BRIDGE_TAINT_DETAILS_KEY) is True


def stamp_details(details: dict[str, Any] | None, latched: bool) -> dict[str, Any] | None:
    """Add the taint stamp to a result's ``details`` when ``latched``.

    Never writes ``False``: a clean result keeps exactly the details it had.
    """
    if not latched:
        return details
    stamped = dict(details or {})
    stamped[BRIDGE_TAINT_DETAILS_KEY] = True
    return stamped


def _tool_payload_stamped(provider_payload: Any) -> bool:
    # ``Message.from_tool_result`` keeps a result's details under
    # ``provider_payload["details"]`` (harness bookkeeping, never sent to a
    # provider), so that is where the stamp lands on a live tool message.
    return isinstance(provider_payload, Mapping) and details_stamped(
        provider_payload.get("details")
    )


def message_is_taint_source(message: Any) -> bool:
    """Whether one live message carries Team Bridge content into context.

    Dispatches on the concrete type FIRST. This runs over every live message
    on every tool call of a clean session, and a ``getattr`` miss on a
    pydantic ``Message`` (it has no ``custom_type``) goes through pydantic's
    ``__getattr__`` and raises internally -- about 5 µs per message, which put
    tens of milliseconds on each call of a long session (review R2). Unknown
    shapes still take the duck-typed path so they are judged, never skipped.
    """
    from local_operator.harness.types import CustomMessage, Message

    if isinstance(message, Message):
        if message.role != "tool":
            return False
        if message.tool_name == BRIDGE_TOOL_NAME:
            return True
        return _tool_payload_stamped(message.provider_payload)
    if isinstance(message, CustomMessage):
        return message.custom_type == TEAM_BRIDGE_MESSAGE_TYPE or details_stamped(message.details)
    if getattr(message, "custom_type", None) == TEAM_BRIDGE_MESSAGE_TYPE:
        return True
    if details_stamped(getattr(message, "details", None)):
        return True
    if getattr(message, "role", None) != "tool":
        return False
    if getattr(message, "tool_name", None) == BRIDGE_TOOL_NAME:
        return True
    return _tool_payload_stamped(getattr(message, "provider_payload", None))


def entry_is_taint_source(entry_type: str, payload: Mapping[str, Any]) -> bool:
    """Whether one durable transcript row is a source, or the latch row itself."""
    if entry_type == "custom":
        return payload.get("custom_type") == BRIDGE_TAINT_LATCH_TYPE
    if entry_type != "message":
        return False
    if payload.get("custom_type") == TEAM_BRIDGE_MESSAGE_TYPE:
        return True
    if details_stamped(payload.get("details")):
        return True
    if payload.get("role") != "tool":
        return False
    if payload.get("tool_name") == BRIDGE_TOOL_NAME:
        return True
    return _tool_payload_stamped(payload.get("provider_payload"))


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


def link_latches(parent: BridgeTaintLatch, child: BridgeTaintLatch) -> Callable[[], None]:
    """Make a live parent/child pair latch together; returns the unlink.

    A child spawned before its parent latched otherwise stays clean for life
    while the latched parent can still write bridge-derived text into it
    (``hub`` send/steer are read-tier; review R1, S-67). And everything a
    latched child produces -- its report, a ``hub`` reply, a ``wait`` or
    ``jobs peek`` result -- reaches the parent. Linking both ways closes every
    such channel at once instead of stamping each result. ``mark`` is
    idempotent and monotonic, so the mutual subscription cannot recurse.
    Each latch writes its own row on its next commit.
    """
    if parent.latched:
        child.mark()
    elif child.latched:
        parent.mark()
    parent_cb = child.mark
    child_cb = parent.mark
    parent.on_latched.append(parent_cb)
    child.on_latched.append(child_cb)

    def unlink() -> None:
        for latch, callback in ((parent, parent_cb), (child, child_cb)):
            try:
                latch.on_latched.remove(callback)
            except ValueError:
                pass

    return unlink


#: Bytes that a source row must contain (compact JSON as the journal writes
#: it, with a tolerated space after the colon). Used only as a prefilter so a
#: clean target costs one regex pass over the file; every hit is parsed and
#: judged by :func:`entry_is_taint_source`, so a mention of these words in
#: ordinary text is not a source.
_SOURCE_NEEDLES = re.compile(rb"team_bridge_message|bridge_taint|\"tool_name\":\s*\"bridge\"")


def session_directory_latched(directory: Any) -> bool:
    """Whether the session stored at ``directory`` holds bridge content.

    For a PULL from another session (``sessions`` peek, ``read``/``grep`` of
    its files). The WHOLE journal is judged, not the peeked window: the
    window can hold the assistant's restatement of an instruction whose
    source row sits much earlier. A session with no transcript has nothing
    to pull (``False``). Any other failure -- unreadable file, archive that
    will not open, a matched row that will not parse -- means latched.
    """
    import json
    import mmap
    from pathlib import Path

    from local_operator.session.transcript import _resolved_transcript_path

    try:
        path = _resolved_transcript_path(Path(directory), read_only=True)
        if path is None:
            return False
        with open(path, "rb") as handle:
            if handle.seek(0, 2) == 0:
                return False
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
                position = 0
                while True:
                    match = _SOURCE_NEEDLES.search(view, position)
                    if match is None:
                        return False
                    start = view.rfind(b"\n", 0, match.start()) + 1
                    end = view.find(b"\n", match.end())
                    if end < 0:
                        end = len(view)
                    line = bytes(view[start:end]).strip()
                    position = end + 1
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        # The last line of a live journal can be mid-append;
                        # anything else that names a source and will not
                        # parse is undecidable, so it counts.
                        if end == len(view):
                            continue
                        return True
                    payload = row.get("payload") if isinstance(row, Mapping) else None
                    if isinstance(payload, Mapping) and entry_is_taint_source(
                        str(row.get("type") or ""), payload
                    ):
                        return True
    except FileNotFoundError:
        return False
    except Exception:  # noqa: BLE001 - §3.6: an undeterminable target means latched
        logger.warning(
            "bridge taint of session %s could not be determined; treating as latched",
            directory,
            exc_info=True,
        )
        return True


#: Tools whose ``path`` argument reads file CONTENT. ``glob`` returns names
#: only, so it is not here.
_CONTENT_READ_TOOLS = frozenset({"read", "grep"})


def _sessions_root() -> Any:
    from local_operator.paths import config_dir

    return (config_dir() / "sessions").resolve()


def _valid_session_id(session_id: str) -> bool:
    from pathlib import PurePath

    return (
        bool(session_id)
        and PurePath(session_id).name == session_id
        and session_id
        not in (
            ".",
            "..",
        )
    )


def _transcript_missing(directory: Any) -> bool:
    """Whether ``directory`` holds no resolvable transcript. Raises on error."""
    from pathlib import Path

    from local_operator.session.transcript import _resolved_transcript_path

    try:
        return _resolved_transcript_path(Path(directory), read_only=True) is None
    except FileNotFoundError:
        return True


def _session_dir_latched(session_id: str) -> bool:
    """Whether the store session ``session_id`` is latched. Fail-closed.

    An id that cannot name a directory directly under the sessions root (empty,
    a separator, ``..``) is not a known owner, so it counts as latched. A
    session directory with no transcript is clean HERE (S-60: a read of the
    session's own files finds nothing to pull); content held ELSEWHERE whose
    owner has no transcript is judged by :func:`_owned_copy_latched`.
    """
    if not _valid_session_id(session_id):
        return True
    return session_directory_latched(_sessions_root() / session_id)


def _owned_copy_latched(session_id: str, copy: Any = None) -> bool:
    """Whether content OWNED by ``session_id`` but stored outside its session
    directory (a spill entry, a decompressed archive copy) is latched.

    FAIL-CLOSED on a missing owner (review S-72 / Y1). The content exists, so
    an owner with no session directory, a deleted directory, or a directory
    with no transcript does not mean "nothing to pull" (that S-60 rule fits a
    read of the directory itself); it means the content's provenance is
    unknown, and unknown counts as latched. The opt-in ``session/cleanup.py``
    can remove a latched session while the shared 64 MiB spill store and the
    decompressed cache keep its content, which is exactly this case.

    ``copy`` is the decompressed copy's own directory: when the owner's journal
    is gone, the byte-identical copy IS that journal, so it is scanned instead
    of being counted blind; a copy with no transcript of its own counts.
    """
    if not _valid_session_id(session_id):
        return True
    directory = _sessions_root() / session_id
    try:
        if not _transcript_missing(directory):
            return session_directory_latched(directory)
        if copy is not None and not _transcript_missing(copy):
            return session_directory_latched(copy)
    except Exception:  # noqa: BLE001 - §3.6: an undeterminable owner means latched
        logger.warning("owner %s of stored content could not be judged", session_id, exc_info=True)
    return True


def _spill_digest_latched(digest: str, reader_id: str = "") -> bool:
    """Whether the spill entry ``digest`` came from a latched session.

    The spill store is shared by every session (``tools/spill.py``) and
    ``_read_spill`` has no owner check, so a spill is judged by the session
    its metadata names (``SpillMeta.session_id``). FAIL-CLOSED on an unknown
    owner (review S-68): an evicted entry, missing or unreadable metadata, or
    an empty ``session_id`` (entries written by a host with no session) all
    count as latched -- the content's provenance is what is unknown, and
    "unknown" is the case the latch exists for. The cost is that a clean
    session reading an ownerless spill asks on its next egress. A spill the
    READER itself wrote is not a pull (this is only asked while the reader is
    clean), which keeps a session's ordinary read-back of its own output free.
    """
    try:
        from local_operator.tools.spill import SPILL_SCHEME, get_store

        meta = get_store().stat(f"{SPILL_SCHEME}{digest}")
        if meta is None:
            return True
        owner = str(meta.session_id or "")
        if owner and owner == reader_id:
            return False
        return _owned_copy_latched(owner)
    except Exception:  # noqa: BLE001 - §3.6: an undeterminable owner means latched
        logger.warning("spill owner of %s could not be determined", digest, exc_info=True)
        return True


#: Ignore files ripgrep reads (in the search root, below it AND in every parent
#: of it) whose ``!`` whitelist un-hides a hidden directory. Measured on rg
#: 14/15 (review S-71): ``!.local-operator`` in a ``.ignore`` or ``.rgignore``
#: in the root or a parent, or in a ``.gitignore`` inside a git work tree, makes
#: ``rg CANARY ~`` reach ``~/.local-operator/sessions``.
_WHITELIST_IGNORE_FILES = (".ignore", ".rgignore", ".gitignore")


def _ignore_file_may_whitelist(directory: Any) -> bool:
    """Whether ``directory`` carries anything that can un-hide a hidden dir.

    A ``.git`` entry counts outright: inside a work tree rg also honours
    ``.git/info/exclude`` and the user's global excludes file, both measured
    to whitelist, and neither is cheap to locate exactly. An ignore file
    counts when any line is a ``!`` negation (a file without one cannot
    whitelist anything); one that cannot be read counts.
    """
    if (directory / ".git").exists():
        return True
    for name in _WHITELIST_IGNORE_FILES:
        file = directory / name
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True
        if any(line.lstrip().startswith("!") for line in text.splitlines()):
            return True
    return False


def _ancestor_search_skips(resolved: Any, tree: Any, arguments: Mapping[str, Any]) -> bool:
    """Whether a recursive search rooted at ``resolved`` provably skips ``tree``.

    ``resolved`` is a strict ancestor of ``tree``. Both grep engines skip a
    hidden directory they were not named into, so with the default
    ``~/.local-operator`` a plain ``grep ~`` never reaches the stores and is
    not a pull. That holds ONLY while nothing un-hides the directory (review
    S-71), so the exemption is withheld -- the search counts as a pull -- when:

    - no component between ``resolved`` and ``tree`` is hidden (a non-hidden
      custom config dir, review X1);
    - the call sets ANY ``include``: rg treats a ``-g`` glob that matches the
      hidden directory's own name (``*``, ``*.*``, ``**``, ``**/*``) as a
      whitelist, and deciding which globs can match is a parser this seam
      should not carry, so every include is treated as one that may;
    - an ignore file in ``resolved``, any parent of it, or a non-hidden
      directory on the way down may whitelist the folder
      (:func:`_ignore_file_may_whitelist`).

    ``RIPGREP_CONFIG_PATH`` (a config carrying ``--hidden``) needs no rule here:
    the grep tool passes ``--no-config`` (``tools/builtin.py``
    ``_ripgrep_scan``). The Python fallback walk prunes every hidden directory
    before it consults includes or ignore rules, so these conditions are
    conservative for it, never short.
    """
    parts = tree.relative_to(resolved).parts
    hidden_at = next((i for i, part in enumerate(parts) if part.startswith(".")), None)
    if hidden_at is None:
        return False
    include = arguments.get("include")
    if include is not None and include != "":
        return False
    on_the_way = [resolved.joinpath(*parts[:i]) for i in range(1, hidden_at + 1)]
    for directory in (*reversed(resolved.parents), resolved, *on_the_way):
        try:
            if _ignore_file_may_whitelist(directory):
                return False
        except OSError:
            return False
    return True


def cross_session_read_latched(tool_name: str, arguments: Any, tool_context: Any) -> bool:
    """Whether a ``read``/``grep`` call pulls a latched session's content.

    A session's content is reachable through four file spellings, and each is
    that session's content as surely as a ``sessions`` peek is (reviews S-60,
    X1, S-68):

    - ``<config>/sessions/<id>/...`` -- its journal and its own files;
    - ``<config>/cache/decompressed/<id>/...`` -- the byte-identical copy
      ``transcript_for_read`` materialises for an archived session
      (``session/compression._cache_root``), judged as ``sessions/<id>``;
    - the shared spill store, ``<config>/spill/<digest>.*`` or
      ``spill://<digest>``, judged by the owner its metadata names
      (fail-closed on an unknown owner, :func:`_spill_digest_latched`);
    - any of those three trees' ROOT, or an ANCESTOR of one (the config dir
      itself, ``<config>/cache``, or a path above the config dir unless the
      search provably skips the hidden config dir,
      :func:`_ancestor_search_skips`):
      a recursive search over many sessions' content is undecidable without
      scanning all of them, so it counts as latched, unconditionally. That is
      a superset of "a latched session exists under that root"; the cost is
      a card on the next egress after such a search in a clean store.

    ``grep`` with no ``path`` searches ``"."``, so the effective path is
    judged. A path naming none of these is not a pull. Never raises: a path
    that will not resolve is judged by its literal spelling and counts if
    that names one of the stores.
    """
    if tool_name not in _CONTENT_READ_TOOLS or not isinstance(arguments, Mapping):
        return False
    raw = arguments.get("path")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = "."
    if not isinstance(raw, str):
        return False
    raw = raw.strip()
    reader = str(getattr(tool_context, "session_id", "") or "")
    if raw.startswith("spill://"):
        digest = raw[len("spill://") :].split("?", 1)[0].strip()
        return _spill_digest_latched(digest, reader)
    if "://" in raw:
        return False
    from pathlib import Path

    try:
        from local_operator.paths import config_dir

        config = config_dir().resolve()
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            cwd = getattr(tool_context, "cwd", None)
            candidate = Path(cwd or ".") / candidate
        resolved = candidate.resolve()
    except Exception:  # noqa: BLE001 - §3.6: undecidable; judged by its spelling
        logger.warning("cross-session read path could not be resolved", exc_info=True)
        return any(word in raw for word in ("sessions", "decompressed", "spill"))
    sessions = config / "sessions"
    decompressed = config / "cache" / "decompressed"
    spills = config / "spill"
    for tree in (sessions, decompressed, spills):
        if resolved == tree:
            return True
        if tree.is_relative_to(resolved):
            # ``resolved`` is an ancestor. At or below the config dir it was
            # named into explicitly, so nothing hidden is skipped on the way.
            if resolved.is_relative_to(config) or not _ancestor_search_skips(
                resolved, tree, arguments
            ):
                return True
    if resolved.is_relative_to(sessions):
        return _session_dir_latched(resolved.relative_to(sessions).parts[0])
    if resolved.is_relative_to(decompressed):
        # A copy of ``sessions/<id>``'s archive, judged as that session; when
        # the session is gone the copy itself is scanned (S-72, fail closed).
        session_id = resolved.relative_to(decompressed).parts[0]
        return _owned_copy_latched(session_id, decompressed / session_id)
    if resolved.is_relative_to(spills):
        # ``<digest>.txt`` (content) or ``<digest>.json`` (metadata); a stray
        # file there names no entry and has no owner, so it counts.
        return _spill_digest_latched(Path(resolved.relative_to(spills).parts[0]).stem, reader)
    return False


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
    #: True when THIS call latched the session by pulling a latched session's
    #: content (``cross_session_read_latched``). The loop stamps the call's
    #: result with it (review S-70): the in-memory latch writes its own row
    #: only on the next commit, and a crash in that window -- or a third
    #: session peeking this one first -- would otherwise see the pulled text
    #: with no source row. A stamped result row IS a source row.
    pulled: bool = False


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
            # index-only, so it always asks while latched (team-bridge-egress.md: "Search
            # providers").
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
    pulled = False
    if not latched and cross_session_read_latched(tool_name, arguments, tool_context):
        # A PULL: this call reads a latched session's files. Latch before it
        # runs (one call early is the safe side); the session writes its own
        # row on the next commit.
        latch = latch_of(tool_context)
        if latch is not None:
            latch.mark()
        latched = True
        pulled = True
    if latched and tool_name in ("web_search", "web_read") and latch_of(tool_context) is None:
        # A host with no latch object is judged by its live messages here,
        # but the search-provider filter (``web_search/tool.py``) sees no
        # messages. Record the decision on the context so the two agree; if
        # it cannot be recorded the filter would run unrestricted, so the
        # call is refused instead (review M3, fail closed).
        try:
            if tool_context is None:
                raise TypeError("no tool context to carry the latch")
            tool_context.inherited_bridge_taint = True
        except Exception:  # noqa: BLE001 - §3.6: could not pass the latch on; refuse
            logger.warning("could not record bridge taint on the tool context", exc_info=True)
            return EgressDecision(tier=tier, bridge_latched=True, refusal=SEARCH_PROVIDER_REFUSAL)
    if not latched:
        return EgressDecision(tier=tier)
    if tier != "read":
        # Already gated; only the card policy (bridge_latched) changes.
        return EgressDecision(tier=tier, bridge_latched=True, pulled=pulled)
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
        return EgressDecision(tier="write", taint_raised=True, bridge_latched=True, pulled=pulled)
    if refusal is not None:
        return EgressDecision(tier=tier, bridge_latched=True, refusal=refusal, pulled=pulled)
    return EgressDecision(
        tier=raised, taint_raised=raised != tier, bridge_latched=True, pulled=pulled
    )


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
    Global auto is left exactly as today: that is the accepted auto/yolo risk.
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


def _copy(name: str, **params: str) -> str:
    from local_operator.i18n.keys import wire_bridge_taint as keys
    from local_operator.i18n.text import tr

    return tr("wire.bridge_taint")(getattr(keys, name)(**params))


def bridge_card_notice() -> str:
    """The line every approval card carries while the session is latched (S-46)."""
    return _copy("card_notice")


def card_options_off_notice() -> str:
    """Why a latched TUI card has no Allow all / ``S`` / ``G``, and the way back.

    Names only exits that work on every surface today: a typed
    ``/approvals auto`` in the terminal (desktop and phone carry no
    ``bridge_ack`` yet) and ``/new``.
    """
    return _copy("card_options_off")


def grant_suspended_notice() -> str:
    """The one-time notice when a pre-latch session auto stops covering calls."""
    return _copy("grant_suspended")


def ack_refused_notice() -> str:
    """The receipt detail when ``approvals auto`` lacks ``bridge_ack`` (§4.2.2)."""
    return _copy("ack_refused")


# -- the full target on a latched card (S-46) ------------------------------------

#: Tools whose card must show the egress target IN FULL while latched: the
#: egress-capable read tools, ``browser`` (its describer shows a URL, but
#: ``evaluate`` code and form values can carry a payload past it), and the two
#: schedulers whose single approval arms unattended future calls
#: (``monitor``/``wake`` creation).
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


#: ``sanitize_prompt_line``'s default clip for the ordinary summary.
_SUMMARY_LIMIT = 500


def _argument_scalars(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _argument_scalars(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _argument_scalars(item)
    elif isinstance(value, bool) or value is None:
        return
    else:
        yield str(value)


def latched_card_summary(summary: str, tool_name: str, arguments: Any) -> str:
    """The summary a latched card shows: the ordinary one, plus the full
    request ONLY when the ordinary one hides part of it (S-46).

    ``web_fetch`` has no describer, so its summary is already the raw
    arguments, and ``browser``'s describer prints the URL: appending the full
    line repeated the target, which wrapped a long URL far enough to scroll
    Deny off the card (design D4). So the line is added only when the summary
    was clipped, or some argument value does not appear in it verbatim -- a
    value with a newline, a control character or a hidden tail is exactly one
    that does not, so the safe direction is kept. Labelled, so the reader
    knows this is the part to check (copy C6).
    """
    full = full_target_line(tool_name, arguments)
    if not full:
        return summary
    clipped = len(summary) >= _SUMMARY_LIMIT - 1
    if not clipped and all(value in summary for value in _argument_scalars(arguments)):
        return summary
    return f"{summary}. {_copy('full_request', request=full)}"
