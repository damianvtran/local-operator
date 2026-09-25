"""The machine-wide desktop event feed: one authenticated stream per BACKEND.

WHY THIS EXISTS. Every notification channel before this one was PER SESSION. A
``notification`` frame rides the SSE stream of a session's bridge, and a bridge
exists only while a route holds one — so a completion in session B, while the
desktop app is displaying session A, produced no composed frame, no banner and
nothing at all. The only remaining announcer was a running TUI's 1 s tick, and
with no TUI running a finished turn was announced by nobody.

This module is the missing channel: ONE stream for the whole process, owned by
``app.state.desktop_feed``, that composes and publishes ``notification`` frames
for sessions that have no bridge.

WHAT MAKES IT SAFE, and the property most likely to be broken by a later edit:

* **It acquires no bridge and spawns no runtime.** Its two dependencies —
  ``AttentionStore.state_many``/``revision`` and ``compose()`` over
  ``sessions/<id>/`` — are both bridge-independent. The tempting "reuse the
  bridge's composer" refactor would make watching a 200-row catalogue build 200
  cold facades and 200 SQLite poll loops, and would take the ``BRIDGE_COUNT``
  ceiling with it. ``tests/unit/server/test_desktop_feed.py`` asserts the
  absence directly (``DesktopSessions.bridges`` stays empty across a feed
  cycle).
* **It mints no second semantics.** The payload is built by the SAME function
  the bridge uses (``notifications.notification_payload``), so
  ``dedupe_key`` is byte-identical and the desktop's local claim map collapses
  the pair into one banner. The one field the feed derives differently is
  ``focus_policy`` — a ROUTING field, not content; see ``_focus_policy_for``.
* **It never replays.** ``notification`` and ``attention`` frames are live-only
  and the connection's first read is a BASELINE: it records the store's current
  revision and announces nothing that predates the connection. A reconnect
  therefore does not flood, which is the same rule the bridge's ``if previous:``
  guard and the store's no-flood bootstrap already apply.

COST. One poller per process, started with the first subscriber and stopped with
the last. Each tick is FOUR ``os.stat`` calls — the store, the two journal
sidecars it commits through, and ``run/mobile``, where the discovery records are
written (the status channel's doorbell) — the doorbell borrowed from
``config_watch.py``'s treatment of ``config.yml`` — with SQL only when one of
them actually moved, plus one bounded authoritative read every
``AUTHORITATIVE_RECOVERY_INTERVAL_S`` and one ``STATUS_PROBE_INTERVAL_S`` status
read. That is what makes detection p50 ~60 ms where the per-session poll's floor
was 1 s.

COST, CONTINUED — the two 1 s invalidation probes are NOT in that four-stat
figure, and the authoring one is O(profiles). The catalogue probe is a readdir
and two stats. The authoring probe cannot be, because the rows it watches are
FILES whose CONTENT is what changes: its price is one ``readdir`` of each
registry plus one ``os.stat`` per row (a few dozen stats on a machine with a few
dozen profiles), and a row's file is READ only when that stat moved — an
unchanged row costs a stat and no read at all. Both run on
``CATALOGUE_PROBE_INTERVAL_S``, one second, NOT on the 100 ms tick, so a quiet
tick still pays exactly the four stats above. The budget is deliberate and
measured (34 ``agent.yml`` read + filter + ``crc32`` = 1.16 ms against 56.95 ms
for a ``yaml.safe_load`` + re-dump of the same rows, both measured on this fleet;
a probe whose stat memory is warm costs 0.31 ms and ZERO file reads — 36
``stat``/``scandir`` calls for those 34 rows and the two readdirs); deleting the
per-file term to "restore" the four-stat profile would re-open the defect this
channel closes, so the count is stated here, in ``docs/DESKTOP_API.md`` and in the
budget test beside it.

THE AUTHORING CHANNEL. ``authoring`` frames say that the PROFILE and TEAM
registries moved — a role an agent just authored from inside a session, a team
created from the app. Nothing in this feed used to mention either: a session could
create a team and the sidebar's Teams/Agents lists kept showing yesterday's rows
until a refresh or a tab switch re-mounted the hook. The frame is the
``catalogue`` one's shape (a monotone ``revision``, no ``session_id``, at most one
frame per tick) and ``open`` carries ``authoring_revision`` beside
``catalogue_revision``. The ``authoring`` section of
``tests/unit/server/test_desktop_feed.py`` holds the two NEGATIVE pins that make it
worth trusting: a turn's in-place ``agent.yml`` rewrite and an identical
``system_prompt.md`` save must publish NOTHING.

THE PER-SESSION STATUS CHANNEL. ``session_status`` frames carry the DERIVED
``{code, label}`` for one session (the list's own precedence, via
``catalog.status_of``) plus a per-session monotone ``revision``, published only
when that pair actually changes. The row's status used to be refreshable only by
re-reading the whole list — up to 30 s, or a window focus — so an answered gate
and a completed turn arrived late on every row the user was not looking at.
Two clocks feed it, and neither is a re-read of the store: the 10 Hz doorbell
above sees every record write, and one 1 s authoritative probe covers the
transitions NO file write announces (``live -> wedged`` is an age crossing, and
``scheduled``/``dormant`` live in the wake index, outside ``run/mobile``). The
comparison is on the DERIVED PAIR, never on a file's mtime, which is what keeps
a 15 s heartbeat rewrite — and any other no-op republish — completely silent.
The comparison is on the pair with its CLOCK removed (``catalog.
status_dedupe_key``), because one label carries a live age and would otherwise
tick once a second on its own.

AND IT IS A READER, INCLUDING WHEN NOTHING HAS EVER RUN. The run directory is
resolved as a plain path and neither status read calls ``registry.run_dir`` — the
probe and the connection baseline decline to scan when the directory is absent —
so THE FEED creates nothing under ``run/``. Stated that narrowly on purpose
(review round 2, MINOR 1): it is not true of the backend, because the sibling
LIST read reaches the same helper — ``load_catalog`` → ``decorate_rows`` →
``registry.scan`` → ``run_dir()`` — so the desktop app's first ``GET
/v1/desktop/sessions`` creates ``run/mobile`` 0700 on a machine that has never
run a session. That is pre-existing at the base commit and unchanged here; the
point of saying so is that an operator must not read this paragraph as "an absent
run directory means no runtime has ever published".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, cast

from local_operator.notifications import notification_payload
from local_operator.notifications.compose import NotificationKind
from local_operator.resume import SessionRow
from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
from local_operator.server.utils.desktop_sessions import (
    BRIDGE_NOTIFIABLE_KINDS,
    REPLAY_BYTES,
    REPLAY_COUNT,
    SUBSCRIBER_COUNT,
    WATCH_TTL,
)
from local_operator.session import catalog
from local_operator.session.attention import AttentionStore
from local_operator.session.model_selection import session_uses_test_hosting
from local_operator.session.runtime import registry
from local_operator.session.runtime.presence import DesktopPresence, desktop_presence
from local_operator.session.runtime.types import RUN_DIRNAME, SessionRecord
from local_operator.tui.sidebar_pins import PINS_FILE
from local_operator.wakes.store import read_index

logger = logging.getLogger(__name__)

#: THE DOORBELL. FOUR ``os.stat`` calls per tick (the attention store, its two
#: journal sidecars, and ``run/mobile``) and no SQL, so the cost of looking is
#: separated from the cost of finding: the store is only opened when its own
#: ``(st_ino, st_size, st_mtime_ns)`` or its journal's moved, and the record
#: directory only moves the tick into reading a record whose OWN file moved.
#: 100 ms is the composure of a 1 s tick with the detection floor of a 10 Hz
#: one; the measured alternative — a 250 ms doorbell — costs 60 ms of p50
#: latency for a quarter of the stat load, which the profile in
#: ``TUI_BACKGROUND_RESPONSIVENESS`` gives no reason to want.
DOORBELL_INTERVAL_S = 0.10

#: Silence after which the stream emits a ``heartbeat`` frame. The client's
#: watchdog is ``heartbeat_seconds`` x 3, so a half-open socket after a
#: sleep/wake is detected by the client rather than reading as a live, quiet
#: stream — which is what the session stream's 15 s heartbeat is for too.
HEARTBEAT_INTERVAL_S = 15.0

#: How often the CATALOGUE revision is recomputed, deliberately slower than the
#: doorbell.
#:
#: The revision is a cheap invalidation token for the sidebar's row set, and it
#: is not free: it is a ``readdir`` of the sessions directory plus one stat, so
#: running it at 10 Hz would spend the doorbell's whole budget on a signal whose
#: consumer shows a page of rows. One second is 5x better than the 5 s
#: ``sessions.list`` poll this replaces and keeps the feed's own I/O profile
#: where the design bounds it (four stats per tick).
CATALOGUE_PROBE_INTERVAL_S = 1.0

#: The two AUTHORING registries under the config dir, and the file each row's
#: metadata lives in. Spelled here rather than imported: ``AgentRegistry`` and
#: ``TeamRegistry`` own these paths for their OWN writes, and importing either
#: module would pull ``dill`` and the rest of the agent-tool stack into the feed's
#: import graph to obtain two string constants.
AGENTS_DIRNAME = "agents"
TEAMS_DIRNAME = "teams"

#: The ``agent.yml`` keys an ordinary TURN rewrites — dropped from the authoring
#: probe's content projection, and the single most load-bearing line in this
#: channel.
#:
#: ``AgentRegistry.update_agent_state`` is the per-turn persistence path
#: (``server/utils/operator.py`` calls it after every turn): it funnels into
#: ``update_agent``, whose ``open("w")`` rewrites ``agent.yml`` IN PLACE — same
#: size or not, a new ``mtime_ns`` every time. What moved in that rewrite is
#: ``last_message``, ``last_message_datetime`` (``update_agent`` stamps it whenever
#: a message is passed) and ``current_working_directory`` (handed through by
#: ``update_agent_state``). Digesting the file WHOLE would therefore fire the
#: ``authoring`` frame on every turn of every chat and refetch the sidebar's
#: profiles and teams 1x per turn — precisely the defect shape this channel exists
#: to remove. Verified against the writers rather than inferred: ``save_agent`` and
#: ``create_agent`` write the whole row, ``update_agent`` the same, and this is the
#: only writer that moves anything without an authored change.
_AGENT_VOLATILE_KEYS = frozenset(
    {"last_message", "last_message_datetime", "current_working_directory"}
)

#: The same projection for ``team.yml`` — EMPTY on purpose rather than absent.
#: Every key that file carries (id, name, created_date, description, manager,
#: members) is authored, and ``save_team`` re-dumps the whole row for a no-op save.
#: Keeping the projection uniform is what lets a future volatile key be added in
#: one place instead of a second digest implementation.
_TEAM_VOLATILE_KEYS: frozenset[str] = frozenset()

#: A top-level YAML key line: ``<key>:`` at column 0, which is what both writers
#: emit for every field. Continuation lines — a block scalar's body, a block
#: sequence's ``- `` items, blank lines — never match it, so "which key is this line
#: under" is answered without a YAML parse.
_YAML_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")

#: THE AUTHORITATIVE STATUS CLOCK. Deliberately slower than the doorbell and
#: equal to ``CATALOGUE_PROBE_INTERVAL_S``: what it exists for is the transitions
#: NO FILE WRITE ANNOUNCES — ``live -> wedged`` is an age crossing
#: (``HEARTBEAT_TIMEOUT_S``), and ``scheduled``/``dormant`` live in the wake
#: index, not in ``run/mobile``. One second bounds both to a value a person
#: cannot distinguish from live, at one record read + one wake-index listdir per
#: second, the same class as the catalogue probe that already runs at this
#: cadence.
#:
#: IT DOES NOT REAP, which is why it is a read the list's own scan cannot be:
#: ``registry.scan(..., reap=False)`` returns the same verdicts while moving
#: nothing, so the feed never becomes the process that destroys another
#: runtime's record. The knob if the record population ever makes this matter is
#: THIS constant — a 5 s value still beats every bound the status channel is
#: designed to (the 30 s safety poll and the 5 s legacy poll) — rather than a
#: second mechanism.
STATUS_PROBE_INTERVAL_S = 1.0

#: THE BOUNDED AUTHORITATIVE RECOVERY PATH (review round 1, R1).
#:
#: The doorbell is an OPTIMISATION over a stat tuple, and R1's whole point is
#: that a stat tuple can be wrong: an ``mtime_ns`` the filesystem reuses, an
#: inode recycled by a rename-over, a sidecar replaced within one timestamp
#: granule.
#: Watching the sidecars closes the WAL case that finding names, but it cannot
#: make a "nothing moved" answer PROOF, and every consumer downstream treats a
#: quiet tick as "there is nothing to publish". So the revision — the value the
#: doorbell exists to avoid having to read — is read on its own slow clock as
#: well, and a tick that finds it moved publishes exactly as a doorbell tick
#: would.
#:
#: 30 s is chosen as a BOUND rather than as a latency: the fix that makes this a
#: safety net rather than a path is the sidecar staleness above, so what this
#: interval buys is the guarantee that no missed change can outlive it. Its cost
#: is one ``SELECT`` on a quiet store every 300 ticks, which is why it can be
#: this slow — a `revision()` over the ledger is the per-row work the doorbell's
#: own comment says must not run at 10 Hz.
AUTHORITATIVE_RECOVERY_INTERVAL_S = 30.0

#: How often a PERSISTENT tick failure is reported at WARNING. ``_poll_loop``
#: keeps the poller alive through a bad tick, which is right, but at DEBUG the
#: one failure mode this channel has — a status channel that has gone silently
#: dead — looked exactly like a quiet machine (QA Q3). The first failure is
#: reported at WARNING immediately, and then at most once per this interval while
#: it persists, so a persistent failure is visible without a traceback per tick.
TICK_FAILURE_WARNING_INTERVAL_S = 60.0

#: The shortest gap between two DOORBELL-fallback probes (review NIT 1). The
#: fallback exists for a session's FIRST record, which has no cached file to
#: compare against; a stream of ticks where the directory moved and no cached
#: record did (a staged write caught mid-rename, a new pid per tick) would
#: otherwise spend a full scan per tick, up to 10 Hz. One per
#: ``STATUS_PROBE_INTERVAL_S`` is the tightest limit that loses nothing: the
#: probe's own clock is stamped by whichever probe ran, so a suppressed fallback
#: is always covered within one interval by the regular one — the fallback can
#: never make a first record slower than the probe's own promise.
UNATTRIBUTED_PROBE_MIN_INTERVAL_S = STATUS_PROBE_INTERVAL_S

#: THE BURST CEILING — at most this many individual banners per doorbell tick.
#:
#: Several user sessions can finish within a second (a fleet of subagent-owning
#: sessions, a machine that was asleep and woke), and an uncapped feed turns
#: that into a stack of banners: the worst possible notification-centre
#: experience and the one thing a user will disable the feature over. The
#: remainder is still claimed-complete — the frames below speak for it — and is
#: reported as ONE digest frame naming the count, so nothing is silently
#: dropped.
#:
#: Kept EQUAL to the TUI's ``_BACKGROUND_NOTIFY_MAX_PER_TICK`` by a test rather
#: than by convention: the two are the same promise on two transports, and two
#: hand-maintained 3s are one edit away from disagreeing about it.
BURST_LIMIT = 3


def _authoring_projection(text: str, volatile: frozenset[str]) -> str:
    """The AUTHORED lines of one registry row, with the volatile ones dropped.

    A LINE FILTER rather than a parse, because it runs on the feed's one-second
    clock: read + filter + ``crc32`` over 34 ``agent.yml`` measures 1.16 ms, where
    ``yaml.safe_load`` + re-dump of the same rows measures 56.95 ms (both measured
    on this fleet). The parse is not affordable at 1 Hz for the amount it buys here:
    the projection only has to answer "did anything the user AUTHORED move", and
    the keys that answer it are exactly the lines.

    WHAT IT CANNOT SEE, stated because a projection is only as good as its
    filter: a key this module does not know is volatile, and that a future writer
    starts stamping per turn, would fire a frame per turn again. The projection is
    therefore pinned by a test (``test_the_authoring_projection_drops_the_turn_keys``)
    and by the two negative pins above it, rather than trusted to stay true.
    """
    kept: list[str] = []
    keeping = True
    for line in text.splitlines():
        match = _YAML_KEY_RE.match(line)
        if match is not None:
            keeping = match.group(1) not in volatile
        if keeping:
            kept.append(line)
    return "\n".join(kept)


def _authoring_digest(path: Path, volatile: frozenset[str]) -> int | None:
    """``crc32`` of one row file's projected content, or ``None`` when it is absent.

    A STABLE digest rather than ``hash()``, for the catalogue token's own reason:
    Python salts string hashing per process, so a ``hash()`` here would report a
    change to every client that reconnects to a restarted backend. Absent is a
    TERM rather than an error (``None``) — the create of a row's file moves the
    token, and the delete of one moves it back.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # A row mid-rewrite, or one this process cannot read: both are "no content
        # to compare yet" rather than a reason to take the poller down. The next
        # probe reads the finished file.
        return None
    return zlib.crc32(_authoring_projection(text, volatile).encode()) & 0x7FFFFFFF


def _fingerprint(path: Path) -> tuple[int, int, int] | None:
    """``(st_ino, st_size, st_mtime_ns)``, or ``None`` when it is not there.

    The comparison this feeds is borrowed wholesale from ``config_watch``, which
    does exactly this for ``config.yml``: three cheap fields that together move
    on any write worth reacting to, and no SQLite connection to open when they
    have not.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


#: The sidecars SQLite commits into, which the doorbell must watch as well.
#:
#: WHY THE MAIN FILE ALONE IS NOT THE DOORBELL (review round 1, R1). A writer's
#: ``_connect`` touches ``attention.db`` BEFORE its transaction commits, and in
#: WAL mode the commit itself lands in ``-wal`` while the main file may not move
#: again until a checkpoint. A tick landing in that window caches the new
#: main-file fingerprint against the OLD database contents, and every later
#: quiet tick then returns early — the completion stays unannounced until some
#: unrelated write happens to move the main file. Journal mode is a property of
#: whoever opened the database (another process may enable WAL under us), so
#: both sidecars are watched unconditionally rather than probed for.
_DB_SIDECAR_SUFFIXES = ("-wal", "-journal")


def _db_fingerprint(path: Path) -> tuple[tuple[int, int, int] | None, ...]:
    """The database's fingerprint TOGETHER WITH its journal sidecars.

    Still pure stats and still no SQLite connection, so the doorbell's cost
    stays in the same class the design bounds it to; it is three stats on a WAL
    database instead of one. A sidecar that does not exist contributes ``None``,
    which is itself a term: a checkpoint that deletes ``-wal`` is a change, and
    reading its absence as "nothing happened" is the same hole in the other
    direction.
    """
    return (_fingerprint(path), *(_fingerprint(Path(f"{path}{s}")) for s in _DB_SIDECAR_SUFFIXES))


@dataclass(eq=False)
class FeedSubscription:
    """One live SSE client of the feed.

    ``baseline_completion_sequence`` is this connection's own floor: a frame that describes
    a completion published BEFORE this client connected is not news to it, and
    the desktop's rule (Q5) is that a completion missed while the app was away
    is recovered by the durable unseen mark rather than by a late banner. The
    filter is per subscription rather than per poller because a second window
    that connects later must not inherit the first one's longer history.

    THE FLOOR IS IN THE DURABLE COMPLETION DOMAIN, NOT THE ENVELOPE'S (review
    round 1, R3, corroborated by QA Q1). It used to be ``self.sequence`` — the
    count of SSE frames this process had emitted — and was then compared against
    ``completions.sequence``, a SQLite AUTOINCREMENT. Those two counters are not
    related: one publish produces several frames, so the envelope counter runs
    ahead by a widening margin. The comparison therefore failed in BOTH
    directions, which is why a reconnecting client lost a genuinely new
    completion (the envelope floor had run past the new row's durable sequence)
    and could equally be handed a pre-connect one (the durable counter had run
    ahead of a quiet envelope). Comparing a durable number to a durable number
    is the whole fix; the baseline is read from ``MAX(completions.sequence)``.
    """

    id: str
    baseline_completion_sequence: int
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    #: Serialized size of each queued entry, in step with ``queue`` because the
    #: two are only ever appended to and popped from together. Kept here rather
    #: than recomputed on read so the backlog bound costs one ``json.dumps`` per
    #: frame per subscriber instead of two.
    queued_sizes: deque[int] = field(default_factory=deque)
    queued_bytes: int = 0
    overflow: bool = False


class DesktopFeed:
    """The process singleton behind ``GET /v1/desktop/events``.

    One instance per HTTP server, created lazily by the route beside the
    ``DesktopSessions`` pool. Not a bridge user: nothing here acquires a
    session, and the only ``DesktopSessions`` contact is the read-only
    ``bridged`` callback, which asks which sessions already have a stream and
    therefore must not be duplicated. ITS KEYS ARE ``session/<id>``, the same
    domain as the rows it is compared against (review round 1, R10) — the route
    used to pass bare session ids here, which made the whole exclusion dead
    code rather than merely wrong.
    """

    def __init__(
        self,
        root: Path,
        *,
        bridged: Callable[[], Any] | None = None,
        presence: DesktopDeliveryPublisher | None = None,
    ) -> None:
        self.root = root
        self.sessions_dir = root / "sessions"
        #: The sidebar's own pin index, watched by the catalogue probe so a pin
        #: written by EITHER surface moves the invalidation token. Taken from the
        #: store rather than spelled here, so the file the TUI writes and the file
        #: this probe stats cannot drift apart.
        self.pins_path = root / PINS_FILE
        #: A fresh epoch per PROCESS start. Frames carry it so a client can tell
        #: "the backend restarted" from "the stream stuttered", exactly as the
        #: session stream's epoch does.
        self.epoch = secrets.token_hex(8)
        self.sequence = 0
        self.subscribers: dict[str, FeedSubscription] = {}
        self.store = AttentionStore(root / "attention.db")
        #: The delivery lease this process publishes for its own subscribers.
        #: Owned here rather than by the route so its lifetime is the feed's: a
        #: claim is only ever believed while the socket that made it is alive.
        self.presence = presence or DesktopDeliveryPublisher(root)
        self._bridged = bridged or (lambda: frozenset())
        self._task: asyncio.Task[None] | None = None
        #: The first subscriber's open snapshot must follow its asynchronous
        #: no-replay baseline. Otherwise a write between the two reads can be
        #: absent from ``open`` and then adopted by the baseline without a frame.
        self._baseline_ready: asyncio.Future[None] | None = None
        self._revision: tuple[int, int, int] | None = None
        self._published_sequence = 0
        self._acknowledgements: dict[str, int] = {}
        #: Widened with the doorbell itself (R1): the fingerprint is now the
        #: database PLUS its journal sidecars, so it is a tuple of tuples rather
        #: than one ``(ino, size, mtime_ns)``.
        self._fingerprint: tuple[tuple[int, int, int] | None, ...] | None = None
        #: Set when a tick read its delta but could not finish processing it
        #: (R5). The cursors committed in ``_emit_delta`` are left untouched in
        #: that case, so the work is still owed — and this flag is what tells
        #: the next tick to retry it even though the doorbell has since gone
        #: quiet. Without it the retry would have to wait for an unrelated write.
        #:
        #: INITIALISED HERE, and that is not bookkeeping: ``_tick`` reads it in an
        #: ``or`` beside the fingerprint comparison, so ``or`` SHORT-CIRCUITS and
        #: the attribute is only reached on a tick where the fingerprint did NOT
        #: move — a genuinely quiet feed. Leaving it implicit would therefore have
        #: passed every test whose ticks follow a write (all of them) and crashed
        #: on the first idle tick in production.
        self._delta_pending = False
        #: The durable completion sequence this client's view already covers.
        #: See ``FeedSubscription`` for why it is not the envelope counter.
        self._supersede_cursor = 0
        #: When the bounded authoritative revision read last ran (R1).
        #: Monotonic, and initialised to 0 so the FIRST tick pays for it.
        self._revision_probed_at = 0.0
        self._catalogue_probed_at = 0.0
        #: THE CATALOGUE REVISION A CLIENT SEES, and it is a monotone COUNTER
        #: rather than the membership token it started as (finding 8). The row
        #: SET can be invalidated by two causes — a session entering or leaving
        #: the catalogue, and a row's derived ORDER KEY changing, which moves it
        #: between "Active chats" and "Previous chats" or to another slot inside
        #: one — and the client's refetch
        #: effect re-runs only on a value it has never seen. A token that is a
        #: hash of the directory can repeat, and cannot express the second cause
        #: at all; a counter that only ever rises cannot do either. The ``open``
        #: snapshot reports this same value, so a connecting client's view is
        #: expressed in the currency the frames use.
        self._catalogue_revision = 0
        self._catalogue_names: tuple[str, ...] = ()
        #: The membership TOKEN the counter's last comparison was made against
        #: (the probe's ``(inode, mtime, dirname set)`` answer). Kept beside the
        #: counter rather than inside it: the token answers "did the set move",
        #: the counter answers "has the client seen this state".
        self._catalogue_token: int | None = None
        #: Set when something OTHER than a membership move invalidates the row
        #: set — an ORDER-KEY transition (finding 8) — and cleared when the frame
        #: carrying it is published.
        self._catalogue_invalidated = False
        #: The tick the counter last published in, so a burst of simultaneous
        #: transitions costs ONE refetch rather than N: the second and later
        #: causes in a tick leave the flag up and the next tick (~100 ms) carries
        #: them.
        self._catalogue_emitted_tick = -1
        self._tick_index = 0

        # -- the authoring channel (profiles and teams) -------------------------
        #: The two registries a session can AUTHOR into: ``agents/<id>/agent.yml``
        #: (the profile/role rows the ``agent`` tool writes) and
        #: ``teams/<id>/team.yml`` (the rows ``POST /v1/desktop/teams`` writes).
        #: Paths only — nothing here constructs a registry, because a registry
        #: mkdirs on construction and this feed must never be the process that
        #: creates a directory (see the module docstring's reader rule).
        self.agents_dir = root / AGENTS_DIRNAME
        self.teams_dir = root / TEAMS_DIRNAME
        #: When the authoring token was last recomputed. Its own clock rather than
        #: the catalogue's, so each probe can be gated independently in a test —
        #: the two measure different claims — while both run on
        #: ``CATALOGUE_PROBE_INTERVAL_S``. No second interval constant: the cadence
        #: is the invalidation cadence, and a second number here would be a second
        #: clock to keep in step for no reason.
        self._authoring_probed_at = 0.0
        #: The monotone counter clients compare against, for the catalogue
        #: counter's own reason (a repeatable hash token cannot tell a client "this
        #: is new"). ``open`` reports this value, so a connecting client's view is
        #: expressed in the currency the frames use.
        self._authoring_revision = 0
        #: The last token the counter was compared against — ``None`` until the
        #: connection baseline sets it, which is what keeps a reconnecting client
        #: from being told about a change that predates its connection.
        self._authoring_token: int | None = None
        #: Set when the token moved and cleared when the frame carrying it is
        #: published.
        self._authoring_invalidated = False
        #: The tick the counter last published in, so a burst of authored rows — a
        #: plan that creates four profiles — costs ONE refetch rather than four.
        self._authoring_emitted_tick = -1
        #: Per-row stat memory for the content projection: ``path -> (stat
        #: fingerprint, projected digest)``. This is what makes the probe O(stats)
        #: rather than O(reads) — a row whose stat did not move is never re-read,
        #: and an idle probe reads nothing at all (pinned by
        #: ``test_the_authoring_probe_reads_nothing_when_nothing_moved``). Rebuilt
        #: per probe from the rows that probe actually saw, so a deleted row cannot
        #: leave an entry behind.
        self._authoring_files: dict[Path, tuple[tuple[int, int, int] | None, int | None]] = {}

        # -- the per-session status channel ------------------------------------
        #: The discovery-record directory (``run/mobile``). Resolved HERE as a
        #: plain path rather than through ``registry.run_dir``, which mkdirs and
        #: chmods, and the resolution is only half the promise: ``registry.scan``
        #: opens with its own ``run_dir`` call, so the two reads below decline to
        #: SCAN while the directory is absent (review round 1, MAJOR 1 / QA Q2 —
        #: ``_probe_status`` and ``_prime_status``). Absent therefore stays absent:
        #: ``_fingerprint`` answers ``None``, the doorbell is silent, and the
        #: directory is created by the runtime that publishes the first record —
        #: through ``registry.publish``, i.e. by a WRITER, never by this reader.
        self._registry_dir = root / RUN_DIRNAME
        #: ``(st_ino, st_size, st_mtime_ns)`` of that directory. EVERY record
        #: write is a staged write + rename IN it (``registry._staged_write``),
        #: so this one stat sees every status edge and every 15 s heartbeat.
        self._registry_fingerprint: tuple[int, int, int] | None = None
        #: ``session_id -> (record, verdict)``: the last thing each record said.
        #: Keyed by SESSION rather than by path because every consumer here asks
        #: by session (the frame carries one), and because it copies the list's
        #: own rule: ``decorate_rows`` folds its scan into ``live[session_id]``
        #: too, so a session with two records — a resumed conversation under a
        #: new pid, briefly — resolves the same way on both surfaces.
        self._records: dict[str, tuple[Any, str]] = {}
        #: ``session_id -> Path``, so the doorbell can re-stat one record without
        #: a ``readdir`` of the run directory.
        self._record_paths: dict[str, Path] = {}
        #: ``session_id -> the fingerprint its record was read AT``. THIS is what
        #: makes a quiet tick cost exactly one stat: a record is re-read only
        #: when its own file moved, never because a neighbour did.
        self._record_fingerprints: dict[str, tuple[int, int, int] | None] = {}
        #: ``session_id -> attention state``, kept between the delta and the
        #: probe: the published pair is derived from it, and the store is asked
        #: only for ids it has never answered for.
        self._attention: dict[str, dict[str, Any]] = {}
        #: ``session_id -> wake-index entry``, refreshed by the 1 s probe. Only
        #: ``wakes``/``wakes_dormant`` are read out of it, and it MUST live on
        #: this clock: the index is outside ``run/mobile``, so no record write
        #: announces an armed or dormant wake.
        self._wake_index: dict[str, dict[str, Any]] = {}
        #: ``session_id -> (code, label)`` as last PUBLISHED. The whole dedupe:
        #: a heartbeat rewrite moves ``heartbeat_at`` and nothing the pair is
        #: derived from, so it publishes nothing.
        #:
        #: The key stored here is ``catalog.status_dedupe_key`` — the same pair
        #: with the CLOCK term removed — so the one label that carries a live age
        #: (the ``wedged`` sentence) cannot turn the clock itself into an edge
        #: (review round 1, MINOR 2).
        self._status_seen: dict[str, tuple[str, str]] = {}
        #: ``session_id -> rank tuple``, the derived ORDER KEY
        #: (``catalog.order_key_of``) as of the last edge published for that row.
        #: A change here is a change the row's own frame cannot express —
        #: placement travels on a LIST read — so it invalidates the catalogue
        #: (finding 8).
        #:
        #: The KEY, not the SECTION, because a row can move WITHIN its section and
        #: that move is just as invisible to the client: a session that is already
        #: Active and finishes goes tier 4 -> 1 with ``active`` True -> True, so a
        #: section comparison publishes nothing for it — measured on the real
        #: backend as ZERO catalogue frames across ~100 accelerated ticks while the
        #: client's next list read led with the completed row. Comparing the key
        #: fires iff ``rank_entries``' sort key moved and it subsumes the section
        #: rule, since a section move always changes the key's first term.
        #:
        #: COMPARED FOR EVERY CANDIDATE, NOT ONLY THE PAIR-CHANGED ONES (review
        #: round 1, M1). The dedupe pair is NOT a superset of this key, which is why
        #: the comparison cannot sit behind the pair gate: a reachable collision
        #: spans ``('scheduled', 'Scheduled (N wakes)')`` at rank ``(6, 0, …)`` (a
        #: cold row with an armed wake, ``active`` False) and rank ``(5, 2, …)``
        #: (that row live and detached with the same wake still armed, ``active``
        #: True) — a SECTION move whose pair is byte-identical, reproduced through
        #: the real writers with the comparison gated and ZERO frames published. So
        #: the key is derived for every candidate the tick sees, before the gate: one
        #: ``entry_for`` build per candidate per tick (measured ``order_key_of``
        #: 1.0 µs per candidate = +0.50 ms per tick over 485 candidates, against
        #: the 3.50 ms 1 Hz membership probe this tick already pays on the same
        #: store), never per row per frame, and
        #: the once-per-tick coalescing below is unchanged.
        #:
        #: THE DIAL, if a flapping wake index ever makes this too eager: compare
        #: ``key[0]`` (the category) only. For Active rows that is exactly
        #: equivalent — Active's ``wake_rank`` is the constant — and it removes the
        #: one new churn vector, at the price of leaving an intra-Previous wake
        #: reorder to the client's own 30 s safety poll. That churn is REAL, not
        #: theoretical (qa round 1, Q1): a genuine ``PermissionError`` on the wake
        #: index makes ``read_index`` return ``{}``, a COLD armed row's band flips
        #: armed -> plain and back, and each flip costs one invalidation (measured
        #: 1 + 1 frames against the base's 0 + 0), self-healing in one probe. It is
        #: accepted rather than dialled down because in that same degraded state the
        #: row's scheduled GLYPH already flaps, so the position agreeing with the
        #: glyph is consistent rather than a new class of lie.
        self._position_seen: dict[str, tuple[int, int, float, str]] = {}
        #: ``session_id -> monotone counter``, bumped only when a frame is
        #: actually published and travelled to the list route (``status_stamps``)
        #: so a client can discard a list that was computed before the frame it
        #: has already applied.
        self._status_revision: dict[str, int] = {}
        #: When the authoritative status probe last ran. Monotonic, and
        #: initialised to 0 so the FIRST tick pays for it.
        self._status_probed_at = 0.0
        #: When the unattributed-move FALLBACK last probed, on its own clock
        #: rather than the probe's (review NIT 1). See
        #: ``_maybe_probe_unattributed``: the fallback is what carries a
        #: session's first record on the 10 Hz doorbell, so it cannot be gated by
        #: a stamp the connection baseline set a moment ago — but a stream of
        #: unattributed moves must not cost a scan per tick either.
        self._unattributed_probed_at = 0.0
        #: Set when a tick found a record had moved but could not finish
        #: publishing it, mirroring ``_delta_pending`` and for the same reason:
        #: the reads that DID succeed have already advanced
        #: ``_record_fingerprints``, so the retry gate has to be this flag rather
        #: than the stat comparison.
        self._record_pending = False
        #: ``session_id -> is_user_session``. The 1 s probe asks this per record
        #: and per wake-index id, and the underlying read is a per-directory
        #: marker read — affordable once per connection (``_snapshot``) and not
        #: once per session per second. Primed in a worker thread by the
        #: catalogue probe, which already lists the name set at 1 Hz, so the
        #: usual miss here is a directory created inside the last second.
        self._user_cache: dict[str, bool] = {}

    # -- subscribers -------------------------------------------------------

    def subscribe(self) -> FeedSubscription:
        """Register a subscriber and start the poller if it is not running.

        Raising when the table is full is deliberate and matches the session
        stream: a client that cannot be served must be told, not quietly given a
        stream that will never carry anything.
        """
        if len(self.subscribers) >= SUBSCRIBER_COUNT:
            raise RuntimeError("too many desktop feed subscribers")
        # THE FLOOR IS READ BEFORE THE SUBSCRIPTION IS PUBLISHED, deliberately
        # (R3). ``self.subscribers`` is what the fan-out iterates, so a
        # completion that lands between the two statements below must be treated
        # as POST-connect: it happened after this client asked to be told about
        # things. Reading the floor afterwards would instead swallow exactly that
        # completion, which is the suppression QA Q1 reproduced. The leftover
        # window is the harmless direction — a completion landing mid-handshake
        # is both announced and present in the ``open`` snapshot, and the
        # desktop's own claim map collapses the pair into one banner.
        subscription = FeedSubscription(
            id=secrets.token_hex(8), baseline_completion_sequence=self.store.revision()[0]
        )
        self.subscribers[subscription.id] = subscription
        self._ensure_poller()
        return subscription

    def unsubscribe(self, subscription: FeedSubscription) -> None:
        """Drop a subscriber, its presence claim, and the poller if it was last."""
        self.subscribers.pop(subscription.id, None)
        self.presence.drop(subscription.id)

    async def events(self, subscription: FeedSubscription) -> AsyncIterator[dict[str, Any]]:
        """Yield ``open`` then every subsequent frame, with heartbeats.

        ``open`` is produced HERE rather than fanned out, because its payload is
        this connection's snapshot (attention state and the catalogue revision)
        and there is nothing to share: two clients that connect a second apart
        must take two snapshots, or the second one's baseline is a lie.
        """
        try:
            yield await self._open_frame(subscription)
            while True:
                try:
                    frame = await asyncio.wait_for(
                        subscription.queue.get(), timeout=HEARTBEAT_INTERVAL_S
                    )
                except TimeoutError:
                    yield self._frame("heartbeat", {"ts": time.time()})
                    continue
                if frame is None:
                    # The backlog bound tripped. Say so and close: the client
                    # reconnects and takes a fresh snapshot, which is the only
                    # honest recovery from a gap it will never see the middle of.
                    yield self._frame(
                        "gap", {"reason": "overflow", "subscription_id": subscription.id}
                    )
                    return
                if subscription.queued_sizes:
                    subscription.queued_bytes -= subscription.queued_sizes.popleft()
                yield frame
        finally:
            self.unsubscribe(subscription)

    async def close(self) -> None:
        """Stop the poller and withdraw the lease. Idempotent."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown
                pass
        self.subscribers.clear()
        self.presence.close()

    # -- frame construction ------------------------------------------------

    def _frame(
        self, frame_type: str, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """Advance the receipt cursor and stamp one frame.

        The envelope's SHAPE is the session stream's (``session_id``, ``epoch``,
        ``seq``, ``type``, ``payload``) so the desktop's existing relay needs no
        new parser. ``session_id`` rides only the types that concern one
        session: the feed is not a session, and a fabricated id on ``open``
        would make the client's ``observe(sessionId, frame)`` API look like it
        had a session to attribute a catalogue event to.
        """
        self.sequence += 1
        frame: dict[str, Any] = {"epoch": self.epoch, "seq": self.sequence, "type": frame_type}
        if session_id is not None:
            frame["session_id"] = session_id
        frame["payload"] = payload
        return frame

    def _publish(
        self,
        frame_type: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        since_sequence: int | None = None,
    ) -> None:
        """Fan a frame out to every subscriber that should see it.

        ``since_sequence`` is the per-subscriber baseline filter, applied only
        to frames that describe a completion: ``attention`` is a level (a stale
        one is corrected by the next, and the merge is revision-guarded) while
        ``notification`` is an edge whose whole value is timeliness. Both sides
        of the comparison are the durable completion sequence (R3).
        """
        frame = self._frame(frame_type, payload, session_id=session_id)
        size = len(json.dumps(frame))
        for subscription in list(self.subscribers.values()):
            if (
                since_sequence is not None
                and subscription.baseline_completion_sequence >= since_sequence
            ):
                continue
            if subscription.overflow:
                continue
            if (
                subscription.queued_bytes + size > REPLAY_BYTES
                or subscription.queue.qsize() >= REPLAY_COUNT
            ):
                subscription.overflow = True
                subscription.queue.put_nowait(None)
                subscription.queued_sizes.append(0)
                continue
            subscription.queued_bytes += size
            subscription.queued_sizes.append(size)
            subscription.queue.put_nowait(frame)

    # -- the poller --------------------------------------------------------

    def _ensure_poller(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._baseline_ready = loop.create_future()
        self._task = loop.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Tick while anybody is listening; stop when nobody is.

        Registered with the first subscriber and torn down with the last
        (``_expire_watches``' shape): an idle backend must not hold a 10 Hz
        timer for a stream nobody is reading.
        """
        baseline_ready = self._baseline_ready
        try:
            try:
                await asyncio.to_thread(self._take_baseline)
            except Exception as error:  # noqa: BLE001 — fail the connection, not hang it
                if baseline_ready is not None and not baseline_ready.done():
                    baseline_ready.set_exception(error)
                raise
            else:
                # ``open`` waits on this handshake before taking its snapshot:
                # the baseline remains the connection boundary, and intervening
                # authoring writes can no longer be hidden by a later token read.
                if baseline_ready is not None and not baseline_ready.done():
                    baseline_ready.set_result(None)
            failures = 0
            # ``-inf`` rather than 0.0: the comment below says the FIRST failure
            # is reported immediately, and with 0.0 that was only true on a host
            # whose monotonic clock had already passed the interval (review round
            # 2, NIT 1).
            warned_at = float("-inf")
            while True:
                await asyncio.sleep(DOORBELL_INTERVAL_S)
                if not self.subscribers:
                    return
                try:
                    await self._tick()
                except Exception:  # noqa: BLE001 — one bad tick is not the feature
                    failures += 1
                    # ONE BAD TICK IS NOT THE FEATURE, but a bad tick EVERY tick
                    # is the feature going silently dead — and at DEBUG it looked
                    # exactly like a quiet machine (QA Q3). The first failure is
                    # reported at WARNING at once; while it persists the report is
                    # rate-limited, so a dead channel is visible without a
                    # traceback per tick.
                    logger.debug("desktop feed tick failed", exc_info=True)
                    now = time.monotonic()
                    if now - warned_at >= TICK_FAILURE_WARNING_INTERVAL_S:
                        warned_at = now
                        logger.warning(
                            "desktop feed tick failed %d time(s): the status and "
                            "attention channels are publishing nothing",
                            failures,
                            exc_info=True,
                        )
                else:
                    if failures:
                        logger.warning("desktop feed recovered after %d failed tick(s)", failures)
                        failures = 0
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("desktop feed poller stopped", exc_info=True)

    def _take_baseline(self) -> None:
        """Record the store's state as of the connection, and announce none of it.

        A completion published before the first subscriber arrived is HISTORY.
        Announcing it on connect is the flood the desktop's own Q5 decision
        refuses: what recovers a completion missed while the app was away is the
        durable ``unseen`` mark in the snapshot, not a banner about last night.
        """
        revision = self.store.revision()
        self._revision = revision
        self._published_sequence = revision[0]
        self._acknowledgements = self.store.acknowledgement_map()
        # The heal cursor baselines the same way and for the same reason (R4): a
        # heal that predates the connection is not news, and the ``open``
        # snapshot this client is about to receive already carries the corrected
        # state. Replaying it would republish a correction nobody is stale for.
        healed = self.store.superseded_since(0)
        self._supersede_cursor = int(healed[-1]["sequence"]) if healed else 0
        # THE STATUS BASELINE, on the same rule: a status that predates this
        # connection is history. Primed here WITHOUT publishing — the ``open``
        # snapshot's list read is what the client sees the current pair on, and
        # announcing it would be a frame per live session per connect.
        self._prime_status()

    def _prime_status(self) -> None:
        """Prime the status caches from one authoritative read, publishing none of it.

        Runs inside ``_take_baseline``, in the same worker thread, so the whole
        connection's cold reads stay off the event loop. The scan is the list's
        own (``check_zombie=None``, so the verdicts behind every later frame are
        the ones ``decorate_rows`` shows) and never reaps; the fingerprints taken
        here are what make the first doorbell tick after a connection cost
        exactly one stat.

        ``_status_seen`` is filled, not just the caches: that is the NO-REPLAY
        rule for this channel. A session whose status was already ``approval``
        when the client connected produces no frame until the pair CHANGES,
        which is the same rule ``_take_baseline`` applies to the attention
        revision and the attention baseline. Its ORDER KEY is primed beside it for
        the same reason (finding 8): a position change is a catalogue invalidation,
        and a baseline that did not record where each row already was would fire
        one on the first edge for every row.

        A MACHINE THAT HAS NEVER RUN A SESSION IS NOT ONE THIS CREATES THE RUN
        DIRECTORY FOR (review round 1, MAJOR 1). ``registry.scan`` opens with
        ``run_dir()``, which mkdirs and chmods, so the scan is skipped while the
        directory is absent — the rest of the prime still runs, because a session
        with no record has a status (``recent``) whether or not the directory
        exists, and a status the client's list already shows must not be
        announced as news when the first record does appear.

        PRIMED OVER EVERY USER SESSION, not only over the candidates an event
        could name. A session with no record, no wake and no attention has a
        status (``recent``) that the client's list already shows, and priming
        the pair as "unknown" would make the FIRST record write for that session
        -- even a proven-dead one, which is not a record at all for status
        purposes -- announce a pair nothing changed. The read is one
        ``state_many`` over the same name set ``_snapshot`` asks about, paid once
        per connection rather than per tick, which is the budget the feed's own
        docstring sets for connection-time work.
        """
        scanned = (
            registry.scan(self.root, check_zombie=None, reap=False)
            if self._registry_dir.is_dir()
            else []
        )
        records: dict[str, tuple[Any, str]] = {}
        paths: dict[str, Path] = {}
        fingerprints: dict[str, tuple[int, int, int] | None] = {}
        for record, state in scanned:
            session_id = str(getattr(record, "session_id", "") or "")
            if not session_id:
                continue
            path = self._registry_dir / f"{record.pid}.json"
            records[session_id] = (record, state)
            paths[session_id] = path
            fingerprints[session_id] = _fingerprint(path)
        self._records = records
        self._record_paths = paths
        self._record_fingerprints = fingerprints
        self._registry_fingerprint = _fingerprint(self._registry_dir)
        self._wake_index = read_index(self.root)
        # The user-session cache is primed by ``_catalogue_probe`` itself, which
        # is why this asks for no names of its own: the marker read is the one
        # per-directory cost ``_snapshot`` already calls affordable once per
        # connection, and the two now share it instead of paying it twice.
        _token, names = self._catalogue_probe()
        # The TOKEN is recorded as part of the baseline, not merely consulted: the
        # first tick compares against the row set the client was told about, so a
        # set that has not moved since the connection is not an invalidation —
        # the same no-replay rule the attention baseline applies to the revision.
        self._catalogue_token = _token
        # The AUTHORING baseline, on the same rule and for the same reason: a
        # profile or a team that predates this connection is not news to it (its
        # sidebar is loading both lists right now), and priming the token rather
        # than the clock means the first tick re-probes and still publishes
        # nothing — so a change landing in that first second is still caught.
        self._authoring_token = self._authoring_probe()
        candidates = [*records, *self._wake_index, *names]
        states = self.store.state_many([f"session/{session_id}" for session_id in candidates])
        for identity, state in states.items():
            self._attention[self._session_id(identity)] = state
        for session_id in candidates:
            row = self._row_for(session_id)
            if row is not None:
                attention = self._attention.get(session_id)
                self._status_seen[session_id] = catalog.status_dedupe_key(row, attention)
                # The ORDER KEY is primed BESIDE the pair rather than left absent:
                # an absent entry reads as "changed" on the row's first edge, which
                # would cost a connecting client one invalidation per row.
                self._position_seen[session_id] = catalog.order_key_of(row, attention)

    async def _tick(self) -> None:
        # The tick's own number, so a frame that may only be published once per
        # tick can say which tick it is in (the catalogue invalidation, finding 8).
        self._tick_index += 1
        # 1. THE DOORBELL. Three stats, no SQL, no connection — the database and
        # each journal sidecar (R1; see ``_db_fingerprint`` for why the main file
        # alone is not a doorbell).
        fingerprint = await asyncio.to_thread(_db_fingerprint, self.store.path)
        # THE BOUNDED AUTHORITATIVE RECOVERY (R1). A "nothing moved" answer is not
        # proof, so the revision is ALSO read on its own slow clock and a move
        # found there publishes exactly as a doorbell move would. Without this the
        # only recovery from a doorbell that cannot see a change is another,
        # unrelated change — which is what the finding says must not be the cure.
        now = time.monotonic()
        recovered = now - self._revision_probed_at >= AUTHORITATIVE_RECOVERY_INTERVAL_S
        # A tick that could not finish its delta left the cursors where they
        # were, so the work is still owed even though nothing has moved since:
        # the retry gate is the pending flag, NOT the stat comparison. Gating the
        # retry on the doorbell is exactly the hole R5 reports — the fingerprint
        # had already been committed against an event that was then dropped.
        if fingerprint != self._fingerprint or self._delta_pending or recovered:
            self._revision_probed_at = now
            # 2. THE REVISION GATE IS THE AUTHORITY, and the delta below is only
            # an optimisation. A heal moves neither `MAX(sequence)` nor
            # `SUM(acknowledged)` — it moves the `supersedes` counter — so a tick
            # that trusted the delta alone would miss it entirely.
            revision = await asyncio.to_thread(self.store.revision)
            if revision != self._revision:
                # Deliberately NOT wrapped in a try/except here: the exception has
                # to reach `_poll_loop`, which keeps the poller alive and logs it.
                # What matters is the ORDER — the pending flag goes up and the
                # cursors stay uncommitted until the emission has actually
                # happened, so a transient read failure costs a retry rather than
                # the event.
                try:
                    await self._emit_delta()
                except Exception:
                    self._delta_pending = True
                    # Retried on the next tick, which is ``DOORBELL_INTERVAL_S``
                    # away, so a persistently unreadable store costs one failing
                    # read per tick — the same rate a continuously-written store
                    # already pays for its revision.
                    raise
                self._revision = revision
            self._delta_pending = False
            self._fingerprint = fingerprint
        # 1b. THE RECORD DOORBELL — the status channel's edge clock. ONE stat on
        # ``run/mobile``: a staged write + rename in that directory moves its
        # mtime, so every status edge (a gate armed, a turn started, a routine
        # leaving) AND every 15 s heartbeat rings here. The tick is FOUR stats
        # now, still no SQL on a quiet tick, and the read below happens only for
        # a record whose OWN fingerprint moved — or, when NOTHING cached moved but
        # the directory did, once through the probe below (a session's first
        # record has no cached file to compare, and that is the case that
        # fallback exists for).
        registry_fingerprint = await asyncio.to_thread(_fingerprint, self._registry_dir)
        if registry_fingerprint != self._registry_fingerprint or self._record_pending:
            # Deliberately NOT wrapped: the exception has to reach ``_poll_loop``,
            # which keeps the poller alive. The ORDER is what matters — the
            # pending flag goes up and the cached fingerprints stay uncommitted
            # for the records that were not read, so a transient read failure
            # costs a retry rather than the event.
            try:
                moved = await asyncio.to_thread(self._reread_moved_records)
                if moved:
                    await self._publish_status_changes(moved)
                elif registry_fingerprint is not None:
                    # THE DIRECTORY MOVED AND NO CACHED RECORD DID. That is the
                    # session's FIRST record (there was no cached file to compare
                    # against, so the fast path has nothing to re-read), and it is
                    # also what a write caught mid-stage looks like. Looking
                    # properly now is what keeps that first edge on the 10 Hz
                    # clock instead of the probe's one-second one — rate-limited,
                    # because the second case can repeat on every tick (NIT 1).
                    await self._maybe_probe_unattributed()
            except Exception:
                self._record_pending = True
                raise
            self._record_pending = False
            self._registry_fingerprint = registry_fingerprint
        # 3. THE CATALOGUE HAS ITS OWN SCHEDULE (R2). Deliberately outside the
        # doorbell branch above: it was reached only when the attention database
        # had moved, so on a quiet store — which is the common case — session
        # directory create/remove never ran the one-second probe at all, and the
        # sidebar fell back to its 30 s safety poll for membership changes the
        # design promises in about a second.
        await self._maybe_emit_catalogue()
        # 4. THE STATUS CLOCK (``STATUS_PROBE_INTERVAL_S``). Outside the doorbell
        # branch for the SAME reason as step 3, and it is load-bearing here: the
        # two transitions this exists for — ``live -> wedged`` (an age crossing)
        # and a wake being armed or stopped (the index lives outside
        # ``run/mobile``) — never move the record directory at all, so a quiet
        # store is exactly the case the doorbell cannot cover.
        await self._maybe_probe_status()
        # 5. THE INVALIDATION THE STATUS CHANNEL OWES (finding 8). The probe above
        # is a second source of status edges — the ones no write announces — and a
        # status edge can also move its row between sections, which only a LIST
        # read carries. Publishing the invalidation HERE and not only in step 3 is
        # what keeps a section move inside the same tick as the frame that caused
        # it; the once-per-tick guard inside ``_maybe_emit_catalogue`` keeps a
        # burst of simultaneous transitions at one refetch.
        await self._maybe_emit_catalogue()
        # 6. THE AUTHORING INVALIDATION. Outside the doorbell branch, like steps 3
        # and 4, because neither registry lives under a directory the doorbell
        # stats at all — a profile an agent authored from a session, or a team
        # created from the app, moves nothing this tick already looks at. The rows
        # are FILES rather than a directory listing, so this probe is O(rows) in
        # stats: the number is stated in the module docstring and in the budget
        # test rather than left to be read as a regression of the four-stat tick
        # above — it runs on the 1 s cadence, not on the tick.
        await self._maybe_emit_authoring()

    async def _maybe_probe_unattributed(self) -> None:
        """The doorbell's unattributed-move fallback, RATE-LIMITED (NIT 1).

        The fallback is worth having: a session's first record has no cached file
        to compare against, so without it every session start would wait for the
        one-second clock instead of riding the 10 Hz doorbell. What the docstring
        used to claim was "one-shot" is in fact "on any tick that satisfies the
        condition" — a write caught mid-stage (temp file created, rename not yet
        landed) or a new pid per tick fires it at the doorbell's own rate, and
        each firing is a full scan that also stamps the probe's clock.

        #: The limit is the FALLBACK's own clock rather than the probe's: the
        #: probe's stamp is set by the connection baseline moments before the
        #: first record of a session is written, so gating on that one would
        #: suppress exactly the case the fallback exists for. A suppressed
        #: fallback is covered within one interval by the regular probe, because
        #: the clock that gates THAT one was stamped by the last probe of either
        #: kind — so the worst case for a first record is the probe's own promise
        #: (1 s) and not worse, while a stream of unattributed moves costs one
        #: scan per second instead of ten.
        """
        now = time.monotonic()
        if now - self._unattributed_probed_at < UNATTRIBUTED_PROBE_MIN_INTERVAL_S:
            return
        self._unattributed_probed_at = now
        await self._probe_status()

    async def _emit_delta(self) -> None:
        """Publish one ``attention`` frame per changed session, then banners.

        COMMIT-AFTER-PROCESSING, deliberately (R5). Every read this needs is
        gathered first and every cursor is committed LAST: ``_published_sequence``
        and ``_acknowledgements`` used to advance while the loop that consumed
        them was still running, so a ``state_many`` that raised on a locked
        database had already consumed the row — the poll loop stayed alive, the
        next tick saw an unchanged fingerprint and returned early, and that
        completion was then silent on every surface forever. Nothing here may
        write a cursor before the emission it describes has happened.
        """
        published, acknowledgements, superseded = await asyncio.gather(
            asyncio.to_thread(self.store.published_since, self._published_sequence),
            asyncio.to_thread(self.store.acknowledgement_map),
            asyncio.to_thread(self.store.superseded_since, self._supersede_cursor),
        )
        changed: list[str] = []
        fresh: list[dict[str, Any]] = []
        #: The cursors this tick would commit IF it finishes; held here rather
        #: than written to ``self`` until the publishes below have happened.
        published_sequence = self._published_sequence
        supersede_cursor = self._supersede_cursor
        for row in published:
            published_sequence = max(published_sequence, int(row["sequence"]))
            changed.append(str(row["conversation"]))
            if row["kind"] in BRIDGE_NOTIFIABLE_KINDS:
                fresh.append(row)
        for conversation, acknowledged in acknowledgements.items():
            if self._acknowledgements.get(conversation) != acknowledged:
                changed.append(conversation)
        # R4: the identities the ``supersedes`` term moved, which appear in
        # NEITHER of the two deltas above because a heal updates its row in
        # place. They join ``changed`` and so get a corrected ``attention`` frame
        # — and deliberately NOT ``fresh``, which is what keeps a heal from
        # producing a second banner: the healed row's own sequence is untouched,
        # so its ``unseen`` mark is unchanged and the read watermark still
        # governs it.
        for row in superseded:
            supersede_cursor = max(supersede_cursor, int(row["sequence"]))
            changed.append(str(row["conversation"]))

        identities = [
            item for item in dict.fromkeys(changed) if self._user_session(self._session_id(item))
        ]
        states: dict[str, dict[str, Any]] = {}
        if identities:
            states = await asyncio.to_thread(self.store.state_many, identities)
            for identity in identities:
                state = states.get(identity)
                if state is not None:
                    # No `supported` key here, deliberately: only a live runtime
                    # can answer it and the feed has none. The renderer's merge
                    # preserves the value it already holds rather than clearing
                    # it, which is what keeps the read receipt working for the
                    # session on screen.
                    self._publish("attention", state, session_id=self._session_id(identity))
            # THE SAME TICK'S ATTENTION FACTS ARE ALSO A STATUS EDGE. The derived
            # pair changes here more often than anywhere else — ``complete``,
            # ``error`` and ``interrupted`` are read off the completion the frame
            # above is about — and the states are already in hand, so this costs
            # no second read and no second derivation. It sits BEFORE the commit
            # point below, so a failure here retries the whole tick rather than
            # publishing half an edge.
            await self._publish_status_changes(identities, states)
        if fresh:
            await self._emit_notifications(fresh, states)

        # THE COMMIT POINT. Reached only if every read and every publish above
        # succeeded; an exception before this line leaves the previous cursors in
        # place and ``_tick`` raises the pending flag for the retry.
        self._published_sequence = published_sequence
        self._acknowledgements = acknowledgements
        self._supersede_cursor = supersede_cursor

    async def _emit_notifications(
        self, fresh: list[dict[str, Any]], states: dict[str, dict[str, Any]]
    ) -> None:
        """Compose a banner for each newly published, unseen, unbridged session.

        ``self._bridged()`` must answer in ``session/<id>`` keys, and must list a
        session only when its bridge has a LIVE subscriber that can notify:
        a bridge retained in the pool with nobody attached announces nothing, so
        excluding it here would leave the completion unannounced everywhere.

        TWO GATES KEEP THE TEST HOSTING OFF THE WIRE, and they are different
        questions rather than belt-and-braces. The process switch is "am I
        allowed to notify at all" — a backend started by a rig or a test has
        it on, and a backend the OPERATOR is running does not. The per-session
        read is "was this conversation ever run on the mock", which is the one
        the operator's own backend needs: it polls a store a rig may have left
        mock sessions in, from a process that never touched the mock itself.
        Without it, that store's conversations banner the operator with "Hello
        from the mock provider!" on the machine-wide channel.

        AND A THIRD, WHICH IS NOT ABOUT THE MOCK: the identity test
        (``desktop_belongs_to_this_process``) refuses the OFFER from a backend
        that is not the user's own run. The banner this composes is raised by
        another process — the desktop app attached here — under ITS bundle
        identity, so withholding the frame is the only place this repository can
        decline it, and a backend under a redirected ``HOME`` is a rig or a
        sandbox by the same reasoning the TUI's own OS legs use.
        """
        from local_operator.tui.notify import (
            desktop_belongs_to_this_process,
            notifications_enabled,
        )

        if not notifications_enabled():
            return
        if not desktop_belongs_to_this_process():
            return
        bridged = set(self._bridged())
        candidates: list[tuple[str, str, str, str, int]] = []
        # See below: filled on first use, then reused for every remaining row.
        presence: DesktopPresence | None = None
        for row in fresh:
            identity = str(row["conversation"])
            if identity in bridged:
                # THE STEADY-STATE GUARD against two banners for one completion.
                # A session with a live bridge has its own composer and its own
                # `focus_policy`; the feed yields to it rather than racing it.
                continue
            state = states.get(identity)
            if state is None or not state.get("unseen"):
                continue
            # ONE presence read per candidate SET (review round 2, R14), taken
            # lazily so a tick whose rows are all bridged still pays none: the
            # answer is identical for every row in this tick, and the read is a
            # mkdir + chmod + readdir + one read per record on disk, so doing it
            # per candidate made a fleet burst — the scenario this channel
            # exists for — pay N of them inside one tick against 1 before.
            if presence is None:
                presence = desktop_presence(self.root, cached=False)
            policy = self._focus_policy_for(self._session_id(identity), presence)
            if policy is None:
                continue
            if await asyncio.to_thread(
                session_uses_test_hosting, self.sessions_dir / self._session_id(identity)
            ):
                # A mock conversation in this store, whoever ran it. Asked LAST
                # of the candidate filters, because it is the only one that
                # reads a file: a store with a hundred finished real sessions
                # pays nothing for them (they stop at the cheap checks above),
                # and one banner's worth of work for the row this saves.
                #
                # OFF THE LOOP, because that read is not cheap: it is a backward
                # walk to the newest v2 row, measured at 57-745 ms on this
                # operator's three largest journals, and this line runs for every
                # candidate row. `session_uses_test_hosting` memoises its verdict
                # on the journal's `(mtime_ns, size)`, so the steady tick is a
                # `stat`; the thread hop is for the misses, which is the whole
                # reason a serve backend must not do this inline.
                continue
            candidates.append(
                (identity, str(row["token"]), str(row["kind"]), policy, int(row["sequence"]))
            )
        if not candidates:
            return
        candidates.sort(key=lambda item: item[4])
        for identity, token, kind, policy, sequence in candidates[:BURST_LIMIT]:
            try:
                payload = await asyncio.to_thread(
                    notification_payload,
                    cast(NotificationKind, kind),
                    session_dir=self.sessions_dir / self._session_id(identity),
                    token=token,
                    session_id=self._session_id(identity),
                    focus_policy=policy,
                )
            except Exception:  # noqa: BLE001 — chrome must not stop the channel
                # ONE UNREADABLE SESSION COSTS ONE BANNER, NEVER THE FEED. This
                # loop is driven by the poller task, so an exception escaping
                # here would end the tick that keeps every subscriber's attention
                # frames flowing — a background completion would then be silent
                # on every surface, which is the defect this module exists to
                # close. The same rule the bridge applies at `_maybe_publish_
                # notification` (T-B13), applied to the machine-wide channel.
                logger.debug("feed compose failed for %s", identity, exc_info=True)
                continue
            self._publish(
                "notification",
                payload,
                session_id=self._session_id(identity),
                since_sequence=sequence,
            )
        overflow = candidates[BURST_LIMIT:]
        if overflow:
            self._publish(
                "notification",
                self._digest_payload(overflow),
                session_id=self._session_id(overflow[-1][0]),
                since_sequence=overflow[-1][4],
            )

    def _digest_payload(self, overflow: list[tuple[str, str, str, str, int]]) -> dict[str, Any]:
        """One banner standing in for the completions the ceiling held back.

        Composed from the TUI's own digest vocabulary
        (``background_digest_title`` / ``BODY_BACKGROUND_DIGEST`` /
        ``digest_subtitle``) rather than minted here: the two transports make
        the same promise, and a second wording for "several sessions finished"
        is a second thing to keep in step. The count is the whole remainder —
        an absolute number, not one relative to the ceiling.
        """
        from local_operator.tui.notify import (
            BODY_BACKGROUND_DIGEST,
            background_digest_title,
            digest_subtitle,
        )

        kinds = [kind for _identity, _token, kind, _policy, _sequence in overflow]
        # The ids ride along so the client can land the click on the catalogue
        # rather than on an arbitrary member of the set — which is the one
        # decision a digest banner cannot make for the user.
        return {
            "contract": 1,
            "kind": "complete",
            "title": background_digest_title(len(overflow)),
            "status": digest_subtitle(kinds),
            "body": BODY_BACKGROUND_DIGEST,
            "body_is_snippet": False,
            "body_is_failure": False,
            "title_is_session_name": False,
            # No single completion owns this frame, so its dedupe key is keyed
            # on the SET. It must not collide with any member's own key: a
            # digest is not a duplicate of a per-session banner, it is the
            # announcement that several happened.
            "dedupe_key": "burst:" + ",".join(sorted(token for _i, token, _k, _p, _s in overflow)),
            "completion_token": None,
            "session_name": None,
            "focus_policy": "always",
            "burst_count": len(overflow),
            "session_ids": [self._session_id(identity) for identity, *_rest in overflow],
            # THE MEMBERS' OWN TOKENS (review round 1, R8), which is what makes a
            # digest arbitrable at all. The digest itself has no single
            # completion to claim — ``completion_token`` is deliberately None, so
            # the desktop's claim step skips it — and it used to carry only
            # member IDS. Nothing then marked those members delivered: the real
            # feed's own reproduction showed all three overflow members still
            # claimable after the digest had been emitted, so any later individual
            # frame for one of them (from another feed instance, from the TUI, or
            # from a re-delivery) was free to raise a SECOND banner for a
            # completion this digest had already announced, and the per-burst cap
            # did not bound OS banners at all.
            #
            # The contract is the SAME one a single frame uses, one level down:
            # a member is claimed through ``sessions.notified``, atomically, at
            # the moment the digest is about to be delivered — never when the
            # frame is merely queued, which is the preclaim the review forbids and
            # which would burn a completion for a banner the client then
            # suppressed by its own focus rule. A member another surface already
            # won simply is not this digest's to announce; the count stays the
            # backend's statement of what happened, and its click still lands on
            # the catalogue, where all of them are listed.
            "member_tokens": [
                {"session_id": self._session_id(identity), "completion_token": token}
                for identity, token, _kind, _policy, _sequence in overflow
            ],
        }

    def _focus_policy_for(self, session_id: str, presence: DesktopPresence) -> str | None:
        """``focus_policy`` for this completion, or ``None`` for "raise nothing".

        WHY THIS IS DERIVED AND NOT COPIED. ``focus_policy`` is a ROUTING field,
        not content. The per-session frame hard-codes ``when_unfocused``, and
        the desktop suppresses exactly that value while any window is focused —
        so shipping the bridge's payload verbatim meant the commonest state of
        all (user in the app on session A while B finishes) announced nothing,
        on any surface, at all: rung 2 had already silenced the runtime and the
        TUI. That is the operator's original symptom, made permanent by the
        presence mechanism that was supposed to fix it.

        So: a completion for a session the app is NOT displaying is ``always`` —
        the window's focus says nothing about whether the user wants to hear
        that a DIFFERENT conversation finished. A completion for the session the
        app IS attendedly displaying is rung 1: the card is in band on its own
        stream, no banner is raised, and ``None`` says so.

        READ UNCACHED, deliberately (review round 1, R1's presence half; QA round
        1's ``presence-unfocused``/``presence-hidden`` rows). This decision is
        TERMINAL — a suppressed completion is not re-decided later, the frame is
        simply never published — so it must not be made on up to
        ``PRESENCE_CACHE_TTL_S`` of focus the user has already left. Reading the
        cache here suppressed a completion that landed just after the user switched
        away from the window, and the runtime's own rung 4 defers whenever a desktop is
        reachable, so no surface raised it at all.

        THE READ IS THE CALLER'S, and it arrives as ``presence`` (review round
        2, R14). It used to be taken here, which made it one uncached read per
        CANDIDATE; the ticks that decide more than one banner therefore paid
        N filesystem reads where one serves the set. The terminal-decision
        argument is untouched by the move: the caller still reads on the tick
        that decides, so the answer is never the cache's.
        """
        if presence.attended and presence.session_id == session_id:
            return None
        return "always"

    # -- catalogue ---------------------------------------------------------

    async def _maybe_emit_catalogue(self) -> None:
        """The catalogue invalidation: the row SET moved, or a row's ORDER KEY did.

        TWO CAUSES, ONE COUNTER (finding 8). The membership token says the row
        set moved (one readdir, one second). The second cause is what the token
        cannot express at all: a row whose derived ORDER KEY
        (``catalog.order_key_of``) changed needs a LIST read to change PLACE —
        which SECTION it is filed in, or its slot inside one — and with "Previous
        chats" collapsed by default the user sees nothing at all until one
        happens, while a client only re-runs its refetch effect on a revision it
        has never seen. So the value published is a monotone counter, bumped once
        per invalidation whatever caused it, and the ``open`` snapshot reports that
        same counter.

        AT MOST ONE FRAME PER TICK: a burst of simultaneous transitions — a
        fleet starting, a batch finishing — costs one refetch, not N. The flag
        stays up and the next tick carries it, i.e. ~100 ms later.
        """
        now = time.monotonic()
        if now - self._catalogue_probed_at >= CATALOGUE_PROBE_INTERVAL_S:
            self._catalogue_probed_at = now
            token, names = await asyncio.to_thread(self._catalogue_probe)
            if token != self._catalogue_token:
                self._catalogue_token = token
                self._catalogue_names = names
                self._catalogue_invalidated = True
        if not self._catalogue_invalidated:
            return
        if self._catalogue_emitted_tick == self._tick_index:
            return
        self._catalogue_invalidated = False
        self._catalogue_emitted_tick = self._tick_index
        self._catalogue_revision += 1
        self._publish("catalogue", {"revision": self._catalogue_revision})

    def _catalogue_probe(self) -> tuple[int, tuple[str, ...]]:
        """A cheap invalidation token for the sidebar's ROW SET.

        The sessions directory's own ``(inode, mtime_ns)`` moves when a session
        is created or removed, and the NAME SET catches a create/delete that a
        same-nanosecond mtime would hide. Both come from one ``readdir``, with
        no per-directory stat: walking the store to notice that a transcript
        grew would be the per-row scan the 5 s ``sessions.list`` poll was
        retired for, and it would grow with the store exactly as that one did.

        What this therefore does NOT do, stated because it is a real limit: an
        in-place append to an existing transcript (a preview or a title that
        changed under a stable row set) does not move either term. The sidebar's
        30 s safety poll and its refetch on window focus are what cover that,
        and the row content a waiting user actually needs — the unseen mark —
        rides its own ``attention`` frame, which is not gated on this at all.

        ONE TERM IS NOT ABOUT THE ROW SET AT ALL, and it is here because the
        desktop plane's refetch on this token is the only mechanism that can
        deliver a PIN change made on another surface: the pin index is shared
        with the TUI, and its file is therefore read by this probe too. See the
        key below.
        """
        names: list[str] = []
        try:
            with os.scandir(self.sessions_dir) as entries:
                for entry in entries:
                    names.append(entry.name)
        except OSError:
            names = []
        names.sort()
        # PRIME THE USER-SESSION CACHE HERE, in the worker thread the caller
        # already put this on, and only for names it has never judged. The
        # marker read is the per-directory cost ``_snapshot`` calls affordable
        # once per connection; the status channel asks the same question per
        # record and per wake-index id EVERY second, so this is what turns that
        # from a per-session-per-second read into one read per NEW directory.
        #
        # REBUILT, not appended to: what this cache may hold is bounded by the
        # store AS IT IS, so a backend that stays up for weeks does not accumulate
        # one entry per directory the machine has ever created. The cost is a dict
        # rebuild per probe over the names this call just listed, and a name that
        # leaves the store simply costs one marker read if it ever comes back.
        previous = self._user_cache
        self._user_cache = {name: previous[name] for name in names if name in previous}
        for name in names:
            if name not in self._user_cache:
                self._user_cache[name] = self._is_user_session(name)
        # A STABLE digest, not `hash()`: the token is opaque to the client but it
        # is compared across reconnects, and Python's string hashing is salted
        # per process — so a `hash()` here would report a change to every client
        # that reconnects to a restarted backend, for no reason at all.
        key = (
            ",".join(names)
            + "|"
            + repr(_fingerprint(self.sessions_dir))
            # THE CROSS-SURFACE PIN STORE, and this term is a CORRECTNESS
            # requirement rather than an optimisation. The pins file is shared by
            # two front ends, so a pin made in the TUI has to reach the desktop
            # app with no manual refresh — and this token is the only thing that
            # can tell the feed to publish the `catalogue` frame the app's
            # sidebar already refetches on. Without it the pins move under a
            # token that did not change, the doorbell never rings, and the app
            # falls back to its 30 s safety poll: bounded, but not the "I just
            # pinned this in my terminal" feel this exists for.
            #
            # Cost: ONE extra `os.stat` per probe, and the probe runs once a
            # second (CATALOGUE_PROBE_INTERVAL_S), not once a tick. The store
            # writes by `os.replace`, so a pin (or unpin) always moves the
            # fingerprint and always publishes exactly one frame.
            + "|"
            + repr(_fingerprint(self.pins_path))
        )
        return zlib.crc32(key.encode()) & 0x7FFFFFFF, tuple(names)

    # -- authoring (profiles and teams) --------------------------------------

    async def _maybe_emit_authoring(self) -> None:
        """The authoring invalidation: a profile or a team was authored, or removed.

        THE THIRD COPY OF THE ``catalogue`` SHAPE, deliberately rather than a
        shared abstraction: the two channels differ in their probe (a readdir
        against a readdir plus an O(rows) content projection), in what a changed
        row means, and in nothing else — and the shape they do share is four
        lines of counter arithmetic whose only invariant is "monotone, at most
        one per tick". A helper taking a probe callable would hide the one thing
        a reader has to see here: WHICH probe can move without a write, and why.

        THE FRAME CARRIES ONLY THE REVISION. A burst collapses to at most one
        frame per tick, so a per-name diff — "these three profiles appeared" — is
        not expressible: the frame says "your profile and team lists are stale",
        and the lists the client already knows how to fetch are the answer. A
        second read per frame to build a diff would put back exactly the cost
        this channel removed.

        ``CATALOGUE_PROBE_INTERVAL_S``, not a constant of its own: both probes are
        invalidations of a sidebar list, the catalogue's own comment already argues
        one second as the cadence a person cannot distinguish from live, and a
        second interval would be a second clock to keep in step for no gain.
        """
        now = time.monotonic()
        if now - self._authoring_probed_at >= CATALOGUE_PROBE_INTERVAL_S:
            self._authoring_probed_at = now
            token = await asyncio.to_thread(self._authoring_probe)
            if token != self._authoring_token:
                self._authoring_token = token
                self._authoring_invalidated = True
        if not self._authoring_invalidated:
            return
        if self._authoring_emitted_tick == self._tick_index:
            return
        self._authoring_invalidated = False
        self._authoring_emitted_tick = self._tick_index
        self._authoring_revision += 1
        self._publish("authoring", {"revision": self._authoring_revision})

    def _authoring_probe(self) -> int:
        """A cheap invalidation token for the PROFILES and TEAMS registries.

        THREE TERMS: the row NAME SET of each registry (one ``readdir`` each) and
        the CONTENT PROJECTION DIGEST of each row's metadata file. The file term is
        what the name set cannot express — an edit that changes what a row SAYS —
        and it is a projection over the AUTHORED lines rather than a digest over the
        bytes, because the ordinary per-turn persistence path rewrites
        ``agent.yml`` in place on every turn (see ``_AGENT_VOLATILE_KEYS``):
        digesting the bytes would fire this frame on every turn of every chat.

        WHY NO ``_fingerprint()`` OF THE TWO DIRECTORIES, which is the one place
        this probe departs from the signed-off design — recorded here rather than
        quietly omitted, because a reader will ask why the catalogue's token has a
        directory stat and this one does not. A ``teams/`` mtime is not merely a
        redundant term beside the name set: ``save_team`` publishes EVERY save
        through ``tempfile.mkdtemp`` plus two ``os.replace`` calls INSIDE
        ``teams/`` (``_swap_row_directory_locked``), so its ``mtime_ns`` moves on a
        save that changed nothing at all — measured HERE, at implementation time,
        as a no-op ``save_team`` moving ``teams/``'s ``mtime_ns`` — and the frame
        would then fire once per team save, which is the class of spurious
        refetch this channel exists to remove. ``agents/``'s stat is the same term
        without the same harm (nothing writes a temp entry into it), so it is left
        out for the reason the name set is the whole story for create and delete
        in BOTH registries: a row's directory cannot appear or vanish without its
        NAME moving, and the name set is the readdir we already pay for.

        WHAT IT CANNOT SEE, the honest limit, stated because the client's backstop
        is what covers it: the per-row term is guarded by a STAT comparison, so an
        in-place edit that leaves ``(ino, size, mtime_ns)`` unchanged — a
        coarse-resolution filesystem, a hand-edited file restored with its own
        timestamp — is missed. The catalogue probe carries the same class of miss
        (its own comment says so), and the desktop app's refetch on window focus
        and on mount is the backstop for both. A profile's ``system_prompt.md``
        (which is where the ``agent`` tool stores INSTRUCTIONS, not ``agent.yml``)
        is deliberately NOT a term: it is not what either list renders, and adding
        it would put a second file per profile on the probe.

        A READER IN THE STRONG SENSE, like every other probe here: an absent
        directory is an empty row set and an absent row file is a ``None`` term.
        Nothing on this path mkdirs, touches, or repairs anything — a feed that
        created ``agents/`` would be creating registry state, which is what
        ``tests/unit/server/test_desktop_feed.py`` pins for this probe too.
        """
        parts: list[str] = []
        cache: dict[Path, tuple[tuple[int, int, int] | None, int | None]] = {}
        for directory, filename, volatile in (
            (self.agents_dir, "agent.yml", _AGENT_VOLATILE_KEYS),
            (self.teams_dir, "team.yml", _TEAM_VOLATILE_KEYS),
        ):
            names = self._authoring_row_names(directory)
            parts.append(",".join(names))
            for name in names:
                path = directory / name / filename
                fingerprint = _fingerprint(path)
                previous = self._authoring_files.get(path)
                if previous is not None and previous[0] == fingerprint:
                    digest = previous[1]
                else:
                    digest = _authoring_digest(path, volatile)
                cache[path] = (fingerprint, digest)
                parts.append(f"{name}={digest}")
        # The stat memory is REBUILT from the rows this probe saw, never appended
        # to, for the catalogue's own reason: what it may hold is bounded by the
        # registry as it is, so a backend up for weeks does not accumulate one
        # entry per row the machine has ever had.
        self._authoring_files = cache
        return zlib.crc32("|".join(parts).encode()) & 0x7FFFFFFF

    @staticmethod
    def _authoring_row_names(directory: Path) -> tuple[str, ...]:
        """The row names directly under one registry directory, sorted — one readdir.

        DOT-PREFIXED ENTRIES ARE NOT ROWS, which is ``TeamRegistry._load``'s own
        rule and load-bearing here rather than defensive: the team writer stages
        and swaps every save through ``.<id>.<rand>`` and ``.<id>.backup.<rand>``
        directories INSIDE ``teams/``, so counting them would make the probe
        report a moved token during a save that changed nothing. A team id is
        validated to ``[A-Za-z0-9._-]`` and an agent id is a uuid4, so no authored
        row can be hidden.
        """
        try:
            with os.scandir(directory) as entries:
                return tuple(
                    sorted(entry.name for entry in entries if not entry.name.startswith("."))
                )
        except OSError:
            # Absent, or unreadable: no rows, and no attempt to create it.
            return ()

    # -- the per-session status channel -------------------------------------

    def status_stamps(self) -> tuple[str, dict[str, int]]:
        """``(epoch, {session_id: revision})`` — what the list route stamps rows with.

        THE ONE NEW PUBLIC METHOD, and it exists because the list is a SECOND
        writer of the same fact: ``sessions.list`` ships ``status`` on every row,
        and a response computed before a frame this client has already applied
        would otherwise clobber it (the in-app marker effect fires a list on
        exactly the transition these frames speed up, so that race is the normal
        path, not a hypothetical). The epoch says which process produced the
        counter and the revision says how far it had got; the client keeps the
        frame's value when its epoch matches and its revision is higher.

        Returned as a COPY. The route holds it while ``rows()`` walks the store in
        a worker thread, and a later tick bumping a counter under it must not move
        a number that was already handed out as "the state when this list was
        computed".
        """
        return self.epoch, dict(self._status_revision)

    def _user_session(self, session_id: str) -> bool:
        """``_is_user_session``, with the answer remembered per session.

        Both the 10 Hz delta and the 1 s probe ask this for every candidate, and
        the underlying read opens a file in the session directory — so both go
        through here, including ``_emit_delta``'s identity filter, which is the
        one place on the 10 Hz path that used to ask the uncached predicate
        directly (review round 1, MINOR 4). The two are interchangeable for a
        bare id (the predicate accepts either spelling). The cache is primed by
        ``_catalogue_probe`` in a worker thread; a miss here is therefore a
        directory created inside the last second (or a record for a session the
        catalogue's name set has not listed yet).
        """
        cached = self._user_cache.get(session_id)
        if cached is None:
            cached = self._is_user_session(session_id)
            self._user_cache[session_id] = cached
        return cached

    def _row_for(self, session_id: str) -> SessionRow | None:
        """The row the LIST builds for one session, from this feed's own caches.

        A transcription of ``decorate_rows``' live-state mapping, field for
        field, because the two must agree about the row the pair is derived from
        — the parity test is what holds them together. What is NOT transcribed is
        the precedence itself: that comes from ``catalog.status_of``, which builds
        the same ``CatalogEntry`` the list builds.

        ``None`` for anything that is not a user session, which is the
        subagent/origin filter the rest of the feed already applies.

        ``mtime``/``name``/``created_at`` are left at their defaults: the status
        properties read none of them (they rank rows, which this does not do).
        """
        if not self._user_session(session_id):
            return None
        cached = self._records.get(session_id)
        live_state = ""
        pending: str | None = None
        leaving = ""
        kind = ""
        age: float | None = None
        # The two subagent counts are a hand transcription of
        # ``decorate_rows``' live mapping, exactly like ``live_state``/``leaving``
        # above and for the same reason: the pair this feed publishes on
        # ``session_status`` must be the pair the LIST derives for the same
        # on-disk state, and a row that is ``delegating`` in the list while the
        # feed's copy reads ``idle`` is precisely the divergence the parity test
        # exists to catch. ``None`` when there is no record — never ``0``, which
        # would be a count nobody reported (see ``SessionRow.delegating``).
        subagents_running: int | None = None
        subagents_queued: int | None = None
        if cached is not None and cached[1] != "stale":
            # ``stale`` IS TREATED AS NO RECORD, and that is not a shortcut: the
            # pid is gone, so nothing the record says about work in progress is
            # true any more. ``decorate_rows`` takes the same rule (review round
            # 1, MINOR 1), so the list and the frame agree even in the poll that
            # performs the sweep that moves the record aside — and this feed must
            # never be that sweep (``reap=False``).
            record, state = cached
            if state == "wedged":
                live_state = "wedged"
            elif record.busy:
                live_state = "busy"
            elif not record.detached:
                live_state = "attached"
            else:
                live_state = "idle"
            pending = record.pending or None
            kind = str(record.kind or "")
            leaving = str(record.leaving or "")
            subagents_running = getattr(record, "subagents_running", None)
            subagents_queued = getattr(record, "subagents_queued", None)
            # The age from the same owner the list asks, and without the zombie
            # probe for the same reason ``decorate_rows`` gives: the verdict has
            # already been reached, so a ``ps`` fork here would buy nothing. It
            # is one ``kill(pid, 0)``-class check per candidate.
            age = registry.classify(record, check_zombie=False).heartbeat_age_s
        entry = self._wake_index.get(session_id) or {}
        schedules = entry.get("schedules") or () if isinstance(entry, dict) else ()
        return SessionRow(
            session_id,
            0.0,
            "",
            live_state=live_state,
            pending=pending,
            leaving=leaving,
            subagents_running=subagents_running,
            subagents_queued=subagents_queued,
            wakes=len(schedules),
            wakes_dormant=bool(isinstance(entry, dict) and entry.get("stopped_at")),
            kind=kind,
            heartbeat_age_s=age,
        )

    def _reread_moved_records(self) -> list[str]:
        """Re-read EXACTLY the records whose own file moved, in a worker thread.

        ZERO READS ON A QUIET TICK, which is the whole point of the extra stat:
        the run directory's fingerprint says "something in here changed", and
        this says "and here is which record" — without a ``readdir``, and without
        opening anything the caller did not need.

        A record whose file is GONE is a change too: ``unpublish`` on a clean exit
        removes it, and another process's sweep may move it to ``reaped/``. Both
        drop out of the cache so the pair is recomputed from no record at all,
        which is what the list will show on its own next read.

        The verdict uses ``check_zombie=False``: a record that just moved was
        written by a live owner, and this path runs on a file write. A ``ps``
        fork per write would be a new cost on the commonest event in the system —
        every 15 s per live session — to re-answer a question the write itself
        just answered.

        FINGERPRINT BEFORE READ is the safe order: a write landing in between
        leaves a stale cached fingerprint, and the next tick re-reads. The
        reverse order would cache a value the file had already moved past.
        """
        moved: list[str] = []
        for session_id in list(self._records):
            path = self._record_paths.get(session_id)
            if path is None:
                continue
            fingerprint = _fingerprint(path)
            if fingerprint == self._record_fingerprints.get(session_id):
                continue
            if fingerprint is None:
                self._records.pop(session_id, None)
                self._record_paths.pop(session_id, None)
                self._record_fingerprints.pop(session_id, None)
                moved.append(session_id)
                continue
            try:
                record = SessionRecord.from_json(json.loads(path.read_text()))
            except (OSError, ValueError, TypeError):
                # Unreadable is not a status. Treated as NO RECORD, which is what
                # the list shows for the same file (its scan deletes an
                # unparseable record; this one leaves the file alone because it
                # is a reader) and what the next successful write corrects.
                logger.debug("desktop feed could not read record %s", path, exc_info=True)
                self._records.pop(session_id, None)
                self._record_paths.pop(session_id, None)
                self._record_fingerprints.pop(session_id, None)
                moved.append(session_id)
                continue
            verdict = registry.classify(record, check_zombie=False)
            self._records[session_id] = (record, verdict.state)
            self._record_fingerprints[session_id] = fingerprint
            moved.append(session_id)
        return moved

    async def _maybe_probe_status(self) -> None:
        """The one-second gate in front of :meth:`_probe_status`."""
        now = time.monotonic()
        if now - self._status_probed_at < STATUS_PROBE_INTERVAL_S:
            return
        await self._probe_status()

    async def _probe_status(self) -> None:
        """The authoritative status read: the transitions no write announces.

        ``registry.scan(..., reap=False)`` is the very call ``decorate_rows``
        makes — same read, same classify, ``check_zombie=None`` so the verdicts
        behind every frame are the ones the list shows — minus the sweep, because
        a 1 Hz reaper on the feed's own poller would unlink other processes'
        evidence. The wake index is outside ``run/mobile`` entirely (that is why
        it needs this clock at all), and ``live -> wedged`` is an AGE crossing
        with no file write anywhere.

        The candidate set is a UNION of four populations rather than a diff:
        every record the scan returned, every record that VANISHED since the last
        probe (an exit whose directory move the doorbell happened to miss), every
        session the wake index names — a wake can be armed for a session with no
        record at all, which is the cold-row case the sidebar shows — and every
        session that LEFT the index since the last probe.

        THAT LAST TERM IS NOT OPTIONAL, and it is the one place this differs from
        the obvious reading of "every session in the wake index": a session whose
        wake is DISARMED (fired, or stopped) is no longer in the index, so a
        candidate set built from the current index can only ever announce
        ``scheduled``/``dormant`` and never the way back — which would leave the
        row showing a wake that is not coming until the client's 30 s poll. The
        previous index is already in memory, so the term is a set difference and
        costs no I/O.

        Runs in worker threads, both reads at once. The clock is stamped HERE
        rather than by the caller so the doorbell's unattributed-move fallback
        consumes the same one-second slot: an early probe is a probe.

        A MACHINE THAT HAS NEVER RUN A SESSION KEEPS IT THAT WAY (review round
        1, MAJOR 1 / QA Q2). ``registry.scan`` opens with ``run_dir()``, which
        mkdirs and chmods, so the scan is asked for only while the directory is
        there — this reader must not be the thing that creates ``run/mobile``
        0700, and on a machine with no runtime there is nothing in it to read.
        The wake index is read either way: it lives OUTSIDE ``run/mobile``, and a
        wake armed for a session with no record at all is exactly the cold-row
        case this clock exists for.
        """
        self._status_probed_at = time.monotonic()
        scanned: list[tuple[Any, str]] = []
        if self._registry_dir.is_dir():
            scanned, wake_index = await asyncio.gather(
                asyncio.to_thread(registry.scan, self.root, check_zombie=None, reap=False),
                asyncio.to_thread(read_index, self.root),
            )
        else:
            wake_index = await asyncio.to_thread(read_index, self.root)
        records: dict[str, tuple[Any, str]] = {}
        paths: dict[str, Path] = {}
        fingerprints: dict[str, tuple[int, int, int] | None] = {}
        for record, state in scanned:
            session_id = str(getattr(record, "session_id", "") or "")
            if not session_id:
                continue
            path = self._registry_dir / f"{record.pid}.json"
            # Last-wins, exactly like ``decorate_rows``'s ``live[session_id]``.
            records[session_id] = (record, state)
            paths[session_id] = path
            fingerprints[session_id] = _fingerprint(path)
        vanished = [session_id for session_id in self._records if session_id not in records]
        disarmed = [session_id for session_id in self._wake_index if session_id not in wake_index]
        self._records = records
        self._record_paths = paths
        self._record_fingerprints = fingerprints
        self._wake_index = wake_index
        candidates = [*records, *vanished, *wake_index, *disarmed]
        if candidates:
            await self._publish_status_changes(candidates)

    async def _publish_status_changes(
        self, session_ids: list[str], states: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Publish one ``session_status`` frame per session whose PAIR changed.

        THE WHOLE EMISSION RULE, in one place:

        * a session that is not a user session is never published (the same
          filter the attention delta and the banners use);
        * the pair is ``catalog.status_of``'s — the list's own derivation, not a
          second copy of the precedence;
        * a pair equal to the last one PUBLISHED for that session publishes
          nothing, which is what makes a 15 s heartbeat rewrite (and every other
          no-op republish: a title change, ``started``, a de-duped
          ``set_busy``) completely silent. The comparison is on the derived pair
          with its CLOCK term removed (``catalog.status_dedupe_key``) and never on
          a file's mtime, which is what lets "status changed" mean exactly that
          rather than "a wedged row's age ticked"; the frame still carries the
          pair, age and all;
        * a change to the row's ORDER KEY publishes a catalogue invalidation, in the
          same tick, so the client's list read re-files it instead of leaving it in
          the wrong slot until the 30 s safety poll (finding 8). This comparison is
          NOT inside the pair gate above: the pair is not a superset of the key
          (see the ``_position_seen`` field comment for the collision and the cost),
          so it is made for every candidate this tick derived. A SECTION move is the
          subsumed case: the reported symptom is a row that reorders INSIDE "Active
          chats" — a session that is already Active and finishes is tier 4 -> 1 with
          ``active`` True -> True — which a section comparison cannot see at all;
        * COMMIT LAST: ``_status_seen`` and the revisions advance only after every
          frame has been fanned out, so a failure mid-way costs a retry of the
          whole set rather than a half-published edge — the same rule
          ``_emit_delta`` states for its own cursors.

        ``states`` is the attention map a caller has ALREADY read (``_emit_delta``
        has it in hand), keyed by store identity; ids missing from both it and the
        cache are filled in ONE ``state_many`` read, which is this path's only
        SQL. It is also why a record edge and an attention edge in the same tick
        cost one derivation and no extra query.
        """
        ids = [self._session_id(item) for item in dict.fromkeys(session_ids)]
        ids = [session_id for session_id in ids if session_id]
        if not ids:
            return
        if states:
            for identity, state in states.items():
                self._attention[self._session_id(identity)] = state
        missing = [session_id for session_id in ids if session_id not in self._attention]
        if missing:
            fetched = await asyncio.to_thread(
                self.store.state_many, [f"session/{session_id}" for session_id in missing]
            )
            for session_id in missing:
                state = fetched.get(f"session/{session_id}")
                if state is not None:
                    self._attention[session_id] = state
        pending: list[tuple[str, tuple[str, str], int]] = []
        published_keys: dict[str, tuple[str, str]] = {}
        moved_positions: dict[str, tuple[int, int, float, str]] = {}
        revisions = dict(self._status_revision)
        for session_id in ids:
            row = self._row_for(session_id)
            if row is None:
                continue
            attention = self._attention.get(session_id)
            # POSITION — where the sidebar files the row: the ORDER KEY
            # (``catalog.order_key_of``, the key ``rank_entries`` sorts by), not
            # merely its section. DERIVED AND COMPARED FOR EVERY CANDIDATE, ahead of
            # the pair gate below, because the pair is not a superset of this key: a
            # row moving out of "Previous chats" into "Active chats" can keep a
            # byte-identical pair (``('scheduled', 'Scheduled (N wakes)')`` spans
            # ``(6, 0, …)`` cold-and-armed and ``(5, 2, …)`` live-detached-and-armed),
            # and a gated comparison published nothing for it (review round 1, M1).
            # The row is already built above, so the added cost is one entry build
            # per CANDIDATE per tick, ~1 µs — not per row per frame.
            # The client cannot express this itself: placement travels on a list
            # read, so a row that reorders keeps its old slot until the next one —
            # measured at 7.5-8.9 s for a section move, and NEVER for a move inside a
            # section, which is the reported bug (a completed row kept its place in
            # "Active chats" while the backend's own list had already led with it).
            #
            # RECORDED, NOT COMMITTED, until the frames are out — see below.
            position = catalog.order_key_of(row, attention)
            if self._position_seen.get(session_id) != position:
                moved_positions[session_id] = position
            key = catalog.status_dedupe_key(row, attention)
            if self._status_seen.get(session_id) == key:
                continue
            pair = catalog.status_of(row, attention)
            revision = revisions.get(session_id, 0) + 1
            revisions[session_id] = revision
            pending.append((session_id, pair, revision))
            published_keys[session_id] = key
        for session_id, pair, revision in pending:
            self._publish(
                "session_status",
                {"code": pair[0], "label": pair[1], "revision": revision},
                session_id=session_id,
            )
        # COMMIT LAST (see the docstring), and it applies to the POSITION map too
        # (review round 2, MINOR 2): advancing ``_position_seen`` up in the build
        # loop meant a raising ``_publish`` lost the move for good — the next tick
        # re-derives the same pair, finds the position already recorded, leaves the
        # invalidation unset, and the row sits in the wrong slot until the 30 s
        # poll, which is the symptom this mechanism exists to remove. Committing it
        # beside ``_status_seen`` gives the retry the status side already had: a
        # failure costs the whole set a retry rather than a half-committed edge.
        for session_id, key in published_keys.items():
            self._status_seen[session_id] = key
        for session_id, position in moved_positions.items():
            self._position_seen[session_id] = position
        self._status_revision = revisions
        if moved_positions:
            self._catalogue_invalidated = True

    # -- identity helpers --------------------------------------------------

    def _is_user_session(self, identity: str) -> bool:
        """Whether this store key is a conversation a person started.

        A subagent child is a machine's delegated run, not a conversation to
        banner about, and children live as siblings under ``sessions/`` — the
        one filter keeps them off the feed entirely. The ``session/`` prefix is
        checked too because an agent transcript's key is ``agent/<name>``, which
        is NOT unique across parents and must never be mistaken for a session
        id.

        Accepts either a store key (``session/<id>``) or a bare directory name,
        because both callers have a different one in hand.
        """
        from local_operator.resume import is_user_session

        session_id = self._session_id(identity)
        if identity != session_id and not identity.startswith("session/"):
            return False
        try:
            return is_user_session(self.sessions_dir / session_id)
        except Exception:  # noqa: BLE001 — an unreadable marker is not a session
            return False

    @staticmethod
    def _session_id(identity: str) -> str:
        """The 12-hex session id a store key names."""
        return identity.split("/", 1)[1] if "/" in identity else identity

    # -- the open frame ----------------------------------------------------

    async def _open_frame(self, subscription: FeedSubscription) -> dict[str, Any]:
        """The connection's snapshot: attention state and the two invalidation counters.

        The poller's no-replay baseline must finish first. If the snapshot ran
        before it, an authoring write between those reads could be adopted by
        ``_take_baseline`` without appearing in either the snapshot or a frame.
        """
        baseline_ready = self._baseline_ready
        if baseline_ready is not None:
            # A disconnect cancels this waiter, not the feed-wide handshake that
            # other subscribers still need to build their own open snapshots.
            await asyncio.shield(baseline_ready)
        elif self._authoring_token is None:
            # A synchronous subscriber has no poller task, but still needs the
            # same first-connection boundary before its open snapshot.
            await asyncio.to_thread(self._take_baseline)
        attention, catalogue_revision, authoring_revision = await asyncio.to_thread(self._snapshot)
        return self._frame(
            "open",
            {
                "subscription_id": subscription.id,
                "heartbeat_seconds": HEARTBEAT_INTERVAL_S,
                "lease_seconds": WATCH_TTL,
                "watch_ttl_seconds": WATCH_TTL,
                "catalogue_revision": catalogue_revision,
                "authoring_revision": authoring_revision,
                "attention": attention,
            },
        )

    def _snapshot(self) -> tuple[dict[str, dict[str, Any]], int, int]:
        """Every user session's attention state, plus both invalidation counters.

        Deliberately NOT the catalogue's rows: those carry a preview read per
        row, and putting that on the feed would move the sidebar's cost rather
        than remove it. The client already has its rows; what it cannot know
        without this is which of them are unread.

        Runs once per connection, so the per-directory marker read behind
        ``is_user_session`` is affordable here in a way it is not on the
        doorbell — this is one ``sessions.list``-shaped scan minus the previews,
        paid by a client that is opening a stream rather than by a 10 Hz timer.
        THAT READ IS SHARED with the status baseline (``_prime_status`` primes
        ``_user_cache`` through ``_catalogue_probe``), so the marker is read once
        per session per connection rather than once per caller.

        The probe's TOKEN is recorded and the COUNTER is reported (finding 8):
        the token is what the feed compares against next tick, while the counter
        is the value a client compares against, so a snapshot that reported the
        token would hand a connecting client a number no ``catalogue`` frame has
        ever published — and the next real invalidation could then repeat it.
        """
        token, names = self._catalogue_probe()
        self._catalogue_token = token
        self._catalogue_names = names
        self._catalogue_probed_at = time.monotonic()
        identities = [f"session/{name}" for name in names if self._user_session(name)]
        states = self.store.state_many(identities) if identities else {}
        # The AUTHORING counter is reported WITHOUT re-probing the registries, which
        # is the one place the two invalidation channels' snapshots differ. The
        # catalogue probe above is already being paid here (its names are what the
        # attention baseline is built from), so priming ITS token is free; the
        # authoring probe is an O(rows) stats walk with no other caller on this
        # path, and paying it per connection to set a token whose only job is to
        # suppress the next tick's frame would be buying one refetch's silence with
        # a scan. The no-replay rule is instead owned by the connection BASELINE
        # (`_prime_status`, which every poller start runs): a profile or a team that
        # predates the connection is already in the lists the client is loading.
        return states, self._catalogue_revision, self._authoring_revision
