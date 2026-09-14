"""The ``serve`` daemon's rendezvous record: which install is listening where.

``local-operator serve`` used to be invisible to every other process on the
machine. Three daemons could be live at once — ``:1111``, ``:7341`` and
``:8080``, each one a different build — and each answered ``/health`` with
nothing but a version string, so "the backend" was whatever answered 200 first.
Two questions had no answer at all: WHICH install is this, and is it the one I
was talking to a minute ago?

This module owns the record that answers them. It lives in its own namespace
(``SERVE_RUN_DIRNAME`` — see the comment there for why it is not ``run/mobile``)
and carries the pid, the address actually bound, the identity of the INSTALL
(``prefix``, ``install_kind``, ``version``, ``source_ref``), and an
``instance_id`` minted at startup that a probe can compare against the
``/health`` of the process it actually reached.

**What is NOT here, deliberately:** the staged write, the 0600 file under a
0700 directory, and the ``live``/``wedged``/``stale`` classification. Those are
:mod:`local_operator.session.runtime.registry`'s, shared by handing it this
namespace's ``dirname`` and deserializer. A second implementation is how the
two would come to disagree about what "alive" means — and the reader that got
the other answer would be the one nobody was looking at.

**Import-light:** stdlib plus that session registry (itself stdlib plus two
light local modules already on the CLI startup path). ``local_operator.update``
is imported FUNCTION-LOCALLY, house style for a module that sits beside
``server/app.py``'s startup path: it reaches ``importlib.metadata``,
``urllib`` and ``subprocess``, none of which a process that merely reads a
record should pay for.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from local_operator.session.runtime import registry as session_registry
from local_operator.session.runtime.types import HEARTBEAT_INTERVAL_S, SERVE_RUN_DIRNAME

#: ``app.state`` attribute carrying the address ``cli.serve_command`` announced.
#:
#: A state attribute and not a module global, because two ASGI apps can live in
#: one process (a test's and a daemon's) and an announcement must not leak from
#: one to the other; and not the environment, because this is the announce that
#: makes the environment unnecessary — see :func:`announce_address`.
ANNOUNCED_STATE_ATTR = "serve_announced_address"

#: The ``--reload``-only announcement channel: ``"<announcer pid> <host> <port>"``.
#:
#: Under ``--reload`` uvicorn re-imports ``server.app`` in a CHILD process from
#: an import string, so the app object cannot be reached and the inherited
#: environment is the only channel that survives into it. The announcer's pid
#: travels INSIDE the value rather than beside it, because the two must not be
#: separable: an announcement is only ours when the process that made it also
#: spawned us (:func:`advertised_address`). That is exactly uvicorn's reload
#: child — ``uvicorn._subprocess.get_subprocess`` spawns it with
#: ``multiprocessing`` — and it is what stops a process that merely INHERITED
#: the variable from publishing its ancestor's listener as its own.
SERVE_ANNOUNCE_ENV = "LOCAL_OPERATOR_SERVE_ANNOUNCE"

#: ``app.state`` flag: this process's address arrived through
#: :data:`SERVE_ANNOUNCE_ENV`, i.e. uvicorn's ``--reload`` supervisor spawned us.
#:
#: Set by :func:`advertised_address` at the one moment the answer is knowable —
#: the announcement is read-and-CLEARED there, so "where did my address come
#: from" cannot be asked again afterwards. It is what lets ``server/app.py``
#: keep a DEV-MODE supervisor out of the build watch: see
#: :func:`is_reload_child` for why that is a correctness rule and not a
#: convenience.
RELOAD_CHILD_STATE_ATTR = "serve_reload_child"

#: Port ``0`` in a record means "not announced", i.e. undialable. Nothing in
#: this module writes one any more — a boot that was not announced publishes no
#: record at all — but a record written by another build still has to parse, and
#: a reader must treat it as no address rather than as port zero.
ANNOUNCED_PORT_UNKNOWN = 0


#: Hosts that name "every interface" rather than a dialable address, mapped to
#: the loopback form of their own family. See :func:`_dialable_host`.
_WILDCARD_LOOPBACK = {"0.0.0.0": "127.0.0.1", "::": "::1"}


def _dialable_host(host: str) -> str:
    """The host a reader should DIAL for a listener bound to ``host``.

    ``--host 0.0.0.0`` (or ``::``) means "listen on every interface"; it is not
    an address anybody can connect to, and the design's dial string
    (``http://{host}:{port}/health``) cannot name a wildcard. The same listener
    is reachable on its own family's loopback, so that is what the record
    carries — a record is only useful if the address in it works — and the port
    is unchanged.

    Explicit addresses are recorded verbatim: rewriting them would be guessing
    at a route, and an IPv6 literal needs brackets when a dialer turns it into a
    URL, which is the dialer's job (as it already is for ``::1``).
    """
    return _WILDCARD_LOOPBACK.get(host, host)


def announce_address(app: Any, host: str, port: int) -> None:
    """Announce the bound address to the app THIS process is about to serve.

    Called by ``cli.serve_command`` after it has bound the listener, so the
    record and ``/health`` name the port the kernel actually gave us rather
    than the ``--port`` argument. That distinction is the whole point of
    ``--port 0``: the UI's own child asks for an ephemeral port, and a record
    that named ``0`` would be unusable exactly where it is needed most.

    Deliberately NOT written to the environment: this process may spawn
    children (an agent's shell tool, a wrapper, another entry point booting the
    same app) and an inherited variable would let any of them publish OUR
    listener as its own — a rendezvous record for a daemon that is not there.
    """
    setattr(app.state, ANNOUNCED_STATE_ATTR, (host, port))
    # An announcement inherited from an ancestor is not ours. The explicit one
    # above wins, and consuming the inherited value here means no later reader
    # in this process can reach for it.
    _take_reload_announcement()


def announce_to_reload_child(host: str, port: int) -> None:
    """Announce the bound address to the ``--reload`` child we are about to spawn.

    The one path an in-process announce cannot cover, and the only writer of
    :data:`SERVE_ANNOUNCE_ENV`. The value names THIS process, so the announce is
    honoured only by a process this one spawned (see
    :func:`_take_reload_announcement`).
    """
    os.environ[SERVE_ANNOUNCE_ENV] = f"{os.getpid()} {host} {port}"


def _spawner_pid() -> int | None:
    """The pid of the process that spawned this one, or ``None``.

    ``multiprocessing``'s own bookkeeping, deliberately: uvicorn spawns the
    ``--reload`` child through ``multiprocessing.get_context("spawn")``
    (``uvicorn._subprocess.get_subprocess``), so "the process that spawned me"
    and "the process that announced my address" are the same process there.
    Imported lazily — only a process that FOUND an announcement pays for it,
    which is never the ordinary daemon.
    """
    try:
        import multiprocessing

        parent = multiprocessing.parent_process()
    except Exception:  # noqa: BLE001 — an unavailable channel is "not announced"
        return None
    return parent.pid if parent is not None else None


def _take_reload_announcement() -> tuple[str, int] | None:
    """Read-and-CLEAR the reload announcement, when it is addressed to us.

    Cleared unconditionally, and that is the point: the ``--reload`` child is
    the only process the announcement was addressed to, so once it has read the
    value nothing IT spawns later can find it and re-publish its parent's
    address.

    A garbled value, or one naming somebody other than our own spawner, is
    consumed and ignored — never guessed at.
    """
    raw = os.environ.pop(SERVE_ANNOUNCE_ENV, "")
    if not raw:
        return None
    try:
        announcer_raw, host, port_raw = raw.split(" ", 2)
        announcer, port = int(announcer_raw), int(port_raw)
    except ValueError:
        return None
    if announcer != _spawner_pid():
        return None
    return host, port


def advertised_address(app: Any = None) -> tuple[str, int] | None:
    """The address THIS process was told it is serving on, or ``None``.

    Two channels, tried in this order:

    1. the app object this process is serving — ``serve_command`` without
       ``--reload`` holds the app it hands uvicorn, and ``app.state`` is shared
       with the lifespan;
    2. the ``--reload`` child's environment, read-and-cleared, honoured only
       when the announcing pid is our own spawner. A hit there also raises
       :data:`RELOAD_CHILD_STATE_ATTR` on the app, because that is the one
       moment the channel is knowable (:func:`is_reload_child`).

    ``None`` when nothing announced one, and it is load-bearing rather than a
    ``("", 0)`` placeholder: a record exists so another process can DIAL a
    daemon, so a boot that cannot name a truthful address publishes no record at
    all (see the lifespan in ``server/app.py``). A record naming the wrong
    address is the class of error this module exists to remove — a reader would
    dial it, reach a different daemon, and have no way to tell.
    """
    if app is not None:
        announced = getattr(app.state, ANNOUNCED_STATE_ATTR, None)
        if announced is not None:
            return announced
    taken = _take_reload_announcement()
    if taken is not None and app is not None:
        # Remember WHICH channel answered, because the value above is gone now.
        # Not a side effect for its own sake: `--reload` is the one boot that
        # must not run the build watch (:func:`is_reload_child`), and this is the
        # only point at which that boot is distinguishable.
        setattr(app.state, RELOAD_CHILD_STATE_ATTR, True)
    return taken


def is_reload_child(app: Any) -> bool:
    """Was this app booted by uvicorn's ``--reload`` supervisor's child?

    True for the child process ``lop serve --reload`` gets: ``serve_command``
    hands the address to it through :data:`SERVE_ANNOUNCE_ENV` rather than to an
    app object, and that is the same event.

    WHY THE LIFESPAN ASKS, and why the answer must be kept out of the build
    watch: under ``--reload`` the PORT belongs to the supervisor, not to us.
    uvicorn's reloader holds the listening socket and runs a child that serves
    through it, so a child that retired would remove its record, ask its own
    process to stop — and leave the parent alive, still accepting connections on
    that socket with nothing behind them. A reader then sees a daemon with no
    record and a ``/health`` that times out, which is a state the record cannot
    describe and no client can act on (QA round 1, Q3). A dev-mode supervisor is
    also not a production daemon: it has no successor to hand a socket to, and
    the operator is watching its console.
    """
    state: Any = getattr(app, "state", None)
    return bool(getattr(state, RELOAD_CHILD_STATE_ATTR, False))


@dataclass
class ServeRecord:
    """What one ``lop serve`` process publishes about itself.

    Keyed by pid, like the session records, because a process serving HTTP is
    one process: the pid is its uniqueness token and a ``kill -9`` leaves
    exactly one file for the next scan to reap. It is NOT a session record and
    never becomes one — a daemon outlives the sessions it hosts, holds no
    control socket anyone attaches to, and is found by a different question.

    Two readers matter and they want different fields. A UI that found the file
    wants ``host``/``port`` to dial and ``instance_id`` to confirm that the
    process answering is the one this record describes. An updater wants
    ``prefix``/``install_kind``/``version`` to know WHICH install is serving,
    which is not the same as the one on ``PATH`` — this host has run three at
    once.

    Every field is additive with respect to a reader built before it: unknown
    keys are dropped by :meth:`from_json` and absent ones that carry defaults
    fall back to them, so a record written by a newer binary is readable here
    (the contract documented at ``session/runtime/types.py``'s record section).
    """

    #: ``os.getpid()`` — the serving process, and the record's filename.
    pid: int
    #: The address the listener is bound to, as announced by ``serve_command``
    #: and reduced to a DIALABLE host (:func:`_dialable_host`) — see
    #: :func:`advertised_address`. A record is only published when one was
    #: announced.
    host: str
    #: The port ACTUALLY BOUND, which is not the ``--port`` argument when that
    #: was ``0``.
    port: int
    #: Minted at startup and returned by ``/health``. The reader's identity
    #: check: a 200 alone is not identification, because a dev server on
    #: another port answered one too.
    instance_id: str
    #: ``update.installed_build()``'s two halves: the distribution version and
    #: the git ref ``lop-update`` recorded, ``""`` for PyPI/pipx/editable
    #: installs. Both are needed because this host's common drift is a
    #: same-version rebuild.
    version: str
    source_ref: str
    #: ``sys.prefix`` — WHICH install this daemon runs. The value the update
    #: path needs and cannot derive from ``which local-operator``: the install
    #: that is SERVING is the install to update.
    prefix: str
    #: ``update.install_kind()``: ``uv-tool`` / ``pipx`` / ``pip`` / ``editable``
    #: / ``unknown``. Tells the updater both how to update and whether it may.
    install_kind: str
    #: Is the desktop plane governed RIGHT NOW, i.e. did the desktop app start
    #: this daemon (``LOCAL_OPERATOR_DESKTOP_TOKEN`` in its environment) or has
    #: an app claimed it since. A daemon started by the app is not the same
    #: animal as one a person started from a shell, and the difference is not
    #: otherwise visible from outside the process.
    #:
    #: Refreshed by the claim when it is accepted (``routes/desktop_claim.py``
    #: republishes through the record's own publisher), so it is not a
    #: boot-time snapshot: a reader that saw ``false`` and finds ``true`` here
    #: is looking at a daemon whose plane was claimed in between, which is
    #: also why its ``claim_key`` is spent. Regenerating the record to refresh
    #: it would RE-MINT that key and destroy the app's proof of ownership, so
    #: nothing may refresh this field by rebuilding the record.
    desktop: bool
    #: Reserved for the claim handshake: minted by :func:`build_record` at
    #: startup and published in this record, ``""`` when the desktop app itself
    #: started the daemon (an env token already governs the plane) or when the
    #: reader is looking at a record from a build that predates the handshake.
    #: Its ONLY lawful channel is this file: a reader that has the record can
    #: already attach to the user's sessions, so publishing it here hands that
    #: principal no new class of secret, while a page in a browser — which can
    #: read no files — can neither see it nor guess it.
    claim_key: str = ""
    #: When this record was first written, and when its owner last proved it
    #: was alive. ``heartbeat_at`` is stamped by every write (the shared
    #: ``publish``), and the difference between it and now is what makes a live
    #: pid's record ``wedged`` rather than ``live``.
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
    #: Set only while this daemon is LEAVING, to the build it loaded
    #: (``retiring_from``) and the build now on disk that it is making room for
    #: (``retiring_to``); both ``""`` for the whole of a normal life. See
    #: :mod:`local_operator.server.retire`.
    #:
    #: WHY THE DAEMON ANNOUNCES RATHER THAN JUST DISAPPEARING. A reader that
    #: finds no record cannot tell a retire from a crash, a ``kill -9``, or a
    #: machine that is coming back — so a daemon that vanished on an update
    #: would look broken exactly when it is working correctly. These make the
    #: handover legible: the record is still there, still heartbeating, and it
    #: says which build is coming. Nothing else about the record changes, so
    #: ``live`` stays the truthful classification until the clean exit removes
    #: the file (see the lifespan's ``finally``).
    #:
    #: Additive, like every field here: a reader built before them drops the
    #: keys and sees the daemon it always saw, and this changes no protocol
    #: version (see ``session/runtime/types.py``'s record section — the daemon
    #: record is not the attach protocol).
    retiring_from: str = ""
    retiring_to: str = ""

    def to_json(self) -> dict[str, Any]:
        # ``asdict`` like the session record: one serialization spelling for
        # the whole record, so a field added here is published without a
        # second place to remember.
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ServeRecord":
        # The session record's contract, applied to this type (see
        # ``SessionRecord.from_json``): drop unknown keys, so a record written
        # by a NEWER binary is readable by this one mid-upgrade. Absent keys
        # take the dataclass defaults, so a field ADDED later is readable by
        # the binary that predates it. A payload missing a field with no
        # default is not a record of this shape at all and raises, which
        # ``scan`` treats as a torn file and reaps.
        known = {f for f in ServeRecord.__dataclass_fields__}
        return ServeRecord(**{k: v for k, v in data.items() if k in known})


def build_record(
    *, instance_id: str, announced: tuple[str, int], desktop_governed: bool | None = None
) -> ServeRecord:
    """Assemble the record for THIS process, reading identity fresh.

    ``announced`` is required rather than read from the environment here: the
    caller resolves it once (:func:`advertised_address`) and uses the same
    answer to decide whether to publish at all, so the reload channel's single
    read-and-clear cannot be spent twice, and this function stays a pure
    assembler.

    ``local_operator.update`` is imported here rather than at module scope: the
    install identity is read exactly once per process, at startup, and every
    other importer of this module (a reader, a test of the record shape) should
    not pay for ``importlib.metadata`` and ``urllib``.

    ``desktop_governed`` is a test seam for the record's ``desktop`` field. The
    desktop plane's own predicate is
    :func:`~local_operator.server.desktop.desktop_posture`, which a test cannot
    pin without mutating the process's environment for every other test in the
    worker. The seam is named for the POSTURE rather than for the variable that
    usually produces it, because the field it pins is true of two different
    daemons: one the app started, and one an app claimed. Passing ``False`` for
    a governed plane therefore publishes a record whose ``desktop`` and
    ``claim_key`` contradict each other — a state a test may construct, never
    one production can be in.

    **The claim key is minted HERE, and only when the plane is nobody else's**
    (``desktop_posture().enabled`` false). Two reasons for the placement: the
    key is the daemon's, not the HTTP app's — a record is written by the serve
    process and read by a stranger, and the key must exist from the instant the
    record is published or a UI that discovers the daemon in the same
    millisecond could find a record with nothing to claim; and the condition is
    the desktop plane's own, asked of the desktop module rather than restated,
    so a daemon the desktop app started (env capability present) publishes
    ``""`` and can never be claimed out from under it.

    ``secrets.token_urlsafe(32)`` is 256 bits from the OS CSPRNG, the same
    primitive ``session/runtime/types.py`` mints ``control_key`` with. It is
    published ONLY through the record — never logged, never returned by a
    route, never written to a second file — because the record's
    ``0600``-under-``0700`` permissions ARE the authorization story (see
    ``server/desktop.py``'s module docstring).
    """
    from local_operator.server.desktop import desktop_posture
    from local_operator.update import install_kind, installed_build

    host, port = announced
    build = installed_build()
    governed = desktop_posture().enabled if desktop_governed is None else desktop_governed
    return ServeRecord(
        pid=os.getpid(),
        host=_dialable_host(host),
        port=port,
        instance_id=instance_id,
        version=build.version,
        source_ref=build.source_ref,
        prefix=sys.prefix,
        install_kind=install_kind().value,
        desktop=governed,
        # ``""``, never a regenerated key: an env-governed daemon has no claim
        # to publish, and a reader must be able to tell that from a key.
        claim_key="" if governed else secrets.token_urlsafe(32),
    )


async def heartbeat_loop(publisher: session_registry.RecordPublisher) -> None:
    """Rewrite the record every ``HEARTBEAT_INTERVAL_S`` until cancelled.

    The same discipline the session runtime uses (``RuntimeServer``'s
    ``_heartbeat_loop``), and for the same reason: the two checks a reader
    makes are independent, and a live pid with a stale heartbeat is a daemon
    that is stuck — which the reader must be able to SEE as ``wedged`` rather
    than discover by timing out on a request. Rewriting the whole record is the
    shared ``publish``'s job, including its atomicity and permissions.

    Never raises: a failed heartbeat leaves the previous record in place, which
    is exactly the degraded state a reader should observe. Dying here would
    stop the heartbeat AND leave the record looking live until it aged out.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        try:
            publisher.heartbeat()
        except Exception:  # noqa: BLE001 — a missed beat is self-healing
            continue


def record_path(pid: int, root: Path | None = None) -> Path:
    """This namespace's ``<pid>.json`` under ``root`` (default: config root)."""
    return session_registry.record_path(pid, root, SERVE_RUN_DIRNAME)


def publisher(record: ServeRecord, root: Path | None = None) -> session_registry.RecordPublisher:
    """A publisher for THIS namespace, which publishes ``record`` on the spot.

    The shared class does the work; binding the dirname here is what keeps the
    namespace out of every caller (and out of the daemon's lifecycle code,
    which should not have to know that records live in directories at all).
    """
    return session_registry.RecordPublisher(record, root, SERVE_RUN_DIRNAME)


def publish(record: ServeRecord, root: Path | None = None) -> Path:
    """Publish (or refresh) ``record`` in the serve namespace."""
    return session_registry.publish(record, root, SERVE_RUN_DIRNAME)


def unpublish(pid: int, root: Path | None = None) -> None:
    """Remove ``pid``'s record. Best-effort, like the shared one."""
    session_registry.unpublish(pid, root, SERVE_RUN_DIRNAME)


def scan(root: Path | None = None) -> list[tuple[ServeRecord, str]]:
    """Every serve record, classified ``live`` / ``wedged`` / ``stale``.

    The classification is the shared one, asked for this namespace and this
    record type — ``stale`` means the pid is gone and the file has just been
    reaped, ``wedged`` means the pid lives but its heartbeat stopped, and only
    ``live`` is a daemon to talk to.
    """
    return session_registry.scan(root, SERVE_RUN_DIRNAME, ServeRecord.from_json)
