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
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from local_operator.session.runtime import registry as session_registry
from local_operator.session.runtime.types import HEARTBEAT_INTERVAL_S, SERVE_RUN_DIRNAME

#: How ``serve_command`` tells the app which address it actually bound.
#:
#: An environment variable because it is the ONLY channel that reaches both
#: hosts the app runs under: the in-process ASGI object (where the CLI already
#: has a reference, but the app does not have one back) and, under
#: ``--reload``, the supervisor's child, which re-imports ``server.app`` from
#: an import string and shares nothing but the inherited environment.
#:
#: Read once, at startup, by :func:`build_record`. Nothing else reads it, so an
#: agent's shell tool inheriting it is inert — and a nested ``lop serve``
#: overwrites it before starting its own uvicorn, so it cannot inherit a stale
#: address by accident.
SERVE_HOST_ENV = "LOCAL_OPERATOR_SERVE_HOST"
SERVE_PORT_ENV = "LOCAL_OPERATOR_SERVE_PORT"

#: Port ``0`` in a record means "not announced" — see :func:`advertised_address`.
ANNOUNCED_PORT_UNKNOWN = 0


def announce_address(host: str, port: int) -> None:
    """Announce the bound address to the app in this process (and its children).

    Called by ``cli.serve_command`` after it has bound the listener, so the
    record and ``/health`` name the port the kernel actually gave us rather
    than the ``--port`` argument. That distinction is the whole point of
    ``--port 0``: the UI's own child asks for an ephemeral port, and a record
    that named ``0`` would be unusable exactly where it is needed most.
    """
    os.environ[SERVE_HOST_ENV] = host
    os.environ[SERVE_PORT_ENV] = str(port)


def advertised_address() -> tuple[str, int]:
    """The address this process was told it is serving on.

    ``("", 0)`` when nothing announced one, which is what a process started
    through some other ASGI runner gets — and the record says so rather than
    guessing ``127.0.0.1:1111``, because a record that names the WRONG address
    is the class of error this module exists to remove: a reader would dial it,
    reach a different daemon, and have no way to tell. An announced-but-garbled
    port is treated the same way, for the same reason.
    """
    port_raw = os.environ.get(SERVE_PORT_ENV, "")
    try:
        port = int(port_raw)
    except ValueError:
        port = ANNOUNCED_PORT_UNKNOWN
    return os.environ.get(SERVE_HOST_ENV, ""), port


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
    #: (``""``/``0`` when nothing announced one; see
    #: :func:`advertised_address`).
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
    #: Is the desktop plane live right now, i.e. was
    #: ``LOCAL_OPERATOR_DESKTOP_TOKEN`` set for this process. A daemon started
    #: by the app is not the same animal as one a person started from a shell,
    #: and the difference is not otherwise visible from outside the process.
    desktop: bool
    #: Reserved for the claim handshake (a later PR mints it; ``""`` means no
    #: claim governs this daemon). Present now, before anything writes it, so
    #: that PR does not have to change this schema and every reader written
    #: against this one already drops it correctly.
    claim_key: str = ""
    #: When this record was first written, and when its owner last proved it
    #: was alive. ``heartbeat_at`` is stamped by every write (the shared
    #: ``publish``), and the difference between it and now is what makes a live
    #: pid's record ``wedged`` rather than ``live``.
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)

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


def build_record(*, instance_id: str, desktop_token_set: bool | None = None) -> ServeRecord:
    """Assemble the record for THIS process, reading identity fresh.

    ``local_operator.update`` is imported here rather than at module scope: the
    install identity is read exactly once per process, at startup, and every
    other importer of this module (a reader, a test of the record shape) should
    not pay for ``importlib.metadata`` and ``urllib``.

    ``desktop_token_set`` is a test seam — the desktop plane's own predicate is
    the environment, and a test must be able to pin the answer without
    mutating the process's environment for every other test in the worker.
    """
    from local_operator.update import install_kind, installed_build

    host, port = advertised_address()
    build = installed_build()
    desktop = (
        bool(os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN"))
        if desktop_token_set is None
        else desktop_token_set
    )
    return ServeRecord(
        pid=os.getpid(),
        host=host,
        port=port,
        instance_id=instance_id,
        version=build.version,
        source_ref=build.source_ref,
        prefix=sys.prefix,
        install_kind=install_kind().value,
        desktop=desktop,
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
