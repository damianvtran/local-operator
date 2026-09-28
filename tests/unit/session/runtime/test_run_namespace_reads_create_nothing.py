"""The ``run/`` namespaces: a READ resolves a path and creates nothing.

THE OTHER HALF OF THE CLASS, and it was still live after the network plane was
fixed — measured on a fresh isolated root against the desktop plane (review round
1, R1-1; QA Q-1): five GET routes created ``run/mobile``, ``run/peers`` and
``run/viewers`` on a machine that had never run a session or a relay.

* ``registry.run_dir`` mkdir'd, and ``registry.scan`` opened with it, so EVERY
  scanner did: ``/v1/desktop/info`` and ``/sessions`` via ``registry.scan``,
  ``/networks`` via ``network.store.find_own_relay`` → ``scan_peer_records``,
  ``/runtimes`` twice over (``reclaim.read_fleet`` and ``roster.build_roster``);
* ``viewers.viewer_run_dir`` mkdir'd under ``scan_viewers``, which
  ``reclaim.read_fleet`` reaches — ``/v1/desktop/runtimes`` again;
* ``presence.desktop_run_dir``/``delivery_dir`` carried the same written
  justification ("a reader reaches it too ... so this stays mkdir-on-read"), which
  is true about the ANSWER and wrong about the WRITE.

The three modules now follow the split this PR established: the plain resolver
returns a PATH, the ``ensure_*`` twin creates 0700, and the scanners additionally
short-circuit on an absent directory (the rule ``server/utils/desktop_feed.py``
had already applied by hand, review round 1 MAJOR 1 there).
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.session.runtime import presence, registry, viewers

RUN_NAMESPACES = ("run", "run/mobile", "run/peers", "run/serve", "run/viewers", "run/desktop")


def _record(pid: int, cwd: str = "/tmp") -> registry.SessionRecord:
    """A minimal valid session record, spelled as ``test_registry_namespaces`` does.

    Constructed here rather than imported from that module: a test file that
    imports another test file's helper makes one file's edits break the other's
    collection, and this is six fields.
    """
    from local_operator.session.runtime.types import SessionRecord

    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id="a" * 12,
        conversation_name="demo",
        cwd=cwd,
        model_label="anthropic/claude-opus-5",
        control_port=12345,
        control_key="k" * 64,
    )


def _created(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


# ---------------------------------------------------------------------------
# The read half
# ---------------------------------------------------------------------------

RESOLVERS: list[tuple[str, Callable[[Path], Any]]] = [
    ("registry.run_dir", lambda root: registry.run_dir(root)),
    ("registry.run_dir (nested)", lambda root: registry.run_dir(root, "run/peers")),
    ("registry.record_path", lambda root: registry.record_path(4242, root)),
    ("registry.record_path (nested)", lambda root: registry.record_path(4242, root, "run/serve")),
    ("viewers.viewer_run_dir", lambda root: viewers.viewer_run_dir(root)),
    ("viewers.viewer_record_path", lambda root: viewers.viewer_record_path(4242, root)),
    ("presence.desktop_run_dir", lambda root: presence.desktop_run_dir(root)),
    ("presence.delivery_path", lambda root: presence.delivery_path(root)),
    ("presence.delivery_dir", lambda root: presence.delivery_dir(root)),
    ("presence.delivery_record_path", lambda root: presence.delivery_record_path("inst-1", root)),
]


@pytest.mark.parametrize(("name", "resolve"), RESOLVERS, ids=[name for name, _ in RESOLVERS])
def test_resolving_a_run_path_creates_nothing(
    tmp_path: Path, name: str, resolve: Callable[[Path], Any]
) -> None:
    resolve(tmp_path)
    assert _created(tmp_path) == [], f"{name} created something under the config root"


def test_scanning_an_absent_namespace_answers_empty_and_creates_nothing(tmp_path: Path) -> None:
    """The call every desktop GET made, at the source: a scan of a namespace that
    was never created answers ``[]`` — the same thing an empty one answers — and
    leaves the tree exactly as it found it."""
    assert registry.scan(tmp_path) == []
    assert registry.scan(tmp_path, "run/peers") == []
    assert viewers.scan_viewers(tmp_path) == []
    assert _created(tmp_path) == []


def test_the_peer_record_reads_create_nothing(tmp_path: Path) -> None:
    """``network.store``'s ``run/peers`` readers, which is the path
    ``GET /v1/desktop/networks`` takes (``desktop_mesh._relay_record`` →
    ``find_own_relay`` → ``peer_records`` → ``scan_peer_records``)."""
    from local_operator.network import store

    assert store.scan_peer_records(tmp_path) == []
    assert store.peer_records(tmp_path) == []
    assert store.find_own_relay(tmp_path) is None
    # ``scan_own_relay`` answers ``(record, verdict)`` and ``(None, "")`` when this
    # install has no relay process at all — the diagnostic's own vocabulary, and
    # the point here is that asking costs no mkdir.
    assert store.scan_own_relay(tmp_path) == (None, "")
    store.run_record_path(4242, tmp_path)  # the resolver the relay writes through
    assert _created(tmp_path) == []


def test_a_wedged_record_is_still_found_once_the_directory_exists(tmp_path: Path) -> None:
    """The fix must not turn "wedged" into "absent".

    A record whose heartbeat stopped is the state the stop ladder acts on, so a
    reader that answered "no runtime" merely because it declined to CREATE the
    namespace would be a worse bug than the one being fixed. Written directly (a
    ``publish`` would stamp a fresh heartbeat), on a directory the WRITER's
    spelling created.
    """
    import json
    import os
    import time

    record = _record(os.getpid(), cwd=str(tmp_path))
    record.heartbeat_at = time.time() - (registry.HEARTBEAT_TIMEOUT_S + 5.0)
    registry.ensure_run_dir(tmp_path)
    registry.record_path(record.pid, tmp_path).write_text(json.dumps(record.to_json()))

    assert [(found.pid, state) for found, state in registry.scan(tmp_path)] == [
        (record.pid, "wedged")
    ]


# ---------------------------------------------------------------------------
# The write half — the opposite failure this split could have introduced
# ---------------------------------------------------------------------------

ENSURE_TWINS: list[tuple[str, Callable[[Path], Path], str]] = [
    ("registry.ensure_run_dir", lambda root: registry.ensure_run_dir(root), "run/mobile"),
    (
        "registry.ensure_run_dir (serve)",
        lambda root: registry.ensure_run_dir(root, "run/serve"),
        "run/serve",
    ),
    (
        "viewers.ensure_viewer_run_dir",
        lambda root: viewers.ensure_viewer_run_dir(root),
        "run/viewers",
    ),
    (
        "presence.ensure_desktop_run_dir",
        lambda root: presence.ensure_desktop_run_dir(root),
        "run/desktop",
    ),
    (
        "presence.ensure_delivery_dir",
        lambda root: presence.ensure_delivery_dir(root),
        "run/desktop/delivery",
    ),
]


@pytest.mark.parametrize(
    ("ensure", "relative"),
    [(ensure, relative) for _name, ensure, relative in ENSURE_TWINS],
    ids=[name for name, _ensure, _relative in ENSURE_TWINS],
)
def test_the_ensure_twins_still_create_at_0700(
    tmp_path: Path, ensure: Callable[[Path], Path], relative: str
) -> None:
    path = ensure(tmp_path)
    assert path == tmp_path / relative
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_a_published_record_is_scannable_and_the_directory_is_private(tmp_path: Path) -> None:
    """The writer path end to end: ``publish`` creates what it needs, and the record
    it writes is 0600 inside a 0700 directory — the pair the run plane has always
    had, preserved through the split.
    """
    import os

    record = _record(4242, cwd=str(tmp_path))
    path = registry.publish(record, root=tmp_path)

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(registry.run_dir(tmp_path).stat().st_mode) == 0o700
    assert [found.pid for found, _state in registry.scan(tmp_path)] == [record.pid]

    # ``scan_viewers`` filters on ``pid_alive``, so the fixture uses THIS process's
    # pid: a record naming a dead pid is correctly not offered.
    viewers.publish_viewer(
        viewers.ViewerRecord(pid=os.getpid(), surface="tui", control_port=1, control_key="k" * 64),
        root=tmp_path,
    )
    assert [row.pid for row in viewers.scan_viewers(tmp_path)] == [os.getpid()]
