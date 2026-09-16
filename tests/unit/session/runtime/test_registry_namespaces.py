"""The shared registry, parameterised by namespace.

One staged write and one liveness rule now serve two record kinds — the
session records at ``run/mobile`` and the ``serve`` daemon's at ``run/serve``
— so these tests pin the property that made that possible without a second
implementation: the parameter is a DIRECTORY and nothing else, and every
default still means exactly what it meant before it existed. The session
registry's own tests (``test_registry.py``) are deliberately untouched by this
change; if they ever need editing for a namespace parameter, the parameter
changed behaviour it promised not to.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

from local_operator.session.runtime import registry
from local_operator.session.runtime.types import (
    HEARTBEAT_TIMEOUT_S,
    RUN_DIRNAME,
    SERVE_RUN_DIRNAME,
    SessionRecord,
)


def make_session_record(pid: int | None = None) -> SessionRecord:
    return SessionRecord(
        pid=pid or os.getpid(),
        kind="tui",
        session_id="s1",
        conversation_name="demo",
        cwd="/tmp",
        model_label="anthropic/claude-opus-5",
        control_port=12345,
        control_key="k" * 64,
    )


def test_the_defaults_are_still_the_session_namespace(tmp_path: Path) -> None:
    """Every parameter that gained a ``dirname`` defaults to ``run/mobile``.

    Asserted through the PUBLIC functions rather than by reading signatures:
    what existing call sites depend on is where the file lands, not what the
    parameter is called.
    """
    assert RUN_DIRNAME == "run/mobile"
    record = make_session_record()
    path = registry.publish(record, root=tmp_path)
    assert path == tmp_path / RUN_DIRNAME / f"{record.pid}.json"
    assert registry.run_dir(tmp_path) == tmp_path / RUN_DIRNAME
    assert registry.record_path(record.pid, tmp_path) == path
    assert [r.pid for r, _state in registry.scan(tmp_path)] == [record.pid]
    registry.unpublish(record.pid, root=tmp_path)
    assert not path.exists()


def test_a_non_default_dirname_is_an_isolated_namespace(tmp_path: Path) -> None:
    """The parameter threads all the way through, and the namespaces do not mix.

    This is the property the serve record rests on: a daemon record must never
    appear in a reader of sessions, because every reader of ``run/mobile``
    treats a file there as a session and ``kind`` is a ``Literal`` it does not
    validate — a daemon in that directory would be a phantom session row, with
    no error anywhere.

    The two records deliberately share a pid, as a daemon and a session in
    different boots can: the namespace, not the pid, is what keeps them apart,
    and the SAME pid must therefore resolve to two different files.
    """
    session = make_session_record()
    serve_like = make_session_record()
    session_path = registry.publish(session, root=tmp_path)
    serve_path = registry.publish(serve_like, root=tmp_path, dirname=SERVE_RUN_DIRNAME)

    assert session_path == tmp_path / RUN_DIRNAME / f"{session.pid}.json"
    assert serve_path == tmp_path / SERVE_RUN_DIRNAME / f"{serve_like.pid}.json"
    assert session_path != serve_path
    for path in (session_path, serve_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    assert [r.pid for r, _state in registry.scan(tmp_path)] == [session.pid]
    assert [r.pid for r, _state in registry.scan(tmp_path, SERVE_RUN_DIRNAME)] == [serve_like.pid]

    # ``unpublish`` is scoped the same way: one namespace's removal must not
    # reach into the other, where a same-pid record is a different process's.
    registry.unpublish(serve_like.pid, root=tmp_path, dirname=SERVE_RUN_DIRNAME)
    assert not serve_path.exists()
    assert session_path.exists()
    assert [r.pid for r, _state in registry.scan(tmp_path)] == [session.pid]
    assert registry.scan(tmp_path, SERVE_RUN_DIRNAME) == []

    registry.unpublish(session.pid, root=tmp_path)
    assert not session_path.exists()


def test_scan_parses_with_the_callers_deserializer(tmp_path: Path) -> None:
    """The classification is shared; the RECORD TYPE is the caller's.

    A second namespace needs a second record type, and handing the deserializer
    in is what lets ``scan`` stay one implementation of "is this alive" rather
    than becoming one per record kind.
    """
    seen: list[dict[str, object]] = []

    class Other:
        # A LIVE pid, so the shared rule has something to classify as live: it
        # is the pid liveness probe, not the record type, that decides.
        pid = os.getpid()
        heartbeat_at = time.time()

        def to_json(self) -> dict[str, object]:
            return {"pid": self.pid, "heartbeat_at": self.heartbeat_at}

        @staticmethod
        def from_json(data: dict[str, object]) -> "Other":
            seen.append(data)
            record = Other()
            record.pid = int(str(data["pid"]))
            record.heartbeat_at = float(str(data["heartbeat_at"]))
            return record

    registry.run_dir(tmp_path, "run/other")
    registry.publish(Other(), root=tmp_path, dirname="run/other")
    results = registry.scan(tmp_path, "run/other", Other.from_json)

    assert seen, "the caller's deserializer is what parsed the file"
    assert [(type(r).__name__, state) for r, state in results] == [("Other", "live")]


def test_a_wedged_record_is_classified_for_any_namespace(tmp_path: Path) -> None:
    """The heartbeat rule reads the record's fields, not its type.

    Written directly rather than through ``publish``, which stamps a fresh
    heartbeat by design — a wedged record is exactly one whose heartbeat
    stopped arriving.
    """
    record = make_session_record()
    directory = registry.run_dir(tmp_path, SERVE_RUN_DIRNAME)
    record.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 1
    (directory / f"{record.pid}.json").write_text(json.dumps(record.to_json()))

    assert [(r.pid, s) for r, s in registry.scan(tmp_path, SERVE_RUN_DIRNAME)] == [
        (record.pid, "wedged")
    ]
    # And the other namespace, which has no such record, is unaffected.
    assert registry.scan(tmp_path) == []
