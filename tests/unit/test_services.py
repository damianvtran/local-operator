"""Bringing the non-runtime fleet onto the current build.

The assertions here are about the two ways this command can do HARM, which is why
they are so specific:

* **Never signal a daemon that cannot take it.** The request is ``SIGUSR1``, whose
  default disposition is to terminate, so a daemon built before the capability
  existed would be KILLED by a well-meaning update. The record's ``reloadable``
  field is the guard, and one of these tests exists to keep it a guard.
* **Never claim a move that did not happen.** ``instance_id`` is minted once per
  process, so a changed one PROVES the daemon was replaced; the version does not
  (this host's ordinary update is a same-version rebuild). Reading the version
  here would report success for a daemon that never moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import services
from local_operator.server import registry as serve_registry
from local_operator.server.reload import RELOAD_SIGNAL
from local_operator.update import BuildStamp

NEW = BuildStamp(version="0.59.0", source_ref="")
OLD = BuildStamp(version="0.56.14", source_ref="")


def _record(**over: Any) -> serve_registry.ServeRecord:
    fields: dict[str, Any] = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 1111,
        "instance_id": "instance-one",
        "version": OLD.version,
        "source_ref": OLD.source_ref,
        "prefix": "/opt/gen/old/tools/local-operator",
        "install_kind": "uv-tool",
        "desktop": True,
        "reloadable": True,
    }
    fields.update(over)
    return serve_registry.ServeRecord(**fields)


@pytest.fixture
def pointer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What the install pointer names. The SUBJECT of every comparison below."""
    state: dict[str, Any] = {"stamp": NEW}
    monkeypatch.setattr("local_operator.update.disk_build", lambda *_a, **_k: state["stamp"])
    return state


def _run(
    records: list[Any],
    *,
    kills: list[tuple[int, int]],
    after: dict[int, Any] | None = None,
    wait_s: float = 1.0,
    clock: list[float] | None = None,
    probe: Any = None,
) -> list[services.ServiceRefresh]:
    """Drive one reload pass with every seam injected.

    ``scan`` answers differently AFTER the signal, which is how a real
    replacement announces itself: the same pid, a new ``instance_id``.

    ``probe`` defaults to ASSENT, because the identity check is its own test and
    every other test here is about something else: without that default each of
    them would open a real loopback connection to the port in its own fixture and
    fail for a reason that has nothing to do with what it asserts.
    """
    after = after or {record.pid: record for record in records}
    ticks = clock if clock is not None else [0.0]
    answers = probe if probe is not None else (lambda record: None)

    def scan() -> list[Any]:
        return records if not kills else list(after.values())

    def kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    def sleep(_seconds: float) -> None:
        ticks[0] += 0.05

    return services.reload_serve_daemons(
        wait_s=wait_s,
        sleep=sleep,
        scan=scan,
        kill=kill,
        probe=answers,
        monotonic=lambda: ticks[0],
    )


def test_an_unproven_process_is_never_signalled(pointer: dict[str, Any]) -> None:
    """THE IDENTITY GUARD (review round 1, R1-4).

    A record is a file whose name is a pid, and a pid is recycled: review
    constructed a live `sleep` named by a hand-written live, reloadable record and
    this command killed it (rc -30). Nothing is signalled unless the address
    answers AS the instance the record names — and the refusal is a warning that
    says nothing was sent, not a silence.
    """
    record = _record()
    kills: list[tuple[int, int]] = []
    out = _run(
        [record],
        kills=kills,
        wait_s=0.0,
        probe=lambda _record: "127.0.0.1:1111 is answering as nobody, not the instance one…",
    )
    assert kills == []
    warnings = [warning for refresh in out for warning in refresh.warnings]
    assert any("was not asked to reload" in warning for warning in warnings)
    assert any("Nothing was signalled" in warning for warning in warnings)


def test_the_identity_check_is_a_real_health_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default probe reads /health and compares `instance_id`, not the port."""
    record = _record(port=61111, host="127.0.0.1", instance_id="expected-instance")

    class _Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"status":200,"result":{"instance_id":"a-different-process"}}'

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: _Response())
    mismatch = services._answers_as_record(record)
    assert mismatch is not None and "a-different-process" in mismatch

    class _Matching(_Response):
        def read(self) -> bytes:
            return b'{"status":200,"result":{"instance_id":"expected-instance"}}'

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: _Matching())
    assert services._answers_as_record(record) is None


def test_a_missing_stamp_moves_nothing_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2-1's fence: a comparison against an absent right-hand side is not a verdict.

    With no readable stamp, `_serves_current_build` answers False for EVERY
    daemon — so without this guard the whole fleet looks stale to a caller that
    has no build to move anything onto. Review demonstrated it end to end with a
    fabricated-root daemon, and the reachable caller is a source checkout, whose
    `disk_build()` is None.
    """
    monkeypatch.setattr("local_operator.update.disk_build", lambda *_a, **_k: None)
    kills: list[tuple[int, int]] = []
    records = [_record(pid=1), _record(pid=2)]
    out = services.reload_serve_daemons(
        scan=lambda: records, kill=lambda pid, sig: kills.append((pid, sig))
    )
    assert kills == []
    assert len(out) == 1
    assert "NOTHING was signalled" in out[0].warnings[0]


def test_the_probe_url_brackets_an_ipv6_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bracket join is load-bearing and was untested (review round 2, NIT-2).

    `http://::1:1111/health` is not a URL: an IPv6 literal must be bracketed, and
    a probe that cannot parse its own URL is a probe that answers "did not
    identify itself" for a daemon that is answering perfectly.
    """
    import json as json_mod

    seen: list[str] = []

    class _Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def read(self) -> bytes:
            return json_mod.dumps({"result": {"instance_id": "v6"}}).encode()

    def _open(url: str, timeout: float = 0) -> Any:
        seen.append(url)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _open)
    assert services._answers_as_record(_record(host="::1", port=1111, instance_id="v6")) is None
    assert seen == ["http://[::1]:1111/health"]


def test_a_daemon_already_on_the_current_build_is_left_alone(pointer: dict[str, Any]) -> None:
    """Nothing to move, so nothing is interrupted — no signal, no stream cut."""
    record = _record(version=NEW.version, source_ref=NEW.source_ref)
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills)
    assert kills == []
    assert out[0].lines == (f"serve pid 4242 is already on {NEW.label()}",)


def test_a_daemon_that_cannot_reload_is_reported_and_never_signalled(
    pointer: dict[str, Any],
) -> None:
    """THE guard. ``SIGUSR1`` kills a process that never installed a handler."""
    record = _record(reloadable=False)
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, wait_s=0.0)
    assert kills == []
    assert len(out) == 1
    assert out[0].lines == ()
    assert "predates in-place reload" in out[0].warnings[0]
    assert "restart that server by hand" in out[0].warnings[0]


def test_a_stale_daemon_is_asked_and_reported_once_it_republishes(pointer: dict[str, Any]) -> None:
    """The happy path, and the reason ``instance_id`` is the proof."""
    record = _record()
    replacement = _record(
        version=NEW.version, source_ref=NEW.source_ref, instance_id="instance-two"
    )
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, after={record.pid: replacement})
    assert kills == [(record.pid, RELOAD_SIGNAL)]
    assert any("is now serving 0.59.0" in line for refresh in out for line in refresh.lines)


def test_a_same_version_rebuild_is_not_read_as_a_no_op(pointer: dict[str, Any]) -> None:
    """This host's ORDINARY update: same version, new ``source_ref``.

    A version-only comparison would call the daemon current and never move it,
    which is the whole failure this feature exists to fix.
    """
    pointer["stamp"] = BuildStamp(version="0.59.0", source_ref="d808fc67")
    record = _record(version="0.59.0", source_ref="")
    kills: list[tuple[int, int]] = []
    _run([record], kills=kills, wait_s=0.0)
    assert kills == [(record.pid, RELOAD_SIGNAL)]


def test_a_daemon_that_never_comes_back_is_a_warning_not_a_failure(pointer: dict[str, Any]) -> None:
    """The install already succeeded; a nudge that did not land is a warning."""
    record = _record()
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, after={record.pid: record}, wait_s=0.25)
    warnings = [warning for refresh in out for warning in refresh.warnings]
    assert kills == [(record.pid, RELOAD_SIGNAL)]
    assert any("did not come back on the current build" in warning for warning in warnings)
    assert any("lop services restart" in warning for warning in warnings)


def test_a_daemon_that_vanishes_while_waiting_is_not_reported_as_moved(
    pointer: dict[str, Any],
) -> None:
    """A pid that is gone may have exited for its own reasons; that is not a move."""
    record = _record()
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, after={}, wait_s=0.25)
    assert all("is now serving" not in line for refresh in out for line in refresh.lines)


def test_one_deadline_covers_the_whole_fleet(pointer: dict[str, Any]) -> None:
    """Three daemons must not cost three budgets: the wait is shared.

    Serialising the waits would add their drains together and report the last
    daemon's answer as if it were the fleet's.
    """
    records = [_record(pid=pid) for pid in (11, 22, 33)]
    kills: list[tuple[int, int]] = []
    ticks = [0.0]
    out = _run(
        records,
        kills=kills,
        after={record.pid: record for record in records},
        wait_s=0.2,
        clock=ticks,
    )
    assert len(kills) == 3
    assert ticks[0] <= 0.2 + 0.05
    assert sum(len(refresh.warnings) for refresh in out) == 3


def test_live_serve_daemons_drops_everything_that_is_not_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wedged`` is a stuck process, and a stuck process is not signalled."""
    live, wedged, stale = _record(pid=1), _record(pid=2), _record(pid=3)
    monkeypatch.setattr(
        serve_registry,
        "scan",
        lambda root=None: [(wedged, "wedged"), (stale, "stale"), (live, "live")],
    )
    assert [record.pid for record in services.live_serve_daemons()] == [1]


def test_status_lines_name_the_drift_and_the_capability(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader is an operator who was just told "older build than the install"."""
    monkeypatch.setattr(
        services,
        "live_serve_daemons",
        lambda: [_record(), _record(pid=9, version=NEW.version, reloadable=False)],
    )
    monkeypatch.setattr(
        services,
        "_supervised_daemon_plists",
        lambda: [Path("/tmp/com.local-operator.mobile.plist")],
    )
    lines = services.status_lines()
    assert "serve daemon pid 4242 on 127.0.0.1:1111 — STALE (0.56.14, reloadable)" in lines
    assert "serve daemon pid 9 on 127.0.0.1:1111 — current (0.59.0, not reloadable)" in lines
    assert "supervised daemon: com.local-operator.mobile" in lines


def test_status_lines_says_so_when_there_are_no_daemons(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    assert services.status_lines() == ["serve daemons: none running"]
