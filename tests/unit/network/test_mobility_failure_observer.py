"""Focused observer contracts: no relays, runtime children, network, or recovery probe.

These cells protect diagnostics, not the historic flake. Independent QA owns the
real target node and its same-code observer-disabled control, once each.
"""

from __future__ import annotations

import ast
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.network.mobility_failure_observer import (
    MAX_BYTES,
    MAX_EVENTS,
    TARGET,
    MobilityFailureObserver,
    emit_before_cleanup,
    observe_failure,
)


def test_pass_is_quiet_and_preserves_return_value(capsys: pytest.CaptureFixture[str]) -> None:
    observer = MobilityFailureObserver()
    sentinel = object()

    @observe_failure
    def cell(*, mobility_failure_observer: MobilityFailureObserver) -> object:
        observer.record("worker.begin", ("root-1", "instance-1"))
        emit_before_cleanup(observer, None)
        return sentinel

    assert cell(mobility_failure_observer=observer) is sentinel
    assert capsys.readouterr() == ("", "")
    assert not observer._sealed


@pytest.mark.parametrize("broken", ["before_cleanup", "_frames", "_sink", "record"])
def test_observer_failure_preserves_original_before_teardown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], broken: str
) -> None:
    order: list[str] = []
    observer = MobilityFailureObserver(lambda text: order.append("emitted"))
    original = AssertionError("original assertion message")

    def unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("PRIVATE observer exception value")

    monkeypatch.setattr(observer, broken, unavailable)

    @observe_failure
    def cell(*, mobility_failure_observer: MobilityFailureObserver) -> None:
        try:
            observer._safe(observer.record, "worker.begin", ("root-1", "instance-1"))
            raise original
        finally:
            emit_before_cleanup(observer, sys.exc_info()[1])
            # This models the boundary before a stop/finalizer that might hang;
            # inspect evidence HERE, not after the synthetic teardown completed.
            captured = capsys.readouterr()
            assert order == ["emitted"] or "diagnostics unavailable" in captured.err
            assert "PRIVATE" not in captured.err
            order.append("teardown")

    with pytest.raises(AssertionError) as caught:
        cell(mobility_failure_observer=observer)
    assert caught.value is original
    assert type(caught.value) is AssertionError
    assert str(caught.value) == "original assertion message"
    assert "teardown" in order


def test_failure_seals_and_emits_once_before_unwind() -> None:
    order: list[str] = []
    observer = MobilityFailureObserver(lambda text: order.append("emitted"))
    original = AssertionError("unchanged")

    @observe_failure
    def cell(*, mobility_failure_observer: MobilityFailureObserver) -> None:
        try:
            raise original
        finally:
            emit_before_cleanup(observer, sys.exc_info()[1])
            order.append("stop")

    with pytest.raises(AssertionError) as caught:
        cell(mobility_failure_observer=observer)
    assert caught.value is original
    assert order == ["emitted", "stop"]
    observer.record("late", ("root-1", "none"))
    assert observer._events == []


def test_wrappers_keep_results_and_exception_identity_when_recording_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = MobilityFailureObserver()
    original = ValueError("PRIVATE product exception value")

    def product() -> None:
        raise original

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("observer")

    monkeypatch.setattr(observer, "record", broken)
    wrapped = observer.wrap(product, "cleanup", lambda: ("root-1", "none"), worker=True)
    with pytest.raises(ValueError) as caught:
        wrapped()
    assert caught.value is original
    assert observer._workers == {}
    sentinel = object()
    assert observer.wrap(lambda: sentinel, "staging", lambda: ("root-1", "none"))() is sentinel


def test_threadsafe_event_bound_and_explicit_truncation() -> None:
    emitted: list[str] = []
    observer = MobilityFailureObserver(emitted.append)
    def record_batch() -> None:
        for _ in range(96):
            observer.record("worker.begin", ("root-1", "instance-1"))

    threads = [threading.Thread(target=record_batch) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(observer._events) == MAX_EVENTS
    assert observer._dropped == 4 * 96 - MAX_EVENTS
    emit_before_cleanup(observer, AssertionError("PRIVATE"))
    document = json.loads(emitted[0])
    assert document["truncated"] is True
    assert document["dropped_events"] == 4 * 96 - MAX_EVENTS
    assert len(document["events"]) <= MAX_EVENTS
    assert len(emitted[0].encode()) <= MAX_BYTES
    assert "PRIVATE" not in emitted[0]


def test_byte_bound_drops_whole_records() -> None:
    emitted: list[str] = []
    observer = MobilityFailureObserver(emitted.append)
    for _ in range(MAX_EVENTS):
        observer.record("x" * 128, ("root-1", "instance-1"), "E" * 128)
    emit_before_cleanup(observer, AssertionError("not serialized"))
    assert len(emitted[0].encode()) <= MAX_BYTES
    document = json.loads(emitted[0])
    assert document["truncated"] is True
    assert document["dropped_events"] > 0
    assert len(document["events"]) + document["dropped_events"] == MAX_EVENTS


def test_privacy_aliases_and_selected_allowlisted_frames_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    observer = MobilityFailureObserver(emitted.append)
    context = observer._context(Path("/PRIVATE/root"), "PRIVATE-instance")
    observer.record("worker.begin", context)
    repo = Path(__file__).resolve().parents[3]
    allowed = SimpleNamespace(
        f_code=SimpleNamespace(
            co_filename=str(repo / "local_operator/network/mobility.py"), co_name="reconcile"
        ),
        f_lineno=123,
        f_back=None,
    )
    unrelated = SimpleNamespace(
        f_code=SimpleNamespace(co_filename="/PRIVATE/unrelated.py", co_name="PRIVATE"),
        f_lineno=1,
        f_back=allowed,
    )
    monkeypatch.setattr(sys, "_current_frames", lambda: {7: unrelated, 99: object()})
    observer._workers[7] = context
    emit_before_cleanup(observer, AssertionError("PRIVATE failure value"))
    assert "PRIVATE" not in emitted[0]
    document = json.loads(emitted[0])
    assert document["frames"] == [
        {
            "file": "local_operator/network/mobility.py",
            "function": "reconcile",
            "line": 123,
            "root": "root-1",
            "instance": "instance-1",
        }
    ]
    # A synthetic over-deep/cyclic stack exercises the finite traversal budget.
    allowed.f_back = allowed
    bounded = MobilityFailureObserver(emitted.append)
    bounded._workers[7] = context
    emit_before_cleanup(bounded, AssertionError())
    clipped = json.loads(emitted[-1])
    assert len(clipped["frames"]) <= MAX_EVENTS
    assert clipped["frames_truncated"] is True
    assert clipped["truncated"] is True


@pytest.mark.parametrize("phase", ["cleanup", "staging", "reconcile", "journal.write"])
def test_worker_and_phase_events_report_exception_type_not_value(phase: str) -> None:
    emitted: list[str] = []
    observer = MobilityFailureObserver(emitted.append)

    def fails() -> None:
        raise FileNotFoundError("PRIVATE payload /path/token")

    cleanup = observer.wrap(fails, phase, lambda: ("root-1", "instance-1"))
    worker = observer.wrap(cleanup, "worker", lambda: ("root-1", "instance-1"), worker=True)
    with pytest.raises(FileNotFoundError):
        worker()
    emit_before_cleanup(observer, AssertionError())
    assert "PRIVATE" not in emitted[0]
    events = json.loads(emitted[0])["events"]
    assert [event["event"] for event in events] == [
        "worker.begin",
        f"{phase}.begin",
        f"{phase}.exception",
        "worker.exception",
        "worker.exit",
    ]
    assert [event["type"] for event in events if event["type"]] == [
        "FileNotFoundError", "FileNotFoundError"
    ]


def test_install_wraps_registration_worker_phases_and_journal_without_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual installer with synthetic modules, not a recovery probe."""
    emitted: list[str] = []
    observer = MobilityFailureObserver(emitted.append)
    calls: list[str] = []
    sentinel = object()

    class Relay:
        root = Path("/PRIVATE/root")
        instance_id = "PRIVATE-instance"

        def register_ops(self, handler: object, **kwargs: object) -> object:
            calls.append("register")
            return sentinel

    def cleanup(root: Path) -> None:
        calls.append("cleanup")

    def staging(root: Path) -> None:
        calls.append("staging")

    def journal(path: Path, entries: object) -> None:
        calls.append("journal")

    placement = SimpleNamespace(_write_journal=journal)

    def reconcile(root: Path, *, server: Relay) -> None:
        calls.append("reconcile")
        placement._write_journal(root / "network/handoffs.json", {"PRIVATE": "payload"})

    mobility = SimpleNamespace(sweep_staging=staging, reconcile=reconcile)

    def worker(server: Relay) -> object:
        cleanup_module.sweep_handed_away(server.root)
        mobility.sweep_staging(server.root)
        mobility.reconcile(server.root, server=server)
        return sentinel

    mobility._start_recovery = worker
    cleanup_module = SimpleNamespace(sweep_handed_away=cleanup)
    monkeypatch.setitem(
        sys.modules,
        "local_operator.network",
        SimpleNamespace(mobility=mobility, relay=SimpleNamespace(RelayServer=Relay)),
    )
    monkeypatch.setitem(
        sys.modules,
        "local_operator.session",
        SimpleNamespace(cleanup=cleanup_module, placement=placement),
    )
    observer.install(monkeypatch)
    server = Relay()
    assert server.register_ops(object(), on_start={"mobility-recovery": worker}) is sentinel
    assert mobility._start_recovery(server) is sentinel
    assert calls == ["register", "cleanup", "staging", "reconcile", "journal"]
    assert observer._workers == {}
    emit_before_cleanup(observer, AssertionError("PRIVATE assertion"))
    assert "PRIVATE" not in emitted[0]
    events = json.loads(emitted[0])["events"]
    names = [event["event"] for event in events]
    for label in ("hook.registration", "worker", "cleanup", "staging", "reconcile", "journal.write"):
        assert label + ".begin" in names
        assert label + ".complete" in names
    assert names[-1] == "worker.exit"
    worker_events = events[names.index("worker.begin") :]
    assert all(event["root"] == "root-1" for event in worker_events)
    assert all(event["instance"] == "instance-1" for event in worker_events)


def test_exact_target_keeps_original_deadline_poll_and_assertions() -> None:
    source = Path(__file__).with_name("test_mobility_integrity.py").read_text()
    module = ast.parse(source)
    target = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == TARGET
    )
    text = ast.get_source_segment(source, target)
    assert text is not None
    for expression in (
        "deadline = time.monotonic() + 10",
        "while SESSION in read_handoff_journal(server_a.root) and time.monotonic() < deadline:",
        "time.sleep(0.05)",
        '), "a relay start left a stale handoff entry in place"',
        'assert "being handed to" not in refused, refused',
    ):
        assert expression in text
    assertions = [ast.unparse(node) for node in ast.walk(target) if isinstance(node, ast.Assert)]
    assert assertions == [
        "assert 'being handed to' not in refused, refused",
        "assert SESSION not in read_handoff_journal(server_a.root), "
        "'a relay start left a stale handoff entry in place'",
    ]
    assert text.index("emit_before_cleanup(") < text.index("fresh.stop()")
    fixture = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "mobility_failure_observer"
    )
    assert ast.unparse(fixture.decorator_list[0]) == "pytest.fixture(autouse=True)"
    assert "request.node.name != TARGET" in ast.get_source_segment(source, fixture)
