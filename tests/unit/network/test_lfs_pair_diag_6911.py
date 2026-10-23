"""Offhost-only real PAIR negatives. Located under tests/unit/network so normal
suite config/runtime/session isolation and network fixtures apply. Never ship
this file in the one-file PAIR PR. No product or threshold changes.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest
import tests.conftest as suite_policy
from tests.unit.network import conftest as network_policy
from tests.unit.network import test_relay_e2e as cell
from scripts.diagnostics.lfs_pair_6911 import controller


@pytest.fixture()
def verified_identity(isolate_environment, monkeypatch):
    root = Path(os.environ["LFS_PAIR_ROOT"]).resolve()
    repository = Path(__file__).resolve().parents[3]
    # Existing suite autouse isolation intentionally scrubs config vars. Re-bind
    # only synthetic children AFTER that policy, never bypass or replace it.
    for key, name in (("LOCAL_OPERATOR_CONFIG_DIR", "config"), ("TMPDIR", "tmp"),
                      ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                      ("XDG_DATA_HOME", "data"), ("XDG_RUNTIME_DIR", "runtime")):
        monkeypatch.setenv(key, str(root / name))
    modules = {"tests/unit/network/test_lfs_pair_diag_6911.py": __file__,
               "tests/unit/network/test_relay_e2e.py": cell.__file__,
               "tests/unit/network/conftest.py": network_policy.__file__,
               "tests/conftest.py": suite_policy.__file__,
               "local_operator/network/relay.py": cell.relay.__file__,
               "local_operator/network/audit.py": cell.audit_mod.__file__,
               "scripts/diagnostics/lfs_pair_6911/controller.py": controller.__file__}
    hashes = {}
    for label, actual in modules.items():
        path = Path(actual).resolve()
        assert path == repository / label, "loaded path mismatch"
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    isolated = True
    for key in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "TMPDIR", "XDG_CONFIG_HOME",
                "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "LOCAL_OPERATOR_RUNTIME_ROOT"):
        assert Path(os.environ[key]).resolve().is_relative_to(root), "isolation root mismatch"
    actual = dict(python=list(sys.version_info[:3]),
                  executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                  pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), sources=hashes, isolated=isolated)
    expected = json.loads(os.environ["LFS_PAIR_EXPECTED"])
    expected["pid"] = os.getpid()  # Parent independently compares against Popen PID.
    controller.validate_identity(actual, expected)
    return actual


@pytest.fixture()
def devices(verified_identity, root, monkeypatch):
    # Identity/isolation MUST qualify before any real relay is started. Reuse the
    # existing fixture generator including its real two-relay stop finally.
    fixture = cell.devices.__wrapped__(root, monkeypatch)
    value = next(fixture)
    try:
        yield value
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


@pytest.mark.parametrize("variant", controller.VARIANTS)
def test_pair_audit_boundary(devices, monkeypatch, verified_identity, variant):
    server_a = devices[0]
    entered, release, sampling, append_done, wrapper_exit = (threading.Event() for _ in range(5))
    original_record = cell.audit_mod.AuditLog.record
    original_events, original_await = cell._events, cell._await_event
    waiter, failure, wait_elapsed = None, None, None
    answered_before = set(cell._pair_answered)
    observations = []
    result = dict(variant=variant, observation_reached=False, target_pass=False,
                  failure=None, append_done=False, wrapper_exited=False,
                  release_thread_joined=False, wall_s=0, wait_s=None,
                  errors=[], identity=verified_identity)
    started = time.monotonic()

    def record(owner, event):
        if owner is not server_a.audit or event.event != "member_admitted":
            return original_record(owner, event)
        entered.set()
        try:
            if variant == "new-omission":
                return
            if not release.wait(20):
                return
            original_record(owner, event)
            append_done.set()  # Publication can precede this: readers MUST wait.
        finally:
            wrapper_exit.set()  # Wrapper-finally witness, NOT Thread.join.

    def sampled_events(server):
        events = original_events(server)
        if server is server_a:
            observations.append("member_admitted" in events)
            if not sampling.is_set():
                assert "member_admitted" not in events
                sampling.set()
        return events

    def observe(server, event, *, timeout=5.0):
        nonlocal waiter, wait_elapsed
        assert server is server_a and event == "member_admitted" and timeout == 5.0
        assert entered.wait(5), "producer did not reach post-success boundary"
        result["observation_reached"] = True
        assert "member_admitted" not in original_events(server)
        if variant == "old-held":
            return original_events(server)
        if variant == "new-release":
            def release_after_read():
                if sampling.wait(5):
                    release.set()
            waiter = threading.Thread(target=release_after_read, daemon=False)
            waiter.start()
        monkeypatch.setattr(cell, "_events", sampled_events)
        begin = time.monotonic()
        try:
            return original_await(server, event, timeout=timeout)
        finally:
            wait_elapsed = time.monotonic() - begin

    def join_waiter():
        if waiter is not None:
            waiter.join(5)
        return waiter is None or not waiter.is_alive()

    def producer_exit():
        # No one-shot is_set after seeing publication: the producer may be between
        # publication and its finally. The existing five-second bound covers it.
        return entered.is_set() and wrapper_exit.wait(5)

    def restore_keys():
        for key in set(cell._pair_answered) - answered_before:
            del cell._pair_answered[key]

    def restore_wrappers():
        cell.audit_mod.AuditLog.record = original_record
        cell._events, cell._await_event = original_events, original_await

    def emit():
        result.update(failure=failure, append_done=append_done.is_set(),
                      wrapper_exited=wrapper_exit.is_set(),
                      release_thread_joined=waiter is None or not waiter.is_alive(),
                      wall_s=round(time.monotonic() - started, 4),
                      wait_s=None if wait_elapsed is None else round(wait_elapsed, 4))
        out = Path(os.environ["LFS_PAIR_RECEIPTS"])
        out.mkdir(parents=True, exist_ok=True)
        (out / (variant + ".json")).write_text(json.dumps(result) + "\n")

    try:
        # Source/AST attribution is owned by the same unconditional cleanup as the
        # patch install and target invocation; failure cannot skip restoration.
        target = cell.test_pairing_admits_the_joiner_after_both_people_confirm
        source, first = inspect.getsourcelines(target)
        final_assert = next(n for n in ast.walk(ast.parse("".join(source))) if isinstance(n, ast.Assert)
                            and ast.unparse(n.test) == "'member_admitted' in events")
        expected_line = first + final_assert.lineno - 1
        monkeypatch.setattr(cell.audit_mod.AuditLog, "record", record)
        monkeypatch.setattr(cell, "_await_event", observe)
        try:
            target(devices, monkeypatch)
        except AssertionError as exc:
            trace = exc.__traceback__
            while trace.tb_next is not None:
                trace = trace.tb_next
            assert trace.tb_frame.f_code is target.__code__ and trace.tb_lineno == expected_line
            failure = "original_member_admitted_assertion"
        result["target_pass"] = failure is None
        assert result["observation_reached"]
        if variant == "new-release":
            assert failure is None and producer_exit() and append_done.is_set()
            assert False in observations and observations[-1] is True
        else:
            assert failure == "original_member_admitted_assertion"
        if variant == "new-omission":
            assert wait_elapsed is not None and wait_elapsed >= 5.0
    finally:
        actions = [("release", release.set), ("waiter", join_waiter),
                   ("producer-exit", producer_exit), ("keys", restore_keys)]
        result["errors"].extend(controller.cleanup_steps(actions))
        # Restore only after wrapper-exit witness; on unsettled producer leave
        # patch until the fixture stops real relay threads (monkeypatch teardown
        # follows devices teardown). Never call the Event physical settlement.
        if wrapper_exit.is_set() or not entered.is_set():
            result["errors"].extend(controller.cleanup_steps([("restore", restore_wrappers)]))
        try:
            emit()
        finally:
            assert not result["errors"], "diagnostic cleanup failed"
