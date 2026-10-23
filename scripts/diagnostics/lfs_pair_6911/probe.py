"""DIAGNOSTIC ONLY: actual pairing with one held/suppressed audit append.

Run only this file explicitly, offhost, serially. No product or fixture copy;
all admission/person assertions execute in the existing target test. The old
counterfactual substitutes its former snapshot observation, not the relay.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import threading
import time

import pytest
from tests.unit.network import test_relay_e2e as cell
from tests.unit.network.conftest import root  # noqa: F401 — existing pytest fixture
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — existing fixture


@pytest.mark.parametrize("variant", ["old-held", "new-release", "new-omission"])
def test_pair_audit_boundary(devices, monkeypatch, variant):
    server_a = devices[0]
    entered, release, sampling, append_done = (threading.Event() for _ in range(4))
    original_record = cell.audit_mod.AuditLog.record
    original_events, original_await = cell._events, cell._await_event
    waiter = None
    result = {"variant": variant, "observation_reached": False}
    real_observations = []
    failure = None
    answered_before = set(cell._pair_answered)
    wait_elapsed = None
    started = time.monotonic()

    def record(owner, event):
        if owner is server_a.audit and event.event == "member_admitted":
            entered.set()  # Producer reached this only AFTER success send.
            if variant == "new-omission":
                return  # Controlled missing event; never suppress other events.
            if not release.wait(20):
                return  # Outer assertions reject absent append_done; no raised relay error.
            original_record(owner, event)
            append_done.set()
            return
        original_record(owner, event)

    def sampled_events(server):
        events = original_events(server)
        if server is server_a:
            real_observations.append("member_admitted" in events)
            if not sampling.is_set():
                assert "member_admitted" not in events
                sampling.set()  # Release only AFTER a real reader observed absence.
        return events

    def observe(server, event, *, timeout=5.0):
        nonlocal waiter, wait_elapsed
        assert server is server_a and event == "member_admitted" and timeout == 5.0
        assert entered.wait(5), "audit producer never reached post-success boundary"
        result["observation_reached"] = True  # ALL earlier original assertions passed.
        assert "member_admitted" not in original_events(server)
        if variant == "old-held":
            return original_events(server)  # EXACT former one-shot observation.
        if variant == "new-release":
            def release_after_read():
                if sampling.wait(5):
                    release.set()
            waiter = threading.Thread(target=release_after_read, daemon=False)
            waiter.start()
        monkeypatch.setattr(cell, "_events", sampled_events)
        wait_started = time.monotonic()
        try:
            return original_await(server, event, timeout=timeout)
        finally:
            wait_elapsed = time.monotonic() - wait_started

    target = cell.test_pairing_admits_the_joiner_after_both_people_confirm
    # Missing-event negative must fail the ORIGINAL final assertion, not any setup,
    # earlier admission assertion, boundary condition, import or helper exception.
    source_lines, first = inspect.getsourcelines(target)
    tree = ast.parse("".join(source_lines))
    final_assert = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert)
                        and ast.unparse(n.test) == "'member_admitted' in events")
    expected_line = first + final_assert.lineno - 1
    monkeypatch.setattr(cell.audit_mod.AuditLog, "record", record)
    monkeypatch.setattr(cell, "_await_event", observe)
    try:
        try:
            target(devices, monkeypatch)
        except AssertionError as exc:
            trace = exc.__traceback__
            while trace.tb_next is not None:
                trace = trace.tb_next
            assert trace.tb_frame.f_code is target.__code__ and trace.tb_lineno == expected_line, (
                "negative failed somewhere other than the original missing-event assertion"
            )
            failure = "original_member_admitted_assertion"
        result["target_pass"] = failure is None
        assert result["observation_reached"]
        if variant == "new-release":
            assert failure is None and append_done.is_set()
            assert False in real_observations and real_observations[-1] is True
        else:
            assert failure == "original_member_admitted_assertion"
        if variant == "new-omission":
            assert wait_elapsed is not None and wait_elapsed >= 5.0
    finally:
        release.set()
        if waiter is not None:
            waiter.join(5)
            assert not waiter.is_alive(), "owned release thread did not settle"
        # Settle the relay's real append BEFORE devices fixture stops its workers.
        if entered.is_set() and variant != "new-omission":
            assert append_done.wait(5), "audit append did not settle after release"
        result.update(failure=failure, append_done=append_done.is_set(),
                      release_thread_joined=waiter is None or not waiter.is_alive(),
                      wall_s=round(time.monotonic() - started, 4),
                      wait_s=None if wait_elapsed is None else round(wait_elapsed, 4))
        for network_id in set(cell._pair_answered) - answered_before:
            del cell._pair_answered[network_id]
        # Fixed synthetic labels/counts only: no identities, payloads, audit contents,
        # exception messages, tokens, transcripts, environment or database paths.
        out = Path(os.environ["LFS_PAIR_RECEIPTS"])
        out.mkdir(parents=True, exist_ok=True)
        (out / (variant + ".json")).write_text(json.dumps(result) + "\n")
