"""The target-side signal receipt (``signal_receipt``) and its registry artifact.

Synthetic throughout: a tmp conversation directory, hand-built markers, and the
production functions. Nothing here signals a process or reads an operator store.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.macos_disclaim import ENV_SPAWN_CHAIN
from local_operator.session.runtime import registry, signal_receipt

SESSION = "recv00000001"
PID = 424242
STARTED = 1_760_000_000.0


def _marker(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": SESSION,
        "pid": PID,
        "started_at": STARTED,
        "at": time.time() - 1.0,
        "rung": "sigterm",
        "deliberate": True,
        "killer": {"pid": 77, "argv0": "lop", "command": "lop stop"},
    }
    payload.update(overrides)
    return payload


def _record(directory: Path | None, **overrides: Any) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {
        "kind": "runtime",
        "session_id": SESSION,
        "pid": PID,
        "started_at": STARTED,
        "name": "SIGTERM",
        "number": 15,
        "in_flight": True,
        "action": "drain",
    }
    kwargs.update(overrides)
    return signal_receipt.record(directory, **kwargs)


def test_an_unsanctioned_signal_writes_the_v1_receipt(tmp_path: Path) -> None:
    receipt = _record(tmp_path)
    on_disk = registry.read_signal_receipt(tmp_path)
    assert on_disk == json.loads((tmp_path / "runtime-signal.json").read_text())
    assert receipt is not None and on_disk is not None
    assert (on_disk["v"], on_disk["kind"], on_disk["count"]) == (1, "runtime", 1)
    assert (on_disk["session_id"], on_disk["pid"], on_disk["started_at"]) == (
        SESSION,
        PID,
        STARTED,
    )
    entry = on_disk["signals"][0]
    assert (entry["name"], entry["number"], entry["sanction"]) == ("SIGTERM", 15, "none")
    assert entry["stop_marker"] is None
    assert entry["sender"] == {
        "state": "unavailable",
        "reason": "no-siginfo",
        "could_be": "same-uid-or-root",
    }
    assert {"uid", "ppid", "pgid", "argv0"} <= set(on_disk["receiver"])
    assert stat.S_IMODE(os.stat(tmp_path / "runtime-signal.json").st_mode) == 0o600


def test_a_covering_marker_sanctions_the_signal_and_is_snapshotted(tmp_path: Path) -> None:
    registry.write_stop_marker(tmp_path, _marker(sweep_id="abc123abc123"))
    _record(tmp_path)
    entry = registry.read_signal_receipt(tmp_path)["signals"][0]  # type: ignore[index]
    assert entry["sanction"] == "marker"
    assert entry["stop_marker"]["deliberate"] is True
    assert entry["stop_marker"]["killer_pid"] == 77
    assert entry["stop_marker"]["sweep_id"] == "abc123abc123"
    assert signal_receipt.stop_class_of(entry) == "deliberate"
    assert signal_receipt.cut_off_verdict(entry) is None


@pytest.mark.parametrize(
    "case",
    [
        ("another-run", {"pid": PID + 1}),
        ("another-session", {"session_id": "someone-else"}),
        ("another-start", {"started_at": STARTED + 500}),
        # THE TWO CLOCK CASES COMPUTE THEIR STAMP IN THE BODY, never in the
        # decorator: a value evaluated at COLLECTION time ages while the module is
        # imported, and the "staged after the signal" case then measures a marker
        # staged BEFORE it as soon as the run is a minute old — which is how this
        # cell went red on CI (agent review round 1, MINOR 1). The offsets are
        # relative to the moment the marker is written.
        ("stale", {"at": -(signal_receipt.PAIR_WINDOW_S + 30)}),
        ("staged-after-the-signal", {"at": +60.0}),
    ],
)
def test_a_marker_that_does_not_cover_the_run_or_the_moment_sanctions_nothing(
    tmp_path: Path, case: tuple[str, dict[str, Any]]
) -> None:
    """A marker must describe THIS run AND have been staged before the signal reached it.

    The last row is the ordering invariant rather than a nicety: a marker written
    AFTER the signal did not sanction it, so it may not narrate the death as a stop
    somebody asked for.
    """
    _name, overrides = case
    overrides = dict(overrides)
    if "at" in overrides:
        overrides["at"] = time.time() + overrides["at"]
    registry.write_stop_marker(tmp_path, _marker(**overrides))
    _record(tmp_path)
    entry = registry.read_signal_receipt(tmp_path)["signals"][0]  # type: ignore[index]
    assert entry["sanction"] == "none" and entry["stop_marker"] is None
    assert signal_receipt.stop_class_of(entry) == "unattributed-signal"


def test_an_involuntary_marker_is_attributed_and_not_deliberate(tmp_path: Path) -> None:
    registry.write_stop_marker(
        tmp_path,
        _marker(deliberate=False, mechanism="generation-prune", actor="lop install prune"),
    )
    _record(tmp_path)
    entry = registry.read_signal_receipt(tmp_path)["signals"][0]  # type: ignore[index]
    assert signal_receipt.stop_class_of(entry) == "attributed-involuntary"
    cause, detail = signal_receipt.cut_off_verdict(entry)  # type: ignore[misc]
    assert cause == "runtime-killed" and "lop install prune" in detail


def test_repeats_append_within_a_run_and_are_bounded_and_a_new_run_replaces(
    tmp_path: Path,
) -> None:
    for _ in range(signal_receipt.MAX_SIGNALS + 3):
        _record(tmp_path)
    receipt = registry.read_signal_receipt(tmp_path)
    assert receipt is not None
    assert len(receipt["signals"]) == signal_receipt.MAX_SIGNALS
    assert receipt["count"] == signal_receipt.MAX_SIGNALS + 3
    _record(tmp_path, pid=PID + 9)
    fresh = registry.read_signal_receipt(tmp_path)
    assert fresh is not None and fresh["pid"] == PID + 9 and fresh["count"] == 1


def test_no_conversation_directory_is_logged_loudly_and_not_created(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "not-materialised"
    with caplog.at_level(logging.WARNING):
        receipt = _record(missing)
    assert receipt is not None and not missing.exists()
    assert "logged only" in caplog.text and "SIGTERM" in caplog.text


def test_a_failing_write_is_a_warning_with_the_facts_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("disk says no")

    monkeypatch.setattr(registry, "write_signal_receipt", boom)
    with caplog.at_level(logging.WARNING):
        assert _record(tmp_path) is None
    assert "could not be written" in caplog.text
    assert SESSION in caplog.text and str(PID) in caplog.text and "SIGTERM" in caplog.text


def test_a_receipt_for_another_run_is_refused_by_the_reader(tmp_path: Path) -> None:
    class Dead:
        pid = PID
        started_at = STARTED

    _record(tmp_path)
    receipt = registry.read_signal_receipt(tmp_path)
    assert signal_receipt.covers_run(receipt, SESSION, Dead())
    assert not signal_receipt.covers_run(receipt, "other", Dead())
    Dead.pid = PID + 1
    assert not signal_receipt.covers_run(receipt, SESSION, Dead())
    assert not signal_receipt.covers_run(receipt, SESSION, None)
    assert not signal_receipt.covers_run(None, SESSION, Dead())


def test_an_unsanctioned_verdict_is_the_signal_story_on_existing_tokens() -> None:
    entry = signal_receipt.build_signal(
        "SIGTERM", 15, at=time.time(), in_flight=True, action="stop", marker=None, covered=False
    )
    cause, detail = signal_receipt.cut_off_verdict(entry)  # type: ignore[misc]
    assert cause == "runtime-shutdown"
    assert "SIGTERM received" in detail and "unidentified sender" in detail
    assert "nobody asked for a stop" in detail
    assert signal_receipt.describe({"signals": [entry], "count": 1})


def test_an_install_shaped_caller_pairs_through_the_real_involuntary_writer(tmp_path: Path) -> None:
    """Wave C (2026-10-01 01:00Z): runtimes were disposed minutes after an install.

    The install/prune paths stage their marker with ``control.note_involuntary_stop``
    before they signal. This cell uses that REAL writer (no hand-built payload), then
    the receipt a runtime writes on arrival, and pins both halves: with the marker the
    signal is ``attributed-involuntary`` naming the install mechanism and NOT a user
    stop; the same signal with no marker (an install that did not stage one, or an
    unrelated sweep) is ``unattributed-signal``. Nothing here signals a process.
    """
    from local_operator.session.runtime import control

    root = tmp_path / "cfg"
    root.mkdir()

    class _Target:
        session_id = SESSION
        pid = PID
        started_at = STARTED
        version = "0.64.11"
        source_ref = "abc1234"

    conversation = root / "sessions" / SESSION
    conversation.mkdir(parents=True, exist_ok=True)
    assert control.note_involuntary_stop(
        _Target(), root, mechanism="in-place-install", actor="lop update"
    )
    receipt = _record(conversation)
    assert receipt is not None
    entry = receipt["signals"][0]
    assert entry["sanction"] == "marker"
    assert entry["stop_marker"]["deliberate"] is False
    assert entry["stop_marker"]["mechanism"] == "in-place-install"
    assert signal_receipt.stop_class_of(entry) == "attributed-involuntary"
    cause, detail = signal_receipt.cut_off_verdict(entry)  # type: ignore[misc]
    assert cause == "runtime-killed" and "lop update" in detail

    bare = tmp_path / "bare"
    bare.mkdir()
    unmarked = _record(bare)
    assert unmarked is not None
    assert signal_receipt.stop_class_of(unmarked["signals"][0]) == "unattributed-signal"


def test_a_recorded_spawn_chain_rides_the_receipt_with_both_readings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direction fact a target can carry, both halves (design round 1, D1).

    Each entry keeps the liveness RECORDED at spawn (``alive_at_spawn``) and
    gains the one measured at arrival (``alive_now``) — the renderer's clause
    gates on both. A disclaimed ancestor (the fix's whole point) simply never
    appears as the app, because the chain was recorded at spawn.
    """
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps(
            [
                {"pid": os.getpid(), "argv0": "lop exec", "alive_at_spawn": True},
                {
                    "pid": 2_147_483_600,
                    "argv0": "/Applications/Local Operator.app/Contents/MacOS/Local Operator",
                    "alive_at_spawn": True,
                },
            ]
        ),
    )
    directory = tmp_path / "chain"
    directory.mkdir()
    receipt = _record(directory)
    assert receipt is not None
    chain = receipt["signals"][0].get("spawn_chain")
    assert isinstance(chain, list) and len(chain) == 2
    assert chain[0]["alive_at_spawn"] is True and chain[0]["alive_now"] is True
    assert chain[1]["alive_at_spawn"] is True and chain[1]["alive_now"] is False


def test_a_receipt_without_a_chain_omits_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    directory = tmp_path / "nochain"
    directory.mkdir()
    receipt = _record(directory)
    assert receipt is not None
    assert "spawn_chain" not in receipt["signals"][0]


def test_an_unattributed_verdict_names_a_gone_app_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-10-09: the runtime cannot name the sender; it CAN say the app outlived."""
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps(
            [
                {"pid": os.getpid(), "argv0": "lop", "alive_at_spawn": True},
                {
                    "pid": 2_147_483_600,
                    "argv0": "/Applications/Local Operator.app/Contents/MacOS/Local Operator",
                    "alive_at_spawn": True,
                },
            ]
        ),
    )
    directory = tmp_path / "gone"
    directory.mkdir()
    receipt = _record(directory)
    assert receipt is not None
    verdict = signal_receipt.cut_off_verdict(receipt["signals"][0], count=1)
    assert verdict is not None
    _cause, detail = verdict
    assert (
        "the app this runtime descends from (Local Operator.app, pid 2147483600)"
        " was running when the runtime started and was no longer running when the signal arrived"
    ) in detail
    # The recorded command line is persisted nowhere the reader sees it (D3).
    assert "/Applications/" not in detail


def test_an_unattributed_verdict_without_the_clause_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ALIVE app ancestor — or one already gone at spawn — renders the old sentence."""
    directory = tmp_path / "alive"
    directory.mkdir()
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps(
            [
                {
                    "pid": os.getpid(),
                    "argv0": "/Applications/Local Operator.app/Contents/MacOS/Local Operator",
                    "alive_at_spawn": True,
                }
            ]
        ),
    )
    receipt = _record(directory)
    assert receipt is not None
    verdict = signal_receipt.cut_off_verdict(receipt["signals"][0], count=1)
    assert verdict is not None
    _cause, detail = verdict
    assert "was no longer running" not in detail
    assert "from an unidentified sender" in detail

    # Already gone when the chain was written for this runtime: it did not
    # outlive anything, so the gate stays shut (design round 1, D1).
    gone_at_spawn = tmp_path / "gone-at-spawn"
    gone_at_spawn.mkdir()
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps(
            [
                {
                    "pid": 2_147_483_600,
                    "argv0": "/Applications/Local Operator.app/Contents/MacOS/Local Operator",
                    "alive_at_spawn": False,
                }
            ]
        ),
    )
    receipt = _record(gone_at_spawn)
    assert receipt is not None
    verdict = signal_receipt.cut_off_verdict(receipt["signals"][0], count=1)
    assert verdict is not None
    _cause, detail = verdict
    assert "was no longer running" not in detail
