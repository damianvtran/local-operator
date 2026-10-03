"""The carried-wake drill's remedy at the promote: install, start, VERIFY, and say so.

The drill (2026-10-03, two real devices) moved a session carrying a daily wake and
found the destination's supervisor loaded-but-stopped: the overdue wake fired only
after a hand-run `lop wake install`, and nothing in the flow had said a word. The
cells here pin the remedy's halves:

* a promote that carries wakes runs the §5.3 ``ensure`` step and reports the VERIFIED
  supervisor state — ``running``, or the exact loud fallback when it cannot be
  (including an ensure that raises: there is no silent branch through the block);
* a promote that carries no wakes never installs anything;
* the monitors statement ("state does not travel") renders where it is written — on
  the receipt, and in the two documents that own the carry contract.

Everything runs against faked installer seams: the real ones shell out to
launchctl/systemd, which a test must never do (``wakes/install.py``'s own guard).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import carry, mobility
from local_operator.wakes import install as wakes_install

REPO = Path(__file__).resolve().parents[3]

SESSION = "aabbccddeeff"


def _custom_entry(custom_type: str, details: dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": f"e-{custom_type}",
            "ts": 1.0,
            "type": "custom",
            "payload": {"custom_type": custom_type, "details": details},
        },
        separators=(",", ":"),
    )


def _wake_row() -> dict[str, Any]:
    # A far-future due time: this cell is about the runner, not the due date, and a
    # past due would be rewritten by whichever runtime opened the session.
    return {
        "id": "w1",
        "message": "carried",
        "next_due_at": 9_999_999_999_999,
        "every_ms": 86_400_000,
        "fired_count": 1,
        "created_at": 1,
        "notify": False,
    }


def _stage(root: Path, *, wakes: int) -> Path:
    """A verified copy, staged where a promote expects it, with ``wakes`` wake rows."""
    staging = root / "network" / "staging" / SESSION
    staging.mkdir(parents=True)
    lines = [json.dumps({"id": "e1", "ts": 1.0, "type": "message", "payload": {}})]
    if wakes:
        lines.append(_custom_entry("wake_schedules", {"schedules": [_wake_row()]}))
    (staging / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return staging


def _server(root: Path, name: str = "dest-device") -> Any:
    return SimpleNamespace(
        root=root,
        identity=SimpleNamespace(device_id="d_destination"),
        _own_label=lambda: name,
    )


def _fake_installer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    installed: bool = True,
    reason: str = "installed",
    running: bool = True,
    detail: str = "active/running",
    ensure_raises: Exception | None = None,
) -> list[Path]:
    """Fake the installer seams, record ensure calls, and answer the verify shape."""
    calls: list[Path] = []

    def fake_installed(config_dir: Path) -> wakes_install.InstallOutcome:
        calls.append(Path(config_dir))
        if ensure_raises is not None:
            raise ensure_raises
        return wakes_install.InstallOutcome(installed=installed, reason=reason)

    def fake_state(config_dir: Path) -> wakes_install.SupervisorState:
        return wakes_install.SupervisorState(loaded=True, running=running, detail=detail)

    monkeypatch.setattr(wakes_install, "ensure_supervisor_installed", fake_installed)
    monkeypatch.setattr(wakes_install, "supervisor_state", fake_state)
    return calls


# ---------------------------------------------------------------------------
# (a) carried wakes leave a RUNNING supervisor — or the loud fallback, exactly
# ---------------------------------------------------------------------------


def test_a_promote_that_carries_wakes_ensures_and_reports_the_running_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy's happy path, through the REAL rebuild and ensure choreography."""
    from local_operator.wakes import store as wake_store

    calls = _fake_installer(monkeypatch)
    root = tmp_path / "dest"
    root.mkdir()
    staging = _stage(root, wakes=1)

    outcome = mobility._promote(_server(root), staging, SESSION)

    assert outcome.ok is True
    # The §5.3 ensure ran, FOR THIS STORE, because wakes were carried.
    assert calls == [root]
    # The carried schedule is live on this device: the rebuilt index holds the row.
    entry = wake_store.read_entry(root, SESSION)
    assert entry is not None and len(entry["schedules"]) == 1
    # And the receipt's half: verified running, no loud line needed.
    assert outcome.carry == {"wakes": 1, "supervisor": "running"}


def test_a_supervisor_that_is_not_running_produces_the_exact_fallback_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drill's own shape: wakes arrived, nothing would fire them, and now it says so.

    The sentence is asserted EXACTLY — the drill lane's requirement — because it is
    the one line that turns a silent stopped supervisor into a next step.
    """
    _fake_installer(
        monkeypatch,
        installed=False,
        reason="unit written; the user manager is not addressable from here; "
        "it takes effect at your next login",
        running=False,
        detail="inactive/dead",
    )
    root = tmp_path / "dest"
    root.mkdir()
    staging = _stage(root, wakes=1)

    outcome = mobility._promote(_server(root), staging, SESSION)

    assert outcome.ok is True
    assert outcome.carry is not None
    assert outcome.carry["wakes"] == 1
    # ``supervisor`` is the machine answer (verified or not); ``notice`` is the
    # loud line — byte for byte the copy the drill requires, and the one thing
    # that turns a silent stopped supervisor into a next step.
    assert outcome.carry["supervisor"] == "not running"
    assert outcome.carry["notice"] == (
        "1 wakes carried; supervisor not running on dest-device — run `lop wake install`"
    )


def test_an_ensure_that_raises_still_produces_the_loud_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No silent branch: a broken installer is exactly when the line matters most."""
    _fake_installer(monkeypatch, ensure_raises=RuntimeError("boom"))
    root = tmp_path / "dest"
    root.mkdir()
    staging = _stage(root, wakes=1)

    outcome = mobility._promote(_server(root), staging, SESSION)

    assert outcome.ok is True
    assert outcome.carry is not None
    assert outcome.carry["wakes"] == 1
    assert outcome.carry["supervisor"] == "not running"
    assert outcome.carry["notice"] == (
        "1 wakes carried; supervisor not running on dest-device — run `lop wake install`"
    )


def test_a_catastrophic_ensure_failure_still_produces_the_loud_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inner layer's own belt-and-braces: even ensure_supervisor raising is spoken."""
    monkeypatch.setattr(
        carry,
        "ensure_supervisor",
        lambda root: (_ for _ in ()).throw(RuntimeError("kernel exploded")),
    )
    root = tmp_path / "dest"
    root.mkdir()
    staging = _stage(root, wakes=1)

    outcome = mobility._promote(_server(root), staging, SESSION)

    assert outcome.ok is True
    assert outcome.carry is not None
    assert outcome.carry["notice"] == (
        "1 wakes carried; supervisor not running on dest-device — run `lop wake install`"
    )


def test_an_unexpected_ensure_answer_still_produces_the_loud_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seam answering a shape this module did not expect must not fail the move.

    Measured 2026-10-03: a test stub answering the OLD string shape raised inside
    the promote's fallback logging, out of the destination's handler, and the move
    failed ``relay_unavailable`` — the exact failure the loud line exists to
    prevent, one layer up. The promote's contract is "never fail here", so the
    old shape is driven through the real path and must still end in the sentence.
    """
    monkeypatch.setattr(carry, "ensure_supervisor", lambda root: "stubbed")
    root = tmp_path / "dest"
    root.mkdir()
    staging = _stage(root, wakes=1)

    outcome = mobility._promote(_server(root), staging, SESSION)

    assert outcome.ok is True
    assert outcome.carry is not None
    assert outcome.carry["supervisor"] == "not running"
    assert outcome.carry["notice"] == (
        "1 wakes carried; supervisor not running on dest-device — run `lop wake install`"
    )


def test_a_promote_that_carries_no_wakes_never_installs_a_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No wakes, no install: the hook exists so sleepers pay nothing for it."""
    calls = _fake_installer(monkeypatch)
    root = tmp_path / "dest"
    root.mkdir()
    staging = _stage(root, wakes=0)

    outcome = mobility._promote(_server(root), staging, SESSION)

    assert outcome.ok is True
    assert calls == []
    assert outcome.carry is None


def test_the_verify_half_reports_a_stopped_supervisor_even_when_the_installer_claimed_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``installed`` is the installer's claim; ``running`` is the fact the move relies on.

    The drill measured a supervisor that was started and then gone; an installer that
    says "installed" while the state probe reads stopped must not read as success.
    """
    _fake_installer(monkeypatch, installed=True, reason="already installed", running=False)
    root = tmp_path / "dest"
    root.mkdir()

    result = carry.ensure_supervisor(root)

    assert result["installed"] is True
    assert result["running"] is False


# ---------------------------------------------------------------------------
# (c) the monitors statement — where the contract lives, and how it renders
# ---------------------------------------------------------------------------


def test_the_monitors_statement_is_written_where_the_carry_contract_lives() -> None:
    """The drill's second finding was that NOTHING in the product said this.

    Pinned in both documents the task names (the session-mobility design doc and the
    network guide); the receipt's own rendering is the cell below.
    """
    statement = "Monitor state does not travel with a move"
    design = (REPO / "docs/design/mesh-session-mobility.md").read_text(encoding="utf-8")
    guide = (REPO / "local_operator/guides/network/GUIDE.md").read_text(encoding="utf-8")
    assert statement in design
    assert statement in guide
    # And the remedy is named beside it: re-baselining, not recreation (the spec rows
    # ride the transcript; telling someone to recreate them would double the monitor).
    assert "re-baselines" in design and "re-baselines" in guide


def test_the_monitors_line_renders_on_the_move_receipt(tmp_path: Path) -> None:
    """``carry.monitor_count`` feeds the sentence, and the CLI renders it as a line."""
    from local_operator import cli

    root = tmp_path / "source"
    monitors_dir = root / "monitors"
    monitors_dir.mkdir(parents=True)
    assert carry.monitor_count(root, SESSION) == 0
    (monitors_dir / f"{SESSION}.json").write_text(
        json.dumps({"schema": 1, "monitors": [{"id": "m1"}, {"id": "m2"}]}),
        encoding="utf-8",
    )
    assert carry.monitor_count(root, SESSION) == 2

    sentence = carry.monitors_notice(2)
    assert sentence == (
        "2 monitors here will not travel with their state — the destination re-baselines them"
    )
    result = {
        "ok": True,
        "session_id": SESSION,
        "new_session_id": SESSION,
        "mode": "move",
        "from_device": {"device_id": "d_source", "name": "this-device"},
        "to_device": {"device_id": "d_destination", "name": "dest-device"},
        "phase": "done",
        "phases": [{"phase": "done", "at": 0.0}],
        "carry": {"monitors": 2, "monitors_notice": sentence},
    }
    lines = cli._sessions_move_words(result, session_id=SESSION, to="dest-device")
    assert sentence in lines
    # A move with no carry block renders exactly the lines it always did.
    bare = cli._sessions_move_words(
        {key: value for key, value in result.items() if key != "carry"},
        session_id=SESSION,
        to="dest-device",
    )
    assert sentence not in bare
