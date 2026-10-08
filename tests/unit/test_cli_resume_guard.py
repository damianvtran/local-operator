"""Cold-resume guards: full TUI uses the shared session factory; exec refuses."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.cli import main as cli_main


@pytest.fixture
def config(tmp_path: Path, monkeypatch) -> Path:
    cfg = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    return cfg


def _own(config: Path, session_id: str, pid: int) -> None:
    d = config / "sessions" / session_id
    d.mkdir(parents=True, exist_ok=True)
    # resume_dir requires a transcript to consider the session resumable.
    (d / "transcript.jsonl").write_text("")
    (d / ".session.pid").write_text(str(pid))


def test_exec_resume_owned_refuses_exit_1(config: Path, monkeypatch, capsys) -> None:
    sleeper_pid = os.getppid()  # a live pid that is not this process
    _own(config, "sess-owned", sleeper_pid)
    monkeypatch.setattr(
        "sys.argv",
        ["local-operator", "exec", "--resume", "sess-owned", "do the thing"],
    )
    code = cli_main()
    assert code == 1
    err = capsys.readouterr().err
    assert "already open in another process" in err
    assert str(sleeper_pid) in err


def test_exec_resume_unowned_proceeds_to_exec(config: Path, monkeypatch) -> None:
    # No marker: the guard must not interfere. Patch the preflight AND run_exec
    # to observe the call rather than standing up a provider.
    _own(config, "sess-free", os.getpid())  # owner == self: not "another process"
    seen: list[tuple[object, object]] = []

    def fake_run_exec(command, args):  # noqa: ANN001
        seen.append((command, args))
        return 0

    monkeypatch.setattr(
        "local_operator.exec_mode.resolve_hosting_model_dry",
        lambda a: ("anthropic", "claude-x"),
        raising=False,
    )
    import local_operator.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_preflight_api_key", lambda *a, **k: None)
    monkeypatch.setattr("local_operator.exec_mode.run_exec", fake_run_exec)
    monkeypatch.setattr("sys.argv", ["local-operator", "exec", "--resume", "sess-free", "task"])
    code = cli_main()
    assert code == 0
    assert seen and seen[0][0] == "task"


def test_exec_resume_zombie_owner_proceeds_to_exec(config: Path, monkeypatch) -> None:
    """A killed runtime's corpse is not "another process".

    The guard's premise is that a pid in the marker means someone is hosting
    the session and this process must not become a second writer. Signal 0 is
    not enough to establish that: it succeeds against an exited-but-unreaped
    process, which is exactly what a SIGKILLed runtime leaves behind when its
    parent (a long-lived TUI) never reaps it. The reported symptom was a
    session refused by every interface with "already open in another process
    (pid N)", where N was a corpse — so the guard must not fire here, and the
    exec must proceed.
    """
    from tests.unreaped import unreaped_child

    seen: list[tuple[object, object]] = []

    def fake_run_exec(command, args):  # noqa: ANN001
        seen.append((command, args))
        return 0

    monkeypatch.setattr(
        "local_operator.exec_mode.resolve_hosting_model_dry",
        lambda a: ("anthropic", "claude-x"),
        raising=False,
    )
    import local_operator.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_preflight_api_key", lambda *a, **k: None)
    monkeypatch.setattr("local_operator.exec_mode.run_exec", fake_run_exec)
    # Held for the whole assertion: the marker must name a pid that is a zombie
    # at the moment the guard probes it, not one that has been reaped since.
    with unreaped_child() as zombie_pid:
        _own(config, "sess-zombie", zombie_pid)
        monkeypatch.setattr(
            "sys.argv",
            ["local-operator", "exec", "--resume", "sess-zombie", "do the thing"],
        )
        code = cli_main()
    assert code == 0
    assert seen and seen[0][0] == "do the thing"


def test_standalone_attach_modules_are_deleted() -> None:
    """Cold resume cannot regress to the projection screen or exit-75 shim."""
    import importlib.util

    assert importlib.util.find_spec("local_operator.cli_attach") is None
    assert importlib.util.find_spec("local_operator.tui.attach_screen") is None


def test_startup_import_weight_unchanged() -> None:
    """The CLI startup guard: importing cli must not pull Textual or the
    mobile package's heavy half (attach imports are lazy on the owned branch)."""
    import subprocess
    import sys

    code = (
        "import sys, local_operator.cli; "
        "bad = [m for m in sys.modules if m.startswith('textual') "
        "or m.startswith('local_operator.mobile.attach_client') "
        "or m.startswith('local_operator.session.attached')]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=repo_root
    )
    assert out.stdout.strip() == "CLEAN", out.stdout


# --- the startup ``--resume`` arm and a SILENT device (mesh-wire-honesty S2, R2-2) ----------
#
# ``cli.main()``'s pre-check answers the ``--resume`` FLAG before any session factory
# or TUI exists (it is a sibling of the subprocess-subcommand branch, so the TUI launch
# takes it too). It used to do a row-only lookup, which discards the silence, so an id
# that did not resolve BECAUSE a device stayed silent printed the generic "no session
# to resume" copy — the claim of absence the refusal family exists to avoid. These
# cells drive the real ``main()`` over the real producer (TTL, ``_read_all``, the miss's
# live read); only the relay's catalogue is injected, the seam the producer's own suites use.


def _silent_mesh(monkeypatch: pytest.MonkeyPatch, *, silent: bool, config: Path):
    """A relay record behind ``config`` whose one device either stays silent or answers."""
    from local_operator.network import projection, store
    from local_operator.session import peer_rows
    from tests.unit.session.test_peer_rows import _Catalog, _Facts

    peer_rows.clear_cache()
    facts = [
        _Facts(
            "d_build",
            "build-box",
            reachable=not silent,
            reason="connect_failed:ConnectionRefusedError" if silent else "",
        )
    ]
    catalog = _Catalog(facts, [])
    monkeypatch.setattr(store, "find_own_relay", lambda root=None: object())
    monkeypatch.setattr(projection, "RelayPeerCatalog", lambda root: catalog)
    return catalog


@pytest.mark.parametrize("tty", [True, False], ids=["tui-launch", "non-tui"])
def test_resume_of_an_unresolved_id_beside_a_silent_device_says_the_silence(
    config: Path, monkeypatch, capsys, tty: bool
) -> None:
    """A silent miss prints the unresolved sentence, returns 1, and boots nothing."""
    import local_operator.cli as cli_mod
    from local_operator.session.remote_open import unresolved_peer_sentence

    config.mkdir(parents=True, exist_ok=True)
    catalog = _silent_mesh(monkeypatch, silent=True, config=config)
    monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: tty)

    async def _no_boot(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("a silent miss must not build ANY session, local or viewer")

    monkeypatch.setattr(cli_mod, "create_session", _no_boot)
    monkeypatch.setattr("sys.argv", ["local-operator", "--resume", "a1b2c3d4e5f6"])

    assert cli_main() == 1

    err = capsys.readouterr().err
    assert "build-box did not answer" in err, err
    assert "may be on that device" in err, err
    assert "no session" not in err, "the generic absence copy was printed"
    assert "recent sessions" not in err, "this machine's list is not the help for a peer's id"
    from local_operator.session.peer_rows import UnansweredPeer

    expected = unresolved_peer_sentence(
        "a1b2c3d4e5f6", (UnansweredPeer("d_build", "build-box", "connect_failed"),)
    )
    assert expected in err
    assert catalog.calls == 1, "the pre-check paid more than one fan-out"


def test_resume_of_an_unresolved_id_every_device_answered_keeps_the_generic_copy(
    config: Path, monkeypatch, capsys
) -> None:
    """The control: nobody silent is a real miss, and the typo copy is unchanged."""
    config.mkdir(parents=True, exist_ok=True)
    _silent_mesh(monkeypatch, silent=False, config=config)
    monkeypatch.setattr("sys.argv", ["local-operator", "--resume", "a1b2c3d4e5f6"])

    assert cli_main() == 1

    err = capsys.readouterr().err
    assert "no session 'a1b2c3d4e5f6' to resume" in err, err
    assert "did not answer" not in err, err


@pytest.mark.parametrize(
    ("requested", "expected", "tty"),
    [
        # The bare ``--resume`` flag (argparse hands the sentinel) with no local sessions.
        ("@latest", "no previous session to resume", False),
        ("@latest", "no previous session to resume", True),
        # A path-shaped string is not an id at all.
        ("../x", "not a session id", False),
        ("a/b", "not a session id", False),
    ],
    ids=["bare-latest", "bare-latest-tty", "dotdot", "slash"],
)
def test_a_silent_device_does_not_rewrite_the_refusals_of_a_non_id(
    config: Path, monkeypatch, capsys, requested: str, expected: str, tty: bool
) -> None:
    """F-1: silence is evidence about an id a device could HOLD, not about a non-id.

    With a silent device and no local sessions the startup arm used to speak for
    ``@latest`` and for path-shaped strings too, replacing "no previous session to
    resume" / "not a session id" with the peer sentence.
    """
    import local_operator.cli as cli_mod

    config.mkdir(parents=True, exist_ok=True)
    _silent_mesh(monkeypatch, silent=True, config=config)
    monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: tty)
    monkeypatch.setattr("sys.argv", ["local-operator", "--resume", requested])

    assert cli_main() == 1

    err = capsys.readouterr().err
    assert expected in err, err
    assert "did not answer" not in err, err
