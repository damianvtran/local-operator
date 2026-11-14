"""Turning on phone approvals from the desktop/TUI (design v2 §6, C3).

These cells drive :func:`authority_setup.setup_authority` with the in-process anchor
root seam, fake root ownership and a STUB gesture runner (§8 E2E step 1). The stub
stands in for the person answering the OS prompt; the host prechecks it must not be
able to bypass (privileged process, no GUI prompt) run before it is ever called.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.operator import authority_setup, elevate, handlers, trust
from local_operator.operator.elevate import ElevationResult


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """An isolated config root and anchor root; the anchor reads as root-owned."""
    root = tmp_path / "config"
    root.mkdir()
    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir()
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", anchor_root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    real_open = trust._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        if opened is None:
            return None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    monkeypatch.setattr(trust, "_open_no_follow", as_root)
    # A Linux host WITH pkexec: the server form may prompt. pkexec is resolved from
    # fixed paths, not PATH (R1-5), so the host's pkexec is a SHIM reached through
    # the test-launcher-only seam; it is never executed (every cell uses a runner).
    # Cells that need a host without it point the seam at nothing.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(trust, "_on_windows", lambda: False)
    pkexec = tmp_path / "bin" / "pkexec"
    pkexec.parent.mkdir()
    pkexec.write_text("#!/bin/sh\nexit 99\n")
    pkexec.chmod(0o755)
    monkeypatch.setattr(elevate, "_PKEXEC_CANDIDATES_OVERRIDE", (str(pkexec),))
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **k: "/usr/bin/sudo" if name == "sudo" else real_which(name),
    )
    calls: list[list[list[str]]] = []

    def approve(command: list[list[str]]) -> ElevationResult:
        """The person approved: land the staged bytes where the runtime reads them."""
        calls.append(command)
        trust.anchor_path().write_bytes(trust.staging_path(root).read_bytes())
        return ElevationResult(0)

    return SimpleNamespace(root=root, calls=calls, approve=approve)


def _audit(root: Path) -> list[dict[str, Any]]:
    path = root / "operator" / "authority-setup-audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_gesture_installs_a_listed_anchor_with_receipts_and_audit(host: SimpleNamespace) -> None:
    result = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)

    assert result["outcome"] == "on" and result["ok"] is True
    assert [r["step"] for r in result["receipts"]] == [
        "proposed",
        "consent",
        "generated",
        "installed",
        "verified",
    ]
    assert all(r["ok"] for r in result["receipts"])
    assert len(host.calls) == 1
    loaded = trust.load_anchor()
    assert loaded.usable
    # §6 item 4: the `init` shape, so every later phone still needs its admit prompt.
    assert loaded.anchor is not None
    assert trust.admission_enforced(loaded.anchor)
    assert loaded.anchor.admission == trust.ADMISSION_LISTED
    # The privileged argv pins the digest of the bytes staged (staging-swap race).
    flat = " ".join(" ".join(argv) for argv in host.calls[0])
    assert str(trust.anchor_path()) in flat
    # §6 item 6: one audit row per receipt, each naming the actor and the attempt.
    rows = _audit(host.root)
    assert [r["step"] for r in rows] == [r["step"] for r in result["receipts"]]
    assert {r["actor"] for r in rows} == {"desktop"}
    assert {r["attempt"] for r in rows} == {result["attempt"]}
    assert (
        oct((host.root / "operator" / "authority-setup-audit.jsonl").stat().st_mode)[-3:] == "600"
    )
    # §6 item 5: the level is reported as achieved; a file-only host is told so.
    assert result["file_only"] is (result["level"] == "operator-file-only")
    assert result["presence_gated"] is False


def test_installed_anchor_is_idempotent_and_never_prompts_again(host: SimpleNamespace) -> None:
    authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)
    before = trust.anchor_path().read_bytes()

    again = authority_setup.setup_authority(host.root, actor="tui", runner=host.approve)

    assert again["outcome"] == "already_on" and again["ok"] is True
    assert len(host.calls) == 1, "a working anchor is never replaced"
    assert trust.anchor_path().read_bytes() == before


def test_privileged_process_is_refused_not_shortcut(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handlers, "_is_privileged", lambda: True)

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)

    assert result["outcome"] == "privileged_process" and result["ok"] is False
    assert host.calls == []
    assert not trust.anchor_path().exists()
    assert not trust.staging_path(host.root).exists(), "refused before any key or staging"


def test_server_on_linux_without_pkexec_is_refused_with_the_sentence(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sudo present (even passwordless) is NOT a prompt for a server: no TTY.
    monkeypatch.setattr(elevate, "_PKEXEC_CANDIDATES_OVERRIDE", ())
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: "/usr/bin/sudo" if name == "sudo" else None
    )

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)

    assert result["outcome"] == "no_password_prompt"
    assert host.calls == [], "the stub gesture cannot bypass the precheck"
    assert not trust.staging_path(host.root).exists()
    sentence = authority_setup.outcome_message("no_password_prompt").envelope(locale="en")["text"]
    assert sentence == (
        "This computer can't show a password prompt, so approvals can't be turned on from here."
    )
    # The TUI owns a terminal, so the same host may use sudo there.
    tui = authority_setup.setup_authority(
        host.root, actor="tui", terminal_sudo=True, runner=host.approve
    )
    assert tui["outcome"] == "on"


def test_run_elevated_refuses_terminal_sudo_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: "/usr/bin/sudo" if name == "sudo" else None
    )
    launched: list[Any] = []
    monkeypatch.setattr(elevate.subprocess, "run", lambda *a, **k: launched.append(a))

    result = elevate.run_elevated([["sudo", "true"]], windows=False, terminal_sudo=False)

    assert result.exit_code == 2 and result.refusal_code == elevate.REFUSAL_NO_PASSWORD_PROMPT
    assert launched == []


def test_no_stored_password_parameter_exists_on_this_path() -> None:
    params = inspect.signature(authority_setup.setup_authority).parameters
    assert "sudo_secret" not in params
    with pytest.raises(ValueError):
        authority_setup.setup_authority(
            Path("/nonexistent"), actor="desktop", admin_route="ask-once"  # type: ignore[arg-type]
        )


def test_cancelled_prompt_leaves_finish_setup_and_nothing_staged(host: SimpleNamespace) -> None:
    result = authority_setup.setup_authority(
        host.root, actor="desktop", runner=lambda command: ElevationResult(126, cancelled=True)
    )

    assert result["outcome"] == "cancelled"
    assert result["receipts"][-1] == {**result["receipts"][-1], "step": "installed", "ok": False}
    assert not trust.staging_path(host.root).exists()
    state = authority_setup.describe_state(host.root)
    assert state["state"] == "finish_setup" and state["can_prompt"] is True
    # Finish setup: the key from the cancelled attempt is reused, not replaced.
    key_id = result["receipts"][2]["key_id"]
    done = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)
    assert done["outcome"] == "on" and done["receipts"][2]["key_id"] == key_id
    assert authority_setup.describe_state(host.root)["state"] == "on"


def test_a_second_attempt_while_one_runs_is_refused(host: SimpleNamespace) -> None:
    entered, release = threading.Event(), threading.Event()
    outcome: dict[str, Any] = {}

    def slow(command: list[list[str]]) -> ElevationResult:
        entered.set()
        release.wait(5)
        return host.approve(command)

    worker = threading.Thread(
        target=lambda: outcome.update(
            authority_setup.setup_authority(host.root, actor="desktop", runner=slow)
        )
    )
    worker.start()
    assert entered.wait(5)
    assert authority_setup.in_progress(host.root) is True
    with pytest.raises(authority_setup.SetupInProgress):
        authority_setup.setup_authority(host.root, actor="tui", runner=host.approve)
    release.set()
    worker.join(5)
    assert outcome["outcome"] == "on"
    assert authority_setup.in_progress(host.root) is False


def test_os_prompt_reason_follows_the_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    command = [["sudo", "install", "-m", "0644", "/a", "/b"]]
    reasons = {purpose: elevate._macos_prompt(command, purpose)[2] for purpose in elevate.PURPOSES}
    assert "to confirm a phone on this computer." in reasons["confirm_phone"]
    assert "to remove a phone from this computer." in reasons["remove_phone"]
    # D7: one plain sentence, no doubled "approval"/"on" inside the OS dialog.
    assert reasons["phone_approvals"].endswith(
        'with prompt "Local Operator wants to turn on phone approvals for this computer."'
    )
    # The default keeps every existing caller's wording.
    assert elevate._macos_prompt(command)[2] == reasons["confirm_phone"]
    with pytest.raises(ValueError):
        elevate._macos_prompt(command, "anything a client sends")
    assert elevate.confirm_card_copy(purpose="phone_approvals")[0] == "Turn on phone approvals"


def test_a_translated_reason_is_escaped_for_applescript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elevate, "_purpose_copy", lambda purpose: ('say "hi" \\ end', "t"))
    script = elevate._macos_prompt([["true"]], "phone_approvals")[2]
    assert script.endswith('with prompt "say \\"hi\\" \\\\ end"')


# --- Round-1 remediation (#3950): R1-1..R1-5, Q1 -------------------------------------

_REAL_INSTALL = shutil.which("install")
_REAL_SHA256SUM = shutil.which("sha256sum")
needs_posix_install = pytest.mark.skipif(
    sys.platform == "win32" or _REAL_INSTALL is None or _REAL_SHA256SUM is None,
    reason="runs the real POSIX install script (install + sha256sum)",
)


def _real_script(tmp_path: Path) -> Any:
    """Run the PRIVILEGED commands for real, minus root.

    ``sudo`` is dropped and ``install`` is wrapped to drop ``-o``/``-g`` (a non-root
    user cannot chown); everything else \u2014 the shell script, its hash checks, its
    compare-and-swap, the rename \u2014 is the shipped text, run by ``sh``.
    """
    shim_dir = tmp_path / "install-shim"
    shim_dir.mkdir(exist_ok=True)
    wrapper = shim_dir / "install"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args, out, i = sys.argv[1:], [], 0\n"
        "while i < len(args):\n"
        "    if args[i] in ('-o', '-g'):\n"
        "        i += 2\n"
        "        continue\n"
        "    out.append(args[i]); i += 1\n"
        f"os.execv({_REAL_INSTALL!r}, ['install', *out])\n"
    )
    wrapper.chmod(0o755)

    def run(command: list[list[str]]) -> ElevationResult:
        import os
        import subprocess

        env = {**os.environ, "PATH": f"{shim_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
        for argv in command:
            assert argv[0] == "sudo"
            done = subprocess.run(argv[1:], env=env, check=False, capture_output=True, text=True)
            if done.returncode != 0:
                return ElevationResult(done.returncode, diagnosis=done.stderr)
        return ElevationResult(0)

    return run


def _other_anchor_bytes(tmp_path: Path, *, admission: str = "listed") -> bytes:
    """A well-formed anchor for a key this attempt did NOT create (the attacker's)."""
    from local_operator.operator import create_key
    from local_operator.operator.sign import anchor_for_handle

    other = tmp_path / "attacker-config"
    other.mkdir(exist_ok=True)
    handle = create_key(config_root=other)
    from dataclasses import replace

    return trust.anchor_bytes(replace(anchor_for_handle(handle), admission=admission))


@needs_posix_install
def test_r1_1_real_install_script_lands_exactly_the_in_memory_anchor(
    host: SimpleNamespace, tmp_path: Path
) -> None:
    calls: list[list[list[str]]] = []
    real = _real_script(tmp_path)

    def runner(command: list[list[str]]) -> ElevationResult:
        calls.append(command)
        return real(command)

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=runner)

    assert result["outcome"] == "on", result
    # R1-4: a first install pins "no anchor yet" as the target's expected state.
    assert calls[0][1][-1] == trust.EXPECT_ABSENT
    # R1-1: the pinned digest is sha256 of the IN-MEMORY bytes, and the verified
    # receipt records what was READ back from the installed anchor.
    import hashlib

    generated = result["receipts"][2]
    installed = trust.anchor_path().read_bytes()
    assert generated["sha256"] == hashlib.sha256(installed).hexdigest() == calls[0][1][-3]
    verified = result["receipts"][-1]
    assert verified["key_id"] == generated["key_id"] and verified["admission"] == "listed"


def test_r1_1_a_fifo_at_the_staging_path_is_refused_with_no_prompt(
    host: SimpleNamespace,
) -> None:
    import os

    staged = trust.staging_path(host.root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(staged)
    outcome: dict[str, Any] = {}
    worker = threading.Thread(
        target=lambda: outcome.update(
            authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)
        ),
        daemon=True,
    )
    worker.start()
    worker.join(10)

    assert not worker.is_alive(), "a FIFO must never block the attempt"
    assert outcome["outcome"] == "unsafe_staging" and outcome["ok"] is False
    assert host.calls == [], "no OS prompt for a staging path we did not write"
    assert not trust.anchor_path().exists()


def test_r1_1_a_symlink_at_the_staging_path_is_refused_and_not_followed(
    host: SimpleNamespace, tmp_path: Path
) -> None:
    staged = trust.staging_path(host.root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(b"untouched")
    staged.symlink_to(elsewhere)

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)

    assert result["outcome"] == "unsafe_staging" and host.calls == []
    assert elsewhere.read_bytes() == b"untouched"


def test_r1_1_staged_bytes_swapped_before_the_prompt_are_refused_with_no_prompt(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_stage = authority_setup._stage_exact
    attacker = _other_anchor_bytes(tmp_path)

    def stage_then_swap(path: Path, expected: bytes) -> None:
        real_stage(path, expected)
        path.write_bytes(attacker)  # a same-user process wins the race here

    monkeypatch.setattr(authority_setup, "_stage_exact", stage_then_swap)

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)

    assert result["outcome"] == "unsafe_staging" and host.calls == []
    assert not trust.anchor_path().exists()


@needs_posix_install
def test_r1_1_staged_bytes_swapped_during_the_prompt_fail_the_install_hash(
    host: SimpleNamespace, tmp_path: Path
) -> None:
    attacker = _other_anchor_bytes(tmp_path)
    real = _real_script(tmp_path)

    def swap_then_install(command: list[list[str]]) -> ElevationResult:
        trust.staging_path(host.root).write_bytes(attacker)  # while the prompt is open
        return real(command)

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=swap_then_install)

    assert result["outcome"] == "failed" and result["ok"] is False
    assert not trust.anchor_path().exists(), "the install refused the swapped bytes"


@pytest.mark.parametrize("tamper", ["attacker_key", "open_admission"])
def test_r1_1_a_tampered_installed_anchor_is_not_verified(
    host: SimpleNamespace, tmp_path: Path, tamper: str
) -> None:
    def install_something_else(command: list[list[str]]) -> ElevationResult:
        if tamper == "attacker_key":
            trust.anchor_path().write_bytes(_other_anchor_bytes(tmp_path))
        else:
            # Our own key, but with admission opened: every phone admitted unlisted.
            staged = json.loads(trust.staging_path(host.root).read_bytes())
            staged.pop("admission", None)
            trust.anchor_path().write_bytes(json.dumps(staged).encode())
        return ElevationResult(0)

    result = authority_setup.setup_authority(
        host.root, actor="desktop", runner=install_something_else
    )

    assert result["outcome"] == "not_verified" and result["ok"] is False
    assert result["receipts"][-1]["step"] == "verified"
    assert result["receipts"][-1]["ok"] is False


def test_r1_2_an_elevated_windows_token_is_refused_not_shortcut(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    monkeypatch.setattr(authority_setup, "_windows_token_elevated", lambda: True)

    result = authority_setup.setup_authority(host.root, actor="desktop", runner=host.approve)

    assert result["outcome"] == "privileged_process"
    assert host.calls == [] and not trust.staging_path(host.root).exists()


def test_r1_2_windows_token_check_reads_isuseranadmin_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    for answer, elevated in ((1, True), (0, False)):
        monkeypatch.setattr(
            ctypes,
            "windll",
            SimpleNamespace(shell32=SimpleNamespace(IsUserAnAdmin=lambda answer=answer: answer)),
            raising=False,
        )
        assert authority_setup._windows_token_elevated() is elevated
    monkeypatch.delattr(ctypes, "windll", raising=False)
    assert authority_setup._windows_token_elevated() is True, "unknown reads as elevated"


@pytest.mark.skipif(sys.platform == "win32", reason="flock is the POSIX cross-process half")
def test_r1_3_a_refused_attempt_keeps_the_running_attempts_marker(
    host: SimpleNamespace,
) -> None:
    import fcntl
    import os

    operator = host.root / "operator"
    operator.mkdir(parents=True, exist_ok=True)
    marker = operator / "authority-setup.running"
    # Another process's attempt: it holds the lock and wrote its own pid.
    holder = open(operator / "authority-setup.lock", "a+")  # noqa: SIM115
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker.write_text(str(os.getppid()))

        with pytest.raises(authority_setup.SetupInProgress):
            authority_setup.setup_authority(host.root, actor="tui", runner=host.approve)

        assert marker.read_text() == str(os.getppid()), "the refused attempt left it alone"
        assert authority_setup.in_progress(host.root) is True
    finally:
        holder.close()
    # Q1: the refused attempt is audited, as its own one-row attempt.
    rows = _audit(host.root)
    assert rows[-1]["code"] == "in_progress" and rows[-1]["actor"] == "tui"
    assert host.calls == []


@needs_posix_install
def test_r1_4_an_anchor_landing_during_the_prompt_is_never_overwritten(
    host: SimpleNamespace, tmp_path: Path
) -> None:
    other = _other_anchor_bytes(tmp_path)
    real = _real_script(tmp_path)

    def another_setup_lands_first(command: list[list[str]]) -> ElevationResult:
        trust.anchor_path().write_bytes(other)  # e.g. `lop operator init` meanwhile
        return real(command)

    result = authority_setup.setup_authority(
        host.root, actor="desktop", runner=another_setup_lands_first
    )

    assert result["outcome"] == "changed_elsewhere" and result["ok"] is False
    assert trust.anchor_path().read_bytes() == other, "the other anchor was kept"
    sentence = authority_setup.outcome_message("changed_elsewhere").envelope(locale="en")["text"]
    assert "nothing was replaced" in sentence


def test_r1_5_pkexec_is_resolved_from_fixed_paths_never_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import subprocess

    assert elevate._TRUSTED_PKEXEC_PATHS == ("/usr/bin/pkexec", "/bin/pkexec")
    planted = tmp_path / "user-bin" / "pkexec"
    planted.parent.mkdir()
    planted.write_text("#!/bin/sh\nexit 0\n")
    planted.chmod(0o755)
    monkeypatch.setenv("PATH", f"{planted.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    trusted = tmp_path / "system-bin" / "pkexec"
    monkeypatch.setattr(elevate, "_PKEXEC_CANDIDATES_OVERRIDE", None)
    monkeypatch.setattr(elevate, "_TRUSTED_PKEXEC_PATHS", (str(trusted),))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(trust, "_on_windows", lambda: False)

    assert shutil.which("pkexec") == str(planted)
    assert elevate.pkexec_path() is None, "a PATH pkexec is no pkexec"
    assert elevate.gui_prompt_refusal(windows=False) is not None

    trusted.parent.mkdir()
    trusted.write_text("#!/bin/sh\nexit 0\n")
    trusted.chmod(0o755)
    launched: list[list[str]] = []
    monkeypatch.setattr(
        elevate.subprocess,
        "run",
        lambda argv, **k: launched.append(argv) or subprocess.CompletedProcess(argv, 126),
    )
    result = elevate.run_elevated([["sudo", "true"]], windows=False, terminal_sudo=False)

    assert launched[0][0] == str(trusted), "the absolute trusted path is what runs"
    assert result.cancelled is True


def test_d4_d7_copy_names_the_feature_not_a_key() -> None:
    for outcome in ("cannot_install", "key_failed", "not_verified"):
        text = authority_setup.outcome_message(outcome).envelope(locale="en")["text"]
        assert "key" not in text.casefold(), (outcome, text)
    assert authority_setup.file_only_message().envelope(locale="en")["text"] == (
        "Note: on this computer itself, approvals won't ask for a fingerprint or password."
    )
