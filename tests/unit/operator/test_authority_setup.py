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
    # A Linux host WITH pkexec: the server form may prompt. Cells that need a host
    # without it override `which`.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(trust, "_on_windows", lambda: False)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **k: (
            f"/usr/bin/{name}" if name in ("pkexec", "sudo") else real_which(name)
        ),
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
    assert "to turn on phone approvals on this computer." in reasons["phone_approvals"]
    # The default keeps every existing caller's wording.
    assert elevate._macos_prompt(command)[2] == reasons["confirm_phone"]
    with pytest.raises(ValueError):
        elevate._macos_prompt(command, "anything a client sends")
    assert elevate.confirm_card_copy(purpose="phone_approvals")[0] == "Turn on phone approvals"


def test_a_translated_reason_is_escaped_for_applescript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elevate, "_purpose_copy", lambda purpose: ('say "hi" \\ end', "t"))
    script = elevate._macos_prompt([["true"]], "phone_approvals")[2]
    assert script.endswith('with prompt "say \\"hi\\" \\\\ end"')
