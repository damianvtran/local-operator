"""The node-browser harness's launch guards (``scripts/e2e/node_browser_harness.py``).

The harness writes operator keys, Chrome policy and profiles. These cells pin
the three refusals that keep it off the operator's real node: it runs only with
``--harness``, only in a fresh subdirectory of the session scratchpad (never in
or above the real config root), and the inner run aborts when the real managed
policy is visible (i.e. it is not inside its mount namespace). They also pin
the hidden ``--inner``/``--operator`` modes to the same refusals plus netns and
run-root containment (writing nothing on refusal), the interrupt cleanup (exit
handlers, the named scope stopped, the /tmp dir removed), and the negative-
control rows' specific refusals. No Chrome, daemon or namespace is started
here: the guards are pure functions on purpose, and the rest is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e import node_browser_harness as harness


@pytest.fixture()
def layout(tmp_path: Path) -> dict[str, Path]:
    real = tmp_path / "home" / ".local-operator"
    pad = real / "sessions" / "abc123" / "scratchpad"
    pad.mkdir(parents=True)
    return {"real": real, "pad": pad}


def test_no_harness_flag_is_refused(layout: dict[str, Path]) -> None:
    reason = harness.refusal(
        harness_flag=False,
        root=layout["pad"] / "run",
        scratchpad=str(layout["pad"]),
        real_config=layout["real"],
    )
    assert reason is not None and "--harness" in reason


def test_the_real_config_root_is_refused(layout: dict[str, Path]) -> None:
    # Even with the scratchpad pointed at the real root, the root itself is refused.
    reason = harness.refusal(
        harness_flag=True,
        root=layout["real"],
        scratchpad=str(layout["real"]),
        real_config=layout["real"],
    )
    assert reason is not None and "real config root" in reason


def test_a_root_outside_the_scratchpad_is_refused(layout: dict[str, Path], tmp_path: Path) -> None:
    for root in (tmp_path / "elsewhere", layout["real"] / "sessions", layout["real"]):
        reason = harness.refusal(
            harness_flag=True,
            root=root,
            scratchpad=str(layout["pad"]),
            real_config=layout["real"],
        )
        assert reason is not None and "not inside the scratchpad" in reason, root


def test_a_missing_scratchpad_and_the_bare_scratchpad_are_refused(layout: dict[str, Path]) -> None:
    assert "not set" in str(
        harness.refusal(
            harness_flag=True,
            root=layout["pad"] / "run",
            scratchpad=None,
            real_config=layout["real"],
        )
    )
    assert "fresh subdirectory" in str(
        harness.refusal(
            harness_flag=True,
            root=layout["pad"],
            scratchpad=str(layout["pad"]),
            real_config=layout["real"],
        )
    )


def test_a_fresh_scratchpad_subdirectory_is_accepted(layout: dict[str, Path]) -> None:
    assert (
        harness.refusal(
            harness_flag=True,
            root=layout["pad"] / "node-browser-harness-1",
            scratchpad=str(layout["pad"]),
            real_config=layout["real"],
        )
        is None
    )


@pytest.mark.parametrize("subtree", ["browser", "networks", "sessions/live", "notifications.db"])
def test_misset_scratchpad_cannot_bless_real_stores(layout: dict[str, Path], subtree: str) -> None:
    pad = layout["real"] / subtree
    reason = harness.refusal(
        harness_flag=True, root=pad / "run", scratchpad=str(pad), real_config=layout["real"]
    )
    assert reason is not None and "inside the real config root" in reason


def test_mesh_preflight_rejects_drive_member_before_signed_browser_dispatch() -> None:
    from local_operator.network.types import (
        MemberRecord,
        NetworkRecord,
        capabilities_for_role,
    )
    from scripts.e2e.node_browser_mesh import require_operator_member

    record = NetworkRecord(network_id="n_synthetic", name="synthetic")
    member = MemberRecord(
        device_id="d_synthetic",
        name="operator",
        role="drive",
        capabilities=sorted(capabilities_for_role("drive")),
    )
    record.members = [member]
    with pytest.raises(RuntimeError, match="approve"):
        require_operator_member(record, member.device_id)
    member.capabilities = sorted(capabilities_for_role("admin"))
    assert "approve" in require_operator_member(record, member.device_id)
    record.members = []
    with pytest.raises(RuntimeError, match="not an active mesh member"):
        require_operator_member(record, member.device_id)


def test_remote_notice_assertion_names_the_node_session_and_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.node_browser_mesh import MeshRig

    rig = MeshRig(tmp_path / "node", tmp_path / "operator")
    rig.node_id = "d_synthetic_node"
    expected = "remote-notice:d_synthetic_node:synthetic_sid:remote-ask:a-synthetic"
    items = [
        {"session_id": "synthetic_sid", "key": expected},
        {"session_id": "other_sid", "key": expected},
        {"session_id": "synthetic_sid", "key": expected.replace("d_synthetic_node", "d_other")},
    ]
    monkeypatch.setattr(rig, "observe", lambda: items)
    result = rig.notice("synthetic_sid", "a-synthetic")
    assert result == {"expected_key": expected, "count": 1, "items": [items[0]]}
    items.append(dict(items[0]))
    assert rig.notice("synthetic_sid", "a-synthetic")["count"] == 2


def test_a_visible_real_policy_aborts_the_inner_run(tmp_path: Path) -> None:
    policy = tmp_path / "local-operator.json"
    assert harness.namespace_refusal(policy) is None
    policy.write_text("{}")
    reason = harness.namespace_refusal(policy)
    assert reason is not None and "NOT inside the harness mount namespace" in reason


def test_main_without_the_flag_exits_2_and_starts_nothing(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    monkeypatch.setattr(harness, "outer", lambda _a: pytest.fail("outer must not run"))
    assert harness.main(["--root", str(layout["pad"] / "run")]) == 2
    assert "--harness" in capsys.readouterr().err


def test_a_loading_frame_does_not_count_as_a_rendered_theme(tmp_path: Path) -> None:
    """Run 6's false pass: a spinner frame differs by hash but shows no story.

    The light/dark row now also needs each frame to carry content. A frame that
    is background plus a small spinner sits far below the bound; one with a
    panel of text-like content sits above it.
    """
    from PIL import Image, ImageDraw

    spinner = Image.new("RGB", (780, 493), (255, 255, 255))
    ImageDraw.Draw(spinner).ellipse((375, 231, 405, 261), outline=(200, 200, 200), width=2)
    spinner_path = tmp_path / "spinner.png"
    spinner.save(spinner_path)

    story = Image.new("RGB", (780, 493), (242, 237, 227))
    draw = ImageDraw.Draw(story)
    draw.rectangle((16, 16, 336, 493), fill=(248, 245, 238))
    for top in range(40, 480, 20):
        draw.rectangle((30, top, 300, top + 8), fill=(40, 40, 40))
    story_path = tmp_path / "story.png"
    story.save(story_path)

    assert harness.rendered_fraction(spinner_path) < harness.RENDERED_MIN_FRACTION
    assert harness.rendered_fraction(story_path) >= harness.RENDERED_MIN_FRACTION


def _frame(path: Path, background: tuple[int, int, int], *, content: bool) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (780, 493), background)
    if content:
        draw = ImageDraw.Draw(image)
        for top in range(40, 480, 20):
            draw.rectangle((30, top, 300, top + 8), fill=(128, 64, 200))
    image.save(path)
    return path


def _request_with_tool_text(text: str) -> object:
    from types import SimpleNamespace

    block = SimpleNamespace(text=text)
    return SimpleNamespace(messages=[SimpleNamespace(role="tool", content=[block])])


def _tool_call_name_and_args(turn: list[object]) -> tuple[str, dict[str, object]]:
    import json

    call = next(event for event in turn if getattr(event, "name", None))
    return str(getattr(call, "id")), json.loads(getattr(call, "argument_delta"))


def test_a_capture_snapshots_then_screenshots_once_the_story_root_is_there(
    tmp_path: Path,
) -> None:
    """Run 6's spinner fix: no screenshot counts until the story has rendered."""
    import asyncio

    shot = tmp_path / "dark.png"
    capture = harness.Capture("dark", str(shot))

    async def drive() -> list[str]:
        ids = []
        turn = await capture.next_turn(_request_with_tool_text(""))
        assert turn is not None
        ids.append(_tool_call_name_and_args(turn)[0])
        # The snapshot found the story root: the next call is the screenshot.
        turn = await capture.next_turn(_request_with_tool_text("heading 'Approvals'"))
        assert turn is not None
        call_id, args = _tool_call_name_and_args(turn)
        ids.append(call_id)
        assert args == {"action": "screenshot", "path": str(shot)}
        _frame(shot, (20, 22, 28), content=True)
        assert await capture.next_turn(_request_with_tool_text("saved")) is None
        return ids

    ids = asyncio.run(drive())
    assert ids[0].startswith("snap_dark_") and ids[1].startswith("shot_dark_")
    assert capture.outcome["rendered"] is True
    assert capture.outcome["background_luminance"] < harness.DARK_MAX_LUMINANCE


def test_a_capture_that_never_renders_its_theme_records_not_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A light page (or a spinner) captured for the DARK story never counts."""
    import asyncio

    monkeypatch.setattr(harness, "RENDER_WAIT_S", 0.0)
    shot = _frame(tmp_path / "dark.png", (242, 237, 227), content=True)
    capture = harness.Capture("dark", str(shot))
    capture._phase = "shot"  # a screenshot was just taken
    capture._started = 1.0  # long ago: the wait is spent

    assert asyncio.run(capture.next_turn(_request_with_tool_text("saved"))) is None
    assert capture.outcome["rendered"] is False
    assert "not rendered" in capture.outcome["error"]
    assert capture.outcome["background_luminance"] > harness.LIGHT_MIN_LUMINANCE


def test_the_withdrawal_is_proven_by_the_extensions_cancelled_reply() -> None:
    """Run 7's over-strict gate: ``cancelled`` alone proves the entry is gone."""
    run7 = {
        "error": False,
        "details": {"origin": "http://127.0.0.1:57651", "state": "cancelled", "pending_count": 0},
    }
    assert harness.withdrawal_confirmed(run7)
    # Run 6's identity-less call matched nothing: the extension says "none".
    assert not harness.withdrawal_confirmed({"error": False, "details": {"state": "none"}})
    assert not harness.withdrawal_confirmed({"error": True, "details": {"state": "cancelled"}})


# ---------------------------------------------------------------------------
# Hidden modes (agent review round 1 R1-1 / QA Q2): ``--inner`` and
# ``--operator`` are reachable from any shell, so they re-apply every launcher
# refusal plus namespace and containment checks, and write nothing on refusal.
# ---------------------------------------------------------------------------

HOST_NETNS = 4026531833
HARNESS_NETNS = 4026532796


@pytest.fixture()
def harness_netns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pretend to be inside the harness namespace: private netns, only lo, mask on."""
    monkeypatch.setattr(harness, "netns_inode", lambda: HARNESS_NETNS)
    monkeypatch.setattr(harness, "netns_interfaces", lambda: ["lo"])
    monkeypatch.setattr(harness, "REAL_POLICY", tmp_path / "masked" / "local-operator.json")


def _hidden_argv(mode: str, root: Path, node_root: Path, ask: Path) -> list[str]:
    argv = [mode, "--harness", "--root", str(root), "--host-netns", str(HOST_NETNS)]
    argv += ["--storybook-dir", str(root / "sb")]
    if mode == "--operator":
        argv += ["--node-root", str(node_root), "--session", "s", "--ask-file", str(ask)]
        argv += ["--harness-netns", str(HARNESS_NETNS)]
    return argv


@pytest.fixture()
def no_mode_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _never(_args: object) -> int:
        pytest.fail("a refused hidden mode must not start")

    monkeypatch.setattr(harness, "inner", _never)
    monkeypatch.setattr(harness, "operator_role", _never)


@pytest.mark.parametrize("mode", ["--inner", "--operator"])
def test_a_hidden_mode_without_the_flag_is_refused_and_writes_nothing(
    mode: str,
    layout: dict[str, Path],
    harness_netns: None,
    no_mode_runs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    root = layout["pad"] / "run"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "operator"))
    argv = [
        a for a in _hidden_argv(mode, root, root / "node", root / "ask.json") if a != "--harness"
    ]
    assert harness.main(argv) == 2
    assert "--harness" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("mode", ["--inner", "--operator"])
def test_a_hidden_mode_with_a_root_outside_the_scratchpad_is_refused(
    mode: str,
    layout: dict[str, Path],
    tmp_path: Path,
    harness_netns: None,
    no_mode_runs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # QA Q2's repro: an unset scratchpad and an outside root used to reach the run.
    root = tmp_path / "x"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "operator"))
    argv = _hidden_argv(mode, root, root / "node", root / "ask.json")
    monkeypatch.delenv("LOCAL_OPERATOR_SCRATCHPAD", raising=False)
    assert harness.main(argv) == 2
    assert "LOCAL_OPERATOR_SCRATCHPAD is not set" in capsys.readouterr().err
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    assert harness.main(argv) == 2
    assert "not inside the scratchpad" in capsys.readouterr().err
    assert not (root / "results.json").exists()


def test_the_operator_role_refuses_a_node_root_or_key_root_outside_the_run_root(
    layout: dict[str, Path],
    tmp_path: Path,
    harness_netns: None,
    no_mode_runs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    root = layout["pad"] / "run"
    root.mkdir()
    ask = root / "ask.json"
    # The review's impact case: --node-root / the key root at the real config.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "operator"))
    assert harness.main(_hidden_argv("--operator", root, layout["real"], ask)) == 2
    assert "--node-root" in capsys.readouterr().err
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(layout["real"]))
    assert harness.main(_hidden_argv("--operator", root, root / "node", ask)) == 2
    assert "LOCAL_OPERATOR_CONFIG_DIR" in capsys.readouterr().err
    # A symlink inside the run root resolves outside it and is refused too.
    (root / "node").symlink_to(layout["real"], target_is_directory=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "operator"))
    assert harness.main(_hidden_argv("--operator", root, root / "node", ask)) == 2
    assert "is not under the run root" in capsys.readouterr().err


def test_the_inner_run_refuses_a_symlinked_config_root(
    layout: dict[str, Path],
    harness_netns: None,
    no_mode_runs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    root = layout["pad"] / "run"
    root.mkdir()
    (root / "operator").symlink_to(layout["real"], target_is_directory=True)
    assert harness.main(_hidden_argv("--inner", root, root / "node", root / "ask.json")) == 2
    assert "operator" in capsys.readouterr().err
    assert not (root / "results.json").exists()


@pytest.mark.parametrize("mode", ["--inner", "--operator"])
def test_a_hidden_mode_in_the_host_netns_is_refused(
    mode: str,
    layout: dict[str, Path],
    harness_netns: None,
    no_mode_runs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    root = layout["pad"] / "run"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "operator"))
    argv = _hidden_argv(mode, root, root / "node", root / "ask.json")
    monkeypatch.setattr(harness, "netns_inode", lambda: HOST_NETNS)
    assert harness.main(argv) == 2
    assert "shares the launcher's network namespace" in capsys.readouterr().err
    # A lying --host-netns does not help: the host netns has a real interface.
    monkeypatch.setattr(harness, "netns_interfaces", lambda: ["lo", "enp39s0"])
    lying = ["1" if a == str(HOST_NETNS) else a for a in argv]
    assert harness.main(lying) == 2
    assert "holds only lo" in capsys.readouterr().err
    assert not root.exists()


def test_the_netns_refusal_needs_every_check() -> None:
    ok = {"host_netns": HOST_NETNS, "own_netns": HARNESS_NETNS, "interfaces": ["lo"]}
    assert harness.netns_refusal(**ok) is None
    assert harness.netns_refusal(**ok, harness_netns=HARNESS_NETNS) is None
    assert "missing --host-netns" in str(harness.netns_refusal(**{**ok, "host_netns": None}))
    assert "shares" in str(harness.netns_refusal(**{**ok, "own_netns": HOST_NETNS}))
    assert "holds only lo" in str(harness.netns_refusal(**{**ok, "interfaces": ["lo", "eth0"]}))
    assert "operator role" in str(harness.netns_refusal(**ok, harness_netns=HARNESS_NETNS + 1))


def test_operator_relay_root_cannot_be_a_symlink_to_live_config(
    layout: dict[str, Path],
    harness_netns: None,
    no_mode_runs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = layout["pad"] / "run"
    root.mkdir()
    (root / "operator").symlink_to(layout["real"], target_is_directory=True)
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "wrong-key-operator"))
    assert harness.main(_hidden_argv("--operator", root, root / "node", root / "ask.json")) == 2
    assert "operator relay root" in capsys.readouterr().err
    assert not (root / "ask.json").exists()


def test_a_hidden_mode_passing_every_check_runs(
    layout: dict[str, Path],
    harness_netns: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control: the checks above refuse for their stated reason only."""
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    root = layout["pad"] / "run"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "operator"))
    ran: list[str] = []

    async def _ran(_args: object) -> int:
        ran.append("ran")
        return 0

    monkeypatch.setattr(harness, "operator_role", _ran)
    monkeypatch.setattr(harness, "install_exit_handlers", lambda: None)
    monkeypatch.setattr(harness, "inner", _ran)
    for mode in ("--operator", "--inner"):
        assert harness.main(_hidden_argv(mode, root, root / "node", root / "ask.json")) == 0
    assert ran == ["ran", "ran"]


def test_the_real_config_root_comes_from_passwd_not_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = harness.real_config_root()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert harness.real_config_root() == before
    assert tmp_path not in harness.real_config_root().parents


def test_a_run_without_a_storybook_dir_is_refused(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    monkeypatch.setattr(harness, "outer", lambda _a: pytest.fail("outer must not run"))
    assert harness.main(["--harness", "--root", str(layout["pad"] / "run")]) == 2
    assert "--storybook-dir" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Interrupt cleanup (R1-2): signals unwind, and the launcher stops its scope.
# ---------------------------------------------------------------------------


def test_exit_handlers_turn_sigterm_and_sigint_into_system_exit() -> None:
    import signal

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        harness.install_exit_handlers()
        for sig in saved:
            handler = signal.getsignal(sig)
            assert callable(handler) and handler is harness._exit_on_signal
            with pytest.raises(SystemExit) as raised:
                handler(sig, None)
            assert raised.value.code == 128 + sig
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_an_interrupted_launcher_stops_its_named_scope_and_removes_the_tmp_dir(
    layout: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    launched: list[list[str]] = []
    monkeypatch.setattr(harness, "host_snapshot", lambda: {})
    monkeypatch.setattr(harness, "CHROME_TMP_BASE", tmp_path / "tmpbase")
    (tmp_path / "tmpbase").mkdir()
    kept = tmp_path / "tmpbase" / "nbh-other-run"
    kept.mkdir()

    killed: list[int] = []
    monkeypatch.setattr(harness.os, "killpg", lambda pid, _sig: killed.append(pid))

    class _Interrupted:
        """A launch whose wait is interrupted by SIGTERM (as SystemExit)."""

        pid = 4242

        def __init__(self, argv: list[str], **_kw: object) -> None:
            launched.append(argv)
            stamp = next(a for a in argv if a.startswith("--unit="))[len("--unit=nbh-") :]
            # The inner run's Chrome TMPDIR, as left behind by a killed inner.
            (tmp_path / "tmpbase" / f"nbh-{stamp}-leaked").mkdir()
            self.running = True
            self.waits = 0

        def wait(self) -> int:
            self.waits += 1
            if self.waits == 1:
                raise SystemExit(128 + 15)  # SIGTERM arrives mid-wait
            self.running = False  # the finally's wait, after killpg
            return -9

        def poll(self) -> int | None:
            return None if self.running else -9

    monkeypatch.setattr(harness.subprocess, "Popen", _Interrupted)
    monkeypatch.setattr(harness.subprocess, "run", lambda argv, **_kw: calls.append(argv))
    args = harness.parse(
        ["--harness", "--root", str(layout["pad"] / "run"), "--storybook-dir", "x"]
    )
    with pytest.raises(SystemExit):
        harness.outer(args)

    unit_arg = next(a for a in launched[0] if a.startswith("--unit="))
    unit = unit_arg[len("--unit=") :]
    assert unit.startswith("nbh-") and harness.STAMP_RE.match(unit[len("nbh-") :])
    assert ["systemctl", "--user", "stop", f"{unit}.scope"] in calls
    assert ["systemctl", "--user", "reset-failed", f"{unit}.scope"] in calls
    assert f"--stamp {unit[len('nbh-'):]}" in launched[0][-1]
    assert killed == [4242]
    assert sorted(p.name for p in (tmp_path / "tmpbase").iterdir()) == ["nbh-other-run"]


def test_the_scope_command_is_named_capped_and_backstopped() -> None:
    command = harness.scope_command("nbh-20261116T000000-abc123", "true")
    assert command[:5] == [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "--unit=nbh-20261116T000000-abc123",
    ]
    assert "MemoryMax=1536M" in command and "MemorySwapMax=0" in command
    assert command[command.index("timeout") + 1] == "600"
    assert command[-4:] == ["-Urmn", "sh", "-c", "true"]


# ---------------------------------------------------------------------------
# Negative-control rows (R1-3): the SPECIFIC refusal, never "any failure".
# ---------------------------------------------------------------------------


def _rendered(key: str) -> str:
    return harness._site_refusal(key)


def test_authority_refusal_needs_the_old_signer_sentence() -> None:
    # QA round 1's actual wrong-key and plain verdicts.
    actual = {
        "connected": True,
        "ok": False,
        "error": f"RuntimeError: {_rendered('refused_old_signer')}",
    }
    assert harness.authority_refused(actual)
    assert not harness.authority_refused({**actual, "connected": False})
    assert not harness.authority_refused({**actual, "ok": True})
    assert not harness.authority_refused(
        {"connected": True, "ok": False, "error": "TimeoutError: "}
    )
    assert not harness.authority_refused(
        {"connected": False, "ok": False, "error": "operator child rc=1: Traceback"}
    )


def test_unsigned_refusal_needs_the_unsigned_decision_code() -> None:
    actual = {"ok": False, "error": _rendered("refused_unsigned"), "code": "unsigned_decision"}
    assert harness.unsigned_refused(actual)
    assert not harness.unsigned_refused({**actual, "code": "bridge_unreachable"})
    assert not harness.unsigned_refused({"ok": False, "error": "TimeoutError"})


def test_member_withdrawal_requires_the_real_http_queue_refusal() -> None:
    from local_operator.asks.render import refusal_copy

    actual = {
        "connected": True,
        "ok": False,
        "http_status": 409,
        "surface": "desktop-http-remote-session",
        "error": refusal_copy({"status": "withdrawn"}),
    }
    assert harness.withdrawn_refused(actual)
    assert not harness.withdrawn_refused({**actual, "http_status": 503})
    assert not harness.withdrawn_refused({**actual, "surface": "direct-attach"})
    assert not harness.withdrawn_refused({**actual, "error": _rendered("refused_old_signer")})
    assert not harness.withdrawn_refused({**actual, "error": "TimeoutError"})


def test_expired_refusal_needs_the_expiry_sentence() -> None:
    actual = {
        "connected": True,
        "ok": False,
        "error": f"RuntimeError: {_rendered('refused_expired')}",
    }
    assert harness.expired_refused(actual)
    assert not harness.expired_refused(
        {
            "connected": True,
            "ok": False,
            "error": f"RuntimeError: {_rendered('refused_old_signer')}",
        }
    )
    assert not harness.expired_refused({"connected": True, "ok": False, "error": "TimeoutError"})


def test_every_child_env_carries_the_notification_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness drives scripted sessions, so no child may notify the operator.

    ``_child_env`` is the mapping the inner run, the node/operator roles, the
    bridge daemon and Chrome are built from. It is a FILTERED copy, so the gate
    must be set by the builder rather than inherited: here the ambient switch
    is removed first, and an ``extra`` override must not drop it either. The
    literal is pinned to the product's one definition so a rename cannot leave
    these children silently un-gated.
    """
    from local_operator.tui.notify import ENV_DISABLE, ENV_DISABLE_VALUE

    monkeypatch.delenv(ENV_DISABLE, raising=False)
    monkeypatch.setenv("CMUX_SURFACE_ID", "synthetic")

    env = harness._child_env(LOCAL_OPERATOR_CONFIG_DIR="/synthetic/node")

    assert harness.NO_NOTIFY_CHILD_ENV == {ENV_DISABLE: ENV_DISABLE_VALUE}
    assert env[ENV_DISABLE] == ENV_DISABLE_VALUE
    assert env["LOCAL_OPERATOR_CONFIG_DIR"] == "/synthetic/node"
    assert "CMUX_SURFACE_ID" not in env
