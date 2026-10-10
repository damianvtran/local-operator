"""Notification-click durability: does the CLICK actually take the user somewhere?

The click half of the Aida check-in banner (PR #2110 made it appear; this is the
half that makes it land). Each cell below is one per-OS failure that was
MEASURED or read in code before it was fixed, and each is pinned by a test that
fails when the fix is reverted (the mutation table is in the PR).

What these tests CAN and CANNOT show, stated so nobody reads them as more:

- macOS: the helper's click-wait window is asserted on the REAL compiled
  binary (its dry-run seam prints the window it would use) where a compiler
  exists, and on the argv the product builds everywhere. That a Notification
  Centre click reaches the helper is a property of macOS, shown only by the one
  operator-run banner probe recorded in the PR.
- Linux: argv and probe logic only. There is no Linux host or notification
  daemon here; `notify-send` is a double.
- Windows: nothing — there is no runtime banner there (see
  ``detached_notify``'s docstring). The one test is that the absence is
  documented in the place a reader would look.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.tui import notifier_app, notify
from local_operator.tui import resume_click as rc

# ---------------------------------------------------------------------------
# macOS: the helper's wait window is a parameter
# ---------------------------------------------------------------------------


def test_the_notify_argv_carries_the_activation_window_in_slot_five() -> None:
    app = Path("/x/LocalOperator.app")
    argv = notifier_app.notify_command(app, "T", "B", "lop resume-click abc", "Sub", 3600)

    # title body click subtitle window — the positions notifier.m reads.
    assert argv[1:] == ["T", "B", "lop resume-click abc", "Sub", "3600"]


def test_no_window_means_the_helper_default_and_no_extra_slot() -> None:
    """Every pre-existing caller must produce the argv it always produced."""
    app = Path("/x/LocalOperator.app")

    assert notifier_app.notify_command(app, "T", "B", "click", "Sub")[1:] == [
        "T",
        "B",
        "click",
        "Sub",
    ]
    assert notifier_app.notify_command(app, "T", "B")[1:] == ["T", "B"]


def test_a_window_with_no_subtitle_keeps_its_positional_slot() -> None:
    """An empty subtitle must still occupy slot 4, or the window shifts into it."""
    argv = notifier_app.notify_command(Path("/x/A.app"), "T", "B", "click", "", 60)

    assert argv[1:] == ["T", "B", "click", "", "60"]


def test_the_requested_window_is_clamped_to_the_helpers_ceiling() -> None:
    app = Path("/x/A.app")
    huge = notifier_app.notify_command(app, "T", "B", "c", "", 10**9)
    tiny = notifier_app.notify_command(app, "T", "B", "c", "", -5)

    assert huge[-1] == str(int(notifier_app.MAX_ACTIVATION_WINDOW_S))
    assert tiny[-1] == "1"


def test_the_stamp_forces_a_rebuild_of_the_helper_that_ignores_the_window() -> None:
    """A stamp-"2" binary exits 30 s after posting whatever argv[5] says."""
    assert notifier_app.BUILD_STAMP != "2"


def test_the_helper_source_reads_the_window_argument() -> None:
    """Runs everywhere: the source must parse argv[5] and bound it.

    The compiled-binary test below is the real proof but is darwin-only; this
    one keeps a revert to a hard-coded 30 s from passing on the Linux CI legs.
    """
    source = (Path(notifier_app.__file__).parent / "notifier.m").read_text(encoding="utf-8")

    assert "argv[5]" in source
    assert "kMaxActivationWindow" in source
    # The old fixed constant must be gone, not merely shadowed.
    assert "kActivationWindow " not in source.replace("kDefaultActivationWindow", "")


@pytest.mark.skipif(
    sys.platform != "darwin" or not __import__("shutil").which("clang"),
    reason="needs macOS and a compiler to build the real helper",
)
def test_the_compiled_helper_uses_the_window_it_is_given(tmp_path: Path) -> None:
    """The REAL binary, built from the shipped source, reports the window it
    would wait for. Its dry-run seam exits before touching Notification Centre,
    so nothing is posted to the desktop."""
    # Compiled directly, NOT through `build_bundle`: that also runs `lsregister
    # -f`, which would point LaunchServices' record for our one real bundle id
    # (`me.damiantran.localoperator`) at a throwaway pytest directory and could
    # change which binary the operator's real banners are attributed to.
    binary = str(tmp_path / "notifier")
    subprocess.run(
        [
            "clang",
            "-framework",
            "Foundation",
            "-Wno-deprecated-declarations",
            "-o",
            binary,
            str(Path(notifier_app.__file__).parent / "notifier.m"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    env = {"PATH": os.environ["PATH"], "LOCAL_OPERATOR_NOTIFIER_DRY_RUN": "1"}

    def window(*extra: str) -> str:
        out = subprocess.run(
            [binary, "T", "B", "echo hi", "", *extra],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=True,
        ).stdout
        return out.split()[0]

    assert window() == "window=30"  # default preserved for every other banner
    assert window("5400") == "window=5400"  # 90 minutes: 08:30 -> 10:00
    assert window("999999999") == "window=86400"  # bounded
    assert window("30s") == "window=30"  # malformed is not half-read
    assert window("-5") == "window=1"


def test_a_durable_banner_asks_for_the_long_window_and_a_blocking_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """detached_notify(durable_click_s=…) → block the build, pass the window."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(notify.ENV_DISABLE, raising=False)
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    seen: dict[str, object] = {}
    app = Path("/x/LocalOperator.app")

    def ensure(config_dir, *, block=False):  # noqa: ANN001
        seen["block"] = block
        return app

    monkeypatch.setattr(notifier_app, "ensure_bundle", ensure)
    spawned: list[list[str]] = []
    monkeypatch.setattr(notify, "_spawn_detached_ok", lambda argv: spawned.append(argv) or True)

    assert notify.detached_notify("Aida", "body", session_id="abc123def456", durable_click_s=7200)

    assert seen["block"] is True
    assert spawned[0][-1] == "7200", spawned
    assert "resume-click abc123def456" in " ".join(spawned[0])


def test_an_ordinary_banner_keeps_the_nonblocking_bundle_and_the_default_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost of durability is opt-in: nobody else pays a compile or a
    resident helper."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(notify.ENV_DISABLE, raising=False)
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    seen: dict[str, object] = {}

    def ensure(config_dir, *, block=False):  # noqa: ANN001
        seen["block"] = block
        return Path("/x/LocalOperator.app")

    monkeypatch.setattr(notifier_app, "ensure_bundle", ensure)
    spawned: list[list[str]] = []
    monkeypatch.setattr(notify, "_spawn_detached_ok", lambda argv: spawned.append(argv) or True)

    assert notify.detached_notify("Done", "body", session_id="abc123def456")

    assert seen["block"] is False
    # title body click subtitle? — no window slot at all.
    assert spawned[0][1:] == ["Done", "body", spawned[0][3]] or len(spawned[0]) <= 5
    assert not spawned[0][-1].isdigit()


def test_a_window_without_a_click_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """No session → nothing to click → no resident helper to pay for."""
    monkeypatch.setattr(notifier_app, "ensure_bundle", lambda *a, **k: Path("/x/LocalOperator.app"))

    argv = notify._identity_notifier("T", "B", "", "", 3600)

    assert argv is not None and argv[1:] == ["T", "B"]


def test_a_blocking_ensure_builds_under_the_single_builder_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-warm and a background builder must never run two compilers into one
    bundle directory."""
    monkeypatch.setattr(sys, "platform", "darwin")
    lock = tmp_path / "notifier" / ".building"
    during: list[bool] = []

    def fake_build(config_dir: Path) -> Path:
        during.append(lock.exists())
        return notifier_app.bundle_root(config_dir)

    monkeypatch.setattr(notifier_app, "build_bundle", fake_build)

    assert notifier_app.ensure_bundle(tmp_path, block=True) is not None
    assert during == [True], "the build ran without holding the marker"
    assert not lock.exists(), "the marker must be released after the build"


def test_a_blocking_ensure_waits_for_a_build_someone_else_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(notifier_app, "_BLOCKING_BUILD_WAIT_S", 5.0)
    lock = tmp_path / "notifier" / ".building"
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    built = {"n": 0}

    def fake_is_built(app: Path) -> bool:
        built["n"] += 1
        return built["n"] >= 3  # the other builder finishes while we wait

    monkeypatch.setattr(notifier_app, "is_built", fake_is_built)
    monkeypatch.setattr(
        notifier_app, "build_bundle", lambda d: pytest.fail("must not compile twice")
    )

    assert notifier_app.ensure_bundle(tmp_path, block=True) is not None


def test_prewarm_builds_synchronously_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(notifier_app, "build_bundle", lambda d: notifier_app.bundle_root(d))
    assert notifier_app.prewarm(tmp_path) is True

    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise RuntimeError("compiler on fire")

    monkeypatch.setattr(notifier_app, "ensure_bundle", boom)
    assert notifier_app.prewarm(tmp_path) is False


# ---------------------------------------------------------------------------
# Linux: the --action probe survives the process
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_notify_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A real file standing in for notify-send (the probe keys on its stat), and
    an isolated config dir for the persisted answer."""
    binary = tmp_path / "notify-send"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path / "cfg")
    notify._ACTION_SUPPORT.clear()
    yield str(binary)
    notify._ACTION_SUPPORT.clear()


def _probe_returns(monkeypatch: pytest.MonkeyPatch, text: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv, **_kwargs):  # noqa: ANN001
        calls.append(list(argv))
        return SimpleNamespace(stdout=text, stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_a_fresh_process_reads_the_persisted_answer_and_is_clickable_at_once(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE LINUX DEFECT. Process one probes (as a TUI would); process two — a
    one-turn runtime posting a single banner — must not need a probe at all."""
    _probe_returns(monkeypatch, "  --action=KEY=LABEL\n")
    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is True

    notify._ACTION_SUPPORT.clear()  # a new process: empty memory, same disk
    calls = _probe_returns(monkeypatch, "(must not be asked)")
    threads_before = threading.active_count()

    assert notify._notify_send_supports_actions(fake_notify_send) is True
    assert calls == [], "the second process re-probed instead of reading the disk"
    assert threading.active_count() <= threads_before


def test_a_changed_binary_invalidates_the_persisted_answer(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upgrade is exactly when --action can appear; a stale 'no' must not stick."""
    _probe_returns(monkeypatch, "  --urgency\n")
    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is False

    notify._ACTION_SUPPORT.clear()
    Path(fake_notify_send).write_text("#!/bin/sh\n# libnotify 0.8\n")  # new size+mtime
    calls = _probe_returns(monkeypatch, "  --action=KEY=LABEL\n")

    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is True
    assert len(calls) == 1


def test_a_torn_persisted_file_is_treated_as_absent(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "cfg" / "notifier" / notify._ACTION_PROBE_FILE
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    calls = _probe_returns(monkeypatch, "  --action=KEY=LABEL\n")

    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is True
    assert len(calls) == 1
    assert json.loads(path.read_text())["entries"][fake_notify_send]["supports"] is True


def test_the_default_path_still_never_blocks(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate path runs ON the event loop: with nothing persisted it must
    answer False immediately (and probe behind), exactly as before."""
    release = threading.Event()

    def slow(argv, **_kwargs):  # noqa: ANN001
        release.wait(5)
        return SimpleNamespace(stdout="  --action=KEY=LABEL\n", stderr="")

    monkeypatch.setattr(subprocess, "run", slow)
    try:
        assert notify._notify_send_supports_actions(fake_notify_send) is False
    finally:
        release.set()


def test_a_durable_banner_on_linux_is_clickable_on_a_cold_machine(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aida's banner, first run on a machine with nothing persisted."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(notify.ENV_DISABLE, raising=False)
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    monkeypatch.setattr(notify.shutil, "which", lambda name: fake_notify_send)
    _probe_returns(monkeypatch, "  --action=KEY=LABEL\n")
    spawned: list[list[str]] = []
    monkeypatch.setattr(notify, "_spawn_detached_ok", lambda argv: spawned.append(argv) or True)

    assert notify.detached_notify("Aida", "body", session_id="abc123def456", durable_click_s=3600)

    assert spawned[0][:2] == ["sh", "-c"], f"the banner was not clickable: {spawned[0]}"
    assert "--action=default=" in spawned[0][2]


def test_windows_has_no_runtime_banner_and_says_so() -> None:
    """The documented posture, in the place a maintainer reading the function
    would look. A Windows banner path may be added later; until it is, this
    keeps the gap stated instead of implied."""
    doc = notify.detached_notify.__doc__ or ""

    assert "WINDOWS HAS NO RUNTIME BANNER" in doc
    assert "desktop-notifier.ts" in doc


# ---------------------------------------------------------------------------
# The last rung: the terminal the user last attended
# ---------------------------------------------------------------------------


class _Recorder:
    """A spawn backend that records and answers as told."""

    def __init__(self, name: str, answer: bool, order: list[str]) -> None:
        self.name = name
        self._answer = answer
        self._order = order

    def spawn(self, launch, env) -> bool:  # noqa: ANN001
        self._order.append(self.name)
        return self._answer


def _rung4(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, remembered: str | None):
    """Rung 4 with no detectable terminal, macOS, and a memory of `remembered`."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: None)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    if remembered:
        from local_operator.spawn import remembered as memory

        assert memory.remember(tmp_path, remembered)


def test_the_remembered_terminal_is_tried_before_terminal_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE macOS DEFECT: the last resort was always Terminal.app."""
    _rung4(monkeypatch, tmp_path, remembered="ghostty")
    order: list[str] = []
    from local_operator.spawn import apple, ghostty

    monkeypatch.setattr(
        ghostty.GhosttyBackend, "spawn", lambda s, launch, env: order.append("ghostty") or True
    )
    monkeypatch.setattr(
        apple.TerminalAppBackend,
        "spawn",
        lambda s, launch, env: order.append("terminal.app") or True,
    )

    assert rc._spawn_terminal("a1b2c3d4e5f6") is True
    assert order == ["ghostty"], order


def test_a_remembered_terminal_that_has_gone_falls_through_to_terminal_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale memory (uninstalled terminal) costs one refused spawn, never the click."""
    _rung4(monkeypatch, tmp_path, remembered="kitty")
    order: list[str] = []
    from local_operator.spawn import apple, kitty

    monkeypatch.setattr(
        kitty.KittyBackend, "spawn", lambda s, launch, env: order.append("kitty") or False
    )
    monkeypatch.setattr(
        apple.TerminalAppBackend,
        "spawn",
        lambda s, launch, env: order.append("terminal.app") or True,
    )

    assert rc._spawn_terminal("a1b2c3d4e5f6") is True
    assert order == ["kitty", "terminal.app"], order


def test_with_no_memory_the_ladder_is_exactly_what_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _rung4(monkeypatch, tmp_path, remembered=None)
    order: list[str] = []
    from local_operator.spawn import apple

    monkeypatch.setattr(
        apple.TerminalAppBackend,
        "spawn",
        lambda s, launch, env: order.append("terminal.app") or True,
    )

    assert rc._spawn_terminal("a1b2c3d4e5f6") is True
    assert order == ["terminal.app"]


def test_a_remembered_terminal_is_withheld_over_ssh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No window server over ssh: a remembered emulator would claim a landing
    it never made, the defect the ladder exists to remove."""
    _rung4(monkeypatch, tmp_path, remembered="ghostty")
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 5000 10.0.0.2 22")
    order: list[str] = []
    from local_operator.spawn import ghostty

    monkeypatch.setattr(
        ghostty.GhosttyBackend, "spawn", lambda s, launch, env: order.append("ghostty") or True
    )

    assert rc._spawn_terminal("a1b2c3d4e5f6") is False
    assert order == []


def test_the_remembered_backend_is_not_tried_twice_when_it_is_the_detected_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _rung4(monkeypatch, tmp_path, remembered="ghostty")
    order: list[str] = []
    from local_operator.spawn import apple, ghostty

    detected = ghostty.GhosttyBackend()
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: detected)
    monkeypatch.setattr(
        ghostty.GhosttyBackend, "spawn", lambda s, launch, env: order.append("ghostty") or False
    )
    monkeypatch.setattr(
        apple.TerminalAppBackend,
        "spawn",
        lambda s, launch, env: order.append("terminal.app") or True,
    )

    assert rc._spawn_terminal("a1b2c3d4e5f6") is True
    assert order == ["ghostty", "terminal.app"], order


def test_cmux_is_never_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmux spawns UNFOCUSED by design; as a click's landing it would look like
    a click that did nothing.

    Detection is FORCED to cmux rather than driven through markers: whether the
    registry picks cmux depends on a cmux binary existing on the host, and a test
    that only passed on a host without one would pin nothing.
    """
    from local_operator.spawn import remembered
    from local_operator.spawn.cmux import CmuxBackend

    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: CmuxBackend())

    assert remembered.remember_current(tmp_path, {}) is False
    assert remembered.recall(tmp_path) is None


def test_remember_current_stores_the_detected_backend_name(tmp_path: Path) -> None:
    from local_operator.spawn import remembered

    assert remembered.remember_current(tmp_path, {"GHOSTTY_RESOURCES_DIR": "/g"}) is True
    assert remembered.recall(tmp_path) == "ghostty"
    # Idempotent: a TUI that boots daily in one terminal rewrites nothing.
    assert remembered.remember_current(tmp_path, {"GHOSTTY_RESOURCES_DIR": "/g"}) is False
    # No terminal detectable → memory untouched.
    assert remembered.remember_current(tmp_path, {}) is False
    assert remembered.recall(tmp_path) == "ghostty"


# ---------------------------------------------------------------------------
# Pre-warm at an attended moment
# ---------------------------------------------------------------------------


def test_preparing_for_clicks_remembers_the_terminal_and_prewarms_the_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(notify.ENV_DISABLE, raising=False)
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    did: list[str] = []
    monkeypatch.setattr(
        "local_operator.spawn.remembered.remember_current",
        lambda root, env=None, **_kw: did.append("remember"),
    )
    monkeypatch.setattr(notifier_app, "prewarm", lambda root: did.append("prewarm") or True)

    rc.prepare_for_clicks()

    assert did == ["remember", "prewarm"]


def test_preparing_for_clicks_is_inert_when_notifications_are_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A user who turned banners off pays no compile and leaves no file."""
    monkeypatch.setenv(notify.ENV_DISABLE, "1")
    # The HOME gate is made PASS, so the kill switch is the only thing that can
    # be doing the refusing (under pytest's redirected HOME it would otherwise
    # refuse on its own and this test would pin nothing about the switch).
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        notifier_app, "prewarm", lambda root: pytest.fail("compiled for a muted user")
    )

    rc.prepare_for_clicks()

    assert not (tmp_path / "notifier").exists()


def test_preparing_for_clicks_is_inert_under_a_foreign_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rig (redirected HOME) must not build a bundle into the operator's
    one real Notification Centre identity."""
    monkeypatch.delenv(notify.ENV_DISABLE, raising=False)
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: tmp_path / "someone-else")
    monkeypatch.setattr(
        notifier_app, "prewarm", lambda root: pytest.fail("compiled under a foreign HOME")
    )

    rc.prepare_for_clicks()


@pytest.mark.asyncio
async def test_the_tui_schedules_click_preparation_off_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from local_operator.tui import _schedule_click_preparation

    ran_on: list[threading.Thread] = []
    kwargs_seen: list[dict[str, object]] = []

    def record(**kwargs: object) -> None:
        kwargs_seen.append(kwargs)
        ran_on.append(threading.current_thread())

    monkeypatch.setattr(rc, "prepare_for_clicks", record)
    app = SimpleNamespace(_click_prep_task=None)

    task = _schedule_click_preparation(app)
    assert app._click_prep_task is task
    await task

    assert len(ran_on) == 1
    assert ran_on[0] is not threading.main_thread(), "the compile ran on the event loop"
    # The TUI IS a person's terminal, so it alone may clear a stale memory.
    assert kwargs_seen == [{"attended_terminal": True}]
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Wiring: the preparation is actually reachable from the attended surfaces
# ---------------------------------------------------------------------------


def test_both_attended_boot_paths_call_the_preparation() -> None:
    """A helper nobody calls fixes nothing.

    Source-level on purpose: booting the real TUI or the server lifespan here
    would start a Textual app / a daemon to prove one call site. These two
    functions are the boot hooks the module docstrings name, and a refactor that
    drops the call must trip this rather than silently re-cold the first banner.
    """
    import inspect

    from local_operator import tui
    from local_operator.server import app as server_app

    assert "_schedule_click_preparation(app)" in inspect.getsource(tui.run_tui)
    # The call itself, off the loop — not merely the import beside it.
    lifespan = inspect.getsource(server_app.lifespan)
    assert "await prepare_for_clicks_detached()" in lifespan
    # The daemon has no emulator markers because it has no emulator; letting it
    # "forget an undetectable terminal" would wipe a good memory every boot.
    assert "attended_terminal" not in lifespan


# ---------------------------------------------------------------------------
# Remediation round 1 (review R1-R6, QA Q1/Q2, design D1-D4)
# ---------------------------------------------------------------------------


def _slow_then_fast_probe(monkeypatch: pytest.MonkeyPatch, text: str):
    """First call TIMES OUT; later calls answer ``text``. Returns the call log."""
    calls: list[list[str]] = []

    def run(argv, **kwargs):  # noqa: ANN001
        calls.append(list(argv))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))
        return SimpleNamespace(stdout=text, stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_a_timed_out_probe_is_not_persisted_so_the_next_process_asks_again(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F1 / R1 = Q1. One slow `notify-send --help` must not become a permanent 'no'.

    The persisted record is keyed on the binary's identity, which a slow moment
    does not change — so persisting the failure made a clickable `notify-send`
    plain-toast until it was replaced. Driven as QA drove it: the stub supports
    `--action`, its first answer times out, then the host recovers.
    """
    calls = _slow_then_fast_probe(monkeypatch, "  --action=KEY=LABEL\n")

    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is False
    assert not (
        tmp_path / "cfg" / "notifier" / notify._ACTION_PROBE_FILE
    ).exists(), "a timeout was written to disk as if it were an answer"

    notify._ACTION_SUPPORT.clear()  # the next process
    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is True
    assert len(calls) == 2, "the recovered host was never asked again"


def test_a_nonzero_or_empty_probe_is_not_persisted_either(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "cfg" / "notifier" / notify._ACTION_PROBE_FILE
    for result in (
        SimpleNamespace(stdout="", stderr="", returncode=0),  # nothing readable
        SimpleNamespace(stdout="  --action", stderr="", returncode=1),  # failed run
    ):
        notify._ACTION_SUPPORT.clear()
        monkeypatch.setattr(subprocess, "run", lambda *a, _r=result, **k: _r)
        notify._notify_send_supports_actions(fake_notify_send, block=True)
        assert not path.exists(), result


def test_a_conclusive_probe_is_persisted_in_both_directions(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other half of F1: persisting only conclusive answers must still persist them."""
    path = tmp_path / "cfg" / "notifier" / notify._ACTION_PROBE_FILE
    for text, expected in (("  --action=KEY=LABEL\n", True), ("  --urgency\n", False)):
        notify._ACTION_SUPPORT.clear()
        path.unlink(missing_ok=True)  # the first answer would otherwise be read back
        _probe_returns(monkeypatch, text)
        assert notify._notify_send_supports_actions(fake_notify_send, block=True) is expected
        record = json.loads(path.read_text())["entries"][fake_notify_send]
        assert record["supports"] is expected


def test_two_notify_send_binaries_do_not_evict_each_other(
    fake_notify_send: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review nit: a single-record file made two alternating binaries re-probe forever."""
    other = tmp_path / "other-notify-send"
    other.write_text("#!/bin/sh\n# a second one on PATH\n")
    other.chmod(0o755)
    _probe_returns(monkeypatch, "  --action=KEY=LABEL\n")
    assert notify._notify_send_supports_actions(fake_notify_send, block=True) is True
    assert notify._notify_send_supports_actions(str(other), block=True) is True

    notify._ACTION_SUPPORT.clear()
    calls = _probe_returns(monkeypatch, "(must not be asked)")

    assert notify._notify_send_supports_actions(fake_notify_send) is True
    assert notify._notify_send_supports_actions(str(other)) is True
    assert calls == []


# --- remembered terminal: age, forget, and the read-side filter (F2 / D2 / Q2) ---


def test_a_memory_older_than_the_limit_is_not_recalled(tmp_path: Path) -> None:
    from local_operator.spawn import remembered

    assert remembered.remember(tmp_path, "ghostty", now=1_000_000)
    assert remembered.recall(tmp_path, now=1_000_000 + remembered.MAX_AGE_S - 1) == "ghostty"
    assert remembered.recall(tmp_path, now=1_000_000 + remembered.MAX_AGE_S + 1) is None


def test_a_memory_with_no_usable_date_is_not_recalled(tmp_path: Path) -> None:
    """An undated memory cannot be shown to be fresh, so it is not believed."""
    from local_operator.spawn import remembered

    path = tmp_path / "notifier" / "last-terminal.json"
    path.parent.mkdir(parents=True)
    for payload in ({"backend": "ghostty"}, {"backend": "ghostty", "recorded_at": "yesterday"}):
        path.write_text(json.dumps(payload))
        assert remembered.recall(tmp_path) is None, payload


def test_a_daily_user_refreshes_the_date_instead_of_aging_out(tmp_path: Path) -> None:
    from local_operator.spawn import remembered

    assert remembered.remember(tmp_path, "ghostty", now=0)
    # Same terminal, inside the refresh interval: no rewrite.
    assert remembered.remember(tmp_path, "ghostty", now=60) is False
    # Same terminal, past it: rewritten with the new date.
    later = remembered._REFRESH_AFTER_S + 1
    assert remembered.remember(tmp_path, "ghostty", now=later) is True
    assert remembered.recall(tmp_path, now=later + remembered.MAX_AGE_S - 1) == "ghostty"


def test_an_attended_boot_that_cannot_name_its_terminal_forgets_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2 / R2(a): booted in Ghostty once, now lives in VS Code's terminal."""
    from local_operator.spawn import remembered

    assert remembered.remember(tmp_path, "ghostty")
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: None)

    # A caller that is NOT a terminal (the desktop daemon) must leave it alone.
    assert remembered.remember_current(tmp_path, {}) is False
    assert remembered.recall(tmp_path) == "ghostty"
    # The TUI is a terminal: not naming it means the user has left the old one.
    assert remembered.remember_current(tmp_path, {}, forget_if_undetected=True) is True
    assert remembered.recall(tmp_path) is None


def test_a_cmux_boot_also_forgets_the_older_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.spawn import remembered
    from local_operator.spawn.cmux import CmuxBackend

    assert remembered.remember(tmp_path, "ghostty")
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: CmuxBackend())

    assert remembered.remember_current(tmp_path, {}, forget_if_undetected=True) is True
    assert remembered.recall(tmp_path) is None


def test_a_hand_written_cmux_memory_is_not_honoured_on_read(tmp_path: Path) -> None:
    """Q2: the write-side filter alone trusts every file that ever reached the disk."""
    from local_operator.spawn import remembered

    path = tmp_path / "notifier" / "last-terminal.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"backend": "cmux", "recorded_at": int(__import__("time").time())}))

    assert remembered.recall(tmp_path) is None


def test_prepare_for_clicks_forgets_only_when_the_caller_is_a_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(notify.ENV_DISABLE, raising=False)
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr(notifier_app, "prewarm", lambda root: True)
    seen: list[object] = []
    monkeypatch.setattr(
        "local_operator.spawn.remembered.remember_current",
        lambda root, env=None, *, forget_if_undetected=False: seen.append(forget_if_undetected),
    )

    rc.prepare_for_clicks()
    rc.prepare_for_clicks(attended_terminal=True)

    assert seen == [False, True]


# --- F3: a terminal that is GONE must actually fall through ---


class _OpenAndOsascript:
    """Doubles ``subprocess.Popen`` the way macOS behaves for a missing Ghostty.

    ``open -na Ghostty.app`` STARTS fine and then exits 1 for a bundle that is
    not installed (measured: rc=1 in 0.3 s); ``osascript`` opens Terminal.app.
    Neither backend is stubbed — the real ``GhosttyBackend`` and
    ``TerminalAppBackend`` run, and only the OS process is faked — so the
    outcome is decided by the code under test, not by the stub.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, open_rc: int | None) -> None:
        self.argv: list[list[str]] = []
        recorder = self

        class _Stdin:
            def write(self, data: bytes) -> None:
                pass

            def close(self) -> None:
                pass

        class _Process:
            stdin = _Stdin()

            def __init__(self, argv: list[str]) -> None:
                self._argv = argv

            def wait(self, timeout=None) -> int:  # noqa: ANN001
                if self._argv[0] == "open":
                    if open_rc is None:
                        raise subprocess.TimeoutExpired(self._argv, timeout or 0.0)
                    return open_rc
                return 0

        def fake_popen(argv, **_kwargs):  # noqa: ANN001
            recorder.argv.append(list(argv))
            return _Process(list(argv))

        monkeypatch.setattr(subprocess, "Popen", fake_popen)


def test_a_missing_ghostty_falls_through_to_terminal_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F3 / R2(b): the claim "a gone terminal costs one refused spawn" is now true."""
    _rung4(monkeypatch, tmp_path, remembered="ghostty")
    spawns = _OpenAndOsascript(monkeypatch, open_rc=1)

    assert rc._spawn_terminal("a1b2c3d4e5f6") is True
    assert [a[0] for a in spawns.argv] == ["open", "osascript"], spawns.argv


def test_an_installed_ghostty_is_not_followed_by_terminal_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control: a launch that succeeded (or is still running) opens ONE window."""
    for open_rc in (0, None):  # None = still alive at the bound, which counts as success
        run_dir = tmp_path / f"open-{open_rc}"  # a fresh memory: one write per run
        run_dir.mkdir()
        _rung4(monkeypatch, run_dir, remembered="ghostty")
        spawns = _OpenAndOsascript(monkeypatch, open_rc=open_rc)

        assert rc._spawn_terminal("a1b2c3d4e5f6") is True
        assert [a[0] for a in spawns.argv] == ["open"], (open_rc, spawns.argv)


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS's real `open`")
def test_the_real_open_reports_a_missing_bundle_as_a_refusal() -> None:
    """The premise of F3, on the real binary: no mock decides this."""
    from local_operator.spawn import ghostty

    assert (
        ghostty._open_reported(["open", "-na", "NoSuchTerminalBundle-click-durability.app"])
        is False
    )


# --- F4: the click runs the command carried by the banner ---


@pytest.fixture(scope="module")
def dry_run_helper(tmp_path_factory: pytest.TempPathFactory) -> str:
    if sys.platform != "darwin" or not __import__("shutil").which("clang"):
        pytest.skip("needs macOS and a compiler to build the real helper")
    binary = str(tmp_path_factory.mktemp("helper") / "notifier")
    subprocess.run(
        [
            "clang",
            "-framework",
            "Foundation",
            "-Wno-deprecated-declarations",
            "-o",
            binary,
            str(Path(notifier_app.__file__).parent / "notifier.m"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return binary


def _dry(binary: str, *argv: str) -> dict[str, str]:
    out = subprocess.run(
        [binary, *argv],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "LOCAL_OPERATOR_NOTIFIER_DRY_RUN": "1"},
        timeout=30,
        check=True,
    ).stdout.strip()
    # "window=5400 command=echo MINE carried=echo MINE foreign=echo MINE"
    import re

    return dict(re.findall(r"(\w+)=(.*?)(?= \w+=|$)", out))


def test_the_banner_carries_its_own_command_and_a_foreign_helper_runs_it(
    dry_run_helper: str,
) -> None:
    """THE PREMISE, shown on the real binary rather than asserted.

    Several helpers of one bundle id are alive at once (her 24 h helper outlives
    every 30 s toast posted after it), and macOS routes a click by bundle id. The
    dry-run builds the notification exactly as a post would, then asks
    `commandToRun` what a DIFFERENT live helper — whose own command is
    `OTHER-HELPER` — would run for this banner's click. It must run THIS banner's
    command, which only works if the command rode the notification (`carried`).
    What it cannot show is macOS actually mis-delivering a click; that needs a
    real Notification Centre and was not probed.
    """
    got = _dry(dry_run_helper, "T", "B", "lop resume-click abc123", "", "5400")

    assert got["carried"] == "lop resume-click abc123"
    assert got["foreign"] == "lop resume-click abc123"
    assert got["command"] == "lop resume-click abc123"


def test_a_banner_with_no_click_falls_back_to_the_helpers_own_command(
    dry_run_helper: str,
) -> None:
    """No carried command (a banner with no click action): the fallback is `ownCommand`."""
    got = _dry(dry_run_helper, "T", "B")

    assert got["carried"] == ""
    assert got["foreign"] == "OTHER-HELPER"


def test_the_helper_source_carries_and_reads_the_command() -> None:
    """Runs on every platform: the two halves of the carry, by name.

    Dropping the write OR the read leaves the other half compiling and the old
    single-helper behaviour intact, which is exactly how review found the first
    version unpinned (31 tests passed with both removed).
    """
    source = (Path(notifier_app.__file__).parent / "notifier.m").read_text(encoding="utf-8")

    assert '@{@"command" : command}' in source  # the write
    assert 'objectForKey:@"command"' in source  # the read
    assert "commandToRun(notification, self.command)" in source  # the delegate uses it


# --- F7: the dry-run seam fires only for exactly "1" ---


def test_the_dry_run_seam_requires_exactly_one() -> None:
    """An inherited `=0` or empty value must not silently drop every banner.

    Source-level because the behavioural cell (`=0` -> a REAL post) is exactly
    what must never be run from a test.
    """
    source = (Path(notifier_app.__file__).parent / "notifier.m").read_text(encoding="utf-8")

    assert 'strcmp(value, "1") == 0' in source
    assert 'getenv("LOCAL_OPERATOR_NOTIFIER_DRY_RUN") != NULL' not in source


@pytest.mark.skipif(
    sys.platform != "darwin" or not __import__("shutil").which("clang"),
    reason="needs macOS and a compiler",
)
def test_the_compiled_dry_run_predicate_accepts_only_one(tmp_path: Path) -> None:
    """The REAL `dryRunRequested`, compiled and called directly — never via a post."""
    harness = tmp_path / "harness.m"
    harness.write_text(
        "#define main notifier_main\n"
        f'#include "{Path(notifier_app.__file__).parent / "notifier.m"}"\n'
        "#undef main\n"
        "int main(void) {\n"
        '    const char *cases[] = {"1", "0", "", "11", "true", "01"};\n'
        "    for (int i = 0; i < 6; i++) {\n"
        '        setenv("LOCAL_OPERATOR_NOTIFIER_DRY_RUN", cases[i], 1);\n'
        '        printf("[%s]=%d\\n", cases[i], dryRunRequested());\n'
        "    }\n"
        '    unsetenv("LOCAL_OPERATOR_NOTIFIER_DRY_RUN");\n'
        '    printf("[unset]=%d\\n", dryRunRequested());\n'
        "    return 0;\n"
        "}\n"
    )
    binary = tmp_path / "harness"
    subprocess.run(
        [
            "clang",
            "-framework",
            "Foundation",
            "-Wno-deprecated-declarations",
            "-o",
            str(binary),
            str(harness),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    out = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30).stdout

    assert out.split() == [
        "[1]=1",
        "[0]=0",
        "[]=0",
        "[11]=0",
        "[true]=0",
        "[01]=0",
        "[unset]=0",
    ], out


# --- F6: quitting during the first-run compile must not wait for it ---


def test_the_click_preparation_does_not_delay_loop_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5: `asyncio.run` joins the default executor, so `to_thread` made quitting wait.

    Measured before the fix with a plain `to_thread` sleep: `asyncio.run` returned
    only after the sleep. Here the preparation takes 3 s and the loop is asked to
    shut down immediately; it must return well inside that.
    """
    import asyncio
    import time

    started = threading.Event()

    def slow_prepare(**_kwargs: object) -> None:
        started.set()
        time.sleep(3.0)

    monkeypatch.setattr(rc, "prepare_for_clicks", slow_prepare)

    async def boot() -> None:
        asyncio.get_running_loop().create_task(rc.prepare_for_clicks_detached())
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

    t0 = time.monotonic()
    asyncio.run(boot())
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, f"shutdown waited {elapsed:.1f}s for the compile"
    thread = next(t for t in threading.enumerate() if t.name == "click-preparation")
    assert thread.daemon is True
    thread.join(timeout=5)
