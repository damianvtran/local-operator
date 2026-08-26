"""Multiplexer resume publication — the safety rules, not the plumbing.

The properties pinned here are the ones whose failure is INVISIBLE until a
crash, which is the worst possible time to find out:

- **Restore-and-idle.** The published command reopens a transcript and waits.
  A command that continued the interrupted turn would, on a fifteen-pane
  restore, resume tool execution in fifteen sessions with nobody watching.
- **Never a subagent's session.** A child runs in its PARENT's pane, so
  publishing one silently replaces the user's binding with a delegated review.
- **Never fatal, never blocking.** Every failure mode a multiplexer can
  present must leave the session running.
- **A clean exit wins over an in-flight re-assert.** Otherwise quitting can
  resurrect the binding it just cleared, and the next shell replays a session
  the user closed.
"""

from __future__ import annotations

import json
import shlex
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.multiplexer.broadcast import (
    _PENDING_POLL_S,
    RESUME_FLAG,
    SWAP_DRAIN_TIMEOUT_S,
    SessionBroadcast,
    broadcast_session,
    build_binding,
    is_resumable_session,
    is_user_owned_session,
    multiplexer_resume_enabled,
    resume_argv,
    resume_executable,
    retire_session,
)
from local_operator.multiplexer.cmux import REASSERT_INTERVAL_S, CmuxBackend
from local_operator.multiplexer.markers import (
    COMMAND_OPTION,
    SESSION_OPTION,
    ScreenBackend,
    TmuxBackend,
    WezTermBackend,
    ZellijBackend,
)
from local_operator.multiplexer.registry import active_backend
from local_operator.multiplexer.types import SessionBinding

WORKSPACE = "11111111-2222-3333-4444-555555555555"
SURFACE = "66666666-7777-8888-9999-000000000000"


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app's config dir at a tmp tree, as `paths` documents."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _make_session(config_dir: Path, session_id: str, *, subagent: bool = False) -> Path:
    """A session directory shaped exactly like the real thing."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    if subagent:
        (directory / "origin.json").write_text(json.dumps({"origin": "subagent"}), encoding="utf-8")
    return directory


def _binding(session_id: str = "abc123abc123") -> SessionBinding:
    return SessionBinding(
        session_id=session_id,
        executable="/home/u/.local/bin/lop",
        argv=("/home/u/.local/bin/lop", RESUME_FLAG, session_id),
        cwd="/work",
    )


class TestRestoreAndIdle:
    def test_the_published_argv_only_ever_resumes(self) -> None:
        """The safety boundary: no prompt, no exec, no continue flag."""
        argv = resume_argv("abc123abc123", "/bin/lop")
        assert argv == ("/bin/lop", "--resume", "abc123abc123")

    def test_no_flag_can_continue_an_interrupted_turn(self) -> None:
        """Pins the ABSENCE of the dangerous flags, not just the shape."""
        argv = resume_argv("abc123abc123", "/bin/lop")
        forbidden = {"--exec", "-e", "--yolo", "--continue", "--prompt"}
        assert not forbidden.intersection(argv)

    def test_the_executable_is_the_launcher_not_the_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`sys.executable` would restore a bare Python REPL, not lop."""
        launcher = tmp_path / "lop"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr("sys.argv", [str(launcher)])
        assert resume_executable() == str(launcher.resolve())

    def test_a_missing_argv0_falls_back_to_a_resolvable_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An embedder has no usable argv[0]; a bad path is worse than a name."""
        monkeypatch.setattr("sys.argv", [])
        assert resume_executable() == "lop"


class TestSubagentGuard:
    def test_a_subagent_session_is_never_publishable(self, config_dir: Path) -> None:
        _make_session(config_dir, "child0000001", subagent=True)
        assert is_user_owned_session("child0000001") is False

    def test_a_user_session_is_publishable(self, config_dir: Path) -> None:
        _make_session(config_dir, "abc123abc123")
        assert is_user_owned_session("abc123abc123") is True

    def test_broadcast_refuses_a_subagent_session(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end-to-end refusal: a child must not touch the pane's binding."""
        _make_session(config_dir, "child0000001", subagent=True)
        monkeypatch.setenv("CMUX_WORKSPACE_ID", WORKSPACE)
        monkeypatch.setenv("CMUX_SURFACE_ID", SURFACE)
        published: list[Any] = []
        monkeypatch.setattr(CmuxBackend, "publish", lambda self, b, e: published.append(b) or True)
        monkeypatch.setattr(CmuxBackend, "detect", lambda self, env: True)
        assert broadcast_session("child0000001") is None
        assert published == []


class TestResumabilityIsRechecked:
    def test_a_session_with_no_transcript_is_not_yet_resumable(self, config_dir: Path) -> None:
        """A cold session: the directory exists, the transcript does not."""
        (config_dir / "sessions" / "cold00000001").mkdir(parents=True)
        assert is_resumable_session("cold00000001") is False

    def test_it_becomes_resumable_once_the_first_turn_persists(self, config_dir: Path) -> None:
        """Why the check is per-publish and not once at startup."""
        directory = config_dir / "sessions" / "cold00000001"
        directory.mkdir(parents=True)
        assert is_resumable_session("cold00000001") is False
        (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
        assert is_resumable_session("cold00000001") is True

    def test_a_cold_session_publishes_nothing_until_it_can_be_resumed(
        self, config_dir: Path
    ) -> None:
        """The bug this guards: publishing a command --resume would refuse."""
        directory = config_dir / "sessions" / "cold00000001"
        directory.mkdir(parents=True)
        calls: list[Any] = []

        class Backend:
            name = "fake"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                calls.append(binding)
                return True

            def retire(self, binding: Any, env: Any) -> bool:
                return True

        broadcast = SessionBroadcast(_binding("cold00000001"), Backend(), env={})
        broadcast._publish_once()
        assert calls == []
        (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
        broadcast._publish_once()
        assert len(calls) == 1


class TestNeverFatal:
    def test_a_backend_that_raises_on_detect_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend bug costs that backend, never the user's session."""

        def boom(self: Any, env: Any) -> bool:
            raise RuntimeError("scanner exploded")

        monkeypatch.setattr(CmuxBackend, "detect", boom)
        monkeypatch.setattr(TmuxBackend, "detect", lambda self, env: False)
        monkeypatch.setattr(ZellijBackend, "detect", lambda self, env: False)
        monkeypatch.setattr(WezTermBackend, "detect", lambda self, env: False)
        monkeypatch.setattr(ScreenBackend, "detect", lambda self, env: False)
        assert active_backend({}) is None

    def test_a_backend_that_raises_on_publish_does_not_escape(self, config_dir: Path) -> None:
        _make_session(config_dir, "abc123abc123")

        class Backend:
            name = "fake"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                raise RuntimeError("socket died")

            def retire(self, binding: Any, env: Any) -> bool:
                raise RuntimeError("socket died")

        broadcast = SessionBroadcast(_binding(), Backend(), env={})
        broadcast._publish_once()  # must not raise
        broadcast.stop()  # must not raise

    def test_a_host_with_no_multiplexer_publishes_nothing(self) -> None:
        """The ordinary case: a plain terminal is not an error."""
        assert active_backend({}) is None

    def test_a_broken_cmux_binary_does_not_break_startup(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cmux surface whose CLI is missing must simply not publish."""
        _make_session(config_dir, "abc123abc123")
        monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: None)
        env = {"CMUX_WORKSPACE_ID": WORKSPACE, "CMUX_SURFACE_ID": SURFACE}
        assert CmuxBackend().detect(env) is False
        assert broadcast_session("abc123abc123", env=env) is None

    def test_the_kill_switch_stops_everything(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_session(config_dir, "abc123abc123")
        env = {
            "CMUX_WORKSPACE_ID": WORKSPACE,
            "CMUX_SURFACE_ID": SURFACE,
            "LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME": "1",
        }
        assert multiplexer_resume_enabled(env) is False
        assert broadcast_session("abc123abc123", env=env) is None


class TestSwapOrdering:
    """F7: a /new or /resume swap must leave the pane advertising the NEW session.

    The failure mode these tests pin is invisible until a crash: the outgoing
    binding's withdrawal lands AFTER the incoming binding's publish, deletes
    it (both name the same pane), and the pane then advertises nothing — so
    the crash this feature exists for finds no binding to restore. Every real
    backend retire is a subprocess spawn (6-75ms on this host), which is far
    past the ~5ms threshold where the unsequenced swap starts losing, so the
    stub-Handle swap test in the TUI suite cannot see this at all: it needs a
    backend with realistic retire latency.
    """

    #: One subprocess spawn, the floor for a real backend retire on this host
    #: (screen -V 6.7ms). High enough to lose every unsequenced swap, low
    #: enough to keep the suite fast.
    RETIRE_LATENCY_S = 0.05

    @staticmethod
    def _marker_session(config_dir: Path, pane_file: str) -> str | None:
        path = config_dir / "multiplexer" / pane_file
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return payload.get("session_id")

    def test_a_slow_retire_cannot_delete_the_successors_binding(self, config_dir: Path) -> None:
        """The zellij marker after a swap names the NEW session, not nothing.

        Mirrors `app.py`'s swap exactly — stop the outgoing broadcast, then
        start the successor with it as predecessor — against the real
        `ZellijBackend` with one subprocess spawn of retire latency. Before
        the fix the pane advertised NOTHING after the swap (20/20 trials at
        5ms of latency in review round 2).
        """
        _make_session(config_dir, "aaaaaaaaaaaa")
        _make_session(config_dir, "bbbbbbbbbbbb")
        env = {"ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}

        class SlowRetireZellij(ZellijBackend):
            def retire(self, binding: Any, env: Any) -> bool:
                time.sleep(TestSwapOrdering.RETIRE_LATENCY_S)
                return super().retire(binding, env)

        backend = SlowRetireZellij()
        outgoing = SessionBroadcast(_binding("aaaaaaaaaaaa"), backend, env=env, interval_s=3600.0)
        outgoing.start()
        deadline = time.monotonic() + 5.0
        while self._marker_session(config_dir, "zellij-main-0.json") != "aaaaaaaaaaaa":
            assert time.monotonic() < deadline, "outgoing binding never published"
            time.sleep(0.005)

        # app.py's swap: stop() returns immediately, the successor sequences.
        outgoing.stop(retire=True)
        incoming = SessionBroadcast(
            _binding("bbbbbbbbbbbb"),
            backend,
            env=env,
            interval_s=3600.0,
            predecessor=outgoing,
        )
        incoming.start()
        incoming.join(timeout=10.0)
        outgoing.join(timeout=10.0)

        assert (
            self._marker_session(config_dir, "zellij-main-0.json") == "bbbbbbbbbbbb"
        ), "swap left the pane advertising nothing — a crash now loses the session"

    def test_the_swap_drain_is_bounded_by_the_number_it_documents(self, config_dir: Path) -> None:
        """F9: a wedged predecessor delays the successor by SWAP_DRAIN_TIMEOUT_S, once.

        Nothing pinned this bound before, which is how a 2x survived to review.
        The successor used to drain its predecessor with `join`, whose timeout
        is PER WORKER over the retire worker and the timer, so the real bound
        on the first publish was 12s against a documented 6s (measured 12.02s).
        `_drain_retire` waits only on the withdrawal, which is the sole worker
        carrying swap ordering.

        The assertion is one-sided on purpose: the LOWER bound would pin the
        drain actually happening, but that is `test_a_slow_retire_cannot_
        delete_the_successors_binding`'s job. This one pins the ceiling.
        """
        _make_session(config_dir, "aaaaaaaaaaaa")
        _make_session(config_dir, "bbbbbbbbbbbb")
        published = threading.Event()
        release = threading.Event()

        class WedgedBackend:
            name = "wedged"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                if binding.session_id == "bbbbbbbbbbbb":
                    published.set()
                    return True
                # The predecessor's publish parks, so its retire queues behind
                # a call already inside a subprocess timeout — the state that
                # makes the drain bound observable at all.
                release.wait(timeout=30.0)
                return False

            def retire(self, binding: Any, env: Any) -> bool:
                release.wait(timeout=30.0)
                return True

        backend = WedgedBackend()
        outgoing = SessionBroadcast(_binding("aaaaaaaaaaaa"), backend, env={}, interval_s=3600.0)
        outgoing.start()
        time.sleep(0.2)  # let the publish get INTO the wedged call
        outgoing.stop(retire=True)

        started = time.monotonic()
        incoming = SessionBroadcast(
            _binding("bbbbbbbbbbbb"),
            backend,
            env={},
            interval_s=3600.0,
            predecessor=outgoing,
        )
        incoming.start()
        assert published.wait(timeout=30.0), "the successor never published at all"
        elapsed = time.monotonic() - started

        # One drain budget plus scheduling slack, and well under the 12s the
        # per-worker join produced on this same scenario.
        assert elapsed < SWAP_DRAIN_TIMEOUT_S + 2.0, (
            f"the successor's first publish waited {elapsed:.2f}s, past the "
            f"documented {SWAP_DRAIN_TIMEOUT_S}s swap drain bound"
        )
        release.set()
        incoming.stop(retire=False)

    def test_a_swap_chain_does_not_compound_the_drain(self, config_dir: Path) -> None:
        """F9, the compounding half: A->B->C pays ONE drain, not two.

        Draining the predecessor's TIMER meant waiting out that timer's own
        drain of ITS predecessor, so a second swap against a wedged socket
        stacked the bounds (measured 11.70s for C's first publish). Only the
        withdrawal is drained now, and a withdrawal never waits on another
        broadcast, so the chain cannot accumulate.
        """
        for session_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"):
            _make_session(config_dir, session_id)
        published_c = threading.Event()
        release = threading.Event()

        class WedgedBackend:
            name = "wedged"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                if binding.session_id == "cccccccccccc":
                    published_c.set()
                    return True
                release.wait(timeout=30.0)
                return False

            def retire(self, binding: Any, env: Any) -> bool:
                release.wait(timeout=30.0)
                return True

        backend = WedgedBackend()
        first = SessionBroadcast(_binding("aaaaaaaaaaaa"), backend, env={}, interval_s=3600.0)
        first.start()
        time.sleep(0.2)
        first.stop(retire=True)
        second = SessionBroadcast(
            _binding("bbbbbbbbbbbb"),
            backend,
            env={},
            interval_s=3600.0,
            predecessor=first,
        )
        second.start()
        time.sleep(0.2)
        second.stop(retire=True)

        started = time.monotonic()
        third = SessionBroadcast(
            _binding("cccccccccccc"),
            backend,
            env={},
            interval_s=3600.0,
            predecessor=second,
        )
        third.start()
        assert published_c.wait(timeout=30.0), "the third session never published"
        elapsed = time.monotonic() - started

        assert elapsed < SWAP_DRAIN_TIMEOUT_S + 2.0, (
            f"a two-swap chain delayed the pane by {elapsed:.2f}s — the drain "
            f"is compounding across the chain again"
        )
        release.set()
        third.stop(retire=False)

    def test_a_scoped_retire_never_removes_a_foreign_marker(self, config_dir: Path) -> None:
        """The class fix, not the instance: retire refuses a marker it did not write.

        Sequencing makes the race rare; scoping makes it harmless. A retire
        arriving after the successor has published must withdraw nothing,
        because the marker on disk is no longer its binding.
        """
        _make_session(config_dir, "aaaaaaaaaaaa")
        _make_session(config_dir, "bbbbbbbbbbbb")
        env = {"ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}
        backend = ZellijBackend()
        assert backend.publish(_binding("bbbbbbbbbbbb"), env) is True

        # The outgoing session's retire arrives late — after the swap.
        assert backend.retire(_binding("aaaaaaaaaaaa"), env) is False
        assert (
            self._marker_session(config_dir, "zellij-main-0.json") == "bbbbbbbbbbbb"
        ), "a late retire deleted the successor's marker"

    def test_a_scoped_retire_still_removes_its_own_marker(self, config_dir: Path) -> None:
        """Scoping must not break the ordinary quit: our own marker goes."""
        _make_session(config_dir, "aaaaaaaaaaaa")
        env = {"ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}
        backend = ZellijBackend()
        assert backend.publish(_binding("aaaaaaaaaaaa"), env) is True
        assert backend.retire(_binding("aaaaaaaaaaaa"), env) is True
        assert self._marker_session(config_dir, "zellij-main-0.json") is None

    def test_an_unreadable_marker_is_never_deleted(self, config_dir: Path) -> None:
        """Corrupt-or-foreign reads as not-ours, because the feed is a DELETE."""
        _make_session(config_dir, "aaaaaaaaaaaa")
        env = {"ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}
        marker = config_dir / "multiplexer" / "zellij-main-0.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{not json", encoding="utf-8")
        assert ZellijBackend().retire(_binding("aaaaaaaaaaaa"), env) is False
        assert marker.exists(), "an unidentifiable marker must not be removed"

    def test_tmux_retire_reads_back_and_scopes_to_its_own_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The option backend scopes the same way, via `display-message` readback."""
        options: dict[str, str] = {}

        def fake_run(argv: list[str]) -> bool:
            # ``-u`` puts the option name at a different index than a set does,
            # so the fake mirrors the real argv shapes: set is ``... <option>
            # <value>`` and unset is ``... -u <option>``.
            if "-u" in argv:
                options.pop(argv[-1], None)
            else:
                options[argv[-2]] = argv[-1]
            return True

        monkeypatch.setattr("local_operator.multiplexer.markers._run", fake_run)
        monkeypatch.setattr(
            "local_operator.multiplexer.markers._capture",
            lambda argv: options.get("@lop_session"),
        )
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/tmux")
        env = {"TMUX": "/tmp/tmux-501/default,123,0", "TMUX_PANE": "%3"}
        backend = TmuxBackend()

        # Successor published first: the pane holds another session's binding.
        options["@lop_session"] = "bbbbbbbbbbbb"
        assert backend.retire(_binding("aaaaaaaaaaaa"), env) is False
        assert "@lop_session" in options, "a foreign clear deleted the fresh binding"

        # Our own binding: cleared, both halves.
        options["@lop_session"] = "aaaaaaaaaaaa"
        assert backend.retire(_binding("aaaaaaaaaaaa"), env) is True
        assert "@lop_session" not in options

    def test_a_wezterm_clear_cannot_scope_and_relies_on_sequencing(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """wezterm's write-only user vars make scoping impossible there.

        This pins WHY the swap is sequenced rather than only scoped: without
        the predecessor wait, this backend's unconditional clear deletes the
        successor's binding every time. The sequencing is exercised end to
        end in the TUI swap test; this asserts the property it protects.
        """
        _make_session(config_dir, "aaaaaaaaaaaa")
        _make_session(config_dir, "bbbbbbbbbbbb")
        options: dict[str, str] = {}

        def fake_run(argv: list[str]) -> bool:
            # wezterm has no unset: a clear is a set with an empty value, so
            # the fake mirrors that shape (empty value = absent).
            if len(argv) > 6 and argv[6] == "":
                options.pop(argv[5], None)
            else:
                options[argv[5]] = argv[6]
            return True

        monkeypatch.setattr("local_operator.multiplexer.markers._run", fake_run)
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/wezterm")
        env = {"WEZTERM_PANE": "7"}

        class SlowRetireWezTerm(WezTermBackend):
            def retire(self, binding: Any, env: Any) -> bool:
                time.sleep(TestSwapOrdering.RETIRE_LATENCY_S)
                return super().retire(binding, env)

        backend = SlowRetireWezTerm()
        outgoing = SessionBroadcast(_binding("aaaaaaaaaaaa"), backend, env=env, interval_s=3600.0)
        outgoing.start()
        deadline = time.monotonic() + 5.0
        while options.get("@lop_session") != "aaaaaaaaaaaa":
            assert time.monotonic() < deadline, "outgoing binding never published"
            time.sleep(0.005)

        outgoing.stop(retire=True)
        incoming = SessionBroadcast(
            _binding("bbbbbbbbbbbb"),
            backend,
            env=env,
            interval_s=3600.0,
            predecessor=outgoing,
        )
        incoming.start()
        incoming.join(timeout=10.0)
        outgoing.join(timeout=10.0)
        assert options.get("@lop_session") == "bbbbbbbbbbbb"


class TestCleanExitWinsOverReassert:
    def test_retire_prevents_a_later_republish(self, config_dir: Path) -> None:
        """The inverted-retirement bug: quitting must not resurrect a binding."""
        _make_session(config_dir, "abc123abc123")
        events: list[str] = []

        class Backend:
            name = "fake"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                events.append("publish")
                return True

            def retire(self, binding: Any, env: Any) -> bool:
                events.append("retire")
                return True

        broadcast = SessionBroadcast(_binding(), Backend(), env={})
        broadcast.stop(retire=True)
        # The retire runs on a worker now (it must never block the event
        # loop), so settle it before asserting on the call order.
        broadcast.join(timeout=5.0)
        broadcast._publish_once()
        assert events == ["retire"]

    def test_stop_without_retire_leaves_the_binding_standing(self, config_dir: Path) -> None:
        """A crash path keeps the binding: that IS the feature."""
        _make_session(config_dir, "abc123abc123")
        events: list[str] = []

        class Backend:
            name = "fake"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                return True

            def retire(self, binding: Any, env: Any) -> bool:
                events.append("retire")
                return True

        broadcast = SessionBroadcast(_binding(), Backend(), env={})
        broadcast.stop(retire=False)
        broadcast.join(timeout=5.0)
        assert events == []

    def test_the_timer_thread_does_not_outlive_stop(self, config_dir: Path) -> None:
        """No timer may survive the app; the thread is joined on stop."""
        _make_session(config_dir, "abc123abc123")

        class Backend:
            name = "fake"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                return True

            def retire(self, binding: Any, env: Any) -> bool:
                return True

        before = threading.active_count()
        broadcast = SessionBroadcast(_binding(), Backend(), env={}, interval_s=0.05)
        broadcast.start()
        broadcast.stop()
        # `stop` no longer joins (that would stall the event loop), so the
        # settle is explicit here. The property under test is unchanged: no
        # timer survives the broadcast.
        broadcast.join(timeout=5.0)
        assert threading.active_count() <= before

    def test_retire_session_tolerates_none(self) -> None:
        retire_session(None)  # must not raise

    def test_stop_does_not_block_the_caller_on_a_wedged_backend(self, config_dir: Path) -> None:
        """`stop` runs on the Textual event loop, so it may never wait.

        Both callers (`_adopt_session` on a `/new` swap, `on_unmount` on quit)
        are on the loop. Before this was fixed, `stop` called `retire`
        synchronously and then joined the timer, blocking ~9.8s against a
        multiplexer whose socket is mid-restart — exactly the post-crash
        window the feature targets.
        """
        _make_session(config_dir, "abc123abc123")
        release = threading.Event()
        retired = threading.Event()

        class WedgedBackend:
            name = "wedged"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                # Mirrors a call parked in `subprocess.run(timeout=...)`.
                release.wait(timeout=30.0)
                return False

            def retire(self, binding: Any, env: Any) -> bool:
                release.wait(timeout=30.0)
                retired.set()
                return True

        broadcast = SessionBroadcast(_binding(), WedgedBackend(), env={}, interval_s=90.0)
        broadcast.start()
        # Let the immediate publish get INTO the wedged call, so stop() is
        # racing a genuinely stuck backend rather than an idle one.
        time.sleep(0.2)

        started = time.perf_counter()
        broadcast.stop(retire=True)
        elapsed = time.perf_counter() - started

        # Generous bound: the point is "does not wait on the socket at all",
        # and the pre-fix path took ~9.8s here.
        assert elapsed < 1.0, f"stop() blocked the caller for {elapsed:.2f}s"
        # And the withdrawal is still genuinely dispatched, just not awaited.
        release.set()
        broadcast.join(timeout=10.0)
        assert retired.is_set()

    def test_the_retire_latch_is_set_before_stop_returns(self, config_dir: Path) -> None:
        """Moving the retire off-thread must not weaken the clean-exit rule.

        The latch — not the completion of the retire call — is what stops a
        queued re-assert resurrecting a binding the user quit out of, so it
        has to be set synchronously even though the call is not.
        """
        _make_session(config_dir, "abc123abc123")
        block = threading.Event()
        events: list[str] = []

        class SlowRetire:
            name = "slow"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                events.append("publish")
                return True

            def retire(self, binding: Any, env: Any) -> bool:
                block.wait(timeout=30.0)
                events.append("retire")
                return True

        broadcast = SessionBroadcast(_binding(), SlowRetire(), env={})
        broadcast.stop(retire=True)
        # The retire has NOT run yet (still blocked), and a re-assert landing
        # in this window must still refuse to publish.
        broadcast._publish_once()
        assert events == []
        block.set()
        broadcast.join(timeout=10.0)
        assert events == ["retire"]

    def test_a_failing_publish_backs_off_to_the_reassert_cadence(self, config_dir: Path) -> None:
        """A refusing backend must not pin the loop to the 5s poll.

        The cadence keys on RESUMABILITY, not on publish success. Keying it on
        success meant a resumable session whose publish kept failing spawned a
        `cmux rpc` subprocess every 5s forever, in every pane.
        """
        _make_session(config_dir, "abc123abc123")
        attempts: list[float] = []

        class AlwaysFails:
            name = "failing"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                attempts.append(time.perf_counter())
                return False

            def retire(self, binding: Any, env: Any) -> bool:
                return True

        # An interval far larger than the pending poll, so any extra attempt
        # can only come from the fast cadence.
        broadcast = SessionBroadcast(_binding(), AlwaysFails(), env={}, interval_s=3600.0)
        broadcast.start()
        try:
            # Long enough for several _PENDING_POLL_S windows to elapse.
            time.sleep(_PENDING_POLL_S * 2 + 1.5)
        finally:
            broadcast.stop()
            broadcast.join(timeout=10.0)

        # Exactly one: the immediate publish at start. Everything after it is
        # governed by the hour-long re-assert interval.
        assert len(attempts) == 1, f"failing publish retried {len(attempts)}x on the fast poll"

    def test_a_cold_session_still_polls_until_its_transcript_appears(
        self, config_dir: Path
    ) -> None:
        """The back-off must not cost a cold session its fast poll.

        This is the counterpart to the test above: the fast cadence still
        exists, and it is what makes a binding appear seconds after the first
        turn persists rather than an interval later.
        """
        session_id = "abc123abc123"
        published: list[str] = []

        class Backend:
            name = "fake"

            def detect(self, env: Any) -> bool:
                return True

            def publish(self, binding: Any, env: Any) -> bool:
                published.append(binding.session_id)
                return True

            def retire(self, binding: Any, env: Any) -> bool:
                return True

        broadcast = SessionBroadcast(_binding(), Backend(), env={}, interval_s=3600.0)
        broadcast.start()
        try:
            time.sleep(0.3)
            assert published == []  # no transcript yet
            # The first turn lands mid-flight, as it does in a real session.
            _make_session(config_dir, session_id)
            time.sleep(_PENDING_POLL_S + 1.5)
            assert published == [session_id]
        finally:
            broadcast.stop()
            broadcast.join(timeout=10.0)

    def test_the_withdrawal_lands_before_interpreter_exit(self, tmp_path: Path) -> None:
        """F8: a deliberate quit must actually withdraw, not just decide to.

        The retire worker is a daemon thread, and daemon threads are killed at
        interpreter exit without running what is left of their target. Before
        the exit drain, `stop(retire=True)` on quit returned in microseconds
        having only STARTED the worker — the process then exited, the
        withdrawal never ran, and the pane kept advertising a session the user
        had deliberately closed until their next shell replayed it.

        Run in a SUBPROCESS because the property IS process death: an
        in-process test would have to exit to observe it.
        """
        import subprocess
        import sys

        child = textwrap.dedent(
            """
            import os, sys, time
            from pathlib import Path
            sys.path.insert(0, {repo!r})
            os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = {config!r}
            from local_operator.multiplexer.broadcast import (
                SessionBroadcast, build_binding,
            )
            from local_operator.multiplexer.markers import ZellijBackend

            class SlowRetireZellij(ZellijBackend):
                # One subprocess spawn's worth of latency, as every real
                # backend retire pays; before the drain this was all it took
                # for interpreter exit to eat the withdrawal.
                def retire(self, binding, env):
                    time.sleep(0.05)
                    return super().retire(binding, env)

            env = {{"ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}}
            # The session must be resumable or nothing publishes; the child
            # has to create it itself because the fixture's monkeypatching
            # does not cross the process boundary.
            session = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / "sessions" / "cccccccccccc"
            session.mkdir(parents=True, exist_ok=True)
            # The trailing newline is spelled \\n in the source: a real one
            # inside this literal would break dedent's common-prefix
            # computation and the child would arrive still-indented.
            (session / "transcript.jsonl").write_text("{{}}\\n")
            broadcast = SessionBroadcast(
                build_binding("cccccccccccc"), SlowRetireZellij(),
                env=env, interval_s=3600.0,
            )
            broadcast.start()
            path = None
            from local_operator.multiplexer.markers import marker_dir
            deadline = time.monotonic() + 5.0
            while not (marker_dir() / "zellij-main-0.json").exists():
                assert time.monotonic() < deadline
                time.sleep(0.005)
            # The clean quit, exactly as on_unmount performs it: stop and
            # return, letting the process end. Nothing else may run.
            broadcast.stop(retire=True)
            """.format(
                repo=str(Path(__file__).resolve().parent.parent.parent),
                config=str(tmp_path),
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, timeout=60
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

        marker = tmp_path / "multiplexer" / "zellij-main-0.json"
        assert not marker.exists(), (
            "the withdrawal never ran: a cleanly-quit session is still " "advertised in its pane"
        )

    def test_a_wedged_retire_bounds_the_quit_delay(self, tmp_path: Path) -> None:
        """The exit drain is bounded: a wedged socket delays quit, it cannot hang it.

        The drain exists to land the F8 withdrawal; this pins its other half —
        the worst case it may add to a deliberate quit. The bound is generous
        against the 2s budget (interpreter startup is inside the measurement)
        and far under the 30s a wedged backend would otherwise add.
        """
        import subprocess
        import sys

        child = textwrap.dedent(
            """
            import os, sys, time
            sys.path.insert(0, {repo!r})
            os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = {config!r}
            from local_operator.multiplexer.broadcast import (
                SessionBroadcast, build_binding,
            )

            class Wedged:
                name = "wedged"
                def detect(self, env): return True
                def publish(self, binding, env): return True
                def retire(self, binding, env):
                    time.sleep(30.0)   # a socket that never answers
                    return True

            broadcast = SessionBroadcast(
                build_binding("abc123abc123"), Wedged(), env={{}}, interval_s=3600.0
            )
            broadcast.start()
            time.sleep(0.2)
            broadcast.stop(retire=True)
            """.format(
                repo=str(Path(__file__).resolve().parent.parent.parent),
                config=str(tmp_path),
            )
        )
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, timeout=60
        )
        elapsed = time.monotonic() - started
        assert completed.returncode == 0, completed.stderr[-2000:]
        # 2s drain budget + interpreter startup + scheduling slack.
        assert elapsed < 10.0, f"a wedged socket delayed quit by {elapsed:.1f}s"


class TestReassertInterval:
    def test_it_exceeds_the_cmux_index_cache_ttl(self) -> None:
        """cmux caches its live-agent index for 60s and retirement is one-way.

        A re-assert faster than that TTL would re-publish against the SAME
        stale snapshot: it would cost a subprocess and fix nothing, leaving
        the binding permanently retired.
        """
        assert REASSERT_INTERVAL_S > 60.0


class TestCmuxPayload:
    def test_it_publishes_agent_hook_with_auto_resume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`source: agent-hook` + `auto_resume` is the ONLY path to auto.

        A `cli`-sourced binding resolves to manual/false in cmux, so this is
        pinned: a future simplification to the `cmux surface resume set` CLI
        would leave auto-resume silently dead.
        """
        sent: dict[str, Any] = {}

        def fake_rpc(binary: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
            sent["method"] = method
            sent["params"] = params
            return {"resume_binding": {"auto_resume": True}}

        monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: "/bin/cmux")
        monkeypatch.setattr("local_operator.multiplexer.cmux._rpc", fake_rpc)
        env = {"CMUX_WORKSPACE_ID": WORKSPACE, "CMUX_SURFACE_ID": SURFACE}
        assert CmuxBackend().publish(_binding(), env) is True
        assert sent["method"] == "surface.resume.set"
        params = sent["params"]
        assert params["source"] == "agent-hook"
        assert params["auto_resume"] is True
        assert params["kind"] == "local-operator"
        assert params["checkpoint_id"] == "abc123abc123"
        assert params["workspace_id"] == WORKSPACE
        assert params["surface_id"] == SURFACE
        # The structured launch command must name the launcher, not python.
        assert params["launch_command"]["executable_path"].endswith("lop")
        assert params["launch_command"]["arguments"][1] == "--resume"

    def test_the_shell_fallback_command_is_quoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cmux re-tokenises the `command` string, so a launcher path with a
        space must survive the round trip. The marker backends already use
        `shlex.join` for the same reason; this is the one that did not."""
        sent: dict[str, Any] = {}

        def fake_rpc(binary: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
            sent["params"] = params
            return {"resume_binding": {"auto_resume": True}}

        monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: "/bin/cmux")
        monkeypatch.setattr("local_operator.multiplexer.cmux._rpc", fake_rpc)
        spaced = "/Applications/My Tools/lop"
        binding = SessionBinding(
            session_id="abc123abc123",
            executable=spaced,
            argv=(spaced, RESUME_FLAG, "abc123abc123"),
            cwd="/work",
        )
        env = {"CMUX_WORKSPACE_ID": WORKSPACE, "CMUX_SURFACE_ID": SURFACE}
        assert CmuxBackend().publish(binding, env) is True
        command = sent["params"]["command"]
        assert "'/Applications/My Tools/lop'" in command
        # The round trip a shell (and cmux's canonicalizer) performs.
        assert shlex.split(command) == [spaced, RESUME_FLAG, "abc123abc123"]

    def test_the_clear_is_scoped_to_our_own_binding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Expectations, so a quit cannot wipe another agent's binding."""
        sent: dict[str, Any] = {}

        def fake_rpc(binary: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
            sent["method"] = method
            sent["params"] = params
            return {"cleared": True}

        monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: "/bin/cmux")
        monkeypatch.setattr("local_operator.multiplexer.cmux._rpc", fake_rpc)
        env = {"CMUX_WORKSPACE_ID": WORKSPACE, "CMUX_SURFACE_ID": SURFACE}
        assert CmuxBackend().retire(_binding(), env) is True
        assert sent["method"] == "surface.resume.clear"
        assert sent["params"]["checkpoint_id"] == "abc123abc123"
        assert sent["params"]["source"] == "agent-hook"
        assert sent["params"]["agent_session_ended"] is True

    @pytest.mark.parametrize(
        "env",
        [
            {},
            {"CMUX_WORKSPACE_ID": WORKSPACE},
            {"CMUX_SURFACE_ID": SURFACE},
            {"CMUX_WORKSPACE_ID": "not-a-uuid", "CMUX_SURFACE_ID": SURFACE},
        ],
    )
    def test_an_incomplete_or_malformed_target_publishes_nothing(
        self, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inherited-but-stale CMUX_* (a container, an ssh hop) is not a surface."""
        monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: "/bin/cmux")
        assert CmuxBackend().detect(env) is False
        assert CmuxBackend().publish(_binding(), env) is False


class TestMarkerBackends:
    def test_tmux_sets_both_pane_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "local_operator.multiplexer.markers._run", lambda argv: calls.append(argv) or True
        )
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/tmux")
        env = {"TMUX": "/tmp/tmux-501/default,123,0", "TMUX_PANE": "%3"}
        assert TmuxBackend().publish(_binding(), env) is True
        assert [c[4] for c in calls] == ["%3", "%3"]
        assert calls[0][5] == SESSION_OPTION
        assert calls[0][6] == "abc123abc123"
        assert calls[1][5] == COMMAND_OPTION
        assert "--resume abc123abc123" in calls[1][6]

    def test_tmux_unsets_rather_than_blanking_on_retire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank option reads as a session with an empty id, not as absence."""
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "local_operator.multiplexer.markers._run", lambda argv: calls.append(argv) or True
        )
        # The readback our own scoping performs: the pane is assumed to hold
        # THIS binding, so the retire proceeds to the unset under test.
        monkeypatch.setattr(
            "local_operator.multiplexer.markers._capture",
            lambda argv: _binding().session_id,
        )
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/tmux")
        env = {"TMUX": "/tmp/tmux-501/default,123,0", "TMUX_PANE": "%3"}
        assert TmuxBackend().retire(_binding(), env) is True
        assert all("-u" in call for call in calls)

    def test_tmux_needs_a_live_server_not_just_a_pane_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TMUX_PANE survives into a process that has left tmux behind."""
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/tmux")
        assert TmuxBackend().detect({"TMUX_PANE": "%3"}) is False

    def test_the_command_marker_is_shell_quoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A restore script hands this to a shell; a spacey path must survive."""
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "local_operator.multiplexer.markers._run", lambda argv: calls.append(argv) or True
        )
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/tmux")
        spaced = SessionBinding(
            session_id="abc123abc123",
            executable="/Applications/My Tools/lop",
            argv=("/Applications/My Tools/lop", "--resume", "abc123abc123"),
            cwd="/work",
        )
        env = {"TMUX": "/tmp/tmux-501/default,123,0", "TMUX_PANE": "%3"}
        TmuxBackend().publish(spaced, env)
        assert "'/Applications/My Tools/lop'" in calls[1][6]

    def test_zellij_writes_a_state_file(self, config_dir: Path) -> None:
        env = {"ZELLIJ": "0", "ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}
        assert ZellijBackend().publish(_binding(), env) is True
        marker = config_dir / "multiplexer" / "zellij-main-0.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["session_id"] == "abc123abc123"
        assert payload["command"][1] == "--resume"

    def test_zellij_removes_the_file_on_retire(self, config_dir: Path) -> None:
        env = {"ZELLIJ": "0", "ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "0"}
        ZellijBackend().publish(_binding(), env)
        assert ZellijBackend().retire(_binding(), env) is True
        assert not (config_dir / "multiplexer" / "zellij-main-0.json").exists()

    def test_two_zellij_panes_in_one_session_do_not_share_a_marker(self, config_dir: Path) -> None:
        """One file per pane, or a pane's exit deletes its siblings' bindings.

        `ZELLIJ_SESSION_NAME` names a SESSION. Verified on zellij 0.42.2 that
        two panes of one session differ only in `ZELLIJ_PANE_ID` (0 and 1), so
        keying on the session name alone gave both panes one marker: the last
        publisher won and the first pane to exit cleanly unlinked the other
        pane's binding.
        """
        base = {"ZELLIJ": "0", "ZELLIJ_SESSION_NAME": "main"}
        pane0 = {**base, "ZELLIJ_PANE_ID": "0"}
        pane1 = {**base, "ZELLIJ_PANE_ID": "1"}

        assert ZellijBackend().publish(_binding("aaaaaaaaaaaa"), pane0) is True
        assert ZellijBackend().publish(_binding("bbbbbbbbbbbb"), pane1) is True

        markers = sorted(p.name for p in (config_dir / "multiplexer").glob("zellij-*.json"))
        assert markers == ["zellij-main-0.json", "zellij-main-1.json"]

        # Neither pane's binding was overwritten by the other's.
        expected_markers = (
            ("zellij-main-0.json", "aaaaaaaaaaaa"),
            ("zellij-main-1.json", "bbbbbbbbbbbb"),
        )
        for name, expected in expected_markers:
            payload = json.loads((config_dir / "multiplexer" / name).read_text(encoding="utf-8"))
            assert payload["session_id"] == expected

        # And pane 0 quitting cleanly leaves pane 1 still advertising.
        assert ZellijBackend().retire(_binding("aaaaaaaaaaaa"), pane0) is True
        assert not (config_dir / "multiplexer" / "zellij-main-0.json").exists()
        assert (config_dir / "multiplexer" / "zellij-main-1.json").is_file()

    def test_zellij_without_a_pane_id_publishes_nothing(self, config_dir: Path) -> None:
        """No coarser fallback: a silent collision is worse than no marker."""
        env = {"ZELLIJ": "0", "ZELLIJ_SESSION_NAME": "main"}
        assert ZellijBackend().detect(env) is False
        assert ZellijBackend().publish(_binding(), env) is False

    def test_screen_uses_the_sty_and_window_identity(self, config_dir: Path) -> None:
        env = {"STY": "1234.pts-0.host", "WINDOW": "2"}
        assert ScreenBackend().publish(_binding(), env) is True
        assert (config_dir / "multiplexer" / "screen-1234.pts-0.host-2.json").is_file()

    def test_two_screen_windows_in_one_session_do_not_share_a_marker(
        self, config_dir: Path
    ) -> None:
        """Same collision as zellij's, same fix. Verified on screen 4.00.03
        that a process inside a session exports both `STY` and `WINDOW`."""
        window0 = {"STY": "1234.pts-0.host", "WINDOW": "0"}
        window1 = {"STY": "1234.pts-0.host", "WINDOW": "1"}
        assert ScreenBackend().publish(_binding("aaaaaaaaaaaa"), window0) is True
        assert ScreenBackend().publish(_binding("bbbbbbbbbbbb"), window1) is True
        markers = sorted(p.name for p in (config_dir / "multiplexer").glob("screen-*.json"))
        assert len(markers) == 2

    def test_screen_falls_back_to_sty_when_window_is_absent(self, config_dir: Path) -> None:
        """`WINDOW` comes from the shell's startup rather than from screen in
        every configuration, and one window per session still restores fine."""
        env = {"STY": "1234.pts-0.host"}
        assert ScreenBackend().publish(_binding(), env) is True
        assert (config_dir / "multiplexer" / "screen-1234.pts-0.host.json").is_file()

    def test_a_pane_id_that_could_escape_the_directory_is_refused(self, config_dir: Path) -> None:
        """The id becomes a FILENAME; writing the right data to the wrong
        pane is worse than writing nothing."""
        env = {"STY": "../../etc/passwd"}
        assert ScreenBackend().detect(env) is False
        assert ScreenBackend().publish(_binding(), env) is False

    def test_every_component_of_a_composite_id_is_validated(self, config_dir: Path) -> None:
        """Each part is checked separately, not the joined string: `-` is legal
        inside a component, so validating only the result would let one
        component forge another pane's filename."""
        env = {"ZELLIJ": "0", "ZELLIJ_SESSION_NAME": "../../etc", "ZELLIJ_PANE_ID": "0"}
        assert ZellijBackend().detect(env) is False
        assert ZellijBackend().publish(_binding(), env) is False
        env = {"ZELLIJ": "0", "ZELLIJ_SESSION_NAME": "main", "ZELLIJ_PANE_ID": "../x"}
        assert ZellijBackend().detect(env) is False
        assert ZellijBackend().publish(_binding(), env) is False

    def test_no_multiplexer_env_means_no_detection(self) -> None:
        for backend in (TmuxBackend(), ZellijBackend(), WezTermBackend(), ScreenBackend()):
            assert backend.detect({}) is False


class TestRegistryOrdering:
    def test_cmux_wins_over_a_nested_tmux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiplexers nest; the OUTERMOST host is the one that restores panes."""
        monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: "/bin/cmux")
        monkeypatch.setattr("local_operator.multiplexer.markers._which", lambda b: "/bin/tmux")
        env = {
            "CMUX_WORKSPACE_ID": WORKSPACE,
            "CMUX_SURFACE_ID": SURFACE,
            "TMUX": "/tmp/tmux-501/default,123,0",
            "TMUX_PANE": "%3",
        }
        backend = active_backend(env)
        assert backend is not None and backend.name == "cmux"


class TestBuildBinding:
    def test_an_empty_session_id_builds_nothing(self) -> None:
        assert build_binding("") is None

    def test_it_records_the_working_directory(self) -> None:
        binding = build_binding("abc123abc123", cwd="/work")
        assert binding is not None
        assert binding.cwd == "/work"
        assert binding.argv[1] == "--resume"

    def test_no_environment_or_secrets_ride_along(self) -> None:
        """Session id and cwd only: this reaches a shell-adjacent surface."""
        binding = build_binding("abc123abc123", cwd="/work")
        assert binding is not None
        assert set(vars(binding)) == {"session_id", "executable", "argv", "cwd", "name"}


def test_importing_the_package_does_not_drag_in_the_engine() -> None:
    """The package must stay cheap enough to sit on a startup path.

    Checked in a FRESH interpreter, because by the time this test body runs
    pytest has already imported most of the tree and an in-process
    `sys.modules` assertion would pass on a real regression. Same reasoning as
    `test_import_graph.py`, which pins the CLI's own graph.
    """
    import subprocess
    import sys

    probe = (
        "import json, importlib, sys;"
        "importlib.import_module('local_operator.multiplexer');"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    modules = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    # The engine, the providers and the TUI are what this must not pull in:
    # publication is bookkeeping, and it is reached from startup.
    assert "local_operator.harness" not in modules
    assert "local_operator.tui.app" not in modules
    assert "local_operator.tools.builtin" not in modules
    assert "asyncio" not in modules
