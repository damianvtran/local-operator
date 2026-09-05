"""OperatorApp Pilot tests — boot, prompt dispatch, slash commands, quit.

Uses a ``FakeSession`` implementing ``SessionProtocol`` so the TUI runs
without providers/network. The factory shape mirrors production: the app
paints first, then awaits the session in a worker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from local_operator.harness.types import (
    AgentMessage,
    ImageContent,
    NoticeEvent,
    TextContent,
    ToolResult,
)
from local_operator.paths import config_dir
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import (
    BOOT_LAYOUT_CLASS,
    COMPOSER_FOCUSED_CLASS,
    COMPOSER_SHELL_CLASS,
    MODEL_SWITCH_MID_TURN_NOTICE,
    PERSIST_HINT,
    PROMPT_CHEVRON,
    SHELL_CHEVRON,
    SLASH_COMMANDS,
    OperatorApp,
    _splash_toast_headline,
    _TerminalFrontendReaper,
)
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.events import TurnEnded, TurnStarted
from local_operator.tui.widgets.approval import ApprovalPrompt
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import (
    ASIDE_PLACEHOLDER,
    DEFAULT_PLACEHOLDER,
    SHELL_PLACEHOLDER,
    Editor,
)
from local_operator.tui.widgets.session_picker import SessionPickerScreen
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    RichBlock,
    TranscriptView,
    UserBlock,
)
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.conftest import caret_cells, chevron_colour, composer_cells


def _set_editor_line(editor, text: str) -> None:
    """Set the buffer AND park the caret at the end, as typing it would.

    ``editor.text = x`` leaves the caret at ``(0, 0)``; a user who typed ``x``
    has it at the end, and the slash pickers are caret-anchored (inline
    detection: which slash token is live depends on where the caret is). A test
    that sets the text without moving the caret asserts about a state the UI
    never produces — the caret sitting before the slash, where no command is
    being edited. This is the faithful shortcut for "the user typed this line".
    """
    editor.text = text
    editor.move_cursor(editor._end_of_buffer())
    editor._sync_picker()


class _FakeJobs:
    """The slice of ``AsyncJobManager`` the app reads: ``list()`` of rows.

    Rows are derived from ``running_children`` so a test states the fact it
    cares about ("two children are up") instead of assembling job models.

    ``running_bash_jobs`` does the same for backgrounded shell work, which the
    stop ladder deliberately SPARES and now names when confirming a stop — so a
    test needs to be able to say "a build is also running" as plainly.
    """

    def __init__(self, session: "FakeSession") -> None:
        self._session = session

    def list(self) -> list[Any]:
        rows = [
            SimpleNamespace(id=f"job{i}", status="running", type="task", queued=False)
            for i in range(self._session.running_children)
        ]
        rows += [
            SimpleNamespace(id=f"bash{i}", status="running", type="bash", queued=False)
            for i in range(self._session.running_bash_jobs)
        ]
        return rows


def test_idle_mounted_terminal_never_reaps() -> None:
    """Idle duration is irrelevant while the owning TUI reader is alive."""
    reaper = _TerminalFrontendReaper(grace_s=10.0)
    assert (
        reaper.observe(
            reader_alive=True, busy=False, gate_pending=False, remote_holders=False, now=0.0
        )
        is None
    )
    assert (
        reaper.observe(
            reader_alive=True,
            busy=False,
            gate_pending=False,
            remote_holders=False,
            now=10_000.0,
        )
        is None
    )
    assert reaper.quiescent_since is None


def test_frontend_gone_busy_defers_reap_until_work_finishes() -> None:
    reaper = _TerminalFrontendReaper(grace_s=10.0)
    reaper.observe(reader_alive=True, busy=False, gate_pending=False, remote_holders=False, now=0.0)
    assert (
        reaper.observe(
            reader_alive=False, busy=True, gate_pending=False, remote_holders=False, now=100.0
        )
        is None
    )
    assert reaper.quiescent_since is None
    assert (
        reaper.observe(
            reader_alive=False, busy=False, gate_pending=False, remote_holders=False, now=110.0
        )
        is None
    )
    assert (
        reaper.observe(
            reader_alive=False, busy=False, gate_pending=False, remote_holders=False, now=119.9
        )
        is None
    )
    assert (
        reaper.observe(
            reader_alive=False,
            busy=False,
            gate_pending=False,
            remote_holders=False,
            now=120.0,
        )
        == "exit"
    )


def test_frontend_gate_gets_grace_then_fresh_exit_grace() -> None:
    reaper = _TerminalFrontendReaper(grace_s=3.0)
    reaper.observe(reader_alive=True, busy=False, gate_pending=False, remote_holders=False, now=0.0)
    assert (
        reaper.observe(
            reader_alive=False, busy=False, gate_pending=True, remote_holders=False, now=1.0
        )
        is None
    )
    assert (
        reaper.observe(
            reader_alive=False,
            busy=False,
            gate_pending=True,
            remote_holders=False,
            now=30.9,
        )
        is None
    )
    assert (
        reaper.observe(
            reader_alive=False,
            busy=True,
            gate_pending=True,
            remote_holders=False,
            now=31.0,
        )
        == "settle"
    )
    reaper.gates_settled()
    assert (
        reaper.observe(
            reader_alive=False, busy=False, gate_pending=False, remote_holders=False, now=31.0
        )
        is None
    )
    assert (
        reaper.observe(
            reader_alive=False,
            busy=False,
            gate_pending=False,
            remote_holders=False,
            now=34.0,
        )
        == "exit"
    )


class _FrontendRegistrant:
    def __init__(
        self,
        *,
        watch_supported: bool,
        phone_watchers: int = 0,
        attach_clients: int = 0,
    ) -> None:
        self.watch_supported = watch_supported
        self.phone_watchers = phone_watchers
        self._attach_clients = attach_clients

    def attach_clients(self) -> int:
        return self._attach_clients


@pytest.mark.asyncio
@pytest.mark.parametrize("work", ["model", "tool", "compaction", "subagent"])
async def test_terminal_loss_settles_gate_without_exiting_during_active_work(work: str) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    app._terminal_frontend_reaper = _TerminalFrontendReaper(grace_s=3.0, reader_seen=True)
    approval = ApprovalPrompt("bash", "run a command")
    app._approval = approval
    session = FakeSession()
    app._session = session
    if work in ("model", "tool"):
        session.streaming = True
    elif work == "compaction":
        app._compacting = True
    else:
        session.running_children = 1
    exits: list[bool] = []
    with (
        patch.object(app, "_terminal_reader_alive", return_value=False),
        patch.object(app, "exit", side_effect=lambda: exits.append(True)),
        patch("local_operator.tui.app.time.monotonic", side_effect=[0.0, 100.0]),
    ):
        app._check_terminal_frontend()
        app._check_terminal_frontend()
    # The 30-second authority deadline is independent of unrelated activity;
    # settling it does not abort that activity and therefore cannot exit.
    assert approval.answered is True
    assert exits == []


@pytest.mark.asyncio
async def test_pending_gate_waits_for_reconnect_then_settles_and_reaps_after_work() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    app._terminal_frontend_reaper = _TerminalFrontendReaper(grace_s=3.0, reader_seen=True)
    approval = ApprovalPrompt("bash", "run a command")
    app._approval = approval
    session = FakeSession()
    app._session = session
    exits: list[bool] = []
    readers = iter([False, True, False, False, False, False, False])
    times = iter([0.0, 2.0, 3.0, 33.0, 34.0, 40.0, 43.0])

    def deny_and_resume() -> None:
        approval.resolve(False, answer="n")
        session.streaming = True

    with (
        patch.object(app, "_terminal_reader_alive", side_effect=lambda: next(readers)),
        patch.object(app, "_deny_queued_approvals", side_effect=deny_and_resume),
        patch.object(app, "_settle_ask_picker"),
        patch.object(app, "_settle_key_prompt"),
        patch.object(app, "exit", side_effect=lambda: exits.append(True)),
        patch("local_operator.tui.app.time.monotonic", side_effect=lambda: next(times)),
    ):
        app._check_terminal_frontend()  # gate grace starts
        app._check_terminal_frontend()  # reconnect preserves the gate
        assert approval.answered is False
        app._check_terminal_frontend()  # replacement grace starts
        app._check_terminal_frontend()  # grace expires and resumes work
        assert approval.answered is True
        app._check_terminal_frontend()  # resumed work cannot exit
        session.streaming = False
        app._check_terminal_frontend()  # fresh clean-exit grace starts
        app._check_terminal_frontend()  # only now may the owner exit
    assert exits == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registrant",
    [
        _FrontendRegistrant(watch_supported=True, attach_clients=1),
        _FrontendRegistrant(watch_supported=True, phone_watchers=1),
        _FrontendRegistrant(watch_supported=False),
    ],
)
async def test_terminal_loss_reaps_independently_of_remote_viewers(
    registrant: _FrontendRegistrant,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    app._mobile_registrant = registrant
    app._session = FakeSession()
    app._terminal_frontend_reaper = _TerminalFrontendReaper(grace_s=3.0, reader_seen=True)
    exits: list[bool] = []
    with (
        patch.object(app, "_terminal_reader_alive", return_value=False),
        patch.object(app, "exit", side_effect=lambda: exits.append(True)),
        patch("local_operator.tui.app.time.monotonic", side_effect=[0.0, 3.0]),
    ):
        app._check_terminal_frontend()
        app._check_terminal_frontend()
    assert exits == [True]


class FakeSession:
    """Records prompts/aborts; satisfies SessionProtocol."""

    def context_breakdown(self) -> dict[str, int]:
        return getattr(self, "_context_breakdown", {})

    def __init__(self) -> None:
        self.prompts: list[str] = []
        #: The images each prompt carried, index-aligned with ``prompts``. Kept
        #: so a test can prove a screenshot reached the model through a command
        #: that submits a prompt (``/team``/``/agent``), not just the text.
        self.prompt_images: list[list[Any]] = []
        self.aborts: list[str] = []
        #: Bang-mode commands this fake was asked to persist. A list of
        #: ``(command, result)`` so a test can say the TUI recorded what ran
        #: without standing up a real transcript.
        self.shell_records: list[tuple[str, Any]] = []
        #: Reasons passed to `cancel_subagents`, and how many children the next
        #: call should report stopping. Staged by tests that exercise the Esc
        #: ladder's second press.
        self.subagent_cancels: list[str] = []
        self.running_children = 0
        #: Backgrounded `bash` jobs, which Esc never stops (`background=true`
        #: exists so a build outlives the turn). Staged separately from
        #: `running_children` because the ladder treats them as opposites.
        self.running_bash_jobs = 0
        #: The job ledger the app counts children in. A real manager's rows are
        #: what `_job_count` reads, so the fake presents the same shape rather
        #: than having the app special-case a test double.
        self.jobs = _FakeJobs(self)
        self.completions: list[tuple[str, str]] = []
        self.asides: list[list[Any]] = []
        self.adopted: list[list[Any]] = []
        self.disposed = False
        self._handlers: list[Any] = []
        self._history: list[Any] = []
        #: `/compact` requests this fake was asked for, and the answer it gives.
        #: A fake carries no history, so the honest default is the refusal a
        #: real session returns for an empty conversation; tests that want a
        #: successful pass stage their own outcome.
        self.compactions = 0
        self.compact_outcome = CompactionOutcome(
            ran=False,
            reason="nothing_to_compact",
            detail="nothing to compact: the whole conversation is ~0 tokens",
        )
        self.preflight_calls = 0
        self.preflight_notice: str | None = None
        self._steering_queue: list[Any] = []
        #: Whether a turn is running. Settable because the `/model` receipt now
        #: reads it: a switch made mid-turn says when it starts applying, and a
        #: hard-coded False could never exercise that branch.
        self.streaming = False
        # The REAL holder, not a bare string: `user_set` precedence (a human
        # rename outranks every generated title, forever) is behaviour the TUI
        # relies on, and a fake that reimplements it as a plain assignment
        # would let a regression in that rule pass every pilot test.
        self._name_state = ConversationName()
        #: Optional team registry for ``/team``. Tests that exercise teams
        #: assign a real ``TeamRegistry``; everyone else pays nothing.
        self.team_registry: Any | None = None
        self.attached_teams: list[Any] = []
        #: Optional agent registry for ``/agent``, same opt-in shape as
        #: ``team_registry``: tests that exercise profiles assign a real
        #: ``AgentRegistry``; everyone else pays nothing.
        self.agent_registry: Any | None = None
        #: Names ``attach_agent_profile`` was asked for, so a `/agent` test
        #: can assert the attach happened without a real system-prompt stamp.
        self.attached_agents: list[str] = []
        #: The tail ``attach_agent_profile`` last stamped, read by the TUI's
        #: A2 empty-instructions notice path. Mirrors the real ``agent_brief``.
        self.agent_brief: str = ""
        #: How many times ``clear_agent_profile`` was called, so a `/agent
        #: clear` test can assert the detach reached the session.
        self.cleared_agents: int = 0

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return self.streaming

    @property
    def model_label(self) -> str:
        return "test/model"

    @property
    def model(self) -> Any:
        return None

    @property
    def effective_model(self) -> Any:
        """The fake never falls back, so selection and effective agree —
        which is also what keeps the protocol's degrade honest."""
        return self.model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    def attach_team(self, team: Any) -> None:
        self.attached_teams.append(team)
        # Track the active roster so the band's active-team segment (U2) can be
        # driven through the real accessor path, exactly as the real session
        # exposes it via ``active_team``/``active_team_name``.
        self._active_team = team

    @property
    def active_team_name(self) -> str:
        team = getattr(self, "_active_team", None)
        if team is None:
            return ""
        return str(getattr(team, "name", "") or "")

    @property
    def active_agent(self) -> str:
        # Mirror the real session: the display NAME of the profile in force, ""
        # once cleared. Stamped alongside ``agent_brief`` below so a `/agent`
        # test can assert the band segment as well as the brief (U2).
        return getattr(self, "_active_agent", "")

    def attach_agent_profile(self, name: str) -> str | None:
        # Mirror the real session's resolution ORDER (own role, then own
        # specialist, then packaged seed) and stamp ``agent_brief`` exactly as
        # the real one does, so `/agent` tests exercise the real filtering
        # rules AND the empty-instructions notice path (A2).
        from local_operator.agent_profiles import is_specialist, resolve_profile

        profile = resolve_profile(name, registry=self.agent_registry)
        if profile is not None and profile.agent_id is not None:
            self.agent_brief = profile.preamble.strip()
            self._active_agent = profile.name
            self.attached_agents.append(profile.name)
            return profile.name
        if self.agent_registry is not None:
            agent = self.agent_registry.get_agent_by_name(name)
            if agent is not None and is_specialist(agent):
                prompt = (self.agent_registry.get_agent_system_prompt(agent.id) or "").strip()
                self.agent_brief = f"[agent: {agent.name}]\n{prompt}" if prompt else ""
                self._active_agent = str(agent.name)
                self.attached_agents.append(str(agent.name))
                return str(agent.name)
        if profile is not None:
            self.agent_brief = profile.preamble.strip()
            self._active_agent = profile.name
            self.attached_agents.append(profile.name)
            return profile.name
        return None

    def clear_agent_profile(self) -> None:
        # Mirror the real detach: blank the tail AND the active name (U2), and
        # record the call so a `/agent clear` test can assert both the effect
        # and that it happened.
        self.agent_brief = ""
        self._active_agent = ""
        self.cleared_agents += 1

    @property
    def variables(self) -> Any:
        """Memory-only store for ``/credential``. Created on first use so
        tests that never touch credentials pay nothing for the property."""
        store = getattr(self, "_variables", None)
        if store is None:
            from local_operator.variables import VariableStore

            store = self._variables = VariableStore(cwd="/tmp", env={})
        return store

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    async def preflight_usage(self) -> None:
        self.preflight_calls += 1
        if self.preflight_notice is not None:
            self.emit(NoticeEvent(text=self.preflight_notice, kind="warning"))

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)
        # Parallel to ``prompts``: the pixels the turn carried, so a test can
        # assert an image survived a route (e.g. `/team <name> <request>` with
        # a pasted screenshot) rather than only that the words did.
        self.prompt_images.append(list(images or []))

    async def record_shell(self, command: str, result: Any) -> None:
        self.shell_records.append((command, result))

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    # The recall seam, mirroring `Session.steer_message`/`recall_steering`:
    # the TUI's Esc lifts a queued steer back into the composer, and tests
    # for it drive these exactly as the real session's queue would.
    def queued_steering(self) -> list[Any]:
        return list(self._steering_queue)

    def steer_message(self, message: Any) -> None:
        self._steering_queue.append(message)

    def recall_steering(self, message: AgentMessage) -> bool:
        for index, held in enumerate(self._steering_queue):
            if held is message:
                del self._steering_queue[index]
                return True
        return False

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def set_ask_handler(self, handler: object | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot, and that
        # install is what makes the tool exist; fakes only need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def running_subagents(self) -> int:
        """The count the stop ladder offers. Staged by tests via
        ``running_children``, mirroring the real session's single predicate."""
        return self.running_children

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """Record the wider stop and report how many children it stopped.

        Reports the SAME number ``running_subagents`` would, which is the
        invariant the real session guarantees by sharing one predicate.
        """
        self.subagent_cancels.append(reason)
        stopped = self.running_children
        self.running_children = 0
        return stopped

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    @property
    def conversation_name(self) -> str:
        return self._name_state.text

    @property
    def conversation_name_state(self) -> ConversationName:
        return self._name_state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return self._name_state.set(text, user_set=user_set)

    async def complete_once(self, system: str, prompt: str) -> str:
        # No title: the naming worker must be inert in the pilot tests, and
        # an empty completion is exactly the "model said nothing usable"
        # path that generate_title resolves to None.
        self.completions.append((system, prompt))
        return ""

    async def dispose(self) -> None:
        self.disposed = True

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Recorded, not answered. The aside's own suites drive the real
        # Session; here the only contract that matters is that the app can
        # call it, so a fake that returned prose would invite pilot tests to
        # assert on words no model produced.
        self.asides.append(list(turns))
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        self.adopted.append(list(messages))

    async def compact_now(self) -> CompactionOutcome:
        return self._answer_compaction()

    def _answer_compaction(self) -> CompactionOutcome:
        """Count the request and answer with the staged outcome.

        Split out so subclasses can override :meth:`compact_now` (to raise, or
        to block) without losing the count every test reads.
        """
        self.compactions += 1
        return self.compact_outcome

    def history(self) -> list[Any]:
        return getattr(self, "_history", [])

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)


async def _factory(session: FakeSession) -> FakeSession:
    return session


def _renderable_plain(renderable) -> str:
    """Recursively flatten a Rich renderable (incl. Group/Padding) to text."""
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_renderable_plain(child) for child in renderable.renderables)
    if isinstance(renderable, Padding):
        return _renderable_plain(renderable.renderable)
    if isinstance(renderable, str):
        return renderable
    return ""


def _transcript_text(app) -> str:
    transcript = app.query_one(TranscriptView)
    parts = []
    for b in transcript.blocks():
        parts.append(_renderable_plain(getattr(b, "renderable", "")))
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_boot_typing_sends_prompt() -> None:
    """Boot the app, type text, press Enter: the session records the prompt."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("h", "i")
        await pilot.press("enter")
        # Poll until the prompt is actually recorded rather than betting two
        # frames is enough: under parallel CPU load the submit worker had not
        # yet reached the session when a fixed tick count expired.
        for _ in range(200):
            await pilot.pause()
            if session.prompts:
                break
        assert session.prompts == ["hi"]
        assert session.preflight_calls == 1
        # A user block was appended for the submitted prompt (the boot hint
        # is lifted by the first real block, D9).
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 1


async def _await_session(app: Any, pilot: Any) -> None:
    """Pause until the boot task has actually built the session.

    Same hazard as ``_await_setup_state`` below, on the other branch of boot.
    ``app._session`` is set by a worker task, and a command that reaches the
    app before it lands is REFUSED ("session is still starting…") rather than
    queued — so a test that sends a slash command after a single
    ``pilot.pause()`` asserts on the refusal's after-state and fails only
    under load. Two of these were caught in CI shards on separate runs
    (`/model default`, `/model saved`) while passing in isolation every time.

    Polling for the state the test depends on removes the timing from the
    assertion while still failing, on timeout, if the session never arrives.
    """
    for _ in range(200):
        await pilot.pause()
        if getattr(app, "_session", None) is not None:
            return


async def _await_setup_state(app: Any, pilot: Any) -> None:
    """Pause until the boot task has actually reached the setup state.

    A fixed number of pauses plus a sleep is not enough: boot runs as a worker
    task, and under a loaded runner (xdist, or simply a filtered selection that
    packs these cases together) it routinely needs longer than the budget. That
    made the assertion a race that passed when the test ran alone and failed
    when it ran beside others -- observed on origin/main before this change,
    not introduced by it. Polling for the state the test is about removes the
    timing from the assertion while still failing (on timeout) if the state is
    never reached.
    """
    for _ in range(100):
        await pilot.pause()
        if app._setup_state:
            return
        await asyncio.sleep(0.05)
    await pilot.pause()


@pytest.mark.asyncio
async def test_first_run_setup_state_when_hosting_unconfigured() -> None:
    """A boot that fails with HostingNotConfiguredError enters the guided setup
    state (splash notice + 'setup' band), NOT a red 'session failed' (item 1)."""
    from local_operator.session_factory import HostingNotConfiguredError

    async def _no_hosting_factory():
        raise HostingNotConfiguredError("Hosting platform is not configured.")

    app = OperatorApp(_no_hosting_factory, provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        # The setup flag is set and the band says "setup", not a model or error.
        assert app._setup_state is True
        assert app._status is not None
        assert app._status._model_label == "setup"
        # The splash carries the guided /login notice, not a failure.
        assert app._splash_notice is not None
        assert "/login" in app._splash_notice
        assert "no provider configured" in app._splash_notice
        # The splash was NOT retired: it is still the empty-state block.
        assert app._welcome_visible is True


@pytest.mark.asyncio
async def test_setup_state_when_hosting_names_an_unknown_provider() -> None:
    """A corrupted `hosting:` value lands in the SAME recoverable setup state.

    The hotfix regression guard. A typo'd provider used to reach the TUI as a
    bare ValueError out of `configure_model`, which painted a red "session
    failed to start: Unsupported hosting platform: anthropicxyq" and left
    `_session` None -- so `/model`, `/provider` and the rest all answered
    "session is still starting..." and the user could not switch away from the
    bad value without hand-editing YAML outside the app.
    """
    from local_operator.session_factory import HostingUnknownError

    async def _bad_hosting_factory():
        raise HostingUnknownError(
            "Hosting 'anthropicxyq' ... not a known provider.", "anthropicxyq"
        )

    app = OperatorApp(_bad_hosting_factory, provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        # Setup state, not a dead session: the band must not say "session error".
        assert app._setup_state is True
        assert app._status is not None
        assert app._status._model_label == "setup"
        # The bad value is remembered so the post-login rebuild can overwrite it.
        assert app._invalid_hosting == "anthropicxyq"
        assert app._splash_notice is not None
        notice = app._splash_notice
        # Action-first: the splash truncates from the right on a narrow
        # terminal, so the remedy must precede the diagnosis or it is what
        # drops. Asserted by POSITION, not just presence.
        assert notice.index("/login") < notice.index("anthropicxyq")
        # LENGTH, not just order. The first version of this line ran to 141
        # characters, so at 80 and 100 columns it was cut mid-clause and at 60
        # the bad value never rendered at all -- the user was told to fix
        # "this" without being told what "this" was.
        #
        # What this number actually buys (D6): the notice row paints into
        # `terminal width - 6` cells, not `width - 2`. MEASURED against the real
        # widget across widths 74..95, not derived from the "! " prefix: a
        # 77-char notice first renders whole at 83 columns and a 78-char one
        # needs 84. So 78 is a REGROWTH CEILING (it fails the 141-char
        # regression it was written for), NOT a promise of 80-column survival --
        # the budget for that is 74, and the line asserted here is 77, so it is
        # cut by three cells on an 80-column terminal today. Stated plainly
        # because the previous comment claimed the opposite, which is how a
        # future notice written to "the limit" would be silently truncated while
        # this suite stayed green. New notices should target 74; see
        # `test_repaired_config_boots_into_an_escapable_state`, which holds its
        # own line to that.
        assert len(notice) <= 78, f"notice must not regrow past the 84-col ceiling: {len(notice)}"
        # The value must be inside the part that survives the narrowest
        # terminal we capture (60 cols), which is the whole point of naming it.
        assert notice.index("anthropicxyq") + len("anthropicxyq") <= 58
        # It names the offending value and does not read as a crash.
        assert "anthropicxyq" in notice
        assert "not a known provider" in notice
        assert "failed" not in notice.lower()
        # It must NOT claim nothing is configured -- something is, just wrongly.
        assert "no provider configured" not in notice
        assert app._welcome_visible is True


@pytest.mark.asyncio
async def test_setup_state_does_not_promise_a_login_it_cannot_honour() -> None:
    """A bad value from --hosting or an agent record must not recommend /login.

    Precedence is agent > flag > config, but `/login` writes the CONFIG FILE.
    Recommending it for a value that came from argv or an agent record is a
    loop the user cannot see the shape of: the login succeeds, writes config,
    the next boot resolves the same flag/agent value, and the app returns to
    setup having claimed it was fixed.
    """
    from local_operator.session_factory import HostingUnknownError

    async def _flag_hosting_factory():
        raise HostingUnknownError("…not a known provider.", "anthropicxyq", "flag")

    app = OperatorApp(_flag_hosting_factory, provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        notice = app._splash_notice
        assert notice is not None
        # Names the real source, and does NOT tell the user to run /login.
        assert "--hosting" in notice
        assert "anthropicxyq" in notice
        assert "/login" not in notice
        # Regrowth ceiling, not an 80-column guarantee -- see the measured note
        # on the sibling assertion above for what this number does and does not
        # buy (D6).
        assert len(notice) <= 78


@pytest.mark.asyncio
async def test_login_from_bad_provider_setup_state_repairs_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The RECOVERY half: `/login` from the bad-provider setup state must write.

    Landing in setup state is only half a fix. `_apply_login_defaults` adopts a
    provider ONLY when hosting is empty, and in this state it is not empty --
    it is wrong -- so without the repair branch the login wrote nothing, the
    rebuild resolved the same bad value, and the user was told "starting
    session..." before dropping straight back into setup.
    """
    from local_operator.config import ConfigManager
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session_factory import HostingUnknownError

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    seed = ConfigManager(tmp_path)
    seed.set_config_value("hosting", "anthropicxyq")
    seed.set_config_value("model_name", "claude-sonnet-4-5")

    async def _bad_hosting_factory():
        raise HostingUnknownError(
            "Hosting 'anthropicxyq' ... not a known provider.", "anthropicxyq"
        )

    app = OperatorApp(_bad_hosting_factory, provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        assert app._setup_state is True

        # The write half of the `/login` flow, run against the real config.
        receipt = app._apply_login_defaults("deepseek")

    assert receipt is not None
    assert "deepseek" in receipt
    repaired = ConfigManager(tmp_path)
    assert repaired.get_config_value("hosting") == "deepseek"
    # The model belonged to the provider that was replaced, so it is replaced
    # too -- otherwise a real provider is pointed at a model that never existed.
    assert repaired.get_config_value("model_name") == "deepseek-chat"


@pytest.mark.parametrize(
    "provider, expect_setup",
    [
        # The two loginable providers with no default model: `/login` here
        # writes a registry-VALID hosting with a CLEARED model, which is the
        # config the repair produces on purpose.
        ("alibaba-token-plan", True),
        ("alibaba-token-plan-oauth", True),
        # Control: an ordinary provider brings its own default, so the repaired
        # config boots straight through and must NOT land in setup. Without it
        # this test would pass on a build that sent every login to setup.
        ("deepseek", False),
    ],
)
@pytest.mark.asyncio
async def test_repaired_config_boots_into_an_escapable_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    expect_setup: bool,
) -> None:
    """A repaired config must BOOT, and its setup state must be escapable (B1).

    The coverage gap this closes: every other assertion about the repair stops
    at the plan or at the config file, and none asserted that the config the
    repair writes can actually start a session. `alibaba-token-plan` has no
    default model, so the repair clears the model deliberately — and the
    resolver used to answer that with a plain `ValueError`, which misses the
    recoverable-error gate in `_on_boot_failed` and painted the red "session
    failed to start" with `_session=None` AND `_setup_state=False`. That is the
    exact terminal state this PR exists to remove, reached by following the
    PR's own remedy, so the user was stuck HARDER after the repair than before.

    Asserted end to end (repair -> config -> boot -> recovery command) because
    that is the only sequence in which the defect is visible: each step in
    isolation looked correct, which is precisely how it shipped.

    The DEFECT is what is pinned, not the mechanism: the assertions are "not the
    dead state" and "the user can get out", so a different recoverable shape
    still passes and a regression to a dead end cannot.
    """
    from local_operator.config import ConfigManager
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers.login_defaults import plan_login_defaults
    from local_operator.session_factory import resolve_hosting_model

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    seed = ConfigManager(tmp_path)
    seed.set_config_value("hosting", "anthropicxyq")
    seed.set_config_value("model_name", "claude-sonnet-4-5")

    # The repair the app performs on a successful `/login`, through the real
    # planner rather than a hand-written config: the point is that this exact
    # pair is what the product writes.
    plan = plan_login_defaults(
        provider, seed.get_config_value("hosting"), seed.get_config_value("model_name")
    )
    assert plan.hosting is not None
    seed.set_config_value("hosting", plan.hosting)
    if plan.model_name is not None:
        seed.set_config_value("model_name", plan.model_name)

    # The REAL startup resolver decides what the next boot does, so the boot
    # error under test is the one the resolver actually raises.
    boot_error: Exception | None = None
    try:
        resolve_hosting_model(
            None, argparse.Namespace(hosting=None, model=None), ConfigManager(tmp_path)
        )
    except Exception as exc:  # noqa: BLE001 — the condition under test
        boot_error = exc

    if not expect_setup:
        assert boot_error is None, f"control provider must boot: {boot_error}"
        return

    assert boot_error is not None
    session = FakeSession()

    async def _factory():
        # Re-reads config every call, so the rebuild after the recovery sees
        # what the recovery wrote rather than a value captured at construction.
        if not ConfigManager(tmp_path).get_config_value("model_name"):
            raise boot_error
        return session

    app = OperatorApp(_factory, provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)

        # NOT the dead state. Both halves are asserted: `_setup_state` False
        # with `_session` None is the red branch, and it is the conjunction
        # that made every recovery command answer "session is still starting…".
        assert app._setup_state is True
        assert app._status is not None
        assert app._status._model_label == "setup"
        assert app._status._model_label != "session error"

        notice = app._splash_notice
        assert notice is not None
        # It names the provider and points at the command that can actually
        # fix this. `/login` must NOT be promised: hosting is already
        # registry-valid, so a login writes nothing and the user loops.
        assert provider.startswith("alibaba")
        assert "alibaba-token-plan" in notice
        assert "/model" in notice
        assert "failed" not in notice.lower()
        # Budget check, same rule as the sibling splash assertions. The notice
        # row paints into `terminal width - 6` cells (the splash's gutter and
        # the "! " glyph), MEASURED against the real widget, so a line of 74
        # renders whole at 80 columns and one of 78 needs 84. The guard is 74
        # rather than 78 so a notice written to the old number is caught here
        # instead of being silently cut on the terminal width most people use.
        assert len(notice) <= 74, f"notice must survive an 80-col terminal: {len(notice)}"

        # THE SUBSTANCE: the state must be escapable. A setup state you cannot
        # leave is the same bug wearing a different colour.
        app._cmd_model("deepseek/deepseek-chat", lambda body, kind="info": None)
        for _ in range(200):
            await pilot.pause()
            if app._session is not None:
                break
            await asyncio.sleep(0.02)

        assert app._session is session, "the recovery command must BUILD the session"
        assert app._setup_state is False

    # The escape persisted the pair, so the next launch does not return here.
    recovered = ConfigManager(tmp_path)
    assert recovered.get_config_value("hosting") == "deepseek"
    assert recovered.get_config_value("model_name") == "deepseek-chat"


@pytest.mark.asyncio
async def test_missing_model_is_recoverable_not_fatal_on_every_surface() -> None:
    """The no-model error must carry the recoverable ancestry, not bare ValueError.

    The type is the fix: `_on_boot_failed` and the CLI preflight both classify by
    `isinstance`, so a plain `ValueError` here bypassed the guided setup state no
    matter how good its message was. Asserted at the type level as well as
    through the app because that ancestry is load-bearing for two callers, and a
    later "simplification" back to `ValueError` would silently restore the dead
    end.
    """
    from local_operator.session_factory import (
        HostingNotConfiguredError,
        HostingUnknownError,
        ModelNotConfiguredError,
        resolve_hosting_model,
    )

    class _Cfg:
        def get_config_value(self, key, default=None):
            return {"hosting": "alibaba-token-plan", "model_name": ""}.get(key, default)

    with pytest.raises(ModelNotConfiguredError) as caught:
        resolve_hosting_model(
            None,
            argparse.Namespace(hosting=None, model=None),
            cast(Any, _Cfg()),
        )

    error = caught.value
    # Recoverable family, so the setup-state gate picks it up …
    assert isinstance(error, HostingNotConfiguredError)
    # … but DISTINCT from the unknown-provider case, whose wording would tell
    # the user to fix the one part of their config that is correct.
    assert not isinstance(error, HostingUnknownError)
    # Legacy `except ValueError` callers keep working.
    assert isinstance(error, ValueError)
    # The informative text survives: it names concrete model ids, which is what
    # the non-interactive paths print.
    assert "alibaba-token-plan" in str(error)
    assert "gpt-4o" in str(error)
    assert error.hosting == "alibaba-token-plan"


def test_non_interactive_preflight_still_fails_fast_without_a_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Recoverable in the TUI must NOT mean permissive in a script.

    A scripted or CI run has nobody to answer a model picker, so limping along
    on a model nobody chose is how a cron job silently bills a different
    provider. `allow_setup_state` is the ONLY thing that separates the two, so
    both directions are asserted here rather than trusting the flag's name.
    """
    from local_operator.cli import _preflight_hosting_model
    from local_operator.config import ConfigManager

    config = ConfigManager(tmp_path)
    config.set_config_value("hosting", "alibaba-token-plan")
    config.set_config_value("model_name", "")
    args = argparse.Namespace(hosting=None, model=None)

    # Non-interactive: fatal, with the message that names concrete model ids.
    result = _preflight_hosting_model(
        config, cast(Any, None), cast(Any, None), None, cast(Any, args)
    )
    assert result == 1
    assert "no default is known" in capsys.readouterr().err

    # Interactive TUI: opens, so the user can reach `/model`.
    assert (
        _preflight_hosting_model(
            config,
            cast(Any, None),
            cast(Any, None),
            None,
            cast(Any, args),
            allow_setup_state=True,
        )
        is None
    )


@pytest.mark.asyncio
async def test_login_outside_setup_state_still_leaves_hosting_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The repair must not leak into the ordinary add-a-second-provider login.

    `_invalid_hosting` is None outside the bad-provider setup state, so a user
    logging into another provider keeps the default they chose.
    """
    from local_operator.config import ConfigManager
    from local_operator.paths import CONFIG_DIR_ENV

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    seed = ConfigManager(tmp_path)
    seed.set_config_value("hosting", "openai")
    seed.set_config_value("model_name", "gpt-4o")

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._invalid_hosting is None
        assert app._apply_login_defaults("deepseek") is None

    unchanged = ConfigManager(tmp_path)
    assert unchanged.get_config_value("hosting") == "openai"
    assert unchanged.get_config_value("model_name") == "gpt-4o"


def test_splash_toast_headline_names_the_fallback_target() -> None:
    """The toast is a glance; the splash row keeps the reason."""
    assert (
        _splash_toast_headline("anthropic quota low (8% remaining) — falling back to zai/glm-5.3")
        == "Fell back to zai/glm-5.3"
    )
    assert (
        _splash_toast_headline(
            "anthropic quota exhausted — falling back to openai/gpt-5.3-codex (high effort)"
        )
        == "Fell back to openai/gpt-5.3-codex (high effort)"
    )
    assert _splash_toast_headline("session failed to start") == "session failed to start"
    assert _splash_toast_headline("") == "Notice"


def test_splash_toast_headline_prefers_an_explicit_headline() -> None:
    """A caller that knows the state names it; no prose sniffing required.

    This is the durable form of the D5 fix. That finding was addressed by
    matching the literal substring "no provider configured", which silently
    stopped applying when the sibling unknown-provider message was added: it
    contains no such phrase, so it fell through to the blind 35-cell cut and
    reproduced the dangling-fragment defect D5 existed to remove. An explicit
    headline cannot regress by being reworded.
    """
    long_line = "Run /login openai — 'anthropicxyq' is not a known provider (/provider lists them)."
    # Without a headline the long line is blind-cut mid-parenthetical.
    assert _splash_toast_headline(long_line).endswith("…")
    # With one, the toast is a clean glance carrying the bad value — and the
    # toast is the one element that is never truncated by terminal width.
    assert (
        _splash_toast_headline(long_line, "Unknown provider 'anthropicxyq'")
        == "Unknown provider 'anthropicxyq'"
    )
    # The explicit headline wins over the legacy substring probe too, so the
    # two setup states are headlined by the same mechanism rather than one
    # each.
    assert (
        _splash_toast_headline("… no provider configured …", "No provider configured")
        == "No provider configured"
    )


@pytest.mark.asyncio
async def test_boot_renders_non_blocking_quota_warning() -> None:
    """A quota fallback is harness news, not a conversation.

    Routing it through the default notice path retired the splash for a
    single yellow line over an empty screen — a launch that looked like
    a conversation that had already started. The splash stays; the
    warning is a toast and a splash row.
    """
    session = FakeSession()
    session.preflight_notice = (
        "anthropic quota exhausted — falling back to openai/gpt-5.3-codex (high effort)"
    )
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            await pilot.pause()
            if app._splash_notice:
                break

        assert session.prompts == []
        welcome = app.query_one(WelcomeView)
        assert welcome.display is True, "a quota notice must not retire the splash"
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert app.query_one(TranscriptView).blocks() == []
        assert "anthropic quota exhausted" in (welcome._info.notice or "")
        assert "falling back to openai/gpt-5.3-codex (high effort)" in (welcome._info.notice or "")
        toast = app.query_one(Toast)
        assert toast.display is True
        # Headline, not the full sentence: the splash row already carries
        # the reason, and repeating it made a two-row card that sat on
        # the mark (design round 1, D1/D2).
        assert toast.message == "Fell back to openai/gpt-5.3-codex (high effort)"
        assert "quota exhausted" not in toast.message


@pytest.mark.asyncio
async def test_a_notice_after_the_conversation_starts_is_a_transcript_row() -> None:
    """Once a user message has retired the splash, a later notice is just
    another line — the toast/splash path is only for the empty state."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("hello"))
        await pilot.pause()
        assert app.query_one(WelcomeView).display is False
        session.emit(NoticeEvent(text="anthropic quota low — falling back", kind="warning"))
        await pilot.pause()
        assert "anthropic quota low" in _transcript_text(app)
        assert app.query_one(WelcomeView).display is False
        assert app.query_one(Toast).display is False


@pytest.mark.asyncio
async def test_exit_command_quits() -> None:
    """``/exit`` handled synchronously quits the app without prompting."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "e", "x", "i", "t", "enter")
        await pilot.pause()
        assert not app.is_running
    assert session.prompts == []


@pytest.mark.asyncio
async def test_quit_alias_quits() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "q", "u", "i", "t", "enter")
        await pilot.pause()
        assert not app.is_running


def test_exit_quit_collapsed_to_one_command() -> None:
    """TUI-014: ONE registry entry; ``quit`` rides as an alias of ``exit``."""
    names = [c.name for c in SLASH_COMMANDS]
    assert "exit" in names
    assert "quit" not in names  # not a separate command
    exit_command = next(c for c in SLASH_COMMANDS if c.name == "exit")
    assert exit_command.aliases == ("quit",)


@pytest.mark.asyncio
async def test_every_authoritative_slash_routes_to_owner_with_supported_images() -> None:
    from local_operator.harness.types import ImageContent
    from local_operator.session.frontend_state import (
        CommandScope,
        FrontendSessionState,
        SlashCapability,
    )
    from local_operator.tui.widgets.editor import Attachment

    class RoutedSession(FakeSession):
        frontend_state: FrontendSessionState

        async def route_shared_slash(self, command: str, args: str, images=()):  # noqa: ANN001
            routed.append((command, args, len(images)))
            return "routed"

    routed: list[tuple[str, str, int]] = []
    session = RoutedSession()
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="owner",
        slash_capabilities=[
            SlashCapability(
                command=command,
                scope=CommandScope.AUTHORITATIVE_SESSION,
                operation="slash",
                supports_images=command in {"agent", "team"},
            )
            for command in (
                "agent",
                "team",
                "loop",
                "compact",
                "goal",
                "model",
                "effort",
                "rename",
                "approvals",
                "btw",
            )
        ],
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        image = ImageContent(data="cG5n", mime_type="image/png")
        attachment = Attachment(image=image, marker="[Image #1]")
        for capability in session.frontend_state.slash_capabilities:
            attachments = {1: attachment} if capability.supports_images else None
            arg = "coder inspect [Image #1]" if attachments else "value"
            app._run_slash_command(f"/{capability.command} {arg}", attachments)
            await pilot.pause()
    assert [name for name, _, _ in routed] == [
        capability.command for capability in session.frontend_state.slash_capabilities
    ]
    assert [(name, count) for name, _, count in routed if name in {"agent", "team"}] == [
        ("agent", 1),
        ("team", 1),
    ]


@pytest.mark.asyncio
async def test_follower_renders_typed_slash_results_locally() -> None:
    """A follower's routed slash renders the owner's typed outcome HERE, not a
    transport receipt — and bare /model opens the invoker's own picker."""
    from local_operator.session.frontend_state import (
        CommandScope,
        FrontendSessionState,
        SlashCapability,
    )
    from local_operator.tui.widgets.transcript import NoticeBlock

    class RoutedSession(FakeSession):
        frontend_state: FrontendSessionState

        async def route_shared_slash(self, command: str, args: str, images=()):  # noqa: ANN001
            routed.append(command)
            if command == "goal":
                return {
                    "kind": "notice",
                    "text": "goal set — applies from the next turn",
                    "style": "info",
                    "data": {"stored": args},
                }
            if command == "mcp":
                return {
                    "kind": "notice",
                    "text": "authenticated MCP server 'linear'; 12 tools available.",
                    "style": "success",
                }
            return {"kind": "noop"}

    routed: list[str] = []
    session = RoutedSession()
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="owner",
        slash_capabilities=[
            SlashCapability(
                command=name, scope=CommandScope.AUTHORITATIVE_SESSION, operation="slash"
            )
            for name in ("goal", "model", "mcp")
        ],
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        # /goal renders the typed notice in the INVOKING transcript.
        app._run_slash_command("/goal ship it")
        for _ in range(60):
            await pilot.pause()
            if any("goal set" in (b.text() or "") for b in app.query(NoticeBlock)):
                break
        assert any(
            "goal set — applies from the next turn" in (b.text() or "")
            for b in app.query(NoticeBlock)
        )
        assert "goal" in routed

        # /mcp login renders the grant receipt locally; no AttributeError.
        app._run_slash_command("/mcp login linear")
        for _ in range(60):
            await pilot.pause()
            if any("authenticated MCP server" in (b.text() or "") for b in app.query(NoticeBlock)):
                break
        assert any("authenticated MCP server" in (b.text() or "") for b in app.query(NoticeBlock))

        # Bare /model opens the invoker's own picker; nothing routes.
        routed.clear()
        app._run_slash_command("/model")
        await pilot.pause()
        await pilot.pause()
        editor = app.query_one(Editor)
        assert editor.text.startswith("/model ")
        assert routed == []


@pytest.mark.asyncio
async def test_follower_mcp_grant_routes_instead_of_crashing_on_snapshot_manager() -> None:
    """The follower's MCP facade is read-only (no get_server_config), so a grant
    subcommand must ROUTE to the owner rather than reach the local handler —
    the exact AttributeError crash from review U1/BLOCKER-1."""
    from local_operator.session.frontend_state import (
        CommandScope,
        FrontendSessionState,
        SlashCapability,
        SnapshotMcpManager,
        _slash_capabilities,
    )

    class RoutedSession(FakeSession):
        frontend_state: FrontendSessionState
        # The read-only facade the production RemoteSession hands the app — the
        # one whose get_server_config absence crashed the local /mcp handler.
        mcp_manager: Any

        async def route_shared_slash(self, command: str, args: str, images=()):  # noqa: ANN001
            routed.append((command, args))
            return {
                "kind": "notice",
                "text": "authenticated MCP server 'linear'; 3 tools available.",
                "style": "success",
            }

    routed: list[tuple[str, str]] = []
    session = RoutedSession()
    session.mcp_manager = SnapshotMcpManager()
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="owner",
        slash_capabilities=[
            SlashCapability(
                command="mcp", scope=CommandScope.AUTHORITATIVE_SESSION, operation="slash"
            )
        ],
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        for subcommand in ("login linear", "logout linear", "reauth linear"):
            app._run_slash_command(f"/mcp {subcommand}")
            await pilot.pause()
        for _ in range(60):
            await pilot.pause()
            if len(routed) == 3:
                break
        # All three grant subcommands routed; none reached the crashing local
        # handler. The bare listing stays local (renders, does not route).
        assert routed == [
            ("mcp", "login linear"),
            ("mcp", "logout linear"),
            ("mcp", "reauth linear"),
        ]
        routed.clear()
        app._run_slash_command("/mcp")
        await pilot.pause()
        await pilot.pause()
        assert routed == []

        # Round 5, MAJOR: against a full-capability owner the bare-/mcp
        # pullback sets remote_capability = None, and the U10 refusal guard
        # used to read that deliberate pullback as "unadvertised" — every
        # production follower lost the listing behind a refusal notice. The
        # carve-out must render the canonical LOCAL listing: no refusal, no
        # route, and the MCP servers block actually mounted in the transcript.
        from local_operator.mcp.config import MCPStdioServerConfig

        full_capabilities = FrontendSessionState(
            session_id=session.session_id,
            epoch="owner",
            slash_capabilities=_slash_capabilities(),
        )
        session.frontend_state = full_capabilities
        for _ in range(20):
            await pilot.pause()
            if getattr(app._session, "frontend_state", None) is full_capabilities:
                break
        configs = {"linear": MCPStdioServerConfig(command="linear-mcp")}
        with patch("local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})):
            app._run_slash_command("/mcp")
            for _ in range(20):
                await pilot.pause()
                if app.query(RichBlock):
                    break
        listing_blocks = [b for b in app.query(RichBlock)]
        assert listing_blocks, "bare /mcp must mount the canonical local listing block"
        listing = "\n".join(_renderable_plain(b.renderable) for b in listing_blocks)
        assert "MCP servers" in listing
        assert "linear" in listing
        assert not any(
            "not available from this session's owner" in (b.text() or "")
            for b in app.query(NoticeBlock)
        )
        assert routed == []

        # The grant form still routes authoritatively under the full list.
        with patch("local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})):
            app._run_slash_command("/mcp login linear")
            for _ in range(60):
                await pilot.pause()
                if any(
                    "authenticated MCP server" in (b.text() or "") for b in app.query(NoticeBlock)
                ):
                    break
        assert routed == [("mcp", "login linear")]
        assert any("authenticated MCP server" in (b.text() or "") for b in app.query(NoticeBlock))


@pytest.mark.asyncio
async def test_unadvertised_shared_slash_refuses_instead_of_running_locally() -> None:
    """U10 (UX round 4): a shared command the owner's capability list omits
    must not silently run on the follower — it would mutate only this
    process and never reach the owner."""
    from local_operator.session.frontend_state import (
        CommandScope,
        FrontendSessionState,
        SlashCapability,
    )

    class RoutedSession(FakeSession):
        frontend_state: FrontendSessionState

        async def route_shared_slash(self, command: str, args: str, images=()):  # noqa: ANN001
            routed.append(command)
            return {"kind": "noop"}

    routed: list[str] = []
    session = RoutedSession()
    # The owner advertises /rename as authoritative but NOT /goal — the
    # version-skew/reduced-owner shape U10 walked.
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="owner",
        slash_capabilities=[
            SlashCapability(
                command="rename", scope=CommandScope.AUTHORITATIVE_SESSION, operation="slash"
            )
        ],
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        app._run_slash_command("/goal ship it")
        for _ in range(60):
            await pilot.pause()
            if any(
                "not available from this session's owner" in (b.text() or "")
                for b in app.query(NoticeBlock)
            ):
                break
        # The shared slash was REFUSED in clear copy — it did not route (the
        # owner never advertised it) and it did not run locally (the
        # follower's goal is untouched).
        assert any(
            "not available from this session's owner" in (b.text() or "")
            for b in app.query(NoticeBlock)
        )
        assert routed == []
        assert session.goal == ""

        # A command the owner DID advertise still routes.
        app._run_slash_command("/rename New title")
        for _ in range(60):
            await pilot.pause()
            if routed:
                break
        assert routed == ["rename"]

        # A follower-local command (not in the capability list at all) still
        # runs locally — the guard only refuses SHARED commands.
        app._run_slash_command("/help")
        await pilot.pause()


@pytest.mark.asyncio
async def test_owner_slash_result_producers_match_local_handler_vocabulary() -> None:
    """The owner's typed-result producers say what the local handlers say —
    goal, rename, context, effort — so a follower's receipt is identical."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        goal = await app.run_slash_authoritative("goal", "ship it")
        assert goal["kind"] == "notice"
        assert "goal set — applies from the next turn" in goal["text"]
        assert session.goal == "ship it"

        bare_goal = await app.run_slash_authoritative("goal", "")
        assert "goal: ship it" in bare_goal["text"]

        renamed = await app.run_slash_authoritative("rename", "New title")
        assert "renamed: New title" in renamed["text"]
        assert session.conversation_name == "New title"

        unknown = await app.run_slash_authoritative("mcp", "bogus x")
        assert "unknown mcp subcommand" in unknown["text"]
        assert unknown["style"] == "warning"


@pytest.mark.asyncio
async def test_every_authoritative_capability_has_a_real_owner_producer() -> None:
    """MAJOR-1 (round 2): no routed command may answer success without acting.

    The capability builder marks every non-frontend-local slash authoritative;
    the owner's typed dispatcher must therefore implement every one of them —
    a hand-picked subset is how ``/loop``, ``/compact``, ``/approvals`` and
    non-bare ``/model`` answered ``ran /…`` while doing nothing. This walks
    the REAL capability list so a new command without a producer fails here.
    """
    from local_operator.session.frontend_state import CommandScope, _slash_capabilities

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        authoritative = [
            cap.command
            for cap in _slash_capabilities()
            if cap.scope is CommandScope.AUTHORITATIVE_SESSION
        ]
        assert authoritative, "the capability list must exist"
        for command in authoritative:
            if command == "context":
                # The context producer needs a REAL breakdown shape; the fake
                # session reports none, which is itself a legitimate outcome.
                session._context_breakdown = {  # type: ignore[attr-defined]
                    "instructions": 1,
                    "tool_inventory": 1,
                    "tool_schemas": 1,
                    "environment": 1,
                    "knowledge_mcp_goal": 1,
                    "messages": 1,
                    "total": 6,
                    "context_window": 100,
                    "cache_read": 0,
                }
            outcome = await app.run_slash_authoritative(command, "bogus value")
            text = str(outcome.get("text", ""))
            # The old fallback receipt is a false success: any producer that
            # still answers it is an unimplemented command.
            assert (
                text != f"ran /{command}"
            ), f"/{command} has no owner producer — it reported success without acting"


@pytest.mark.asyncio
async def test_routed_loop_compact_approvals_and_model_act_on_the_owner() -> None:
    """MAJOR-1 (round 2): the four round-2 commands execute, not no-op."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        # /approvals auto: the OWNER's gate mode actually changes.
        outcome = await app.run_slash_authoritative("approvals", "auto")
        assert app._approve_all is True
        assert "tool approvals: auto" in outcome["text"]
        outcome = await app.run_slash_authoritative("approvals", "ask")
        assert app._approve_all is False
        # Bare /approvals reports the live mode honestly.
        report = await app.run_slash_authoritative("approvals", "")
        assert "tool approvals: ask" in report["text"]

        # /compact: the owner's session is asked for a REAL pass.
        session.compact_outcome = CompactionOutcome(ran=True, detail="done")
        outcome = await app.run_slash_authoritative("compact", "")
        for _ in range(60):
            await pilot.pause()
            if session.compactions:
                break
        assert session.compactions == 1, "the compact pass must actually run"
        assert "compacting context" in outcome["text"]

        # /loop with no goal refuses BEFORE starting; with a goal it runs a
        # real iteration through the session (the worker drives prompts).
        refused = await app.run_slash_authoritative("loop", "2")
        assert "set a goal first" in refused["text"]
        session.set_goal("ship it")
        outcome = await app.run_slash_authoritative("loop", "1")
        assert "looping toward the goal" in outcome["text"]
        for _ in range(200):
            await pilot.pause()
            if not app._loop_running:
                break
        assert session.prompts, "the loop must drive at least one real turn"

        # Non-bare /model: the owner's session switches for real.
        switched = await app.run_slash_authoritative("model", "openrouter/deepseek/deepseek-chat")
        assert "model:" in switched["text"]
        assert switched["kind"] == "notice"
        # /model default is DECLINED, not silently applied on the wrong machine.
        declined = await app.run_slash_authoritative("model", "default openrouter/x")
        assert declined["style"] == "warning"
        assert "persists to the local machine" in declined["text"]


@pytest.mark.asyncio
async def test_a_failed_remote_cancel_never_prints_a_confirmed_success() -> None:
    """MAJOR-2 (round 2): the -1 failure sentinel renders an honest retry line."""
    from local_operator.session.frontend_state import FrontendSessionState
    from local_operator.tui.widgets.transcript import NoticeBlock

    class RemoteishSession(FakeSession):
        is_remote = True
        frontend_state: FrontendSessionState

        def __init__(self) -> None:
            super().__init__()
            self._resolver: Any | None = None

        def set_cancel_resolution(self, resolver: Any | None) -> None:
            self._resolver = resolver

    session = RemoteishSession()
    session.frontend_state = FrontendSessionState(session_id="sess", epoch="owner")
    session.streaming = True
    session.running_children = 2
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        # The owner call FAILS: the resolver the second press installed gets
        # the failure sentinel, exactly as RemoteSession._resolve_cancel
        # delivers it on a socket/owner exception. Staged after the presses
        # rather than inside cancel_subagents so the app's own resolver is
        # what receives it (the real seam is asynchronous).
        resolver = session._resolver
        assert resolver is not None, "the second press must arm the resolution seam"
        resolver(-1)
        # The failure resolution is one hop past the press: the offer frame
        # paints "stopping 2 subagents…" first, then the sentinel rewrites
        # it. Wait for the rewrite, not a fixed number of pauses.
        for _ in range(100):
            await pilot.pause()
            texts = [b.text() or "" for b in app.query(NoticeBlock)]
            if any("could not confirm" in text for text in texts):
                break
        texts = [b.text() or "" for b in app.query(NoticeBlock)]
        assert any("could not confirm" in text for text in texts), texts
        assert not any("stopped 2 subagents" in text for text in texts), texts


@pytest.mark.asyncio
async def test_frontend_update_burst_coalesces_to_latest_snapshot() -> None:
    """Queued canonical updates repaint once, from the newest complete state."""
    from local_operator.session.frontend_state import FrontendSessionState

    class StatefulSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.frontend_state = FrontendSessionState(session_id="sess", epoch="owner")

    session = StatefulSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        scheduled: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []
        applied: list[FrontendSessionState] = []

        def schedule(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
            scheduled.append((callback, args))
            return True

        def apply(state: Any) -> None:
            applied.append(cast(FrontendSessionState, state))

        app.call_later = schedule
        app._apply_frontend_state = apply

        session.frontend_state = session.frontend_state.model_copy(
            update={"conversation_title": "first"}
        )
        app._on_frontend_update(object())
        session.frontend_state = session.frontend_state.model_copy(
            update={"conversation_title": "latest"}
        )
        app._on_frontend_update(object())

        assert len(scheduled) == 1
        callback, args = scheduled[0]
        callback(*args)
        assert [state.conversation_title for state in applied] == ["latest"]


@pytest.mark.asyncio
async def test_raw_subagent_progress_waits_for_canonical_band_update() -> None:
    """Owner progress has the same single visual invalidation as a follower."""
    from local_operator.session.frontend_state import FrontendSessionState, JobState
    from local_operator.tui.events import SubagentProgress

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        refreshes = 0

        def refreshed() -> None:
            nonlocal refreshes
            refreshes += 1

        app._refresh_band = refreshed
        app.on_subagent_progress(SubagentProgress("child", "child", "reading files"))
        assert refreshes == 0

        # The canonical apply is the owner/follower-common route and owns the
        # one repaint after the manager's <=50 ms coalescing window.
        app._apply_frontend_state(
            FrontendSessionState(
                session_id="sess",
                epoch="owner",
                jobs=[
                    JobState(
                        id="child",
                        type="task",
                        status="running",
                        latest_details={"progress": "reading files"},
                    )
                ],
            )
        )
        assert refreshes == 1


@pytest.mark.asyncio
async def test_pending_frontend_update_cannot_repaint_after_session_adoption() -> None:
    """An old queued callback is retired before a replacement snapshot paints."""
    from local_operator.session.frontend_state import FrontendSessionState

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        scheduled: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []
        applied: list[str] = []

        def schedule(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
            scheduled.append((callback, args))
            return True

        app.call_later = schedule
        app._apply_frontend_state = lambda state: applied.append(state.conversation_title)

        class StatefulSession(FakeSession):
            frontend_state: FrontendSessionState

        old = StatefulSession()
        old.frontend_state = FrontendSessionState(
            session_id="old", epoch="old", conversation_title="old"
        )
        app._session = old
        app._on_frontend_update(object())
        assert len(scheduled) == 1

        app._invalidate_pending_frontend_state()
        replacement = FakeSession()
        app._session = replacement
        app._apply_frontend_state(
            FrontendSessionState(session_id="new", epoch="new", conversation_title="new")
        )
        callback, args = scheduled[0]
        callback(*args)

        assert applied == ["new"]


@pytest.mark.asyncio
async def test_takeover_preserves_the_status_band_verbatim() -> None:
    """D2 (round 2): a transport rotation must not move the band's segments."""
    from local_operator.session.frontend_state import (
        CostKnowledge,
        FrontendSessionState,
    )

    state = FrontendSessionState(
        session_id="sess",
        epoch="owner",
        conversation_title="Unified session parity",
        active_agent="coder",
        active_team="lopdev",
        cumulative_parent_cost=3.92,
        cost_knowledge=CostKnowledge.FLOOR,
        context_tokens=402_000,
        context_is_estimate=False,
        context_window=1_000_000,
    )

    class StatefulSession(FakeSession):
        def __init__(self, st: FrontendSessionState) -> None:
            super().__init__()
            self._st = st

        @property
        def frontend_state(self) -> FrontendSessionState:
            return self._st

        @property
        def active_agent(self) -> str:
            return self._st.active_agent

        @property
        def active_team_name(self) -> str:
            return self._st.active_team

        @property
        def conversation_name(self) -> str:
            return self._st.conversation_title

    class Remoteish(StatefulSession):
        is_remote = True

        def set_takeover_callback(self, callback: Any) -> None:
            self.takeover_callback = callback

    remote = Remoteish(state)
    replacement = StatefulSession(state.model_copy(update={"epoch": "new-owner"}))
    app = OperatorApp(lambda: _factory(remote))
    async with app.run_test(size=(118, 36)) as pilot:
        for _ in range(60):
            await pilot.pause()
            if app._session is remote:
                break
        status = app._status
        assert status is not None
        # A REAL follower boots with the canonical cost already painted (the
        # sync snapshot installs it before the first frame); a fake without
        # the frontend-subscribe seam has to be settled into that state
        # explicitly, or the test would measure boot cost ≠ post-takeover
        # cost and pass on a band that started WRONG rather than stayed right.
        app._apply_frontend_state(remote.frontend_state)
        await pilot.pause()
        before = {
            "agent_profile": status._agent_profile,
            "team": status._team,
            "conversation_name": status._conversation_name,
            "cost": status._cost,
        }
        assert before["cost"] == "≥$3.92", before
        await app._adopt_takeover_session(replacement)
        for _ in range(5):
            await pilot.pause()
        after = {
            "agent_profile": status._agent_profile,
            "team": status._team,
            "conversation_name": status._conversation_name,
            "cost": status._cost,
        }
        assert after == before, f"the band must not shift through takeover: {before} → {after}"


@pytest.mark.asyncio
async def test_follower_model_picker_lists_the_owners_models() -> None:
    """D3 (round 2): the picker's rows come from the OWNER's catalogue, so a
    credential-less follower's bare /model still lists selectable models."""
    from local_operator.session.frontend_state import FrontendSessionState

    class OwnerCatalogueSession(FakeSession):
        frontend_state: FrontendSessionState

        def owner_model_catalogue(self) -> list[dict[str, Any]]:
            return [
                {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-6",
                    "label": "Claude Sonnet 4.6",
                    "context_window": 1_000_000,
                    "input_price": 3.0,
                    "output_price": 15.0,
                    "connected": True,
                    "aggregated": False,
                }
            ]

    session = OwnerCatalogueSession()
    session.frontend_state = FrontendSessionState(session_id="sess", epoch="owner")
    # The follower's OWN controller has no usable provider at all: without
    # the owner merge the picker would render zero selectable rows.
    app = OperatorApp(lambda: _factory(session), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is session:
                break
        app._populate_model_picker()
        for _ in range(30):
            await pilot.pause()
        picker = app.query_one(Editor).model_picker
        selectors = [row.selector for row in picker._rows]
        assert "anthropic/claude-sonnet-4-6" in selectors, selectors


@pytest.mark.asyncio
async def test_resume_owned_session_adopts_remote_in_standard_app(monkeypatch, tmp_path) -> None:
    """A live owner becomes a RemoteSession in the existing OperatorApp.

    There is no pushed screen or attach vocabulary: the standard transcript and
    composer remain the only surface, and the old local writer is disposed.
    """
    from local_operator.mobile.types import PROTOCOL_VERSION, SessionRecord
    from local_operator.session.remote import RemoteSession
    from local_operator.session.runtime import registry as mobile_registry
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = FakeSession()

    class _RemoteFake(FakeSession):
        is_remote = True

        def __init__(self) -> None:
            super().__init__()
            self.takeover_callback: Any | None = None

        def set_takeover_callback(self, callback: Any) -> None:
            self.takeover_callback = callback

    remote = _RemoteFake()

    async def resume_factory(resume_id):
        assert resume_id == "sess-owned"
        return FakeSession()

    async def connect(*args, **kwargs):  # noqa: ANN002, ANN003
        return remote

    monkeypatch.setattr(RemoteSession, "connect", connect)
    app = OperatorApp(lambda: _factory(session), resume_factory=resume_factory)
    owner = os.getppid()
    marker_dir = config_dir() / "sessions" / "sess-owned"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / ".session.pid").write_text(str(owner))
    record = SessionRecord(
        pid=owner,
        kind="tui",
        session_id="sess-owned",
        conversation_name="Owned",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        protocol=PROTOCOL_VERSION,
    )
    mobile_registry.publish(record, root=config_dir())
    async with app.run_test(size=(80, 24)) as pilot:
        for _ in range(50):
            await pilot.pause()
            if app._session is session:
                break
        assert app._session is session
        app._resume_session("sess-owned", lambda *a, **k: None)
        for _ in range(50):
            await pilot.pause()
            if app._session is remote:
                break
            await asyncio.sleep(0.05)
        assert app._session is remote
        assert session.disposed
        assert app.screen is app.screen_stack[0]
        assert app.query_one(Editor).disabled is False
        assert callable(remote.takeover_callback)
        assert not any(
            "attach" in (notice.text() or "").lower() for notice in app.query(NoticeBlock)
        )


@pytest.mark.asyncio
async def test_live_resume_atomically_gates_submission_then_recovers_success_and_failure(
    monkeypatch, tmp_path
) -> None:
    """Pending live resume never routes a draft to the session being left."""
    from local_operator.mobile.types import PROTOCOL_VERSION, SessionRecord
    from local_operator.session.remote import RemoteSession
    from local_operator.session.runtime import registry as mobile_registry

    old = FakeSession()
    replacement = FakeSession()
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    async def resume_factory(resume_id):  # noqa: ANN001, ANN202
        return FakeSession()

    async def connect(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal attempts
        attempts += 1
        started.set()
        await release.wait()
        if attempts == 1:
            return replacement
        raise ConnectionError("owner sync failed")

    monkeypatch.setattr(RemoteSession, "connect", connect)
    app = OperatorApp(lambda: _factory(old), resume_factory=resume_factory)
    owner = os.getppid()
    marker_dir = config_dir() / "sessions" / "sess-transition"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / ".session.pid").write_text(str(owner))
    mobile_registry.publish(
        SessionRecord(
            pid=owner,
            kind="tui",
            session_id="sess-transition",
            conversation_name="Owned",
            cwd=str(tmp_path),
            model_label="test/model",
            control_port=1,
            control_key="k",
            protocol=PROTOCOL_VERSION,
        ),
        root=config_dir(),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        app._resume_session("sess-transition", lambda *a, **k: None)
        await asyncio.wait_for(started.wait(), timeout=1)
        _set_editor_line(editor, "must reach replacement")
        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == "must reach replacement"
        assert old.prompts == []
        assert editor.disabled is False

        release.set()
        for _ in range(50):
            await pilot.pause()
            if app._session is replacement:
                break
        assert app._session is replacement
        await pilot.press("enter")
        await pilot.pause()
        assert replacement.prompts == ["must reach replacement"]

        # A failed transition restores the old/current session's ordinary input
        # boundary. No visible mode or status is needed to make it routable again.
        started.clear()
        release.clear()
        app._resume_session("sess-transition", lambda *a, **k: None)
        await asyncio.wait_for(started.wait(), timeout=1)
        _set_editor_line(editor, "send after failure")
        await pilot.press("enter")
        assert replacement.prompts == ["must reach replacement"]
        release.set()
        for _ in range(50):
            await pilot.pause()
            if not app._session_transition_pending:
                break
        assert app._session is replacement
        assert editor.text == "send after failure"
        await pilot.press("enter")
        await pilot.pause()
        assert replacement.prompts == ["must reach replacement", "send after failure"]


@pytest.mark.asyncio
async def test_resume_owned_session_without_record_keeps_refusal(monkeypatch, capsys) -> None:
    """No record (old binary / registrant failure) keeps today's refusal copy
    verbatim — graceful degradation."""
    session = FakeSession()

    async def resume_factory(resume_id):
        return session

    app = OperatorApp(lambda: _factory(session), resume_factory=resume_factory)
    owner = os.getppid()
    marker_dir = config_dir() / "sessions" / "sess-owned2"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / ".session.pid").write_text(str(owner))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._resume_session("sess-owned2", lambda *a, **k: None)
        for _ in range(20):
            await pilot.pause()
        capsys.readouterr()
        # The refusal lands as a transcript notice (rendered), not stdout;
        # assert on the app's notice bookkeeping instead of the pipe.
        from local_operator.tui.widgets.transcript import NoticeBlock

        notices = list(app.query(NoticeBlock))
        assert any("older Local Operator process" in (n.text() or "") for n in notices)
        assert session.prompts == []


@pytest.mark.asyncio
async def test_model_opens_the_picker_instead_of_reporting_a_label() -> None:
    """``/model`` opens the model list, and still never prompts.

    Typing ``/model`` and pressing Enter goes through the command picker, whose
    completion adds the terminating space — and that space is exactly the handover
    that opens the model list. So the keystrokes a user actually presses reach the
    catalogue without ``/model`` ever being submitted as a command.

    Reporting the current label here was a dead end: the status band already shows
    it, while "which models could I switch to" had no way to be asked at all.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o", "d", "e", "l", "enter")
        await pilot.pause()
        assert editor.text == "/model ", editor.text
        assert editor.model_picker.is_open()
        # NOTHING was submitted: completing a command whose argument drives its
        # own list is not running it, so there is no echoed UserBlock and no
        # notice — just the list.
        assert app.query_one(TranscriptView).blocks() == []
    assert session.prompts == []


@pytest.mark.asyncio
async def test_clear_resets_transcript_and_bookkeeping() -> None:
    """TUI-009: /clear resets _streaming_block/_tool_cards AND posts a
    notice that history is untouched (cosmetic clear)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 1
        # Simulate live bookkeeping so we can prove the reset.
        card = ToolCard("t9", "bash", {"command": "ls"})
        app._tool_cards["t9"] = card
        await pilot.press("slash", "c", "l", "e", "a", "r", "enter")
        await pilot.pause()
        blocks = transcript.blocks()
        assert len(blocks) == 1  # only the "history untouched" notice
        assert isinstance(blocks[0], NoticeBlock)
        assert "untouched" in blocks[0].renderable.plain  # type: ignore[attr-defined]
        assert app._streaming_block is None
        assert app._tool_cards == {}


@pytest.mark.asyncio
async def test_ctrl_l_clears_and_resets() -> None:
    """Ctrl+L runs the same clear path as /clear (TUI-009)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        app._streaming_block = object()  # type: ignore[assignment]
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert app._streaming_block is None
        assert app._tool_cards == {}


@pytest.mark.asyncio
async def test_session_disposed_on_exit() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("slash", "e", "x", "i", "t", "enter")
        await pilot.pause()
    assert session.disposed


# --- keybinding pilot tests (TUI-026) -------------------------------------


@pytest.mark.asyncio
async def test_ctrl_c_interrupts_and_app_stays_alive() -> None:
    """Ctrl+C posts InterruptRequested (abort the turn) and never exits."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running  # the app stays alive
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_without_submit() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("a")
        await pilot.press("shift+enter")
        await pilot.press("b")
        await pilot.pause()
        # No submit happened; the buffer carries a newline.
        assert session.prompts == []
        assert editor.text == "a\nb"


@pytest.mark.asyncio
async def test_bang_on_empty_composer_enters_shell_mode() -> None:
    """opencode's dedicated-mode half: ``!`` on an empty field is consumed,
    the placeholder becomes the command invitation, and the dock class the
    stylesheet reads for the chevron's string-green is on."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("!")
        await pilot.pause()
        assert editor.shell_mode is True
        assert editor.text == ""
        assert editor.placeholder == SHELL_PLACEHOLDER
        assert app.query_one("#input-dock").has_class(COMPOSER_SHELL_CLASS)
        assert chevron_colour(composer_cells(app)) == theme_mod.semantic_color("string").lower()
        assert chevron_colour(composer_cells(app)) != theme_mod.semantic_color("accent").lower()
        # Shape, not hue: bang-mode changes what Enter does, and a colour-only
        # cue is invisible under NO_COLOR / monochrome / colour-vision
        # deficiency (#385). The glyph is the surviving signal.
        cells = composer_cells(app)
        assert any(SHELL_CHEVRON in text for text, _fg, _bg in cells)
        assert not any(PROMPT_CHEVRON in text for text, _fg, _bg in cells)


@pytest.mark.asyncio
async def test_shell_mode_is_legible_without_hue_once_the_buffer_has_text() -> None:
    """The placeholder is gone the moment the buffer has text, so the mode
    has to be carried by the frame itself — a hue-only chevron was the
    whole remaining signal, and two frames that differed only in ink
    were TEXT-IDENTICAL (#385).

    Asserted against the compositor, not against the widget: a reader
    who cannot see colour sees the glyphs, and the glyphs have to
    disagree. Mutation-checked: restoring ``❯`` while the mode is on
    fails this test, which is the defect.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("l", "s")
        await pilot.pause()
        prompt = "".join(text for text, _fg, _bg in composer_cells(app))
        editor.clear_content()
        await pilot.press("!")
        await pilot.press("l", "s")
        await pilot.pause()
        shell = "".join(text for text, _fg, _bg in composer_cells(app))
        assert editor.shell_mode is True
        assert SHELL_CHEVRON in shell
        assert PROMPT_CHEVRON not in shell
        assert PROMPT_CHEVRON in prompt
        assert SHELL_CHEVRON not in prompt
        # The two frames must disagree in TEXT, not only in ink. A
        # colour-only swap made this comparison True, which is the gap.
        assert prompt != shell, (
            "shell-mode and prompt-mode frames are text-identical once the "
            "buffer has text, so a reader who cannot see hue has no cue "
            "that Enter will run a command (#385)"
        )


def test_composer_markers_match_the_app() -> None:
    """``conftest`` mirrors the glyphs so it does not import the app
    module. Drift here would make every colour/caret assertion look at
    the wrong cell the moment bang-mode is on."""
    from tests.unit.tui import conftest as tui_conftest

    assert tui_conftest.PROMPT_CHEVRON == PROMPT_CHEVRON
    assert tui_conftest.SHELL_CHEVRON == SHELL_CHEVRON
    assert PROMPT_CHEVRON != SHELL_CHEVRON


@pytest.mark.asyncio
async def test_bang_inside_a_draft_is_just_a_character() -> None:
    """A ``!`` in `echo hi!` or mid-sentence is not a mode switch."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("h", "i", "!")
        await pilot.pause()
        assert editor.shell_mode is False
        assert editor.text == "hi!"
        assert editor.placeholder == DEFAULT_PLACEHOLDER


@pytest.mark.asyncio
async def test_escape_and_empty_backspace_leave_shell_mode() -> None:
    """opencode's exits: Esc keeps a draft, empty-buffer backspace is the
    inverse of the bang that entered. Neither posts a stop."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("!")
        await pilot.press("l", "s")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert editor.shell_mode is False
        assert editor.text == "ls"
        assert editor.placeholder == DEFAULT_PLACEHOLDER
        assert not app.query_one("#input-dock").has_class(COMPOSER_SHELL_CLASS)
        assert session.aborts == []

        editor.clear_content()
        await pilot.press("!")
        await pilot.pause()
        await pilot.press("backspace")
        await pilot.pause()
        assert editor.shell_mode is False
        assert editor.text == ""
        assert session.aborts == []


@pytest.mark.asyncio
async def test_shell_mode_submit_runs_the_command_not_a_prompt() -> None:
    """Enter in bang-mode runs bash locally: a user row, a tool card, no
    ``prompt()``, and the result is persisted so the next turn can see it."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("!")
        for key in "echo bang-mode-ok":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        # The worker has to actually spawn /bin/sh; a couple of pauses is
        # not a bound, so wait for the card to settle.
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        assert session.prompts == []
        assert editor.shell_mode is False
        assert editor.text == ""
        transcript = app.query_one(TranscriptView)
        blocks = transcript.blocks()
        assert any(isinstance(block, UserBlock) for block in blocks)
        cards = [block for block in blocks if isinstance(block, ToolCard)]
        assert len(cards) == 1
        assert cards[0].tool_name == "bash"
        assert cards[0]._state == "success"
        assert session.shell_records
        command, result = session.shell_records[0]
        assert command == "echo bang-mode-ok"
        assert "bang-mode-ok" in result.text
        # History stored WITH the bang so Up-arrow recall re-runs as a command.
        assert editor.prompt_history()[-1] == "! echo bang-mode-ok"


@pytest.mark.asyncio
async def test_a_recalled_bang_line_still_runs_as_a_command() -> None:
    """omp's submit-of-a-leading-bang half: history stores the bang, so Up
    then Enter re-runs as a command even though dedicated mode is off.

    ``!`` on an empty composer is consumed, so the only way a leading bang
    lands in the buffer is a recall (or a paste). That is the path this
    pins — without it, Up after ``! echo hi`` would send ``! echo hi`` as
    a prompt.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("!")
        for key in "echo first":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        assert len(session.shell_records) == 1
        await pilot.press("up")
        await pilot.pause()
        assert editor.shell_mode is False
        assert editor.text == "! echo first"
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if len(session.shell_records) >= 2:
                break
        assert session.prompts == []
        assert [command for command, _ in session.shell_records] == [
            "echo first",
            "echo first",
        ]


@pytest.mark.asyncio
async def test_empty_shell_submit_is_a_noop() -> None:
    """A user who entered bang-mode by accident and pressed Enter lands
    back on the resting composer, not on a red notice."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("!")
        await pilot.press("enter")
        await pilot.pause()
        assert editor.shell_mode is False
        assert session.prompts == []
        assert session.shell_records == []
        assert app.query_one(TranscriptView).blocks() == []


@pytest.mark.asyncio
async def test_esc_aborts_a_running_shell_command() -> None:
    """Esc on a live bang-mode command stops it the same way it stops a
    turn: the card is interrupted on this press, not after the process
    reaps."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        # The bang path captures ``self._session`` at submit time and persists
        # through it; submitting before adoption lands makes ``persist`` a
        # silent no-op, so the record the test asserts never exists. A single
        # pause bets adoption takes one tick — it is the first app's
        # import-and-construct cost on a fresh interpreter. Wait on the
        # condition instead.
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause(0.02)
        assert app._session is not None, "the session was never adopted"
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("!")
        for key in "sleep 30":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        # Wait until the card is on screen and still running — the whole
        # point of this test is the in-flight state, not the settled one.
        card = None
        for _ in range(200):
            await pilot.pause()
            cards = [
                block
                for block in app.query_one(TranscriptView).blocks()
                if isinstance(block, ToolCard)
            ]
            if cards:
                card = cards[0]
                if card._state == "running":
                    break
        assert card is not None
        assert card._state == "running"
        await pilot.press("escape")
        # Read the interrupted frame WITHOUT yielding first: the abort marks
        # the card and keeps ownership synchronously on the key press, so the
        # assertions must observe that on-press state before the event loop
        # hands control to the shell worker. A `pause()` here raced under
        # parallel load — the aborted worker reaped `sleep 30` and cleared
        # `_shell_card` to None before the ownership assertion read it.
        assert card._state == "interrupted"
        # The card stays owned until execute_bash has reaped the process and
        # persisted its interrupted result. Clearing it on the key press made
        # the receipt disappear on resume — so the invariant is that ownership
        # is never cleared WITHOUT the record: if the reaper has already run
        # by the time we look (it can, under load, inside the press's own
        # yield), the persisted record must already be there. A bare
        # ``_shell_card is card`` asserted that the reaper had NOT run, which
        # is a bet on how fast the kill lands rather than on the contract.
        if app._shell_card is None:
            assert (
                session.shell_records
            ), "ownership was cleared without persisting the interrupted result"
        else:
            assert app._shell_card is card
        for _ in range(200):
            await pilot.pause()
            if app._shell_card is None:
                break
        assert app._shell_card is None
        assert session.shell_records
        command, result = session.shell_records[0]
        assert command == "sleep 30"
        assert result.is_error is True
        assert "interrupted" in result.text.lower()


@pytest.mark.asyncio
async def test_a_failing_command_marks_the_card_failed_and_persists_it_so() -> None:
    """A nonzero exit is a FAILURE on the human-facing surface: the collapsed
    card says ✗, and the persisted result carries is_error so a resumed
    session restores the same mark (design round 1, D1)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "exit 3":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        assert session.shell_records, "the failed command must still be persisted"
        command, result = session.shell_records[0]
        assert command == "exit 3"
        assert result.is_error is True
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert cards and cards[-1]._state == "error"


@pytest.mark.asyncio
async def test_a_signal_killed_command_marks_the_card_failed() -> None:
    """A negative exit code (SIGKILL — the same shape a timeout's kill
    produces) is a failure on the human-facing surface, not a `✓` with a
    scary body (round 3 minor)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "kill -9 $$":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        assert session.shell_records
        command, result = session.shell_records[0]
        assert command == "kill -9 $$"
        assert result.is_error is True
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert cards and cards[-1]._state == "error"


@pytest.mark.asyncio
async def test_a_bang_card_settles_open() -> None:
    """The user ran the command to READ its output, so the settled card is
    already open — a collapsed receipt hides exactly the bytes the command
    was run for behind a click nobody asked for."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "echo open-me":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert cards
        assert cards[-1]._state == "success"
        assert cards[-1].expanded is True
        # The open is a settle-time decision, not a stuck mode: the flag is
        # consumed, so a user collapse is final.
        assert cards[-1]._open_on_settle is False


@pytest.mark.asyncio
async def test_a_replayed_bang_card_opens_and_ordinary_calls_stay_shut() -> None:
    """Resume parity for the open-on-settle contract: a bang receipt replays
    open, while an agent-issued bash call in the same history stays a normal
    collapsible row."""
    session = FakeSession()
    session._history = [
        SimpleNamespace(
            role="user", text="! echo hi", tool_calls=None, content=[], custom_type=None
        ),
        SimpleNamespace(
            role="assistant",
            text="",
            tool_calls=[
                SimpleNamespace(id="shell-1", name="bash", arguments={"command": "echo hi"})
            ],
            custom_type=None,
        ),
        SimpleNamespace(
            role="tool",
            tool_call_id="shell-1",
            text="exit code: 0\nhi",
            is_error=False,
            provider_payload=None,
            content=[],
        ),
        SimpleNamespace(
            role="user", text="what did I just run?", tool_calls=None, content=[], custom_type=None
        ),
        SimpleNamespace(
            role="assistant",
            text="",
            tool_calls=[SimpleNamespace(id="agent-1", name="bash", arguments={"command": "ls"})],
            custom_type=None,
        ),
        SimpleNamespace(
            role="tool",
            tool_call_id="agent-1",
            text="exit code: 0\nfile",
            is_error=False,
            provider_payload=None,
            content=[],
        ),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert len(cards) == 2
        assert cards[0].expanded is True  # the bang receipt
        assert cards[1].expanded is False  # an ordinary agent call


@pytest.mark.asyncio
async def test_a_bang_card_says_who_ran_it() -> None:
    """A user-run receipt is visually distinguishable from an agent-run bash
    row: the summary leads with a `you:` chip, live and after resume, and the
    chip survives truncation at narrow widths."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "echo attribution":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert cards
        assert cards[-1].user_run is True
        assert "you:" in cards[-1]._build_row(80).plain
        # The chip is inside the summary budget, so a narrow row keeps it.
        assert "you:" in cards[-1]._build_row(40).plain
        # An ordinary agent card carries no chip.
        plain = ToolCard("x", "bash", {"command": "ls"})
        assert "you:" not in plain._build_row(80).plain


@pytest.mark.asyncio
async def test_a_replayed_bang_card_says_who_ran_it() -> None:
    """Resume parity for the attribution: the replayed bang row carries the
    same `you:` chip the live one showed, and an ordinary bash call does not."""
    session = FakeSession()
    session._history = [
        SimpleNamespace(
            role="user", text="! echo hi", tool_calls=None, content=[], custom_type=None
        ),
        SimpleNamespace(
            role="assistant",
            text="",
            tool_calls=[
                SimpleNamespace(id="shell-1", name="bash", arguments={"command": "echo hi"})
            ],
            custom_type=None,
        ),
        SimpleNamespace(
            role="tool",
            tool_call_id="shell-1",
            text="exit code: 0\nhi",
            is_error=False,
            provider_payload=None,
            content=[],
        ),
        SimpleNamespace(
            role="assistant",
            text="",
            tool_calls=[SimpleNamespace(id="agent-1", name="bash", arguments={"command": "ls"})],
            custom_type=None,
        ),
        SimpleNamespace(
            role="tool",
            tool_call_id="agent-1",
            text="exit code: 0\nfile",
            is_error=False,
            provider_payload=None,
            content=[],
        ),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert len(cards) == 2
        assert cards[0].user_run is True
        assert "you:" in cards[0]._build_row(80).plain
        assert cards[1].user_run is False
        assert "you:" not in cards[1]._build_row(80).plain


@pytest.mark.asyncio
async def test_a_failing_bang_card_settles_open_too() -> None:
    """Failure is exactly when the output matters most, so the open-on-settle
    contract covers the error settle as well."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "exit 3":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert cards
        assert cards[-1]._state == "error"
        assert cards[-1].expanded is True


@pytest.mark.asyncio
async def test_a_replayed_aborted_bang_card_stays_shut_and_dim() -> None:
    """The user's own Esc must not come back as a red failure on resume: the
    persisted abort result is an error-shaped message, but the live frame it
    came from was the dim shut `interrupted` row (design round 1, D1)."""
    session = FakeSession()
    session._history = [
        SimpleNamespace(
            role="user", text="! sleep 30", tool_calls=None, content=[], custom_type=None
        ),
        SimpleNamespace(
            role="assistant",
            text="",
            tool_calls=[
                SimpleNamespace(id="shell-1", name="bash", arguments={"command": "sleep 30"})
            ],
            custom_type=None,
        ),
        SimpleNamespace(
            role="tool",
            tool_call_id="shell-1",
            text="aborted (interrupted): sleep 30",
            is_error=True,
            provider_payload=None,
            content=[],
        ),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert len(cards) == 1
        assert cards[0]._state == "interrupted"
        assert cards[0].expanded is False


@pytest.mark.asyncio
async def test_a_bang_call_with_no_recorded_result_stays_shut() -> None:
    """A session killed between the call and its answer replays `interrupted`
    with nothing absorbed — open-on-settle must not force such a card open."""
    session = FakeSession()
    session._history = [
        SimpleNamespace(
            role="user", text="! sleep 30", tool_calls=None, content=[], custom_type=None
        ),
        SimpleNamespace(
            role="assistant",
            text="",
            tool_calls=[
                SimpleNamespace(id="shell-1", name="bash", arguments={"command": "sleep 30"})
            ],
            custom_type=None,
        ),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        # Poll until the boot worker has rendered the resumed history rather
        # than betting one frame is enough: under parallel CPU load the
        # transcript was still empty when a single tick expired.
        cards: list[ToolCard] = []
        for _ in range(200):
            await pilot.pause()
            cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
            if cards:
                break
        assert len(cards) == 1
        assert cards[0]._state == "interrupted"
        assert cards[0].expanded is False


@pytest.mark.asyncio
async def test_a_user_collapse_after_settle_is_final() -> None:
    """Open-on-settle is a settle-time decision, not a stuck mode: the flag is
    consumed at the first settle, so closing the card stays closed."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "echo hi":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert cards and cards[-1].expanded is True
        cards[-1].toggle_expanded()
        await pilot.pause()
        assert cards[-1].expanded is False
        for _ in range(10):
            await pilot.pause()
        assert cards[-1].expanded is False


@pytest.mark.asyncio
async def test_a_refused_second_command_comes_back_in_shell_mode() -> None:
    """The refused line is handed back IN the mode it was submitted from: the
    placeholder matches the buffer, and Enter re-runs it as a command once
    the running one is settled — not as a prompt to the model."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "sleep 30":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if app._shell_card is not None:
                break
        assert app._shell_card is not None

        await pilot.press("!")
        for key in "echo second":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        await pilot.pause()

        assert editor.shell_mode is True
        assert editor.text == "! echo second"
        assert editor.placeholder == SHELL_PLACEHOLDER
        await pilot.press("escape")  # leave the mode
        await pilot.press("escape")  # abort the running command
        for _ in range(200):
            await pilot.pause()
            if session.shell_records:
                break
        # The aborted first command persists; the refused second did not run.
        assert [c for c, _ in session.shell_records] == ["sleep 30"]


@pytest.mark.asyncio
async def test_reload_aborts_and_settles_shell_before_disposing_its_session() -> None:
    """A session swap cannot strand the bang-mode subprocess or let its
    completion write into a session that `/reload` already disposed."""
    first = FakeSession()
    second = FakeSession()
    order: list[str] = []

    async def record_shell(command: str, result: Any) -> None:
        assert first.disposed is False
        order.append("recorded")
        first.shell_records.append((command, result))

    async def dispose() -> None:
        order.append("disposed")
        first.disposed = True

    first.record_shell = record_shell  # type: ignore[method-assign]
    first.dispose = dispose  # type: ignore[method-assign]
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "sleep 30":
            await pilot.press("space" if key == " " else key)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if app._shell_card is not None:
                break
        assert app._shell_card is not None

        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        assert order == ["recorded", "disposed"]
        assert first.shell_records[0][0] == "sleep 30"
        assert app._shell_worker is None
        assert app._shell_card is None
        assert app._session is second


@pytest.mark.asyncio
async def test_tab_completes_slash_without_losing_focus() -> None:
    """Tab completes /he -> /help (trailing space = the argument slot) and
    focus stays on the editor (TUI-013)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("slash", "h", "e")
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "/help "
        assert editor.has_focus  # completion never moves focus


# --- boot failure + reload (TUI-012) --------------------------------------


@pytest.mark.asyncio
async def test_boot_failure_posts_error_and_reload_retries() -> None:
    attempts = {"n": 0}
    session = FakeSession()

    async def flaky_factory() -> FakeSession:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("provider is down")
        return session

    app = OperatorApp(flaky_factory)
    async with app.run_test(size=(80, 24)) as pilot:
        for _ in range(40):
            await pilot.pause()
            transcript = app.query_one(TranscriptView)
            texts = "\n".join(
                getattr(getattr(b, "renderable", None), "plain", "") for b in transcript.blocks()
            )
            if "provider is down" in texts:
                break
        # Boot failure surfaces as an error notice + 'session error' status.
        transcript = app.query_one(TranscriptView)
        kinds = [type(b).__name__ for b in transcript.blocks()]
        texts = "\n".join(
            getattr(getattr(b, "renderable", None), "plain", "") for b in transcript.blocks()
        )
        assert "NoticeBlock" in kinds
        assert "provider is down" in texts
        assert app._session is None
        # And the splash SURVIVES it. A session that never constructed is the most
        # infrastructure-y report in the app, and the worst moment to lose the one
        # block that says what to do next: the credential warning and the boot
        # hints both live there. Retiring the empty state here left a single red
        # line over an empty screen.
        assert app.query_one(WelcomeView).display is True
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        # /reload now re-execs (exit 75) rather than retrying in-process.
        # A boot that never constructed still relaunches as a cold start.
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "r", "e", "l", "o", "a", "d", "enter")
        await pilot.pause()
        await pilot.pause()
        from local_operator.reexec import REEXEC_CODE

        assert app.return_code == REEXEC_CODE
        assert attempts["n"] == 1
        assert app._session is None


# --- login/logout through the provider controller --------------------------


@pytest.mark.asyncio
async def test_login_lists_providers_from_the_controller() -> None:
    """Bare /login lists loginable providers — the controller is the only
    path now that the CLI login_handler seam is gone.

    Esc before Enter is what makes it BARE: completing `/login` now opens the
    provider list instead, so dismissing that list is the only way left to run
    the command with no argument.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "escape", "enter")
        await pilot.pause()
        text = _transcript_text(app)
    assert "openrouter" in text and "deepseek" in text


@pytest.mark.asyncio
async def test_logout_routes_to_the_controller() -> None:
    controller = FakeProviderController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        for ch in "/logout openrouter":
            await pilot.press("slash" if ch == "/" else ("space" if ch == " " else ch))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert controller.logouts == ["openrouter"]


def _provider_rows(app) -> list[tuple[str, str]]:
    """``(id, detail)`` for every row the provider list is offering."""
    rows: list[tuple[str, str]] = []
    for name, choice in app.query_one(Editor).picker.suggestions():
        assert isinstance(choice, ArgumentChoice), "the picker is not in argument mode"
        rows.append((name, choice.detail))
    return rows


@pytest.mark.asyncio
async def test_login_list_reports_all_three_credential_states() -> None:
    """`/login ` offers every loginable provider with where the user stands.

    Three states, not two: an env key runs a turn but is not a login, so calling
    it one would promise a stored account `/logout` could remove.
    """

    class EnvKeyController(FakeProviderController):
        def is_usable(self, provider):
            # deepseek has DEEPSEEK_API_KEY in the environment but no stored login.
            return self.has_any_credential(provider) or provider == "deepseek"

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=EnvKeyController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/login ")
        await pilot.pause()
        assert _provider_rows(app) == [
            ("openrouter", "logged in"),
            ("deepseek", "env key"),
            ("xai-oauth", "needs login"),
        ]


@pytest.mark.asyncio
async def test_a_search_alias_reaches_the_provider_it_names() -> None:
    """`grok` is how users refer to xAI; the row still completes to the id."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/login grok")
        await pilot.pause()
        assert [name for name, _ in _provider_rows(app)] == ["xai-oauth"]


@pytest.mark.asyncio
async def test_logout_rows_name_the_credential_they_will_remove() -> None:
    """`/logout` offers only what can be removed, and each row says what goes.

    Not "logged in": this list is FILTERED to providers holding a credential, so
    that state is true of every row by construction — a column with no bits in
    it, holding cells the description needs at narrow widths. The kind is what
    differs between rows and what the keystroke destroys.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/logout ")
        await pilot.pause()
        assert _provider_rows(app) == [("openrouter", "remove api key")]


@pytest.mark.asyncio
async def test_logout_with_nothing_stored_says_so_where_the_list_would_have_been() -> None:
    """An empty set and "nothing matched your query" render identically — as
    nothing at all. Only the first is worth a sentence, because no amount of
    retyping would have produced a row.

    It is said in the PICKER, not the transcript. The sentence answers a UI event,
    so a transcript line repeats on every re-entry into the argument state (see the
    test below) — and a transcript is a record, while an empty list is a transient
    state of the input. The row is dim and unselectable: `is_open()` stays False, so
    Enter still submits the buffer and no click can action a sentence.
    """

    class NoCredentials(FakeProviderController):
        def credentials(self):
            return []

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=NoCredentials())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/logout ")
        await pilot.pause()
        picker = app.query_one(Editor).picker
        assert not picker.is_open(), "a sentence is not a suggestion"
        assert picker.display is True, "and it is on screen anyway"
        assert "log out of" in picker.render_text(60).plain
        assert picker._index_at(0) is None, "the row cannot be clicked into a command"
        assert "log out of" not in _transcript_text(app), "the transcript is a record"
        # And it did NOT end the empty state. Opening a list is not the
        # conversation starting: a fresh session that collapsed its boot
        # composition to report that a command the user has not run yet has
        # nothing to offer would spend the whole empty state on that sentence.
        assert app.query_one(WelcomeView).display is True
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)


@pytest.mark.asyncio
async def test_re_entering_an_empty_argument_state_leaves_the_transcript_alone() -> None:
    """Ten re-entries, zero transcript blocks.

    The sentence is raised from a UI event, so every route back into the argument
    state raises it again: typing `/logout `, backspacing, typing the space. As a
    transcript notice that stacked four identical rows in as many keystrokes, each
    one also taking a row off the splash that shares the region. In the picker the
    tenth re-entry looks exactly like the first.
    """

    class NoCredentials(FakeProviderController):
        def credentials(self):
            return []

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=NoCredentials())
    async with app.run_test(size=(96, 28)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        for _ in range(10):
            _set_editor_line(editor, "/logout")
            await pilot.pause()
            _set_editor_line(editor, "/logout ")
            await pilot.pause()
            await pilot.pause()
            assert transcript.blocks() == []
        assert "log out of" in editor.picker.render_text(60).plain
        # One row, however many times it was set: the picker holds a string, not a
        # list it appends to.
        assert editor.picker.styles.height is not None
        assert editor.picker.region.height == 1


@pytest.mark.asyncio
async def test_an_unreadable_credential_store_says_that_instead() -> None:
    """ "You have no credentials" and "I cannot tell" are different answers, and the
    informational row carries whichever one is true."""

    class RaisingStore(FakeProviderController):
        def credentials(self):
            raise RuntimeError("database is locked")

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RaisingStore())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/logout ")
        await pilot.pause()
        rendered = app.query_one(Editor).picker.render_text(80).plain
        assert "unreadable" in rendered
        assert "no stored credentials" not in rendered


@pytest.mark.asyncio
async def test_choosing_a_row_runs_the_existing_logout_path() -> None:
    """The list is a way to reach `_cmd_logout`, not a second implementation.

    Two Enters, even with a single row: on a destructive list "there is only one
    match" is not evidence that the user meant it — an empty query matches
    everything, and here everything happens to be one credential. The first
    Enter names it in the buffer, the second removes it.
    """
    controller = FakeProviderController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/logout ")
        await pilot.pause()
        assert len(app.query_one(Editor).picker.suggestions()) == 1, "premise: one match"
        await pilot.press("enter")
        await pilot.pause()
        assert controller.logouts == [], "an unnamed row is completed, not removed"
        assert app.query_one(Editor).text == "/logout openrouter"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert controller.logouts == ["openrouter"]


@pytest.mark.asyncio
async def test_choosing_a_row_runs_the_existing_login_path() -> None:
    controller = FakeProviderController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/login deepseek")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert controller.logins == ["deepseek"]


@pytest.mark.asyncio
async def test_login_still_lists_every_provider_when_the_store_cannot_be_read() -> None:
    """An unreadable credential store costs the STATE column, not the app.

    The handler runs on a keystroke, so an exception out of it takes the whole
    TUI down — and the moment the store is unreadable is exactly the moment a
    user reaches for `/login`. The catalogue comes from the in-memory registry
    and is still entirely answerable, so every provider is still offered; only
    the state is blank, because a blank claims nothing and any of the three
    states would claim something the app cannot know.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RaisingStoreController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/login ")
        await pilot.pause()
        assert app.is_running, "a locked credential store must not take the app down"
        assert _provider_rows(app) == [
            ("openrouter", ""),
            ("deepseek", ""),
            ("xai-oauth", ""),
        ]


@pytest.mark.asyncio
async def test_logout_says_the_store_is_unreadable_rather_than_claiming_it_is_empty() -> None:
    """`/logout` asks a question only the store can answer, so there is no
    degraded list — but "you have no credentials" is a different answer from "I
    cannot tell", and only one of them is true when the file is locked."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RaisingStoreController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/logout ")
        await pilot.pause()
        assert app.is_running
        assert _provider_rows(app) == []
        picker = app.query_one(Editor).picker
        rendered = picker.render_text(60).plain
        assert "store unreadable" in rendered, rendered
        assert "no stored credentials" not in rendered


@pytest.mark.asyncio
async def test_logout_offers_one_row_per_credential_not_per_provider() -> None:
    """`openai` and `openai-device` share one stored credential, so logging out
    of either removes the same account. Two rows for one outcome is a choice the
    user cannot make correctly — while `/login` must still offer both, because
    they are two different ways to sign in."""
    app = OperatorApp(
        lambda: _factory(FakeSession()), provider_controller=CollidingStorageController()
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/logout ")
        await pilot.pause()
        assert _provider_rows(app) == [("openai", "remove oauth")]

        _set_editor_line(app.query_one(Editor), "/login ")
        await pilot.pause()
        assert [name for name, _ in _provider_rows(app)] == ["openai", "openai-device"]


@pytest.mark.asyncio
async def test_a_provider_row_describes_what_its_id_does_not_already_say() -> None:
    """The registry name restated the id on twelve rows out of twelve.

    `openai / OpenAI (ChatGPT Plus/Pro)` spent half the description column
    re-spelling the id in title case and parenthesised the only part that
    distinguishes the row. That parenthetical is also the ONLY thing telling
    `openai` from `openai-device` and `xai` from `xai-oauth` apart, so it is
    what makes those four near-duplicates choosable. Where the name says nothing
    the id does not, the cell is blank — that is the honest answer.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RealRegistryController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/login ")
        await pilot.pause()
        described = {
            name: choice.description for name, choice in app.query_one(Editor).picker.suggestions()
        }
    assert described["openai"] == "ChatGPT Plus/Pro"
    assert described["openai-device"] == "ChatGPT device code"
    assert described["xai"] == "Grok API key"
    assert described["xai-oauth"] == "Grok OAuth"
    assert described["anthropic"] == "Claude Pro/Max"
    assert described["deepseek"] == ""
    assert described["openrouter"] == ""
    assert described["radient"] == ""


@pytest.mark.asyncio
async def test_a_provider_query_the_user_typed_over_is_not_answered() -> None:
    """The opening message is one message-loop tick old, and a tick is enough to
    abandon the command. Answering it anyway attached a notice — and a list — to
    a command that no longer exists in the buffer."""

    class NoCredentials(FakeProviderController):
        def has_any_credential(self, provider):
            return False

        def credentials(self):
            return []

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=NoCredentials())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        editor = app.query_one(Editor)
        _set_editor_line(editor, "/logout ")
        editor.text = "how do I write a parser?"
        await pilot.pause()
        await pilot.pause()
        picker = editor.picker
        assert not picker.is_open()
        assert picker.display is False, "nothing is said about an abandoned command"
        assert "log out" not in _transcript_text(app)


@pytest.mark.asyncio
async def test_login_without_controller_points_at_the_cli() -> None:
    """Degrading to a pointer notice is the contract when the TUI is embedded
    without a controller — it must never crash or silently do nothing.

    The two routes degrade to different PLACES, because they answer different
    things. Opening the list is a UI event that repeats on every re-entry into the
    argument state, so it says why the list is empty inside the list. Running the
    bare command is something the user did once, so it lands in the transcript,
    which is the record of what they did.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))  # no controller
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "enter")  # opens the list
        await pilot.pause()
        picker = app.query_one(Editor).picker
        assert not picker.is_open(), "no controller, no rows"
        assert "local-operator login" in picker.render_text(70).plain
        assert "local-operator login" not in _transcript_text(app)

        # No Esc needed: with no rows the list never opened, so Enter on the
        # completed `/login ` goes straight to the command dispatch — and THAT is
        # a command the user ran, so it is recorded.
        await pilot.press("enter")
        await pilot.pause()
        assert _transcript_text(app).count("local-operator login") == 1


@pytest.mark.asyncio
async def test_skills_and_mcp_commands_never_crash() -> None:
    """Whatever the discovery layer returns, the handlers stay graceful."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "s", "k", "i", "l", "l", "s", "enter")
        await pilot.pause()
        await pilot.press("slash", "m", "c", "p", "enter")
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) >= 2  # echoes + notices/listings
    assert session.prompts == []


# --- autocomplete scoring (sync, I/O-free) --------------------------------
from local_operator.tui.autocomplete import (  # noqa: E402
    SlashCommand,
    match_commands,
    score_command_text_match,
)


def test_scoring_exact_beats_prefix_beats_fuzzy() -> None:
    assert score_command_text_match("help", "help") == 1000
    assert score_command_text_match("he", "help") == 900
    fuzzy = score_command_text_match("hp", "help")
    assert 1 <= fuzzy <= 40
    assert 1000 > 900 > fuzzy


def test_scoring_case_insensitive_and_no_match() -> None:
    assert score_command_text_match("HELP", "help") == 1000
    assert score_command_text_match("zzz", "help") == 0
    assert score_command_text_match("", "help") == 0


def test_match_orders_by_score_then_registry() -> None:
    commands = [SlashCommand("help"), SlashCommand("history")]
    # "h" is a prefix of both -> flat 900; registry order breaks the tie.
    matches = match_commands("/h", commands)
    assert [name for name, _ in matches] == ["help", "history"]
    # "hi" prefixes only history.
    assert [name for name, _ in match_commands("/hi", commands)] == ["history"]


def test_completion_takes_the_top_match_when_ambiguous() -> None:
    """With several matches the picker highlights the top-ranked one and Tab
    applies it — registration order breaking ties, same ranking the scoring
    produces (there is no more "refuse when ambiguous" completion)."""
    editor = Editor(commands=[SlashCommand("help"), SlashCommand("history"), SlashCommand("exit")])
    editor.text = "/h"  # help and history tie at 900; registry order wins
    assert editor.picker.highlighted_name() == "help"
    editor.text = "/hello"  # nothing matches: the picker closes
    assert not editor.picker.is_open()


def test_completion_matches_alias() -> None:
    """TUI-014: the collapsed exit/quit command still completes via alias —
    the alias wins the ranking slot, so the inserted word is the alias."""
    editor = Editor(commands=[SlashCommand("exit", "Quit", aliases=("quit",))])
    _set_editor_line(editor, "/q")
    assert editor.picker.highlighted_name() == "quit"
    editor.picker.choose(0)  # the mouse path: select row 0 and complete
    assert editor.text == "/quit "


# --- provider-controller slash commands -----------------------------------


class FakeModel:
    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id


class FakeProviderController:
    """Minimal stand-in for ProviderController (sync + immediate fetches)."""

    def __init__(self) -> None:
        self.set_model_calls: list[Any] = []
        self.usage_reports: list[Any] = []
        # Every `fetch_usage` argument list, so a test can prove the panel's
        # refresh key actually re-fetches rather than repainting stale numbers.
        self.usage_calls: list[Any] = []
        self.usage_error: Exception | None = None
        self.logins: list[str] = []
        self.logouts: list[str] = []
        #: Staged cache rows for the instant-open path. Empty by default so
        #: existing tests exercise the cold (fetching…) open.
        self.cached_reports: list[Any] = []
        #: Age in ms of the staged cache row, or None when there is none.
        #: The warmer reads this to decide whether to fetch.
        self.usage_cache_age: int | None = None

    def login_providers(self) -> list[Any]:
        return [
            _FakeDef("openrouter", "OpenRouter", None, ("router",)),
            _FakeDef("deepseek", "DeepSeek", None, ("ds",)),
            _FakeDef("xai-oauth", "xAI OAuth", "xai", ("grok",)),
        ]

    def provider(self, pid):
        for d in self.login_providers():
            if d.id == pid:
                return d
        return None

    def is_usable(self, provider):
        # An env key counts, so this is wider than has_any_credential. The fake
        # answers both because the app asks the narrow one for "logged in" and the
        # wide one for "would a turn work".
        return self.has_any_credential(provider)

    def usable_providers(self) -> set[str] | None:
        # The set shape the picker's filter asks for: one answer for the whole
        # registry instead of one probe per provider. `None` would mean the store
        # could not be read at all, which this fake never simulates.
        return {d.id for d in self.login_providers() if self.is_usable(d.id)}

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="openrouter",
                model_id="deepseek/deepseek-chat",
                label="DeepSeek Chat",
                context_window=64_000,
                input_price=0.14,
                output_price=0.28,
                connected=True,
                aggregated=True,
            )
        ]

    async def live_catalogue(self, *, ttl_s=None):
        return self.static_catalogue(), {"openrouter": "ok"}

    def has_any_credential(self, provider):
        return provider in ("openrouter",)

    def credentials(self):
        return [
            _FakeCred(1, "openrouter", "api_key", {"source": "login"}),
            _FakeCred(2, "deepseek", "oauth", {"expires": 9999999999999, "email": "a@b.c"}),
        ]

    def usage_enabled_providers(self):
        return ["openrouter", "zai"]

    def usage_reportable_providers(self):
        # The real controller narrows "has an endpoint" by "and a credential that
        # can reach it"; `/provider` renders this one, not the wider list.
        return [p for p in self.usage_enabled_providers() if self.has_any_credential(p)]

    def can_report_usage(self, provider):
        # The warmer's gate: "has an endpoint AND a credential to reach it".
        return provider in self.usage_reportable_providers()

    def resolve_model(self, provider, model_id):
        return FakeModel(provider, model_id)

    def set_login_callbacks(self, factory):
        # The TUI installs its own transcript-rendering callbacks before every
        # flow; without this the flow dies on an AttributeError the app catches
        # and reports as "login failed", which looks exactly like a real failure.
        self.login_callbacks = factory

    async def login(self, provider):
        self.logins.append(provider)
        return f"logged in {provider}"

    async def logout(self, provider):
        self.logouts.append(provider)
        return f"removed {provider}"

    async def fetch_usage(self, provider_ids=None, *, force_refresh: bool = False):
        self.usage_calls.append(provider_ids)
        if self.usage_error is not None:
            raise self.usage_error
        return self.usage_reports

    def cached_usage_reports(self, provider=None):
        # The shared-cache fast path. Empty by default so the existing pilot
        # tests exercise the cold (fetching…) open; a test that wants the
        # instant-paint path stages rows here.
        return list(getattr(self, "cached_reports", []))

    def usage_cache_age_ms(self, provider):
        # Settable so a test can simulate a warm row (0) or a cold one (None).
        return getattr(self, "usage_cache_age", None)


class _FakeDef:
    def __init__(self, pid, name, store_as, aliases=()):
        self.id = pid
        self.name = name
        self.store_credentials_as = store_as
        self.login = object()  # truthy -> has interactive login
        # Mirrors ProviderDefinition.search_aliases: the other names a user would
        # type for this provider, which is what makes `grok` reach `xai-oauth`.
        self.search_aliases = aliases


class _FakeCred:
    def __init__(self, ident, provider, ctype, data):
        self.id = ident
        self.provider = provider
        self.credential_type = ctype
        self.data = data
        self.identity_key = None


class RaisingStoreController(FakeProviderController):
    """A controller whose credential store cannot be read.

    `database is locked` is one other local-operator process away, and every
    read below goes to the same SQLite file.
    """

    def has_any_credential(self, provider):
        raise RuntimeError("database is locked")

    def is_usable(self, provider):
        raise RuntimeError("database is locked")

    def credentials(self):
        raise RuntimeError("database is locked")


class CollidingStorageController(FakeProviderController):
    """Two providers that file their credential under ONE storage id.

    The real registry has two such pairs (openai/openai-device and
    xai/xai-oauth); the default fake had a `store_credentials_as` provider with
    nothing to collide with, so the dedupe branch never ran.
    """

    def login_providers(self):
        return [
            _FakeDef("openai", "OpenAI (ChatGPT Plus/Pro)", None, ("gpt",)),
            _FakeDef("openai-device", "OpenAI (ChatGPT device code)", "openai", ("gpt",)),
        ]

    def has_any_credential(self, provider):
        return provider in ("openai", "openai-device")

    def credentials(self):
        return [_FakeCred(1, "openai", "oauth", {})]


class RealRegistryController(FakeProviderController):
    """The fake controller over the REAL provider registry.

    The descriptions are derived from registry names, so the only test that can
    fail when a name changes is one that reads the actual registry.
    """

    def login_providers(self):
        from local_operator.providers.registry import list_login_providers

        return list_login_providers()

    def has_any_credential(self, provider):
        return False

    def is_usable(self, provider):
        return False

    def credentials(self):
        return []


@pytest.mark.asyncio
async def test_model_switch_calls_session_set_model() -> None:
    session = FakeSession()
    set_models: list[Any] = []

    def set_model(spec, *, explicit=False):
        set_models.append(spec)

    session.set_model = set_model  # type: ignore[attr-defined]
    ctrl = FakeProviderController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        # /model openrouter/deepseek/deepseek-chat
        for key in "s", "p", "a", "c", "e":
            pass
        await pilot.press(
            "slash",
            "m",
            "o",
            "d",
            "e",
            "l",
            "space",
            "o",
            "p",
            "e",
            "n",
            "r",
            "o",
            "u",
            "t",
            "e",
            "r",
            "slash",
            "d",
            "e",
            "e",
            "p",
            "s",
            "e",
            "e",
            "k",
            "slash",
            "d",
            "e",
            "e",
            "p",
            "s",
            "e",
            "e",
            "k",
            "-",
            "c",
            "h",
            "a",
            "t",
            "enter",
        )
        await pilot.pause()
    assert len(set_models) == 1
    assert set_models[0].provider == "openrouter"


@pytest.mark.asyncio
async def test_provider_command_renders_listing() -> None:
    session = FakeSession()
    ctrl = FakeProviderController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("slash")
        for key in "p", "r", "o", "v", "i", "d", "e", "r":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        texts = _transcript_text(app)
    assert "openrouter" in texts
    assert "OpenRouter" in texts
    assert session.prompts == []


@pytest.mark.asyncio
async def test_accounts_command_renders_credentials() -> None:
    session = FakeSession()
    ctrl = FakeProviderController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("slash")
        for key in "a", "c", "c", "o", "u", "n", "t", "s":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        texts = _transcript_text(app)
    assert "openrouter" in texts
    assert "api_key (login)" in texts


def _failover_config(tmp_path, chains) -> None:
    """Write a real config.yml with ``chains`` under ``values.retry``.

    Through ConfigManager rather than a hand-written YAML file: the loader
    validates the document, and a hand-rolled one missing a bookkeeping key
    fails in the app as an unrelated error.
    """
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "anthropic")
    manager.set_config_value("model_name", "claude-opus-5")
    retry = dict(manager.get_config_value("retry", {}) or {})
    retry.update(chains)
    manager.set_config_value("retry", retry)


async def _failover_text(app) -> str:
    """Run `/failovers` once the session has attached, and read the transcript.

    The command reports the model's cascade, so running it before the session
    lands captures the "still starting" degrade rather than the listing.
    """
    for _ in range(200):
        if app._session is not None:
            break
        await asyncio.sleep(0.01)
    app._run_slash_command("/failovers")
    return _transcript_text(app)


@pytest.mark.asyncio
async def test_failovers_command_renders_the_cascade(monkeypatch, tmp_path) -> None:
    """The populated case: order, matched key, accounts, and what is serving."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(
        tmp_path,
        {
            "fallbackChains": {
                "test/*": [
                    {"provider": "openrouter", "model": "deepseek/deepseek-chat", "effort": "high"},
                    "deepseek/deepseek-reasoner",
                ]
            }
        },
    )
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    assert "failover cascade" in text
    # The matched key AND how it matched: an exact entry silently outranking a
    # wildcard is the confusion this row exists to end.
    assert "test/* (wildcard)" in text
    # Configured order is the routing order, numbered so it reads as a cascade.
    assert "1. openrouter/deepseek/deepseek-chat" in text
    assert "2. deepseek/deepseek-reasoner" in text
    # CONFIGURED effort, never a resolved spec (that would be a network call on
    # the paint path). Only stated when it DIFFERS from the model default: the
    # phrase was identical on every row of a real config, which is a column
    # carrying no bits and the width that pushed the serving marker into a wrap.
    assert "high effort" in text
    assert "model default effort" not in text
    # The fake never falls back, so the primary is what serves.
    assert "← serving" in text
    assert text.index("← serving") < text.index("1. openrouter")
    assert "ask the agent to change models, effort or providers" in text


@pytest.mark.asyncio
async def test_failovers_counts_accounts_without_printing_identities(monkeypatch, tmp_path) -> None:
    """Counts collapse login flavours; `/accounts` keeps the identities.

    The fake holds an openrouter key and a deepseek OAuth row carrying an email.
    Printing `row.data` here would leak an address onto a surface that never
    promised one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {"default": ["deepseek/deepseek-reasoner"]}})
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    assert "1 account" in text
    assert "a@b.c" not in text


@pytest.mark.asyncio
async def test_failovers_without_a_chain_points_at_the_key_to_set(monkeypatch, tmp_path) -> None:
    """The default install — nothing configured — is the most likely caller."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {}})
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    # The primary is still shown: "you have no cascade" is only useful beside
    # what would have had one.
    assert "primary" in text
    assert "values.retry.fallbackChains.default" in text


@pytest.mark.asyncio
async def test_failovers_names_the_key_that_disabled_the_cascade(monkeypatch, tmp_path) -> None:
    """Switched off reads as switched off, and names WHICH switch."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(
        tmp_path,
        {"modelFallback": False, "fallbackChains": {"default": ["deepseek/x"]}},
    )
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    assert "retry.modelFallback is false" in text
    # The configured chain must NOT be listed as a route: nothing would use it.
    assert "deepseek/x" not in text


@pytest.mark.asyncio
async def test_failovers_without_a_provider_facade_warns(monkeypatch, tmp_path) -> None:
    """Same shape as `/accounts`: name the terminal command that can answer."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {}})
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _failover_text(app)
        await pilot.pause()
        notices = [b._text for b in app.query(NoticeBlock)]

    assert any("local-operator config list" in n for n in notices)


@pytest.mark.asyncio
async def test_failovers_reports_an_unreadable_config_as_an_error(monkeypatch, tmp_path) -> None:
    """A config read that raises is named, not swallowed into an empty tree."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {}})
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for _ in range(200):
            if app._session is not None:
                break
            await asyncio.sleep(0.01)
        import local_operator.config as config_mod

        def explode(*args, **kwargs):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(config_mod, "ConfigManager", explode)
        app._run_slash_command("/failovers")
        await pilot.pause()
        notices = [b._text for b in app.query(NoticeBlock)]

    assert any("failover list failed: database is locked" in n for n in notices)


@pytest.mark.asyncio
async def test_failovers_before_the_session_attaches_warns(monkeypatch, tmp_path) -> None:
    """The pre-attach state, which the other tests deliberately wait past.

    Reachable in practice: the command is typeable while the session is still
    booting, and the cascade is a property of the session's model.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {}})
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        # NO wait for attach — that is the state under test.
        app._session = None
        app._run_slash_command("/failovers")
        await pilot.pause()
        notices = [b._text for b in app.query(NoticeBlock)]

    assert any("session is still starting" in n for n in notices)


@pytest.mark.asyncio
async def test_failovers_on_a_follower_whose_model_raises_warns(monkeypatch, tmp_path) -> None:
    """`RemoteSession.model` is a property that RAISES before the owner syncs.

    `getattr(session, "model", None)` does NOT suppress an exception raised
    inside a property, so this used to surface as `failover list failed: owner
    has no selected model spec` — an error notice carrying developer vocabulary
    for a state that has a warning phrasing.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {}})

    class _RaisingSpec(FakeSession):
        """A follower whose owner has not published a model spec yet.

        Subclasses the fake so the session plumbing the app installs on adopt
        (subscribe/dispose) still exists; only the spec accessors raise.
        """

        @property
        def model_label(self) -> str:
            return ""

        @property
        def model(self):
            raise RuntimeError("owner has no selected model spec")

        @property
        def effective_model(self):
            raise RuntimeError("owner has no effective model spec")

        @property
        def effective_model_label(self) -> str:
            return ""

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for _ in range(200):
            if app._session is not None:
                break
            await asyncio.sleep(0.01)
        # Swapped in AFTER attach: `_adopt_session` reads `.model` on the boot
        # path too, and that unrelated call site is out of this slice's scope.
        app._session = _RaisingSpec()
        app._run_slash_command("/failovers")
        await pilot.pause()
        # `_token` is where NoticeBlock keeps the resolved kind; asserting on it
        # is what pins the REGISTER rather than just the words.
        notices = [(b._text, b._token) for b in app.query(NoticeBlock)]

    matched = [(text, token) for text, token in notices if "no model selected yet" in text]
    assert matched, notices
    # Nothing is broken, so this must not be the error tier.
    assert matched[0][1] != "error"
    assert not any("failover list failed" in text for text, _ in notices)


@pytest.mark.asyncio
async def test_failovers_matched_key_with_only_the_current_model_is_not_no_match(
    monkeypatch, tmp_path
) -> None:
    """A key CAN match and still expand to nothing, and that is not "no match".

    `expand_fallback_targets` drops a legacy entry equal to the current
    selector, so `default: ["test/model"]` on a `test/model` primary matched
    `default` and produced []. Saying "set fallbackChains.default" there names a
    key that is already set and suggests a no-op.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {"default": ["test/model"]}})
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    assert "default matched but lists only the current model" in text
    # The no-op suggestion must NOT be offered for a key that already matched.
    assert "set values.retry.fallbackChains.default" not in text


@pytest.mark.asyncio
async def test_failovers_marks_exactly_one_serving_row(monkeypatch, tmp_path) -> None:
    """Same model at a different effort is a real route, and only ONE serves.

    `FallbackTarget` is (selector, effort), so a selector-only comparison put
    the marker on both the primary and the target — worst in exactly the case
    the marker exists to disambiguate ("am I on high or low?").
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(
        tmp_path,
        {"fallbackChains": {"default": [{"provider": "test", "model": "model", "effort": "low"}]}},
    )

    class LowEffortSpec:
        provider = "test"
        model_id = "model"
        reasoning_effort = "low"
        display_name = ""

    class ServingLowSession(FakeSession):
        @property
        def model(self):
            spec = LowEffortSpec()
            spec.reasoning_effort = "high"  # the SELECTION is high
            return spec

        @property
        def effective_model(self):
            return LowEffortSpec()  # the low-effort route is answering

    app = OperatorApp(
        lambda: _factory(ServingLowSession()), provider_controller=FakeProviderController()
    )
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    assert text.count("← serving") == 1, text
    # It belongs to the low-effort TARGET, not the high-effort primary.
    low_row = [line for line in text.splitlines() if "low effort" in line]
    assert low_row and "← serving" in low_row[0], text


@pytest.mark.asyncio
async def test_failovers_does_not_call_a_usable_provider_accountless(monkeypatch, tmp_path) -> None:
    """A routing listing must not report a working hop as `no accounts`.

    The fake's `deepseek` has an OAuth row; `openrouter` has a key. A provider
    that needs no credential at all (`allows_missing_api_key`) is a healthy
    route, and calling it `no accounts` reads as "this hop will fail" and sends
    the user to a pointless login.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"fallbackChains": {"default": ["ollama/llama3"]}})

    class OllamaDef:
        id = "ollama"
        name = "Ollama"
        store_credentials_as = None
        allows_missing_api_key = True
        login = None
        search_aliases = ()

    class OllamaController(FakeProviderController):
        def provider(self, pid):
            return OllamaDef() if pid == "ollama" else super().provider(pid)

        def is_usable(self, provider):
            return provider == "ollama" or super().is_usable(provider)

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=OllamaController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    ollama_row = [line for line in text.splitlines() if "ollama/llama3" in line]
    assert ollama_row, text
    assert "no credential needed" in ollama_row[0]
    assert "no accounts" not in ollama_row[0]


@pytest.mark.asyncio
async def test_failovers_describes_what_the_session_routes_on(monkeypatch, tmp_path) -> None:
    """The listing must describe what the SESSION routes on, never the file.

    The session's ``routing_settings`` is the mapping the stream will actually
    use — kept current by the process config watcher — so a listing built from
    disk would answer a different question. The old ``stale · /reload`` row is
    gone with the gap it described: nothing must reintroduce a "reload to
    apply" hint for a change that already applied.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # On DISK: two targets. The session (a test double the watcher cannot
    # update) reports only the first — the listing follows the session.
    _failover_config(tmp_path, {"fallbackChains": {"default": ["zai/glm-5.3", "kimi/k3"]}})

    class SnapshotSession(FakeSession):
        @property
        def routing_settings(self):
            return {"retry": {"fallbackChains": {"default": ["zai/glm-5.3"]}}}

    app = OperatorApp(
        lambda: _factory(SnapshotSession()), provider_controller=FakeProviderController()
    )
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        text = await _failover_text(app)
        await pilot.pause()

    # What the SESSION will do, not what the file says.
    assert "1. zai/glm-5.3" in text
    assert "kimi/k3" not in text
    # No drift row and no reload hint: the change path is live now.
    assert "stale" not in text
    assert "/reload" not in text


@pytest.mark.asyncio
async def test_failovers_offers_the_agent_affordance_in_every_state(monkeypatch, tmp_path) -> None:
    """The user with NO cascade needs the "ask the agent" hint most."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _failover_config(tmp_path, {"modelFallback": False, "fallbackChains": {}})
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        disabled = await _failover_text(app)
        await pilot.pause()

    assert "ask the agent to turn the cascade back on" in disabled


def _usage_reports(*, used: float = 5.0):
    from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport

    return [
        UsageReport(
            provider="openrouter",
            limits=[
                UsageLimit(
                    id="openrouter:credits",
                    label="Credits",
                    amount=UsageAmount(used=used, limit=50.0, unit="usd"),
                )
            ],
        )
    ]


class _ControlledUsageController(FakeProviderController):
    """Keeps the first network request pending so ordering is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.first_release = asyncio.Event()
        self.first_started = asyncio.Event()
        self.first_cancelled = False

    async def fetch_usage(self, provider_ids=None, *, force_refresh: bool = False):
        self.usage_calls.append(provider_ids)
        if len(self.usage_calls) == 1:
            self.first_started.set()
            try:
                await self.first_release.wait()
            except asyncio.CancelledError:
                self.first_cancelled = True
                raise
            return _usage_reports(used=5.0)
        return _usage_reports(used=42.0)


async def _run_usage_command(pilot, app) -> None:
    """Type `/usage` and submit it, the way a user reaches the panel."""
    app.query_one(Editor).focus()
    await pilot.press("slash")
    for key in "u", "s", "a", "g", "e":
        await pilot.press(key)
    await pilot.press("enter")
    for _ in range(4):
        await pilot.pause()


@pytest.mark.asyncio
async def test_usage_command_opens_the_panel_with_the_report() -> None:
    """`/usage` opens the popup rather than appending to the transcript.

    A quota report is reference material: appended as a block it was pushed off
    screen by the next turn and could not be re-read without re-fetching.
    """
    from local_operator.tui.widgets.usage_panel import UsagePanel

    session = FakeSession()
    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        assert panel.is_open
        text = "\n".join(panel.render_lines_for_test())
        assert app.focused is panel
    assert "openrouter" in text
    assert "Credits" in text


@pytest.mark.asyncio
async def test_usage_opens_instantly_from_the_cache_when_a_row_is_warm() -> None:
    """The whole point of the shared cache: when a row is on hand, `/usage`
    paints it immediately (with its age) rather than showing "fetching…" and
    making the user wait for a network round they did not need."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    session = FakeSession()
    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    # Stage a warm cache row: the panel must paint THIS before the fetch lands.
    ctrl.cached_reports = _usage_reports(used=5.0)
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        assert panel.is_open
        text = "\n".join(panel.render_lines_for_test())
    # The cached row painted immediately: no "fetching…", the report is there.
    assert "fetching…" not in text
    assert "openrouter" in text
    assert "Credits" in text


@pytest.mark.asyncio
async def test_an_empty_successful_fetch_replaces_cached_numbers() -> None:
    """`[]` from the controller is a REAL answer after the empty-over-data
    acceptance window ("this provider reports nothing"). Treating it as a
    failed refresh would keep the stale numbers the cache just stopped serving
    and pin a "refresh failed" note on the wrong arm."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    session = FakeSession()
    ctrl = FakeProviderController()
    ctrl.cached_reports = _usage_reports(used=5.0)
    ctrl.usage_reports = []  # the fetch's answer: nothing to report
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        for _ in range(6):
            await pilot.pause()
        text = "\n".join(panel.render_lines_for_test())
    assert "refresh failed" not in text
    assert "Credits" not in text  # the stale numbers were replaced
    assert "no usage" in text


@pytest.mark.asyncio
async def test_the_background_warmer_only_fetches_when_the_row_is_stale() -> None:
    """The warmer is an optimisation, not a fetch-per-interval: it must skip the
    network when the shared cache already holds a fresh row for the active
    provider, and only fire when that row is missing or going stale."""

    class _SessionWithModel(FakeSession):
        @property
        def model(self):
            return FakeModel("openrouter", "deepseek/deepseek-chat")

    session = _SessionWithModel()
    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # No cached row yet (the fake's `usage_cache_age_ms` returns None), so
        # the warmer must fire a background fetch for the active provider.
        before = len(ctrl.usage_calls)
        app._warm_usage_background()
        for _ in range(4):
            await pilot.pause()
        assert len(ctrl.usage_calls) == before + 1
        assert ctrl.usage_calls[-1] == ["openrouter"]

        # Now the row is warm: a second warm tick must NOT fetch again.
        ctrl.usage_cache_age = 0  # pretend the row is fresh
        before = len(ctrl.usage_calls)
        app._warm_usage_background()
        for _ in range(4):
            await pilot.pause()
        assert len(ctrl.usage_calls) == before


@pytest.mark.asyncio
async def test_escape_closes_the_usage_panel_and_returns_focus() -> None:
    """The panel takes focus to receive its keys, so it must hand focus back —
    otherwise dismissing it leaves the user typing into nothing."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        assert panel.is_open
        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        assert not panel.is_open
        assert isinstance(app.focused, Editor)


@pytest.mark.asyncio
async def test_dismissed_usage_request_cannot_reopen_the_panel() -> None:
    """Esc closes the request as well as its card; a late network response must
    not reverse an explicit user action."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = _ControlledUsageController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.first_started.wait()
        await pilot.press("escape")
        for _ in range(4):
            await pilot.pause()
        ctrl.first_release.set()
        for _ in range(4):
            await pilot.pause()

        assert ctrl.first_cancelled
        assert not panel.is_open


@pytest.mark.asyncio
async def test_usage_result_ready_between_close_and_dismiss_handler_is_ignored() -> None:
    """Closing invalidates the request synchronously, before the app can receive
    the dismissal message and cancel its worker."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = _ControlledUsageController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.first_started.wait()

        # This is the exact cross-queue gap inside `action_dismiss`: the panel is
        # already closed, while the app has not yet handled `UsageDismissed`.
        panel.close()
        ctrl.first_release.set()
        for _ in range(4):
            await pilot.pause()

        assert not ctrl.first_cancelled
        assert not panel.is_open


@pytest.mark.asyncio
async def test_usage_refresh_supersedes_a_slower_request() -> None:
    """A stale first response must not overwrite the report returned by the
    refresh that replaced it."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = _ControlledUsageController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.first_started.wait()
        await pilot.press("r")
        for _ in range(6):
            await pilot.pause()
        ctrl.first_release.set()
        for _ in range(4):
            await pilot.pause()
        text = "\n".join(panel.render_lines_for_test())

        assert ctrl.first_cancelled
        assert len(ctrl.usage_calls) == 2
        assert "84%" in text
        assert "10%" not in text


@pytest.mark.asyncio
async def test_r_refetches_without_closing_the_panel() -> None:
    """Refresh is the whole reason the panel holds focus: the numbers go stale
    while they are being read, and re-typing the command to see new ones would
    make the panel worse than the transcript block it replaced."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        before = len(ctrl.usage_calls)
        await pilot.press("r")
        for _ in range(4):
            await pilot.pause()
        assert len(ctrl.usage_calls) == before + 1
        assert panel.is_open


@pytest.mark.asyncio
async def test_scrolling_during_an_in_flight_usage_fetch_keeps_place_when_it_lands() -> None:
    """Open `/usage` from cache, scroll while the confirming fetch is in flight:
    when it lands the view must not jump to the top."""
    from local_operator.tui.widgets.usage_panel import UsagePanel
    from tests.unit.tui.test_usage_panel import _many_reports

    class _HeldMany(FakeProviderController):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()
            self.started = asyncio.Event()
            self.cached_reports = _many_reports()

        async def fetch_usage(self, provider_ids=None, *, force_refresh: bool = False):
            self.usage_calls.append(provider_ids)
            self.started.set()
            await self.release.wait()
            return _many_reports()

    ctrl = _HeldMany()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.started.wait()
        panel.action_scroll_page(1)
        offset = panel.view_offset
        assert offset > 0
        ctrl.release.set()
        for _ in range(6):
            await pilot.pause()
        assert panel.view_offset == offset
        assert "refreshing…" not in "\n".join(panel.render_lines_for_test())


@pytest.mark.asyncio
async def test_r_on_a_scrolled_usage_panel_does_not_jump_to_the_top() -> None:
    """`keep_offset` holds during ``refreshing…``; the finished fetch must too."""
    from local_operator.tui.widgets.usage_panel import UsagePanel
    from tests.unit.tui.test_usage_panel import _many_reports

    ctrl = FakeProviderController()
    ctrl.usage_reports = _many_reports()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        panel.action_scroll_page(1)
        offset = panel.view_offset
        assert offset > 0
        await pilot.press("r")
        for _ in range(6):
            await pilot.pause()
        assert panel.view_offset == offset
        assert panel.is_open
        assert "refreshing…" not in "\n".join(panel.render_lines_for_test())


@pytest.mark.asyncio
async def test_r_moves_the_header_even_when_one_account_stays_stuck() -> None:
    """The reported bug, driven through the key the operator actually pressed.

    Reported against v0.44.38: five Anthropic logins refreshed 1.8 minutes
    earlier, one Kimi account serving 169-minute-old last-good from its
    per-account backoff, and a title stuck on ``2h ago`` that ``r`` would not
    move. It could not move: a forced re-probe that misses again keeps the
    PREVIOUS report object (``ProviderController._mark_account_failure``) with
    its old ``fetched_at``, so a header taken from the oldest stamp returned the
    same 169-minute reading on every press.

    So the assertion is that the header ADVANCES across successive forced
    refreshes while the stuck account is still missing — the fetch really is
    running, and the title has to show it.
    """
    from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
    from local_operator.tui.widgets.usage_panel import UsagePanel

    now_ms = 200 * 60_000.0

    def _limit(limit_id: str, percent: float) -> UsageLimit:
        return UsageLimit(
            id=limit_id,
            label="5 hour",
            amount=UsageAmount(
                used=percent,
                limit=100.0,
                remaining=100.0 - percent,
                used_fraction=percent / 100.0,
                unit="percent",
            ),
            shared=True,
        )

    # Created once and handed back by identity on every fetch: that object
    # identity IS the mechanism, not an approximation of it.
    stuck = UsageReport(
        provider="kimi",
        identity="cred:8",
        fetched_at=int(now_ms - 169 * 60_000),
        limits=[_limit("kimi:5h", 64.0)],
    )
    stuck.consecutive_failures = 1

    class _OneStuckAccount(FakeProviderController):
        def __init__(self) -> None:
            super().__init__()
            self.now = now_ms

        def _set(self):
            healthy = UsageReport(
                provider="anthropic",
                identity="a@example.com",
                fetched_at=int(self.now),
                limits=[_limit("anthropic:5h", 20.0)],
            )
            stuck.consecutive_failures += 1  # the miss is re-counted, the stamp is not
            return [healthy, stuck]

        async def fetch_usage(self, provider_ids=None, *, force_refresh: bool = False):
            self.usage_calls.append(provider_ids)
            return self._set()

    ctrl = _OneStuckAccount()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        for _ in range(6):
            await pilot.pause()
        opened_ms = panel.fetched_ms

        seen = []
        for step in (1, 2):
            ctrl.now = now_ms + step * 5 * 60_000
            await pilot.press("r")
            for _ in range(8):
                await pilot.pause()
            seen.append(panel.fetched_ms)

        panel.set_clock(ctrl.now)
        panel._repaint()
        await pilot.pause()
        panel.action_scroll_end()
        for _ in range(4):
            await pilot.pause()
        rows = panel.render_lines_for_test()

    assert opened_ms == now_ms
    # Each forced refresh reports the confirmation it just obtained.
    assert seen == [now_ms + 5 * 60_000, now_ms + 10 * 60_000], seen
    # And the account that is genuinely stale still says so, on its own block.
    assert any("last known" in row for row in rows), rows
    assert stuck.fetched_at == int(now_ms - 169 * 60_000)  # never advanced


@pytest.mark.asyncio
async def test_a_failed_fetch_is_reported_inside_the_panel() -> None:
    """The panel is what has focus and what carries the key that retries, so an
    error anywhere else asks the user to look away from the fix."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = FakeProviderController()
    ctrl.usage_error = RuntimeError("network is down")
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        text = "\n".join(panel.render_lines_for_test())
        assert panel.is_open
    assert "network is down" in text
    assert "r refresh" in text


@pytest.mark.asyncio
async def test_all_three_usage_surfaces_agree_for_an_api_key_only_install(monkeypatch) -> None:
    """`/provider`'s "report quota" list, bare `/usage`'s targets and
    `/usage <provider>`'s up-front warning are three surfaces answering one
    question, and they used to give three answers: with only `ANTHROPIC_API_KEY`
    set, `/provider` advertised anthropic, bare `/usage` rendered "no usage data",
    and `/usage anthropic` correctly said it needs a login."""
    from local_operator.providers.controller import (
        ControllerAuthStore,
        ProviderController,
    )
    from tests.unit.providers.test_controller import _USAGE_ENV_VARS, FakeAuthStore

    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    controller = ProviderController(
        cast(ControllerAuthStore, FakeAuthStore()), login_callbacks=None
    )
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)

    # Surface 1: `/provider` must not advertise what `/usage` cannot deliver.
    assert app._provider_usage_state() == []
    # Surface 2: the bare `/usage` target list is the same list.
    assert controller.usage_reportable_providers() == []
    # Surface 3: `/usage anthropic` refuses up front, with the actionable reason.
    #
    # Run under `run_test` rather than with an injected notice callback: the
    # REFUSING branches go through `_system_notice`, so they keep the boot
    # composition a rejected command never earned the right to collapse (see
    # `_cmd_usage`), and that method writes to the transcript rather than to the
    # `notice` parameter.
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._cmd_usage("anthropic", app._notice)
        await pilot.pause()
        notices = [
            (block._text, block._token)
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        welcome_kept = app.query_one(WelcomeView).display
    assert notices == [("anthropic reports usage only after /login anthropic", "warning")]
    assert welcome_kept is True


@pytest.mark.asyncio
async def test_provider_and_the_login_list_report_credentials_in_the_same_words() -> None:
    """Two surfaces, one question, one vocabulary.

    `/provider` rendered a provider with no credential as `—` while the `/login`
    picker called the same provider `needs login`. A dash is not an answer: a
    user with no credential reads a dash and cannot tell "none", "unknown" and
    "not supported" apart.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        _set_editor_line(app.query_one(Editor), "/provider")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        listed = [
            line
            for line in _transcript_text(app).split("\n")
            if "deepseek" in line or "xai-oauth" in line
        ]
        _set_editor_line(app.query_one(Editor), "/login ")
        await pilot.pause()
        states = dict(_provider_rows(app))

    assert listed, "premise: /provider rendered its listing"
    assert all("needs login" in line for line in listed), listed
    assert all("—" not in line for line in listed), listed
    assert states["deepseek"] == "needs login", "and the picker says the same thing"


@pytest.mark.asyncio
async def test_run_tui_forwards_provider_controller(monkeypatch) -> None:
    """F3 regression: run_tui must pass provider_controller to OperatorApp so
    the slash-command surface is live (not a pointer, not a crash)."""
    import local_operator.tui.app as app_mod
    from local_operator.tui.app import OperatorApp

    seen: dict[str, Any] = {}
    fake_controller = object()

    class _SpyApp(OperatorApp):
        def __init__(self, *a, **kw):
            seen["controller"] = kw.get("provider_controller")
            super().__init__(*a, **kw)

        async def run_async(self, **kwargs: Any) -> None:
            return None

    # run_tui lazy-imports OperatorApp from local_operator.tui.app at call
    # time, so patching that module attribute is what routes the spy in.
    monkeypatch.setattr(app_mod, "OperatorApp", _SpyApp)
    called = []

    async def factory():
        called.append(1)
        return _SpyApp()  # type: ignore[return-value]

    # Await run_tui with a fake session factory that must not await forever.
    async def factory2() -> Any:
        return object()

    from local_operator.tui import run_tui

    await run_tui(factory2, theme_name="dark", provider_controller=fake_controller)
    assert seen["controller"] is fake_controller


# --- /goal and /loop -------------------------------------------------------


class GoalSession(FakeSession):
    """FakeSession with the goal surface and a recording prompt()."""

    def __init__(self) -> None:
        super().__init__()
        self._goal = ""
        self.fail_on_prompt = False
        #: Verdicts the goal-mode judge returns, consumed one per call. Default
        #: (empty list) => after the staged verdicts run out, answer ACHIEVED,
        #: so a test that stages nothing terminates rather than spinning the
        #: pilot forever.
        self.judge_verdicts: list[str] = []
        #: Number of consecutive `complete_aside` calls that should RAISE before
        #: any staged verdict is returned — exercises the judge-error path.
        self.judge_raises = 0
        self.judge_calls = 0
        #: When set, `prompt` awaits it before recording — lets a test observe
        #: the loop mid-flight (the fake is otherwise instantaneous and the
        #: worker settles before the test can look at `_loop_running`).
        self.prompt_gate: asyncio.Event | None = None
        #: When set to the running app, `prompt` POSTS a real `TurnEnded` via
        #: `post_message` on settle, exactly like the live controller
        #: (`tui/events.py` `_post`). This reproduces the queued-dispatch
        #: ordering that the plain fake could not — the R1 regression relies on
        #: it, because the completion toast fires from `on_turn_ended` off that
        #: queued event, not synchronously from `prompt`.
        self.post_turn_ended_to: Any = None

    @property
    def goal(self) -> str:
        return self._goal

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        if self.fail_on_prompt:
            raise RuntimeError("boom")
        if self.prompt_gate is not None:
            await self.prompt_gate.wait()
        self.prompts.append(text)
        if self.post_turn_ended_to is not None:
            # Mirror the live controller: a settled turn is announced through the
            # message pump, not returned inline, so the completion notify races
            # the worker's suppress-flag reset.
            self.post_turn_ended_to.post_message(
                TurnEnded(aborted=False, error=None, context_tokens=1000)
            )

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Scriptable judge: raise `judge_raises` times, then serve staged
        # verdicts, then default to ACHIEVED so an unstaged test terminates.
        self.judge_calls += 1
        if on_usage is not None:
            on_usage(SimpleNamespace())
        if self.judge_raises > 0:
            self.judge_raises -= 1
            raise RuntimeError("judge boom")
        if self.judge_verdicts:
            return self.judge_verdicts.pop(0)
        return "VERDICT: ACHIEVED\ndefault stop"


async def _type_command(pilot, app, command: str) -> None:
    app.query_one(Editor).focus()
    await pilot.press("slash")
    for ch in command:
        await pilot.press("space" if ch == " " else ch)
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_goal_set_show_and_clear() -> None:
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "goal ship it")
        assert session.goal == "ship it"
        await _type_command(pilot, app, "goal")
        assert "ship it" in _transcript_text(app)
        await _type_command(pilot, app, "goal clear")
        assert session.goal == ""


@pytest.mark.asyncio
async def test_loop_requires_a_goal() -> None:
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop")
        assert "set a goal first" in _transcript_text(app)
    assert session.prompts == []


@pytest.mark.asyncio
async def test_loop_runs_bounded_iterations() -> None:
    session = GoalSession()
    session.set_goal("finish the parser")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 2")
        for _ in range(12):
            await pilot.pause()
            if not app._loop_running:
                break
        text = _transcript_text(app)
    assert len(session.prompts) == 2
    assert "loop finished after 2" in text


@pytest.mark.asyncio
async def test_loop_rejects_out_of_range() -> None:
    # An out-of-range INTEGER is still an error. Non-integer text is no longer
    # an error at all — it is a goal (see the goal-mode dispatch tests below),
    # the one user-facing behaviour this feature removes.
    session = GoalSession()
    session.set_goal("g")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 99")
        text = _transcript_text(app)
    assert "between 1 and" in text
    assert session.prompts == []


@pytest.mark.asyncio
async def test_loop_botched_count_does_not_launch_goal_loop() -> None:
    # U2 — a number-shaped typo (`3e`, `5x`, `3.5`) must NOT silently start an
    # unbounded goal loop toward the literal typo; it errors with a hint.
    for typo in ("3e", "5x", "3.5", "12."):
        session = GoalSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await _type_command(pilot, app, f"loop {typo}")
            # Give any (erroneously launched) worker a chance to run a turn.
            for _ in range(6):
                await pilot.pause()
            text = _transcript_text(app)
        assert app._loop_running is False, f"{typo!r} launched a loop"
        assert session.prompts == [], f"{typo!r} ran a turn"
        assert "mistyped count" in text, f"{typo!r} gave no disambiguating hint"


@pytest.mark.asyncio
async def test_loop_goal_with_digit_prefix_words_still_launches() -> None:
    # A real goal that merely begins with a digit ("2fa the login flow") is
    # written in words and must still start goal mode, not hit the typo guard.
    session = GoalSession()
    session.prompt_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 2fa the login flow")
        assert app._loop_running is True
        assert app._loop_goal == "2fa the login flow"
        session.prompt_gate.set()
        await _settle_loop(pilot, app)


@pytest.mark.asyncio
async def test_loop_goal_launch_notice_names_the_goal() -> None:
    # U1 — the goal text must appear on screen at launch so there is a record of
    # what the loop is pursuing.
    session = GoalSession()
    session.prompt_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop finish the parser")
        text = _transcript_text(app)
        assert "looping toward: finish the parser" in text
        session.prompt_gate.set()
        await _settle_loop(pilot, app)


@pytest.mark.asyncio
async def test_loop_no_goal_hint_surfaces_inline_form() -> None:
    # U3 — bare `/loop` with no standing goal should point at the inline goal
    # form, not only at `/goal`.
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop")
        # Collapse wrap so an 80-col line break inside the hint doesn't matter.
        text = " ".join(_transcript_text(app).split())
    assert "loop toward one now: /loop <goal text>" in text
    assert session.prompts == []


@pytest.mark.asyncio
async def test_loop_stops_on_turn_error() -> None:
    session = GoalSession()
    session.set_goal("g")
    session.fail_on_prompt = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 5")
        for _ in range(12):
            await pilot.pause()
            if not app._loop_running:
                break
        text = _transcript_text(app)
    assert "loop stopped" in text  # did not spin through all 5


@pytest.mark.asyncio
async def test_interrupt_cancels_running_loop() -> None:
    session = GoalSession()
    session.set_goal("g")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._loop_running = True
        app.action_interrupt()
        assert app._loop_cancelled is True


# -- /loop goal mode ---------------------------------------------------------


async def _settle_loop(pilot, app, limit: int = 60) -> None:
    """Pump the pilot until the loop worker settles.

    Bounded so a regression that never releases fails as a timeout in the
    assertions rather than hanging the suite forever.
    """
    for _ in range(limit):
        await pilot.pause()
        if not app._loop_running:
            return


def _record_completion_notifies(app) -> list[int | None]:
    """Wrap `app._notify` so a test can count 'complete' notifications.

    Returns the list the wrapper appends to; the real notify still runs (it is a
    no-op headless, but the call is what a test asserts on).
    """
    fired: list[int | None] = []
    real_notify = app._notify

    def rec(kind: str, *, running_children: int | None = None) -> bool:
        if kind == "complete":
            fired.append(running_children)
        return real_notify(kind, running_children=running_children)

    app._notify = rec  # type: ignore[assignment]
    return fired


@pytest.mark.asyncio
async def test_loop_goal_dispatch_is_not_numeric() -> None:
    # `/loop <non-integer>` is a GOAL, not a usage error — the whole feature.
    session = GoalSession()
    # Hold the first turn so the loop is observable mid-flight.
    session.prompt_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop finish the parser")
        # The goal is captured before the worker exits; assert while it runs.
        assert app._loop_running is True
        assert app._loop_goal == "finish the parser"
        # Release the turn; the default ACHIEVED verdict then ends the loop.
        session.prompt_gate.set()
        await _settle_loop(pilot, app)
        text = _transcript_text(app)
    assert "usage: /loop" not in text
    assert session.prompts  # at least one turn ran


@pytest.mark.asyncio
async def test_loop_goal_releases_when_judge_says_achieved() -> None:
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        session.judge_verdicts = [
            "VERDICT: CONTINUE\nnot yet",
            "VERDICT: ACHIEVED\ndone",
        ]
        await _type_command(pilot, app, "loop do the thing")
        await _settle_loop(pilot, app)
        text = _transcript_text(app)
    assert len(session.prompts) == 2
    assert session.judge_calls == 2
    assert "goal achieved after 2" in text
    # All three fields reset by the worker's finally.
    assert app._loop_running is False
    assert app._loop_goal == ""
    assert app._loop_suppress_completion is False


@pytest.mark.asyncio
async def test_loop_goal_suppresses_turn_ended_toast() -> None:
    # The crux, tested at the seam: while the suppress flag is set, a settling
    # TurnEnded fires NO completion notify; with it clear, the same event does.
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        completion_notifies: list[int | None] = []
        real_notify = app._notify

        def _record_notify(kind: str, *, running_children: int | None = None) -> bool:
            if kind == "complete":
                completion_notifies.append(running_children)
            return real_notify(kind, running_children=running_children)

        app._notify = _record_notify  # type: ignore[assignment]
        # Held loop: a per-turn settle must not notify.
        app._loop_suppress_completion = True
        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=1_000))
        await pilot.pause()
        await pilot.pause()
        assert completion_notifies == []
        # Not held: the same event notifies as usual (numeric mode / normal turns).
        app._loop_suppress_completion = False
        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=1_000))
        await pilot.pause()
        await pilot.pause()
    assert len(completion_notifies) == 1


@pytest.mark.asyncio
async def test_loop_goal_notifies_once_on_release() -> None:
    # End to end: N held turns produce exactly one completion notify, on release.
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Post a real TurnEnded per turn (live-controller ordering) so the per-
        # turn toasts would fire if suppression leaked — they must not.
        session.post_turn_ended_to = app
        completion_notifies = _record_completion_notifies(app)
        session.judge_verdicts = [
            "VERDICT: CONTINUE\nnot yet",
            "VERDICT: CONTINUE\nstill not",
            "VERDICT: ACHIEVED\ndone",
        ]
        await _type_command(pilot, app, "loop do the thing")
        await _settle_loop(pilot, app)
        for _ in range(6):
            await pilot.pause()
    assert len(session.prompts) == 3
    # Exactly one completion notify — the release toast, not a per-turn toast.
    assert len(completion_notifies) == 1


@pytest.mark.asyncio
async def test_loop_goal_ignores_standing_goal_guard() -> None:
    # Goal mode starts with NO standing goal set; numeric mode still refuses.
    session = GoalSession()
    session.prompt_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert session.goal == ""
        await _type_command(pilot, app, "loop do the thing")
        assert app._loop_running is True
        session.prompt_gate.set()
        await _settle_loop(pilot, app)
        # Numeric with no goal still refuses.
        await _type_command(pilot, app, "loop 2")
        text = _transcript_text(app)
    assert "set a goal first" in text


@pytest.mark.asyncio
async def test_loop_goal_does_not_clobber_standing_goal() -> None:
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "goal keep me")
        assert session.goal == "keep me"
        session.judge_verdicts = ["VERDICT: ACHIEVED\ndone"]
        await _type_command(pilot, app, "loop something else")
        await _settle_loop(pilot, app)
    assert session.goal == "keep me"


@pytest.mark.asyncio
async def test_loop_goal_judge_error_continues_with_warning() -> None:
    # A judge that raises once must NOT stop or release; it continues with a
    # visible warning, then releases on the next readable verdict.
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        session.judge_raises = 1
        session.judge_verdicts = ["VERDICT: ACHIEVED\ndone"]
        await _type_command(pilot, app, "loop do the thing")
        await _settle_loop(pilot, app)
        text = _transcript_text(app)
    # Turn 1 -> judge raises (continue), turn 2 -> ACHIEVED release.
    assert len(session.prompts) == 2
    assert "judge unavailable, continuing" in text
    assert "goal achieved after 2" in text


@pytest.mark.asyncio
async def test_loop_goal_judge_failure_breaker() -> None:
    # A judge that never returns a readable verdict trips the breaker after
    # MAX_LOOP_JUDGE_FAILURES, rather than spinning forever.
    from local_operator.tui.app import MAX_LOOP_JUDGE_FAILURES

    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Empty string => no VERDICT line => unreadable => failure strike.
        session.judge_verdicts = [""] * (MAX_LOOP_JUDGE_FAILURES + 5)
        await _type_command(pilot, app, "loop do the thing")
        await _settle_loop(pilot, app)
        text = _transcript_text(app)
    assert len(session.prompts) == MAX_LOOP_JUDGE_FAILURES
    assert "judge could not decide" in text
    assert app._loop_suppress_completion is False


@pytest.mark.asyncio
async def test_loop_goal_stop_cancels() -> None:
    session = GoalSession()
    # Hold the first turn so the stop lands while the loop is genuinely running.
    session.prompt_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # R1 regression: the fake POSTS a real TurnEnded on settle like the live
        # controller, so a spurious completion toast after the suppress-flag
        # reset would fire here. Assert exactly ZERO 'complete' notifies.
        session.post_turn_ended_to = app
        completion_notifies = _record_completion_notifies(app)
        session.judge_verdicts = ["VERDICT: CONTINUE\nnot yet"] * 50
        await _type_command(pilot, app, "loop do the thing")
        assert app._loop_running is True
        await _type_command(pilot, app, "loop stop")
        session.prompt_gate.set()
        await _settle_loop(pilot, app)
        # Extra pumps so any still-queued TurnEnded has dispatched.
        for _ in range(6):
            await pilot.pause()
        text = _transcript_text(app)
    assert "loop cancelled" in text
    assert app._loop_running is False
    assert app._loop_goal == ""
    assert app._loop_suppress_completion is False
    assert completion_notifies == []  # no false "task complete" on a cancelled loop


@pytest.mark.asyncio
async def test_loop_goal_reload_safety() -> None:
    # Swapping app._session mid-loop must stop the worker cleanly and reset all
    # three fields (esp. the suppress flag, which would otherwise mute the NEXT
    # session's completion toasts).
    session = GoalSession()
    session.prompt_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # R1 regression on the reload path too: a queued TurnEnded from the
        # disposed session must not fire a completion toast after reset.
        session.post_turn_ended_to = app
        completion_notifies = _record_completion_notifies(app)
        session.judge_verdicts = ["VERDICT: CONTINUE\nnot yet"] * 50
        await _type_command(pilot, app, "loop do the thing")
        assert app._loop_running is True
        # Simulate a reload: the session the worker captured is no longer live.
        app._session = GoalSession()
        session.prompt_gate.set()
        await _settle_loop(pilot, app)
        for _ in range(6):
            await pilot.pause()
        text = _transcript_text(app)
    assert "stopped by reload" in text
    assert app._loop_running is False
    assert app._loop_goal == ""
    assert app._loop_suppress_completion is False
    assert completion_notifies == []  # no false "task complete" on a reloaded loop


@pytest.mark.asyncio
async def test_loop_goal_turn_error_stops() -> None:
    session = GoalSession()
    session.fail_on_prompt = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop do the thing")
        await _settle_loop(pilot, app)
        text = _transcript_text(app)
    assert "loop stopped" in text
    assert app._loop_running is False
    assert app._loop_goal == ""
    assert app._loop_suppress_completion is False


def test_parse_loop_verdict_units() -> None:
    from local_operator.tui.app import _parse_loop_verdict

    # Clean verdicts.
    assert _parse_loop_verdict("VERDICT: ACHIEVED\nall done") == (True, "all done")
    assert _parse_loop_verdict("VERDICT: CONTINUE\nnot yet") == (False, "not yet")
    # Lowercase / mixed case is fine (parser upper-cases).
    assert _parse_loop_verdict("verdict: achieved\nok")[0] is True
    # A "not achieved" phrasing must NOT read as achieved — the substring trap.
    assert _parse_loop_verdict("VERDICT: NOT ACHIEVED\nnope")[0] is not True
    # CONTINUE wins even if both tokens somehow appear.
    assert _parse_loop_verdict("VERDICT: CONTINUE ACHIEVED")[0] is False
    # Leading prose before the verdict line is tolerated.
    assert _parse_loop_verdict("thinking...\nVERDICT: ACHIEVED\nreason")[0] is True
    # No verdict line at all => unreadable => None.
    assert _parse_loop_verdict("") == (None, "")
    assert _parse_loop_verdict("just some prose\nno verdict here") == (None, "")
    # A verdict with no reason line => (bool, "").
    assert _parse_loop_verdict("VERDICT: ACHIEVED") == (True, "")

    # R2 — token-level negation. An ordinary word that merely CONTAINS "not"/
    # "n't" (nothing, cannot, another, note) must NOT flip a genuine ACHIEVED to
    # CONTINUE: substring matching did exactly that and spun the loop forever.
    assert _parse_loop_verdict("VERDICT: ACHIEVED, nothing else remains")[0] is True
    assert _parse_loop_verdict("VERDICT: ACHIEVED - on to another goal")[0] is True
    assert _parse_loop_verdict("VERDICT: ACHIEVED. Cannot do more.")[0] is True
    assert _parse_loop_verdict("VERDICT: ACHIEVED, note the caveat")[0] is True
    # But a real whole-word negator still reads as CONTINUE.
    assert _parse_loop_verdict("VERDICT: NOT ACHIEVED")[0] is False
    assert _parse_loop_verdict("VERDICT: NOT_ACHIEVED")[0] is False
    assert _parse_loop_verdict("VERDICT: ACHIEVED but can't verify")[0] is False
    # R6 — a curly apostrophe (U+2019, what editors/phones autocorrect to) in
    # the contraction must count as a negator too, or `can’t` reads as a false
    # RELEASE. False-continue is the safe side; a false release is the bug.
    assert _parse_loop_verdict("VERDICT: ACHIEVED but can\u2019t verify")[0] is False


# -- MCP status band + startup toast -----------------------------------------


class FakeMcpManager:
    """The five methods the app asks of a manager, plus a way to drive a drop."""

    def __init__(self, configured: list[str], connected: list[str]) -> None:
        self._configured = list(configured)
        self._connected = list(connected)
        self._callback: Any = None
        self.inner_calls: list[list[Any]] = []
        # Set by tests that exercise the login/logout/reauth paths: the
        # config lookup the app does before dispatching a subcommand, and the
        # record of what the workers were asked to do.
        self._configs: dict[str, Any] = {}
        self.disconnects: list[str] = []
        self.connects: list[tuple[str, Any]] = []

    def get_all_server_names(self) -> list[str]:
        return sorted(self._configured)

    def get_connected_servers(self) -> list[str]:
        return sorted(self._connected)

    def get_connection_status(self, name: str) -> str:
        return "connected" if name in self._connected else "disconnected"

    def get_server_config(self, name: str) -> Any:
        return self._configs.get(name)

    async def disconnect_server(self, name: str) -> None:
        self.disconnects.append(name)
        if name in self._connected:
            self._connected.remove(name)

    async def connect_configured_server(self, name: str, *, timeout_ms: Any = None) -> Any:
        # No browser in tests: the fake answers as an already-valid grant
        # would, so the login worker's success path is what gets exercised.
        self.connects.append((name, timeout_ms))
        if name not in self._connected:
            self._connected.append(name)
        return SimpleNamespace(tools=["tool-a", "tool-b"])

    def set_on_tools_changed(self, callback: Any) -> None:
        self._callback = callback

    @property
    def on_tools_changed(self) -> Any:
        return self._callback

    def install_incumbent(self) -> None:
        """Stand in for the composition root's own subscriber, which the app
        must chain rather than clobber."""
        self._callback = self.inner_calls.append

    def fire(self) -> None:
        """What ``set_on_tools_changed`` does on connect/disconnect."""
        assert self._callback is not None
        self._callback([])

    def drop(self, name: str) -> None:
        self._connected.remove(name)
        self.fire()


class McpSession(FakeSession):
    """A session carrying the two attributes the composition root records."""

    def __init__(self, manager: Any = None, startup: Any = None) -> None:
        super().__init__()
        self.mcp_manager = manager
        self.mcp_startup = startup


def _band(app) -> str:  # type: ignore[no-untyped-def]
    from textual.widgets import Static

    return app.query_one("#status-band", Static).render().plain


@pytest.mark.asyncio
async def test_no_mcp_means_no_segment_and_no_toast() -> None:
    """The whole feature is invisible on a machine that does not use MCP. A
    ``⊙ 0 MCP`` and a "0 servers" toast on every launch would be pure noise."""
    session = McpSession(manager=None, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "MCP" not in _band(app)
        assert app.query_one(Toast).display is False


@pytest.mark.asyncio
async def test_the_band_counts_connected_servers_and_the_toast_reports_startup() -> None:
    manager = FakeMcpManager(["github", "linear", "slack"], ["github", "linear"])
    startup = McpStartupOutcome(
        configured=("github", "linear", "slack"),
        connected=("github", "linear"),
        failures={"slack": "command not found: slack-mcp"},
        tool_count=31,
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "⊙ 2 MCP" in _band(app)
        toast = app.query_one(Toast)
        assert toast.display is True
        assert "2 of 3 servers up, 31 tools" in toast.message
        assert "slack" in toast.message


@pytest.mark.asyncio
async def test_a_server_dropping_updates_the_count_live() -> None:
    """The reference snapshots this count at boot and lets it go stale. Here
    ``set_on_tools_changed`` drives a repaint, so a server dying is visible."""
    manager = FakeMcpManager(["github", "linear"], ["github", "linear"])
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "⊙ 2 MCP" in _band(app)
        manager.drop("linear")
        for _ in range(4):
            await pilot.pause()
        band = _band(app)
        assert "⊙ 1 MCP" in band
        # …and the surviving server's neighbour being down turns the lamp: the
        # count alone cannot say whether 1 of 2 is a failure or a config change.
        assert app._mcp_status().failed is True


@pytest.mark.asyncio
async def test_the_app_chains_the_composition_roots_subscriber() -> None:
    """Clobbering the incumbent callback would freeze the agent's TOOL LIST at
    boot — a far worse bug than a stale counter. The app reads the incumbent and
    calls it from its own wrapper."""
    manager = FakeMcpManager(["github"], ["github"])
    manager.install_incumbent()
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        manager.fire()
        for _ in range(4):
            await pilot.pause()
        assert manager.inner_calls, "the incumbent tool-merge callback was dropped"


@pytest.mark.asyncio
async def test_a_failure_survives_the_toast_dismissing() -> None:
    """A toast that dismisses is not a record. The failure lands in the
    transcript as a notice AND is reachable through ``/mcp``, so the information
    outlives the five or ten seconds the overlay is up."""
    manager = FakeMcpManager(["slack"], [])
    startup = McpStartupOutcome(
        configured=("slack",),
        failures={"slack": "command not found: slack-mcp"},
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        assert toast.display is True
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.display is False
        assert toast.message == ""
        # The durable half — appended, but WITHOUT ending the empty state: the
        # conversation has not started just because a server failed to start, and
        # collapsing the boot composition on launch would mean a user with one
        # broken server never saw the centred prompt the toast interrupted.
        text = _transcript_text(app)
        assert "MCP slack failed: command not found: slack-mcp" in text
        welcome = app.query_one(WelcomeView)
        assert welcome.display is True, "an infrastructure notice must not retire the splash"


@pytest.mark.asyncio
async def test_mcp_command_reports_per_server_state_not_just_the_config() -> None:
    """``/mcp`` used to dump the configured command and never say whether it
    worked, which is the only question it gets run to answer."""
    from local_operator.mcp.config import MCPStdioServerConfig

    manager = FakeMcpManager(["slack"], [])
    startup = McpStartupOutcome(
        configured=("slack",),
        failures={"slack": "command not found: slack-mcp"},
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        block = None
        with patch(
            "local_operator.mcp.config.load_all_mcp_configs",
            return_value=({"slack": MCPStdioServerConfig(command="slack-mcp")}, {}),
        ):
            block = app._mcp_block()
        assert block is not None
        listing = _renderable_plain(block.renderable)
        assert "slack" in listing
        assert "disconnected" in listing
        assert "command not found: slack-mcp" in listing


@pytest.mark.asyncio
async def test_mcp_command_puts_the_status_in_a_column() -> None:
    """Crammed into the detail string the status landed wherever the name ended,
    so the shorter name pushed the longer status LEFT and the two facts a reader
    scans for formed no column. Both fields are padded to their widest."""
    from local_operator.mcp.config import MCPStdioServerConfig

    manager = FakeMcpManager(["github", "gh"], ["github"])
    startup = McpStartupOutcome(
        configured=("github", "gh"),
        connected=("github",),
        failures={"gh": "command not found: gh"},
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        configs = {
            "github": MCPStdioServerConfig(command="npx -y server-github"),
            "gh": MCPStdioServerConfig(command="gh mcp serve"),
        }
        with patch("local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})):
            block = app._mcp_block()
        assert block is not None
        # Branch rows only: the listing now leads with a dim caption naming what
        # it lists (the block is its own receipt, so it has to say what it is),
        # and this test is about the SERVER rows' column alignment.
        rows = [
            row
            for row in _renderable_plain(block.renderable).split("\n")
            if row.lstrip().startswith(("├─", "└─"))
        ]
        assert len(rows) == 2
        # The caption is what makes the block its own receipt now that `/mcp`
        # no longer echoes; the filter above discards it, so it is pinned here.
        assert _renderable_plain(block.renderable).splitlines()[0] == "MCP servers"
        # `connected` is a substring of `disconnected`, so each row is located by
        # its own status word and the two start columns compared directly. Before
        # the fix the SHORTER name pushed the LONGER status four cells left.
        connected_at = next(row.index("connected") for row in rows if "disconnected" not in row)
        disconnected_at = next(row.index("disconnected") for row in rows if "disconnected" in row)
        assert connected_at == disconnected_at, rows
        # The detail after the status column lines up too, or the padding only
        # moved the ragged edge one field to the right.
        assert rows[0].index("npx") == rows[1].index("command not found"), rows


@pytest.mark.asyncio
async def test_mcp_logout_removes_the_credential_and_disconnects() -> None:
    """``/mcp logout <name>`` forgets the stored OAuth row AND tears the live
    connection down — deleting the row alone would leave the session's tools
    authenticated with a grant the user just revoked."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        )
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, {}),
            ),
            patch("local_operator.mcp.auth.mcp_logout_server", return_value=None) as logout,
        ):
            await _type_command(pilot, app, "mcp logout linear")
            for _ in range(6):
                await pilot.pause()
        assert logout.call_count == 1
        assert manager.disconnects == ["linear"]
        text = _transcript_text(app)
        assert "logged out of MCP server 'linear'" in text


@pytest.mark.asyncio
async def test_mcp_logout_of_a_server_with_no_credential_refuses() -> None:
    """The destructive row must not report success when nothing was removed:
    'logged out' after a no-op deletion would claim a grant was revoked that
    never existed."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        )
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, {}),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logout_server",
                return_value="no stored credential for MCP server 'linear' — nothing to log out of",
            ),
        ):
            await _type_command(pilot, app, "mcp logout linear")
            for _ in range(6):
                await pilot.pause()
        assert manager.disconnects == [], "a failed removal must not disconnect anyway"
        # The notice wraps across transcript rows at this width, so match the
        # phrase that cannot split.
        assert "no stored credential" in _transcript_text(app)


@pytest.mark.asyncio
async def test_mcp_reauth_removes_then_runs_a_fresh_grant() -> None:
    """Reauth is logout + login as ONE step: plain login reuses a stored
    client registration or refreshable token, which is exactly what an account
    switch cannot afford. The login only starts once the old row is gone."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        )
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, {}),
            ),
            patch("local_operator.mcp.auth.mcp_logout_server", return_value=None) as logout,
        ):
            await _type_command(pilot, app, "mcp reauth linear")
            for _ in range(20):
                await pilot.pause()
                if manager.disconnects and manager.connects:
                    break
        assert logout.call_count == 1
        assert manager.disconnects == ["linear"]
        assert manager.connects == [("linear", 600_000)]
        assert "authenticated MCP server 'linear'" in _transcript_text(app)


@pytest.mark.asyncio
async def test_mcp_login_abandoned_in_the_browser_gets_a_cancel_receipt() -> None:
    """The "logging in…" line must always get an ending.

    The reported defect: close the browser tab (or walk away from the consent
    screen) and the transcript kept a "logging in to MCP server linear…" line
    that nothing ever answered — the flow sat on its idle clock in silence.
    The flow now settles that as McpLoginCancelledError and the worker reports
    it as a cancellation, not as a red failure: the user is the one who left.
    """
    from local_operator.mcp.auth import McpLoginCancelledError
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        )
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs

    async def abandoned_grant(name: str, *, timeout_ms: Any = None) -> Any:
        raise McpLoginCancelledError(
            "no redirect arrived within 10 minutes — the login was probably "
            "cancelled (browser tab closed, or the authorization left unfinished)."
        )

    manager.connect_configured_server = abandoned_grant  # type: ignore[method-assign]
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        with patch(
            "local_operator.mcp.config.load_all_mcp_configs",
            return_value=(configs, {}),
        ):
            await _type_command(pilot, app, "mcp login linear")
            for _ in range(8):
                await pilot.pause()
        text = _transcript_text(app)
        assert "logging in to MCP server linear" in text
        assert "MCP login for 'linear' cancelled" in text
        assert "tab closed" in text
        assert "MCP login failed" not in text


@pytest.mark.asyncio
async def test_mcp_login_worker_cancellation_is_acknowledged() -> None:
    """A worker cancelled by its exclusive group must still close the receipt.

    Textual never delivers the CancelledError to a worker cancelled this way —
    the coroutine simply stops — so without the handler the second login left
    the first one's "logging in…" line permanently unanswered.
    """
    import asyncio as _asyncio

    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        )
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        # The grant must still be in flight when the cancel lands, or the test
        # cancels a finished worker and proves nothing — so the fake connect
        # parks the way a real browser round trip does.
        entered = _asyncio.Event()

        async def hanging_grant(name: str, *, timeout_ms: Any = None) -> Any:
            entered.set()
            await _asyncio.Event().wait()  # the browser never answers
            raise AssertionError("unreachable")

        manager.connect_configured_server = hanging_grant  # type: ignore[method-assign]
        # Drive the worker directly: the group-exclusivity cancellation that
        # strands the receipt in the real UI is a worker.cancel(), which is
        # what this reproduces without racing two typed commands.
        worker = app._mcp_login_worker(manager, "linear")
        task = _asyncio.ensure_future(worker)
        await _asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(_asyncio.CancelledError):
            await task
        await pilot.pause()
        assert "MCP login for 'linear' cancelled" in _transcript_text(app)


@pytest.mark.asyncio
async def test_mcp_reauth_never_logs_in_on_top_of_a_surviving_row() -> None:
    """A login over a row that failed to delete is NOT a re-auth — the stored
    registration would short-circuit the grant — so a failed removal stops the
    chain instead of popping a misleading browser tab."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        )
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, {}),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logout_server",
                return_value="no stored credential for MCP server 'linear' — nothing to log out of",
            ),
        ):
            await _type_command(pilot, app, "mcp reauth linear")
            for _ in range(8):
                await pilot.pause()
        assert manager.connects == [], "the grant must not start when removal failed"
        assert "MCP reauth failed" in _transcript_text(app)


@pytest.mark.asyncio
async def test_mcp_unknown_subcommand_names_the_three_verbs() -> None:
    """Before logout/reauth existed, mistyping `/mcp relogin` was told only
    about login — the one discoverability surface the verbs have besides the
    argument list."""
    manager = FakeMcpManager(["linear"], ["linear"])
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        await _type_command(pilot, app, "mcp relogin linear")
        for _ in range(4):
            await pilot.pause()
        text = _transcript_text(app)
        assert "unknown mcp subcommand" in text
        # Every verb the command accepts, so the one discoverability surface
        # besides the argument list stays complete as verbs are added.
        assert "list|add|remove|login|logout|reauth" in text


@pytest.mark.asyncio
async def test_mcp_argument_list_offers_subcommands_then_servers() -> None:
    """The suggestion UX: typing `/mcp ` offers the three verbs; once a verb
    is chosen the SAME list turns into the OAuth servers that verb can act
    on, as ready-to-run `login notion`-style rows."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "stdio": MCPHttpServerConfig(url="https://stdio.example/mcp"),
    }
    manager = FakeMcpManager(["linear", "stdio"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        editor = app.query_one(Editor)
        with patch(
            "local_operator.mcp.config.load_all_mcp_configs",
            return_value=(configs, {}),
        ):
            _set_editor_line(editor, "/mcp ")
            for _ in range(6):
                await pilot.pause()
            verb_rows = [name for name, _ in editor.picker.suggestions()]
            # `list` leads: the safe, non-destructive row is the one a stray
            # Enter lands on.
            assert verb_rows == [
                "list",
                "add",
                "remove",
                "login",
                "logout",
                "reauth",
            ], verb_rows

            _set_editor_line(editor, "/mcp login ")
            for _ in range(6):
                await pilot.pause()
            server_rows = [name for name, _ in editor.picker.suggestions()]
            assert server_rows == ["login linear"], server_rows

            # Narrowing still matches against the WHOLE argument: `login lin`
            # must keep offering the compound row, or the row the user is
            # looking at would vanish the moment they filtered to it.
            _set_editor_line(editor, "/mcp login lin")
            for _ in range(6):
                await pilot.pause()
            assert [name for name, _ in editor.picker.suggestions()] == ["login linear"]


@pytest.mark.asyncio
async def test_mcp_logout_list_offers_only_servers_holding_a_credential() -> None:
    """The `/logout` rule applied to MCP: a logout row for a server with no
    stored grant promises a removal that can only end in a warning, so the
    list is pre-filtered by the credential store (keyed by URL)."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "notion": MCPHttpServerConfig(
            url="https://mcp.notion.com/mcp", auth=MCPAuthConfig(type="oauth")
        ),
    }
    manager = FakeMcpManager(["linear", "notion"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        editor = app.query_one(Editor)
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, {}),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logged_out_servers",
                return_value={"https://mcp.linear.app/mcp"},
            ),
        ):
            _set_editor_line(editor, "/mcp logout ")
            for _ in range(6):
                await pilot.pause()
            assert [name for name, _ in editor.picker.suggestions()] == ["logout linear"]


async def _type_into_editor(pilot, app, text: str) -> None:
    """Type ``text`` one REAL keystroke at a time, leaving it in the buffer.

    Deliberately NOT ``_set_editor_line``. That helper assigns ``editor.text``
    and calls ``_sync_picker()`` directly, which BYPASSES the
    ``RefreshArgumentChoices`` message the picker relies on to swap its rows
    when a two-level command crosses into its second slot. A whole class of
    wiring bug is therefore invisible to it: the `/mcp <verb> ` server rows
    were unreachable by typing while six tests using that helper passed, and
    the captured screenshots showed a state the UI never produced.

    Unlike ``_type_command`` this does not press Enter, because the state under
    test is the OPEN picker rather than the executed command.
    """
    editor = app.query_one(Editor)
    editor.text = ""
    editor.move_cursor(editor._end_of_buffer())
    for _ in range(4):
        await pilot.pause()
    editor.focus()
    for ch in text:
        await pilot.press("space" if ch == " " else ch)
    for _ in range(6):
        await pilot.pause()


@pytest.mark.asyncio
async def test_mcp_server_rows_are_reachable_by_typing_not_just_by_setting_text() -> None:
    """The rows must appear for a user at the KEYBOARD, not only when a test
    assigns the buffer.

    `/mcp` is two-level: the first argument slot holds a verb, and the space
    after it opens the SERVER slot. The picker only refills when its tracked
    sub-slot changes, so a tracker keyed on the verb TOKEN alone never fires on
    that space — `remove` and `remove ` are the same token — and the stale verb
    rows get filtered to nothing by the server query, closing the list.

    Every other server-row test here uses `_set_editor_line`, which calls
    `_sync_picker()` directly and cannot observe that message at all. This test
    types, so it fails when the wiring is broken even though those pass.
    """
    from local_operator.mcp.config import (
        MCPAuthConfig,
        MCPHttpServerConfig,
        MCPStdioServerConfig,
    )

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "filesystem": MCPStdioServerConfig(command="npx"),
    }
    sources = {"linear": "/tmp/x/.local-operator/mcp.json", "filesystem": "/tmp/x/.claude.json"}
    manager = FakeMcpManager(["linear", "filesystem"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, sources),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logged_out_servers",
                return_value={"https://mcp.linear.app/mcp"},
            ),
        ):
            # The verb slot still works when typed.
            await _type_into_editor(pilot, app, "/mcp ")
            assert [n for n, _ in editor.picker.suggestions()] == list(OperatorApp.MCP_SUBCOMMANDS)

            # Each verb -> server transition, reached by the space alone.
            for typed, expected in (
                ("/mcp remove ", ["remove linear", "remove filesystem"]),
                ("/mcp login ", ["login linear"]),
                ("/mcp logout ", ["logout linear"]),
                ("/mcp reauth ", ["reauth linear"]),
            ):
                await _type_into_editor(pilot, app, typed)
                rows = [n for n, _ in editor.picker.suggestions()]
                assert sorted(rows) == sorted(expected), f"typing {typed!r} gave {rows}"
                assert editor.picker.display, f"typing {typed!r} left the picker closed"

            # And narrowing still works from the typed state.
            await _type_into_editor(pilot, app, "/mcp remove fs")
            assert [n for n, _ in editor.picker.suggestions()] == ["remove filesystem"]


@pytest.mark.asyncio
async def test_home_then_end_keeps_an_open_argument_list() -> None:
    """#393: ``home`` then ``end`` closed an ARGUMENT list and never reopened it.

    The COMMAND list survives the same pair of keys, so the two lists disagreed
    about what a round trip to the start of the line and back should do. Driven
    through the real ``OperatorApp`` with real key presses — the same sequence
    the issue names — because the defect is a caret-move / queued-refresh race
    that assigning the buffer never reaches.

    ``home`` is an ordinary caret gesture, not a dismissal. Escape is the key
    that means "I do not want this list".
    """
    from unittest.mock import patch

    from local_operator.mcp.config import (
        MCPAuthConfig,
        MCPHttpServerConfig,
        MCPStdioServerConfig,
    )

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "filesystem": MCPStdioServerConfig(command="npx"),
    }
    sources = {"linear": "/tmp/x/.local-operator/mcp.json", "filesystem": "/tmp/x/.claude.json"}
    manager = FakeMcpManager(["linear", "filesystem"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)
        picker = editor.picker
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, sources),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logged_out_servers",
                return_value={"https://mcp.linear.app/mcp"},
            ),
        ):
            # COMMAND list: the control that already works.
            await _type_into_editor(pilot, app, "/mc")
            assert picker.is_open()
            command_rows = [n for n, _ in picker.suggestions()]
            await pilot.press("home")
            for _ in range(4):
                await pilot.pause()
            await pilot.press("end")
            for _ in range(8):
                await pilot.pause()
            assert picker.is_open(), "COMMAND list closed on home+end"
            assert [n for n, _ in picker.suggestions()] == command_rows

            # ARGUMENT list: the one that closed and never came back.
            await _type_into_editor(pilot, app, "/mcp ")
            assert picker.is_open()
            argument_rows = [n for n, _ in picker.suggestions()]
            assert argument_rows, "the verb list never opened"
            await pilot.press("home")
            for _ in range(4):
                await pilot.pause()
            await pilot.press("end")
            for _ in range(8):
                await pilot.pause()
            assert picker.is_open(), "ARGUMENT list closed on home+end and did not reopen"
            assert [n for n, _ in picker.suggestions()] == argument_rows
            assert editor.text == "/mcp "
            assert editor.selection.end == (0, 5)


@pytest.mark.asyncio
async def test_the_destructive_gate_is_armed_on_the_TYPED_path() -> None:
    """#378's fuzzy-Enter guard reads the highlighted row's ``alert`` flag, so
    it is only as good as the list being open.

    While the server rows were unreachable by typing there was no highlighted
    choice, and the gate returned ``False`` for every `/mcp` row on the real
    key path — the flag was set correctly and protected nothing. Asserted here
    on TYPED input so the safety property is proven where a user meets it:
    armed for the verbs that destroy, and off for `login`, which does not.
    """
    from local_operator.mcp.config import (
        MCPAuthConfig,
        MCPHttpServerConfig,
        MCPStdioServerConfig,
    )

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "filesystem": MCPStdioServerConfig(command="npx"),
    }
    sources = {"linear": "/tmp/x/.local-operator/mcp.json", "filesystem": "/tmp/x/.claude.json"}
    manager = FakeMcpManager(["linear", "filesystem"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, sources),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logged_out_servers",
                return_value={"https://mcp.linear.app/mcp"},
            ),
        ):
            for typed, destroys in (
                ("/mcp remove fs", True),
                ("/mcp logout lin", True),
                # The fuzzy shape that motivated the gate: `lnr` spells nothing
                # and still narrows to one row.
                ("/mcp reauth lnr", True),
                ("/mcp login lin", False),
            ):
                await _type_into_editor(pilot, app, typed)
                assert editor.picker.suggestions(), f"typing {typed!r} opened no list"
                assert editor._argument_is_destructive() is destroys, (
                    f"typing {typed!r}: gate={editor._argument_is_destructive()}, "
                    f"expected {destroys}"
                )


@pytest.mark.asyncio
async def test_every_mcp_verb_that_destroys_something_flags_its_rows() -> None:
    """`alert` is the SAFETY bit, so the set it covers must match what the
    verbs actually destroy — not what they are named.

    `reauth` reads like a login and IS a deletion: `_cmd_mcp` runs
    `_mcp_logout` first (the docstring calls it "the two composed — forget
    first"), so the stored grant leaves the shared auth.db the moment the row
    runs, and an abandoned browser round trip leaves the server with no
    credential at all. It was flagged `False` while the PR that made this flag
    load-bearing claimed every destructive row carried it.

    Asserted as a whole table rather than one row, because the claim the
    editor's destructive gate relies on is about the SET: `login` is the only
    verb here that destroys nothing, and it is the only one left unflagged.
    """
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    configs = {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
    }
    manager = FakeMcpManager(["linear"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        editor = app.query_one(Editor)
        with (
            patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(configs, {"linear": "/tmp/x/.local-operator/mcp.json"}),
            ),
            patch(
                "local_operator.mcp.auth.mcp_logged_out_servers",
                return_value={"https://mcp.linear.app/mcp"},
            ),
        ):
            # verb -> does choosing a row DESTROY persistent state?
            # `logout` last: its list is filtered by the credential store, so
            # ordering it after the others keeps a failure here pointing at the
            # alert flag rather than at an empty list.
            for verb, destroys in (
                ("login", False),
                ("reauth", True),
                ("remove", True),
                ("logout", True),
            ):
                # Clear between verbs: the picker tracks the sub-slot and only
                # refills when it CHANGES, so hopping verb-to-verb without
                # passing through the bare form leaves the previous list up.
                _set_editor_line(editor, "/mcp ")
                for _ in range(4):
                    await pilot.pause()
                _set_editor_line(editor, f"/mcp {verb} ")
                for _ in range(8):
                    await pilot.pause()
                rows = editor.picker.suggestions()
                assert rows, f"/mcp {verb} offered no rows"
                for name, choice in rows:
                    assert isinstance(choice, ArgumentChoice)
                    assert choice.alert is destroys, (
                        f"/mcp {verb} row {name!r} has alert={choice.alert}, "
                        f"but the verb destroys={destroys}"
                    )


def _mcp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A sandbox HOME holding one owned config and one Claude-imported server.

    The `/mcp add|remove` paths WRITE, so every test touching them needs its
    own home — a test that mutated the developer's real
    ``~/.local-operator/mcp.json`` would delete servers off their machine.
    """
    home = tmp_path / "home"
    (home / ".local-operator").mkdir(parents=True)
    (home / ".local-operator" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "filesystem": {"type": "stdio", "command": "npx"},
                    "grafana": {"type": "http", "url": "https://grafana.example/mcp"},
                }
            }
        )
    )
    # Visible to `/mcp`, and none of local-operator's business to delete.
    (home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"notion": {"type": "http", "url": "https://n.example/mcp"}}})
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.asyncio
async def test_mcp_remove_list_offers_every_server_not_just_the_oauth_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this list had: it was built from `oauth_server_names`, so a
    stdio server and a non-OAuth http server were invisible to the picker even
    though `remove` is exactly the verb that can act on them. `remove` reads
    the CONFIG, which is the thing it edits."""
    home = _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager = FakeMcpManager(["filesystem", "grafana", "notion"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        editor = app.query_one(Editor)
        _set_editor_line(editor, "/mcp remove ")
        for _ in range(6):
            await pilot.pause()
        rows = {}
        for name, choice in editor.picker.suggestions():
            # An ARGUMENT list only ever carries ArgumentChoice rows;
            # suggestions() is typed as the union it shares with the
            # command-word list, so assert the mode rather than casting.
            assert isinstance(choice, ArgumentChoice), "the picker is not in argument mode"
            rows[name] = choice
        assert sorted(rows) == ["remove filesystem", "remove grafana", "remove notion"]
        # The detail column carries the SOURCE FILE, abbreviated: that is what
        # turns the refusal into something the user saw coming.
        assert rows["remove filesystem"].detail == "~/.local-operator/mcp.json"
        assert rows["remove notion"].detail == "~/.claude.json"
        assert str(home) not in rows["remove notion"].detail
        # `alert` is LOAD-BEARING SAFETY here, not a tint (see
        # _mcp_remove_choices): the editor's destructive gate reads it to make
        # Enter fill rather than fire.
        assert all(choice.alert for choice in rows.values())


@pytest.mark.asyncio
async def test_mcp_remove_refuses_a_server_defined_by_a_foreign_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing a Claude-imported server would either fail or write a
    local-operator file that silently shadows a config the user still
    maintains in Claude Code. The refusal names the FILE and the TOOL and says
    where to go instead, and leaves both configs untouched."""
    home = _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    claude_before = (home / ".claude.json").read_text()
    manager = FakeMcpManager(["filesystem", "grafana", "notion"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        await _type_command(pilot, app, "mcp remove notion")
        for _ in range(6):
            await pilot.pause()
        text = _transcript_text(app)
        assert "~/.claude.json" in text
        assert "imported from Claude Code" in text
        assert "Remove it there" in text
    # Neither file was touched.
    assert (home / ".claude.json").read_text() == claude_before
    owned = json.loads((home / ".local-operator" / "mcp.json").read_text())
    assert sorted(owned["mcpServers"]) == ["filesystem", "grafana"]


@pytest.mark.asyncio
async def test_mcp_remove_refuses_a_codex_imported_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Codex refusal is PERMANENT, not a policy choice: `tomllib` parses
    TOML and cannot emit it (`tomli_w` is not a dependency), so a server
    defined by `~/.codex/config.toml` can be listed but never removed in
    place. The refusal has to name that file and Codex CLI, because "remove it
    there" is the only action left to the user (issues #367/#368)."""
    home = _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    (home / ".codex").mkdir(parents=True)
    codex_config = home / ".codex" / "config.toml"
    codex_config.write_text(
        '[mcp_servers.codexy]\nurl = "https://codexy.example/mcp"\n', encoding="utf-8"
    )
    codex_before = codex_config.read_text()
    manager = FakeMcpManager(["filesystem", "grafana", "notion", "codexy"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        await _type_command(pilot, app, "mcp remove codexy")
        for _ in range(6):
            await pilot.pause()
        text = _transcript_text(app)
        assert "~/.codex/config.toml" in text
        # The origin phrase soft-wraps at the transcript width, so collapse
        # whitespace before matching rather than asserting on a line break.
        assert "imported from Codex CLI" in " ".join(text.split())
        assert "Remove it there" in text
        # A `not found` here would be the confusing failure #368 replaced.
        assert "is not configured" not in text
    assert codex_config.read_text() == codex_before


@pytest.mark.asyncio
async def test_mcp_add_writes_both_transports_and_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grammar discriminates on whether the third token is an http(s) URL.
    The receipt NAMES the file written, because the global scope is otherwise
    an invisible default the user has no way to confirm.

    OAuth is deliberately NOT inferred from the URL: real configs carry
    non-OAuth http servers, and inferring auth would silently change how a
    server authenticates."""
    home = _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager = FakeMcpManager(["filesystem", "grafana"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        await _type_command(pilot, app, "mcp add demo-stdio npx -y demo-mcp")
        for _ in range(6):
            await pilot.pause()
        await _type_command(pilot, app, "mcp add demo-http https://demo.example/mcp")
        for _ in range(6):
            await pilot.pause()
        assert "~/.local-operator/mcp.json" in _transcript_text(app)
    servers = json.loads((home / ".local-operator" / "mcp.json").read_text())["mcpServers"]
    assert servers["demo-stdio"] == {"type": "stdio", "command": "npx", "args": ["-y", "demo-mcp"]}
    assert servers["demo-http"] == {"type": "http", "url": "https://demo.example/mcp"}
    assert "auth" not in servers["demo-http"]


@pytest.mark.asyncio
async def test_mcp_list_is_an_alias_for_the_bare_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lop mcp list` has always existed, so a user who guesses `/mcp list`
    used to get `unknown mcp subcommand: list` for a correctly-formed command.

    Reserving `list` is safe HERE because `/mcp`'s first token is a CLOSED verb
    set — nothing can be named `list` in that slot. `/team` and `/agent` take
    an OPEN namespace and must NOT copy this fix."""
    _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager = FakeMcpManager(["filesystem", "grafana"], ["grafana"])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        await _type_command(pilot, app, "mcp list")
        for _ in range(6):
            await pilot.pause()
        text = _transcript_text(app)
        assert "MCP servers" in text
        assert "unknown mcp subcommand" not in text
        assert "filesystem" in text and "grafana" in text


@pytest.mark.asyncio
async def test_a_fuzzy_mcp_remove_row_fills_rather_than_deleting_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `/logout` hazard, applied to config deletion. The matcher is a
    SUBSEQUENCE matcher, so a query that spells nothing can still leave one
    survivor: `fsy` reaches `filesystem`. Enter on that row must COMPLETE it —
    leaving the user looking at the full name they never typed — rather than
    deleting a server off disk on a single keystroke.

    Verified as a real data-loss defect before this test existed: `/mcp remove
    fsy` + Enter removed `filesystem` from mcp.json outright."""
    home = _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager = FakeMcpManager(["filesystem", "grafana"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)
        _set_editor_line(editor, "/mcp remove fsy")
        for _ in range(6):
            await pilot.pause()
        assert [name for name, _ in editor.picker.suggestions()] == ["remove filesystem"]
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()
        # Completed into the buffer, not run.
        assert editor.text.strip() == "/mcp remove filesystem"
    servers = json.loads((home / ".local-operator" / "mcp.json").read_text())["mcpServers"]
    assert "filesystem" in servers, "a fuzzy match must never delete a server on one keystroke"


@pytest.mark.asyncio
async def test_mcp_add_refuses_to_shadow_or_no_op_over_an_existing_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`add` honours the SAME ownership rule `remove` enforces, and reports the
    two cases differently because their consequences differ.

    A foreign, lower-priority definition (`~/.claude.json`): our write would
    WIN and silently repoint a server the user still maintains in Claude Code —
    the outcome `_mcp_remove_result` refuses to cause from the other side.

    A higher-priority definition (`<cwd>/.mcp.json`): our write lands and
    changes nothing observable, so an unqualified "added" would be a receipt
    that LIES about an effect the user will never see. That is the worse half.
    """
    home = _mcp_home(tmp_path, monkeypatch)
    project = home / "proj"
    project.mkdir()
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"proj": {"type": "http", "url": "https://proj.example/mcp"}}})
    )
    monkeypatch.chdir(project)
    manager = FakeMcpManager(["filesystem", "grafana", "notion", "proj"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    claude_before = (home / ".claude.json").read_text()
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()

        await _type_command(pilot, app, "mcp add notion https://evil.example/mcp")
        for _ in range(8):
            await pilot.pause()
        text = " ".join(_transcript_text(app).split())
        assert "~/.claude.json" in text and "shadow" in text

        marker = len(_transcript_text(app))
        await _type_command(pilot, app, "mcp add proj https://mine.example/mcp")
        for _ in range(8):
            await pilot.pause()
        # The transcript hard-wraps, so compare on collapsed whitespace.
        second = " ".join(_transcript_text(app)[marker:].split())
        assert "takes priority" in second and "no effect" in second
        # The no-op case must NOT claim success.
        assert "added MCP server" not in second

    # Neither foreign file was touched, and no shadowing entry was written.
    assert (home / ".claude.json").read_text() == claude_before
    owned = json.loads((home / ".local-operator" / "mcp.json").read_text())["mcpServers"]
    assert "notion" not in owned and "proj" not in owned


@pytest.mark.asyncio
async def test_mcp_add_receipt_suggests_a_command_that_can_actually_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt used to end with `/mcp login <name>`, which was guaranteed
    to fail for every server this command creates: `add` deliberately writes no
    `auth` block and `_resolve_mcp_server` refuses any server whose `auth.type`
    is not `oauth`. The hint now names the CLI flag that writes the block."""
    _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager = FakeMcpManager(["filesystem", "grafana"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        await _type_command(pilot, app, "mcp add gw https://gw.example/mcp")
        for _ in range(8):
            await pilot.pause()
        text = _transcript_text(app)
        assert "--oauth" in text
        # The suggestion that cannot work must be gone.
        assert "/mcp login gw" not in text


@pytest.mark.asyncio
async def test_mcp_refuses_trailing_tokens_on_fixed_arity_verbs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently dropping the tail is the same class of mistake this command
    exists to avoid — acting on something other than what the user described.
    It matters most on `remove`, where the ignored token could be the name they
    actually meant to delete."""
    home = _mcp_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager = FakeMcpManager(["filesystem", "grafana"], [])
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()

        marker = len(_transcript_text(app))
        await _type_command(pilot, app, "mcp list junk")
        for _ in range(6):
            await pilot.pause()
        listing = " ".join(_transcript_text(app)[marker:].split())
        assert "takes no arguments" in listing and "MCP servers" not in listing

        marker = len(_transcript_text(app))
        await _type_command(pilot, app, "mcp remove filesystem extra")
        for _ in range(6):
            await pilot.pause()
        assert "takes one server name" in " ".join(_transcript_text(app)[marker:].split())

    # The destructive verb refused, so the server is still configured.
    servers = json.loads((home / ".local-operator" / "mcp.json").read_text())["mcpServers"]
    assert "filesystem" in servers


@pytest.mark.asyncio
async def test_a_discovery_failure_keeps_an_alarm_in_the_band() -> None:
    """Discovery raising leaves no manager and no server list, so the band used
    to render exactly like a machine that never configured MCP — while the toast
    saying otherwise dismissed itself ten seconds later."""
    startup = McpStartupOutcome(failures={"discovery": "config unreadable"})
    session = McpSession(manager=None, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        band = _band(app)
        assert "⊙ MCP" in band, band
        # No count: the config layer never produced one, so any number is a lie.
        assert "0 MCP" not in band
        app.query_one(Toast).dismiss_toast()
        await pilot.pause()
        assert "⊙ MCP" in _band(app)


@pytest.mark.asyncio
async def test_the_band_refreshes_even_when_the_incumbent_callback_raises() -> None:
    """The chained wrapper calls the composition root's subscriber first, and
    ``McpManager._fire_tools_changed`` swallows and logs whatever comes out of
    it — so a raising incumbent used to leave the band asserting a count that was
    no longer true, which is the exact staleness the live segment exists to
    remove. The repaint is scheduled in a ``finally``."""
    manager = FakeMcpManager(["github", "linear"], ["github", "linear"])

    def exploding_incumbent(tools: list[Any]) -> None:
        raise RuntimeError("refresh_tools blew up")

    manager.set_on_tools_changed(exploding_incumbent)
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "⊙ 2 MCP" in _band(app)
        manager._connected.remove("linear")
        # The manager swallows the incumbent's exception; the app must not lose
        # the repaint with it.
        with pytest.raises(RuntimeError):
            manager.fire()
        for _ in range(4):
            await pilot.pause()
        assert "⊙ 1 MCP" in _band(app)


@pytest.mark.asyncio
async def test_a_toast_erases_no_transcript_row_outside_its_own_columns() -> None:
    """A/B on the COMPOSITED frame, which is the only place this was visible.

    The toast host was ``width: 1fr``: a widget owns its whole region and Textual
    blanks all of it, so a 35-cell card on a 96-cell screen erased the other 59
    cells of every row it covered — the transcript row read
    `· line 0 ABCDEFGHIJ…` out to column 90 before the toast and nothing at all
    after it. The layer keeps the toast out of the LAYOUT; it does not keep it
    off the screen.
    """
    session = McpSession(manager=None, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(96, 28)) as pilot:
        for _ in range(4):
            await pilot.pause()
        transcript = app.query_one(TranscriptView)
        # ONE tall block rather than many: scrolled to the bottom, a block taller
        # than the viewport fills every row of it with text, including the row the
        # card lands on. Separate blocks put their adaptive gap row there instead,
        # which would make this A/B pass by having nothing to lose.
        transcript.append_block(
            NoticeBlock(
                "\n".join(f"line {index} " + "ABCDEFGHIJ" * 8 for index in range(40)), "info"
            )
        )
        app._set_welcome_visible(False)
        for _ in range(4):
            await pilot.pause()

        def rows() -> list[str]:
            return [
                "".join(segment.text for segment in strip)
                for strip in app.screen._compositor.render_strips()
            ]

        before = rows()
        toast = app.query_one(Toast)
        toast.show(
            "⊙ MCP: 1 of 2 servers up, 9 tools\nfailed: b — spawn ENOENT", duration_ms=60_000
        )
        for _ in range(4):
            await pilot.pause()
        after = rows()

        card = toast.region
        assert card.height == 2, card
        # NOT vacuous: at least one row the card covers carries transcript text to
        # the left of it, which is exactly what the full-width host used to wipe.
        # Only SOME of them — the transcript's own top padding row is blank by
        # design and the card starts on it.
        assert any(
            before[y][: card.x].strip() for y in range(card.y, card.bottom)
        ), f"nothing to lose on rows {card.y}..{card.bottom - 1}"
        # The DAMAGE: every column outside the card is byte-identical.
        for y in range(card.y, card.bottom):
            for column in range(len(before[y])):
                if card.x <= column < card.right:
                    continue
                assert before[y][column] == after[y][column], (
                    f"row {y} column {column} was erased outside the card "
                    f"{card}: {before[y]!r} -> {after[y]!r}"
                )
        # …then the mechanism that guarantees it, and proof the card really is
        # showing there, so this is not a vacuous pass.
        assert app.query_one("#toast-host").region == card
        assert "MCP: 1 of 2 servers up" in after[card.y]


# --- model access: what the picker offers, what a switch confirms ------------


class _AccessController(FakeProviderController):
    """A catalogue spanning the three credential situations the picker must tell
    apart: a stored credential, none at all, and a local server that needs none.

    ``store_error`` is the store failing to answer — SQLite locked, file gone —
    which the real controller reports as ``None`` from ``usable_providers`` rather
    than as an empty set, because "I cannot tell" and "you have nothing" are
    different answers.
    """

    def __init__(
        self, stored: tuple[str, ...] = ("openrouter",), store_error: bool = False
    ) -> None:
        super().__init__()
        self.stored = set(stored)
        self.store_error = store_error

    def login_providers(self):
        return [
            _FakeDef("openrouter", "OpenRouter", None, ("router",)),
            _FakeDef("anthropic", "Anthropic", None, ("claude",)),
            _FakeDef("ollama", "Ollama", None, ()),
        ]

    def has_any_credential(self, provider):
        if self.store_error:
            raise RuntimeError("credential store locked")
        return provider in self.stored

    def is_usable(self, provider):
        # Ollama stands in for `allows_missing_api_key`: a local server runs with
        # no credential at all, so a filter keyed on credentials alone would hide
        # the one provider that always works.
        return provider == "ollama" or self.has_any_credential(provider)

    def usable_providers(self):
        if self.store_error:
            return None
        return {d.id for d in self.login_providers() if self.is_usable(d.id)}

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        usable = self.usable_providers()
        return [
            CatalogueEntry(
                provider=provider,
                model_id=model_id,
                label=model_id,
                context_window=200_000,
                input_price=3.0,
                output_price=15.0,
                connected=usable is None or provider in usable,
                aggregated=provider == "openrouter",
            )
            for provider, model_id in (
                ("openrouter", "deepseek/deepseek-chat"),
                ("anthropic", "claude-opus-5"),
                ("ollama", "qwen3:8b"),
            )
        ]

    async def live_catalogue(self, *, ttl_s=None):
        return self.static_catalogue(), {}


async def _open_model_picker(app, pilot):
    """Type ``/model `` the way a user does and settle the live refresh."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    await pilot.press("slash", "m", "o", "d", "e", "l", "space")
    # Twice: the first frame is the static catalogue, the second is the worker's
    # live one. Asserting on the first would test the wrong list.
    await pilot.pause()
    await pilot.pause()
    return editor.model_picker


class _TtlRecordingController(_AccessController):
    """Records what listing TTL the picker asks for."""

    def __init__(self) -> None:
        super().__init__(stored=("openrouter",))
        self.asked: list[float | None] = []

    async def live_catalogue(self, *, ttl_s=None):
        self.asked.append(ttl_s)
        return self.static_catalogue(), {}


@pytest.mark.asyncio
async def test_the_picker_asks_for_a_fifteen_minute_listing() -> None:
    """The user opening `/model` is the one moment a fresh list is worth a request.

    Discovery's 24h default is what hid a model published that morning from the
    picker all day; the fetch is off-loop behind painted rows, so a short TTL
    costs nothing visible.
    """
    from local_operator.providers.controller import PICKER_TTL_S

    ctrl = _TtlRecordingController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await _open_model_picker(app, pilot)
    assert ctrl.asked == [PICKER_TTL_S], ctrl.asked


@pytest.mark.asyncio
async def test_the_model_list_offers_what_the_user_can_actually_run() -> None:
    """The list is a set of choices. A row whose only outcome is a login prompt is
    not one, and it costs a line of a fourteen-row window."""
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
    # openrouter has a credential; ollama needs none by definition; anthropic has
    # neither and is the one the user cannot act on.
    assert offered == {"openrouter/deepseek/deepseek-chat", "ollama/qwen3:8b"}, offered


class _PruningController(_AccessController):
    """A catalogue that withdrew the model the session is running.

    This is what an authoritative account-scoped listing does: the account's own
    catalogue replaces the bundled ids, so a session started on a bundled id has
    no entry in the list at all. The registry still describes it and ``/model``
    still accepts it, which is exactly the disagreement the rescue exists to
    close.
    """

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="openrouter",
                model_id="deepseek/deepseek-chat",
                label="DeepSeek Chat",
                context_window=64_000,
                input_price=0.14,
                output_price=0.28,
                connected=True,
                aggregated=True,
            )
        ]

    def entry_for(self, provider, model_id):
        from local_operator.providers.controller import CatalogueEntry

        if (provider, model_id) != ("test", "model"):
            return None
        return CatalogueEntry(
            provider=provider,
            model_id=model_id,
            label="Test Model",
            context_window=128_000,
            input_price=1.0,
            output_price=2.0,
            connected=True,
        )


class _PruningWithHiddenController(_PruningController):
    """One usable row, one filtered row, and one rescued row not in either."""

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="openrouter",
                model_id="deepseek/deepseek-chat",
                label="DeepSeek Chat",
                context_window=64_000,
                input_price=0.14,
                output_price=0.28,
                connected=True,
                aggregated=True,
            ),
            CatalogueEntry(
                provider="anthropic",
                model_id="claude-opus-5",
                label="Claude Opus 5",
                context_window=1_000_000,
                input_price=15.0,
                output_price=75.0,
                connected=False,
            ),
        ]


@pytest.mark.asyncio
async def test_a_rescued_row_does_not_hide_a_real_hidden_catalogue_entry() -> None:
    """``hidden`` counts catalogue rows filtered out, not final rows painted.

    A rescued current row was never in ``entries``. Subtracting it anyway made
    one real filtered catalogue entry disappear from the footer; flooring at
    zero prevented ``-1`` but did not preserve what the count means.
    """
    ctrl = _PruningWithHiddenController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
        chrome = picker.render_text(90).plain

    assert offered == {"openrouter/deepseek/deepseek-chat", "test/model"}, offered
    assert "1 hidden — /login <provider>" in chrome, chrome


@pytest.mark.asyncio
async def test_the_running_model_is_offered_even_when_the_listing_withdrew_it() -> None:
    """The band names the model; the list must not deny it.

    An authoritative listing may prune bundled ids, and exempting the current row
    from the credential filter does not help when no entry arrives to exempt. The
    session read `test/model` on the band while the picker showed no current row
    and typing the id answered "no matching models" — the list disagreeing with
    the set `/model` accepts, with the list being the discovery surface.
    """
    ctrl = _PruningController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
        chrome = picker.render_text(90).plain

    assert "test/model" in offered, offered
    # The `●` is the whole point: it is what answers "what am I on", and a row
    # without it would leave the band and the list contradicting each other.
    assert "test/model" in chrome and "●" in chrome, chrome
    # And the count stays honest: the rescued row is not one of the catalogue's
    # entries, so a naive subtraction reports a NEGATIVE number of hidden rows.
    assert not re.search(r"-\d+\s+hidden", chrome), chrome


@pytest.mark.asyncio
async def test_a_host_facade_without_the_rescue_still_paints_the_list() -> None:
    """The provider facade is duck-typed, so an embedding host need not implement
    ``entry_for``. A missing courtesy must never cost the picker."""
    ctrl = _AccessController()  # no entry_for at all
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}

    assert offered == {"openrouter/deepseek/deepseek-chat", "ollama/qwen3:8b"}, offered


@pytest.mark.asyncio
async def test_the_hidden_models_are_counted_with_the_command_that_reveals_them() -> None:
    """Discoverability was the whole argument for the old show-everything list.
    The footer chrome keeps it without crowding the persistence instruction."""
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        chrome = picker.render_text(90).plain
    assert "1 hidden" in chrome, chrome
    assert "/login <provider>" in chrome, chrome


@pytest.mark.asyncio
async def test_an_unreadable_credential_store_shows_every_model_not_none() -> None:
    """An empty picker claims the user owns no models, which is exactly what the
    app failed to find out. Showing an unfiltered list is the recoverable error."""
    ctrl = _AccessController(store_error=True)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
        chrome = picker.render_text(90).plain
    assert "anthropic/claude-opus-5" in offered, offered
    assert len(offered) == 3, offered
    # …and it says so, rather than quietly presenting a list it could not filter.
    assert "credential check unavailable" in chrome, chrome


@pytest.mark.asyncio
async def test_a_new_credential_reaches_the_list_without_a_restart() -> None:
    """`/login anthropic` then `/model` is one continuous action. Rows built once
    at boot would make the user restart the app to see what they just unlocked."""
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        assert not any(row.provider == "anthropic" for row in picker.rows())

        ctrl.stored.add("anthropic")  # what a completed /login stores
        editor = app.query_one(Editor)
        editor.text = ""  # close the list; the next `/model` rebuilds it
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
    assert "anthropic/claude-opus-5" in offered, offered


class _SwitchableSession(FakeSession):
    """A session whose label follows ``set_model``, as the real one's does — the
    confirmation names the model, so a frozen label would not test it."""

    def __init__(self) -> None:
        super().__init__()
        self._label = "openrouter/deepseek/deepseek-chat"

    @property
    def model_label(self) -> str:
        return self._label

    def set_model(self, model, *, explicit: bool = False) -> None:
        self._label = f"{model.provider}/{model.model_id}"


@pytest.mark.asyncio
async def test_switching_confirms_access_instead_of_warning_about_it() -> None:
    """The old line told the user to go and check something the app knew, on every
    provider change including the ones that were fine."""
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert "anthropic/claude-opus-5" in text, text
    assert "anthropic logged in" in text, text
    assert "make sure you are logged in" not in text, text


@pytest.mark.asyncio
async def test_switching_without_a_credential_names_the_one_fix() -> None:
    """ "needs login" is the same word `/provider` and the `/login` picker use, and
    the command is the entire remedy — no second surface to go and consult."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert "anthropic needs login — /login anthropic" in text, text
    assert "make sure you are logged in" not in text, text


@pytest.mark.asyncio
async def test_a_hidden_model_is_still_reachable_by_typing_its_selector() -> None:
    """Filtering the LIST is not a lock on the command. A user who knows the id —
    from `/provider`, from docs, from the model they used yesterday — types it and
    gets the switch plus the one thing they are missing, not a refusal."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        _set_editor_line(editor, "/model anthropic/claude-opus-5")
        await pilot.pause()
        assert editor.model_picker.suggestions() == [], "premise: the row is hidden"
        await pilot.press("enter")
        await pilot.pause()
        text = _transcript_text(app)
    assert session.model_label == "anthropic/claude-opus-5"
    assert "anthropic needs login — /login anthropic" in text, text


class _OpenAILiveController(_AccessController):
    """Logged into OpenAI with the account-scoped Codex catalogue, not the
    shipped gpt-4o/o3 registry. That is the ChatGPT OAuth path: the public
    ``/v1/models`` 403s a subscription token, so gpt-5.6 only exists here."""

    def __init__(self) -> None:
        super().__init__(stored=("openai",))

    def login_providers(self):
        return [
            _FakeDef("openai", "OpenAI", None, ("gpt",)),
            _FakeDef("anthropic", "Anthropic", None, ("claude",)),
        ]

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="openai",
                model_id="gpt-4o",
                label="GPT-4o",
                context_window=128_000,
                input_price=2.5,
                output_price=10.0,
                connected=True,
            )
        ]

    async def live_catalogue(self, *, ttl_s=None):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="openai",
                model_id=model_id,
                label=label,
                context_window=272_000,
                input_price=0.0,
                output_price=0.0,
                connected=True,
            )
            for model_id, label in (
                ("gpt-5.6-sol", "GPT-5.6-Sol"),
                ("gpt-5.6-terra", "GPT-5.6-Terra"),
                ("gpt-5.6-luna", "GPT-5.6-Luna"),
                ("gpt-5.5", "GPT-5.5"),
            )
        ], {"openai": "ok"}


@pytest.mark.asyncio
async def test_chatgpt_oauth_offers_the_live_gpt_5_family() -> None:
    """The reported `/model gpt-5.6` miss. A ChatGPT login whose live
    catalogue is the Codex list must surface those slugs, not the shipped
    gpt-4o/o3 registry that the public API would have left behind."""
    ctrl = _OpenAILiveController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        editor = app.query_one(Editor)
        _set_editor_line(editor, "/model gpt-5.6")
        await pilot.pause()
        offered = {row.model_id for row in picker.suggestions()}
        chrome = picker.render_text(90).plain
    assert {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} <= offered, offered
    assert "gpt-4o" not in offered
    assert "no matching models" not in chrome


@pytest.mark.asyncio
async def test_a_failed_credential_check_is_reported_as_itself() -> None:
    """Neither a confirmation the app cannot make nor the old blanket warning: the
    store is what broke, and naming it is what makes it fixable."""
    session = _SwitchableSession()
    ctrl = _AccessController(store_error=True)
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert "cannot check anthropic credentials: credential store locked" in text, text
    assert "logged in" not in text, text


@pytest.mark.asyncio
async def test_a_failing_turn_shows_the_providers_own_error() -> None:
    """The 400 that started this: a switch-time guess about logins was on screen
    while the real reason — a rejected parameter — came from the provider. Only
    the provider's own words tell the user what to change."""
    session = FakeSession()

    async def prompt(text, images=None):
        raise RuntimeError("HTTP 400: `temperature` is deprecated for this model.")

    session.prompt = prompt  # type: ignore[assignment]
    app = OperatorApp(lambda: _factory(session), provider_controller=_AccessController())
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        text = _transcript_text(app)
    assert "HTTP 400: `temperature` is deprecated for this model." in text, text


# --- /resume ---------------------------------------------------------------


def _resume_factory(
    boots: list[str | None],
    history_text: str = "resumed history",
    assistant_text: str = "resumed answer",
):
    """A resume factory that records the id it was asked to boot.

    ``None`` is a real value here, not a missing one: it is what ``/new`` asks
    for, and ``create_session`` reads it as "start a fresh conversation".
    """

    async def resume_factory(resume_id: str | None):
        boots.append(resume_id)
        session = FakeSession()
        session._history = [
            SimpleNamespace(role="user", text=history_text, tool_calls=None),
            # Production Message objects use an empty list, not None, for a
            # prose-only assistant reply.
            SimpleNamespace(role="assistant", text=assistant_text, tool_calls=[]),
        ]
        return session

    return resume_factory


def _seed_session(tmp_path: Path, session_id: str, prompt: str = "") -> None:
    """Lay down one resumable session transcript under a temp config dir.

    ``recent_sessions`` globs ``<config_dir>/sessions/*`` for transcripts, so
    a real file is the only honest way to make the listing and the resume
    both resolve — same convention the ``--resume`` CLI path trusts.

    ``prompt`` writes a real opening user message, which is what the picker
    names the row by; without one the session is legitimately nameless.
    """
    sess_dir = tmp_path / "sessions" / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    body = ""
    if prompt:
        body = (
            json.dumps(
                {
                    "id": "e1",
                    "ts": 0,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"text": prompt}],
                    },
                }
            )
            + "\n"
        )
    (sess_dir / "transcript.jsonl").write_text(body)


def _append_turn(tmp_path: Path, session_id: str, text: str) -> None:
    """Add a later message to an existing seeded session.

    The picker names a row by its FIRST user message, so anything appended here
    lands in the searchable body without changing the visible name. That is the
    distinction some fixtures need: on a real store an incidental keyword hit is
    a word buried in a conversation, not the title of one.
    """
    line = (
        json.dumps(
            {
                "id": "e2",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": text}],
                },
            }
        )
        + "\n"
    )
    with (tmp_path / "sessions" / session_id / "transcript.jsonl").open("a") as handle:
        handle.write(line)


@pytest.mark.asyncio
async def test_a_bare_resume_opens_the_picker_naming_each_session(tmp_path, monkeypatch) -> None:
    """A bare ``/resume`` opens the picker instead of printing ids.

    The old behaviour dumped ``<hex id>  3h ago`` rows into the transcript,
    which pushed the conversation up, could not be navigated, and left the
    user to retype an id read off the scrollback. The picker is a two-way
    surface and names each row by its opening message.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc", prompt="Make an asteroids game")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
        await pilot.pause()
        await pilot.pause()
        assert boots == [], "a bare /resume must not boot a session"
        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)
        card = "\n".join(picker.render_lines_for_test())
        # Named by the opening message, not just listed by id.
        assert "Make an asteroids game" in card, card
        assert "aabbcc" in card, card


@pytest.mark.asyncio
async def test_the_picker_lists_a_store_far_larger_than_any_default_limit(
    tmp_path, monkeypatch
) -> None:
    """The picker shows the WHOLE store, exercised through the real command.

    This is the regression that two review rounds and a full CI run missed,
    because every other fixture here has fewer rows than the smallest cap: the
    reach fix removed ``RESUME_PICKER_LIMIT`` but ``recent_session_rows``
    defaulted to ``limit=10`` and ``_cmd_resume`` passed no argument, so the
    "uncapped" picker showed ten rows on a 236-session store — strictly worse
    than the cap of 200 it replaced.

    250 sessions, chosen to exceed every limit in the codebase (10 recovery,
    100 daemon summaries, 200 daemon search) so no default can satisfy it
    accidentally. Driven by typing ``/resume`` rather than by calling
    ``recent_session_rows`` directly: a parameterised stand-in for the product
    path is exactly what let this through the first time.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    total = 250
    for index in range(total):
        _seed_session(tmp_path, f"s{index:04d}", prompt=f"session number {index}")

    session = FakeSession()
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory([]),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        for key in "/resume":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)

        # Every session is HELD, which is what makes it filterable: the filter
        # only ever scans the rows the screen was handed.
        assert (
            len(picker.visible_rows) == total
        ), f"picker holds {len(picker.visible_rows)} of {total} sessions"
        # And the header says so, rather than reporting a truncated total.
        assert f"{total:,} sessions" in "\n".join(picker.render_lines_for_test())

        # The oldest session — the one furthest past every cap — is reachable by
        # filtering, which is the user-visible failure being fixed ("a session I
        # know exists cannot be found").
        oldest = f"s{total - 1:04d}"
        picker.set_query(oldest)
        await pilot.pause()
        assert [row.id for row in picker.visible_rows] == [oldest]


@pytest.mark.asyncio
async def test_typing_a_typo_reaches_the_soft_tier(tmp_path, monkeypatch) -> None:
    """The soft tier must be REACHABLE from a keyboard, not merely harmless.

    Two prior regression tests pinned the absence of harm (no row withdrawn, no
    cursor swap) and passed with the soft tier replaced by ``return False`` —
    deleting the feature satisfies them. Nothing asserted the tier ever runs.

    It did not. A gate that decided once at the START of a typing run froze the
    tier off for every word a user can type: a run begins at one character, and
    every single character has exact hits in a real store, so the tier was
    unreachable. Zero ``SoftSearchIndex.search`` calls across 25 typed runs.

    Drives real ``pilot.press`` keystrokes, because ``set_query("classifer")``
    jumps straight to the final query and cannot observe a per-run latch at
    all — which is exactly how that bug was validated as working.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # One session whose body carries the correctly-spelled word, plus filler so
    # the early keystrokes have exact hits and the run latches off under the
    # old gate.
    _seed_session(tmp_path, "aaaa0001", prompt="improve the adm classifier throughput")
    for index in range(5):
        _seed_session(tmp_path, f"bbbb{index:04d}", prompt="classroom scheduling notes")

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), resume_factory=_resume_factory([]))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        for key in "/resume":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)

        calls: list[str] = []
        real_search = picker._soft_index.search

        def counting(digests, query):
            calls.append(query)
            return real_search(digests, query)

        picker._soft_index.search = counting  # type: ignore[method-assign]

        # `classifer` is a typo: no session contains it as a substring, so only
        # the soft tier can find the `classifier` session.
        for key in "classifer":
            await pilot.press(key)
            await pilot.pause()

        assert calls, "the soft tier never ran while the user typed a typo"
        assert "aaaa0001" in {
            row.id for row in picker.visible_rows
        }, "a typo'd query found nothing; soft matching is unreachable by keyboard"


@pytest.mark.asyncio
async def test_choosing_in_the_picker_resumes_that_session(tmp_path, monkeypatch) -> None:
    """Enter on a row is what actually resumes it — the picker's whole job."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc", prompt="the only session")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert boots == ["aabbcc"], boots


@pytest.mark.asyncio
async def test_escaping_the_picker_resumes_nothing(tmp_path, monkeypatch) -> None:
    """A cancelled picker leaves the session on screen exactly as it was."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc", prompt="the only session")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
    assert boots == [], boots


@pytest.mark.asyncio
async def test_resume_id_rebinds_and_reloads(tmp_path, monkeypatch) -> None:
    """``/resume <id>`` swaps the factory to that id and reboots the app."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "cafe01")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press(
            "/", "r", "e", "s", "u", "m", "e", " ", "c", "a", "f", "e", "0", "1", "enter"
        )
        await pilot.pause()
        await pilot.pause()
        assert boots == ["cafe01"], boots


@pytest.mark.asyncio
async def test_resume_replaces_the_visible_transcript(tmp_path, monkeypatch) -> None:
    """Switching sessions must show only the resumed conversation's history."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "cafe01")

    session = FakeSession()
    session._history = [
        SimpleNamespace(role="user", text="history from the current session", tool_calls=None)
    ]

    async def dispose_with_terminal_event() -> None:
        # A real session emits terminal events while dispose is awaited. The
        # old controller must already be detached or this notice is queued
        # behind the transcript replacement and lands in the resumed session.
        session.emit(NoticeEvent(text="stale event from disposed session", kind="warning"))
        session.disposed = True

    session.dispose = dispose_with_terminal_event  # type: ignore[method-assign]
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots, "history from the resumed session"),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press(
            "/", "r", "e", "s", "u", "m", "e", " ", "c", "a", "f", "e", "0", "1", "enter"
        )
        await pilot.pause()
        await pilot.pause()

        text = _transcript_text(app)
        assert boots == ["cafe01"], boots
        assert "history from the resumed session" in text, text
        assistant_texts = [
            block.text()
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, AssistantBlock)
        ]
        assert assistant_texts == ["resumed answer"], assistant_texts
        assert "history from the current session" not in text, text
        assert "/resume cafe01" not in text, text
        assert "resuming session cafe01" not in text, text
        assert "stale event from disposed session" not in text, text


@pytest.mark.asyncio
async def test_a_session_swap_never_shows_the_splash(tmp_path, monkeypatch) -> None:
    """Conversation A goes straight to conversation B — no logo in between.

    The splash is the transcript's EMPTY STATE, and a swap clears the ledger
    before repopulating it, so for a while "the transcript has no content" was
    true of a screen that was only mid-substitution. It was true for the whole
    of the session factory too, because the clear used to happen BEFORE the
    await: the user watched conversation A, then the centred logo, then
    conversation B.

    Asserted at every frame Textual paints rather than at the end, which is the
    only way to catch a state that exists only in the middle: ``post_display_hook``
    is Textual's own after-a-frame callback. The factory is made slow on purpose,
    because a free one hides the very window this is about.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "swap01")

    current = FakeSession()
    current._history = [
        SimpleNamespace(role="user", text="the conversation being left", tool_calls=None)
    ]

    async def slow_resume_factory(_resume_id: str | None):
        await asyncio.sleep(0.05)  # a real factory is not free
        resumed = FakeSession()
        resumed._history = [
            SimpleNamespace(role="user", text="the conversation arrived at", tool_calls=None)
        ]
        return resumed

    app = OperatorApp(lambda: _factory(current), resume_factory=slow_resume_factory)
    splash_frames: list[bool] = []
    watching = False
    original_hook = type(app).post_display_hook

    def hook(self) -> None:
        original_hook(self)
        if watching:
            splash_frames.append(self.query_one(WelcomeView).display)

    monkeypatch.setattr(type(app), "post_display_hook", hook)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one(TranscriptView).blocks(), "the session being left never rendered"
        assert app.query_one(WelcomeView).display is False

        watching = True
        # Exactly what `_resume_session` does before running the worker.
        app._session_factory = lambda: slow_resume_factory("swap01")  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(6):
            await pilot.pause()
        watching = False

        assert splash_frames, "no frame was painted across the swap"
        assert not any(splash_frames), (
            f"the splash was painted on {sum(splash_frames)} of "
            f"{len(splash_frames)} frames across the swap"
        )
        assert "the conversation arrived at" in _transcript_text(app)


@pytest.mark.asyncio
async def test_a_swap_onto_an_empty_session_still_lands_on_the_splash(tmp_path) -> None:
    """The suppression is about the TRANSITION, not about the destination.

    ``/new`` swaps onto a conversation with no history, and that is a genuinely
    empty app: the splash is right for it. A fix that simply stopped the swap
    from ever showing the splash would leave the fresh session on a blank
    screen with no boot card and no hints.
    """
    current = FakeSession()
    current._history = [SimpleNamespace(role="user", text="something", tool_calls=None)]

    async def new_factory(_resume_id: str | None):
        return FakeSession()  # no history: this is what /new asks for

    app = OperatorApp(lambda: _factory(current), resume_factory=new_factory)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one(WelcomeView).display is False

        app._session_factory = lambda: new_factory(None)  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        assert app.query_one(WelcomeView).display is True
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert app.query_one(TranscriptView).blocks() == []


@pytest.mark.asyncio
async def test_a_swap_leaves_the_ledger_matching_the_new_sessions_history() -> None:
    """The screen is a PROJECTION of the model's context, after the reorder too.

    The swap now builds the replacement session BEFORE clearing the ledger, so
    that the substitution paints in one frame. That reordering must not weaken
    the rule it moved around: what is on screen at the end is the new session's
    ``history()``, all of it and nothing else — including on the path where the
    replacement fails to construct, which leaves no conversation at all rather
    than the previous one under an error saying there is no session.
    """
    first = FakeSession()
    first._history = [
        SimpleNamespace(role="user", text="first-session prompt", tool_calls=None),
        SimpleNamespace(role="assistant", text="first-session answer", tool_calls=[]),
    ]
    second = FakeSession()
    second._history = [
        SimpleNamespace(role="user", text="second-session prompt", tool_calls=None),
        SimpleNamespace(role="assistant", text="second-session answer", tool_calls=[]),
        SimpleNamespace(role="user", text="second-session follow-up", tool_calls=None),
    ]

    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert len(app.query_one(TranscriptView).blocks()) == 2

        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        text = _transcript_text(app)
        assert len(app.query_one(TranscriptView).blocks()) == 3
        assert "first-session prompt" not in text, text
        assert "first-session answer" not in text, text
        for line in ("second-session prompt", "second-session answer", "second-session follow-up"):
            assert line in text, text

        # And the failure path clears too: a swap that lands on nothing must
        # not leave the previous conversation standing under the error.
        async def broken_factory():
            raise RuntimeError("factory exploded")

        app._session_factory = broken_factory  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        after = _transcript_text(app)
        assert "second-session prompt" not in after, after
        assert "factory exploded" in after, after
        assert app.query_one(WelcomeView).display is True


@pytest.mark.asyncio
async def test_resume_long_history_opens_at_the_latest_turn(tmp_path, monkeypatch) -> None:
    """A replay batch settles with the resumed conversation's tail in view."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "long01")

    async def resume_factory(_resume_id: str | None):
        resumed = FakeSession()
        resumed._history = [
            SimpleNamespace(
                role="user",
                text=f"resumed history line {index}",
                tool_calls=None,
            )
            for index in range(30)
        ]
        return resumed

    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=resume_factory,
    )
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for key in "/resume long01":
            await pilot.press(key if key != " " else "space")
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()

        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 30
        max_scroll_y = max(0, transcript.virtual_size.height - transcript.size.height)
        assert transcript.scroll_offset.y == max_scroll_y


@pytest.mark.asyncio
async def test_resume_at_latest_passes_the_sentinel_verbatim(tmp_path, monkeypatch) -> None:
    """``/resume @latest`` must hand the factory the ``@latest`` symbol, not
    strip the ``@`` (C14-02). resume.py only resolves the newest session on an
    EXACT ``@latest`` match; a stripped ``latest`` falls through to a literal
    ``sessions/latest`` path and fails to boot."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        # Type ``/resume @latest``
        for key in "/resume @latest":
            await pilot.press(key if key != " " else "space")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert boots == ["@latest"], boots


@pytest.mark.asyncio
async def test_a_tall_usage_overlay_never_scrolls_the_screen_or_steals_width() -> None:
    """A floating overlay must leave the layout beneath it alone.

    It did not. The `/usage` card is sized by the widget and positioned by an
    offset on the `toast` layer; layers keep it out of the FLOW, but nothing
    kept it out of the SCROLLABLE REGION. A card taller than its resting
    offset allowed pushed the screen's virtual height past its size, so
    Textual drew a screen scrollbar — which the app has no use for (the
    transcript is the scrolling surface and the input is docked) and which
    cost two cells of width, reflowing the transcript behind a popup that is
    supposed to change nothing. `overflow: hidden` on `Screen` is the guard.
    """
    from local_operator.tui.widgets.usage_panel import UsagePanel
    from tests.unit.tui.test_usage_panel import _many_reports

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        width_before = app.screen.size.width
        panel = app.query_one(UsagePanel)
        panel.display = True
        # Deliberately taller than the terminal so the overlay MUST overflow.
        panel.show_reports(_many_reports())
        await pilot.pause()
        await pilot.pause()
        assert app.screen.show_vertical_scrollbar is False
        assert app.screen.size.width == width_before
        # And the card's own content box still gets every row it composed:
        # the pinned height carries the padding, it does not eat the footer.
        assert panel.size.height == len(panel.render_lines_for_test())


# --- the boot default: is it settable, and did it land -----------------------


def _unwrapped(text: str) -> str:
    """Text with all whitespace removed.

    The transcript wraps a notice to the widget's width, breaking both prose and
    a tmp-dir path across lines. Comparing what was WRITTEN with what was
    RENDERED has to ignore the wrap, or the assertion is really about whichever
    terminal size the test happened to pick.
    """
    return "".join(text.split())


@pytest.mark.asyncio
async def test_a_bare_model_names_the_current_pair_and_the_command_that_keeps_it() -> None:
    """The complaint this closes: nothing on the model surfaces said a default
    existed, so `/model default` was reachable only by already knowing the word.
    Both the notice above the list and the list's own footer now say it."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model")
        await pilot.pause()
        await pilot.pause()
        text = _transcript_text(app)
        picker = app.query_one(Editor).model_picker
        footer = picker.render_text(90).plain.split("\n")[-1]
    # The subject of the sentence: "make THIS the default" needs a this.
    assert "openrouter/deepseek/deepseek-chat" in _unwrapped(text), text
    assert _unwrapped(PERSIST_HINT) in _unwrapped(text), text
    # …and the list itself, which is the surface the user is actually reading.
    assert PERSIST_HINT in footer, footer


@pytest.mark.asyncio
async def test_rejected_model_commands_keep_the_boot_composition() -> None:
    """A typo changed no state and must not permanently collapse the splash."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        welcome = app.query_one(WelcomeView)
        for command in ("/model missing-slash", "/model bogus-provider/model"):
            app._run_slash_command(command)
            await pilot.pause()
            assert welcome.display is True, command
            assert app.screen.has_class("boot"), command


@pytest.mark.asyncio
async def test_an_entered_rejected_model_command_keeps_the_boot_composition() -> None:
    """The SAME rule as the direct-call test above, now that the submit path
    agrees with it.

    This test used to assert the opposite — that entering the command dismissed
    the splash — because the submit handler echoed every slash command and
    retired the splash on its own. Two paths into one command answering "has the
    session started?" differently was the bug underneath: the echo, not the
    command, was deciding. With the echo on the registry (`SlashCommand.echo`)
    and the empty-state edge back on `_append_block`, a rejected selector is a
    rejected selector however it was typed.
    """
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        _set_editor_line(editor, "/model missing-slash")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        welcome_display = app.query_one(WelcomeView).display
        boot = app.screen.has_class("boot")

    # The rejection is still ON SCREEN, under the surviving splash: keeping the
    # boot composition must not cost the user the answer to what went wrong.
    assert "usage: /model <provider>/<model-id>" in painted, painted
    assert welcome_display is True
    assert boot


def test_help_uses_one_column_wider_than_every_command_name() -> None:
    """The two longest aliases used to consume the literal 14-cell column and
    glue directly onto descriptions: `/model, /modelsSwitch model…`."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    rows = _renderable_plain(app._help_block().renderable).splitlines()
    for command in SLASH_COMMANDS:
        names = ", ".join(f"/{name}" for name in command.names)
        row = next(row for row in rows if row.startswith(names))
        assert row[len(names) :].startswith("  "), row


def test_help_mentions_the_window_title_toggle() -> None:
    """The title lives outside the app's frame, so the help is where a user who
    notices it and wants it off can discover the toggle without source-diving."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    text = _renderable_plain(app._help_block().renderable)
    assert "window title" in text
    assert "lop config edit display.terminal_title false" in text
    assert "LOCAL_OPERATOR_NO_TERMINAL_TITLE" in text
    assert "lo ›" in text and "lo ⣾" in text and "lo !" in text


@pytest.mark.asyncio
async def test_the_paste_key_rows_do_not_wrap_at_eighty_columns() -> None:
    """Both key-reference rows fit on ONE line at the commonest terminal width.

    This defect has now shipped twice. `paste_note`'s copy was shortened from
    77 cells for it (#402, design round 1 D2), and the `cmd+v` row added beside
    it reintroduced it at 75 cells — caught independently by all three round-1
    reviewers (D1/F1/U2). Neither occurrence had a test, which is why the
    second one was possible.

    The failure shape is what makes it worth pinning rather than eyeballing:
    these notes `ljust(name_width)` the FIRST line only and have no
    continuation indent, so an overflowing description wraps to column 0 — the
    KEY gutter — and the tail reads as another key row rather than as the rest
    of a sentence. At 80 columns the user saw `Terminal.app)` sitting where a
    keybinding belongs.

    Asserted against the REAL PAINTED FRAME, not against a string length. The
    gutter is `name_width`, which is derived from the longest command name, so
    the budget shrinks whenever a command with a longer name is added: a test
    pinning today's copy to today's arithmetic would keep passing while the
    row silently wrapped again. Reading the compositor asks the only question
    that matters — is this row one line or two.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 44)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("/help")
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()

        painted = [
            "".join(segment.text for segment in strip).rstrip()
            for strip in app.screen._compositor.render_strips()
        ]

    # The whole description has to be ON the key's own row. A wrap is visible
    # as the row losing its tail, so assert the last word of each note is
    # present on the same line the key opens — the shape a hanging continuation
    # breaks by construction.
    for key, tail in (
        ("ctrl+v", "system clipboard"),
        ("cmd+v", "not Terminal.app"),
        ("!", "shell command"),
        ("option+left/right", "by word"),
        ("shift+tab", "reasoning effort"),
        ("ctrl+d", "empty composer"),
    ):
        row = next(
            (row for row in painted if row.strip().startswith(f"{key} ")),
            None,
        )
        assert row is not None, f"the {key} row is missing from /help"
        assert tail in row, (
            f"the {key} help row wrapped at 80 columns: {row.strip()!r} does not "
            f"carry {tail!r}, so the tail hangs at column 0 in the KEY gutter "
            "where it reads as another key row (design round 1 D2, again as "
            "D1/F1/U2)"
        )


@pytest.mark.asyncio
async def test_a_switch_admits_it_is_session_only_and_names_the_persist_command() -> None:
    """A switch that looks permanent and is not is the actual bug: the old
    "(next turn)" said WHEN it applied and never said for how long, so the next
    launch coming back on the old model read as the switch having been lost."""
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert _unwrapped("this session") in _unwrapped(text), text
    # #369 repointed the persist breadcrumb at the picker's `d` affordance, so
    # the receipt names PERSIST_HINT rather than the old `/model default`
    # spelling whose elided form no longer writes.
    assert _unwrapped(PERSIST_HINT) in _unwrapped(text), text
    # The access clause is unchanged by the new one sharing the line.
    assert _unwrapped("anthropic logged in") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_model_default_confirms_both_keys_and_the_file_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare "saved" is a claim the user cannot check without relaunching.
    The provider is the half that rides along silently — it is written from the
    selector's left side and never typed as a setting of its own."""
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # A file on disk, so ConfigManager builds its own Config rather than handing
    # back (and mutating) the module-level DEFAULT_CONFIG singleton.
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model default anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert written["hosting"] == "anthropic", written
    assert written["model_name"] == "claude-opus-5", written
    # What it wrote, under the names the config file uses…
    assert _unwrapped("hosting anthropic, model_name claude-opus-5") in _unwrapped(text), text
    # …and where, so the user can go and read or undo it.
    assert str(tmp_path / "config.yml") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_model_default_alone_confirms_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#369. A bare `/model default` used to write the CURRENT model straight
    into config.yml, replacing a default the user may have relied on, with no
    undo and a receipt that arrived after the write. Two intentions shared one
    spelling and the destructive one won.

    The assertion that matters is that the FILE IS UNCHANGED — byte for byte,
    not merely that a notice appeared. A confirmation that still wrote would
    pass any check made against the transcript alone.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    before = (tmp_path / "config.yml").read_bytes()
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        # Poll for the session rather than betting one frame is enough. A
        # single `pause()` lost the race under parallel CPU load: `/model`
        # refused with "session is still starting…" and the assertion below
        # then read a transcript containing only that refusal. Same cause and
        # same remedy as the submit-worker polls elsewhere in this file —
        # observed failing 4 of 6 runs on clean origin/main, so it is the test
        # that is racy, not the command.
        for _ in range(200):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/model anthropic/claude-opus-5")
        await _await_session(app, pilot)
        app._run_slash_command("/model default")
        await pilot.pause()
        text = _transcript_text(app)
    assert (tmp_path / "config.yml").read_bytes() == before, "bare /model default wrote to config"
    # It names the model it WOULD save, so the confirmation has a subject…
    assert _unwrapped("anthropic/claude-opus-5") in _unwrapped(text), text
    # …and both other readings, which is what removes the ambiguity rather
    # than merely deferring it.
    assert _unwrapped("/model saved") in _unwrapped(text), text
    assert _unwrapped("/model default anthropic/claude-opus-5") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_model_saved_switches_to_the_configured_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#369's missing reading: "put me back on my configured default".

    `/team` and `/agent` already have a detach verb; `/model` had none, so this
    intention was inexpressible and users reached for `/model default`, which
    did the opposite.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: anthropic\n  model_name: claude-opus-5\n"
    )
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model openrouter/deepseek/deepseek-chat")
        await _await_session(app, pilot)
        app._run_slash_command("/model saved")
        await pilot.pause()
        label = session.model_label
    assert label == "anthropic/claude-opus-5", label


@pytest.mark.asyncio
async def test_model_saved_says_so_when_no_default_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doing nothing silently would read as the switch having happened."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: ''\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model saved")
        await pilot.pause()
        text = _transcript_text(app)
    assert _unwrapped("no boot default saved yet") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_model_default_explicit_selector_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#369 changed the ELIDED form only. `/model default <p>/<id>` is
    unambiguous — the user named the model — so it keeps writing immediately."""
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model default anthropic/claude-opus-5")
        await pilot.pause()
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert (written["hosting"], written["model_name"]) == ("anthropic", "claude-opus-5"), written


@pytest.mark.asyncio
async def test_model_picker_d_saves_the_highlighted_row_as_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#369's PRIMARY fix: the affordance that removes the need to type the
    command, and with it the ambiguity. `d` saves; it deliberately does NOT
    switch, because Enter already means switch and a key that did both would
    make the two indistinguishable."""
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 30)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model")
        await pilot.pause()
        await pilot.pause()
        picker = app.query_one(Editor).model_picker
        row = picker.highlighted()
        assert row is not None
        before_label = session.model_label
        app._persist_default_from_picker()
        await pilot.pause()
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert (written["hosting"], written["model_name"]) == (row.provider, row.model_id), written
    # The session was NOT switched: `d` and Enter must stay distinguishable.
    assert session.model_label == before_label, session.model_label


_MODEL_DEFAULT_KEYS = {" ": "space", "/": "slash", "-": "minus"}


@pytest.mark.asyncio
async def test_typing_model_default_letter_by_letter_is_not_eaten_by_the_d_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 1, U8 (QA Q4): the advertised command, typed rather than pasted.

    `/model ` opens the picker with an empty query and the current model
    highlighted — which is also the state one keystroke into `/model default
    <p>/<id>`. The `d` gate (#369) read that `d` as "save the highlighted row":
    a silent write of the CURRENT model to config, then Enter switched to a
    model literally named `efault anthropic/…`. Measured before the fix:
    composer `/model efault anthropic/claude-opus-5`, config `model_name:
    deepseek/deepseek-chat`, notice `unknown provider: efault anthropic`.

    The gate now also asks whether the user NAVIGATED the list: arrowing to a
    row is what makes `d` a key; typing straight after the space is spelling.
    """
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 30)) as pilot:
        await _await_session(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for char in "/model default anthropic/claude-opus-5":
            await pilot.press(_MODEL_DEFAULT_KEYS.get(char, char))
        await pilot.pause()
        # Every letter reached the composer — the picker did not swallow one.
        assert editor.text == "/model default anthropic/claude-opus-5", editor.text
        await pilot.press("enter")
        for _ in range(5):
            await pilot.pause()
        text = _transcript_text(app)
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    # The NAMED model was persisted, not the one that happened to be highlighted.
    assert (written["hosting"], written["model_name"]) == ("anthropic", "claude-opus-5"), written
    assert session.model_label == "anthropic/claude-opus-5", session.model_label
    assert "efault anthropic" not in text, text
    assert "unknown provider" not in text, text


@pytest.mark.asyncio
async def test_d_still_saves_after_arrowing_to_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of U8: the #369 affordance survives, through real keys.

    ``test_model_picker_d_saves_the_highlighted_row_as_default`` calls the
    handler directly; this drives the KEY, because the gate it exercises now
    lives on the key path (empty query AND navigated) and a handler-level test
    could not tell a gate that never fires from one that fires correctly.
    """
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 30)) as pilot:
        await _await_session(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o", "d", "e", "l", "space")
        await pilot.pause()
        picker = editor.model_picker
        assert picker.is_open() and not picker.navigated()
        await pilot.press("down")
        await pilot.pause()
        assert picker.navigated()
        row = picker.highlighted()
        assert row is not None
        before_label = session.model_label
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        # The letter was the KEY: it did not land in the composer.
        assert editor.text == "/model ", editor.text
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert (written["hosting"], written["model_name"]) == (row.provider, row.model_id), written
    assert session.model_label == before_label


@pytest.mark.asyncio
async def test_every_model_default_surface_says_it_the_same_way() -> None:
    """D14. One instruction had four wordings on four surfaces a user meets
    within two keystrokes. The canonical sentence now names the consequence —
    future sessions — rather than merely saying an unspecified pair is saved.

    The defect is the divergence, so all four surfaces are checked together.
    The footer is checked unwrapped and whole because it is the tightest site:
    an instruction that is consistent only after truncation is not consistent.
    """
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 30)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model")
        await pilot.pause()
        await pilot.pause()
        picker = app.query_one(Editor).model_picker
        footer = picker.render_text(picker.size.width or 90).plain.split("\n")[-1]
        bare_notice = _transcript_text(app)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        receipt = _transcript_text(app)
        app._run_slash_command("/help")
        await pilot.pause()
        help_text = _transcript_text(app)

    # 1. the notice a bare `/model` prints above the list, 2. the switch receipt,
    # 3. the picker's own footer, 4. the `/help` row.
    assert _unwrapped(PERSIST_HINT) in _unwrapped(bare_notice), bare_notice
    assert _unwrapped(PERSIST_HINT) in _unwrapped(receipt), receipt
    assert PERSIST_HINT in footer, footer
    assert _unwrapped(PERSIST_HINT) in _unwrapped(help_text), help_text
    model_row = next(c for c in SLASH_COMMANDS if c.name == "model")
    assert PERSIST_HINT in model_row.description, model_row.description

    # And the four it replaced are gone from all of them, so there is no second
    # phrasing left for the user to meet.
    everything = _unwrapped(bare_notice + receipt + footer + help_text)
    for stale in (
        "as the boot default",
        "to make it the boot default",
        "saves the boot default",
        "/model default persists it",
    ):
        assert _unwrapped(stale) not in everything, stale

    # The receipt is two clauses now, not a run-on of four separators — and it
    # still says the two things that made it necessary: the scope, and the access
    # state of the provider it just switched to.
    switch_line = next(line for line in _unwrapped(receipt).split("·") if _unwrapped("→") in line)
    assert _unwrapped("(this session)") in switch_line, switch_line
    assert _unwrapped("from the next turn") not in _unwrapped(receipt), receipt
    assert _unwrapped("anthropic logged in") in _unwrapped(receipt), receipt


@pytest.mark.asyncio
async def test_a_mid_turn_switch_says_when_it_starts_applying() -> None:
    """Switched while the agent is working, the receipt says WHEN it applies.

    Mid-turn is the one moment "starting when" is a live question: the request
    already streaming cannot be re-targeted, so for a few seconds the user
    watches the OLD model keep working after the band has repainted to the new
    one. Without a word here that reads as the switch having been ignored,
    which is the complaint this whole change exists to fix.
    """
    session = _SwitchableSession()
    session.streaming = True
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _unwrapped(_transcript_text(app))
        blocks = list(app.query_one(TranscriptView).blocks())

    assert _unwrapped(MODEL_SWITCH_MID_TURN_NOTICE) in text, text
    # On its OWN row: the receipt keeps its two-clause budget, which is what
    # keeps it to two wrapped lines at the widths this app supports. Folding the
    # timing into the scope parenthetical measured three.
    assert _unwrapped("(this session)") in text, text
    # #369: the persist breadcrumb is PERSIST_HINT's `d` affordance now, not the
    # old `/model default` spelling.
    assert _unwrapped(PERSIST_HINT) in text, text
    # SAME INK as the receipt it qualifies (design review D3). At `note` the
    # subordinate row measured 8.62:1 against the receipt's 4.55:1 and the eye
    # landed on the qualifier first; the token is asserted rather than the
    # colour so this survives a palette change.
    notices = [b for b in blocks if isinstance(b, NoticeBlock)]
    qualifier = next(
        b
        for b in notices
        if MODEL_SWITCH_MID_TURN_NOTICE in _renderable_plain(getattr(b, "renderable", ""))
    )
    receipt = next(
        b for b in notices if "\u2192" in _renderable_plain(getattr(b, "renderable", ""))
    )
    assert qualifier._token == receipt._token, (qualifier._token, receipt._token)


@pytest.mark.asyncio
async def test_model_default_mid_turn_also_says_when_it_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/model default p/id`` switches the LIVE session too, so it owes the same answer.

    Design review D1: the timing row was nested in the plain-switch branch, so
    this spelling printed only "used from the next launch" while the status band
    beside it had already repainted to the new model — a receipt contradicting
    the band on the same frame, which is the exact class of bug this PR fixes.
    """
    import yaml  # noqa: F401  (parity with the sibling default tests)

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    session.streaming = True
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model default anthropic/claude-opus-5")
        await pilot.pause()
        text = _unwrapped(_transcript_text(app))

    assert _unwrapped(MODEL_SWITCH_MID_TURN_NOTICE) in text, text
    # Still the persistence receipt, not the session one: this asserts the row
    # was ADDED to that branch rather than the branch being changed.
    assert _unwrapped("used from the next launch") in text, text


@pytest.mark.asyncio
async def test_reselecting_the_model_already_in_force_promises_no_handover() -> None:
    """A no-op switch must not promise a handover that will never happen (D4).

    "this one finishes on the old model" describes a transition between two
    models. Re-picking the running model — easy to do from the picker, where the
    current row is one Enter away — has no old model to finish on.
    """
    session = _SwitchableSession()
    session.streaming = True
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        # Whatever the fake reports as current, selected again.
        app._run_slash_command(f"/model {session.model_label}")
        await pilot.pause()
        text = _unwrapped(_transcript_text(app))

    assert _unwrapped(MODEL_SWITCH_MID_TURN_NOTICE) not in text, text


@pytest.mark.asyncio
async def test_an_idle_switch_does_not_talk_about_steps() -> None:
    """Between turns there is no step to wait for, so the timing clause is noise.

    Paired with the test above: together they pin that the clause is
    CONDITIONAL, not that it was simply added to the string.
    """
    session = _SwitchableSession()
    session.streaming = False
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await _await_session(app, pilot)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _unwrapped(_transcript_text(app))

    assert _unwrapped("(this session)") in text, text
    assert _unwrapped(MODEL_SWITCH_MID_TURN_NOTICE) not in text, text


@pytest.mark.asyncio
async def test_model_default_hint_survives_every_supported_narrow_footer() -> None:
    """More terminal width must never hide words from the approved persistence
    instruction; the 50-column footer must keep the command whole."""
    for size in ((50, 20), (60, 22), (80, 24), (100, 30)):
        ctrl = _AccessController()
        app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            await pilot.pause()
            painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert PERSIST_HINT in painted, (size, painted)


@pytest.mark.asyncio
async def test_boot_warms_session_imports_before_awaiting_the_factory() -> None:
    """The import cost must be paid in a thread, ahead of the factory await.

    ``create_session`` does not yield until it has imported the engine, the
    provider stack and the MCP SDK — measured at ~700 ms — so awaiting it
    directly from the boot worker freezes the compositor and the key handler
    for that whole window: the user's first keystrokes land in a dead screen
    and appear in a burst afterwards. Ordering is the contract (warm, THEN
    build) and the thread hop is what makes the warm-up non-blocking, so both
    are asserted here.
    """
    import threading

    calls: list[str] = []
    warm_thread: list[int] = []

    def fake_warm() -> None:
        warm_thread.append(threading.get_ident())
        calls.append("warm")

    async def factory() -> FakeSession:
        calls.append("factory")
        return FakeSession()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("local_operator.session_factory.warm_session_imports", fake_warm)
    try:
        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for _ in range(50):
                if calls.count("factory"):
                    break
                await pilot.pause()
            loop_thread = threading.get_ident()
    finally:
        monkeypatch.undo()

    assert calls[:2] == ["warm", "factory"], calls
    assert warm_thread and warm_thread[0] != loop_thread


@pytest.mark.asyncio
async def test_quitting_during_boot_never_paints_into_the_dismantled_screen() -> None:
    """The boot worker must be cancelled BEFORE the widget tree is pruned.

    Textual's shutdown prunes every screen and widget first and cancels workers
    afterwards, so a boot worker that wakes in between resumes into
    ``_adopt_session`` → ``_render_resumed_history`` → ``_transcript_view()``,
    where the ``#transcript`` re-lookup for a transcript that is no longer in the
    tree raises ``NoMatches`` — and ``exit_on_error`` turns it into
    ``WorkerFailed``, i.e. a traceback for anyone who quits while the session is
    still starting (the import warm-up alone is ~700 ms; see
    ``_warm_session_imports``). It is that crash which surfaced in this suite's
    per-width app loops as a ``NoMatches`` on a screen still wearing ``boot``:
    sweeping only the hop's duration across an app's lifetime hit it 6 times in
    301 apps, so a loop that tears one app down per width meets it eventually.

    So the ordering is the contract, and the arrangement below pins it rather
    than sampling it: the factory is held until the prune has FINISHED, then
    released with ticks to spare before Textual's own ``cancel_all``. Patching
    ``App._close_all`` is the only seam that brackets the prune — nothing public
    runs between it and the cancellation it races. Restoring the app's order
    (dropping ``OperatorApp._shutdown``, or textual renaming what it overrides)
    fails here again.
    """
    from textual.app import App
    from textual.worker import WorkerState

    factory_gate = asyncio.Event()

    async def warm() -> None:
        await factory_gate.wait()

    pruned_then_released: list[str] = []
    original_close_all = App._close_all

    async def close_all(self: App[None]) -> None:
        await original_close_all(self)
        pruned_then_released.append("pruned")
        factory_gate.set()
        # Ticks for the boot worker to be scheduled in the window Textual leaves
        # open between the prune and its own `workers.cancel_all()`.
        for _ in range(4):
            await asyncio.sleep(0)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(OperatorApp, "_warm_session_imports", staticmethod(warm))
    monkeypatch.setattr(App, "_close_all", close_all)
    try:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            boot_worker = next(w for w in app.workers if w.group == "session")
            # The premise: boot is still in flight, so teardown genuinely races
            # it. Without this the test could pass by never having a race.
            assert app._session is None
    finally:
        monkeypatch.undo()

    assert pruned_then_released == ["pruned"]
    # Cancelled at its own await instead of resuming into the dead tree: no
    # adoption, no query, and the worker carries no error for `_handle_exception`
    # to exit the app with.
    assert boot_worker.state is WorkerState.CANCELLED, boot_worker.state
    assert app._session is None
    assert app._exception is None, app._exception


class _MeasuredSession(FakeSession):
    """A session that can report what it is already carrying."""

    def __init__(self, tokens: int = 42_318) -> None:
        super().__init__()
        self.tokens = tokens
        self.measure_calls = 0

    @property
    def model(self) -> Any:
        return SimpleNamespace(context_window=1_000_000, reasoning_effort="", reasoning=False)

    async def measure_preloaded_context(self) -> int:
        self.measure_calls += 1
        return self.tokens


def _ctx_estimate(app: OperatorApp) -> bool:
    """Whether the band's reading is a local estimate rather than the wire."""
    status = app._status
    assert status is not None
    return status.context_is_estimate


def _ctx_tokens(app: OperatorApp) -> int:
    """The band's context reading, with the ``_status is None`` case ruled out.

    The attribute is Optional until compose runs; every caller here is inside
    ``run_test``, where it never is.
    """
    status = app._status
    assert status is not None
    return status.context_tokens


async def _settle(pilot, predicate, tries: int = 60) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await pilot.pause()
    return predicate()


@pytest.mark.asyncio
async def test_context_reads_before_the_first_message() -> None:
    """The band must not claim an empty context while the prompt is loaded.

    The provider's exact ``prompt_tokens`` only exists after a turn, so without
    a boot measurement the one number a user checks before deciding what to ask
    is blank precisely when the system prompt, skills index and every tool
    schema are already spent.
    """
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)
        assert _ctx_estimate(app) is True
        assert session.prompts == [], "measuring must not send anything"


@pytest.mark.asyncio
async def test_a_real_turn_supersedes_the_estimate() -> None:
    """An estimate is a stand-in, not a competitor: the wire number wins."""
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=51_007))
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 51_007)
        assert _ctx_estimate(app) is False

        # And a later re-measure (an MCP server connecting) must not walk it back.
        session.tokens = 9_999
        app._measure_preloaded_context(session)
        for _ in range(10):
            await pilot.pause()
        assert _ctx_tokens(app) == 51_007


@pytest.mark.asyncio
async def test_a_turn_landing_mid_measurement_still_wins() -> None:
    """The race the cheap outer check cannot cover.

    A measurement that started while the context was unknown can finish AFTER
    a turn has reported the provider's exact count. Deciding only at dispatch
    would let the stale estimate overwrite the better number, so the result is
    re-checked against the state at the moment it lands.
    """
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        # Park a second measurement mid-flight, with the reading still an
        # estimate so the dispatch-time check lets it through.
        release = asyncio.Event()

        async def slow_measure() -> int:
            await release.wait()
            return 9_999

        session.measure_preloaded_context = slow_measure  # type: ignore[method-assign]
        app._measure_preloaded_context(session)
        await pilot.pause()

        # The turn lands while that is parked...
        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=51_007))
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 51_007)

        # ...and the late estimate must not undo it.
        release.set()
        for _ in range(15):
            await pilot.pause()
        assert _ctx_tokens(app) == 51_007
        assert _ctx_estimate(app) is False


@pytest.mark.asyncio
async def test_a_growing_tool_inventory_re_measures() -> None:
    """MCP schemas are the biggest term, and they land after boot."""
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        session.tokens = 88_000
        app._measure_preloaded_context(session)
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 88_000)
        assert _ctx_estimate(app) is True


@pytest.mark.asyncio
async def test_a_session_without_the_capability_is_not_a_crash() -> None:
    """Reduced hosts (embedders, these fakes) have no measurement to offer."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(10):
            await pilot.pause()
        assert _ctx_tokens(app) == 0
        assert _ctx_estimate(app) is False


@pytest.mark.asyncio
async def test_reload_drops_the_dead_sessions_context() -> None:
    """An exact count belongs to one conversation, and dies with it.

    The dispatch guard reads "a non-zero reading that is not an estimate" as
    "the band already knows better than you". Left standing across a reload
    that is a lie about a session that no longer exists, and it also suppresses
    the replacement session's own measurement.
    """
    first = _MeasuredSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=51_007))
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 51_007)

        second = _MeasuredSession(tokens=20_000)
        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()

        assert await _settle(pilot, lambda: _ctx_tokens(app) == 20_000)
        assert second.measure_calls >= 1, "the new session must be measured"


@pytest.mark.asyncio
async def test_the_newest_measurement_wins_not_the_slowest() -> None:
    """Several measurements can be in flight; they finish out of order.

    Neither precedence guard can break that tie — both only ask whether the
    reading is exact, which is false for every estimate — so without
    cancellation the LAST to land wins rather than the newest, and the band
    settles on a smaller inventory than the session actually has.
    """
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        slow_release = asyncio.Event()

        async def slow() -> int:
            await slow_release.wait()
            return 30_000

        async def fast() -> int:
            return 90_000

        session.measure_preloaded_context = slow  # type: ignore[method-assign]
        app._measure_preloaded_context(session)
        await pilot.pause()

        session.measure_preloaded_context = fast  # type: ignore[method-assign]
        app._measure_preloaded_context(session)
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 90_000)

        # Releasing the superseded measurement must not walk the band back.
        slow_release.set()
        for _ in range(15):
            await pilot.pause()
        assert _ctx_tokens(app) == 90_000


@pytest.mark.asyncio
async def test_a_dead_sessions_turn_cannot_restore_its_context() -> None:
    """The race the reload reset alone does not close.

    A plain ``/reload`` deliberately keeps the controller SUBSCRIBED across
    ``dispose()`` so the dying session's ``agent_end`` can settle its live tool
    cards. That same event carries a ``context_tokens``, and it is posted to
    the message pump — so whether it arrives before or after the reload's reset
    is scheduling. Arriving after, it reinstates an exact reading for a
    conversation that no longer exists, and the exact-count guard then
    suppresses the replacement session's own measurement: the reset undone.
    """
    first = _MeasuredSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        second = _MeasuredSession(tokens=20_000)
        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 20_000)

        # The dying session's turn lands late, carrying its own exact count.
        app._session = None
        app.post_message(TurnEnded(aborted=True, error=None, context_tokens=51_007))
        for _ in range(15):
            await pilot.pause()

        assert _ctx_tokens(app) == 20_000, "a dead session's count was adopted"
        assert _ctx_estimate(app) is True


def _resumed(*messages: Any) -> Any:
    """A session whose ``history()`` is the given messages."""

    class _Resumed(FakeSession):
        def history(self) -> list[Any]:
            return list(messages)

    return _Resumed()


def _blocks_by_type(app: OperatorApp) -> dict[str, int]:
    from local_operator.tui.widgets.assistant import AssistantBlock as _AB
    from local_operator.tui.widgets.tool_card import ToolCard as _TC
    from local_operator.tui.widgets.transcript import NoticeBlock as _NB
    from local_operator.tui.widgets.transcript import UserBlock as _UB

    counts = {"user": 0, "assistant": 0, "tool": 0, "notice": 0}
    for block in app.query_one(TranscriptView).blocks():
        if isinstance(block, _UB):
            counts["user"] += 1
        elif isinstance(block, _AB):
            counts["assistant"] += 1
        elif isinstance(block, _TC):
            counts["tool"] += 1
        elif isinstance(block, _NB):
            counts["notice"] += 1
    return counts


class TestResumeReplaysTheWholeConversation:
    """What ``--resume`` puts back on screen.

    The old rule was "prompts, plus assistant messages carrying no tool calls".
    An agent turn is text AND tool_calls in ONE message, so that excluded the
    prose too: on a real 396-message session it mounted 6 blocks and dropped 74
    assistant messages, 215 calls and 215 results. The screen read as a list of
    questions nobody had answered.
    """

    @pytest.mark.asyncio
    async def test_prose_that_accompanies_a_tool_call_is_kept(self) -> None:
        from local_operator.harness.types import Message, ToolCall

        session = _resumed(
            Message.user("find the bug"),
            Message(
                role="assistant",
                content=[TextContent(text="Reading the file first.")],
                tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "a.py"})],
            ),
            Message(
                role="tool",
                content=[TextContent(text="file contents")],
                tool_call_id="c1",
                tool_name="read",
            ),
            Message.assistant("Found it."),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 1)
            counts = _blocks_by_type(app)
        # The old rule scored assistant=1 here; the turn's own sentence was lost.
        assert counts == {"user": 1, "assistant": 2, "tool": 1, "notice": 0}

    @pytest.mark.asyncio
    async def test_a_tool_call_is_paired_with_its_result(self) -> None:
        """Results can arrive several messages after the call that asked for
        them, so they are indexed rather than matched positionally."""
        from local_operator.harness.types import Message, ToolCall
        from local_operator.tui.widgets.tool_card import ToolCard

        session = _resumed(
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="a", name="read", arguments={"path": "x"}),
                    ToolCall(id="b", name="bash", arguments={"command": "ls"}),
                ],
            ),
            Message(
                role="tool",
                content=[TextContent(text="boom")],
                tool_call_id="b",
                tool_name="bash",
                is_error=True,
            ),
            Message(
                role="tool", content=[TextContent(text="ok")], tool_call_id="a", tool_name="read"
            ),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 2)
            cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert [c.tool_name for c in cards] == ["read", "bash"], "call order, not result order"
        assert [c._state for c in cards] == ["success", "error"]

    @pytest.mark.asyncio
    async def test_a_call_with_no_result_is_interrupted_not_complete(self) -> None:
        """A session killed mid-turn leaves a call with no answer. Rendering it
        as done would invent an outcome that never happened."""
        from local_operator.harness.types import Message, ToolCall
        from local_operator.tui.widgets.tool_card import ToolCard

        session = _resumed(
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="orphan", name="bash", arguments={"command": "sleep 99"})],
            ),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 1)
            card = next(
                b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)
            )
        assert card._state == "interrupted"

    @pytest.mark.asyncio
    async def test_a_replayed_card_reports_no_duration(self) -> None:
        """The transcript records what a tool did, never how long it took.
        ``0.0s`` on every row is a wrong number, not a missing one."""
        from local_operator.harness.types import Message, ToolCall
        from local_operator.tui.widgets.tool_card import ToolCard

        session = _resumed(
            Message(role="assistant", tool_calls=[ToolCall(id="c", name="read", arguments={})]),
            Message(role="tool", content=[TextContent(text="ok")], tool_call_id="c"),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 1)
            card = next(
                b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)
            )
            rendered = card._build_row(100).plain
        assert card._duration is None
        assert "0.0s" not in rendered, rendered

    @pytest.mark.asyncio
    async def test_a_failed_turn_says_so_instead_of_vanishing(self) -> None:
        """The reported symptom: two identical prompts and nothing between them.

        The turn had errored, and an assistant message with neither prose nor a
        call was skipped — so the screen implied the second prompt was a
        duplicate rather than a retry.
        """
        from local_operator.harness.types import Message

        session = _resumed(
            Message.user("do the thing"),
            Message(role="assistant", stop_reason="error"),
            Message.user("do the thing"),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["notice"] == 1)
            counts = _blocks_by_type(app)
        assert counts["user"] == 2 and counts["notice"] == 1

    @pytest.mark.asyncio
    async def test_a_refused_turn_replays_the_providers_refusal(self) -> None:
        """A refusal is not a generic "turn failed": the loop stashes the
        provider's own words on the assistant message so a resumed session can
        show WHAT was refused — the half of the diagnosis that decides between
        rephrasing and switching models."""
        from local_operator.harness.types import Message
        from local_operator.tui.widgets.transcript import NoticeBlock

        session = _resumed(
            Message.user("do the thing"),
            Message(
                role="assistant",
                stop_reason="refusal",
                provider_payload={"refusal": "model refused: I can't help with that. [marker]"},
            ),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["notice"] == 1)
            notice = next(
                b for b in app.query_one(TranscriptView).blocks() if isinstance(b, NoticeBlock)
            )
        assert "I can't help with that." in notice._text

    @pytest.mark.asyncio
    async def test_a_refused_turn_with_partial_prose_still_shows_the_refusal(self) -> None:
        """Gemini safety stops often cut a partial answer: the prose alone
        reads as a complete, oddly short reply, so the refusal notice must
        appear even when text was streamed."""
        from local_operator.harness.types import Message
        from local_operator.harness.types import TextContent as _Text

        session = _resumed(
            Message.user("do the thing"),
            Message(
                role="assistant",
                content=[_Text(text="Here is the beginning of")],
                stop_reason="refusal",
                provider_payload={"refusal": "model refused and sent no message [SAFETY]"},
            ),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(
                pilot,
                lambda: _blocks_by_type(app)["notice"] == 1
                and _blocks_by_type(app)["assistant"] == 1,
            )
            counts = _blocks_by_type(app)
        assert counts["assistant"] == 1 and counts["notice"] == 1

    @pytest.mark.asyncio
    async def test_a_fresh_session_still_replays_nothing(self) -> None:
        """The splash must not be retired by an empty history."""
        app = OperatorApp(lambda: _factory(_resumed()))
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(10):
                await pilot.pause()
            assert _blocks_by_type(app) == {"user": 0, "assistant": 0, "tool": 0, "notice": 0}


@pytest.mark.asyncio
async def test_new_starts_a_fresh_conversation() -> None:
    """``/new`` had no equivalent: ``/clear`` keeps the conversation the model
    sees, ``/reload`` reboots the same one, ``/resume`` moves to another
    existing one. Starting fresh meant quitting the app."""
    boots: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/new")
        assert await _settle(pilot, lambda: boots == [None])
    # None, not "" and not a sentinel id: create_session branches on
    # `resume is not None`, so this is the cold-launch path.
    assert boots == [None]


@pytest.mark.asyncio
async def test_new_replaces_the_visible_ledger() -> None:
    """The transcript must not outlive the conversation it describes."""
    boots: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._append_block(NoticeBlock("from the old conversation", "info"))
        assert any(
            "old conversation" in _renderable_plain(getattr(b, "renderable", ""))
            for b in app.query_one(TranscriptView).blocks()
        )
        app._run_slash_command("/new")
        assert await _settle(pilot, lambda: boots == [None])
        for _ in range(10):
            await pilot.pause()
        assert not any(
            "old conversation" in _renderable_plain(getattr(b, "renderable", ""))
            for b in app.query_one(TranscriptView).blocks()
        )


@pytest.mark.asyncio
async def test_new_without_a_capable_launcher_says_so() -> None:
    """Embedders may supply no session factory beyond the first."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/new")
        for _ in range(6):
            await pilot.pause()
        text = "\n".join(
            _renderable_plain(getattr(b, "renderable", ""))
            for b in app.query_one(TranscriptView).blocks()
        )
    assert "unavailable" in text


# --- "am I focused?", both sides of it ----------------------------------------
#
# The composer answered neither side. It drew no caret at all while the buffer
# was empty, so clicking into it changed nothing on the frame; and its bright
# chevron — the affordance meant to carry focus (D23) — was driven by
# `#input-dock:focus-within`, which Textual never re-applied on blur, so once
# lit it stayed lit for the life of the process. Both directions are pinned
# here, off a real composed frame rather than off widget state: what the user
# was missing is what the terminal was SENT.


@pytest.mark.asyncio
async def test_the_focused_composer_shows_a_caret_and_the_blurred_one_shows_none() -> None:
    """The reported case exactly: an EMPTY composer, clicked into.

    A caret on an empty field has nowhere to go but the placeholder's first
    cell, so the placeholder starts one column later while the caret is drawn
    and both survive — the block is on a blank cell, and the copy is unbroken.
    Blurring takes the caret away again, which is the half that makes the
    presence of one mean anything.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        assert editor.text == "", "premise: the reported state is an EMPTY composer"
        focused = composer_cells(app)
        assert caret_cells(focused) == [" "], "the focused composer drew no caret"
        placeholder = [text for text, _, _ in focused if "Message Local Operator" in text]
        assert placeholder, f"the caret broke the placeholder up: {focused}"

        app.set_focus(None)
        await pilot.pause()
        assert not caret_cells(composer_cells(app)), "the caret survived the blur"

        editor.focus()
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "the caret did not come back"


@pytest.mark.asyncio
async def test_the_caret_holds_still_across_consecutive_frames() -> None:
    """A caret that is absent from half the frames is not a focus signal.

    Sampled across four stock blink periods (500 ms each), empty and with text:
    the empty state is new, and it is the one where a reintroduced blink would
    strobe against a static splash.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert editor.cursor_blink is False

        for text in ("", "hello"):
            editor.load_text(text)
            await pilot.pause()
            samples = set()
            for _ in range(8):
                await asyncio.sleep(0.25)
                await pilot.pause()
                samples.add(tuple(composer_cells(app)))
            assert len(samples) == 1, f"the composer row changed between frames: {text!r}"
            assert caret_cells(next(iter(samples))), f"no caret at all with text={text!r}"


@pytest.mark.asyncio
async def test_the_chevron_brightens_on_focus_and_never_spends_the_accent() -> None:
    """The second affordance, in both directions — it used to only go on.

    And it used to go on GREEN. The accent means "a turn is live" and the
    composer is focused in nearly every frame, so the chevron sat permanently
    accent two rows above the band's streaming spinner, which is accent for a
    different reason (D5). Focus is a brightness step in the same neutral ramp
    now: `fg` focused, `dim` blurred, a 3.86x luminance move where the accent
    was 2.15x. The accent assertion is the load-bearing one — a future pass
    reaching for green here reintroduces the collision, and the two-meanings
    bug is invisible in any frame where no turn is running.

    The class is asserted alongside the painted colour because the class is the
    mechanism the stylesheet reads: a bright chevron with the class off would be
    a stale frame, and the class on with a dim chevron would be a broken rule.
    The blur is a tool card taking focus — a real product surface, the one a
    user reaches by clicking or tabbing to a tool row, and it leaves the
    composer on screen, which is exactly when "are my keys still going to the
    composer?" is a live question.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        dock = app.query_one("#input-dock")
        focused_ink = theme_mod.semantic_color("fg").lower()
        accent = theme_mod.semantic_color("accent").lower()
        dim = theme_mod.semantic_color("dim").lower()

        app.query_one(Editor).focus()
        await pilot.pause()
        assert dock.has_class(COMPOSER_FOCUSED_CLASS)
        assert chevron_colour(composer_cells(app)) == focused_ink
        assert chevron_colour(composer_cells(app)) != accent, "focus spent the accent"

        card = ToolCard("t1", "bash", {"command": "ls"})
        app._append_block(card)
        await pilot.pause()
        card.focus()
        await pilot.pause()
        assert not dock.has_class(COMPOSER_FOCUSED_CLASS)
        assert chevron_colour(composer_cells(app)) == dim, "the chevron stayed lit"

        app.query_one(Editor).focus()
        await pilot.pause()
        assert dock.has_class(COMPOSER_FOCUSED_CLASS)
        assert chevron_colour(composer_cells(app)) == focused_ink


@pytest.mark.asyncio
async def test_the_read_only_composer_shows_neither_caret_nor_bright_chevron() -> None:
    """The subagent page's composer refuses every key, and looks it.

    Today this holds because `_set_composer_read_only` drops `can_focus` and
    pushes focus off the widget, so `TextArea._draw_cursor`'s read-only branch
    (`show_cursor and has_focus`, a DIFFERENT expression from the one every
    other state takes) resolves False. That is a chain of three decisions in
    two files, and restoring focusability at either end would put a caret in a
    field that ignores it — the most misleading thing this mode could paint.
    Pinned on the frame so it goes red there rather than in a bug report.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "premise: it starts focused"

        app._set_composer_read_only(True)
        await pilot.pause()
        assert editor.read_only is True
        cells = composer_cells(app)
        assert not caret_cells(cells), "a caret is pointing into a field that refuses keys"
        assert chevron_colour(cells) == theme_mod.semantic_color("dim").lower()

        app._set_composer_read_only(False)
        app.query_one(Editor).focus()
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "the composer never came back"


@pytest.mark.asyncio
async def test_the_aside_keeps_the_caret_because_it_keeps_the_composer() -> None:
    """`/btw` opens a card the composer types INTO, so focus never moves.

    It is the one mode where the two affordances have to stay ON while the
    conversation's own surface has visibly stepped aside, and the placeholder
    is a sentence about the mode rather than an invitation — so this is also
    the proof that the caret's cell is taken from the field, not from the copy,
    whatever the copy happens to say.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()

        assert app._open_aside() is not None
        await pilot.pause()
        assert app.query_one(Editor).placeholder == ASIDE_PLACEHOLDER
        cells = composer_cells(app)
        assert caret_cells(cells) == [" "], "the aside took the caret with it"
        assert chevron_colour(cells) == theme_mod.semantic_color("fg").lower()
        # The CONSTANT, not a copy literal: the claim is that the caret takes
        # its cell from the field rather than out of the placeholder, which is
        # true whatever the placeholder says. Spelling the sentence out here
        # made this test fail on a wording change that did not touch the caret.
        assert [
            text for text, _, _ in cells if ASIDE_PLACEHOLDER in text
        ], f"the caret broke the aside's own placeholder up: {cells}"


def _osc_titles(writes: Sequence[str]) -> list[str]:
    """The OSC 0 title payloads, in order, deduped only by the caller.

    Saved titles (`\x1b[22;0t`) and restores (`\x1b[23;0t`) are different
    evidence and are asserted separately, so this helper extracts only the
    "what would the tab title read" payloads.
    """
    out: list[str] = []
    for data in writes:
        if data.startswith("\x1b]0;") and data.endswith("\x07"):
            out.append(data.removeprefix("\x1b]0;").removesuffix("\x07"))
    return out


def _spy_driver_writes(app: OperatorApp, writes: list[str]) -> None:
    """Record the real driver's writes while still forwarding them.

    Tests want the app-level contract (`_start_terminal_title` writes through
    the driver and `on_unmount` restores through it), not a fake driver with a
    growing list of Textual methods to impersonate.
    """
    assert app._driver is not None
    original = app._driver.write

    def spy(data: str) -> None:
        writes.append(data)
        original(data)

    app._driver.write = spy  # type: ignore[method-assign]


def _isolate_tui_settings(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    """Point display-setting reads at a disposable config dir for one test.

    The title path consults `display.terminal_title`, and these tests are about
    app wiring, not the developer's personal config. Using the shared config-dir
    override exercises the real setting path while guaranteeing the test can
    neither vary with nor create anything in `~/.local-operator`.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "lo-config"))
    from local_operator.tui.settings import settings_reload

    settings_reload()


@pytest.mark.asyncio
async def test_app_wires_the_terminal_title_to_boot_and_turn_state(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    """The app, not just the band, must prove the wiring works.

    ``run_test`` uses Textual's headless driver, so `_start_terminal_title`
    normally exits early and the app-level contract would otherwise go
    unexercised: boot attaches a title writer, the writer sees the cwd fallback
    on first stable paint, and a started turn flips it to the working state.
    """
    writes: list[str] = []
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        _spy_driver_writes(app, writes)
        app._stop_terminal_title()
        app._start_terminal_title()
        await pilot.pause()
        titles = _osc_titles(writes)
        assert app._status is not None
        cwd_label = Path(app._status._cwd).name  # the band's own fallback label
        assert titles[-1] == f"lo › {cwd_label}"
        assert "lo ›" not in titles

        app.on_turn_started(TurnStarted())
        await pilot.pause()
        latest = _osc_titles(writes)[-1]
        assert latest.endswith(f" {cwd_label}")
        assert latest.split(" ")[1] in {"⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"}


@pytest.mark.asyncio
async def test_session_swap_clears_a_parked_approval_title(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    """A dead session must not leave `lo ! …` standing on the replacement.

    This is the bug the review found: `_approval` was cleared during a session
    swap, but the title's derived `attention` state was not, so `/reload` or
    `/new` from a parked approval left the next idle session claiming it still
    owed one.
    """
    writes: list[str] = []
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        _spy_driver_writes(app, writes)
        app._stop_terminal_title()
        app._start_terminal_title()
        await pilot.pause()
        assert app._status is not None
        cwd_label = Path(app._status._cwd).name
        # The LIVE prompt, which is what `_approval` holds: the transcript's
        # `ApprovalBlock` is the receipt written after the answer, and the
        # title's "owes an answer" state is derived from the live one.
        approval = ApprovalPrompt("bash", "run a command")
        app._approval = approval
        app._refresh_working_activity()
        await pilot.pause()
        assert _osc_titles(writes)[-1] == f"lo ! {cwd_label}"

        await app._reload_session()
        await pilot.pause()
        assert _osc_titles(writes)[-1] == f"lo › {cwd_label}"


@pytest.mark.asyncio
async def test_unmount_restores_the_terminals_own_title(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    """The app owns the terminal and therefore the restore.

    Pinned here rather than in the pure title unit tests because the ordering is
    app-level: `on_unmount` must restore before any awaited teardown can fail
    and strand the shell under this session's title.
    """
    writes: list[str] = []
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        _spy_driver_writes(app, writes)
        app._stop_terminal_title()
        app._start_terminal_title()
        await pilot.pause()
    push_index = writes.index("\x1b[22;0t")
    pop_index = writes.index("\x1b[23;0t")
    assert push_index < pop_index


@pytest.mark.asyncio
async def test_context_command_renders_wire_schema_breakdown() -> None:
    """`/context` is the visible answer to MCP/tool-schema context cost: all
    wire categories, total/window percent, and cache-read row when present."""
    session = FakeSession()
    setattr(
        session,
        "_context_breakdown",
        {
            "instructions": 1200,
            "tool_inventory": 400,
            "tool_schemas": 6600,
            "environment": 50,
            "knowledge_mcp_goal": 900,
            "messages": 10950,
            "total": 20100,
            "context_window": 200000,
            "cache_read": 12800,
        },
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        block = app._context_block()
        assert block is not None
        listing = _renderable_plain(block.renderable)
        assert "Estimated next request" in listing
        assert "Tool schemas" in listing and "~6.6k" in listing
        assert "Messages" in listing and "~10.9k" in listing
        assert "Total" in listing and "~20.1k / 200k (10.1%)" in listing
        assert "Last cache read (exact)" in listing and "12.8k" in listing


@pytest.mark.asyncio
async def test_effective_model_change_repaints_the_band() -> None:
    """A fallback edge repaints the band's model segment with the model
    actually serving, and the recovery edge paints the selection back.

    The band is the composer's one account of "who is replying"; leaving it
    on the selected model while every request goes to a fallback is the stale
    frame this event exists to prevent.
    """
    from local_operator.tui.events import EffectiveModelChanged

    class _Effective(FakeSession):
        """A session whose effective surface follows the edge, the way the
        real one's does — the handler re-reads name/effort/window through it."""

        def __init__(self) -> None:
            super().__init__()
            # Declared on the class so assigning it below is not an implicit
            # attribute-creation (pyright), and so the property has a stable
            # value before the first edge arrives.
            self._eff: Any = None

        @property
        def effective_model(self):
            return getattr(self, "_eff", None)

        @property
        def effective_model_label(self):
            spec = getattr(self, "_eff", None)
            return f"{spec.provider}/{spec.model_id}" if spec else "test/model"

    session = _Effective()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        session._eff = SimpleNamespace(
            provider="zai",
            model_id="glm-5.3",
            display_name="GLM 5.3",
            context_window=200_000,
            reasoning_effort=None,
            reasoning_efforts=(),
            reasoning=False,
        )
        app.post_message(EffectiveModelChanged("zai", "glm-5.3", None, "provider failure", True))
        await pilot.pause()
        assert app._status is not None
        assert app._status._model_label == "zai/glm-5.3"
        assert app._status._context_window == 200_000

        # Recovery: the selection comes back, with its own metadata.
        session._eff = None
        app.post_message(
            EffectiveModelChanged("test", "model", None, "primary model recovered", False)
        )
        await pilot.pause()
        assert app._status._model_label == "test/model"


@pytest.mark.asyncio
async def test_route_independence_holds_on_a_real_store_shape(tmp_path, monkeypatch) -> None:
    """Route independence, exercised against a store shaped like a real one.

    D12's finding, taken seriously: the previous guards ran on 16- and 50-row
    fixtures whose digests were two hand-written strings, and three consecutive
    rounds shipped defects those fixtures could not express. This builds a store
    with the properties that actually produce the bug — many sessions, digests
    that share prefixes, and a query whose exact matches vanish partway through
    so the soft tier engages mid-word.

    Drives real keystrokes through `/resume` and compares two routes to the same
    visible query. It fails on `bc55a183`, where ranking read run history.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # 120 sessions: 20 that match `watch` exactly and stop there, 100 that only
    # a bounded edit-distance search can reach. Enough rows that the page window
    # matters and the recency tie-break has something to order.
    # `watch` and `watchl` both have exact hits, `watchlq` does not. So the two
    # routes reach the final query having latched the soft tier at DIFFERENT
    # points: typing straight to `watchl` never latches, while `watchlq` latches
    # and then backspaces into `watchl` carrying that state. Any rule that reads
    # run history diverges here; a rule that is a function of the query cannot.
    # Getting this wrong is why the first version of this guard passed on the
    # broken head — the fixture has to make the two routes actually differ.
    for index in range(20):
        _seed_session(tmp_path, f"aa{index:06d}", prompt=f"watchl the retention sweep {index}")
    for index in range(100):
        _seed_session(tmp_path, f"bb{index:06d}", prompt=f"wotchel batch rollup {index}")

    async def route(extra: str | None) -> list[str]:
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session), resume_factory=_resume_factory([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one(Editor).focus()
            for key in "/resume":
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, SessionPickerScreen)
            for key in "watchl" if extra is None else "watchl" + extra:
                await pilot.press(key)
                await pilot.pause()
            if extra is not None:
                for _ in extra:
                    await pilot.press("backspace")
                    await pilot.pause()
            return [row.id for row in picker.visible_rows]

    forward = await route(None)
    backspaced = await route("q")

    assert forward == backspaced, (
        "the same visible query rendered differently depending on the route: "
        f"{forward[:5]} vs {backspaced[:5]}"
    )
    # And the cursor row specifically, since that is what Enter resumes.
    assert (forward or [None])[0] == (backspaced or [None])[0]


@pytest.mark.asyncio
async def test_a_typo_is_findable_despite_an_incidental_body_hit(tmp_path, monkeypatch) -> None:
    """The recall regression that four rounds of guards did not catch (F15).

    Gating the soft tier on "the exact tiers found nothing" reads as correct and
    destroys typo search on a real store. The exact tier also admits BODY
    substring hits, and almost every typed token appears incidentally somewhere
    in a corpus of conversations — so one unrelated session mentioning the typo
    in passing silenced the tier for the whole query, and the session the user
    was actually looking for became unreachable.

    The fixture encodes exactly that: `mispelled` appears as a body substring in
    one unrelated session (the incidental hit) while the target session is
    reachable only by edit distance. A gate that stops at "some exact hit
    exists" returns the decoy and hides the target.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # The decoy: the typo appears in the BODY only, never in the row's name.
    # That distinction is the whole point — the picker names a row by its first
    # user message, so a one-line fixture would put the typo in the name and
    # test a different gate than the one that fails on a real store, where the
    # incidental hit is a word buried in a long conversation.
    # THREE decoys, not one: the gate tolerates a couple of incidental hits
    # before it treats them as a real answer, so a single-decoy fixture cannot
    # tell the shipped floor from a floor of one. Each carries the typo in the
    # BODY only — the picker names a row by its first user message, so a
    # one-line fixture would put the typo in the NAME and exercise a different
    # path than the one that fails on a real store.
    for decoy in range(3):
        _seed_session(tmp_path, f"dec{decoy:05d}", prompt=f"quarterly planning notes {decoy}")
        _append_turn(tmp_path, f"dec{decoy:05d}", "a paragraph that mentions mispelled in passing")
    # The target: contains the correct spelling only, reachable by soft match.
    for index in range(5):
        _seed_session(tmp_path, f"tgt{index:05d}", prompt=f"the misspelled word report {index}")

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), resume_factory=_resume_factory([]))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        for key in "/resume":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)

        for key in "mispelled":
            await pilot.press(key)
            await pilot.pause()

        shown = {row.id for row in picker.visible_rows}
        targets = {f"tgt{index:05d}" for index in range(5)}
        assert shown & targets, (
            "a typo'd query returned only the incidental body hit; the session the "
            f"user was looking for is unreachable. shown={sorted(shown)}"
        )


@pytest.mark.asyncio
async def test_one_incidental_name_hit_does_not_silence_the_typo_tier(
    tmp_path, monkeypatch
) -> None:
    """The floor, tested where it actually applies.

    A single exact hit on a NAME is as often incidental as deliberate: on the
    real store `spit` matches "Failover Triggering Despite Available Account",
    and treating that as a real answer hid every `split` session behind it. So
    the tier keeps running until a few precise hits accumulate.

    The sibling body-hit test cannot cover this: with no name/id match at all
    the tier runs whatever the floor is, so that fixture cannot tell a floor of
    three from a floor of one. This one puts the typo in a NAME, which is the
    only shape where the floor is the deciding factor.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # One incidental NAME hit: "despite" contains "spit".
    _seed_session(tmp_path, "dec00001", prompt="Failover triggering despite available account")
    # The sessions the user means, reachable only by edit distance from `spit`.
    for index in range(5):
        _seed_session(tmp_path, f"tgt{index:05d}", prompt=f"split the transcript window {index}")

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), resume_factory=_resume_factory([]))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        for key in "/resume":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)

        for key in "spit":
            await pilot.press(key)
            await pilot.pause()

        shown = {row.id for row in picker.visible_rows}
        targets = {f"tgt{index:05d}" for index in range(5)}
        assert shown & targets, (
            "one incidental name hit silenced the soft tier, so the sessions the "
            f"user meant are unreachable. shown={sorted(shown)}"
        )


@pytest.mark.asyncio
async def test_help_documents_the_system_clipboard_paste_key() -> None:
    """``ctrl+v`` is the ONLY way to attach a clipboard image outside cmux, and
    it has no other durable surface: it is not a slash command, so it cannot
    appear in the command table, and this app shows no key reference in the
    footer.

    That matters because the gesture users try first is unhookable. With an
    image on the pasteboard, Terminal.app and Ghostty deliver zero bytes on
    Cmd+V and beep, so a user who tried it saw nothing at all and had no way to
    discover the key that works. The paste notice teaches it once at the moment
    of failure; ``/help`` is where it is still findable an hour later.
    """
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # `RichBlock` wraps its Group in a Padding, so the rows are one level
        # in; read them rather than rendering to a frame, since what is under
        # test is the CONTENT of the help, not its layout. Cast because the
        # renderable chain is typed as the broad `RenderableType` union.
        padding = cast(Padding, app._help_block().renderable)
        group = cast(Group, padding.renderable)
        text = "\n".join(cast(Text, row).plain for row in group.renderables)

        assert "ctrl+v" in text, "/help must name the system clipboard paste key"
        line = next(row for row in text.splitlines() if "ctrl+v" in row)
        assert "clipboard" in line, f"{line!r} names the key without saying what it does"


@pytest.mark.asyncio
async def test_help_documents_the_composer_copy_key_and_its_release() -> None:
    """``ctrl+c`` is the composer's copy gesture and needs a durable surface.

    Same shape of gap as ``ctrl+v`` above: not a slash command, so the command
    table cannot carry it, and its only other advertisement is a ``welcome``
    tip on a splash that stops being displayed after the first message. The
    user who wants the key is mid-draft, which is exactly when the splash is
    gone (#169).

    The RULE is asserted, not merely the key. A live range makes every ctrl+c a
    copy, so a user holding a highlight has to know what gives the key back —
    otherwise the honest reading of a one-line "ctrl+c copies" entry is that
    the interrupt has disappeared. ``esc`` is on the row because it stops the
    agent regardless of what is highlighted.
    """
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        padding = cast(Padding, app._help_block().renderable)
        group = cast(Group, padding.renderable)
        rows = [cast(Text, row).plain for row in group.renderables]
        text = "\n".join(rows)

        assert "ctrl+c" in text, "/help must name the composer copy key"
        index = next(i for i, row in enumerate(rows) if "ctrl+c" in row)
        assert "copies" in rows[index], f"{rows[index]!r} names the key without its effect"
        # The continuation line carries the half users get wrong: how the key
        # stops being a copy, and what always interrupts.
        release = rows[index + 1]
        assert "caret" in release, f"{release!r} does not say what hands the key back"
        assert "esc" in release, f"{release!r} does not name the key that always interrupts"

        # THE WRAP CEILING IS 74, MEASURED through the painted compositor
        # rather than derived: a composed row of 74 fits and 75 wraps to two
        # lines. A wrapped tail lands at the key gutter, where it reads as
        # another key row — #402's design round 1 D2, the defect `paste_note`
        # was shortened for and which the `cmd+v` row reintroduced once.
        #
        # The rule is `terminal width - 6` (confirmed at 80/90/100 -> 74/84/94),
        # and the six cells are four reservations declared in different places:
        # `TranscriptView`'s left padding, the reserved scrollbar column, the
        # block's `SPINE_INDENT`, and `RichBlock`'s own `Padding`. Stated here
        # because a one-term derivation gives 76 and that wrong number has been
        # asserted twice (review round 2 F3, round 3 F4); see `_help_block`.
        #
        # A looser bound here would be worse than none: these rows are the ones
        # this test is advertised as guarding, so a window that excludes 75 and
        # 76 cannot fire at the boundary where the defect first appears (review
        # round 2, F3).
        for row in (rows[index], release):
            assert len(row) <= 74, f"{row!r} is {len(row)} cells and wraps at 80 columns"


@pytest.mark.asyncio
async def test_help_documents_shell_mode_and_the_composer_chords() -> None:
    """``/help`` is the in-app key reference (#385). Until these rows
    existed the README was the only channel, and a user already inside
    the TUI had no way to ask what keys exist.

    Tightly scoped: another change (#430) may also add discovery for
    ``ctrl+v`` against this same block, so this asserts CONTENT of the
    new rows rather than their exact neighbours. The wrap ceiling is
    the same 74-cell bound the copy/paste rows pin, mutation-checked
    the same way: a row padded to 75 fails it.
    """
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        padding = cast(Padding, app._help_block().renderable)
        group = cast(Group, padding.renderable)
        rows = [cast(Text, row).plain for row in group.renderables]

        # Bang-mode's marker is named because the glyph is the
        # colour-independent signal. A row that names ``!`` without
        # saying what ``$`` on the composer means leaves the same gap
        # the hue-only chevron did.
        bang = next((row for row in rows if row.lstrip().startswith("! ")), None)
        assert bang is not None, "/help must name bang-mode"
        assert "shell" in bang.lower() or "command" in bang.lower(), bang
        marker = next((row for row in rows if "$" in row and "❯" in row), None)
        assert marker is not None, "/help must name the $ / ❯ swap"

        required = (
            "option+left/right",
            "option+up/down",
            "shift+tab",
            "ctrl+l",
            "ctrl+t",
            "ctrl+g",
            "ctrl+b",
            "ctrl+pageup",
            "ctrl+r",
            "esc",
            "ctrl+d",
        )
        # Matched against the KEY GUTTER, not against the whole help text. A
        # substring test over everything passes on a mention anywhere: the
        # command table below names chords inside descriptions, so a bare
        # `"<key>" in text` stayed true with the chord's own row deleted (caught
        # by mutating a row away and watching this test stay green). These rows
        # are the block's contract; a guard that cannot see the row go missing
        # is not guarding it.
        gutters = {row[:20].strip() for row in rows}
        missing = [key for key in required if key not in gutters]
        assert not missing, f"/help is missing key rows: {missing}"

        # Same wrap ceiling as the copy/paste rows. A hanging tail at
        # column 0 reads as another key row (#402 D2). Continuations
        # (empty gutter) are included: they wrap at the same bound.
        keys = (
            "ctrl+c",
            "ctrl+v",
            "cmd+v",
            "!",
            "option+left/right",
            "option+up/down",
            "shift+tab",
            "ctrl+l",
            "ctrl+t",
            "ctrl+g",
            "ctrl+b",
            "ctrl+pageup",
            "ctrl+r",
            "esc",
            "ctrl+d",
        )
        keyed: list[str] = []
        capturing = False
        for row in rows:
            lead = row[:20].strip()
            if lead in keys or (lead == "!" and "!" in keys):
                capturing = True
                keyed.append(row)
            elif capturing and not lead:
                keyed.append(row)
            else:
                capturing = False
        assert keyed, "no key-reference rows found in /help"
        for row in keyed:
            assert len(row) <= 74, f"{row!r} is {len(row)} cells and wraps at 80 columns"


# -- the aside's keyboard scroll chord (D3) --------------------------------
#
# The card is `can_focus = False` and binds nothing, so before these chords the
# WHEEL was its only scroll gesture — and a wheel event does not exist under
# `tmux set -g mouse off`, with mouse reporting disabled, or over `screen(1)`
# and non-SGR terminals. These tests are the no-mouse half of the contract:
# every one of them drives the REAL binding through `pilot.press`, never the
# action or the panel API, because "the key reaches the app with the composer
# focused" is the whole claim and calling the action directly assumes it.
def _aside_answer_rows(rows: int, tag: str = "ANSWER") -> str:
    """`rows` uniquely identifiable lines, short enough never to wrap.

    Same device as the investigation file's `_long_answer`: "row" in these
    assertions means one source line, so the counts are not a function of the
    terminal's column count.

    A markdown LIST, and not bare lines, because the answer is rendered as
    markdown: consecutive bare lines are one paragraph to a parser, which
    reflows them into as many markers as fit a row and breaks the one-line
    one-row premise every count here rests on. A list item is its own block, so
    the source line survives as a row. Same reason `test_aside.py` feeds its
    overflow fixtures as `- line`.
    """
    return "\n".join(f"- {tag}-ROW-{index:03d}" for index in range(rows))


async def _open_long_aside(pilot, app: OperatorApp, question: str = "explain the loop"):
    """Open a real aside through the composer and settle its answer."""
    for _ in range(80):
        await pilot.pause()
        if app._session is not None:
            break
    assert app._session is not None, "the session never booted"
    app.query_one(Editor).focus()
    await pilot.pause()
    app.query_one(Editor).load_text(f"/btw {question}")
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()
    from local_operator.tui.widgets.aside_panel import AsidePanel

    panel = app.query_one(AsidePanel)
    assert panel.is_open, "the aside never opened"
    return panel


def _aside_rows_on_screen(panel) -> set[int]:
    """Which answer rows the card is painting right now."""
    blob = "\n".join(panel.render_lines_for_test())
    return {index for index in range(200) if f"ANSWER-ROW-{index:03d}" in blob}


@pytest.mark.asyncio
async def test_the_aside_chord_scrolls_without_touching_the_composer() -> None:
    """ctrl+pageup reaches rows unreachable at rest, and the caret does not move.

    The acceptance criterion the whole slice exists for: the Editor is FOCUSED
    (the aside's premise is that the user keeps typing), so the chord has to
    reach app level past a focused `TextArea` without the key landing in the
    buffer. Both halves are asserted — a chord that scrolled the card AND typed
    into the composer would pass a movement-only test.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app)

        editor = app.query_one(Editor)
        assert app.screen.focused is editor, "the aside must keep the composer focused"
        # A real draft, so "unchanged" is a claim about content and not about
        # an empty buffer staying empty.
        editor.load_text("half a thought")
        await pilot.pause()
        caret_before = editor.cursor_location

        at_rest = _aside_rows_on_screen(panel)
        assert at_rest, "the card is painting no answer rows at all"

        await pilot.press("ctrl+pageup")
        await pilot.pause()
        after = _aside_rows_on_screen(panel)

        # The point of the chord: rows that offset 0 could not show.
        assert after - at_rest, (
            "ctrl+pageup reached no new rows — "
            f"at rest {min(at_rest)}-{max(at_rest)}, after {min(after)}-{max(after)}"
        )
        # And the composer is untouched: text AND caret.
        assert editor.text == "half a thought", "the chord typed into the composer"
        assert editor.cursor_location == caret_before, "the chord moved the caret"


@pytest.mark.asyncio
async def test_the_aside_chord_returns_to_the_newest_rows() -> None:
    """ctrl+pagedown is the way back, and the pair is symmetric.

    Without this the chord is a one-way trip: a reader who paged up to check
    something has no keyboard route back to the live tail, which is the same
    gap `ctrl+end` fills for the transcript.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app)
        at_rest = _aside_rows_on_screen(panel)

        for _ in range(3):
            await pilot.press("ctrl+pageup")
            await pilot.pause()
        scrolled = _aside_rows_on_screen(panel)
        assert scrolled != at_rest, "ctrl+pageup did not move the card"

        for _ in range(12):
            await pilot.press("ctrl+pagedown")
            await pilot.pause()
        # Clamped at the tail, back where it started — not merely "somewhere
        # else". The wheel clamps rather than wrapping and the key must agree.
        assert _aside_rows_on_screen(panel) == at_rest, "ctrl+pagedown did not return home"


@pytest.mark.asyncio
async def test_the_aside_chord_pages_rather_than_creeping_by_a_row() -> None:
    """One press turns over a screenful, because there is no wheel to spin.

    A row-per-press chord is technically "reachable" and useless: the measured
    worst case is 200 rows against a 3-row budget. This is the assertion that
    stops the page size being quietly reduced to a row.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app)
        before = _aside_rows_on_screen(panel)
        await pilot.press("ctrl+pageup")
        await pilot.pause()
        after = _aside_rows_on_screen(panel)

        moved = min(before) - min(after)
        # Most of a screenful, not all of it: the panel owns the exact page
        # size and may hold a row back for its overflow marker or a pinned
        # question row, so this is bounded below rather than pinned.
        assert moved >= max(1, len(before) // 2), (
            f"one press moved {moved} rows against a {len(before)}-row window; "
            "the chord is creeping, not paging"
        )


@pytest.mark.asyncio
async def test_the_aside_chord_is_a_no_op_with_the_aside_closed() -> None:
    """Closed, the chord does nothing AND steals nothing.

    The reason it bubbles rather than sitting at `priority`. Asserted against
    the neighbouring chords it sits between in `BINDINGS`: `ctrl+up`/`ctrl+down`
    page the todos and `ctrl+home`/`ctrl+end` page the transcript, and a chord
    that swallowed a key on the common path would be a regression in a feature
    that has nothing to do with the aside.
    """
    from local_operator.tui.widgets.aside_panel import AsidePanel

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("still typing")
        await pilot.pause()
        caret_before = editor.cursor_location

        assert not app._aside_is_open()
        # No exception, no focus change, no edit to the buffer.
        await pilot.press("ctrl+pageup")
        await pilot.pause()
        await pilot.press("ctrl+pagedown")
        await pilot.pause()

        assert not app._aside_is_open(), "the chord opened an aside"
        assert editor.text == "still typing"
        assert editor.cursor_location == caret_before
        assert app.screen.focused is editor
        assert app.query_one(AsidePanel).is_open is False

        # The keys it must not have taken still work.
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app._todo_panel is not None


@pytest.mark.asyncio
async def test_the_aside_chord_is_silent_when_there_is_nothing_to_scroll() -> None:
    """Open but fully on screen: the press is a harmless miss, not a notice.

    The `_scroll_todos` rule. An aside stays open across whole conversations,
    so a receipt on every stray press would append warning rows to a transcript
    from a card whose title says nothing here joins it — and they would be
    drawn BEHIND the card, so the user would find them only after dismissing
    the thing they were reading. Silence is the deliberate choice.
    """
    from tests.unit.tui.test_aside import AsideSession

    # Two rows against a 40-row screen: nothing is ever hidden.
    session = AsideSession(answer="one line.")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app, "short one")
        before = panel.render_lines_for_test()

        await pilot.press("ctrl+pageup")
        await pilot.pause()
        await pilot.press("ctrl+pagedown")
        await pilot.pause()

        after = panel.render_lines_for_test()
        assert after == before, "a no-op press repainted the card"
        # And it did not invent an overflow marker for a card with no overflow.
        # Pinned on the marker's own noun rather than on the word "scroll",
        # which the card's hint row legitimately carries.
        assert not any("earlier" in line for line in after), after


# -- the aside's copy key (Option E) ---------------------------------------
#
# `omp` ships copy as a first-class action beside branch and dismiss, and this
# is the port of that idea rather than of its keybinding: its bare `c` cannot
# work here (see `ASIDE_COPY_KEY` — Textual gives the focused TextArea the
# character first, measured). What ports is the CONTRACT: the full answer off
# the model, never the painted rows, and nothing written to the session.
@pytest.mark.asyncio
async def test_the_aside_copy_key_takes_the_whole_answer_not_the_painted_rows() -> None:
    """The acceptance case, and the reason the payload is not read off screen.

    The card paints a WINDOW — that is the defect this branch exists to fix —
    so a copy sourced from the frame would hand back the same fragment the user
    is complaining they cannot see past. 200 rows are asserted against the
    clipboard while the card is painting a fraction of them.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app, "explain the whole loop")
        app._clipboard = ""

        painted = _aside_rows_on_screen(panel)
        assert len(painted) < 200, "the card fits everything; this proves nothing"

        await pilot.press("ctrl+r")
        await pilot.pause()

        copied = app._clipboard
        got = {index for index in range(200) if f"ANSWER-ROW-{index:03d}" in copied}
        assert got == set(range(200)), f"clipboard is missing {sorted(set(range(200)) - got)[:5]}…"
        # The question came with it, so the exchange reads as one.
        assert "explain the whole loop" in copied
        # And NO card chrome: not the title, not the rule, not the hint row.
        # The `Chrome.ALLOW_SELECT` rule applied to this path.
        assert "─" not in copied, copied[:200]
        assert "esc" not in copied.lower(), copied[:200]


@pytest.mark.asyncio
async def test_the_aside_copy_key_writes_nothing_to_the_session() -> None:
    """The off-the-record contract, which is the whole point of the key.

    `^f` rescues the text by writing it permanently into the context AND the
    transcript. This rescues it without touching either — the clipboard is
    outside the session. Asserted against the message list itself, not against
    a notice.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(50))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_long_aside(pilot, app)
        app._clipboard = ""

        before = list(session._history)
        forked_before = list(session.forked)

        await pilot.press("ctrl+r")
        await pilot.pause()

        assert app._clipboard, "nothing was copied, so this proves nothing"
        assert list(session._history) == before, "copy wrote to the conversation"
        assert list(session.forked) == forked_before, "copy forked the exchange"


@pytest.mark.asyncio
async def test_the_aside_copy_key_works_while_the_answer_is_still_streaming() -> None:
    """Streaming is when the key matters MOST, because `^f` is refused there.

    `_aside_can_fork` returns False while the session streams — splicing a
    message into a live batch produces a request no provider accepts. That is a
    constraint on WRITING to the session, and copying does not write, so this
    path must stay open. Routing the payload through `fork_messages()` would
    have failed exactly here: it is gated on `AsideTurn.forkable`, which needs
    `state == "done"`, so mid-stream it returns an empty list.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer="")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app, "why is it slow")

        # A turn mid-flight: asked, partially answered, not settled.
        session.streaming = True
        generation = panel.ask("and what about startup")
        panel.append_answer(generation, "PARTIAL-TEXT so far")
        await pilot.pause()
        assert panel.turns[-1].state == "running"
        assert app._aside_can_fork() is False, "precondition: ^f is refused here"

        app._clipboard = ""
        await pilot.press("ctrl+r")
        await pilot.pause()

        assert "PARTIAL-TEXT so far" in app._clipboard, repr(app._clipboard)
        assert "and what about startup" in app._clipboard


@pytest.mark.asyncio
async def test_the_aside_copy_key_is_inert_outside_the_aside() -> None:
    """Closed, it copies nothing and — the part that matters — types nothing.

    `ctrl+r` was chosen because `TextArea` does not bind it, so the composer
    keeps every editing key it had. A regression that turned this into a
    character would be invisible until someone typed it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("a draft I am writing")
        await pilot.pause()
        app._clipboard = ""

        assert not app._aside_is_open()
        await pilot.press("ctrl+r")
        await pilot.pause()

        assert app._clipboard == "", "copied something with no aside open"
        assert editor.text == "a draft I am writing", "ctrl+r reached the buffer"


@pytest.mark.asyncio
async def test_the_aside_copy_key_leaves_out_a_turn_that_failed() -> None:
    """A failed turn's partial text is NOT the model's answer, so it is not copied.

    `fail_answer` keeps whatever streamed before the failure in `.answer` and
    puts the reason in a separate `.error` field, precisely so the fragment can
    never be handed on as though the model had said it — the same rule
    `fork_messages` applies when it drops non-`forkable` turns ("half an
    exchange is not one").

    A filter written against "the answer is non-empty" rather than against the
    turn's STATE passes every other test in this file and still leaks here,
    which is why this case is pinned at the KEY level: it asserts what the user
    actually gets on the clipboard, not what a helper returns.

    The running turn in the same card must still come through. That is the
    distinction this test exists to hold: exclude error and cancelled, include
    running — `^f` is refused mid-stream, so copy is the only way out there.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(20, tag="GOOD"))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app, "the settled question")

        # A turn that streamed some text and then failed.
        failed = panel.ask("the doomed question")
        panel.append_answer(failed, "HALF-AN-ANSWER before it broke")
        panel.fail_answer(failed, "provider exploded")
        await pilot.pause()
        assert panel.turns[-1].state == "error"
        assert panel.turns[-1].answer, "precondition: the fragment survived on the turn"

        app._clipboard = ""
        await pilot.press("ctrl+r")
        await pilot.pause()

        copied = app._clipboard
        assert "GOOD-ROW-000" in copied, "the settled turn should still be copied"
        assert "HALF-AN-ANSWER" not in copied, f"a failed turn's fragment leaked: {copied!r}"
        assert "the doomed question" not in copied, copied
        # The failure text itself is this app's prose, never the model's.
        assert "provider exploded" not in copied, copied


@pytest.mark.asyncio
async def test_the_aside_copy_key_leaves_out_a_cancelled_turn() -> None:
    """Cancelled is the other half of the same rule, and it is a separate state.

    A filter that special-cased only ``error`` would pass the failure test above
    and still leak here.
    """
    from tests.unit.tui.test_aside import AsideSession

    session = AsideSession(answer=_aside_answer_rows(20, tag="GOOD"))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        panel = await _open_long_aside(pilot, app, "the settled question")

        abandoned = panel.ask("the abandoned question")
        panel.append_answer(abandoned, "ABANDONED-TEXT")
        panel.turns  # the accessor copies, so mutate the turn the card holds
        panel._turns[-1].state = "cancelled"
        await pilot.pause()

        app._clipboard = ""
        await pilot.press("ctrl+r")
        await pilot.pause()

        copied = app._clipboard
        assert "GOOD-ROW-000" in copied, "the settled turn should still be copied"
        assert "ABANDONED-TEXT" not in copied, f"a cancelled turn leaked: {copied!r}"


class _TTLBoundController(_AccessController):
    """A provider whose listing cache behaves the way the real one does.

    Serves whatever the cache holds, and re-reads upstream only when the caller
    asks for a document younger than the cache -- which is the contract
    ``model/catalogue.py`` implements on disk and the reason ``PICKER_TTL_S``
    exists. ``upstream`` is what Anthropic would answer if asked right now.
    """

    #: How old the cached document is when the picker opens. Between the 15-minute
    #: picker TTL and the 24h default, so it is the ONE window where the two
    #: disagree: the old default served this happily for the rest of the day.
    CACHE_AGE_S = 3 * 60 * 60

    def __init__(self) -> None:
        super().__init__(stored=("anthropic",))
        # Written before `claude-fable-5-1` shipped.
        self.cached = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5"]
        self.upstream = ["claude-fable-5-1"] + self.cached
        self.fetches = 0

    def login_providers(self):
        return [_FakeDef("anthropic", "Anthropic", None, ("claude",))]

    def _entries(self, ids):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="anthropic",
                model_id=i,
                label=i,
                context_window=1_000_000,
                input_price=5.0,
                output_price=25.0,
                connected=True,
            )
            for i in ids
        ]

    def static_catalogue(self):
        return self._entries(self.cached)

    async def live_catalogue(self, *, ttl_s=None):
        if ttl_s is not None and ttl_s < self.CACHE_AGE_S:
            self.fetches += 1
            self.cached = list(self.upstream)
            return self._entries(self.cached), {"anthropic": "ok"}
        return self._entries(self.cached), {"anthropic": "cached"}


@pytest.mark.asyncio
async def test_the_picker_shows_a_model_released_since_the_listing_was_cached() -> None:
    """The reported defect, at the surface the user actually touches.

    `claude-fable-5-1` shipped after the cached listing was written. With the 24h
    default the document was still fresh, so the live-refresh worker asked for the
    cache, got yesterday's answer, and the model was unreachable through `/model`
    all day. The picker now asks for a fifteen-minute document.
    """
    ctrl = _TTLBoundController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.model_id for row in picker.rows()}
    assert "claude-fable-5-1" in offered, offered
    assert ctrl.fetches == 1, "and it cost exactly one live listing"


@pytest.mark.asyncio
async def test_a_bang_command_carries_the_conversation_title_to_its_tools() -> None:
    """Bang mode builds its OWN ToolContext, outside any turn.

    ``Session._build_tool_context`` is what carries the title to every ordinary
    tool call, and this path deliberately does not use it (there is no turn to
    build one for). It therefore has to carry the title itself, or every
    display-only consumer — the browser tab group above all — sees an unnamed
    session for the whole of a `! cmd`.

    The provider is asserted alongside the snapshot because a `! cmd` can
    outlive the naming errand that titles the conversation.
    """
    session = FakeSession()
    session.set_conversation_name("Debug browser extension port binding issue")
    seen: list[Any] = []

    async def fake_execute_bash(tool_call_id, args, signal, on_update, context):
        seen.append(context)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bash",
            content=[TextContent(text="exit code: 0")],
        )

    app = OperatorApp(lambda: _factory(session))
    with patch("local_operator.tools.builtin.execute_bash", fake_execute_bash):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.press("!")
            for key in "echo hi":
                await pilot.press("space" if key == " " else key)
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause()
                if seen:
                    break

    assert seen, "the bang command never reached a tool"
    context = seen[-1]
    assert context.session_name == "Debug browser extension port binding issue"
    # Live too: a title that lands while a long `! cmd` runs must still reach
    # the tab group, exactly as it does on the per-turn path.
    session.set_conversation_name("Renamed mid-command")
    assert context.session_name_provider is not None
    assert context.session_name_provider() == "Renamed mid-command"
