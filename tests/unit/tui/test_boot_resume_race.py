"""A ``/resume`` made while the TUI is still booting must survive boot finishing.

The defect (architect's in-process reproduction on d0301a22fba2): ``_boot_session``
fenced only against SIDEBAR navigation (``navigation_generation``) before adopting
its freshly built session. ``/resume <id>`` never bumps that generation, so a
resume that LANDED while boot was still building was then overwritten by boot's
session — the user saw a fresh conversation and their next prompt went there.

The fix is an adoption epoch: ``_adopt_session`` (the one place a session becomes
current) bumps ``_adoption_epoch``, and boot steps aside when it moved. These
tests drive the real slash seam (``/resume s1`` through ``_run_slash_command``)
with only the runtime lookup and ``AttachedSession.connect`` stubbed, and a real
offline ``AttachedSession`` as the viewer — the rig the architect used.

They also pin the requester's product decision for what boot does with a prompt
it would have sent when it steps aside (both step-aside branches): never sent,
never silently dropped — an unsent draft in an empty composer, or quoted in the
notice when the user already has a draft.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.types import SessionRecord
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_boot_layout import FakeSession
from tests.unit.tui.waiting import settle

RETURNED = "Your startup prompt wasn't sent — it's in the composer."
KEPT_PREFIX = "Your startup prompt wasn't sent: "


def _record() -> SessionRecord:
    return SessionRecord(
        pid=90909,
        kind="tui",
        session_id="s1",
        conversation_name="the remote",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
    )


def _offline_viewer(config_root: Path) -> Any:
    """A real ``AttachedSession`` that never dials: the shape CI's witness builds."""
    import local_operator.session.attached as attached_module
    from local_operator.session.attached import AttachedSession
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
        FrontendStateStore,
    )

    async def no_takeover() -> Any:
        raise attached_module.SessionLeaseHeldError(config_root / "s1", 123)

    viewer = AttachedSession(
        config_dir=config_root, session_id="s1", takeover_factory=no_takeover, surface="desktop"
    )
    model = FrontendModelSpec(provider="test", model_id="model")
    viewer._frontend_store = FrontendStateStore(
        FrontendSessionState(
            session_id="s1", epoch="test", selected_model=model, effective_model=model
        )
    )
    return viewer


class _Rig:
    """Boot held on ``gate``; ``/resume s1`` resolves to ``viewer`` (or cold)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, live: bool) -> None:
        import local_operator.resume as resume_mod

        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        self.gate = asyncio.Event()
        self.connect_gate = asyncio.Event()
        self.connect_gate.set()
        self.boot = FakeSession()
        self.events: list[str] = []
        self.submitted: list[str] = []
        if live:
            self.viewer: Any = _offline_viewer(tmp_path)
        else:

            class Cold(FakeSession):
                @property
                def session_id(self) -> str:
                    return "s1"

            self.viewer = Cold()
        self._spy_dispose(self.boot, "boot")
        self._spy_dispose(self.viewer, "viewer")

        async def factory() -> Any:
            await self.gate.wait()
            return self.boot

        async def resume_factory(_sid: str) -> Any:
            if live:
                raise AssertionError("takeover not expected on the live path")
            return self.viewer

        async def connect(*_a: Any, **_k: Any) -> Any:
            await self.connect_gate.wait()
            return self.viewer

        monkeypatch.setattr(
            resume_mod,
            "live_runtime_pid",
            (lambda root, sid: 90909) if live else (lambda r, s: None),
        )
        monkeypatch.setattr(
            "local_operator.mobile.attach_client.find_runtime_record",
            lambda root, sid: (_record(), 90909),
        )
        monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
        self.app = OperatorApp(factory)
        self.app._resume_factory = resume_factory

        async def no_retirement(_s: Any) -> None:
            return None

        monkeypatch.setattr(self.app, "_retire_unused_runtime", no_retirement)
        # Every door a prompt could leave through: the composer's own submit
        # and the command path `/aida` uses. Neither may fire for a stashed
        # prompt once boot has stepped aside.
        monkeypatch.setattr(
            self.app, "_submit_prompt", lambda text, *a, **k: self.submitted.append(text)
        )
        monkeypatch.setattr(
            self.app, "_submit_command_prompt", lambda text, *a, **k: self.submitted.append(text)
        )

        # The notice is what the user reads; record the call AND let it paint
        # so the rendered frame and the assertion see the same thing.
        self.notice_log: list[str] = []
        real_notice = self.app._system_notice

        def notice(body: str, kind: Any = "info") -> None:
            self.notice_log.append(body)
            real_notice(body, kind)

        monkeypatch.setattr(self.app, "_system_notice", notice)

    def _spy_dispose(self, session: Any, name: str) -> None:
        real = session.dispose

        async def spy() -> None:
            # Record whether the app had adopted it at the moment of disposal:
            # boot's session must go while still UNADOPTED, never after
            # replacing the viewer.
            self.events.append(f"{name}.dispose adopted={self.app._session is session}")
            await real()

        session.dispose = spy  # type: ignore[method-assign]

    def notices(self) -> list[str]:
        return [body for body in self.notice_log if "startup prompt" in body]


async def _boot_released_after_resume(rig: _Rig, pilot: Any) -> None:
    app = rig.app
    await settle(pilot, lambda: app._session is None, what="boot held")
    app._run_slash_command("/resume s1")
    # No `workers=` here on purpose: the resume transition shares the
    # "session" group with the boot worker, which is HELD, so the group can
    # never drain at this point. The predicate on `_session` is the signal.
    await settle(pilot, lambda: app._session is rig.viewer, what="/resume landed")
    rig.gate.set()
    # Either outcome ends the wait, so the defect reads as a failed ASSERTION
    # on who is current rather than as a timeout: before the fix boot adopted
    # its own session here and replaced the resumed one.
    await settle(
        pilot,
        lambda: app._session is rig.boot or any(e.startswith("boot.dispose") for e in rig.events),
        what="boot finished",
        workers=("session",),
    )


@pytest.mark.asyncio
async def test_resume_during_boot_survives_boot_finishing(monkeypatch, tmp_path) -> None:
    rig = _Rig(monkeypatch, tmp_path, live=True)
    async with rig.app.run_test(size=(100, 30)) as pilot:
        await _boot_released_after_resume(rig, pilot)
        assert rig.app._session is rig.viewer
        assert rig.app._adopted_session_id == "s1"
        assert rig.events == ["boot.dispose adopted=False"]
    # The viewer is disposed only by quit, never by boot.
    assert rig.events[-1].startswith("viewer.dispose")
    assert rig.submitted == []


@pytest.mark.asyncio
async def test_resume_still_dialling_when_boot_finishes_replaces_boot(
    monkeypatch, tmp_path
) -> None:
    """Ordering control: an UNLANDED resume does not make boot step aside."""
    rig = _Rig(monkeypatch, tmp_path, live=True)
    rig.connect_gate.clear()
    app = rig.app
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot, lambda: app._session is None, what="boot held")
        app._run_slash_command("/resume s1")
        rig.gate.set()
        await settle(pilot, lambda: app._session is rig.boot, what="boot adopted first")
        assert rig.events == []
        rig.connect_gate.set()
        await settle(
            pilot,
            lambda: app._session is rig.viewer and bool(rig.events),
            what="viewer replaced boot",
            workers=("session",),
        )
        assert app._adopted_session_id == "s1"
        # Boot WAS current when the resume's own swap retired it: proof that
        # boot did not step aside for a resume that had not landed yet.
        assert rig.events == ["boot.dispose adopted=True"]


@pytest.mark.asyncio
async def test_cold_resume_during_boot_survives_boot_finishing(monkeypatch, tmp_path) -> None:
    rig = _Rig(monkeypatch, tmp_path, live=False)
    async with rig.app.run_test(size=(100, 30)) as pilot:
        await _boot_released_after_resume(rig, pilot)
        assert rig.app._session is rig.viewer
        assert rig.app._adopted_session_id == "s1"
        assert rig.events == ["boot.dispose adopted=False"]


def _stash_boot_prompt(rig: _Rig, tmp_path: Path, text: str) -> None:
    from local_operator.fork import write_boot_prompt

    session_dir = tmp_path / "sessions" / rig.boot.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    write_boot_prompt(session_dir, text)


async def _step_aside(rig: _Rig, pilot: Any, branch: str) -> None:
    if branch == "epoch":
        await _boot_released_after_resume(rig, pilot)
        return
    # The sidebar branch: a navigation committed while boot was building is
    # what `SessionNavigation.select` records as a generation bump.
    app = rig.app
    await settle(pilot, lambda: app._session is None, what="boot held")
    app._sidebar_navigation.generation += 1
    rig.gate.set()
    await settle(
        pilot,
        lambda: any(e.startswith("boot.dispose") for e in rig.events),
        what="boot stepped aside",
        workers=("session",),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["epoch", "sidebar"])
async def test_stepped_aside_boot_prompt_becomes_unsent_draft(
    monkeypatch, tmp_path, branch
) -> None:
    rig = _Rig(monkeypatch, tmp_path, live=True)
    _stash_boot_prompt(rig, tmp_path, "summarise the release notes")
    async with rig.app.run_test(size=(100, 30)) as pilot:
        await _step_aside(rig, pilot, branch)
        await settle(pilot, lambda: bool(rig.notices()), what="notice shown")
        assert rig.app._editor().text == "summarise the release notes"
        assert rig.notices() == [RETURNED]
        assert rig.submitted == []
        assert rig.boot.prompts == []
    # Consumed, so a later resume of that conversation cannot auto-run it.
    assert not (tmp_path / "sessions" / rig.boot.session_id / "boot-prompt.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["epoch", "sidebar"])
async def test_stepped_aside_boot_prompt_keeps_users_draft(monkeypatch, tmp_path, branch) -> None:
    rig = _Rig(monkeypatch, tmp_path, live=True)
    _stash_boot_prompt(rig, tmp_path, "summarise the release notes")
    async with rig.app.run_test(size=(100, 30)) as pilot:
        app = rig.app
        await settle(pilot, lambda: app._session is None, what="boot held")
        if branch == "epoch":
            app._run_slash_command("/resume s1")
            await settle(pilot, lambda: app._session is rig.viewer, what="/resume landed")
        else:
            app._sidebar_navigation.generation += 1
        app._editor().load_text("my own words")
        rig.gate.set()
        await settle(pilot, lambda: bool(rig.notices()), what="notice shown", workers=("session",))
        assert app._editor().text == "my own words"
        assert rig.notices() == [KEPT_PREFIX + "summarise the release notes"]
        assert rig.submitted == []
        assert rig.boot.prompts == []


@pytest.mark.asyncio
async def test_stepped_aside_aida_stash_for_boot_session_is_returned(monkeypatch, tmp_path) -> None:
    """Her stash for boot's OWN session rides the same rule; another id's stays."""
    rig = _Rig(monkeypatch, tmp_path, live=True)
    rig.app._pending_aida_prompt = (rig.boot.session_id, "plan my day", None)
    async with rig.app.run_test(size=(100, 30)) as pilot:
        await _step_aside(rig, pilot, "sidebar")
        await settle(pilot, lambda: bool(rig.notices()), what="notice shown")
        assert rig.app._editor().text == "plan my day"
        assert rig.app._pending_aida_prompt is None
        assert rig.submitted == []


@pytest.mark.asyncio
async def test_stepped_aside_aida_stash_for_other_session_is_left_alone(
    monkeypatch, tmp_path
) -> None:
    """A stash keyed to ANOTHER id belongs to a transition still in flight.

    The other half of the docstring above: boot's step-aside must neither hand
    it back (it is not boot's prompt) nor clear it (its own transition would
    then land without it).
    """
    rig = _Rig(monkeypatch, tmp_path, live=True)
    stash = ("someone-else", "plan my day", None)
    rig.app._pending_aida_prompt = stash
    async with rig.app.run_test(size=(100, 30)) as pilot:
        await _step_aside(rig, pilot, "sidebar")
        await pilot.pause()
        assert rig.app._pending_aida_prompt == stash
        assert rig.app._editor().text == ""
        assert rig.notices() == []
        assert rig.submitted == []


@pytest.mark.asyncio
async def test_kept_draft_notice_quotes_long_prompt_in_full(monkeypatch, tmp_path) -> None:
    """The notice is the prompt's ONLY remaining copy, so it is never truncated.

    With the user's draft in the composer and the sidecar consumed, any cap on
    the quote would lose the prompt's tail for good — the silent drop the
    product rule forbids (design D1 on PR #3787). ~3000 characters over
    several paragraphs, so a cap anywhere near the old 500 shows up.
    """
    paragraphs = [
        f"Paragraph {n}: " + " ".join(f"word{n}-{i}" for i in range(40)) for n in range(1, 9)
    ]
    paragraphs.append("LAST LINE: ship it only after the smoke test passes.")
    prompt = "\n\n".join(paragraphs)
    assert len(prompt) > 2500
    rig = _Rig(monkeypatch, tmp_path, live=True)
    _stash_boot_prompt(rig, tmp_path, prompt)
    async with rig.app.run_test(size=(100, 30)) as pilot:
        app = rig.app
        await settle(pilot, lambda: app._session is None, what="boot held")
        app._sidebar_navigation.generation += 1
        app._editor().load_text("my own words")
        rig.gate.set()
        await settle(pilot, lambda: bool(rig.notices()), what="notice shown", workers=("session",))
        assert app._editor().text == "my own words"
        assert rig.notices() == [KEPT_PREFIX + prompt]
        assert rig.notices()[0].endswith("LAST LINE: ship it only after the smoke test passes.")
        assert rig.submitted == []


# The in-process swap (`_reload_session`) has its own sidebar step-aside, and
# the requester's decision covers every branch of that class: the prompt its
# replacement carried (a switch-mode `/fork <message>` sidecar, or an `/aida
# <request>` stash riding a `/resume`) is handed back, never sent and never
# dropped. Before the fix that branch disposed the replacement bare.


class _Replacement(FakeSession):
    @property
    def session_id(self) -> str:
        return "s2"


async def _reload_stepped_aside(
    rig: _Rig, pilot: Any, *, draft: str | None = None, aida: str | None = None
) -> _Replacement:
    """Boot normally, start a swap, click the sidebar while it builds, release."""
    app = rig.app
    rig.gate.set()
    await settle(pilot, lambda: app._session is rig.boot, what="boot adopted", workers=("session",))
    if aida is not None:
        # Armed AFTER boot adopted, the way `/aida <request>` stashes it just
        # before its `/resume`: boot's own adoption consumes any earlier stash.
        app._pending_aida_prompt = ("s2", aida, None)
    replacement = _Replacement()
    rig._spy_dispose(replacement, "replacement")
    building = asyncio.Event()
    release = asyncio.Event()

    async def swap_factory() -> Any:
        building.set()
        await release.wait()
        return replacement

    app._session_factory = swap_factory
    app._run_session_transition(app._reload_session())
    await settle(pilot, building.is_set, what="reload building")
    app._sidebar_navigation.generation += 1
    if draft is not None:
        app._editor().load_text(draft)
    release.set()
    await settle(
        pilot,
        lambda: any(e.startswith("replacement.dispose") for e in rig.events),
        what="reload stepped aside",
        workers=("session",),
    )
    return replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("stash", ["fork", "aida"])
async def test_reload_step_aside_returns_prompt_as_unsent_draft(
    monkeypatch, tmp_path, stash
) -> None:
    rig = _Rig(monkeypatch, tmp_path, live=True)
    sidecar = tmp_path / "sessions" / "s2" / "boot-prompt.json"
    if stash == "fork":
        from local_operator.fork import write_boot_prompt

        sidecar.parent.mkdir(parents=True, exist_ok=True)
        write_boot_prompt(sidecar.parent, "summarise the release notes")
    async with rig.app.run_test(size=(100, 30)) as pilot:
        replacement = await _reload_stepped_aside(
            rig, pilot, aida="summarise the release notes" if stash == "aida" else None
        )
        await settle(pilot, lambda: bool(rig.notices()), what="notice shown")
        assert rig.app._editor().text == "summarise the release notes"
        assert rig.notices() == [RETURNED]
        assert rig.submitted == []
        assert replacement.prompts == []
        assert rig.boot.prompts == []
        assert rig.app._pending_aida_prompt is None
    assert not sidecar.exists()


@pytest.mark.asyncio
async def test_reload_step_aside_keeps_users_draft_and_quotes_prompt(monkeypatch, tmp_path) -> None:
    from local_operator.fork import write_boot_prompt

    rig = _Rig(monkeypatch, tmp_path, live=True)
    session_dir = tmp_path / "sessions" / "s2"
    session_dir.mkdir(parents=True, exist_ok=True)
    write_boot_prompt(session_dir, "summarise the release notes")
    async with rig.app.run_test(size=(100, 30)) as pilot:
        replacement = await _reload_stepped_aside(rig, pilot, draft="my own words")
        await settle(pilot, lambda: bool(rig.notices()), what="notice shown")
        assert rig.app._editor().text == "my own words"
        assert rig.notices() == [KEPT_PREFIX + "summarise the release notes"]
        assert rig.submitted == []
        assert replacement.prompts == []
        assert rig.boot.prompts == []
