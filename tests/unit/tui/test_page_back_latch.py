"""Edge-triggered page-back latch, pinned directly and deterministically.

The cascade the operator reported ("I scroll to the top and it just starts
loading chunks one after another") lived in the TRIGGER SHAPE, not in any one
gesture: both page-back paths tested the scroll offset as a LEVEL (``at the
top ⇒ load``) instead of as an EDGE (``arrived at the top ⇒ load once``).
Anything that re-fired the scroll watch while the reader sat at the top — the
anchor restore a prepend performs, the settle frames that follow it, a
resize — mounted another page, and a wheel still in motion walked the whole
history.

Driving that through Textual animation is flaky by construction (the loop is
frame-timed), so these tests pin the STATE MACHINE directly: a watch firing
while parked at the top loads nothing; a gesture re-arms; the next crossing
loads exactly one. The end-to-end arrival behaviour is covered in
``test_resume_render.py`` and ``test_subagent_view.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.session.transcript import Transcript
from local_operator.tui.app import RESUME_PAGE_MESSAGES, OperatorApp
from local_operator.tui.widgets.subagent_view import SubagentView
from local_operator.tui.widgets.transcript import TranscriptView

from .test_app_pilot import FakeSession as _PilotFakeSession
from .test_app_pilot import _factory
from .test_band_panels import FakeSession, _async_factory, _fake_jobs
from .test_subagent_view import _job_with, _open, _wait_history


@pytest.mark.asyncio
async def test_subagent_latch_ignores_watch_firings_while_parked_at_top(
    tmp_path, monkeypatch
) -> None:
    """A scroll watch firing at the top loads ONCE, then nothing.

    Pre-fix the trigger was the level ``scroll_y <= 1`` with no latch, so
    every firing at the top requested another page — the loop the operator
    saw. Post-fix the crossing consumes the latch and only a GESTURE
    (``_note_history_gesture``) re-arms it, which settle frames never are.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(320):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        reads = {"n": 0}
        original = SubagentView.__module__
        import importlib

        module = importlib.import_module(original)
        real_read = module.read_transcript_page

        def counting_read(*args: Any, **kwargs: Any) -> Any:
            reads["n"] += 1
            return real_read(*args, **kwargs)

        monkeypatch.setattr(original + ".read_transcript_page", counting_read)
        reads["n"] = 0

        # FIRST crossing at the top: one load, latch consumed.
        view._body.scroll_to(y=0, animate=False)
        await _wait_history(pilot, view)
        assert reads["n"] == 1

        # The reader stays parked at the top and the watch keeps firing —
        # the settle frames, a resize, the restore itself. None of these
        # are gestures, so none may re-arm the latch.
        for _ in range(10):
            view._scroll_changed()
        for _ in range(5):
            await pilot.pause()
            view._scroll_changed()
        assert reads["n"] == 1, "watch firings while parked at top must not load"

        # A GESTURE re-arms; the reader leaves and comes back: one more page.
        view._note_history_gesture()
        view._body.scroll_to(y=40, animate=False)
        await pilot.pause()
        view._body.scroll_to(y=0, animate=False)
        await _wait_history(pilot, view)
        assert reads["n"] == 2
        # And it holds again: one page per arrival, never a cascade.
        for _ in range(10):
            view._scroll_changed()
        assert reads["n"] == 2


@pytest.mark.asyncio
async def test_subagent_latch_does_not_rearm_while_a_page_is_in_flight(
    tmp_path, monkeypatch
) -> None:
    """The notches of the SAME wheel that requested a page cannot re-arm.

    A drag's momentum tail arrives while the first page is still mounting;
    re-arming on it made one gesture request a second page the moment the
    first landed. The re-arm is suppressed while ``_history_loading`` is set.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(320):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        reads = {"n": 0}
        import importlib

        module = importlib.import_module(SubagentView.__module__)
        real_read = module.read_transcript_page

        def counting_read(*args: Any, **kwargs: Any) -> Any:
            reads["n"] += 1
            return real_read(*args, **kwargs)

        monkeypatch.setattr(SubagentView.__module__ + ".read_transcript_page", counting_read)
        reads["n"] = 0

        # Cross into the top: the load starts (loading=True) but has not
        # landed. The worker read runs on a thread, so the window is held
        # open manually here exactly as a real drag's momentum tail sees it:
        # the latch's suppression is what is under test, not the scheduling.
        view._body.scroll_to(y=0, animate=False)
        await pilot.pause()
        await _wait_history(pilot, view)
        assert reads["n"] == 1
        view._history_loading = True  # a page is in flight, as during a mount
        # The momentum tail: gestures arriving while that page is mounting.
        for _ in range(5):
            view._note_history_gesture()
            view._scroll_changed()
        view._history_loading = False
        for _ in range(20):
            await pilot.pause()
        assert reads["n"] == 1, "in-flight gestures must not arm a second page"


@pytest.mark.asyncio
async def test_resume_latch_ignores_watch_firings_while_parked_in_zone() -> None:
    """The parent view's page check mounts ONCE per arrival, then holds.

    Pre-fix the check was the level ``offset <= trigger rows`` gated only by
    ``_resume_paging``; once a mount's settle released that gate with the
    reader still parked inside the zone, the next watch firing mounted again.
    Post-fix the latch consumes on the mount and only a gesture re-arms.
    """
    from tests.unit.tui.test_resume_render import _history, _wait_for_resume

    session = _PilotFakeSession()
    session._history = _history(200)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        before = len(app._resume_pending_head)
        assert before

        # Arrive at the top through the real watch path: one page.
        view.scroll_to(y=0, animate=False)
        for _ in range(20):
            await pilot.pause()
        assert len(app._resume_pending_head) == before - RESUME_PAGE_MESSAGES

        # Parked in the zone, nothing in flight: repeated checks mount nothing.
        for _ in range(10):
            app._check_resume_page()
        for _ in range(10):
            await pilot.pause()
            app._check_resume_page()
        assert len(app._resume_pending_head) == before - RESUME_PAGE_MESSAGES

        # A gesture re-arms; leaving and returning mounts exactly one more.
        app._resume_in_zone = True  # what a real gesture does via the hook
        view.scroll_to(y=40, animate=False)
        await pilot.pause()
        view.scroll_to(y=0, animate=False)
        for _ in range(20):
            await pilot.pause()
        assert len(app._resume_pending_head) == before - 2 * RESUME_PAGE_MESSAGES
        for _ in range(10):
            app._check_resume_page()
        assert len(app._resume_pending_head) == before - 2 * RESUME_PAGE_MESSAGES


@pytest.mark.asyncio
async def test_finish_history_mount_refuses_a_stale_generation(tmp_path) -> None:
    """A settle callback landing after a retarget must not touch the new job.

    Review round 2, M4: `_finish_history_mount` was the only history
    completion callback without a generation guard. A page's insert-settle can
    land after `show()` has retargeted the page to another job (the generation
    bumped, that job's own initial read in flight); an unguarded clear took
    down the NEW job's `_history_loading` mid-read — hint stuck on
    "loading earlier…", latch un-suppressed. Every sibling completion path
    (`_apply_history_page`, `_finish_history_error`,
    `_finish_history_unavailable`) guards the same way; this pins the fourth.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(230):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    other = Transcript(tmp_path / "other")
    for index in range(230):
        await other.append_message(Message.assistant(f"other {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    # The comms must answer PER JOB. A single-directory stub made the app's own
    # refresh of `job-other` arrive carrying the FIRST job's directory, which
    # takes `show`'s changed-directory branch and fires a third `_reset_history`
    # mid-test — clearing the very `_history_loading` this test holds open and
    # failing the assertion for a reason that has nothing to do with the guard
    # (a test-only race, seen ~1 in 8 under load and on CI's 3.13 leg).
    directories = {"job-other": other.directory}
    session._subagent_comms = type(
        "Comms",
        (),
        {"session_dir_of": (lambda self, job_id: directories.get(job_id, transcript.directory))},
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        stale = view._history_generation
        # Retarget to a different job: bumps the generation and starts that
        # job's own initial read, so `_history_loading` is legitimately True.
        view.show(
            job_id="job-other",
            label="other",
            status="running",
            queued=False,
            elapsed="1s",
            outcome="",
            events=[],
            transcript_directory=str(other.directory),
        )
        await _wait_history(pilot, view)
        assert view._history_generation != stale
        # Represent the mid-read state the guard exists to protect. The real
        # window is thread-scheduling-wide and completes in microseconds, so
        # the test holds the flag open exactly as a slow disk read would (the
        # same technique the in-flight latch test above uses).
        view._history_loading = True
        # The stale settle lands NOW, after the retarget.
        view._finish_history_mount(stale)
        await pilot.pause()
        assert view._history_loading, "a stale settle must not clear the new job's in-flight read"
        assert "loading earlier" in view._history_state_text()
