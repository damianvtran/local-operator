"""Aida's TUI surfaces, driven through the real app rather than the helpers.

Two things this file exists for, both learned the hard way:

* the PINNED-FIRST picker claim is about the row PIPELINE (``_cmd_resume``
  builds rows, overlays live state, then hands them to the screen). A test of
  the reorder function alone proved the function and shipped nothing — the
  first cut called it from nowhere, and only a rendered picker showed her
  last. So these tests type ``/resume`` and read the screen.
* the receipts are the feature's whole visible surface for pause/resume/status
  (decision (c): no extra chrome), so they are asserted as the text a user
  reads, on the transcript, in a booted app.
"""

from __future__ import annotations

import pytest

from tests.unit.tui.test_app_pilot import (
    FakeSession,
    _factory,
    _resume_factory,
    _seed_session,
    _transcript_text,
)


def _boot(tmp_path, monkeypatch, *, resume_boots=None):
    """An app on an isolated root, with the real config-dir resolution."""
    from local_operator.tui.app import OperatorApp

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=_resume_factory(resume_boots if resume_boots is not None else []),
    )
    return app


@pytest.mark.asyncio
async def test_the_picker_leads_with_her_pinned_row(tmp_path, monkeypatch) -> None:
    """R27's picker half, through the pipeline that actually builds the rows.

    Three real sessions: hers (created and pinned by ``ensure_session``) plus
    two ordinary ones whose transcripts are FRESHER than her directory's
    mtime, so a pure-recency order would put her last — which is exactly what
    the first cut of this feature did.
    """
    import os

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None, "the isolate must be enabled by default"
    _seed_session(tmp_path, "aaaaaaaaaaa1", prompt="fix the flaky login test")
    _seed_session(tmp_path, "bbbbbbbbbbb2", prompt="draft the release notes")
    # Her directory is made the OLDEST on purpose: the picker's native order is
    # ``(-mtime, id)``, so without the pin she would land last (and the id
    # tiebreak would not save her either — '6' sorts before 'a'). The pin is
    # therefore the only reason she can be first, which is what R27 claims.
    old = os.stat(tmp_path / "sessions" / her_id).st_mtime - 3600
    os.utime(tmp_path / "sessions" / her_id, (old, old))

    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one("Editor")
        editor.focus()
        editor.text = "/resume"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(120):
            await pilot.pause()
            if isinstance(app.screen, SessionPickerScreen) and app.screen._all:
                break

        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)
        ids = [row.id for row in picker._all]
        assert ids[0] == her_id, (
            "the pinned conversation leads the picker; a pure-recency order "
            f"would list her last. Order was {ids}"
        )
        # The rest keep their order — the reorder moves pinned rows, it does
        # not sort the remainder. In recency order the seeds are newer than
        # her directory, so both follow her.
        assert ids[1:] == ["bbbbbbbbbbb2", "aaaaaaaaaaa1"], ids
        # And the FIRST painted row is hers, not just the first in a list.
        painted = "\n".join(picker.render_lines_for_test())
        assert painted.index("Aida") < painted.index("fix the flaky"), painted


@pytest.mark.asyncio
async def test_the_pause_resume_status_receipts(tmp_path, monkeypatch) -> None:
    """``/aida pause``, ``resume`` and ``status`` each answer in one line.

    The pause receipt must also be true on disk (config flipped, hold marker
    written) — a receipt that only says the words would pass a transcript
    assertion and hold nothing.
    """
    from local_operator import aida as aida_pkg
    from local_operator.config import ConfigManager

    her_id = await aida_pkg.ensure_session(tmp_path)
    # The cadence arm is what gives ``pause`` an index entry to stamp: her
    # ensure writes ``wakes/<id>.json`` with the ``aida-cadence`` one-shot.
    from local_operator.wakes import store as wake_store

    assert wake_store.read_entry(tmp_path, her_id), "ensure must arm the cadence"

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one("Editor")

        async def run(command: str) -> str:
            editor.focus()
            editor.text = command
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
            return _transcript_text(app)

        body = await run("/aida status")
        assert "Aida: active" in body, body
        assert "next check-in" in body, body

        body = await run("/aida pause")
        assert "Aida paused" in body, body
        # ``get_nested_value``, not ``get_config_value``: the settings registry
        # stores this under the nested path (("aida","cadence","paused")), and
        # the dotted string is a verbatim top-level key that nothing writes —
        # the mismatch config.py's own docstring warns about.
        assert (
            ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "cadence", "paused"))
            is True
        )
        # Pause CANCELS her rows (the freeze: remembered in state for resume);
        # the held_at marker itself is exercised where it exists for a reason —
        # against a surviving owner row (test_aida_proactive's pause test).
        held = wake_store.read_entry(tmp_path, her_id) or {}
        assert [row["id"] for row in held.get("schedules") or []] == [], held

        body = await run("/aida resume")
        assert "Aida is active again" in body, body
        again = wake_store.read_entry(tmp_path, her_id) or {}
        assert "held_at" not in again, again
        assert [row["id"] for row in again.get("schedules") or []] == ["aida-cadence"], again
        assert (
            ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "cadence", "paused"))
            is False
        )

        body = await run("/aida =pause now")  # the escape sends her the WORD
        assert "Aida paused" not in body, "the `=` escape must not be parsed as the verb"


@pytest.mark.asyncio
async def test_a_request_rides_the_adoption_onto_her_conversation(tmp_path, monkeypatch) -> None:
    """``/aida <request>`` types ONCE: the text is delivered to her session.

    The stash-then-consume seam (``_pending_aida_prompt`` →
    ``_submit_aida_prompt``) is the one place a request can be silently
    dropped, so the assertion is on the SESSION's recorded prompt, not on a
    receipt: a dropped request leaves a perfectly quiet screen.
    """
    # The app resolves ``config_dir()`` from the ENV, so the isolate has to be
    # the env's root BEFORE the app is built: with the two out of step, the
    # app creates a SECOND Aida next to the one this test made — which is how
    # this test first failed, and it is the whole reason every test here goes
    # through ``_boot``'s env or sets it before construction.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)

    built: list[FakeSession] = []
    boots: list[str | None] = []

    class HerSession(FakeSession):
        """A fake whose ``session_id`` is the one it was booted FOR.

        ``FakeSession.session_id`` is a read-only property returning ``"sess"``
        — assignment raises — and the adoption seam compares exactly this
        attribute, so the id has to be the boot's.
        """

        def __init__(self, sid: str) -> None:
            super().__init__()
            self._sid = sid

        @property
        def session_id(self) -> str:
            return self._sid

    async def resume_factory(session_id):
        boots.append(session_id)
        session = HerSession(session_id or "")
        built.append(session)
        return session

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one("Editor")
        editor.focus()
        editor.text = "/aida summarise everything in flight"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(120):
            await pilot.pause()
            if built and built[-1].prompts:
                break

        assert boots == [her_id], f"the factory must be asked for HER conversation: {boots}"
        assert built[0].prompts == ["summarise everything in flight"], built[0].prompts
        assert app._conversation_id() == her_id
