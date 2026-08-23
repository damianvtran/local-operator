"""A pasted image survives a prompt-sending slash command.

The bug: attach an image, then route it to a team or persona with
``/team <name> <request>`` or ``/agent <name> <message>``, and the screenshot
was silently dropped — the model received the bare ``[Image #N]`` marker as
text with no pixels behind it. ``on_editor_submitted`` handed slash-shaped
lines to ``_run_slash_command(text)`` and discarded ``message.attachments``,
and the two commands whose argument reaches the model (``_cmd_team`` /
``_cmd_agent``) called ``_submit_prompt(request)`` with no images.

These are end-to-end pilot tests, not editor unit tests: the whole point is
that the attachment survives the SUBMIT path from composer to session, which is
exactly the seam that dropped it. The image is pasted from a real PNG so the
marker the app writes is the same one a user's paste produces, and the
assertion reads the pixels off ``FakeSession.prompt_images`` — proof the bytes
travelled, not merely that a marker did.
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from textual import events

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _agent_registry(tmp: Path) -> AgentRegistry:
    """A registry with one role that carries real instructions.

    Mirrors ``test_slash_echo._agent_registry`` but minimal: ``/agent auditor``
    resolves to a role whose non-empty system prompt makes the attach layer a
    persona (the ``agent_brief`` path), so the message-carrying branch — the one
    that must forward the image — is the branch this test exercises.
    """
    registry = AgentRegistry(tmp)
    # AgentEditFields requires every field spelled out (no defaults), so build a
    # None-filled base and override only what this fixture needs — the same
    # shape ``test_slash_echo`` uses for its registry.
    base = dict(
        name=None,
        security_prompt=None,
        hosting=None,
        model=None,
        description=None,
        tags=None,
        categories=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    role = registry.create_agent(
        AgentEditFields(**{**base, "name": "auditor", "description": "Audit", "tags": ["role"]})
    )
    registry.set_agent_system_prompt(role.id, "You audit changes.")
    return registry


def _png(path: Path, width: int = 1568, height: int = 200) -> str:
    Image.new("RGB", (width, height), (30, 30, 40)).save(path)
    return str(path)


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _paste(app: OperatorApp, pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_team_request_carries_the_pasted_image(tmp_path) -> None:
    """`/team <name> <request>` sends the screenshot the request cites."""
    path = _png(tmp_path / "shot.png", 1568, 410)
    session = FakeSession()
    reg = TeamRegistry(Path(tempfile.mkdtemp()))
    reg.create_team(
        TeamEditFields(name="ops", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = reg
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        # Attach the image FIRST — the exact order the bug report describes:
        # image on the composer, then a slash command typed around it.
        await _paste(app, pilot, path)
        assert editor.referenced_images(), "the image was not attached to the composer"
        # The composer now holds `[Image #1, …] ` — type the command around it,
        # mirroring `/team ops <request about the image>`.
        marker = editor.text.strip()
        editor.load_text(f"/team ops look at {marker} and fix it")
        await pilot.press("enter")
        await pilot.pause()

        assert [t.name for t in session.attached_teams] == ["ops"]
        assert session.prompts == ["look at [Image #1, 1568x410] and fix it"]
        # The crux: one image, and it is the pixels that were pasted.
        (sent,) = session.prompt_images[0]
        decoded = Image.open(io.BytesIO(base64.b64decode(sent.data)))
        assert decoded.size[0] > 0 and decoded.size[1] > 0


@pytest.mark.asyncio
async def test_team_without_a_request_sends_nothing(tmp_path) -> None:
    """A bare attach (`/team ops` with no request) sends no turn, no image.

    Guards the resolve-from-text rule from the other side: the image marker
    lives in the request tail, so an attach with no tail cites nothing and must
    not smuggle the composer's leftover attachment into a turn that never runs.
    """
    path = _png(tmp_path / "shot.png")
    session = FakeSession()
    reg = TeamRegistry(Path(tempfile.mkdtemp()))
    reg.create_team(
        TeamEditFields(name="ops", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = reg
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        # Attach only — the image marker is present but the command has no
        # request tail citing it.
        editor.load_text("/team ops")
        await pilot.press("enter")
        await pilot.pause()

        assert [t.name for t in session.attached_teams] == ["ops"]
        assert session.prompts == []
        assert session.prompt_images == []


@pytest.mark.asyncio
async def test_team_request_carries_multiple_images_in_text_order(tmp_path) -> None:
    """Two images, but only the one the REQUEST text still cites is sent.

    Locks the resolve-from-text contract the fix leans on: markers are keys, not
    positions, so citing only ``#2`` (the user deleted ``#1`` before sending)
    must send exactly one image — the ``#2`` pixels, not ``#1`` and not both.
    Distinct sizes make the surviving image identifiable by its decoded
    dimensions. (Ordering across several surviving markers is already pinned by
    ``resolve_markers``'s own tests in ``test_paste_images``; this guards the
    slash-command seam that used to drop the pixels entirely.)
    """
    first = _png(tmp_path / "a.png", 40, 20)
    second = _png(tmp_path / "b.png", 80, 60)
    session = FakeSession()
    reg = TeamRegistry(Path(tempfile.mkdtemp()))
    reg.create_team(
        TeamEditFields(name="ops", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = reg
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        await _paste(app, pilot, second)
        # Both markers are now in the composer, `#1` then `#2`.
        assert editor.text == "[Image #1, 40x20] [Image #2, 80x60] "

        # Cite ONLY #2 in the request — the user kept the second screenshot and
        # dropped the first before routing it.
        editor.load_text("/team ops just this one [Image #2, 80x60]")
        await pilot.press("enter")
        await pilot.pause()

        (sent,) = session.prompt_images[0]
        kept = Image.open(io.BytesIO(base64.b64decode(sent.data)))
        assert kept.size == (80, 60), "the wrong (or extra) image was sent"


@pytest.mark.asyncio
async def test_team_marker_in_the_name_position_is_not_an_image(tmp_path) -> None:
    """F2: an image marker where the NAME goes resolves nothing and finds no team.

    Pins the (correct, but silent) behaviour that only the request TAIL carries
    images: ``/team [Image #1] …`` treats the bracketed marker as a team name,
    which no team matches, so it is an error rather than a smuggled image. Keeps
    a future refactor from quietly making the name position image-bearing.
    """
    path = _png(tmp_path / "shot.png")
    session = FakeSession()
    reg = TeamRegistry(Path(tempfile.mkdtemp()))
    reg.create_team(
        TeamEditFields(name="ops", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = reg
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.strip()
        # Marker in the NAME slot, not the request tail.
        editor.load_text(f"/team {marker} describe it")
        await pilot.press("enter")
        await pilot.pause()

        # No team matched a bracket-shaped name, so nothing was attached and no
        # turn (and no image) was sent.
        assert session.attached_teams == []
        assert session.prompts == []
        assert session.prompt_images == []


@pytest.mark.asyncio
async def test_agent_message_carries_the_pasted_image(tmp_path) -> None:
    """`/agent <name> <message>` sends the screenshot the message cites."""
    path = _png(tmp_path / "shot.png", 1568, 410)
    session = FakeSession()
    session.agent_registry = _agent_registry(Path(tempfile.mkdtemp()))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        assert editor.referenced_images(), "the image was not attached to the composer"
        marker = editor.text.strip()
        editor.load_text(f"/agent auditor what do you make of {marker}")
        await pilot.press("enter")
        await pilot.pause()

        assert session.attached_agents == ["auditor"]
        assert session.prompts == ["what do you make of [Image #1, 1568x410]"]
        (sent,) = session.prompt_images[0]
        decoded = Image.open(io.BytesIO(base64.b64decode(sent.data)))
        assert decoded.size[0] > 0 and decoded.size[1] > 0
