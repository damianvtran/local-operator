"""``/agent class <name> proactive|reactive`` — the R36 switch in the TUI.

The storage half is pinned in ``tests/unit/test_action_class.py``; what these
pin is the slash surface: the reserved-word grammar (including the ``=``
escape), the report form, the flip's receipt, and the immediate cleanup call
on the session. Driven at the real ``OperatorApp`` dispatch, the way the other
slash tests are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.action_class import class_from_tags
from local_operator.agents import AgentRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _settle(pilot, app: OperatorApp) -> list[str]:
    """Pause a bounded number of frames so the worker can land its notice.

    Fixed-count rather than stop-at-first-notice: boot itself may have painted
    a notice, and returning early would race the worker this test is about.
    """
    for _ in range(120):
        await pilot.pause()
    return _notices(app)


@pytest.mark.asyncio
async def test_the_report_form_names_the_class_without_changing_anything(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/agent class aida")
        notices = await _settle(pilot, app)

    assert any("class proactive" in text for text in notices), notices
    # Seed resolution, not an install: the report must not write anything.
    assert AgentRegistry(tmp_path).get_agent_by_name("aida") is None


@pytest.mark.asyncio
async def test_the_flip_writes_the_tag_and_receipts_the_stop(tmp_path: Path) -> None:
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/agent class aida reactive")
        notices = await _settle(pilot, app)
        for _ in range(100):  # the worker's registry write is a thread hop
            await pilot.pause()
            row = AgentRegistry(tmp_path).get_agent_by_name("aida")
            if row is not None:
                break

    assert any("now reactive" in text and "stopped" in text for text in notices), notices
    row = AgentRegistry(tmp_path).get_agent_by_name("aida")
    assert row is not None
    assert class_from_tags(row.tags) == "reactive"


@pytest.mark.asyncio
async def test_a_bad_class_word_is_refused_before_any_write(tmp_path: Path) -> None:
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/agent class aida sideways")
        notices = await _settle(pilot, app)

    assert any("must be one of" in text for text in notices), notices
    assert AgentRegistry(tmp_path).get_agent_by_name("aida") is None


def _fields(**overrides: Any):
    """``AgentEditFields`` with every field spelled out (strict mode)."""
    from local_operator.agents import AgentEditFields

    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
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
    base.update(overrides)
    return AgentEditFields(**base)


@pytest.mark.asyncio
async def test_the_settings_pane_rows_state_each_profiles_class(tmp_path: Path) -> None:
    """The browse surface is honest about the class (design §8.1.3).

    ``_agent_profile_rows`` is the ONE enumeration feeding both the settings
    page's Agents pane and the ``/agent`` argument picker, so the class riding
    its facts string is what lets a user see, while browsing, which profiles
    may message them unprompted. Pinned because the pane showed only
    ``role``/model/effort before this slice, and a regression there would be
    invisible: the pane is read-only and nothing else asserts its text.
    """
    registry = AgentRegistry(tmp_path)
    registry.create_agent(
        _fields(
            name="steadier",
            description="Reaches out when something needs the operator.",
            tags=["role", "class:proactive"],
            # A CONFIGURED role: model pinned, so the facts string is longer
            # than the settings pane's 27-cell line and the class only
            # survives if it leads the optional facts (design round 1, D1).
            model="claude-opus-5",
        )
    )
    session = FakeSession()
    session.agent_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        rows = app._agent_profile_rows()

    facts = {name: kind for name, _label, kind, _summary in rows}
    assert "proactive" in facts.get("steadier", ""), facts
    # The CLASS leads the optional facts, so the pane's 27-cell truncation
    # cannot cut it — and it must still be there after that truncation.
    from local_operator.tui.widgets.settings_view import truncate_cells

    assert facts["steadier"].startswith("role · proactive"), facts["steadier"]
    assert "proactive" in truncate_cells(facts["steadier"], 27)
    # The control: absence of the tag must NOT invent a marker.
    registry.create_agent(
        _fields(
            name="plainrole",
            description="A reactive role.",
            tags=["role"],
        )
    )
    app2 = OperatorApp(lambda: _factory(session))
    async with app2.run_test(size=(120, 40)) as pilot2:
        for _ in range(40):
            await pilot2.pause()
            if app2._session is not None:
                break
        rows2 = app2._agent_profile_rows()
    facts2 = {name: kind for name, _label, kind, _summary in rows2}
    assert "plainrole" in facts2, facts2
    assert "proactive" not in facts2["plainrole"], facts2


def test_the_pane_paints_the_class_marker_one_rung_brighter() -> None:
    """D1's ink half: the marker is `muted`, the rest of the line `faint`.

    The pane's single facts line used to paint wholly `faint`, which measures
    1.97:1 against the pane — so even a marker that survived truncation was
    effectively erased on the surface §8.1.3 puts forward as where the class
    is visible (design round 1, D1). The unmarked case must be BYTE-identical
    to the old line, so this is a marker change and not a repaint.
    """
    from rich.style import Style

    from local_operator.tui.widgets.settings_view import _paint_facts

    muted = Style(color="red")
    faint = Style(color="blue")
    line = _paint_facts("role · proactive · claude-opus-5", 27, muted=muted, faint=faint)
    assert ("proactive", muted) in line
    assert all(style is faint for text, style in line if text.strip(" ·") != "proactive"), line
    plain = _paint_facts("role · claude-opus-5", 27, muted=muted, faint=faint)
    assert plain == [("    role · claude-opus-5", faint)]


def test_the_facts_indent_is_paid_once_however_many_parts() -> None:
    """R10: only the FIRST part carries the 4-cell indent.

    The caller's budget reserves exactly ONE indent (``room - 4``), so an
    indent repeated per part overruns it by 4*(parts-1) cells and the tail
    drops whole: a populated 4-part line cost 57 instead of 45 cells and
    clipped the final fact (QA round-2 measured +12). This pins the budget
    arithmetic against the real helper, marker class intact.
    """
    from rich.cells import cell_len
    from rich.style import Style

    from local_operator.tui.widgets.settings_view import _paint_facts

    muted = Style(color="red")
    faint = Style(color="blue")
    for facts in (
        "proactive · model · high",
        "proactive · medium · 3 tools · model/name",
    ):
        line = _paint_facts(facts, 80, muted=muted, faint=faint)
        plain = "".join(part for part, _style in line)
        assert plain == "    " + facts
        assert cell_len(plain) == cell_len(facts) + 4
        marker = [part for part, style in line if style is muted]
        assert marker == ["    proactive"]


def test_a_budgeted_facts_line_drops_the_tail_whole_and_keeps_the_marker() -> None:
    """R10 narrow case: at the caller's exact budget the MARKER survives and
    the tail gives way inside the same contract (``room`` passed is the
    caller's ``room - 4``; the indent rides on top, so the line may not
    exceed ``room + 4`` cells and may not re-indent any later part)."""

    from rich.cells import cell_len
    from rich.style import Style

    from local_operator.tui.widgets.settings_view import _paint_facts, truncate_cells

    muted = Style(color="red")
    faint = Style(color="blue")
    facts = "proactive · model · high"
    room = 20  # the caller's own window at this point
    line = _paint_facts(facts, room, muted=muted, faint=faint)
    plain = "".join(part for part, _style in line)
    assert plain.startswith("    proactive")
    assert "high" not in plain
    assert cell_len(plain) == cell_len(truncate_cells(facts, room)) + 4
    assert line[0] == ("    proactive", muted)
    assert all(style is faint for _text, style in line[1:])


class _PickerEditor:
    """The slice ``slash_argument`` and the argument builder read."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._argument_commands = ("agent", "agents")
        self._command_names = frozenset({"agent", "agents"})

    def _caret_offset(self) -> int:
        return len(self.text)


@pytest.mark.asyncio
async def test_the_agent_picker_offers_the_class_verb(tmp_path: Path) -> None:
    """Discoverability: the reserved word is a picker row (UX round 1, U3).

    The word the picker offers comes from ``agent_subcommand_rows()`` — the
    same table the handler reads — so the row a user picks cannot drift from
    the word the handler accepts. Before this the switch was findable only by
    guessing the word or being told it.
    """
    registry = AgentRegistry(tmp_path)
    registry.create_agent(_fields(name="auditor", description="Audit changes", tags=["role"]))
    # R4-2: a name that ITSELF starts with the escape character — its row must
    # complete doubled (``==odd``), because the handlers strip exactly one `=`.
    registry.create_agent(_fields(name="=odd", description="Odd name", tags=["role"]))
    session = FakeSession()
    session.agent_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        rows = app._agent_argument_choices(_PickerEditor("/agent "))
        compounds = app._agent_argument_choices(_PickerEditor("/agent class "))

    # ``=odd`` sorts before every letter, so it also proves the escape row
    # leads — it is a NAME row (the common action), not the verb.
    assert rows[0].name == "==odd", rows
    assert any(row.name == "auditor" for row in rows), rows
    # U4: agent NAMES lead; the reserved verb is present but LAST, so it is
    # never the default Tab completion (never first).
    assert [row.name for row in rows[-3:]] == ["class", "new", "edit"], rows
    assert rows[-3].detail == "subcommand"
    assert "proactive" in rows[-3].description
    # After the verb the names are offered as ``class <name>`` compounds, which
    # complete to the report form.
    assert compounds and all(row.name.startswith("class ") for row in compounds), compounds
    # U7: a name whose own spelling starts with ``=`` completes DOUBLED in the
    # second slot too, so the offered compound resolves the name it shows.
    assert any(row.name == "class ==odd" for row in compounds), compounds
    assert not any(row.name == "class =odd" for row in compounds), compounds


async def _type(pilot, text: str) -> None:
    """Type ``text`` into the focused editor via real key presses.

    The picker's argument detection is caret-anchored, so setting ``editor.text``
    directly no longer opens the argument list — driving real key presses is the
    path the app actually takes (the test_team_chart/test_slash_echo idiom).
    """
    for char in text:
        await pilot.press("slash" if char == "/" else ("space" if char == " " else char))


@pytest.mark.asyncio
async def test_bare_tab_completes_an_agent_not_the_class_verb(tmp_path: Path) -> None:
    """U4: Tab on a bare ``/agent `` fills the first AGENT, never the verb.

    Walk-shaped on purpose — driven through the running composer, because the
    direct builder call above cannot see the Tab path (``_resolve_argument``
    accepts the HIGHLIGHTED row; ``set_name_choices`` only feeds the
    highlighter). The rule is ``/team``'s: the common action on ``/agent `` is
    attaching/messaging, so Tab must never silently land in the class grammar.
    """
    registry = AgentRegistry(tmp_path)
    registry.create_agent(_fields(name="auditor", description="Audit changes", tags=["role"]))
    session = FakeSession()
    session.agent_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app.query_one(Editor).focus()
        await _type(pilot, "/agent ")
        editor = app.query_one(Editor)
        assert editor.picker.is_open(), "the agent list must open"
        first_row = editor.picker._choices[0].name
        assert first_row != "class", editor.picker._choices
        await pilot.press("tab")
        await pilot.pause()
        text = editor.text
    # Tab accepted the HIGHLIGHTED (first) row, and that row is a NAME: the
    # mutation this pins is the old verb-first order, under which both the
    # highlight and the completed buffer were `class`.
    assert text == f"/agent {first_row} ", text


@pytest.mark.asyncio
async def test_picker_second_slot_reoffers_names_feeding_the_class_report(
    tmp_path: Path,
) -> None:
    """U5: ``/agent class `` crosses into the second slot and repaints.

    The slot tracker must post a refresh on the boundary (the `chart ` rule),
    or the compounds the builder returns are never fetched by the running
    editor — the round-2 finding: no rows at all under a query that no longer
    matches the first-slot choice set.
    """
    registry = AgentRegistry(tmp_path)
    registry.create_agent(_fields(name="auditor", description="Audit changes", tags=["role"]))
    session = FakeSession()
    session.agent_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app.query_one(Editor).focus()
        await _type(pilot, "/agent class ")
        editor = app.query_one(Editor)
        names = [c.name for c in editor.picker._choices]
    assert "class auditor" in names, names
    assert all(name.startswith("class ") for name in names), names


@pytest.mark.asyncio
async def test_the_class_verb_shows_no_switch_hint(tmp_path: Path) -> None:
    """U5 second half: ``/agent class `` must not print the switch/send hint.

    The hint promises a switch-or-send choice; after the verb the slot is a
    report argument whose list is up — the hint's own contract excludes the
    reserved word (it excluded ``/team chart`` for the same reason).
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app.query_one(Editor).focus()
        await _type(pilot, "/agent class ")
        editor = app.query_one(Editor)
        notice = editor.picker._notice or ""
        hint_shown = "Enter to switch" in notice
    assert not hint_shown, notice


def test_the_scope_token_takes_a_second_facts_row_when_it_cannot_fit() -> None:
    """LOH-9 S2 round 2 (D1/U2): the class marker AND the full scope survive.

    At the pane's 27-cell facts budget ``specialist · proactive · this
    conversation`` is 42 cells. Dropping the token kept the class but erased the
    one fact saying the row exists only here; clipping it mid-word read as
    breakage. The bound is a second facts line in the same ink, so both facts
    are readable at the narrowest width.
    """
    from rich.style import Style

    from local_operator.tui.widgets.settings_view import _paint_facts_lines

    muted = Style(color="red")
    faint = Style(color="blue")

    fits = _paint_facts_lines("role · this conversation", 27, muted=muted, faint=faint)
    assert len(fits) == 1
    assert ("this conversation", muted) in fits[0]

    narrow = _paint_facts_lines(
        "specialist · proactive · this conversation", 27, muted=muted, faint=faint
    )
    assert len(narrow) == 2
    assert "".join(part for part, _style in narrow[0]) == "    specialist · proactive"
    assert ("proactive", muted) in narrow[0], "the class marker must survive"
    assert narrow[1] == [("    this conversation", muted)], "the token is whole and muted"

    # THE REAL CONFIGURED-ROLE STRING (round 2, D1): `agent_listing_rows` puts
    # model/effort facts AFTER the scope token, so the token is mid-string. A
    # suffix-only lift left this — the commonest populated role — clipped.
    # 52 cells: the full line is 68, and the head WITHOUT the token is 47 plus
    # the 4-cell indent — so this pins the lift, not a truncation (at the pane's
    # 27 the head itself must truncate, which is correct and is the case above).
    configured = _paint_facts_lines(
        "role · proactive · this conversation · claude-opus-5 · effort high",
        52,
        muted=muted,
        faint=faint,
    )
    assert len(configured) == 2
    assert "this conversation" not in "".join(part for part, _style in configured[0])
    assert "effort high" in "".join(part for part, _style in configured[0])
    assert configured[1] == [("    this conversation", muted)]
