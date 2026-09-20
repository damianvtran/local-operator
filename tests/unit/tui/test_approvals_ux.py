"""Choosing an approval mode: the scope, the list, and what the band claims.

Three complaints answered here, and each maps to a group below.

* The mode applied to the running session only, so "auto-approve everything
  forever" had no spelling at all — while ``/model default`` had one, on the
  same word, one command away.
* The modes were named in the command's DESCRIPTION and nowhere else, so a user
  who could not remember them typed blind into an argument the app knew the
  answers to. ``/effort`` had the identical shape.
* An owner's frame showed ``! auto-approve`` beside two tool calls reporting
  ``User denied approval``. The denial turned out to be a swallowed exception
  elsewhere, but the hour spent establishing whether the band was lying is the
  cost of a band nothing ties to the gate. So the last group asserts the tie.

Everything here drives the REAL editor and the REAL command dispatch. The
argument list is a keystroke-level feature: a test that called the handler
directly would pass with the list wired to nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import SLASH_COMMANDS, OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice, ArgumentMode
from local_operator.tui.widgets.approval import ApprovalBlock, ApprovalPrompt
from local_operator.tui.widgets.command_picker import PickerMode
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _band, _factory
from tests.unit.tui.test_effort import EffortSession


class GatedSession(FakeSession):
    """A fake that keeps the approval handler the app installs on it."""

    def __init__(self) -> None:
        super().__init__()
        self.approval_handler: Any | None = None

    def set_approval_handler(self, handler: object | None) -> None:
        self.approval_handler = handler


def _gate(session: GatedSession):
    """The gate the app installed, narrowed — this is what a tool call reaches."""
    handler = session.approval_handler
    assert handler is not None, "the app installed no approval handler"
    return cast("Any", handler)


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _wait_prompt(pilot, app: OperatorApp) -> None:
    """Settle until the in-flight gate's question is mounted.

    Replaces a fixed `pilot.pause(0.3)` that bet on how long the gate takes to
    arm and raise its card; the observable the next assertion needs is the
    ``ApprovalPrompt`` on screen, so wait for exactly that. The ceiling is a
    deadlock guard, not a timing assumption.
    """
    for _ in range(200):
        await pilot.pause()
        if app.query(ApprovalPrompt):
            return


async def _settle(pilot, ticks: int = 6) -> None:
    """Let the event loop drain for a bounded number of idle ticks.

    Used where the assertion is a NEGATIVE — that no prompt mounted — which
    cannot be polled for (there is no state to wait to appear); a fixed handful
    of idle waits gives any erroneous mount the chance to happen so its absence
    is meaningful.
    """
    for _ in range(ticks):
        await pilot.pause()


async def _type(pilot, app: OperatorApp, text: str) -> None:
    """Put ``text`` in the composer the way typing does, and settle the lists.

    Assignment rather than per-character presses: the editor funnels every
    mutation through ``edit()``/``load_text()`` into the same ``_sync_picker``,
    so the list state is identical and the test does not spend a second on
    keystrokes. The tests that are ABOUT keystrokes press real keys.

    The caret is parked at the end and the pickers re-derived, because the
    ``text`` setter syncs with the caret still at the origin and the slash
    detection is caret-anchored (inline detection) — a text-set that left the
    caret before the slash would sit in no command at all.
    """
    editor = app.query_one(Editor)
    editor.text = text
    editor.move_cursor(editor._end_of_buffer())
    editor._sync_picker()
    await pilot.pause()
    await pilot.pause()


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line and send it, dismissing the list first.

    Enter on an open list completes the highlighted row instead of submitting
    what was typed, so a test that skipped the Esc would exercise the
    completion rather than the command it meant to run.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _rows(app: OperatorApp) -> list[tuple[str, str, str]]:
    """``(name, description, detail)`` for every row the argument list offers."""
    picker = app.query_one(Editor).picker
    assert picker.mode is PickerMode.ARGUMENT, "the picker is not in argument mode"
    rows: list[tuple[str, str, str]] = []
    for name, choice in picker.suggestions():
        assert isinstance(choice, ArgumentChoice)
        rows.append((name, choice.description, choice.detail))
    return rows


def _saved_mode(config_dir: Path) -> str | None:
    """The mode as it is on DISK — read back through YAML, not through the app.

    Asserting on the app's own attribute would prove only that it remembers
    what it was told; the claim a receipt makes is about a file.
    """
    config_file = config_dir / "config.yml"
    if not config_file.is_file():
        return None
    return yaml.safe_load(config_file.read_text(encoding="utf-8"))["values"].get(
        "tool_approval_mode"
    )


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config at a temp dir — these tests WRITE one."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


# -- scope: this session versus every session ---------------------------------


@pytest.mark.asyncio
async def test_a_session_mode_changes_nothing_on_disk(config_dir: Path) -> None:
    """The default of the two scopes is the reversible one.

    `/approvals auto` is the same command it always was, and it still ends when
    the window does. The receipt has to SAY so, because nothing else on the
    frame distinguishes it from the durable form — that indistinguishability is
    the complaint.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals auto")

        assert app._approve_all is True
        assert app._approvals_default_auto is False
        assert _saved_mode(config_dir) is None, "a session switch wrote to the config file"
        receipt = _notices(app)[-1]
        assert "(this session)" in receipt
        assert "/approvals default auto" in receipt, "the durable form is unreachable"


@pytest.mark.asyncio
async def test_the_session_switch_stops_offering_a_save_the_file_already_has(
    config_dir: Path,
) -> None:
    """UX round 1, U5: the hint is dropped once it is not news.

    A `/approvals auto` usually follows the refusal notice now — the file already
    says `auto` and only this session is holding back — so "saves it for new
    sessions" was instructing the user to do what was already done. The receipt
    keeps the live half exactly as the runtime words it.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert _saved_mode(config_dir) == "auto", "premise: the file already says auto"
        await _submit(pilot, app, "/approvals ask")
        await _submit(pilot, app, "/approvals auto")

        receipt = _notices(app)[-1]
        assert receipt == (
            "tool approvals: auto — every tool runs without asking (this session)"
        ), receipt
        assert app._approve_all is True


@pytest.mark.asyncio
async def test_the_default_form_writes_the_config_and_names_the_file(config_dir: Path) -> None:
    """`/approvals default auto` — the promotion, spelled `/model`'s way.

    The receipt names the file AND the key for the reason `/model default`'s
    does: "saved" alone is a claim the user cannot check without quitting.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals default auto")

        # Both halves: the durable one AND this session, so nobody has to run
        # two commands to end up in the state they asked for.
        assert _saved_mode(config_dir) == "auto"
        assert app._approve_all is True
        assert app._approvals_default_auto is True
        receipt = _notices(app)[-1]
        assert "config.yml" in receipt and "tool_approval_mode auto" in receipt
        assert "every new one" in receipt


@pytest.mark.asyncio
async def test_bare_default_keeps_the_mode_the_session_is_already_in(config_dir: Path) -> None:
    """ "Make THIS the default" is the sentence a user has right after switching.

    Same affordance as bare `/model default`, and for the same reason: making
    them retype the word they just typed is a transcription exercise.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals auto")
        assert _saved_mode(config_dir) is None
        await _submit(pilot, app, "/approvals default")

        assert _saved_mode(config_dir) == "auto"


@pytest.mark.asyncio
async def test_the_saved_default_is_in_force_in_the_next_session(config_dir: Path) -> None:
    """A config value nothing reads back is a file the app writes to itself.

    So this boots a SECOND app against the same config dir and asks the gate
    itself — not the flag, the callable a tool call actually awaits.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")
    assert _saved_mode(config_dir) == "auto"

    session = GatedSession()
    relaunched = OperatorApp(lambda: _factory(session))
    async with relaunched.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, relaunched)

        assert relaunched._approve_all is True
        assert relaunched._approvals_default_auto is True
        # The gate, not the flag: a write-only default would pass every
        # assertion above and still stop the first tool of the session.
        assert await _gate(session)("bash", "run: ls") is True
        # `ApprovalPrompt`, not `ApprovalBlock`: the live question is the docked
        # card now, and the block is only the receipt written after an answer.
        assert not relaunched.query(ApprovalPrompt), "auto-approve still mounted a prompt"
        # The band's trailing cell is the alarm. It no longer spells the mode out
        # — the session NAME owns that slot now — so `auto` and `always` are one
        # glyph here, and `/approvals` is what distinguishes them.
        assert _band(relaunched).rstrip().endswith("!")


@pytest.mark.asyncio
async def test_a_session_switched_away_from_its_default_says_both(config_dir: Path) -> None:
    """The two-valued state, reported without leaving the screen.

    A user who booted on a saved `auto` and turned it off for this session is
    one relaunch away from a mode they last chose days ago. Nothing else on the
    frame would ever mention it: the band goes quiet when the gate is armed,
    which is correct, and correct is not the same as complete.

    The saved half names ``config.yml`` and is read from the FILE at report
    time rather than from the cached default (UX round 1, U1/U2). Two reasons,
    both from the live-config change: the cached value now MOVES on a config
    tick, so comparing against it reported a matched pair for exactly the
    divergence this sentence exists to disclose; and with the asymmetric
    approvals rule a session can legitimately hold a mode the file disagrees
    with, which makes "what does the file say" the question a user is actually
    asking here. Naming the file also makes the claim checkable without
    quitting, which "new sessions" never was.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals ask")
        await _submit(pilot, app, "/approvals")

        report = _notices(app)[-1]
        assert "ask (this session)" in report
        assert "config.yml says auto" in report
        # The remedy is named, and it is the one that MATCHES the file — not
        # `/approvals default ask`, which would rewrite the machine-wide default
        # to clear one session's surprise (UX round 1, U3).
        assert "/approvals auto adopts it in this session" in report, report


@pytest.mark.asyncio
async def test_a_matched_pair_reports_one_state(config_dir: Path) -> None:
    """When the session and the default agree there is one fact, said once.

    The split sentence is for the split state; using it unconditionally would
    make every bare `/approvals` read like a discrepancy report.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals")

        report = _notices(app)[-1]
        assert "tool approvals: ask" in report
        assert "new sessions open the same way" in report
        assert "this session" not in report


@pytest.mark.asyncio
async def test_an_unwritable_config_still_switches_the_session(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    """A read-only config dir is a reason not to promise the next launch
    anything, not a reason to refuse this session the mode it asked for.

    And the band must not claim `always` off a write that failed — that is the
    same class of lie as the band the last group is about.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        monkeypatch.setattr(
            "local_operator.config.ConfigManager.set_config_value",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only file system")),
        )
        await _submit(pilot, app, "/approvals default auto")

        assert app._approve_all is True, "the session was denied the mode it asked for"
        assert app._approvals_default_auto is False
        assert "could not save default" in _notices(app)[-1]
        # Still alarmed: the session got the mode, only the promise about the
        # NEXT session failed, and the band never claimed that part anyway.
        assert _band(app).rstrip().endswith("!")


# -- the list: offered, not remembered ----------------------------------------


@pytest.mark.asyncio
async def test_approvals_offers_both_modes_and_both_scopes(config_dir: Path) -> None:
    """The reported UX bug: the modes were named in prose and typed from memory.

    Four rows, because scope is the axis that was invisible. A `default` row
    leading to a second list would put the mode back behind a keystroke the
    list cannot show, which is the same failure one level down.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/approvals ")

        assert [row[0] for row in _rows(app)] == ["ask", "auto", "default ask", "default auto"]
        scopes = {name: detail for name, _description, detail in _rows(app)}
        assert scopes["ask"].startswith("this session")
        assert scopes["auto"].startswith("this session")
        assert scopes["default ask"].startswith("every session")
        assert scopes["default auto"].startswith("every session")
        # Every row says what the mode DOES, not just what it is called.
        assert all(description for _name, description, _detail in _rows(app))


@pytest.mark.asyncio
async def test_the_list_marks_the_live_mode_and_the_saved_one_separately(
    config_dir: Path,
) -> None:
    """Two marks, because the user can be in two places at once.

    `· current` is the mode running now and `· saved` is what the next launch
    opens in; a list carrying only one of them cannot show a session that has
    been switched away from its default, which is the state most worth seeing
    at the moment of choosing.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals ask")
        await _type(pilot, app, "/approvals ")

        marks = {name: detail for name, _description, detail in _rows(app)}
        assert marks["ask"] == "this session · current"
        assert marks["auto"] == "this session"
        assert marks["default auto"] == "every session · saved"
        assert marks["default ask"] == "every session"


def test_the_option_list_marks_without_moving_a_column(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused row is marked by TINT, because words re-flow the list.

    Design round 1 (D2) asked for the loosening rows to stop being offered as if
    they would work on a surface that refuses them, and round 1's answer put a
    suffix in the row's ``detail``. Round 2 (D7) measured what that cost: the
    picker sizes its columns from the widest cell in the SET, so the mark took
    the detail column from 22 cells to 47 — invisible at 44 columns, and at 60 it
    dropped the scope column for ALL FOUR rows. ``alert`` paints the cell it
    already has, so no width loses anything.

    The property pinned here is exactly that: the two topologies offer
    byte-identical rows, and only the mark moves. ``_may_loosen_gate_here`` is
    the input the app reads for the decision, so staging it is staging the input
    rather than the answer — the live topologies (owning pane, attached pane,
    phone) are driven against real runtimes by
    ``tests/unit/session/runtime/test_approval_authority_seam.py``.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    owner = app._approval_choices()
    monkeypatch.setattr(app, "_may_loosen_gate_here", lambda: False)
    follower = app._approval_choices()

    def shape(choices: list[ArgumentChoice]) -> list[tuple[str, str, str]]:
        return [(choice.name, choice.description, choice.detail) for choice in choices]

    # Nothing about the rows moves: not the names, not the descriptions, not the
    # details, and therefore not the column widths the picker derives from them.
    assert shape(follower) == shape(owner)
    assert max(len(choice.detail) for choice in follower) == max(
        len(choice.detail) for choice in owner
    )

    marks = {choice.name: choice.alert for choice in follower}
    assert marks["auto"] is True, "the row that will be refused is not marked"
    # `ask` is never marked: tightening works from every surface, and a warning
    # tint on it would say the opposite.
    assert marks["ask"] is False
    assert {choice.name: choice.alert for choice in owner}["auto"] is False


def _block_shape(block: NoticeBlock) -> str:
    """What a height outlier needs in one line (design round 4b, D20).

    A height reading alone says nothing about WHICH text produced it: at 44
    columns a 5-row block is the card's sentence without the ``not applied —``
    correction, or an ordinary receipt — another block entirely, not a settle
    race. The opening characters and the width distinguish those in one line
    instead of costing the next reader a round of re-measurement.
    """
    return f"h={block.size.height} w={block.content_size.width} text[:32]={block.text()[:32]!r}"


@pytest.mark.asyncio
async def test_the_refused_card_notice_reaches_the_screen(config_dir: Path) -> None:
    """D9's host half, on the screen, at the narrowest width the product has.

    Three findings meet here. The refusal has to REACH the operator at all
    (design round 2 D9 left it swallowed, and the host half was rig-verified but
    unpinned — agent round 3, R3-4); it has to carry the CARD's sentence rather
    than the command's (UX U8); and the receipt the card's own keypress wrote
    above it has to be corrected, because "✓ allowed" is the strongest "it
    worked" affordance the transcript has and it is briefly wrong (UX U13).
    """
    from local_operator.harness.approval import CARD_APPROVAL_REFUSED_NOTICE
    from local_operator.session.errors import OperatorAuthorityRequired

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(44, 20)) as pilot:
        await _boot(pilot, app)
        app._note_gate_refusal_on_app_loop(OperatorAuthorityRequired(trigger="approval_answer"))
        await pilot.pause()

        notices = _notices(app)
        assert notices, "the refused card's notice never reached the transcript"
        shown = notices[-1]
        # The CARD's sentence, and the correction of the receipt above it.
        assert CARD_APPROVAL_REFUSED_NOTICE in shown, shown
        assert shown.startswith("not applied — "), shown

        block = [
            item for item in app.query_one(TranscriptView).blocks() if isinstance(item, NoticeBlock)
        ][-1]
        # The card's own sentence is EXACTLY 8 rows at 44 columns measured on this
        # widget, in the revision-2 copy (228 characters of copy plus the
        # "not applied — " receipt correction, at a 40-cell content width; the
        # rendered block text is 242 characters). It was 6 rows at 176 rendered
        # characters under the revision-1 copy, and the growth is the cost of the
        # sentence naming the levers that actually work — re-measured and re-pinned
        # rather than widened to a tolerant bound, because a one-row growth is the
        # regression this pin exists for (agent review round 4, R4-2/R4-3: the
        # previous `<= 8` and `<= 13` let a row slip through). Frames:
        # ``before-card``/``after-card`` in the PR's evidence.
        assert block.size.height == 8, _block_shape(block)

        # The command's copy is NOT what a card reader is told.
        from local_operator.harness.approval import OPERATOR_AUTHORITY_REQUIRED_NOTICE

        assert OPERATOR_AUTHORITY_REQUIRED_NOTICE not in shown, shown

        # AND THE LONG ONE, measured rather than computed (design round 3, D14;
        # agent R3-5). The block wraps at its OWN content width — 40 cells at a
        # 44-column terminal, not 44 — so the revision-2 copy's 288 characters
        # render as 9 rows, against a transcript area that is 13 rows in this
        # staging (`region [1,1,42,13] size [41,13] virtual [40,21] scroll_y=8`).
        # It was 12 rows at 345 characters under the revision-1 copy, and the
        # SHORTER block is not an accident: the retired remedy sentence was the
        # longest clause in it. What stays on screen is the reason and the
        # remedies, which is why the copy leads with them. A wrap-based pin once
        # said "9 rows" and measured a wrapping the frame does not do; this one
        # measures the widget. The transcript is given a row of its own first so
        # the two areas are the same shape.
        app._system_notice("a row of its own", "info")
        app._note_gate_refusal_on_app_loop(OperatorAuthorityRequired())
        await pilot.pause()
        tall = [
            item for item in app.query_one(TranscriptView).blocks() if isinstance(item, NoticeBlock)
        ][-1]
        # 9, measured: the command's 288 characters at the same 40-cell content
        # width. The block is pinned; the AREA is not, because it is a property of
        # what else is in the transcript — 13 rows in this staging, fewer with the
        # re-armed card docked, where the painted rows are the notice's TAIL rather
        # than a middle slice (design rounds 4b/5, D19: the claim that it fits at
        # every height was wider than the frames, which is the class of defect this
        # PR exists to fix). The ORDER of the copy is what carries the narrow
        # frames: the reason and the remedies are what a reader reaches first, and
        # the detail is recoverable once the card is answered.
        assert tall.size.height == 9, _block_shape(tall)
        assert OPERATOR_AUTHORITY_REQUIRED_NOTICE in tall._text


@pytest.mark.asyncio
async def test_choosing_a_row_runs_the_command_it_spells(config_dir: Path) -> None:
    """The list completes into the ARGUMENT and submits the same line a typist
    would have typed — one implementation of what `/approvals default auto`
    means, not a second path that can drift from the first."""
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/approvals ")
        # Arrow onto the row rather than trusting the matcher's pick: an
        # explicit move is what the editor's ambiguity gate accepts as "the
        # user chose this", and it is the gesture the feature is for.
        for _ in range(3):
            await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert app.query_one(Editor).text == ""
        assert app._approve_all is True
        assert _saved_mode(config_dir) == "auto"


@pytest.mark.asyncio
async def test_typing_the_mode_still_works_without_the_list(config_dir: Path) -> None:
    """The list is an addition for people who do not remember the options, not
    a gate in front of people who do.

    Both the canonical word and an accepted alias, pressed as real keys, with
    the list dismissed — the muscle memory that existed before this change must
    survive it.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        for character in "/approvals yolo":
            await pilot.press(
                "slash" if character == "/" else ("space" if character == " " else character)
            )
        await pilot.pause()
        await pilot.press("escape")  # dismiss the list; the typist does not need it
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert app._approve_all is True

        await _submit(pilot, app, "/approvals ask")
        assert app._approve_all is False


@pytest.mark.asyncio
async def test_a_bare_approvals_still_reports_instead_of_opening_a_list(
    config_dir: Path,
) -> None:
    """`/approvals` answers "what am I on", so Enter on its row SENDS it.

    That is the difference between an OPTIONAL argument and `/login`'s REQUIRED
    one, and it is why the two are separate values rather than one boolean: a
    command with a useful bare form must not have Enter silently repurposed
    into "open a list".
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        for character in "/approvals":
            await pilot.press("slash" if character == "/" else character)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert any("tool approvals: ask" in notice for notice in _notices(app))


@pytest.mark.asyncio
async def test_effort_offers_this_models_rungs_with_the_current_one_marked(
    config_dir: Path,
) -> None:
    """The second customer of the same mechanism, converted in the same change.

    `/effort` printed a ladder the user then had to transcribe. The rungs are
    the model's, read off the spec the request is built from, so a list that
    offered a fixed set would be wrong on the next model.
    """
    session = EffortSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/effort ")

        names = [name for name, _description, _detail in _rows(app)]
        assert names[0] == "auto", "the way back has to be offered, not remembered"
        assert names[1:] == list(session.model.reasoning_efforts)
        marked = [name for name, _description, detail in _rows(app) if detail == "current"]
        assert marked == [session.model.reasoning_effort or "auto"]


@pytest.mark.asyncio
async def test_choosing_an_effort_row_puts_it_on_the_spec(config_dir: Path) -> None:
    """The rung has to reach the REQUEST, not just the transcript."""
    session = EffortSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/effort low")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert session.model.reasoning_effort == "low"


def test_the_registry_states_which_commands_offer_values() -> None:
    """One declaration per command, and the two kinds are not interchangeable.

    Pinned as a table for the reason ``ECHO_POLICY`` is: the field has a safe
    default, so without this a new command silently inherits "no list" and the
    absence never shows up as a failure.
    """
    modes = {
        command.name: command.arguments
        for command in SLASH_COMMANDS
        if command.arguments is not ArgumentMode.NONE
    }
    assert modes == {
        "effort": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/effort`: bare `/fast` TOGGLES the dial, and the space
        # offers on/off/status for a user who would rather name the resulting
        # state than flip into it.
        "fast": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/effort`: bare `/theme` answers with the active theme,
        # and the space opens the ramp list with live preview.
        "theme": ArgumentMode.OPTIONAL,
        "approvals": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/approvals`: bare `/mcp` lists the servers, and the
        # space opens the login/logout/reauth subcommand list.
        "mcp": ArgumentMode.OPTIONAL,
        "login": ArgumentMode.REQUIRED,
        "logout": ArgumentMode.REQUIRED,
        # OPTIONAL like `/effort`: bare `/credential` lists the names in
        # memory; the space opens a KEY argument, then a masked paste.
        "credential": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/approvals`: bare `/stop` stops THIS session; the
        # list offers the other sessions and `all`.
        "stop": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/stop`: bare `/move` opens the directory picker, which
        # is the discoverable route, and the space offers the same suggestions
        # inline for a user who would rather type. Deliberately not REQUIRED —
        # Enter on the bare command does something useful, which is the line
        # `/login` sits on the other side of.
        "move": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/mcp`: bare `/team` lists the teams, and the space
        # opens the team-name argument list with roster details.
        "team": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/team`, which `/agent` mirrors: bare `/agent` lists
        # the roles/specialists, and the space opens the name argument list.
        "agent": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/effort`: bare `/analytics` opens the default (usage)
        # view, and the space offers the analytics-view list (today just
        # `usage`); the screen it opens IS the receipt, so it never echoes.
        "analytics": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/theme`: bare `/title` reports the name the
        # conversation currently carries, and the space offers the `--refresh`
        # row — the one word a user could not guess. Deliberately not REQUIRED:
        # Enter on the bare command answers, which is the line `/login` sits on
        # the other side of. Keyed by PRIMARY name, so `rename`, not `title`.
        "rename": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/rename`, whose flag row this mirrors: the space offers
        # `/goal --clear` to a user who has a goal to unset, while bare `/goal`
        # still reports the current one. NOT a name slot — the row is a flag and
        # the argument stays free text, which is what `name_argument` says.
        "goal": ArgumentMode.OPTIONAL,
        # OPTIONAL for the same reason: the running loop's `--stop` row is an
        # offer beside the iteration count and the goal text, not a gate in
        # front of them.
        "loop": ArgumentMode.OPTIONAL,
        # OPTIONAL like `/rename`: bare `/notifications` LISTS the unread
        # completions, so Enter on the word already answers and the space is an
        # offer of the one clearing form (`read`) rather than a gate in front of
        # it. Deliberately not REQUIRED — that would make the listing
        # unreachable from the completion path, which is the `/login` line this
        # sits on the other side of.
        "notifications": ArgumentMode.OPTIONAL,
    }
    # `/provider` was the third candidate and is deliberately not here: it takes
    # no argument at all — `_cmd_providers` ignores what follows it — so a list
    # would offer values the handler discards.
    assert next(c for c in SLASH_COMMANDS if c.name == "provider").arguments is ArgumentMode.NONE


# -- the band cannot outrun the gate ------------------------------------------


@pytest.mark.asyncio
async def test_the_band_and_the_gate_agree_through_every_route(config_dir: Path) -> None:
    """The invariant the owner's confusing frame is the argument for.

    Each route that can change the mode — the prompt's `A`, the command, both
    directions — is followed by asking the GATE what it does and the BAND what
    it says. They are set in one place precisely so this cannot drift; the test
    is what stops the next author reintroducing a second writer.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        gate = _gate(session)

        async def gate_runs_without_asking() -> bool:
            pending = asyncio.ensure_future(gate("bash", "run: echo hi"))
            # Wait for whichever terminal state the mode produces: an auto gate
            # returns immediately, an asking gate mounts a prompt. Polling both
            # replaces a fixed `pause(0.2)` that bet the slower one had settled.
            for _ in range(200):
                await pilot.pause()
                if pending.done() or app.query(ApprovalPrompt):
                    break
            if pending.done():
                return await pending
            # A prompt is on screen: the gate is armed. Answer it so the test
            # leaves nothing parked on a future.
            await pilot.press("n")
            await asyncio.wait_for(pending, 2)
            return False

        def band_is_alarmed() -> bool:
            """Whether the band's trailing cell is the disarmed-gate alarm.

            The segment is a bare `!` now — the session name took the words —
            so both scopes look identical here on purpose: the band's promise is
            "no tool will ask", and "until when" is `/approvals`' answer.
            """
            return _band(app).rstrip().endswith("!")

        assert await gate_runs_without_asking() is False
        assert band_is_alarmed() is False

        await _submit(pilot, app, "/approvals auto")
        assert await gate_runs_without_asking() is True
        assert band_is_alarmed() is True

        await _submit(pilot, app, "/approvals default auto")
        assert await gate_runs_without_asking() is True
        assert band_is_alarmed() is True

        await _submit(pilot, app, "/approvals ask")
        assert await gate_runs_without_asking() is False
        assert band_is_alarmed() is False


@pytest.mark.asyncio
async def test_the_live_prompt_is_untouched_by_the_default_machinery(
    config_dir: Path,
) -> None:
    """The in-flight gate is a SEPARATE mechanism and stays exactly as it was.

    Three behaviours pinned here because this change touched the mode they read:
    a question still mounts and still waits; `n` refuses that one tool and
    leaves the turn's next ask free to ask again; and the turn-scoped deny latch
    still drains a stopped turn's queued asks without a card. None of them is
    about the default — that is the point.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        gate = _gate(session)

        first = asyncio.ensure_future(gate("bash", "run: one"))
        await _wait_prompt(pilot, app)
        # The live question is the docked card; the transcript keeps the
        # RECEIPT once it is answered. Two widgets, two jobs.
        assert app.query(ApprovalPrompt), "the prompt no longer mounts"
        assert not first.done(), "the gate stopped waiting for an answer"
        await pilot.press("n")
        assert await asyncio.wait_for(first, 2) is False

        # …and one refusal is not a mode: the next ask still asks.
        second = asyncio.ensure_future(gate("write", "write: two"))
        await _wait_prompt(pilot, app)
        assert app.query(ApprovalPrompt)
        await pilot.press("y")
        assert await asyncio.wait_for(second, 2) is True
        # The turn-scoped latch: a stop drains what the stopped turn queued,
        # with no card, and that is still true with a saved default in play.
        await _submit(pilot, app, "/approvals default ask")
        receipts = len(app.query(ApprovalBlock))
        app._deny_queued_approvals()
        third = asyncio.ensure_future(gate("bash", "run: three"))
        assert await asyncio.wait_for(third, 2) is False
        await _settle(pilot)
        # No question was raised for the stopped turn's ask...
        assert not app.query(ApprovalPrompt), "a stopped turn's ask mounted a question"
        # ...and no receipt was written for a decision the user never made.
        # Counted rather than queried for emptiness: the two answered prompts
        # above left receipts and stay on screen, which is what a ledger is for.
        assert len(app.query(ApprovalBlock)) == receipts


@pytest.mark.asyncio
async def test_the_allow_all_key_reports_and_paints_like_the_command(
    config_dir: Path,
) -> None:
    """`A` on the prompt is the other route into auto, and it is session-scoped.

    It must not write a default — a keystroke answering one question cannot
    reasonably be a standing preference — but it must paint the same band the
    command does, through the same writer.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        await _wait_prompt(pilot, app)
        await pilot.press("A")
        assert await asyncio.wait_for(pending, 2) is True

        assert app._approve_all is True
        # One glyph for both scopes: the band says "no tool will ask", and the
        # keystroke's session-only scope is proved by the config below staying
        # unwritten rather than by a word in the band.
        assert _band(app).rstrip().endswith("!")
        assert _saved_mode(config_dir) is None


@pytest.mark.asyncio
async def test_a_stopped_session_settles_its_live_approval() -> None:
    """A stop must not leave a gate live enough to eat the next keystroke.

    Round-5 MAJOR-4: the viewer's stop handler retired the band and the tool
    cards but never settled the gate, so the docked prompt survived with its
    key routing intact. The user's next keystroke was then read as an ANSWER —
    the `y` of a typed word approved the tool — and the transcript kept a
    receipt saying the user approved a command nobody approved. The owner's
    own `/stop` already denies queued approvals; this is the viewer path
    catching up to it.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        gate = _gate(session)
        pending = asyncio.ensure_future(gate("bash", "bash: rm -rf /tmp/x"))
        await _wait_prompt(pilot, app)
        assert app.query(ApprovalPrompt), "the gate never mounted its question"

        app._stopped_session_id = session.session_id
        app._on_watched_session_stopped()
        await _settle(pilot)

        # The card is gone and the gate answered NO by the stop, not by a user.
        assert not app.query(ApprovalPrompt), "the stop left the question live"
        assert await asyncio.wait_for(pending, 2) is False
        # A keystroke after the stop is text, never an answer.
        receipts_before = len(app.query(ApprovalBlock))
        await pilot.press("y")
        await _settle(pilot)
        assert len(app.query(ApprovalBlock)) == receipts_before
