"""``/network`` — one vocabulary, three readers, and a CLI behind all of them.

The design (``docs/design/mesh-ui.md`` §1.1) makes three claims this file exists
to hold:

1. **The picker's words and the handler's words are ONE list.** Drift between the
   word a row offers and the word the handler accepts is a defect class this repo
   has paid for, so the picker's rows are asserted equal to ``NETWORK_SUBCOMMANDS``
   rather than eyeballed against the table in the design.
2. **``/network`` is FRONTEND-LOCAL.** Every verb is a fact about THIS device's
   mesh, and the family must be answerable with no runtime at all (a device whose
   peers are unreachable is the device that needs it) — so the classification is
   asserted here, where the registry's own complement rule can see it.
3. **The destructive verbs need their word.** ``disconnect``, ``member rm``,
   ``rm`` and ``panic`` run NOTHING without a typed confirmation, and ``panic``'s
   token is the network's own NAME rather than ``yes`` because a panic is not
   undoable by the same command. Each is driven through the real editor and the
   real submit handler — the reported path — with the CLI stubbed at
   ``run_network``, so the assertion is about what the SURFACE decided to run.

The CLI itself is not stubbed at the process boundary in the drives that prove it
(``tests/unit/network`` and the PR's own before/after runs do that): here the
question is which argv the handler spells, which a fake answer answers exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.session.frontend_state import _FRONTEND_LOCAL_SLASHES
from local_operator.slash_commands import (
    NETWORK_SUBCOMMANDS,
    command_argument_is_used,
    command_argument_refusal,
    network_subcommand_rows,
    slash_command_for,
)
from local_operator.tui.network_cli import NetworkRun

# ---------------------------------------------------------------------------
# the registry: one list, three readers
# ---------------------------------------------------------------------------


def test_the_registry_entry_declares_the_vocabulary_and_no_destination() -> None:
    spec = slash_command_for("/network")
    assert spec is not None
    assert spec.name == "network"
    assert spec.subcommands == NETWORK_SUBCOMMANDS
    # `echo=False`: the listing or the receipt IS the answer.
    assert spec.echo is False
    # No desktop surface in this pass — see `mesh-ui.md` §2.7/§2.8. An advertised
    # destination with no adapter behind it is the offered-but-dead failure the
    # `/mobile` entry documents.
    assert spec.desktop_destination == ""


def test_the_picker_offers_exactly_the_words_the_handler_accepts() -> None:
    """Claim 1, mechanically. Both halves through their own public reader."""
    handler_words = set(NETWORK_SUBCOMMANDS)
    picker_words = {word for word, _help in network_subcommand_rows()}
    assert picker_words == handler_words, "the picker and the handler disagree"
    # The help line is part of the contract rather than decoration: a word with no
    # description is a row that teaches nothing, so the reader is total over the
    # vocabulary (it raises on a missing key) and this asserts the same fact from
    # the other side.
    assert all(help_text for _word, help_text in network_subcommand_rows())
    assert len(NETWORK_SUBCOMMANDS) == len(set(NETWORK_SUBCOMMANDS)), "a duplicate verb"


def test_network_is_frontend_local() -> None:
    """Claim 2: this device's own mesh state, never the runtime's."""
    assert "network" in _FRONTEND_LOCAL_SLASHES
    # The installation verbs are NOT in the vocabulary: a composer row that boots
    # out the operator's relay, or deletes this device's identity keypair, is the
    # one-keystroke class of mistake the incident verbs' confirmation refuses.
    for installation in ("serve", "start", "stop", "restart", "uninstall"):
        assert installation not in NETWORK_SUBCOMMANDS
    # Nor `pool`: `lop network` has no such parser (compute-pool is not
    # implemented), and a row for a verb the CLI lacks is offered-but-broken.
    assert "pool" not in NETWORK_SUBCOMMANDS


def test_the_shape_accepts_a_subcommand_and_keeps_a_sentence_prose() -> None:
    spec = slash_command_for("/network")
    assert spec is not None
    assert command_argument_is_used(spec, "ls")
    assert command_argument_is_used(spec, "status")
    assert command_argument_is_used(spec, "doctor devon-laptop")
    # THE BOUNDARY IS THE THIRD TOKEN, and this is the shape's one real cost
    # (`mesh-ui.md` §2.8.3): the predicate is MCP's — at most two tokens, the
    # second name-shaped — so a family verb that needs its own two arguments,
    # `member rm <network> <device>`, is PROSE to the messages endpoint. It is
    # asserted here rather than left as a surprise, and it is harmless while no
    # desktop surface queries this family: the TUI dispatches on the first word,
    # so the command runs either way.
    assert not command_argument_is_used(spec, "member rm net dev")
    assert not command_argument_is_used(spec, "ls and then tell me a story")
    assert not command_argument_is_used(spec, "disconnect?! please")


def test_the_route_refusal_names_this_familys_words_not_mcps() -> None:
    """A `/network` misuse must not be told to use the MCP setup form."""
    spec = slash_command_for("/network")
    assert spec is not None
    refusal = command_argument_refusal(spec, "git push")
    assert refusal is not None
    assert "ls" in refusal and "panic" in refusal
    assert "MCP" not in refusal


def test_argparse_does_not_own_a_verb_the_cli_lacks() -> None:
    """Every word in the vocabulary is a parser this tree actually has.

    The vocabulary is a front end, so a word with no parser behind it would be a
    row that runs ``lop network <unknown>`` and answers with argparse's usage
    text — the "offered but broken" shape, caught here rather than by a user.

    `subcommands_of` rather than walking `parser._subparsers` directly: argparse
    types that attribute `Optional` and an action's `choices` as
    `Iterable[Any] | None`, so the direct walk is four non-narrowable sites (the
    checkout's own `pyright` gate flagged them). The helper proves the mapping is
    the dict `add_subparsers` built once, in one place, for every test module
    that asks — including this one.
    """
    import argparse

    from local_operator.network.cli import add_parser
    from tests.unit.network import conftest as net_fixtures

    parser = argparse.ArgumentParser(prog="lop", add_help=False)
    subparsers = parser.add_subparsers(dest="command")
    add_parser(subparsers)
    network = net_fixtures.subcommands_of(parser)["network"]
    available = set(net_fixtures.subcommands_of(network))
    # `new` is the user-facing word and `init` is the CLI's verb (the receipt says
    # which); `rename`/`rm` map straight through.
    mapping = {"new": "init"}
    for word in NETWORK_SUBCOMMANDS:
        assert mapping.get(word, word) in available, word


# ---------------------------------------------------------------------------
# the handler: what it runs, and what it refuses to run
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """A ``run_network`` stand-in: records argv, answers with a fixed line."""

    calls: list[list[str]] = field(default_factory=list)
    answer: str = "ok"
    returncode: int = 0

    def __call__(self, args: list[str], **kwargs: Any) -> NetworkRun:
        self.calls.append(list(args))
        return NetworkRun(tuple(args), self.returncode, stdout=self.answer + "\n")

    @property
    def argv(self) -> list[str]:
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


def _isolated_network_store(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    """Give ``_network_records`` a fixed set of records, with no disk behind it."""

    class _Record:
        def __init__(self, network_id: str, name: str) -> None:
            self.network_id = network_id
            self.name = name

    records = [_Record(f"n_{index:02d}", name) for index, name in enumerate(names)]
    monkeypatch.setattr(
        "local_operator.network.store.list_networks", lambda root=None: list(records)
    )


async def _boot(pilot: Any, app: Any) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot: Any, text: str) -> None:
    """Type ``text`` as real keystrokes, so the caret lands where a user's does."""
    await pilot.press(*("space" if char == " " else char for char in text))


async def _submit(pilot: Any, app: Any, text: str) -> None:
    from local_operator.tui.widgets.editor import Editor

    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _app_fixture() -> Any:
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


def _notices(app: Any) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _painted(app: Any) -> str:
    """The whole rendered frame as text — receipts are painted, not stored.

    ``RichBlock`` carries a rich renderable rather than a string attribute, so the
    frame is the only place its line exists: reading it from the compositor is
    what makes the assertion about what the user READS (the same reader
    ``test_slash_echo`` uses).
    """
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


#: The cells a row may occupy in the narrowest card the band draws: `_card_width()`
#: at 56x20, the smallest size `_BAND_SIZES` pins (43, measured on the frame below
#: rather than recomputed from the percentage). It is named because two tests hold
#: the first-run block to it and the wording cannot be allowed to drift past it.
_NARROWEST_CARD = 43


def _visible_body_rows(app: Any, screen: Any) -> list[str]:
    """The panel's body rows AS PAINTED — clipped to the scroll viewport.

    The distinction this exists for: ``render_lines_for_test()`` returns the block
    the model holds, and the block holds these rows at every size — so a check
    against it passes on a frame that has scrolled the device cue out of sight
    (UX round 1, U5). The compositor's strips are the only place "is it on the
    screen" is answerable, and the SCROLL's region is the rectangle of them that
    is on the screen: the body widget's own region is the whole block, so slicing
    by it walks past the viewport and reads the footer row that is docked under
    it.

    Trailing padding and the vertical scrollbar glyph are removed, because both
    are painted OVER the row rather than part of it: every row is padded to the
    body's width, and the bar's thumb is drawn in that padding. What is left is
    the text as the reader sees it, which is what the assertions compare.
    """
    rows = _painted(app).split("\n")
    scroll = screen.query_one("#network-scroll")
    left = screen.query_one("#network-body").region.x
    window = rows[scroll.region.y : scroll.region.y + scroll.region.height]
    return [row[left:] if len(row) > left else "" for row in window]


#: Textual's vertical scrollbar thumb, one glyph per position. Stripped from a
#: painted row before it is compared: the bar is drawn in the row's padding.
_SCROLLBAR_GLYPHS = "▁▂▃▄▅▆▇█"


def _row_text(row: str) -> str:
    """A painted body row as text: no padding, no scrollbar overlay."""
    return row.rstrip().rstrip(_SCROLLBAR_GLYPHS).rstrip()


@pytest.mark.asyncio
async def test_read_verbs_run_the_cli_with_the_argv_the_design_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each read verb, through the real editor, asserting the EXACT argv."""
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network peers")
        await app.workers.wait_for_complete()
        assert run.argv == ["peers"]

        run.calls.clear()
        await _submit(pilot, app, "/network log")
        await app.workers.wait_for_complete()
        assert run.argv == ["log", "--limit", "20"]

        run.calls.clear()
        await _submit(pilot, app, "/network doctor devon-laptop")
        await app.workers.wait_for_complete()
        assert run.argv == ["doctor", "--peer", "devon-laptop"]

        run.calls.clear()
        await _submit(pilot, app, "/network show devmesh")
        await app.workers.wait_for_complete()
        assert run.argv == ["show", "devmesh"]

        run.calls.clear()
        await _submit(pilot, app, "/network invite")
        await app.workers.wait_for_complete()
        assert run.argv == ["invite", "--role", "drive"]

        # The session plane (review round 4, MINOR 3): the ONE way the session
        # `/new remote <peer>` creates is observable and controllable from the
        # composer. The tail is passed through, because the same CLI verb lists,
        # creates and acts on a session.
        run.calls.clear()
        await _submit(pilot, app, "/network sessions --peer radiant-m4")
        await app.workers.wait_for_complete()
        assert run.argv == ["sessions", "--peer", "radiant-m4"]

        # A NAME IS A SENTENCE (review round 4, MINOR 2): both verbs take a
        # free-text positional, and slicing the tail to its first word silently
        # created a network called `My`. The whole tail goes in, and the arity
        # check refuses a SHORT command without refusing a longer name.
        run.calls.clear()
        await _submit(pilot, app, "/network new My Fancy Net")
        await app.workers.wait_for_complete()
        assert run.argv == ["init", "My Fancy Net"]

        run.calls.clear()
        await _submit(pilot, app, "/network rename devmesh My Fancy Name")
        await app.workers.wait_for_complete()
        assert run.argv == ["rename", "devmesh", "My Fancy Name"]


@pytest.mark.asyncio
async def test_approval_verbs_run_the_cli_with_the_argv_the_design_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approval cards from the composer, one hop onto the CLI's own verbs.

    Slice (e)'s TUI half (remote-onboarding §2.3): the badge reads (`list`,
    `show`) and the two decisions (`approve`, `deny`) are the SAME verbs every
    other surface runs — the store, the signature gate and the audit line are
    the CLI's, and this surface decides only the argv (plus the gesture-shaped
    budget `approve` gets).
    """
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network approvals")
        await app.workers.wait_for_complete()
        assert run.argv == ["approvals", "list"]

        run.calls.clear()
        await _submit(pilot, app, "/network approvals list")
        await app.workers.wait_for_complete()
        assert run.argv == ["approvals", "list"]

        run.calls.clear()
        await _submit(pilot, app, "/network approvals show a_123")
        await app.workers.wait_for_complete()
        assert run.argv == ["approvals", "show", "a_123"]

        run.calls.clear()
        await _submit(pilot, app, "/network approvals approve a_123")
        await app.workers.wait_for_complete()
        assert run.argv == ["approvals", "approve", "a_123"]

        run.calls.clear()
        await _submit(pilot, app, "/network approvals deny a_123")
        await app.workers.wait_for_complete()
        assert run.argv == ["approvals", "deny", "a_123"]


@pytest.mark.asyncio
async def test_approval_words_this_surface_does_not_carry_read_the_usage_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`request`/`run` stay the agent's path; a half-typed sub-verb reads the
    surface's own usage line, not argparse's sentence from one hop down."""
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network approvals request --host 10.0.0.9")
        await pilot.pause()
        assert run.calls == []
        assert any("Use /network approvals list" in text for text in _notices(app))

        await _submit(pilot, app, "/network approvals run")
        await pilot.pause()
        assert run.calls == []

        await _submit(pilot, app, "/network approvals show")
        await pilot.pause()
        assert run.calls == []

        await _submit(pilot, app, "/network approvals list extra")
        await pilot.pause()
        assert run.calls == []


@pytest.mark.asyncio
async def test_the_receipt_is_the_clis_own_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output is a BLOCK carrying the CLI's line, not a paraphrase of it."""
    run = _Recorder(answer="the relay is not running; start it with `lop network start`")
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network peers")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "start it with `lop network start`" in _painted(app)


@pytest.mark.asyncio
async def test_join_names_the_cli_instead_of_half_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairing needs a terminal; the surface says so and runs NOTHING.

    ``network/cli.py::_cmd_join`` shows this device's own code and reads the other
    device's code from stdin — the SAS ceremony has a human in the middle by
    design, and ``--sas-stdin`` exists only behind the test-mode environment
    variable. A subprocess with no stdin cannot pair, so the honest answer is the
    command to run: the same call the installation verbs get.
    """
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network join @some/token")
        await app.workers.wait_for_complete()
        assert run.calls == [], "join must not spawn a pairing subprocess"
        assert any("lop network join @some/token" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_network_name_with_a_space_is_addressable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q-R10-4: the addressing verbs resolve their target as a LONGEST PREFIX.

    ``/network new`` takes the WHOLE tail as a name, so a network called
    ``Gamma Mesh`` is creatable from this surface — and the first-token rule
    answered "this device is not in a network called 'Gamma'" for the network
    the sibling verb had just made, while ``lop network rename "Gamma Mesh" …``
    renamed it. Two networks are staged on purpose: ``Gamma`` beside
    ``Gamma Mesh`` is the case where a shorter-prefix-first rule would silently
    rename the wrong one.
    """
    _isolated_network_store(monkeypatch, ("Gamma Mesh", "Gamma"))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network rename Gamma Mesh Gamma Renamed")
        await app.workers.wait_for_complete()
        assert run.calls[-1] == ["rename", "Gamma Mesh", "Gamma Renamed"]

        await _submit(pilot, app, "/network rename Gamma Gamma Prime")
        await app.workers.wait_for_complete()
        assert run.calls[-1] == ["rename", "Gamma", "Gamma Prime"]

        await _submit(pilot, app, "/network rm Gamma Mesh yes")
        await app.workers.wait_for_complete()
        assert run.calls[-1] == ["rm", "Gamma Mesh"]

        await _submit(pilot, app, "/network disconnect Gamma Mesh yes")
        await app.workers.wait_for_complete()
        assert run.calls[-1] == ["disconnect", "Gamma Mesh"]

        await _submit(pilot, app, "/network member rm Gamma Mesh d_peer yes")
        await app.workers.wait_for_complete()
        assert run.calls[-1] == ["member", "rm", "Gamma Mesh", "d_peer"]


@pytest.mark.asyncio
async def test_an_unknown_verb_refuses_without_running_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network install")
        await app.workers.wait_for_complete()
        assert run.calls == []
        assert any("Use /network" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_disconnect_rehearses_then_runs_on_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 3: the rehearsal names the act, and only the exact word runs it."""
    _isolated_network_store(monkeypatch, ("devmesh",))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network disconnect")
        await app.workers.wait_for_complete()
        assert run.calls == [], "a bare disconnect must not leave the network"
        rehearsals = [text for text in _notices(app) if "To confirm" in text]
        assert rehearsals, _notices(app)
        assert "/network disconnect yes" in rehearsals[-1]

        # A wrong word is refused, and still runs nothing.
        await _submit(pilot, app, "/network disconnect ok")
        await app.workers.wait_for_complete()
        assert run.calls == []

        await _submit(pilot, app, "/network disconnect yes")
        await app.workers.wait_for_complete()
        assert run.argv == ["disconnect"]


@pytest.mark.asyncio
async def test_panic_demands_the_networks_name_not_a_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The design's typed token, and the reason it is not ``yes``.

    A panic rotates the network's secret: every OTHER device must be re-invited
    before it can come back, which is not something the same command can undo. A
    confirmation that any word would satisfy is not a confirmation for that act.
    """
    _isolated_network_store(monkeypatch, ("devmesh",))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network panic n_00")
        await app.workers.wait_for_complete()
        assert run.calls == []
        assert any("/network panic n_00 devmesh" in text for text in _notices(app))

        await _submit(pilot, app, "/network panic n_00 yes")
        await app.workers.wait_for_complete()
        assert run.calls == [], "`yes` is not the token for a panic"
        assert any("Type the network's name" in text for text in _notices(app))

        await _submit(pilot, app, "/network panic n_00 devmesh")
        await app.workers.wait_for_complete()
        assert run.argv == ["panic", "n_00"]


@pytest.mark.asyncio
async def test_member_rm_is_rehearsed_and_takes_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolated_network_store(monkeypatch, ("devmesh",))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/network member rm devmesh d_abc")
        await app.workers.wait_for_complete()
        assert run.calls == []
        assert any("rotates the network secret" in text for text in _notices(app))

        await _submit(pilot, app, "/network member rm devmesh d_abc yes")
        await app.workers.wait_for_complete()
        assert run.argv == ["member", "rm", "devmesh", "d_abc"]


@pytest.mark.asyncio
async def test_member_without_rm_names_the_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network member show devmesh")
        await app.workers.wait_for_complete()
        assert run.calls == []
        assert any("member rm <network> <device>" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_named_network_runs_without_a_rehearsal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The named form is unambiguous, so ``rm`` runs on the file it names."""
    _isolated_network_store(monkeypatch, ("devmesh", "homelab"))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network rm devmesh yes")
        await app.workers.wait_for_complete()
        # Ambiguity is only a problem for the BARE forms; a named network is
        # unambiguous and runs.
        assert run.argv == ["rm", "devmesh"]


@pytest.mark.asyncio
async def test_the_panel_opens_from_the_bare_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/network`` and ``/network status`` push the screen — one surface, two doors."""
    _isolated_network_store(monkeypatch, ("devmesh",))
    _isolate_panel(monkeypatch)
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network")
        from local_operator.tui.widgets.network_panel import NetworkScreen

        assert isinstance(app.screen, NetworkScreen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any("Mesh networks" in line for line in app.screen.render_lines_for_test())
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, NetworkScreen)


def _isolate_panel(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...] = ("devmesh",)) -> None:
    """Point the PANEL at a fixed first frame and at nothing that spawns.

    The panel is a second reader of both halves, so both are redirected:

    * ``capture_local`` — its first frame reads this device's identity file, its
      relay record and its member lists off disk. A test must not paint whatever
      the machine happens to hold, and it must not depend on a store it does not
      own.
    * ``network_panel.run_network`` — the screen's own worker calls the CLI. Left
      alone, an assertion about what the SCREEN ran would spawn real
      ``lop network`` subprocesses from a unit test.

    The app's own ``run_network`` is patched separately by each test, so the two
    recorders never see each other's calls.
    """
    from local_operator.tui.widgets.network_panel import NetworkEntry, NetworkLocal

    monkeypatch.setattr(
        "local_operator.tui.widgets.network_panel.capture_local",
        lambda root=None: NetworkLocal(
            device_id="d_self",
            device_name="this-mbp",
            identity_present=True,
            relay_state="",
            networks=[
                NetworkEntry(
                    network_id=f"n_{index:02d}",
                    name=name,
                    epoch=1,
                    role="admin",
                    members=1,
                    trust="active",
                )
                for index, name in enumerate(names)
            ],
        ),
    )
    monkeypatch.setattr(
        "local_operator.tui.widgets.network_panel.run_network",
        lambda args, **kwargs: NetworkRun(tuple(args), 0, stdout="", stderr=""),
    )


@pytest.mark.asyncio
async def test_the_picker_arm_offers_the_vocabulary_through_the_real_editor() -> None:
    """The rows behind ``/network `` are that list, read off the real editor.

    The reader being one list is asserted above; this is the other half — that the
    ARM uses it, so what a user sees is that vocabulary rather than a second list
    assembled somewhere else.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _app_fixture()
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        # REAL KEYSTROKES, not a text assignment: the argument list opens from the
        # caret being past the terminating space, and `editor.text = "/network "`
        # leaves the caret where it was — so assigning the buffer would test a
        # state no typing user reaches.
        await _type(pilot, "/network ")
        editor = app.query_one(Editor)
        await pilot.pause()
        assert editor._picker.is_open(), "a space after /network must open its list"
        painted = "\n".join(row.plain for row in editor._picker.render_rows(70))
        for word in NETWORK_SUBCOMMANDS:
            assert word in painted, word
        # And the help line travels with the row: a picker of bare words teaches
        # nothing about which one to press.
        assert "Networks this device is in" in painted


@pytest.mark.asyncio
async def test_the_panels_panic_key_fills_the_composer_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shift+P`` hands the typed command over UNSUBMITTED.

    The panel selects and the command confirms: a screen that both chose a network
    and fired the revoke would be the one-step accident the typed confirmation
    exists to prevent, so this asserts the buffer, not an effect.
    """
    _isolated_network_store(monkeypatch, ("devmesh",))
    _isolate_panel(monkeypatch)
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network")
        await app.workers.wait_for_complete()
        await pilot.pause()
        from local_operator.tui.widgets.editor import Editor

        await pilot.press("shift+p")
        await pilot.pause()
        assert app.query_one(Editor).text == "/network panic n_00"
        assert run.calls == [], "the panel never runs a destructive verb itself"


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "cut"), [((100, 30), False), ((64, 30), True)])
async def test_the_panel_footer_says_when_it_has_been_cut(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int], cut: bool
) -> None:
    """U8: the footer fits, or it says it does not.

    Measured before the fix, at 64x30: the line ended after ``shift+p`` with no
    ellipsis and no fallback, so the only hints for refresh, copy and panic were
    gone exactly where a cramped terminal wanted them — and nothing on the frame
    said more existed. A cut line that does not say it was cut is the defect; a
    cut line that does is the fix.
    """
    _isolated_network_store(monkeypatch, ("devmesh",))
    _isolate_panel(monkeypatch)
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network")
        await app.workers.wait_for_complete()
        await pilot.pause()
        hint = app.screen.query_one("#network-hint")
        text = str(hint.render())
        if cut:
            assert text.endswith("…"), text
            assert "ctrl+r copy" not in text, text
        else:
            assert text.endswith("ctrl+r copy"), text
            assert "…" not in text, text


@pytest.mark.asyncio
async def test_the_panels_disconnect_key_names_the_selected_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolated_network_store(monkeypatch, ("devmesh",))
    _isolate_panel(monkeypatch)
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network")
        await app.workers.wait_for_complete()
        await pilot.pause()
        from local_operator.tui.widgets.editor import Editor

        await pilot.press("d")
        await pilot.pause()
        assert app.query_one(Editor).text == "/network disconnect n_00"
        assert run.calls == []


# ---------------------------------------------------------------------------
# design round 2, D20 — the loading frame's one relay fact
# ---------------------------------------------------------------------------


def _panel_local() -> Any:
    from local_operator.tui.widgets.network_panel import NetworkLocal

    return NetworkLocal(
        device_id="d_self", device_name="this-mbp", identity_present=True, relay_state=""
    )


def test_the_pending_frame_does_not_call_the_relay_not_running() -> None:
    """MEASURED ON THE LOADING FACE, which is the one it was filed against.

    ``relay: not running on this device`` painted under This device while the
    Relay section seven lines below painted its own pending line — the same frame
    contradicting itself about the one fact both lines are about. While the live
    answer is out, the disk record is the input the answer is about, not an
    answer, so the pending sentence is the same on both lines.

    The pending word is ONE word since design round 3's D23 (``checking``; it was
    ``asking the relay…`` on this section and ``relay: checking…`` on the device
    line), which is why the two assertions below are about the same verb.
    """
    from local_operator.tui.widgets.network_panel import build_network_report

    text = build_network_report(_panel_local()).plain
    assert "relay: checking…" in text, text
    assert "not running on this device" not in text, text
    assert "  checking…" in text, text


def test_a_section_does_not_promise_to_check_a_relay_known_to_be_down() -> None:
    """The other half of D20: once the answer IS in, the sections inherit it.

    ``checking with the relay…`` is a promise the frame has already broken when
    the Relay section's own answer says the relay is not running — there is
    nothing to check with — and the Peers section then also has to paint the rows
    it already holds rather than nothing at all.
    """
    import json

    from local_operator.tui.widgets.network_panel import NetworkRun, NetworkScreen

    screen = NetworkScreen(_panel_local())
    screen.status_run = NetworkRun(
        ("status",),
        0,
        stdout=json.dumps({"installed": True, "identity_present": True, "relay_running": False}),
    )
    text = "\n".join(screen.render_lines_for_test())
    assert "checking with the relay…" not in text, text
    assert "from this device's records" in text, text
    assert "no peers yet" in text, text

    # AND THAT ROW NAMES A CONSEQUENCE, NOT A COMMAND (design round 1, D1): it used
    # to say `/network invite mints a token`, which is the step the Networks block
    # above already names — one remedy under two headings, the shape this PR's own
    # `test_the_unpaired_empty_state_teaches_the_whole_pairing_path` files as a
    # defect one block away. Asserted HERE rather than there because the row only
    # paints once the relay has answered: the pending frame's Peers block is its
    # `checking with the relay…` line and nothing else.
    peers = text.split("Peers", 1)[1]
    assert "/network invite" not in peers, peers


#: The sizes the unpaired empty state is expected to fit WHOLE. The card is 83
#: cells at 100x30, 81 at 98x30, 79 at 96x30, 68 at 84x16, 65 at 80x24 and 43 at
#: 56x20, and no row of the block is wider than 40 — the longest of them being the
#: pairing-sequence row. The This-device row held that title at 41 until review
#: round 2's R2-1: its `<name>` placeholder came back and the `mints it` clause
#: paid for it, which puts the row at 39 — two cells NARROWER than the incomplete
#: wording it replaced.
#:
#: 56x20 IS IN THIS LIST NOW and was not before (UX round 1, U5): the block's rows
#: were 62 and 60 cells, so every line of it wrapped at 43 — which is what put the
#: device cue below the fold at the size a laptop in a split pane actually
#: reports. At 41 cells the block fits the narrowest band, so the claim is made
#: there rather than excluded from it. The two sizes below this one (50x18, body
#: 38; 40x20, body 29) cannot hold a wording that names both commands at all, and
#: `_UNPAIRED_HANG_SIZES` is the property asserted for them instead.
_UNPAIRED_FIT_SIZES = ((100, 30), (98, 30), (96, 30), (84, 16), (80, 24), (56, 20))

#: The sizes at which the pairing SEQUENCE has to be on the screen rather than
#: merely in the body. MEASURED, per size, as the scroll viewport and the row the
#: device cue lands on: 16 rows (cue at 6) at 100x30, 98x30 and 96x30, 11 at
#: 80x24 (6), 8 at 60x20 (6) and 8 at 56x20 (6). The two sizes that are NOT here
#: are excluded by measurement, not by convenience: 84x16's viewport is FOUR rows
#: (the title block and the footer are docked inside sixteen rows, so the body
#: gets 'This device', its identity row and the relay row), and 50x18's is six
#: with the block itself at 38 cells — neither can show the sequence whatever it
#: says, and the property asserted for them is the hang instead.
_UNPAIRED_FOLD_SIZES = ((100, 30), (98, 30), (96, 30), (80, 24), (60, 20), (56, 20))

#: The two sizes narrower than any wording that names both commands (`/network
#: new <name>` and `/network invite` with the device cue are 39-40 cells with
#: their lead, against a body of 38 and 29 here). There the block wraps — 15 and
#: 17 body rows — so the property is that the WRAP keeps the row's indent.
_UNPAIRED_HANG_SIZES = ((50, 18), (40, 20))

#: The three of those whose BASE frame paints the empty state without a vertical
#: scrollbar, and therefore the three the change must leave scrollbar-free: at
#: 84x16 and 80x24 the block already scrolled before this PR (measured on a
#: throwaway worktree at ``origin/main`` — body 15 rows, bar true), so a bar
#: appearing there is not this change's doing and is not asserted away.
_UNPAIRED_SCROLLBAR_FREE = ((100, 30), (98, 30), (96, 30))


@pytest.mark.asyncio
async def test_the_unpaired_empty_state_fits_its_card_at_every_legible_size() -> None:
    """EVERY ROW OF THE FIRST-RUN FRAME HAS TO FIT THE CARD IT IS PAINTED INTO.

    This is the assertion whose absence let an 82-cell row reach review (review
    round 1, MAJOR, and MINOR 3 asks for it by name). The row was written against
    the 100x30 frame, where the body is 83 cells; the panel is drawn across
    ``_BAND_SIZES``, where it is 81 at 98x30 and 79 at 96x30 — so it wrapped at two
    sizes inside that list, grew the body to 17 rows in a 16-row viewport and put
    the vertical scrollbar back on, which is the very thing the empty state had
    just been arranged to avoid. Order was asserted and fit was not, and a wrapped
    row is exactly what an order-only assertion cannot see.

    Measured on the PAINTED widget rather than the model: ``render_lines_for_test()``
    returns unwrapped lines at every size, so a model-side check passes on a row
    that paints as two. The bound is the body's own ``size.width`` — the frame
    under test — rather than a constant copied out of it, and the scroll region's
    virtual height is the second half of the same claim, because a row that wraps
    is a row that makes the region overflow.
    """
    from local_operator.tui.widgets.network_panel import NetworkLocal, NetworkScreen

    local = NetworkLocal(device_id="", device_name="", identity_present=False, relay_state="")
    for size in _UNPAIRED_FIT_SIZES:
        app = _app_fixture()
        async with app.run_test(size=size) as pilot:
            screen = NetworkScreen(local)
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()

            body = screen.query_one("#network-body")
            too_wide = [
                row for row in screen.render_lines_for_test() if cell_len(row) > body.size.width
            ]
            assert not too_wide, (size, body.size.width, too_wide)

            scroll = screen.query_one("#network-scroll")
            if size in _UNPAIRED_SCROLLBAR_FREE:
                assert scroll.virtual_size.height <= scroll.size.height, (
                    size,
                    scroll.virtual_size,
                    scroll.size,
                )
                assert not scroll.show_vertical_scrollbar, (size, scroll.virtual_size, scroll.size)


def test_every_unpaired_row_fits_the_narrowest_card_the_band_draws() -> None:
    """EVERY ROW OF THE FIRST-RUN BLOCK FITS THE 43 CELLS OF THE NARROWEST BAND.

    WHAT THIS TESTED BEFORE, and why the replacement is stronger: the old
    invariant was relative — no row wider than the Networks line they shared at
    62 cells — so it let the block sit at a threshold that wrapped at 56x20 and
    said nothing about the size a split pane reports. The budget is now the
    narrowest card the panel is drawn in (43 cells, measured off the frame), the
    rows are named here WHERE THE WORDS ARE so a reword cannot quietly grow past
    it, and the failure it pins is unchanged: a row written against the 100-column
    frame that no narrower size can hold.

    It is a model-side check and it is honest about that — `cell_len` of the row
    as built, not of the frame. The FOLD is what the painted test below measures;
    this one states the budget that makes it possible.
    """
    from local_operator.tui.widgets.network_panel import (
        NetworkLocal,
        build_network_report,
    )

    local = NetworkLocal(device_id="", device_name="", identity_present=False, relay_state="")
    # At the narrowest card, not at the default: a narrower width would PRE-BREAK
    # the rows (that is `_hanging_row`'s job) and this test is about how wide one
    # line of the block is when it is not broken at all.
    rows = [
        row
        for row in build_network_report(local, width=_NARROWEST_CARD).plain.splitlines()
        if row.strip()
    ]
    identity = next(row for row in rows if "no identity yet" in row)
    networks = next(row for row in rows if "no networks yet" in row)
    sequence = next(row for row in rows if "join on the peer" in row)
    assert _NARROWEST_CARD == 43, _NARROWEST_CARD
    for row in (identity, networks, sequence):
        assert cell_len(row) <= _NARROWEST_CARD, (cell_len(row), row)


@pytest.mark.asyncio
async def test_the_pairing_sequence_survives_the_fold_at_every_size_that_can_hold_it() -> None:
    """U5, AT THE FOLD: the sequence has to be ON THE SCREEN, not in the body.

    Measured on the real frame before the fix (56x20: body 43 cells, viewport
    eight rows): the two teaching rows were 62 and 60 cells, wrapped to four
    lines inside that viewport, and the device cue — the half the row exists for,
    `on the peer` — was the first row PAST the fold, on a frame whose body gave
    no sign it scrolled. At 39 and 40 cells the block went 22 -> 19 virtual rows
    and the cue paints at every size in `_UNPAIRED_FOLD_SIZES`, 56x20 included.

    Asserted on the COMPOSITOR'S strips — the painted frame — because the body
    holds these rows at every size: a model-side reading of the same block passes
    on a frame that clips it, which is exactly how this reached review. The
    assertion is the presence of the cue in the body's own visible rows, so a row
    that exists and is scrolled out of the viewport fails it.
    """
    from local_operator.tui.widgets.network_panel import NetworkLocal, NetworkScreen

    local = NetworkLocal(device_id="", device_name="", identity_present=False, relay_state="")
    for size in _UNPAIRED_FOLD_SIZES:
        app = _app_fixture()
        async with app.run_test(size=size) as pilot:
            screen = NetworkScreen(local)
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()
            visible = _visible_body_rows(app, screen)
            assert any("join on the peer" in row for row in visible), (size, visible)
            assert any("/network new" in row for row in visible), (size, visible)
            assert any("/network invite" in row for row in visible), (size, visible)


def test_an_overlong_row_breaks_on_words_and_keeps_its_lead() -> None:
    """The helper the two frame tests rest on, read directly.

    ``_hanging_row`` is the one place a prose row is broken now, so its two
    promises are pinned here rather than inferred from a frame: every line after
    the first carries the row's lead, and the sentence survives the break intact —
    no word is cut and none is lost. The second is the half that separates it from
    :func:`_indented_value`, which cuts on exact CELLS because the value it breaks
    has to re-join to the string it came from; a sentence that broke "creates"
    across two rows would re-join with a space that was never in it.
    """
    from local_operator.tui.widgets.network_panel import _hanging_row

    row = _hanging_row("  ", "no networks yet — /network new <name>", 30)
    lines = row.split("\n")
    assert len(lines) > 1, lines
    assert all(line.startswith("  ") for line in lines), lines
    assert " ".join(line.strip() for line in lines) == "no networks yet — /network new <name>"
    # A row that fits comes back untouched, so applying it is never a change of
    # wording — which is what lets the block keep its one-cell budget stable.
    assert _hanging_row("  ", "no networks yet — /network new <name>", 43) == (
        "  no networks yet — /network new <name>"
    )
    # …and the cell it keeps clear at the end is the SCROLLBAR's, so a row written
    # to the card exactly still fits the body the card is painted over: at 50x18
    # the card is 39 and the body 38, and a row of 39 would come back re-wrapped.
    assert _hanging_row("  ", "no networks yet — /network new <name>", 39) == (
        "  no networks yet — /network new\n  <name>"
    )


@pytest.mark.asyncio
async def test_a_row_that_wraps_below_the_band_floor_keeps_its_indent() -> None:
    """U5's first half, at the two sizes whose card cannot hold the whole block.

    Below the fold sizes the block still wraps — the body is 38 cells at 50x18 and
    29 at 40x20, against rows of 39-41 — and what the round measured there was
    that the continuation landed at column 0, flush with the section headers, so
    `mints it` and `<name> creates one` read as rows of their own. The mechanism is
    now explicit (`_hanging_row` breaks each sentence itself, against the same
    width the caller measures its rows in), and this is the assertion that ties it
    to the frame: the identity row — the block's FIRST, so its wrapped lines are
    inside the viewport at both sizes — is read back out of the painted frame, and
    every line after its first has to carry the row's own two cells. It fails on
    the pre-fix tree at the same assertion: the continuation was Rich's, so it
    began at column 0.
    """
    from local_operator.tui.widgets.network_panel import NetworkLocal, NetworkScreen

    local = NetworkLocal(device_id="", device_name="", identity_present=False, relay_state="")
    for size in _UNPAIRED_HANG_SIZES:
        app = _app_fixture()
        async with app.run_test(size=size) as pilot:
            screen = NetworkScreen(local)
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()
            rows = [row for row in screen.render_lines_for_test() if row.strip()]
            visible = [_row_text(row) for row in _visible_body_rows(app, screen) if row.strip()]
            # The identity row is the block's FIRST, so its wrapped lines are
            # inside the viewport at BOTH sizes — the witness this property needs,
            # and the reason the assertion is about it rather than about the two
            # teaching rows below it, which these sizes scroll off. It is read off
            # the FRAME rather than off the model, because the model is not where
            # the defect was: the row was one long line and the break was Rich's.
            sentence = "no identity yet — /network new <name>"
            first = next(i for i, row in enumerate(visible) if "no identity yet" in row)
            lines: list[str] = []
            for row in visible[first:]:
                lines.append(row)
                if " ".join(lines).split() == sentence.split():
                    break
            assert len(lines) > 1, (size, lines)
            for row in lines[1:]:
                assert row.startswith("  "), (size, visible)
            assert " ".join(lines).split() == sentence.split(), (size, visible)
            # The mechanism is shared, so the witness above carries the paint; this
            # half says all three first-run rows go through it, which is what a row
            # added to the block has to keep doing.
            block = [
                row
                for row in rows
                if row.lstrip().startswith(("no identity yet", "no networks yet", "then /network"))
            ]
            assert len(block) == 3, (size, block)
            assert all(len(row) <= screen.query_one("#network-body").size.width for row in block), (
                size,
                block,
            )


@pytest.mark.asyncio
async def test_the_first_run_rows_advice_runs_at_the_caret_that_printed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-1: THE FIRST-RUN ROW NAMES A STEP, AND THIS RUNS IT.

    WHAT THE FRAME SAID BEFORE THIS ROUND, measured on the real panel at 100x30:
    ``no identity yet — /network new mints it``. The `new` arm is `needs=1`, so the
    reader who followed that row literally — typing the command it names — read
    ``That is not a complete /network new command``: the step named did not run,
    which is U2's defect one row up. So the guard is U2's shape rather than a
    string assertion: the row is read off the PAINTED frame, its own words are
    handed to the real editor and the real submit handler, and the assertion is
    what the arm did with them.

    WHY THE EXECUTION HALF ALONE IS NOT ENOUGH, which is the part worth stating:
    the `new` arm joins the WHOLE tail into the name, so a row whose command is
    followed by prose still RUNS — it runs with the prose as the network's name
    (measured on the pre-remediation row: its tail ``/network new mints it``
    reaches the CLI as ``init "mints it"``). "Something ran" therefore passes on
    the defect. The assertion that observes it is the argv: the words the row
    prints have to BE the command, so the name the arm takes from them is the
    row's own placeholder and nothing else. `<name>` is the family's spelling —
    the Networks row one block down, `TIP_MESH`, and `/network new`'s own usage
    line all print it.

    NOTHING RUNS ON DISK: the first frame is a fixed `NetworkLocal` in the state
    this row is painted in, and both `run_network` seams (the app's and the
    panel's) are recorders — unstubbed, `/network new` starts the relay, and a
    unit test must not leave a daemon behind. What is NOT stubbed is the path the
    words travel: the real `/network` arm, the real editor, the real handler.
    """
    from local_operator.tui.widgets.network_panel import NetworkLocal, NetworkScreen

    monkeypatch.setattr(
        "local_operator.tui.widgets.network_panel.capture_local",
        lambda root=None: NetworkLocal(
            device_id="", device_name="", identity_present=False, relay_state=""
        ),
    )
    # The panel's own worker and the app's arm call the same module, so both are
    # redirected: one recorder would otherwise miss the calls the other makes.
    monkeypatch.setattr(
        "local_operator.tui.widgets.network_panel.run_network",
        lambda args, **kwargs: NetworkRun(tuple(args), 0, stdout="", stderr=""),
    )
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)

    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network")
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, NetworkScreen), screen
        painted = [_row_text(row) for row in _visible_body_rows(app, screen)]
        row = next(row for row in painted if "no identity yet" in row)
        await pilot.press("escape")
        await pilot.pause()

        # Every row of this block is `state — instruction`, so the tail is what a
        # reader copies out of it.
        assert "—" in row, row
        advice = row.split("—", 1)[1].strip()
        assert advice.startswith("/network new "), row
        before = len(run.calls)
        await _submit(pilot, app, advice)
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = "\n".join(_notices(app))

    assert "not a complete /network new command" not in notices, (row, notices)
    assert run.calls[before:] == [["init", "<name>"]], (row, run.calls[before:])


def test_a_receipt_is_read_back_in_the_composers_spelling() -> None:
    """THE RECEIPT IS THE CLI'S; THE READER IS AT A COMPOSER (design round 1, D2).

    Measured on the real path before the fix: `/network new devmesh` printed
    ``next: lop network invite --role drive``, so the surface that had just
    accepted the family's own command handed its reader another surface's
    spelling as the next step — while the panel's empty state and the README both
    say `/network invite`.

    The second half is the reason the translation consults a vocabulary instead of
    replacing a string: the same receipts name `lop network start`, `serve` and
    `install`, which `NETWORK_SUBCOMMANDS` deliberately withholds (a composer row
    that boots out the operator's relay is the one-keystroke mistake the family's
    typed confirmations exist to prevent). Those keep the CLI's spelling, because
    translating them would offer a word this front end then refuses.
    """
    from local_operator.tui.network_cli import tui_spelling

    assert tui_spelling(
        "next: lop network invite --role drive   (the token is written to a file, not printed)"
    ) == ("next: /network invite --role drive   (the token is written to a file, not printed)")

    for withheld in (
        "the relay is not running; start it with `lop network start`",
        "no launchd here: run `lop network serve` in the foreground",
        "nothing was installed",
    ):
        assert tui_spelling(withheld) == withheld, withheld

    # A verb this front end runs is rewritten, and the rest of the sentence is
    # untouched — the receipt is still the CLI's own wording.
    assert tui_spelling("then, on the other device: lop network show devmesh") == (
        "then, on the other device: /network show devmesh"
    )

    # `join` is the subtler half of the same rule: it IS in the vocabulary (the
    # picker offers it) but the composer does not run it — pairing needs a
    # terminal, so its answer here is a sentence telling the reader to use one —
    # and the receipt naming it is addressed to the OTHER device. Rewriting it
    # would hand the reader a word this front end then refuses.
    for untranslated in (
        "then, on the other device: lop network join @<token-file>",
        "run: lop network join @/tmp/token.invite",
    ):
        assert tui_spelling(untranslated) == untranslated, untranslated

    # THE VERB IS PART OF THE SPELLING (UX round 1, U4). `init` is the shell's word
    # for the composer's `new`, and it is the word a REFUSAL now hands the reader as
    # the way out (the empty-network refusal ends with it) — so a reader at a
    # composer who is told to run `lop network init` is being handed a subcommand
    # `NETWORK_SUBCOMMANDS` refuses, which is D2 one token down. Both directions are
    # pinned here: renamed when it is carried, and left alone when it is not.
    assert tui_spelling(
        "name a network: this device is in none — `lop network init <name>` creates the first one"
    ) == ("name a network: this device is in none — `/network new <name>` creates the first one")
    assert tui_spelling("then run `lop network init devmesh` on the other device") == (
        "then run `/network new devmesh` on the other device"
    )


@pytest.mark.asyncio
async def test_the_printed_advice_runs_at_the_caret_that_printed_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U2: THE RECEIPT NAMES A COMMAND, AND THIS RUNS IT.

    What the assertion above cannot do, and why this exists: it pins the STRING
    `tui_spelling` returns and stops there — so the suite stayed green while
    pasting that line verbatim answered ``✗ cli.py network invite: error: argument
    --network: expected one argument``. The invite arm read `rest[0]` as the
    network's name, handed argparse `--network --role`, and refused the very step
    the receipt had just named, in argparse's vocabulary, on the surface this
    family exists to keep plain. A guard that pins the text of an instruction
    nobody executes cannot fail on the defect it is there for.

    NOTHING HERE IS A STUB OR A FIXTURE. The `next:` line is the REAL CLI's output
    from `network init` in an isolated root; the paste is driven through the real
    editor into the real submit handler; the handler spawns the real `lop network
    invite`. Both halves that have to agree — what the CLI prints and what the
    composer accepts — are the shipped ones, and the assertion is the invite that
    came back plus the token file it wrote, not the argv that was sent.
    """
    from local_operator.tui.network_cli import run_network, tui_spelling

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))

    created = run_network(["init", "devmesh", "--no-start"])
    assert created.ok, created.lines
    advice = next(line for line in created.lines if line.startswith("next:"))
    assert "lop network invite --role drive" in advice, advice
    assert tui_spelling(advice) == (
        "next: /network invite --role drive   (the token is written to a file, not printed)"
    )
    # What a reader copies out of that line: the command, without its parenthetical.
    pasted = tui_spelling(advice).split("next:", 1)[1].split("  ", 1)[0].strip()
    assert pasted == "/network invite --role drive", pasted

    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, pasted)
        await app.workers.wait_for_complete()
        await pilot.pause()
        painted = _painted(app)

    assert "expected one argument" not in painted, painted
    assert re.search(r"invite \S+ for role drive", painted), painted
    # …and the invite's own effect, which is the channel the receipt sends the
    # reader to: the token is written to a file and never printed here.
    outbox = tmp_path / ".local-operator" / "network" / "outbox"
    assert [path.suffix for path in sorted(outbox.iterdir())] == [".invite"], list(outbox.iterdir())


@pytest.mark.asyncio
async def test_the_invite_arm_accepts_every_line_its_own_receipt_prints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arm's grammar, shape by shape, against the CLI's OWN parser.

    The test above proves the receipt's line RUNS; this one says which shapes the
    arm accepts, and — the half no string assertion can see — that every argv it
    builds parses under the real `lop network` parser, built here the way the CLI's
    own tests build it. A shape the arm INVENTED would otherwise reach a user as an
    argparse sentence, which is what the round measured.

    The two refused shapes are refused BY THIS SURFACE: `/network invite --role`
    has half a flag, and `/network invite devmesh --role admin` puts the flag where
    the CLI's help does not print it. Both would be silently mistranslated if the
    tail were passed through blind — the first as the network's NAME (the defect
    this branch was filed for), the second as a dropped word — so the assertion is
    that nothing ran AND that the sentence the user reads is this family's usage
    line rather than argparse's. The third is the same silence in the other half
    (review round 2, R2-2): a surplus WORD ran the first name's invite and said
    nothing about the rest, which is the accepted-and-then-dropped class the flag
    refusal above exists for, one branch away from it.
    """
    import argparse

    from local_operator.network import cli as net_cli
    from tests.unit.network import conftest as net_fixtures

    _isolated_network_store(monkeypatch, ("devmesh",))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))
    # The SUBparser, taken the way `tests/unit/network/test_cli.py` takes it, so
    # the argv below is checked against the verb's own grammar rather than by
    # re-listing the flags this test already expects.
    network = net_fixtures.subcommands_of(parser)["network"]

    accepted = {
        # The bare form, and the line `/network new` prints as its next step.
        "/network invite": ["invite", "--role", "drive"],
        "/network invite --role drive": ["invite", "--role", "drive"],
        # A role the CLI offers and the composer does not pin: it is passed on
        # rather than dropped (the minted token says what it is).
        "/network invite --role admin": ["invite", "--role", "admin"],
        # The named form: the first token that is not part of a flag pair.
        "/network invite devmesh": ["invite", "--role", "drive", "--network", "devmesh"],
    }
    refused = (
        "/network invite --role",
        "/network invite devmesh --role admin",
        "/network invite devmesh extra",
    )

    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        for command, expected in accepted.items():
            before = len(run.calls)
            await _submit(pilot, app, command)
            await app.workers.wait_for_complete()
            assert run.calls[before:] == [expected], (command, run.calls[before:])
            network.parse_args(run.calls[before])
        for command in refused:
            before = len(run.calls)
            await _submit(pilot, app, command)
            await app.workers.wait_for_complete()
            assert run.calls[before:] == [], (command, run.calls[before:])
        notices = "\n".join(_notices(app))
        assert "--role needs a value" in notices, notices
        assert "takes one network name and a leading --role" in notices, notices
        # The surplus word's own refusal, and the one thing it must not do: run.
        assert "devmesh extra is more than one name" in notices, notices
        assert "error: argument" not in notices, notices


@pytest.mark.asyncio
async def test_the_invite_arm_addresses_a_multi_word_name_the_way_its_siblings_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-1: a name `new` accepted has to be a name `invite` can address.

    `new` takes the WHOLE tail, so `/network new My Fancy Net` makes a mesh a space
    wide, and `rename`, `rm`, `disconnect` and `member rm` each resolve a
    longest-prefix target so that name stays reachable. `invite` was the fifth and
    the only one that read the space as a surplus word, answering *My Fancy Net is
    more than one name* — a sentence that is false about a name this same surface
    creates, lists and renames, and that left the 26-character id (copied out of a
    DIFFERENT refusal) as the only route to the mesh the user had just named.

    The resolve is its siblings', and the LEFTOVER is what still refuses, which is
    what keeps R2-2 whole: `/network invite My Fancy Net extra` must not quietly
    mint against the two words it could resolve, and an unresolvable multi-word tail
    must not run either — a local refusal is the only answer that cannot drop a
    word. The single misspelt token still degrades to the CLI (which names the
    networks this device is in), because that refusal is the CLI's own.

    The argv goes through the real `lop network` parser exactly as its sibling above
    does: a name with a space reaches argparse as ONE token only if the arm joins it,
    and a name split across two argv entries is the same defect one layer down.
    """
    import argparse

    from local_operator.network import cli as net_cli
    from tests.unit.network import conftest as net_fixtures

    # BOTH names are in the store, so the LONGEST match has to win: the arm must
    # address the mesh the user typed rather than the one that is merely a prefix.
    _isolated_network_store(monkeypatch, ("My Fancy", "My Fancy Net"))
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))
    network = net_fixtures.subcommands_of(parser)["network"]

    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/network invite My Fancy Net")
        await app.workers.wait_for_complete()
        assert run.calls == [["invite", "--role", "drive", "--network", "My Fancy Net"]], run.calls
        network.parse_args(run.calls[0])
        # The shorter name is still addressable beside it, and a leading flag pair
        # rides with a multi-word name the way the receipt's own line does.
        await _submit(pilot, app, "/network invite --role admin My Fancy")
        await app.workers.wait_for_complete()
        flagged = ["invite", "--role", "admin", "--network", "My Fancy"]
        assert run.calls[1:] == [flagged], run.calls[1:]
        # A SURPLUS THE RESOLVE CAN SEE: the name resolved, the extra word refused.
        await _submit(pilot, app, "/network invite My Fancy Net extra")
        await app.workers.wait_for_complete()
        assert len(run.calls) == 2, run.calls
        notices = "\n".join(_notices(app))
        assert "My Fancy Net extra is more than one name" in notices, notices
        # A SURPLUS IT CANNOT: nothing in this tail names a network, and the arm may
        # not hand the first word to the CLI and drop the rest on the way.
        await _submit(pilot, app, "/network invite Gamma Mesh extra")
        await app.workers.wait_for_complete()
        assert len(run.calls) == 2, run.calls
        assert "Gamma Mesh extra is more than one name" in "\n".join(_notices(app))
        # ONE misspelt token is not a surplus: it is passed through, so the refusal
        # that comes back names the networks this device is actually in.
        await _submit(pilot, app, "/network invite My-Fancy-Typo")
        await app.workers.wait_for_complete()
        assert run.calls[2:] == [
            ["invite", "--role", "drive", "--network", "My-Fancy-Typo"]
        ], run.calls[2:]


@pytest.mark.asyncio
async def test_the_invite_refusal_names_a_remedy_the_caret_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U4: the refusal used to say what was missing and stop.

    Measured on a device that had just been told to invite: ``✗ name a network:
    this device is in none``. Every other refusal in this family ends with the
    command that gets the reader out — the CLI's own join failures do (``mint one
    on the other device with `lop network invite`, bring the file across, then run
    …``) — and this one named no way forward on the surface where `invite` is a
    one-word command away from the empty state that does not cover it.

    The remedy is asserted WHERE IT IS ADDRESSED: the same sentence reaches a
    shell and a composer, so the composer's copy has to name a command the caret
    runs — `tui_spelling` renames the CLI's `init` to this front end's `new` — and
    the remedy the frame carries is then handed to the real arm to prove it is
    accepted. The create itself is not EXECUTED here: the composer's `new` arm
    starts the relay (`_cmd_init` without `--no-start`), and a unit test must not
    leave a daemon behind — the executed half is the receipt test above.
    """
    from local_operator.tui.network_cli import run_network, tui_spelling

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))

    refused = run_network(["invite", "--role", "drive"])
    assert not refused.ok, refused.lines
    sentence = refused.lines[-1]
    assert "name a network: this device is in none" in sentence, sentence
    assert "lop network init <name>" in sentence, sentence
    composer = tui_spelling(sentence)
    assert composer.endswith("`/network new <name>` creates the first one"), composer

    remedy = composer.split("`", 2)[1]
    assert remedy == "/network new <name>", remedy
    recorder = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", recorder)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, remedy.replace(" <name>", " devmesh"))
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert recorder.argv == ["init", "devmesh"], recorder.argv


def test_the_unpaired_empty_state_teaches_the_whole_pairing_path() -> None:
    """A device with no identity and no networks is where every device starts,
    and the frame it painted named two commands with nothing between them.

    The ORDER is the property, not the wording. On this device `/network invite`
    is refused outright — ``name a network: this device is in none`` — so an
    empty state that offers `invite` first (the Peers row below it does name it)
    hands the reader a command that fails, and one that named `new` alone under
    two headings left the other half of the path unstated: a token, written to a
    file, redeemed by the OTHER device with a command this one never prints.
    That second half is the whole of pairing and it is the reason the empty
    state is a sequence rather than a sentence.

    The device row is asserted NOT to repeat the Networks row's clause: two rows
    carrying the same remedy verb under two headings read as two steps, which is
    the defect the sequence above is meant to remove rather than to add to.

    WHAT THE THIRD STEP IS ASSERTED BY, since UX round 1's U5 shortened these rows
    to fit the narrowest card: the block still names both commands THIS device runs
    (`/network new`, `/network invite`) in order, and the third step by the fact
    that has no other home — WHICH DEVICE runs the join (`join on the peer`). The
    CLI spelling of that command (`lop network join @<file>`) is the 23 cells that
    did not fit beside the cue at 43, and it is printed by the side that runs it
    (``invite``'s own receipt ends ``then, on the other device: lop network join
    @<token-file>``). The replacement is not the weaker half: the cue is asserted
    here AND at the fold — painted at every size in `_UNPAIRED_FOLD_SIZES` — while
    the literal it replaced was clipped off the frame at 56x20 and never checked
    for being on it.
    """
    from local_operator.tui.widgets.network_panel import (
        NetworkLocal,
        build_network_report,
    )

    local = NetworkLocal(device_id="", device_name="", identity_present=False, relay_state="")
    text = build_network_report(local, width=_NARROWEST_CARD).plain

    networks = text.split("Networks", 1)[1].split("Peers", 1)[0]
    create = networks.index("/network new")
    invite = networks.index("/network invite")
    cue = networks.index("join on the peer")
    assert create < invite < cue, networks

    device_rows = [line for line in text.splitlines() if "no identity yet" in line]
    assert len(device_rows) == 1, text
    assert "creates one" not in device_rows[0], device_rows[0]


def test_the_peers_row_names_the_next_command_once_a_network_exists() -> None:
    """U3: the surface that taught step one went quiet at step two.

    Measured on the real flow: `/network new devmesh` prints the network row and
    the Peers block states a consequence (`no peers yet — a device you pair appears
    here`) — design round 1's D1 wording, written for the frame where BOTH blocks
    are empty states and `invite` is named one block up. One command later the
    reader has a network and no peers, and the row that named `/network invite`
    before this PR named nothing: a NET REMOVAL at the moment the command is next.
    The consequence keeps the state D1 was about; the command comes back for the
    state D1 does not cover, and both facts are already computed for the same
    render (`_entries()` is what the Networks block above shows).
    """
    import json

    from local_operator.tui.widgets.network_panel import (
        NetworkEntry,
        NetworkLocal,
        NetworkRun,
        NetworkScreen,
    )

    # SETTLED, because the row only paints once the relay has answered: the pending
    # frame's Peers block is its own `checking with the relay…` line and nothing
    # else (that is the D20 test above). A status answer saying the relay is not
    # running is the state where the sections inherit the answer.
    settled = NetworkRun(
        ("status",),
        0,
        stdout=json.dumps({"installed": True, "identity_present": True, "relay_running": False}),
    )

    def peers_section(local: NetworkLocal) -> str:
        screen = NetworkScreen(local)
        screen.status_run = settled
        text = "\n".join(screen.render_lines_for_test())
        # Read as one line: the row wraps below the card width the screen uses
        # before it is mounted, and this assertion is about the WORDS (the wrap is
        # the two tests above).
        return " ".join(text.split("Peers", 1)[1].split())

    # A device with NO network: the consequence wording D1 was written for, where
    # the Networks block above is an empty state whose own rows name `invite`.
    empty = NetworkLocal(
        device_id="d_self", device_name="this-mbp", identity_present=True, relay_state=""
    )
    peers = peers_section(empty)
    assert "no peers yet — a device you pair appears here" in peers, peers
    assert "/network invite" not in peers, peers

    # One command later: a network on this device, still no peers. The Networks
    # block above is a network ROW now, so naming the command here is not the
    # duplication D1 was removing.
    paired = NetworkLocal(
        device_id="d_self",
        device_name="this-mbp",
        identity_present=True,
        relay_state="",
        networks=[
            NetworkEntry(
                network_id="n_3985570272c803eeb85a3e23",
                name="devmesh",
                epoch=1,
                role="admin",
                members=1,
                trust="active",
            )
        ],
    )
    peers = peers_section(paired)
    assert "no peers yet — /network invite mints a token" in peers, peers


# ---------------------------------------------------------------------------
# Design round 3 D23 / UX round 3 U22 — the pending word, and the log path's indent
# ---------------------------------------------------------------------------


def test_one_pending_word_for_one_pending_fact() -> None:
    """D23: the frame said "the relay has not answered yet" three ways.

    ``relay: checking…`` on the device line, ``checking with the relay…`` on the
    two sections that inherit the answer, and ``asking the relay…`` in the Relay
    section — three spellings of one state on one screen. The verb is one word
    now; what differs is only the OBJECT, which is the fact each line is about.
    """
    from local_operator.tui.widgets.network_panel import build_network_report

    text = build_network_report(_panel_local()).plain
    assert "asking the relay…" not in text, text
    assert "relay: checking…" in text, text
    assert "checking with the relay…" in text, text
    # The Relay section's own pending line: the same verb, and its subject is the
    # relay the heading three lines above already names.
    assert "  checking…" in text, text


def test_a_wrapped_log_path_keeps_the_sections_indent() -> None:
    """U22: the value dropped to column 0 when it wrapped, so the tail of a path
    read as a different section."""
    from local_operator.tui.widgets.network_panel import _indented_value

    prefix = "  log:        "
    long_path = "/Users/someone/Library/Application Support/local-operator/logs/network.log"
    wrapped = _indented_value(prefix, long_path, 60)
    lines = wrapped.splitlines()
    assert len(lines) > 1, wrapped
    assert lines[0].startswith(prefix)
    for continuation in lines[1:]:
        assert continuation.startswith(" " * len(prefix)), repr(continuation)
    # Nothing is lost: the wrapped form re-joins to the original value.
    assert "".join([lines[0][len(prefix) :], *(ln[len(prefix) :] for ln in lines[1:])]) == long_path
    # A value that fits is untouched — there is no second code path for it.
    assert _indented_value(prefix, "/short/log", 60) == prefix + "/short/log"


# ---------------------------------------------------------------------------
# UX round 3 U23 — the reason, in words
# ---------------------------------------------------------------------------


def test_a_reason_token_is_said_in_words_and_a_sentence_is_left_alone() -> None:
    """U23, at the function every surface now shares.

    ``connect_failed:ConnectionRefusedError`` is a stage, a colon with no space
    and a Python class name: a user surface that prints it names the transport's
    failure mode where the reader needs "that device is not there" (design round 1,
    D3). ``asked, and it did not answer`` is a SENTENCE the relay already wrote for
    a person, and glossing that one away would replace a specific answer with a
    blander one — which is why the rule splits on what the field holds rather than
    on which surface is reading it.
    """
    from local_operator.resume import peer_reason_words

    assert peer_reason_words("connect_failed:ConnectionRefusedError") == "it did not answer"
    assert peer_reason_words("connect_failed:TimeoutError") == "it did not answer"
    assert peer_reason_words("no_endpoint") == "no address published for it"
    assert peer_reason_words("") == "it did not answer"
    # A sentence survives, and a ``stage: <sentence>`` keeps its sentence and
    # loses only the stage word.
    assert peer_reason_words("asked, and it did not answer") == "asked, and it did not answer"
    assert (
        peer_reason_words("not_attempted: the listing budget ran out before this member was probed")
        == "the listing budget ran out before this member was probed"
    )
    # NEVER EMPTY is the contract every caller paints into a clause.
    for token in ("", "  ", "connect_failed:", "unreachable:TimeoutError"):
        assert peer_reason_words(token).strip(), token


#: The one prose reason whose body EMBEDS an endpoint, built by its producer so the
#: case cannot drift out of this file (round 10, MAJOR-1).
_HANDSHAKE_ANSWERED = (
    "handshake_not_attempted: 127.0.0.1:39223 answered and the listing budget ran out "
    "before the handshake"
)


def test_a_sentence_carrying_an_address_is_still_a_sentence() -> None:
    """Round 10, MAJOR-1, at the shared function: the shape test read prose as a list.

    The relay writes this one reason as a SENTENCE with the winning endpoint inside
    it, and the winner is by construction a bare ``host:port``. The previous shape
    test counted any colon-bearing field as a wire token, so this input came back as
    "no address of it answered" — for an address that answered — which is the
    inverse of the truth (round 10's own reproduction, at this function).

    The reason is built by the REAL producer, and the two neighbours it must not be
    confused with are asserted beside it: a machine list is still glossed whole, and
    a bare endpoint-plus-detail segment with no ``;`` at all is still a list.
    """
    import local_operator.resume as resume
    from local_operator.network import relay

    assert relay.handshake_not_attempted_reason("127.0.0.1:39223") == _HANDSHAKE_ANSWERED
    words = resume.peer_reason_words(_HANDSHAKE_ANSWERED)
    assert words == "it answered, and the listing ran out of time before the handshake"
    # Neither the address nor the stage word survives — and the meaning does.
    assert "127.0.0.1" not in words and "handshake_not_attempted" not in words, words
    assert "no address of it answered" not in words, words
    # THE OTHER DIRECTION. A machine list is a ``;``-separated sequence of
    # ``<endpoint> <detail>`` pairs, and every one of its segments ends in a code the
    # probe can write — so a list is glossed WHOLE, including a one-segment list,
    # which is the shape closest to the prose above.
    for tail in (
        "127.0.0.1:39223 connect_failed:ConnectionRefusedError",
        "10.0.0.1:7 no_answer; 10.0.0.2:7 bad_endpoint",
        "10.0.0.1:7 not_attempted",
    ):
        assert resume.peer_reason_words(f"unreachable: {tail}") == "no address of it answered", tail
        assert resume.peer_reason_words(f"half_broken: {tail}") == "no address of it answered", tail
    # THE DISCRIMINATOR ITSELF, both directions, because the arm above would hide a
    # regression in it: the relay's sentence is NOT a machine list, and a list of
    # ``<endpoint> <detail>`` pairs IS — whichever stage word prefixes either one.
    assert (
        resume._carries_wire_tokens(  # noqa: SLF001 — the rule under test
            "127.0.0.1:39223 answered and the listing budget ran out before the handshake"
        )
        is False
    )
    assert (
        resume._carries_wire_tokens("127.0.0.1:39223 connect_failed:ConnectionRefusedError") is True
    )
    assert resume._carries_wire_tokens("10.0.0.1:7 no_answer") is True
    assert (
        resume._carries_wire_tokens("the listing budget ran out before this member was probed")
        is False
    )


def test_the_member_reading_fails_closed_on_prose_that_names_an_address() -> None:
    """Round 11's R11-2 and Q-R25-3, and the trap R11-4 called a dead arm.

    THREE INPUTS, ONE PROPERTY: prose the recogniser cannot read must still not reach a
    person carrying an address, a class name or a stage word.

    * R11-2 — a declared endpoint containing ``;``. Endpoints are stored verbatim and
      unvalidated at both write points (an operator's ``advertise_hosts``, a peer's
      handshake ``peer_endpoints``), so one ``;`` inside an address splits an
      ``<endpoint> <detail>`` pair in two and left a segment whose last field is not a
      code. The rule required EVERY segment to end in a code, so a genuine machine
      list was handed back as prose. It now requires ANY, which is the fail-closed
      direction: at worst a mixed tail reads blander than it is, never wider.
    * Q-R25-3 — the doctor's PRE-round-24 spelling of ``handshake_not_attempted``,
      which an old stored reason or a peer on an older build can still hand the member
      table. It is not a machine list (no ``;`` at all), so the recogniser said prose
      and the sentence kept the endpoint that ANSWERED.
    * R11-4 — the bare stage word ``unreachable`` is an identifier, so the bare-code
      fallback matched it first and read it as a REFUSAL, the opposite state, while
      the arm written for it sat unreachable below.

    The other direction is asserted beside them, because a rule that glosses
    everything is not a fix: the relay's own member-level sentence (no address in it)
    is still returned as written.
    """
    import local_operator.resume as resume
    from local_operator.network import relay

    # R11-2, through the real producer.
    compound = relay.probe_reason(
        [
            relay.CandidateAttempt("10.0.0.1;evil", False, "connect_failed:ConnectionRefusedError"),
            relay.CandidateAttempt("10.0.0.2:7", False, "no_answer"),
        ]
    )
    assert ";" in compound, compound
    read = resume.peer_reason_words(compound)
    assert read == "no address of it answered", read
    for leaked in ("10.0.0.1", "10.0.0.2", "evil", "ConnectionRefusedError", "no_answer"):
        assert leaked not in read, (leaked, read)
    # The discriminator itself, so the arm above cannot hide a regression in it.
    assert resume._carries_wire_tokens(compound.rsplit(": ", 1)[-1]) is True  # noqa: SLF001

    # Q-R25-3: the legacy spelling, built as the relay's own producer builds it.
    legacy = (
        "not_attempted: 127.0.0.1:64994 answered and the doctor budget ran out "
        "before the handshake"
    )
    words = resume.peer_reason_words(legacy)
    assert words == "it answered, and the listing ran out of time before the handshake", words
    assert "127.0.0.1" not in words and "64994" not in words, words
    assert "not_attempted" not in words, words
    # Prose with no address in it is still prose, from the member's own producer.
    untouched = resume.peer_reason_words(relay.NOT_ATTEMPTED_REASON)
    assert untouched == "the listing budget ran out before this member was probed", untouched

    # R11-4: the bare stage word is not a refusal.
    assert resume.peer_reason_words("unreachable") == "no address of it answered"
    assert resume.peer_reason_words("unreachable") != resume._BARE_CODE_WORDS  # noqa: SLF001
    assert resume.peer_reason_words("connect_failed") == "it did not answer"


def test_the_membership_table_speaks_in_words_too() -> None:
    """Round 11, Step 1's own enumeration: a THIRD vocabulary, missed by all of 9-11.

    ``MembershipReport``'s silent rows carry a reason for a member's TABLE not arriving,
    and two renderers printed it raw beside a 34-character device id: the sentence
    `lop network show` prints, and the marker `lop network ls` and the agent tool's
    digest append to each row. Rounds 9-11 each swept a surface that renders a PEER's
    reason and never looked at the surfaces that render a MEMBERSHIP row's, which is
    the enumeration failure this round is about rather than a fourth instance of it.
    """
    import local_operator.resume as resume
    from local_operator.network import relay

    device = "d_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
    report = relay.MembershipReport(network_id="n_" + "a" * 22, refreshed_at=0.0)
    report.silent.append({"device_id": device, "reason": "no_live_link"})
    report.silent.append(
        {"device_id": "d_ffffffffffffffffffffffffffffffff", "reason": "no_table:error"}
    )
    sentence = report.sentence()
    assert sentence.startswith("members NOT verified"), sentence
    for leaked in (device, "no_live_link", "no_table", "d_ffffffffffffffffffffffffffffffff"):
        assert leaked not in sentence, (leaked, sentence)
    assert "nothing is connected to it" in sentence, sentence

    row = {
        "members": 3,
        "membership": {
            "table": {
                "complete": False,
                "answered": [],
                "not_answered": [{"device_id": device, "reason": "no_live_link"}],
            }
        },
    }
    marker = relay.membership_marker(row)
    assert "NOT verified" in marker, marker
    assert device not in marker and "no_live_link" not in marker, marker
    # A reason the producer wrote for a reader survives; one that names an address
    # does not, whichever shape it arrives in.
    assert (
        resume.table_reason_words("not_asked: the refresh budget ran out before this peer's turn")
        == "the refresh budget ran out before this peer's turn"
    )
    assert resume.table_reason_words("nothing at 127.0.0.1:9 answered") == (
        "it did not answer the table read"
    )


def test_the_gloss_never_hands_a_bare_wire_code_back_to_a_person() -> None:
    """Round 10, MAJOR-2: the single-code path's tokens had no table entry.

    A member whose addresses are all black holes and whose budget expires is
    reported with the bare ``no_answer`` code, and that code — like ``bad_endpoint``
    and ``not_attempted`` — had no reading, so a human row printed it. The codes are
    enumerated FROM THE SOURCE here rather than typed: the probe's own vocabulary
    (``relay.PROBE_DETAIL_CODES`` plus the open ``connect_failed:`` family), the
    record facts the relay names as reasons, the handshake's refusal constants
    (``handshake.REASON_*``) and the dial's own phase guard. Every one has a stated
    reading — a table entry, a stage arm, or one of the two documented fallbacks —
    so none of them can reach a reader as itself.
    """
    import local_operator.resume as resume
    from local_operator.network import handshake, relay

    codes = set(relay.PROBE_DETAIL_CODES) - {relay.DETAIL_OK}
    # ``ok`` is a detail an attempt carries, never a reason: a reason exists only
    # when no candidate connected, so ``probe_reason`` drops it.
    assert relay.DETAIL_OK not in codes
    codes |= {
        f"{relay.CONNECT_FAILED_PREFIX}{name}"
        for name in ("OSError", "TimeoutError", "ConnectionRefusedError")
    }
    codes |= {value for name, value in vars(handshake).items() if name.startswith("REASON_")}
    codes |= {"pair_phase_requires_the_ceremony", "no_endpoint", "not_a_member", "member_removed"}
    codes |= {
        relay.NOT_ATTEMPTED_REASON,
        relay.handshake_not_attempted_reason("127.0.0.1:39223"),
        "handshake_refused:OSError",
        # The one arrival a bare identifier cannot be told apart from: a peer's own
        # refusal message in a single word. It reads as that peer's refusal, which
        # is what it is.
        "denied",
    }
    assert len(codes) > 20, codes  # a truncated enumeration is not evidence
    for code in sorted(codes):
        words = resume.peer_reason_words(code)
        assert words.strip(), code
        # Never the code itself, and never anything that still LOOKS like one: no
        # stage colon, no snake_case.
        assert words != code, code
        assert ":" not in words and "_" not in words, (code, words)
    # AND THE CODES WHOSE READING IS ALREADY KNOWN HAVE THAT READING, in the table
    # rather than through the function: "something other than the code" is satisfied
    # by the bare-code fallback, which is a sentence about a REFUSED link and simply
    # the wrong one for a dial that produced no answer at all. A probe code added
    # without an entry fails here, which is the gap this round found.
    for code in sorted(relay.PROBE_DETAIL_CODES - {relay.DETAIL_OK}):
        assert code in resume.PEER_REASON_WORDS, code
    assert resume.PEER_REASON_WORDS["no_answer"] == resume._NO_ADDRESS_ANSWERED
    assert resume.PEER_REASON_WORDS["not_attempted"] == (
        "the listing ran out of time before it was tried"
    )
    assert resume.PEER_REASON_WORDS["bad_endpoint"] == (
        "the address it publishes cannot be dialled"
    )
    # AND THE FAMILIES THAT ARE ONLY EVER WRITTEN AFTER A CONNECT HAVE THE RIGHT
    # READING, which is the assertion "not the code" cannot make: the silence default
    # satisfies it too, which is how ``handshake_refused:`` — the one arrival whose
    # own name says the peer ANSWERED — shipped reading as silence, and how the bare
    # refusal codes could regress to it (round 24, Q-R24-1).
    refused = {value for name, value in vars(handshake).items() if name.startswith("REASON_")} | {
        "pair_phase_requires_the_ceremony"
    }
    # ``not_a_member`` is the one refusal with a DIAGNOSIS rather than the refusal
    # reading (the peer answered, refused, and said WHICH way this device is wrong),
    # so it keeps its table entry; the assertion below it covers that.
    refused -= {"not_a_member"}
    refused |= {
        relay.handshake_refused_reason(exc)
        for exc in (TimeoutError(), ConnectionResetError("the peer closed it"))
    }
    assert len(refused) >= 11, refused  # a truncated enumeration is not evidence
    for reason in sorted(refused):
        assert resume.peer_reason_words(reason) == resume._BARE_CODE_WORDS, reason  # noqa: SLF001
    assert resume.peer_reason_words("not_a_member") == resume.PEER_REASON_WORDS["not_a_member"]


# ---------------------------------------------------------------------------
# Design round 3 D40 — the audit row, and the row it is allowed to cost
# ---------------------------------------------------------------------------


def test_the_audit_row_is_painted_where_the_fold_cannot_reach_it() -> None:
    """THE FOLD IS THE CONSTRAINT, so the row left the scrolled region entirely.

    Round 3 painted the audit news in the Relay block and paid for it out of the
    block's separator blank. That held only while the content above held still: the
    block is the LAST thing in ``#network-scroll``, whose height grows with every
    network and peer row, so two more peers pushed the block — and the row with it —
    under the fold, and the distinction vanished again in exactly the state the row
    exists for (design round 4, D46, measured on the real wedged payload and on the
    fixture with two more peers: ``virtual_size`` 18 and 20 in an 84x16 region).

    The row is the title block's third line now, outside ``#network-scroll`` and out
    of its budget, in the row the title's own blank padding used to hold — so what this
    pins is a model fact rather than a pixel: **the body is the same in every audit
    state**, and the news is a property of the panel. A steady panel paints the third
    line empty, which is the frame every geometry comparison in the round (and the
    committed README figure) is made against.
    """
    import json

    from local_operator.tui.widgets.network_panel import NetworkRun, NetworkScreen

    def block(relay: Any, **over: Any) -> list[str]:
        screen = NetworkScreen(_panel_local())
        payload = {
            "installed": True,
            "identity_present": True,
            "relay_running": True,
            "relay_answering": True,
            "relay_state": "live",
            "relay": relay,
            "log": "/tmp/iso/logs/network.log",
        }
        payload.update(over)
        screen.status_run = NetworkRun(("status",), 0, stdout=json.dumps(payload))
        return screen.render_lines_for_test()

    steady = block({"pid": 4711, "audit_recorded_through": 13, "audit_published_through": 13})
    lagging = block({"pid": 4711, "audit_recorded_through": 13, "audit_published_through": 12})

    # The steady panel carries no news anywhere, and the title block contributes no
    # third line: that row is the title's own padding, which is what keeps the steady
    # frame's styles — and its bytes — exactly as they were.
    assert not [line for line in steady if line.startswith("  audit:")], steady
    assert steady[0] == "Mesh networks", steady
    assert steady[_body_start(steady)] == "This device", steady[:5]

    # The news is on the TITLE: after the rule and before the body, in the same words
    # at the same cell 14 the block's rows use — so a reader moving between the panel
    # and `lop network status` reads the same field twice.
    body_at = _body_start(lagging)
    assert lagging[0] == "Mesh networks", lagging
    assert lagging[1] == steady[1], lagging[:2]
    news_lines = lagging[2:body_at]
    assert len(news_lines) == 1, lagging[:6]
    assert news_lines[0].startswith("  audit:"), lagging[:5]
    assert lagging[2].index("13 recorded") == 14, lagging[2]
    # AND IT IS ONE ROW, ELIDED AND MARKED. This rig's title is built at the panel's
    # 40-cell floor — the screen is never mounted, so ``_card_width`` falls back to
    # ``_MIN_CARD_WIDTH`` — and the sentence is 53 cells of value against a 26-cell
    # budget, so the row is cut and ends in the ``…`` that says so (``_one_row_value``).
    # Round 4 wrapped here instead, which is what grew the title block 3 -> 8 rows on a
    # real terminal at 50x18 and took the body's six rows down to one (review round 4
    # MAJOR 1; design round 5, D48).
    assert news_lines[0] == "  audit:      13 recorded, published th…", news_lines

    # THE BODY IS STATE-INDEPENDENT, and the separator is the section rhythm in every
    # state — the two halves of D47: the news no longer spends the blank, so `Relay` no
    # longer abuts the peers list in the states that carry news.
    steady_at = _body_start(steady)
    assert lagging[body_at:] == steady[steady_at:]
    relay_at = next(n for n, line in enumerate(steady) if line == "Relay") - steady_at
    assert steady[steady_at + relay_at - 1] == "", steady[steady_at + relay_at - 2 :]
    assert lagging[body_at + relay_at - 1] == "", lagging[body_at + relay_at - 2 :]


def _body_start(lines: list[str]) -> int:
    """Where the scrolled body begins — the first row after the title block."""
    return next(n for n, line in enumerate(lines) if line == "This device")


def _band(app: Any, screen: Any) -> dict[str, Any]:
    """What the title band OWNS, read off the painted frame rather than the model.

    The rows come from the compositor's own strips — the title's region is a rectangle
    of the frame — because the property under test is that a row of the scrolled body
    is still on screen, and no widget's own model can answer that: the block's BOX grew
    while its content did not, which is a thing only the painted frame shows (review
    round 4 MAJOR 1). ``content`` is the border box's content area, so it is the cells
    a row can actually occupy — the number the card width has to stay inside.
    """
    title = screen.query_one("#network-title")
    scroll = screen.query_one("#network-scroll")
    rows = _painted(app).split("\n")
    top, height = title.region.y, title.region.height
    return {
        "box": (title.region.width, title.region.height),
        "content": (title.size.width, title.size.height),
        "rows": rows[top : top + height],
        "region": scroll.region.height,
        "virtual": scroll.virtual_size.height,
        "bar": bool(scroll.show_vertical_scrollbar),
    }


#: The sizes the band is pinned at — the six the review round measured (its own
#: table, 100x30 down to the 50x18 floor where the region went 6 -> 1), the 80x24
#: default-width case, the 84x16 region this file's other geometry tests cite, and
#: the 40x20 floor where the rule itself was wider than the box. A single width is
#: the defect this replaced: the claim held at 100x30 and at no other size, so a
#: test pinned there passed while the panel grew 3 -> 8 rows on a real terminal
#: (review round 4 MAJOR 1; design round 5, D48).
_BAND_SIZES = ((100, 30), (98, 30), (96, 30), (84, 16), (80, 24), (56, 20), (50, 18), (40, 20))


@pytest.mark.asyncio
async def test_the_news_row_costs_the_scrolled_body_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE ROW IS OUTSIDE THE REGION, and the region does not shrink to pay for it.

    The claim is about the PAINTED frame — the row the reader loses is a row of the
    scrolled body, and no model assertion can see it — so this is measured off the
    compositor at every size in ``_BAND_SIZES``, in both states: the title block is a
    three-row box with news and a three-row box without it, and ``#network-scroll``
    is the same region with the same scrollbar in both. ``1fr`` makes that the second
    half of the claim: the region is whatever the title and hint leave, so a row the
    title took would be a body row lost, and the designer's "nothing that was on-page
    was pushed off it" is a measurement here rather than a property (round 4, D46/D47).

    THE BOX IS THE BOUND, and it is what round 4's ``height: auto`` broke: the block
    grew to fit a wrapped sentence (3 -> 4 -> 8 rows over this list of sizes) and the
    growth came out of a ``1fr`` region that went 16 -> 10 -> 1 at 50x18. The sentence
    is elided to one row now, so the assertion is that the box is three rows at every
    size rather than three rows at the one size where nothing wraps — and the rule is
    checked for the same reason, because a card one cell wider than the box made the
    rule wrap onto a row of its own below itself at the floor.
    """
    import json

    import local_operator.tui.widgets.network_panel as panel_mod
    from local_operator.network.relay import audit_status_words
    from local_operator.tui.widgets.network_panel import NetworkRun, NetworkScreen

    def status_dict(published_through: int) -> dict[str, Any]:
        return {
            "installed": True,
            "identity_present": True,
            "relay_running": True,
            "relay_answering": True,
            "relay_state": "live",
            "relay": {
                "pid": 4711,
                "audit_recorded_through": 13,
                "audit_published_through": published_through,
            },
            "record": {"pid": 4711},
            "log": "/tmp/iso/logs/network.log",
        }

    def stub(published_through: int) -> Any:
        def run(args: list[str], **kwargs: Any) -> NetworkRun:
            if not args or args[0] != "status":
                return NetworkRun(tuple(args), 0, stdout="")
            return NetworkRun(tuple(args), 0, stdout=json.dumps(status_dict(published_through)))

        return run

    # The relay's own words for this payload: the panel paints them, so the elision is
    # checked against the sentence rather than against a copy of it written here.
    words = audit_status_words(status_dict(12), omit_steady=True)
    #: The label column the block's own rows use, and so the cells of value a row has.
    label = "  audit:      "

    bands: dict[bool, dict[tuple[int, int], dict[str, Any]]] = {False: {}, True: {}}
    for news in (False, True):
        for size in _BAND_SIZES:
            monkeypatch.setattr(panel_mod, "run_network", stub(12 if news else 13))
            app = _app_fixture()
            async with app.run_test(size=size) as pilot:
                screen = NetworkScreen(_panel_local())
                app.push_screen(screen)
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.pause()
                band = _band(app, screen)
                bands[news][size] = band

                # THREE ROWS, and the box is exactly as tall as the content it was
                # given: the steady title is a 3-row box holding 2 rows and a padding
                # row, the news state a 3-row box holding all 3 rows.
                assert band["box"] == (band["content"][0], 3), (size, news, band["box"])
                assert band["content"][1] == (3 if news else 2), (size, news, band["content"])
                # ONE rule row, never two: the rule is built at the card width, which
                # may not exceed the box it is painted in, or it wraps below itself.
                assert [n for n, row in enumerate(band["rows"]) if "─" in row] == [1], (
                    size,
                    news,
                    band["rows"],
                )
                # The frame row carries the panel's own offset (a 90% card, centred,
                # inside a padding box), so the block's rows are read from the label the
                # block itself writes. ``label`` is the cell-14 column the block's rows
                # use; from the frame, the label is what is left after the offset.
                third = band["rows"][2].rstrip()
                if news:
                    # One row, and it is the relay's sentence elided to the cells it
                    # has — the whole sentence where it fits, and the tail replaced by
                    # the ``…`` that says it was cut (never a second row: a wrapped
                    # line is what grew the block instead of marking the cut).
                    row = third.lstrip()
                    assert row.startswith(label.lstrip()), (size, third)
                    value = row[len(label.lstrip()) :]
                    # AND THE CUT IS AT THE CELL THE BUDGET ENDS ON, not merely inside
                    # the row (review round 5, NIT): the cells the value may use are
                    # the panel's own card width minus the label, so an elision that
                    # cut EARLY and still stayed inside the box would fail here, where
                    # "some prefix ending in an ellipsis" passed. Both fixtures are
                    # ASCII, so cells and characters are the same number for them.
                    budget = screen._card_width() - cell_len(label)
                    assert cell_len(value) == min(cell_len(words), budget), (
                        size,
                        third,
                        budget,
                        cell_len(value),
                    )
                    if cell_len(words) <= cell_len(value):
                        assert value == words, (size, third, words)
                    else:
                        assert value == words[: len(value) - 1] + "…", (size, third, words)
                else:
                    assert third.strip() == "", (size, band["rows"])

    for size in _BAND_SIZES:
        # THE REGION, as the reader experiences it: same rows, same thumb, same state.
        assert bands[False][size]["region"] == bands[True][size]["region"], size
        assert bands[False][size]["bar"] == bands[True][size]["bar"], size
        assert bands[True][size]["box"][1] == 3, (size, bands[True][size])
        # And the body's own EXTENT is the same number of rows in both states — the
        # claim D46/D47 rests on, and the one a row moved out of the body has to keep:
        # the sentence is painted in the band, not added to the region's content.
        assert bands[False][size]["virtual"] == bands[True][size]["virtual"], (size, bands)


@pytest.mark.asyncio
async def test_a_re_layout_leaves_the_band_the_same_three_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RESIZE THAT CHANGES NOTHING IN THE MODEL MUST LEAVE THE FRAME ALONE.

    Q-R4-1, measured: in the news state any re-layout left the title holding four rows
    for three rows of content — an empty row under the sentence — and the row did not
    come back until the news cleared, so ``#network-scroll`` sat one row short (16 -> 15)
    with a scrollbar it did not have. It was not the sentence wrapping at the
    destination width: bouncing through 110x34, where nothing wrapped on this tree
    either, returned the same 4/15. The empty row was the block's own padding row,
    which ``height: auto`` re-added to a three-row content measurement on the way back.

    So the assertion is the pair: a fresh frame at a size and the frame a resize
    returns to that same size are THE SAME FRAME, and the news clearing gives the row
    back to the padding it took it from. Driven through the pilot's own
    ``resize_terminal``, which enters the same ``on_resize`` -> ``_repaint`` path a
    terminal resize does.
    """
    import json

    import local_operator.tui.widgets.network_panel as panel_mod
    from local_operator.tui.widgets.network_panel import NetworkRun, NetworkScreen

    def payload(published_through: int) -> str:
        return json.dumps(
            {
                "installed": True,
                "identity_present": True,
                "relay_running": True,
                "relay_answering": True,
                "relay_state": "live",
                "relay": {
                    "pid": 4711,
                    "audit_recorded_through": 13,
                    "audit_published_through": published_through,
                },
                "record": {"pid": 4711},
                "log": "/tmp/iso/logs/network.log",
            }
        )

    def run(args: list[str], **kwargs: Any) -> NetworkRun:
        if not args or args[0] != "status":
            return NetworkRun(tuple(args), 0, stdout="")
        return NetworkRun(tuple(args), 0, stdout=payload(12))

    monkeypatch.setattr(panel_mod, "run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        screen = NetworkScreen(_panel_local())
        app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
        fresh = _band(app, screen)
        assert fresh["rows"][2].strip().startswith("audit:"), fresh["rows"]

        # Away and back through every size the round found a defect at: the wider
        # 110x34 (nothing wraps at either end), the capture width's neighbours, and
        # the 50x18 floor where the region collapsed to a single row.
        for via in ((110, 34), (96, 30), (80, 24), (50, 18), (40, 20)):
            await pilot.resize_terminal(*via)
            await pilot.pause()
            await pilot.pause()
            at = _band(app, screen)
            assert at["box"][1] == 3, (via, at)
            assert [n for n, row in enumerate(at["rows"]) if "─" in row] == [1], (via, at["rows"])
            assert at["rows"][2].strip().startswith("audit:"), (via, at["rows"])

        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        await pilot.pause()
        back = _band(app, screen)
        assert back["box"] == fresh["box"], (fresh, back)
        assert back["content"] == fresh["content"], (fresh, back)
        assert back["region"] == fresh["region"], (fresh, back)
        assert back["rows"] == fresh["rows"], (fresh["rows"], back["rows"])

        # THE ROW COMES BACK when the news clears — the sentence and the class go
        # together, through the panel's own "the worker answered" door. The two
        # non-status runs are the ones the stubbed fill already published.
        assert screen.relay is not None and screen.peers_run is not None, "the fill did not settle"
        answered = NetworkRun(("status",), 0, stdout=payload(13))
        screen.set_relay(screen.relay, screen.peers_run, answered)
        await pilot.pause()
        await pilot.pause()
        cleared = _band(app, screen)
        assert not screen.query_one("#network-title").has_class("audit-news"), cleared
        assert cleared["rows"][2].rstrip() == "", cleared["rows"]
        assert cleared["box"] == fresh["box"], (fresh, cleared)
        assert cleared["content"] == (fresh["content"][0], 2), (fresh, cleared)
        assert cleared["region"] == fresh["region"], (fresh, cleared)


@pytest.mark.asyncio
async def test_a_repair_notice_renders_in_this_device_block_or_not_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner-side notice (decision memo item D), when open — and only then.

    The panel reads the ``doctor`` receipt's ``credential_repair`` checks and paints
    the producer's sentence in the device block; a frame with no such check must not
    gain a row, which is every existing panel cell's frame (they would catch a row
    that leaked in unconditionally).
    """
    import json

    import local_operator.tui.widgets.network_panel as panel_mod
    from local_operator.tui.widgets.network_panel import NetworkRun, NetworkScreen

    sentence = (
        "laptop needs '/mcp login https://mcp.example.com' here — "
        "its borrowed credential cannot be refreshed"
    )

    def stub(with_repair: bool) -> Any:
        checks: list[dict[str, Any]] = (
            [
                {
                    "check": "credential_repair",
                    "network_id": "n_0123456789abcdef01234567",
                    "ok": False,
                    "credential_name": "mcp:https://mcp.example.com",
                    "device_id": "d_peer",
                    "device_name": "laptop",
                    "detail": sentence,
                    "remedies": ["run `/mcp login https://mcp.example.com` on this device"],
                }
            ]
            if with_repair
            else []
        )
        doctor = {"ok": False, "identity_present": True, "checks": checks}

        def run(args: list[str], **kwargs: Any) -> NetworkRun:
            if not args or args[0] != "doctor":
                return NetworkRun(tuple(args), 0, stdout="")
            return NetworkRun(tuple(args), 0, stdout=json.dumps(doctor))

        return run

    seen: dict[bool, str] = {}
    for with_repair in (False, True):
        monkeypatch.setattr(panel_mod, "run_network", stub(with_repair))
        app = _app_fixture()
        async with app.run_test(size=(100, 30)) as pilot:
            screen = NetworkScreen(_panel_local())
            app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()
            seen[with_repair] = "\n".join(screen.render_lines_for_test())

    assert "repair:" not in seen[False], seen[False]
    # The producer's sentence arrives whole — the wrap is the hanging-row helper's
    # (its continuation repeats the row's own lead, so stitch it back and compare
    # the lot, rather than pinning where this build happens to break the line).
    assert "repair: laptop needs" in seen[True], seen[True]
    # The wrap is indent-only under the label (design round 1, D1), so the
    # continuation is the lead's own cell count of spaces — stitch that back.
    flat = seen[True].replace("\n" + " " * cell_len("  repair: "), " ")
    assert (
        "laptop needs '/mcp login https://mcp.example.com' here — its borrowed "
        "credential cannot be refreshed"
    ) in flat, seen[True]


def test_a_wedged_relay_gets_a_sentence_in_the_panel_where_the_numbers_would_be() -> None:
    """The SIGSTOP case: ``relay: null`` and no ``audit*`` key at all.

    A reader who has just found a row missing from ``audit.jsonl`` is, by construction,
    in a state where the relay may not answer — so the panel must not go quiet here,
    because its silence is indistinguishable from a healthy writer's. This is the one
    surface case D40 said it would not accept as a follow-up.

    AND IT NAMES THE PROCESS IT IS TALKING ABOUT (design round 4, D45). The wedged
    payload is the one where ``relay`` is null, which is exactly why the pid cannot be
    read from that block alone: it printed the literal ``None`` above a sentence saying
    the process is up. The record on disk carries the pid, and the panel reads the pair
    in the order the CLI and the agent digest already do.
    """
    import json

    from local_operator.tui.widgets.network_panel import NetworkRun, NetworkScreen

    screen = NetworkScreen(_panel_local())
    screen.status_run = NetworkRun(
        ("status",),
        0,
        stdout=json.dumps(
            {
                "installed": True,
                "identity_present": True,
                "relay_running": True,
                "relay_answering": False,
                "relay_state": "wedged",
                "relay": None,
                "record": {"pid": 35292},
                "log": "/tmp/iso/logs/network.log",
            }
        ),
    )
    text = "\n".join(screen.render_lines_for_test())
    assert "relay:      running, pid 35292 — NOT answering" in text, text
    assert "pid None" not in text, text
    assert "audit:" in text, text
    # The sentence is present even at this cell's 40-cell panel width (a 26-cell value
    # budget), as ONE elided row — the words it can carry, and the ``…`` that says it
    # could not carry the rest. It used to wrap here, which cost the body a row per
    # wrapped line on a real terminal (review round 4 MAJOR 1).
    assert "  audit:      unavailable — relay not a…" in text, text
    assert "audit.jsonlholds" not in text.replace(" ", ""), text
    # And the numbers are never invented for it: no counter survives a null relay.
    assert "recorded," not in text, text


# ---------------------------------------------------------------------------
# the argv the runner builds
# ---------------------------------------------------------------------------


def test_the_json_flag_lands_before_any_payload_the_tail_carries() -> None:
    """The pilot verbs take their text as an argparse REMAINDER, so a flag APPENDED
    to a tail that carries an act becomes part of the prompt: `--send <s> hi --json`
    would send the words "hi --json" and leave the payload channel empty — and this
    runner is the one that appends it (`json_output=True`, added here so no caller
    can forget it).

    Asserted on the argv rather than through a spawned child: the PLACEMENT is the
    behaviour, and a subprocess cell would be measuring the parser that the pilot
    suites already pin.
    """
    from local_operator.tui.network_cli import _argv_for

    argv = _argv_for(
        ["sessions", "--peer", "cloud-node-1", "--send", "abc", "hi", "there"],
        json_output=True,
    )
    start = argv.index("sessions")
    # Directly after the subcommand: where `network sessions` declares it, and where
    # nothing after it can read it as text.
    assert argv[start + 1] == "--json", argv
    assert argv[start + 2 :] == [  # noqa: E203 — black's slice spacing
        "--peer",
        "cloud-node-1",
        "--send",
        "abc",
        "hi",
        "there",
    ], argv

    # And a call that did not ask for JSON is untouched.
    plain = _argv_for(["sessions", "--peer", "cloud-node-1"], json_output=False)
    assert "--json" not in plain, plain
