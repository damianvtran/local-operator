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

from dataclasses import dataclass, field
from typing import Any

import pytest

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
