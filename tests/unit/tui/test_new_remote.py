"""``/new remote <peer>`` — the picker's rows, the grammar, and what runs.

``docs/design/mesh-ui.md`` §1.4 makes the surface's job narrow on purpose: the
SESSION is created by ``lop network sessions --peer <peer> --create`` (the session
plane's own verb, the one the agent guide drives), and this front end's work is
(i) resolving a typed word to ONE device, (ii) offering the words a device this
machine actually knows, and (iii) refusing the cases it cannot resolve instead of
guessing which of two memberships the user meant.

Three claims, one per section below:

* **The rows come from this device's own member lists, read off disk.** The
  autofill opens on the keystroke after a space, and ``/new remote`` is the one
  form that reaches a device whose relay is down — so a vocabulary that needed a
  dial would be empty exactly when it is useful (``network/peers.py``).
* **The ROW carries the whole argument** (``remote <peer>``), because the picker's
  completion replaces the argument it is filling. A row named only ``devon`` would
  leave a buffer whose meaning depended on which shape the registry happened to
  parse.
* **A bare ``/new`` is still a local session.** Reading the argument (which the
  handler used to drop) must not disturb the single-word form the DESKTOP picker
  owns, which is why the registry's new shape accepts one token as well as two.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.peers import KnownPeer
from local_operator.slash_commands import (
    command_argument_is_used,
    command_argument_refusal,
    slash_command_for,
)
from local_operator.tui.network_cli import NetworkRun

# ---------------------------------------------------------------------------
# the grammar (no app needed)
# ---------------------------------------------------------------------------


def test_the_shape_accepts_the_legacy_word_and_the_remote_form() -> None:
    """The superset that keeps ``/new foo`` a control on the desktop.

    ``WORD`` alone would plan ``/new remote devon`` as prose and spend a paid
    model turn on a control the user typed deliberately; ``_is_remote_peer`` alone
    would drop the desktop picker's own selection. Both halves are asserted
    because a future tightening that removed either one would be silent
    otherwise.
    """
    spec = slash_command_for("/new")
    assert spec is not None
    assert command_argument_is_used(spec, "my-project")  # the desktop's selection
    assert command_argument_is_used(spec, "remote devon-laptop")
    assert command_argument_is_used(spec, "remote devon-laptop#d_9f2c")
    # A third token is prose again — the boundary the shape draws.
    assert not command_argument_is_used(spec, "remote devon and then summarise it")


def test_the_refusal_states_the_form_rather_than_listing_peers() -> None:
    spec = slash_command_for("/new")
    assert spec is not None
    refusal = command_argument_refusal(spec, "remote devon and then a sentence")
    assert refusal == "Use /new remote <peer> — or /new for a session on this device"


def test_peer_name_tokens_are_sorted_names_then_ids_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vocabulary a user types: the name they recognise, the id as fallback.

    Read through the real ``known_peer_names`` over a store fixture rather than a
    reimplementation of its loop — a test that spells the logic out again proves
    only that the test and the code were written by the same person.
    """
    from local_operator.network import peers as peers_mod

    record = _FakeRecord(
        network_id="n_1",
        name="devmesh",
        self_device_id="d_self",
        members=[
            _FakeMember(device_id="d_aaaa", name="damian-mbp"),
            _FakeMember(device_id="d_bbbb", name=""),
        ],
    )
    # The same device in a SECOND network: still one token, because the picker
    # offers a peer rather than a membership.
    second = _FakeRecord(
        network_id="n_2",
        name="homelab",
        self_device_id="d_self",
        members=[_FakeMember(device_id="d_aaaa", name="damian-mbp")],
    )
    monkeypatch.setattr(peers_mod.store, "list_networks", lambda root=None: [record, second])
    assert peers_mod.known_peer_names() == ("damian-mbp", "d_aaaa", "d_bbbb")
    # One entry per membership, so a graph can tell the two apart even though the
    # autofill cannot.
    assert [peer.network_id for peer in peers_mod.known_peers()] == ["n_1", "n_1", "n_2"]


def test_split_peer_token_reads_both_halves() -> None:
    from local_operator.network.peers import split_peer_token

    assert split_peer_token("devon") == ("devon", "")
    assert split_peer_token("devon#d_9f2c") == ("devon", "d_9f2c")
    # An id alone is not a name#id pair.
    assert split_peer_token("d_9f2c") == ("d_9f2c", "")


def test_known_peers_excludes_this_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device is never offered as its own peer: ``/new remote <self>`` routes nowhere."""
    from local_operator.network import peers as peers_mod

    record = _FakeRecord(
        network_id="n_1",
        name="devmesh",
        self_device_id="d_self",
        members=[
            _FakeMember(device_id="d_self", name="this-mbp"),
            _FakeMember(device_id="d_other", name="damian-mbp"),
            _FakeMember(device_id="d_gone", name="retired", active=False),
        ],
    )
    monkeypatch.setattr(peers_mod.store, "list_networks", lambda root=None: [record])
    known = peers_mod.known_peers()
    assert [peer.device_id for peer in known] == ["d_other"]
    assert peers_mod.known_peer_names() == ("damian-mbp", "d_other")


def test_a_store_failure_is_an_empty_vocabulary_not_an_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No networks is the ordinary case, and a picker must survive it.

    ``/new remote`` is offered on a device with none — the row exists so the form
    can be discovered, and the picker says what to do about it (the notice below)
    rather than showing nothing.
    """
    from local_operator.network import peers as peers_mod

    monkeypatch.setattr(peers_mod.store, "list_networks", lambda root=None: [])
    assert peers_mod.known_peers() == []
    assert peers_mod.known_peer_names() == ()


# ---------------------------------------------------------------------------
# the app surface
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeMember:
    device_id: str
    name: str = ""
    role: str = "drive"
    kind: str = "device"
    active: bool = True


@dataclasses.dataclass
class _FakeRecord:
    network_id: str
    name: str
    self_device_id: str
    members: list[Any]

    def active_members(self) -> list[Any]:
        return [member for member in self.members if member.active]


@dataclasses.dataclass
class _Recorder:
    calls: list[list[str]] = dataclasses.field(default_factory=list)
    ok: bool = True
    #: What the CLI printed. The real receipt's shape (its first line is
    #: ``session: <id>``) because the surface now READS that line to open what it
    #: created — a stand-in the reader cannot parse would exercise the miss path
    #: in every test that is not about the miss.
    stdout: str = "session: 9f2ac1e0b7d2\ncreated on damian-mbp\nprompt admitted\n"

    def __call__(self, args: list[str], **kwargs: Any) -> NetworkRun:
        self.calls.append(list(args))
        if self.ok:
            return NetworkRun(tuple(args), 0, stdout=self.stdout)
        return NetworkRun(
            tuple(args),
            1,
            stderr="the relay is not running; start it with `lop network start`\n",
        )

    @property
    def argv(self) -> list[str]:
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


def _app_fixture() -> Any:
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


def _peers(monkeypatch: pytest.MonkeyPatch, peers: list[KnownPeer]) -> None:
    from local_operator.network import peers as peers_mod

    monkeypatch.setattr(peers_mod, "known_peers", lambda root=None: list(peers))
    monkeypatch.setattr(
        peers_mod,
        "resolve_peer",
        lambda target, root=None: [
            peer for peer in peers if peer.device_id == target or peer.name == target
        ],
    )


async def _boot(pilot: Any, app: Any) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot: Any, text: str) -> None:
    await pilot.press(*("space" if char == " " else char for char in text))


def _notices(app: Any) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _transcript_text(app: Any) -> str:
    """Everything the transcript holds, notices and receipts together."""
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    def flatten(renderable: Any) -> str:
        if isinstance(renderable, Text):
            return renderable.plain
        if isinstance(renderable, Group):
            return "\n".join(flatten(child) for child in renderable.renderables)
        if isinstance(renderable, Padding):
            return flatten(renderable.renderable)
        return renderable if isinstance(renderable, str) else ""

    return "\n".join(
        flatten(getattr(block, "renderable", "")) for block in app._transcript_view().blocks()
    )


#: The id the stubbed CLI's receipt names — the session the peer "minted".
MINTED = "9f2ac1e0b7d2"


@pytest.fixture(autouse=True)
def _peer_catalogue_is_stubbed(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The catalogue READ, stubbed, and the ttl of every read recorded.

    ``peer_session_rows`` dials this device's relay: a TUI test that let it dial
    would reach whatever relay the machine running the suite happens to have, which
    is the same reason ``run_network`` is stubbed in the tests below. The real read
    is exercised by the network tests that own it; what these tests are about is
    what THIS surface asks for — and the recorded ``ttl_s`` is how the forced
    read after a create (``0``) is asserted rather than asserted about.
    """
    from local_operator.session import peer_rows as peer_rows_mod

    reads: list[float] = []

    def read(root: Any = None, *, ttl_s: float = 20.0, **kwargs: Any) -> tuple[Any, ...]:
        reads.append(ttl_s)
        return ()

    monkeypatch.setattr(peer_rows_mod, "peer_session_rows", read)
    return reads


@pytest.mark.asyncio
async def test_the_picker_offers_remote_then_every_known_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed keystroke for keystroke, then the rows read off the real picker."""
    from local_operator.tui.widgets.editor import Editor

    _peers(
        monkeypatch,
        [
            KnownPeer(device_id="d_aaaa", name="damian-mbp", role="admin"),
            KnownPeer(device_id="d_bbbb", name="", role="drive"),
        ],
    )
    app = _app_fixture()
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, "/new ")
        editor = app.query_one(Editor)
        await pilot.pause()
        assert editor._picker.is_open()
        painted = "\n".join(row.plain for row in editor._picker.render_rows(70))
        # THE ROW PAINTS THE DEVICE, AND INSERTS THE ARGUMENT (design round 2,
        # D15). The keyword stays in the VALUE — taking it out is how a row the
        # picker offers starts a LOCAL session (UX round 1, U4) — so it is
        # asserted ABSENT from the painted name and PRESENT in what choosing the
        # row would fill. Painting it spent the one column that answers "which
        # device" restating the command the user had just typed.
        assert "remote " not in painted, painted
        assert "damian-mbp" in painted
        # The peer with no name is NAMED, not ellipsized: an id-shaped token used
        # to fill both columns with a truncated hex string and never reached the
        # shared ``unnamed device`` string the panel and the sidebar already use
        # (design round 2, D15; the D8 vocabulary).
        assert "unnamed device" in painted
        # …and the device sits in the NAME column, which is the column a reader
        # scans down, rather than in the description.
        rows = editor._picker.render_rows(70)
        assert rows[0].plain.split()[1] == "damian-mbp", rows[0].plain
        assert "unnamed device" in rows[1].plain, rows[1].plain
        # The row NAMES — what choosing a row puts in the buffer — are the tokens
        # the shape accepts: the two halves of ONE contract, asserted together so
        # a row that filled something the validator then refused could not pass.
        inserted = [name for name, _item in editor._picker._matches]
        assert inserted == ["remote damian-mbp", "remote d_bbbb"], inserted
        spec = slash_command_for("/new")
        assert spec is not None
        for token in inserted:
            assert command_argument_is_used(spec, token)
        # ...AND THE ROW UNDER THE CURSOR IS ONE THE HANDLER RUNS (UX round 1,
        # U4). The list opens with its first row highlighted, so a leading
        # keyword-only row meant the first Enter filled the buffer with a bare
        # `remote` and the second answered a red "Use /new remote <peer>…" — the
        # picker refusing the syntax it had just offered. With a peer to offer,
        # every row is a whole working argument and the bare row is not listed.
        assert (
            "Create it on another device" not in painted
        ), "the keyword-only row is back, and it is the row the picker pre-selects"


@pytest.mark.asyncio
async def test_the_picker_says_what_to_do_with_no_peers() -> None:
    """THE STATE EVERY NEW USER STARTS IN (design round 2, D16).

    The list used to open with ONE row — a bare ``remote`` keyword row that
    existed "to be SEEN rather than to be RUN" — and that row was PRE-SELECTED, so
    the first Enter a user presses ran it and landed on a red "Use /new remote
    <peer>…". It also suppressed the notice: ``set_notice`` returns early while
    rows are showing, so the one sentence that says how a first peer comes to
    exist was the sentence the row hid. No row, and the notice paints.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _app_fixture()
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, "/new ")
        editor = app.query_one(Editor)
        await pilot.pause()
        assert not editor._picker.is_open(), "a row is under the cursor with no peers to offer"
        painted = editor._picker.render_text(70).plain
        assert "/network invite" in painted, painted
        # The key the notice leaves the user holding: nothing here runs, so the
        # first Enter cannot be answered with a refusal.
        assert not editor._picker._matches


@pytest.mark.asyncio
async def test_remote_creates_the_session_on_the_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp")
        await app.workers.wait_for_complete()
        # THE USER'S OWN WORD GOES ON THE WIRE, AND COMES BACK IN THE RECEIPT
        # (QA round 11 Q-R11-2 / UX round 2 U12). ``--create`` echoes whatever
        # ``--peer`` it was given into ``created on <x>`` and into the remedy
        # command, so passing the device id made the one surface the user
        # addressed by name answer in a 34-character wire token — a word the
        # sidebar, `/network peers` and `/network sessions` all replace. The
        # relay resolves a name against the same member store this device just
        # resolved it against.
        assert run.argv == [
            "sessions",
            "--peer",
            "damian-mbp",
            "--create",
        ]
        # AND NO ``--name``, which is the fix rather than an omission (UX round 3,
        # U19). It used to pass the peer's own label, so the session was TITLED
        # after the device: the sidebar then painted a session row reading
        # ``pixel-8`` directly under the ``⇄ pixel-8`` heading that already said
        # which device holds it, and it survived a restart because the name went
        # to disk. An unnamed session is named by the owner's own auto-namer on
        # its first substantive turn — exactly what a bare local ``/new`` gives.
        assert "--name" not in run.argv, run.argv


@pytest.mark.asyncio
async def test_an_id_addressed_create_keeps_the_id_when_the_name_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The friendly word may only be used where it still names ONE device.

    The receipt fix (QA round 11, Q-R11-2) must not turn a working create into a
    refusal: an id-addressed create for a peer whose NAME is shared with another
    device would be answered by the relay with its own ambiguous-peer refusal, so
    the id stays on the wire whenever the name does not resolve back to exactly
    this device. Asserted because a one-word change that reads better is one
    committer away from breaking a create that already worked.
    """
    _peers(
        monkeypatch,
        [
            KnownPeer(device_id="d_aaaa", name="laptop", network_id="n_1"),
            KnownPeer(device_id="d_bbbb", name="laptop", network_id="n_2"),
        ],
    )
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote d_aaaa")
        await app.workers.wait_for_complete()
        assert run.argv[:3] == ["sessions", "--peer", "d_aaaa"]


@pytest.mark.asyncio
async def test_a_prompt_rides_along_as_the_remote_first_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp summarise the mesh brief")
        await app.workers.wait_for_complete()
        assert run.argv[-2:] == ["--prompt", "summarise the mesh brief"]


@pytest.mark.asyncio
async def test_the_id_form_addresses_an_unnamed_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _peers(monkeypatch, [KnownPeer(device_id="d_bbbb", name="")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote d_bbbb")
        await app.workers.wait_for_complete()
        assert run.argv[:3] == ["sessions", "--peer", "d_bbbb"]
        # NO ``--name`` FOR AN UNNAMED PEER EITHER (UX round 3, U19). It used to
        # fall back to the peer's own token — and the assertion here was that a
        # nameless peer must not produce a session titled ``""``. Sending NO
        # name is the better answer to the same worry: the relay's own
        # ``name = str(frame.get("name") or "")`` writes no sidecar at all, so
        # the session is UNTITLED (the shared ``Untitled conversation`` string on
        # every surface) rather than titled with a 12-hex token no user typed —
        # and it gets its real name from the owner's auto-namer on its first
        # substantive turn, like a local ``/new``.
        assert "--name" not in run.argv, run.argv


@pytest.mark.asyncio
async def test_an_unknown_peer_is_refused_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo costs a sentence, not a 120-second dial."""
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote devon-lapto")
        await app.workers.wait_for_complete()
        assert run.calls == []
        assert any("No peer named devon-lapto" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_name_in_two_networks_asks_for_the_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguity is refused, and the refusal hands over the disambiguators.

    Guessing which membership the user meant would create the session on a device
    in a network they may not have been thinking about — and the id is the one
    word that names a device rather than a membership.
    """
    _peers(
        monkeypatch,
        [
            KnownPeer(device_id="d_aaaa", name="damian-mbp", network_id="n_1"),
            KnownPeer(device_id="d_bbbb", name="damian-mbp", network_id="n_2"),
        ],
    )
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp")
        await app.workers.wait_for_complete()
        assert run.calls == []
        refusal = [text for text in _notices(app) if "more than one network" in text]
        assert refusal, _notices(app)
        assert "d_aaaa" in refusal[-1] and "d_bbbb" in refusal[-1]


@pytest.mark.asyncio
async def test_bare_remote_asks_for_a_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote")
        await app.workers.wait_for_complete()
        assert run.calls == []
        assert any("Use /new remote <peer>" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_the_peers_own_refusal_is_shown_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable peer is refused by the RELAY, with its own reason."""
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp")])
    run = _Recorder(ok=False)
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp")
        await app.workers.wait_for_complete()
        assert run.argv[:3] == ["sessions", "--peer", "damian-mbp"]
        assert any("start it with `lop network start`" in text for text in _notices(app)), _notices(
            app
        )


@pytest.mark.asyncio
async def test_bare_new_is_still_a_local_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression this signature change had to avoid.

    ``/new`` used to ignore its argument entirely; reading it must not touch the
    zero-argument form, which is the cold-launch path the sidebar and the
    transcript both depend on.
    """
    from tests.unit.tui.test_app_pilot import (
        FakeSession,
        OperatorApp,
        _factory,
        _resume_factory,
    )

    boots: list[str | None] = []
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new")
        for _ in range(60):
            await pilot.pause()
            if boots == [None]:
                break
        assert boots == [None]
        assert run.calls == []


@pytest.mark.asyncio
async def test_a_single_word_stays_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/new my-project`` is the desktop's selection, not a peer."""
    from tests.unit.tui.test_app_pilot import (
        FakeSession,
        OperatorApp,
        _factory,
        _resume_factory,
    )

    boots: list[str | None] = []
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new my-project")
        for _ in range(60):
            await pilot.pause()
            if boots == [None]:
                break
        assert boots == [None]
        assert run.calls == []


def test_every_peerless_notice_rung_names_the_joiner_verb() -> None:
    """UX round 3, U17: the joiner's verb was unshowable at every width in use.

    The ladder's first rung names both verbs and needs ~91 cells, which the dock's
    notice row only has on a very wide terminal — so at 110, 80 and 60 columns the
    user with a token in hand was told about ``invite`` and nothing else, and the
    ``/network join`` they needed was one verb away in a surface they were not
    standing in. Every rung names ``join`` now; ``invite`` rides along while the
    cells allow it rather than being promised and then cropped (a rung that names
    one verb and loses the other is the failure D16 recorded).
    """
    from local_operator.tui.app import NO_PEERS_NOTICE_RUNGS

    assert NO_PEERS_NOTICE_RUNGS, "the peerless notice has no rungs at all"
    for rung in NO_PEERS_NOTICE_RUNGS:
        assert "/network join" in rung, rung
    # The JOINER'S verb comes FIRST in every rung that has to COMPRESS — the
    # widest rung is the full sentence, where both commands are on screen whole
    # and the order hides nothing (design round 2, D16's rung, unchanged). The
    # user standing at an empty `/new` with a token in hand is the one who cannot
    # act, and the inviter is already told what to run by the invite the mint
    # prints, so `join` is what leads wherever the row is cutting something.
    compressed = [rung for rung in NO_PEERS_NOTICE_RUNGS[1:] if "invite" in rung]
    assert compressed, "no rung names how a first peer is minted"
    for rung in compressed:
        assert rung.index("/network join") < rung.index("invite"), rung
    # Monotone: the ladder is ordered widest-first, which is what lets
    # `_fitted_notice` pick by `cell_len` rather than by index.
    widths = [len(rung) for rung in NO_PEERS_NOTICE_RUNGS]
    assert widths == sorted(widths, reverse=True), widths


# ---------------------------------------------------------------------------
# the create and the view are ONE act (QA round 1 integration, Q-INT-2)
# ---------------------------------------------------------------------------


def test_the_create_receipt_names_the_id_the_surface_opens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reader is pinned to the WRITER, not to a shape this end guessed.

    ``/new remote`` opens the session it created, and the id reaches this surface in
    ``lop network sessions --create``'s own receipt (its first line is
    ``session: <id>``) — the CLI is a subprocess here, so a payload would have to be
    printed into the transcript instead. This drives the CLI's create branch, the
    handler argv dispatches to, with only the relay's answer faked, and asserts the
    id the peer minted is the id ``created_session_id`` recovers from the lines it
    printed. A reworded receipt fails HERE rather than silently leaving every
    ``/new remote`` on the session it stood in before.
    """
    import argparse

    from local_operator.cli import build_cli_parser
    from local_operator.network import cli as network_cli
    from local_operator.tui.network_cli import created_session_id
    from tests.unit.network import conftest as net_fixtures

    parser: Any = build_cli_parser()
    for name in ("network", "sessions"):
        parser = net_fixtures.subcommands_of(parser)[name]
    assert isinstance(parser, argparse.ArgumentParser)

    monkeypatch.setattr(
        network_cli,
        "_relay_answer",
        lambda op, **fields: {"session_id": MINTED, "admitted": True, "detail": "prompt admitted"},
    )
    args = parser.parse_args(["--peer", "damian-mbp", "--create", "--prompt", "hello"])
    assert network_cli._cmd_sessions(args) == 0
    printed = capsys.readouterr().out.splitlines()
    assert any(line.strip() == f"session: {MINTED}" for line in printed), printed
    assert created_session_id(printed) == MINTED
    # ``_reported`` refuses a create whose answer carries no id, so a receipt with
    # no id line at all is the shape to read as "nothing to open" — never as an id
    # to go looking for.
    assert created_session_id(["created on damian-mbp", "prompt admitted"]) == ""
    assert created_session_id(["the session: is not this line"]) == ""


@pytest.mark.asyncio
async def test_a_remote_create_opens_the_session_it_created(
    monkeypatch: pytest.MonkeyPatch, _peer_catalogue_is_stubbed: list[float]
) -> None:
    """Q-INT-2's headline: the pane follows the session onto the peer.

    The receipt was always right and the pane was always somewhere else, so the
    next prompt, a steer and the prompt after that all landed in the LOCAL session
    the user was standing in. A local ``/new`` never left that gap — it rebuilds
    onto the session it made — so the remote form now opens what it created.
    """
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp", role="admin")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    opened: list[str] = []

    def _record(self: Any, session_id: str, root: Any) -> bool:
        opened.append(session_id)
        return True

    monkeypatch.setattr("local_operator.tui.app.OperatorApp._open_session_or_refuse", _record)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp")
        await app.workers.wait_for_complete()
        # THE RECEIPT IS STILL THE USER'S, and it is painted BEFORE the open: the CLI's
        # own lines are what the verb reported.
        assert f"session: {MINTED}" in _transcript_text(app), _transcript_text(app)
        assert opened == [MINTED], "the session this create made on the peer was not opened"
        # THE ROW IS RE-READ, not taken from the sidebar's 20 s cache: the session was
        # minted a moment ago, and a cache-first miss would read as "cannot open what it
        # just made".
        assert _peer_catalogue_is_stubbed == [0.0], _peer_catalogue_is_stubbed


@pytest.mark.asyncio
async def test_a_session_this_device_cannot_list_yet_says_where_it_went(
    monkeypatch: pytest.MonkeyPatch, _peer_catalogue_is_stubbed: list[float]
) -> None:
    """Q-INT-2's second half: the pane must not IMPLY the remote session is driven.

    The peer answered, the receipt named the session and the id, and this device's
    own catalogue does not carry a row for it (the stub returns none) — so nothing
    was opened, and the user is standing in a LOCAL conversation. Silence there is
    the finding: they would type on believing the pane is the session they named.
    Says the id, the device, where they are, and the one command that opens it.
    """
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp", role="admin")])
    run = _Recorder()
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp")
        await app.workers.wait_for_complete()
        said = [text for text in _notices(app) if MINTED in text]
        assert said, f"nothing told the user where the session went: {_notices(app)}"
        assert f"/resume {MINTED}" in said[-1], said[-1]
        assert "damian-mbp" in said[-1], said[-1]


@pytest.mark.asyncio
async def test_a_refused_create_opens_nothing(
    monkeypatch: pytest.MonkeyPatch, _peer_catalogue_is_stubbed: list[float]
) -> None:
    """A refusal is the CLI's sentence and there is no session to open."""
    _peers(monkeypatch, [KnownPeer(device_id="d_aaaa", name="damian-mbp", role="admin")])
    run = _Recorder(ok=False)
    monkeypatch.setattr("local_operator.tui.app.run_network", run)
    opened: list[str] = []
    monkeypatch.setattr(
        "local_operator.tui.app.OperatorApp._open_session_or_refuse",
        lambda self, session_id, root: (opened.append(session_id), True)[1],
    )
    app = _app_fixture()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/new remote damian-mbp")
        await app.workers.wait_for_complete()
        assert opened == [], opened
        assert _peer_catalogue_is_stubbed == [], "a refusal re-read the catalogue"
        assert any("relay is not running" in text for text in _notices(app)), _notices(app)
