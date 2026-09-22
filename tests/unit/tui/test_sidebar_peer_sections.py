"""The sidebar's REMOTE rows: the `⇄` slot, the peer sections, the tooltip.

R6's TUI half (``docs/design/mesh-ui.md`` §1.3) is one list with a local/remote
annotation, and the design makes two claims this file holds:

1. **The mark lives in the CURSOR-PREFIX slot (columns 0-1)**, never in column 2 —
   that column belongs to ``row_state_mark``'s urgency ladder, and locality is a
   durable property of the same kind the pin is. So the pin's own docstring's rule
   ("a durable property must not displace the ladder") applies here too, and the
   mark costs zero new cells: the title still starts at column 4 in every case.
2. **A device with no peers paints byte-identically to before.** The rows are
   local, they carry no mark, no peer heading exists, and the four tier headings
   are the same four strings. That is the *before* frame of the visual pair, and
   it is asserted here as well as by pixels because a regression in it is the one
   this change must not cause.

WHY THE FIXTURES CARRY THE FIELDS BY HAND: these rows are driven the way the
design's evidence plan drives them — stamped with the mobility fields — which is
the contract the producer has to satisfy, and it keeps the rendering half of the
test independent of the read half.

THE PRODUCER NOW EXISTS (``local_operator/session/peer_rows.py``, review round 4
MINOR 3): the sidebar's poll appends what it returns, which is what this file's
last test drives. ``network/projection.py``'s catalogue is still a live relay
read the sidebar may not make from a frame, which is why that producer caches and
bounds it and why the rows here are still hand-stamped for everything that is
about RENDERING.
"""

from __future__ import annotations

import time

import pytest

from local_operator.resume import SessionRow
from local_operator.session.catalog import CatalogEntry
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_session_sidebar import _plain, _sidebar_with, _sub

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _remote(
    sid: str,
    *,
    device: str = "d_aaaa",
    label: str = "damian-mbp",
    active: bool = False,
    reachable: bool = True,
    reason: str = "",
    stale: bool = False,
    name: str = "",
) -> CatalogEntry:
    """A row on ANOTHER device: the field set the projection must supply."""
    return CatalogEntry(
        SessionRow(
            sid,
            time.time(),
            name or f"Session {sid}",
            live_state="busy" if active else "",
            locality="remote",
            owner_device=device,
            owner_device_name=label,
            reachable=reachable,
            unreachable_reason=reason,
            placement_stale=stale,
        )
    )


TIER_HEADINGS = {"★ Pinned", "Active Sessions", "Previous Sessions", "⌥ Subagent Runs"}


def _section_headings(sidebar) -> list[str]:
    """The section headings the painter would emit, in order, with their keys."""
    return [
        kind.removeprefix("header:")
        for kind, _entry in sidebar._display_rows()
        if kind.startswith("header:")
    ]


def _headings(lines: list[str]) -> list[str]:
    return [
        line.strip()
        for line in lines
        if line.strip() in TIER_HEADINGS or line.strip().startswith("⇄")
    ]


def _line_with(lines: list[str], needle: str) -> str:
    for line in lines:
        if needle in line:
            return line
    raise AssertionError(f"{needle!r} not in:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# the mark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_remote_row_carries_the_mark_and_the_title_does_not_move() -> None:
    """Claim 1: `⇄` occupies columns 0-1, and column 4 is still the title."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [_plain("mine", active=True), _remote("theirs", name="Remote work")]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        remote = _line_with(lines, "Remote work")
        local = _line_with(lines, "Session mine")
        # The remote row: the locality mark in cell 1 of the reserved slot, with
        # cell 0 blank (this row is neither the cursor nor pinned). Design round
        # 1, D4: the mark keeps cell 1 at EVERY state of cell 0, so the row the
        # user is about to act on is not the one row that stops saying it is
        # remote — and because cell 0 was already reserved, no title moves.
        assert remote[:2] == " ⇄"
        assert local[:2] == "  "
        # Same column for the title on both rows: the mark is not paid for by
        # narrowing the title, which is what "zero new cells" means.
        assert remote.index("Remote work") == local.index("Session mine") == 4


@pytest.mark.asyncio
async def test_a_local_row_gains_no_mark() -> None:
    """The absent-claim case: `locality=""` renders exactly as it always has."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # One row with `locality="local"` explicitly and one with the DEFAULT:
        # both must paint byte-identically, because "" means "not stated" and a
        # row from this machine's store is local either way.
        entries = [
            _plain("mine", active=True),
            CatalogEntry(
                SessionRow(
                    "stated",
                    time.time(),
                    "Session stated",
                    live_state="busy",
                    locality="local",
                )
            ),
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        assert "⇄" not in "\n".join(lines)
        first = _line_with(lines, "Session mine")
        second = _line_with(lines, "Session stated")
        # Columns 0-1 blank on BOTH rows — the mark's absence is the whole
        # claim, and the state mark in column 2 is the ladder's business (a busy
        # row spins, so it is not asserted to a literal here).
        assert first[:2] == second[:2] == "  "
        assert first.index("Session mine") == second.index("Session stated")


@pytest.mark.asyncio
async def test_the_caret_and_the_pin_take_cell_zero_and_never_the_mark() -> None:
    """Cell 0 is the caret's or the pin's; cell 1 is ALWAYS the locality mark.

    Design round 1, D4. The first cut gave the caret and the pin the WHOLE
    two-cell slot, so a remote row under the cursor — the one row a user is
    deciding about — painted no mark, and the peer heading that carries the same
    fact independently sat two lines above it on a screen being scrolled. Both
    facts now ride the slot that was already reserved, and the title still starts
    at column 4 in every one of the four states.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [
            _remote("pinned-remote", device="d_aaaa", label="A", name="Pinned remote"),
            _remote("plain-remote", device="d_aaaa", label="A", name="Plain remote"),
        ]
        sidebar = await _sidebar_with(pilot, app, entries, pins=("pinned-remote",))
        # The cursor on a REMOTE row: the pair the design asked for a frame of.
        # Focused, because the caret is only painted when the list HAS the focus
        # (`render`: `cursor = self.has_focus and entry.id == self.cursor_id`) —
        # which is exactly why the design's round-1 frames, all unfocused, could
        # not show this state at all.
        sidebar.focus()
        sidebar.cursor_id = "plain-remote"
        await pilot.pause()
        assert sidebar.has_focus, "the caret is only painted on a FOCUSED list"
        lines = sidebar.render().plain.splitlines()
        pinned = _line_with(lines, "Pinned remote")
        plain = _line_with(lines, "Plain remote")
        assert pinned[:2] == "★⇄", repr(pinned)
        assert plain[:2] == "›⇄", repr(plain)
        # Nothing moved: the title is at column 4 on both, exactly as a local row.
        assert pinned.index("Pinned remote") == plain.index("Plain remote") == 4


# ---------------------------------------------------------------------------
# the sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_peer_forms_one_section_after_previous_and_before_subagent() -> None:
    """Claim 1's ordering, and contiguity for each device separately."""
    from local_operator.tui.widgets.session_sidebar import _SECTION_PEER_RANK

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        entries = [
            _plain("mine", active=True),
            _sub("run1", label="ship it", agent="coder"),
            _remote("a1", device="d_aaaa", label="damian-mbp", name="Peer A one"),
            _plain("old"),
            _remote("b1", device="d_bbbb", label="radiant-m4", name="Peer B one"),
            _remote("a2", device="d_aaaa", label="damian-mbp", name="Peer A two"),
        ]
        sidebar = await _sidebar_with(pilot, app, entries, show_subagents=True, total=1)
        ranks = [
            sidebar._section_of(entry)
            for _kind, entry in sidebar._display_rows()
            if entry is not None
        ]
        # Ranks are ordered, so no section is split by another.
        assert ranks == sorted(ranks)
        assert set(ranks) == {1, _SECTION_PEER_RANK, 2, 4}
        lines = sidebar.render().plain.splitlines()
        # Two devices are two sections, each heading naming its own device, and
        # each pair of rows sitting under its own heading.
        headings = _headings(lines)
        assert "⇄ damian-mbp" in headings
        assert "⇄ radiant-m4" in headings
        a_one = lines.index(_line_with(lines, "Peer A one"))
        a_two = lines.index(_line_with(lines, "Peer A two"))
        heading = lines.index(_line_with(lines, "⇄ damian-mbp"))
        # Peer A's rows are contiguous BELOW its heading and above Peer B's.
        assert heading < a_one < a_two
        assert a_two < lines.index(_line_with(lines, "⇄ radiant-m4"))


@pytest.mark.asyncio
async def test_the_peer_heading_stacks_in_the_rows_own_mark_column() -> None:
    """Design round 2, D18: one mark column for a heading and the rows under it.

    Rows paint the locality glyph in cell 1 — cell 0 is the caret-or-pin slot —
    and the heading painted it in cell 0, so the one glyph that says "everything
    under this line is another device" started at a different x from every mark
    it governed and the column did not stack. The heading has no caret and no
    pin, so its cell 0 is empty by definition: indent it, and the tier headings
    above (which do own cell 0) stay exactly where they were.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [_remote("a1", label="damian-mbp", name="Peer A one")]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        heading = _line_with(lines, "⇄ damian-mbp")
        row = _line_with(lines, "Peer A one")
        assert heading.index("⇄") == row.index("⇄") == 1, (heading, row)


@pytest.mark.asyncio
async def test_an_unreachable_peer_says_so_in_the_heading() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [
            _remote(
                "a1",
                label="damian-mbp",
                name="Peer A one",
                reachable=False,
                reason="connect_failed:ConnectionRefusedError",
            )
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        assert "⇄ damian-mbp (unreachable)" in _headings(lines)


@pytest.mark.asyncio
async def test_a_nameless_peer_gets_a_heading_from_its_id() -> None:
    """An unnamed device must not produce a heading with nothing after the glyph."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [_remote("a1", device="d_9f2c1a4b", label="", name="Peer A one")]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        assert "⇄ d_9f2c1a" in _headings(lines)


@pytest.mark.asyncio
async def test_the_page_never_overruns_with_several_peer_sections() -> None:
    """``_header_lines`` counts each peer SECTION, not the peer RANK.

    Two peers share a rank and each has a heading of its own, so counting distinct
    ranks would charge one heading for two sections and the painted frame would be
    a line taller than the height it was computed for — the overrun this
    accounting exists to prevent.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [_plain("mine", active=True)]
        entries += [
            _remote(f"p{index}", device=f"d_{index:04d}", label=f"device-{index}", name=f"R{index}")
            for index in range(3)
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        rows = sidebar._display_rows()
        headings = [kind for kind, _entry in rows if kind.startswith("header:")]
        # THREE peer headings and one tier heading. Counting distinct RANKS would
        # see two (active + peer), which is the bug this asserts against.
        assert len(headings) == 4, headings
        # Chrome = headings + blanks, per section, and the painted frame fits.
        assert sidebar._header_lines() == 4 * 2 + 3
        assert len(rows) <= sidebar.size.height


# ---------------------------------------------------------------------------
# the tooltip, and the zero-peer invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tooltip_names_the_device_and_the_reason() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [
            _remote("a1", name="Peer A one"),
            _remote(
                "a2",
                name="Peer A two",
                reachable=False,
                reason="asked, and it did not answer",
                stale=True,
            ),
            _plain("mine"),
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        by_id = {entry.id: entry for entry in entries}
        reachable = sidebar._describe(by_id["a1"])
        assert "on damian-mbp" in reachable
        unreachable = sidebar._describe(by_id["a2"])
        assert "on damian-mbp — unreachable: asked, and it did not answer" in unreachable
        assert "(last known state)" in unreachable
        local = sidebar._describe(by_id["mine"])
        # No device clause at all: the local row's tooltip is exactly the three
        # lines it has always been (name, state, id).
        assert local.splitlines()[1:] == ["Recent", "mine"]


@pytest.mark.asyncio
async def test_a_device_with_no_peers_paints_exactly_as_before() -> None:
    """Claim 2, the R9/R16 invariant: no peer, no mark, no peer heading.

    Asserted on the headless four-tier fixture: the headings are the four tier
    names, the rows' first four columns are the caret/mark pair, and no ``⇄``
    appears anywhere. This is the same frame the *before* capture must be
    byte-identical to (§4.1).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        entries = [
            _plain("a1", active=True),
            _sub("s1", label="one", agent="scout"),
            _plain("p1"),
            _plain("pin1", active=True),
        ]
        sidebar = await _sidebar_with(pilot, app, entries, pins=("pin1",), show_subagents=True)
        lines = sidebar.render().plain.splitlines()
        assert _headings(lines) == [
            "★ Pinned",
            "Active Sessions",
            "Previous Sessions",
            "⌥ Subagent Runs",
        ]
        assert "⇄" not in "\n".join(lines)
        # And `_section_of` is unchanged for every row: 0/1/2/4, never the peer rank.
        ranks = {sidebar._section_of(entry) for entry in entries}
        assert ranks == {0, 1, 2, 4}


@pytest.mark.asyncio
async def test_the_poll_adopts_what_the_producer_returns(monkeypatch) -> None:
    """The producer's rows reach the list: the WIRING, not the rendering.

    The rendering half is pinned above against hand-stamped rows. This pins the
    other end of the same claim — `_refresh_sidebar` appends what
    `session.peer_rows` returns — so the `⇄` mark and the per-device heading have
    a live source rather than only a contract fixture, which is what review round
    4's MINOR 3 was about (the session `/new remote <peer>` creates had no surface
    that could see it).
    """
    import local_operator.session.peer_rows as peer_rows_mod

    monkeypatch.setattr(
        peer_rows_mod,
        "peer_session_rows",
        lambda root=None, **kwargs: (
            SessionRow(
                "peer-session-1",
                time.time(),
                "Federated catalogue fan-out",
                live_state="idle",
                locality="remote",
                owner_device="d_radiant",
                owner_device_name="radiant-m4",
            ),
        ),
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        sidebar = app._session_sidebar
        sidebar.set_open(True)
        if app._sidebar_timer is not None:
            app._sidebar_timer.pause()
        app._refresh_sidebar()
        for _ in range(60):
            await pilot.pause()
            if not app._sidebar_refresh_pending:
                break
        lines = sidebar.render().plain.splitlines()
        joined = "\n".join(lines)
        assert "⇄ radiant-m4" in joined, joined
        row = next(line for line in lines if "Federated catalogue" in line)
        # The mark rides cell 1 of the reserved prefix slot: the caret/pin owns
        # cell 0, and the title still starts at column 4.
        assert row[1] == "⇄", repr(row)
        assert row[4] == "F", repr(row)


@pytest.mark.asyncio
async def test_a_peer_that_stopped_answering_keeps_a_heading_with_no_rows() -> None:
    """UX round 3, U16: "my peer has nothing" and "my peer is gone" must differ.

    §8.3 says a peer that does not answer contributes NO ROWS rather than stale
    ones — right, and not what was filed. The section was built from rows, so the
    tier went with them: six sessions the user had been looking at simply
    vanished, and the list read as complete. The heading is the one sentence that
    explains the frame, and it must be the SAME string a live section carries with
    ``reachable`` false, because the state is the same state.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        sidebar = await _sidebar_with(pilot, app, [_plain("mine", active=True)])
        assert "⇄ damian-mbp" not in _headings(sidebar.render().plain.splitlines())
        sidebar.set_silent_peers([("damian-mbp", "connect_failed:ConnectionRefusedError")])
        await pilot.pause()
        lines = sidebar.render().plain.splitlines()
        assert "⇄ damian-mbp (unreachable)" in _headings(lines)
        # THE REASON IS NOT PAINTED. It is the relay's wire token
        # (``connect_failed:ConnectionRefusedError``), which UX round 3 filed as
        # U23 on the listing surface; a heading is not where it gets a second,
        # unlocalised spelling.
        assert "ConnectionRefusedError" not in "\n".join(lines)
        # ...and the peer answering again takes it away, so the line cannot
        # outlive the state it describes.
        sidebar.set_silent_peers([])
        await pilot.pause()
        assert "⇄ damian-mbp" not in _headings(sidebar.render().plain.splitlines())


@pytest.mark.asyncio
async def test_a_silent_peer_keeps_the_peer_rank_on_both_sides_of_answering() -> None:
    """Design round 4, D27: the peer axis' rank is not a function of liveness.

    ``mesh-ui.md`` decision 1 puts the peer axis after `previous` and BEFORE
    `subagent`, and a live peer's section obeys it. Built at the END of the row
    list instead, a silent peer's heading sat below `⌥ Subagent Runs` and moved
    above it the moment the peer recovered — so the list re-ordered itself around
    an event the user did not cause, and two devices stacked in one order
    re-ordered themselves.

    Asserted BOTH ways round, which is what makes it a claim about place rather
    than about one frame: the silent heading and the equivalent answering one
    (a remote row for the same device, ``reachable=False``) put the peer section
    in the same position relative to the subagent tier.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 45)) as pilot:
        await pilot.pause()
        entries = [_plain("mine", active=True), _sub("s1")]
        sidebar = await _sidebar_with(pilot, app, entries, show_subagents=True)
        sidebar.set_silent_peers([("radiant-m4", "connect_failed:ConnectionRefusedError")])
        await pilot.pause()
        silent = _section_headings(sidebar)
        silent_at = silent.index("peer: ⇄ radiant-m4 (unreachable)")
        assert silent_at < silent.index("subagent"), silent

        sidebar.set_silent_peers([])
        sidebar.set_entries([*entries, _remote("p1", label="radiant-m4", reachable=False)])
        await pilot.pause()
        answering = _section_headings(sidebar)
        answering_at = answering.index("peer: ⇄ radiant-m4 (unreachable)")
        assert answering_at < answering.index("subagent"), answering

        # AND THE TWO DEVICES STAY IN ONE ORDER. A second peer's section is
        # placed by its NAME, not by when it was last heard from, so two peers do
        # not swap places when one of them comes back.
        sidebar.set_silent_peers([("radiant-m4", "connect_failed:ConnectionRefusedError")])
        sidebar.set_entries(
            [
                *entries,
                _remote("p1", device="d_aaaa", label="radiant-m4", reachable=False),
                _remote("p2", device="d_bbbb", label="pixel-8", reachable=False),
            ]
        )
        await pilot.pause()
        both = _section_headings(sidebar)
        assert both.index("peer: ⇄ pixel-8 (unreachable)") < both.index(
            "peer: ⇄ radiant-m4 (unreachable)"
        ), both


@pytest.mark.asyncio
async def test_two_row_less_sections_are_separated_by_one_blank() -> None:
    """Design round 4, D28: every boundary in the list is ONE blank row.

    The per-section chrome is `blank, heading, blank`, which is right for a
    section with rows under it and one blank too many between two that have
    none: two unreachable peers came out three rows apart where every other
    boundary is two. The rule is now the builder's — a section declines its
    leading blank when the previous row is already a blank — so the frame a user
    with several lost devices actually gets is the same shape as every other.

    The invariant is asserted over the WHOLE row list, not just the pair, because
    "never two blanks in a row" is what the rule is; the pair is the case that
    exposed it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 45)) as pilot:
        await pilot.pause()
        sidebar = await _sidebar_with(pilot, app, [_plain("mine", active=True)])
        sidebar.set_silent_peers(
            [
                ("radiant-m4", "connect_failed:ConnectionRefusedError"),
                ("pixel-8", "connect_failed:ConnectionRefusedError"),
            ]
        )
        await pilot.pause()
        kinds = [kind for kind, _entry in sidebar._display_rows()]
        doubled = [
            index
            for index, (first, second) in enumerate(zip(kinds, kinds[1:]))
            if first == second == "blank"
        ]
        assert doubled == [], f"two blank rows in a row at {doubled}: {kinds}"
        first = kinds.index("header:peer: ⇄ pixel-8 (unreachable)")
        second = kinds.index("header:peer: ⇄ radiant-m4 (unreachable)")
        assert second - first == 2, kinds[first : second + 1]
        assert kinds[first + 1] == "blank", kinds[first : second + 1]
        # The headers are chrome the frame still fits: the page size is computed
        # from the same model, so a silent section cannot overrun the height.
        assert len(sidebar._display_rows()) <= sidebar.size.height


@pytest.mark.asyncio
async def test_the_silent_peers_heading_is_chrome_the_keyboard_cannot_land_on() -> None:
    """A heading-only section must stay OUT of ``entries``.

    ``action_move``, ``_cursor_index`` and ``_switch_session_from`` all index
    ``self.entries``, so a chrome row that leaked into it would let
    ``ctrl+shift+down`` "switch" to a device and desync open-from-closed
    navigation — the exact hazard ``_display_rows`` documents for every header.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        sidebar = await _sidebar_with(pilot, app, [_plain("mine", active=True)])
        sidebar.set_silent_peers([("damian-mbp", "connect_failed:ConnectionRefusedError")])
        await pilot.pause()
        lines = sidebar.render().plain.splitlines()
        heading_y = next(
            y for y, line in enumerate(lines) if line.strip() == "⇄ damian-mbp (unreachable)"
        )
        assert sidebar._entry_at(heading_y) is None, "the silent-peer heading is a click target"
        assert all(entry.id != "" for entry in sidebar.entries)
