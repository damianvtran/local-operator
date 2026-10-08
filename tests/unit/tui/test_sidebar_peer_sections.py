"""The sidebar's REMOTE rows: the `↗` slot, the merged bins, the tooltip.

R6's TUI half (``docs/design/mesh-ui.md`` §1.3, revised by the operator's
convergence report, 2026-10-05, and by design round 1 of this change) is one
list with a local/remote annotation, and the design makes these claims this
file holds:

1. **The marks live in the cursor slot's locality cell** — `↗` on a remote
   row, `↛` on a remote row that is UNREACHABLE (design round 1, D2/D5: the
   pair must be decodable AT REST, because a Textual tooltip needs a mouse;
   `↗` reads "elsewhere", the external-link convention, where `⇄` read as
   exchange/sync) — never in the mark column: that column belongs to
   ``row_state_mark``'s urgency ladder, and locality is a durable property of
   the same kind the pin is. So the pin's own docstring's rule ("a durable
   property must not displace the ladder") applies here too, and the mark
   costs the title nothing: a remote row and a local row start their titles at
   the same column (ahead of both, the pin cell's own two columns sit since
   issue #1357 slice 2a).
2. **REMOTE ROWS ARE FIRST-CLASS**: they file into the ordinary bins
   (``SessionSidebar._unpinned_rank`` — the place that carries the SHARED
   CONVENTION sentence the desktop sidebar's ``feat/sidebar-remote-rows`` names)
   under the ordinary ordering rule, interleaved with this device's own rows by
   the same ``rank`` key — no per-device section, no heading, no rank of their
   own.
3. **A device with no peers paints byte-identically to before.** The rows are
   local, they carry no mark, no peer heading exists, and the four tier headings
   are the same four strings. That is the *before* frame of the visual pair, and
   it is asserted here as well as by pixels.

WHY THE FIXTURES CARRY THE FIELDS BY HAND: these rows are driven the way the
design's evidence plan drives them — stamped with the mobility fields — which is
the contract the producer has to satisfy, and it keeps the rendering half of the
test independent of the read half.

THE PRODUCER EXISTS (``local_operator/session/peer_rows.py``, review round 4
MINOR 3): the sidebar's poll appends what it returns, which is what this file's
wiring test drives. ``network/projection.py``'s catalogue is still a live relay
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
from tests.unit.tui.test_session_sidebar import (
    _plain,
    _section_of_line,
    _sidebar_with,
    _sub,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _remote(
    sid: str,
    *,
    device: str = "d_aaaa",
    label: str = "damian-mbp",
    network: str = "devmesh",
    active: bool = False,
    born_min_ago: float | None = None,
    reachable: bool = True,
    reason: str = "",
    stale: bool = False,
    name: str = "",
) -> CatalogEntry:
    """A row on ANOTHER device: the field set the projection must supply.

    ``born_min_ago`` stamps ``created_at`` the way the producer does — from the
    peer's ``started`` claim (``session/peer_rows.py``'s row construction) —
    because ordering a remote row among local ones is one of the claims below.
    ``None`` leaves the zero the pre-convergence producer emitted.
    """
    now = time.time()
    return CatalogEntry(
        SessionRow(
            sid,
            now,
            name or f"Session {sid}",
            live_state="busy" if active else "",
            locality="remote",
            owner_device=device,
            owner_device_name=label,
            owner_network_name=network,
            reachable=reachable,
            unreachable_reason=reason,
            placement_stale=stale,
            created_at=0.0 if born_min_ago is None else now - born_min_ago * 60.0,
        )
    )


TIER_HEADINGS = {"★ Pinned", "Active Sessions", "Previous Sessions", "⌥ Subagent Runs"}


def _tier_headings(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip() in TIER_HEADINGS]


def _header_kinds(sidebar) -> list[str]:
    return [kind for kind, _entry in sidebar._display_rows() if kind.startswith("header:")]


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
    """Claim 1: `↗` sits in the locality cell, and both titles start together."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [_plain("mine", active=True), _remote("theirs", name="Remote work")]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        remote = _line_with(lines, "Remote work")
        local = _line_with(lines, "Session mine")
        # The remote row: the locality mark in the slot's second cell — after
        # the pin cell's two columns and the caret's — with the first three
        # blank (this row is neither the cursor nor pinned). Design round 1,
        # D4: the mark keeps its own cell at EVERY state, so the row the user
        # is about to act on is not the one row that stops saying it is remote.
        assert remote[:4] == "   ↗", repr(remote)
        assert local[:4] == "    "
        # Same column for the title on both rows: the mark is not paid for by
        # narrowing the title relative to a local row, which is what the
        # "costs zero new cells" claim means (the pin cell ahead of both is
        # issue #1357 slice 2a's own two columns, priced once for every row).
        assert remote.index("Remote work") == local.index("Session mine") == 6


@pytest.mark.asyncio
async def test_an_unreachable_remote_row_carries_the_struck_mark_at_rest() -> None:
    """Design round 1, D2: unreachable is decodable WITHOUT a hover.

    A Textual tooltip needs a mouse, so the row itself must say it: the
    locality cell paints `↛` — the same arrow family as `↗` with the stroke
    that says "does not get there" — in the same cell, so the unreachable row
    is not byte-identical to a live one and nothing reflows between the two
    states. The tooltip's `unreachable · <reason>` line (D4) remains the
    available expansion; this is the at-rest half.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [
            _remote("live", name="Live elsewhere"),
            _remote(
                "gone",
                name="Gone elsewhere",
                reachable=False,
                reason="connect_failed:ConnectionRefusedError",
                stale=True,
            ),
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        live = _line_with(lines, "Live elsewhere")
        gone = _line_with(lines, "Gone elsewhere")
        assert live[:4] == "   ↗", repr(live)
        assert gone[:4] == "   ↛", repr(gone)
        # One cell, one column, no reflow: the two states differ in the glyph
        # alone, and the title starts at the same place on both.
        assert live.index("Live elsewhere") == gone.index("Gone elsewhere") == 6


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
        assert "↗" not in "\n".join(lines)
        assert "↛" not in "\n".join(lines)
        first = _line_with(lines, "Session mine")
        second = _line_with(lines, "Session stated")
        # The pin cell and the caret cell are blank on BOTH rows — the mark's
        # absence is the whole claim — and the state mark in the next column is
        # the ladder's business (a busy row spins, so it is not asserted to a
        # literal here).
        assert first[:4] == second[:4] == "    "
        assert first.index("Session mine") == second.index("Session stated")


@pytest.mark.asyncio
async def test_the_pin_and_the_caret_ride_their_own_cells_and_never_the_mark() -> None:
    """The pin has its own cell pair; the caret keeps the slot beside it.

    Issue #1357 slice 2a moved the durable star out of the cursor-prefix slot's
    cell 0, which it shared with the caret — so a pinned row under the cursor
    could draw only one of the two, and neither column was a stable pointer
    target. The locality mark keeps its cell in EVERY state (design round 1,
    D4: the one row a user is deciding about may not be the one row that stops
    saying it is remote), and the mark column stays the urgency ladder's alone.
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
        assert pinned[:4] == "★  ↗", repr(pinned)
        assert plain[:4] == "  ›↗", repr(plain)
        # THE STATE THE OLD SHARED CELL COULD NOT DRAW: pinned AND the cursor.
        sidebar.cursor_id = "pinned-remote"
        await pilot.pause()
        both = _line_with(sidebar.render().plain.splitlines(), "Pinned remote")
        assert both[:4] == "★ ›↗", repr(both)
        # Nothing moved: the title starts at the same column on every state —
        # the pin cell's two columns, then the caret and locality cells.
        assert plain.index("Plain remote") == both.index("Pinned remote") == 6


# ---------------------------------------------------------------------------
# the bins (the convention this change exists for)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_rows_file_into_the_ordinary_bins_by_their_own_state() -> None:
    """Claim 2: a remote row is binned by ``active``, exactly like a local row.

    One remote row busy (active) and one cold (previous), beside local rows in
    the same two bins, and the frame's headings are the tier names ALONE — no
    per-device ``↗ <device>`` section (the retired heading's glyph was `⇄`),
    which is the segregation the convergence report
    removed. The two remote rows are on DIFFERENT devices and still share the
    ordinary bins, so nothing per-device is left anywhere.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        entries = [
            _plain("mine", active=True),
            _remote("a1", label="damian-mbp", name="Peer A busy", active=True),
            _plain("old"),
            _remote("b1", device="d_bbbb", label="radiant-m4", name="Peer B cold"),
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        assert _tier_headings(lines) == ["Active Sessions", "Previous Sessions"]
        # The chrome IS the tier list: a kind outside the table would be a
        # section the names do not own — which is exactly what the per-device
        # peer headings were.
        assert _header_kinds(sidebar) == ["header:active", "header:previous"]
        active = lines.index(_line_with(lines, "Active Sessions"))
        previous = lines.index(_line_with(lines, "Previous Sessions"))
        # Busy files with busy; cold with cold — each by its OWN state, and the
        # remote busy row sits INSIDE the active run rather than in a block of
        # its own.
        assert active < lines.index(_line_with(lines, "Peer A busy")) < previous
        assert previous < lines.index(_line_with(lines, "Peer B cold"))
        # Every painted section is one contiguous run (nothing is split).
        ranks = [
            sidebar._section_of(entry)
            for _kind, entry in sidebar._display_rows()
            if entry is not None
        ]
        assert ranks == sorted(ranks)


@pytest.mark.asyncio
async def test_remote_rows_interleave_with_local_rows_by_the_same_ordering_key() -> None:
    """The SAME ordering rule: position inside a bin comes from ``rank``'s
    ``-created_at`` key, so a remote row born BETWEEN two local rows sorts
    between them rather than below both.

    ``-created_at`` is the contested half of this: the pre-convergence producer
    stamped remote rows with no birth, which parked every one of them at the
    BOTTOM of its bin — the soft form of the same segregation. The producer now
    stamps the peer's ``started`` claim (``session/peer_rows.py``), and this is
    what that buys on the frame.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        now = time.time()
        entries = [
            CatalogEntry(
                SessionRow(
                    "mine-new",
                    now,
                    "Newest local",
                    live_state="busy",
                    created_at=now - 60,
                )
            ),
            _remote("theirs-mid", name="Middle remote", active=True, born_min_ago=30),
            CatalogEntry(
                SessionRow(
                    "mine-old",
                    now,
                    "Oldest local",
                    live_state="busy",
                    created_at=now - 5400,
                )
            ),
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        lines = sidebar.render().plain.splitlines()
        first = lines.index(_line_with(lines, "Newest local"))
        second = lines.index(_line_with(lines, "Middle remote"))
        third = lines.index(_line_with(lines, "Oldest local"))
        assert first < second < third, "\n".join(lines)


@pytest.mark.asyncio
async def test_a_pinned_remote_row_lifts_into_pinned_and_keeps_its_mark() -> None:
    """First-class includes the pin: a pinned remote row paints under
    ``★ Pinned`` with its `↗` mark kept — the pin's lift is the ordinary one."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entries = [_plain("mine", active=True), _remote("a1", name="Peer A one")]
        sidebar = await _sidebar_with(pilot, app, entries, pins=("a1",))
        lines = sidebar.render().plain.splitlines()
        assert _section_of_line(lines, "Peer A one") == "★ Pinned"
        assert "↗" in _line_with(lines, "Peer A one")


@pytest.mark.asyncio
async def test_several_remote_devices_add_no_chrome() -> None:
    """Three devices, two sections: the per-device heading cost is gone.

    The old accounting charged ONE heading per device, so this asserted
    ``len(headings) == 4`` for three peers plus a tier. There is nothing per
    device to charge now — the section count is the tier count — and the same
    invariant as before still holds: the frame fits the height it was computed
    for.
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
        assert _header_kinds(sidebar) == ["header:active", "header:previous"]
        # Chrome = headings + blanks, per section — exact, because every
        # section has rows.
        assert sidebar._header_lines() == 2 * 2 + 1
        assert len(rows) <= sidebar.size.height


# ---------------------------------------------------------------------------
# the tooltip, and the zero-peer invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tooltip_names_the_device_the_network_and_the_reason() -> None:
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
            _remote("a3", device="d_9f2c1a4b", label="", network="", name="Peer A three"),
            _plain("mine"),
        ]
        sidebar = await _sidebar_with(pilot, app, entries)
        by_id = {entry.id: entry for entry in entries}
        reachable = sidebar._describe(by_id["a1"])
        # DEVICE AND NETWORK, in one clause: this is the TUI half of the pair
        # whose sentence `_unpinned_rank` records — the desktop sidebar's
        # `feat/sidebar-remote-rows` reads the same two facts on its hover and
        # accessible name.
        assert "on damian-mbp · devmesh" in reachable
        # THE REASON GETS ITS OWN TERSE LINE (design round 1, D4): the device
        # clause stays intact, and the unreachable fact keeps the same `·`
        # separator and the shared gloss (`peer_reason_words`) as its own line
        # under it — the fused `— unreachable: <prose>` sentence is gone.
        unreachable = sidebar._describe(by_id["a2"])
        assert unreachable.splitlines()[2] == "on damian-mbp · devmesh (last known state)"
        assert unreachable.splitlines()[3] == "unreachable · asked, and it did not answer"
        assert unreachable.splitlines()[4] == "a2"
        # DEGRADED, NOT GUESSED: a nameless device falls back to the id's tail
        # (`owner_label`) and an unreadable membership omits the network clause
        # rather than printing an id or a placeholder.
        nameless = sidebar._describe(by_id["a3"])
        assert nameless.splitlines()[2] == "on d_9f2c1a"
        local = sidebar._describe(by_id["mine"])
        # No device clause at all: the local row's tooltip is exactly the three
        # lines it has always been (name, state, id).
        assert local.splitlines()[1:] == ["Recent", "mine"]


@pytest.mark.asyncio
async def test_a_device_with_no_peers_paints_exactly_as_before() -> None:
    """Claim 3, the R9/R16 invariant: no peer, no mark, no peer heading.

    Asserted on the headless four-tier fixture: the headings are the four tier
    names, the rows carry only the pin/caret/mark columns, and no ``↗`` or
    ``↛`` appears
    anywhere. This is the same frame the *before* capture must be byte-identical
    to (§4.1).
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
        assert _tier_headings(lines) == [
            "★ Pinned",
            "Active Sessions",
            "Previous Sessions",
            "⌥ Subagent Runs",
        ]
        assert "↗" not in "\n".join(lines)
        assert "↛" not in "\n".join(lines)
        # And `_section_of` is unchanged for every row: 0/1/2/4. Rank 3 is
        # UNASSIGNED — the retired peer axis held it — and `subagent` keeps 4
        # rather than renumbering for a hole nothing reads.
        ranks = {sidebar._section_of(entry) for entry in entries}
        assert ranks == {0, 1, 2, 4}


@pytest.mark.asyncio
async def test_the_poll_adopts_what_the_producer_returns(monkeypatch) -> None:
    """The producer's rows reach the list: the WIRING, not the rendering.

    The rendering half is pinned above against hand-stamped rows. This pins the
    other end of the same claim — `_refresh_sidebar` appends what
    `session.peer_rows` returns — so the `↗` mark has a live source rather than
    only a contract fixture, which is what review round 4's MINOR 3 was about
    (the session `/new remote <peer>` creates had no surface that could see it).
    The row is ``idle``, which is an ACTIVE state, so it must adopt into the
    ordinary ``Active Sessions`` bin: the bins claim, end to end, from producer
    to painted frame.
    """
    import local_operator.session.peer_rows as peer_rows_mod

    monkeypatch.setattr(
        peer_rows_mod,
        "read_listing",
        lambda root=None, **kwargs: (
            (
                SessionRow(
                    "peer-session-1",
                    time.time(),
                    "Federated catalogue fan-out",
                    live_state="idle",
                    locality="remote",
                    owner_device="d_radiant",
                    owner_device_name="radiant-m4",
                    owner_network_name="devmesh",
                ),
            ),
            (),
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
        # No device heading exists, and the row files under the ordinary bin
        # for its state — the two halves of claim 2, through the REAL poll.
        assert "↗ radiant-m4" not in joined, joined
        assert _header_kinds(sidebar) == ["header:active"]
        row = next(line for line in lines if "Federated catalogue" in line)
        # The mark rides the locality cell — after the pin cell's two columns
        # and the caret's — and the title starts after the mark column.
        assert row[3] == "↗", repr(row)
        assert row[6] == "F", repr(row)
