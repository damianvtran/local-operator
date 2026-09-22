"""The ``/network`` panel: this device's mesh, two-phase like ``/info``.

WHY A PANEL AND NOT A NOTICE LIST (``docs/design/mesh-ui.md`` §1.1.3/§1.1.4). A
notice is one line and retires; the mesh question — which networks am I in, which
peers do I have, is my relay answering — is a table the user reads and then acts
on, from the same screen. ``InfoScreen`` is the shape this copies: pushed BEFORE
the slow work, filled by a worker, cancellable, and never blank.

TWO PHASES, and the split is which reads can block:

* the FIRST frame is DISK ONLY — this device's identity file, its relay record
  and the member lists it holds (``network/store.py``, ``network/identity.py``).
  Those are in-memory or small-file reads, so the panel is useful from frame one
  with the relay stopped, which is the state people open it in.
* the SECOND frame is the RELAY's answer: ``lop network ls --json`` and
  ``peers --json`` dial every member with a 12 s probe budget. That runs in a
  worker, off the loop, because 12 s of dialing on the paint path is three
  hundred frames of freeze. Both halves are labelled so a reader can tell the
  device's own record from what the relay just verified — that provenance is the
  same distinction ``lop network ls`` marks as ``stale``.

THE DESTRUCTIVE VERBS ARE HANDED TO THE COMPOSER, never executed here. ``d`` and
``shift+P`` post the typed command (``NetworkCommandRequested``) and the app puts
it in the composer unsubmitted: the incident controls are deliberately words a
human types (§1.5), and a screen that both selected a network and fired the
revoke would be the one-step accident that decision exists to prevent. The
confirmation therefore lives in ONE place — the command's own — and the panel
cannot drift from it.

EVERY NETWORK IMPORT IS FUNCTION-LOCAL. ``local_operator/cli.py`` imports this
package lazily so a run that never touches the mesh pays nothing for it; a widget
that imported ``store`` at module scope would put that cost back on every TUI
start, including the overwhelming majority with no mesh at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.resume import UNNAMED_DEVICE
from local_operator.tui.network_cli import LISTING_TIMEOUT_S, NetworkRun, run_network
from local_operator.tui.widgets.aside_panel import ASIDE_COPY_KEY

#: The minimum content width a row may be laid out in. The panel is 90% of the
#: terminal (the shared `.analytics-panel` rule), so this only bites on a very
#: narrow terminal, where it keeps the columns from being computed to zero.
_MIN_CARD_WIDTH = 40


#: How much of an id the panel shows, and WHY ONE NUMBER FOR ALL THREE.
#:
#: The frame mixed three widths — the device's full 34 cells, a network's last
#: 12, a peer's first 12 — so a reader could not compare a peer's id with the
#: device's, which is the one comparison these columns exist for (design round 1,
#: D6). ``<kind>_`` + 10 hex is the shape: the prefix says what kind of id it is,
#: the ten hex are enough to tell two of them apart by eye, and the FULL value is
#: what every command takes (``/network show``, the ``#`` form of a peer token).
#: This shortens a COLUMN, never the data.
_ID_CELLS = 12


#: The relay's ``reason`` tokens, in the words a person reads. See
#: :func:`peer_reason_words` for why the token itself is not the answer.
_PEER_REASON_WORDS: dict[str, str] = {
    "no_endpoint": "no address published for it",
    "not_a_member": "it is not a member of this network",
    "member_removed": "it was removed from this network",
}


def short_id(value: str) -> str:
    """The first :data:`_ID_CELLS` cells of an id — the ONE abbreviation.

    FIRST cells, not last: the prefix carries the kind (``d_``/``n_``) and is
    what makes a peer's id comparable with the device's, which the last-N form
    threw away. ``_network_id_tail``'s old rationale ("the prefix carries no
    information a reader can use") was true of one id in isolation and false of
    the three columns read together.
    """
    return value[:_ID_CELLS] if len(value) > _ID_CELLS else value


def peer_reason_words(reason: str) -> str:
    """The relay's ``reason`` token said in words, and never empty.

    WHY NOT THE TOKEN ITSELF (design round 1, D3). ``reason`` is a PROTOCOL
    detail: ``connect_failed:ConnectionRefusedError`` is a stage, a colon with no
    space after it, and a Python exception class. The panel is a user surface —
    it named the transport's failure mode where the reader needs "that device is
    not there" — and it was the only place in the product that spelled the
    condition that way, while the sidebar says the same fact as ``(unreachable)``.
    The raw token is not lost: ``lop network peers`` prints it per candidate, and
    that is the detail view this panel is a summary OF.

    Prefix-matched on the stage before the ``:`` rather than on the exception
    class, because the tail is whatever the dial raised: a vocabulary of Python
    class names would be a second registry to keep, and the distinction a reader
    needs is "nothing answered", not which exception said so.
    """
    token = (reason or "").strip()
    if token in _PEER_REASON_WORDS:
        return _PEER_REASON_WORDS[token]
    if token.startswith("unreachable:"):
        return "no address of it answered"
    # The default is also the answer for a stage whose tail names a failure this
    # module has never seen: an unreadable reason is still "it did not answer" to
    # the person reading the row.
    return "it did not answer"


#: The copy chord, the SAME constant `/info` and the aside bind rather than a
#: second literal: "copy this surface out" is one gesture across the app, and
#: importing the constant is what stops the three from drifting. §1.1.4 says this
#: screen copies ``InfoScreen``, and ``InfoScreen`` carries this binding — the
#: first cut did not, which left the one screen a person pastes into a support
#: message as the only one without a way to get it there (review round 4, NIT 4).
NETWORK_COPY_KEY = ASIDE_COPY_KEY


@dataclass
class NetworkEntry:
    """One network as this device holds it, plus whatever the relay added.

    ``verified`` is the provenance flag and the reason this type exists rather
    than a dict: the panel has to be able to say "the relay answered this" beside
    a row, and a row that came from the local record must not be able to claim it.
    """

    network_id: str
    name: str = ""
    epoch: int = 0
    role: str = ""
    members: int = 0
    trust: str = ""
    stale: bool = False
    verified: bool = False

    @property
    def label(self) -> str:
        return self.name or self.network_id


@dataclass
class PeerEntry:
    """One peer membership, as the local member list records it."""

    device_id: str
    name: str = ""
    role: str = ""
    network_id: str = ""
    reachable: bool | None = None
    reason: str = ""

    @property
    def label(self) -> str:
        return self.name or self.device_id


@dataclass
class NetworkLocal:
    """Everything the first frame can be painted from, with no dial at all."""

    device_id: str = ""
    device_name: str = ""
    identity_present: bool = False
    relay_state: str = ""
    relay_pid: int = 0
    networks: list[NetworkEntry] = field(default_factory=list)
    peers: list[PeerEntry] = field(default_factory=list)


def capture_local(root: Path | None = None) -> NetworkLocal:
    """The disk-only half of the panel. Safe to call synchronously on the loop.

    Deliberately NOT ``relay.status()``: that one shells launchd and forks probes
    (``network/cli.py``'s ``status`` verb pays it, and it is documented as the
    first command an agent runs rather than something a paint path may do). This
    reads the relay RECORD and the identity file, which is what a first frame
    needs, and leaves the live verdict to the worker's own ``status --json``.
    """
    from local_operator.network import identity, store

    local = NetworkLocal()
    device = identity.load(root)
    if device is not None:
        local.identity_present = True
        local.device_id = getattr(device, "device_id", "") or ""
        local.device_name = getattr(device, "name", "") or ""
    record, state = store.scan_own_relay(root)
    local.relay_state = state
    local.relay_pid = int(getattr(record, "pid", 0) or 0)

    for net in store.list_networks(root):
        self_member = net.self_member()
        local.networks.append(
            NetworkEntry(
                network_id=net.network_id,
                name=net.name,
                epoch=int(net.epoch or 0),
                role=str(getattr(self_member, "role", "") or ""),
                members=len(net.active_members()),
                trust=str(net.trust or ""),
            )
        )
    # One row per membership rather than per device, and NOT folded: a device in
    # two networks is two rows here because the member list said so, and this
    # capture is the place that must not collapse what the store did not
    # (mesh-ui.md §2.8.1 — the flat peer catalogue is the surface that does).
    from local_operator.network.peers import known_peers

    for peer in known_peers(root):
        local.peers.append(
            PeerEntry(
                device_id=peer.device_id,
                name=peer.name,
                role=peer.role,
                network_id=peer.network_id,
            )
        )
    return local


class NetworkCommandRequested(Message):
    """The panel asks for a typed command to be placed in the composer.

    A message rather than a direct call, because the composer belongs to the app
    and a screen that walked up to reach it would be the same coupling the
    Screen/handler split exists to avoid. The command is handed over UNSUBMITTED
    — see the module docstring for why the confirmation must stay in one place.
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command


class NetworkScreen(ModalScreen[None]):
    """``/network [ls|status]`` — the mesh table, pushed before its slow half."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=False),
        Binding("q", "dismiss_screen", "Back", show=False),
        Binding("r", "refresh_report", "Refresh", show=False),
        Binding(NETWORK_COPY_KEY, "copy_report", "Copy", show=False),
        Binding("enter", "open_selected", "Members", show=False),
        Binding("d", "request_disconnect", "Disconnect", show=False),
        Binding("shift+p", "request_panic", "Panic", show=False),
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, local: NetworkLocal | None = None, *, scroll_to: str = "") -> None:
        super().__init__()
        self.local = local if local is not None else capture_local()
        #: The worker's answer: the relay's own ``ls`` rows, ``peers`` rows and
        #: ``status`` payload. ``None`` means "not back yet", which the body
        #: paints as ``checking…`` rather than as an empty, confident table.
        self.relay: NetworkRun | None = None
        self.peers_run: NetworkRun | None = None
        self.status_run: NetworkRun | None = None
        #: The selected network's member table, filled by a second worker after
        #: ``enter``. Keyed by network id so a late answer for a network the user
        #: has moved off cannot be painted under the wrong header.
        self.detail: dict[str, Any] | None = None
        self.detail_for = ""
        self.selected = 0
        self.presentation_cancelled = False
        #: ``status`` is ``ls`` scrolled to the relay section (§1.1.1) — one
        #: screen, two entry points, rather than a second screen with a subset of
        #: the same rows.
        self._focus_relay = scroll_to == "status"

    # -- composition --------------------------------------------------------

    def compose(self) -> Any:
        with Container(classes="analytics-panel"):
            # Held on `self` because the worker's answer has to REPAINT them:
            # a `Static` built inline and never stored is a body that keeps the
            # first frame forever, which is the failure mode a two-phase screen
            # has instead of crashing.
            self._title = Static(self._title_text(), id="network-title")
            yield self._title
            with VerticalScroll(id="network-scroll") as scroll:
                self._scroll = scroll
                self._body = Static(self._report_text(), id="network-body")
                yield self._body
            self._hint = Static(self._hint_text(), id="network-hint")
            yield self._hint

    def on_mount(self) -> None:
        self._repaint()
        self.call_after_refresh(self._dismiss_if_cancelled)
        self.run_worker(self._fill(), thread=False, group="network", exit_on_error=False)

    def on_unmount(self) -> None:
        self.presentation_cancelled = True

    def on_resize(self, _event: Any) -> None:
        """Re-derive the width-sensitive halves, so a resize is not painted stale.

        The title rule and the footer both measure the card, and both are built
        once in ``compose`` before anything can be measured (UX round 1, U8 — the
        footer used to be FIXED at that moment, so it could never notice a narrow
        terminal at all).
        """
        self._repaint()

    def on_screen_resume(self) -> None:
        self._dismiss_if_cancelled()

    def invalidate(self) -> None:
        """Retire this presentation without popping a newer modal above it."""
        self.presentation_cancelled = True
        self._dismiss_if_cancelled()

    def _dismiss_if_cancelled(self) -> None:
        # Only when RESUMED: ``Screen.dismiss()`` pops the current screen, which
        # is not necessarily this instance — the same guard ``InfoScreen`` keeps.
        if self.presentation_cancelled and self.is_mounted and self.app.screen is self:
            self.dismiss(None)

    def set_relay(self, run: NetworkRun, peers: NetworkRun, status: NetworkRun) -> None:
        """Publish the worker's answer, while this presentation is still owned."""
        if self.presentation_cancelled:
            return
        self.relay, self.peers_run, self.status_run = run, peers, status
        self._repaint()

    def set_detail(self, network_id: str, payload: dict[str, Any] | None) -> None:
        if self.presentation_cancelled:
            return
        self.detail_for = network_id
        self.detail = payload
        self._repaint()

    # -- workers ------------------------------------------------------------

    async def _fill(self) -> None:
        """Ask the relay for its own view: three calls, each in its own thread.

        Sequential rather than gathered: the relay serialises its own control
        socket, and three concurrent listings would each pay the peer probe
        budget while queueing behind one another. The timeout is the CLI's own
        listing budget plus slack, stated in ``network_cli`` rather than invented
        here.
        """
        # Explicit keywords rather than a ``dict(**listing)`` splat: the splat
        # widened every value to one union and pyright caught the timeout being
        # passed as a bool. Three calls, written out, are also the only place the
        # panel states which verbs it asks for.
        run = await asyncio.to_thread(
            run_network, ["ls"], timeout=LISTING_TIMEOUT_S, json_output=True
        )
        peers = await asyncio.to_thread(
            run_network, ["peers"], timeout=LISTING_TIMEOUT_S, json_output=True
        )
        status = await asyncio.to_thread(
            run_network, ["status"], timeout=LISTING_TIMEOUT_S, json_output=True
        )
        self.set_relay(run, peers, status)

    async def _fill_detail(self, network_id: str) -> None:
        run = await asyncio.to_thread(
            run_network, ["show", network_id], timeout=LISTING_TIMEOUT_S, json_output=True
        )
        self.set_detail(network_id, run.payload())

    # -- actions ------------------------------------------------------------

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    def action_refresh_report(self) -> None:
        if self.presentation_cancelled or self not in self.app.screen_stack:
            return
        self.relay = self.peers_run = self.status_run = None
        self._repaint()
        self.run_worker(self._fill(), thread=False, group="network", exit_on_error=False)

    def action_copy_report(self) -> None:
        """Copy exactly what the panel PAINTS, built from the model not the pixels.

        Composed by :meth:`render_lines_for_test` — the same function the frame's
        text comes from — rather than scraped off the painted rows, following
        ``InfoScreen.action_copy_report``: two implementations of one idea drift,
        and the one that drifts here would decide what a user pastes into a
        support message. The write goes through the app's single clipboard path
        (OSC 52, with its receipt toast), so a copy survives ssh and a
        multiplexer and is never silent. ``/info`` and the aside bind the same
        chord (``NETWORK_COPY_KEY``), which is what makes it one gesture.

        The report is copied WITHOUT waiting for the relay: the first frame is
        this device's own records, and it is a true statement about the device —
        the same reason the screen is useful with the relay stopped.
        """
        payload = "\n".join(self.render_lines_for_test())
        put = getattr(self.app, "_put_on_clipboard", None)
        if put is None:
            self.app.copy_to_clipboard(payload)
            return
        put(payload)

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        entries = self._entries()
        if not entries:
            return
        # CLAMPED, not wrapped: this is a full-page mode whose list is the page
        # (the documented exception in AGENTS.md "TUI conventions"), and a list
        # whose bottom is a destination should not throw the user off the top.
        self.selected = max(0, min(len(entries) - 1, self.selected + delta))
        self._repaint()

    def action_open_selected(self) -> None:
        entries = self._entries()
        if not entries:
            return
        network_id = entries[self.selected].network_id
        if self.detail_for == network_id and self.detail is not None:
            self.detail = None
            self.detail_for = ""
            self._repaint()
            return
        self.detail = None
        self.detail_for = network_id
        self._repaint()
        self.run_worker(
            self._fill_detail(network_id), thread=False, group="network-detail", exit_on_error=False
        )

    def action_request_disconnect(self) -> None:
        entries = self._entries()
        if not entries:
            return
        # The command is handed over UNSUBMITTED and WITHOUT its confirmation
        # word: the composer's line is the rehearsal, and the user's own Enter is
        # what runs it. `/network disconnect <net> yes` typed here would be the
        # one-keystroke incident control §1.5 refuses.
        self.post_message(
            NetworkCommandRequested(f"/network disconnect {entries[self.selected].network_id}")
        )

    def action_request_panic(self) -> None:
        entries = self._entries()
        if not entries:
            return
        self.post_message(
            NetworkCommandRequested(f"/network panic {entries[self.selected].network_id}")
        )

    def action_scroll_up(self) -> None:
        self._scroll_page(-1)

    def action_scroll_down(self) -> None:
        self._scroll_page(1)

    def action_page_up(self) -> None:
        self._scroll_page(-1)

    def action_page_down(self) -> None:
        self._scroll_page(1)

    def action_scroll_home(self) -> None:
        scroll = getattr(self, "_scroll", None)
        if scroll is not None:
            scroll.scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        scroll = getattr(self, "_scroll", None)
        if scroll is not None:
            scroll.scroll_end(animate=False)

    def _scroll_page(self, delta: int) -> None:
        scroll = getattr(self, "_scroll", None)
        if scroll is None:
            return
        step = max(1, scroll.size.height - 1)
        scroll.scroll_to(y=scroll.scroll_offset.y + delta * step, animate=False)

    # -- rendering ----------------------------------------------------------

    def _card_width(self) -> int:
        """The content cells a row may occupy, MEASURED off the mounted scroll.

        Measured rather than recomputed, for the reason ``InfoScreen`` records:
        a formula and a stylesheet drift, and a row built one cell too wide folds
        a value onto a second line — the "one record reads as two" fault these
        screens exist to remove.
        """
        scroll = getattr(self, "_scroll", None)
        if scroll is not None and scroll.is_mounted and scroll.size.width:
            return max(_MIN_CARD_WIDTH, scroll.size.width - 1)
        try:
            return max(_MIN_CARD_WIDTH, min(140, int(self.app.size.width * 0.9)) - 7)
        except Exception:  # noqa: BLE001 — before mount there is no app size
            return _MIN_CARD_WIDTH

    def _title_text(self) -> Text:
        return Text(
            "Mesh networks\n" + "─" * max(1, self._card_width()),
            no_wrap=True,
            overflow="crop",
        )

    #: The footer's keys, in order, as SEGMENTS rather than one string: every
    #: prefix of this tuple is a valid shorter footer, which is what lets a narrow
    #: terminal keep the leftmost keys and say that it dropped the rest (UX round
    #: 1, U8). One string could only be cut, and a cut line does not say it was
    #: cut.
    HINT_SEGMENTS = (
        "esc",
        "↑↓",
        "enter members",
        "d disconnect",
        "shift+p panic",
        "r refresh",
        "ctrl+r copy",
    )

    def _hint_text(self, width: int = 0) -> str:
        """The footer: the keys, in ONE casing, and the row-sensitive half named.

        ``shift+P`` used to be the only capitalised key on a line whose other four
        were lowercase (design round 1, D9). All five are lowercase now, which is
        how the rest of the app's footers spell them.

        ``enter members`` STAYS, and the reviewer's alternative reading is worth
        answering rather than adopting: §1.1.4 has ``enter`` on a PEER open
        ``/new remote <peer>``, but the panel's cursor walks NETWORK rows only —
        the peer rows are painted fields, not selectable rows — so there is no
        peer row for the hint to be about. Making peers selectable is the same
        producer gap MINOR 3 / §1.2 record, and the hint will follow the cursor
        the day that lands rather than describing a binding that does not exist.

        THE LINE HAS A BUDGET, which is why two of the labels are terse. The body
        is 83 cells at 100x30 (the panel is 90% of the terminal, minus its own
        padding) and the previous hint measured 78 of them; adding the copy chord
        is 14 more, so ``esc close``/``↑↓ move`` gave up ten cells of verb to keep
        the whole line on screen rather than ending in an ellipsis. Nothing is
        dropped at a width that fits: every binding the screen carries is named.

        AT A WIDTH THAT DOES NOT FIT, THE LINE SAYS SO (UX round 1, U8). Measured
        at 64x30 the frame ended after ``shift+p`` with no ellipsis and no
        fallback, so the only hints for refresh, copy and panic were gone exactly
        where a cramped terminal wants them — and nothing on the frame said more
        existed. ``width`` is the cells available (``_card_width``, the same
        measurement the title uses); zero means "not measured yet" and returns the
        whole line, which is what ``compose`` renders before the first layout.
        """
        full = " · ".join(self.HINT_SEGMENTS)
        if not width or len(full) <= width:
            return full
        kept: list[str] = []
        for segment in self.HINT_SEGMENTS:
            candidate = " · ".join([*kept, segment])
            # +2 for the marker itself: a line that names six keys and says
            # nothing about the seventh is the defect, not the truncation.
            if len(candidate) + 2 > width:
                break
            kept.append(segment)
        return " · ".join(kept) + " …"

    def _entries(self) -> list[NetworkEntry]:
        """The network rows on screen: the relay's when it has answered, else ours.

        ONE list, so the cursor cannot point at a row the relay replaced. The
        relay's rows carry ``verified=True``, and that flag gates the SECTION NOTE
        in :meth:`_networks_section` — it is read off the section, not off a row's
        trailing column, because there is no such column in the frame (review
        round 4, NIT 5: the docstring claimed one).
        """
        payload = self.relay.payload() if self.relay is not None else None
        if payload is not None:
            rows = payload.get("networks")
            if isinstance(rows, list) and rows:
                entries: list[NetworkEntry] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    entries.append(
                        NetworkEntry(
                            network_id=str(row.get("network_id") or ""),
                            name=str(row.get("name") or ""),
                            epoch=int(row.get("epoch") or 0),
                            role=str(row.get("role") or ""),
                            members=int(row.get("members") or 0),
                            trust=str(row.get("trust") or ""),
                            stale=bool(row.get("stale")),
                            verified=True,
                        )
                    )
                return entries
        return list(self.local.networks)

    def _report_text(self, width: int | None = None) -> Text:
        width = self._card_width() if width is None else max(_MIN_CARD_WIDTH, width)
        body = Text()
        if self._focus_relay:
            self._relay_section(body, width)
            body.append("\n")
        self._device_section(body, width)
        body.append("\n")
        self._networks_section(body, width)
        if self.detail_for:
            body.append("\n")
            self._detail_section(body, width)
        body.append("\n")
        self._peers_section(body, width)
        if not self._focus_relay:
            body.append("\n")
            self._relay_section(body, width)
        # NO TRAILING BLANK INSIDE THE SCROLL REGION (design round 1, D2). Every
        # section ends its last row with a newline, so the body's line count was
        # one MORE than the lines it painted: on a body that already fitted, the
        # scroll region's ``virtual_size`` came out one above its ``size`` and the
        # panel drew a scrollbar with nothing to scroll — and only in the LOADED
        # frame, which is why the two phases of one screen disagreed about whether
        # there was more to see. Stripping the trailing newline makes the region's
        # extent the content's extent, which is what makes "no bar" mean "nothing
        # more".
        #
        # ``Text.rstrip`` MUTATES IN PLACE and returns ``None`` in this Rich
        # version: calling it as an expression handed ``None`` to
        # ``Static.update`` and the screen raised instead of painting (caught by
        # the first capture after the change, which is why the frame is the proof
        # and not the unit test).
        body.rstrip()
        return body

    def _header(self, body: Text, title: str, note: str = "") -> None:
        body.append(title, style="bold")
        if note:
            body.append("  ")
            body.append(note, style="dim")
        body.append("\n")

    def _device_section(self, body: Text, width: int) -> None:
        self._header(body, "This device")
        if not self.local.identity_present:
            body.append("  no identity yet — /network new <name> creates one\n", style="yellow")
        else:
            # LABEL FIRST, then the abbreviated id (design round 1, D6). The line
            # used to lead with the full 34-cell id, putting the longest token on
            # the screen in front of the one thing a person recognises; §1.1.4
            # already asked for the short form here. The full id is what
            # ``/network show`` and the ``#`` peer token take, so this shortens a
            # COLUMN rather than the data.
            body.append("  ")
            body.append(self.local.device_name or UNNAMED_DEVICE)
            if self.local.device_id:
                body.append(f"  {short_id(self.local.device_id)}", style="dim")
            body.append("\n")
        # THE RELAY'S STANDING HAS ONE VOICE, and once the worker has answered it
        # is the Relay section's: that half read the live process, this half read
        # a record on disk, and printing both put "relay: not running" five lines
        # above "installed: yes \u00b7 pid 4711" in the same frame — two answers to
        # one question, the older one wrong. Before the answer lands, the record is
        # the only thing known and it says so.
        if self.status_run is not None:
            return
        if not self.local.relay_state:
            # NO BACKTICKS, AND NO DEFERRED ANSWER (design round 1, D3). This line
            # used to carry a shell command between literal backticks — the
            # backticks painted onto a terminal, confirmed at 250% zoom — and it
            # spent its words pushing the one thing the reader wants, why the mesh
            # is down, into a command they have to leave the panel to run.
            # "No record" is a fact this device holds, and it IS the answer: there
            # is no relay process here. The CLI keeps its place as a next step in
            # the footer, beside the other next steps, rather than as the answer.
            body.append("  relay: not running on this device\n", style="dim")
        else:
            verdict = (
                "running"
                if self.local.relay_state == "live"
                else f"running but not answering ({self.local.relay_state})"
            )
            body.append(f"  relay: {verdict}  pid {self.local.relay_pid}\n")

    @staticmethod
    def _short(network_id: str) -> str:
        """A network id in the panel's one abbreviation (:func:`short_id`).

        Kept as a thin alias rather than deleted because the SECTION's call sites
        read better naming what they hold; the width and the direction now come
        from the module-level helper, which is what makes the device, network and
        peer columns comparable (design round 1, D6).
        """
        return short_id(network_id)

    def _row(self, body: Text, fields: Sequence[str], width: int, *, lead: str = "  ") -> None:
        """One row, fitted: trailing FIELDS are dropped rather than wrapped.

        A wrapped row is the failure this panel's whole measurement exists to
        avoid (``mesh-ui.md`` §1.1.4): two lines for one network reads as two
        records. The fields are ordered by what a reader needs first — name, id,
        epoch, role, members, trust — so dropping from the right loses the least.
        A row is never cut in the MIDDLE of a field: half a word is worse than a
        missing one, because it looks like data.
        """
        kept: list[str] = []
        for column in fields:
            candidate = " ".join([*kept, column])
            if len(lead) + len(candidate) > width and kept:
                break
            kept.append(column)
        body.append(lead)
        body.append(" ".join(kept) or fields[0])
        body.append("\n")

    def _networks_section(self, body: Text, width: int) -> None:
        entries = self._entries()
        if self.relay is None:
            note = "checking with the relay…"
        elif entries and not entries[0].verified:
            note = "from this device's records"
        else:
            note = ""
        self._header(body, "Networks", note)
        if not entries:
            body.append(
                "  no networks on this device — /network new <name> creates one\n", style="dim"
            )
            return
        for index, entry in enumerate(entries):
            cursor = "›" if index == self.selected else " "
            fields = [
                f"{entry.label[:24]}",
                self._short(entry.network_id),
            ]
            if entry.role:
                fields.append(entry.role)
            # ``4 members`` / ``1 member``, not ``member(s)`` (design round 1,
            # D7): the plural is knowable from the number that is already in the
            # field, so writing both spellings is copy nobody writes. The EPOCH
            # left this row and went to the detail heading — it is protocol
            # vocabulary with no legend here, and the detail view is where a
            # reader who wants it can see it beside the members it counts.
            fields.append(f"{entry.members} member" + ("" if entry.members == 1 else "s"))
            if entry.trust and entry.trust != "active":
                fields.append(f"trust {entry.trust}")
            if entry.stale:
                fields.append("[stale]")
            self._row(body, fields, width, lead=f"{cursor} ")

    def _detail_section(self, body: Text, width: int) -> None:
        label = self._detail_label()
        # THE EPOCH IS READ HERE (design round 1, D7). It is protocol vocabulary —
        # the membership generation — and this is the view a reader opens when
        # they want the detail rather than a row: it sits beside the member count
        # it counts, instead of pushing a word nobody can look up into a summary
        # line beside the name.
        epoch = next((e.epoch for e in self._entries() if e.network_id == self.detail_for), None)
        suffix = "" if epoch is None else f" · epoch {epoch}"
        self._header(body, f"{label} · members{suffix}")
        if self.detail is None:
            body.append("  asking the relay…\n", style="dim")
            return
        members = self.detail.get("members_detail")
        if not isinstance(members, list) or not members:
            body.append("  no members reported\n", style="dim")
            return
        for member in members:
            if not isinstance(member, dict):
                continue
            mark = "active" if member.get("active") else "REMOVED"
            body.append(f"  {mark:7} ")
            body.append(f"{str(member.get('device_id') or '')[:12]:12}")
            body.append(f" {str(member.get('role') or ''):5}")
            caps = member.get("capabilities") or []
            body.append(f" {', '.join(str(cap) for cap in caps)}")
            if member.get("suspect"):
                body.append("  [suspect: key may be copied]", style="yellow")
            body.append("\n")

    def _detail_label(self) -> str:
        for entry in self._entries():
            if entry.network_id == self.detail_for:
                return entry.label
        return self.detail_for

    def _peers_section(self, body: Text, width: int) -> None:
        payload = self.peers_run.payload() if self.peers_run is not None else None
        rows = payload.get("peers") if payload is not None else None
        relay_rows = isinstance(rows, list) and bool(rows)
        # THE NOTE IS KEYED ON A USABLE PEERS PAYLOAD, NOT ON THE ``ls`` RUN
        # (review round 4, MINOR 1). It read ``as the relay found them`` whenever
        # any relay call had finished, so in the state the module docstring says
        # people open the panel in — the relay down — the section fell back to the
        # device's OWN member lists and still printed them as the relay's answer.
        # ``_networks_section`` already labels its fallback correctly; this is the
        # same rule, and ``found by the relay`` is a state rather than the
        # dependent clause it was (design round 1, D11).
        if self.peers_run is None:
            note = "checking with the relay…"
        elif relay_rows:
            note = "found by the relay"
        else:
            note = "from this device's records"
        self._header(body, "Peers", note)
        # The `isinstance` here REPEATS the one folded into `relay_rows` on
        # purpose: `relay_rows` is a bool, so a checker cannot carry the
        # narrowing through it, and iterating a `None` is what this loop would
        # otherwise look like. The conjunction is the same condition — a false
        # `relay_rows` only ever short-circuits it — so the printed rows and the
        # note still come from one verdict.
        if relay_rows and isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                reachable = bool(row.get("reachable"))
                fields = [
                    short_id(str(row.get("device_id") or "")),
                    # One fallback string across the surfaces (design round 1,
                    # D8): a peer with no label was a dangling id here while the
                    # sidebar called the same device ``another device``.
                    str(row.get("name") or UNNAMED_DEVICE),
                ]
                if not reachable:
                    # THE WORDS, NOT THE TOKEN (design round 1, D3).
                    fields.append(peer_reason_words(str(row.get("reason") or "")))
                self._row(
                    body,
                    [f"{'reachable' if reachable else 'unreachable':11}", *fields],
                    width,
                )
            return
        if self.peers_run is None:
            # NOTHING IN THE BODY: the section-subtitle slot already says it
            # (``checking with the relay…``), and painting the same sentence
            # twice — once on the heading, once as a row — was this section's
            # own copy of the redundancy the two-phase design avoids elsewhere.
            # The other two sections use the subtitle for exactly this state.
            return
        if not self.local.peers:
            body.append(
                "  no peers yet — /network invite mints a token\n",
                style="dim",
            )
            return
        for peer in self.local.peers:
            self._row(
                body,
                [short_id(peer.device_id), peer.label, peer.role],
                width,
            )

    def _relay_section(self, body: Text, width: int) -> None:
        self._header(body, "Relay")
        if self.status_run is None:
            body.append("  asking the relay…\n", style="dim")
            return
        payload = self.status_run.payload()
        if payload is None:
            # A refused ``status`` prints its sentence to stderr; showing the
            # CLI's own words is the one honest rendering of a failure whose
            # reason the CLI owns.
            for line in self.status_run.lines:
                body.append(f"  {line}\n", style="yellow")
            return
        body.append(f"  installed:  {'yes' if payload.get('installed') else 'no'}\n")
        identity = "present" if payload.get("identity_present") else "missing"
        body.append(f"  identity:   {identity}\n")
        relay = payload.get("relay") or {}
        if payload.get("relay_running"):
            state = "answering" if payload.get("relay_answering") else "NOT answering"
            body.append(f"  relay:      running, pid {relay.get('pid')} — {state}\n")
        else:
            body.append("  relay:      not running\n")
        log = str(payload.get("log") or "")
        if log:
            body.append(f"  log:        {log}\n", style="dim")

    def render_lines_for_test(self) -> list[str]:
        """The screen as plain strings — what a user actually reads.

        The TITLE is included, unlike ``InfoScreen``'s: this panel's title is where
        the surface names itself ("Mesh networks"), so a reader asserting what the
        screen says would otherwise have to know that half of it lives in a
        different widget.
        """
        return (self._title_text().plain + "\n" + self._report_text().plain).split("\n")

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is not None and body.is_mounted:
            body.update(self._report_text())
        title = getattr(self, "_title", None)
        if title is not None and title.is_mounted:
            title.update(self._title_text())
        # THE FOOTER IS WIDTH-SENSITIVE, so it is re-derived here rather than only
        # built once in `compose`: `compose` runs before the first layout, when
        # there is nothing to measure, and a hint fixed at that moment cannot
        # notice a resize or a narrow screen (UX round 1, U8).
        hint = getattr(self, "_hint", None)
        if hint is not None and hint.is_mounted:
            hint.update(self._hint_text(self._card_width()))


def build_network_report(local: NetworkLocal, width: int = _MIN_CARD_WIDTH) -> Text:
    """The first frame's text, for a test or a capture that has no app.

    Kept as a module function so a unit test can assert the table without
    mounting a screen, exactly as ``build_info_report`` is for ``/info``.
    """
    return NetworkScreen(local)._report_text(width)
