"""The full-page ``/community`` view and the ``/orgs`` switcher (TUI frames T1–T10).

WHAT IT IS. A sibling MODE of ``/notifications`` and ``/library``: it takes the
transcript's region, the session composer goes inert, and the page owns the
keyboard until ``esc`` closes it. Closing the page is never leaving the
community — leaving lives only in Settings → Leave, which points at the org
switcher first (frame T9).

THE PAGE'S OWN COMPOSER. It always says who will see the message, and it is a
separate widget from the session composer, so community text can't reach the
agent by accident and an agent prompt can't be posted by accident. A leading
``/`` never posts (``//`` posts a literal slash). Drafts stay in memory only.

STATE vs PAINT. :mod:`local_operator.tui.community_model` decides every
composer outcome and the org default; :mod:`local_operator.tui.community_backend`
does every network call. This widget routes keys and paints.

OVERLAYS (one at a time, drawn in place of the rail and pane): ``join``
(guidelines + explicit agreement, T2), ``browse`` (``ctrl+k``, T3), ``report``
(``!``, T8), ``why`` (``f2`` on a refusal, T5), ``settings`` (``s``, T9),
``orgs`` (``g`` / ``/orgs``, the Amendment A switcher) and ``keys`` (``?``).

CONTRACT GAPS, SHOWN HONESTLY. Reports, people search, profile fields and
agent settings are ``x-status: planned`` on the hub, so those screens say the
feature isn't available on this server yet instead of pretending to send.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static

from local_operator.community import service
from local_operator.i18n import keys as i18n_keys
from local_operator.tui import community_model as model
from local_operator.tui.i18n_status import display_tr

_t = display_tr("tui.community")
_k = i18n_keys.tui_community

#: Below this the rail becomes a drawer (frame T1 "60x20"): ``tab`` opens it.
DRAWER_COLUMNS = 72
#: The rail's width at 80 columns (frame T1); 28 from 150 columns up.
RAIL_WIDTH = 22
RAIL_WIDTH_WIDE = 28
WIDE_COLUMNS = 150

_BROWSE_FILTERS = ("all", "joined", "not_joined", "private")


class CommunityViewDismissed(Message):
    """``esc`` at the page's top level: leave the mode."""


class CommunityOrgSelected(Message):
    """The switcher picked a private org (the app records it; nothing is left)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


def _hhmm(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone().strftime("%H:%M")
    except (TypeError, ValueError):
        return ""


def _now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _wait(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _category(category: str | None) -> str:
    labels = {
        "unsafe_or_illegal": _k.category_unsafe_or_illegal,
        "harassment_or_abuse": _k.category_harassment_or_abuse,
        "spam_or_promotion": _k.category_spam_or_promotion,
        "private_information": _k.category_private_information,
    }
    maker = labels.get(category or "")
    return _t(maker()) if maker else (category or "")


class CommunityView(Vertical):
    """The community page. ``backend`` is the I/O seam (see module docstring)."""

    can_focus = True

    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("tab", "cycle_focus", "Next pane", show=False),
        Binding("ctrl+k", "browse", "Browse", show=False),
        Binding("f2", "details", "Details", show=False),
        Binding("up,k", "move(-1)", "Up", show=False),
        Binding("down,j", "move(1)", "Down", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("enter", "activate", "Open", show=False),
        Binding("space", "toggle_check", "Toggle", show=False),
        Binding("ctrl+f", "cycle_filter", "Filter", show=False),
    ]

    def __init__(self, backend: Any, *, mode: str = "page", taint_note: bool = False) -> None:
        super().__init__(classes="settings-view community-view")
        self.backend = backend
        self.home: Any = None
        self.home_failure: Any = None
        self.home_loading = True
        #: ``rail`` | ``messages`` | ``composer``: where keys go (T1 ``tab`` cycle).
        self.area = "rail"
        self.channels: list[Any] = []
        self.channel_index = 0
        self.messages: dict[str, list[Any] | None] = {}
        self.failures: dict[str, Any] = {}
        self.message_index = -1
        self.composers: dict[str, model.Composer] = {}
        self.overlay: str | None = None if mode == "page" else mode
        #: The overlay was the reason the page opened (``/orgs``): esc closes all.
        self._overlay_is_page = mode != "page"
        self.drawer_open = False
        self.notice = ""
        # overlay state
        self.join_agree = False
        self.join_focus = 0  # 0 checkbox, 1 join, 2 cancel
        self.join_busy = False
        self.join_error = ""
        self.browse_filter = "all"
        self.browse_rows: list[Any] = []
        self.browse_state = "idle"
        self.browse_index = 0
        self.report_index = 1
        self.report_target: Any = None
        self.settings_tab = 1
        self.leave_focus = 0
        self.orgs_private: list[model.OrgEntry] = []
        self.orgs_communities: list[model.OrgEntry] = []
        self.orgs_index = 0
        self.orgs_state = "loading"
        self.taint_note = taint_note
        self._attempts: dict[str, str] = {}

        self._header = Static(classes="community-view-header")
        self._rail = Static(classes="community-view-rail")
        self._pane_text = Static(classes="community-view-text")
        self._pane = VerticalScroll(self._pane_text, classes="community-view-pane")
        self._audience = Static(classes="community-view-audience")
        self._input = Input(classes="community-view-input", id="community-composer")
        self._status = Static(classes="community-view-status")
        self._column = Vertical(
            self._pane,
            self._audience,
            self._input,
            self._status,
            classes="community-view-column",
        )
        self._body = Horizontal(self._rail, self._column, classes="community-view-body")
        self._overlay_title = Static(classes="community-view-overlay-title")
        self._overlay_text = Static(classes="community-view-overlay-text")
        self._overlay_input = Input(classes="community-view-overlay-input")
        self._overlay_scroll = VerticalScroll(
            self._overlay_text, classes="community-view-overlay-body"
        )
        self._overlay_controls = Static(classes="community-view-overlay-controls")
        self._overlay = Vertical(
            self._overlay_title,
            self._overlay_input,
            self._overlay_scroll,
            self._overlay_controls,
            classes="community-view-overlay",
        )
        self._footer = Static(classes="community-view-footer")

    # -- composition ---------------------------------------------------------

    def compose(self):  # type: ignore[override]
        yield self._header
        yield self._body
        yield self._overlay
        yield self._footer

    def on_mount(self) -> None:
        self._overlay.display = False
        self._apply_layout()
        self.paint()
        self.focus()
        self.set_interval(1.0, self._tick)
        self.run_worker(self._load_home(), exclusive=False, group="community-home")
        if self.overlay == "orgs":
            self._open_orgs()

    def on_resize(self) -> None:
        self._apply_layout()
        self.paint()

    @property
    def narrow(self) -> bool:
        return self.size.width and self.size.width < DRAWER_COLUMNS  # type: ignore[return-value]

    def _apply_layout(self) -> None:
        width = RAIL_WIDTH_WIDE if self.screen.size.width >= WIDE_COLUMNS else RAIL_WIDTH
        self._rail.styles.width = width
        # No rail before there is a community to list (loading, error, not
        # joined): an empty CHANNELS heading reads as "you have no channels".
        member = self.home is not None and self.home.joined
        self._rail.display = member and ((not self.narrow) or self.drawer_open)

    # -- data ----------------------------------------------------------------

    async def _load_home(self) -> None:
        self.home_loading = True
        self.home_failure = None
        self.paint()
        try:
            # Every backend read enters through ``service.display_safe``: the
            # page paints server text in dozens of places (R1-3).
            self.home = service.display_safe(await self.backend.home())
        except Exception as exc:  # noqa: BLE001 - painted as the page error state
            self.home_failure = exc
            self.home = None
        self.home_loading = False
        self._apply_layout()
        if self.home is not None:
            self.channels = model.rail_order(self.home.channels)
            if not self.home.joined and self.overlay is None:
                self.overlay = None  # the not-joined page offers ``j``
            if self.channels:
                # Open on the first channel a member can post in: landing on
                # read-only #welcome-and-rules would greet everyone with a
                # composer that refuses them.
                self.channel_index = next(
                    (i for i, c in enumerate(self.channels) if not model.is_read_only(c)), 0
                )
                self._open_channel()
        self.paint()

    @property
    def community_id(self) -> str:
        return self.home.community.id if self.home is not None else ""

    @property
    def channel(self) -> Any:
        if 0 <= self.channel_index < len(self.channels):
            return self.channels[self.channel_index]
        return None

    def composer(self) -> model.Composer | None:
        channel = self.channel
        if channel is None:
            return None
        current = self.composers.get(channel.id)
        if current is None:
            current = model.Composer()
            self.composers[channel.id] = current
        return model.composer_for_channel(current, channel)

    def _open_channel(self) -> None:
        channel = self.channel
        if channel is None:
            return
        self.message_index = -1
        composer = self.composer()
        if composer is not None:
            self._input.value = composer.draft
        if channel.id not in self.messages or self.failures.get(channel.id) is not None:
            self.messages[channel.id] = None
            self.failures.pop(channel.id, None)
            self.run_worker(self._load_history(channel), group="community-history")
        self.paint()

    async def _load_history(self, channel: Any) -> None:
        try:
            rows = service.display_safe(await self.backend.history(self.community_id, channel.id))
        except Exception as exc:  # noqa: BLE001 - the pane's error state
            self.failures[channel.id] = exc
            self.messages[channel.id] = self.messages.get(channel.id) or []
        else:
            self.failures.pop(channel.id, None)
            self.messages[channel.id] = list(rows)
        self.paint()
        if channel is self.channel:
            # Newest at the bottom, like any chat: the pane opens at its end
            # (T1 "end jumps"); after the repaint so the new height counts.
            self.call_after_refresh(self._pane.scroll_end, animate=False)

    # -- composer --------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is self._input:
            composer = self.composer()
            if composer is not None:
                composer.draft = event.value
        elif event.input is self._overlay_input and self.overlay == "browse":
            self.run_worker(self._browse(), exclusive=True, group="community-browse")
        elif event.input is self._overlay_input:
            self.paint()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if event.input is self._overlay_input:
            self.action_activate()
            return
        self._send()

    def _send(self) -> None:
        composer = self.composer()
        channel = self.channel
        if composer is None or channel is None or self.home is None:
            return
        text = self._input.value
        if composer.status in ("moderation_unavailable", "offline", "error", "guidelines"):
            composer.status = "idle"
        postable = composer.postable_text(text)
        if postable is None:
            if text.strip().startswith("/"):
                self.notice = _t(_k.status_local_command())
                self.paint()
            return
        if not composer.can_send():
            self.paint()
            return
        from local_operator.community.client import new_idempotency_key

        key = new_idempotency_key()
        composer.begin(postable, key)
        self._input.value = ""
        self._attempts[channel.id] = key
        self.paint()
        self.run_worker(
            self._submit(channel, postable, key), group=f"community-submit-{channel.id}"
        )

    async def _submit(self, channel: Any, text: str, key: str) -> None:
        version = self.home.accepted_version if self.home is not None else None
        outcome = service.display_safe(
            await self.backend.submit(self.community_id, channel.id, text, key, version)
        )
        composer = self.composers.get(channel.id)
        if composer is None or composer.in_flight_key != key:
            # The attempt was cancelled first; a late "published" is the
            # T5 "cancel was too late" case, anything else is already settled.
            if composer is not None and outcome.kind == "posted" and composer.status == "cancelled":
                composer.settle(service.PostOutcome("too_late", submission_id=key))
                composer.draft = ""
                self._input.value = "" if channel is self.channel else self._input.value
            self.paint()
            return
        composer.settle(outcome)
        composer.at = _now_hhmm()
        if outcome.kind == "posted":
            self.run_worker(self._load_history(channel), group="community-history")
        if channel is self.channel:
            self._input.value = composer.draft
        self.paint()

    async def _cancel(self, channel: Any, composer: model.Composer) -> None:
        key = composer.in_flight_key
        if key is None:
            return
        outcome = service.display_safe(await self.backend.cancel(self.community_id, key))
        if composer.in_flight_key != key:
            self.paint()
            return
        composer.settle(outcome)
        composer.at = _now_hhmm()
        if channel is self.channel:
            self._input.value = composer.draft
        self.paint()

    def _tick(self) -> None:
        composer = self.composer()
        if composer is not None and composer.status == "rate_limited":
            composer.tick()
            self._paint_status()

    # -- actions -------------------------------------------------------------

    def action_back(self) -> None:
        if self.overlay is not None:
            if self._overlay_is_page:
                self.post_message(CommunityViewDismissed())
                return
            self._close_overlay()
            return
        composer = self.composer()
        if self.area == "composer" and composer is not None and composer.status == "checking":
            # T5: the first esc cancels the checking post, never leaves.
            self.run_worker(self._cancel(self.channel, composer), group="community-cancel")
            return
        if self.area == "composer":
            self._set_area("messages")
            return
        if self.area == "messages":
            self._set_area("rail")
            return
        self.post_message(CommunityViewDismissed())

    def action_cycle_focus(self) -> None:
        if self.overlay == "join":
            self.join_focus = (self.join_focus + 1) % 3
            self.paint()
            return
        if self.overlay == "settings":
            self.settings_tab = (self.settings_tab + 1) % 3
            self.paint()
            return
        if self.overlay is not None:
            return
        order = ["rail", "messages", "composer"]
        nxt = order[(order.index(self.area) + 1) % 3]
        if self.narrow:
            self.drawer_open = nxt == "rail"
            self._apply_layout()
        self._set_area(nxt)

    def _set_area(self, area: str) -> None:
        self.area = area
        if area == "composer" and self.channel is not None:
            self._input.focus()
        else:
            self.focus()
        self.paint()

    def action_move(self, delta: int) -> None:
        if self.overlay == "browse":
            self.browse_index = max(0, min(len(self.browse_rows) - 1, self.browse_index + delta))
        elif self.overlay == "report":
            self.report_index = max(0, min(3, self.report_index + delta))
        elif self.overlay == "orgs":
            total = len(self.orgs_private) + len(self.orgs_communities)
            self.orgs_index = max(0, min(total - 1, self.orgs_index + delta))
        elif self.overlay == "settings" and self.settings_tab == 2:
            self.leave_focus = max(0, min(1, self.leave_focus + delta))
        elif self.overlay == "join":
            self.join_focus = max(0, min(2, self.join_focus + delta))
        elif self.overlay is None and self.area == "rail" and self.channels:
            self.channel_index = max(0, min(len(self.channels) - 1, self.channel_index + delta))
            self._open_channel()
        elif self.overlay is None and self.area == "messages":
            rows = self.messages.get(getattr(self.channel, "id", ""), None) or []
            if rows:
                start = len(rows) if self.message_index < 0 else self.message_index
                self.message_index = max(0, min(len(rows) - 1, start + delta))
        self.paint()

    def action_page(self, delta: int) -> None:
        if self.overlay is not None:
            self._overlay_scroll.scroll_relative(
                y=delta * max(1, self._overlay_scroll.size.height - 1), animate=False
            )
        elif self.overlay is None:
            self._pane.scroll_relative(y=delta * max(1, self._pane.size.height - 1))

    def action_toggle_check(self) -> None:
        if self.overlay == "join" and self.join_focus == 0:
            self.join_agree = not self.join_agree
            self.paint()

    def action_activate(self) -> None:
        if self.overlay == "join":
            self._join_activate()
        elif self.overlay == "browse":
            self._browse_open()
        elif self.overlay == "orgs":
            self._orgs_pick()
        elif self.overlay == "report":
            # Nothing can be sent until the hub serves reports (createReport is
            # planned); the receipt says so instead of implying a send.
            self._close_overlay()
            composer = self.composer()
            if composer is not None:
                composer.status = "report_sent"
        elif self.overlay == "settings" and self.settings_tab == 2:
            if self.leave_focus == 1:
                self.run_worker(self._leave(), group="community-leave")
            else:
                self._close_overlay()
        elif self.overlay in ("why", "keys"):
            self._close_overlay()
        elif self.overlay is None and self.area == "rail":
            self._set_area("messages")
        self.paint()

    def action_details(self) -> None:
        composer = self.composer()
        if self.overlay is not None:
            return
        if composer is not None and composer.status == "refused":
            self._open_overlay("why")
        elif composer is not None and composer.restore_set_aside():
            self._input.value = composer.draft
        else:
            self._open_overlay("keys" if self.area != "composer" else "info")
        self.paint()

    def action_browse(self) -> None:
        if self.home is None or not self.home.joined:
            return
        self._open_overlay("browse")
        self._overlay_input.value = ""
        self._overlay_input.focus()
        self.run_worker(self._browse(), exclusive=True, group="community-browse")

    def action_cycle_filter(self) -> None:
        if self.overlay != "browse":
            return
        index = _BROWSE_FILTERS.index(self.browse_filter)
        self.browse_filter = _BROWSE_FILTERS[(index + 1) % len(_BROWSE_FILTERS)]
        self.run_worker(self._browse(), exclusive=True, group="community-browse")

    def on_key(self, event: Any) -> None:
        """Letter shortcuts — only when nothing is accepting typed text (T1)."""

        if self.screen.focused in (self._input, self._overlay_input):
            return
        key = getattr(event, "character", None) or ""
        handled = True
        if key == "g":
            self._open_orgs()
        elif key == "s" and self.overlay is None and self.home is not None and self.home.joined:
            self.settings_tab = 1
            self._open_overlay("settings")
        elif key == "!" and self.overlay is None:
            self._open_report()
        elif key == "?" and self.overlay is None:
            self._open_overlay("keys")
        elif key == "j" and self.overlay is None and self.home is not None and not self.home.joined:
            self._open_overlay("join")
        elif key == "r" and self.overlay is None:
            if self.home is None:
                self.run_worker(self._load_home(), group="community-home")
            else:
                self._retry_channel()
        else:
            handled = False
        if handled:
            event.stop()
            event.prevent_default()
            self.paint()

    def _retry_channel(self) -> None:
        channel = self.channel
        if channel is not None:
            self.failures[channel.id] = True  # forces a reload in _open_channel
            self._open_channel()

    # -- overlays --------------------------------------------------------------

    def _open_overlay(self, name: str) -> None:
        self.overlay = name
        self._body.display = False
        self._overlay.display = True
        self._overlay_input.display = name in ("browse", "report")
        if name == "join":
            self.join_agree = False
            self.join_focus = 0
            self.join_error = ""
        if name == "report":
            self._overlay_input.value = ""
            self._overlay_input.placeholder = _t(_k.report_note(note=""))
            self._overlay_input.focus()
        elif name == "browse":
            self._overlay_input.placeholder = _t(_k.browse_search(query=""))
        elif name != "browse":
            self.focus()
        self.paint()

    def _close_overlay(self) -> None:
        self.overlay = None
        self._overlay.display = False
        self._body.display = True
        self._set_area(self.area)

    def _open_report(self) -> None:
        rows = self.messages.get(getattr(self.channel, "id", ""), None) or []
        self.report_target = (
            rows[self.message_index] if 0 <= self.message_index < len(rows) else None
        )
        self.report_index = 1
        self._open_overlay("report")

    def _join_activate(self) -> None:
        if self.join_focus == 2:
            self._close_overlay()
            return
        if self.join_focus == 0:
            self.join_agree = not self.join_agree
            return
        if not self.join_agree:
            # T2: the disabled button can be focused so its reason reads;
            # ↵ on it moves focus back to the checkbox.
            self.join_focus = 0
            return
        if not self.join_busy and self.home is not None:
            self.join_busy = True
            self.run_worker(self._join(self.home.guidelines_version), group="community-join")

    async def _join(self, version: str) -> None:
        try:
            await self.backend.join(self.community_id, version)
        except Exception as exc:  # noqa: BLE001
            self.join_busy = False
            self.join_error = service.error_word(exc)
            self.paint()
            return
        self.join_busy = False
        self._close_overlay()
        await self._load_home()

    async def _leave(self) -> None:
        try:
            await self.backend.leave(self.community_id)
        except Exception as exc:  # noqa: BLE001
            self.notice = _t(_k.leave_failed(code=service.error_word(exc)))
            self.paint()
            return
        self.post_message(CommunityViewDismissed())

    async def _browse(self) -> None:
        if self.home is None:
            return
        self.browse_state = "searching"
        self.paint()
        try:
            rows = service.display_safe(
                await self.backend.browse(
                    self.community_id, self._overlay_input.value.strip(), self.browse_filter
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.browse_state = f"failed:{service.error_word(exc)}"
        else:
            self.browse_rows = model.rail_order(rows)
            self.browse_index = 0
            self.browse_state = "idle"
        self.paint()

    def _browse_open(self) -> None:
        if not self.browse_rows:
            return
        picked = self.browse_rows[self.browse_index]
        for index, channel in enumerate(self.channels):
            if channel.id == picked.id:
                self.channel_index = index
                break
        else:
            self.channels.append(picked)
            self.channel_index = len(self.channels) - 1
        self._close_overlay()
        self._open_channel()

    def open_mode(self, mode: str) -> None:
        """A second ``/community <word>`` or ``/orgs`` while the page is up."""

        if mode == "orgs":
            self._open_orgs()
        elif mode == "browse":
            self.action_browse()
        elif mode == "settings" and self.home is not None and self.home.joined:
            self.settings_tab = 1
            self._open_overlay("settings")
        elif mode == "join" and self.home is not None and not self.home.joined:
            self._open_overlay("join")
        elif self.overlay is not None:
            self._close_overlay()

    def _open_orgs(self) -> None:
        self._open_overlay("orgs")
        self.orgs_state = "loading"
        self.run_worker(self._load_orgs(), exclusive=True, group="community-orgs")

    async def _load_orgs(self) -> None:
        try:
            memberships, communities = service.display_safe(await self.backend.orgs())
        except Exception as exc:  # noqa: BLE001
            self.orgs_state = f"failed:{service.error_word(exc)}"
            self.paint()
            return
        self.orgs_private, self.orgs_communities = model.org_entries(memberships, communities)
        chosen = model.default_selection(
            self.orgs_private, self.orgs_communities, self.backend.last_selection()
        )
        every = [*self.orgs_private, *self.orgs_communities]
        self.orgs_index = every.index(chosen) if chosen in every else 0
        self.orgs_state = "ready"
        self.paint()

    def _orgs_pick(self) -> None:
        every = [*self.orgs_private, *self.orgs_communities]
        if not every:
            return
        entry = every[self.orgs_index]
        self.backend.remember_selection(entry.key)
        if entry.kind == "community":
            self._overlay_is_page = False
            self._close_overlay()
            if self.home is None or self.home.community.id != entry.key:
                self.run_worker(self._load_home(), group="community-home")
            return
        name = entry.name or _t(_k.orgs_all())
        self.post_message(CommunityOrgSelected(name))

    # -- paint -----------------------------------------------------------------

    def paint(self) -> None:
        if not self.is_mounted:
            return
        self._paint_header()
        if self.overlay is not None:
            self._overlay_text.update(self._paint_overlay())
        else:
            self._rail.update(self._paint_rail())
            self._pane_text.update(self._paint_pane())
            self._paint_composer()
        self._footer.update(self._paint_footer())

    def _typing(self) -> bool:
        return self.screen.focused in (self._input, self._overlay_input)

    def _paint_header(self) -> None:
        name = self.home.community.name if self.home is not None else ""
        offline = self.home_failure is not None
        state = _t(_k.state_offline() if offline else _k.state_connected())
        left = _t(_k.header(community=name, state=state)) if name else ""
        right = _t(_k.header_keys_typing() if self._typing() else _k.header_keys())
        width = max(20, self.size.width or 80)
        # The key hints yield whole segments, last first (R1-7): a narrow
        # header used to run the name into a half-cut hint ("connected g orgs").
        # Two cells of gap at least, so the name and the hints never touch.
        segments = right.split(" · ")
        while segments and cell_len(left) + cell_len(" · ".join(segments)) + 4 > width:
            segments.pop()
        right = " · ".join(segments)
        gap = max(2, width - cell_len(left) - cell_len(right) - 2)
        line = Text(f" {left}{' ' * gap}{right} ", style="bold", no_wrap=True, overflow="ellipsis")
        line.truncate(width, overflow="ellipsis")
        self._header.update(line)

    def _paint_rail(self) -> Text:
        text = Text()
        text.append(_t(_k.rail_channels()) + "\n", style="dim")
        width = int(self._rail.styles.width.value) - 2 if self._rail.styles.width else RAIL_WIDTH
        private_started = False
        for index, channel in enumerate(self.channels):
            if channel.visibility == "private" and not private_started:
                text.append(_t(_k.rail_private()) + "\n", style="dim")
                private_started = True
            mark = (
                "▸"
                if index == self.channel_index and self.area == "rail"
                else ("•" if index == self.channel_index else " ")
            )
            unread = str(channel.unread) if channel.unread else ""
            label = f"# {channel.name}"
            room = width - 3 - len(unread)
            if len(label) > room:
                label = label[: max(1, room - 1)] + "…"
            line = f"{mark} {label}".ljust(width - len(unread)) + unread
            text.append(line + "\n", style="bold" if index == self.channel_index else "")
        text.append("─" * (width - 1) + "\n", style="dim")
        text.append(_t(_k.rail_browse()) + "\n")
        text.append(_t(_k.rail_settings()))
        return text

    def _author_line(self, message: Any, width: int, selected: bool) -> Text:
        author = message.author
        if author.kind == "former_member":
            name = _t(_k.former_member())
        else:
            name = (
                author.agent_handle or author.display_name
                if author.kind == "agent"
                else author.display_name
            )
        badge = ""
        if author.kind == "agent":
            badge = (
                _t(_k.badge_agent(owner=author.display_name))
                if author.agent_handle
                else _t(_k.badge_agent_plain())
            )
        when = _hhmm(message.created_at)
        left = f"{'▸' if selected else ' '}{name}" + (f"  {badge}" if badge else "")
        line = Text(left, style="bold")
        line.append(" " * max(1, width - len(left) - len(when)) + when, style="dim")
        return line

    def _paint_pane(self) -> Text:
        text = Text()
        if self.home_loading and self.home is None:
            return Text(_t(_k.page_loading()))
        if self.home is None:
            failure = self.home_failure
            if failure is not None and getattr(failure, "code", "") == "community_signin_required":
                return Text(_t(_k.page_signin()))
            if failure is not None:
                code = service.error_word(failure)
                return Text(_t(_k.page_error(code=code)) + "\n" + _t(_k.page_error_keys()))
            return Text(_t(_k.page_none()))
        if not self.home.joined:
            name = self.home.community.name
            text.append(_t(_k.not_joined_title(community=name)) + "\n", style="bold")
            text.append(_t(_k.not_joined_body()) + "\n\n")
            text.append(_t(_k.not_joined_keys()), style="dim")
            return text
        channel = self.channel
        if channel is None:
            return Text(_t(_k.no_channel()))
        rows = self.messages.get(channel.id)
        failure = self.failures.get(channel.id)
        view = model.pane_state(rows, failure=failure if failure is not True else None)
        name = channel.name
        if view.state == "loading":
            text.append(_t(_k.loading(channel=name)) + "\n\n")
            text.append(_t(_k.loading_detail()), style="dim")
            return text
        if view.state == "error":
            if view.error_status:
                text.append(_t(_k.error_title(channel=name, status=str(view.error_status))) + "\n")
            else:
                text.append(_t(_k.error_title_plain(channel=name)) + "\n")
            text.append(_t(_k.error_unaffected()) + "\n")
            text.append(_t(_k.error_keys()), style="dim")
            return text
        if view.state == "offline":
            text.append(_t(_k.offline_title()) + "\n")
            text.append(_t(_k.offline_keys()) + "\n\n", style="dim")
        composer = self.composer()
        if composer is not None and composer.status == "moderation_unavailable":
            text.append(_t(_k.degraded_title()) + "\n")
            text.append(_t(_k.degraded_not_you()) + "\n")
            text.append(_t(_k.degraded_draft()) + "\n\n")
        if view.state == "empty":
            text.append(_t(_k.empty_title(channel=name)) + "\n")
            if composer is None or composer.status not in ("read_only", "locked"):
                text.append(_t(_k.empty_checked()) + "\n")
                text.append(_t(_k.empty_start()), style="dim")
            return text
        width = max(30, (self._pane.size.width or 56) - 2)
        for index, message in enumerate(view.messages):
            selected = self.area == "messages" and index == self.message_index
            text.append_text(self._author_line(message, width, selected))
            text.append("\n")
            body = message.text or ""
            for line in body.splitlines() or [""]:
                text.append(f"{'▸' if selected else ' '}  {line}\n")
            extras = []
            if message.asset_ids:
                extras.append(_t(_k.images(count=len(message.asset_ids))))
            if message.reply_count:
                extras.append(_t(_k.replies(count=message.reply_count)))
            if message.edited:
                extras.append(_t(_k.edited()))
            if message.reactions:
                extras.append("  ".join(f"{r.emoji} {r.count}" for r in message.reactions))
            if extras:
                text.append("   " + "  ·  ".join(extras) + "\n", style="dim")
        return text

    def _paint_composer(self) -> None:
        channel = self.channel
        composer = self.composer()
        joined = self.home is not None and self.home.joined
        show = joined and channel is not None
        self._audience.display = show
        self._input.display = show
        self._status.display = show
        if not show or channel is None:
            return
        if composer is not None and composer.status in ("read_only", "locked"):
            # T10: a channel nobody here can post in says so where the
            # audience line would be; the status row keeps the reason.
            self._audience.update(Text(""))
        elif channel.visibility == "private":
            self._audience.update(_t(_k.audience_private(channel=channel.name)))
        else:
            self._audience.update(_t(_k.audience_public(channel=channel.name)))
        self._input.placeholder = _t(_k.composer_placeholder(channel=channel.name))
        self._input.disabled = composer is not None and composer.status in ("read_only", "locked")
        self._paint_status()

    def _paint_status(self) -> None:
        composer = self.composer()
        channel = self.channel
        if composer is None or channel is None:
            return
        if self.notice:
            self._status.update(Text(self.notice, style="dim"))
            self.notice = ""
            return
        self._status.update(self._status_text(composer, channel.name))

    def _status_text(self, composer: model.Composer, channel: str) -> Text:
        outcome = composer.outcome
        kind = composer.status
        if kind == "checking":
            return Text(_t(_k.status_checking()))
        if kind == "posted":
            return Text(_t(_k.status_posted(time=composer.at or _now_hhmm())), style="green")
        if kind == "too_late":
            return Text(_t(_k.status_too_late(time=composer.at or _now_hhmm())))
        if kind == "cancelled":
            return Text(
                _t(_k.status_cancelled_aside() if composer.set_aside else _k.status_cancelled())
            )
        if kind == "refused":
            return Text(
                _t(_k.status_refused(category=_category(outcome.category if outcome else None))),
                style="red",
            )
        if kind == "rate_limited":
            wait = _wait(composer.seconds_left())
            scope = outcome.scope if outcome else None
            maker = {
                "account_burst": _k.status_rate_burst,
                "conversation": _k.status_rate_conversation,
                "account_daily": _k.status_rate_daily,
                "probation": _k.status_rate_probation,
            }.get(scope or "", _k.status_rate_other)
            return Text(_t(maker(wait=wait)), style="yellow")
        if kind == "moderation_unavailable":
            return Text(_t(_k.status_moderation()), style="yellow")
        if kind == "offline":
            return Text(_t(_k.status_offline()), style="yellow")
        if kind == "restricted":
            return Text(_t(_k.status_restricted(channel=channel)), style="red")
        if kind == "read_only":
            return Text(_t(_k.status_read_only(channel=channel)), style="dim")
        if kind == "locked":
            return Text(_t(_k.status_locked(channel=channel)), style="dim")
        if kind == "guidelines":
            return Text(
                _t(_k.status_guidelines(version=(outcome.version if outcome else "") or ""))
            )
        if kind == "report_sent":
            return Text(_t(_k.status_report_sent()), style="dim")
        if kind == "error":
            if outcome is not None and outcome.code == "community_signin_required":
                return Text(_t(_k.status_signin()))
            return Text(
                _t(_k.status_error(code=(outcome.code if outcome else "") or "")), style="red"
            )
        return Text(_t(_k.status_idle()), style="dim")

    def _paint_footer(self) -> Text:
        if self.overlay is not None:
            return Text(" " + self._overlay_keys(), style="dim")
        if self.home is None or not self.home.joined:
            # The pane already names this state's keys (retry / join).
            return Text("")
        maker = {
            "rail": _k.footer_rail,
            "messages": _k.footer_messages,
            "composer": _k.footer_composer,
        }[self.area]
        return Text(" " + _t(maker()), style="dim")

    # -- overlay paint ---------------------------------------------------------

    #: Overlays whose controls are pinned under a scrolling body and whose key
    #: hints move to the page footer (T2: "the agreement and Join rows stay
    #: pinned at the bottom so they never scroll out of view").
    _PINNED = ("join", "browse", "report", "settings")

    def _paint_overlay(self) -> Text:
        painter = {
            "join": self._paint_join,
            "browse": self._paint_browse,
            "report": self._paint_report,
            "why": self._paint_why,
            "settings": self._paint_settings,
            "orgs": self._paint_orgs,
            "keys": self._paint_keys,
            "info": self._paint_info,
        }.get(self.overlay or "")
        self._overlay_title.update(Text(self._overlay_heading(), style="bold"))
        self._overlay_title.display = self.overlay in self._PINNED
        controls = self._overlay_controls_text()
        self._overlay_controls.display = bool(controls.plain)
        self._overlay_controls.update(controls)
        return painter() if painter else Text("")

    def _overlay_heading(self) -> str:
        community = self.home.community.name if self.home is not None else ""
        if self.overlay == "join":
            return _t(_k.join_title(community=community))
        if self.overlay == "browse":
            return _t(_k.browse_title())
        if self.overlay == "report":
            return _t(_k.report_title())
        if self.overlay == "settings":
            return f"{_t(_k.settings_title())} · {community}"
        return ""

    def _overlay_keys(self) -> str:
        return {
            "join": _t(_k.join_keys()),
            "browse": _t(_k.browse_keys()),
            "report": _t(_k.report_keys()),
            "settings": _t(_k.settings_keys()),
            "orgs": _t(_k.orgs_keys()),
            "why": _t(_k.why_keys()),
            "info": _t(_k.why_keys()),
        }.get(self.overlay or "", "")

    def _overlay_controls_text(self) -> Text:
        text = Text()
        if self.overlay == "join" and self.home is not None:
            version = self.home.guidelines_version
            box = "[x]" if self.join_agree else "[ ]"
            text.append(f"{'▸' if self.join_focus == 0 else ' '} {box} ")
            text.append(_t(_k.join_agree(version=version)) + "\n")
            button = f"[ {_t(_k.join_button())} ]"
            text.append(f"{'▸' if self.join_focus == 1 else ' '} ")
            text.append(button, style="bold" if self.join_agree else "dim")
            if not self.join_agree:
                text.append("  " + _t(_k.join_disabled()), style="dim")
            text.append(f"    {'▸' if self.join_focus == 2 else ' '} [ {_t(_k.cancel())} ]")
            if self.join_busy:
                text.append("\n" + _t(_k.joining()), style="dim")
            if self.join_error:
                text.append("\n" + _t(_k.join_failed(code=self.join_error)), style="red")
        elif self.overlay == "settings" and self.settings_tab == 2:
            # T9: Stay has focus by default, so ↵ never leaves by accident.
            text.append(f"{'▸' if self.leave_focus == 0 else ' '} [ {_t(_k.leave_stay())} ]")
            text.append(f"    {'▸' if self.leave_focus == 1 else ' '} [ {_t(_k.leave_confirm())} ]")
        elif self.overlay == "report" and self.report_target is not None:
            text.append(_t(_k.report_unavailable()) + "\n", style="yellow")
            text.append(f"  [ {_t(_k.report_send())} ]", style="dim")
            text.append(f"    ▸ [ {_t(_k.cancel())} ]")
        return text

    def _paint_join(self) -> Text:
        text = Text()
        if self.home is None:
            return text
        version = self.home.guidelines_version
        text.append(_t(_k.join_guidelines(version=version)) + "\n", style="bold")
        for line in (self.home.guidelines_text or "").splitlines():
            text.append(f"  {line}\n")
        text.append(_t(_k.join_moderation()) + "\n")
        text.append(_t(_k.join_profile()))
        return text

    def _paint_browse(self) -> Text:
        text = Text()
        marks = {name: "(•)" if self.browse_filter == name else "( )" for name in _BROWSE_FILTERS}
        text.append(_t(_k.browse_tab_channels(count=str(len(self.browse_rows)))) + "   ")
        text.append(_t(_k.browse_people_unavailable()) + "\n", style="dim")
        text.append(
            _t(
                _k.browse_filter(
                    all=marks["all"],
                    joined=marks["joined"],
                    not_joined=marks["not_joined"],
                    private=marks["private"],
                )
            )
            + "\n"
        )
        if self.browse_state.startswith("failed:"):
            text.append(
                _t(_k.browse_failed(code=self.browse_state.split(":", 1)[1])) + "\n", style="red"
            )
        elif self.browse_state == "searching" and not self.browse_rows:
            text.append(_t(_k.browse_searching()) + "\n", style="dim")
        if not self.browse_rows and self.browse_state == "idle":
            text.append(_t(_k.browse_empty(query=self._overlay_input.value.strip())) + "\n")
            if self.browse_filter != "all":
                text.append(_t(_k.browse_empty_filter()) + "\n")
            text.append(_t(_k.browse_suggest()) + "\n")
        width = max(40, (self._overlay_scroll.size.width or 70) - 2)
        for index, row in enumerate(self.browse_rows):
            mark = "▸" if index == self.browse_index else " "
            tail = _t(_k.browse_joined()) if row.member else ""
            purpose = row.purpose or ""
            if row.visibility == "private":
                purpose = _t(_k.browse_private()) + (f" · {purpose}" if purpose else "")
            name = f"{mark} # {row.name:<20} "
            room = max(0, width - len(name) - len(tail) - 1)
            purpose = purpose[:room].ljust(room)
            text.append(
                f"{name}{purpose} {tail}\n", style="dim" if self.browse_state == "searching" else ""
            )
        if self.browse_rows:
            text.append(_t(_k.browse_count(count=len(self.browse_rows))), style="dim")
        return text

    def _paint_report(self) -> Text:
        text = Text()
        target = self.report_target
        if target is None:
            text.append(_t(_k.report_none()), style="dim")
            return text
        excerpt = (target.text or "").splitlines()[0][:50] if target.text else ""
        text.append(f'{target.author.display_name} · {_hhmm(target.created_at)} · "{excerpt}"\n')
        text.append(_t(_k.report_what()) + "\n")
        for index, category in enumerate(service.REFUSAL_CATEGORIES):
            mark = "▸" if index == self.report_index else " "
            radio = "(•)" if index == self.report_index else "( )"
            text.append(f"{mark} {radio} {_category(category)}\n")
        text.append("  " + _t(_k.report_note_help()) + "\n", style="dim")
        text.append(_t(_k.report_anonymous(name=target.author.display_name)))
        return text

    def _paint_why(self) -> Text:
        composer = self.composer()
        outcome = composer.outcome if composer is not None else None
        text = Text()
        text.append(_t(_k.why_title()) + "\n", style="bold")
        if outcome is not None:
            text.append(_t(_k.why_category(category=_category(outcome.category))) + "\n")
            text.append(
                _t(
                    _k.why_reference(
                        reference=outcome.reference or "", channel=getattr(self.channel, "name", "")
                    )
                )
                + "\n"
            )
        text.append(_t(_k.why_only_you()) + "\n\n")
        text.append(_t(_k.why_keys()), style="dim")
        return text

    def _paint_info(self) -> Text:
        text = Text()
        text.append(_t(_k.info_title()) + "\n", style="bold")
        text.append(_t(_k.info_body()) + "\n")
        composer = self.composer()
        if composer is not None and composer.set_aside:
            text.append(_t(_k.info_restore()) + "\n")
        text.append("\n" + _t(_k.why_keys()), style="dim")
        return text

    def _paint_keys(self) -> Text:
        return Text(_t(_k.keys_title()) + "\n", style="bold") + Text(_t(_k.keys_body()))

    def _paint_settings(self) -> Text:
        text = Text()
        community = self.home.community.name if self.home is not None else ""
        tabs = [_k.settings_tab_profile, _k.settings_tab_agents, _k.settings_tab_leave]
        labels = [
            f"[{_t(maker())}]" if index == self.settings_tab else f" {_t(maker())} "
            for index, maker in enumerate(tabs)
        ]
        text.append(
            _t(_k.settings_tabs(profile=labels[0], agents=labels[1], leave=labels[2])) + "\n"
        )
        text.append("─" * 60 + "\n", style="dim")
        if self.settings_tab == 0:
            text.append(_t(_k.settings_profile_unavailable()) + "\n")
        elif self.settings_tab == 1:
            text.append(_t(_k.settings_agents_head()) + "\n")
            pad = max((len(c.name) for c in self.channels), default=0)
            for channel in self.channels:
                text.append(
                    "  " + _t(_k.settings_agents_row(channel=channel.name.ljust(pad))) + "\n"
                )
            text.append(_t(_k.settings_agents_no_grants()) + "\n", style="dim")
            text.append(_t(_k.settings_agents_taint()) + "\n")
        else:
            text.append(_t(_k.leave_title(community=community)) + "\n", style="bold")
            text.append(_t(_k.leave_switch()) + "\n\n")
            text.append(_t(_k.leave_effects()))
        return text

    def _current_org_key(self) -> str | None:
        """The org being VIEWED, which the switcher checks apart from its cursor (R1-6, D21).

        Opened over the community page, that is the community on screen. Opened
        on its own (``/orgs`` from a session), it is the last selection — the
        private org this machine was last pointed at — else the home org.
        """
        if self.home is not None and not self._overlay_is_page:
            return str(self.home.community.id)
        every = [*self.orgs_private, *self.orgs_communities]
        last = self.backend.last_selection()
        if last and any(entry.key == last for entry in every):
            return last
        home = next((entry for entry in self.orgs_private if entry.home), None)
        return home.key if home is not None else None

    def _org_marks(self, index: int, entry: Any, current: str | None) -> str:
        """Cursor then check: ``▸ ✓``. Two columns, so moving never reflows a row."""
        cursor = "▸" if index == self.orgs_index else " "
        return f"{cursor} {'✓' if current is not None and entry.key == current else ' '}"

    def _paint_orgs(self) -> Text:
        text = Text()
        text.append(_t(_k.orgs_title()) + "\n", style="bold")
        if self.orgs_state == "loading":
            return text + Text(_t(_k.orgs_loading()), style="dim")
        if self.orgs_state.startswith("failed:"):
            return text + Text(
                _t(_k.orgs_failed(code=self.orgs_state.split(":", 1)[1])), style="red"
            )
        index = 0
        text.append(_t(_k.orgs_group_private()) + "\n", style="dim")
        if not self.orgs_private:
            text.append("  " + _t(_k.orgs_none_private()) + "\n", style="dim")
        current = self._current_org_key()
        for entry in self.orgs_private:
            if entry.kind == "all":
                label = _t(_k.orgs_all())
            elif entry.invited:
                label = _t(_k.orgs_invited(name=entry.name))
            else:
                label = _t(_k.orgs_team(name=entry.name))
            text.append(f"{self._org_marks(index, entry, current)} {label}\n")
            index += 1
        text.append(_t(_k.orgs_group_communities()) + "\n", style="dim")
        if not self.orgs_communities:
            text.append("  " + _t(_k.orgs_none_communities()) + "\n", style="dim")
        for entry in self.orgs_communities:
            text.append(
                f"{self._org_marks(index, entry, current)} "
                f"{_t(_k.orgs_community(name=entry.name))}\n"
            )
            index += 1
        text.append("\n" + _t(_k.orgs_note()) + "\n\n", style="dim")
        text.append(_t(_k.orgs_keys()), style="dim")
        return text


__all__ = ["CommunityOrgSelected", "CommunityView", "CommunityViewDismissed"]
