"""``/guard``: the TUI's Radient Guard screen (Guard RFC v0.9 §5.2.3, §5.4, §14', slice L2b).

WHAT IT SHOWS, in priority order (one decision on screen at a time, so a person
never answers the wrong card):

1. a queued consent prompt (§15.4' C4) -- "Don't share" / "Decide later" /
   "Share", or on a widening "Keep current" / "Decide later" / "Share the new
   items";
2. an agent-drafted report (``guard/reports.py``) -- "Send report" / "Discard";
3. the overview: the passive indicator per org, "Answered on <surface>",
   "Stop sharing" (→ the C2 confirm) or "Share again";
4. for an account with no Team org, the non-Team card (§15.4' C1).

PRESENTATION SAFETY (§5.2.3). Focus starts on the non-sharing choice. Esc is
"Decide later": the prompt stays pending, it is NOT a decline. Any button press
within :data:`DEBOUNCE_S` of a card rendering is ignored, so a typed-ahead Enter
meant for the composer cannot answer it (a UX debounce, not a security bound).

NEVER INSIDE A TURN. The screen opens only on ``/guard`` (or a key the person
presses); the arrival of a prompt is a one-line notice plus a status-bar badge
(``OperatorApp._poll_guard``), and nothing here touches the session, so a
running turn keeps streaming underneath.

WHO MAY ACCEPT. A person at this terminal: the TUI process is the interactive
host on this computer, so it answers with the host capability
(``capability=True``) and :func:`harness.approval.admit_increasing` decides. The
model has no path here -- no tool, peer ``send`` or ``bridge_decide`` reaches a
Textual button.

NETWORK. An accept and a revoke send ONE request each, from a worker thread
(``guard.consent``); the non-Team card reads ``GET /v1/team-trial/eligibility``
through the injected loader. Nothing on this screen calls ``/v1/guard/*`` for a
non-Team account.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from local_operator.guard import surface
from local_operator.i18n.keys import tui_guard as _keys
from local_operator.i18n.text import tr

_t = tr("tui.guard")

#: §5.2.3: input within 500 ms of a card rendering is ignored.
DEBOUNCE_S = 0.5
#: How often an open card re-reads its prompt row, so an answer given on the
#: desktop or the phone closes it with "Answered on <surface>".
RECHECK_S = 2.0

EligibilityLoader = Callable[[], Mapping[str, Any] | None]

_SCOPE_LINE = {
    "decisions": _keys.scope_decisions,
    "task_time": _keys.scope_task_time,
    "review_cycles": _keys.scope_review_cycles,
    "posture": _keys.scope_posture,
    "trace_detail": _keys.scope_trace_detail,
}
_SCOPE_ORDER = (
    "decisions",
    "posture",
    "task_time",
    "review_cycles",
    "reports",
    "trace_detail",
    "admin_share",
)
_SCOPE_LABEL = {
    "decisions": _keys.scope_label_decisions,
    "posture": _keys.scope_label_posture,
    "task_time": _keys.scope_label_task_time,
    "review_cycles": _keys.scope_label_review_cycles,
    "reports": _keys.scope_label_reports,
    "trace_detail": _keys.scope_label_trace_detail,
    "admin_share": _keys.scope_label_admin_share,
}
_REASON = {
    "new_org": lambda org, scopes: _keys.prompt_reason_new_org(org=org),
    "widened": lambda org, scopes: _keys.prompt_reason_widened(org=org, scopes=scopes),
    "reactivated": lambda org, scopes: _keys.prompt_reason_reactivated(org=org),
    "plan_gap": lambda org, scopes: _keys.prompt_reason_plan_gap(org=org),
    "new_device": lambda org, scopes: _keys.prompt_reason_new_device(),
    "rejoined": lambda org, scopes: _keys.prompt_reason_rejoined(org=org),
}
_PAUSE_REASON = {
    "entitlement_lapsed": lambda org: _keys.reason_entitlement_lapsed(),
    "org_off": lambda org: _keys.reason_org_off(org=org),
    "membership_ended": lambda org: _keys.reason_membership_ended(org=org),
    "scope_revoked": lambda org: _keys.reason_scope_revoked(),
    "revoke_pending": lambda org: _keys.reason_revoke_pending(),
}
_SURFACE = {
    "tui": _keys.surface_tui,
    "desktop": _keys.surface_desktop,
    "mobile": _keys.surface_mobile,
}
_REPORT_KIND = {
    "credential_exposure": _keys.report_kind_credential_exposure,
    "excessive_guardrail": _keys.report_kind_excessive_guardrail,
    "other": _keys.report_kind_other,
}


# ---------------------------------------------------------------------------
# Text (pure; the status bar and the notice reuse these)
# ---------------------------------------------------------------------------


def indicator_text(state: surface.OrgState) -> str:
    """The passive indicator line (§5.3). Never "paused" for a pending widening."""

    if state.state == surface.STATE_SHARING:
        return _t(_keys.indicator_sharing(org=state.org))
    if state.state == surface.STATE_SHARING_PENDING:
        return _t(_keys.indicator_sharing_pending(org=state.org))
    if state.state == surface.STATE_WAITING:
        return _t(_keys.indicator_waiting(org=state.org))
    if state.state == surface.STATE_NOT_SHARING:
        return _t(_keys.indicator_not_sharing(org=state.org))
    reason = _PAUSE_REASON.get(state.reason)
    if reason is None:  # an unknown server reason: say paused, never the enum
        return _t(_keys.indicator_paused(reason=_t(_keys.reason_scope_revoked())))
    return _t(_keys.indicator_paused(reason=_t(reason(state.org))))


def badge_text(states: list[surface.OrgState]) -> str:
    """The status-bar badge: a waiting choice outranks a sharing state. ``""`` = none."""

    if any(s.state == surface.STATE_WAITING for s in states):
        return _t(_keys.badge_waiting())
    if any(s.state == surface.STATE_SHARING_PENDING for s in states):
        # Still sharing (§13.3' consent_stale): never "paused", never "choose" alone.
        return _t(_keys.badge_sharing_pending())
    if any(s.state == surface.STATE_SHARING for s in states):
        return _t(_keys.badge_sharing())
    return ""


def notice_text(state: surface.OrgState) -> str:
    """§15.4' C6, the one-line host notice."""

    return _t(_keys.notice_org_on(org=state.org))


def surface_name(via: str) -> str:
    key = _SURFACE.get(via)
    return _t(key()) if key is not None else via


def prompt_lines(row: Any) -> tuple[str, list[str]]:
    """``(title, body lines)`` for one queued prompt, per §15.4' C4."""

    org = row.org_display_name
    # The C4 order (decisions, posture, task_time, review_cycles, reports,
    # trace_detail, admin_share), never the queue's sorted order.
    asked = [s for s in _SCOPE_ORDER if s in set(row.asked_scopes)]
    labels = ", ".join(_t(_SCOPE_LABEL[s]()) for s in asked if s in _SCOPE_LABEL)
    reason = _REASON.get(row.reason, _REASON["new_org"])
    lines = [_t(reason(org, labels)), "", _t(_keys.prompt_would_send())]
    for scope in asked:
        if scope == "reports":
            text = _t(_keys.scope_reports(org=org))
        elif scope == "admin_share":
            text = _t(_keys.scope_admin_share(org=org))
        elif scope in _SCOPE_LINE:
            text = _t(_SCOPE_LINE[scope]())
        else:
            continue
        lines.append(f"  • {text}")
    lines.append("")
    lines.append(_t(_keys.prompt_never_sent()))
    offered = set(row.grant_scopes)
    posture, share = "posture" in offered, "admin_share" in offered
    if posture and share:
        lines.append(_t(_keys.prompt_visibility_admin_share_posture()))
    elif share:
        lines.append(_t(_keys.prompt_visibility_admin_share()))
    elif posture:
        lines.append(_t(_keys.prompt_visibility_posture()))
    else:
        lines.append(_t(_keys.prompt_visibility()))
    lines.append(_t(_keys.prompt_background()))
    return _t(_keys.prompt_title(org=org)), lines


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


class GuardScreen(ModalScreen[str]):
    """One Guard decision at a time; Esc = Decide later (never a decline)."""

    BINDINGS = [Binding("escape", "later", "Decide later", show=False)]

    def __init__(
        self,
        config_dir: Path | str,
        *,
        eligibility_loader: EligibilityLoader | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._config_dir = Path(config_dir)
        self._eligibility_loader = eligibility_loader
        self._clock = clock
        self._armed_at = 0.0
        self._mode = "overview"
        self._row: Any = None
        self._draft: Any = None
        self._revoke: surface.OrgState | None = None
        self._states: list[surface.OrgState] = []
        self._card: surface.NonTeamCard | None = None
        self._message = ""
        self._busy = False
        self._load()

    # -- state ---------------------------------------------------------------

    def _load(self) -> None:
        from local_operator.guard.reports import GuardReports

        self._states = surface.org_states(self._config_dir)
        self._row = next((s.prompt for s in self._states if s.prompt is not None), None)
        drafts = GuardReports(self._config_dir).drafts() if self._states else []
        known = {s.tenant for s in self._states}
        self._draft = next((d for d in drafts if d.tenant in known), None)
        if self._revoke is not None:
            self._mode = "revoke"
        elif self._row is not None:
            self._mode = "prompt"
        elif self._draft is not None:
            self._mode = "report"
        elif self._states:
            self._mode = "overview"
        elif not surface.insights_tenants(self._config_dir) and not _has_team(self._config_dir):
            self._mode = "card"
            if self._card is None:
                self._card = surface.non_team_card(None)
        else:
            self._mode = "overview"

    def _org_name(self, tenant: str) -> str:
        for state in self._states:
            if state.tenant == tenant:
                return state.org
        return tenant

    # -- compose -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Container(classes="guard-card"):
            yield Static(self._title(), id="guard-title", markup=False)
            with VerticalScroll(id="guard-body", classes="gap-above"):
                for line in self._body():
                    # A bullet keeps its hanging indent when it wraps: the
                    # indent is the widget's padding, the bullet its first cell
                    # (the `/move --to` card's per-line padding, review D3).
                    depth = len(line) - len(line.lstrip(" "))
                    text = line.lstrip(" ")
                    if text.startswith("• "):
                        with Horizontal(classes="guard-bullet") as bullet:
                            bullet.styles.padding = (0, 0, 0, depth)
                            yield Static("•", classes="guard-bullet-mark", markup=False)
                            yield Static(Content(text[2:]), classes="guard-line", markup=False)
                        continue
                    row = Static(Content(text), classes="guard-line", markup=False)
                    row.styles.padding = (0, 0, 0, depth)
                    yield row
            with Horizontal(id="guard-buttons", classes="gap-above"):
                yield from self._buttons()

    def _title(self) -> str:
        if self._mode == "prompt":
            return prompt_lines(self._row)[0]
        if self._mode == "report":
            return _t(_keys.report_title(org=self._org_name(self._draft.tenant)))
        if self._mode == "revoke" and self._revoke is not None:
            return _t(_keys.revoke_title(org=self._revoke.org))
        if self._mode == "card":
            return _t(_keys.card_title())
        return _t(_keys.screen_title())

    def _body(self) -> list[str]:
        lines: list[str] = []
        if self._message:
            lines += [self._message, ""]
        if self._mode == "prompt":
            return lines + prompt_lines(self._row)[1]
        if self._mode == "report":
            kind = _REPORT_KIND.get(self._draft.kind, _keys.report_kind_other)
            return lines + [f"  • {_t(kind())}", "", _t(_keys.report_body())]
        if self._mode == "revoke" and self._revoke is not None:
            consent = _consent_scopes(self._config_dir, self._revoke.tenant)
            body = _keys.revoke_body_trace() if "trace_detail" in consent else _keys.revoke_body()
            return lines + [_t(body), _t(_keys.revoke_retained()), _t(_keys.revoke_totals())]
        if self._mode == "card":
            return lines + [_t(_keys.card_body()), "", _t(_keys.card_footer())]
        if not self._states:
            return lines + [_t(_keys.screen_none())]
        for state in self._states:
            lines.append(indicator_text(state))
            if state.answered_via and state.answered_via != "tui":
                lines.append(f"  {_t(_keys.answered_on(surface=surface_name(state.answered_via)))}")
        return lines

    def _buttons(self) -> ComposeResult:
        if self._mode == "prompt":
            if self._row.widening:
                yield Button(_t(_keys.button_keep_current()), id="guard-no")
                yield Button(_t(_keys.button_decide_later()), id="guard-later")
                yield Button(_t(_keys.button_share_new()), id="guard-yes", variant="primary")
            else:
                yield Button(_t(_keys.button_dont_share()), id="guard-no")
                yield Button(_t(_keys.button_decide_later()), id="guard-later")
                yield Button(_t(_keys.button_share()), id="guard-yes", variant="primary")
            return
        if self._mode == "report":
            yield Button(_t(_keys.button_discard()), id="guard-no")
            yield Button(_t(_keys.button_decide_later()), id="guard-later")
            yield Button(_t(_keys.button_send_report()), id="guard-yes", variant="primary")
            return
        if self._mode == "revoke":
            yield Button(_t(_keys.button_keep_sharing()), id="guard-no")
            yield Button(_t(_keys.button_stop_sharing()), id="guard-yes", variant="error")
            return
        if self._mode == "card":
            yield Button(_t(_keys.screen_close()), id="guard-close")
            label = (
                _t(_keys.card_cta_trial(trial_days=str(self._card.trial_days)))
                if self._card is not None and self._card.trial_days
                else _t(_keys.card_cta_see())
            )
            yield Button(label, id="guard-cta", variant="primary")
            return
        yield Button(_t(_keys.screen_close()), id="guard-close")
        sharing = next(
            (
                s
                for s in self._states
                if s.state in (surface.STATE_SHARING, surface.STATE_SHARING_PENDING)
            ),
            None,
        )
        stopped = next(
            (
                s
                for s in self._states
                if s.state in (surface.STATE_NOT_SHARING, surface.STATE_PAUSED)
                and s.reason != "revoke_pending"
            ),
            None,
        )
        if sharing is not None:
            yield Button(_t(_keys.button_stop_sharing()), id="guard-stop", variant="error")
        elif stopped is not None:
            yield Button(_t(_keys.button_share_again()), id="guard-again")

    # -- lifecycle -----------------------------------------------------------

    def on_mount(self) -> None:
        self._arm()
        self.set_interval(RECHECK_S, self._recheck)
        if self._mode == "card" and self._eligibility_loader is not None:
            self.run_worker(self._load_eligibility(), exclusive=True, group="guard-card")

    def _arm(self) -> None:
        """Start the debounce and put focus on the non-sharing choice."""

        self._armed_at = self._clock() + DEBOUNCE_S
        for selector in ("#guard-no", "#guard-close"):
            found = self.query(selector)
            if found:
                found.first(Button).focus()
                return

    async def _rerender(self) -> None:
        self._load()
        await self.recompose()
        self._arm()

    async def _load_eligibility(self) -> None:
        loader = self._eligibility_loader
        if loader is None:
            return
        try:
            body = await asyncio.to_thread(loader)
        except Exception:  # noqa: BLE001 - the card falls back to "See Radient Team"
            body = None
        self._card = surface.non_team_card(body)
        if self._mode == "card":
            await self.recompose()
            self._arm()

    async def _recheck(self) -> None:
        """Close a card answered on another surface with "Answered on <surface>"."""

        if self._mode != "prompt" or self._busy or self._row is None:
            return
        from local_operator.guard.consent import GuardConsentHost

        host = GuardConsentHost(self._config_dir, transport=None, mirror=False)
        try:
            host.reconcile_mirrors()
        except Exception:  # noqa: BLE001
            pass
        current = host.prompt(self._row.tenant, self._row.collection_rev)
        if current is not None and current.state == "pending":
            return
        via = ""
        for state in surface.org_states(self._config_dir):
            if state.tenant == self._row.tenant:
                via = state.answered_via
        # The overview that follows carries "Answered on <surface>" under the
        # org, so the card closes into it without repeating the line.
        self._message = (
            ""
            if via and via != "tui"
            else _t(_keys.answered_on(surface=surface_name(via or "desktop")))
        )
        await self._rerender()

    # -- answers -------------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._busy or self._clock() < self._armed_at:
            return  # typed-ahead input (§5.2.3), or an answer already in flight
        button = event.button.id or ""
        if button in ("guard-close", "guard-later"):
            if button == "guard-later" and self._mode == "prompt":
                self.app.notify(_t(_keys.screen_later()))
            self.dismiss("later")
            return
        if button == "guard-cta" and self._card is not None:
            self.app.open_url(self._card.url)
            return
        if button == "guard-stop":
            self._revoke = next(
                (
                    s
                    for s in self._states
                    if s.state in (surface.STATE_SHARING, surface.STATE_SHARING_PENDING)
                ),
                None,
            )
            self._message = ""
            await self._rerender()
            return
        if button == "guard-again":
            stopped = next(
                (
                    s
                    for s in self._states
                    if s.state in (surface.STATE_NOT_SHARING, surface.STATE_PAUSED)
                ),
                None,
            )
            if stopped is not None:
                self._host().share_again(stopped.tenant)
            self._message = _t(_keys.screen_later())
            await self._rerender()
            return
        await self._answer(yes=button == "guard-yes")

    def action_later(self) -> None:
        if self._mode == "revoke":
            self._revoke = None
        self.dismiss("later")

    def _host(self) -> Any:
        from local_operator.guard.consent import GuardConsentHost, default_transport

        root = self._config_dir
        return GuardConsentHost(root, transport=lambda: default_transport(root))

    async def _answer(self, *, yes: bool) -> None:
        self._busy = True
        self._message = _t(_keys.screen_working())
        try:
            if self._mode == "prompt":
                self._message = await self._answer_prompt(yes)
            elif self._mode == "report":
                self._message = self._answer_report(yes)
            elif self._mode == "revoke" and self._revoke is not None:
                org = self._revoke.org
                tenant = self._revoke.tenant
                self._revoke = None
                if yes:
                    outcome = await asyncio.to_thread(self._host().revoke, tenant)
                    self._message = (
                        _t(_keys.screen_revoked(org=org))
                        if outcome.status == "granted" or outcome.status == "revoked"
                        else indicator_text(
                            surface.OrgState(
                                tenant, org, "", surface.STATE_PAUSED, "revoke_pending"
                            )
                        )
                    )
                else:
                    self._message = ""
        except Exception:  # noqa: BLE001 - say so; nothing was decided
            self._message = _t(_keys.screen_failed())
        finally:
            self._busy = False
        await self._rerender()

    async def _answer_prompt(self, yes: bool) -> str:
        row = self._row
        org = row.org_display_name
        host = self._host()
        if not yes:
            host.decline(row.tenant, row.collection_rev, surface="tui")
            return _t(_keys.screen_declined(org=org))
        # The person pressed Share at this terminal: the host capability.
        outcome = await asyncio.to_thread(
            host.accept,
            row.tenant,
            row.collection_rev,
            capability=True,
            signature=None,
            source="tui",
        )
        if outcome.status == "granted":
            return _t(_keys.screen_shared(org=org))
        return _t(_keys.screen_failed())

    def _answer_report(self, yes: bool) -> str:
        from local_operator.guard.reports import GuardReports

        reports = GuardReports(self._config_dir)
        org = self._org_name(self._draft.tenant)
        if yes:
            reports.confirm(self._draft.id, capability=True, signature=None, surface="tui")
            return _t(_keys.report_confirmed(org=org))
        reports.discard(self._draft.id, surface="tui")
        return _t(_keys.report_discarded())


def _has_team(config_dir: Path) -> bool:
    from local_operator.hub_org import eligibility

    return eligibility.is_eligible(config_dir)


def _consent_scopes(config_dir: Path, tenant: str) -> set[str]:
    from local_operator.guard.consent import GuardConsentHost

    consent = GuardConsentHost(config_dir, transport=None).consent(tenant)
    return set(consent.get("scopes") or []) if consent else set()


def load_trial_eligibility(config_dir: Path | str) -> Mapping[str, Any] | None:
    """``GET /v1/team-trial/eligibility`` as the signed-in person, or ``None``.

    The non-Team card's only read (§14', §0.13). It is NOT a ``/v1/guard/*``
    call and writes nothing. Run it off the event loop (it resolves the
    person's OAuth access on a private loop).
    """

    from local_operator.hub_org import eligibility

    resolved = eligibility.person_client(Path(config_dir))
    if resolved is None:
        return None
    return resolved[0].team_trial_eligibility()


__all__ = [
    "DEBOUNCE_S",
    "GuardScreen",
    "badge_text",
    "indicator_text",
    "load_trial_eligibility",
    "notice_text",
    "prompt_lines",
    "surface_name",
]
