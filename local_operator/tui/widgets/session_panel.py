"""Read-only current-session diagnostics, using the analytics ledger's units.

The report is a close/reopen snapshot, not a live bill. Runtime scalars are
captured before the worker reads SQLite; neither prompts nor tool payloads ever
enter this view. Keep this screen independent of AnalyticsScreen's calendar
charts and metric-toggle bindings while sharing its visual chrome and formatters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.analytics.model import COMPONENT_LABELS, SessionReport
from local_operator.session.protocol import SessionProtocol
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.analytics_panel import (
    format_cost,
    format_percent,
    format_tokens,
)


@dataclass(frozen=True)
class SessionDiagnostics:
    """Only scalar public state; never hold a mutable session across a disk read."""

    session_id: str
    name: str
    selected_model: str
    effective_model: str
    streaming: bool
    context_tokens: int | None = None
    context_window: int | None = None
    context_is_estimate: bool | None = None
    generation: int | None = None
    epoch: str | None = None

    @classmethod
    def capture(cls, session: SessionProtocol) -> SessionDiagnostics:
        # The canonical frontend snapshot is already mirrored on RemoteSession.
        # Reduced SDK facades need not have it; don't traverse private context or
        # tokenize history just to fill a missing diagnostic.
        state = getattr(session, "frontend_state", None)
        model = session.effective_model
        return cls(
            session_id=session.session_id,
            name=session.conversation_name,
            selected_model=session.model_label,
            effective_model=session.effective_model_label,
            streaming=session.is_streaming,
            context_tokens=getattr(state, "context_tokens", None),
            context_window=getattr(state, "context_window", None)
            or getattr(model, "context_window", None),
            context_is_estimate=getattr(state, "context_is_estimate", None),
            generation=getattr(state, "generation", None),
            epoch=getattr(state, "epoch", None),
        )


def _stamp(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "unknown"
    return datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _milliseconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:,.0f} ms"


def build_session_report(report: SessionReport, runtime: SessionDiagnostics) -> Text:
    """Stacked, wrapping rows keep full IDs readable even in a 50-column pane."""
    fg = Style(color=theme_mod.semantic_color("fg"))
    muted = Style(color=theme_mod.semantic_color("muted"))
    result = Text(style=fg, overflow="fold")

    def line(label: str, value: str = "") -> None:
        result.append(label, style=muted)
        if value:
            result.append("  " + value, style=fg)
        result.append("\n")

    def section(title: str) -> None:
        result.append(
            "\n" + title + "\n", style=Style(color=theme_mod.semantic_color("fg"), bold=True)
        )

    line(runtime.name or "Untitled session")
    line("ID", runtime.session_id)
    line("Current session only. Close and reopen to refresh.")
    aggregate = report.aggregate
    if not report.available:
        section("Ledger unavailable")
        line("Could not read local usage records. Close and reopen to try again.")
    elif not aggregate.calls:
        section("No recorded requests")
        line("No retained usage records for this session yet.")
    else:
        section("Recorded usage")
        line("Requests", f"{aggregate.calls:,}")
        line("Cost (reported / est.)", format_cost(aggregate))
        line("Tokens", f"{aggregate.total_tokens:,} total")
        line("Input context", f"{aggregate.context_tokens:,}")
        line(
            "Output",
            f"{aggregate.output_tokens:,} (includes {aggregate.reasoning_tokens:,} reasoning)",
        )
        line(
            "Cache read",
            f"{aggregate.cache_read_tokens:,} / {aggregate.context_tokens:,} "
            f"context ({format_percent(aggregate.cache_hit_rate)})",
        )
        line("Cache write", f"{aggregate.cache_write_tokens:,}")
        line(
            "Usage missing",
            f"{report.missing_usage_calls:,} requests; {report.unknown_usage_calls:,} unknown",
        )
        line("Period", _stamp(report.first_ts_ms) + " to " + _stamp(report.last_ts_ms))
        line("Cost is provider-reported or a list-price estimate. Source is not retained per row.")
        line(
            "+ means a partial cost; unknown cost is not free. "
            "Input/output/tool dollars are unavailable separately."
        )

        section("Provider / model")
        for (provider, model), group in sorted(
            report.by_model.items(), key=lambda item: -item[1].total_tokens
        ):
            line(f"{provider}/{model}")
            line(
                "  Usage",
                f"{group.calls:,} requests · {format_tokens(group.total_tokens)} "
                f"tokens · {format_cost(group)}",
            )
            line(
                "  Input / output",
                f"{format_tokens(group.context_tokens)} / {format_tokens(group.output_tokens)}",
            )

        section("Estimated input breakdown")
        line(
            "Character-weighted attribution of recorded input "
            "context, not separately billed tokens or dollars."
        )
        for key, value in sorted(aggregate.components.items(), key=lambda item: -item[1]):
            line(COMPONENT_LABELS.get(key, key), f"~{format_tokens(value)}")
        unattributed = max(0, aggregate.context_tokens - sum(aggregate.components.values()))
        if unattributed:
            line("Unattributed (older records)", format_tokens(unattributed))

        section("Request purpose / outcome")
        for (purpose, outcome), count in sorted(report.by_purpose_outcome.items()):
            line(f"{purpose} / {outcome}", f"{count:,}")

        section("Logical request timings")
        line(
            "Observed wall time includes retries and consumer "
            "backpressure; not provider compute or per-attempt timing. "
            "First output is the first text or tool-call delta."
        )
        for key, label in (
            ("duration_ms", "Duration"),
            ("ttft_ms", "First output"),
            ("preparation_ms", "Preparation"),
        ):
            timing = report.timings.get(key)
            if timing is None or not timing.samples:
                line(label, "unknown (0 samples)")
            else:
                line(label, f"{_milliseconds(timing.mean_ms)} mean ({timing.samples:,} samples)")
                line("  Range", f"{_milliseconds(timing.min_ms)} to {_milliseconds(timing.max_ms)}")

    section("Runtime snapshot")
    line("Selected", runtime.selected_model)
    line("Effective", runtime.effective_model)
    line("State", "streaming" if runtime.streaming else "idle")
    line(
        "Turn generation",
        str(runtime.generation) if runtime.generation is not None else "unavailable",
    )
    context = "unavailable"
    if runtime.context_tokens is not None:
        context = ("~" if runtime.context_is_estimate else "") + f"{runtime.context_tokens:,}"
    limit = f"{runtime.context_window:,}" if runtime.context_window else "unavailable"
    line("Context / limit", context + " / " + limit)
    line("Compaction count", "unavailable")

    if report.recent:
        section(f"Recent requests ({len(report.recent)} of {aggregate.calls:,})")
        line("Newest first; request IDs identify logical requests, not retry attempts.")
        for row in report.recent:
            line(_stamp(row.ts_ms))
            line("ID", row.request_id or "not recorded")
            line(f"{row.provider}/{row.model_id}")
            line(f"{row.purpose} / {row.outcome}")
            usage = (
                "unknown"
                if row.usage_reported is None
                else "reported" if row.usage_reported else "missing"
            )
            line(
                "Usage",
                f"{usage} · input {format_tokens(row.context_tokens)} "
                f"· output {format_tokens(row.output_tokens)}",
            )
            line(
                "Duration / first output",
                f"{_milliseconds(row.duration_ms)} / {_milliseconds(row.ttft_ms)}",
            )
            line("Preparation", _milliseconds(row.preparation_ms))
            result.append("\n")
    section("Scope and limits")
    line(
        "Retained local ledger rows for this exact ID only. Child sessions and copied "
        "fork history are excluded; resuming the same ID includes its retained records."
    )
    line(
        "Older records may have been pruned. In-flight requests and pending "
        "recorder writes may not appear yet. This command makes no model request."
    )
    return result


class SessionScreen(ModalScreen[None]):
    """A snapshot with analytics chrome, but no all-session charts or toggles."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=False),
        Binding("q", "dismiss_screen", "Back", show=False),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, report: SessionReport, runtime: SessionDiagnostics) -> None:
        super().__init__()
        self.report = report
        self.runtime = runtime

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            yield Static(
                Text("Session diagnostics\n" + "─" * 140, no_wrap=True, overflow="crop"),
                id="session-report-title",
            )
            with VerticalScroll(id="session-report-scroll") as scroll:
                self._scroll = scroll
                yield Static(
                    build_session_report(self.report, self.runtime), id="session-report-body"
                )
            self._hint = Static("esc / q back", id="session-report-hint")
            yield self._hint

    def on_mount(self) -> None:
        self.call_after_refresh(self._sync_hint)

    def on_resize(self) -> None:
        self.call_after_refresh(self._sync_hint)

    def _sync_hint(self) -> None:
        # Like analytics, don't advertise a dead scroll control when an empty
        # report fits a tall terminal. Layout must settle before measuring it.
        scroll = getattr(self, "_scroll", None)
        hint = getattr(self, "_hint", None)
        if scroll is not None and hint is not None and hint.is_mounted:
            hint.update("esc / q back" + (" · ↑↓ scroll" if scroll.max_scroll_y > 0 else ""))

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    def action_scroll_up(self) -> None:
        self._scroll.scroll_up()

    def action_scroll_down(self) -> None:
        self._scroll.scroll_down()

    def action_page_up(self) -> None:
        self._scroll.scroll_page_up()

    def action_page_down(self) -> None:
        self._scroll.scroll_page_down()

    def action_scroll_home(self) -> None:
        self._scroll.scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll.scroll_end()
