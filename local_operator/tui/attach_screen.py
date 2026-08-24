"""The attach screen: follow and steer a session another process owns.

``/resume <id>`` of a live session used to refuse — "already open in another
process" — because two writers on one transcript is the corruption case the
refusal exists to prevent. But the mobile stack already solved multi-front-end
for the phone: the owner runs a registrant whose loopback socket carries
whole-projection repaints and steering ops. This screen is a SECOND TERMINAL's
window onto that socket: it renders the owner's ``SessionProjection`` and
routes submits to ``prompt``/``steer``, never opening the transcript directory
for writing. The owner stays the only writer; this process is a front end.

Deliberately a ``Screen``, not a second ``OperatorApp``: full parity would
mean inverse-folding projections into AgentEvents — lossy and drift-prone —
and the projection is already a legitimate render surface (the phone renders
exactly this shape). The bargain is the phone's: a follower view with a
banner that says so.

Two hosting modes share the one screen:

- **in-app** — pushed over a running ``OperatorApp`` by
  ``OperatorApp._resume_session``; ``/detach`` pops back to the user's own
  conversation, whose claim and registrant were never touched.
- **standalone** — a minimal Textual app (``run_attach_app``) for cold-boot
  ``lop --resume <live-id>``, where there is no TUI of the user's own to push
  onto; ``/detach`` exits.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.binding import Binding
from textual.widgets import Input, Static
from local_operator.mobile.attach_client import AttachClient
from local_operator.mobile.types import SessionProjection, SessionRecord, TranscriptEntry

#: Submitted text that starts with this word is a screen command, not a steer.
#: Only one exists (/detach); parsed by prefix so the composer stays a plain
#: Editor with no slash registry of its own — the owner owns slash commands.
DETACH_COMMAND = "/detach"


def _entry_line(entry: TranscriptEntry) -> str:
    """One text line for one transcript row, folded like the phone renders it.

    The projection is pre-folded (one line per tool call, notices as text);
    this only chooses the lead glyph. Kept dumb on purpose: this screen is a
    follower view, not a second implementation of the TUI's render semantics.
    """
    if entry.kind == "user":
        return f"❯ {entry.text}"
    if entry.kind == "steer":
        return f"↪ {entry.text}"
    if entry.kind == "tool":
        label = entry.summary or entry.tool_name or "tool"
        return f"· {label}"
    if entry.kind == "notice":
        return f"· {entry.text}"
    # assistant rows and compaction markers render as their own text
    return entry.text


class _Banner(Static):
    """The one-line 'you are a follower' band segment.

    Dim on purpose (design §3): 'attached' is the load-bearing word, and a
    follower signal that outshone the transcript would invert the frame. On a
    phone-watch-less world this banner is the only persistent reminder that
    submits here steer a conversation whose owner is elsewhere.
    """


class _Transcript(Vertical):
    """Read-only rows rendered from the projection's transcript tail."""

    def render_projection(self, projection: SessionProjection) -> None:
        self.remove_children()
        for entry in projection.transcript:
            text = _entry_line(entry)
            classes = "attach-row"
            if entry.kind == "user":
                classes += " attach-user"
            elif entry.kind == "steer":
                classes += " attach-steer"
            elif entry.kind == "tool":
                classes += " attach-tool"
            elif entry.kind == "notice":
                classes += " attach-notice"
            self.mount(Static(text, classes=classes))
        # The working line: what the turn is doing right now, when streaming.
        if projection.streaming:
            activity = projection.activity or "working"
            self.mount(Static(f"… {activity}", classes="attach-row attach-working"))


class _PendingCard(Static):
    """The front approval/ask gate, answerable with minimal keys.

    V1 keeps multi-question asks to the owner (the phone/owner terminal);
    a single-question answer or a y/n approval is fine from a follower.
    """


class AttachScreen(Screen[None]):  # noqa: D101 — the module docstring is the doc
    CSS = """
    AttachScreen {
        layout: vertical;
    }
    #attach-banner {
        height: 1;
        color: $lo-dim;
    }
    #attach-transcript {
        height: 1fr;
        overflow-y: auto;
    }
    .attach-row {
        height: auto;
        color: $lo-fg;
    }
    .attach-user {
        color: $lo-fg;
        text-style: bold;
    }
    .attach-steer {
        color: $lo-muted;
    }
    .attach-tool {
        color: $lo-muted;
    }
    .attach-notice {
        color: $lo-dim;
    }
    .attach-working {
        color: $lo-dim;
    }
    #attach-pending {
        height: auto;
        color: $lo-warning;
    }
    #attach-composer {
        dock: bottom;
        height: 3;
    }
    #attach-composer Input {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "detach", "Detach", show=False),
    ]

    def __init__(
        self,
        record: SessionRecord,
        session_id: str,
        *,
        on_detached: Optional[Callable[[], None]] = None,
        on_resume_here: Optional[Callable[[str], None]] = None,
        standalone: bool = False,
    ) -> None:
        super().__init__()
        self._record = record
        self._session_id = session_id
        self._standalone = standalone
        self._on_detached = on_detached
        self._on_resume_here = on_resume_here
        self._client: AttachClient | None = None
        self._projection: SessionProjection | None = None
        self._owner_dead = False

    # -- lifecycle ---------------------------------------------------------------

    def on_mount(self) -> None:
        self._banner = _Banner(id="attach-banner")
        self._transcript = _Transcript(id="attach-transcript")
        self._pending = _PendingCard(id="attach-pending")
        self._composer = Input(placeholder="steer the attached session — /detach to leave")
        self._composer.id = "attach-composer"
        self.mount(self._banner, self._transcript, self._pending, self._composer)
        self._set_banner()
        self._client = AttachClient(
            on_projection=self._on_projection,
            on_disconnected=self._on_disconnected,
        )
        # The reader task must live on the app's loop; connect() is async and
        # cheap (loopback), so create_task is right here.
        self.run_task = asyncio.get_event_loop().create_task(self._connect())

    async def _connect(self) -> None:
        try:
            await self._client.connect(self._record, self._session_id)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — any connect failure is owner-death copy
            self._on_disconnected("could not attach")

    def on_unmount(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- socket callbacks (fire on the client's reader task; hop to the UI) -----

    def _on_projection(self, projection: SessionProjection) -> None:
        self._projection = projection
        self.call_from_thread(self._render_projection, projection) if self.is_running else (
            self._render_projection(projection)
        )

    def _on_disconnected(self, reason: str) -> None:
        try:
            self.call_from_thread(self._owner_exited, reason)
        except Exception:  # noqa: BLE001 — screen may already be unmounting
            pass

    # -- rendering ----------------------------------------------------------------

    def _set_banner(self, dead: bool = False) -> None:
        name = (self._projection.conversation_name if self._projection else "") or "untitled"
        pid = self._record.pid
        if dead:
            self._banner.update(f"owner exited (pid {pid}) — r: resume here · esc: detach")
        else:
            self._banner.update(f"attached · pid {pid} · {name} — /detach to release")

    def _render_projection(self, projection: SessionProjection) -> None:
        self._projection = projection
        self._set_banner()
        self._transcript.render_projection(projection)
        pending = projection.pending
        if pending is None:
            self._pending.display = False
        else:
            if pending.kind == "approval":
                body = f"{pending.title} — y/n"
            else:
                opts = " · ".join(o.label for o in pending.options) or "type an answer"
                body = f"{pending.title} — {opts}"
            self._pending.update(body)
            self._pending.display = True

    def _owner_exited(self, reason: str) -> None:
        """Owner death: swap the banner and offer the two exits inline.

        No modal (design §3): the transcript stays readable behind the choice.
        'resume here' re-runs the resume path — by now the claim marker names
        a dead pid, so this process becomes the legitimate FIRST writer.
        """
        if self._owner_dead:
            return
        self._owner_dead = True
        self._set_banner(dead=True)
        self._pending.update("owner exited — r: resume here · esc: detach")
        self._pending.display = True
        self._composer.placeholder = "owner exited — r to resume here, esc to detach"

    # -- input ---------------------------------------------------------------------

    async def on_input_submitted(self, event: Any) -> None:
        text = str(getattr(event, "value", "")).strip()
        if not text:
            return
        if self._composer is not None:
            self._composer.value = ""
        if text.lower() in (DETACH_COMMAND, "esc"):
            self.action_detach()
            return
        if self._owner_dead:
            # Inputs while dead other than the two commands are meaningless;
            # the banner restates the choice rather than silently eating keys.
            self._pending.update("owner exited — r: resume here · esc: detach")
            return
        client = self._client
        if client is None or not client.connected:
            return
        # Route by owner state: prompt when idle, steer when streaming. The
        # alternative — sending prompt and showing the busy error — is what
        # the owned handle does, but from a follower terminal the busy case IS
        # the normal case mid-turn, and reading it as a failure teaches the
        # user the screen is broken when it is working as designed.
        streaming = bool(self._projection and self._projection.streaming)
        try:
            if streaming:
                detail = await client.steer(text)
                self._transcript.mount(
                    Static(f"↪ {text}", classes="attach-row attach-steer")
                )
                self._banner.update(
                    f"attached · pid {self._record.pid} — queued as steering ({detail})"
                )
            else:
                detail = await client.prompt(text)
                self._transcript.mount(
                    Static(f"❯ {text}", classes="attach-row attach-user")
                )
                self._banner.update(
                    f"attached · pid {self._record.pid} — {detail}"
                )
        except (RuntimeError, ConnectionError) as exc:
            self._banner.update(f"attached · pid {self._record.pid} — {exc}")

    def on_key(self, event: Any) -> None:
        """Minimal key answers for the pending gate, plus owner-death actions."""
        key = str(getattr(event, "key", ""))
        if self._owner_dead:
            if key == "r" and self._on_resume_here is not None:
                event.prevent_default()
                event.stop()
                self._on_resume_here(self._session_id)
            return
        pending = self._projection.pending if self._projection else None
        if pending is None or self._composer is None:
            return
        if self._composer.has_focus:
            return  # typing goes to the composer
        client = self._client
        if client is None:
            return

        async def answer() -> None:
            try:
                if pending.kind == "approval":
                    approved = key == "y"
                    await client.approval_answer(pending.request_id, approved)
                elif pending.options:
                    for opt in pending.options:
                        if opt.label.lower().startswith(key.lower()) and key:
                            await client.ask_answer(pending.request_id, opt.label)
                            break
            except (RuntimeError, ConnectionError):
                pass

        if pending.kind == "approval" and key in ("y", "n"):
            event.prevent_default()
            event.stop()
            asyncio.get_event_loop().create_task(answer())
        elif pending.kind == "ask" and pending.options and len(key) == 1:
            event.prevent_default()
            event.stop()
            asyncio.get_event_loop().create_task(answer())

    # -- actions --------------------------------------------------------------------

    def action_detach(self) -> None:
        """Leave the attach: pop (in-app) or exit (standalone)."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
        if self._standalone:
            self.app.exit()
        else:
            self.app.pop_screen()
        if self._on_detached is not None:
            self._on_detached()


class AttachApp(App[None]):
    """The minimal standalone host for cold-boot ``lop --resume <live-id>``.

    Carries the app-level theme variables (``$lo-*``) so the screen renders in
    the product's palette rather than Textual defaults — the same seam
    ``OperatorApp.get_css_variables`` provides, duplicated here because the
    standalone host deliberately does not boot an OperatorApp (no session, no
    workers, no claim)."""

    CSS_PATH = "../local_operator.tcss"

    def __init__(
        self,
        record: SessionRecord,
        session_id: str,
        on_resume_here: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._record = record
        self._session_id = session_id
        self._on_resume_here = on_resume_here

    def get_css_variables(self) -> dict[str, str]:
        from local_operator.tui import theme as theme_mod

        return {
            **theme_mod.tcss_variable_map(theme_mod.current_theme()),
            **super().get_css_variables(),
        }

    def on_mount(self) -> None:
        self.push_screen(
            AttachScreen(
                self._record,
                self._session_id,
                standalone=True,
                on_resume_here=self._resume_here,
            )
        )

    async def _resume_here(self, session_id: str) -> None:
        """Owner died and the user chose to resume: replace this process's
        mission. The standalone host cannot resume in-place (it owns no
        session machinery), so it exits with a sentinel the CLI reads to
        relaunch the real TUI on that transcript."""
        from local_operator.reexec import REEXEC_CODE, plan_reexec, take_plan  # noqa: F401

        # Simplest correct v1: exit 75 (EX_TEMPFAIL) with the id on stdout;
        # the CLI wrapper re-runs itself WITHOUT --resume, which now finds the
        # owner gone and opens normally.
        self.exit(75, return_code=75)


def run_attach_app(record: SessionRecord, session_id: str) -> int:
    """Run the standalone attach app to completion; return a process code.

    Exit 75 means the user chose 'resume here' after owner death — the caller
    (the CLI's owned-resume branch) treats it as 'relaunch without --resume'.
    """
    app = AttachApp(record, session_id)
    app.run()
    code = app.return_code or 0
    return int(code)
