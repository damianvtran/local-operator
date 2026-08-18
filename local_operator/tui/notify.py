"""Desktop notifications — telling a user who is looking somewhere else.

A terminal UI can only be seen by someone looking at it. The whole value of a
long agentic turn is that the user does something else while it runs, which
means the two moments that matter — *the turn is finished* and *the agent is
waiting on you* — happen in a window nobody is watching. The status band says
both, and the window title (``tui/terminal_title.py``) carries them one surface
further out to a tab strip; this module carries them the last step, to the
notification centre of the OS, which is the only surface visible from another
application.

Companion, not a replacement. The title is a persistent, zero-cost *state* — it
is correct at every instant and costs nothing to leave on. A notification is a
one-shot *edge* that interrupts the user, so it is emitted only on transitions
the user actually owes something for, and everything here exists to keep that
promise narrow.

What is delivered
-----------------

Delivery is deliberately layered, because no single mechanism reaches every
terminal:

1. **cmux** (:func:`cmux_command`) — when ``CMUX_SURFACE_ID`` names a concrete
   surface, ``cmux notify --surface`` is used. It is the only path that ties
   the toast to the *pane* that produced it, so clicking it lands the user on
   the right session out of a sidebar of many. This mirrors the fork of omp
   (`packages/tui/src/terminal-capabilities.ts`), and it must win over the
   in-band escapes: cmux hosts Ghostty, so the OSC 9 below would also fire and
   the user would get the same event twice.
2. **In-band OSC** (:func:`notification_sequence`) — OSC 99 for kitty, OSC 9
   for Ghostty/iTerm2/WezTerm/Warp, BEL for everything else. Written through
   Textual's driver like every other escape this app emits.
3. **libnotify** (:func:`desktop_notify_command`) — on Linux, terminals whose
   only protocol is BEL (VTE family, Alacritty, plain xterm) cannot carry toast
   *text* at all, so the body is fanned out over D-Bus via ``notify-send``. The
   BEL is still written, because it is what raises tmux's ``monitor-bell`` flag
   and X11 urgency hints. macOS needs no equivalent: every terminal here either
   speaks OSC 9 natively or is running under cmux.

Multiplexers are handled where they break things: tmux does not forward a bare
OSC 9/99 to the outer terminal, so the sequence is wrapped in tmux's DCS
passthrough envelope and followed by a BEL (which flags the pane even when
``allow-passthrough`` is off). Zellij has no passthrough envelope but does
raise its bell flag, so it gets the OSC plus a BEL.

When it is delivered
--------------------

Three gates, and each one is a real reported failure mode of notification
systems rather than defensiveness:

- **Only a terminal.** Notifications are emitted by the TUI and by nothing
  else. ``local-operator serve`` — the backend behind local-operator-ui — must
  stay silent: the UI owns its own notification surface, and a server that also
  notified would double every alert and would do it on whichever machine the
  *backend* runs on, which is not necessarily where the user is. This module is
  imported only from ``tui/`` for that reason, and :class:`Notifier` still
  refuses to construct without a live driver sink.
- **Only when unfocused** (:attr:`Notifier.set_focused`). A toast for a session
  the user is already staring at is pure interruption; Textual reports focus
  through ``AppFocus``/``AppBlur``, which every terminal here supports.
- **Only for the parent's own edges** — see below.

Whose events count
------------------

The parent agent's completion is the notifiable event. A subagent finishing is
NOT: the parent is still working, the user has nothing to do, and a session
that delegates five children would fire five toasts for one task. Worse, the
parent's *own* ``agent_end`` frequently arrives while children are still
running — the harness's ``task`` tool returns as soon as the job is registered
— so "the model stopped talking" is not the same fact as "the work is done".
:meth:`Notifier.notify_turn_complete` therefore takes the count of running
child jobs and stays quiet while any are alive, leaving the notification to the
completion that follows the last child's delivery (settled jobs re-enter the
conversation as a new turn, see ``Session._on_job_completed``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import uuid
from typing import Callable, Literal, Mapping

from local_operator.tui.settings import settings_get

#: Environment kill switch, mirroring ``LOCAL_OPERATOR_NO_TERMINAL_TITLE`` and
#: the shimmer/nerd-icon gates. Wanted by anything that records raw terminal
#: output (a demo capture, CI, ``script(1)``) and by a user who simply does not
#: want to be interrupted, without editing config.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_NOTIFICATIONS"

#: The app name every backend attributes the toast to. The window title spends
#: its budget on `lo` because a sidebar row clips at ~24 cells; a notification
#: centre has room for the product's real name and is read out of context, so
#: the two deliberately differ.
APP_NAME = "Local Operator"

#: Notification body for a finished turn, and for a turn parked on the user.
#: Fixed strings rather than the model's own words: a notification is delivered
#: by the OS to a surface this app cannot sanitise twice, and the point of the
#: toast is the STATE, which the user reads in under a second. The session name
#: (the title) says which conversation it was.
BODY_COMPLETE = "Task complete"
BODY_APPROVAL = "Waiting for approval"
BODY_ASK = "Waiting for your answer"
BODY_ERROR = "Stopped with an error"

#: Notification kinds. ``complete`` is an edge the user may ignore; the two
#: waiting kinds are edges the turn is BLOCKED on, which is why they are
#: separated — see :func:`urgency_for`.
NotifyKind = Literal["complete", "approval", "ask", "error"]

#: Bodies keyed by kind, so a call site names the event rather than the prose.
BODIES: dict[str, str] = {
    "complete": BODY_COMPLETE,
    "approval": BODY_APPROVAL,
    "ask": BODY_ASK,
    "error": BODY_ERROR,
}

#: BEL. Every terminal ever made honours it; it carries no text, but it is what
#: raises tmux's ``monitor-bell``, Zellij's ``[!]`` flag and X11 urgency hints,
#: which is the only signal a BACKGROUNDED pane can give.
BEL = "\x07"

#: OSC 99 (kitty's desktop-notification protocol) and OSC 9 (iTerm2's, adopted
#: by Ghostty, WezTerm and Warp). Both carry arbitrary text; OSC 99 additionally
#: carries structured metadata, of which we use the fields kitty documents as
#: safe to send unconditionally.
OSC99_PREFIX = "\x1b]99;"
OSC9_PREFIX = "\x1b]9;"

#: String Terminator. Preferred over BEL to close an OSC *notification* (unlike
#: the title's OSC 0, which uses BEL for old-emulator compatibility): kitty's
#: OSC 99 grammar is specified with ST, and every terminal that implements
#: either notification protocol is recent enough to accept it.
ST = "\x1b\\"

#: Which in-band protocol a terminal speaks. Ordered by richness; the resolver
#: returns exactly one.
NotifyProtocol = Literal["osc99", "osc9", "bell"]

#: cmux injects this per surface, and it is the ONLY marker that identifies the
#: pane to notify — ``CMUX_WORKSPACE_ID`` names a workspace of many surfaces and
#: ``CMUX_SOCKET_PATH`` is a CLI override that can be set outside cmux
#: entirely. Validated as a UUID before it reaches an argv (see
#: :func:`cmux_surface_id`).
_CMUX_SURFACE_ENV = "CMUX_SURFACE_ID"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$", re.IGNORECASE)

#: Control characters stripped from any text that reaches an escape sequence.
#: This is a security boundary, exactly as in ``terminal_title``: session names
#: are MODEL-GENERATED, and both BEL and ESC terminate an OSC string — a name
#: containing either would close the sequence early and leave the remainder
#: being executed by the terminal as commands. ESC (0x1b) and BEL (0x07) are
#: inside the swept ranges.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

#: Hard cap on a notification title. No notification centre shows more, and an
#: unbounded model-generated string does not belong on the wire.
MAX_TITLE_CHARS = 80


#: The type every env-reading helper here accepts. ``os.environ`` satisfies it,
#: and so does a plain dict — which is what makes each of these testable
#: without mutating the process's real environment.
EnvMap = Mapping[str, str]


def notifications_enabled() -> bool:
    """Whether notifications may be emitted at all (env gate + config flag).

    Same two-tier shape as ``terminal_title_enabled``/``nerd_icons_enabled``:
    an environment kill switch for a capture or a CI run, and a
    ``display.notifications`` config flag for a persistent preference.
    """
    if os.environ.get(_ENV_DISABLE):
        return False
    return bool(settings_get("display.notifications", True))


def sanitize_text(value: str | None, limit: int = MAX_TITLE_CHARS) -> str:
    """``value`` with control characters removed and whitespace collapsed.

    Shares its contract with ``terminal_title.sanitize_label`` because it
    guards the same wire against the same untrusted source; kept separate
    because the two have different length budgets and this one also feeds
    argv (cmux, notify-send), where the risk is different but the answer is
    the same.
    """
    if not value:
        return ""
    cleaned = " ".join(_CONTROL_CHARS.sub(" ", value).split())
    return cleaned[:limit]


def detect_protocol(env: EnvMap | None = None) -> NotifyProtocol:
    """The richest in-band notification protocol this terminal speaks.

    Detected from the environment markers each emulator injects rather than by
    querying the terminal: a capability query needs a reply read off stdin,
    and stdin belongs to Textual's input loop while the app is running.

    Unknown terminals get ``bell``, which is correct rather than merely safe —
    an unrecognised terminal that silently swallowed OSC 9 would give the user
    nothing at all, whereas a BEL at worst rings.
    """
    source = os.environ if env is None else env
    if source.get("KITTY_WINDOW_ID") or source.get("TERM", "").startswith("xterm-kitty"):
        return "osc99"
    if source.get("GHOSTTY_RESOURCES_DIR") or source.get("GHOSTTY_BIN"):
        return "osc9"
    if source.get("WEZTERM_PANE") or source.get("WEZTERM_EXECUTABLE"):
        return "osc9"
    if source.get("ITERM_SESSION_ID"):
        return "osc9"
    term_program = source.get("TERM_PROGRAM", "").lower()
    if term_program in ("ghostty", "iterm.app", "wezterm", "warpterminal"):
        return "osc9"
    return "bell"


#: freedesktop urgency for every notification this app sends. ``critical`` is
#: reserved by the spec for toasts that must never expire on their own, and on
#: GNOME it produces a card the user has to dismiss by hand — a session waiting
#: on an approval does not warrant hijacking the desktop, and a user who walks
#: away from three sessions would come back to a stack of undismissable cards.
#: ``low`` is filtered out entirely by some daemons, which would silently undo
#: the feature. So: ``normal``, for every kind.
URGENCY = "normal"


def is_inside_tmux(env: EnvMap | None = None) -> bool:
    """Whether this process is inside tmux (read fresh: a session can attach)."""
    source = os.environ if env is None else env
    return bool(source.get("TMUX"))


def is_inside_zellij(env: EnvMap | None = None) -> bool:
    """Whether this process is inside Zellij (read fresh, as for tmux)."""
    source = os.environ if env is None else env
    return bool(source.get("ZELLIJ"))


def wrap_tmux_passthrough(sequence: str) -> str:
    """Wrap ``sequence`` in tmux's DCS passthrough envelope.

    tmux does not forward OSC 9/99 to the outer terminal, so an unwrapped
    notification is simply eaten by the multiplexer. The envelope is
    ``\\x1bPtmux;<escaped>\\x1b\\\\`` and every ESC inside the payload must be
    DOUBLED, or tmux reads the first one as the end of its own DCS.

    Only reaches the outer terminal when the user has ``allow-passthrough on``.
    That is why callers append a BEL as well (see :func:`notification_writes`):
    the BEL is what flags the pane for everyone else.
    """
    return f"\x1bPtmux;{sequence.replace(chr(27), chr(27) * 2)}\x1b\\"


def osc99_id() -> str:
    """A fresh OSC 99 notification id, unique per notification.

    kitty treats the ``i=`` key as an IDENTITY: a notification reusing a live
    id REPLACES the one on screen. A constant id therefore let a later "task
    complete" silently overwrite an unanswered "waiting for approval" — losing
    exactly the toast the user most needed — and made every session on the
    machine share one id, which is the multiplexer-routing collision kitty's
    spec warns about and the opposite of this feature's "five sessions must be
    tellable apart" goal.

    A UUID stem rather than a counter because the uniqueness has to hold ACROSS
    processes, not just within one: several sessions run side by side, and each
    would start its counter at the same place. Restricted to the
    ``[a-zA-Z0-9_+-.]`` set the spec allows.
    """
    return f"lo-{uuid.uuid4().hex[:12]}"


def osc99_sequence(title: str, body: str, notification_id: str | None = None) -> str:
    """A structured OSC 99 notification: one payload for title, one for body.

    kitty's protocol chunks a notification by id: metadata rides on the first
    payload, ``d=0`` holds display until a later chunk arrives, and ``p=body``
    marks the body payload. ``i=`` groups them. We send the fixed metadata
    kitty defines (application name, urgency) and nothing dynamic — the
    interesting variability here is the text, and every field we would
    otherwise vary is already expressed by it. The application name (``f=``) is
    deliberately NOT sent: the spec requires it base64-encoded, and it buys
    nothing here because the title already names the session.

    Sent WITHOUT base64: both fields are sanitised to printable text by
    :func:`sanitize_text`, so there is nothing left that would need escaping,
    and plain payloads stay readable in a terminal capture.
    """
    nid = notification_id or osc99_id()
    if not body:
        return f"{OSC99_PREFIX}i={nid}:u=1;{title}{ST}"
    return f"{OSC99_PREFIX}i={nid}:u=1:d=0;{title}{ST}" f"{OSC99_PREFIX}i={nid}:p=body;{body}{ST}"


def notification_sequence(
    protocol: NotifyProtocol,
    title: str,
    body: str,
    notification_id: str | None = None,
) -> str:
    """The in-band escape for one notification under ``protocol``.

    ``bell`` returns a bare BEL: it cannot carry text, and pretending otherwise
    by writing the words to stdout would paint them over the app's own frame.
    """
    if protocol == "bell":
        return BEL
    if protocol == "osc99":
        # `notification_id` is injectable ONLY so a test can pin the wire; every
        # production caller leaves it None and gets a fresh id per notification.
        return osc99_sequence(title, body, notification_id)
    # OSC 9 has no title/body split — one line is all it takes. The title leads
    # because a notification centre truncates from the right, and which session
    # this is matters more than which of four fixed states it reached.
    line = f"{title}: {body}" if title and body else (title or body)
    return f"{OSC9_PREFIX}{line}{ST}"


def notification_writes(
    protocol: NotifyProtocol,
    title: str,
    body: str,
    *,
    in_tmux: bool = False,
    in_zellij: bool = False,
) -> list[str]:
    """Everything that should be written to the terminal for one notification.

    A list rather than a string so a test can see the *shape* of the delivery,
    and because the multiplexer cases genuinely emit two things: the escape the
    outer terminal may or may not receive, and the BEL that flags the pane
    regardless. Under ``bell`` the sequence IS a BEL, so no second one is added
    — a double ring for one event is a bug users report.
    """
    sequence = notification_sequence(protocol, title, body)
    if protocol == "bell":
        return [sequence]
    if in_tmux:
        # Checked BEFORE Zellij, and correct when both are set (Zellij hosting
        # a tmux session, or the reverse): tmux is the INNER multiplexer that
        # would otherwise eat the OSC, so it is the one needing the envelope,
        # and the BEL appended here is exactly what the Zellij branch would
        # have contributed anyway. Stated rather than left to be rediscovered,
        # because "the second branch is unreachable" reads like a bug.
        return [wrap_tmux_passthrough(sequence), BEL]
    if in_zellij:
        # No passthrough envelope exists for Zellij; it swallows the OSC and
        # raises its own `[!]` flag on the BEL, which is the signal a
        # backgrounded Zellij pane can actually give.
        return [sequence, BEL]
    return [sequence]


def cmux_surface_id(env: EnvMap | None = None) -> str | None:
    """The cmux surface this process belongs to, or ``None``.

    Validated as a UUID before returning: the value reaches an argv, and a
    UUID is the exact shape cmux mints. Workspace/socket markers deliberately
    do not qualify — a workspace holds several surfaces, so a notification sent
    to one would name the wrong pane.
    """
    source = os.environ if env is None else env
    value = (source.get(_CMUX_SURFACE_ENV) or "").strip()
    if not value or not _UUID_RE.match(value):
        return None
    return value


def argv_safe(value: str) -> str:
    """``value`` with any leading dashes neutralised for an argv position.

    Complements :func:`sanitize_text`, which closes the ESCAPE hole but says
    nothing about a string's SHAPE. A conversation name is model-generated, so
    it can begin with ``-``; most argument parsers then read it as an option.
    A leading space is enough to make it unambiguous while leaving the text
    readable, which matters because this is a user-facing toast title.
    """
    return f" {value}" if value.startswith("-") else value


def cmux_command(surface_id: str, title: str, body: str) -> list[str]:
    """argv delivering one notification to a specific cmux surface.

    Pure so a test asserts the exact wire shape without spawning anything.
    """
    return [
        "cmux",
        "notify",
        "--surface",
        surface_id,
        # Each value follows its own flag, so cmux consumes it positionally
        # rather than re-parsing it as an option — but the title is still
        # model-written, so it is passed through `argv_safe` for the same reason
        # the notify-send path uses `--`: a value that begins with a dash is
        # ambiguous to most parsers, and being right here costs one function
        # call. The surface id is already UUID-validated by `cmux_surface_id`.
        "--title",
        argv_safe(title or APP_NAME),
        "--body",
        argv_safe(body),
    ]


def desktop_notify_command(
    notifier: str, title: str, body: str, urgency: str = URGENCY
) -> list[str]:
    """argv for the Linux D-Bus fallback through ``notify-send``.

    Only used where the in-band protocol is BEL — i.e. the VTE family,
    Alacritty and bare xterm, none of which can surface arbitrary toast text.
    ``--expire-time`` keeps the toast transient; the notification is a nudge,
    not a record.
    """
    return [
        notifier,
        "--app-name",
        APP_NAME,
        f"--urgency={urgency}",
        "--expire-time=5000",
        # `--` ends option parsing. The title is derived from a MODEL-WRITTEN
        # conversation name, and sanitisation strips control characters without
        # constraining the string's shape — so a session named `--help` or
        # `-u critical` otherwise lands in notify-send's option namespace
        # instead of its summary, silently breaking delivery or reaching the
        # urgency this module deliberately refuses to send. Not a shell risk
        # (`Popen` without `shell=True`), but model-controlled input reaching an
        # argument parser is the same class of defect.
        "--",
        title or APP_NAME,
        body,
    ]


def should_use_desktop_fallback(
    protocol: NotifyProtocol,
    platform: str,
    env: EnvMap | None = None,
) -> bool:
    """Whether to ALSO fan a notification out over D-Bus.

    True only when every one of these holds: the in-band protocol is BEL (a
    terminal that speaks OSC 9/99 already delivered the toast, and a second one
    would duplicate it), the platform is Linux (macOS terminals all speak OSC 9
    or run under cmux, and there is no session bus), and a session bus is
    actually reachable. Resolving the binary is the caller's job.

    ``platform`` is ``sys.platform``-shaped, so Linux is matched by prefix:
    the value is plain ``linux`` on every supported interpreter but was
    ``linux2`` historically, and a prefix test costs nothing to be right about.
    """
    if protocol != "bell":
        return False
    if not platform.startswith("linux"):
        return False
    source = os.environ if env is None else env
    if source.get("DBUS_SESSION_BUS_ADDRESS"):
        return True
    runtime_dir = source.get("XDG_RUNTIME_DIR")
    return bool(runtime_dir and os.path.exists(os.path.join(runtime_dir, "bus")))


def _spawn_detached(argv: list[str]) -> None:
    """Fire-and-forget a notifier process; never raise, never block.

    ``start_new_session`` plus fully redirected stdio is what keeps a slow or
    hung notifier (a stalled D-Bus activation, a cmux socket mid-restart) from
    holding the TUI's event loop or writing bytes into the frame. The child is
    never waited on: it is a side effect, and its exit status tells this app
    nothing it can act on.
    """
    try:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        # Best-effort by design: a missing binary or a spawn failure must never
        # surface as an error in a session, because the user asked for a task
        # and not for a toast.
        pass


class Notifier:
    """Decides whether an event is notifiable, then delivers it.

    Constructed by the app with a ``write`` sink — ``driver.write``, the same
    door ``TerminalTitle`` and Textual's own OSC 52 use, and for the same
    reason: Textual serialises output through a writer thread, and a second
    writer interleaves escape bytes into the middle of a painted frame.

    Holds the policy the module docstring describes, so call sites state facts
    ("the turn ended, N children are running") rather than re-deriving whether
    a toast is warranted. Two pieces of state make that possible:

    - ``focused`` — whether the terminal has OS focus, fed from Textual's
      ``AppFocus``/``AppBlur``. Notifying a session the user is watching is
      pure interruption.
    - ``label`` — the conversation name, so a toast names the session it came
      from. A user with five sessions open gets five identically-titled
      notifications otherwise, which is the same failure the window title was
      built to fix.
    """

    def __init__(
        self,
        write: Callable[[str], None],
        *,
        enabled: bool = True,
        env: EnvMap | None = None,
        platform: str | None = None,
    ) -> None:
        self._write = write
        #: The environment every delivery decision is read from. Defaults to
        #: ``os.environ`` ITSELF rather than a copy of it, which keeps two
        #: properties that would otherwise conflict: a test injects a plain
        #: dict and gets a deterministic notifier regardless of the terminal
        #: the suite happens to run in (this suite runs inside cmux, whose
        #: `CMUX_SURFACE_ID` would otherwise capture every delivery), while
        #: production still reads FRESH — tmux and Zellij sessions can be
        #: attached and detached under a running app.
        self._env: EnvMap = os.environ if env is None else env
        self._platform = sys.platform if platform is None else platform
        #: A disabled notifier still accepts every state change and simply
        #: never delivers — a null object rather than an ``Optional`` at each
        #: call site, matching ``TerminalTitle``.
        self._enabled = enabled
        self._label = ""
        #: Start focused. The app is launched from the terminal the user is
        #: typing in, and Textual only reports focus on a CHANGE — assuming
        #: unfocused would notify on the first turn of every session.
        self._focused = True
        #: Annotated rather than inferred: the attribute is mutable, so type
        #: inference widens the literal this returns to plain ``str``.
        #:
        #: Resolved ONCE, unlike the multiplexer checks below: a terminal
        #: emulator cannot change under a running process, and re-deriving it
        #: per notification would only add a way for two toasts in one session
        #: to disagree.
        self._protocol: NotifyProtocol = detect_protocol(self._env)

    @property
    def enabled(self) -> bool:
        """Whether this instance delivers anything at all."""
        return self._enabled

    @property
    def protocol(self) -> NotifyProtocol:
        """The in-band protocol resolved for this terminal (tests/diagnostics)."""
        return self._protocol

    @property
    def focused(self) -> bool:
        """Whether the terminal currently has OS focus."""
        return self._focused

    def set_label(self, label: str) -> None:
        """Name the session a toast comes from (``""`` falls back to the brand)."""
        self._label = sanitize_text(label)

    def set_focused(self, focused: bool) -> None:
        """Record terminal focus, from ``AppFocus``/``AppBlur``."""
        self._focused = focused

    def notify_turn_complete(self, *, running_children: int) -> bool:
        """The parent finished a turn. Returns whether a toast was delivered.

        ``running_children`` is the count of live ``task`` jobs, and it is the
        whole reason this method takes an argument. The harness's ``task`` tool
        returns as soon as a child is registered, so a parent that delegates
        reaches ``agent_end`` with its children still working: the model has
        stopped talking, but the WORK the user asked for is not done, and a
        toast then is a false finish. Staying quiet costs nothing, because each
        settled job re-enters the conversation as a fresh turn
        (``Session._on_job_completed``) whose own completion is notifiable —
        so the user is told once, when the last child has landed.

        Deliberately counts children rather than asking whether any single
        child finished: a child's own completion is never notifiable on its
        own. It is an implementation detail of the parent's task, the user has
        nothing to do about it, and one delegating turn would otherwise fire a
        toast per child.
        """
        if running_children > 0:
            return False
        return self.send("complete")

    def notify_waiting(self, kind: Literal["approval", "ask"]) -> bool:
        """The turn is parked on the user. Returns whether a toast was delivered.

        Unconditional on subagents, unlike completion: an unanswered approval
        blocks THIS turn no matter what else is running, and it is the case
        where a missed notification costs the most — the session sits parked
        indefinitely, which is exactly how an agent run gets abandoned.
        """
        return self.send(kind)

    def notify_error(self) -> bool:
        """The turn stopped with an error. Returns whether a toast was delivered."""
        return self.send("error")

    def send(self, kind: NotifyKind) -> bool:
        """Deliver one notification of ``kind``; return whether anything was sent.

        The gates, in order of cheapness: disabled, focused (nothing to tell a
        user who is looking at the session), then delivery. cmux is checked
        before the in-band write because cmux hosts Ghostty — both paths would
        otherwise fire and the user would be told twice about one event.
        """
        if not self._enabled:
            return False
        if self._focused:
            return False
        title = self._label or APP_NAME
        body = BODIES.get(kind, BODY_COMPLETE)

        surface = cmux_surface_id(self._env)
        if surface is not None:
            _spawn_detached(cmux_command(surface, title, body))
            return True

        for chunk in notification_writes(
            self._protocol,
            title,
            body,
            in_tmux=is_inside_tmux(self._env),
            in_zellij=is_inside_zellij(self._env),
        ):
            self._write(chunk)

        if should_use_desktop_fallback(self._protocol, self._platform, self._env):
            notifier = shutil.which("notify-send")
            if notifier:
                _spawn_detached(desktop_notify_command(notifier, title, body, URGENCY))
        return True
