"""Viewer discovery: which PROCESS is displaying which session, right now.

A session record (:mod:`local_operator.session.runtime.registry`) answers "where
does this transcript live". This module answers a different question that
nothing could answer before: **"which running window can put session X on
screen"**. They are not the same question, and conflating them is what this
module exists to avoid.

WHY IT IS NEEDED. A desktop notification's click used to spawn a brand-new
terminal, because :mod:`local_operator.tui.resume_click` was written when a
notification was only ever posted with nothing watching. PR #724 broke that
premise: a live TUI now posts a toast for a *background* session it can display
through its own sidebar. The click had no way to find that TUI — so it opened a
second window running a second process for a session already one keystroke
away. The operator hit this and reported it; an orphaned Ghostty from exactly
that path was still resident when this was written.

WHY IT IS A SEPARATE DIRECTORY, AND NOT A ``kind="viewer"`` SESSION RECORD.
This was the design's first proposal and it is unshippable, for three reasons
worth recording so nobody re-proposes it:

1. **The path collides.** ``registry.record_path`` is ``run/mobile/<pid>.json``,
   keyed by pid. A TUI that owns a local session already writes that exact file.
   A viewer record from the same process IS that file; one silently overwrites
   the other.
2. **Old binaries cannot be taught to ignore it.** ``SessionRecord.from_json``
   drops unknown *keys*; it does not validate ``kind``'s *value*. So a process
   running a previous build parses ``kind="viewer"`` as an ordinary session
   record. It would then be a target for ``lop stop --all``'s SIGTERM ladder
   (``session/runtime/control.py::_stop_targets``), a routable peer for ``lop
   send`` (``mobile/peer_send.resolve_peer_target``), and a session row on the
   phone. A change that makes an OLDER peer misbehave is not additive, and this
   host routinely runs a dozen mixed-version sessions at once.
3. **There is no single scan site to filter.** ``registry.scan`` has around a
   dozen non-test callers; a rule enforced in twelve places is a rule that will
   be broken in the thirteenth.

A separate directory has none of those properties: an old binary globs
``run/mobile/*.json`` and never sees these files at all. The precedent is
already in the tree — ``browser_bridge/state.py`` keeps its own discovery file
under ``run/browser`` for the same isolation reason.

Stdlib-only and import-light by the same contract as ``registry``: the viewer
publishes on the TUI startup path, and ``tests/unit/test_import_graph.py`` pins
what that path may pull in.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir
from local_operator.session.runtime.registry import pid_alive

#: Deliberately NOT ``run/mobile``. See the module docstring: sharing that
#: directory would put viewer records in front of every older binary's
#: ``registry.scan``, which treats whatever it finds as a stoppable, routable
#: session. Treat this literal as a wire constant, exactly as ``RUN_DIRNAME``
#: is treated — two builds coexist in running processes and there is no
#: migration that avoids it.
VIEWER_RUN_DIRNAME = "run/viewers"

#: Version of the *viewer* control surface, independent of the session
#: runtime's ``PROTOCOL_VERSION`` (which stays at 5; this adds no session
#: frame). A viewer client refuses a record whose ``protocol`` it does not
#: know rather than dialing a socket whose frames it cannot predict — see
#: ``KNOWN_VIEWER_PROTOCOLS``, which is where that refusal is implemented.
VIEWER_PROTOCOL = 1

#: Every viewer protocol this build can hold a conversation in. ``choose_viewer``
#: skips a record outside this set, so a FUTURE viewer publishing ``protocol=2``
#: is routed past rather than dialed with frames it may not understand — the
#: click then falls through to the spawn, which works against any build.
#:
#: This exists as a set rather than a ``<=`` comparison because the useful
#: question is "can I speak this", not "is it older than me": a build that drops
#: support for a retired version answers correctly by removing it from here, and
#: nothing else has to change. The comment above used to describe this check
#: while no code performed it; a stale docstring on this exact path is what
#: shipped the bug this module was written to fix, so the prose is now backed by
#: an implementation rather than the other way round.
KNOWN_VIEWER_PROTOCOLS = frozenset({VIEWER_PROTOCOL})

#: Advertised by a viewer whose host can activate its own window, and CONSULTED
#: by the client before it spends a round trip asking. Lives here, beside the
#: record that carries it, so neither half has to import the other's module to
#: agree on the spelling.
FOCUS_WINDOW_CAPABILITY = "focus-window-v1"

#: THE OUTER BOUND ON ONE ``resume_session``, owned here because BOTH halves
#: must agree on it and they live in different processes.
#:
#: The viewer server bounds its whole hop-into-the-app-and-wait-for-the-switch
#: with this (``tui/app.py::OperatorApp.viewer_resume_session``); the client
#: derives its own ack deadline from it below. The ordering is the invariant:
#: **the client must never give up on a switch the server is still going to
#: land.** When it did, the click produced BOTH outcomes — the running TUI
#: switched sessions *and* a second terminal window opened for the same
#: session, which is precisely the bug this feature exists to remove. That was
#: not hypothetical: with an 8.0 s client bound against a 10.0 s server bound,
#: a ``/resume`` measured at 8.5 s reproduced the duplicate window every time.
#:
#: Expressing the relationship as arithmetic rather than as two hand-maintained
#: literals two files apart is deliberate. A comment asking a future editor to
#: check the other side is a rule that gets broken; a derived constant cannot be
#: inverted without deleting the derivation.
VIEWER_RESUME_TIMEOUT_S = 10.0

#: How much longer than the server's bound the client waits before concluding
#: that no ack is coming. It only has to cover the socket round trip and
#: scheduling on a loaded host — the server has already given up by then, so
#: this is slack, not patience.
VIEWER_ACK_GRACE_S = 5.0

#: How long the server may spend STOPPING a switch that overran the bound
#: above, before it answers the click.
#:
#: The bound alone never made "never both" true, it only moved the delay at
#: which both happened: ``wait_for`` cancels the waiter, not the navigation, so
#: a switch slower than ``VIEWER_RESUME_TIMEOUT_S`` still committed after the
#: endpoint had already answered failure — the caller spawned a window AND the
#: session switched. Measured on the derived bounds: clean at 9.8 s, both
#: outcomes at 10.2 s and every value above it.
#:
#: So the timeout now cancels the navigation and reads what is on screen once
#: it has stopped, and this is the budget for that. It is spent INSIDE the
#: client's grace — the server has to answer before the client stops
#: listening — which is why it is a fraction of the grace rather than a third
#: literal: the ordering ``RESUME + ABANDON < RESUME + GRACE`` is arithmetic
#: that cannot be edited into an inversion on one side, and the remaining half
#: stays with the socket round trip the grace was sized for.
VIEWER_ABANDON_SETTLE_S = VIEWER_ACK_GRACE_S / 2

#: Same cadence and grace as the session runtime's, so "is this alive" reads
#: the same way in both places and an operator debugging one has learned the
#: other. Divergence here would be a second thing to remember for no gain.
VIEWER_HEARTBEAT_INTERVAL_S = 15.0
VIEWER_HEARTBEAT_TIMEOUT_S = 45.0


@dataclass
class ViewerRecord:
    """One record per PROCESS that can put a session on screen.

    Keyed by pid like a session record, and for the same reason: a process
    hosts one viewer, so the pid is the natural uniqueness token and a
    ``kill -9`` leaves exactly one stale file to reap.

    ``control_key`` carries the whole authorization story, again as the session
    record does: the file is 0600 under a 0700 directory, so anything that can
    read the key is already the owning account.

    **``current_session`` is the only session id here, deliberately.** The
    design this implements had the record enumerate every session the viewer
    could display. A sidebar TUI can display anything in the catalogue, so that
    list would be large and would be rewritten on every 2 s catalogue poll —
    write amplification of precisely the kind that makes the sidebar laggy in
    the first place. ``can_switch`` says "this viewer can be told to display
    something else", which is the fact routing actually needs; the id it happens
    to be showing is the separate fact that lets a click skip a redundant
    switch.
    """

    pid: int
    #: What kind of surface this is. Only ``"tui"`` exists today; the field is
    #: here because the Electron desktop app is a viewer with a strictly better
    #: focus ceiling and will want to publish one of these unchanged.
    surface: str
    control_port: int
    control_key: str
    #: The session on screen RIGHT NOW, or ``""`` while the app is still
    #: booting and has bound nothing. A click for this id needs no switch —
    #: only focus — which is both the cheapest outcome and the one that does
    #: not disturb the user's scroll position.
    current_session: str = ""
    #: Whether this viewer honours ``resume_session``. False for a surface that
    #: can only be focused, so resolution can prefer one that can actually take
    #: the session rather than dialing and being refused.
    #:
    #: **No producer in this repository writes ``False`` today** — every viewer
    #: is a TUI and every TUI can switch. It is published anyway because the
    #: Electron desktop app is expected to be a focus-only viewer at first, and
    #: a routing field that appears later cannot be read by builds already
    #: running. ``choose_viewer`` honours it now so the older client does the
    #: right thing on the day something publishes ``False``.
    can_switch: bool = True
    #: When this viewer last held OS focus (``time.time()``). The best
    #: available proxy for "the window the user expects to land in" when
    #: several viewers could take the same session. 0.0 means never focused.
    focused_at: float = 0.0
    protocol: int = VIEWER_PROTOCOL
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
    #: Advertised capability strings, ``hasattr``-gated by the publisher exactly
    #: as ``SessionRecord.capabilities`` is, and READ by the client before it
    #: asks for an optional op (``deliver_click`` skips ``focus_window`` when
    #: ``FOCUS_WINDOW_CAPABILITY`` is absent).
    #:
    #: A viewer that cannot activate its window must therefore not advertise
    #: that it can: the client would spend a round trip and its focus timeout on
    #: a request that can only be refused. The reverse under-advertisement is
    #: harmless — focus is chrome, and the switch has already happened.
    capabilities: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ViewerRecord":
        # Tolerate unknown keys for the same forward-compatibility reason
        # SessionRecord does: a newer viewer's record must be readable by an
        # older client, which is what lets the two coexist mid-upgrade.
        known = set(ViewerRecord.__dataclass_fields__)
        return ViewerRecord(**{k: v for k, v in data.items() if k in known})


def viewer_run_dir(root: Path | None = None) -> Path:
    """The viewer record directory, created 0700 on first use.

    Creating is the WRITER's business, but readers reach it too and a missing
    directory is an ordinary answer (no viewer has ever run here), so this
    stays mkdir-on-read like ``registry.run_dir`` rather than raising.
    """
    path = (root or config_dir()) / VIEWER_RUN_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def viewer_record_path(pid: int, root: Path | None = None) -> Path:
    """Where one viewer's record lives."""
    return viewer_run_dir(root) / f"{pid}.json"


def publish_viewer(record: ViewerRecord, root: Path | None = None) -> Path:
    """Write (or refresh) a viewer record, staged so a scanner reads either the
    old file or the new one and never a half-written one."""
    directory = viewer_run_dir(root)
    record.heartbeat_at = time.time()
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{record.pid}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record.to_json(), handle)
        os.chmod(tmp, 0o600)
        target = directory / f"{record.pid}.json"
        os.replace(tmp, target)
        return target
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def unpublish_viewer(pid: int, root: Path | None = None) -> None:
    """Remove a viewer's record on clean exit. Best-effort by contract: an exit
    path must never raise over a missing file."""
    try:
        viewer_record_path(pid, root).unlink()
    except OSError:
        pass


def scan_viewers(root: Path | None = None) -> list[ViewerRecord]:
    """Every LIVE viewer, freshest focus first, reaping what is dead.

    Only live records are returned — unlike ``registry.scan``, which reports
    ``wedged`` because a user needs to *see* a wedged session in ``lop
    sessions``. Nothing here is user-facing: a wedged viewer is one that cannot
    answer a dial, and reporting it would only make the click try a socket that
    will time out before falling back to the spawn it should have gone to
    directly. So a stale heartbeat is filtered out rather than surfaced.

    Ordering is the routing preference, applied once here so every caller
    resolves identically: most recently focused first (the window the user was
    last in is where they expect to land), then lowest pid as a deterministic
    tiebreak so two never-focused viewers do not alternate between scans.
    """
    directory = viewer_run_dir(root)
    now = time.time()
    out: list[ViewerRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = ViewerRecord.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            # The only writer is the staged write above, so an unparseable file
            # means an interrupted crash rather than a format worth preserving.
            try:
                path.unlink()
            except OSError:
                pass
            continue
        if not pid_alive(record.pid):
            try:
                path.unlink()
            except OSError:
                pass
            continue
        if now - record.heartbeat_at > VIEWER_HEARTBEAT_TIMEOUT_S:
            # Alive but quiet: the process is wedged or stopped. Leave the file
            # (its owner may recover and resume beating) but do not offer it as
            # a routing target.
            continue
        out.append(record)
    out.sort(key=lambda rec: (-rec.focused_at, rec.pid))
    return out
