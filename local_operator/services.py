"""Every non-runtime service on this machine, moved onto the build ``current`` names.

WHAT THIS IS FOR
----------------
``lop update`` has always replaced the install and left the processes alone. The
supervised ones (mobile, browser bridge, tunnel, wakes) were repaired afterwards
by :func:`local_operator.update.refresh_daemons_after_upgrade`, but a ``lop
serve`` daemon — the process a desktop app actually talks to — kept serving the
build it had loaded, indefinitely and by design: :mod:`local_operator.server
.retire` refuses to exit on a build change because a marker proves neither a
ready successor nor that scheduler-owned work can stop.

The result was a machine whose INSTALL was current and whose BACKEND was not, and
an app whose update button reported exactly that honestly and then had nothing to
offer: it had no way to move a server it did not start, and no way to make the
install it *did* move become the one being served. This module is the missing
step, and :mod:`local_operator.server.reload` is the mechanism it drives — the
daemon replaces its own image while keeping its pid and its listening socket, so
nothing has to own, kill or respawn it.

WHAT IS AND IS NOT A SERVICE
----------------------------
A SERVICE is a long-lived non-conversational process: a ``serve`` daemon, the
mobile daemon, the browser bridge, the tunnel, the wakes supervisor. A RUNTIME
is a conversation (``session/runtime/process.py``), and this module never touches
one: runtimes are detached, own their transcript leases, and converge on the new
build by themselves when they next go idle (``buildwatch`` + ``launch
._spawn_interpreter``). Restarting a runtime would drop the turn it is running;
restarting the daemons around it does not.

ORDER IS LOAD-BEARING, AND IT IS THE SAME ONE ``lop update`` USES
-----------------------------------------------------------------
The supervised daemons are repaired FIRST and by a CHILD running the NEW build,
because the repair RENDERS plists: run in-process from a superseded interpreter
it would write the previous build's plist shape and report success (see
``update._post_upgrade_invocation``). The ``serve`` reloads follow, and they are
requests rather than actions — a daemon that is mid-spawn refuses and says so.
"""

from __future__ import annotations

import http.client
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from local_operator import procname
from local_operator.procstate import pid_liveness
from local_operator.server import registry as serve_registry
from local_operator.server import reload as serve_reload

logger = logging.getLogger("local_operator.services")

#: How long one daemon has to come back on the new build before it is reported.
#:
#: Sized against the phases it covers, MEASURED rather than assumed (serve-reload
#: review round 2, R2-2: the first version was 30 s while the daemon's own budgets
#: summed to 40 s, so the caller could give up while the daemon was still
#: working and report a failure for a reload that then succeeded):
#:
#:   drain   up to ``reload.DRAIN_BUDGET_S``  = 10 s   (asked TWICE — see below)
#:   smoke   up to ``reload.SMOKE_TIMEOUT_S`` = 10 s
#:   start   interpreter + record publish     = ~1.5-2 s on the reporting host
#:
#: THE DRAIN IS PAID TWICE (serve-reload review round 4, R4-2: this and the reload's own
#: comment both described one drain, which invites a future smoke bump that
#: silently overruns the caller — a reload is drained, smoke-checked, and drained
#: AGAIN, because the smoke is off the loop and a spawn can arrive during it).
#: Worst honest sum ≈ 32 s against 45 s. A change to either budget has to re-do
#: that arithmetic rather than assume this number still covers it.
#:
#: 45 s is a margin over that sum rather than a restatement of it, and it is the
#: CALLER's patience, not a phase's deadline: an expiry is a warning on a
#: successful update, never a kill.
RELOAD_WAIT_S = 45.0

#: The poll period while waiting for a daemon's replacement to publish.
RELOAD_POLL_S = 0.2

#: How long the identity probe before a signal is given.
#:
#: One loopback round trip to a daemon that is answering right now; the timeout is
#: the same order as ``update._mobile_healthz_answers``'s, for the same reason —
#: the answer is either immediate or it is not an answer.
HEALTH_TIMEOUT_S = 1.0


#: The ADDRESS axis, which the shared ``live``/``wedged``/``stale`` vocabulary
#: cannot express.
#:
#: The shared classifier answers "is the OWNER there" from the record alone, with
#: no network cost, and that two-fact contract is what keeps ``registry.scan``
#: cheap enough to run on every ``lop`` invocation. It therefore cannot say the
#: one thing that mattered on 2026-09-23: whether anything is SERVING the address
#: the record names. A serve daemon whose loop is alive and whose heartbeat is
#: fresh while nothing accepts on its port reads as ``live`` — which is exactly
#: the state that cost this machine twelve minutes, because the desktop app's
#: spawn gate believes a live record and the operator was told nothing.
#:
#: So the axis is added HERE, beside the identity probe that already owns "prove
#: the process before touching it", and never in the shared classifier. The
#: tokens below are the whole vocabulary of the axis and are rendered from these
#: constants by every surface that prints them — a second spelling of "not
#: serving" is the defect these exist to prevent.
SERVING = "serving"

#: pid alive, heartbeat fresh, and the identity probe did not answer.
#:
#: THE INCIDENT'S STATE, and it had no name anywhere in the product. It is NOT
#: ``wedged``: a wedged owner stopped reporting, and the honest reading there is
#: "a long turn or a starved loop"; this owner is still reporting and its port is
#: still silent, which is a different fact with a different remedy.
DEAF = "deaf"

#: Something else answered at the record's address. Distinct from ``deaf``
#: because the incident passed through both and the operator needs to know which:
#: ``deaf`` is "nobody is there", this is "somebody else is", and only this one
#: names another daemon to go and look at.
SQUATTED = "squatted"

#: The shared tokens, re-exported as names so a caller that composes a sentence
#: per verdict imports one vocabulary rather than spelling strings twice.
WEDGED = "wedged"
STALE = "stale"

#: How many consecutive probes must agree before ``deaf``/``squatted`` may
#: authorize a SIGNAL (never before they may be REPORTED).
#:
#: A single refused connection is not evidence that a daemon is down: the loop
#: may be mid-restart, the probe may have lost a race with a reload's exec, or
#: the machine may simply be starved (this host ran at a load average of 130 for
#: hours). Acting on one reading is how a healthy daemon gets killed by the
#: command that exists to clean up the unhealthy ones, so the signal path asks
#: three times and an unbroken run of agreement is required.
PROBE_CONFIRMATIONS = 3

#: The gap between those confirmations. Small enough that a refusal stays a
#: command's worth of latency, large enough that two readings are two readings.
PROBE_CONFIRM_GAP_S = 0.25


@dataclass(frozen=True)
class AddressProbe:
    """One address's reading, before any classification is applied to it.

    ``detail`` is the sentence fragment that says WHY, and it is empty exactly
    when the verdict is :data:`SERVING` — the shape :func:`_answers_as_record`
    has always had, kept so its callers did not have to learn a new one.
    """

    verdict: str
    detail: str = ""
    #: The instance id that answered, when one did and it was not ours.
    answered_as: str = ""


@dataclass(frozen=True)
class ServeDaemonReport:
    """One serve record as a reader needs it: its shared state, plus its address.

    ``verdict`` is the COMPOSED answer, and the probe outranks the beat wherever a
    probe ran: ``serving`` and ``squatted`` come from the address, ``deaf`` is a
    ``live`` record the address could not reach, and ``wedged``/``stale`` keep the
    shared classifier's word when the address agrees nobody is serving. Composing it
    in one place is what keeps ``lop services status`` and ``lop services reclaim``
    from disagreeing about what they are looking at.
    """

    record: Any
    state: str
    probe: AddressProbe | None = None

    @property
    def verdict(self) -> str:
        """The one word for this daemon's address.

        THE PROBE OUTRANKS THE BEAT, and that is a composition rule rather than a
        preference. ``wedged`` says the owner stopped REPORTING — a long turn, a
        starved loop — and the shared classifier is explicit that this is not a
        verdict on the process (``registry``: a stale beat is not proof the
        workload stopped). So a wedged record whose address answers as its own
        instance IS serving: the strongest evidence there is, one loopback round
        trip of it.

        Reading the shared state first here made ``reclaim``'s only guard against
        ending a working plane unreachable for every state but ``live`` — a wedged
        record with 0 address probes and no confirmation was signalled while its
        receipt asserted "it is not serving that address" (review round 1, R1-1).
        """
        if self.probe is None:
            # Nothing was asked (a ``stale`` pid has no address to ask), so the
            # shared classification is the whole story.
            return self.state
        if self.probe.verdict in (SERVING, SQUATTED):
            # Both are answers the probe OWNS: nothing else can overrule "this
            # address is served by X" or "by somebody who is not X".
            return self.probe.verdict
        # The probe found nobody answering. ``deaf`` is then exactly the state a
        # ``live`` record names — the incident's state, and the reason this axis
        # exists; a wedged or stale record already has a truer, more specific word.
        return DEAF if self.state == "live" else self.state

    def __getattr__(self, name: str) -> Any:
        # The record's own fields are read through the report (``record.pid``,
        # ``record.host``, …) because every renderer wants them; ``__getattr__``
        # keeps that from becoming a copy of the record's field list here.
        return getattr(self.record, name)


def probe_address(record: Any) -> AddressProbe:
    """Ask the record's address who is serving it — ONE loopback round trip.

    Structured rather than sentence-shaped so the address axis can be composed
    with the shared classification (see :class:`ServeDaemonReport`); the
    sentence every existing caller wants is still produced by
    :func:`_answers_as_record`, which is now a renderer over this.

    FAIL-CLOSED: an unreadable probe is :data:`DEAF`, never :data:`SERVING`.
    """
    import urllib.error
    import urllib.request

    # AN IPv6 LITERAL MUST BE BRACKETED, and the first version of this line was
    # not: `f"http://{[record.host]}:…"` renders the list `['::1']`, so every
    # probe of a v6 daemon asked a URL that cannot parse and the daemon was
    # reported as "did not identify itself" while answering perfectly. Found by
    # the test the review asked for (serve-reload review round 2, R2-6), which is the
    # whole reason it was asked for.
    authority = f"[{record.host}]" if ":" in record.host else record.host
    url = f"http://{authority}:{record.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_S) as response:
            import json

            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # SOMETHING ANSWERED, and that is the whole difference between this arm
        # and the one below. ``HTTPError`` is a response — a 500 from an orphaned
        # daemon, a 403 from somebody else's server — and reporting it as "did not
        # identify itself" would tell the operator nobody is there while a
        # process they can see in `lsof` is answering (measured on 2026-09-23: a
        # stray rig answered 500 on 8080 and nothing in the product named it).
        return AddressProbe(
            SQUATTED,
            f"{record.host}:{record.port} answered {exc.code} — something is there "
            "that is not this record's daemon",
        )
    except (OSError, urllib.error.URLError) as exc:
        # NO ANSWER AT ALL — refused, timed out, or the connection died before a
        # response (``RemoteDisconnected`` is both a ``ConnectionResetError`` and a
        # ``BadStatusLine``, and it lands here because nothing was answered). This is
        # the incident's ``deaf``, and the ONLY arm that may say "nothing is
        # answering there".
        return AddressProbe(DEAF, f"{record.host}:{record.port} did not answer ({exc})")
    except http.client.HTTPException as exc:
        # AN ANSWER ARRIVED AND WAS UNINTELLIGIBLE (``BadStatusLine``,
        # ``IncompleteRead``) — so this is SQUATTED, not DEAF. The difference is
        # what the operator does next, and the first version of this fix filed a
        # TALKING port under "nothing is answering there": the same defect class as
        # R1-3, one bucket over, and catching the exception is what made it
        # printable (review round 2, R2-3).
        return AddressProbe(
            SQUATTED,
            f"{record.host}:{record.port} answered with something that is not this "
            f"product's health endpoint ({exc})",
        )
    except ValueError as exc:
        # Bytes arrived and could not be read as JSON: an answer, not a silence, for
        # the reason above.
        return AddressProbe(
            SQUATTED,
            f"{record.host}:{record.port} answered with a body that is not a health "
            f"report ({exc})",
        )
    if not isinstance(payload, dict):  # pragma: no cover - a non-object body
        return AddressProbe(
            SQUATTED,
            f"{record.host}:{record.port} answered with something that is not a health report",
        )
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    served = result.get("instance_id") if isinstance(result, dict) else None
    if not served:
        return AddressProbe(
            SQUATTED,
            f"{record.host}:{record.port} answered as no instance at all",
        )
    if served != record.instance_id:
        return AddressProbe(
            SQUATTED,
            f"{record.host}:{record.port} is answering as {served}, "
            f"not the {record.instance_id[:8]}… this record names",
            answered_as=str(served),
        )
    return AddressProbe(SERVING)


def _answers_as_record(record: Any) -> str | None:
    """Why this record must NOT be trusted, or ``None`` when it is proven.

    THE RECORD IS NOT PROOF OF WHO IS LISTENING (serve-reload review round 1, R1-4). It is a
    file whose name is a pid, and a pid is recycled: review constructed a live
    ``sleep`` named by a hand-written live, ``reloadable: true`` record, and
    ``reload_serve_daemons`` signalled it — rc ``-30``, a process killed by a
    command whose entire purpose is to NOT kill anything. The ``/health``
    ``instance_id`` is the one value that ties the address to the process, which
    is the check daemon discovery already performs for the same reason
    (``server/registry``'s module docstring: "a 200 alone is not
    identification").

    FAIL-CLOSED, and asymmetrically so against the rule that governs the
    DAEMON's own probes: there, an unreadable probe means "stay" because leaving
    is the irreversible act. Here the irreversible act is SIGNALLING somebody
    else's process, so an unreadable answer means "do not signal" — the daemon
    keeps serving and the operator is told the address did not identify itself.
    """
    return probe_address(record).detail or None


@dataclass(frozen=True)
class ServiceRefresh:
    """One service's outcome, in the shape the CLI already prints.

    Deliberately the same three buckets ``update.DaemonRefresh`` uses — ``lines``
    for what happened, ``warnings`` for what did not and what to do about it —
    because the two lists are printed by one printer and a second vocabulary
    would make an operator read two formats for one command's output.
    """

    name: str
    lines: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ServeOutcome:
    """Internal: what happened to one daemon, before it is worded for a reader."""

    record: Any
    moved: bool = False
    skipped: bool = False
    warning: str = ""
    state: dict[str, Any] = field(default_factory=dict)


def live_serve_daemons() -> list[Any]:
    """Every LIVE ``serve`` record on this machine, in pid order.

    ``live`` only — the shared registry's classification, which is the same one
    every other reader acts on: a record whose pid is gone has already been reaped
    by ``scan``, and a ``wedged`` daemon (pid alive, heartbeat stopped) is not one
    to signal, because a stuck process is not a process that will handle SIGUSR1.

    Sorted by pid so the output is stable run to run: an operator comparing two
    invocations should not have to diff a shuffled list.
    """
    found = []
    for record, state in serve_registry.scan():
        if state == "live":
            found.append(record)
    return sorted(found, key=lambda record: record.pid)


#: The spawn contract of a ``lop serve`` daemon, matched as a WORD SEQUENCE.
#:
#: The same rule :func:`local_operator.session.runtime.reclaim.parse_process_row`
#: states for runtimes — the match is the spawn contract, never a substring —
#: because the census this proves against is read from the process table, where a
#: person running ``grep 'local_operator.cli serve'`` is otherwise
#: indistinguishable from the daemon. A stray serve daemon is exactly what the
#: operator ends up holding when this command is needed, and ending a stranger is
#: worse than reclaiming nothing (this module's recoverability rule).
SERVE_SPAWN_MARKER = ("-m", "local_operator.cli", "serve")

#: The other spelling: a daemon whose process IS the ``lop`` launcher.
#:
#: ``lop serve`` normally re-execs into the branded interpreter (``procname``), so
#: the module marker above is what a live daemon shows. A process that never made
#: that hop (a platform where branding is unavailable, or a checkout run as
#: ``bin/lop serve``) keeps the launcher's argv, and a proof that recognised only
#: the module form would refuse to reclaim precisely the daemons on those hosts.
#: Both arms are the same claim — "this is this product's serve entry point" —
#: which is why they live in one predicate, and a third spelling added later must
#: be added HERE rather than at a call site.
SERVE_LAUNCHER_NAMES = frozenset({"lop", "local-operator"})

#: The verb that makes a launcher invocation a serve daemon rather than, say, a
#: ``lop stop``.
SERVE_LAUNCHER_VERB = "serve"

#: The first words of the label :mod:`local_operator.procname` writes over ``argv[0]``,
#: DERIVED from that module's own template rather than typed again: a label change
#: must not be able to leave this proof behind (review round 3, NIT-2).
_SERVE_LABEL_PREFIX = re.compile(
    "^"
    + re.escape(procname.LABEL_SERVE)
    .replace(re.escape("{brand}"), re.escape(procname.BRAND))
    .replace(re.escape("{port}"), r"\d+")
    + "$"
)

#: The desktop app's managed backend script, word by word. The app spawns
#: ``<interpreter> -c "from local_operator.cli import main; main()" serve --port N``
#: (``local-operator-ui``, ``src/main/backend/owned-serve-launch.ts``) and that shape
#: is DELIBERATELY never branded — ``procname`` refuses a ``-c`` launch on purpose,
#: because the app verifies its backend by asking the same ``-c`` string to report
#: ``sys.executable`` and a re-exec through the branded link would break that check
#: on every machine. So the proof has to accept the spelling itself, or the app's own
#: backend is the one daemon ``reclaim`` cannot end — which was the daemon holding
#: 1111 in the incident this change exists for (review round 3, R3-1).
#:
#: Matched as a WORD SEQUENCE, not as a substring of the command: a ``-c`` that
#: merely prints this text is not a serve daemon.
SERVE_ENTRYPOINT_WORDS = ("from", "local_operator.cli", "import", "main;", "main()")

#: How long a reclaimed daemon is given to leave after ``SIGTERM``.
#:
#: Sized to uvicorn's own shutdown rather than to a drain bound: a serve daemon
#: holds no transcript lease (that is a runtime's) and answers ``SIGTERM`` by
#: closing its listener and exiting. It is NOT the session runtime's 2.5-minute
#: drain, and must not become it — this command exists because the daemon is
#: already refusing to answer, so patience here only delays the operator.
RECLAIM_TERM_GRACE_S = 10.0

#: How long a ``SIGKILL``ed pid is given to disappear. The same 3 s
#: ``control.SIGKILL_CONFIRM_S`` uses, for the same reason: ``SIGKILL`` cannot be
#: refused, so this covers only a process wedged in an uninterruptible syscall.
RECLAIM_KILL_CONFIRM_S = 3.0

#: The poll period while waiting for either exit.
RECLAIM_POLL_S = 0.2

#: No record of OURS claims the address this process serves.
#:
#: The stray case, and the one the incident produced: the process holding the
#: operator's port belonged to a rig whose records lived in its own config root,
#: so nothing in the operator's root described it. Distinct from ``squatted``,
#: which needs one of our records to compare against.
STRAY = "stray"


@dataclass(frozen=True)
class ReclaimReport:
    """What ``lop services reclaim`` did, and the evidence it decided on."""

    pid: int
    verdict: str
    address: str = ""
    acted: bool = False
    lines: tuple[str, ...] = ()
    #: Why nothing was done, when nothing was. Empty on a completed reclaim, and
    #: on a refusal alike — a refusal is a completed decision, and its sentence
    #: is in ``lines``; this field is the machine-readable half.
    problem: str = ""

    @property
    def refused(self) -> bool:
        return bool(self.problem) and not self.acted


def is_serve_command(command: str) -> bool:
    """Is this ``ps`` command line one of this product's serve entry points?

    A PROOF OF BRAND, not a capability: it decides whether the operator's own
    ``reclaim`` may act on a pid they named. The security boundary is elsewhere —
    same uid, and the fact that a person asked for this specific pid — and this
    exists so a mistyped or recycled pid is refused instead of signalled.
    """
    words = command.split()
    marker = SERVE_SPAWN_MARKER
    for index in range(len(words) - len(marker) + 1):
        if tuple(words[index : index + len(marker)]) == marker:
            return True
    # THE LAUNCHER FORM IS ADJACENT — the process IS `lop serve`, so the two words
    # are neighbours — AND IT SITS AT argv[0] OR RIGHT AFTER THE procname LABEL.
    #
    # Two measurements are both load-bearing here, and each one alone was wrong:
    #
    # * matching the pair ANYWHERE in argv accepted ``/bin/zsh -c "lop serve …"``
    #   and ``grep -rn lop serve`` — a person or a script that merely MENTIONS the
    #   command (review round 1, R1-8). That matters here and nowhere else because
    #   a STRAY is signalled on this proof ALONE: no record of ours describes it,
    #   so this brand test and the argv reading are the entire case for acting.
    # * requiring argv[0] killed the proof for the daemons it exists for, because
    #   ``procname`` REPLACES argv[0] with a label. Measured on this host
    #   (2026-09-24): a live daemon reads
    #   ``Local Operator [serve] port=18490 /Users/…/local-operator serve --host …``
    #   and the launcher is four words in, so the round-1 fix refused it and sent
    #   the operator back to killing by hand (review round 2, R2-1).
    #
    # The BASENAME, because a launcher is reached by PATH: the machine's own ``lop``
    # is ``/Users/<me>/.local/bin/lop serve``, and a comparison against the bare word
    # ``lop`` recognised only a daemon started from a directory on PATH — the exact
    # daemons (a rig's, an installer's, a launchd agent's) this proof has to
    # recognise. ``Path(...).name`` is the one place that takes a word and answers
    # "which program is this".
    labelled = _procname_label_length(words)
    for index in range(len(words) - 1):
        if Path(words[index]).name not in SERVE_LAUNCHER_NAMES:
            continue
        if words[index + 1] != SERVE_LAUNCHER_VERB:
            continue
        if index in (0, labelled):
            return True
    # THE DESKTOP APP'S OWN BACKEND: ``<interpreter> -c "<entrypoint>" serve …``.
    # Its verb sits after the interpreter, the ``-c`` and the five script words, so
    # the two arms above cannot see it, and it is the daemon the incident's operator
    # most needed to be able to end (review round 3, R3-1: this predicate refused the
    # app's live backend on 1111 while `services status` told the operator to reclaim
    # that exact pid).
    if _is_serve_entrypoint(words):
        return True
    return False


def _procname_label_length(words: list[str]) -> int:
    """How many leading words are the ``procname`` label; ``0`` when there is none.

    ``procname`` is what makes this project's processes identifiable in the process
    listing, and on this platform it works by REPLACING argv[0] (see
    :mod:`local_operator.procname`), so a labelled daemon's real argv begins one
    label further in. The pattern comes from that module's own ``LABEL_SERVE``
    template, and the length is discovered rather than hard-coded because the brand
    is a template field (``Local Operator [serve] port=18490`` is four words today).
    """
    for length in range(1, min(len(words), 8) + 1):
        if _SERVE_LABEL_PREFIX.match(" ".join(words[:length])):
            return length
    return 0


def _is_serve_entrypoint(words: list[str]) -> bool:
    """Is this argv the desktop app's ``-c`` entry point AND its ``serve`` verb?

    ``<interpreter> -c "from local_operator.cli import main; main()" serve …`` — the
    app's managed backend, which is never branded (see :data:`SERVE_ENTRYPOINT_WORDS`).

    THE ``-c`` IS FOUND, NOT ASSUMED TO BE ``words[1]``. The interpreter comes from
    the console script's shebang, which is a home-derived path, so a home with a
    space in it (``/Users/John Doe/…``) makes the interpreter two ``ps`` words and
    the ``-c`` the third — and the round-3 version refused the app's own backend for
    exactly those users. Measured (review round 4, R4-1).

    A BOOLEAN, not the index it used to return (review round 5, R5-2): the only
    consumer tested truthiness, and a returned index reads like a length.

    The accepted trade-off, recorded because it widens a proof: because the ``-c`` is
    SEARCHED FOR, this also admits an argv carrying an EARLIER, unrelated ``-c`` and
    the app's own pre-exec wrapper (``bash -c 'exec "$@"' owned-serve <interpreter>
    -c … serve``) — both still carry this product's serve entry point followed by the
    verb (review round 5, R5-1). That costs nothing here: the app's plan execs, so no
    nameable pid is ever that ``bash``, and an interpreter handed these exact words as
    ``-c`` *is* our serve entry point, which is what a STRAY proof has to recognise.
    """
    window = len(SERVE_ENTRYPOINT_WORDS)
    for index in range(len(words) - window - 1):
        if words[index] != "-c":
            continue
        if tuple(words[index + 1 : index + 1 + window]) != SERVE_ENTRYPOINT_WORDS:
            continue
        if words[index + 1 + window] == SERVE_LAUNCHER_VERB:
            return True
    return False


def serve_process(pid: int, *, timeout_s: float = HEALTH_TIMEOUT_S) -> str | None:
    """ONE pid's command line, or ``None`` when it is not a serve daemon now.

    The re-read discipline the runtime sweep already applies: the thing about to
    be signalled must be re-identified at signal time, because a pid is
    recyclable and a snapshot is not evidence about this instant. ``None`` is
    therefore a REFUSAL at every call site, never "probably gone".
    """
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["/bin/ps", "-p", str(pid), "-o", "uid=,command="],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:  # noqa: BLE001 — an unreadable process table is doubt
        return None
    for line in result.stdout.splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2 or not fields[0].lstrip("-").isdigit():
            continue
        return fields[1].strip()
    return None


def serve_process_uid(pid: int, *, timeout_s: float = HEALTH_TIMEOUT_S) -> int | None:
    """One pid's uid, or ``None`` when it cannot be read."""
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["/bin/ps", "-p", str(pid), "-o", "uid="],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    text = result.stdout.strip()
    return int(text) if text.isdigit() else None


def _address_from_argv(command: str) -> tuple[str, int] | None:
    """The ``--host``/``--port`` a serve daemon was started with, if it says.

    Used only when no record of ours describes the pid — the stray case. A
    ``--listener-fd`` child carries no ``--port`` of its own (its port came from
    the descriptor), and ``None`` there is honest: the caller refuses rather than
    probing an address it guessed.
    """
    words = command.split()
    host = ""
    port: int | None = None
    for index, word in enumerate(words[:-1]):
        if word == "--host":
            host = words[index + 1]
        elif word == "--port":
            try:
                port = int(words[index + 1])
            except ValueError:
                return None
    if port is None or port == 0:
        return None
    return (host or "127.0.0.1", port)


def serve_daemon_reports(
    root: Path | None = None,
    *,
    probe: Callable[[Any], AddressProbe] = probe_address,
) -> list[ServeDaemonReport]:
    """Every serve record this machine's own root(s) hold, with its verdict.

    ONE scan and — for records that claim to be alive — one identity probe each.
    A ``stale`` record costs nothing (its pid is gone; there is no address to
    ask), which matters because this is called by ``lop services status`` and by
    the update path.

    ``live`` and ``wedged`` are both probed, and that is deliberate: the whole
    point of the address axis is that a FRESH HEARTBEAT IS NOT PROOF OF SERVING.
    A wedged record whose address answers as somebody else is a squatted plane,
    which is a different thing to tell the operator than a quiet owner.
    """
    # ``root=None`` is the caller's own config root: the one daemon a front end or
    # a rig owns, and the only one this reader may act on. There is deliberately
    # NO machine-wide sweep here — an address is not owned by a config root (see
    # the design doc), and reading every root on the machine to decide a refusal
    # would make the same address mean different things to two installs.
    reports: list[ServeDaemonReport] = []
    for record, state in serve_registry.scan(root):
        if state == "stale":
            # A gone pid has no address to ask, and probing it would spend a
            # timeout on a daemon that has already been reaped.
            reports.append(ServeDaemonReport(record=record, state=state))
            continue
        reports.append(ServeDaemonReport(record=record, state=state, probe=probe(record)))
    return reports


#: One entry per verdict that has a ROW: the sentence it renders, and the adjective
#: the receipt uses. ONE dictionary, because two of them is how two surfaces come to
#: describe one daemon differently (review round 1, N1) — and it is read with `[]`
#: rather than `.get()`, so a verdict added without a row is a loud KeyError in a
#: test rather than a silently borrowed sentence on the operator's terminal.
#:
#: THE SHAPE IS "recorded on X, but …", DELIBERATELY. The reader has just been
#: shown the ordinary daemon line for this same record, which names the build the
#: daemon says it serves; a second line reading "is not answering on X" beside it
#: looks like a contradiction rather than the two halves of one statement — the
#: RECORD claims an address (that is what the line above repeats), and the ADDRESS
#: is where the evidence disagrees. Measured on the live evidence run for this
#: change: "serving 0.62.24@eeeeeee" followed by "is not answering on" read as the
#: product arguing with itself.
@dataclass(frozen=True)
class _Verdict:
    """One verdict's row sentence (taking ``{address}``) and its receipt adjective."""

    row: str
    phrase: str


VERDICTS: dict[str, _Verdict] = {
    DEAF: _Verdict("recorded on {address}, but nothing is answering there", "deaf"),
    SQUATTED: _Verdict(
        "recorded on {address}, but what answers there is not this record's daemon",
        "squatted",
    ),
    WEDGED: _Verdict("recorded on {address}, but the daemon stopped reporting", "wedged"),
    STALE: _Verdict("recorded on {address}, but its process has exited", "stale"),
    STRAY: _Verdict("serves {address}, which no record of this install claims", "stray"),
}


def _not_serving_sentence(report: ServeDaemonReport) -> str:
    """One daemon's sentence, from the record's own fields — one renderer for
    ``services status``'s rows and ``services restart``'s warnings."""
    address = f"{report.record.host}:{report.record.port}"
    return VERDICTS[report.verdict].row.format(address=address)


def _remedy_lines(report: ServeDaemonReport) -> list[str]:
    """The remedy, WHERE THERE IS ONE.

    A ``stale`` record needs no command (its pid is gone and the next scan reaps
    the file), and telling a reader to reclaim a pid that no longer exists is the
    kind of advice this repo refuses to print.
    """
    if report.verdict not in (DEAF, SQUATTED, WEDGED, STRAY):
        return []
    return [
        f"    end it with `lop services reclaim {report.record.pid}`"
        " (nothing else on this machine can)"
    ]


def not_serving_lines(report: ServeDaemonReport) -> list[str]:
    """The rows for one daemon that is NOT serving its address.

    It always ends with the remedy, because the failure this whole axis exists
    for was silent: the operator had to reconstruct "who is on 1111" by hand
    while their app was dead, and the product said nothing at all.
    """
    lines = [f"  {_serve_name(report.record)}: {_not_serving_sentence(report)}"]
    if report.probe is not None and report.probe.detail:
        lines.append(f"    {report.probe.detail}")
    return lines + _remedy_lines(report)


def stuck_report_lines(report: ServeDaemonReport) -> tuple[str, ...]:
    """What a command that MOVES services prints about one it cannot move.

    Rendered from the same sentence and the same remedy as the status rows
    (review round 1, R1-3). The first version typed its own sentence here, and it
    said "nothing is answering there" about a SQUATTED address (something was),
    told the reader to reclaim a STALE record's pid (which is gone), and printed
    the raw verdict token where every other surface renders a sentence.
    """
    return (
        f"warning: {_serve_name(report.record)}: {_not_serving_sentence(report)} — it"
        " cannot be asked to move",
        *_remedy_lines(report),
    )


def reclaim_serve_daemon(
    pid: int,
    *,
    probe: Callable[[Any], AddressProbe] = probe_address,
    reports: Callable[[], list[ServeDaemonReport]] = serve_daemon_reports,
    read_command: Callable[[int], str | None] = serve_process,
    read_uid: Callable[[int], int | None] = serve_process_uid,
    kill: Callable[[int, int], None] = os.kill,
    alive: Callable[[int], bool | None] = pid_liveness,
    sleep: Callable[[float], None] = time.sleep,
    term_grace_s: float = RECLAIM_TERM_GRACE_S,
    kill_confirm_s: float = RECLAIM_KILL_CONFIRM_S,
    poll_s: float = RECLAIM_POLL_S,
    confirmations: int = PROBE_CONFIRMATIONS,
    confirm_gap_s: float = PROBE_CONFIRM_GAP_S,
) -> ReclaimReport:
    """End ONE serve daemon the operator named — asked for, proven, bounded.

    THE MISSING OWNER. Until this command existed, nothing in the product could
    end a serve daemon: ``lop stop`` resolves through the SESSION registry,
    ``lop sessions reclaim`` refuses any candidate whose record exists, and
    ``lop services restart`` moves daemons it can still talk to. The daemon that
    cost this machine twelve minutes on 2026-09-23 — alive, recorded, not
    answering, holding the desktop app's port — was reachable by no command at
    all; the operator had to find its pid by hand and ``kill`` it.

    NEVER AUTOMATIC, and that is a design decision rather than an omission. A
    serve daemon may own scheduled work and may be supervising session runtimes
    mid-spawn, and no reader on this machine has a successor-readiness proof (see
    :mod:`local_operator.server.retire`). So the destructive, irreversible step
    stays where this repo puts destructive steps: behind an explicit request that
    names the process.

    PROOF, THEN VERDICT, THEN A RE-READ — in that order, and none of them is
    skippable:

    1. the pid must be THIS product's serve daemon right now (``is_serve_command``
       over a fresh ``ps``), and the same uid as the caller;
    2. its verdict must NOT be ``serving``. A working plane is never ended by
       this command, however it was named — that is the one outcome worse than
       the outage this exists to repair;
    3. for the two verdicts that a transient could mimic — ``deaf`` and
       ``squatted`` — the same probe must agree ``confirmations`` times in a row
       (see :data:`PROBE_CONFIRMATIONS`);
    4. the pid is re-read immediately before the signal, so a leftover pid whose
       process died and was recycled is refused rather than signalled.

    The signal goes to the DAEMON ONLY, never its process tree. A serve daemon's
    children are, on this machine, the session runtimes it spawned for the
    desktop app — conversations with their own journals and their own lives —
    and taking a stuck backend down must not also cut the turns those sessions
    are running. ``procstate.terminate_process_tree`` is therefore deliberately
    NOT used here; that helper is for a tree this process owns.
    """
    command = read_command(pid)
    if command is None:
        return ReclaimReport(
            pid=pid,
            verdict=STALE,
            problem="not-running",
            lines=(f"pid {pid} is not a running process",),
        )
    if not is_serve_command(command):
        return ReclaimReport(
            pid=pid,
            verdict="",
            problem="not-a-serve-daemon",
            lines=(
                f"pid {pid} is not a `lop serve` daemon, so this command will not end it",
                f"    it is running: {command}",
            ),
        )
    uid = read_uid(pid)
    ours = getattr(os, "getuid", None)
    if uid is not None and ours is not None and uid != ours():
        return ReclaimReport(
            pid=pid,
            verdict="",
            problem="foreign-user",
            lines=(f"pid {pid} belongs to uid {uid}, not this account — end it with sudo",),
        )

    # WHICH RECORD DESCRIBES THIS PID — and therefore what its address is.
    #
    # Two shapes, and both are real on this machine. A daemon of THIS root is the
    # ordinary case: its record names the address and supplies the verdict. A
    # daemon that no record of ours describes is the case this command was asked
    # for — a rig's stray, holding an address our records still claim — and it
    # takes its address from the argv the proof above already read.
    described = next((item for item in reports() if item.record.pid == pid), None)
    # ``basis`` is the ADDRESS READING the signal will rest on, when there is one: it
    # is what gets confirmed and what gets re-read last, and it is deliberately not
    # ``verdict`` — the word the receipt uses. The two differ in exactly one place: a
    # stray is NAMED for what it is (no record of ours describes it) while its
    # evidence is what its address answers (review round 1, R1-4/R1-5).
    if described is not None:
        subject = described.record
        address = f"{subject.host}:{subject.port}"
        basis = described.probe.verdict if described.probe is not None else None
        verdict = described.verdict
    else:
        argv_address = _address_from_argv(command)
        if argv_address is None:
            return ReclaimReport(
                pid=pid,
                verdict="",
                problem="no-address",
                lines=(
                    f"pid {pid} is a `lop serve` daemon but its address could not be read",
                    "    (a daemon started with --listener-fd does not name its port in",
                    "    argv), and no record of this install describes it; nothing was sent",
                ),
            )
        host, port = argv_address
        address = f"{host}:{port}"
        claimant = next(
            (item for item in reports() if (item.record.host, item.record.port) == (host, port)),
            None,
        )
        if claimant is None:
            # Nothing of ours claims the address, so there is no identity to compare
            # against: the brand proof above is the whole of what is known, and the
            # verdict says exactly that rather than borrowing a word that claims more.
            #
            # The subject carries an EMPTY instance id, so ``probe`` reads any answer
            # at all as SQUATTED — "something is there" — which is the one fact the
            # sentence needs. Before this the no-claimant arm never asked and asserted
            # "it is not serving that address" about a process that (a rig's backend,
            # on this very machine) can perfectly well be serving a plane of its own.
            subject = SimpleNamespace(host=host, port=port, pid=pid, instance_id="")
            basis = probe(subject).verdict
            verdict = STRAY
        else:
            # Our record describes the ADDRESS but not this pid: the daemon it named is
            # gone (or was replaced) and this process is where that daemon should be.
            # The verdict still comes from a READING of that address rather than from
            # the shape of the record (review round 1, R1-5): the claimant's owner may
            # have come back.
            subject = claimant.record
            basis = probe(subject).verdict
            if basis == SERVING:
                return ReclaimReport(
                    pid=pid,
                    verdict=SERVING,
                    address=address,
                    problem="serving",
                    lines=(
                        f"{address} IS being served, by the daemon pid {subject.pid} names — "
                        "nothing was sent",
                        f"    pid {pid} is a different process, so ending it would not free the",
                        "    address the record is about",
                    ),
                )
            verdict = basis if basis in (DEAF, SQUATTED) else STRAY

    # AN ALLOW-LIST, not a deny-list: only the states whose remedy IS this command may
    # be signalled, so a ``stale`` record (its pid is gone; the brand re-read below
    # would refuse it a moment later) and any verdict added later are refused by
    # default rather than by remembering to add them here.
    if verdict == SERVING:
        return ReclaimReport(
            pid=pid,
            verdict=verdict,
            address=address,
            problem="serving",
            lines=(
                f"pid {pid} IS the daemon serving {address} — nothing was sent",
                "    a working plane is never ended by this command; use `lop stop` for a",
                "    session, or `lop services restart` to move this daemon onto the current build",
            ),
        )
    if verdict not in (DEAF, SQUATTED, WEDGED, STRAY):
        return ReclaimReport(
            pid=pid,
            verdict=verdict,
            address=address,
            problem="not-actionable",
            lines=(
                f"pid {pid} reads as {verdict or 'an unknown state'} on {address}, which is not a",
                "    state this command ends — nothing was sent. Run `lop services status` for",
                "    what this install knows about the address.",
            ),
        )

    # CONFIRM THE EVIDENCE, not the word: ``basis`` can be DEAF for a WEDGED record
    # (the beat says the owner stopped reporting; the address says nobody is there) and
    # it is the ADDRESS that must not change its mind under the signal.
    if basis in (DEAF, SQUATTED):
        for _ in range(max(0, confirmations - 1)):
            sleep(confirm_gap_s)
            again = probe(subject)
            if again.verdict != basis:
                return ReclaimReport(
                    pid=pid,
                    verdict=again.verdict,
                    address=address,
                    problem="unconfirmed",
                    lines=(
                        f"pid {pid} read as {basis} on {address}, but the next reading was "
                        f"{again.verdict} — nothing was sent",
                        "    run it again to act on the newer reading",
                    ),
                )

    # SIGNAL TIME: the process must still be the one this verdict was measured on.
    # ONE read, held in a variable: two reads are two instants, and the second one
    # could belong to a different process than the one that was just proven.
    again = read_command(pid)
    if again is None or not is_serve_command(again):
        return ReclaimReport(
            pid=pid,
            verdict=verdict,
            address=address,
            problem="changed",
            lines=(
                f"pid {pid} is no longer the serve daemon this was measured on — "
                "nothing was sent",
            ),
        )

    # THE LAST READING BEFORE THE SIGNAL IS AN ADDRESS READING (review round 1,
    # R1-5). Between the confirmation above and this instant a daemon can BEGIN
    # SERVING — the recovery this whole command must not punish — and the brand
    # re-read cannot see that, because the process is the same one either way.
    if basis is not None:
        last = probe(subject)
        if last.verdict == SERVING:
            holder = subject.instance_id[:8] if subject.instance_id else "another process"
            return ReclaimReport(
                pid=pid,
                verdict=last.verdict,
                address=address,
                problem="serving",
                lines=(
                    f"{address} began answering as {holder}… since the last reading — "
                    "nothing was sent",
                    "    a working plane is never ended by this command; run it again if that",
                    "    changes",
                ),
            )
        if last.verdict != basis:
            return ReclaimReport(
                pid=pid,
                verdict=last.verdict,
                address=address,
                problem="changed",
                lines=(
                    f"{address} changed while this was being confirmed ({basis} to "
                    f"{last.verdict}) — nothing was sent",
                    "    run it again to act on the newer reading",
                ),
            )

    # WHAT THE EVIDENCE ACTUALLY SAYS, in the receipt rather than a claim that
    # flattens three states into one (review round 1, R1-3): a stray CAN be serving a
    # plane of its own, and ``deaf`` and ``squatted`` are different facts.
    if verdict == STRAY:
        evidence = (
            f"it is serving {address} under records that are not this install's"
            if basis == SQUATTED
            else f"nothing is answering {address}"
        )
    elif basis == SQUATTED:
        evidence = f"{address} is answered by a process this install does not claim"
    else:
        evidence = f"nothing is answering {address}"

    lines = [
        f"ending {VERDICTS[verdict].phrase} serve daemon on {address} (pid {pid})",
        f"    {evidence}, and any session runtimes it spawned keep running",
    ]
    kill(pid, signal.SIGTERM)
    outcome = _await_exit(pid, alive, sleep, term_grace_s, poll_s)
    if outcome is True:
        lines.append("    ended on SIGTERM")
        return ReclaimReport(
            pid=pid, verdict=verdict, address=address, acted=True, lines=tuple(lines)
        )
    lines.append(f"    did not leave within {term_grace_s:g}s; sending SIGKILL")
    try:
        kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        # It left between the last read and this signal: the escalation found no
        # process, which is the outcome that was asked for rather than an error.
        lines.append("    ended on SIGKILL")
        return ReclaimReport(
            pid=pid, verdict=verdict, address=address, acted=True, lines=tuple(lines)
        )
    outcome = _await_exit(pid, alive, sleep, kill_confirm_s, poll_s)
    if outcome is True:
        lines.append("    ended on SIGKILL")
        return ReclaimReport(
            pid=pid, verdict=verdict, address=address, acted=True, lines=tuple(lines)
        )
    if outcome is None:
        # DOUBT, NOT A SURVIVOR (review round 1, R1-7). ``_await_exit`` answers three
        # ways and the third used to be folded into "it is wedged in the kernel; a
        # reboot is the only way left" — a diagnosis from a reading that did not
        # arrive, which is the one thing a fail-closed command must never print.
        lines.append("    the process table could not be read after SIGKILL: this is DOUBT, not a")
        lines.append("    survivor — check `lop services status` and the process table by hand")
        return ReclaimReport(
            pid=pid,
            verdict=verdict,
            address=address,
            acted=True,
            problem="unreadable-after-signal",
            lines=tuple(lines),
        )
    lines.append("    STILL RUNNING — it is wedged in the kernel; a reboot is the only way left")
    return ReclaimReport(
        pid=pid,
        verdict=verdict,
        address=address,
        acted=True,
        problem="survived-sigkill",
        lines=tuple(lines),
    )


def _await_exit(
    pid: int,
    alive: Callable[[int], bool | None],
    sleep: Callable[[float], None],
    budget_s: float,
    poll_s: float,
) -> bool | None:
    """Wait for a pid to disappear inside ``budget_s``.

    ``True`` it is gone; ``False`` it is still there; ``None`` the LAST read could
    not be taken. Three answers, and the third is DOUBT rather than death or
    survival: ``alive`` is itself three-valued (``None`` = unreadable) and an
    unreadable read keeps waiting rather than reporting an exit that was never
    observed — the same rule the record classifier states for a heartbeat it could
    not read. The answer describes the FINAL read rather than the worst one along
    the way, because that is the reading the caller prints (review round 2, N-4: a
    sticky "something was unreadable" flag could announce a doubt the last read had
    already resolved).
    """
    waited = 0.0
    while waited < budget_s:
        if alive(pid) is False:
            return True
        sleep(poll_s)
        waited += poll_s
    seen = alive(pid)
    if seen is False:
        return True
    return None if seen is None else False


def _current_stamp() -> Any:
    """The build the pointer names, or ``None`` when it cannot be read.

    The SUBJECT of the whole module, and read from the pointer rather than from
    this process (``update.disk_build()``'s argument) on purpose: the caller may
    itself be a superseded build — the desktop app runs ``lop update`` through the
    stable shim, and the shim's interpreter is whatever ``current`` named when the
    process started. Asking ``sys.prefix`` would answer "every daemon is already
    current" for the one process that just made them all stale.
    """
    from local_operator import update

    try:
        return update.disk_build()
    except Exception:  # noqa: BLE001 — an unreadable install is "no answer"
        logger.debug("install stamp unreadable", exc_info=True)
        return None


def _serves_current_build(record: Any, stamp: Any) -> bool:
    """Is this daemon already running the build the pointer names?

    Both halves, because this machine's common drift is a SAME-VERSION rebuild
    (``lop-update`` rebuilds the checkout, so ``pyproject.toml`` names the last
    release and only ``source_ref`` moves) — a version-only test would call a
    stale daemon current on exactly the host this is for.
    """
    if stamp is None:
        return False
    return (record.version, record.source_ref) == (stamp.version, stamp.source_ref)


def reload_serve_daemons(
    *,
    wait_s: float = RELOAD_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
    scan: Callable[[], list[Any]] = live_serve_daemons,
    kill: Callable[[int, int], None] = os.kill,
    probe: Callable[[Any], str | None] = _answers_as_record,
    monotonic: Callable[[], float] = time.monotonic,
    stuck: Callable[[], list[ServeDaemonReport]] = serve_daemon_reports,
) -> list[ServiceRefresh]:
    """Ask every stale ``serve`` daemon to move itself onto the current build.

    ONE REQUEST PER DAEMON, THEN ONE WAIT — not request/wait/request/wait. The
    daemons move in parallel by construction (each is an independent process), so
    serialising the waits would add their drains together: three busy daemons
    would cost three times the budget to discover something that was true two
    waits ago. The whole fleet gets one deadline instead.

    NEVER RAISES. A service that could not be moved is a warning on a successful
    update, exactly like ``update.refresh_daemons_after_upgrade``'s contract, and
    for the same reason: the install has already been replaced, and a failed
    nudge must not be reported as a failed update.
    """
    # SCAN FIRST, so "nothing to say" stays silent. A machine with no serve
    # daemons running is not a machine with a broken install, and a warning on
    # every `lop update` on such a host would be noise that trains its reader to
    # ignore the line that matters.
    daemons = scan()
    # AND THE DAEMONS THAT ARE NOT THERE TO MOVE, reported FIRST and independently
    # of the loop below — which iterates LIVE records, so a daemon that is alive
    # and answering nothing is invisible to it, and the early return under it would
    # swallow the report entirely on the machine where it matters most (the
    # incident's own host: `live_serve_daemons()` was empty because the stuck record
    # classified as `wedged`, so `lop services status` printed "none running" while
    # 1111 was held). Reported, never signalled: the decision is the operator's.
    stuck_reports = [report for report in stuck() if report.verdict != SERVING]
    if not daemons and not stuck_reports:
        return []
    refreshes: list[ServiceRefresh] = [
        ServiceRefresh(
            _serve_name(report.record),
            warnings=stuck_report_lines(report),
        )
        for report in stuck_reports
    ]
    # NO EARLY RETURN FOR A PLATFORM WITHOUT SIGUSR1 (serve-reload review round 8,
    # R8-1). The first version refused the whole fleet up front, which was redundant
    # for signalling — every daemon reports `reloadable: false` there, so the loop
    # below takes its "cannot move itself" branch for each one anyway — and LOSSY for
    # the report, because it replaced those per-daemon lines with a single sentence.
    # The operator on such a platform now sees exactly which daemons are running and
    # what each needs, which is the same information the POSIX path gives.
    stamp = _current_stamp()
    if stamp is None:
        # NO STAMP MEANS NO PREDICATE (serve-reload review round 2, R2-1). ``_serves_current_build``
        # answers False when the stamp is unreadable, so without this guard EVERY
        # daemon looks stale and the whole fleet gets signalled by a caller that
        # has no build to move anything onto. That is not hypothetical: it was
        # reachable the moment the services stage started running on ``lop
        # update``'s "nothing to install" path, and a source checkout is exactly
        # such a caller — its ``disk_build()`` is None, so a developer running
        # ``lop update`` in their worktree would have signalled this machine's serve
        # daemon. Demonstrated end to end in review with a fabricated-root daemon:
        # stale → signalled → really reloaded. (serve-reload review round 4, R4-4: the first two
        # versions of this sentence listed the mobile daemon, the browser bridge and
        # the tunnel as well, and both overclaimed — see below.)
        #
        # THE BLAST RADIUS IS THE SERVES, not the whole fleet (serve-reload review
        # round 3, R3-4: an earlier version of this comment listed the mobile daemon, the
        # browser bridge and the tunnel, and overclaimed — `_repair_refusal`
        # already refuses an editable caller inside the plist refresh child).
        #
        # AND NOT THE MOBILE DAEMON EITHER, except on one path (serve-reload review
        # round 4, R4-4): a checkout's `lop update --no-services` does still reach the mobile
        # bounce, which has no guard of this kind and never did. That is
        # pre-existing behaviour rather than anything this change added —
        # `--no-services` is deliberately the pre-change path — so it is named here
        # and left alone rather than quietly changed under a flag whose whole job is
        # to reproduce the old behaviour.
        #
        # The invariant this restores: "stale" is a comparison, and a comparison
        # against an absent right-hand side is not a verdict. Fail-closed, and
        # said out loud — an operator whose pointer is unreadable AND who has
        # daemons running has a real problem and should hear about it.
        # The stuck-daemon reports are carried out through this return too: they are
        # a fact about the machine, not about the pointer, and the one host where
        # this arm fires is a source checkout — where a stuck daemon holding the
        # app's address is exactly what the reader needs to see.
        refreshes.append(
            ServiceRefresh(
                "serve daemons",
                warnings=(
                    "warning: this install has no build the pointer can name, so no "
                    "service can be moved onto it and NOTHING was signalled. Run "
                    "`lop services status` to see what is running, and `lop update` "
                    "from the install that owns them.",
                ),
            )
        )
        return refreshes
    pending: list[Any] = []

    # The SCAN IS DONE ONCE and reused: scanning twice would let a daemon arrive
    # or depart between the guard and the loop, which is exactly the kind of gap
    # the guard exists to close.
    for record in daemons:
        name = _serve_name(record)
        if _serves_current_build(record, stamp):
            refreshes.append(ServiceRefresh(name, lines=(f"{name} is already on {_label(stamp)}",)))
            continue
        # ONE WAY TO BE UNMOVABLE, TWO REASONS — and the platform is one of them
        # (serve-reload review round 9, R9-1; found by the test that also covers
        # R8-1's arm). Removing R8-1's early return was right for the REPORT and
        # wrong for the SIGNAL: it left the decision resting on the record's
        # `reloadable`, so a daemon whose record claims the capability — a record
        # written by a build from before this change, or a hand-edited one — would be
        # sent a signal this platform does not have, with `None` reaching `os.kill`.
        # The per-daemon report R8-1 asked for is kept; the signal is decided here,
        # where it cannot be delegated to a field.
        unmovable = serve_reload.RELOAD_SIGNAL is None or not getattr(record, "reloadable", False)
        if unmovable:
            # The capability is absent for every daemon built before this
            # existed, and for a --reload child whose port belongs to uvicorn's
            # supervisor. Sending the signal anyway is NOT harmless: SIGUSR1's
            # default disposition is to terminate, so signalling a daemon that
            # never installed the handler kills the backend this command exists
            # to bring along. Report it by name instead.
            refreshes.append(
                ServiceRefresh(
                    name,
                    warnings=(
                        f"warning: {name} is on {record.version or 'an older build'} and "
                        "cannot move itself ("
                        + (
                            "this platform has no SIGUSR1"
                            if serve_reload.RELOAD_SIGNAL is None
                            else "it predates in-place reload"
                        )
                        + "); restart that "
                        "server by hand, or stop and start whatever supervises it",
                    ),
                )
            )
            continue
        try:
            # PROVE THE PROCESS BEFORE TOUCHING IT (serve-reload review round 1, R1-4). A pid
            # is not an identity, and the request is a signal whose default
            # disposition is death.
            mismatch = probe(record)
            if mismatch is not None:
                refreshes.append(
                    ServiceRefresh(
                        name,
                        warnings=(
                            f"warning: {name} was not asked to reload — {mismatch}. "
                            "Nothing was signalled.",
                        ),
                    )
                )
                continue
            # The family: a mismatch here is silent, which is why it has its own test.
            kill(record.pid, serve_reload.RELOAD_SIGNAL)  # type: ignore[arg-type]  # guarded above
        except OSError as exc:
            refreshes.append(
                ServiceRefresh(
                    name,
                    warnings=(f"warning: {name} could not be asked to reload: {exc}",),
                )
            )
            continue
        pending.append(record)
        refreshes.append(
            ServiceRefresh(
                name,
                lines=(
                    f"{name} was asked to move onto {_label(stamp)}; "
                    "its conversations keep running",
                ),
            )
        )

    if pending:
        refreshes.extend(
            _await_relocations(pending, wait_s=wait_s, sleep=sleep, scan=scan, monotonic=monotonic)
        )
    return refreshes


def _await_relocations(
    pending: Sequence[Any],
    *,
    wait_s: float,
    sleep: Callable[[float], None],
    scan: Callable[[], list[Any]],
    monotonic: Callable[[], float],
) -> list[ServiceRefresh]:
    """Wait for the asked daemons to republish under a NEW ``instance_id``.

    ``instance_id`` is the proof and the version is not: it is minted once per
    process (``server/app.py``), so a changed one can only mean the process that
    published before has been replaced — whereas the version is equal across a
    same-version rebuild, which is this host's ordinary case. Comparing versions
    here would report "did not move" for a reload that worked.

    A daemon that vanishes while we wait is NOT reported as a failure of the
    request: a pid that is gone may have exited for its own reasons, and the
    honest report is that it stopped answering — the reader can see the record is
    gone too.
    """
    before = {record.pid: record.instance_id for record in pending}
    deadline = monotonic() + wait_s
    settled: dict[int, Any] = {}
    while True:
        for record in scan():
            if record.pid in before and record.pid not in settled:
                if record.instance_id != before[record.pid]:
                    settled[record.pid] = record
        if len(settled) == len(before):
            break
        if monotonic() >= deadline:
            break
        sleep(RELOAD_POLL_S)

    out: list[ServiceRefresh] = []
    for record in pending:
        name = _serve_name(record)
        moved = settled.get(record.pid)
        if moved is not None:
            out.append(
                ServiceRefresh(
                    name,
                    lines=(f"{name} is now serving {moved.version or 'the current build'}",),
                )
            )
        else:
            out.append(
                ServiceRefresh(
                    name,
                    warnings=(
                        f"warning: {name} did not come back on the current build within "
                        f"{_seconds_label(wait_s)}; it is still serving the build it loaded "
                        "and the next `lop services restart` will try again",
                    ),
                )
            )
    return out


def restart_services(*, wait_s: float = RELOAD_WAIT_S) -> list[ServiceRefresh]:
    """Bring EVERY non-runtime service onto the current build.

    THE ONE OWNER of that sentence, so ``lop update`` and ``lop services
    restart`` cannot drift into two different ideas of what "everything" means.

    The supervised daemons go first and are repaired by a child of the NEW build
    (``update.refresh_daemons_after_upgrade``, which owns that rule and its
    reasons); the ``serve`` reloads follow. A run that moved nothing is a
    success with nothing to say — the list comes back with skips in it rather
    than with an error.
    """
    from local_operator import update

    refreshes: list[ServiceRefresh] = []
    try:
        for refresh in update.refresh_daemons_after_upgrade():
            refreshes.append(
                ServiceRefresh(
                    refresh.name, lines=tuple(refresh.lines), warnings=tuple(refresh.warnings)
                )
            )
    except Exception:  # noqa: BLE001 — a failed repair must not fail the command
        logger.warning("supervised daemon refresh failed", exc_info=True)
        refreshes.append(
            ServiceRefresh(
                "service daemons",
                warnings=(
                    "warning: the installed daemons could not be refreshed; run "
                    "`lop browser restart`, `lop mobile restart` and `lop tunnel restart`",
                ),
            )
        )
    refreshes.extend(reload_serve_daemons(wait_s=wait_s))
    return refreshes


#: The width `status` keeps every line inside. The house budget, and the one the
#: check in this module's tests asserts — a soft-wrapping terminal breaks mid-word, so
#: 137 columns rendered as ``curre`` / ``nt`` (design review D6).
_LINE_BUDGET = 80

#: The mobile relay's LaunchAgent label. It is the one supervised daemon that `restart`
#: bounces even when the repair guard refuses to repoint the others, which is why the
#: note names it separately and only when its plist is installed.
_MOBILE_AGENT_LABEL = "com.local-operator.mobile"


def _install_may_repoint_daemons() -> bool:
    """May THIS caller rewrite the supervised plists?

    The repair guard's own question, and only IT — because the guard does not speak for
    the other half of ``restart``. Measured on 2026-09-18 by bouncing the real daemon
    from a checkout: ``update.refresh_daemons_after_upgrade`` composes
    ``refresh_service_daemons_after_upgrade`` (the plist REPAIR, which consults this
    guard) with ``refresh_mobile_after_upgrade`` (a plain ``lop mobile restart`` BOUNCE,
    which consults nothing). So a checkout refuses to repoint browser/tunnel/wakes and
    still bounces the mobile relay onto the current build.

    Drawing that distinction too coarsely is what round 11's R11-3 fix got wrong in the
    other direction: it suppressed a clause that is TRUE in every state, and gated the
    serve-daemon advice — which never consults this guard at all — on it as well.

    An unanswerable guard is NOT permission, so an exception reads as "no".
    """
    from local_operator import update

    try:
        return update._repair_refusal() is None
    except Exception:  # noqa: BLE001 - an unreadable guard must not licence a promise
        logger.debug("repair refusal unreadable", exc_info=True)
        return False


def status_lines() -> list[str]:
    """One line per non-runtime service, with the build it is serving.

    The reader this exists for is an operator who has just been told "the server is on
    an older build than the install" by a desktop app and wants to see WHICH process
    said so. Nothing here acts, so it is safe to run at any time.

    BOTH SIDES OF EVERY COMPARISON ARE NAMED, which the first version did not do
    (design review D1): it printed the daemon's own version and nothing else, so on a
    SAME-VERSION REBUILD — this host's ordinary drift, and the whole reason
    ``BuildStamp`` compares a ref — the output read ``STALE (0.59.2)`` beside an
    install that was also 0.59.2 with no ref anywhere to explain it, and after an
    upgrade the new build appeared nowhere at all. A verdict whose two operands are
    not both visible is not a report; it reads as a fault in the tool that printed it.

    NOTHING HERE PROMISES A MOVE THE TOOL CANNOT MAKE. Every sentence that names
    ``lop services restart`` is gated on that command being able to act — the action
    line and the supervised note both (D7, D11, R11-2, R11-3) — because on a checkout
    the guard refuses and on a fleet where every stale daemon is non-reloadable the
    verb skips the very daemons the line is about.
    """
    stamp = _current_stamp()
    lines: list[str] = [
        (
            f"install: {_label(stamp)}"
            if stamp is not None
            else "install: no build the pointer can name, so nothing can be compared"
        )
    ]
    # ONE SCAN, ONE ROW PER RECORD (review round 1, R1-6). Reading the fleet from
    # ``live_serve_daemons()`` and the stuck set from a SECOND scan let a daemon that
    # is live-and-deaf print twice — once as an ordinary row and once as a stuck one
    # — which breaks the property ``_fleet_action_lines`` documents ("exactly one
    # `serve daemon` line per record", the thing `grep`/`awk` counting relies on)
    # and showed the incident's own daemon as two daemons in the evidence run. The
    # row a record gets is now decided by its composed verdict, which is also what
    # makes the probe authoritative for the ordinary row: a record the classifier
    # calls ``wedged`` but whose address answers as its own instance is SERVING
    # here, and it is listed as what it is rather than both ways.
    reports = serve_daemon_reports()
    daemons = [report.record for report in reports if report.verdict == SERVING]
    stuck = [report for report in reports if report.verdict != SERVING]
    if not reports:
        lines.append("serve daemons: none running")
    elif not daemons:
        # "none serving", NOT "not answering": the stuck set mixes records whose
        # address is silent with records whose address is held by somebody else, and
        # the summary is printed directly above rows that say which — so it must not
        # claim the one reason while a `squatted` row sits two lines below it (review
        # round 2, R2-4). The word for the state is the verdict; the count is the only
        # thing this line adds.
        lines.append(
            f"serve daemons: none serving ({len(stuck)} recorded, none serving their address)"
            if len(stuck) > 1
            else "serve daemons: none serving (1 recorded, not serving its address)"
        )
    for record in daemons:
        lines.extend(_daemon_status_lines(record, stamp))
    for report in stuck:
        lines.extend(not_serving_lines(report))
    lines.extend(_fleet_action_lines(daemons, stamp))
    supervised = _supervised_daemon_plists()
    for path in supervised:
        lines.append(f"supervised daemon: {path.stem}")
    if supervised:
        lines.append("note: a supervised daemon resolves the install when it starts,")
        lines.append("      so its build is not in the plist.")
        if _install_may_repoint_daemons():
            lines.append("      `lop services restart` puts them on the current build")
        elif any(path.stem == _MOBILE_AGENT_LABEL for path in supervised):
            # TRUE IN BOTH HALVES, which the blunt gate was not: the bounce happens
            # whatever this caller is, the repoint does not. Gated on the mobile plist
            # being among the installed ones (round 13, R13-1; the design round's D21):
            # with no mobile plist `refresh_mobile_after_upgrade` returns without
            # bouncing anything, so promising a bounce would over-warn about a daemon
            # the reader does not have.
            lines.append("      `lop services restart` still bounces the mobile relay;")
            lines.append("      the rest are repointed only by the install that owns them")
        else:
            lines.append("      `lop services restart` repoints them only from the")
            lines.append("      install that owns them")
    return lines


def _fleet_action_lines(daemons: Sequence[Any], stamp: Any) -> list[str]:
    """What the reader should DO about the fleet, or nothing at all.

    The advice is only printed when ``restart`` would actually act on a serve daemon:
    there is a build to move onto (an unnameable install refuses the whole half) and at
    least one stale daemon is reloadable. The repair guard is deliberately NOT consulted
    here: ``reload_serve_daemons`` never asks it, so gating on it suppressed correct
    advice in a durable second install, where serve reloads work and only the plist
    repoint is refused.
    ``reloadable`` defaults to False and records written before the capability existed
    do not carry it, so "every stale daemon is unmovable" is the COMMON case, not an
    edge one: naming the verb there sends the reader to a command that skips the very
    daemons the line is about (design review D11, round 11 R11-2).
    """
    if not daemons:
        return []
    if stamp is None:
        # A POINTER, OR NOTHING (design review D16, code round 12 R12-1). The first
        # version of this fallback was not stamp-gated, so on a checkout — which is
        # what `disk_build()` returns None for by design, so it is the DEFAULT
        # developer render — it called the fleet "the stale ones" and prescribed a HAND
        # restart, while the install line two lines above said the comparison was
        # impossible. No daemon in that render was labelled STALE:, so the line had no
        # evidence for its own noun, and the action it prescribed is the destructive
        # one: a hand restart drops the runtimes the in-place reload exists to keep. It
        # now says only what it knows, about the half it is about.
        return ["no build to compare against, so no serve daemon can be moved automatically"]
    stale = [r for r in daemons if not _serves_current_build(r, stamp)]
    if not stale:
        return []
    movable = [r for r in stale if getattr(r, "reloadable", False)]
    if not movable:
        # Written to the budget like every other line here — the first version of this
        # sentence was 86 columns, over the budget it was added to protect. And the noun
        # agrees with its count (round 13, R13-3): "the stale ones" over ONE daemon is
        # the same defect as "1 need".
        if len(stale) == 1:
            return [
                "the stale daemon cannot be moved by `lop services restart`; restart it by hand"
            ]
        return ["the stale ones cannot be moved by `lop services restart`; restart them by hand"]
    if len(movable) == len(stale):
        return ["run `lop services restart` to move the stale ones onto the current build"]
    # ``need`` agrees with its count (code round 12, R12-3): the first version said
    # "1 need restarting by hand" at one, which is right at two and wrong at one.
    needs = len(stale) - len(movable)
    return [
        f"run `lop services restart` for the {len(movable)} it can move; "
        f"{needs} need{'s' if needs == 1 else ''} restarting by hand"
    ]


def _daemon_status_lines(record: Any, stamp: Any) -> list[str]:
    """One daemon's lines: where it is, what it serves, and what that means.

    SHORT LINES, AND THE VERDICT LEADS (design review D6). Folding everything onto one
    line took the old 70-72 columns to **137** — a fix for a reader who could not see
    both operands, producing a line that soft-wraps mid-word in an 80-column terminal
    and pushes the action clause, the part that says what to DO, onto the continuation
    where adjacent daemons' clauses start mid-sentence. So a daemon that needs nothing
    stays on ONE line when it fits, and one that needs something gets its identity on a
    line of its own followed by INDENTED lines that open with the verdict — the widest
    anchor the eye has. Exactly one `serve daemon` line per record, so `grep`/`awk`
    still see a row per daemon.

    THE JOINED FORM IS WIDTH-CHECKED, not assumed (round 11 R11-1): a long authority —
    a link-local or global IPv6 host — takes ``serve daemon pid N on [host]:port —
    current (X@y)`` past the budget on its own, so the verdict moves to its own line
    when the joined one will not fit. The check is on the rendered length, so no host
    or label can slip past it.

    ``reloadable`` is not reader vocabulary: it is rendered as the action it decides,
    and omitted where it decides nothing (D1). The authority is bracketed for an IPv6
    host, where ``::1:56569`` is ambiguous (D4).
    """
    authority = (
        f"[{record.host}]:{record.port}" if ":" in record.host else f"{record.host}:{record.port}"
    )
    serving = _record_label(record)
    where = f"serve daemon pid {record.pid} on {authority}"
    if stamp is None:
        # The install line above has already said WHY nothing can be compared, so this
        # does not repeat it (design review D13) — and it is not a tautology about the
        # current build either, which is what the first version printed here because
        # `_label(None)` falls back to a phrase.
        return [where, f"  serving {serving}"]
    if _serves_current_build(record, stamp):
        joined = f"{where} — current ({serving})"
        if len(joined) <= _LINE_BUDGET:
            return [joined]
        return [where, f"  current ({serving})"]
    action = (
        "`lop services restart` will move it"
        if getattr(record, "reloadable", False)
        else "cannot move itself; restart it by hand"
    )
    return [
        where,
        f"  STALE: serving {serving}, current is {_label(stamp)}",
        f"  {action}",
    ]


def _record_label(record: Any) -> str:
    """``version@ref7`` for a record, matching ``BuildStamp.label()``'s shape.

    The ref is the part that distinguishes a same-version rebuild from a current
    daemon, so dropping it made the common case unreadable (design review D1).
    """
    version = record.version or "unknown build"
    ref = str(getattr(record, "source_ref", "") or "")[:7]
    return f"{version}@{ref}" if ref else version


def _seconds_label(value: float) -> str:
    """``45s``, ``0.5s`` — never ``0s`` for a budget that is not zero (D2).

    The wait is echoed back in a warning, and ``{value:.0f}`` printed ``within
    0s`` for a half-second budget the caller had genuinely asked for.
    """
    return f"{value:g}s"


def _supervised_daemon_plists() -> list[Any]:
    """The installed LaunchAgents this product owns, via ``update``'s own probe.

    Asked of ``update`` rather than re-derived so the two cannot disagree about
    which labels are ours: ``_installed_daemon_plists`` is also what decides
    whether the repair is worth a child process at all.
    """
    from local_operator import update

    return list(update._installed_daemon_plists())


def _serve_name(record: Any) -> str:
    """``serve pid <pid>``, the name every line about one daemon is prefixed with."""
    return f"serve pid {record.pid}"


def _label(stamp: Any) -> str:
    """``BuildStamp.label()`` when there is a stamp, else a phrase that says so."""
    return stamp.label() if stamp is not None else "the current build"


def print_refreshes(refreshes: Sequence[ServiceRefresh]) -> None:
    """Print a refresh list the way ``lop update`` already prints its daemons.

    ``lines`` first and in the order the callers built them, then ``warnings`` —
    the same discipline ``update._print_daemon_refreshes`` states, and for the
    same reason: a warning is what the reader has to act on, so it goes where the
    eye lands last rather than being interleaved with success.
    """
    for refresh in refreshes:
        for line in refresh.lines:
            print(line)
    for refresh in refreshes:
        for warning in refresh.warnings:
            print(warning, file=sys.stderr)
