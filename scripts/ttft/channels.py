"""The six channels, each driven the way its real front end drives it.

WHAT A CHANNEL IS HERE
======================
A channel is a path from "the user pressed send" to "a front end has something new
to paint", and each one is driven through its REAL surface rather than through a
re-implementation: the desktop arm posts to the real daemon and reads its real
SSE, the mobile arm drives the real phone routes (``/api/sessions/start``,
``/api/sessions/{id}/command``, ``/api/sessions/{id}/events``) on the real
``build_app``, the jobs arm posts to the real ``/v1/chat/async`` and follows the
real ``/v1/sse/jobs/{id}``, and the exec arm runs the real console entry point.

Where a channel's front end cannot be driven here, that is stated per driver
rather than hidden — see ``drive_exec`` (no runtime-side stamp) and ``drive_desktop``
(the admission-ack frame does not exist on this tree yet).

WARMTH, DEFINED PER CHANNEL FAMILY
==================================
``cold`` means the channel pays its cold-start cost inside this measurement, and
``warm`` means the same channel on a state that already exists. For the
session-shaped channels (``tui``, ``desktop``, ``mobile``) that is the first
message of a session versus a later message of the same session — the distinction
a user feels. For ``exec`` and ``sse-jobs`` the channel is one fresh process per
turn BY CONTRACT, so a warm session does not exist: ``warm`` there means a turn
processed after one has already run in this run group, i.e. the page and bytecode
caches are warm and the process is not. That is the same convention the
pre-existing bench used for its cold-process TUI arm, and the table labels it
rather than leaving a reader to assume it.

WHY RUNS ARE SEQUENTIAL AND CONCURRENCY IS NOT
==============================================
``concurrency=N`` means N turns in flight at once — N sessions for the
session-shaped channels — and all N samples pool into the cell, so a cell's p95
describes "one user of N simultaneous ones". RUNS are sequential, by necessity:
the in-process drivers point this process at the run's own ``HOME`` and config
dir, so two runs at once would measure each other's state. That is a property of
the harness, not of the product, and it is why concurrency is expressed as
concurrent TURNS inside a run.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, cast

from scripts.ttft import metrics as M
from scripts.ttft.isolate import IsolatedRun, child_env
from scripts.ttft.loopback import LoopbackProvider

#: How long a single frame may take before the run is called a failure rather
#: than silently lengthening a percentile.
FRAME_TIMEOUT_S = 60.0

#: A turn that needs a priming turn first, or a whole turn, gets this multiple.
TURN_TIMEOUT_S = FRAME_TIMEOUT_S * 4

#: The prompt every measured turn submits. The token is what correlates a turn's
#: wire traffic with the provider's stamps; it is fresh per turn so a helper call
#: (auto-naming, effort classification, a compaction summary) can never be
#: mistaken for the measured turn.
PROMPT_SUFFIX = "Reply with one short sentence."

#: Frame/event names that carry REASONING. The agreed vocabulary for the
#: streamed-reasoning change is ``reasoning.delta`` on the SSE and desktop planes
#: plus a matching agent event; the extra spellings are here so these detectors
#: start reporting the moment that change lands, with no edit in this file. On the
#: tree this harness was written against every channel reports ``-1`` for
#: reasoning, and that IS the finding — see the ``metrics`` module docstring.
REASONING_NAMES = frozenset(
    {"reasoning.delta", "reasoning_update", "reasoning_delta", "reasoning", "thinking.delta"}
)

#: Transcript row kinds the phone paints reasoning into. No such kind exists on
#: this tree; listed here with the reasoning names above for the same reason.
REASONING_ROW_KINDS = frozenset({"reasoning", "thinking"})


def names_reasoning(kind: str) -> bool:
    """True when a frame or event name is in the reasoning vocabulary."""
    return kind.lower() in REASONING_NAMES


def new_token() -> str:
    return secrets.token_hex(8)


def prompt_for(token: str) -> str:
    return f"[bench:{token}] {PROMPT_SUFFIX}"


@dataclass(frozen=True)
class ChannelConfig:
    """Which provider to point the measured app at."""

    hosting: str
    model: str
    provider: LoopbackProvider | None


# ---------------------------------------------------------------------------
# Per-turn marks
# ---------------------------------------------------------------------------


@dataclass
class TurnMarks:
    """The stamps of ONE measured turn, every metric as a delta from its submit.

    ``paints`` and ``carries_text`` are separate flags on purpose: on the phone
    the first visible motion can be the activity line ("thinking") with no
    assistant text yet, and that is a real paint which must NOT be recorded as
    first TEXT. Folding them would overstate the phone's text latency and hide
    exactly the state the streamed-reasoning change moves.
    """

    token: str
    submit_monotonic: float
    submit_epoch: float
    marks: dict[str, float] = field(default_factory=dict)

    def note_paint_at(
        self,
        at: float,
        *,
        paints: bool,
        carries_text: bool = False,
        carries_reasoning: bool = False,
        emit_monotonic: float | None = None,
    ) -> None:
        """Record the front end receiving something for this turn, at instant ``at``.

        ``at`` is passed in rather than read here because the reader owns the
        instant: it timestamps the frame the moment it decodes it, and a caller
        that let this method read its own clock would fold the caller's
        scheduling back into the measurement.
        """
        if paints and M.FIRST_PAINT not in self.marks:
            self.marks[M.FIRST_PAINT] = (at - self.submit_monotonic) * 1000
            self.marks[M.RUNTIME_EMIT] = (
                (emit_monotonic - self.submit_monotonic) * 1000
                if emit_monotonic is not None
                else M.UNAVAILABLE
            )
        if carries_text and M.FIRST_TEXT not in self.marks:
            self.marks[M.FIRST_TEXT] = (at - self.submit_monotonic) * 1000
        if carries_reasoning and M.FIRST_REASONING not in self.marks:
            self.marks[M.FIRST_REASONING] = (at - self.submit_monotonic) * 1000

    def note_paint(self, **kwargs: Any) -> None:
        """As :meth:`note_paint_at`, stamping "now"."""
        self.note_paint_at(time.monotonic(), **kwargs)

    def note_runtime_reasoning(self) -> None:
        """The RUNTIME produced a reasoning delta — below the front end, so this is
        the observation that shows the delta EXISTS while no user can see it."""
        self.marks.setdefault(
            M.RUNTIME_REASONING, (time.monotonic() - self.submit_monotonic) * 1000
        )

    def note_stream_entered(self) -> None:
        """The runtime entered its stream function: all local work before the call."""
        self.marks.setdefault(M.STREAM_ENTERED, (time.monotonic() - self.submit_monotonic) * 1000)

    def note_admitted(self) -> None:
        """The submit request returned — admission, which is not visible output."""
        self.marks.setdefault(M.ADMITTED, (time.monotonic() - self.submit_monotonic) * 1000)

    def note_turn_end(self) -> None:
        self.marks[M.TURN] = (time.monotonic() - self.submit_monotonic) * 1000

    def sample(self, *, arm: str, channel: str, **extra: Any) -> dict[str, Any]:
        """The finished sample, with every unobserved metric stated as ``-1``."""
        out: dict[str, Any] = {
            "channel": channel,
            "arm": arm,
            "token": self.token,
            "submit_epoch": self.submit_epoch,
        }
        for name in M.METRICS:
            out[name] = float(self.marks.get(name, M.UNAVAILABLE))
        out.update(extra)
        return out


# ---------------------------------------------------------------------------
# Wire reading
# ---------------------------------------------------------------------------


async def read_until(
    lines: Any, predicate: Callable[[dict[str, Any]], bool], *, timeout: float = FRAME_TIMEOUT_S
) -> tuple[dict[str, Any], float]:
    """The first frame matching ``predicate``, with the instant it was PARSED.

    The timestamp is taken immediately after the line is decoded and before the
    predicate runs, because that decode is what the front end pays for.
    """

    async def read() -> tuple[dict[str, Any], float]:
        async for line in lines:
            if not line.startswith("data: "):
                continue
            at = time.monotonic()
            frame = json.loads(line[6:])
            if predicate(frame):
                return frame, at
        raise AssertionError("stream ended before the expected frame")

    return await asyncio.wait_for(read(), timeout)


def sse_emit_monotonic(frame: Mapping[str, Any], turn: "TurnMarks") -> float | None:
    """The frame's own ``ts`` (a daemon wall clock) as an instant on THIS process's clock.

    The SSE envelope stamps ``ts = time.time()`` when the event is PUBLISHED, and
    the client and the daemon are different processes, so the wall clock is the one
    clock they share. Converting through the turn's own submit pair keeps the
    reported delta on the harness's monotonic clock without pretending the two
    processes share one.

    A negative or absurdly large delta — a clock step, or a frame belonging to
    another turn — returns ``None``, which the driver reports as unavailable rather
    than as a suspiciously fast frame. (The bound is the turn timeout, not the frame
    timeout: a cold job on a loaded host legitimately takes minutes to publish its
    first delta, and a tighter bound would silently turn that into ``-1``.)
    """
    stamp = frame.get("ts")
    if not isinstance(stamp, (int, float)):
        return None
    delta = float(stamp) - turn.submit_epoch
    if delta < -1.0 or delta > TURN_TIMEOUT_S:
        return None
    return turn.submit_monotonic + delta


# ---------------------------------------------------------------------------
# In-process runtime instrumentation: the runtime-side emission stamp
# ---------------------------------------------------------------------------


class FrameStamp:
    """When the emitting side of a channel had its first paint-worthy frame ready.

    Installed by wrapping one named seam on a real object in THIS process and
    removed at teardown. It is the runtime-side half of the decomposition:
    everything before it is engage + runtime + IPC, everything after it is socket
    and parse. Keyed by session id so N concurrent turns are attributed to the
    session each was submitted on, and OVERWRITTEN each time so a priming turn's
    stamp cannot be read as the measured turn's.

    A renamed seam must not break the harness: the installers report
    ``unavailable_reason`` and return a no-op restore, so the column degrades to
    unavailable and the measurement survives.
    """

    def __init__(self) -> None:
        self.stamps: dict[str, float] = {}
        self.unavailable_reason = ""

    def note(self, session_id: str, paint: bool) -> None:
        if paint and session_id:
            self.stamps[session_id] = time.monotonic()


def install_desktop_frame_stamp(
    stamp: FrameStamp, is_paint: Callable[[Mapping[str, Any]], bool]
) -> Callable[[], None]:
    """Wrap ``DesktopSessionBridge.events`` so every frame it yields is examined.

    That generator is the daemon's serialization boundary for one session's stream:
    the frame has already crossed the child's IPC hop and is about to be written to
    the client, which makes it the exact line between "the runtime produced it" and
    "the client received it".
    """
    from local_operator.server.utils.desktop_sessions import DesktopSessionBridge

    original = getattr(DesktopSessionBridge, "events", None)
    if original is None:
        stamp.unavailable_reason = "DesktopSessionBridge.events is gone; runtime emit not measured"
        return lambda: None

    async def wrapped(self: Any, sub: Any, *, epoch: Any, after_seq: Any) -> Any:
        async for frame in original(self, sub, epoch=epoch, after_seq=after_seq):
            stamp.note(str(getattr(self, "session_id", "")), is_paint(frame))
            yield frame

    DesktopSessionBridge.events = wrapped  # type: ignore[method-assign]

    def restore() -> None:
        DesktopSessionBridge.events = original  # type: ignore[method-assign]

    return restore


def install_projection_frame_stamp(stamp: FrameStamp) -> Callable[[], None]:
    """Wrap ``mobile.daemon._projection_frame`` — the phone plane's serialization boundary.

    One call per repaint per session, so the last call before a frame reaches the
    phone is when the daemon had it ready to ship. Its argument carries the session
    id, which is what makes the stamp attributable under concurrency.
    """
    from local_operator.mobile import daemon as mobile_daemon

    original = getattr(mobile_daemon, "_projection_frame", None)
    if original is None:
        stamp.unavailable_reason = (
            "mobile.daemon._projection_frame is gone; runtime emit not measured"
        )
        return lambda: None

    def wrapped(projection: Any) -> Any:
        stamp.note(str(getattr(projection, "session_id", "")), True)
        return original(projection)

    mobile_daemon._projection_frame = wrapped  # type: ignore[assignment]

    def restore() -> None:
        mobile_daemon._projection_frame = original  # type: ignore[assignment]

    return restore


# ---------------------------------------------------------------------------
# Shared: the daemon plane
# ---------------------------------------------------------------------------


def bind_listener() -> socket.socket:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    return listener


async def serve_daemon(address: socket.socket) -> tuple[Any, asyncio.Task[Any]]:
    """Start the real daemon app on ``address`` in THIS process and await readiness."""
    import uvicorn

    from local_operator.server.app import app

    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[address]))
    for _ in range(1_000_000):
        if server.started:
            return server, task
        if task.done():
            await task
        await asyncio.sleep(0)
    raise AssertionError("daemon did not start")


async def stop_daemon(server: Any, task: asyncio.Task[Any], address: socket.socket) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(task, 30)
    except Exception:  # noqa: BLE001 — an unresponsive daemon is cancelled, not awaited
        task.cancel()
    address.close()


async def gather_or_raise(coros: list[Awaitable[Any]]) -> list[Any]:
    """Await all, re-raising the first failure.

    A failed turn must fail the run rather than vanish into a percentile: this
    harness's whole claim is that its numbers are measurements, and a run whose
    frames never arrived has no number to contribute.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise failures[0]
    return results


# ---------------------------------------------------------------------------
# Desktop
# ---------------------------------------------------------------------------


def desktop_paint(frame: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    """(paints, carries_text, carries_reasoning) for one desktop SSE frame.

    ``message_update`` with a non-empty ``delta`` is the frame the renderer
    appends, i.e. the one the user waits for.

    THE ROLE CHECK IS THE SAME ECHO EXCLUSION AS THE TUI SEAM (QA round 1, Q5).
    The runtime emits ``message_update`` for the user's own row too, and this
    channel reads the wire rather than the object, so a frame carrying the echo of
    what the operator just typed would stamp first paint at the submit round trip.
    A frame whose ``message`` carries no role at all is still counted, deliberately:
    refusing to count it would turn a shape change elsewhere into a silent zero in
    this column, and the reverse self-check in ``_reduce_cells`` is what catches a
    seam that then reads faster than the stream it comes from.
    """
    payload = frame.get("payload")
    if not isinstance(payload, Mapping):
        return False, False, False
    kind = str(payload.get("type") or "")
    if names_reasoning(kind):
        return True, False, True
    if kind == "message_update" and payload.get("delta"):
        message = payload.get("message")
        role = str(message.get("role") or "") if isinstance(message, Mapping) else ""
        if role and role != "assistant":
            return False, False, False
        return True, True, False
    return False, False, False


def is_admission_ack(frame: Mapping[str, Any], request_id: str) -> bool:
    """Is this frame THIS submit's admission acknowledgement?

    Matched on the caller's own ``request_id`` and not on the frame name alone: the
    stream is per session and carries every turn's frames, so a previous turn's
    acknowledgement arriving late would otherwise be read as this one's. The name is
    taken from the product's own constant rather than re-spelled, so a rename cannot
    silently un-gate the acknowledgement.
    """
    if not request_id:
        return False
    from local_operator.server.utils.desktop_sessions import ADMISSION_ACCEPTED_FRAME

    payload = frame.get("payload")
    kind = str(frame.get("type") or "")
    if kind != ADMISSION_ACCEPTED_FRAME:
        if not isinstance(payload, Mapping):
            return False
        if str(payload.get("type") or "") != ADMISSION_ACCEPTED_FRAME:
            return False
    if not isinstance(payload, Mapping):
        return False
    return str(payload.get("request_id") or "") == request_id


def classify_agent_event(event: Any) -> tuple[bool, bool]:
    """(carries_text, carries_reasoning) for one agent event at the front end.

    THE ROLE CHECK IS THE ECHO EXCLUSION, and it is load-bearing. The TUI (and the
    exec renderer, the desktop bridge and the phone projection) all echo the
    submitted user message back as a message update, so a seam that counted any
    delta would stamp first paint at the submit round trip — QA round 1, Q5 mutated
    exactly this and read **3 ms**, which would have passed every gate in the
    harness for the wrong reason. Only ASSISTANT deltas are model output; everything
    else about a turn (the user's own row, a custom row, a tool result) is something
    the front end already had, or something that is not thinking content either.

    Module level rather than closed over by the TUI child so that the exclusion can
    be tested: the finding was that nothing in this repository would notice the
    check being deleted. The product import is function-local like every other one in
    this file, so a bench that is not driving a channel pays nothing for it.
    """
    from local_operator.harness.types import MessageUpdateEvent

    if isinstance(event, MessageUpdateEvent):
        role = str(getattr(getattr(event, "message", None), "role", "") or "")
        if role != "assistant":
            return False, False
        return bool(getattr(event, "delta", "")), False
    name = ""
    for attr in ("type", "event_type", "name"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value:
            name = value
            break
    if names_reasoning(name) or names_reasoning(type(event).__name__):
        return False, True
    return False, False


async def drive_desktop(
    run: IsolatedRun,
    *,
    config: ChannelConfig,
    concurrency: int,
    arms: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """The desktop channel: N sessions on a real daemon, over real HTTP + SSE.

    THE ACKNOWLEDGEMENT IS READ OFF THE WIRE. This driver used to say that the
    admission ACK frame did not exist on this tree and that ``admitted_ms`` (the
    submit POST's return) was the stand-in until it did. It exists now — the
    ack-before-engage change landed it (``ADMISSION_ACCEPTED_FRAME``, emitted by the
    host while the engage is still starting) — so the ack mark comes from that frame,
    correlated on the submit's own request id, and the POST's return is only the
    fallback for a daemon that does not send one. On the cold arm those two are
    seconds apart, which is the whole point of the change.
    """
    import httpx

    address = bind_listener()
    server, task = await serve_daemon(address)
    base_url = f"http://127.0.0.1:{address.getsockname()[1]}"
    headers = {"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]}
    stamp = FrameStamp()
    restore = install_desktop_frame_stamp(stamp, lambda frame: desktop_paint(frame)[0])
    samples: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=300) as client:
            session_ids: list[str] = []
            for _ in range(concurrency):
                created = await client.post(
                    "/v1/desktop/sessions",
                    json={"request_id": str(uuid.uuid4()), "cwd": str(run.cwd)},
                )
                created.raise_for_status()
                session_ids.append(created.json()["result"]["session_id"])

            async def one(session_id: str) -> list[dict[str, Any]]:
                target = f"/v1/desktop/sessions/{session_id}"
                out: list[dict[str, Any]] = []
                async with client.stream("GET", target + "/events") as response:
                    lines = response.aiter_lines()
                    opened = (await read_until(lines, lambda f: f["type"] == "open"))[0]
                    await read_until(lines, lambda f: f["type"] == "snapshot")
                    await client.post(
                        target + "/watch",
                        json={
                            "subscription_id": opened["payload"]["subscription_id"],
                            "visible": True,
                            "can_notify": False,
                        },
                    )
                    for arm in arms:
                        if arm == "warm" and "cold" not in arms:
                            # ``warm`` means a LATER message of a session that
                            # already has a runtime and a transcript. When the cold
                            # arm also ran, its own completed turn IS that priming;
                            # asked for alone, the warm arm has to produce it,
                            # unmeasured — otherwise "warm" would measure a cold
                            # session and quietly mislabel it.
                            await client.post(
                                target + "/messages",
                                json={
                                    "request_id": str(uuid.uuid4()),
                                    "text": prompt_for(new_token()),
                                },
                            )
                            await read_until(
                                lines,
                                lambda f: f.get("type") == "event"
                                and f.get("payload", {}).get("type") == "agent_end",
                                timeout=TURN_TIMEOUT_S,
                            )
                        out.append(
                            await _desktop_submit(
                                client,
                                target,
                                lines,
                                session_id=session_id,
                                stamp=stamp,
                                arm=arm,
                                diagnostics=diagnostics,
                            )
                        )
                return out

            for batch in await gather_or_raise([one(s) for s in session_ids]):
                samples.extend(batch)
        diagnostics["desktop_frame_stamp"] = stamp.unavailable_reason or "installed"
        return samples
    finally:
        restore()
        await stop_daemon(server, task, address)


async def _desktop_submit(
    client: Any,
    target: str,
    lines: Any,
    *,
    session_id: str,
    stamp: FrameStamp,
    arm: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit one message and read to the first paint-worthy frame.

    The frame reader runs CONCURRENTLY with the submit POST, and that is
    load-bearing: reading only after the POST returned would date the arrival of
    socket-buffered frames to the read, which on the cold arm is ~2.6 s of the very
    cost under measurement.
    """
    token = new_token()
    turn = TurnMarks(token=token, submit_monotonic=time.monotonic(), submit_epoch=time.time())
    seen: list[str] = []
    # ONE request id, used by the POST and matched on the wire: the acknowledgement
    # frame carries the caller's own id, so an id generated separately for the
    # request would leave the frame unattributable.
    request_id = str(uuid.uuid4())

    def predicate(frame: Mapping[str, Any]) -> bool:
        # THE ACKNOWLEDGEMENT FRAME IS THE ACK, NOW THAT IT EXISTS. It is emitted by
        # the host while the engage is still starting
        # (``server/utils/desktop_sessions.py::ADMISSION_ACCEPTED_FRAME``), which is
        # what the ack-before-engage change added; the POST's return is the fallback
        # for a daemon that does not send it (``note_admitted`` keeps the first
        # stamp, so the frame wins when both arrive). This is the mark the merged
        # ack gate is scored on, and reading the POST instead would date the ack to
        # the HTTP receipt — seconds away on the cold arm, and not the frame the
        # operator's renderer paints.
        if is_admission_ack(frame, request_id):
            turn.note_admitted()
            seen.append("admission.accepted")
            return False
        paints, carries_text, carries_reasoning = desktop_paint(frame)
        seen.append(
            f"{frame.get('type')}/{(frame.get('payload') or {}).get('type')}"
            if isinstance(frame.get("payload"), Mapping)
            else str(frame.get("type"))
        )
        if paints or carries_text or carries_reasoning:
            turn.note_paint(
                paints=paints,
                carries_text=carries_text,
                carries_reasoning=carries_reasoning,
                emit_monotonic=stamp.stamps.get(session_id),
            )
            return True
        return False

    frame_task = asyncio.create_task(read_until(lines, predicate))
    post_task = asyncio.create_task(
        client.post(
            target + "/messages",
            json={"request_id": request_id, "text": prompt_for(token)},
        )
    )
    await frame_task
    response = await post_task
    response.raise_for_status()
    turn.note_admitted()
    # DRAIN TO THIS TURN'S END BEFORE RETURNING, and that is load-bearing rather
    # than tidy: this session's stream carries the SAME frame types for every turn,
    # so a reader that stops at the first paint-worthy frame leaves the rest of the
    # turn queued — and the next arm's predicate then matches a PREVIOUS turn's
    # delta. Measured: it reported the warm arm at 213 ms against a provider stamp
    # of 433 ms, which is impossible in that direction and is exactly the signature
    # of a leaked frame. Draining also makes ``turn_ms`` the turn's real end rather
    # than the submit's return.
    await read_until(
        lines,
        lambda f: f.get("type") == "event" and f.get("payload", {}).get("type") == "agent_end",
        timeout=TURN_TIMEOUT_S,
    )
    turn.note_turn_end()
    if diagnostics is not None and len(diagnostics.get("desktop_frames", [])) < 16:
        diagnostics.setdefault("desktop_frames", []).extend(seen)
    return turn.sample(arm=arm, channel="desktop", session=session_id)


# ---------------------------------------------------------------------------
# TUI — in-process, the floor channel
# ---------------------------------------------------------------------------


async def drive_tui(
    run: IsolatedRun,
    *,
    config: ChannelConfig,
    concurrency: int,
    arms: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the TUI arm in a fresh interpreter and parse its report.

    One interpreter per run, N sessions inside it, because the cost this channel
    exists to expose is per-PROCESS: a TUI is one long-lived process, so its first
    turn is the only one that pays the cold caches, and an in-process loop would
    measure the settled number for a cost the user meets once.
    """
    env = child_env(
        {
            "HOME": str(run.home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(run.config_dir),
            "TMPDIR": str(run.root),
        }
    )
    # ``child_env`` carries the notification gate (``harness_child_env``) with it:
    # the child below is a real TUI in a fresh interpreter, its session is one
    # nobody is watching, and the hosting's reply is a mock provider's canned line
    # — which is a notification body, and belongs nowhere near the operator.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "bench_ttft.py"),
        "--child-tui",
        "--concurrency-child",
        str(concurrency),
        "--arms",
        ",".join(arms),
        "--hosting",
        config.hosting,
        "--model",
        config.model,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    out, err = await proc.communicate()
    samples: list[dict[str, Any]] = []
    for line in out.decode().splitlines():
        if line.startswith("TTFT_JSON "):
            samples = json.loads(line[len("TTFT_JSON ") :])
    # ONE RETRY FOR THE STORE-INITIALISATION RACE, and only for that (QA round 1,
    # Q1). ``IsolatedRun.seed`` creates ``auth.db`` serially before this child
    # exists, which is the fix; this is the belt for the case where something else
    # reaches the file first — a single, named, retriable failure rather than a
    # blanket retry that would hide a real child crash behind a second attempt.
    attempts = 0
    while not samples and "database is locked" in err.decode() and attempts < 1:
        attempts += 1
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "bench_ttft.py"),
            "--child-tui",
            "--concurrency-child",
            str(concurrency),
            "--arms",
            ",".join(arms),
            "--hosting",
            config.hosting,
            "--model",
            config.model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await proc.communicate()
        for line in out.decode().splitlines():
            if line.startswith("TTFT_JSON "):
                samples = json.loads(line[len("TTFT_JSON ") :])
    if not samples:
        raise AssertionError(f"tui child reported no timing: {err.decode()[-2000:]}")
    diagnostics["tui_child_stderr"] = err.decode()[-800:]
    if attempts:
        diagnostics["tui_child_retried"] = attempts
    return samples


async def tui_child_main(
    *, concurrency: int, arms: tuple[str, ...], hosting: str, model: str
) -> int:
    """The child side of the TUI arm: N sessions, cold then warm, real handler seam.

    The front-end seam is ``Session.subscribe`` — the agent-event subscription the
    exec/headless renderer attaches to directly and the one the TUI's controller
    attaches to as well (``tui/events.py`` ``EventController.subscribe`` calls
    ``session.subscribe(self._on_event)``), so this is the TUI's own input boundary
    rather than a proxy for it.

    ONE QUALIFICATION, stated rather than glossed: the controller hands events to the
    widgets through a COALESCING FLUSH on a ``1/30 s`` timer
    (``tui/events.py`` ``FLUSH_INTERVAL_S``), so the frame the user sees is up to
    ~33 ms after the instant measured here. That is a bounded, UI-side cost this
    harness deliberately does not include — it is the same for every arm, and
    folding it in would require driving a real Textual app, which is a different
    kind of test.
    """
    import argparse as _argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.harness.types import StreamReasoningDelta
    from local_operator.session_factory import create_session, warm_session_imports

    # EXACTLY WHAT THE TUI DOES FIRST: the real app runs this off-loop at boot, and
    # it is the seam that carries the tokenizer warm. A benchmark that skipped it
    # would charge the tokenizer to the first turn.
    await asyncio.to_thread(warm_session_imports)

    args = _argparse.Namespace()
    for key, value in {
        "hosting": hosting,
        "model": model,
        "agent_name": None,
        "agent_id": None,
        "yolo": True,
        "train": False,
        "resume": None,
        "agent": None,
    }.items():
        setattr(args, key, value)

    config_dir = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    samples: list[dict[str, Any]] = []

    def classify(event: Any) -> tuple[bool, bool]:
        return classify_agent_event(event)

    async def one_session() -> list[dict[str, Any]]:
        session = await create_session(
            args,
            ConfigManager(config_dir),
            CredentialManager(config_dir),
            AgentRegistry(config_dir),
            cwd=str(config_dir),
        )
        holder: list[TurnMarks] = []
        try:

            def handler(event: Any) -> None:
                if not holder:
                    return
                carries_text, carries_reasoning = classify(event)
                if carries_text or carries_reasoning:
                    holder[-1].note_paint(
                        paints=True, carries_text=carries_text, carries_reasoning=carries_reasoning
                    )

            session.subscribe(handler)
            inner = cast(Any, session)._stream_fn

            class TimedStream:
                """Wraps the session's stream fn to see the runtime's OWN events.

                This is where a dropped reasoning delta is still visible: the
                provider produced it, the runtime's stream carries it, and no
                front-end handler is ever called for it.
                """

                def __getattr__(self, name: str) -> Any:
                    return getattr(inner, name)

                def __call__(self, request: Any, signal: Any = None) -> Any:
                    if holder:
                        holder[-1].note_stream_entered()
                    source = inner(request, signal)

                    async def relay() -> Any:
                        async for event in source:
                            if holder and isinstance(event, StreamReasoningDelta):
                                holder[-1].note_runtime_reasoning()
                            yield event

                    return relay()

            cast(Any, session)._stream_fn = TimedStream()

            out: list[dict[str, Any]] = []
            for arm in ("cold", "warm"):
                marks = TurnMarks(
                    token=new_token(),
                    submit_monotonic=time.monotonic(),
                    submit_epoch=time.time(),
                )
                holder.append(marks)
                try:
                    await session.prompt(prompt_for(marks.token))
                finally:
                    holder.pop()
                marks.note_turn_end()
                if arm in arms:
                    out.append(marks.sample(arm=arm, channel="tui"))
            return out
        finally:
            await session.dispose()

    for batch in await gather_or_raise([one_session() for _ in range(concurrency)]):
        samples.extend(batch)
    print("TTFT_JSON " + json.dumps(samples))
    return 0


# ---------------------------------------------------------------------------
# Exec / headless
# ---------------------------------------------------------------------------


async def drive_exec(
    run: IsolatedRun,
    *,
    config: ChannelConfig,
    concurrency: int,
    arms: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the real ``local-operator exec`` entry point and time its first JSON line.

    ``--json`` is the streaming surface for exec: the default one-shot mode prints
    the last assistant text only, so it has no first token to be early about, while
    ``--json`` writes one line per agent event as it arrives — which is what a
    supervisor reads.

    NO RUNTIME-SIDE STAMP, and that is a real limit rather than an omission: the
    measured process is a separate interpreter and exec's JSON line carries the
    event but no publish timestamp, so there is nothing to compare the client's
    receipt against. ``runtime_emit`` is ``-1`` for this channel. ``--yolo`` is
    passed so an approval gate can never park the turn; the loopback model calls no
    tool, so it has nothing to approve.

    ``warm`` here is a process launched after one has already run in this run
    group — see the module docstring on warmth.
    """
    entry = Path(sys.executable).parent / "local-operator"
    if not entry.exists():
        raise AssertionError(f"no console script at {entry}")
    diagnostics.setdefault("exec_entry", str(entry))
    env = child_env(
        {
            "HOME": str(run.home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(run.config_dir),
            "TMPDIR": str(run.root),
        }
    )
    # The real CLI below is a child of a bench, so it is gated the same way: see
    # ``child_env``, which routes through ``harness_child_env`` (the gate AND the
    # nested-session waiver the real invocation needs under an agent's shell).

    async def one(index: int, arm: str) -> dict[str, Any]:
        token = new_token()
        turn = TurnMarks(token=token, submit_monotonic=time.monotonic(), submit_epoch=time.time())
        proc = await asyncio.create_subprocess_exec(
            str(entry),
            "exec",
            prompt_for(token),
            "--json",
            "--yolo",
            cwd=str(run.cwd),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert proc.stdout is not None
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), TURN_TIMEOUT_S)
                if not line:
                    break
                at = time.monotonic()
                text = line.decode(errors="replace").strip()
                if not text.startswith("{"):
                    continue
                try:
                    event = json.loads(text)
                except ValueError:
                    continue
                kind = str(event.get("type") or "")
                if kind == "message_update" and event.get("delta"):
                    turn.note_paint_at(at, paints=True, carries_text=True)
                    break
                if names_reasoning(kind):
                    turn.note_paint_at(at, paints=True, carries_reasoning=True)
                    break
            await asyncio.wait_for(proc.wait(), TURN_TIMEOUT_S)
        finally:
            if proc.returncode is None:
                proc.kill()
        turn.note_turn_end()
        return turn.sample(arm=arm, channel="exec", index=index)

    batches = []
    if "warm" in arms and "cold" not in arms:
        # A LONE ``--arms warm`` IS A LIE ABOUT A PROCESS ARM (QA round 1, minor).
        # This channel spawns a real CLI process per sample, so "warm" means "a
        # process has already run in this run group and paid the bytecode and import
        # cost". With cold excluded nothing has, and every sample would be cold while
        # the table said warm. Relabelling would silently change what the arm MEANS,
        # so the harness pays that cost once, unmeasured, and says so.
        await one(0, "prime-unmeasured")
        diagnostics["exec_primed"] = (
            "a lone --arms warm ran one unmeasured turn first, so the arm is warm"
        )
    for arm in arms:
        batches.extend(await gather_or_raise([one(index, arm) for index in range(concurrency)]))
    return batches


# ---------------------------------------------------------------------------
# Mobile (the phone plane, over the daemon's real routes)
# ---------------------------------------------------------------------------


def phone_text(frame: Mapping[str, Any]) -> str:
    """The assistant-visible text of a phone projection frame."""
    rows = frame.get("transcript")
    if not isinstance(rows, list):
        return ""
    parts: list[str] = []
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("kind")) in (
            "assistant",
            "parent_message",
            "subagent_message",
        ):
            parts.append(str(row.get("text") or ""))
    return "".join(parts)


def phone_reasoning(frame: Mapping[str, Any]) -> str:
    """The reasoning-visible text of a phone projection frame.

    The agreed vocabulary for a reasoning row on the phone is a transcript row
    whose kind names it; no such kind exists on this tree, so this returns empty
    and the reasoning column reports ``-1`` — which is the finding.
    """
    rows = frame.get("transcript")
    if not isinstance(rows, list):
        return ""
    parts: list[str] = []
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("kind")).lower() in REASONING_ROW_KINDS:
            parts.append(str(row.get("text") or ""))
    return "".join(parts)


async def drive_mobile(
    run: IsolatedRun,
    *,
    config: ChannelConfig,
    concurrency: int,
    arms: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drive the real phone routes: start sessions, submit from the phone, read its SSE.

    The phone's only realtime channel is ``/api/sessions/{id}/events``, whose frames
    are the daemon's own ``_projection_frame`` payloads — the same bytes a handset
    parses — so this reads exactly what the phone would. The daemon spawns the
    runtime for a session it starts, which is the phone's real cold path.
    """
    from starlette.requests import Request

    from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
    from local_operator.mobile.daemon import MobileDaemon, build_app

    password = "bench-phone-password"
    daemon = MobileDaemon(password=password)
    app = build_app(daemon)
    scanner = asyncio.create_task(daemon.scan_loop())
    stamp = FrameStamp()
    restore = install_projection_frame_stamp(stamp)
    samples: list[dict[str, Any]] = []
    child_pids: list[int] = []

    def endpoint(path: str) -> Any:
        # ``app.routes`` is typed as ``BaseRoute``, which declares neither
        # ``path`` nor ``endpoint``: the daemon's own app is a Starlette app whose
        # routes are ``Route`` objects at runtime. Cast rather than re-implement the
        # lookup, so this drives the REAL registered endpoint.
        return next(
            cast(Any, route).endpoint for route in app.routes if cast(Any, route).path == path
        )

    def request_for(
        path: str,
        *,
        method: str = "GET",
        session_id: str = "",
        body: dict[str, Any] | None = None,
    ) -> Any:
        payload = json.dumps(body).encode("utf-8") if body is not None else b""
        headers = [
            (b"host", b"fixture"),
            (b"cookie", f"{COOKIE_NAME}={sign_cookie(password)}".encode()),
            (b"content-length", str(len(payload)).encode()),
        ]
        if body is not None:
            headers.append((b"content-type", b"application/json"))
        scope: dict[str, Any] = {
            "type": "http",
            "method": method,
            "path": path,
            "path_params": {"session_id": session_id} if session_id else {},
            "query_string": b"",
            "headers": headers,
            "scheme": "http",
            "server": ("fixture", 80),
            "client": ("127.0.0.1", 1),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": payload, "more_body": False}

        return Request(scope, receive)

    async def phone_json(path: str, body: dict[str, Any], session_id: str = "") -> Any:
        response = await endpoint(path)(
            request_for(
                path.replace("{session_id:str}", session_id),
                method="POST",
                session_id=session_id,
                body=body,
            )
        )
        return json.loads(bytes(response.body))

    async def read_frame(
        iterator: Any, predicate: Callable[[dict[str, Any]], bool], *, timeout: float
    ) -> tuple[dict[str, Any], float]:
        async def read() -> tuple[dict[str, Any], float]:
            while True:
                chunk = await anext(iterator)
                for line in str(chunk).splitlines():
                    if not line.startswith("data: "):
                        continue
                    at = time.monotonic()
                    frame = json.loads(line[len("data: ") :])
                    if predicate(frame):
                        return frame, at
                # A keepalive comment yields nothing to match; keep reading.

        return await asyncio.wait_for(read(), timeout)

    async def command(session_id: str, op: str, text: str = "") -> Any:
        return await phone_json(
            "/api/sessions/{session_id:str}/command",
            {"op": op, "text": text, "command_id": str(uuid.uuid4()), "images": []},
            session_id=session_id,
        )

    async def one() -> list[dict[str, Any]]:
        started = await phone_json("/api/sessions/start", {"cwd": str(run.cwd)})
        session_id = str(started["session_id"])
        child_pids.append(int(started["pid"]))
        response = await endpoint("/api/sessions/{session_id:str}/events")(
            request_for(f"/api/sessions/{session_id}/events", session_id=session_id)
        )
        iterator = response.body_iterator
        # Drain the seed frame: the baseline must be the state BEFORE this submit,
        # or the previous turn's text would read as this turn's first event.
        seed, _ = await read_frame(iterator, lambda f: True, timeout=FRAME_TIMEOUT_S)
        baseline = phone_text(seed)
        out: list[dict[str, Any]] = []
        seen: list[str] = []
        for arm in arms:
            if arm == "warm" and "cold" not in arms:
                # ``warm`` means a later message of a session that already has a
                # runtime and a transcript. When the cold arm also ran, its own
                # completed turn IS that priming; asked for alone, the warm arm has
                # to produce it, unmeasured.
                await command(session_id, "prompt", prompt_for(new_token()))
                primed, _ = await read_frame(
                    iterator, lambda f: bool(f.get("stop_reason")), timeout=TURN_TIMEOUT_S
                )
                baseline = phone_text(primed)
            token = new_token()
            turn = TurnMarks(
                token=token, submit_monotonic=time.monotonic(), submit_epoch=time.time()
            )

            #: The projection is CUMULATIVE, so after a turn ends every later frame
            #: still carries ``stop_reason``. "The turn ended" therefore cannot be
            #: read off the frame alone — it is the first end marker seen AFTER this
            #: turn has been observed to start, which is what this flag tracks. Without
            #: it the warm arm matched the previous turn's final repaint 13 ms in and
            #: reported a turn that never streamed.
            started = {"value": False}

            def observe(frame: Mapping[str, Any], at: float) -> bool:
                """Record what this frame shows, and say whether the turn is OBSERVED OUT.

                The phone's projection is a repaint protocol, so a turn produces
                several frames that matter and the FIRST of them is not the last
                observation this harness owes the reader: the activity line arrives
                before any text, and stopping there would leave ``first_text``
                unmeasured on the phone forever. So this keeps reading until both the
                first paint and the first text are recorded, and is bounded by the
                turn's own end so a channel that never carries text cannot hang.
                """
                text = phone_text(frame)
                reasoning = phone_reasoning(frame)
                activity = str(frame.get("activity") or "")
                streaming = bool(frame.get("streaming"))
                seen.append(activity or "idle")
                emit = stamp.stamps.get(session_id)
                if reasoning and reasoning != baseline:
                    turn.note_paint_at(at, paints=True, carries_reasoning=True, emit_monotonic=emit)
                if text and text != baseline:
                    turn.note_paint_at(at, paints=True, carries_text=True, emit_monotonic=emit)
                if activity and M.FIRST_PAINT not in turn.marks:
                    # The phone's working line. A real paint — the user sees motion
                    # — and deliberately NOT recorded as first TEXT; see
                    # TurnMarks.note_paint_at.
                    turn.note_paint_at(at, paints=True, emit_monotonic=emit)
                if streaming or activity or (text and text != baseline):
                    started["value"] = True
                if started["value"] and frame.get("stop_reason") and not streaming:
                    return True
                return M.FIRST_PAINT in turn.marks and M.FIRST_TEXT in turn.marks

            async def read_turn() -> Mapping[str, Any]:
                while True:
                    frame, at = await read_frame(iterator, lambda f: True, timeout=TURN_TIMEOUT_S)
                    if observe(frame, at):
                        return frame

            reader = asyncio.create_task(read_turn())
            reply = asyncio.create_task(command(session_id, "prompt", prompt_for(token)))
            stopped = await reader
            got = await reply
            if isinstance(got, Mapping) and got.get("ok") is False:
                raise AssertionError(f"phone refused the prompt: {got!r}")
            turn.note_admitted()
            # DRAIN TO THIS TURN'S END before the next arm starts. The phone's
            # stream is one long-lived SSE per session carrying the SAME frame
            # shape for every turn, so a reader that stopped at the first
            # paint-worthy frame would leave the rest of the turn queued and the
            # next arm's reader would match a PREVIOUS turn's text. It also makes
            # ``turn_ms`` the turn's real end rather than the submit's return.
            # Skipped when the observation loop was already stopped by the end
            # frame, which is the common case, so the drain cannot wait for the
            # NEXT turn's end.
            if not stopped.get("stop_reason"):
                stopped, _ = await read_frame(
                    iterator, lambda f: bool(f.get("stop_reason")), timeout=TURN_TIMEOUT_S
                )
            baseline = phone_text(stopped)
            turn.note_turn_end()
            out.append(turn.sample(arm=arm, channel="mobile", session=session_id))
        if len(diagnostics.get("mobile_activities", [])) < 16:
            diagnostics.setdefault("mobile_activities", []).extend(seen[:16])
        return out

    try:
        for batch in await gather_or_raise([one() for _ in range(concurrency)]):
            samples.extend(batch)
        diagnostics["mobile_frame_stamp"] = stamp.unavailable_reason or "installed"
        return samples
    finally:
        restore()
        scanner.cancel()
        try:
            await scanner
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown, and a
            # cancelled task re-raises CancelledError, which is a BaseException and
            # would otherwise escape the harness's teardown on every single run.
            pass
        try:
            await daemon.close_phone_views()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass
        dial_tasks = list(getattr(daemon, "_dial_tasks", {}).values())
        for dial_task in dial_tasks:
            dial_task.cancel()
        await asyncio.gather(*dial_tasks, return_exceptions=True)
        for pid in child_pids:
            try:
                os.kill(pid, 15)
            except (ProcessLookupError, PermissionError, OSError):
                pass


# ---------------------------------------------------------------------------
# SSE jobs
# ---------------------------------------------------------------------------


def jobs_paint(frame: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    """(paints, carries_text, carries_reasoning) for one jobs-SSE frame.

    No role check is needed here, and that is a property of the vocabulary rather
    than an oversight: ``message.delta`` is documented on the server's own event
    table as *incremental assistant text* (``server/utils/sse.py``), so a jobs frame
    cannot be the user's echo. If that contract ever widens, the reverse self-check
    in ``_reduce_cells`` (a frame cannot precede the stream it comes from) is the
    tripwire.
    """
    kind = str(frame.get("type") or "")
    if names_reasoning(kind):
        return True, False, True
    if kind == "message.delta" and frame.get("delta"):
        return True, True, False
    return False, False, False


async def drive_sse_jobs(
    run: IsolatedRun,
    *,
    config: ChannelConfig,
    concurrency: int,
    arms: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """``POST /v1/chat/async`` then follow ``GET /v1/sse/jobs/{id}`` — the jobs plane.

    Each request spawns a job PROCESS, so this channel shares the cold-engage shape
    with the desktop arm and ``warm`` means a request processed after one has
    already run in this run group (warm caches, fresh process).

    The runtime-side stamp is the frame's own ``ts``: the SSE envelope stamps it
    when the daemon PUBLISHES the event, which is after the job process crossed the
    queue hop and before the socket write. That makes ``ts`` this channel's
    daemon-side emission with no instrumentation at all.
    """
    import httpx

    address = bind_listener()
    server, task = await serve_daemon(address)
    base_url = f"http://127.0.0.1:{address.getsockname()[1]}"
    samples: list[dict[str, Any]] = []

    async def one() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: list[str] = []
        async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
            for arm in arms:
                token = new_token()
                turn = TurnMarks(
                    token=token, submit_monotonic=time.monotonic(), submit_epoch=time.time()
                )
                posted = await client.post(
                    "/v1/chat/async",
                    json={
                        "prompt": prompt_for(token),
                        "hosting": config.hosting,
                        "model": config.model,
                    },
                )
                posted.raise_for_status()
                job_id = posted.json()["result"]["id"]
                turn.note_admitted()
                async with client.stream("GET", f"/v1/sse/jobs/{job_id}") as response:
                    lines = response.aiter_lines()

                    def predicate(frame: Mapping[str, Any]) -> bool:
                        paints, carries_text, carries_reasoning = jobs_paint(frame)
                        seen.append(str(frame.get("type")))
                        if paints or carries_text or carries_reasoning:
                            turn.note_paint(
                                paints=paints,
                                carries_text=carries_text,
                                carries_reasoning=carries_reasoning,
                                emit_monotonic=sse_emit_monotonic(frame, turn),
                            )
                            return True
                        return False

                    await read_until(lines, predicate, timeout=TURN_TIMEOUT_S)
                    # Drain to the terminal frame: this stream is closed and
                    # discarded after the read either way, but ``turn_ms`` would
                    # otherwise be "when the first text arrived" wearing the name of
                    # the turn's end.
                    await read_until(
                        lines,
                        lambda f: str(f.get("type")) in ("stream.terminal", "stream.gap"),
                        timeout=TURN_TIMEOUT_S,
                    )
                turn.note_turn_end()
                out.append(turn.sample(arm=arm, channel="sse-jobs"))
        if len(diagnostics.get("jobs_frames", [])) < 16:
            diagnostics.setdefault("jobs_frames", []).extend(seen[:16])
        return out

    try:
        if "warm" in arms and "cold" not in arms:
            # Same defect, same fix as the exec arm (QA round 1, minor): each request
            # spawns a fresh job PROCESS, so warm means the shared bytecode cache has
            # already been paid for. Nothing else pays for it when cold is excluded.
            async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
                priming = await client.post(
                    "/v1/chat/async",
                    json={
                        "prompt": "prime the caches",
                        "hosting": config.hosting,
                        "model": config.model,
                    },
                )
                priming.raise_for_status()
                priming_id = priming.json()["result"]["id"]
                async with client.stream("GET", f"/v1/sse/jobs/{priming_id}") as response:
                    await read_until(
                        response.aiter_lines(),
                        lambda frame: str(frame.get("type")) in ("stream.terminal", "stream.gap"),
                        timeout=TURN_TIMEOUT_S,
                    )
            diagnostics["sse_jobs_primed"] = (
                "a lone --arms warm ran one unmeasured request first, so the arm is warm"
            )
        for batch in await gather_or_raise([one() for _ in range(concurrency)]):
            samples.extend(batch)
        return samples
    finally:
        await stop_daemon(server, task, address)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One measurable channel: its arms and how to drive them."""

    name: str
    drive: Callable[..., Awaitable[list[dict[str, Any]]]]


#: The channels this harness drives, by the name the CLI accepts and the table
#: prints. Each entry's driver shares the signature
#: ``drive(run, *, config, concurrency, arms, diagnostics)``.
CHANNELS: dict[str, Channel] = {
    "tui": Channel("tui", drive_tui),
    "desktop": Channel("desktop", drive_desktop),
    "exec": Channel("exec", drive_exec),
    "mobile": Channel("mobile", drive_mobile),
    "sse-jobs": Channel("sse-jobs", drive_sse_jobs),
}


async def drive_channel(
    name: str,
    run: IsolatedRun,
    *,
    config: ChannelConfig,
    concurrency: int,
    arms: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drive one channel, then merge the provider's own floor into every sample.

    The provider pairing happens HERE, for every channel, because it is the one
    measurement whose two halves live in different processes. It is the only
    comparison in this harness made on the wall clock: a submit's epoch against the
    provider's epoch. The provider's numbers are reported and never asserted (see
    ``metrics.BUDGET_MS``), which is what keeps a clock step from becoming a
    failure.
    """
    channel = CHANNELS[name]
    samples = await channel.drive(
        run, config=config, concurrency=concurrency, arms=arms, diagnostics=diagnostics
    )
    provider = config.provider
    for sample in samples:
        token = str(sample.get("token") or "")
        for kind, metric in (
            ("reasoning", M.PROVIDER_REASONING),
            ("text", M.PROVIDER_TEXT),
        ):
            if provider is None:
                sample[metric] = M.UNAVAILABLE
                continue
            sample[metric] = provider.stamps.delta_ms(kind, token, float(sample["submit_epoch"]))
    return samples
