"""A `/goal <text>` issued while a turn is RUNNING: answered, never parked.

WHY THIS IS ITS OWN CELL. The mid-turn path fails differently from the two
neighbours beside it. The dropped-receipt bug (``test_desktop_sessions.py``) lost
the text and there was no turn to notice; this one consumes the receipt
CORRECTLY — the goal is stored and the turn is queued — and the reply is what
breaks: ``admit_prompt`` acks on the owner's durable transcript append, which the
drain performs only when it reaches the command, so awaiting it parks the reply
for the whole of the turn already running. Past the client's ``ACK_TIMEOUT_S``
(15 s) the caller is told the owner is unavailable, and its retry under the same
request id reads as indeterminate — a 503-then-409 ladder for a command that
worked.

So the property under test is a BOUND, and only a real blocked turn can show it:
the provider stream is held open, and the `/commands` reply arriving while it is
still held is the proof, since an implementation that awaited the ack would not
return at all. Everything else is the production stack — real HTTP, real runtime,
real transcript, real loopback attach — with only the model's replies scripted.

THE DISPOSITION IS ASSERTED, NOT ACCEPTED (review round 2, F1). The ack crosses a
real ``AttachClient`` socket here, and this cell asserts the OWNER'S OWN sentence
(``steering queued`` / ``prompt admitted``) rather than whichever of two phrases
the host happened to produce — the previous form passed on the host's own
fallback wording, which is how a REFUSED admission came to be answered
``admitted`` with nobody noticing. Both legs of the bound are exercised: the
STEER leg with a turn held open, and the IDLE leg with nothing running, whose ack
resolves on the turn's own start. The elapsed times are PRINTED rather than
asserted: they are the measurement the bound was sized from, and a wall-clock
assertion on a shared box is the flaky test AGENTS.md names.
"""

import asyncio
import os
import secrets
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from local_operator.server.app import app
from local_operator.server.routes import desktop_sessions as desktop_routes
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e

GOAL_TEXT = "Preserve one identity"
#: A SECOND goal, for the IDLE leg, deliberately a single word: both legs COUNT
#: the durable user rows carrying a text, and a phase-1 text that CONTAINS a
#: phase-2 one makes the phase-2 count match phase 1's row as well. The first
#: draft of this leg used "Preserve the second identity" — a substring of the
#: goal-context wrapper the runtime stores around the goal — and the count then
#: read 2 for one submission. A count whose subject can match another row's text
#: is not a count.
IDLE_GOAL_TEXT = "Idleness"

#: A real 2x2 PNG. The STEER leg carries it, deliberately: an owner that has to
#: bound an image does so through a thread hop BEFORE its reservation and before
#: any refusal (``serving.py::_image_blocks_async``), and a thread hop is exactly
#: what the loop-turn budget could not cover — it was already paid for once
#: (``serving.py:3427-3440``, measured "not done at 2 sleep(0)s, done by 20")
#: and it must not come back as a `pending` receipt for image-carrying goals.
#: Generated inline rather than stored: a fixture file would be a second thing to
#: keep valid, and real bytes are what the bound decodes.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgADAALm"
    "AEAUQs4PAAAAAElFTkSuQmCC"
)


async def until(predicate, *, timeout_s: float = 15.0) -> None:
    """Wait for state the code under test publishes, never on the clock.

    AGENTS.md, "Wait on the event, never on the clock": the deadline is the
    cell's deadlock guard rather than the assertion, and every caller asserts the
    state it waited for afterwards.
    """
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.001)


def _user_rows(entries: list[dict[str, Any]], text: str) -> int:
    """The durable USER MESSAGE rows whose own text is exactly ``text``.

    Filtered to message rows with a matching ``content`` rather than a substring
    search over the whole row, because a durable row is not only a message: the
    frontend state checkpoint embeds the session's GOAL verbatim, so a substring
    count over every entry counts the checkpoint as well as the turn and reports
    two rows for one submission. The question these cells ask is "how many user
    turns carried this text", and that is a question about message rows.
    """
    return sum(
        1
        for entry in entries
        if entry.get("type") == "message"
        and entry.get("payload", {}).get("role") == "user"
        and text
        in [
            block.get("text", "")
            for block in entry.get("payload", {}).get("content", [])
            if isinstance(block, dict)
        ]
    )


def _introduced_in(stream: ScriptedStream, text: str) -> list[int]:
    """The call indexes where ``text`` FIRST appears — i.e. was delivered.

    Every turn after the one that carried it replays the conversation as history,
    so "the text is in this request" is true for the rest of the run. What these
    cells count is DELIVERY, which is a transition: one submitted request is one
    index, and a duplicated submission would show as a second one.
    """
    seen = False
    indexes = []
    for index, request in enumerate(stream.requests):
        present = text in request.model_dump_json()
        if present and not seen:
            seen = True
            indexes.append(index)
    return indexes


def request_id() -> str:
    return str(uuid.uuid4())


class HeldStream(ScriptedStream):
    """Scripted turns whose FIRST call is held open until the test releases it.

    The hold is the whole instrument: the turn is streaming and cannot end until
    ``release()``, so any reply that came back came back without waiting for it.
    Later calls replay the script untouched, which is what lets the cell assert
    that the steered goal text reaches the provider exactly once.
    """

    def __init__(self, turns) -> None:
        super().__init__(turns)
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self._held = False

    def __call__(self, request, signal=None):
        if self._held:
            return super().__call__(request, signal)
        self._held = True
        self.requests.append(request)

        async def held():
            self.started.set()
            await self.released.wait()
            for event in text_turn("The held turn finished."):
                yield event

        return held()

    def release(self) -> None:
        self.released.set()


@pytest.mark.asyncio
async def test_a_goal_mid_turn_is_answered_without_waiting_for_the_turn(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = handle = None
    try:
        await until(lambda: server.started)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            created = await client.post(
                "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = "/v1/desktop/sessions/" + sid

            # Five turns: the held one, the follow-up turn that has to carry the
            # steered goal text, the steer goal's judge, the idle goal's own
            # turn, and the idle goal's judge. A call past the end is a test bug
            # and ``ScriptedStream`` fails loudly on it rather than answering with
            # a bare stop — which is how this cell learned its tape had gone
            # short when the judged goal started forking a judge.
            stream = HeldStream(
                [
                    text_turn("The held turn finished."),
                    text_turn("Carried the steer."),
                    # The STEER goal's forked judge, which answers ACHIEVED so it
                    # settles that goal instead of admitting a continuation: this
                    # cell is about the admission BOUND, and a chain of extra
                    # turns would make its call census a measurement of the judge
                    # rather than of the steer.
                    text_turn("VERDICT: ACHIEVED\nSteered goal verified."),
                    text_turn("Carried the idle goal."),
                    # ...and the IDLE goal's judge, for the same reason. Without
                    # these two, the judged-goal machinery reads prose as an
                    # unreadable verdict and keeps asking.
                    text_turn("VERDICT: ACHIEVED\nIdle goal verified."),
                ]
            )
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            # Named up front, deliberately: an unnamed conversation fires the
            # owner's separate title-model errand, which is a REAL provider call
            # on this same stream — it would be the call this cell holds open,
            # leaving the turn itself queued behind it and the instrument
            # measuring the wrong thing.
            session.set_conversation_name("Mid-turn goal cell", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            # A real turn, streaming and held open.
            started = await client.post(
                target + "/messages",
                json={"request_id": request_id(), "text": "Start a long turn"},
            )
            assert started.status_code == 200, started.text
            await asyncio.wait_for(stream.started.wait(), 15)
            # The turn is streaming and cannot finish: the hold is still set and
            # no second call has been made. Asserted on this cell's own
            # instrument rather than on a session flag, because the flag is the
            # runtime's business and the hold is the fact under test.
            assert not stream.released.is_set()
            assert len(stream.requests) == 1

            # THE ASSERTION THIS FILE EXISTS FOR: the goal command is answered
            # while that turn is still held. The ten seconds are the test's own
            # deadlock guard, not a product bound -- the reply either arrives
            # without the turn or never arrives at all.
            #
            # THE PROBE IS RELEASED BEFORE THE COMMAND, deliberately (review
            # round 2, F2). Holding a bridge ACROSS the POST kept the pool's
            # `users` off zero for the whole command, so the last-user detach
            # could not happen and the cell could not see the race it exists for.
            # Here the POST is its session's ONLY user: its own release is the
            # last one, which is exactly the path that used to dispose the facade
            # — and close the client — under an admission still in flight.
            #
            # The reading the probe takes is what makes the steer choice
            # deterministic: the route reads the same owner state immediately
            # before its dispatch, so a lagging frontend update cannot send the
            # command down the idle path and make the disposition accidental.
            pool = app.state.desktop_sessions
            async with pool.session(sid) as probe:
                await until(lambda: bool(probe.remote.is_streaming))
            goal_id = request_id()
            commanded_at = time.monotonic()
            answered = await asyncio.wait_for(
                client.post(
                    target + "/commands",
                    json={
                        "request_id": goal_id,
                        "command": "goal",
                        "args": GOAL_TEXT,
                        # The image rides the command so the STEER path pays the
                        # owner-side thread hop the loop-turn budget could not
                        # cover; see ``PNG_B64``.
                        "images": [{"data_b64": PNG_B64, "mime_type": "image/png"}],
                    },
                ),
                timeout=10,
            )
            steer_elapsed = time.monotonic() - commanded_at
            assert answered.status_code == 200, answered.text
            assert (
                not stream.released.is_set()
            ), "the turn was released early; the bound is untested"
            admission = answered.json()["result"]["result"]["admission"]
            # THE OWNER'S OWN ACK, by name, on a real socket. Not "whichever of
            # two phrases the host happens to produce": the disposition is what
            # the OWNER answered, and the bound exists to observe it. A pending
            # phrase here means the ack was lost or never waited for, and the
            # idle-path phrase means the turn in flight was not recognised.
            assert admission["status"] == "admitted", admission
            assert admission["detail"] == "steering queued", admission["detail"]
            # The bridge is back to zero users the instant the reply is in hand:
            # the extra reference the admission took was given back on the
            # settled path, exactly once, and the lease is not leaked.
            assert pool.bridges[sid].users == 0, "the admission's hold was not released"
            # The goal itself landed on the owner.
            assert session.goal == GOAL_TEXT

            # And the retry under the same id is a REPLAY, not the 409 that the
            # parked path produced once the ack deadline had passed. The body is
            # repeated BYTE FOR BYTE — the receipt store fingerprints the whole
            # body, so a retry that omitted the image would be a 409 for "already
            # used with different input", which is the store working, not this
            # route failing.
            retried = await client.post(
                target + "/commands",
                json={
                    "request_id": goal_id,
                    "command": "goal",
                    "args": GOAL_TEXT,
                    "images": [{"data_b64": PNG_B64, "mime_type": "image/png"}],
                },
            )
            assert retried.status_code == 200, retried.text
            assert retried.json()["result"]["replayed"] is True

            # RELEASE THE TURN: the steered goal text must reach the provider
            # exactly once, in the follow-up call that carries it — asserted below,
            # where the two calls can be told apart (the text appearing in a later
            # request proves nothing, because history is replayed into it).
            stream.release()
            await until(lambda: len(stream.requests) >= 2)
            entries = (await client.get(target + "/history")).json()["result"]["entries"]
            assert (
                _user_rows(entries, GOAL_TEXT) == 1
            ), "one durable user row for one goal submission"
            # DELIVERED ONCE, to the provider as well: the steer's text was ABSENT
            # from the call that started the turn and appears in exactly one call
            # after it. Counted rather than pinned to an index because the judged
            # goal now forks the judge at the same turn end, so which of the two
            # subsequent calls carries the text first is the judge's business —
            # what must hold is that ONE call carries it and the held turn did not.
            delivered = _introduced_in(stream, GOAL_TEXT)
            assert len(delivered) == 1, (
                "the steered goal text must reach the provider exactly once, never"
                f" twice: delivered in calls {delivered}"
            )
            assert delivered[0] >= 1, (
                "and NOT in the call that started the turn, which was already"
                f" streaming when the goal was set: delivered in call {delivered[0]}"
            )

            # PHASE 2: THE IDLE LEG, the same route with no turn running. The ack
            # there is the TURN'S OWN START rather than a queue insertion, which
            # is the other leg of the bound -- and it is the leg that proves the
            # wait is real, because nothing is ahead of the command: an
            # implementation that only gave the dispatch a few loop turns would
            # answer `pending` here while the turn had in fact started.
            #
            # Idleness is asserted on the RUNTIME's own flag rather than on a
            # fresh facade's: a new bridge's ``is_streaming`` starts False and is
            # filled in from the canonical snapshot, so a wait on it could pass
            # before the sync that would have said otherwise. Waiting on the
            # runtime removes that race from the instrument instead of hoping it
            # does not fire.
            await until(lambda: not session.is_streaming, timeout_s=20)
            idle_id = request_id()
            idle_at = time.monotonic()
            idled = await asyncio.wait_for(
                client.post(
                    target + "/commands",
                    json={"request_id": idle_id, "command": "goal", "args": IDLE_GOAL_TEXT},
                ),
                timeout=10,
            )
            idle_elapsed = time.monotonic() - idle_at
            assert idled.status_code == 200, idled.text
            idle_admission = idled.json()["result"]["result"]["admission"]
            assert idle_admission["status"] == "admitted", idle_admission
            # ``prompt admitted`` is the owner's own receipt for the ``prompt`` op
            # -- the ack that resolves on the durable append, i.e. the turn's own
            # start.
            assert idle_admission["detail"] == "prompt admitted", idle_admission["detail"]
            # WAIT FOR THE DELIVERY ITSELF, not for a request count: the steer
            # goal's judge and the continuation it may admit are calls too, so a
            # count of three can be reached by the machinery around the command
            # rather than by the command's own turn.
            await until(lambda: _introduced_in(stream, IDLE_GOAL_TEXT), timeout_s=20)
            idle_delivered = _introduced_in(stream, IDLE_GOAL_TEXT)
            assert len(idle_delivered) == 1, (
                "the idle goal's text must reach the provider in exactly one call,"
                f" never twice: delivered in calls {idle_delivered}"
            )
            assert idle_delivered[0] >= 2, (
                "and in a call made AFTER the command — the steer leg's own calls"
                f" were already made by then: delivered in call {idle_delivered[0]}"
            )
            idle_entries = (await client.get(target + "/history")).json()["result"]["entries"]
            assert (
                _user_rows(idle_entries, IDLE_GOAL_TEXT) == 1
            ), "one durable user row for the idle goal submission"
            assert (
                _user_rows(idle_entries, GOAL_TEXT) == 1
            ), "and the idle leg added no second row for the steer's text"
            print(
                "HTTP /goal mid-turn: answered 200 while the turn was still held (no 503, no "
                f"ack-deadline wait); admission {admission['status']}/{admission['detail']!r} "
                f"from the owner in {steer_elapsed * 1000:.1f} ms (steer leg, bound "
                f"{desktop_routes._ADMISSION_ACK_BOUND_S:g} s); goal stored; retry replayed; "
                "exactly one provider call carried the text and exactly one durable row "
                f"recorded it; idle leg answered {idle_admission['status']}/"
                f"{idle_admission['detail']!r} in {idle_elapsed * 1000:.1f} ms"
            )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
