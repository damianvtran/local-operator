"""The desktop ``/commands`` route completes every action receipt it declared.

THE INCIDENT THESE TESTS PIN. ``SLASH_ACTION_RECEIPTS`` is the shared
vocabulary of receipts that carry a ``request`` for the INVOKING client to
submit as a user turn, and the ownership rule is one predicate: a client that
DECLARED the type in its attach auth frame submits the request itself, and the
runtime stands down. The desktop viewer is a declaring client
(``AttachedSession`` dials with ``slash_consumers=list(SLASH_ACTION_RECEIPTS)``
on every surface), so the desktop host owes the submit — but this route spelled
the types out by hand as ``{"team_attached", "agent_attached"}``. #796 added
``goal_set`` to the tuple, taught the runtime and the TUI to complete it, and
left this third host behind: ``/goal <text>`` stored the standing goal and
dropped the turn, with no user row and no error.

So there are three guards here, deliberately of different kinds:

* a BEHAVIOURAL one over the REAL router — the goal command admits its argument
  through the route, and a type added to the vocabulary tomorrow is admitted
  without this file knowing its name;
* a BOUND one — the reply does not park on a running turn's durable append, and
  reports in ``admission.status``/``admission.detail`` what the OWNER said: the
  owner's own acknowledgement, ``pending`` when it had not answered inside the
  bound, or ``failed`` when it answered with an error. A failure that lands
  after a ``pending`` receipt is published on the session's stream, because the
  receipt is gone by then and the user's text must not vanish into a log line;
* a STATIC one over the real source, in the spirit of
  ``tests/unit/tui/test_noop_consumers.py`` — the handler must reach its
  decision through the shared helper and must not name a receipt type itself,
  which is the shape of the defect that shipped.

The pair that pins the DECISION is "every vocabulary type is completed" and
"nothing outside the vocabulary is": the first alone is satisfied by a host that
admits everything, which is what an unguarded inversion of ``runtime_must_
complete`` degenerates into (it answers False both for a declared type and for a
non-action notice), and the second alone is satisfied by the drop that started
this.

The bridge is a double (this is a route-contract test); the assembled
HTTP + runtime path is covered end to end by
``tests/e2e/test_desktop_sessions.py``.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import gc
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_sessions
from local_operator.session import attached as attached_module
from local_operator.session.runtime.types import SLASH_ACTION_RECEIPTS

TOKEN = "desktop-goal-admission-token"
#: A real canonical session id shape (12 hex), the shape the pool resolves.
SESSION = "8fd6c6a40934"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"
#: A second id for the cells that issue two commands: the receipt store refuses
#: one id used with two different bodies, and that rule has its own tests.
CLEAR_ID = "33333333-3333-4333-8333-333333333333"
#: A 2x2 PNG, real bytes because the route bounds wire images on the way to the
#: owner (``decode_images``) and a placeholder that does not decode would leave
#: the image clause of the decision untested. Generated once, deliberately
#: inline: a fixture file would be a second thing to keep valid.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgADAALm"
    "AEAUQs4PAAAAAElFTkSuQmCC"
)
ROUTE_SOURCE = Path(desktop_sessions.__file__).read_text(encoding="utf-8")


def _wire_images() -> list[dict[str, str]]:
    return [{"data_b64": PNG_B64, "mime_type": "image/png"}]


async def until(predicate, *, timeout_s: float = 10.0) -> None:
    """Wait for state the code under test publishes, never on the clock.

    AGENTS.md, "Wait on the event, never on the clock": the deadline here is the
    cell's deadlock guard rather than the assertion, and the wait lasts exactly
    as long as the work does — every caller asserts the state it waited for
    afterwards.
    """
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.001)


def _goal_set_receipt(request: str) -> dict[str, Any]:
    """The receipt the runtime's ``_goal_slash`` returns for ``/goal <text>``.

    Spelled as the producer spells it (``serving.py::_goal_slash``) rather than
    imported, because the route's contract is the WIRE shape: a change to the
    producer that this route does not follow is what these tests exist to
    catch, and a shared constructor would hide it on both sides at once.
    """
    return {
        "kind": "notice",
        "text": "goal set",
        "style": "info",
        "data": {"type": "goal_set", "stored": request, "request": request},
    }


def _agent_cleared_receipt() -> dict[str, Any]:
    """The receipt ``serving.py::_agent_slash`` returns for ``/agent clear``.

    THE REAL empty-request action receipt: ``agent_attached`` is in the
    vocabulary, so it reaches the decision, and its ``request`` is empty because
    a detach carries no ask. (``goal_set`` is never emitted empty — the runtime's
    ``/goal`` returns the show notice first and the clear notice with no ``data``
    at all — so a goal-shaped empty receipt would test a shape no owner produces.)
    """
    return {
        "kind": "notice",
        "text": "this session uses its base instructions",
        "style": "info",
        "data": {"type": "agent_attached", "agent": "", "request": ""},
    }


class FakeRemote:
    """The viewer facade the command route reads, recording every call.

    The members the route touches, and the admissions are recorded rather than
    answered away: "did the host submit the request, and how" is the whole
    question these tests ask, and a double that admitted silently would answer it
    vacuously.

    THREE CONTROLS, because the route's answer now depends on what the OWNER does
    and when (review round 2, F1): ``ack_delay_s`` is a REAL wait before the ack
    — the shape of every socket round trip, and the thing no loop-turn budget can
    see; ``ack_error`` is the owner answering with a refusal; and ``park`` is an
    ack that never lands at all, which is how the "does not park" cells drive the
    reply that has to come back without it. ``fail_after_park`` is the parked ack
    that fails once the test releases it, i.e. the DETACHED failure the UI has to
    be told about.
    """

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        self.binds = 0
        self.is_streaming = False
        self.park: asyncio.Event | None = None
        self.ack_delay_s: float | None = None
        self.ack_error: BaseException | None = None
        self.fail_after_park: BaseException | None = None
        self.routed: list[tuple[str, str]] = []
        self.admissions: list[tuple[str, str]] = []
        self.steered: list[bool] = []

    async def bind_runtime(self) -> None:
        self.binds += 1

    async def route_shared_slash(self, command: str, args: str, images: Any = None):
        self.routed.append((command, args))
        return dict(self.receipt)

    async def admit_prompt(self, text: str, *, command_id: str, images: Any = None, steer=False):
        self.admissions.append((text, command_id))
        self.steered.append(steer)
        if self.park is not None:
            await self.park.wait()
            if self.fail_after_park is not None:
                raise self.fail_after_park
        if self.ack_delay_s is not None:
            # A real suspension, so the ack cannot land inside any budget
            # measured in loop turns: this is the socket round trip's shape.
            await asyncio.sleep(self.ack_delay_s)
        if self.ack_error is not None:
            raise self.ack_error
        # The two details the real owner answers with, verbatim.
        return ("steering queued" if steer else "prompt admitted", False)


class FakeBridge:
    """``DesktopSessionBridge``-shaped: the facade AND the lease under it.

    The lease half is not decoration (review round 2, F2). The route must HOLD
    this bridge for the admission's life, because the pool disposes the facade
    when its last user releases it and the reader pump then fails every pending
    request future — the pattern that manufactured this host's own failure out of
    a POST that happened to be its session's only user. So the double counts the
    holds and records the frames the route publishes for the UI, and the cells
    assert both: a hold per command, released exactly once, and an
    ``admission.failed`` frame for a detached failure.
    """

    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote
        self.users = 0
        self.releases = 0
        #: The real bridge detaches on its LAST release only (``users == 0``),
        #: so this is the count the cancellation cell reads: a leaked reference
        #: leaves it at 0 and a double release takes it to 2.
        self.detaches = 0
        #: A release that is still IN PROGRESS, for the second cancellation
        #: window: the real one waits on the bridge's lock (a concurrent cold
        #: ``acquire`` holds it across ``attach_existing``) and then, at
        #: ``users == 0``, on its own tear-down.
        self.release_park: asyncio.Event | None = None
        self.refreshes = 0
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def refresh_watch(self) -> None:
        self.refreshes += 1

    async def acquire(self) -> FakeRemote:
        self.users += 1
        return self.remote

    async def release(self) -> None:
        self.releases += 1
        if self.release_park is not None:
            await self.release_park.wait()
        self.users -= 1
        if self.users == 0:
            self.detaches += 1

    def publish(self, kind: str, payload: dict[str, Any], *, replay: bool = True) -> None:
        self.published.append((kind, payload))


class FakePool:
    """``DesktopSessions``-shaped: the route's only door to a session.

    The real pool's own admissions (the retirement latch) have their own tests;
    what matters here is that the route works THROUGH a bridge facade rather
    than calling a handler directly. ``refresh_watch`` lives on the BRIDGE (the
    route calls it there) and ``bind_runtime`` on the facade it holds, which is
    the real split and the reason this double yields both.
    """

    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote
        self.bridge = FakeBridge(remote)

    @contextlib.asynccontextmanager
    async def session(self, session_id: str):
        if session_id != SESSION:
            raise KeyError("Unknown session")
        yield self.bridge


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The REAL command router over a fake bridge and an isolated config root.

    ``LOCAL_OPERATOR_DESKTOP_TOKEN`` is what the router's bearer gate reads, and
    the redirected ``HOME``/config dir is the house rule for anything that
    builds the real app: the receipt store writes rows, so the isolation is not
    optional here even though no test asserts on those files.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    remote = FakeRemote(_goal_set_receipt("Preserve one identity"))
    pool = FakePool(remote)
    app.state.desktop_sessions = pool
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, remote, pool.bridge


async def _goal(
    client: AsyncClient,
    args: str,
    *,
    request_id: str = REQUEST_ID,
    images: list[dict[str, str]] | None = None,
):
    return await client.post(
        f"/v1/desktop/sessions/{SESSION}/commands",
        json={
            "request_id": request_id,
            "command": "goal",
            "args": args,
            "images": images or [],
        },
    )


# ---------------------------------------------------------------------------
# behaviour: the goal argument becomes a turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_goal_command_admits_its_argument_as_a_user_turn(desktop) -> None:
    """The reproduction, at the route: setting the goal AND submitting the text.

    Before this, the receipt came back with ``kind: notice``, the goal was
    stored, and the request was dropped — the TUI's single-Enter behaviour on
    the desktop path producing half of itself.
    """
    client, remote, _bridge = desktop

    response = await _goal(client, "Preserve one identity")

    assert response.status_code == 200, response.text
    result = response.json()["result"]["result"]
    assert result["admission"]["status"] == "admitted"
    # The command reached the RUNTIME (which is what stored the goal); a
    # shortcut to ``native_action`` would answer 200 with the goal unset.
    assert remote.routed == [("goal", "Preserve one identity")]
    # The ARGUMENT, with the caller's own request id: a second submission under
    # a fresh id would be a second turn the user never asked for.
    assert remote.admissions == [("Preserve one identity", REQUEST_ID)]


@pytest.mark.asyncio
async def test_the_receipt_keeps_its_goal_metadata_beside_the_admission(desktop) -> None:
    """``stored`` and the receipt text survive the admission.

    The goal metadata is what the caller reports the goal with, and the
    admission is additive: a host that replaced the receipt with its admission
    would leave the desktop composer unable to say the goal was set.
    """
    client, _remote, _bridge = desktop

    response = await _goal(client, "Preserve one identity")

    result = response.json()["result"]["result"]
    assert result["kind"] == "notice"
    assert result["text"] == "goal set"
    assert result["data"]["stored"] == "Preserve one identity"
    assert result["data"]["request"] == "Preserve one identity"


@pytest.mark.asyncio
async def test_a_goal_arriving_mid_turn_takes_the_steering_path(desktop) -> None:
    """A turn already running takes the text the way both other hosts take it.

    The TUI's ``_submit_prompt`` and ``serving.py::_complete_unconsumed_action``
    both steer when the session is streaming, and ``/goal <text>`` typed mid-turn
    in the terminal behaves that way, so the desktop host must not invent a third
    answer. It is also what keeps this reply OFF the running turn's ack: ``steer``
    answers on queue insertion where ``prompt`` waits for the durable append.
    """
    client, remote, _bridge = desktop
    remote.is_streaming = True

    response = await _goal(client, "Preserve one identity")

    assert response.status_code == 200, response.text
    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == "admitted"
    assert remote.steered == [True]
    # The owner's own ack, which says the text joined the work in flight rather
    # than starting a turn — the disposition ``status`` cannot carry.
    assert admission["detail"] == "steering queued"


@pytest.mark.asyncio
async def test_a_parked_ack_does_not_park_the_reply(desktop) -> None:
    """THE REGRESSION CELL: the reply must not wait for the owner's ack.

    The ack resolves on the owner's DURABLE APPEND, which the drain performs only
    when it reaches the command — so awaiting it while a turn runs parks this
    reply for the whole of that turn, past the client's 15 s ``ACK_TIMEOUT_S``.
    The caller was then told the owner was unavailable while the goal was set and
    the turn queued, and a retry under the same id read as indeterminate.

    Driven with an ack that never resolves at all: the reply coming back is the
    proof, since an implementation that awaited it would never return. The ten
    seconds are the TEST's deadlock guard, not a product bound — the assertion
    is on the reply's presence and wording, never on how long it took.

    AND THE WORDING IS THE FIX (review round 2, F1). This receipt cannot know
    whether the owner took the text, so it must not say it did: ``pending``, with
    a phrase that names the missing acknowledgement. The previous
    ``admitted; the owner's acknowledgement was still in flight`` asserted the
    admission in the same breath as admitting it had not been observed, which is
    the sentence a REFUSED admission was being handed.
    """
    client, remote, bridge = desktop
    remote.park = asyncio.Event()

    response = await asyncio.wait_for(_goal(client, "Preserve one identity"), timeout=10)

    assert response.status_code == 200, response.text
    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == desktop_sessions.PENDING_ADMISSION_STATUS
    # No turn was running, so this is the idle phrase, not the steer one: the two
    # must not be confused in either direction.
    assert admission["detail"] == desktop_sessions.PENDING_ADMISSION_DETAIL
    assert remote.admissions == [("Preserve one identity", REQUEST_ID)]
    # The admission is still in flight, so it is still HOLDING the bridge — the
    # whole point of the hold (F2): the pool cannot dispose the connection this
    # request is using while the reply has already gone out.
    assert bridge.users == 1
    # Release the parked ack so the detached task is not left pending at teardown.
    remote.park.set()
    await until(lambda: bridge.users == 0)
    # It SUCCEEDED, so nothing is announced: the user row is the confirmation.
    assert bridge.published == []


@pytest.mark.asyncio
async def test_a_parked_ack_mid_turn_is_reported_as_pending(desktop) -> None:
    """And the STEER phrase when a turn was running behind the parked ack.

    The distinction is the point of the two phrases: a renderer that promises
    "sends when this step finishes" needs the steer one, and the same status
    word covers both. Neither claims the owner accepted the text.
    """
    client, remote, _bridge = desktop
    remote.is_streaming = True
    remote.park = asyncio.Event()

    response = await asyncio.wait_for(_goal(client, "Preserve one identity"), timeout=10)

    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == desktop_sessions.PENDING_ADMISSION_STATUS
    assert admission["detail"] == desktop_sessions.PENDING_STEER_ADMISSION_DETAIL
    assert remote.steered == [True]
    remote.park.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_an_ack_that_takes_a_real_step_is_still_reported_verbatim(desktop) -> None:
    """THE POSITIVE HALF OF F1: a wait, not a loop-turn count.

    The owner's answer here arrives after a REAL suspension — the shape of every
    ``AttachClient`` socket round trip, and precisely what the loop-turn budget
    could not observe. The old code therefore fell through to its "still in
    flight" phrase on every socket answer while claiming a refusal arm existed;
    this cell fails if the receipt stops waiting for the owner, and the sibling
    below fails if a refusal that DOES arrive is answered `admitted`.

    Ten milliseconds is not the bound under test — it is long enough that no
    number of ``sleep(0)`` turns can see it, and short enough that the cell stays
    a unit test. ``ack_delay_s`` is deliberately not the bound's size: a cell
    that slept for the bound would assert the bound by waiting it out, which is
    the one thing the bound exists to avoid.
    """
    client, remote, bridge = desktop
    remote.ack_delay_s = 0.01

    response = await _goal(client, "Preserve one identity")

    assert response.status_code == 200, response.text
    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == desktop_sessions.ADMITTED_ADMISSION_STATUS
    assert admission["detail"] == "prompt admitted"
    # Settled inside the bound, so the hold is already given back — exactly once.
    assert bridge.users == 0 and bridge.releases == 1
    assert bridge.published == []


@pytest.mark.asyncio
async def test_a_refusal_after_a_real_step_is_reported_failed_not_admitted(desktop) -> None:
    """THE REFUSAL CELL: the owner says no, and the receipt passes it on.

    This is review round 2's own repro, as a test: a remote whose ack raises after
    one real step — a stand-in for any socket round trip — used to be answered
    ``HTTP 200 status: admitted`` with the failure reaching nothing but a WARNING
    log, while the user's text went nowhere. Now the caller learns, in the
    receipt it is holding, that the request was NOT admitted.

    It is deliberately NOT raised into the error arm. A raise would leave this
    request's receipt unfinished (``DesktopReceipts._claim``), and the caller's
    retry under the same id would then read "outcome is indeterminate" — the
    503-then-409 ladder this route exists to remove.
    """
    client, remote, bridge = desktop
    remote.ack_delay_s = 0.01
    remote.ack_error = ConnectionError("owner socket unreachable: [Errno 61]")

    response = await _goal(client, "Preserve one identity")

    assert response.status_code == 200, response.text
    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == desktop_sessions.FAILED_ADMISSION_STATUS
    # A vetted sentence, never the transport's text: the errno and the port are
    # what ``errors()`` refuses to echo for the same reason.
    assert admission["detail"] == "failed; the session owner could not be reached"
    assert "Errno 61" not in admission["detail"]
    # The goal is still set — that part of the command worked — but the turn was
    # never submitted, which is what the caller now knows.
    assert bridge.users == 0 and bridge.releases == 1
    # Settled within the bound, so the caller has it: no frame is needed.
    assert bridge.published == []


@pytest.mark.asyncio
async def test_a_detached_failure_is_published_where_the_ui_can_see_it(desktop) -> None:
    """A failure AFTER the receipt reaches the UI, not just the log (F1).

    The receipt has already answered ``pending`` by the time this one lands, so
    there is no caller left to tell — the failure used to be a ``logger.warning``
    and nothing else, invisible client-side (an op failure is an error frame to
    the CALLER, never a frame on the desktop plane). It is now published on the
    session's own stream, where the mounted viewer reads it.
    """
    client, remote, bridge = desktop
    remote.park = asyncio.Event()
    remote.fail_after_park = RuntimeError("owner unavailable")

    response = await asyncio.wait_for(_goal(client, "Preserve one identity"), timeout=10)

    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == desktop_sessions.PENDING_ADMISSION_STATUS
    assert bridge.published == []

    remote.park.set()
    await until(lambda: bridge.published)

    kind, payload = bridge.published[0]
    assert kind == desktop_sessions.ADMISSION_FAILED_FRAME
    assert payload == {
        "request_id": REQUEST_ID,
        "command": "goal",
        "status": desktop_sessions.FAILED_ADMISSION_STATUS,
        # The owner's own ``RuntimeError`` text is NOT echoed: only the two
        # vetted shapes are (``_admission_failure_detail``).
        "detail": "failed; the owner did not admit the request",
    }
    # And the hold is given back exactly once, on the same path.
    assert bridge.users == 0 and bridge.releases == 1


@pytest.mark.asyncio
async def test_the_admission_holds_the_bridge_until_it_settles(desktop) -> None:
    """THE F2 CELL: the reply returning must not release the lease (F2).

    The pool disposes the facade when its last user releases it, and the reader
    pump then fails every pending request future — so a POST that was its
    session's only user used to close the connection its own admission was still
    using, manufacturing a spurious failure and warning out of nothing. The route
    takes ONE reference per command and gives it back exactly once, which is what
    the counters here assert in both directions: held while in flight, released
    after.
    """
    client, remote, bridge = desktop
    remote.park = asyncio.Event()

    response = await asyncio.wait_for(_goal(client, "Preserve one identity"), timeout=10)
    assert response.status_code == 200, response.text
    assert bridge.users == 1, "the admission outlived its bridge reference"

    remote.park.set()
    await until(lambda: bridge.users == 0)
    assert bridge.releases == 1, "the hold must be given back exactly once"


@pytest.mark.asyncio
async def test_a_request_cancelled_inside_the_bound_still_gives_the_bridge_back(
    desktop,
) -> None:
    """THE CANCELLATION CELL (review round 3, F1): no exit may leak the hold.

    ``admit_receipt_request`` takes one reference and has exactly one releaser
    per path — itself on the settled path, the detached continuation otherwise.
    A CANCELLATION landing inside ``asyncio.wait`` made both unreachable, and the
    leak is permanent rather than slow: ``release`` is the only thing that drops
    ``users`` and the only trigger for ``_detach``, the pool's eviction path only
    ever considers bridges at ``users == 0``, and so the facade, the owner
    connection and its runtime stayed resident for the process's life. The
    reviewer's double of the real refcount read ``users=2 → door released →
    users=1, detach=0`` and, with a refusing ack, an unretrieved task exception.

    Three things are asserted here, and each was broken before the fix: the
    lease is given back exactly once, the admission is NOT released out from
    under itself while it is still in flight (round 2's F2 — an immediate release
    on this path would re-manufacture it), and the refusal that lands afterwards
    reaches the UI as a frame instead of the loop's exception log.
    """
    _client, remote, bridge = desktop
    remote.park = asyncio.Event()
    # The parked ack REFUSES once this cell releases it: an error is the shape
    # that escaped as an unretrieved task exception when nothing awaited it.
    remote.fail_after_park = ConnectionError("owner socket unreachable")
    entry_users = bridge.users

    # Installed around the work so the loop's own path for an unretrieved task
    # exception (``Task.__del__`` → ``call_exception_handler``) is read here
    # rather than only in a CI log.
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        request = asyncio.ensure_future(
            desktop_sessions.admit_receipt_request(
                bridge,
                "Preserve one identity",
                command="goal",
                command_id=REQUEST_ID,
                images=[],
            )
        )
        await until(lambda: bridge.users == entry_users + 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        # Cancelled mid-flight: the reference is neither dropped (that would
        # dispose the facade this admission is still using — round 2's F2) nor
        # leaked. The continuation that owns the call owns the release.
        assert bridge.users == entry_users + 1
        assert bridge.releases == 0

        # Let the parked ack refuse. The frame can only be published by
        # something that awaited the dispatched task, which is also what marks
        # its exception retrieved.
        remote.park.set()
        try:
            await until(lambda: bridge.releases)
        except TimeoutError:
            # Named, rather than left as a bare wait timeout: this is the leak
            # the cell exists for — before the fix a cancellation here left no
            # releaser at all, so nothing ever moved these numbers.
            raise AssertionError(
                "the cancelled request never gave its reference back: "
                f"users={bridge.users}, releases={bridge.releases}, "
                f"detaches={bridge.detaches}"
            ) from None
        await until(lambda: bridge.published)

        assert bridge.users == entry_users, "the cancelled request leaked its reference"
        assert bridge.releases == 1, "the reference must be given back exactly once"
        assert bridge.detaches == 1, "the last release detaches exactly once"
        kind, payload = bridge.published[0]
        assert kind == desktop_sessions.ADMISSION_FAILED_FRAME
        assert payload["request_id"] == REQUEST_ID
        assert payload["detail"] == "failed; the session owner could not be reached"

        # The cancelled request's own frame is what still references the
        # dispatched task; dropping it is what makes the check below able to fail
        # instead of passing on an object that is still alive.
        del request
        gc.collect()
        assert reported == [], (
            "a refusal that lands after the cancellation must be reported by the "
            f"continuation, never left to the loop: {reported}"
        )
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_a_request_cancelled_during_its_release_still_gives_the_reference_back(
    desktop,
) -> None:
    """THE SECOND CANCELLATION WINDOW (review round 3, F1): the release itself.

    The bound is not the only suspension point a cancellation can land in. A
    release waits on the bridge's lock — contended by every other route on the
    session, and held across a cold ``attach_existing`` by a concurrent
    ``acquire`` — and then, at ``users == 0``, on its own tear-down. A
    cancellation there propagates INTO the awaited release (a task's cancellation
    reaches the future it waits on), so the count was never decremented and the
    facade stayed resident: the same pin, reached by the other door. The release
    is therefore run shielded — the AWAIT is cancelled, the release is not.
    """
    _client, remote, bridge = desktop
    bridge.release_park = asyncio.Event()
    entry_users = bridge.users

    request = asyncio.ensure_future(
        desktop_sessions.admit_receipt_request(
            bridge,
            "Preserve one identity",
            command="goal",
            command_id=REQUEST_ID,
            images=[],
        )
    )
    await until(lambda: bridge.releases == 1)  # the reference is being given back
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    # The request is gone; the release it owed is not. Let it finish.
    bridge.release_park.set()
    try:
        await until(lambda: bridge.users == entry_users)
    except TimeoutError:
        raise AssertionError(
            "cancelling the request took its release with it: "
            f"users={bridge.users}, releases={bridge.releases}, "
            f"detaches={bridge.detaches}"
        ) from None
    assert bridge.releases == 1, "the reference must be given back exactly once"
    assert bridge.detaches == 1


@pytest.mark.asyncio
async def test_an_agent_clear_carries_no_request_and_starts_no_turn(desktop) -> None:
    """THE REAL empty-request case: ``agent_attached`` from ``/agent clear``.

    A detach is a receipt with no action behind it — the type IS in the
    vocabulary (so it reaches this decision), and the request is empty. Nothing
    is submitted with or without images: the body's staged images are the
    CALLER's, not the receipt's, so admitting them would open an image-only turn
    nobody asked for, as a paid provider call and a durable row.
    """
    client, remote, _bridge = desktop
    remote.receipt = _agent_cleared_receipt()

    response = await _goal(client, "clear")

    assert response.status_code == 200, response.text
    # The null says no admission was made. It is a DECLARED field of the response
    # WRAPPER (``OwnerCommandResult.admission``), which is where the null shape
    # comes from; the route writes an extra key onto a dump whose model allows
    # extras, so an absent key and a null are the same thing to a reader and the
    # null is what a client sees.
    assert response.json()["result"]["result"]["admission"] is None
    assert remote.admissions == []

    with_images = await _goal(client, "clear", request_id=CLEAR_ID, images=_wire_images())

    assert with_images.status_code == 200, with_images.text
    assert with_images.json()["result"]["result"]["admission"] is None
    assert remote.admissions == [], (
        "an action-less receipt must not open a turn on the strength of the "
        "caller's staged images"
    )


@pytest.mark.asyncio
async def test_a_status_notice_with_images_still_starts_no_turn(desktop) -> None:
    """The show form, with an image staged: still not an action.

    This is the cell that fails for an implementation which admits whenever the
    receipt type is merely not-declared-by-the-runtime: ``runtime_must_complete``
    answers False for a typeless notice too, so the bare inversion would open a
    turn here — a paid provider call for ``/goal`` alone with a pasted image.
    """
    client, remote, _bridge = desktop
    remote.receipt = {
        "kind": "notice",
        "text": "goal: Preserve one identity",
        "style": "info",
        "data": {"goal": "Preserve one identity"},
    }

    response = await _goal(client, "", images=_wire_images())

    assert response.status_code == 200, response.text
    assert response.json()["result"]["result"]["admission"] is None
    assert remote.admissions == []


@pytest.mark.asyncio
async def test_a_receipt_outside_the_vocabulary_never_opens_a_turn(desktop) -> None:
    """And a typed receipt that is not an action receipt at all.

    A type outside ``SLASH_ACTION_RECEIPTS`` is not this host's to complete —
    the runtime never stood down for it, so admitting here would be the second
    submission of a command a future client declares as its own.
    """
    client, remote, _bridge = desktop
    remote.receipt = {
        "kind": "block",
        "text": "",
        "style": "info",
        "data": {"type": "session_listing", "request": "not an action"},
    }

    response = await _goal(client, "", images=_wire_images())

    assert response.status_code == 200, response.text
    assert response.json()["result"]["result"]["admission"] is None
    assert remote.admissions == []


# ---------------------------------------------------------------------------
# the drift guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_type_added_to_the_shared_vocabulary_is_completed_here(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE DRIFT GUARD: the next receipt type cannot be silently half-adopted.

    ``SLASH_ACTION_RECEIPTS`` is extended in one place, and every host of the
    rule must follow from that same place. This cell extends the vocabulary
    itself — to a name that exists nowhere in the route's source — and requires
    the route to admit it. A route that spells the types out by hand fails here
    on the type the author of the NEXT entry would have forgotten, rather than
    months later in a bug report about a dropped request.

    WHAT IT PINS, exactly: that the decision is not a STALE LIST. It is
    satisfied by an implementation that admits everything, which is why its
    partner cell — nothing outside the vocabulary admits, with images staged —
    is asserted beside it rather than assumed. The monkeypatched name is in the
    tuple on purpose: with the vocabulary gate in front of the predicate, an
    added name stays completable by a host that declares the whole list.

    BOTH HALVES ARE EXTENDED, because that is what a real extension does: the
    declaration this host reads is DEFINED from the vocabulary
    (``attached.py::ATTACHED_SLASH_CONSUMERS``), so a build that adds a type to
    one has added it to the other before any request runs. Extending only the
    vocabulary would describe a state no build can be in and would make this
    cell assert that the host claims a type its own client declared away —
    which is the behaviour review round 2's NIT-1 asked for, not a regression.
    """
    from local_operator.session import attached as attached_module
    from local_operator.session.runtime import types as runtime_types

    extended = (*runtime_types.SLASH_ACTION_RECEIPTS, "synthetic_attached")
    monkeypatch.setattr(runtime_types, "SLASH_ACTION_RECEIPTS", extended)
    monkeypatch.setattr(attached_module, "ATTACHED_SLASH_CONSUMERS", extended)
    client, remote, _bridge = desktop
    remote.receipt = {
        "kind": "notice",
        "text": "attached",
        "style": "info",
        "data": {"type": "synthetic_attached", "request": "Do the new thing"},
    }

    response = await _goal(client, "Do the new thing")

    assert response.status_code == 200, response.text
    assert remote.admissions == [("Do the new thing", REQUEST_ID)]


@pytest.mark.asyncio
async def test_every_declared_action_receipt_is_completed_by_this_route(
    desktop,
) -> None:
    """Every name in the vocabulary, driven through the route itself.

    Parametrising over the tuple rather than over a written-out list is the
    point: an entry added to ``SLASH_ACTION_RECEIPTS`` is covered by this cell
    the moment it is added, with no edit here.
    """
    client, remote, _bridge = desktop

    assert SLASH_ACTION_RECEIPTS, "the vocabulary is empty; this audit has gone blind"
    for index, receipt_type in enumerate(SLASH_ACTION_RECEIPTS):
        remote.admissions.clear()
        # A fresh id per type: the receipt store refuses one id used with two
        # different bodies, and that refusal is a different rule's test.
        request_id = f"22222222-2222-4222-8222-{index:012d}"
        remote.receipt = {
            "kind": "notice",
            "text": "receipt",
            "style": "info",
            "data": {"type": receipt_type, "request": f"request for {receipt_type}"},
        }

        response = await _goal(client, f"request for {receipt_type}", request_id=request_id)

        assert response.status_code == 200, response.text
        assert remote.admissions == [(f"request for {receipt_type}", request_id)], (
            f"{receipt_type} is declared in SLASH_ACTION_RECEIPTS but this route "
            "does not submit its request; the runtime stood down on the "
            "declaration, so the turn is dropped in silence"
        )


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """The module's top-level functions, by name.

    Top level only: the route handlers and this helper's subject all live there,
    and descending would make ``command`` ambiguous with any future nested
    function of the same name.
    """
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(node: ast.AST) -> set[str]:
    """The names CALLED anywhere under ``node``.

    Structural rather than textual, deliberately: the first version of these
    cells searched the handler's source segment for the helper's NAME, and a
    COMMENT naming the helper satisfies a string search — a guard that a rewrite
    can talk its way past. Only a real call is a real call.
    """
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def _loaded_names(node: ast.AST) -> set[str]:
    """The module-level names READ anywhere under ``node``."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _assignment_value(source: str) -> ast.expr:
    """The right-hand side of a one-line probe assignment in ``source``.

    A probe rather than an inline ``ast.parse(...).body[0].value`` so the
    expression has a home the type checker can read ``.value`` from: the parsed
    body is a statement, and only an ``ast.Assign`` exposes one.
    """
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.Assign), "the probe source must be one assignment"
    return node.value


def _slashed_consumer_dials(source: str) -> list[ast.expr]:
    """Every expression ``source`` passes as a ``slash_consumers=`` keyword."""
    return [
        keyword.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "slash_consumers"
    ]


#: The declaration the auth frame must dial with, as a PARSED SHAPE rather than
#: as source text (review round 3, NIT-1): ``black`` re-wraps that call — it
#: already sits at the argument-width boundary — without changing the fact it
#: states, and a substring match failed on the re-wrap alone. ``ast.dump``
#: compares the expression, so line breaks, trailing commas and comments under it
#: are all free.
_EXPECTED_DIAL = ast.dump(_assignment_value("_dialed = list(ATTACHED_SLASH_CONSUMERS)"))


def test_the_command_handler_routes_its_decision_through_the_helper() -> None:
    """The static half: the handler decides through the helper, not by name.

    WHAT THIS CAN AND CANNOT SEE. It holds the SHAPE that made the defect
    reachable — a local tuple of type names beside a shared vocabulary that grew
    without it — by requiring a real CALL to the helper and refusing any receipt
    type spelled out in the handler. It cannot see whether the decision is
    right: a handler that admitted unconditionally and happened to call the
    helper would pass here. That half is the behavioural pair above, which is
    why "nothing outside the vocabulary admits" exists as a cell rather than as
    a remark.
    """
    tree = ast.parse(ROUTE_SOURCE)
    handler = _functions(tree)["command"]

    assert "desktop_viewer_must_submit" in _called_names(handler), (
        "the desktop command route no longer CALLS desktop_viewer_must_submit; a "
        "hand-written set of receipt types is how /goal <text> lost its turn"
    )
    named = sorted(
        {
            node.value
            for node in ast.walk(handler)
            if isinstance(node, ast.Constant) and node.value in SLASH_ACTION_RECEIPTS
        }
    )
    assert not named, (
        f"the command route spells out receipt types ({named}) instead of keying "
        "off SLASH_ACTION_RECEIPTS; the next type added to the vocabulary will be "
        "dropped here"
    )


def test_the_shared_helper_is_derived_from_the_shared_vocabulary() -> None:
    """And the helper itself stays a reading of the ONE rule.

    ``runtime_must_complete`` is that rule and the vocabulary is the list it is
    applied to. Both clauses are asserted separately because they are the two
    halves the review separated: membership in the vocabulary is "is this an
    action receipt at all" (the half whose absence made the inversion admit
    every notice), and the shared predicate is "did the client declare it". A
    helper that consulted anything else — a private set, a configuration flag —
    would answer for this host while the runtime answered for the shared seam,
    and would still leave the runtime completing a subset a future client
    declares.
    """
    from local_operator.session.attached import ATTACHED_SLASH_CONSUMERS
    from local_operator.session.runtime.types import runtime_must_complete

    tree = ast.parse(ROUTE_SOURCE)
    helper = _functions(tree)["desktop_viewer_must_submit"]

    assert "SLASH_ACTION_RECEIPTS" in _loaded_names(helper), (
        "desktop_viewer_must_submit must read the shared vocabulary "
        "(SLASH_ACTION_RECEIPTS), not a list of its own"
    )
    assert "runtime_must_complete" in _called_names(helper), (
        "the completion decision must be the shared rule, so this host and the "
        "runtime cannot answer differently about the same receipt"
    )
    # AND THE DECLARATION IT IS ASKED ABOUT, rather than the vocabulary a second
    # time. ``runtime_must_complete(t, SLASH_ACTION_RECEIPTS)`` is a constant
    # False for everything the first clause admits, so the second clause decided
    # nothing at all (review round 2, NIT-1) — the code was shaped to satisfy
    # this guard, which cannot see that. Reading the value the client actually
    # DIALS with makes the clause decide, and the cell below pins the assumption
    # that used to be unenforceable.
    assert "ATTACHED_SLASH_CONSUMERS" in _loaded_names(helper), (
        "the second clause must read the declaration the client dials with; "
        "passed the vocabulary instead it is inert, and this host would claim a "
        "submit its own client declared away"
    )
    # The two directions the route relies on, asserted on the predicate itself:
    # an undeclaring client leaves the submit to the runtime, a declaring one
    # (this host) takes it.
    assert runtime_must_complete("goal_set", []) is True
    assert runtime_must_complete("goal_set", ATTACHED_SLASH_CONSUMERS) is False


def test_the_client_declares_every_action_receipt_this_route_claims() -> None:
    """The assumption the route's second clause rests on, ENFORCED.

    ``desktop_viewer_must_submit`` answers "this host declared it" by reading
    ``ATTACHED_SLASH_CONSUMERS`` — the list ``AttachedSession`` really puts in its
    auth frame. If a future edit narrowed that declaration, this host would stand
    down for the missing types (correctly) while the docstring still claimed the
    whole vocabulary, and the symptom would be a receipt nobody completes: the
    very drop this route was written to repair. A subset declaration is therefore
    a test failure, from whichever side it is introduced.

    It also pins the DISCOVERY of the previous shape: the declaration must not be
    written out as a second literal, which is one more list to forget.
    """
    from local_operator.session.attached import ATTACHED_SLASH_CONSUMERS

    assert set(SLASH_ACTION_RECEIPTS) <= set(ATTACHED_SLASH_CONSUMERS), (
        "the attached client must declare every action receipt, or this route "
        "stands down for a type nothing else completes"
    )
    attached_source = Path(attached_module.__file__).read_text(encoding="utf-8")
    # STRUCTURAL, not a substring (review round 3, NIT-1): the fact is that this
    # expression is what the auth frame dials with, and a re-wrap of the call
    # changes the text without touching the fact. Exactly ONE such keyword is
    # also the point — a second one is a declaration the route reads but the
    # client does not send.
    dials = _slashed_consumer_dials(attached_source)
    assert [ast.dump(dial) for dial in dials] == [_EXPECTED_DIAL], (
        "the auth frame must declare the shared constant exactly once — "
        "``slash_consumers=list(ATTACHED_SLASH_CONSUMERS)``; a second literal "
        "here is a declaration the route reads but the client does not send"
    )
