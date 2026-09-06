"""The runtime completes an attach receipt's request when the client will not.

**This file exists because `/team lopdev <request>` attached the team and then
dropped the request in total silence, live, against a shipped runtime.**

The mechanism is a version seam rather than a typo, which is why no existing
test could see it. Since #624 the owner answers a routed `/team <name>
<request>` with a ``team_attached`` receipt carrying the request, and the
VIEWER is expected to submit that request as a user turn. A viewer built
before #624 prints the receipt text and has no consumer for
``data["request"]`` — so the team was attached, "sending to <team>. <manager>
is coordinating." was printed, and no user row and no turn ever appeared.

That pairing is not exotic on a developer host: ``lop-update`` replaces the
on-disk install several times a day, a long-lived TUI keeps running the code
it loaded, and the runtime it spawns resolves ``sys.executable`` fresh — so an
OLD terminal routinely drives a NEW runtime. The repair is that the runtime
admits the request itself for any client that did not declare it consumes the
receipt type.

The invariant these tests pin, in both directions:

* **undeclared ⇒ admit** — the request has no other home, so the runtime runs
  it. ``None`` (older client, no field) and ``[]`` (declared, consumes
  nothing) are both undeclared.
* **declared ⇒ do not admit** — the client submits it, and a runtime that also
  admitted would run the user's command twice, which is worse than the bug.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.types import SLASH_ACTION_RECEIPTS
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from tests.unit.session.runtime.test_owned import FakeSession, make_handle


class _Receipt:
    """The ``SlashResult`` slice the completion path reads.

    A stand-in rather than the real pydantic model so a test can produce a
    receipt shape directly without routing a whole slash command through the
    session — the completion predicate is what is under test here, not the
    ``/team`` grammar (``test_owned.py`` and the e2e stage cover that end).
    ``model_copy`` is reproduced because the failure path uses it.
    """

    def __init__(self, kind: str, text: str, data: dict[str, Any] | None, style: str = "info"):
        self.kind = kind
        self.text = text
        self.style = style
        self.data = data

    def model_copy(self, *, update: dict[str, Any]) -> "_Receipt":
        clone = _Receipt(self.kind, self.text, self.data, self.style)
        for key, value in update.items():
            setattr(clone, key, value)
        return clone


def _attach_receipt(receipt_type: str = "team_attached", request: str = "do the thing") -> _Receipt:
    """The exact shape ``owned.py::_team_attach_slash`` returns on attach."""
    return _Receipt(
        kind="notice",
        text="sending to lopdev. manager is coordinating.",
        data={
            "type": receipt_type,
            "team": "lopdev",
            "manager": "manager",
            "request": request,
        },
    )


class _TeamSession(FakeSession):
    """A session double that can genuinely attach a team.

    ``_team_attach_slash`` refuses outright when the session has no
    ``attach_team`` — an owner that cannot attach must not let a request run
    with no roster while the receipt claims a manager is coordinating. So the
    double grows the seam rather than the guard being weakened for the test:
    the receipt these cells assert on only exists on a session that really
    attached. A SUBCLASS rather than attributes bolted on at runtime, so the
    shape is part of the double's declared contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.team_registry: Any = None
        self.attached_teams: list[str] = []
        self.active_team_name: str = ""

    def attach_team(self, team: Any) -> None:
        name = getattr(team, "name", str(team))
        self.attached_teams.append(name)
        self.active_team_name = name


@contextlib.asynccontextmanager
async def _team_handle() -> AsyncIterator[tuple[OwnedSessionHandle, _TeamSession]]:
    """A handle over a session that really can attach ``lopdev``."""
    session = _TeamSession()
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    with tempfile.TemporaryDirectory() as tmp:
        registry = TeamRegistry(Path(tmp))
        registry.create_team(
            TeamEditFields(
                name="lopdev",
                description="d",
                manager="manager",
                members=[TeamMember(role="coder")],
            )
        )
        session.team_registry = registry
        yield handle, session


async def _settle() -> None:
    """Let the handle's prompt drain task reach the session.

    Admission is durable-first: ``prompt`` queues a ``_PromptCommand`` and the
    drain forwards it. Both are on this loop, so yielding is enough — no
    wall-clock budget, which would be a bet on machine load rather than a
    test.
    """
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_an_undeclared_receipt_has_its_request_admitted() -> None:
    """THE INCIDENT, at the unit seam: ``consumers=None`` is the old viewer.

    A client that never heard of ``slash_consumers`` sends no declaration, so
    the runtime must run the request itself. Before the fix this returned the
    receipt and nothing else happened anywhere.
    """
    handle, session = make_handle()
    images = [{"data_b64": "aGk=", "mime_type": "image/png"}]

    result = await handle._complete_unconsumed_action(_attach_receipt(), images, None)

    await _settle()
    assert session.prompt_calls == ["do the thing"], (
        "the runtime must admit the request for a client that did not declare "
        "the receipt type; this is the silent drop the incident reported"
    )
    assert result.style == "info", "a successful admission must not warn"
    assert result.text == "sending to lopdev. manager is coordinating.", (
        "the receipt copy is unchanged because it is now TRUE — the request " "really was sent"
    )


@pytest.mark.asyncio
async def test_declaring_nothing_is_not_the_same_as_consuming() -> None:
    """``consumers=[]`` also admits: the rule is ``type not in declared``.

    Spelled as its own cell because the tempting implementation — "admit when
    the field was absent" — passes the test above and fails here, leaving a
    client that declares an empty list silently dropping every request.
    """
    handle, session = make_handle()

    await handle._complete_unconsumed_action(_attach_receipt(), None, [])

    await _settle()
    assert session.prompt_calls == ["do the thing"]


@pytest.mark.asyncio
async def test_a_declared_receipt_is_never_admitted_by_the_runtime() -> None:
    """The double-submission guard, and the reason the declaration exists.

    A current viewer submits the request from its own terminal so the row
    carries its images and paste expansion. If the runtime also admitted, the
    user would see one typed command produce two turns with no way to tell
    which is which — strictly worse than the bug being repaired.
    """
    handle, session = make_handle()

    result = await handle._complete_unconsumed_action(
        _attach_receipt(), None, list(SLASH_ACTION_RECEIPTS)
    )

    await _settle()
    assert session.prompt_calls == [], "a declared receipt must be left to the client"
    assert session.steer_calls == []
    assert result.style == "info"


@pytest.mark.asyncio
async def test_a_receipt_with_no_request_is_not_an_action() -> None:
    """``/agent clear`` returns ``agent_attached`` with ``request: ""``.

    A detach is a receipt with no action behind it. Admitting an empty prompt
    would start a turn with no words in it.
    """
    handle, session = make_handle()

    await handle._complete_unconsumed_action(
        _attach_receipt("agent_attached", request=""), None, None
    )

    await _settle()
    assert session.prompt_calls == []
    assert session.steer_calls == []


@pytest.mark.asyncio
async def test_a_non_action_receipt_passes_through_untouched() -> None:
    """Only the action-carrying types are completed.

    Every other routed slash — ``/goal``, ``/rename``, a refusal notice —
    returns through this same path and must be handed back byte-identical.
    """
    handle, session = make_handle()
    plain = _Receipt(kind="notice", text="goal set", data={"type": "goal"})

    result = await handle._complete_unconsumed_action(plain, None, None)

    await _settle()
    assert result is plain
    assert session.prompt_calls == []


@pytest.mark.asyncio
async def test_a_busy_session_is_steered_rather_than_prompted() -> None:
    """Mid-turn, the completion STEERS — mirroring what a viewer does.

    ``prompt`` rejects a concurrent call, so admitting one while a turn runs
    would throw the request away. Steering delivers it at the engine's next
    boundary, which is the existing mid-turn channel and the same choice
    ``app.py::_submit_prompt`` makes.
    """
    handle, session = make_handle()
    session.is_streaming = True

    await handle._complete_unconsumed_action(_attach_receipt(), None, None)

    await _settle()
    assert session.steer_calls == ["do the thing"]
    assert session.prompt_calls == []


@pytest.mark.asyncio
async def test_a_failed_admission_reports_a_warning_instead_of_silence() -> None:
    """The attach landed; only the turn did not start. Say so.

    Session closing, a full queue, a rejected prompt: the attach has already
    happened and stays, so the receipt is rewritten as a WARNING naming the
    failure. That is the whole difference from the original defect — the user
    learns the request did not run and can resend, rather than watching
    nothing happen.
    """
    handle, session = make_handle()

    async def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("session is closing; prompt was not admitted")

    handle.prompt = refuse  # type: ignore[method-assign]

    result = await handle._complete_unconsumed_action(_attach_receipt(), None, None)

    assert result.style == "warning"
    assert "the request was not sent" in result.text
    assert "session is closing" in result.text, "the warning must name the actual failure"
    assert result.text.startswith("sending to lopdev."), "the attach receipt is still reported"
    assert session.prompt_calls == []


@pytest.mark.asyncio
async def test_the_full_routed_path_admits_for_an_undeclared_client() -> None:
    """End of the handle's own path: ``run_slash_authoritative`` wires it up.

    The cells above drive the predicate directly. This one goes through the
    public method the server actually calls, so a completion that works but is
    never invoked cannot pass.
    """
    async with _team_handle() as (handle, session):
        outcome = await handle.run_slash_authoritative(
            "team", "lopdev do the thing", [], consumers=None
        )

        await _settle()
        assert outcome["data"]["type"] == "team_attached"
        assert outcome["data"]["request"] == "do the thing"
        assert session.attached_teams == ["lopdev"], "the attach still happens on the owner"
        assert session.prompt_calls == ["do the thing"], (
            "the routed path must complete an undeclared client's request; this "
            "is the exact call the RuntimeServer makes for a pre-#624 viewer"
        )


@pytest.mark.asyncio
async def test_the_full_routed_path_defers_to_a_declaring_client() -> None:
    """The same public path, with the declaration present: no admission.

    The ATTACH still happens — that is the owner's job either way. Only the
    request submission is left to the client.
    """
    async with _team_handle() as (handle, session):
        outcome = await handle.run_slash_authoritative(
            "team", "lopdev do the thing", [], consumers=list(SLASH_ACTION_RECEIPTS)
        )

        await _settle()
        assert outcome["data"]["request"] == "do the thing"
        assert session.attached_teams == ["lopdev"]
        assert session.prompt_calls == []


def test_the_declared_set_matches_what_the_producers_emit() -> None:
    """A cheap sanity pin on the constant itself.

    The static audit in ``tests/unit/tui/test_noop_consumers.py`` proves the
    set equals the producers' and the renderer's. This asserts the value is
    non-empty and stringy, so a refactor that empties it fails HERE with an
    obvious message rather than only as a silent no-op in the completion path.
    """
    assert SLASH_ACTION_RECEIPTS
    assert all(isinstance(item, str) and item for item in SLASH_ACTION_RECEIPTS)


@pytest.mark.asyncio
async def test_images_ride_the_completed_request() -> None:
    """Wire images survive the completion; only paste BODIES degrade.

    The viewer sends resolved image blocks in the ``slash_result`` frame, so
    they are already on this side and go into the admission unchanged. This is
    the documented half of the degradation note: a screenshot the request
    cites still reaches the manager as pixels.
    """
    handle, session = make_handle()
    captured: list[Any] = []

    async def capture(text: str, images: Any = None, command_id: str | None = None) -> str:
        captured.append(images)
        session.prompt_calls.append(text)
        return "ok"

    handle.prompt = capture  # type: ignore[method-assign]

    await handle._complete_unconsumed_action(
        _attach_receipt(), [{"data_b64": "aGk=", "mime_type": "image/png"}], None
    )

    assert session.prompt_calls == ["do the thing"]
    assert captured and captured[0], "the wire images must reach the admission"


@pytest.mark.asyncio
async def test_each_completion_uses_a_fresh_command_id() -> None:
    """Two completions must not collide on one durable command identity.

    The admission path is producer-keyed for replay safety: reusing an id
    would make the second request read as a duplicate of the first and be
    dropped — reintroducing the silence on the second ``/team`` of a session.
    """
    handle, session = make_handle()
    ids: list[str | None] = []

    async def capture(text: str, images: Any = None, command_id: str | None = None) -> str:
        ids.append(command_id)
        session.prompt_calls.append(text)
        return "ok"

    handle.prompt = capture  # type: ignore[method-assign]

    await handle._complete_unconsumed_action(_attach_receipt(), None, None)
    await handle._complete_unconsumed_action(_attach_receipt(request="again"), None, None)

    assert len(ids) == 2
    assert all(ids), "every admission needs an explicit command id"
    assert ids[0] != ids[1], "a reused id makes the second request read as a duplicate"


class _SlowDrainSession(FakeSession):
    """A session whose admission resolves only after a prior turn finishes.

    Models the real shape: ``Session`` resolves a prompt's ``admitted`` future
    on the durable transcript append, which the drain performs when it reaches
    that command — so a turn queued ahead holds the next admission open. The
    flag matters too: ``is_streaming`` is still False during the drain's
    pre-streaming prelude, which is the window the steer guard cannot see.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.started: list[str] = []

    # ``message_id``/``admitted`` are load-bearing on this double, not
    # decoration. The handle probes for ``message_id`` (``owned.py``'s
    # ``legacy_prompt`` check) and, when it is ABSENT, resolves the admission
    # immediately as a compatibility shim for pre-durable sessions — so a
    # double without it never exercises the durable path and cannot reproduce
    # the park at all. The first version of this test passed on the UNFIXED
    # tree for exactly that reason, which is why it is spelled to match
    # production ``Session``: the drain hands over the ``admitted`` future and
    # the session resolves it on the durable append.
    async def prompt(  # noqa: ANN001
        self, text: str, images=None, message_id=None, admitted=None, **_kwargs
    ) -> None:
        self.started.append(text)
        self.prompt_calls.append(text)
        # Deliberately NOT setting is_streaming: this is the prelude window,
        # where a turn is admitted and running but the flag is still False —
        # the window the steer guard cannot see.
        await self.release.wait()
        if admitted is not None and not admitted.done():
            admitted.set_result(None)


@pytest.mark.asyncio
async def test_the_receipt_returns_without_waiting_for_a_queued_turn() -> None:
    """R1-1: the reply must not be held open for the duration of a prior turn.

    ``prompt`` resolves its receipt on the durable append, so AWAITING it here
    parked the ``slash_result`` response behind whatever was already running.
    Measured at 4.35 s in review; anything over the client's ``ACK_TIMEOUT_S``
    (15 s) makes the viewer raise a transport error for a request that was in
    fact admitted — worse than the silent drop this method exists to fix — and
    the per-connection reader is serial, so every later op from that viewer
    queues behind the park too.

    Asserted structurally rather than on a stopwatch: the receipt is produced
    while the session is still blocked, which is a fact about ordering that no
    amount of machine load can change (AGENTS.md: wait on the event, never on
    the clock).
    """
    session = _SlowDrainSession()
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")

    result = await handle._complete_unconsumed_action(_attach_receipt(), None, None)

    assert not session.release.is_set(), "the turn must still be in flight"
    assert result.style == "info", "a dispatched admission is not a failure to report"
    assert result.text == "sending to lopdev. manager is coordinating."

    # And the turn really was admitted, not dropped: release the drain and it
    # completes on its own.
    session.release.set()
    await _settle()
    assert session.prompt_calls == ["do the thing"]


@pytest.mark.asyncio
async def test_a_synchronous_refusal_is_still_reported_after_dispatching() -> None:
    """Dispatching must not cost the warning receipt for FAST failures.

    A closing session, a full queue and a rejected reservation all raise in
    ``prompt``'s synchronous prelude, before it ever awaits — so they are still
    knowable when the receipt is built, and step 4's contract survives the
    change. Only the slow outcome (a turn genuinely queued behind another) is
    left to run detached.
    """
    handle, session = make_handle()
    handle._disposing = True

    result = await handle._complete_unconsumed_action(_attach_receipt(), None, None)

    assert result.style == "warning"
    assert "the request was not sent" in result.text
    assert "session is closing" in result.text
    assert session.prompt_calls == []
