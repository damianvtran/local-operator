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
  says in ``admission.detail`` which of the two dispositions the caller got;
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_sessions
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
    vacuously. ``park`` is the one control a test needs beyond that: an unset
    event makes the owner's ack hang forever, which is how the "does not park"
    cells drive the reply that has to come back without it.
    """

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        self.binds = 0
        self.is_streaming = False
        self.park: asyncio.Event | None = None
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
        # The two details the real owner answers with, verbatim.
        return ("steering queued" if steer else "prompt admitted", False)


class FakePool:
    """``DesktopSessions``-shaped: the route's only door to a session.

    The real pool's own admissions (the retirement latch) have their own tests;
    what matters here is that the route works THROUGH a bridge facade rather
    than calling a handler directly. ``refresh_watch`` lives on the BRIDGE and
    ``bind_runtime`` on the facade it holds, which is the real split and the
    reason this double yields both.
    """

    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote
        self.refreshes = 0

    async def refresh_watch(self) -> None:
        self.refreshes += 1

    @contextlib.asynccontextmanager
    async def session(self, session_id: str):
        if session_id != SESSION:
            raise KeyError("Unknown session")
        yield SimpleNamespace(remote=self.remote, refresh_watch=self.refresh_watch)


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
    app.state.desktop_sessions = FakePool(remote)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, remote


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
    client, remote = desktop

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
    client, _remote = desktop

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
    client, remote = desktop
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
    """
    client, remote = desktop
    remote.park = asyncio.Event()

    response = await asyncio.wait_for(_goal(client, "Preserve one identity"), timeout=10)

    assert response.status_code == 200, response.text
    admission = response.json()["result"]["result"]["admission"]
    assert admission["status"] == "admitted"
    # No turn was running, so this is the "handed over" phrase, not the queued
    # one: the two must not be confused in either direction.
    assert admission["detail"] == desktop_sessions.HANDED_OVER_ADMISSION_DETAIL
    assert remote.admissions == [("Preserve one identity", REQUEST_ID)]
    # Release the parked ack so the detached task is not left pending at teardown.
    remote.park.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_parked_ack_mid_turn_is_reported_as_queued(desktop) -> None:
    """And the queued phrase when a turn WAS running behind the parked ack.

    The distinction is the point of the two phrases: a renderer that promises
    "sends when this step finishes" needs the queued one, and the same status
    word covers both.
    """
    client, remote = desktop
    remote.is_streaming = True
    remote.park = asyncio.Event()

    response = await asyncio.wait_for(_goal(client, "Preserve one identity"), timeout=10)

    admission = response.json()["result"]["result"]["admission"]
    assert admission["detail"] == desktop_sessions.QUEUED_ADMISSION_DETAIL
    assert remote.steered == [True]
    remote.park.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_an_agent_clear_carries_no_request_and_starts_no_turn(desktop) -> None:
    """THE REAL empty-request case: ``agent_attached`` from ``/agent clear``.

    A detach is a receipt with no action behind it — the type IS in the
    vocabulary (so it reaches this decision), and the request is empty. Nothing
    is submitted with or without images: the body's staged images are the
    CALLER's, not the receipt's, so admitting them would open an image-only turn
    nobody asked for, as a paid provider call and a durable row.
    """
    client, remote = desktop
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
    client, remote = desktop
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
    client, remote = desktop
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
    """
    from local_operator.session.runtime import types as runtime_types

    monkeypatch.setattr(
        runtime_types,
        "SLASH_ACTION_RECEIPTS",
        (*runtime_types.SLASH_ACTION_RECEIPTS, "synthetic_attached"),
    )
    client, remote = desktop
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
    client, remote = desktop

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
    # The two directions the route relies on, asserted on the predicate itself:
    # an undeclaring client leaves the submit to the runtime, a declaring one
    # (this host) takes it.
    assert runtime_must_complete("goal_set", []) is True
    assert runtime_must_complete("goal_set", SLASH_ACTION_RECEIPTS) is False
