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

So there are two guards here, deliberately of different kinds:

* a BEHAVIOURAL one over the REAL router — the goal command admits its argument
  through the route, and a type added to the vocabulary tomorrow is admitted
  without this file knowing its name; and
* a STATIC one over the real source, in the spirit of
  ``tests/unit/tui/test_noop_consumers.py`` — the handler must reach its
  decision through the shared helper and must not name a receipt type itself,
  which is the shape of the defect that shipped.

The bridge is a double (this is a route-contract test); the assembled
HTTP + runtime path is covered end to end by
``tests/e2e/test_desktop_sessions.py``.
"""

from __future__ import annotations

import ast
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
ROUTE_SOURCE = Path(desktop_sessions.__file__).read_text(encoding="utf-8")


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


class FakeRemote:
    """The viewer facade the command route reads, recording every call.

    Only the three members the route touches, and the admissions are recorded
    rather than answered away: "did the host submit the request" is the whole
    question these tests ask, and a double that admitted silently would answer
    it vacuously.
    """

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        self.binds = 0
        self.routed: list[tuple[str, str]] = []
        self.admissions: list[tuple[str, str]] = []

    async def bind_runtime(self) -> None:
        self.binds += 1

    async def route_shared_slash(self, command: str, args: str, images: Any = None):
        self.routed.append((command, args))
        return dict(self.receipt)

    async def admit_prompt(self, text: str, *, command_id: str, images: Any = None):
        self.admissions.append((text, command_id))
        return ("prompt admitted", False)


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


async def _goal(client: AsyncClient, args: str, *, request_id: str = REQUEST_ID):
    return await client.post(
        f"/v1/desktop/sessions/{SESSION}/commands",
        json={"request_id": request_id, "command": "goal", "args": args},
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
async def test_a_goal_receipt_without_a_request_starts_no_turn(desktop) -> None:
    """``/goal clear``'s shape: a typed receipt with nothing to submit.

    The goal mutation still landed on the runtime, so the receipt is returned
    unchanged; admitting an empty string would be a turn with no text.
    """
    client, remote = desktop
    remote.receipt = {
        "kind": "notice",
        "text": "goal cleared",
        "style": "info",
        "data": {"type": "goal_set", "stored": "", "request": ""},
    }

    response = await _goal(client, "clear")

    assert response.status_code == 200, response.text
    # ``admission`` is a declared field of the receipt and serialises as null:
    # the absence of an admission is what says no turn was started.
    assert response.json()["result"]["result"]["admission"] is None
    assert remote.admissions == []


@pytest.mark.asyncio
async def test_a_status_notice_is_not_treated_as_an_action(desktop) -> None:
    """The show form: ``/goal`` with no argument carries no receipt type.

    A string a picker or listing happens to call ``request`` is not proof that
    a turn was asked for, which is why the decision keys on the typed
    discriminator rather than on the presence of the key.
    """
    client, remote = desktop
    remote.receipt = {
        "kind": "notice",
        "text": "goal: Preserve one identity",
        "style": "info",
        "data": {"goal": "Preserve one identity"},
    }

    response = await _goal(client, "")

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


def test_the_command_handler_decides_through_the_shared_helper() -> None:
    """The static half: the handler may not name a receipt type itself.

    The behavioural cells above prove what the route does TODAY; this one holds
    the SHAPE that made the defect reachable — a local tuple of type names
    beside a shared vocabulary that grew without it. Anything the handler
    compares a receipt type against is therefore either the shared helper or
    nothing, and re-hardcoding the names fails here.
    """
    tree = ast.parse(ROUTE_SOURCE)
    functions = _functions(tree)
    handler = functions["command"]
    body = ast.get_source_segment(ROUTE_SOURCE, handler) or ""

    assert "desktop_viewer_must_submit" in body, (
        "the desktop command route no longer reaches its completion decision "
        "through desktop_viewer_must_submit; a hand-written set of receipt types "
        "is how /goal <text> lost its turn"
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

    ``runtime_must_complete`` is that rule, and the vocabulary is the list it is
    applied to: a helper that consulted anything else (a private set, a
    configuration flag) would answer for this host while the runtime answered
    for the shared seam.
    """
    from local_operator.session.runtime.types import runtime_must_complete

    tree = ast.parse(ROUTE_SOURCE)
    helper = _functions(tree)["desktop_viewer_must_submit"]
    body = ast.get_source_segment(ROUTE_SOURCE, helper) or ""

    assert "SLASH_ACTION_RECEIPTS" in body, (
        "desktop_viewer_must_submit must read the shared vocabulary "
        "(SLASH_ACTION_RECEIPTS), not a list of its own"
    )
    assert "runtime_must_complete" in body, (
        "the completion decision must be the shared rule, so this host and the "
        "runtime cannot answer differently about the same receipt"
    )
    # The two directions the route relies on, asserted on the predicate itself:
    # an undeclaring client leaves the submit to the runtime, a declaring one
    # (this host) takes it.
    assert runtime_must_complete("goal_set", []) is True
    assert runtime_must_complete("goal_set", SLASH_ACTION_RECEIPTS) is False
