"""The accounting through the REAL product path: ``sdk.open_session`` -> real ``Session``.

Zero-cost: the model wire is the shipped ``hosting="test"`` mock provider with its
``MockClient.stream`` replaced by a scripted turn (the seam that provider exists for),
and the classification service — the REAL one, with its REAL legs — talks to an
``httpx.MockTransport``. No socket is opened and nothing is billed.

THE THREE CLAIMS, each the thing a unit test on the service cannot show:

1. A turn that classifies leaves the classification request in the session's channel
   ledger, and the REQUEST ledger (what ``subscribe_requests`` observes: one row per
   model request through ``SessionStreamFn._record_stream``) is UNCHANGED — the same
   count and purposes as the identical turn with classification off. That is the
   no-double-counting claim: the two ledgers are disjoint by construction and this
   proves it on the real chain.
2. Every surface reads one object. The TUI table, the phone projection and the
   desktop frame are all fed from ``FrontendStateStore``'s published ``spend_channels``
   — the test pulls the row out of each surface's own entry point instead of trusting
   that they share a source.
3. The rows persist like every other channel row and a resume folds them back.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from local_operator.harness.types import StreamEndEvent, StreamTextDelta, Usage
from local_operator.session.channel_spend import CHANNEL_SPEND_CUSTOM_TYPE

pytestmark = pytest.mark.asyncio

PROMPT = "please deploy core to qa using the tunnel guide"


class Wire:
    """The scripted MODEL transport: one text turn per request, counted."""

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: Any, api_key: Any, oauth_access: Any = None):
        self.requests += 1
        yield StreamTextDelta(delta="done")
        yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=5))


@pytest.fixture
def model_wire(monkeypatch: pytest.MonkeyPatch) -> Wire:
    from local_operator.providers.clients import MockClient

    scripted = Wire()
    monkeypatch.setattr(MockClient, "stream", scripted.stream)
    return scripted


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    from local_operator import agent_shell
    from local_operator.sdk import SessionRoots

    home = tmp_path / "home"
    if sys.platform == "darwin":
        import pwd

        # The active seatbelt boundary refuses controller stores under the
        # exposed per-user temp tree. HOME is already pytest-isolated, so use
        # passwd's real home and own only this UUID child, as world-root tests do.
        home = (
            Path(pwd.getpwuid(os.getuid()).pw_dir)
            / ".cache"
            / "lop-accounting-sdk-tests"
            / uuid.uuid4().hex
        )
        request.addfinalizer(lambda: shutil.rmtree(home, ignore_errors=True))
    root = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "decoy-config"))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "decoy-home"))
    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
    for key in [k for k in os.environ if k.startswith(("CMUX_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    # A static decision credential so the radient leg resolves without an OAuth store.
    monkeypatch.setenv("RADIENT_API_KEY", "unit-test-key")
    return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)


def _spec(**overrides: Any) -> Any:
    from local_operator.session.spec import ApprovalPolicy, SessionSpec

    values: dict[str, Any] = {
        "hosting": "test",
        "model": "mock",
        "approvals": ApprovalPolicy.auto(),
    }
    values.update(overrides)
    return SessionSpec(**values)


def _set_auto(scratch: Any, enabled: bool) -> None:
    from local_operator.config import ConfigManager

    ConfigManager(scratch.config_path).set_config_value("classification", {"auto": enabled})


class DecisionWire:
    """The mock DECISION transport, recording what reached it."""

    def __init__(self, *, cost: float | None = 0.000027) -> None:
        self.hosts: list[str] = []
        self._cost = cost

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.hosts.append(req.url.host)
        body = json.loads(req.content)
        question_id, question = next(iter(body["questions"].items()))
        answer: dict[str, Any] = {
            "answers": {question_id: {"choice": next(iter(question["criteria"]))}}
        }
        if self._cost is not None:
            answer["usage"] = {"cost": self._cost, "input_tokens": 100, "output_tokens": 0}
        return httpx.Response(200, json=answer)


async def _settle(session: Any, decisions: DecisionWire, timeout: float = 10.0) -> None:
    """Wait for the decision to land AND its channel append to finish (structural)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if decisions.hosts and not session._channel_tasks and session.channels.rows():
            await asyncio.sleep(0.05)
            if not session._channel_tasks:
                return
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"classification never landed: hosts={decisions.hosts} rows={session.channels.rows()}"
    )


async def _one_turn(scratch: Any, decisions: DecisionWire | None, **spec: Any) -> dict[str, Any]:
    """Run one prompt through a real session; return what each ledger holds."""
    from local_operator import sdk

    seen: list[Any] = []
    async with sdk.open_session(
        _spec(world_root=scratch.cwd_path, **spec), roots=scratch
    ) as session_proto:
        session = cast(Any, session_proto)
        classifier = session._knowledge_hooks.classifier
        if decisions is not None:
            assert classifier is not None
            classifier._http = httpx.AsyncClient(transport=httpx.MockTransport(decisions))
        else:
            assert classifier is None
        session.subscribe_requests(seen.append)
        await session.prompt(PROMPT)
        if decisions is not None:
            await _settle(session, decisions)
        else:
            await asyncio.sleep(0.5)
        # The turn itself must have finished before the request ledger is read.
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.02)
        store = session._frontend_state_store
        return {
            "request_purposes": [row.purpose for row in seen],
            "channel_rows": list(session.channels.rows()),
            "published": store.state.spend_channels,
            "phone_copy": store.spend_channels_copy(),
            "session_id": session.session_id,
            "directory": Path(session._transcript.directory),
        }


async def test_classification_rows_reach_the_ledger_and_the_request_ledger_is_unchanged(
    scratch: Any, model_wire: Wire
) -> None:
    """Claim 1: rows appear; the model-turn request ledger is byte-for-byte the control's."""
    _set_auto(scratch, False)
    control = await _one_turn(scratch, None)
    control_model_requests = model_wire.requests

    model_wire.requests = 0
    _set_auto(scratch, True)
    decisions = DecisionWire()
    treated = await _one_turn(scratch, decisions)

    assert control["request_purposes"] == ["turn"]
    assert (
        treated["request_purposes"] == control["request_purposes"]
    ), "the request ledger must not gain a row for a classification request"
    assert model_wire.requests == control_model_requests == 1
    assert control["channel_rows"] == [], "auto=False: nothing recorded"
    assert decisions.hosts == ["api.radienthq.com"], "exactly one decision request left the layer"
    (row,) = treated["channel_rows"]
    assert (row.channel, row.provider, row.units, row.unit) == (
        "classification",
        "radient",
        1,
        "calls",
    )
    assert (row.amount_micro, row.billing_basis, row.status) == (27, "billed", "ok")
    assert row.session_id == treated["session_id"], "the SESSION stamps the identity"


async def test_every_surface_reads_the_same_classification_row(
    scratch: Any, model_wire: Wire
) -> None:
    """Claim 2: desktop frame, phone projection and TUI table each show the row."""
    from dataclasses import replace

    from local_operator.mobile.projection import ProjectionFold, SessionProjection
    from local_operator.session.frontend_state import FrontendSpendChannels
    from local_operator.tui.widgets.session_panel import _Body, _draw_spend_channels
    from tests.unit.tui.test_session_panel import runtime

    decisions = DecisionWire()
    out = await _one_turn(scratch, decisions)

    def classification_rows(payload: Any) -> list[dict[str, Any]]:
        data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        return [row for row in data["rows"] if row["channel"] == "classification"]

    # Desktop frame: the store's published model object.
    (desktop,) = classification_rows(out["published"])
    # Phone: the store's serialised copy, folded through the projection exactly as
    # ``mobile.tui_handle`` / ``serving`` do.
    fold = ProjectionFold(SessionProjection(session_id="s1", pid=1))
    fold.set_spend_context(
        cumulative_parent_cost=None,
        child_costs={},
        subagent_cost=None,
        subagent_cost_knowledge=None,
        cost_knowledge="unknown",
        context_tokens=None,
        context_window=None,
        context_is_estimate=None,
        usage={},
        spend_channels=out["phone_copy"],
    )
    (phone,) = classification_rows(fold.projection.spend_channels)
    # TUI: the same published object through the /session table.
    body = _Body(width=100)
    diagnostics = replace(
        runtime(),
        spend_channels=FrontendSpendChannels.model_validate(
            out["published"].model_dump(mode="json")
        ),
    )
    assert _draw_spend_channels(body, diagnostics) is True
    table = body.to_text().plain

    assert desktop == phone, "desktop and phone must be the same row"
    assert desktop["amount_micro"] == 27 and desktop["unit"] == "calls" and desktop["units"] == 1
    assert "classification" in table.lower() and "radient" in table


async def test_a_classification_row_with_no_cost_is_a_partial_total_on_every_surface(
    scratch: Any, model_wire: Wire
) -> None:
    """The unknown-never-zero rule, through the real session's published object."""
    decisions = DecisionWire(cost=None)
    out = await _one_turn(scratch, decisions)

    (row,) = out["channel_rows"]
    assert row.amount_micro is None and row.billing_basis == "not_tracked"
    published = out["published"]
    assert published.total_micro == 0
    assert published.knowledge != "exact", "an unsized request must never read as exact"
    assert published.by_basis["not_tracked_calls"] == 1
    assert out["phone_copy"]["knowledge"] == published.knowledge


async def test_the_row_is_journaled_and_a_resume_folds_it_back(
    scratch: Any, model_wire: Wire
) -> None:
    """Claim 3: persistence consistent with every other channel row, and resume."""
    from local_operator import sdk
    from local_operator.session.transcript import Transcript

    decisions = DecisionWire()
    out = await _one_turn(scratch, decisions)

    journal = Transcript(out["directory"])
    rows = journal.channel_spend_rows()
    assert [(r["channel"], r["amount_micro"], r["unit"]) for r in rows] == [
        ("classification", 27, "calls")
    ]
    assert journal.channel_spend_tracked() is True, "the start marker precedes the first row"
    custom = [
        entry.payload
        for entry in journal.entries()
        if entry.payload.get("custom_type") == CHANNEL_SPEND_CUSTOM_TYPE
    ]
    assert len(custom) == 2, "one start marker + one record, exactly like any other channel"

    # A real resume: the new process folds the journal, and classifies nothing new
    # (its decision transport is inert), so the total is the persisted one.
    async with sdk.open_session(
        _spec(resume=out["session_id"], world_root=scratch.cwd_path), roots=scratch
    ) as resumed_proto:
        resumed = cast(Any, resumed_proto)
        folded = [row for row in resumed.channels.rows() if row.channel == "classification"]
        assert [(r.record_id, r.amount_micro, r.status) for r in folded] == [
            (rows[0]["record_id"], 27, "ok")
        ]
        assert resumed.channels_started is True


async def test_an_in_flight_request_abandoned_at_dispose_is_journaled(
    scratch: Any, model_wire: Wire
) -> None:
    """A session ended mid-request still leaves the attempt countable on disk."""
    from local_operator import sdk
    from local_operator.session.transcript import Transcript

    started = asyncio.Event()

    async def hang(req: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(500)

    async with sdk.open_session(_spec(world_root=scratch.cwd_path), roots=scratch) as session_proto:
        session = cast(Any, session_proto)
        session._knowledge_hooks.classifier._http = httpx.AsyncClient(
            transport=httpx.MockTransport(hang)
        )
        await session.prompt(PROMPT)
        await asyncio.wait_for(started.wait(), timeout=10)
        directory = Path(session._transcript.directory)

    deadline = asyncio.get_running_loop().time() + 5
    rows: list[dict[str, Any]] = []
    while asyncio.get_running_loop().time() < deadline:
        rows = Transcript(directory).channel_spend_rows()
        if rows:
            break
        await asyncio.sleep(0.05)
    assert [(r["channel"], r["status"], r["detail"], r["amount_micro"]) for r in rows] == [
        ("classification", "cancelled", "interrupted-in-flight", None)
    ]


async def test_auto_false_opens_no_decision_socket_and_records_no_row(
    scratch: Any, model_wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``auto: false`` stays a real off through the whole product path."""
    posted: list[str] = []
    real_post = httpx.AsyncClient.post

    async def spy(self: httpx.AsyncClient, url: Any, *args: Any, **kwargs: Any) -> Any:
        if "decisions" in str(url) or "systemone" in str(url):
            posted.append(str(url))
            # Fail closed: a regression must never turn this no-network proof into
            # a real paid decision request before the final assertion catches it.
            raise AssertionError("disabled classification attempted a decision POST")
        return await real_post(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", spy)
    _set_auto(scratch, False)
    out = await _one_turn(scratch, None)

    assert posted == []
    assert out["channel_rows"] == []
    from local_operator.session.transcript import Transcript

    assert not [
        row
        for row in Transcript(out["directory"]).channel_spend_rows()
        if row["channel"] == "classification"
    ]
