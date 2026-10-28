"""The per-session Saver chain against a local fake prepare server (contract §3.2, §6, r5 S3-6).

The production route (AS2) is not live while LO1 is built, so every test drives
``SaverChain.apply`` exactly as ``SessionStreamFn`` does, against
``fake_prepare.FakeServer`` — the §3 response shape with reduce-v2 framing —
through an ``httpx.MockTransport``. No socket is opened.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)
from local_operator.saver import chain as chain_mod
from local_operator.saver.chain import (
    MAX_UPLOAD_BYTES,
    SaverChain,
    plan_upload,
    predicted_boundaries,
    spend_record,
)
from local_operator.saver.client import SaverClient
from local_operator.saver.projection import ProjectionCache, shape_only_redaction
from tests.unit.saver.fake_prepare import FakeServer

MODEL = ModelSpec(provider="anthropic", model_id="claude-test")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The process-wide 404/403 pause is a class attribute: reset it per test.
    monkeypatch.setattr(chain_mod._ProcessState, "paused", False)
    monkeypatch.setattr("local_operator.saver.client.destination_allowed", lambda base: True)


def _block(i: int, *, big: int = 0) -> list[Message]:
    output = ("z" * big) if big else "\n".join(f"line {j} of src/mod_{i}.py" for j in range(60))
    return [
        Message(
            role="user",
            content=[TextContent(text=f"step {i}: fix src/mod_{i}.py " + "y" * 200)],
        ),
        Message(
            role="assistant",
            content=[TextContent(text=f"reading mod_{i}")],
            tool_calls=[
                ToolCall(id=f"c{i}", name="bash", arguments={"command": f"cat src/mod_{i}.py"})
            ],
        ),
        Message(
            role="tool",
            tool_call_id=f"c{i}",
            tool_name="bash",
            content=[TextContent(text=output)],
        ),
        Message(role="assistant", content=[TextContent(text=f"fixed mod_{i}")]),
    ]


class Rig:
    """One chain, its fake server, and what it emitted."""

    def __init__(self, server: FakeServer | None = None, *, session_id: str = "s1") -> None:
        self.server = server or FakeServer()
        self.notices: list[str] = []
        self.spend: list[Any] = []

        async def bearer(force_refresh: bool) -> str:
            return "radient-bearer"

        async def notice(message) -> None:
            self.notices.append(message.code.rsplit(".", 1)[-1])

        self.client = SaverClient(
            "https://api.radienthq.com/v1", bearer=bearer, transport=self.server.transport()
        )
        self.chain = SaverChain(
            client=self.client, session_id=session_id, notice=notice, spend=self.spend.append
        )
        self.messages: list[Message] = []

    async def turn(self, *new: Message) -> ChatRequest:
        self.messages.extend(new)
        request = ChatRequest(model=MODEL, messages=list(self.messages))
        out = await self.chain.apply(request)
        await asyncio.sleep(0)  # let a started prefetch run against the mock
        if self.chain._prefetch is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.chain._prefetch.task), 1)
            except Exception:  # noqa: BLE001 — the chain classifies it next turn
                pass
        return out

    async def turns(self, start: int, stop: int, **kw) -> ChatRequest:
        out = None
        for i in range(start, stop):
            out = await self.turn(*_block(i, **({"big": kw["big"]} if kw.get("at") == i else {})))
        assert out is not None
        return out

    async def close(self) -> None:
        await self.chain.aclose()
        await self.client.aclose()


# -- the happy path -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_chain_freezes_splices_and_bills() -> None:
    rig = Rig()
    out = await rig.turns(0, 10)
    await rig.close()
    assert rig.chain.frontier > 0 and rig.chain.epochs
    assert out.frozen_prefix_messages == sum(len(e.messages) for e in rig.chain.epochs)
    assert len(out.messages) < len(rig.messages)
    # The tail after the frontier is the session's own rows, by identity.
    assert out.messages[out.frozen_prefix_messages :] == rig.messages[rig.chain.frontier :]
    assert all(m.role == "user" for m in out.messages[: out.frozen_prefix_messages])
    assert rig.notices == []
    assert rig.spend and all(r.channel == "saver" for r in rig.spend)


@pytest.mark.asyncio
async def test_the_critical_path_never_waits_on_a_slow_prepare() -> None:
    """§6 latency budget: a due prepare in flight is waited on ≤ 250 ms."""
    gate = asyncio.Event()
    server = FakeServer()

    class Slow(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await gate.wait()
            return server(request)

    rig = Rig(server)
    rig.client._http._transport = Slow()  # noqa: SLF001 — inject latency only
    for i in range(6):
        rig.messages.extend(_block(i))
    await rig.chain.apply(ChatRequest(model=MODEL, messages=list(rig.messages)))
    rig.messages.extend(_block(6))
    loop = asyncio.get_running_loop()
    started = loop.time()
    out = await rig.chain.apply(ChatRequest(model=MODEL, messages=list(rig.messages)))
    waited = loop.time() - started
    gate.set()
    await rig.close()
    assert waited < 0.6
    assert out.frozen_prefix_messages is None  # sent without it, unreduced


# -- §6.1 check 7: turn binding ----------------------------------------------------


async def _with_pending_prefetch(rig: Rig, gate: asyncio.Event) -> None:
    server = rig.server

    class Gated(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await gate.wait()
            return server(request)

    rig.client._http._transport = Gated()  # noqa: SLF001
    for i in range(6):
        rig.messages.extend(_block(i))
    await rig.chain.apply(ChatRequest(model=MODEL, messages=list(rig.messages)))
    assert rig.chain._prefetch is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["clear", "compaction", "fork"])
async def test_check7_a_prefetch_arriving_after_a_rewrite_is_discarded(event: str) -> None:
    gate = asyncio.Event()
    rig = Rig()
    await _with_pending_prefetch(rig, gate)
    if event == "clear":
        rig.messages = _block(100)  # /clear: a new history
    elif event == "compaction":
        summary = Message(role="user", content=[TextContent(text="summary of earlier work")])
        rig.messages = [summary, *rig.messages[-4:]]
    else:  # fork: a different conversation's chain receives the result
        rig.chain.session_id = "forked-child"
    gate.set()
    await asyncio.sleep(0.01)
    out = await rig.chain.apply(ChatRequest(model=MODEL, messages=list(rig.messages)))
    await rig.close()
    assert rig.chain.epochs == []
    assert rig.chain.frontier == 0
    assert out.frozen_prefix_messages is None
    assert rig.notices == []  # the conversation moved on; nothing failed


@pytest.mark.asyncio
async def test_check7_a_truncation_bumps_the_generation_and_drops_later_epochs() -> None:
    rig = Rig()
    await rig.turns(0, 12)
    assert rig.chain.epochs
    generation = rig.chain.chain_generation
    rig.messages[1].content = [TextContent(text="edited")]  # a rewrite inside epoch 0
    await rig.turn(*_block(12))
    await rig.close()
    assert rig.chain.chain_generation > generation
    assert rig.chain.frontier == 0 or rig.chain.epochs[0].source_count <= rig.chain.frontier


# -- §6: failures fail open, notices once per class --------------------------------------


@pytest.mark.asyncio
async def test_radient_blackholed_mid_session_notices_once_and_the_turn_succeeds() -> None:
    rig = Rig()
    await rig.turns(0, 10)
    rig.server.raise_exc = httpx.ConnectTimeout("blackholed")
    # A prepare that SUCCEEDED before the blackhole is still adopted on the next
    # turn (correct: its bytes were validated); the failure surface starts after.
    await rig.turn(*_block(10))
    frozen_before = rig.chain.frontier
    assert frozen_before > 0
    rig.chain.call_again_after = 1
    outs = []
    for i in range(11, 17):
        rig.chain._paused_until = 0  # skip the 60 s back-off so every turn tries
        outs.append(await rig.turn(*_block(i)))
    await rig.close()
    assert rig.notices == ["unavailable"]
    assert rig.chain.frontier == frozen_before  # never advanced on failure
    # The existing epochs keep being sent with a longer verbatim tail (§6).
    assert all(out.frozen_prefix_messages for out in outs)
    assert outs[-1].messages[outs[-1].frozen_prefix_messages :] == rig.messages[frozen_before:]


@pytest.mark.asyncio
async def test_a_rejected_response_is_discarded_whole() -> None:
    server = FakeServer()

    def lie(body, payload):
        if payload["epochs"]:
            payload["epochs"][0]["source_digest"] = "sha256:" + "0" * 64
        return payload

    server.mutate = lie
    rig = Rig(server)
    out = await rig.turns(0, 10)
    await rig.close()
    assert rig.chain.epochs == [] and rig.chain.frontier == 0
    assert out.frozen_prefix_messages is None
    assert rig.notices == ["rejected"]
    assert rig.spend == []  # nothing adopted, nothing recorded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,notice,scope",
    [
        (401, "signed_out", "session"),
        (402, "no_credit", "session"),
        (403, "disabled", "process"),
        (404, "disabled", "process"),
        (429, "rate_limited", "timed"),
        (503, "unavailable", "timed"),
    ],
)
async def test_backoff_per_status(status: int, notice: str, scope: str) -> None:
    server = FakeServer()
    server.status = status
    server.headers = {"retry-after": "30"} if status == 429 else {}
    rig = Rig(server)
    refreshed: list[bool] = []
    original = rig.client._bearer

    async def bearer(force_refresh: bool) -> str:
        refreshed.append(force_refresh)
        return await original(force_refresh)

    rig.client._bearer = bearer
    await rig.turns(0, 12)
    calls = len(server.requests)
    await rig.turns(12, 20)
    await rig.close()
    assert rig.notices == [notice]
    assert len(server.requests) == calls  # backed off: no further calls
    if status == 401:
        assert refreshed.count(True) == 1  # exactly one refresh, then pause
        assert calls == 2
    if status == 402:
        assert rig.chain.needs_availability_refresh
    if scope == "process":
        assert chain_mod._ProcessState.paused
    if scope == "timed":
        assert rig.chain._paused_until > 0


@pytest.mark.asyncio
async def test_a_redirect_pauses_saver_and_is_not_followed() -> None:
    server = FakeServer()
    server.status = 307
    server.headers = {"location": "https://evil.example/"}
    rig = Rig(server)
    await rig.turns(0, 12)
    await rig.close()
    assert all(r.url.host == "api.radienthq.com" for r in server.requests)
    assert rig.notices == ["disabled"]


# -- §5a: the spend line ----------------------------------------------------------------


def test_spend_comes_from_the_cost_object_and_is_idempotent() -> None:
    payload = {
        "cost": {
            "v": 1,
            "amount_usd": 0.000043,
            "basis": "billed",
            "tool_type": "context_saver",
            "usage_record_id": "usage-abc",
        }
    }
    first = spend_record(payload, session_id="s1")
    again = spend_record(payload, session_id="s1")
    assert first is not None and first.channel == "saver"
    assert first.record_id == again.record_id == "saver:usage-abc"
    assert first.amount_micro == 43
    assert spend_record({"epochs": []}, session_id="s1") is None  # no row, nothing charged


# -- r5 S3-6: oversized units -------------------------------------------------------


def _project(messages: list[Message]):
    return ProjectionCache().project(messages, shape_only_redaction())


def test_s3_6_the_upload_limit_grows_for_an_over_1mib_unit_and_never_past_the_ceiling() -> None:
    messages: list[Message] = []
    for i in range(3):
        messages += _block(i)
    messages += _block(3, big=int(1.5 * 1024 * 1024))
    for i in range(4, 12):
        messages += _block(i)
    rows, encoded = _project(messages)
    plan = plan_upload(rows, encoded)
    assert plan.oversized_unit == 13  # the call row heading the big result's pair
    assert MAX_UPLOAD_BYTES < plan.limit <= 8 * 1024 * 1024 - 64 * 1024
    assert predicted_boundaries(rows[: plan.count])[-1] > 15  # it freezes past the unit


def test_s3_6_a_unit_past_the_ceiling_is_a_hard_barrier() -> None:
    messages = _block(0, big=9 * 1024 * 1024) + [m for i in range(1, 8) for m in _block(i)]
    rows, encoded = _project(messages)
    plan = plan_upload(rows, encoded)
    # Row 0 is the user turn; the unit is the call (row 1) plus its 9 MiB result.
    assert plan.blocking_unit == 1 and plan.blocking_new and plan.count == 1


@pytest.mark.asyncio
async def test_s3_6_a_1_5_mib_result_is_frozen_through_one_oversized_upload() -> None:
    """The LO1 fixture named in §6.1 check 1."""
    rig = Rig()
    await rig.turns(0, 30, at=3, big=int(1.5 * 1024 * 1024))
    await rig.close()
    big_uploads = [r for r in rig.server.requests if len(r.content) > MAX_UPLOAD_BYTES]
    assert len(big_uploads) == 1  # frozen through ONE oversized-head upload, no retry
    assert rig.chain.frontier > 15  # past the unit (rows 12..15)
    assert rig.notices == []


@pytest.mark.asyncio
async def test_s3_6_a_9_mib_head_gives_one_notice_and_zero_calls_over_20_turns() -> None:
    """The LO1 fixture named in §6.1 check 1: the oversized unit heads the suffix."""
    rig = Rig()
    await rig.turns(0, 21, at=0, big=9 * 1024 * 1024)
    await rig.close()
    assert rig.notices == ["oversize"]
    assert rig.server.requests == []
    assert rig.chain._oversize_mark is not None


@pytest.mark.asyncio
async def test_s3_6_freezes_up_to_the_unit_then_resumes_past_it_once_it_leaves_the_head() -> None:
    """The manager's acceptance case: a session whose head unit becomes > 1 MiB
    beyond any U. It freezes up to the unit, marks it, notices once, never
    retries while it heads the suffix, and keeps freezing past it once lop's
    prune rewrites that row (the source at the frontier changes)."""
    rig = Rig()
    big_at = 10
    await rig.turns(0, big_at + 1, at=big_at, big=9 * 1024 * 1024)
    big_row = big_at * 4 + 2
    await rig.turns(big_at + 1, 25)
    frontier_at_barrier = rig.chain.frontier
    calls_at_barrier = len(rig.server.requests)
    assert 0 < frontier_at_barrier <= big_row - 1  # froze up to the unit, not into it
    assert rig.notices == ["oversize"]
    await rig.turns(25, 35)
    assert len(rig.server.requests) == calls_at_barrier  # never retried
    assert rig.chain.frontier == frontier_at_barrier
    assert all(len(r.content) <= MAX_UPLOAD_BYTES for r in rig.server.requests)
    # lop's prune blanks the stale output in place: the unit leaves the barrier.
    rig.messages[big_row].content = [TextContent(text="[output pruned]")]
    await rig.turns(35, 45)
    await rig.close()
    assert rig.chain.frontier > big_row  # froze past it
    assert rig.notices == ["oversize"]  # still exactly one notice
    assert rig.chain._oversize_mark is None


@pytest.mark.asyncio
async def test_s3_6_a_413_on_an_oversized_upload_marks_instead_of_retrying() -> None:
    server = FakeServer()

    def refuse_big(body, payload):
        total = sum(len(str(m)) for m in body["messages"])
        return httpx.Response(413) if total > MAX_UPLOAD_BYTES else payload

    server.mutate = refuse_big
    rig = Rig(server)
    await rig.turns(0, 40, at=3, big=int(1.5 * 1024 * 1024))
    await rig.close()
    big_uploads = [r for r in server.requests if len(r.content) > MAX_UPLOAD_BYTES]
    assert len(big_uploads) == 1
    assert rig.notices == ["oversize"]
