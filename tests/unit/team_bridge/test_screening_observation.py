"""The observe-only ``screening_observation`` block (agent-server #332, TEAM-BRIDGE.md §8.5).

WHAT IS PINNED. The client reports which screener ACTUALLY answered and at
which stage — ``preferred`` (stage 1 of ``resolve_screening_chain``),
``fallback`` (the receiver's default/fallbackChains cascade), ``unknown`` (no
trusted answering-hop provenance) — on the pair ack and the conversation
admit, including B4b's owner Deliver. A missing provenance omits the block; a
hub that has not advertised it never receives it; and nothing about the block
changes a verdict, a request, a retry or a release.

THE RIGS are the shipped ones: the chain rig of ``test_screening_chain`` (the
real ``run_screening_cycle`` and production factory over a mocked vendor
transport), the real ``TeamBridgeClient`` over ``httpx.MockTransport``, the
pull lane's ``FakeWire``, and B4b's conversation spine from
``test_hold_release``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import pull
from local_operator.team_bridge import screening_observation as obs
from local_operator.team_bridge.model_boundary import (
    DefaultSelection,
    ScreenChainRunner,
)
from local_operator.team_bridge.types import ScreeningObservation, ScreeningReceipt
from tests.unit.team_bridge.test_client import _make_client, _ok
from tests.unit.team_bridge.test_hold_release import rig  # noqa: F401 - the B4b fixture
from tests.unit.team_bridge.test_screening import _setup
from tests.unit.team_bridge.test_screening_chain import (  # noqa: F401 - autouse fixture
    _MONKEYPATCH,
    DELIVER,
    _blocked_credential,
    _chain_rig,
    _cycle,
    _install_wire,
    _Wire,
)

CID = "bcv_abcdefgh"
MID = "bcmsg_abcdef01"
VERSION = "9.9.9-test"
SHA = "ab" * 32


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fresh hub memo per test, a pinned client version, an open destination."""

    obs.reset_hub_support()
    monkeypatch.setattr(obs, "_client_version", lambda: VERSION)
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    yield
    obs.reset_hub_support()


def _receipt(observation: ScreeningObservation | None = None) -> ScreeningReceipt:
    return ScreeningReceipt(
        verdict="allow",
        model="gpt-5.4",
        prompt_version="bridge_screen_v1",
        template_sha256="c" * 64,
        body_sha256="c" * 64,
        screener_preferred=True,
        observation=observation,
    )


PREFERRED = ScreeningObservation("openai", "gpt-5.4", "preferred")


# -- the producer: which screener answered, at which stage -------------------


def _cascade_config(tmp_path: Path, chain: list[str]) -> None:
    """Default ``openai/gpt-5.4`` as the ONLY preferred pair; ``chain`` is stage 2."""

    from local_operator.team_bridge import model_boundary

    lines = [
        "values:",
        "  hosting: openai",
        "  model_name: gpt-5.4",
        "  retry:",
        "    modelFallback: true",
        "    fallbackChains:",
        "      default:",
        *[f"        - {entry}" for entry in chain],
    ]
    (tmp_path / "config.yml").write_text("\n".join(lines) + "\n")
    _MONKEYPATCH[-1].setattr(model_boundary, "SCREENING_EVALUATED_MODELS", (("openai", "gpt-5.4"),))


def test_a_preferred_screener_that_answers_reports_preferred(tmp_path, monkeypatch) -> None:
    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["deepseek/deepseek-flash"])
    _install_wire(monkeypatch, _Wire({"gpt-5.4": ("reply", DELIVER)}))

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver"
    assert result.receipt.observation == ScreeningObservation("openai", "gpt-5.4", "preferred")
    assert result.receipt.screener_preferred is True


def test_a_quota_refusal_reports_the_fallback_that_actually_answered(tmp_path, monkeypatch) -> None:
    """Out of quota on the preferred screener: the cascade serves, and says so.

    The configured/preferred model (gpt-5.4) must NOT be reported: the
    deepseek hop answered, in stage 2.
    """

    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["deepseek/deepseek-flash"])
    wire = _Wire({"deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire, failover={"gpt-5.4": _blocked_credential()})

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver"
    assert result.receipt.model == "deepseek-flash"
    assert result.receipt.observation == ScreeningObservation(
        "deepseek", "deepseek-flash", "fallback"
    )
    assert [s["model"] for s in wire.sends] == ["deepseek-flash"]


def test_a_login_flavour_fallback_reports_the_provider_it_authenticates(
    tmp_path, monkeypatch
) -> None:
    """#343 review: ``anthropic-key`` is a way to log in to ``anthropic``.

    The hub counts an unknown provider string as OTHER, so the client folds
    the flavour through its own alias table. The model is never folded.
    """

    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["anthropic-key/claude-sonnet-5-5"])
    wire = _Wire({"claude-sonnet-5-5": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire, failover={"gpt-5.4": _blocked_credential()})

    result = _cycle(tmp_path, owner)

    assert result.receipt.observation == ScreeningObservation(
        "anthropic", "claude-sonnet-5-5", "fallback"
    )
    released = obs.from_release_row(
        {"screener_provider": "radient-key", "screener_model": "claude-sonnet-5-5"}
    )
    assert released == ScreeningObservation("radient", "claude-sonnet-5-5", "unknown")


def test_a_wire_429_on_the_preferred_hop_also_reports_fallback(tmp_path, monkeypatch) -> None:
    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", 429), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.receipt.observation == ScreeningObservation(
        "deepseek", "deepseek-flash", "fallback"
    )


def test_a_generic_provider_error_on_the_preferred_hop_reports_fallback(
    tmp_path, monkeypatch
) -> None:
    """A plain route-failure ``ProviderError`` (5xx), not a quota: the cascade
    advances and the block names the hop that answered (review m3)."""

    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", 503), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert [s["model"] for s in wire.sends] == ["gpt-5.4", "deepseek-flash"]
    assert result.receipt.observation == ScreeningObservation(
        "deepseek", "deepseek-flash", "fallback"
    )


def test_a_batch_screen_stamps_the_answering_hop_on_every_item(tmp_path, monkeypatch) -> None:
    """The batch conclude reads the same answering hop as the single path (review m3).

    One model call screens both messages, so both receipts carry the one hop
    that answered that call — the fallback here, never the configured model.
    """

    from local_operator.team_bridge import scheduler as _scheduler
    from tests.unit.team_bridge.test_batch_screening import _triples
    from tests.unit.team_bridge.test_screening import _consent_status, _item

    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["deepseek/deepseek-flash"])
    reply = _triples(("m1", "deliver"), ("m2", "deliver"))
    wire = _Wire({"gpt-5.4": ("fail", 503), "deepseek-flash": ("reply", reply)})
    _install_wire(monkeypatch, wire)
    raw = _consent_status(owner, {"version": "bridge_consent_v1", "text": "v1 consent"})
    verified = _scheduler.VerifiedStatus(_scheduler._VERIFIED_TOKEN, raw)

    async def fetch() -> Any:
        return verified

    results = asyncio.run(
        _scheduler.run_screening_cycle(
            config_dir=tmp_path,
            owner=owner,
            items=[_item(body="a", message_id="bmsg_1"), _item(body="b", message_id="bmsg_2")],
            auth_store=None,
            status_fetcher=fetch,
            batch=True,
        )
    )

    assert [s["model"] for s in wire.sends] == ["gpt-5.4", "deepseek-flash"]
    assert [r.disposition for r in results] == ["deliver", "deliver"]
    fallback = ScreeningObservation("deepseek", "deepseek-flash", "fallback")
    assert [r.receipt.observation for r in results] == [fallback, fallback]


def test_without_trusted_hop_provenance_the_stage_is_unknown() -> None:
    """A bare runner, or a hop that did not produce the answer, is ``unknown``."""

    class Bare:
        answered_hop = DefaultSelection("openai", "gpt-5.4", None, preferred=True)

    # A bare runner cannot claim a stage or a provider, whatever it carries.
    assert obs.from_answered_hop(Bare(), "gpt-5.4") == ScreeningObservation(
        "", "gpt-5.4", "unknown"
    )
    chain = ScreenChainRunner([], lambda _hop: None)
    chain.answered_hop = DefaultSelection("openai", "gpt-5.4", None, preferred=True)
    # The hop on record is not the identity that answered: no stage claimed.
    assert obs.from_answered_hop(chain, "deepseek-flash") == ScreeningObservation(
        "", "deepseek-flash", "unknown"
    )
    # Nothing answered at all: there is nothing to report.
    assert obs.from_answered_hop(chain, None) is None
    block = obs.wire_block(ScreeningObservation("", "gpt-5.4", "unknown"))
    assert block == {
        "v": 1,
        "model": "gpt-5.4",
        "selection_stage": "unknown",
        "client_version": VERSION,
    }


def test_the_wire_block_has_the_contract_shape_and_never_an_invented_stage() -> None:
    assert obs.wire_block(PREFERRED) == {
        "v": 1,
        "provider": "openai",
        "model": "gpt-5.4",
        "selection_stage": "preferred",
        "client_version": VERSION,
    }
    odd = obs.wire_block(ScreeningObservation("openai", "gpt-5.4", "primary"))
    assert odd is not None and odd["selection_stage"] == "unknown"
    assert obs.wire_block(None) is None
    assert obs.wire_block(ScreeningObservation("openai", "", "preferred")) is None


# -- the verdict is byte-identical with and without the block ---------------


def test_the_verdict_is_identical_with_and_without_the_observation(tmp_path, monkeypatch) -> None:
    """Strip the producer and re-run the same cycle: only ``observation`` differs."""

    from local_operator.team_bridge import screening

    def run(directory: Path) -> tuple[Any, list[dict[str, Any]]]:
        directory.mkdir()
        _store, owner = _setup(directory)
        _cascade_config(directory, ["deepseek/deepseek-flash"])
        wire = _Wire({"deepseek-flash": ("reply", DELIVER)})
        _install_wire(monkeypatch, wire, failover={"gpt-5.4": _blocked_credential()})
        return _cycle(directory, owner), wire.sends

    with_block, sends_with = run(tmp_path / "with")
    monkeypatch.setattr(screening, "_answered_observation", lambda _runner, _model: None)
    without, sends_without = run(tmp_path / "without")

    assert with_block.receipt.observation is not None and without.receipt.observation is None
    assert with_block.receipt == without.receipt  # observation is compare=False
    assert with_block.receipt.tuple_key() == without.receipt.tuple_key()
    assert json.dumps(with_block.receipt.to_json()) == json.dumps(without.receipt.to_json())
    assert "observation" not in json.dumps(with_block.receipt.to_json())
    for field in ("disposition", "hold_code", "retry_at", "screened_bytes", "attachments_hash"):
        assert getattr(with_block, field) == getattr(without, field)
    assert with_block.receipt.screener_preferred == without.receipt.screener_preferred
    # The model was asked the same thing, the same number of times.
    assert sends_with == sends_without


# -- the wire: real client, MockTransport -----------------------------------


def _capture() -> tuple[list[httpx.Request], Any]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
        if request.url.path.endswith("/ack"):
            # An agreeing hub answers exactly what it answered before #332:
            # nothing about the observation comes back, and none is needed.
            return _ok({"delivery": {"state": "read", "state_at": "t"}})
        return _ok({"status": "recorded"})

    return seen, handler


async def _ack(client: Any, observation: ScreeningObservation | None) -> dict[str, Any]:
    return await client.ack(
        MID,
        tenant_id="tnt_1",
        state="read",
        screening=_receipt().to_json(),
        client_ref="ref-1",
        screening_observation=observation,
    )


async def _admit(client: Any, observation: ScreeningObservation | None, **extra: Any) -> Any:
    return await client.conversation_admit(
        CID,
        MID,
        verdict="allow",
        body_hash=SHA,
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        screening_observation=observation,
        **extra,
    )


def _same_request_but_the_block(with_block: httpx.Request, without: httpx.Request) -> None:
    assert (with_block.method, str(with_block.url)) == (without.method, str(without.url))
    loaded = json.loads(with_block.content)
    block = loaded.pop(obs.WIRE_KEY)
    assert block["selection_stage"] == "preferred" and block["v"] == 1
    # Re-serialised the way the client serialises: the rest is byte-identical.
    assert json.loads(without.content) == loaded
    assert obs.WIRE_KEY.encode() not in without.content
    # BYTE level: the block is appended as the LAST key, so every byte of the
    # block-less body up to its closing brace is reproduced verbatim.
    assert without.content.endswith(b"}")
    assert with_block.content.startswith(without.content[:-1])
    headers = {k: v for k, v in with_block.headers.items() if k != "content-length"}
    assert headers == {k: v for k, v in without.headers.items() if k != "content-length"}


@pytest.mark.asyncio
async def test_status_primes_the_gate_and_the_block_rides_ack_and_admit_beside_the_verdict() -> (
    None
):
    seen, handler = _capture()
    client = _make_client(handler)
    assert not obs.hub_accepts(client.base_url)
    await client.status()
    assert obs.hub_accepts(client.base_url)
    # Same hub spelled without its path (the pull lane's default client).
    assert obs.hub_accepts("https://HUB.example/")
    assert not obs.hub_accepts("https://other.example/v1")

    out_with = await _ack(client, PREFERRED)
    out_without = await _ack(client, None)
    assert out_with == out_without  # the client reads nothing back about the block
    _same_request_but_the_block(seen[1], seen[2])
    ack_body = json.loads(seen[1].content)
    assert "screening_observation" not in ack_body["screening"]  # beside, never inside

    await _admit(client, PREFERRED)
    await _admit(client, None)
    _same_request_but_the_block(seen[3], seen[4])


@pytest.mark.asyncio
async def test_a_hub_that_never_advertised_the_block_never_receives_it() -> None:
    """An older hub: a pre-#332 admit decoder refuses unknown keys (400)."""

    seen: list[httpx.Request] = []

    def strict_old_hub(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {}})
        if obs.WIRE_KEY.encode() in request.content:
            return httpx.Response(400, json={"error": "invalid request body", "code": "invalid"})
        if request.url.path.endswith("/ack"):
            return _ok({"delivery": {"state": "read", "state_at": "t"}})
        return _ok({"status": "recorded"})

    client = _make_client(strict_old_hub)
    await client.status()
    assert await _admit(client, PREFERRED) == {"status": "recorded"}
    await _ack(client, PREFERRED)
    assert all(obs.WIRE_KEY.encode() not in r.content for r in seen)
    assert len(seen) == 3  # no extra request, no retry


@pytest.mark.asyncio
async def test_a_hub_that_stops_advertising_stops_receiving_the_block() -> None:
    features: dict[str, Any] = {obs.OBSERVATION_FEATURE: 1}
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": dict(features)})
        return _ok({"status": "recorded"})

    client = _make_client(handler)
    await client.status()
    await _admit(client, PREFERRED)
    features[obs.OBSERVATION_FEATURE] = 2  # an unknown version is not ours
    await client.status()
    await _admit(client, PREFERRED)
    assert obs.WIRE_KEY.encode() in seen[1].content
    assert obs.WIRE_KEY.encode() not in seen[3].content


@pytest.mark.asyncio
async def test_the_bounded_retries_make_the_same_attempts_with_or_without_the_block() -> None:
    """The refusal_class 400 retry runs exactly as before; the block never decides it."""

    async def attempts(with_block: bool) -> list[dict[str, Any]]:
        obs.reset_hub_support()
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/messaging/status"):
                return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
            body = json.loads(request.content)
            bodies.append(body)
            if "refusal_class" in body:
                return httpx.Response(400, json={"error": "refused", "code": "invalid_field"})
            return _ok({"verdict": "reject"})

        client = _make_client(handler)
        await client.status()
        await client.conversation_admit(
            CID,
            MID,
            verdict="reject",
            body_hash="h",
            refusal_class="abusive",
            screening_observation=PREFERRED if with_block else None,
        )
        return bodies

    with_block, without = await attempts(True), await attempts(False)
    assert [("refusal_class" in b) for b in with_block] == [True, False]
    assert [{k: v for k, v in b.items() if k != obs.WIRE_KEY} for b in with_block] == without
    # The 400 rode the block, so the gate is cleared (S-1) and the existing
    # retry goes out without it: same attempts, the block only on the first.
    assert [(obs.WIRE_KEY in b) for b in with_block] == [True, False]


_INVALID = {"error": "invalid request body", "code": "invalid"}


class _Clock:
    """The latch's monotonic clock, moved by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _admits(seen: list[httpx.Request]) -> list[httpx.Request]:
    return [r for r in seen if r.url.path.endswith("/admit")]


def _carried(request: httpx.Request) -> bool:
    return obs.WIRE_KEY.encode() in request.content


def test_a_full_rollback_admits_on_the_first_drive_and_never_re_screens(
    tmp_path, monkeypatch
) -> None:
    """Q-2, full rollback: the hub advertised, then every pod went pre-#332.

    The quota-refused preferred screener leaves a FALLBACK allow — the receipt
    whose unsent re-drive is retired and screened again. The 400 the block
    draws must not become that re-drive: the admit is resent once without the
    block and recorded, so the vendor is called exactly once, and nothing
    after the 400 carries the block again.
    """

    _store, owner = _setup(tmp_path)
    _cascade_config(tmp_path, ["deepseek/deepseek-flash"])
    wire = _Wire({"deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire, failover={"gpt-5.4": _blocked_credential()})
    result = _cycle(tmp_path, owner)
    assert result.receipt.observation == ScreeningObservation(
        "deepseek", "deepseek-flash", "fallback"
    )

    rolled_back = {"done": False}
    seen: list[httpx.Request] = []

    def hub(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            features = {} if rolled_back["done"] else {obs.OBSERVATION_FEATURE: 1}
            return _ok({"features": features})
        if _carried(request):
            return httpx.Response(400, json=_INVALID)
        return _ok({"status": "recorded"})

    async def drive() -> list[Any]:
        client = _make_client(hub)
        await client.status()
        rolled_back["done"] = True  # lands between the status read and the admit
        answers = [await _admit(client, result.receipt.observation)]
        for _ in range(2):  # later messages on the rolled-back hub
            await client.status()
            answers.append(await _admit(client, result.receipt.observation))
        return answers

    assert asyncio.run(drive()) == [{"status": "recorded"}] * 3
    assert [_carried(r) for r in _admits(seen)] == [True, False, False, False]
    assert len(wire.sends) == 1  # one paid screen: no hold, no re-screen


@pytest.mark.asyncio
async def test_two_hub_versions_behind_one_origin_keep_the_block_off_across_re_drives() -> None:
    """Q-2, mixed versions: status keeps advertising, admits alternate pods.

    The conversation lane reads status before EVERY admit, so a fresh
    advertisement precedes each one; the latch must outrank it, or each admit
    that lands on the pre-#332 pod 400s again.
    """

    seen: list[httpx.Request] = []
    pods = iter(["pre", "post"] * 4)

    def hub(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
        if next(pods) == "pre" and _carried(request):
            return httpx.Response(400, json=_INVALID)
        return _ok({"status": "recorded"})

    client = _make_client(hub)
    for _ in range(3):
        await client.status()
        assert await _admit(client, PREFERRED) == {"status": "recorded"}

    assert [_carried(r) for r in _admits(seen)] == [True, False, False, False]
    assert obs.hub_accepts(client.base_url) is False

    # The pair ack shares the origin's latch.
    await client.status()
    await _ack(client, PREFERRED)
    assert not _carried(seen[-1])


@pytest.mark.asyncio
async def test_the_block_returns_only_once_the_cool_down_has_elapsed(monkeypatch) -> None:
    clock = _Clock()
    # raising=False: the same test must fail on the pre-latch head by behaviour.
    monkeypatch.setattr(obs, "_monotonic", clock, raising=False)
    seen: list[httpx.Request] = []
    upgraded = {"done": False}

    def hub(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
        if not upgraded["done"] and _carried(request):
            return httpx.Response(400, json=_INVALID)
        return _ok({"status": "recorded"})

    client = _make_client(hub)
    await client.status()
    await _admit(client, PREFERRED)  # 400, then resent without the block
    upgraded["done"] = True

    clock.now += obs.REFUSAL_COOLDOWN_S - 1
    await client.status()
    await _admit(client, PREFERRED)
    assert not _carried(seen[-1])  # an advertisement inside the window is outranked

    clock.now += 2
    await client.status()
    await _admit(client, PREFERRED)
    assert _carried(seen[-1])  # the latch expired: the advertisement counts again
    assert [_carried(r) for r in _admits(seen)] == [True, False, False, True]


@pytest.mark.asyncio
async def test_a_400_to_a_request_without_the_block_changes_nothing(monkeypatch) -> None:
    """A 400 the block did not ride says nothing about the block.

    It is raised after one request (no resend), opens no latch, and neither
    clears nor extends one already running.
    """

    clock = _Clock()
    # raising=False: the same test must fail on the pre-latch head by behaviour.
    monkeypatch.setattr(obs, "_monotonic", clock, raising=False)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
        return httpx.Response(400, json=_INVALID)

    client = _make_client(handler)
    await client.status()
    with pytest.raises(tb_client.BridgeRefusal):
        await _admit(client, None)
    assert len(_admits(seen)) == 1
    assert obs.hub_accepts(client.base_url) is True

    # Open a latch (the block's 400 and its resend's 400), then a plain 400
    # late in the window: the latch neither ends nor is pushed back.
    with pytest.raises(tb_client.BridgeRefusal):
        await _admit(client, PREFERRED)
    await client.status()
    assert obs.hub_accepts(client.base_url) is False
    clock.now += obs.REFUSAL_COOLDOWN_S - 1
    await client.status()
    with pytest.raises(tb_client.BridgeRefusal):
        await _admit(client, None)
    assert obs.hub_accepts(client.base_url) is False
    clock.now += 2
    assert obs.hub_accepts(client.base_url) is True


@pytest.mark.asyncio
async def test_an_ack_400_with_the_block_latches_but_is_never_resent() -> None:
    """S-r3-1: no hub version 400s an ack BECAUSE of the block.

    The ack decoder is tolerant both before and after #332, so a 400 there is
    about the body itself and a block-free resend would only repeat it. The
    latch still arms (it costs only telemetry); the refusal is raised after
    exactly one request.
    """

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
        return httpx.Response(400, json=_INVALID)

    client = _make_client(handler)
    await client.status()
    with pytest.raises(tb_client.BridgeRefusal) as refused:
        await _ack(client, PREFERRED)
    assert refused.value.http_status == 400
    acks = [r for r in seen if r.url.path.endswith("/ack")]
    assert [_carried(r) for r in acks] == [True]
    await client.status()
    assert obs.hub_accepts(client.base_url) is False


@pytest.mark.asyncio
async def test_the_latch_and_the_resend_each_log_one_origin_only_line(caplog) -> None:
    """S-r3-2: an operator can see the latch arm and the resend fire.

    Each line names the hub ORIGIN and nothing else — no body, verdict,
    account or model — so the exact rendered messages are pinned here.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messaging/status"):
            return _ok({"features": {obs.OBSERVATION_FEATURE: 1}})
        if _carried(request):
            return httpx.Response(400, json=_INVALID)
        return _ok({"status": "recorded"})

    client = _make_client(handler)
    await client.status()
    caplog.set_level("INFO", logger="local_operator.team_bridge")
    assert await _admit(client, PREFERRED) == {"status": "recorded"}
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("local_operator")]
    assert lines == [
        "team_bridge: hub https://hub.example refused the screening observation;"
        " latched off for this origin",
        "team_bridge: hub https://hub.example refused an admit carrying the screening"
        " observation; resending once without it",
    ]


# -- the pull lane: the block rides the first send of an intent only --------


def _pull_rig(tmp_path: Path) -> tuple[Any, Any, Any]:
    from tests.unit.team_bridge.test_pull import FakeWire
    from tests.unit.team_bridge.test_pull import _setup as pull_setup

    _store, owner, access = pull_setup(tmp_path)
    return owner, access, FakeWire()


@pytest.mark.asyncio
async def test_pull_ack_forwards_the_receipts_observation(tmp_path: Path) -> None:
    owner, access, wire = _pull_rig(tmp_path)
    outcome = await pull.ack(
        owner,
        "bmsg_1",
        "read",
        _receipt(PREFERRED),
        config_dir=tmp_path,
        access=access,
        client=wire,
    )
    assert outcome.state == pull.ACK_CONFIRMED
    ((name, kwargs),) = [c for c in wire.calls if c[0] == "ack"]
    assert kwargs["screening_observation"] == PREFERRED


@pytest.mark.asyncio
async def test_missing_provenance_sends_the_exact_pre_block_call(tmp_path: Path) -> None:
    owner, access, wire = _pull_rig(tmp_path)
    await pull.ack(
        owner, "bmsg_1", "read", _receipt(None), config_dir=tmp_path, access=access, client=wire
    )
    ((_name, kwargs),) = [c for c in wire.calls if c[0] == "ack"]
    assert "screening_observation" not in kwargs
    assert set(kwargs) == {"tenant_id", "state", "screening", "client_ref", "access", "message_id"}


@pytest.mark.asyncio
async def test_a_replayed_or_rechecked_intent_carries_no_block(tmp_path: Path) -> None:
    """The recorded intent never stores the observation: a replay omits it."""

    owner, access, wire = _pull_rig(tmp_path)
    first = await pull.ack(
        owner,
        "bmsg_1",
        "read",
        _receipt(PREFERRED),
        config_dir=tmp_path,
        access=access,
        client=wire,
    )
    # The confirmed intent's standing re-check (a re-driven handoff).
    await pull.flush_ack(
        owner,
        first.intent_id,
        config_dir=tmp_path,
        access=access,
        client=wire,
        recheck_confirmed=True,
    )
    acks = [kwargs for name, kwargs in wire.calls if name == "ack"]
    assert len(acks) == 2
    assert "screening_observation" in acks[0]
    assert "screening_observation" not in acks[1]


# -- B4b: the owner's Deliver reports the released hold's screener -----------


def test_b4b_release_reports_the_screener_whose_hold_was_released() -> None:
    row = {
        "screener_provider": "anthropic",
        "screener_model": "claude-sonnet-5-5",
        "screener_preferred": 1,
    }
    assert obs.from_release_row(row) == ScreeningObservation(
        "anthropic", "claude-sonnet-5-5", "preferred"
    )
    # A stored 0 is the legacy default, not evidence of a fallback.
    legacy = obs.from_release_row({**row, "screener_preferred": 0})
    assert legacy is not None and legacy.selection_stage == "unknown"
    assert obs.from_release_row({**row, "screener_model": ""}) is None


def test_b4b_owner_deliver_admits_with_the_released_screeners_block(
    rig: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811 - the imported B4b fixture
) -> None:
    """The shipped spine: hold, the person's signed Deliver, then the admit."""

    from tests.unit.team_bridge import test_conversation_delivery as tcd
    from tests.unit.team_bridge import test_hold_release as thr

    config_dir, store, owner = rig
    admits: list[dict[str, Any]] = []
    original = tcd.FakeWire.conversation_admit

    async def recording(self: Any, cid: str, mid: str, **kwargs: Any) -> Any:
        admits.append(dict(kwargs))
        return await original(self, cid, mid, **kwargs)

    monkeypatch.setattr(tcd.FakeWire, "conversation_admit", recording)
    server, wakes, run = thr._spine(monkeypatch, config_dir, owner, thr.Runner(thr.NEW_WORK))
    held = run()
    assert held["action"] == "hold" and admits == []

    thr._decide(config_dir, owner, "deliver")
    delivered = run()

    assert delivered["action"] == "delivered" and len(wakes) == 1
    assert server.admits == [
        {"endpoint": "bep_recv", "message_id": thr.MESSAGE_ID, "verdict": "allow"}
    ]
    (admit,) = admits
    assert admit["screening_observation"] == ScreeningObservation(
        "anthropic", "claude-sonnet-5-5", "preferred"
    )
    # The release itself is untouched: same outcome row the B4b suite pins.
    assert store.owner_release(owner, thr.MESSAGE_ID)["outcome"] == "admitted"


def test_a_replayed_owner_deliver_sends_no_block_on_either_setting(
    rig: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811 - the imported B4b fixture
) -> None:
    """S-2 / m2: a re-driven Deliver of an ALREADY consumed release omits the block.

    Same rule as a crash-replayed ack intent: the hub counts per request, so a
    replay must not count the screen again, and on the setting-off path the
    release row is not signature-verified before the replay is answered, so
    nothing may be read from it for the hub.
    """

    from tests.unit.team_bridge import test_hold_release as thr

    config_dir, store, owner = rig
    thr._screen(config_dir, owner, thr.Runner(thr.NEW_WORK))
    thr._decide(config_dir, owner, "deliver")
    first = thr._screen(config_dir, owner, thr.Runner(thr.DELIVER))
    assert first.disposition == "deliver" and first.receipt is not None
    assert first.receipt.observation is not None

    again = thr.Runner(thr.DELIVER)
    replay_on = thr._screen(config_dir, owner, again)
    assert replay_on.disposition == "deliver" and again.calls == 0
    assert replay_on.receipt is not None
    assert replay_on.receipt.observation is None

    thr._setting(config_dir, False)
    later = thr.Runner(thr.DELIVER)
    replay_off = thr._screen(config_dir, owner, later)
    assert replay_off.disposition == "deliver" and later.calls == 0
    assert replay_off.receipt is not None
    assert replay_off.receipt.observation is None
