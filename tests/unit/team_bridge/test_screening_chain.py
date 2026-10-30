"""Screening walks the operator's configured fallback chain (live incident fix).

THE INCIDENT: a screening request is ``isolated=True`` and failover disables
its own retry/fallback for isolated requests, so screening made ONE attempt on
the configured default. 47 ProviderErrors on that default reached the
screening breaker and parked inbound messages while the configured
``retry.fallbackChains.default`` was never tried.

THE RIG: the SHIPPED composition (``run_screening_cycle``) with the REAL
production factory (``make_screening_runner``): its wire guard, the charging
transport proxy, the selected-route check at ``_transport_for_url``, the
per-hop (provider, model) tuple check, the pipeline's step-13 identity check,
the durable receipt, breaker and hold rows. Only two things are faked: the
vendor wire client that ``create_stream_fn`` would build (it POSTs through the
dedicated guarded client exactly as a real client does) and the physical
transport underneath (``httpx.MockTransport``). No network.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamModelEvent,
    StreamTextDelta,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.failover import ProviderError
from local_operator.team_bridge import scheduler as _scheduler
from local_operator.team_bridge.model_boundary import (
    DefaultSelection,
    ScreenChainRunner,
    ScreenModelRunner,
    _ScreeningStreamRefused,
    resolve_screening_chain,
    screening_failure_class,
)
from local_operator.team_bridge.store import Store, _owner_key, state_db_path
from tests.unit.team_bridge.test_screening import _consent_status, _item, _setup

DELIVER = "VERDICT: deliver\nREASON: ordinary."
HOLD = "VERDICT: hold\nREASON: needs a human."
VENDOR_CANARY = "VENDOR-PROSE-CANARY upstream exploded"


def _config(tmp_path: Path, *, chain: list[str] | None, model_fallback: bool = True) -> None:
    lines = ["values:", "  hosting: openai", "  model_name: gpt-5.4"]
    if chain is not None:
        lines += [
            "  retry:",
            f"    modelFallback: {'true' if model_fallback else 'false'}",
            "    fallbackChains:",
            "      default:",
            *[f"        - {entry}" for entry in chain],
        ]
    (tmp_path / "config.yml").write_text("\n".join(lines) + "\n")


class _Wire:
    """The vendor side: scripted per model id, observed per physical send."""

    def __init__(self, script: dict[str, Any]) -> None:
        # model_id -> ("fail", status) | ("reply", text) | ("reply_as", provider, text)
        self.script = script
        self.sends: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content or b"{}")
        self.sends.append(payload)
        action = self.script[payload["model"]]
        if action[0] == "fail":
            return httpx.Response(action[1], json={"error": VENDOR_CANARY})
        if action[0] == "reply_as":
            return httpx.Response(200, json={"text": action[2], "provider": action[1]})
        return httpx.Response(200, json={"text": action[1]})


def _install_wire(monkeypatch: pytest.MonkeyPatch, wire: _Wire) -> None:
    """Fake ONLY the vendor client and the physical transport."""

    from local_operator.model import configure as _configure

    real_transport = httpx.AsyncHTTPTransport

    def mock_transport(*_args: Any, **_kwargs: Any) -> Any:
        return httpx.MockTransport(wire.handler)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", mock_transport)

    def fake_create_stream_fn(
        auth_store: Any,
        settings: Any = None,
        *,
        session_id: Any = None,
        wrap_client: Any = None,
        http_client: Any = None,
        **_: Any,
    ) -> Any:
        assert session_id == "team-bridge-screen"
        assert wrap_client is not None and http_client is not None

        class VendorClient:
            _http = http_client

            def stream(self, request: Any, api_key: Any, oauth_access: Any = None) -> Any:
                async def gen() -> Any:
                    # Through the dedicated GUARDED client: route check at
                    # selection, then the charging proxy, then the mock wire.
                    response = await http_client.post(
                        f"https://{request.model.provider}.invalid/v1/chat",
                        json={
                            "provider": request.model.provider,
                            "model": request.model.model_id,
                            "temperature": request.temperature,
                            "isolated": request.isolated,
                        },
                    )
                    if response.status_code >= 400:
                        raise ProviderError(response.status_code, VENDOR_CANARY)
                    body = response.json()
                    served = request.model.model_copy(
                        update={"provider": body.get("provider", request.model.provider)}
                    )
                    yield StreamModelEvent(model=served)
                    yield StreamTextDelta(delta=body["text"])
                    yield StreamEndEvent(stop_reason="stop")

                return gen()

        guard = wrap_client(VendorClient())

        class StreamFn:
            def __call__(self, request: Any, signal: Any) -> Any:
                return guard.stream(request, "k")

            async def close(self) -> None:
                return None

        return StreamFn()

    monkeypatch.setattr(_configure, "create_stream_fn", fake_create_stream_fn)
    assert real_transport is not None


def _cycle(tmp_path: Path, owner: Any, message_id: str = "bmsg_1") -> Any:
    raw = _consent_status(owner, {"version": "bridge_consent_v1", "text": "v1 consent"})
    verified = _scheduler.VerifiedStatus(_scheduler._VERIFIED_TOKEN, raw)

    async def fetch() -> Any:
        return verified

    async def drive() -> Any:
        return await _scheduler.run_screening_cycle(
            config_dir=tmp_path,
            owner=owner,
            items=[_item(message_id=message_id)],
            auth_store=None,
            status_fetcher=fetch,
        )

    return asyncio.run(drive())


def _rows(tmp_path: Path, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        return db.execute(sql, args).fetchall()


# -- (a) hop 0 fails, hop 1 serves: one verdict, end-to-end identity accepted --


def test_hop0_provider_error_falls_over_to_hop1_and_delivers(tmp_path, monkeypatch) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", 503), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result is not None and result.disposition == "deliver"
    assert result.receipt.model == "deepseek-flash"
    # Both hops were physically sent THROUGH their own guard (charged rows),
    # each isolated, each at temperature 0.
    assert [(s["provider"], s["model"]) for s in wire.sends] == [
        ("openai", "gpt-5.4"),
        ("deepseek", "deepseek-flash"),
    ]
    assert all(s["isolated"] is True and s["temperature"] == 0.0 for s in wire.sends)
    assert len(_rows(tmp_path, "SELECT 1 FROM screening_attempts WHERE kind = 'http'")) == 2
    # The DURABLE receipt carries hop 1's identity: the per-hop guard, the
    # step-13 check and the finish's approved-map check all accepted it.
    assert _rows(
        tmp_path,
        "SELECT model, verdict FROM screening_receipts WHERE owner_key = ? AND message_id = ?",
        _owner_key(owner),
        "bmsg_1",
    ) == [("deepseek-flash", "allow")]
    assert store.screening_breaker_state(owner) is None
    assert store.screening_hold(owner, "bmsg_1") is None
    # The published policy row stays hop 0's: no epoch/revision churn.
    assert (store.current_screening_policy(owner) or {})["default_model"] == "gpt-5.4"


# -- (b) every hop fails: ONE breaker failure, one hold, chain_exhausted -------


def test_every_hop_failing_is_one_breaker_failure_and_one_classed_hold(
    tmp_path, monkeypatch
) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash", "openrouter/qa-model"])
    wire = _Wire(
        {
            "gpt-5.4": ("fail", 503),
            "deepseek-flash": ("fail", 429),
            "qa-model": ("fail", 500),
        }
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "provider_error"
    assert [s["model"] for s in wire.sends] == ["gpt-5.4", "deepseek-flash", "qa-model"]
    breaker = store.screening_breaker_state(owner)
    assert breaker is not None and int(breaker["failures"]) == 1
    holds = _rows(
        tmp_path,
        "SELECT hold_code, failure_class FROM screening_holds WHERE owner_key = ?",
        _owner_key(owner),
    )
    assert holds == [("provider_error", "chain_exhausted")]
    assert _rows(tmp_path, "SELECT 1 FROM screening_receipts") == []
    # No vendor prose was persisted anywhere.
    assert VENDOR_CANARY.encode() not in state_db_path(tmp_path).read_bytes()


# -- (c) a model HOLD verdict is a verdict: hop 1 is never called --------------


def test_model_hold_verdict_stops_the_walk(tmp_path, monkeypatch) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("reply", HOLD), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "model_hold"
    assert [s["model"] for s in wire.sends] == ["gpt-5.4"]
    assert store.screening_breaker_state(owner) is None
    assert _rows(tmp_path, "SELECT failure_class FROM screening_holds") == [(None,)]


def test_chain_runner_never_builds_hop1_after_a_verdict_or_stream_refusal() -> None:
    hops = [
        DefaultSelection("openai", "gpt-5.4", build_model_spec("openai", "gpt-5.4")),
        DefaultSelection(
            "deepseek", "deepseek-flash", build_model_spec("deepseek", "deepseek-flash")
        ),
    ]
    built: list[str] = []

    class Hop:
        def __init__(self, outcome: Any) -> None:
            self.outcome = outcome

        async def acall(self, prompt: str, *, default_model: str, **_: Any) -> Any:
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    def factory(outcomes: dict[str, Any]) -> Any:
        def build(hop: DefaultSelection) -> Any:
            built.append(hop.model_id)
            return Hop(outcomes[hop.model_id])

        return build

    runner = ScreenChainRunner(
        hops, factory({"gpt-5.4": (HOLD, "gpt-5.4"), "deepseek-flash": (DELIVER, "deepseek-flash")})
    )
    assert asyncio.run(runner.acall("p", default_model="gpt-5.4")) == (HOLD, "gpt-5.4")
    assert built == ["gpt-5.4"]

    # A hop's own reply refused (length cut, tool use) is that model's answer
    # to these bytes: it propagates, and a second screener never overrules it.
    built.clear()
    runner = ScreenChainRunner(
        hops,
        factory(
            {
                "gpt-5.4": _ScreeningStreamRefused("length"),
                "deepseek-flash": (DELIVER, "deepseek-flash"),
            }
        ),
    )
    with pytest.raises(_ScreeningStreamRefused):
        asyncio.run(runner.acall("p", default_model="gpt-5.4"))
    assert built == ["gpt-5.4"]


# -- (d) tuple exactness: a foreign provider serving an approved id is refused --


def test_foreign_provider_serving_an_approved_model_id_is_refused(tmp_path, monkeypatch) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    # hop 0 fails; hop 1's wire claims to be served by ANOTHER provider under
    # the approved id "deepseek-flash". The id alone must not admit.
    wire = _Wire({"gpt-5.4": ("fail", 503), "deepseek-flash": ("reply_as", "openrouter", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "provider_error"
    assert _rows(tmp_path, "SELECT 1 FROM screening_receipts") == []
    assert _rows(tmp_path, "SELECT failure_class FROM screening_holds") == [("chain_exhausted",)]

    # The same refusal at the hop runner itself, without a chain around it.
    selection = DefaultSelection(
        "deepseek", "deepseek-flash", build_model_spec("deepseek", "deepseek-flash")
    )
    foreign = build_model_spec("deepseek", "deepseek-flash").model_copy(
        update={"provider": "openrouter"}
    )

    class Fn:
        def __call__(self, request: Any, signal: Any) -> Any:
            async def gen() -> Any:
                yield StreamModelEvent(model=foreign)
                yield StreamTextDelta(delta=DELIVER)
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    assert ScreenModelRunner(Fn(), selection)("p", default_model="deepseek-flash") == ("", None)


def test_each_hop_approves_only_its_own_tuple(tmp_path) -> None:
    _config(tmp_path, chain=["deepseek/deepseek-flash", "deepseek/deepseek-flash", "openai/*"])
    hops = resolve_screening_chain(tmp_path)
    # Deduped by tuple (``openai/*`` expands to the primary itself and is dropped).
    assert [(h.provider, h.model_id) for h in hops] == [
        ("openai", "gpt-5.4"),
        ("deepseek", "deepseek-flash"),
    ]
    assert hops[1].approved_tuples == frozenset({("deepseek", "deepseek-flash")})
    assert hops[0].approved_tuples == frozenset({("openai", "gpt-5.4")})


# -- (e) modelFallback off: a single attempt, exactly as before ---------------


def test_model_fallback_false_is_a_single_attempt(tmp_path, monkeypatch) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"], model_fallback=False)
    assert [h.model_id for h in resolve_screening_chain(tmp_path)] == ["gpt-5.4"]
    wire = _Wire({"gpt-5.4": ("fail", 503), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "provider_error"
    assert [s["model"] for s in wire.sends] == ["gpt-5.4"]
    breaker = store.screening_breaker_state(owner)
    assert breaker is not None and int(breaker["failures"]) == 1
    # One hop, so the class is the hop's own — not chain_exhausted.
    assert _rows(tmp_path, "SELECT failure_class FROM screening_holds") == [("provider_5xx",)]


def test_failure_class_is_derived_from_type_never_prose() -> None:
    assert screening_failure_class(ProviderError(401, "x", auth_error=True)) == "auth_unusable"
    assert screening_failure_class(ProviderError(429, "x", kind="quota")) == "quota"
    assert screening_failure_class(ProviderError(None, "x", kind="timeout")) == "timeout"
    assert screening_failure_class(ProviderError(502, "x")) == "provider_5xx"
    assert screening_failure_class(_ScreeningStreamRefused("length")) == "stream_refused"
    assert screening_failure_class(asyncio.TimeoutError()) == "timeout"
    assert screening_failure_class(RuntimeError("quota exceeded 503")) == "other"
    tagged = ProviderError(502, "x")
    setattr(tagged, "tb_chain_exhausted", True)
    assert screening_failure_class(tagged) == "chain_exhausted"


# -- (f) migration: a store whose screening_holds predates failure_class -------

_PRE_CHANGE_HOLDS_DDL = """
CREATE TABLE screening_holds (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    hold_code TEXT NOT NULL,
    retry_at REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_key, message_id)
)
"""


def test_old_holds_table_gains_failure_class_and_keeps_its_rows(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    key = _owner_key(owner)
    # Rebuild screening_holds in the exact pre-change shape (schema v8 without
    # the column: the column is additive and moves no stamp) with a live row.
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        db.execute("DROP TABLE screening_holds")
        db.execute(_PRE_CHANGE_HOLDS_DDL)
        db.execute(
            "INSERT INTO screening_holds VALUES (?,?,?,?,?)",
            (key, "bmsg_old", "provider_error", 123.0, 100.0),
        )
    columns = {r[1] for r in _rows(tmp_path, "PRAGMA table_info(screening_holds)")}
    assert "failure_class" not in columns

    reopened = Store(tmp_path)
    # Readers work on the old shape before any writer has run.
    assert reopened.screening_hold(owner, "bmsg_old") == {
        "hold_code": "provider_error",
        "retry_at": 123.0,
        "created_at": 100.0,
    }
    reopened.persist_hold(owner, "bmsg_new", "provider_error", failure_class="quota")

    columns = {r[1] for r in _rows(tmp_path, "PRAGMA table_info(screening_holds)")}
    assert "failure_class" in columns
    assert sorted(
        _rows(
            tmp_path, "SELECT message_id, hold_code, created_at, failure_class FROM screening_holds"
        )
    ) == [
        ("bmsg_new", "provider_error", pytest.approx(0.0, abs=1e12), "quota"),
        ("bmsg_old", "provider_error", 100.0, None),
    ]
    assert _rows(tmp_path, "SELECT value FROM schema_meta WHERE key = 'schema_version'") == [("8",)]
    # A re-code to a different reason drops the stale class; the age is kept.
    assert reopened.advance_hold_retry(owner, "bmsg_new", retry_at=999.0, hold_code="model_hold")
    assert _rows(
        tmp_path,
        "SELECT hold_code, failure_class FROM screening_holds WHERE message_id = ?",
        "bmsg_new",
    ) == [("model_hold", None)]
