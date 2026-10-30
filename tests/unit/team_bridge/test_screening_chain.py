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
import time
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
from local_operator.team_bridge.store import (
    Store,
    _owner_key,
    policy_serving_models,
    receipt_authority,
    state_db_path,
)
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
            "qa-model": ("fail", 504),
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


# -- remediation round 1 ------------------------------------------------------


def _policy_and_receipt(tmp_path: Path, owner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        db.row_factory = sqlite3.Row
        policy = db.execute(
            "SELECT * FROM screening_policy WHERE owner_key = ?", (_owner_key(owner),)
        ).fetchone()
        receipt = db.execute(
            "SELECT * FROM screening_receipts WHERE owner_key = ?", (_owner_key(owner),)
        ).fetchone()
    return dict(policy), dict(receipt)


def test_hop1_allow_is_actually_delivered_downstream(tmp_path, monkeypatch) -> None:
    """B1/S-3: a fallback hop's allow must survive EVERY downstream fence.

    Screening stopping at ``deliver`` proved nothing: the receipt said
    ``deepseek-flash`` while the published default stayed ``gpt-5.4``, and each
    fence compared the two for equality. This drives the real request-delivery
    stamp (``write_screened_request_delivery``, the wake hand-off), the envelope
    re-validation the receiving session runs (``verify_original_envelope_current``),
    the epoch re-attestation and the read slice's ``receipt_authority``.
    """

    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", 503), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    item = _item()
    item["message"]["type"] = "request"
    item["message"]["request"] = {"state": "pending"}

    result = _cycle(tmp_path, owner)
    assert result.disposition == "deliver" and result.receipt.model == "deepseek-flash"

    policy, receipt = _policy_and_receipt(tmp_path, owner)
    assert policy["default_model"] == "gpt-5.4"
    assert policy_serving_models(policy) == {"gpt-5.4", "deepseek-flash"}
    authority, damage, _ = receipt_authority(owner, policy, receipt)
    assert damage is None and authority is not None and authority["verdict"] == "allow"
    # The receipt is current: re-attestation has nothing to move, and it
    # matches by membership (it would return False on the pre-fix equality).
    assert store.reattest_receipt_epoch(owner, "bmsg_1") is False

    # The real stamp path, with a real held claim for the designated session.
    store.update_designated_session(owner, "sess_1")
    store.claim(owner, "bmsg_1", "sess_1", "tok-1", time.time() + 300)
    envelope = store.write_screened_request_delivery(
        owner,
        delivery_ref="dref_1",
        message_id="bmsg_1",
        designated_session="sess_1",
        display_digest="d" * 64,
        body_sha256=receipt["body_sha256"],
        claim_token="tok-1",
        request={
            "thread_id": "bth_1",
            "message_kind": "request",
            "request_state": "pending",
            "sender_account_id": "acc_sender1",
            "sender_endpoint_id": "bep_s",
            "recipient_account_id": owner.account_id,
            "recipient_endpoint_id": owner.endpoint_id,
        },
    )
    assert envelope["model"] == "deepseek-flash"
    assert store.verify_original_envelope_current(owner, envelope) == ("", "")
    assert _rows(tmp_path, "SELECT model FROM screened_request_deliveries") == [("deepseek-flash",)]


def test_a_model_outside_the_published_chain_is_still_stale(tmp_path) -> None:
    """The fence widened to the chain, not to anything: a receipt for a model the
    policy never published stays ``receipt_stale``; a legacy NULL means
    ``{default_model}`` only, and a damaged value never widens it."""

    store, owner = _setup(tmp_path)
    base = {"default_model": "gpt-5.4"}
    assert policy_serving_models(base) == {"gpt-5.4"}
    assert policy_serving_models({**base, "approved_serving": None}) == {"gpt-5.4"}
    assert policy_serving_models({**base, "approved_serving": "not json"}) == {"gpt-5.4"}
    assert policy_serving_models({**base, "approved_serving": '[["x"]]'}) == {"gpt-5.4"}
    chain = {**base, "approved_serving": '[["deepseek","deepseek-flash"],["openai","gpt-5.4"]]'}
    receipt = {
        "template_sha256": "t",
        "model": "qa-other",
        "admission_revision": "r",
        "policy_epoch": 1,
        "creator_generation": owner.enrollment_generation,
        "categories": "[]",
        "verdict": "allow",
    }
    policy = {**chain, "template_sha256": "t", "revision": "r", "epoch": 1}
    assert receipt_authority(owner, policy, receipt)[1] == "receipt_stale"
    assert receipt_authority(owner, policy, {**receipt, "model": "deepseek-flash"})[1] is None


def test_legacy_policy_table_gains_approved_serving_and_keeps_its_row(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        db.execute("DROP TABLE screening_policy")
        db.execute(
            "CREATE TABLE screening_policy (owner_key TEXT PRIMARY KEY, version TEXT NOT NULL, "
            "template_sha256 TEXT NOT NULL, default_model TEXT, revision TEXT NOT NULL, "
            "epoch INTEGER NOT NULL, accepted_at REAL NOT NULL)"
        )
        db.execute(
            "INSERT INTO screening_policy VALUES (?,?,?,?,?,?,?)",
            (_owner_key(owner), "v1", "t" * 64, "gpt-5.4", "rev-old", 4, 1.0),
        )
    reopened = Store(tmp_path)
    assert reopened.current_screening_policy(owner) == {
        "version": "v1",
        "template_sha256": "t" * 64,
        "default_model": "gpt-5.4",
        "revision": "rev-old",
        "epoch": 4,
    }
    epoch = reopened.publish_screening_policy(
        owner,
        version="v1",
        template_sha256="t" * 64,
        default_model="gpt-5.4",
        revision="rev-new",
        expected_epoch=4,
        approved_serving=[("openai", "gpt-5.4"), ("deepseek", "deepseek-flash")],
    )
    assert epoch == 5
    assert _rows(tmp_path, "SELECT approved_serving FROM screening_policy") == [
        ('[["deepseek", "deepseek-flash"], ["openai", "gpt-5.4"]]',)
    ]


def test_editing_the_chain_advances_the_admission_revision(tmp_path, monkeypatch) -> None:
    """S-4: the chain in force is part of the authority a receipt binds to."""

    store, owner = _setup(tmp_path)
    wire = _Wire({"gpt-5.4": ("reply", DELIVER), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    revisions = []
    for chain in (None, ["deepseek/deepseek-flash"], ["deepseek/deepseek-flash"], None):
        _config(tmp_path, chain=chain)
        _cycle(tmp_path, owner, message_id=f"bmsg_{len(revisions)}")
        revisions.append((store.current_screening_policy(owner) or {})["revision"])
    single, chained, same, back = revisions
    assert single != chained, "adding a fallback hop must advance the revision"
    assert chained == same, "an unchanged chain must not churn the revision"
    assert back == single, "a one-hop chain keeps the pre-chain revision byte-for-byte"


# -- S-1/S-2: advance only on failures a sender cannot cause -------------------


def test_input_refusal_400_on_hop0_holds_and_never_reaches_hop1(tmp_path, monkeypatch) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", 400), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "provider_error"
    assert [s["model"] for s in wire.sends] == ["gpt-5.4"]
    assert _rows(tmp_path, "SELECT 1 FROM screening_receipts") == []
    assert _rows(tmp_path, "SELECT failure_class FROM screening_holds") == [("other",)]


@pytest.mark.parametrize("status", [400, 404, 413, 422, 500, 501])
def test_sender_triggerable_statuses_stop_the_walk(tmp_path, monkeypatch, status) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", status), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    assert _cycle(tmp_path, owner).disposition == "hold"
    assert [s["model"] for s in wire.sends] == ["gpt-5.4"]


@pytest.mark.parametrize("status", [401, 403, 429, 502, 503, 504, 529])
def test_route_failures_advance_to_hop1(tmp_path, monkeypatch, status) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("fail", status), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    result = _cycle(tmp_path, owner)
    assert result.disposition == "deliver" and result.receipt.model == "deepseek-flash"
    assert [s["model"] for s in wire.sends] == ["gpt-5.4", "deepseek-flash"]


def test_guard_refusal_stops_the_walk(tmp_path, monkeypatch) -> None:
    """S-2: the wire guard's own fail-closed refusal (here: the hourly HTTP
    budget is refused at the send) is local authority, not provider weather — it holds
    with zero sends and is not retried on every hop."""

    store, owner = _setup(tmp_path)
    _config(tmp_path, chain=["deepseek/deepseek-flash"])
    wire = _Wire({"gpt-5.4": ("reply", DELIVER), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    from local_operator.team_bridge.store import StoreRefused

    def refuse(self: Store, *args: Any, **kwargs: Any) -> str:
        # The guard's per-send reservation refusing (the revoked-authority
        # class: owner moved / policy stale / budget spent at the send).
        raise StoreRefused("budget_exhausted", "spent")

    monkeypatch.setattr(Store, "reserve_screening_dispatch", refuse)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "provider_error"
    assert wire.sends == []
    holds = _rows(tmp_path, "SELECT failure_class FROM screening_holds")
    assert holds == [("other",)], "one hop tried, not a chain_exhausted walk"


def test_stream_refusals_stop_the_walk_without_building_hop1() -> None:
    """Every _ScreeningStreamRefused holds (superseding review M1)."""

    hops = [
        DefaultSelection("openai", "gpt-5.4", build_model_spec("openai", "gpt-5.4")),
        DefaultSelection(
            "deepseek", "deepseek-flash", build_model_spec("deepseek", "deepseek-flash")
        ),
    ]
    for reason in (
        "images_unsupported",
        "error",
        "aborted",
        "missing_stop",
        "missing_end",
        "over_cap",
        "length",
        "refusal",
        "tool_use",
    ):
        built: list[str] = []

        class Hop:
            async def acall(self, prompt: str, *, default_model: str, **_: Any) -> Any:
                raise _ScreeningStreamRefused(reason)

        def build(hop: DefaultSelection, built: list[str] = built) -> Any:
            built.append(hop.model_id)
            return Hop()

        with pytest.raises(_ScreeningStreamRefused):
            asyncio.run(ScreenChainRunner(hops, build).acall("p", default_model="gpt-5.4"))
        assert built == ["gpt-5.4"], reason


def test_advance_classifier_reads_raw_types_never_prose() -> None:
    from local_operator.providers.auth_store import AuthStoreError
    from local_operator.team_bridge.model_boundary import (
        _scrub_wire_failure,
        _sender_cannot_cause,
    )

    assert _sender_cannot_cause(httpx.ConnectError("x"))
    assert _sender_cannot_cause(httpx.ReadTimeout("x"))
    assert _sender_cannot_cause(asyncio.TimeoutError())
    assert _sender_cannot_cause(AuthStoreError("refresh failed"))
    assert _sender_cannot_cause(ProviderError(503, "x"))
    assert _sender_cannot_cause(ProviderError(429, "x"))
    assert _sender_cannot_cause(ProviderError(401, "x", auth_error=True))
    for status in (400, 404, 413, 422, 500, 501):
        assert not _sender_cannot_cause(ProviderError(status, "x")), status
    assert not _sender_cannot_cause(RuntimeError("503 timeout quota"))
    assert not _sender_cannot_cause(ProviderError(None, "upstream 503 timed out"))
    assert not _sender_cannot_cause(
        ProviderError(409, "x", kind="request", request_sent=False, fail_closed=True)
    )
    # The scrubber carries the raw decision as a typed flag across the scrub.
    assert getattr(_scrub_wire_failure(ProviderError(503, "x")), "tb_screen_advance") is True
    assert getattr(_scrub_wire_failure(ProviderError(400, "x")), "tb_screen_advance") is False


def test_store_clamps_failure_class_to_the_closed_set(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    store.persist_hold(owner, "bmsg_a", "provider_error", failure_class="VENDOR PROSE")
    store.persist_hold(owner, "bmsg_b", "provider_error", failure_class="quota")
    store.persist_hold(owner, "bmsg_c", "model_hold")
    assert sorted(_rows(tmp_path, "SELECT message_id, failure_class FROM screening_holds")) == [
        ("bmsg_a", "other"),
        ("bmsg_b", "quota"),
        ("bmsg_c", None),
    ]
