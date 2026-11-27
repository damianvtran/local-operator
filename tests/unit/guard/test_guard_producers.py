"""Radient Guard background producers (Guard RFC v0.9 §5.3, §10', §23; slice L3).

The consent is granted through L2a's REAL admitted-accept path against a local
stand-in for agent-server, and every upload goes through the real
``RadientClient`` to the same stand-in, so the request log IS the wire. The
install key lives in an in-memory store: no test reaches the macOS Keychain.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.clients.radient import RadientClient
from local_operator.guard import activity as act
from local_operator.guard import consent as C
from local_operator.guard import decisions as gd
from local_operator.guard import ingest, install_key
from local_operator.guard import posture as pst
from local_operator.guard import producers as P
from local_operator.guard.outbox import GuardOutbox

TENANT = "ten_acme"
TERMS = "sha256:" + "ab" * 32
ACCOUNT = "acct_member_1"
#: 2026-11-27 12:00 UTC (a Friday, ISO week 2026-W48).
NOW = dt.datetime(2026, 11, 27, 12, 0, tzinfo=dt.UTC).timestamp()
YESTERDAY = "2026-11-26"
TODAY = "2026-11-27"


# ---------------------------------------------------------------------------
# Stand-in server
# ---------------------------------------------------------------------------


class _Fixture:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.ingest_replies: list[tuple[int, dict[str, Any]]] = []
        self.lock = threading.Lock()
        self.base = ""

    def ingests(self) -> list[dict[str, Any]]:
        with self.lock:
            return [r for r in self.requests if r["path"] == "/v1/guard/ingest"]


def _handler(fx: _Fixture) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            with fx.lock:
                fx.requests.append(
                    {
                        "path": self.path,
                        "body": body,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
            if self.path == "/v1/guard/consents":
                self._reply(201, {"result": {"kind": "started"}})
            elif self.path == "/v1/guard/consents/revoke":
                self._reply(200, {"result": {"revoked_consents": 1}})
            elif self.path == "/v1/guard/ingest":
                with fx.lock:
                    reply = fx.ingest_replies.pop(0) if fx.ingest_replies else None
                if reply is None:
                    assert body is not None
                    batch = body["payload"]["batch_id"]
                    reply = (202, {"result": {"batch_id": batch, "accepted": 1}})
                self._reply(*reply)
            else:
                self._reply(404, {"code": "not_found", "error": "refused"})

    return Handler


@pytest.fixture
def fx() -> Iterator[_Fixture]:
    fixture = _Fixture()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fixture))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fixture.base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        yield fixture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class MemoryKeyStore:
    name = "memory"

    def __init__(self) -> None:
        self.items: dict[str, str] = {}
        self.writes = 0

    def load(self, install_id: str) -> str | None:
        return self.items.get(install_id)

    def store(self, install_id: str, secret_hex: str) -> None:
        self.writes += 1
        self.items[install_id] = secret_hex

    def delete(self, install_id: str) -> None:
        self.items.pop(install_id, None)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv(install_key.BACKEND_ENV, "file")
    previous_sink = gd.set_decision_sink(None)
    previous_task = act.set_task_sink(None)
    yield config
    gd.set_decision_sink(previous_sink)
    act.set_task_sink(previous_task)


def _client(fx: _Fixture) -> RadientClient:
    return RadientClient(api_key=SecretStr("jwt-person"), base_url=fx.base)


def _grant(
    root: Path, fx: _Fixture, store: MemoryKeyStore, scopes: list[str], rev: int = 1
) -> C.GuardConsentHost:
    host = C.GuardConsentHost(root, transport=_client(fx), key_store=store, mirror=False)
    host.offer(
        C.ConsentOffer.from_wire(
            {
                "tenant": TENANT,
                "org_display_name": "Acme",
                "collection_rev": rev,
                "revocation_gen": 3,
                "scopes": scopes,
                "added_scopes": [],
                "rendered_terms_digest": TERMS,
                "reason": "new_org",
            }
        )
    )
    if host.sticky(TENANT):
        host.share_again(TENANT)
        host.offer(
            C.ConsentOffer.from_wire(
                {
                    "tenant": TENANT,
                    "org_display_name": "Acme",
                    "collection_rev": rev,
                    "revocation_gen": 3,
                    "scopes": scopes,
                    "added_scopes": [],
                    "rendered_terms_digest": TERMS,
                    "reason": "new_org",
                }
            )
        )
    outcome = host.accept(TENANT, rev, capability=True, signature=None, source="tui")
    assert outcome.status == "granted", outcome
    return host


@dataclass
class _Clock:
    now: float = NOW

    def __call__(self) -> float:
        return self.now


@dataclass
class _Rng:
    """Deterministic draws: pops values, then repeats the last."""

    values: list[float] = field(default_factory=lambda: [0.1])

    def __call__(self) -> float:
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def _producers(root: Path, fx: _Fixture, store: MemoryKeyStore, **kw: Any) -> P.GuardProducers:
    calls = kw.pop("calls", None)

    def factory():
        if calls is not None:
            calls.append(1)
        return _client(fx), ACCOUNT

    return P.GuardProducers(
        root,
        transport_factory=factory,
        key_store=store,
        clock=kw.pop("clock", _Clock()),
        rng=kw.pop("rng", _Rng()),
        **kw,
    )


def _decision(day: str = YESTERDAY, outcome: str = "block") -> gd.GuardDecisionRecord:
    return gd.GuardDecisionRecord(outcome, gd.RULE_CLASS_DANGEROUS_DELETE, "shell", day=day)


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# ---------------------------------------------------------------------------
# Zero cost without consent
# ---------------------------------------------------------------------------


def test_no_consent_store_means_no_file_no_call_no_sink(root: Path, fx: _Fixture) -> None:
    calls: list[int] = []
    producers = _producers(root, fx, MemoryKeyStore(), calls=calls)
    before = _files(root)
    for _ in range(3):
        report = producers.tick()
        assert report.skipped == "no_consent_store"
    assert _files(root) == before == []
    assert calls == [] and fx.requests == []
    assert gd.get_decision_sink() is gd.NOOP_SINK and act.get_task_sink() is None


# ---------------------------------------------------------------------------
# Decisions: signed, closed days only, idempotent, revoke stops
# ---------------------------------------------------------------------------


def test_live_consent_uploads_closed_day_decisions_signed_by_the_install_key(
    root: Path, fx: _Fixture
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    store = MemoryKeyStore()
    _grant(root, fx, store, ["decisions"])
    producers = _producers(root, fx, store)
    producers.tick()  # sets the floor and installs the sink
    assert isinstance(gd.get_decision_sink(), P.GatedDecisionSink)
    outbox = GuardOutbox(root)
    old, fresh = _decision(YESTERDAY), _decision(TODAY)
    outbox.append(old)
    outbox.append(fresh)
    report = producers.tick()
    assert report.sent == {"decisions": 1}
    (req,) = fx.ingests()
    env = req["body"]
    payload = env["payload"]
    assert payload["tenant"] == TENANT and payload["collection_rev"] == 1
    assert payload["revocation_gen"] == 3 and payload["schema"] == ingest.BATCH_SCHEMA
    assert [r["record_id"] for r in payload["decisions"]] == [old.record_id]  # today stays local
    assert set(payload) == {
        "schema",
        "tenant",
        "install_id",
        "batch_id",
        "collection_rev",
        "revocation_gen",
        "nonce",
        "ts",
        "decisions",
    }
    assert "origin" not in req["headers"]
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(next(iter(store.items.values()))))
    sig = base64.urlsafe_b64decode(env["signature"] + "==")
    key.public_key().verify(sig, ingest.BATCH_SIGNING_DOMAIN + C.canonical_json(payload))
    # Settled: never sent twice.
    producers.tick()
    assert len(fx.ingests()) == 1


def test_records_queued_before_the_grant_are_never_sent_under_it(root: Path, fx: _Fixture) -> None:
    store = MemoryKeyStore()
    pre = _decision()
    GuardOutbox(root).append(pre)  # an outbox left from an earlier authority
    _grant(root, fx, store, ["decisions"])
    producers = _producers(root, fx, store)
    producers.tick()
    assert fx.ingests() == []


def test_retry_reuses_the_batch_id_and_backs_off(root: Path, fx: _Fixture) -> None:
    store = MemoryKeyStore()
    _grant(root, fx, store, ["decisions"])
    clock = _Clock()
    producers = _producers(root, fx, store, clock=clock)
    producers.tick()
    GuardOutbox(root).append(_decision())
    fx.ingest_replies.append((503, {"code": "unavailable", "error": "try later"}))
    assert producers.tick().results == ["retry"]
    clock.now += 10  # inside the backoff window: nothing sent
    producers.tick()
    assert len(fx.ingests()) == 1
    clock.now += P.BACKOFF_CAP_S
    producers.reset_transport()
    assert producers.tick().results == ["accepted"]
    first, second = fx.ingests()
    assert first["body"]["payload"]["batch_id"] == second["body"]["payload"]["batch_id"]
    assert first["body"]["payload"]["nonce"] != second["body"]["payload"]["nonce"]


def test_revoke_pending_stops_uploads_and_drops_the_queue(root: Path, fx: _Fixture) -> None:
    store = MemoryKeyStore()
    host = _grant(root, fx, store, ["decisions"])
    producers = _producers(root, fx, store)
    producers.tick()
    GuardOutbox(root).append(_decision())
    # The revoke call fails: the consent is left revoke_pending (N-3 = stopped).
    host._transport = None
    host.revoke(TENANT)
    assert (host.consent(TENANT) or {}).get("state") == "revoke_pending"
    report = producers.tick()
    assert report.skipped == "no_live_consent" and fx.ingests() == []
    assert GuardOutbox(root).count() == 0
    assert gd.get_decision_sink() is gd.NOOP_SINK


def test_server_revocation_refusal_records_a_console_revoke(root: Path, fx: _Fixture) -> None:
    store = MemoryKeyStore()
    host = _grant(root, fx, store, ["decisions"])
    producers = _producers(root, fx, store)
    producers.tick()
    GuardOutbox(root).append(_decision())
    fx.ingest_replies.append((409, {"code": "guard_revocation_stale", "error": "revoked"}))
    assert producers.tick().results == ["revoked"]
    assert (host.consent(TENANT) or {}).get("state") == "revoked"
    assert host.sticky(TENANT) == C.STICKY_REVOKED
    producers.tick()
    assert len(fx.ingests()) == 1 and GuardOutbox(root).count() == 0


def test_paused_consent_sends_nothing_until_a_new_grant(root: Path, fx: _Fixture) -> None:
    store = MemoryKeyStore()
    _grant(root, fx, store, ["decisions"])
    producers = _producers(root, fx, store)
    producers.tick()
    GuardOutbox(root).append(_decision())
    fx.ingest_replies.append((403, {"code": "guard_org_not_on", "error": "off"}))
    assert producers.tick().results == ["paused"]
    GuardOutbox(root).append(_decision())
    producers.tick()
    assert len(fx.ingests()) == 1


# ---------------------------------------------------------------------------
# Posture: bit only, memo per (account, fact, value), survives everything
# ---------------------------------------------------------------------------


def test_memo_is_reused_on_value_return_and_drawn_only_for_a_new_value(root: Path) -> None:
    draws = _Rng([0.1, 0.9, 0.1])  # truthful, then lie, then (unused)
    memo = pst.PostureMemo(root, rng=draws)
    fact = pst.FACT_GUARDRAILS_ENABLED
    first = memo.bit(ACCOUNT, fact, pst.VALUE_RECOMMENDED)
    assert first == 1  # 0.1 < 0.75: truthful
    other = memo.bit(ACCOUNT, fact, pst.VALUE_OTHER)
    assert other == 1  # 0.9 >= 0.75: complement of 0
    # R -> O -> R -> O: the memo answers; no further draw.
    for _ in range(3):
        assert memo.bit(ACCOUNT, fact, pst.VALUE_RECOMMENDED) == first
        assert memo.bit(ACCOUNT, fact, pst.VALUE_OTHER) == other
    assert memo.memo_count(ACCOUNT) == 2
    assert draws.values == [0.1]  # exactly two draws consumed
    # Another account draws its own.
    memo.bit("acct_other", fact, pst.VALUE_RECOMMENDED)
    assert memo.memo_count("acct_other") == 1


def test_randomised_response_rate_is_p_three_quarters(root: Path) -> None:
    import random

    rng = random.Random(1234)
    memo = pst.PostureMemo(root, rng=rng.random)
    n = 4000
    truthful = sum(
        memo.bit(f"acct_{i}", pst.FACT_GUARDRAILS_ENABLED, pst.VALUE_RECOMMENDED) for i in range(n)
    )
    assert pst.P_TRUTH == pytest.approx(0.75)
    # 4 sigma of a binomial(4000, 0.75): sqrt(4000*.75*.25) = 27.4
    assert abs(truthful - 0.75 * n) < 4 * 27.4


def test_posture_batch_carries_the_bit_only_and_memo_survives_regrant_and_reset(
    root: Path, fx: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = {
        pst.FACT_GUARDRAILS_ENABLED: pst.VALUE_RECOMMENDED,
        pst.FACT_CRITICAL_OVERRIDE_OFF: pst.VALUE_OTHER,
    }
    monkeypatch.setattr(pst, "read_facts", lambda _c: dict(truth))
    store = MemoryKeyStore()
    host = _grant(root, fx, store, ["posture"])
    clock = _Clock()
    producers = _producers(root, fx, store, clock=clock, rng=_Rng([0.1, 0.1]))
    producers.tick()  # observes week 2026-W48
    assert fx.ingests() == []  # the week is still open
    clock.now += 7 * 86400  # into W49: W48 closed
    producers.reset_transport()
    producers.tick()
    (req,) = fx.ingests()
    posture = req["body"]["payload"]["posture"]
    assert {tuple(sorted(r)) for r in posture} == {("bit", "fact", "schema", "week")}
    assert {r["week"] for r in posture} == {"2026-W48"}
    bits = {r["fact"]: r["bit"] for r in posture}
    assert "recommended" not in json.dumps(req["body"]) and "other" not in json.dumps(req["body"])
    memo_file = pst.memo_path(root)
    before = memo_file.read_bytes()

    # Revoke, namespace reset, re-grant: the memo file is untouched and the
    # next report for the same true values carries the same bits.
    host.revoke(TENANT)
    producers.tick()
    producers.namespace_reset()
    assert memo_file.read_bytes() == before
    _grant(root, fx, store, ["posture"], rev=2)
    producers.tick()  # observe W49 under the new grant
    clock.now += 7 * 86400
    producers.reset_transport()
    producers._rng = _Rng([0.99])  # any fresh draw would now LIE: must not be used
    producers.tick()
    later = [r for r in fx.ingests() if "posture" in r["body"]["payload"]][-1]
    assert {r["fact"]: r["bit"] for r in later["body"]["payload"]["posture"]} == bits
    assert pst.PostureMemo(root).memo_count(ACCOUNT) == 2


# ---------------------------------------------------------------------------
# Task time: interval union
# ---------------------------------------------------------------------------


def test_interval_union_never_double_counts_nested_or_parallel_spans() -> None:
    spans = [(0, 100), (10, 20), (50, 150), (200, 210), (205, 209)]
    assert act.interval_union(spans) == [(0, 150), (200, 210)]
    assert act.active_seconds(spans) == 160
    assert act.active_seconds([]) == 0


@dataclass
class _Children:
    jobs: list[Any]

    def list(self) -> list[Any]:
        return self.jobs


@dataclass
class _Job:
    type: str = "task"
    status: str = "completed"
    agent_id: str | None = "coder"
    started_at: float | None = 1000.0
    settled_at: float | None = 1100.0
    child_jobs: Any = None


def test_task_record_uses_the_union_of_the_job_and_its_children() -> None:
    # Parent 1000-1100; two parallel children 1050-1500 and 1060-1400:
    # union 1000-1500 = 500 s (5m-15m). A sum would be 100+450+340 = 890 s.
    kids = _Children(
        [_Job(started_at=1050, settled_at=1500), _Job(started_at=1060, settled_at=1400)]
    )
    record = act.record_from_job(_Job(child_jobs=kids, settled_at=1100.0))
    assert record is not None
    assert record.active_duration_band == "5m_15m" and record.workflow_class == "coder"
    assert record.terminal_state == "done"
    assert set(record.wire()) == {
        "record_id",
        "schema",
        "day",
        "workflow_class",
        "terminal_state",
        "active_duration_band",
    }
    cancelled = act.record_from_job(_Job(status="cancelled", agent_id="my-secret-agent"))
    assert cancelled is not None and cancelled.active_duration_band is None
    assert cancelled.workflow_class == "other" and "active_duration_band" not in cancelled.wire()
    assert act.record_from_job(_Job(status="running")) is None
    assert act.record_from_job(_Job(type="bash")) is None


def test_task_sink_is_installed_only_under_a_live_task_time_consent(
    root: Path, fx: _Fixture
) -> None:
    store = MemoryKeyStore()
    act.note_job_settled(_Job())  # no sink: nothing, no file
    assert not act.activity_path(root).exists()
    _grant(root, fx, store, ["task_time"])
    producers = _producers(root, fx, store)
    producers.tick()
    assert isinstance(act.get_task_sink(), act.StoreTaskSink)
    assert gd.get_decision_sink() is gd.NOOP_SINK  # decisions not consented
    act.note_job_settled(_Job(settled_at=dt.datetime(2026, 11, 26, tzinfo=dt.UTC).timestamp()))
    producers.tick()
    (req,) = fx.ingests()
    assert list(req["body"]["payload"]["task_time"][0]) == [
        "record_id",
        "schema",
        "day",
        "workflow_class",
        "terminal_state",
        "active_duration_band",
    ]
    assert "decisions" not in req["body"]["payload"]


# ---------------------------------------------------------------------------
# Review cycles
# ---------------------------------------------------------------------------


def _entry(
    state: str, passes: list[dict[str, Any]], key: str = "github.com/o/r#7"
) -> dict[str, Any]:
    return {"key": key, "state": state, "convention": {"pieces": {"comments": passes}}}


def _write_entry(root: Path, name: str, entry: dict[str, Any]) -> None:
    folder = root / "cache" / "code_requests" / "github.com"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(entry))


def test_review_adapter_counts_distinct_agent_rounds_once_per_change(root: Path) -> None:
    passes = [
        {"lane": "agent", "kind": "review", "round": 1},
        {"lane": "agent", "kind": "review", "round": 1},  # fix-verification pass
        {"lane": "agent", "kind": "remediation", "round": 1},
        {"lane": "qa", "kind": "review", "round": 1},
        {"lane": "agent", "kind": "review", "round": 2},
    ]
    store = act.ActivityStore(root)
    _write_entry(root, "o%2Fr__7.json", _entry("open", passes))
    _write_entry(root, "o%2Fr__8.json", _entry("merged", passes, key="github.com/o/r#8"))
    assert store.scan_reviews(today=TODAY) == 0  # #7 tracked, #8 pre-enrolment: skipped
    _write_entry(root, "o%2Fr__7.json", _entry("merged", passes))
    assert store.scan_reviews(today=TODAY) == 1
    assert store.scan_reviews(today=TODAY) == 0  # dedup
    (row,) = store.pending_after("review_cycles", 0, before_day="9999-12-31")
    assert row[1]["rounds_band"] == "2" and row[1]["terminal_outcome"] == "merged"
    assert "o/r" not in json.dumps(row[1]) and "github" not in json.dumps(row[1])
    assert act.rounds_band(7) == "4_plus"


def test_review_change_with_no_parsed_round_is_incomplete_not_zero(root: Path) -> None:
    store = act.ActivityStore(root)
    _write_entry(root, "x.json", _entry("open", []))
    store.scan_reviews(today=TODAY)
    _write_entry(root, "x.json", _entry("closed", []))
    assert store.scan_reviews(today=TODAY) == 0


# ---------------------------------------------------------------------------
# Friction, outbox migration, built-in digest pin
# ---------------------------------------------------------------------------


def test_friction_counter_is_local_and_banded(root: Path, fx: _Fixture) -> None:
    store = MemoryKeyStore()
    _grant(root, fx, store, ["reports"])
    producers = _producers(root, fx, store)
    producers.tick()
    for _ in range(7):
        gd.emit_decision("block", gd.RULE_CLASS_CUSTOM, "shell")
    gd.emit_decision("allow", gd.RULE_CLASS_CUSTOM, "shell")
    assert act.ActivityStore(root).friction(gd.RULE_CLASS_CUSTOM) == "6_20"
    assert not GuardOutbox(root).path.exists()  # decisions not consented
    producers.tick()
    assert fx.ingests() == []


def test_outbox_v1_migrates_and_seq_never_rewinds_after_prune(root: Path) -> None:
    outbox = GuardOutbox(root)
    outbox.path.parent.mkdir(parents=True)
    raw = sqlite3.connect(outbox.path)
    raw.execute(
        "CREATE TABLE decision_records (record_id TEXT PRIMARY KEY, schema TEXT NOT NULL,"
        " day TEXT NOT NULL, outcome TEXT NOT NULL, rule_class TEXT NOT NULL,"
        " tool_class TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', seq INTEGER NOT NULL)"
    )
    rec = _decision()
    raw.execute(
        "INSERT INTO decision_records VALUES (?, ?, ?, ?, ?, ?, 'pending', 5)",
        (rec.record_id, rec.schema, rec.day, rec.outcome, rec.rule_class, rec.tool_class),
    )
    raw.execute("PRAGMA user_version = 1")
    raw.commit()
    raw.close()
    assert outbox.max_seq() == 5
    outbox.mark_sent([rec.record_id])
    outbox.prune()
    assert outbox.count() == 0
    outbox.append(_decision())
    assert outbox.max_seq() == 6  # MAX(seq)+1 would have restarted at 1
    assert outbox.pending_after(5, before_day="9999-12-31")[0][0] == 6


def test_builtin_policy_digest_pin_matches_the_shipped_default() -> None:
    assert pst._builtin_digest() == pst.BUILTIN_POLICY_SHA256


def test_ingest_payload_refuses_unknown_families_and_fields() -> None:
    def build(families: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return ingest.build_payload(
            tenant=TENANT,
            install_id="a" * 32,
            collection_rev=1,
            revocation_gen=0,
            ts=1,
            families=families,
        )

    with pytest.raises(ValueError):
        build({"paths": [{"x": 1}]})
    with pytest.raises(ValueError):
        build({"decisions": [{"record_id": "r", "command": "rm -rf /"}]})
