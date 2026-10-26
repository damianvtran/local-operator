"""Creation retries exercise the real authenticated HTTP route and disk receipts.

Only child construction and owner discovery are controlled: no live runtime,
credential, transcript or operator session store participates. The rules pinned
here are the ones that fail silently and cost a second conversation:

* one key mints one durable identity, including across a lost response, a
  daemon restart and a cancelled request;
* a replay reconciles the CURRENT owner of that identity — a recorded pid is
  history the OS may have recycled, so it is never answered back;
* uncertainty holds the exact SID (typed refusal) instead of minting a new one;
* refusals (auth, origin, mismatch, capacity, corrupt store) happen before any
  side effect;
* an unkeyed caller keeps the shipped behaviour and never touches the store.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path

# ``httpx2`` under the familiar name, as tests/unit/mcp/test_grant_diagnostics.py does:
# Starlette's ``TestClient`` is built on ``httpx2``, so one ``Response`` type covers
# both the TestClient cells and the streamed ``ASGITransport`` cells.
import httpx2 as httpx
import pytest
from starlette.testclient import TestClient

from local_operator.mobile import attach_client
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.mobile.start_receipts import (
    MAX_FIELD_CHARS,
    RECEIPTS_NAME,
    START_BODY_MAX_BYTES,
    StartReceipts,
    canonical_request,
)

KEY = "1a2b3c4d-1111-4222-8333-444455556666"
SID = "abcdef123456"


# --------------------------------------------------------------------------
# Rig: real app, isolated roots, recorded spawn seam
# --------------------------------------------------------------------------


class Rig:
    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, capacity: int | None = None
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
        self.root = tmp_path / "config"
        self.daemon = MobileDaemon(port=0, password="fixture-password")
        if capacity is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.daemon._start_receipts = StartReceipts(self.root, capacity=capacity)
        self.spawns: list[tuple[str, str | None, str | None, str | None]] = []
        self.calls = 0

        async def spawn(cwd, provider=None, model_id=None, resume=None):
            self.calls += 1
            self.spawns.append((cwd, provider, model_id, resume))
            return 970123

        monkeypatch.setattr(self.daemon, "spawn_session", spawn)
        self.client = TestClient(build_app(self.daemon), follow_redirects=False)
        assert self.client.post("/login", data={"password": "fixture-password"}).status_code == 303

    @property
    def receipts_path(self) -> Path:
        return self.root / RECEIPTS_NAME

    def start(self, body: dict[str, object], **kwargs):
        return self.client.post("/api/sessions/start", json=body, **kwargs)

    def rows(self) -> list[tuple[str, str, str, str | None]]:
        """Read the journal WITHOUT creating it: a stray 0-byte file would make
        the daemon's own O_EXCL creation path see an uninitialised store."""
        if not self.receipts_path.exists():
            return []
        connection = sqlite3.connect(f"file:{self.receipts_path}?mode=ro", uri=True)
        try:
            return connection.execute(
                "SELECT request_id, resolved, session_id, result FROM receipts"
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            connection.close()


def completed(row: tuple[str, str, str, str | None]) -> dict[str, object]:
    """The recorded completion of one receipt row; ``result`` is NULL until the
    start completes, so a cell reading it asserts completion first."""
    result = row[3]
    assert result is not None, f"receipt {row[0]} has no recorded completion"
    return json.loads(result)


def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, capacity: int | None = None) -> Rig:
    return Rig(tmp_path, monkeypatch, capacity=capacity)


def spawn_adapter(daemon: MobileDaemon):
    """The route's own two-argument seam, spelled once for direct-start cells.

    ``StartReceipts.start`` hands an adapter the caller's resolved pair and the
    freshly minted SID, exactly as the route does; the daemon method itself takes
    positional provider/model/resume. Keeping one adapter stops a cell from
    silently calling ``spawn(dict, sid)`` as ``cwd=dict, provider=sid``, which
    would exercise nothing the route does.
    """

    async def spawn(chosen, sid):
        return await daemon.spawn_session(
            chosen["cwd"], provider=chosen["provider"], model_id=chosen["model_id"], resume=sid
        )

    return spawn


def body(**over: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "create_request_id": KEY,
        "cwd": "~",
        "provider": "openai",
        "model_id": "fixture-model",
    }
    payload.update(over)
    return payload


# --------------------------------------------------------------------------
# Idempotent creation and replay
# --------------------------------------------------------------------------


def test_same_creation_key_never_mints_a_second_session(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    first = r.start(body())
    sid = first.json()["session_id"]

    async def owner(session_id):
        assert session_id == sid
        return 970123

    monkeypatch.setattr(r.daemon, "recover_started_session", owner)
    second = r.start(body())
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert r.calls == 1
    assert r.spawns[0][3] == sid
    assert len(r.rows()) == 1


def test_concurrent_requests_for_one_key_coalesce(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    gate = asyncio.Event()
    original = r.daemon.spawn_session

    async def blocking(*args, **kwargs):
        await gate.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(r.daemon, "spawn_session", blocking)
    receipts = StartReceipts(r.root)
    r.daemon._start_receipts = receipts

    def resolve() -> dict[str, str]:
        return {"cwd": "/synthetic", "provider": "openai", "model_id": "fixture-model"}

    async def settle(predicate):
        """Deadline-bounded wait: an indefinite loop would hang the suite the
        moment a claim silently failed to commit instead of reporting it."""
        for _ in range(400):
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("condition never became true")

    async def scenario():
        first = asyncio.create_task(
            receipts.start(
                body(), resolve, spawn_adapter(r.daemon), r.daemon.recover_started_session
            )
        )
        await settle(lambda: bool(r.rows()))
        second = asyncio.create_task(
            receipts.start(
                body(), resolve, spawn_adapter(r.daemon), r.daemon.recover_started_session
            )
        )
        await asyncio.sleep(0.05)
        gate.set()
        return await asyncio.gather(first, second)

    answers = asyncio.run(scenario())
    assert answers[0] == answers[1]
    assert r.calls == 1


def test_lost_response_retry_replays_the_exact_session(tmp_path, monkeypatch):
    """The proxy drops the ack: the claim is on disk and the retry must reuse it."""
    r = rig(tmp_path, monkeypatch)
    original = r.daemon.spawn_session

    async def dies_after_claim(cwd, provider=None, model_id=None, resume=None):
        raise ConnectionResetError("client went away mid-spawn")

    monkeypatch.setattr(r.daemon, "spawn_session", dies_after_claim)
    lost = r.start(body())
    assert lost.status_code == 409
    assert lost.json()["code"] == "create_indeterminate"
    held = lost.json()["session_id"]
    assert re.fullmatch(r"[0-9a-f]{12}", held)
    pending = r.rows()
    assert len(pending) == 1 and pending[0][3] is None, "the claim survives, unfinished"

    # Same key, now with the owner verifiable: the SID is recovered, not reminted.
    monkeypatch.setattr(r.daemon, "spawn_session", original)

    async def finds_owner(session_id):
        assert session_id == held
        return 970123

    monkeypatch.setattr(r.daemon, "recover_started_session", finds_owner)
    retried = r.start(body())
    assert retried.status_code == 200
    assert retried.json() == {"ok": True, "pid": 970123, "session_id": held}
    assert len(r.rows()) == 1


def test_completed_replay_reconciles_current_owner_not_the_recorded_pid(tmp_path, monkeypatch):
    """A pid in a finished receipt is history; PID reuse would address another session."""
    r = rig(tmp_path, monkeypatch)
    assert r.start(body()).status_code == 200
    (row,) = r.rows()
    recorded = completed(row)
    assert recorded["pid"] == 970123

    async def current_owner(session_id):
        assert session_id == recorded["session_id"]
        return 975555  # restarted runtime, same durable identity

    monkeypatch.setattr(r.daemon, "recover_started_session", current_owner)
    replay = r.start(body())
    assert replay.status_code == 200
    assert replay.json() == {"ok": True, "pid": 975555, "session_id": recorded["session_id"]}
    assert r.calls == 1  # no fresh spawn


def test_completed_replay_without_a_live_owner_holds_the_sid(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    assert r.start(body()).status_code == 200
    (row,) = r.rows()
    recorded = completed(row)

    async def no_owner(_session_id):
        return None

    monkeypatch.setattr(r.daemon, "recover_started_session", no_owner)
    replay = r.start(body())
    assert replay.status_code == 409
    assert replay.json() == {
        "error": "create_indeterminate",
        "code": "create_indeterminate",
        "session_id": recorded["session_id"],
    }
    assert "pid" not in replay.json()
    assert r.calls == 1


def test_pending_claim_adopts_only_a_verified_exact_owner(tmp_path, monkeypatch):
    """Restart with an unfinished claim: adopt, complete, and never re-spawn."""
    r = rig(tmp_path, monkeypatch)
    (r.root).mkdir(parents=True, exist_ok=True)
    receipts = StartReceipts(r.root)
    _, encoded = canonical_request(body())
    fresh, chosen, sid, answer = receipts._claim(
        KEY,
        encoded,
        lambda: {"cwd": "/synthetic", "provider": "openai", "model_id": "fixture-model"},
    )
    assert fresh and answer is None
    r.daemon._start_receipts = receipts

    record = type("Record", (), {"pid": 972222})()
    seen: dict[str, str] = {}

    def find(_root, session_id):
        seen["sid"] = session_id
        return record, None

    async def prepared(_record):
        return None

    monkeypatch.setattr(attach_client, "find_runtime_record", find)
    monkeypatch.setattr(r.daemon, "_prepare_phone_record", prepared)
    adopted = r.start(body())
    assert adopted.status_code == 200
    assert adopted.json() == {"ok": True, "pid": 972222, "session_id": sid}
    assert seen["sid"] == sid
    assert r.calls == 0
    assert completed(r.rows()[0])["session_id"] == sid


def test_pending_claim_with_no_owner_stays_unresolved_without_spawning(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    receipts = StartReceipts(r.root)
    _, encoded = canonical_request(body())
    _, _, sid, _ = receipts._claim(
        KEY,
        encoded,
        lambda: {"cwd": "/synthetic", "provider": "openai", "model_id": "fixture-model"},
    )
    r.daemon._start_receipts = receipts

    held = r.start(body())
    assert held.status_code == 409
    assert held.json()["session_id"] == sid
    assert r.calls == 0
    # Still queryable afterwards: the same key answers the same held identity.
    assert r.start(body()).json()["session_id"] == sid


def test_cancelled_request_leaves_a_queryable_receipt(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    gate = asyncio.Event()
    released: list[str] = []

    async def slow(cwd, provider=None, model_id=None, resume=None):
        # The receipt path always resumes under the SID it claimed; a missing
        # one would be the defect this cell exists to catch.
        assert resume is not None
        released.append(resume)
        await gate.wait()
        return 970123

    monkeypatch.setattr(r.daemon, "spawn_session", slow)
    receipts = StartReceipts(r.root)
    r.daemon._start_receipts = receipts

    async def scenario():
        pending = asyncio.create_task(
            receipts.start(
                body(),
                lambda: {"cwd": "/synthetic", "provider": "openai", "model_id": "fixture-model"},
                spawn_adapter(r.daemon),
                r.daemon.recover_started_session,
            )
        )
        while not r.rows():
            await asyncio.sleep(0.01)
        pending.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        # The shielded operation finishes and records its identity.
        for _ in range(200):
            if r.rows() and r.rows()[0][3] is not None:
                break
            await asyncio.sleep(0.01)
        return r.rows()[0][2]

    sid = asyncio.run(scenario())
    assert released == [sid]
    assert completed(r.rows()[0])["session_id"] == sid

    async def owner(_session_id):
        return 970123

    monkeypatch.setattr(r.daemon, "recover_started_session", owner)
    retried = r.start(body())
    assert retried.json()["session_id"] == sid


# --------------------------------------------------------------------------
# Refusals happen before side effects
# --------------------------------------------------------------------------


def test_mismatched_body_under_the_same_key_is_refused(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    assert r.start(body()).status_code == 200
    conflict = r.start(body(model_id="another-model"))
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "create_conflict"
    assert r.calls == 1


def test_invalid_key_and_unknown_fields_are_refused(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    invalid = r.start(body(create_request_id="not-a-uuid"))
    assert invalid.json()["code"] == "invalid_create_request_id"
    assert r.start(body(account="someone-else")).json()["code"] == "invalid_start_request"
    assert r.calls == 0
    assert not r.receipts_path.exists()


def test_authentication_and_origin_precede_claim_and_replay(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    anonymous = TestClient(build_app(r.daemon), follow_redirects=False)
    denied = anonymous.post("/api/sessions/start", json=body())
    assert denied.status_code in (401, 403)
    cross = r.client.post(
        "/api/sessions/start", json=body(), headers={"Origin": "http://evil.example"}
    )
    assert cross.status_code in (401, 403)
    assert r.calls == 0
    assert not r.receipts_path.exists()


# --------------------------------------------------------------------------
# Body byte cap: refused before buffering or parsing, with no side effect
# --------------------------------------------------------------------------

TOO_LARGE = {"error": "start_request_too_large", "code": "start_request_too_large"}


def at_size(payload: dict[str, object], size: int) -> bytes:
    """``payload`` as JSON padded to exactly ``size`` bytes.

    The padding is insignificant whitespace, so the parsed body is unchanged and
    a body AT the cap can be a genuinely admissible start, not merely "not 413".
    """
    raw = json.dumps(payload, separators=(",", ":")).encode()
    assert len(raw) < size
    return raw[:-1] + b" " * (size - len(raw)) + b"}"


def stream_post(r: Rig, raw: bytes, chunk: int = 64 * 1024) -> tuple[httpx.Response, int]:
    """POST ``raw`` chunked with NO Content-Length, through a real ASGI stream.

    ``TestClient`` hands the app the whole body in one message, which cannot show
    whether the route stops reading early. ``httpx.ASGITransport`` delivers each
    chunk as its own ``http.request`` message, and the generator counts what the
    route actually pulled before answering.
    """
    pulled = 0

    async def chunks():
        nonlocal pulled
        for start in range(0, len(raw), chunk):
            piece = raw[start : start + chunk]
            pulled += len(piece)
            yield piece

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=build_app(r.daemon))
        # The authenticated session cookie, copied by value: the sync client's
        # jar is not an async client's jar.
        cookies = {name: value for name, value in r.client.cookies.items()}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", cookies=cookies
        ) as client:
            return await client.post(
                "/api/sessions/start",
                content=chunks(),
                headers={"content-type": "application/json"},
            )

    response = asyncio.run(scenario())
    return response, pulled


def assert_no_side_effect(r: Rig) -> None:
    assert r.calls == 0
    assert not r.receipts_path.exists()


def test_oversized_declared_body_is_refused_before_parsing(tmp_path, monkeypatch):
    """The reviewer's reproduction: an ~8 MiB keyed body with an honest length."""
    r = rig(tmp_path, monkeypatch)
    raw = json.dumps(body(cwd="x" * (8 * 1024 * 1024))).encode()
    refused = r.client.post(
        "/api/sessions/start", content=raw, headers={"content-type": "application/json"}
    )
    assert refused.status_code == 413
    assert refused.json() == TOO_LARGE
    assert_no_side_effect(r)


def test_lying_content_length_is_bounded_by_the_bytes_read(tmp_path, monkeypatch):
    """The header is the client's claim, never the bound."""
    r = rig(tmp_path, monkeypatch)
    raw = at_size(body(), START_BODY_MAX_BYTES + 1)
    refused = r.client.post(
        "/api/sessions/start",
        content=raw,
        headers={"content-type": "application/json", "content-length": "64"},
    )
    assert refused.status_code == 413
    assert refused.json() == TOO_LARGE
    assert_no_side_effect(r)


def test_chunked_body_without_a_length_stops_reading_at_the_cap(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    raw = at_size(body(), 8 * 1024 * 1024)
    refused, pulled = stream_post(r, raw)
    assert "content-length" not in refused.request.headers
    assert refused.status_code == 413
    assert refused.json() == TOO_LARGE
    # Stopped within one chunk of the cap rather than buffering all 8 MiB.
    assert pulled <= START_BODY_MAX_BYTES + 64 * 1024 < len(raw)
    assert_no_side_effect(r)


def test_unkeyed_oversized_body_is_refused_too(tmp_path, monkeypatch):
    """The route cannot know a body is keyed until it parses it."""
    r = rig(tmp_path, monkeypatch)
    raw = at_size({"cwd": "~"}, START_BODY_MAX_BYTES + 1)
    refused, _ = stream_post(r, raw)
    assert refused.status_code == 413
    assert refused.json() == TOO_LARGE
    assert_no_side_effect(r)


@pytest.mark.parametrize("framing", ["content-length", "chunked"])
def test_byte_cap_boundary_admits_at_and_refuses_one_over(tmp_path, monkeypatch, framing):
    r = rig(tmp_path, monkeypatch)

    def send(raw: bytes) -> httpx.Response:
        if framing == "chunked":
            return stream_post(r, raw, chunk=4096)[0]
        return r.client.post(
            "/api/sessions/start", content=raw, headers={"content-type": "application/json"}
        )

    over = send(at_size(body(), START_BODY_MAX_BYTES + 1))
    assert over.status_code == 413
    assert over.json() == TOO_LARGE
    assert_no_side_effect(r)

    at_cap = send(at_size(body(), START_BODY_MAX_BYTES))
    assert at_cap.status_code == 200
    assert r.calls == 1
    assert [row[2] for row in r.rows()] == [at_cap.json()["session_id"]]


def test_cap_never_refuses_the_largest_schema_valid_body(tmp_path, monkeypatch):
    """Worst-case encoding of three maximal fields still reaches field validation.

    An astral character is one code point but 12 JSON bytes when escaped, so
    this is the most bytes a body inside the per-field bounds can take.
    """
    r = rig(tmp_path, monkeypatch)
    widest = "\U0001f600" * MAX_FIELD_CHARS
    payload = body(cwd=widest, provider=widest, model_id=widest)
    raw = json.dumps(payload, ensure_ascii=True, indent=1).encode()
    assert len(raw) <= START_BODY_MAX_BYTES
    reply = r.client.post(
        "/api/sessions/start", content=raw, headers={"content-type": "application/json"}
    )
    assert reply.status_code != 413
    assert reply.json()["code"] != "start_request_too_large"


def test_capacity_refuses_new_keys_before_side_effects_and_still_replays_known(
    tmp_path, monkeypatch
):
    r = rig(tmp_path, monkeypatch, capacity=1)
    assert r.start(body()).status_code == 200
    other = body(create_request_id="9a8b7c6d-1111-4222-8333-444455556666")
    refused = r.start(other)
    assert refused.status_code == 503
    assert refused.json()["code"] == "create_capacity"
    assert r.calls == 1
    assert len(r.rows()) == 1
    assert r.start(body()).json()["session_id"] == r.rows()[0][2]


def test_corrupt_store_fails_closed(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    r.root.mkdir(parents=True, exist_ok=True)
    r.receipts_path.write_bytes(b"this is not a sqlite database")
    broken = r.start(body())
    assert broken.status_code == 503
    assert broken.json()["code"] == "create_receipts_unavailable"
    assert r.calls == 0


def test_unwritable_store_path_fails_closed(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    r.receipts_path.mkdir(parents=True)  # a directory where the store file belongs
    broken = r.start(body())
    assert broken.status_code == 503
    assert broken.json()["code"] == "create_receipts_unavailable"
    assert r.calls == 0


# --------------------------------------------------------------------------
# Frozen intent and legacy compatibility
# --------------------------------------------------------------------------


def test_omitted_defaults_are_resolved_once_and_frozen(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    omitted = body(model_id=None)
    first = r.start(omitted)
    assert first.status_code == 200
    resolved = json.loads(r.rows()[0][1])
    assert resolved["model_id"] and resolved["provider"] == "openai"

    # A later default change cannot rewrite what the first admission chose.
    import local_operator.model.defaults as defaults

    monkeypatch.setattr(defaults, "default_model_for", lambda _provider: "changed-default")
    replay = r.start(omitted)
    assert replay.json()["session_id"] == first.json()["session_id"]
    assert json.loads(r.rows()[0][1])["model_id"] == resolved["model_id"]
    assert r.calls == 1


def test_legacy_callers_keep_the_shipped_response_and_never_touch_the_store(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    legacy = r.start({"cwd": "~", "provider": "openai", "model_id": "fixture-model"})
    assert legacy.status_code == 200
    payload = legacy.json()
    assert set(payload) == {"ok", "pid", "session_id"}
    assert payload["ok"] is True and payload["pid"] == 970123
    assert re.fullmatch(r"[0-9a-f]{12}", payload["session_id"])
    assert r.calls == 1
    assert not r.receipts_path.exists()


def test_capability_is_advertised_only_on_an_authenticated_listing(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch)
    payload = r.client.get("/api/sessions").json()
    assert payload["capabilities"]["session_start"] == {
        "version": 1,
        "durable_idempotency": True,
        "receipt_capacity": 4096,
    }
    anonymous = TestClient(build_app(r.daemon), follow_redirects=False)
    assert anonymous.get("/api/sessions").status_code in (401, 403)


def test_protocol_doc_matches_the_advertised_contract():
    """The doc is the contract native codes against; keep the two in step."""
    doc_path = Path(__file__).resolve().parents[3] / "docs" / "design" / "mobile-start-receipts.md"
    doc = doc_path.read_text()
    for token in (
        "create_request_id",
        "session_start",
        "create_conflict",
        "create_indeterminate",
        "create_capacity",
        "create_receipts_unavailable",
        "start_request_too_large",
    ):
        assert token in doc
    # The documented byte cap is the enforced one.
    assert f"{START_BODY_MAX_BYTES:,} bytes" in doc
