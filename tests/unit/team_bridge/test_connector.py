"""H4 connector tests: lifecycle, loop, spine — targeted and controlled.

No network, no services, no runtime spawns: the transport is faked at the
``pull``/``client`` seam, the screening/delivery primitives are replaced at
the ``connector.wake`` seam, and the store is REAL native SQLite under the
test's own tmp_path. The connector's privacy shape is asserted directly
(nothing but the bounded payload leaves the spine).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import plistlib
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge import pull as pull_mod
from local_operator.team_bridge import store as store_mod
from local_operator.team_bridge.connector import install as install_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.connector import wake as wake_mod
from local_operator.team_bridge.connector.lock import ConnectorLock, ConnectorLockBusy
from local_operator.team_bridge.screening import ScreenResult
from local_operator.team_bridge.store import OwnerReceipt, Store, StoreRefused
from local_operator.team_bridge.types import (
    InboxCursor,
    OwnerSnapshot,
    ScreeningReceipt,
    feed_seq,
)


def _access(owner: Any) -> SimpleNamespace:
    """The authenticated access double the producer fence PINS (TB-14).

    A context that cannot name its credential row is refused as
    ``credential_changed`` — fail-closed, deliberately — so a case that means to
    exercise a boundary must carry the owner's own credential id, exactly as
    the production connector always does.
    """

    return SimpleNamespace(credential_id=int(owner.credential_id))


OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://api.radienthq.com/v1",
    enrollment_generation=1,
)


def _enroll(tmp_path: Path, *, session: str | None = "sess_cos") -> tuple[Store, OwnerSnapshot]:
    store = Store(tmp_path)
    store.commit_enrollment(
        OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=session
    )
    return store, OWNER


def _item(body: str = "hello from the org") -> dict[str, Any]:
    return {
        "message": {
            "id": "bmsg_1",
            "thread_id": "bth_1",
            "type": "note",
            "body": {"text": body},
            "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        },
        "sender": {"account_id": "acc_sender", "name": "Org Member", "organization": "Acme"},
        "thread": {"id": "bth_1", "tenant_id": "tnt_1"},
        "delivery": {"state": "unread"},
    }


def _receipt(body: str, *, verdict: str = "allow") -> ScreeningReceipt:
    return ScreeningReceipt(
        verdict=verdict,
        model="qa-default",
        prompt_version="bridge-screen-v1",
        template_sha256="ab" * 32,
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        categories=(),
    ).validated()


# -- H4a: lock ---------------------------------------------------------------


def test_machine_lock_refuses_second_holder_and_releases(tmp_path) -> None:
    first = ConnectorLock(tmp_path, deadline_s=0.3)
    first.acquire()
    second = ConnectorLock(tmp_path, deadline_s=0.3)
    with pytest.raises(ConnectorLockBusy):
        second.acquire()
    first.release()
    with second:
        pass
    assert (tmp_path / "team_bridge" / "connector.lock").exists()


# -- H4a: install rendering (never installs) ---------------------------------


def test_install_renders_own_label_to_temp_only(tmp_path) -> None:
    config_dir = tmp_path / "cfg"
    target = tmp_path / "rendered"
    path = install_mod.write_plist(config_dir, target)
    assert path.parent == target
    plan = plistlib.loads(path.read_bytes())
    assert plan["Label"] == "com.local-operator.bridge-connector"
    assert plan["Label"] != "com.local-operator.wakes"
    assert plan["KeepAlive"] is True
    assert plan["ProgramArguments"][1:3] == ["-m", "local_operator.team_bridge.connector"]
    unit = install_mod.render_systemd(config_dir)
    assert "Restart=always" in unit and unit.startswith("[Unit]")
    assert install_mod.SYSTEMD_UNIT.endswith(".service")
    task = install_mod.render_task_xml(config_dir)
    assert install_mod.TASK_NAME in task and "<LogonTrigger>" in task
    # Rendering is not installation: the real target MAY exist from an
    # install, but a render to a temp dir must not change it. The lifecycle is
    # deliberately the separate, guarded ``sync_connector_service`` API.
    assert install_mod.plist_path().name == "com.local-operator.bridge-connector.plist"
    existed_before = install_mod.plist_path().exists()
    install_mod.write_plist(config_dir, tmp_path / "rendered-again")
    assert install_mod.plist_path().exists() == existed_before
    assert not hasattr(install_mod, "install") and not hasattr(install_mod, "bootstrap")


def test_missing_or_mismatched_credential_is_credential_changed(tmp_path) -> None:
    """TB-14's `credential_changed` arm (m2): the sweep must not delete it.

    The fence refuses a context that cannot name its credential row, and one
    that names a DIFFERENT row than the owner's committed one — both are the
    same fail-closed verdict, and this case keeps the second arm exercised now
    that every other call site carries the owner's own id.
    """

    from local_operator.team_bridge.connector import wake as wake_mod

    store, owner = _enroll(tmp_path)
    _enable_bridge(tmp_path)  # the fence reads the DISK config (S-8a) first
    missing = SimpleNamespace()  # no credential_id at all
    wrong = SimpleNamespace(credential_id=int(owner.credential_id) + 1)
    assert wake_mod._producer_fence(store, owner, missing, None) == "credential_changed"
    assert wake_mod._producer_fence(store, owner, wrong, None) == "credential_changed"
    assert wake_mod._producer_fence(store, owner, _access(owner), None) == ""


def test_connector_import_stays_bounded() -> None:
    """The import pin: module scope must not pull the session/model graphs."""

    heavy = [
        "local_operator.session.session",
        "local_operator.model.configure",
        "local_operator.mobile.peer_send",
        "local_operator.session.runtime.launch",
        "local_operator.team_bridge.scheduler",
        "local_operator.team_bridge.screening",
    ]
    code = (
        "import local_operator.team_bridge.connector as c,"
        " local_operator.team_bridge.connector.loop,"
        " local_operator.team_bridge.connector.wake, sys;"
        f"heavy = {heavy!r};"
        "loaded = [m for m in heavy if m in sys.modules];"
        "assert not loaded, loaded;"
        # The ONE light exception at module scope is pinned in the positive:
        "assert 'local_operator.harness.message_types' in sys.modules;"
        "print('BOUND_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[3]), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BOUND_OK" in proc.stdout


# -- H4b: loop + cursor + resync ---------------------------------------------


def test_loop_reissue_and_cursor_persistence(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    calls: list[dict[str, Any]] = []

    async def fake_events(owner_, *, config_dir, access, after=None, wait=None, client=None):
        calls.append({"after": after})
        if len(calls) == 1:
            return None  # 204 → reissue
        return {"cursor": 7, "refs": [], "resync": False}

    monkeypatch.setattr(pull_mod, "events", fake_events)
    first = asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    # A fresh enrolment's cursor is 0 and is SENT as 0 (live E2E F2): an
    # omitted ``after`` is the hub's no-cursor arm and skips the first message.
    assert first["reissued"] is True and calls[0]["after"] == 0
    # Simulate the durable cursor having advanced (as record_feed_batch would):
    with store_mod.Store(tmp_path)._connect() as conn:  # noqa: SLF001 — test introspection
        store_mod.Store(tmp_path)._init(conn)  # noqa: SLF001
        conn.execute(
            "UPDATE sync SET feed_seq = 7 WHERE owner_key = ?",
            (store_mod._owner_key(owner),),  # noqa: SLF001
        )
    second = asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert second["cursor"] == 7 and second["reissued"] is False
    assert calls[1]["after"] == 7


def test_pending_feed_refs_retry_on_204_without_poison_starvation(tmp_path, monkeypatch, caplog):
    """A committed feed cursor is not proof that its refs reached screening.

    Synthetic upgrade/receive shape: seven durable refs, one materialized
    unread message, no hold/attempt. A pre-screen policy refusal must retain
    only that failed ref, not strand it until somebody sends another message
    or starve the six other refs. No live account or message ids are used.
    """
    from local_operator.team_bridge.types import project_message_ref

    store, owner = _enroll(tmp_path)
    ids = [f"bmsg_synthetic_{index}" for index in range(7)]
    refs = [f"message_available:message_id={mid}" for mid in ids]
    item = _item()
    item["message"]["id"] = ids[0]
    item["message"]["recipient"] = {
        "account_id": owner.account_id,
        "endpoint_id": owner.endpoint_id,
    }
    item["message"]["seq"] = 5  # message sequence is independent of feed cursor 7
    store.merge_refs(owner, [project_message_ref(item)])
    calls = []
    polls = []
    refusal = StoreRefused("policy_stale", "PRIVATE_EXCEPTION_CANARY")

    async def events(owner_, **kwargs):
        polls.append(int(kwargs["after"]))
        if len(polls) == 1:
            store.record_feed_batch(owner, feed_seq(7), refs, resync_required=False)
            return {"cursor": 7, "refs": refs, "resync": False}
        return None  # no new traffic, only the already committed ref remains

    async def handle(owner_, mid, **kwargs):
        calls.append(mid)
        if mid == ids[0] and calls.count(mid) == 1:
            raise refusal
        return {"action": "delivered", "message_id": mid}

    monkeypatch.setattr(pull_mod, "events", events)
    monkeypatch.setattr(loop_mod, "handle_message_available", handle)
    with caplog.at_level("WARNING"), pytest.raises(StoreRefused) as caught:
        asyncio.run(
            loop_mod.run_iteration(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )
    assert caught.value is refusal  # steady-state failure backoff still owns retry pacing
    assert calls == ids  # a poison first ref does not starve later messages
    assert store.drain_feed_refs(owner, 7) == refs[:1]
    assert store.screening_hold(owner, ids[0]) is None
    assert "stage=pair_handling code=policy_stale" in caplog.text
    assert "PRIVATE_EXCEPTION_CANARY" not in caplog.text
    with store._connect() as conn:  # noqa: SLF001 — metadata-only test assertions
        assert conn.execute("SELECT last_observed_at FROM sync").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM screening_attempts").fetchone()[0] == 0

    result = asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert polls == [0, 7]
    assert result["reissued"] is True
    assert result["handled"] == [{"action": "delivered", "message_id": ids[0]}]
    assert store.drain_feed_refs(owner, 7) == []
    asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert calls == ids + ids[:1]  # a following 204 does not replay terminal work


def test_mixed_feed_failures_retain_exact_refs_and_retry_on_204(tmp_path, monkeypatch, caplog):
    """Q1: both delivery lanes retire neighbours without losing failed refs.

    Two independent failures prove that the first original exception still
    controls the iteration's backoff. Conversation labels exercise the encoded
    ref identity; they must never leak into the closed failure diagnostic.
    """
    from local_operator.team_bridge.types import (
        ConversationFeedRef,
        parse_conversation_ref_value,
    )

    store, owner = _enroll(tmp_path)
    pair_failed = "bmsg_pair_failed"
    pair_ok = "bmsg_pair_ok"
    conv_ok = ConversationFeedRef(
        conversation_id="bcv_abcdefgh",
        message_id="bcmsg_abcdef02",
        conversation_seq=2,
        mention=True,
    ).validated()
    pair_ref = f"message_available:message_id={pair_failed}"
    legacy_failed = (
        "message_available:conversation_id=bcv_abcdefgh/bcmsg_abcdef01/1/0/"
        "PRIVATE_CONVERSATION_CANARY%20%2F%20name/PRIVATE_SENDER_CANARY/human"
    )
    conv_failed = parse_conversation_ref_value(legacy_failed.split("=", 1)[1])
    assert conv_failed is not None and conv_failed.legacy_alias is not None
    refs = [
        pair_ref,
        f"message_available:message_id={pair_ok}",
        legacy_failed,
        conv_ok.encode(),
    ]
    store.record_feed_batch(owner, feed_seq(9), refs, resync_required=False)
    # The writer canonicalizes even an older in-memory key. Restore a v6
    # fixture on disk to exercise migration plus #2835's idle-poll retry.
    with sqlite3.connect(store._path) as conn:
        conn.execute("UPDATE schema_meta SET value='6' WHERE key='schema_version'")
        conn.execute(
            "UPDATE feed_refs SET ref=? WHERE ref=?", (legacy_failed, conv_failed.encode())
        )
    first = StoreRefused("policy_stale", "PRIVATE_PAIR_CANARY")
    second = RuntimeError("PRIVATE_CONVERSATION_EXCEPTION_CANARY")
    pair_calls = []
    conv_calls = []

    async def timeout(*args, **kwargs):
        return None

    async def pair_handler(owner_, mid, **kwargs):
        pair_calls.append(mid)
        if mid == pair_failed and pair_calls.count(mid) == 1:
            raise first
        return {"action": "delivered", "message_id": mid}

    async def conv_handler(owner_, ref, **kwargs):
        conv_calls.append(ref.encode())
        if ref == conv_failed and conv_calls.count(ref.encode()) == 1:
            raise second
        return {"action": "delivered", "message_id": ref.message_id}

    monkeypatch.setattr(pull_mod, "events", timeout)
    monkeypatch.setattr(loop_mod, "handle_message_available", pair_handler)
    monkeypatch.setattr(loop_mod, "handle_conversation_available", conv_handler)
    with caplog.at_level("WARNING"), pytest.raises(StoreRefused) as caught:
        asyncio.run(
            loop_mod.run_iteration(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )
    assert caught.value is first
    assert set(pair_calls) == {pair_failed, pair_ok}
    assert set(conv_calls) == {conv_failed.encode(), conv_ok.encode()}
    assert set(store.drain_feed_refs(owner, 9)) == {pair_ref, conv_failed.encode()}
    assert "stage=pair_handling code=policy_stale" in caplog.text
    assert "stage=conversation_handling code=unexpected_failure" in caplog.text
    assert "PRIVATE_" not in caplog.text
    for _ in range(2):  # retry the exact failures, then prove terminal consumption
        asyncio.run(
            loop_mod.run_iteration(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )
    assert pair_calls.count(pair_failed) == 2 and pair_calls.count(pair_ok) == 1
    assert conv_calls.count(conv_failed.encode()) == 2 and conv_calls.count(conv_ok.encode()) == 1
    assert store.drain_feed_refs(owner, 9) == []
    assert store.sync_state(owner)["feed_seq"] == 9


def test_204_empty_queue_has_no_delivery_or_resync_effects(tmp_path, monkeypatch):
    store, owner = _enroll(tmp_path)
    before = store.sync_state(owner)

    async def timeout(*args, **kwargs):
        return None

    async def forbidden(*args, **kwargs):
        pytest.fail("empty queue must not fetch, screen, wake or resync")

    monkeypatch.setattr(pull_mod, "events", timeout)
    monkeypatch.setattr(loop_mod, "handle_message_available", forbidden)
    monkeypatch.setattr(loop_mod, "handle_conversation_available", forbidden)
    monkeypatch.setattr(loop_mod, "resync_walk", forbidden)
    result = asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert result["handled"] == [] and result["events"] == 0
    assert store.sync_state(owner) == before


@pytest.mark.parametrize("action", ["claimed_elsewhere", "delivered", "bounce", "skipped"])
def test_204_pending_ref_keeps_existing_once_only_outcomes(tmp_path, monkeypatch, action):
    """The fix changes scheduling, not the spine's claim/terminal authority."""
    store, owner = _enroll(tmp_path)
    store.record_feed_batch(
        owner,
        feed_seq(3),
        ["message_available:message_id=bmsg_synthetic"],
        resync_required=False,
    )
    calls = []

    async def timeout(*args, **kwargs):
        return None

    async def handle(owner_, mid, **kwargs):
        calls.append(mid)
        return {"action": action, "message_id": mid}

    monkeypatch.setattr(pull_mod, "events", timeout)
    monkeypatch.setattr(loop_mod, "handle_message_available", handle)
    for _ in range(2):
        asyncio.run(
            loop_mod.run_iteration(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )
    assert calls == ["bmsg_synthetic"]
    assert store.drain_feed_refs(owner, 3) == []


def test_204_pending_ref_stale_generation_cannot_fetch_or_wake(tmp_path, monkeypatch):
    store, owner = _enroll(tmp_path)
    _enable_bridge(tmp_path)
    store.record_feed_batch(
        owner,
        feed_seq(3),
        ["message_available:message_id=bmsg_synthetic"],
        resync_required=False,
    )
    store.invalidate(owner)

    async def timeout(*args, **kwargs):
        return None

    async def forbidden(*args, **kwargs):
        pytest.fail("old owner generation must not cross a traffic boundary")

    monkeypatch.setattr(pull_mod, "events", timeout)
    monkeypatch.setattr(pull_mod, "message", forbidden)
    monkeypatch.setattr(wake_mod, "screen_item", forbidden)
    monkeypatch.setattr(wake_mod, "deliver_wake", forbidden)
    with pytest.raises(StoreRefused):
        asyncio.run(
            loop_mod.run_iteration(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )
    assert store.drain_feed_refs(owner, 3) == ["message_available:message_id=bmsg_synthetic"]


@pytest.mark.parametrize("code", ["policy_stale", "PRIVATE_CODE_CANARY", None])
def test_pre_screen_diagnostic_is_closed_and_never_logs_exception_prose(caplog, code):
    failure = StoreRefused(code, "PRIVATE_BODY_CANARY token=PRIVATE_TOKEN_CANARY")
    with caplog.at_level("WARNING"):
        loop_mod._log_delivery_failure("pair_handling", failure)
    expected = "policy_stale" if code == "policy_stale" else "unexpected_failure"
    assert f"stage=pair_handling code={expected}" in caplog.text
    assert "PRIVATE_" not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


def test_resync_walk_completes_only_after_terminal_page(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    from local_operator.team_bridge.types import InboxPage

    pages = [
        InboxPage(items=(), next_cursor=InboxCursor("c1")),
        InboxPage(items=(), next_cursor=None),
    ]
    seen = []

    async def fake_inbox(owner_, *, config_dir, access, after=None, **kw):
        seen.append(after)
        return pages.pop(0)

    finishes = []

    def fake_finish(owner_, checkpoint_, *, config_dir):
        finishes.append(checkpoint_)

    monkeypatch.setattr(pull_mod, "inbox", fake_inbox)
    monkeypatch.setattr(pull_mod, "finish_resync", fake_finish)
    result = asyncio.run(loop_mod.resync_walk(owner, config_dir=tmp_path, access=_access(owner)))
    assert result["pages"] == 2 and len(finishes) == 1
    # Witness: a premature completion was IMPOSSIBLE by construction until the
    # terminal page merged; and calling finish early on a fresh walk refuses.
    store2 = Store(tmp_path)
    with pytest.raises(StoreRefused):
        store2.finish_resync(owner, InboxCursor(""))


# -- H4c: the spine ----------------------------------------------------------


def _enable_bridge(tmp_path: Path) -> None:
    """Turn the feature ON the way the fence reads it: the DISK config (S-8a).

    The fence re-reads ``config.yml`` at every boundary, so a case that wants to
    exercise anything BEHIND it must write the real file shape. Without this the
    spine refuses ``bridge_off`` before the first seam — and a case that awaits a
    seam signal then waits forever, which hides the misconfiguration.
    """

    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
    )


def _wire_spine(monkeypatch, tmp_path, *, disposition="deliver", body=None):
    """Replace the heavy seams with controlled fakes; return the recorder.

    The settings seam is patched ENABLED here (the spine's config gate); the
    dedicated off-arm below re-patches it to prove ZERO new traffic. The ack
    fake returns a CONFIRMED AckOutcome — the S-12 gate reads the outcome, so
    a bare ``None`` (the old shape) would now correctly HOLD.
    """

    from types import SimpleNamespace

    body = body or "hello from the org"
    record: dict[str, Any] = {"acks": [], "delivered": [], "screened": 0}

    async def fake_fetch(owner, message_id, *, config_dir, access, auth_store):
        return _item(body)

    async def fake_screen(owner, item, *, config_dir, auth_store, settings=None):
        record["screened"] += 1
        if disposition == "deliver":
            return ScreenResult(disposition="deliver", receipt=_receipt(body))
        if disposition == "bounce":
            return ScreenResult(disposition="bounce", receipt=_receipt(body, verdict="reject"))
        return ScreenResult(disposition="hold", hold_code="screening_hold")

    async def fake_message(owner, message_id, *, config_dir, access, client=None):
        return SimpleNamespace(message_id=message_id)

    deliveries_seen_sender_display: list[object] = []

    async def fake_ack(owner, message_id, target, receipt, *, config_dir, access, **_seam_kwargs):
        record["acks"].append(target)
        # The REAL frozen intent: the confirmed outcome must resolve to THIS
        # row by id (S-12), so the default fake records the real one — the
        # binding arms below deliberately break the tuple instead.
        from local_operator.team_bridge.store import Store as _AckStore

        intent_id = _AckStore(config_dir).record_ack_intent(
            owner, message_id, target, receipt, "connector-test"
        )
        # S-12 FULL TUPLE: the ledger row must read CONFIRMED, so the nominal
        # fake completes the real write path (record → mark) instead of
        # returning a confirmed outcome over a pending row.
        _AckStore(config_dir).mark_ack_result(owner, intent_id, state="confirmed", error_code=None)
        return SimpleNamespace(state="confirmed", intent_id=intent_id)

    async def fake_deliver(
        owner, payload, *, session_id, config_dir, bridge_meta=None, sender_display=None
    ):
        # D1 grew this seam: the wake carries the member's SCREENED display
        # fields so the receiving row can name them. Recorded so a case can
        # assert the carriage, not just tolerate it.
        deliveries_seen_sender_display.append(sender_display)
        record["delivered"].append(
            {"session": session_id, "payload": payload, "bridge_meta": bridge_meta}
        )
        return "delivered"

    monkeypatch.setattr(pull_mod, "message", fake_message)
    monkeypatch.setattr(wake_mod, "fetch_item", fake_fetch)
    monkeypatch.setattr(wake_mod, "screen_item", fake_screen)
    monkeypatch.setattr(wake_mod, "ack_read", fake_ack)
    monkeypatch.setattr(wake_mod, "deliver_wake", fake_deliver)
    # These cases model an ENGAGED designated session; F8's unused-session
    # gate has its own cases below.
    monkeypatch.setattr(wake_mod, "engagement_hold", lambda designation, config_dir: "")
    # Every hand-off passes ``pre_handoff_gate`` (PR #3063 review r7 M9),
    # which asks the hub for its CURRENT status whatever produced the verdict:
    # the rig answers a lawful one. A case that withdraws it re-patches.
    _hub_status(monkeypatch, OWNER)
    _enable_bridge(tmp_path)  # S-8a: the fence reads the DISK config
    return record, body


def test_spine_delivers_from_screened_bytes_and_releases_claim(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "delivered" and out["session"] == "sess_cos"
    assert record["acks"] == ["read"]
    payload = record["delivered"][0]["payload"]
    assert "team_bridge_message" in payload and "bmsg_1" in payload
    assert hashlib.sha256(body.encode()).hexdigest() in payload
    assert "Org Member" in payload and "Acme" in payload
    # Claim released: a fresh claim with a NEW token succeeds immediately.
    store.claim(owner, "bmsg_1", "sess_x", "tok2", expiry=time.time() + 60)


def test_spine_off_settings_start_zero_new_traffic(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    # The operator's config says off: fence #0 holds BEFORE the designation
    # read, the claim, the authenticated fetch or the screening. S-8a: the
    # fence reads the switch from DISK on every boundary (a cached in-memory
    # settings object is deliberately ignored), so the off arm must flip the
    # real file, not patch a seam the fence no longer consults.
    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": False}}}), encoding="utf-8"
    )
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold" and out["code"] == "bridge_off"
    assert record["screened"] == 0
    assert record["delivered"] == []
    assert record["acks"] == []


def test_spine_ack_pending_holds_without_delivery(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)

    async def pending_ack(
        owner_, message_id, target, receipt, *, config_dir, access, **_seam_kwargs
    ):
        record["acks"].append(target)
        # S-12: an ORDINARY return is not a confirmed read — the transport
        # RETURNS pending; the spine must HOLD, never fabricate the ack.
        return SimpleNamespace(state="pending", intent_id="intent-bmsg_1")

    monkeypatch.setattr(wake_mod, "ack_read", pending_ack)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold"
    assert out["code"] == "ack_not_confirmed:pending"
    assert record["delivered"] == []


def test_spine_ack_binding_mismatch_arms_hold_without_delivery(tmp_path, monkeypatch) -> None:
    """S-12: confirmed state alone is not enough — the frozen tuple must match.

    Each arm gives the spine a CONFIRMED outcome whose intent id resolves to a
    DIFFERENT frozen tuple (or a foreign owner's row): all must HOLD with zero
    delivery — never a fabricated ack.
    """
    from types import SimpleNamespace

    from local_operator.team_bridge.store import Store as _AckStore

    def _arm(case: Path, *, record_kwargs: dict[str, Any], record_owner=None):
        case.mkdir()
        store, owner = _enroll(case)
        record, body = _wire_spine(monkeypatch, case)
        receipt = _receipt(body)
        other_owner = record_owner or owner
        if other_owner is not owner:
            store.commit_enrollment(
                other_owner, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
            )
        intent_id = _AckStore(case).record_ack_intent(
            other_owner,
            record_kwargs.get("message_id", "bmsg_1"),
            record_kwargs.get("target", "read"),
            record_kwargs.get("receipt", receipt),
            "connector-test",
        )

        async def bound_ack(
            owner_, message_id, target, receipt_, *, config_dir, access, **_seam_kwargs
        ):
            record["acks"].append(target)
            return SimpleNamespace(state="confirmed", intent_id=intent_id)

        monkeypatch.setattr(wake_mod, "ack_read", bound_ack)
        out = asyncio.run(
            wake_mod.handle_message_available(
                owner, "bmsg_1", config_dir=case, auth_store=None, access=_access(owner)
            )
        )
        assert out["action"] == "hold", out
        assert out["code"] == "ack_binding_mismatch", out
        assert record["delivered"] == []

    # Wrong message: a real read intent for ANOTHER message id.
    _arm(tmp_path / "wrong_msg", record_kwargs={"message_id": "bmsg_other"})
    # Wrong target: a real BLOCKED intent (reject receipt) answering our read.
    _arm(
        tmp_path / "wrong_target",
        record_kwargs={
            "target": "blocked",
            "receipt": _receipt("hello from the org", verdict="reject"),
        },
    )
    # Wrong BODY HASH: a read intent for different screened bytes.
    _arm(tmp_path / "wrong_body", record_kwargs={"receipt": _receipt("a different body")})
    # Foreign owner: the row exists only in ANOTHER owner's namespace; the run's
    # lookup is keyed by ITS OWN owner, so it misses -> hold.
    case = tmp_path / "foreign_owner"
    case.mkdir()
    store, owner = _enroll(case)
    record, body = _wire_spine(monkeypatch, case)
    foreign = OwnerSnapshot(
        canonical_base_url="https://hub.example",
        account_id="acc_1",
        tenant_id="tnt_1",
        endpoint_id="ep_other",
        credential_id=9,
        enrollment_generation=3,
    )
    store.commit_enrollment(
        foreign, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
    )
    foreign_id = _AckStore(case).record_ack_intent(
        foreign, "bmsg_1", "read", _receipt(body), "connector-test"
    )
    # ``active_owner`` treats two committed enrollments as AMBIGUOUS authority
    # (None → the producer fence answers owner_changed before the binding check
    # is ever reached). The foreign row must only exist as ANOTHER namespace's
    # ledger entry, so retire its enrollment after the intent is recorded; the
    # intent row stays keyed by the foreign owner key.
    import sqlite3

    with sqlite3.connect(store._path) as raw:
        raw.execute(
            "UPDATE enrollments SET state = 'superseded' WHERE endpoint_id = ?",
            (foreign.endpoint_id,),
        )

    async def foreign_ack(
        owner_, message_id, target, receipt_, *, config_dir, access, **_seam_kwargs
    ):
        record["acks"].append(target)
        return SimpleNamespace(state="confirmed", intent_id=foreign_id)

    monkeypatch.setattr(wake_mod, "ack_read", foreign_ack)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=case, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold" and out["code"] == "ack_binding_mismatch"
    assert record["delivered"] == []


def test_spine_ack_binding_accepts_the_real_frozen_intent(tmp_path, monkeypatch) -> None:
    """Positive control: the SAME path delivers when the tuple matches."""
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "delivered"
    assert record["delivered"][0]["session"] == "sess_cos"


def test_spine_ack_full_tuple_mismatch_arms_hold(tmp_path, monkeypatch) -> None:
    """S-12 FULL TUPLE: a CONFIRMED row must equal THIS run's receipt.

    The ledger row is recorded with a DIFFERENT model (same message/body) and
    then confirmed; the spine screens the normal receipt and must HOLD before
    any phase advance or delivery — a wrong-model confirmation is not this
    screening result's ack.
    """

    from dataclasses import replace

    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    altered = replace(_receipt(body), model="model-b").validated()

    async def wrong_tuple_ack(
        owner_, message_id, target, receipt_, *, config_dir, access, **_seam_kwargs
    ):
        record["acks"].append(target)
        from local_operator.team_bridge.store import Store as _AckStore

        intent_id = _AckStore(config_dir).record_ack_intent(
            owner_, message_id, target, altered, "connector-test-wrong"
        )
        _AckStore(config_dir).mark_ack_result(owner_, intent_id, state="confirmed", error_code=None)
        return SimpleNamespace(state="confirmed", intent_id=intent_id)

    monkeypatch.setattr(wake_mod, "ack_read", wrong_tuple_ack)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold"
    assert out["code"] == "ack_tuple_mismatch", out
    assert record["delivered"] == []


def test_spine_ack_pending_ledger_with_confirmed_outcome_holds(tmp_path, monkeypatch) -> None:
    """A transport claiming CONFIRMED over a ledger row still PENDING must HOLD."""

    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)

    async def lying_ack(
        owner_, message_id, target, receipt_, *, config_dir, access, **_seam_kwargs
    ):
        record["acks"].append(target)
        from local_operator.team_bridge.store import Store as _AckStore

        intent_id = _AckStore(config_dir).record_ack_intent(
            owner_, message_id, target, receipt_, "connector-test"
        )
        # Deliberately NOT marked confirmed: the outcome's claim is a lie the
        # frozen LEDGER state must catch.
        return SimpleNamespace(state="confirmed", intent_id=intent_id)

    monkeypatch.setattr(wake_mod, "ack_read", lying_ack)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold"
    assert out["code"] == "ack_tuple_mismatch", out
    assert record["delivered"] == []


def test_spine_phase_persistence_refusal_holds_without_delivery(tmp_path, monkeypatch) -> None:
    """S-12: a handoff phase that cannot be persisted must NOT deliver.

    The request path is driven to the phase write (a fake stamped envelope +
    a confirmed bound ack), where ``mark_delivery_phase`` REFUSES: the spine
    must HOLD with zero delivery — never log-and-continue.
    """
    from types import SimpleNamespace

    from local_operator.team_bridge.store import Store as _SpineStore
    from local_operator.team_bridge.store import StoreRefused as _Refused

    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)

    async def request_message(owner_, message_id, *, config_dir, access, client=None):
        return SimpleNamespace(
            message_id=message_id,
            message_type="request",
            request_state="pending",
            thread_id="bth_1",
            request_expires_at="",
            request_ref="req-1",
            sender_account_id="acc_sender",
            sender_endpoint_id="ep_sender",
            recipient_account_id="acc_1",
            recipient_endpoint_id="ep_1",
            tenant_id="tnt_1",
        )

    monkeypatch.setattr(pull_mod, "message", request_message)

    def fake_write(self, owner_, *, delivery_ref, message_id, designated_session, **kwargs):
        return {
            "delivery_ref": delivery_ref,
            "display_digest": "dd44" * 16,
            "phase": "prepared",
        }

    def refuse_phase(self, owner_, delivery_ref, phase):
        raise _Refused("phase_conflict", "the phase write refuses")

    monkeypatch.setattr(_SpineStore, "write_screened_request_delivery", fake_write)
    monkeypatch.setattr(_SpineStore, "mark_delivery_phase", refuse_phase)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold"
    assert str(out["code"]).startswith("phase_refused")
    assert record["delivered"] == []


def test_spine_hold_and_bounce_never_wake(tmp_path, monkeypatch) -> None:
    for disposition, expected in (("hold", "hold"), ("bounce", "bounce")):
        case = tmp_path / disposition
        case.mkdir()
        store, owner = _enroll(case)
        record, _ = _wire_spine(monkeypatch, case, disposition=disposition)
        out = asyncio.run(
            wake_mod.handle_message_available(
                owner, "bmsg_1", config_dir=case, auth_store=None, access=_access(owner)
            )
        )
        assert out["action"] == expected
        assert record["delivered"] == []
        if disposition == "bounce":
            assert record["acks"] == ["blocked"]
        else:
            assert record["acks"] == []


def test_spine_no_designation_is_badge_only(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path, session=None)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "badge_only"
    assert record["screened"] == 0 and record["delivered"] == []


def test_spine_claimed_elsewhere_skips(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    store.claim(owner, "bmsg_1", "sess_other", "tok_other", expiry=time.time() + 300)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "claimed_elsewhere"
    assert record["delivered"] == []


class _FakeClock:
    """``time.time`` under the test's control: the claim lease (900 s) and the
    hold schedule both read it, so a lease can expire without waiting."""

    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


def _one_pending_ref(store: Store, owner: OwnerSnapshot, message_id: str = "bmsg_1") -> None:
    """A durable feed ref the next iteration drains (a 204 still drains it)."""

    store.record_feed_batch(
        owner, feed_seq(3), [f"message_available:message_id={message_id}"], resync_required=False
    )


async def _quiet_events(*args: Any, **kwargs: Any) -> None:
    return None  # 204: no new traffic; only the committed ref is drained


def _iterate(owner: OwnerSnapshot, tmp_path: Path, access: Any = None) -> dict[str, Any]:
    return asyncio.run(
        loop_mod.run_iteration(
            owner, config_dir=tmp_path, access=access or _access(owner), auth_store=None
        )
    )


def test_dead_connector_claim_is_redriven_once_after_the_lease(tmp_path, monkeypatch) -> None:
    """PR #3038 review R2-3: a connector killed mid-spine leaves its claim
    ``held`` (its ``finally`` never ran). The restarted connector's spine is
    refused ``claimed_elsewhere``; the loop used to consume the ref with no
    hold, so after the lease lapsed nothing referenced the message and it was
    never delivered. It must now park a hold due at the lease expiry, then
    deliver once. (At-least-once in general: a holder killed between the
    handoff and its release is woken twice — see loop.HOLD_STALE_CLAIM.)"""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    # What a SIGKILLed spine leaves behind: its connector-minted claim, held.
    store.claim(
        owner, "bmsg_1", "sess_cos", "connector:dead-owner", expiry=clock.now + wake_mod.CLAIM_TTL_S
    )
    _one_pending_ref(store, owner)

    first = _iterate(owner, tmp_path)
    assert [o["action"] for o in first["handled"]] == ["claimed_elsewhere"]
    assert record["delivered"] == []
    assert store.drain_feed_refs(owner, 3) == []
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] == loop_mod.HOLD_STALE_CLAIM
    assert hold["retry_at"] >= clock.now + wake_mod.CLAIM_TTL_S

    # Before the lease lapses the parked message is not due: nothing moves.
    clock.now += wake_mod.CLAIM_TTL_S - 60
    _iterate(owner, tmp_path)
    assert record["delivered"] == []

    clock.now += 60 + loop_mod.STALE_CLAIM_MARGIN_S + 1
    _iterate(owner, tmp_path)
    assert len(record["delivered"]) == 1
    assert store.screening_hold(owner, "bmsg_1") is None
    for _ in range(3):  # later passes never deliver it again
        clock.now += wake_mod.CLAIM_TTL_S
        _iterate(owner, tmp_path)
    assert len(record["delivered"]) == 1


def test_live_second_holder_is_never_double_delivered(tmp_path, monkeypatch) -> None:
    """The parked retry must be a no-op when the claim belonged to a LIVE
    spine: that holder delivers and releases, and a released claim retires the
    parked row instead of re-driving it."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    real_deliver = wake_mod.deliver_wake
    _one_pending_ref(store, owner)

    async def scenario() -> None:
        in_handoff = asyncio.Event()
        finish = asyncio.Event()

        async def held_deliver(*args: Any, **kwargs: Any) -> str:
            in_handoff.set()
            await finish.wait()
            return await real_deliver(*args, **kwargs)

        monkeypatch.setattr(wake_mod, "deliver_wake", held_deliver)
        holder = asyncio.create_task(
            wake_mod.handle_message_available(
                owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
            )
        )
        await in_handoff.wait()  # the live holder is mid-handoff, claim held
        racing = await loop_mod.run_iteration(
            owner, config_dir=tmp_path, access=_access(owner), auth_store=None
        )
        assert [o["action"] for o in racing["handled"]] == ["claimed_elsewhere"]
        hold = store.screening_hold(owner, "bmsg_1")
        assert hold is not None and hold["hold_code"] == loop_mod.HOLD_STALE_CLAIM
        finish.set()
        assert (await holder)["action"] == "delivered"

    asyncio.run(scenario())
    assert len(record["delivered"]) == 1
    clock.now += wake_mod.CLAIM_TTL_S + loop_mod.STALE_CLAIM_MARGIN_S + 1
    retried = _iterate(owner, tmp_path)
    assert retried["retries"]["outcomes"] == [{"message_id": "bmsg_1", "action": "retired"}]
    assert store.screening_hold(owner, "bmsg_1") is None
    clock.now += wake_mod.CLAIM_TTL_S
    _iterate(owner, tmp_path)
    assert len(record["delivered"]) == 1


def test_a_stop_during_the_stale_claim_redrive_never_loses_the_message(
    tmp_path, monkeypatch
) -> None:
    """PR #3063 review M1: the clean stop CANCELS a re-drive, and the spine's
    ``finally`` then releases its OWN fresh claim with no outcome. A
    ``stale_claim`` park read "released" as "the holder's outcome stands",
    retired itself on the restart, and the message (whose ref was consumed
    long before) was never delivered. The interrupted re-drive must leave an
    ordinary retry behind, which delivers once on the restart."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    store.claim(
        owner, "bmsg_1", "sess_cos", "connector:dead-owner", expiry=clock.now + wake_mod.CLAIM_TTL_S
    )
    _one_pending_ref(store, owner)
    _iterate(owner, tmp_path)  # parks the stale claim; the ref is consumed
    real_screen = wake_mod.screen_item

    async def stopped_mid_screen(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError  # what ``_supervise``'s cancel delivers

    monkeypatch.setattr(wake_mod, "screen_item", stopped_mid_screen)
    clock.now += wake_mod.CLAIM_TTL_S + loop_mod.STALE_CLAIM_MARGIN_S + 1
    with pytest.raises(asyncio.CancelledError):
        _iterate(owner, tmp_path)
    released = store.claim_status(owner, "bmsg_1")
    assert released is not None and released["state"] == "released"
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] != loop_mod.HOLD_STALE_CLAIM

    # The restarted connector: screening works again.
    monkeypatch.setattr(wake_mod, "screen_item", real_screen)
    for _ in range(4):
        clock.now += wake_mod.CLAIM_TTL_S
        _iterate(owner, tmp_path)
    assert len(record["delivered"]) == 1
    assert store.screening_hold(owner, "bmsg_1") is None


#: Captured at import, before any case's monkeypatch replaces the seam.
_PRODUCTION_ACK_READ = wake_mod.ack_read
#: The REAL screening seam, captured before any case patches it: the
#: recovery matrix drives the production cycle and cache through it.
_PRODUCTION_SCREEN_ITEM = wake_mod.screen_item


class _AckWire:
    """The hub's ack endpoint, faked at the wire seam so the REAL ``pull.ack``
    (intent ledger, ``client_ref``, replay) runs. The hub treats a repeated
    ``client_ref`` as a replay and a new one for a read message as a conflict.
    """

    base_url = OWNER.canonical_base_url  # the destination guard's evidence

    def __init__(self) -> None:
        self.refs: list[str] = []

    async def ack(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        ref = str(kwargs["client_ref"])
        if self.refs and ref != self.refs[0]:
            raise pull_mod.PullRefused("intent_conflict", "a different client_ref")
        self.refs.append(ref)
        return {"delivery": {"state": kwargs["state"], "state_at": "t"}}


@pytest.mark.parametrize("death", ["stop", "kill"])
def test_a_stop_after_the_read_ack_still_delivers_once(tmp_path, monkeypatch, death) -> None:
    """PR #3063 QA Q-1: a connector stopped (claim released, ref kept) or
    killed (claim held, ``stale_claim`` park) AFTER the confirmed read ACK but
    BEFORE the session handoff. The replay minted a fresh ``client_ref``, the
    ledger refused it as ``intent_conflict``, and the message was never
    handed off. The replay must reuse the stored ref and reach the handoff."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    wire = _AckWire()
    monkeypatch.setattr(pull_mod, "_client_for", lambda config_dir, client: wire)
    _hub_status(monkeypatch, owner)  # the recorded-verdict hand-off asks the hub
    # The real ``pull.ack`` guards the owner on a REAL pinned OAuth access.
    access = OAuthAccess(
        access_token="tok",
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        org_id=owner.tenant_id,
    )
    # The PRODUCTION seam, not the fixture's fake: it is what mints the ref.
    monkeypatch.setattr(wake_mod, "ack_read", _PRODUCTION_ACK_READ)
    delivered_fake = wake_mod.deliver_wake

    async def dies_at_handoff(*args: Any, **kwargs: Any) -> str:
        raise asyncio.CancelledError  # the stop lands after the ACK

    monkeypatch.setattr(wake_mod, "deliver_wake", dies_at_handoff)
    _one_pending_ref(store, owner)
    if death == "stop":
        with pytest.raises(asyncio.CancelledError):
            _iterate(owner, tmp_path, access)
        assert store.drain_feed_refs(owner, 3)  # the ref survived the stop
    else:
        # SIGKILL: the spine's ``finally`` never runs, so do what it leaves —
        # ack confirmed, claim held, then the restart parks the stale claim.
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                wake_mod.handle_message_available(
                    owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=access
                )
            )
        store.claim(
            owner, "bmsg_1", "sess_cos", "connector:killed", expiry=clock.now + wake_mod.CLAIM_TTL_S
        )
    assert len(wire.refs) == 1 and record["delivered"] == []

    monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    for _ in range(4):  # the restarted connector, across a lease or more
        _iterate(owner, tmp_path, access)
        clock.now += wake_mod.CLAIM_TTL_S
    assert len(record["delivered"]) == 1
    assert set(wire.refs) == {wire.refs[0]}  # never a second, conflicting ref
    assert store.screening_hold(owner, "bmsg_1") is None
    assert store.drain_feed_refs(owner, 3) == []


def _real_ack_access(monkeypatch: Any, owner: OwnerSnapshot) -> tuple[_AckWire, OAuthAccess]:
    """Wire the PRODUCTION ack seam (intent ledger + stored ``client_ref``)."""

    wire = _AckWire()
    monkeypatch.setattr(pull_mod, "_client_for", lambda config_dir, client: wire)
    monkeypatch.setattr(wake_mod, "ack_read", _PRODUCTION_ACK_READ)
    access = OAuthAccess(
        access_token="tok",
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        org_id=owner.tenant_id,
    )
    return wire, access


@pytest.mark.parametrize("first_code", ["delivery_failed", "bridge_off"])
def test_a_redriven_hold_wakes_once_across_many_passes(tmp_path, monkeypatch, first_code) -> None:
    """PR #3063 review B1: a hold written AFTER the receipt commit (a failed
    handoff, a fence that closed after screening) re-drives through a cache
    hit, which commits no new receipt — so nothing deleted the row. With the
    stored ACK ref replaying, every later pass reached the handoff again and
    woke the session on EVERY pass. The pass must retire the row by its own
    key on a terminal outcome, whatever its code."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    wire, access = _real_ack_access(monkeypatch, owner)
    _hub_status(monkeypatch, owner)  # the recorded-verdict hand-off asks the hub
    delivered_fake = wake_mod.deliver_wake
    if first_code == "delivery_failed":

        async def fails_once(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("handoff refused")

        monkeypatch.setattr(wake_mod, "deliver_wake", fails_once)
        _one_pending_ref(store, owner)
        _iterate(owner, tmp_path, access)  # screened, acked, then held
        monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    else:
        # Screened and acked, then the feature fence closed before handoff.
        assert (
            asyncio.run(
                wake_mod.handle_message_available(
                    owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=access
                )
            )["action"]
            == "delivered"
        )
        record["delivered"].clear()
        store.persist_hold(owner, "bmsg_1", first_code, retry_at=clock.now)
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] == first_code
    assert record["delivered"] == []
    for _ in range(12):
        clock.now += loop_mod.RETRY_MAX_S
        _iterate(owner, tmp_path, access)
    assert len(record["delivered"]) == 1
    assert store.screening_hold(owner, "bmsg_1") is None
    assert len(set(wire.refs)) == 1


def test_a_stale_claim_retry_redrive_is_bounded(tmp_path, monkeypatch) -> None:
    """PR #3063 review M2: a ``stale_claim_retry`` re-drive whose outcome is
    neither terminal nor a hold (``duplicate`` from the conversation
    reservation witness, ``badge_only``, ``claimed_elsewhere``) left the row
    due with ``retry_at`` frozen, so it was re-driven on EVERY pass. Over 200
    one-second passes it must be re-driven a bounded number of times."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    _enable_bridge(tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    calls: list[str] = []

    async def duplicate_spine(owner_, message_id, **kwargs):
        calls.append(message_id)
        return {"action": "duplicate", "message_id": message_id}

    monkeypatch.setattr(loop_mod, "handle_message_available", duplicate_spine)
    store.persist_hold(owner, "bmsg_1", loop_mod.STALE_CLAIM_REDRIVE_CODE, retry_at=clock.now - 1)
    for _ in range(200):
        clock.now += 1.0
        _iterate(owner, tmp_path)
    assert 1 <= len(calls) <= 2
    assert store.screening_hold(owner, "bmsg_1") is None


@pytest.mark.parametrize("action", ["badge_only", "claimed_elsewhere"])
def test_a_non_terminal_redrive_outcome_backs_off(tmp_path, monkeypatch, action) -> None:
    """M2's other arm: an outcome that is neither terminal nor a hold keeps
    the row (badge-only owners and live claimants are real waits) but on the
    SAME bounded doubling backoff as a hold, never due on every pass."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    _enable_bridge(tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    calls: list[str] = []

    async def waiting_spine(owner_, message_id, **kwargs):
        calls.append(message_id)
        return {"action": action, "message_id": message_id, "code": "claimed"}

    monkeypatch.setattr(loop_mod, "handle_message_available", waiting_spine)
    store.persist_hold(owner, "bmsg_1", loop_mod.STALE_CLAIM_REDRIVE_CODE, retry_at=clock.now - 1)
    for _ in range(200):
        clock.now += 1.0
        _iterate(owner, tmp_path)
    # 60 s, then 120 s: two re-drives inside 200 s, not 200.
    assert 1 <= len(calls) <= 3
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["retry_at"] > clock.now
    # Review n4: the owner sees WHY it waits, not the re-drive mechanism.
    assert hold["hold_code"] == action


def test_a_stop_during_the_read_ack_post_is_flushed_and_delivered_once(
    tmp_path, monkeypatch
) -> None:
    """PR #3063 review m4 (operator: unacked messages must catch up after the
    device was off). A stop landing DURING the read-ACK POST leaves a PENDING
    intent; screening's step 5 then answers ``ack_recovery`` for that message
    on every re-drive, and no connector path ever flushed the intent, so the
    message was held forever. The re-drive must replay the STORED intent (its
    tuple and its ``client_ref``) and continue to the handoff."""

    from local_operator.team_bridge import screening as screening_mod

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    wire, access = _real_ack_access(monkeypatch, owner)
    screened = wake_mod.screen_item

    async def screening_step_5(owner_, item, *, config_dir, auth_store, settings=None):
        # The REAL pipeline's first reconcile (screening.py step 5): a pending
        # read intent is never a fresh admission. The fake screen behind it
        # stands in for the model call only.
        if Store(config_dir).find_pending_intent(owner_, "bmsg_1", "read") is not None:
            return screening_mod.ScreenResult("hold", hold_code="ack_recovery")
        result = await screened(
            owner_, item, config_dir=config_dir, auth_store=auth_store, settings=settings
        )
        return result

    monkeypatch.setattr(wake_mod, "screen_item", screening_step_5)
    _hub_status(monkeypatch, owner)
    real_wire_ack = wire.ack

    async def stopped_mid_post(message_id: str, **kwargs: Any) -> dict[str, Any]:
        # The intent is durable (pull.ack records it first); the POST never
        # completes. A stop's cancellation lands exactly here.
        wire.refs.append(str(kwargs["client_ref"]))
        raise asyncio.CancelledError

    monkeypatch.setattr(wire, "ack", stopped_mid_post)
    _one_pending_ref(store, owner)
    with pytest.raises(asyncio.CancelledError):
        _iterate(owner, tmp_path, access)
    pending = store.find_pending_intent(owner, "bmsg_1", "read")
    assert pending is not None and record["delivered"] == []
    original_ref = pending["client_ref"]

    # The restarted connector: the hub is reachable again.
    wire.refs.clear()
    monkeypatch.setattr(wire, "ack", real_wire_ack)
    outcomes = []
    for _ in range(6):
        outcomes.append(_iterate(owner, tmp_path, access))
        clock.now += loop_mod.RETRY_MAX_S
    held = [o["code"] for out in outcomes for o in out["handled"] if o["action"] == "hold"]
    held += [
        o["action"] for out in outcomes for o in out["retries"].get("outcomes", ())
    ]  # the retry pass reports per-row actions
    assert len(record["delivered"]) == 1, held
    assert set(wire.refs) == {original_ref}  # the ORIGINAL ref only, never a second
    intent = store.ack_intent(owner, pending["intent_id"])
    assert intent is not None and intent["state"] == "confirmed"
    assert store.screening_hold(owner, "bmsg_1") is None
    assert store.drain_feed_refs(owner, 3) == []


def _hub_status(monkeypatch: Any, owner: OwnerSnapshot) -> dict[str, Any]:
    """The hub's CURRENT status as the production sealed fetcher returns it.

    The recorded-verdict path runs the same authority gate a fresh screen
    runs (``scheduler.verified_status_capture``, PR #3063 review M6), so a
    rig that hands off on a recorded verdict must answer a lawful status. The
    returned dict is live: a case flips ``available`` or the endpoint
    ``state`` to model the hub withdrawing authority while the device is off.
    Only the transport is faked; normalization, consent currency and the
    floor advance are the real ones.
    """

    from local_operator.team_bridge import scheduler as scheduler_mod
    from local_operator.team_bridge.client import (
        SCREENING_TEMPLATE_V1,
        SCREENING_VERSION_V1,
    )

    hub = {"available": True, "endpoint_state": "active", "fetches": 0}

    def fetcher(*, config_dir: Path, owner: Any, auth_store: Any = None, transport: Any = None):
        async def _fetch() -> Any:
            hub["fetches"] += 1
            raw = {
                "api_version": "1",
                "available": hub["available"],
                "tenants": [
                    {
                        "tenant_id": owner.tenant_id,
                        "tenant_name": "Org",
                        "eligible": True,
                        "endpoint": {
                            "id": owner.endpoint_id,
                            "label": "e",
                            "state": hub["endpoint_state"],
                        },
                        "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
                    }
                ],
                "guidance": {"version": "bridge-guidance-v1", "text": "g"},
                "screening": {"version": SCREENING_VERSION_V1, "template": SCREENING_TEMPLATE_V1},
                "consent": {"version": "bridge_consent_v1", "text": "c"},
            }
            return scheduler_mod.VerifiedStatus(scheduler_mod._VERIFIED_TOKEN, raw)

        return _fetch

    monkeypatch.setattr(scheduler_mod, "default_status_fetcher", fetcher)
    return hub


def _stop_during_the_read_ack_post(monkeypatch, tmp_path) -> tuple[Any, ...]:
    """A connector stopped DURING the read-ACK POST, the m4 shape: the intent
    is durable and pending, the hub never answered, the ref is still due."""

    from local_operator.team_bridge import screening as screening_mod

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    wire, access = _real_ack_access(monkeypatch, owner)
    screens: list[str] = []
    model = {"name": "qa-default"}

    async def screen(owner_, item, *, config_dir, auth_store, settings=None):
        # screening.py step 5 (a pending read intent is never admitted), then
        # a fresh allow under whatever model is CURRENT — the stand-in for a
        # cache miss after a model/template change.
        if Store(config_dir).find_pending_intent(owner_, "bmsg_1", "read") is not None:
            return screening_mod.ScreenResult("hold", hold_code="ack_recovery")
        screens.append(model["name"])
        receipt = ScreeningReceipt(
            verdict="allow",
            model=model["name"],
            prompt_version="bridge-screen-v1",
            template_sha256="ab" * 32,
            body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            categories=(),
        ).validated()
        return ScreenResult(disposition="deliver", receipt=receipt)

    monkeypatch.setattr(wake_mod, "screen_item", screen)
    _hub_status(monkeypatch, owner)
    real_wire_ack = wire.ack

    async def stopped_mid_post(message_id: str, **kwargs: Any) -> dict[str, Any]:
        wire.refs.append(str(kwargs["client_ref"]))
        raise asyncio.CancelledError

    monkeypatch.setattr(wire, "ack", stopped_mid_post)
    _one_pending_ref(store, owner)
    with pytest.raises(asyncio.CancelledError):
        _iterate(owner, tmp_path, access)
    pending = store.find_pending_intent(owner, "bmsg_1", "read")
    assert pending is not None
    wire.refs.clear()
    monkeypatch.setattr(wire, "ack", real_wire_ack)
    return clock, store, owner, record, wire, access, screens, model, pending


def test_a_changed_screening_tuple_after_the_replay_delivers_on_the_stored_verdict(
    tmp_path, monkeypatch
) -> None:
    """PR #3063 review M3: after the m4 replay CONFIRMS the stored intent, a
    re-screen under a CHANGED tuple (a ``lop`` update shipping a new model or
    template between the stop and the restart) re-recorded the allow under the
    stored ref; the write-once ledger raised ``intent_conflict`` out of the
    spine — hub read, nobody woken, no hold, and the connector stuck on its
    failure backoff. The hub's confirmed verdict IS the screening decision:
    the handoff must carry it, with no re-screen."""

    clock, store, owner, record, wire, access, screens, model, pending = (
        _stop_during_the_read_ack_post(monkeypatch, tmp_path)
    )
    screens.clear()
    model["name"] = "qa-default-v2"  # the update landed while the device was off
    for _ in range(4):
        _iterate(owner, tmp_path, access)
        clock.now += loop_mod.RETRY_MAX_S
    assert len(record["delivered"]) == 1
    assert screens == []  # delivered on the stored verdict, never re-screened
    assert set(wire.refs) == {pending["client_ref"]}  # the ORIGINAL ref only
    assert store.screening_hold(owner, "bmsg_1") is None
    assert store.drain_feed_refs(owner, 3) == []


@pytest.mark.parametrize("change", ["default_model", "template_sha256"])
def test_a_confirmed_read_with_a_stopped_handoff_delivers_on_its_verdict(
    tmp_path, monkeypatch, change
) -> None:
    """PR #3063 review M4 (the r4 probe): the hub CONFIRMED the read ACK, a
    stop cancelled the handoff, and an update changed the screening tuple
    before the restart. With no PENDING intent the spine re-screened, the
    write-once ledger refused the new tuple, and the message was held
    ``ack_refused:intent_conflict`` forever. ONE rule now covers pending and
    confirmed: a recorded read verdict for these bytes is handed off as is.
    Passes cross a claim lease; totals are asserted at the end."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    wire, access = _real_ack_access(monkeypatch, owner)
    tuple_now = {"default_model": "qa-default", "template_sha256": "ab" * 32}
    screens: list[dict[str, str]] = []

    async def screen(owner_, item, *, config_dir, auth_store, settings=None):
        screens.append(dict(tuple_now))
        receipt = ScreeningReceipt(
            verdict="allow",
            model=tuple_now["default_model"],
            prompt_version="bridge-screen-v1",
            template_sha256=tuple_now["template_sha256"],
            body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            categories=(),
        ).validated()
        return ScreenResult(disposition="deliver", receipt=receipt)

    monkeypatch.setattr(wake_mod, "screen_item", screen)
    _hub_status(monkeypatch, owner)
    delivered_fake = wake_mod.deliver_wake

    async def stopped_at_the_handoff(*args: Any, **kwargs: Any) -> str:
        raise asyncio.CancelledError  # after the CONFIRMED read ACK

    monkeypatch.setattr(wake_mod, "deliver_wake", stopped_at_the_handoff)
    _one_pending_ref(store, owner)
    with pytest.raises(asyncio.CancelledError):
        _iterate(owner, tmp_path, access)
    recorded = store.recorded_intent(owner, "bmsg_1", "read")
    assert recorded is not None and recorded["state"] == "confirmed"
    assert len(screens) == 1 and len(wire.refs) == 1

    tuple_now[change] = "qa-default-v2" if change == "default_model" else "cd" * 32
    monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    for _ in range(4):  # across more than one 15-minute claim lease
        _iterate(owner, tmp_path, access)
        clock.now += wake_mod.CLAIM_TTL_S + 5
    assert len(record["delivered"]) == 1
    assert len(screens) == 1  # the restart never re-screened
    # Never a second ref: the hand-off's standing re-check re-sends the SAME
    # stored ref as a same-state repeat (a hub no-op), never a new tuple.
    assert set(wire.refs) == {wire.refs[0]}
    assert store.screening_hold(owner, "bmsg_1") is None
    assert store.drain_feed_refs(owner, 3) == []


def _confirmed_then_stopped_at_the_handoff(monkeypatch, tmp_path) -> tuple[Any, ...]:
    """The r4/r5 probe shape: screened, read ACK CONFIRMED by the hub, then a
    stop cancels the handoff. Returns the rig with the stop already applied."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    wire, access = _real_ack_access(monkeypatch, owner)
    screens: list[int] = []

    async def screen(owner_, item, *, config_dir, auth_store, settings=None):
        screens.append(int(owner_.enrollment_generation))
        receipt = _receipt(body)
        return ScreenResult(disposition="deliver", receipt=receipt)

    monkeypatch.setattr(wake_mod, "screen_item", screen)
    hub = _hub_status(monkeypatch, owner)
    delivered_fake = wake_mod.deliver_wake

    async def stopped_at_the_handoff(*args: Any, **kwargs: Any) -> str:
        raise asyncio.CancelledError

    monkeypatch.setattr(wake_mod, "deliver_wake", stopped_at_the_handoff)
    _one_pending_ref(store, owner)
    with pytest.raises(asyncio.CancelledError):
        _iterate(owner, tmp_path, access)
    recorded = store.recorded_intent(owner, "bmsg_1", "read")
    assert recorded is not None and recorded["state"] == "confirmed"
    monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    return clock, store, owner, record, wire, access, screens, hub


_ACROSS_A_LEASE_S = max(wake_mod.CLAIM_TTL_S, loop_mod.RETRY_MAX_S) + 5


@pytest.mark.parametrize("withdrawn", ["available", "endpoint_state"])
def test_a_recorded_verdict_still_passes_the_hubs_current_authority(
    tmp_path, monkeypatch, withdrawn
) -> None:
    """PR #3063 review M6 (the r5 probe). The invariant: a recorded read
    verdict replaces ONLY the model call. Here the hub withdrew messaging (or
    deactivated the endpoint) while the device was off; the recorded-verdict
    path skipped the screen and with it the hub-status gate, and handed the
    message off anyway. It must hold ``policy_stale`` — visibly — for as long
    as the hub withholds authority, and deliver once on its stored verdict
    (still no model call) when the hub restores it. Totals at the END."""

    clock, store, owner, record, wire, access, screens, hub = (
        _confirmed_then_stopped_at_the_handoff(monkeypatch, tmp_path)
    )
    if withdrawn == "available":
        hub["available"] = False
    else:
        hub["endpoint_state"] = "disabled"
    for _ in range(4):  # across more than one 15-minute claim lease
        _iterate(owner, tmp_path, access)
        clock.now += _ACROSS_A_LEASE_S
    assert record["delivered"] == []
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] == "policy_stale"
    assert hub["fetches"] >= 1  # the gate actually asked the hub

    hub.update(available=True, endpoint_state="active")
    for _ in range(4):
        _iterate(owner, tmp_path, access)
        clock.now += _ACROSS_A_LEASE_S
    assert len(record["delivered"]) == 1
    assert screens == [1]  # the only model call is the original screen
    assert set(wire.refs) == {wire.refs[0]}  # one ref, the original, re-checked
    assert store.screening_hold(owner, "bmsg_1") is None


def test_a_verdict_confirmed_before_a_relogin_is_carried_through_the_gate(
    tmp_path, monkeypatch
) -> None:
    """PR #3063 review M6's re-login variant, under review r7 M8's rule:
    ``owner_key`` survives a same-account re-login, so the recorded intent is
    still found. The hub already HOLDS that read verdict, and its ack ledger
    is write-once per message, so a re-screen under the new generation can
    only mint a second verdict the hub refuses. The verdict is carried — no
    model call — through ``pre_handoff_gate`` under the NEW generation (its
    authority, its owner pin, the hub's per-ack sender check), then delivered
    exactly once. Totals at the end, across leases."""

    clock, store, owner, record, wire, access, screens, _hub = (
        _confirmed_then_stopped_at_the_handoff(monkeypatch, tmp_path)
    )
    relogged = OwnerSnapshot(
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        canonical_base_url=owner.canonical_base_url,
        enrollment_generation=store.invalidate(owner),
    )
    store.record_feed_batch(
        relogged,
        feed_seq(9),
        ["message_available:message_id=bmsg_1"],
        resync_required=False,
    )
    for _ in range(4):
        _iterate(relogged, tmp_path, access)
        clock.now += _ACROSS_A_LEASE_S
    assert screens == [1]  # the hub-held verdict is carried, never re-made
    assert len(record["delivered"]) == 1
    assert store.screening_hold(relogged, "bmsg_1") is None


def test_a_screening_rehold_keeps_the_hold_age(tmp_path, monkeypatch) -> None:
    """PR #3063 review m10: a re-drive held AGAIN by screening (which writes
    through ``Store.persist_hold``) used INSERT OR REPLACE, so the hold's
    ``created_at`` reset on every pass and its age read 0 forever. The age
    must grow across passes while the reason changes."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    codes = iter(["policy_stale", "breaker_open"] * 4)

    async def screening_holds(owner_, item, *, config_dir, auth_store, settings=None):
        code = next(codes)
        Store(config_dir).persist_hold(owner_, "bmsg_1", code)  # screening's own write
        return ScreenResult(disposition="hold", hold_code=code)

    monkeypatch.setattr(wake_mod, "screen_item", screening_holds)
    _one_pending_ref(store, owner)
    ages: list[tuple[str, int]] = []
    for _ in range(4):
        _iterate(owner, tmp_path)
        hold = store.screening_hold(owner, "bmsg_1")
        assert hold is not None
        ages.append((hold["hold_code"], int(clock.now - hold["created_at"])))
        clock.now += 3700
    assert [code for code, _ in ages] == ["policy_stale", "breaker_open"] * 2
    assert [age for _, age in ages] == [0, 3700, 7400, 11100]


def test_recoding_a_hold_keeps_its_age(tmp_path, monkeypatch) -> None:
    """PR #3063 review m9: re-coding a row (a new hold reason, or a
    ``badge_only``/``claimed_elsewhere`` outcome) went through INSERT OR
    REPLACE and reset ``created_at``, so the owner saw an old wait as new."""

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    _enable_bridge(tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    outcomes = iter(
        [{"action": "badge_only"}, {"action": "hold", "code": "retry_later"}]
        + [{"action": "claimed_elsewhere", "code": "claimed"}] * 4
    )

    async def flipping_spine(owner_, message_id, **kwargs):
        return {**next(outcomes), "message_id": message_id}

    monkeypatch.setattr(loop_mod, "handle_message_available", flipping_spine)
    store.persist_hold(owner, "bmsg_1", "screening_hold", retry_at=clock.now - 1)
    first = store.screening_hold(owner, "bmsg_1")
    assert first is not None
    born = first["created_at"]
    codes = []
    for _ in range(3):
        clock.now += loop_mod.RETRY_MAX_S + 1
        _iterate(owner, tmp_path)
        hold = store.screening_hold(owner, "bmsg_1")
        assert hold is not None
        codes.append(hold["hold_code"])
        assert hold["created_at"] == born
    assert codes == ["badge_only", "retry_later", "claimed_elsewhere"]


def test_an_ack_ledger_refusal_is_a_visible_hold_never_an_escape(tmp_path, monkeypatch) -> None:
    """M3 defence in depth: any ``StoreRefused`` out of ``ack_read`` (the
    write-once ledger, a fenced owner) is a typed hold on the bounded backoff,
    never an exception that leaves the message with no hold row."""

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)

    async def refusing_ack(*args: Any, **kwargs: Any) -> Any:
        raise StoreRefused("intent_conflict", "write-once intent tuple differs")

    monkeypatch.setattr(wake_mod, "ack_read", refusing_ack)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out == {"action": "hold", "message_id": "bmsg_1", "code": "ack_refused:intent_conflict"}
    assert record["delivered"] == []


def test_a_bounce_whose_blocked_ack_did_not_confirm_is_a_hold(tmp_path, monkeypatch) -> None:
    """PR #3063 review m7: ``bounce`` was terminal whatever the blocked-ACK
    outcome, so a 503 left a pending ``blocked`` intent nothing ever flushed
    and the re-drive row was retired. Only a CONFIRMED blocked ACK ends it."""

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path, disposition="bounce")

    async def pending_ack(
        owner_, message_id, target, receipt, *, config_dir, access, **_seam_kwargs
    ):
        record["acks"].append(target)
        return SimpleNamespace(state="pending", intent_id="i-1")

    monkeypatch.setattr(wake_mod, "ack_read", pending_ack)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out == {"action": "hold", "message_id": "bmsg_1", "code": "ack_not_confirmed:pending"}
    assert record["acks"] == ["blocked"] and record["delivered"] == []


def test_the_recovery_ack_runs_behind_the_producer_fence(tmp_path, monkeypatch) -> None:
    """PR #3063 review m8: the m4 replay sends an ACK after two network
    round trips; a bridge turned off during them must stop it, exactly as
    FENCE #2 stops the ordinary ACK."""

    clock, store, owner, record, wire, access, screens, model, pending = (
        _stop_during_the_read_ack_post(monkeypatch, tmp_path)
    )
    real_fetch = wake_mod.fetch_item

    async def fetch_then_off(*args: Any, **kwargs: Any) -> Any:
        item = await real_fetch(*args, **kwargs)
        (tmp_path / "config.yml").write_text(
            json.dumps({"values": {"team_bridge": {"enabled": False}}}), encoding="utf-8"
        )
        return item

    monkeypatch.setattr(wake_mod, "fetch_item", fetch_then_off)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=access
        )
    )
    assert out["action"] == "hold" and out["code"] == "bridge_off"
    assert wire.refs == [] and record["delivered"] == []
    intent = store.ack_intent(owner, pending["intent_id"])
    assert intent is not None and intent["state"] == "pending"


def test_a_session_claim_is_never_parked_for_redelivery(tmp_path, monkeypatch) -> None:
    """A session's own ``ack working`` claim proves the message already reached
    a session: the ref is consumed as before, with no parked retry."""

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    store.claim(owner, "bmsg_1", "sess_cos", str(uuid.uuid4()), expiry=time.time() + 300)
    _one_pending_ref(store, owner)
    out = _iterate(owner, tmp_path)
    assert [o["action"] for o in out["handled"]] == ["claimed_elsewhere"]
    assert store.screening_hold(owner, "bmsg_1") is None
    assert record["delivered"] == []


#: The connector entry point run in a REAL child process, parked mid-spine:
#: the claim is taken, then screening blocks forever. Only the transport and
#: screening seams are faked; the signal path, the loop, the claim and the
#: machine lock are the production ones.
_MID_SPINE_CHILD = r"""
import asyncio, sys
from pathlib import Path
from types import SimpleNamespace
from local_operator.team_bridge import pull as pull_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.connector import __main__ as main_mod
from local_operator.team_bridge.connector import wake as wake_mod

config_dir, marker = Path(sys.argv[1]), Path(sys.argv[2])
polls = []

async def events(owner, *, config_dir, access, after=None, wait=None, client=None):
    from local_operator.team_bridge.store import Store
    from local_operator.team_bridge.types import feed_seq
    polls.append(1)
    if len(polls) == 1:
        refs = ["message_available:message_id=bmsg_1"]
        Store(config_dir).record_feed_batch(owner, feed_seq(3), refs, resync_required=False)
        return {"cursor": 3, "refs": refs, "resync": False}
    await asyncio.sleep(3600)

async def access(self):
    return SimpleNamespace(credential_id=1)

async def message(owner, message_id, *, config_dir, access, client=None):
    return SimpleNamespace(message_id=message_id)

async def fetch(owner, message_id, *, config_dir, access, auth_store):
    return {"message": {"id": message_id}}

async def screen(owner, item, *, config_dir, auth_store, settings=None):
    marker.write_text("in-spine")  # the claim is held from here on
    await asyncio.sleep(3600)

pull_mod.events = events
pull_mod.message = message
TeamBridgeClient.current_access = access
wake_mod.fetch_item = fetch
wake_mod.screen_item = screen
wake_mod.engagement_hold = lambda designation, config_dir: ""
sys.exit(main_mod.main(["--config-dir", str(config_dir)]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX stop signals")
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_stop_signal_mid_spine_releases_the_claim_and_exits_zero(tmp_path, signame) -> None:
    """R2-3 part 1: launchd's SIGTERM (bootout, logout, reboot) used to kill
    the connector with the default action, so the spine's ``finally:
    release_claim`` never ran and the claim stayed held for the whole lease.
    A stop signal must now release it and exit 0 within the bounded window."""

    import signal as signal_mod

    from local_operator.team_bridge.connector import __main__ as main_mod

    store, owner = _enroll(tmp_path)
    _enable_bridge(tmp_path)
    marker = tmp_path / "in-spine"
    script = tmp_path / "child.py"
    script.write_text(_MID_SPINE_CHILD, encoding="utf-8")
    root = str(Path(__file__).resolve().parents[3])
    child = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(marker)],
        cwd=root,
        env={"PYTHONPATH": root, "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert child.poll() is None, child.communicate()[1].decode()[-2000:]
            assert time.monotonic() < deadline, "child never reached the spine"
            time.sleep(0.05)
        held = store.claim_status(owner, "bmsg_1")
        assert held is not None and held["state"] == "held" and held["connector"]
        sent = time.monotonic()
        child.send_signal(getattr(signal_mod, signame))
        code = child.wait(timeout=main_mod.STOP_GRACE_S + main_mod.STOP_CANCEL_S + 10)
        elapsed = time.monotonic() - sent
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    _, stderr = child.communicate()
    assert code == 0, stderr.decode()[-2000:]
    assert elapsed < main_mod.STOP_GRACE_S + main_mod.STOP_CANCEL_S
    released = store.claim_status(owner, "bmsg_1")
    assert released is not None and released["state"] == "released"
    # The machine lock went with it: a successor connector can start at once.
    with ConnectorLock(tmp_path):
        pass


def test_spine_body_hash_mismatch_reholds(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path, body="real body")

    # Screen a DIFFERENT body than the one the fetch returns → S-20 refusal.
    async def lying_screen(owner, item, *, config_dir, auth_store, settings=None):
        return ScreenResult(disposition="deliver", receipt=_receipt("other body"))

    monkeypatch.setattr(wake_mod, "screen_item", lying_screen)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out["action"] == "hold" and out["code"] == "body_hash_mismatch"
    assert record["delivered"] == []


# -- payload bounds + refs parsing -------------------------------------------


def test_payload_bounds_and_untrusted_label() -> None:
    big = "x" * 5000
    item = _item(big)
    item["sender"]["label"] = "line1\nline2"
    payload = wake_mod.build_wake_payload(item, message_id="bmsg_1", body_sha256="a" * 64)
    assert "line1 line2" in payload and "line1\nline2" not in payload
    excerpt = payload.split("screened):\n", 1)[1].split("\nbody_sha256:", 1)[0]
    assert len(excerpt) == wake_mod.EXCERPT_MAX_CHARS
    short = wake_mod.build_wake_payload(_item("tiny"), message_id="m", body_sha256="b" * 64)
    assert "tiny" in short


def test_ref_parser_is_strict() -> None:
    assert wake_mod.parse_event_ref("message_available:message_id=bmsg_1") == (
        "message_available",
        "message_id",
        "bmsg_1",
    )
    assert wake_mod.parse_event_ref("message_available:bmsg_1") is None
    assert wake_mod.parse_event_ref("a:b=c:d") is None
    assert wake_mod.parse_event_ref(":b=c") is None


# -- H4d: designation authority ----------------------------------------------


def test_designation_accessor_is_fenced_and_pref_is_not_authority(tmp_path) -> None:
    store, owner = _enroll(tmp_path, session="sess_cos")
    assert store.designated_session_for(owner) == "sess_cos"
    # A generic settings preference is NOT wake authority: an owner at a NEW
    # generation with no commit reads None even while surface keys (settings.py
    # SESSION_PATH) may hold a value.
    bumped = OwnerSnapshot(**{**owner.__dict__, "enrollment_generation": 2})
    assert store.designated_session_for(bumped) is None
    # Empty designation commits as None (badge-only consent).
    store2, _ = _enroll(tmp_path / "empty", session=None)
    assert store2.designated_session_for(OWNER) is None


def test_commit_consent_and_designation_seam(tmp_path) -> None:
    from local_operator.team_bridge.connector import enrollment as enroll_mod

    store = Store(tmp_path)
    enroll_mod.commit_consent_and_designation(
        OWNER,
        OwnerReceipt("bridge_consent_v1", "ref-1"),
        cos_session_id="sess_cos",
        config_dir=tmp_path,
    )
    assert store.designated_session_for(OWNER) == "sess_cos"
    with pytest.raises(StoreRefused):
        enroll_mod.commit_consent_and_designation(
            OWNER, OwnerReceipt("", ""), cos_session_id="sess", config_dir=tmp_path
        )
    with pytest.raises(StoreRefused):
        # Same generation: the monotonic fence refuses a rollback.
        enroll_mod.commit_consent_and_designation(
            OWNER,
            OwnerReceipt("bridge_consent_v1", "ref-2"),
            cos_session_id="sess2",
            config_dir=tmp_path,
        )


def test_new_message_type_registered() -> None:
    from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE

    assert TEAM_BRIDGE_MESSAGE_TYPE == "team_bridge_message"
    assert TEAM_BRIDGE_MESSAGE_TYPE != "peer_message"


# -- H4-A2: due-hold retry with bounded backoff -------------------------------


def test_due_hold_is_redriven_to_deliver(tmp_path, monkeypatch) -> None:
    """A DUE quarantine re-enters the SAME spine and delivers on success."""

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path, disposition="deliver")
    store.persist_hold(owner, "bmsg_h", "screening_hold", retry_at=time.time() - 5)
    result = asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert result["due"] == 1
    assert result["outcomes"] == [{"message_id": "bmsg_h", "action": "delivered"}]
    assert record["delivered"] and record["acks"] == ["read"]
    # Delivered: the backoff counter resets (no stale growth next time).
    assert "bmsg_h" not in loop_mod._RETRY_ATTEMPTS


def test_not_due_hold_is_skipped(tmp_path, monkeypatch) -> None:
    """A future `retry_at` is local metadata, not a reason to spend."""

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path)
    store.persist_hold(owner, "bmsg_future", "screening_hold", retry_at=time.time() + 3600)
    result = asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert result == {"due": 0, "outcomes": []}
    assert record["screened"] == 0  # nothing re-read, nothing re-screened


def test_repeat_failure_advances_bounded_backoff(tmp_path, monkeypatch) -> None:
    """Held again → the gap doubles per failure and stops at the cap."""

    store, owner = _enroll(tmp_path)
    _wire_spine(monkeypatch, tmp_path, disposition="hold")
    store.persist_hold(owner, "bmsg_f", "screening_hold", retry_at=time.time() - 5)
    first = asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert first["outcomes"][0]["action"] == "hold"
    row_one = store.screening_hold(owner, "bmsg_f")
    assert row_one is not None
    gap_one = row_one["retry_at"] - time.time()
    assert 0 < gap_one <= loop_mod.RETRY_BASE_S + 5
    # Make it due again; the second failure must ADVANCE the gap (bounded 2x).
    store.advance_hold_retry(owner, "bmsg_f", retry_at=time.time() - 1)
    asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    row_two = store.screening_hold(owner, "bmsg_f")
    assert row_two is not None
    gap_two = row_two["retry_at"] - time.time()
    assert gap_two > gap_one
    # At the attempt cap the gap can never exceed RETRY_MAX_S.
    loop_mod._RETRY_ATTEMPTS["bmsg_f"] = loop_mod.RETRY_MAX_ATTEMPTS
    store.advance_hold_retry(owner, "bmsg_f", retry_at=time.time() - 1)
    asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    row_three = store.screening_hold(owner, "bmsg_f")
    assert row_three is not None
    assert row_three["retry_at"] - time.time() <= loop_mod.RETRY_MAX_S + 5
    loop_mod._RETRY_ATTEMPTS.pop("bmsg_f", None)


def test_budget_exhaustion_hold_uses_the_same_due_path(tmp_path, monkeypatch) -> None:
    """Early (pre-attempt) holds retry identically — same due mechanism."""

    store, owner = _enroll(tmp_path)
    record, _ = _wire_spine(monkeypatch, tmp_path, disposition="deliver")
    store.persist_hold(owner, "bmsg_b", "budget_exhausted", retry_at=time.time() - 1)
    result = asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert result["outcomes"] == [{"message_id": "bmsg_b", "action": "delivered"}]
    assert record["screened"] == 1


def test_cancel_during_retry_keeps_claim_releasable(tmp_path, monkeypatch) -> None:
    """Cancellation propagates and the spine's finally releases the claim."""

    store, owner = _enroll(tmp_path)
    started = asyncio.Event()

    async def slow_screen(owner_, item, *, config_dir, auth_store, settings=None):
        started.set()
        await asyncio.sleep(30)
        raise AssertionError("never reached")

    async def fake_fetch(owner_, message_id, *, config_dir, access, auth_store):
        return _item("hello")

    async def fake_message(owner_, message_id, *, config_dir, access, client=None):
        from types import SimpleNamespace

        return SimpleNamespace(message_id=message_id)

    monkeypatch.setattr(pull_mod, "message", fake_message)
    monkeypatch.setattr(wake_mod, "fetch_item", fake_fetch)
    monkeypatch.setattr(wake_mod, "engagement_hold", lambda designation, config_dir: "")
    monkeypatch.setattr(wake_mod, "screen_item", slow_screen)
    # The spine sits BEHIND the fence, so this case must enable the feature —
    # and it BOUNDS the wait: a refusal here used to hang the suite instead of
    # failing, which is how the missing config hid for a round.
    _enable_bridge(tmp_path)
    store.persist_hold(owner, "bmsg_c", "screening_hold", retry_at=time.time() - 5)

    async def run() -> None:
        task = asyncio.ensure_future(
            loop_mod.retry_due_holds(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    # The claim was released on the cancellation path: a FRESH token can take
    # the same message without waiting out the TTL.
    store.claim(owner, "bmsg_c", "sess_cos", f"connector:{uuid.uuid4()}", expiry=time.time() + 60)


# -- H4-A2 minor: typed delivery selection (no string inspection) -------------


def test_deliver_wake_selects_branch_by_typed_history_predicate(tmp_path, monkeypatch) -> None:
    """History predicate picks live-vs-cold; both call typed entry points."""

    from types import SimpleNamespace

    from local_operator.mobile import peer_send as peer_mod
    from local_operator.session.runtime import launch as launch_mod

    owner = SimpleNamespace(account_id="acc_x")
    calls: dict[str, Any] = {}

    async def fake_outcome(
        record, *, session_id, text, mode, wake, sender, custom_type=None, bridge_meta=None
    ):
        # The double mirrors the PRODUCTION signature: the live wake carries the
        # producer's locator block (H4e/F3), and a double that omits it would
        # hide a hop that stopped carrying it.
        calls["live"] = {
            "session_id": session_id,
            "wake": wake,
            "text": text,
            "custom_type": custom_type,
            "bridge_meta": bridge_meta,
        }
        return SimpleNamespace(detail="delivered (live)", state="delivered")

    async def fake_engage(session_id, cwd, work, *, config_dir):
        calls["cold"] = {"session_id": session_id, "work": type(work).__name__, "text": work.text}
        calls["cold_errand"] = work
        return SimpleNamespace(state="started")

    monkeypatch.setattr(peer_mod, "session_has_durable_history", lambda sid: True)
    monkeypatch.setattr(peer_mod, "deliver_peer_message_outcome", fake_outcome)
    out_live = asyncio.run(
        wake_mod.deliver_wake(owner, "payload-1", session_id="sess_cos", config_dir=tmp_path)
    )
    assert out_live == "delivered (live)" and calls["live"]["wake"] is True
    # H4e-render: the bridge provenance kind rides the REAL call — the receive
    # side stamps it on the row (§13.4's own card), never a generic peer kind.
    assert calls["live"]["custom_type"] == "team_bridge_message"

    monkeypatch.setattr(peer_mod, "session_has_durable_history", lambda sid: False)
    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    out_cold = asyncio.run(
        wake_mod.deliver_wake(owner, "payload-2", session_id="sess_cos", config_dir=tmp_path)
    )
    assert "started" in out_cold
    assert calls["cold"] == {
        "session_id": "sess_cos",
        "work": "PeerMessageErrand",
        "text": "payload-2",
    }
    assert calls["cold_errand"].custom_type == "team_bridge_message"


# -- Live E2E findings (local-operator-ui #1314 QA): the connector as shipped ---
# never delivered to an agent. Each case below failed on main before its fix.


def test_first_poll_after_enrolment_sends_cursor_zero(tmp_path, monkeypatch) -> None:
    """F2: a fresh enrolment's stored cursor is 0, and 0 is a POSITION.

    Omitting it reads as "no cursor", which the hub answers with the resync
    arm at the CURRENT head — so the first message ever delivered to a fresh
    endpoint was skipped and never woke the agent.
    """

    store, owner = _enroll(tmp_path)
    calls: list[Any] = []

    async def fake_events(owner_, *, config_dir, access, after=None, wait=None, client=None):
        calls.append({"after": after, "wait": wait})
        return None

    monkeypatch.setattr(pull_mod, "events", fake_events)
    asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert calls[0]["after"] == 0 and calls[0]["after"] is not None
    # F1: every connector read is the long poll, never a plain poll.
    assert calls[0]["wait"] == loop_mod.EVENTS_WAIT_S


@pytest.mark.parametrize("reissued", [False, True], ids=["instant-200", "instant-204"])
def test_run_forever_never_hot_loops_on_instant_answers(tmp_path, monkeypatch, reissued) -> None:
    """F1: a hub that answers every read at once (resync arms, a busy feed, a
    proxy that strips the park) must not turn the loop into a request storm.
    Measured on main: 124 reads in under 2 s, then 429 rate_limited. The 204
    arm is review R1-1: a proxy stripping ``wait`` makes every caught-up poll
    an INSTANT 204, which the bare 0.2 s reissue delay let through at
    ~300/min against the hub's 120/min."""

    store, owner = _enroll(tmp_path)
    polls: list[float] = []
    stop = asyncio.Event()

    async def instant_iteration(owner_, **kwargs):
        polls.append(time.monotonic())
        # The fake ends the loop itself after a storm-sized count: the unfixed
        # loop never yields to the event loop, so a timer-driven stop could
        # never fire and the test would spin without bound.
        if len(polls) >= 50:
            stop.set()
        return {"events": 0, "reissued": reissued, "resync": not reissued, "handled": []}

    async def access_provider():
        return _access(owner)

    monkeypatch.setattr(loop_mod, "run_iteration", instant_iteration)

    async def drive() -> None:
        task = asyncio.create_task(
            loop_mod.run_forever(
                owner,
                config_dir=tmp_path,
                auth_store=None,
                access_provider=access_provider,
                stop=stop,
            )
        )
        await asyncio.sleep(1.5)
        stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(drive())
    assert 1 <= len(polls) <= 3, len(polls)


def test_failure_delay_honours_retry_after_and_stays_bounded() -> None:
    """F1: a 429 names its own wait; re-polling sooner only extends the
    refusal. The server's value wins over a shorter local backoff, is never
    shorter than the backoff, and is capped so a hostile header cannot park
    the connector for hours."""

    from local_operator.team_bridge.client import BridgeRefusal

    limited = BridgeRefusal("rate_limited", "slow down", http_status=429, retry_after_seconds=42)
    assert loop_mod._failure_delay(limited, 1.0) == 42.0
    assert loop_mod._failure_delay(limited, 60.0) == 60.0
    huge = BridgeRefusal("rate_limited", "x", http_status=429, retry_after_seconds=10**9)
    assert loop_mod._failure_delay(huge, 1.0) == loop_mod.RETRY_AFTER_MAX_S
    assert loop_mod._failure_delay(RuntimeError("boom"), 4.0) == 4.0


def test_screen_item_hands_screening_the_production_status_source(tmp_path, monkeypatch) -> None:
    """F3: without a status source every screen holds ``policy_stale`` before
    any model call. The connector must pass the module's OWN sealed fetcher
    (the H1 client's status read over the owner's pinned row), not none."""

    from local_operator.team_bridge import scheduler as scheduler_mod

    store, owner = _enroll(tmp_path)
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_default_status_fetcher(*, config_dir, owner, transport=None, auth_store=None):
        captured["factory"] = (config_dir, owner)
        return sentinel

    async def fake_cycle(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(scheduler_mod, "default_status_fetcher", fake_default_status_fetcher)
    monkeypatch.setattr(scheduler_mod, "run_screening_cycle", fake_cycle)
    asyncio.run(wake_mod.screen_item(owner, _item(), config_dir=tmp_path, auth_store=None))
    assert captured["status_fetcher"] is sentinel
    assert captured["factory"] == (tmp_path, owner)


def test_connector_entry_point_builds_the_config_rooted_credential_store(
    tmp_path, monkeypatch
) -> None:
    """F4: the entry point started with ``auth_store=None``, so the screener
    found no provider key and held every message as ``provider_error``."""

    from local_operator.providers.auth_store import AuthStore
    from local_operator.team_bridge.connector import __main__ as main_mod

    _enroll(tmp_path)
    captured: dict[str, Any] = {}

    async def fake_run_forever(owner, *, config_dir, auth_store, access_provider, **kwargs):
        captured["auth_store"] = auth_store
        captured["config_dir"] = config_dir

    monkeypatch.setattr(loop_mod, "run_forever", fake_run_forever)
    assert main_mod.main(["--config-dir", str(tmp_path)]) == 0
    auth_store = captured["auth_store"]
    assert isinstance(auth_store, AuthStore)
    # The SAME root as the enrollment: config_dir alone does not relocate the
    # database, so the explicit db_path is what makes this the right store.
    assert auth_store._db_path == tmp_path / "auth.db"  # noqa: SLF001 — the root is the contract


def test_bare_conversation_hold_never_reaches_the_pair_spine(tmp_path, monkeypatch) -> None:
    """F5: screening records a ``provider_error`` hold under the bare
    ``bcmsg_`` id. The retry handed it to the PAIR spine, which asked the hub
    for a pair message that does not exist and raised on every iteration —
    BEFORE the poll, so the whole connector stalled. The conversation's own
    encoded-ref row is the re-drive authority; the bare row is retired."""

    from local_operator.team_bridge.types import ConversationFeedRef

    store, owner = _enroll(tmp_path)
    ref = ConversationFeedRef(
        conversation_id="bcv_abcdefgh",
        message_id="bcmsg_abcdef01",
        conversation_seq=1,
        mention=False,
    ).validated()
    store.persist_hold(owner, "bcmsg_abcdef01", "provider_error", retry_at=time.time() - 5)
    store.persist_hold(owner, ref.encode(), "provider_error", retry_at=time.time() - 5)
    pair_calls: list[str] = []
    conversation_calls: list[str] = []

    async def pair_spine(owner_, message_id, **kwargs):
        pair_calls.append(message_id)
        from local_operator.team_bridge.client import BridgeRefusal

        raise BridgeRefusal("message_not_found", "This message does not exist.", http_status=404)

    async def conversation_spine(owner_, conversation_ref, **kwargs):
        conversation_calls.append(conversation_ref.message_id)
        return {"action": "delivered", "message_id": conversation_ref.message_id}

    monkeypatch.setattr(loop_mod, "handle_message_available", pair_spine)
    monkeypatch.setattr(loop_mod, "handle_conversation_available", conversation_spine)
    result = asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert pair_calls == []
    assert conversation_calls == ["bcmsg_abcdef01"]
    assert store.screening_hold(owner, "bcmsg_abcdef01") is None
    assert {o["message_id"]: o["action"] for o in result["outcomes"]}["bcmsg_abcdef01"] == (
        "retired"
    )


def test_a_failing_redrive_never_escapes_the_retry_pass(tmp_path, monkeypatch) -> None:
    """F5 (bound): one hold's re-drive raising must not abort the iteration
    before its poll. A message the hub says is gone (404) is terminal and its
    row is retired; any other failure is held again on the bounded backoff."""

    from local_operator.team_bridge.client import BridgeRefusal

    store, owner = _enroll(tmp_path)
    store.persist_hold(owner, "bmsg_gone", "provider_error", retry_at=time.time() - 5)
    store.persist_hold(owner, "bmsg_flaky", "provider_error", retry_at=time.time() - 5)

    async def pair_spine(owner_, message_id, **kwargs):
        if message_id == "bmsg_gone":
            raise BridgeRefusal(
                "message_not_found", "This message does not exist.", http_status=404
            )
        raise RuntimeError("transport hiccup")

    monkeypatch.setattr(loop_mod, "handle_message_available", pair_spine)
    loop_mod._RETRY_ATTEMPTS.clear()
    result = asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    actions = {o["message_id"]: o["action"] for o in result["outcomes"]}
    assert actions == {"bmsg_gone": "retired", "bmsg_flaky": "hold"}
    assert store.screening_hold(owner, "bmsg_gone") is None
    flaky = store.screening_hold(owner, "bmsg_flaky")
    assert flaky is not None and flaky["retry_at"] > time.time()
    loop_mod._RETRY_ATTEMPTS.clear()


# -- Live E2E F8: a designated chief of staff its owner has never messaged ------


def _f8_enroll(tmp_path: Path, consent_version: str, *, enabled: bool = True) -> OwnerSnapshot:
    """A committed enrollment with the given ACCEPTED consent and a designated
    session that has no transcript at all (never engaged)."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(tmp_path)
    store.commit_enrollment(
        OWNER, OwnerReceipt(consent_version, "ref-1"), cos_session_id="sess_cos"
    )
    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": enabled}}}), encoding="utf-8"
    )
    return OWNER


def _engage_transcript(tmp_path: Path, session_id: str = "sess_cos") -> None:
    """Give the session one real user turn on disk (what the owner's first message writes)."""

    from local_operator.session.runtime.engagement import TRANSCRIPT_FILENAME

    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    row = {"type": "message", "payload": {"kind": "message", "role": "user", "content": "hi"}}
    (directory / TRANSCRIPT_FILENAME).write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "version,engaged,expected",
    [
        ("bridge_consent_v1", False, "open_chief_of_staff"),  # v1 + never engaged: hold
        ("bridge_consent_v1", True, ""),  # v1 + engaged: the F1-F6 path
        ("bridge_consent_v2", False, ""),  # v2 + never engaged: may engage
        ("bridge_consent_v3", False, ""),  # v3 repeats v2's engagement disclosure
        ("bridge_consent_v4", False, "open_chief_of_staff"),  # unknown: never inferred
        ("bridge_consent_v10", False, "open_chief_of_staff"),  # not "v2 or later" by order
    ],
)
def test_engagement_hold_is_gated_on_the_accepted_consent(
    tmp_path, version, engaged, expected
) -> None:
    owner = _f8_enroll(tmp_path, version)
    if engaged:
        _engage_transcript(tmp_path)
    assert Store(tmp_path).designated_session_for(owner) == "sess_cos"
    assert wake_mod.engagement_hold("sess_cos", tmp_path) == expected


def test_engagement_never_trusts_a_disabled_feature_or_another_session(tmp_path) -> None:
    from local_operator.team_bridge.engagement import designated_session_may_be_engaged

    _f8_enroll(tmp_path, "bridge_consent_v2")
    assert designated_session_may_be_engaged("sess_cos", config_dir=tmp_path) is True
    # Not the designated session: never an exception.
    assert designated_session_may_be_engaged("sess_other", config_dir=tmp_path) is False
    # No store at all (another machine root): refused, and no store is created.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert designated_session_may_be_engaged("sess_cos", config_dir=empty) is False
    assert not (empty / "team_bridge").exists()
    # Feature switched off on disk: refused.
    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": False}}}), encoding="utf-8"
    )
    assert designated_session_may_be_engaged("sess_cos", config_dir=tmp_path) is False


def test_v1_unengaged_conversation_message_holds_visibly_and_never_wakes(
    tmp_path, monkeypatch
) -> None:
    """F8, v1: the card is held under the visible code before ANY spend — no
    body fetch, no screen, no admit, no wake — and the loop keeps a bounded
    retry row so it is delivered once the owner sends the session a message."""

    from local_operator.team_bridge.connector import conversation as conversation_mod
    from local_operator.team_bridge.types import ConversationFeedRef

    owner = _f8_enroll(tmp_path, "bridge_consent_v1")
    calls: list[str] = []

    async def must_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("no spend before the engagement gate")

    monkeypatch.setattr(conversation_mod, "_wire_for", must_not_run)
    monkeypatch.setattr(conversation_mod, "screen_item", must_not_run)
    monkeypatch.setattr(conversation_mod, "deliver_wake", must_not_run)
    ref = ConversationFeedRef(
        conversation_id="bcv_abcdefgh",
        message_id="bcmsg_abcdef01",
        conversation_seq=1,
        mention=False,
        sender_kind="human",
    ).validated()
    out = asyncio.run(
        conversation_mod.handle_conversation_available(
            owner, ref, config_dir=tmp_path, auth_store=None, access=_access(owner)
        )
    )
    assert out == {"action": "hold", "message_id": "bcmsg_abcdef01", "code": "open_chief_of_staff"}
    assert calls == []
    # Through the loop: a durable, future-dated retry row under the encoded ref.
    loop_mod._persist_hold_backoff(Store(tmp_path), owner, ref.encode(), out["code"])
    row = Store(tmp_path).screening_hold(owner, ref.encode())
    assert row is not None and row["hold_code"] == "open_chief_of_staff"
    assert row["retry_at"] > time.time()
    # The owner's surface shows the specific reason, not ``unknown_hold``.
    from local_operator.team_bridge.read_context import _screening_state

    assert _screening_state(row)["hold_code"] == "open_chief_of_staff"
    loop_mod._RETRY_ATTEMPTS.clear()


def test_open_chief_of_staff_hold_is_counted_for_the_owner_surface(tmp_path) -> None:
    """F8: the held count the desktop status read shows the owner (code + count only)."""

    from local_operator.team_bridge.types import ConversationFeedRef

    owner = _f8_enroll(tmp_path, "bridge_consent_v1")
    store = Store(tmp_path)
    ref = ConversationFeedRef(
        conversation_id="bcv_abcdefgh",
        message_id="bcmsg_abcdef01",
        conversation_seq=1,
        mention=False,
    ).validated()
    assert store.hold_count_readonly(owner, "open_chief_of_staff") == 0
    store.persist_hold(owner, ref.encode(), "open_chief_of_staff", retry_at=time.time() + 60)
    # A bare-id duplicate is never double-counted.
    store.persist_hold(owner, "bcmsg_abcdef01", "open_chief_of_staff", retry_at=time.time() + 60)
    assert store.hold_count_readonly(owner, "open_chief_of_staff") == 1


def test_a_rehold_for_a_new_reason_replaces_the_visible_code(tmp_path, monkeypatch) -> None:
    """F8: once the owner messages the session, a later hold for another reason
    must replace ``open_chief_of_staff`` — the visible code may not keep asking
    them to do what they already did."""

    owner = _f8_enroll(tmp_path, "bridge_consent_v1")
    store = Store(tmp_path)
    store.persist_hold(owner, "bmsg_x", "open_chief_of_staff", retry_at=time.time() - 5)

    async def pair_spine(owner_, message_id, **kwargs):
        return {"action": "hold", "message_id": message_id, "code": "policy_stale"}

    monkeypatch.setattr(loop_mod, "handle_message_available", pair_spine)
    loop_mod._RETRY_ATTEMPTS.clear()
    asyncio.run(
        loop_mod.retry_due_holds(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    row = store.screening_hold(owner, "bmsg_x")
    assert row is not None and row["hold_code"] == "policy_stale" and row["retry_at"] > time.time()
    loop_mod._RETRY_ATTEMPTS.clear()


def test_seed_starts_a_used_endpoint_at_head_and_a_fresh_one_at_zero(tmp_path) -> None:
    """Review R1-2 without reopening F2. The seed is the hub's head when THIS
    machine commits: a brand-new endpoint's head is 0 (its first message, seq 1,
    still arrives); a previously used endpoint's head is N (no 7-day replay).
    The seed never moves a cursor a poll already advanced."""

    fresh = _f8_enroll(tmp_path / "fresh", "bridge_consent_v1")
    store = Store(tmp_path / "fresh")
    assert store.seed_feed_cursor(fresh, feed_seq(0)) is True
    assert store.sync_state(fresh)["feed_seq"] == 0  # F2: after=0 still sent

    used_root = tmp_path / "used"
    used = _f8_enroll(used_root, "bridge_consent_v1")
    used_store = Store(used_root)
    assert used_store.seed_feed_cursor(used, feed_seq(412)) is True
    assert used_store.sync_state(used)["feed_seq"] == 412

    # A poll already advanced the cursor: a late seed is refused.
    polled_root = tmp_path / "polled"
    polled = _f8_enroll(polled_root, "bridge_consent_v1")
    polled_store = Store(polled_root)
    polled_store.record_feed_batch(polled, feed_seq(3), ["ref-a"], resync_required=False)
    assert polled_store.seed_feed_cursor(polled, feed_seq(900)) is False
    assert polled_store.sync_state(polled)["feed_seq"] == 3


def test_pair_holds_count_in_the_owner_surface(tmp_path) -> None:
    """Review R1-3: a PAIR message is held under its bare pair id, and must
    count toward ``delivery_hold`` like a conversation hold does."""

    owner = _f8_enroll(tmp_path, "bridge_consent_v1")
    store = Store(tmp_path)
    store.persist_hold(owner, "bmsg_pair01", "open_chief_of_staff", retry_at=time.time() + 60)
    assert store.hold_count_readonly(owner, "open_chief_of_staff") == 1


def test_a_held_message_is_redriven_on_the_next_pass_once_the_owner_messages_the_cos(
    tmp_path, monkeypatch
) -> None:
    """Review of local-operator-ui #1325: an ``open_chief_of_staff`` hold whose
    reason is gone (the owner sent their chief of staff a message) must be
    re-driven on the NEXT pass, not after the rest of its doubling backoff (up
    to an hour). Every other hold code keeps its backoff."""

    owner = _f8_enroll(tmp_path, "bridge_consent_v1")
    store = Store(tmp_path)
    later = time.time() + 3000
    store.persist_hold(owner, "bmsg_cos001", "open_chief_of_staff", retry_at=later)
    store.persist_hold(owner, "bmsg_other1", "provider_error", retry_at=later)
    driven: list[str] = []

    async def pair_spine(owner_, message_id, **kwargs):
        driven.append(message_id)
        store.delete_hold(owner_, message_id)
        return {"action": "woken", "message_id": message_id}

    monkeypatch.setattr(loop_mod, "handle_message_available", pair_spine)
    loop_mod._RETRY_ATTEMPTS.clear()

    def one_pass() -> None:
        asyncio.run(
            loop_mod.retry_due_holds(
                owner, config_dir=tmp_path, access=_access(owner), auth_store=None
            )
        )

    # Not yet messaged: the hold keeps waiting (no re-drive, backoff intact).
    one_pass()
    assert driven == []
    waiting = store.screening_hold(owner, "bmsg_cos001")
    assert waiting is not None and waiting["retry_at"] == later

    # The owner sends one message: the very next pass re-drives it.
    _engage_transcript(tmp_path)
    one_pass()
    assert driven == ["bmsg_cos001"]
    # The unrelated hold is still on its own backoff.
    other = store.screening_hold(owner, "bmsg_other1")
    assert other is not None and other["retry_at"] == later
    loop_mod._RETRY_ATTEMPTS.clear()


# -- PR #3063 round 6: the stop/restart recovery MATRIX ------------------------
#
# Six rounds each fixed one probe and opened the next defect, because every
# probe was a hand-built rig. This matrix replaces that: ONE driver, the REAL
# ``pull.ack`` (intent ledger, stored ``client_ref``, replay), the REAL
# screening cycle and fingerprint cache (only the model reply is scripted; no
# helper writes a row production would not write), and a hub that answers
# like the server's ack state machine. Every cell crosses claim leases and
# asserts totals at the END.


class _MatrixHub:
    """The hub's ack endpoint, in one of two MODES.

    ``server`` mirrors agent-server ``bridge_ack_service.go`` (read at
    ``782aefeb``): the counterpart's standing is re-checked on EVERY ack,
    repeats included (a revoked sender answers 403 ``recipient_unavailable``);
    ``unread -> read`` records the verdict write-once; a same-state repeat is
    a 200 no-op when its verdict AGREES with the record (or carries none) and
    a 409 ``invalid_transition`` when it disagrees. ``client_ref`` is "carried
    for nothing" there: idempotency is the state machine and the verdict.

    ``strict_ref`` is the older, STRICTER fake: it additionally 409s a repeat
    whose ``client_ref`` differs from the first one, even with an agreeing
    verdict. Production never does this; it is kept so a cell can show its
    outcome does not depend on the hub tolerating a fresh ref.

    ``stop_in_post`` / ``apply_then_lose`` model the two halves of a stop
    during the POST: the hub never saw it, or it APPLIED the read and the
    response was lost (QA/review r7 ``pending_applied``)."""

    base_url = OWNER.canonical_base_url

    def __init__(self, mode: str = "server") -> None:
        assert mode in ("server", "strict_ref")
        self.mode = mode
        self.reads: dict[str, tuple[Any, ...]] = {}
        self.refs: dict[str, str] = {}
        self.applied: list[str] = []
        self.conflicts = 0
        self.revoked = False
        self.stop_in_post = False
        self.apply_then_lose = False

    @staticmethod
    def _tuple(screening: Any) -> tuple[Any, ...]:
        get = (
            screening.get if isinstance(screening, dict) else (lambda key: getattr(screening, key))
        )
        return (
            get("verdict"),
            get("model"),
            get("prompt_version"),
            get("template_sha256"),
            get("body_sha256"),
            tuple(sorted(get("categories") or ())),
        )

    async def ack(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        from local_operator.team_bridge.client import BridgeRefusal

        if self.stop_in_post:
            # The stop lands DURING the POST: the intent is durable, the hub
            # never saw the request.
            self.stop_in_post = False
            raise asyncio.CancelledError
        if self.revoked:
            raise BridgeRefusal("recipient_unavailable", "counterpart", http_status=403)
        attached = self._tuple(kwargs["screening"])
        ref = str(kwargs.get("client_ref") or "")
        stored = self.reads.get(message_id)
        if stored is None:
            self.reads[message_id] = attached
            self.refs[message_id] = ref
            self.applied.append(message_id)
        elif stored != attached or (self.mode == "strict_ref" and ref != self.refs[message_id]):
            self.conflicts += 1
            raise BridgeRefusal("invalid_transition", "disagreeing repeat", http_status=409)
        if self.apply_then_lose:
            # APPLIED, then the response is lost to a stop.
            self.apply_then_lose = False
            raise asyncio.CancelledError
        return {"delivery": {"state": kwargs["state"], "state_at": "t"}}


_MATRIX_SOURCES = ("fresh", "cache_hit")
_MATRIX_INTENTS = ("none", "pending", "pending_applied", "confirmed")
_MATRIX_CHANGES = ("none", "model", "relogin", "status_withdrawn", "sender_revoked")
#: Round 7 (review r7 M8/M9): compound changes and the one transition a real
#: member makes after the fact — renaming their account, which changes the
#: served ``sender.name`` (the hub serves it from the account).
_MATRIX_COMPOUND = ("rename", "rename+model", "relogin+revoked", "rename+revoked")
_MATRIX_HUB_MODES = ("server", "strict_ref")
_MATRIX_STOPS = ("cancel", "kill")
#: Changes the hub or the owner's policy REFUSES: the message must sit in a
#: visible hold while the refusal stands, and clear once it is lifted.
_REFUSING = ("status_withdrawn", "sender_revoked", "relogin+revoked", "rename+revoked")


def _matrix_config(tmp_path: Path, model: str) -> None:
    (tmp_path / "config.yml").write_text(
        json.dumps(
            {"values": {"hosting": "openai", "model_name": model, "team_bridge": {"enabled": True}}}
        ),
        encoding="utf-8",
    )


def _expected_model_calls(source: str, intent: str, change: str) -> int:
    """One screen before the stop (``bmsg_1`` fresh, or ``bmsg_0`` that the
    cache then serves). A stop at ANY point after that screen leaves a recorded
    decision — the message's own receipt (stopped before the ACK), or its read
    intent — and the restart hands off on it with NO second model call, even
    under a changed model. Two cases re-screen, by design: a re-login (a new
    enrollment generation is a new owner, whose decision must be its own; the
    cache is generation-fenced too), and a CACHE-SERVED message stopped before
    its ACK under a model change — the cache writes no row for it and the hub
    was never told, so there is no decision to carry and re-screening under
    the new model is harmless."""

    # A re-login re-screens only where the hub holds no verdict yet (r7 M8):
    # a recorded intent, pending or confirmed, is the hub's write-once
    # decision and is carried, never re-made.
    relogin = "relogin" in change and intent == "none"
    # The fingerprint cache keys on the FULL envelope, display name included
    # (A6), so a rename misses it exactly like a model change does; with no
    # record and nothing at the hub, that re-screen is the only honest path.
    model_miss = (
        source == "cache_hit" and intent == "none" and ("model" in change or "rename" in change)
    )
    return 1 + (1 if relogin or model_miss else 0)


def _run_recovery_cell(
    tmp_path, monkeypatch, source, intent, change, stop, hub_mode="server"
) -> dict[str, Any]:
    from local_operator.team_bridge import model_boundary as boundary_mod

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    _matrix_config(tmp_path, "gpt-5.4")
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    hub = _MatrixHub(hub_mode)
    monkeypatch.setattr(pull_mod, "_client_for", lambda config_dir, client: hub)
    monkeypatch.setattr(wake_mod, "ack_read", _PRODUCTION_ACK_READ)
    monkeypatch.setattr(wake_mod, "screen_item", _PRODUCTION_SCREEN_ITEM)
    status = _hub_status(monkeypatch, owner)
    sender = {"name": "Org Member"}
    access = OAuthAccess(
        access_token="tok",
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        org_id=owner.tenant_id,
    )

    async def fetch(owner_, message_id, *, config_dir, access, auth_store):
        item = _item(body)
        item["message"]["id"] = message_id
        item["sender"]["name"] = sender["name"]  # the hub serves it from the account
        return item

    monkeypatch.setattr(wake_mod, "fetch_item", fetch)
    model_calls: list[str] = []

    def scripted_runner(**kwargs: Any) -> Any:
        # ONLY the model is scripted; the cycle, cache, ledger and receipts
        # are the production ones.
        def runner(prompt: str, *, default_model: str, **_: Any) -> tuple[str, str]:
            model_calls.append(default_model)
            return "VERDICT: deliver\nREASON: ordinary.", default_model

        return runner

    monkeypatch.setattr(boundary_mod, "make_screening_runner", scripted_runner)

    def drive(current: OwnerSnapshot) -> None:
        for _ in range(4):  # across more than one 15-minute claim lease
            try:
                _iterate(current, tmp_path, access)
            except Exception:  # noqa: BLE001 - run_forever's backoff absorbs it
                pass
            clock.now += _ACROSS_A_LEASE_S

    if source == "cache_hit":
        # Another message with the same bytes and envelope screens first; the
        # REAL cache then serves ``bmsg_1`` with no row of its own.
        _one_pending_ref(store, owner, "bmsg_0")
        _iterate(owner, tmp_path, access)
        assert len(record["delivered"]) == 1 and len(model_calls) == 1
        record["delivered"].clear()

    delivered_fake = wake_mod.deliver_wake
    if intent == "none":

        async def stopped_before_the_ack(*args: Any, **kwargs: Any) -> Any:
            raise asyncio.CancelledError

        monkeypatch.setattr(wake_mod, "ack_read", stopped_before_the_ack)
    elif intent == "pending":
        hub.stop_in_post = True
    elif intent == "pending_applied":
        hub.apply_then_lose = True
    else:

        async def stopped_at_the_handoff(*args: Any, **kwargs: Any) -> str:
            raise asyncio.CancelledError

        monkeypatch.setattr(wake_mod, "deliver_wake", stopped_at_the_handoff)
    store.record_feed_batch(
        owner, feed_seq(4), ["message_available:message_id=bmsg_1"], resync_required=False
    )
    if stop == "cancel":
        with pytest.raises(asyncio.CancelledError):
            _iterate(owner, tmp_path, access)
    else:
        # SIGKILL: no ``finally`` runs, so the claim stays held by a dead
        # owner until its lease expires; the durable ref is still there.
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                wake_mod.handle_message_available(
                    owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=access
                )
            )
        store.claim(
            owner, "bmsg_1", "sess_cos", "connector:killed", expiry=clock.now + wake_mod.CLAIM_TTL_S
        )
    monkeypatch.setattr(wake_mod, "ack_read", _PRODUCTION_ACK_READ)
    monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    recorded = store.recorded_intent(owner, "bmsg_1", "read")
    expected_state = {
        "none": None,
        "pending": "pending",
        "pending_applied": "pending",
        "confirmed": "confirmed",
    }[intent]
    assert (recorded or {}).get("state") == expected_state
    assert hub.applied.count("bmsg_1") == (1 if intent in ("pending_applied", "confirmed") else 0)

    # -- while the device is down (a compound change applies each part) ----
    for part in change.split("+"):
        owner = _apply_matrix_change(part, store, owner, status, hub, sender, tmp_path)

    out: dict[str, Any] = {}
    drive(owner)
    if change in _REFUSING:
        out["refused_deliveries"] = len(record["delivered"])
        hold = store.screening_hold(owner, "bmsg_1")
        out["refused_hold"] = hold["hold_code"] if hold else None
        status["available"] = True
        hub.revoked = False
        drive(owner)
    hold = store.screening_hold(owner, "bmsg_1")
    out.update(
        deliveries=len(record["delivered"]),
        hub_reads=hub.applied.count("bmsg_1"),
        conflicts=hub.conflicts,
        model_calls=len(model_calls),
        hold=hold["hold_code"] if hold else None,
        refs=len(store.drain_feed_refs(owner, 10)),
    )
    return out


def _apply_matrix_change(part, store, owner, status, hub, sender, tmp_path) -> OwnerSnapshot:
    """One change while the device is down; returns the (possibly new) owner."""

    if part == "none":
        return owner
    if part == "rename":
        sender["name"] = "Org Member (renamed)"
        return owner
    if part == "model":
        _matrix_config(tmp_path, "gpt-5.4-mini")  # a ``lop`` update moved the default
        return owner
    if part == "relogin":
        owner = OwnerSnapshot(
            credential_id=owner.credential_id,
            account_id=owner.account_id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            canonical_base_url=owner.canonical_base_url,
            enrollment_generation=store.invalidate(owner),
        )
        store.record_feed_batch(
            owner, feed_seq(9), ["message_available:message_id=bmsg_1"], resync_required=False
        )
        return owner
    if part == "status_withdrawn":
        status["available"] = False
        return owner
    if part == "revoked" or part == "sender_revoked":
        hub.revoked = True
        return owner
    raise AssertionError(part)


def _matrix_cases() -> list[Any]:
    """The base cross-product on the server-faithful hub, plus the r7 rows:
    the compound changes on every source/intent, and the decisive M8 cells
    re-run on the stricter ``strict_ref`` hub."""

    cases = [
        (source, intent, change, stop, "server")
        for source in _MATRIX_SOURCES
        for intent in _MATRIX_INTENTS
        for change in _MATRIX_CHANGES + _MATRIX_COMPOUND
        for stop in _MATRIX_STOPS
    ]
    cases += [
        (source, intent, change, stop, "strict_ref")
        for source in _MATRIX_SOURCES
        for intent in ("pending", "pending_applied", "confirmed")
        for change in ("none", "model", "rename", "rename+model")
        for stop in _MATRIX_STOPS
    ]
    return [pytest.param(*case, id="-".join(case)) for case in cases]


@pytest.mark.parametrize("source,intent,change,stop,hub_mode", _matrix_cases())
def test_stop_restart_recovery_matrix(
    tmp_path, monkeypatch, source, intent, change, stop, hub_mode
) -> None:
    """Every (screening source x intent at stop x change while down x stop
    kind) cell ends with exactly ONE delivery, ONE read applied at the hub,
    no conflicting ACK, no leftover hold or ref, and only the model calls the
    change genuinely requires; a refusing change first holds VISIBLY with zero
    deliveries, then clears once lifted."""

    out = _run_recovery_cell(tmp_path, monkeypatch, source, intent, change, stop, hub_mode)
    # One greppable line per cell: the remediation's before/after table.
    cell = f"{source}/{intent}/{change}/{stop}/{hub_mode}"
    print(f"MATRIX {cell} {json.dumps(out, sort_keys=True)}")
    if change in _REFUSING:
        assert (out["refused_deliveries"], out["refused_hold"] is not None) == (0, True), out
    assert out["deliveries"] == 1, out
    assert out["hub_reads"] == 1 and out["conflicts"] == 0, out
    assert out["hold"] is None and out["refs"] == 0, out
    assert out["model_calls"] == _expected_model_calls(source, intent, change), out


def _request_message_seam(monkeypatch: Any) -> None:
    """``pull.message`` answering a PENDING request (the ask-carrying kind)."""

    async def request_message(owner_, message_id, *, config_dir, access, client=None):
        return SimpleNamespace(
            message_id=message_id,
            message_type="request",
            request_state="pending",
            thread_id="bth_1",
            request_expires_at="",
            request_ref="req-1",
            sender_account_id="acc_sender",
            sender_endpoint_id="ep_sender",
            recipient_account_id=OWNER.account_id,
            recipient_endpoint_id=OWNER.endpoint_id,
            tenant_id=OWNER.tenant_id,
        )

    monkeypatch.setattr(pull_mod, "message", request_message)


@pytest.mark.parametrize("stop", _MATRIX_STOPS)
def test_recovery_matrix_request_across_a_policy_republish(tmp_path, monkeypatch, stop) -> None:
    """QA r4 O3, as a matrix cell. A REQUEST has a confirmed verdict and the
    connector dies in the handoff. While it is down the default model changes
    and another message screens first, publishing the new policy, so the
    request's proof is no longer current and its envelope (which grants the
    human an ask) refuses ``stale_proof``. That is a VISIBLE hold, with no
    delivery. Once the original policy is published again (the model change
    is rolled back and another message screens under it), the hold must clear
    and the request deliver exactly once, on its recorded verdict, with one
    read at the hub and no extra model call for it. Totals at the END."""

    from local_operator.team_bridge import model_boundary as boundary_mod

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    _request_message_seam(monkeypatch)
    _matrix_config(tmp_path, "gpt-5.4")
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    hub = _MatrixHub()
    monkeypatch.setattr(pull_mod, "_client_for", lambda config_dir, client: hub)
    monkeypatch.setattr(wake_mod, "ack_read", _PRODUCTION_ACK_READ)
    monkeypatch.setattr(wake_mod, "screen_item", _PRODUCTION_SCREEN_ITEM)
    _hub_status(monkeypatch, owner)
    access = OAuthAccess(
        access_token="tok",
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        org_id=owner.tenant_id,
    )

    def request_item(message_id: str, text: str) -> dict[str, Any]:
        item = _item(text)
        item["message"].update(
            id=message_id, type="request", request={"kind": "approval", "ref": "req-1"}
        )
        return item

    async def fetch(owner_, message_id, *, config_dir, access, auth_store):
        return request_item(message_id, body)

    monkeypatch.setattr(wake_mod, "fetch_item", fetch)
    model_calls: list[str] = []

    def scripted_runner(**kwargs: Any) -> Any:
        def runner(prompt: str, *, default_model: str, **_: Any) -> tuple[str, str]:
            model_calls.append(default_model)
            return "VERDICT: deliver\nREASON: ordinary.", default_model

        return runner

    monkeypatch.setattr(boundary_mod, "make_screening_runner", scripted_runner)

    def publish_by_screening(message_id: str) -> None:
        # The REAL publisher: another message's screen runs the cycle, which
        # publishes the configured policy before it screens.
        asyncio.run(
            _PRODUCTION_SCREEN_ITEM(
                owner,
                request_item(message_id, f"other {message_id}"),
                config_dir=tmp_path,
                auth_store=None,
            )
        )

    async def stopped_at_the_handoff(*args: Any, **kwargs: Any) -> str:
        raise asyncio.CancelledError

    delivered_fake = wake_mod.deliver_wake
    monkeypatch.setattr(wake_mod, "deliver_wake", stopped_at_the_handoff)
    _one_pending_ref(store, owner)
    if stop == "cancel":
        with pytest.raises(asyncio.CancelledError):
            _iterate(owner, tmp_path, access)
    else:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                wake_mod.handle_message_available(
                    owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=access
                )
            )
        store.claim(
            owner, "bmsg_1", "sess_cos", "connector:killed", expiry=clock.now + wake_mod.CLAIM_TTL_S
        )
    monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    recorded = store.recorded_intent(owner, "bmsg_1", "read")
    assert recorded is not None and recorded["state"] == "confirmed"
    request_calls = len(model_calls)

    _matrix_config(tmp_path, "gpt-5.4-mini")  # an update changes the default model
    publish_by_screening("bmsg_8")
    for _ in range(4):
        _iterate(owner, tmp_path, access)
        clock.now += _ACROSS_A_LEASE_S
    refused = len(record["delivered"])
    held = store.screening_hold(owner, "bmsg_1")

    _matrix_config(tmp_path, "gpt-5.4")  # rolled back
    publish_by_screening("bmsg_9")
    for _ in range(4):
        _iterate(owner, tmp_path, access)
        clock.now += _ACROSS_A_LEASE_S
    out = {
        "refused_deliveries": refused,
        "refused_hold": held["hold_code"] if held else None,
        "deliveries": len(record["delivered"]),
        "hub_reads": hub.applied.count("bmsg_1"),
        "conflicts": hub.conflicts,
        "request_model_calls": request_calls,
        "hold": (store.screening_hold(owner, "bmsg_1") or {}).get("hold_code"),
    }
    print(f"MATRIX request/confirmed/policy_republished/{stop} {json.dumps(out, sort_keys=True)}")
    assert out["refused_deliveries"] == 0 and out["refused_hold"] is not None, out
    assert out["deliveries"] == 1 and out["hub_reads"] == 1 and out["conflicts"] == 0, out
    assert out["request_model_calls"] == 1 and out["hold"] is None, out


def _revocation_rig(tmp_path: Path, monkeypatch: Any) -> tuple[Any, ...]:
    """The matrix's production pieces (real cycle, cache, ledger, ``pull.ack``,
    loop) on the server-faithful hub, with the model reply scripted per case."""

    from local_operator.team_bridge import model_boundary as boundary_mod

    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    loop_mod._RETRY_ATTEMPTS.clear()
    store, owner = _enroll(tmp_path)
    record, _body = _wire_spine(monkeypatch, tmp_path)
    _matrix_config(tmp_path, "gpt-5.4")
    monkeypatch.setattr(pull_mod, "events", _quiet_events)
    hub = _MatrixHub("server")
    monkeypatch.setattr(pull_mod, "_client_for", lambda config_dir, client: hub)
    monkeypatch.setattr(wake_mod, "ack_read", _PRODUCTION_ACK_READ)
    monkeypatch.setattr(wake_mod, "screen_item", _PRODUCTION_SCREEN_ITEM)
    _hub_status(monkeypatch, owner)
    reply = {"text": "VERDICT: deliver\nREASON: ordinary."}
    model_calls: list[str] = []

    def scripted_runner(**kwargs: Any) -> Any:
        def runner(prompt: str, *, default_model: str, **_: Any) -> tuple[str, str]:
            model_calls.append(default_model)
            return reply["text"], default_model

        return runner

    monkeypatch.setattr(boundary_mod, "make_screening_runner", scripted_runner)
    access = OAuthAccess(
        access_token="tok",
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        org_id=owner.tenant_id,
    )
    return clock, store, owner, record, hub, access, reply, model_calls


def _passes_under_revocation(owner, tmp_path, access, clock) -> list[Any]:
    """Three loop passes, each past the retry backoff; a pass that RAISES is
    recorded, not swallowed — that is the whole-loop failure M10 measured."""

    outcomes: list[Any] = []
    for _ in range(3):
        try:
            out = _iterate(owner, tmp_path, access)
            outcomes.append([h.get("action") for h in out["handled"]])
        except Exception as failure:  # noqa: BLE001 - the regression under test
            outcomes.append(("raised", getattr(failure, "code", type(failure).__name__)))
        clock.now += _ACROSS_A_LEASE_S
    return outcomes


def test_a_confirmed_blocked_ack_retires_under_a_revoked_sender(tmp_path, monkeypatch) -> None:
    """PR #3063 review r8 M10: the hub CONFIRMED a ``blocked`` ACK, the
    connector stopped before consuming the feed ref, and the owner revoked
    the sender while it was down. Re-driving the rejected message re-POSTed
    the terminal ACK, the hub answered 403 ``recipient_unavailable`` (it
    checks standing before even a same-state repeat), and the refusal escaped
    the spine: the ref stayed, no hold was written, and EVERY later pass
    raised into the connector's whole-loop backoff. A confirmed blocked ACK
    is terminal and hands nothing off; it must retire without asking."""

    clock, store, owner, record, hub, access, reply, _calls = _revocation_rig(tmp_path, monkeypatch)
    reply["text"] = "VERDICT: bounce\nREASON: prompt_injection attempt."
    _one_pending_ref(store, owner)
    first = _iterate(owner, tmp_path, access)
    assert [h.get("action") for h in first["handled"]] == ["bounce"], first
    assert (store.recorded_intent(owner, "bmsg_1", "blocked") or {}).get("state") == "confirmed"
    posts_before = len(hub.applied)

    # The crash window: the hub has the blocked state, the ref was not consumed.
    store.record_feed_batch(
        owner, feed_seq(7), ["message_available:message_id=bmsg_1"], resync_required=False
    )
    hub.revoked = True
    outcomes = _passes_under_revocation(owner, tmp_path, access, clock)

    assert outcomes == [["bounce"], [], []], outcomes  # terminal once, never a raise
    assert store.drain_feed_refs(owner, 10) == []
    assert store.screening_hold(owner, "bmsg_1") is None
    assert record["delivered"] == []
    assert len(hub.applied) == posts_before  # the terminal ACK was not re-sent


def test_a_confirmed_read_ack_under_a_revoked_sender_holds_then_recovers(
    tmp_path, monkeypatch
) -> None:
    """The other side of M10: a confirmed READ is about to be handed off, so
    the standing re-check stays. While the sender is revoked the message is a
    visible hold with no hand-off; once standing is restored it is delivered
    exactly once, on its recorded verdict (no second model call)."""

    clock, store, owner, record, hub, access, _reply, model_calls = _revocation_rig(
        tmp_path, monkeypatch
    )
    delivered_fake = wake_mod.deliver_wake

    async def stopped_at_the_handoff(*args: Any, **kwargs: Any) -> str:
        raise asyncio.CancelledError

    monkeypatch.setattr(wake_mod, "deliver_wake", stopped_at_the_handoff)
    _one_pending_ref(store, owner)
    with pytest.raises(asyncio.CancelledError):
        _iterate(owner, tmp_path, access)
    monkeypatch.setattr(wake_mod, "deliver_wake", delivered_fake)
    assert (store.recorded_intent(owner, "bmsg_1", "read") or {}).get("state") == "confirmed"

    hub.revoked = True
    outcomes = _passes_under_revocation(owner, tmp_path, access, clock)
    assert all(not isinstance(o, tuple) for o in outcomes), outcomes  # no pass raised
    assert record["delivered"] == []
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] == "ack_refused:recipient_unavailable"

    hub.revoked = False
    _passes_under_revocation(owner, tmp_path, access, clock)
    assert len(record["delivered"]) == 1
    assert store.screening_hold(owner, "bmsg_1") is None
    assert len(model_calls) == 1 and hub.applied.count("bmsg_1") == 1
