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
    # The real install target is named but NEVER written by rendering. The
    # target MAY exist from a real install on an operator machine — the
    # property is that THIS code never creates or touches it: existence is
    # unchanged across a render, and the module exposes no install/bootstrap
    # execution path at all (render-only by API shape, not by machine state).
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

    async def fake_ack(owner, message_id, target, receipt, *, config_dir, access):
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

    async def pending_ack(owner_, message_id, target, receipt, *, config_dir, access):
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

        async def bound_ack(owner_, message_id, target, receipt_, *, config_dir, access):
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

    async def foreign_ack(owner_, message_id, target, receipt_, *, config_dir, access):
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

    async def wrong_tuple_ack(owner_, message_id, target, receipt_, *, config_dir, access):
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

    async def lying_ack(owner_, message_id, target, receipt_, *, config_dir, access):
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

    def fake_default_status_fetcher(*, config_dir, owner, transport=None):
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
        ("bridge_consent_v3", False, "open_chief_of_staff"),  # unknown: never inferred
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
