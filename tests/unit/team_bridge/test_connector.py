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

    async def fake_events(owner_, *, config_dir, access, after=None, client=None):
        calls.append({"after": after})
        if len(calls) == 1:
            return None  # 204 → reissue
        return {"cursor": 7, "refs": [], "resync": False}

    monkeypatch.setattr(pull_mod, "events", fake_events)
    first = asyncio.run(
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=_access(owner), auth_store=None)
    )
    assert first["reissued"] is True and calls[0]["after"] is None
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

    async def fake_deliver(owner, payload, *, session_id, config_dir, bridge_meta=None):
        record["delivered"].append(
            {"session": session_id, "payload": payload, "bridge_meta": bridge_meta}
        )
        return "delivered"

    monkeypatch.setattr(pull_mod, "message", fake_message)
    monkeypatch.setattr(wake_mod, "fetch_item", fake_fetch)
    monkeypatch.setattr(wake_mod, "screen_item", fake_screen)
    monkeypatch.setattr(wake_mod, "ack_read", fake_ack)
    monkeypatch.setattr(wake_mod, "deliver_wake", fake_deliver)
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

    def _arm(case: Path, *, record_kwargs: dict, record_owner=None):
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
        record_kwargs={"target": "blocked", "receipt": _receipt("hello from the org", verdict="reject")},
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
        _AckStore(config_dir).mark_ack_result(
            owner_, intent_id, state="confirmed", error_code=None
        )
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

    from local_operator.team_bridge.store import Store as _SpineStore, StoreRefused as _Refused

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
        loop_mod.retry_due_holds(
            owner, config_dir=tmp_path, access=_access(owner), auth_store=None
        )
    )
    row_two = store.screening_hold(owner, "bmsg_f")
    assert row_two is not None
    gap_two = row_two["retry_at"] - time.time()
    assert gap_two > gap_one
    # At the attempt cap the gap can never exceed RETRY_MAX_S.
    loop_mod._RETRY_ATTEMPTS["bmsg_f"] = loop_mod.RETRY_MAX_ATTEMPTS
    store.advance_hold_retry(owner, "bmsg_f", retry_at=time.time() - 1)
    asyncio.run(
        loop_mod.retry_due_holds(
            owner, config_dir=tmp_path, access=_access(owner), auth_store=None
        )
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
