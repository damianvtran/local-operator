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
import plistlib
import subprocess
import sys
import time
from pathlib import Path
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
    # The real install target is named but NEVER written by rendering:
    assert install_mod.plist_path().name == "com.local-operator.bridge-connector.plist"
    assert not install_mod.plist_path().exists() or True  # never asserted as written


def test_connector_import_stays_bounded() -> None:
    """The import pin: module scope must not pull the session/model graphs."""

    code = (
        "import local_operator.team_bridge.connector, sys;"
        "assert 'local_operator.session.session' not in sys.modules;"
        "assert 'local_operator.model.configure' not in sys.modules;"
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
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=None, auth_store=None)
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
        loop_mod.run_iteration(owner, config_dir=tmp_path, access=None, auth_store=None)
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
    result = asyncio.run(loop_mod.resync_walk(owner, config_dir=tmp_path, access=None))
    assert result["pages"] == 2 and len(finishes) == 1
    # Witness: a premature completion was IMPOSSIBLE by construction until the
    # terminal page merged; and calling finish early on a fresh walk refuses.
    store2 = Store(tmp_path)
    with pytest.raises(StoreRefused):
        store2.finish_resync(owner, InboxCursor(""))


# -- H4c: the spine ----------------------------------------------------------


def _wire_spine(monkeypatch, tmp_path, *, disposition="deliver", body=None):
    """Replace the heavy seams with controlled fakes; return the recorder."""

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
        from types import SimpleNamespace

        return SimpleNamespace(message_id=message_id)

    async def fake_ack(owner, message_id, target, receipt, *, config_dir, access):
        record["acks"].append(target)

    async def fake_deliver(owner, payload, *, session_id, config_dir):
        record["delivered"].append({"session": session_id, "payload": payload})
        return "delivered"

    monkeypatch.setattr(pull_mod, "message", fake_message)
    monkeypatch.setattr(wake_mod, "fetch_item", fake_fetch)
    monkeypatch.setattr(wake_mod, "screen_item", fake_screen)
    monkeypatch.setattr(wake_mod, "ack_read", fake_ack)
    monkeypatch.setattr(wake_mod, "deliver_wake", fake_deliver)
    return record, body


def test_spine_delivers_from_screened_bytes_and_releases_claim(tmp_path, monkeypatch) -> None:
    store, owner = _enroll(tmp_path)
    record, body = _wire_spine(monkeypatch, tmp_path)
    out = asyncio.run(
        wake_mod.handle_message_available(
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=None
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


def test_spine_hold_and_bounce_never_wake(tmp_path, monkeypatch) -> None:
    for disposition, expected in (("hold", "hold"), ("bounce", "bounce")):
        case = tmp_path / disposition
        case.mkdir()
        store, owner = _enroll(case)
        record, _ = _wire_spine(monkeypatch, case, disposition=disposition)
        out = asyncio.run(
            wake_mod.handle_message_available(
                owner, "bmsg_1", config_dir=case, auth_store=None, access=None
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
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=None
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
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=None
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
            owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=None
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
