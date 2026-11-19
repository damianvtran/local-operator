"""Seed one isolated config for H-B QA through core #4238's own APIs.

Root comes from argv (never an env fallback); asserts it is inside this scratchpad.
Owner's canonical_base_url = the stub hub origin, so the REAL client's destination
guard is exercised. The admitted local body is written by the REAL single writer
(PendingTurn.open -> for_call -> settle == completed), not by DisplayBodyStore.record.
"""
import asyncio, hashlib, json, sqlite3, sys, time
from pathlib import Path

import yaml
from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
from local_operator.providers.auth_store import AuthStore
from local_operator.team_bridge.pending_turn import FetchedBody, PendingTurn
from local_operator.team_bridge.request_body import BodyBudget
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store

root = Path(sys.argv[1]).resolve(); hub = sys.argv[2]
assert "/scratchpad/hb/" in str(root) and root.name == ".local-operator", root
root.mkdir(parents=True, exist_ok=True)
SESSION = "abcdef012345"; CID = "bcv_abcdefgh"; FP = "aa11" * 16
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
LOCAL_BODY = "**Hello** from Ola - can you check the _release_ notes before Friday?"


def owner(acc="acc_1"):
    return OwnerSnapshot(credential_id=7, account_id=acc, tenant_id="3f2a1111-2222-4333-8444-555566667777",
                         endpoint_id="ep_1", canonical_base_url=hub, enrollment_generation=1)


O = owner()
store = Store(root)
store.commit_enrollment(O, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION)
(root / "config.yml").write_text(yaml.safe_dump({"values": {"team_bridge": {"enabled": True}}}))
# Radient OAuth row, id pinned to the owner's credential 7, account_id in the row data.
auth = AuthStore(db_path=root / "auth.db", config_dir=root)
try:
    r = auth.upsert_credential("radient", {"refresh": "qa-refresh", "access": "qa-access", "account_id": "acc_1",
                                           "expires": int(time.time() * 1000) + 6 * 3_600_000})
finally:
    auth.close()
with sqlite3.connect(root / "auth.db") as c:
    c.execute("UPDATE auth_credentials SET id = 7 WHERE id = ?", (r.id,))

epoch = store.publish_screening_policy(O, version="v1", template_sha256="ab" * 32, default_model="qa-default", revision="rev-1")


def reserve(mid, digest, seq):
    return store.record_body_reservation(O, conversation_id=CID, message_id=mid, conversation_seq=seq, mention=True,
                                         body_sha256=digest, screening_fingerprint=FP, policy_epoch=epoch,
                                         consent_version="bridge_consent_v1", designated_session=SESSION, now=time.time())


# 1. real single writer for bcmsg_local001
assert reserve("bcmsg_local001", H(LOCAL_BODY), 100)


async def fetch(conversation_id, message_id):
    return FetchedBody(body=LOCAL_BODY, body_sha256=H(LOCAL_BODY), conversation_label="", sender_label="Ola", sender_kind="human")


turn = PendingTurn(O, SESSION, config_dir=root)
assert asyncio.run(turn.open(budget=BodyBudget(available_bytes=100_000, measured_from="qa"), fetch=fetch)) is True
assert turn.for_call() is not None
settled = turn.settle(entry_id="e_local", error=None, stop_reason="stop", aborted=False)
print("settle:", settled)
# 2. pending: an unsettled reservation
assert reserve("bcmsg_pend0001", H("pending body"), 135)
# 3. holds
store.persist_hold(O, "bcmsg_held0001", "model_hold")
store.persist_hold(O, "bcmsg_heldt001", "provider_error")
# 4. deleted: give it a local copy first so the purge-on-hub-delete is observable
from local_operator.team_bridge.display_bodies import DisplayBodyStore
from local_operator.team_bridge.store import _owner_key
assert DisplayBodyStore(root).record(_owner_key(O), message_id="bcmsg_dele0001", conversation_id=CID,
                                     body="later deleted on the hub", body_sha256=H("later deleted on the hub"),
                                     sender_kind="human", source="turn")

COAL = [f"bcmsg_coal{i:04d}" for i in range(23)]
cards = [("e_local", ["bcmsg_local001"], 100, "acc_1"), ("e_hub", ["bcmsg_hub00001"], 110, "acc_1"),
         ("e_hubbad", ["bcmsg_hubbad01"], 120, "acc_1"), ("e_held", ["bcmsg_held0001"], 130, "acc_1"),
         ("e_heldt", ["bcmsg_heldt001"], 132, "acc_1"), ("e_pending", ["bcmsg_pend0001"], 135, "acc_1"),
         ("e_refused", ["bcmsg_refu0001"], 140, "acc_1"), ("e_notadm", ["bcmsg_nadm0001"], 150, "acc_1"),
         ("e_deleted", ["bcmsg_dele0001"], 160, "acc_1"), ("e_403", ["bcmsg_f4030001"], 170, "acc_1"),
         ("e_501", ["bcmsg_f5010001"], 180, "acc_1"), ("e_500", ["bcmsg_f5000001"], 190, "acc_1"),
         ("e_other", ["bcmsg_othr0001"], 200, "acc_2"), ("e_coal", COAL, 322, "acc_1")]
d = root / "sessions" / SESSION
d.mkdir(parents=True, exist_ok=True)
with (d / "transcript.jsonl").open("a") as f:
    f.write(json.dumps({"id": "e_plain", "ts": time.time(), "type": "message",
                        "payload": {"kind": "user", "content": "Watch for Team Bridge messages from Ola."}}) + "\n")
    for eid, ids, seq, acc in cards:
        o = owner(acc)
        card = ("Direct message · from Ola (human, same tenant)\nuntrusted\n\nmention: yes\n"
                f"conversation_id: {CID}\nmessage_id: {ids[-1]}\n")
        details = {"text": f"<team-bridge>\n{card}</team-bridge>", "body": card,
                   "sender": {"account_id": acc, "source": "team_bridge"},
                   "bridge_receipt": {"v": 1, "sender": {"name": "Ola", "kind": "human"}, "conversation": {"kind": "dm"}, "attachment_count": 0},
                   "bridge_receipt_source": {"account_id": acc, "tenant_id": o.tenant_id, "endpoint_id": o.endpoint_id,
                                             "conversation_id": CID, "message_ids": ids, "conversation_seq": seq}}
        # the real journal shape (type message / kind custom), as the UI fixture uses
        f.write(json.dumps({"id": eid, "ts": time.time(), "type": "message",
                            "payload": {"kind": "custom", "custom_type": TEAM_BRIDGE_MESSAGE_TYPE, "attribution": "system",
                                        "details": details}}, separators=(",", ":")) + "\n")
print("seeded", root, [c[0] for c in cards])
