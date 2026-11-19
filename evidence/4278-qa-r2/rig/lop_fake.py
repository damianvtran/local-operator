"""Real `lop` CLI from core #4238; ONLY the hub wire (TeamBridgeClient network methods) is a
logging double, answering per card by before_seq. conversation_body (classification fetch) = BOMB."""
import json, os, sys
from types import SimpleNamespace
from local_operator.team_bridge import client as C

LOG = os.environ["QA_WIRE_LOG"]
SEQ = json.load(open(os.environ["QA_SEQMAP"]))


def _log(e):
    with open(LOG, "a") as f:
        f.write(json.dumps(e, default=str) + "\n")


def row(mid, seq, **kw):
    r = {"id": mid, "conversation_id": "bcv_abcdefgh", "seq": seq, "body": None,
         "body_hash": "x" * 64, "sender": {"kind": "human"}}
    r.update(kw)
    return r


async def current_access(self):
    _log({"call": "current_access"})
    return SimpleNamespace(account_id="acc_1", kind="oauth", credential_id=7)


async def conversation_messages(self, conversation_id, **kw):
    seq = kw.get("before_seq", 0) - 1
    card = next((k for k, v in SEQ.items() if v == seq), None)
    _log({"call": "conversation_messages", "card": card, **{k: v for k, v in kw.items() if k != "access"}})
    if card == "e_offline":
        raise C.BridgeRefusal("bridge_unreachable", "qa fake: hub down")
    if card == "e_refused":
        return {"messages": [row("bcmsg_refu0001", seq, body_state={"state": "rejected", "reason": "secret category S7"})], "has_more": False}
    if card == "e_deleted":
        return {"messages": [row("bcmsg_dele0001", seq, deleted_at="2026-11-18T00:00:00Z")], "has_more": False}
    return {"messages": [], "has_more": False}


async def conversation_body(self, *a, **k):
    _log({"call": "BOMB conversation_body"})
    raise AssertionError("classification fetch reached")


async def guarded_access(self, expected_account_id, *, access=None):
    return SimpleNamespace(account_id="acc_1", kind="oauth", credential_id=7)


C.TeamBridgeClient.base_url = property(lambda self: "https://hub.example")
C.TeamBridgeClient.current_access = current_access
C.TeamBridgeClient.conversation_messages = conversation_messages
C.TeamBridgeClient.conversation_body = conversation_body
C.TeamBridgeClient._guarded_access = guarded_access
# Negative control only: QA_NO_CAP=1 drops the capability from the relay's
# advertised features, so the phone must never call the body route.
if os.environ.get("QA_NO_CAP") == "1":
    import local_operator.mobile.daemon as D
    _orig = D._feature_flags
    def _no_cap():
        f = dict(_orig()); f.pop("team_bridge_card_body", None); return f
    D._feature_flags = _no_cap
exec(open(os.path.join(os.path.dirname(__file__), 'shim_extra.py')).read())
from local_operator.cli import main  # noqa: E402

sys.argv[0] = "lop"
sys.exit(main())
