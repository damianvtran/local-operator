"""QA r1 #4278 M1: engage a LIVE idle runtime for the seeded session (WarmErrand), then on
`deliver` deliver ONE inbound Team Bridge message through core's own peer-delivery primitive
(deliver_peer_message_outcome, mode=mailbox wake=True — wake.py's arm) with a display body
recorded first, so the persisted row resolves to `admitted`."""
import asyncio, hashlib, json, os, sys, time
from pathlib import Path
WT = "/home/ec2-user/local-operator/.worktrees/bridge-card-phone-web"
sys.path.insert(0, WT)
root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]); assert "scratchpad/qa/cfg/" in str(root), root
SID = "abcdef012345"

async def warm():
    from local_operator.session.runtime.launch import WarmErrand, engage_runtime
    out = await engage_runtime(SID, str(root), WarmErrand(), config_dir=root, deadline_s=60)
    print("engage", out)

async def deliver(mid):
    import tests.unit.team_bridge.test_card_body as T
    from local_operator.team_bridge.display_bodies import DisplayBodyStore
    from local_operator.team_bridge.store import _owner_key
    from local_operator.mobile.peer_send import deliver_peer_message_outcome
    body = f"**Live** inbound message `{mid}` — delivered to an idle session at {time.strftime('%H:%M:%S')}Z."
    key = _owner_key(T._owner())
    assert DisplayBodyStore(root).record(key, message_id=mid, conversation_id=T.CONVERSATION_ID, body=body,
        body_sha256=hashlib.sha256(body.encode()).hexdigest(), sender_kind="human", source="turn")
    d = T._details([mid], seq=900)
    sender = dict(d["sender"]); sender["bridge_receipt"] = d["bridge_receipt"]; sender["bridge_receipt_source"] = d["bridge_receipt_source"]
    t0 = time.time()
    out = await deliver_peer_message_outcome(None, session_id=SID, text=d["text"], mode="mailbox", wake=True,
        sender=sender, custom_type="team_bridge_message")
    print("deliver", round(time.time() - t0, 3), out)

if sys.argv[1] == "warm": asyncio.run(warm())
else: asyncio.run(deliver(sys.argv[2]))
