"""Rewrite the seeded session's journal through core's own writer (Transcript.append_messages),
so the rows are the shape a real session journals (type=message, payload.kind=custom) and the
phone's durable fold replays them. Same ids/details as seed.py (test_card_body._details)."""
import asyncio, json, os, sys
from pathlib import Path
WT = "/home/ec2-user/local-operator/.worktrees/bridge-card-phone-web"
sys.path.insert(0, WT)
import tests.unit.team_bridge.test_card_body as T
from local_operator.harness.types import CustomMessage, Message
from local_operator.session.transcript import Transcript
root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]); assert "scratchpad/qa/cfg/" in str(root), root
d = root / "sessions" / T.SESSION
old = d / "transcript.jsonl"
if old.exists(): old.rename(d / "transcript.seed-v1.jsonl.bak")
seq = json.load(open(os.environ["QA_SEQMAP"]))
coal = [f"bcmsg_coal{i:04d}" for i in range(23)]
cards = [("e_admit", ["bcmsg_admit001"]), ("e_long", ["bcmsg_long0001"]), ("e_held", ["bcmsg_held0001"]),
         ("e_heldt", ["bcmsg_heldt001"]), ("e_refused", ["bcmsg_refu0001"]), ("e_deleted", ["bcmsg_dele0001"]),
         ("e_offline", ["bcmsg_offl0001"]), ("e_coal", coal), ("e_n1", ["bcmsg_n1sent01"]), ("e_n1neg", ["bcmsg_n1neg001"]), ("e_empty", ["bcmsg_empty001"])]
msgs = [Message.model_validate({"role": "user", "content": [{"type": "text", "text": "Watch for Team Bridge messages from Ola."}], "id": "e_plain"})]
for eid, ids in cards:
    det = T._details(ids, seq=seq[eid])
    if eid == "e_empty": det.pop("bridge_receipt_source")  # pre-receipt card: nothing to resolve
    msgs.append(CustomMessage(custom_type="team_bridge_message", attribution="user", details=det, id=eid))
async def main():
    t = Transcript(d)
    await t.append_messages(msgs)
asyncio.run(main())
print([json.loads(l)["type"] + ":" + json.loads(l)["id"] for l in open(old)])
