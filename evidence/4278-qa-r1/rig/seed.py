"""QA r1 core #4278 (phone web relay): seed an isolated config with Team Bridge cards + display_bodies rows
through core #4238's own store APIs (helpers from its test module, as core QA did)."""
import hashlib, json, os, sys, time
from pathlib import Path
WT = "/home/ec2-user/local-operator/.worktrees/bridge-card-phone-web"
sys.path.insert(0, WT)
import tests.unit.team_bridge.test_card_body as T
from local_operator.team_bridge.display_bodies import DisplayBodyStore
from local_operator.team_bridge.store import _owner_key

root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
assert "scratchpad/qa/cfg/" in str(root), root
SHORT = ("**Hello** from Ola — can you check the _release_ notes before Friday?\n\n"
         "- confirm the `v0.108` changelog\n- link: [release board](https://example.com/board)")
LONG = "## Release checklist\n\n" + "\n".join(
    f"{i}. Line {i}: the quick brown fox jumps over the lazy dog, then checks item `{i}`." for i in range(1, 31))
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
store = T._enroll(root)
T._sign_in_radient(root)
owner = T._owner()
key = _owner_key(owner)
disp = DisplayBodyStore(root)


def local(mid, body):
    assert disp.record(key, message_id=mid, conversation_id=T.CONVERSATION_ID, body=body,
                       body_sha256=h(body), sender_kind="human", source="turn"), mid


local("bcmsg_admit001", SHORT)
local("bcmsg_long0001", LONG)
local("bcmsg_dele0001", "this body was later deleted on the hub")
store.persist_hold(owner, "bcmsg_held0001", "model_hold")
store.persist_hold(owner, "bcmsg_heldt001", "provider_error")
coal = [f"bcmsg_coal{i:04d}" for i in range(23)]
for i, mid in enumerate(coal):
    local(mid, f"Coalesced message **{i + 1}** of 23.")
d = root / "sessions" / T.SESSION
d.mkdir(parents=True, exist_ok=True)
with (d / "transcript.jsonl").open("a") as f:
    f.write(json.dumps({"id": "e_plain", "ts": time.time(), "type": "message",
                        "payload": {"kind": "user", "content": "Watch for Team Bridge messages from Ola."}}) + "\n")
cards = [("e_admit", ["bcmsg_admit001"]), ("e_long", ["bcmsg_long0001"]), ("e_held", ["bcmsg_held0001"]),
         ("e_heldt", ["bcmsg_heldt001"]), ("e_refused", ["bcmsg_refu0001"]), ("e_deleted", ["bcmsg_dele0001"]),
         ("e_offline", ["bcmsg_offl0001"]), ("e_coal", coal)]
seqmap = {}
for i, (eid, ids) in enumerate(cards):
    seq = 100 + 10 * i
    seqmap[eid] = seq
    T._write_card(root, T._details(ids, seq=seq), entry_id=eid)
json.dump(seqmap, open(os.environ["QA_SEQMAP"], "w"))
print("seeded", T.SESSION, seqmap)
