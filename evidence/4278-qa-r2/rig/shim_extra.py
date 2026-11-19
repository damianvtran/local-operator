# QA r2 #4278: the body route stays real; only its resolver answers "not written yet" for the
# first QA_LATE[entry] reads per epoch (CardBodyNotFound -> the route's real 404 'no such bridge
# card'), and $S/unk_on makes the route's own session check answer 404 'unknown session'.
import json, os, time
import local_operator.team_bridge.card_body as CB
import local_operator.mobile.daemon as D
_LOG = os.environ["QA_WIRE_LOG"]; _EP = os.environ["QA_EPOCH"]; _UNK = os.environ["QA_UNK"]
LATE = {k: int(v) for k, v in (x.split(":") for x in os.environ.get("QA_LATE", "").split(",") if x)}
COUNT = {}
def _lg(e):
    with open(_LOG, "a") as f: f.write(json.dumps(e) + "\n")
def _epoch():
    try: return open(_EP).read().strip()
    except FileNotFoundError: return ""
_orig = CB.resolve_card_body
async def resolve(root, sid, entry, **kw):
    ep = _epoch(); k = (ep, entry); COUNT[k] = COUNT.get(k, 0) + 1
    late = COUNT[k] <= LATE.get(entry, 0)
    _lg({"call": "resolve", "entry": entry, "epoch": ep, "n": COUNT[k], "late": late, "t": time.time()})
    if late: raise CB.CardBodyNotFound(entry)
    return await _orig(root, sid, entry, **kw)
CB.resolve_card_body = resolve
_orig_dir = D._durable_user_session_dir
def durable(session_id, **kw):
    if os.path.exists(_UNK):
        _lg({"call": "durable-dir-unknown", "sid": session_id, "t": time.time()}); return None
    return _orig_dir(session_id, **kw)
D._durable_user_session_dir = durable
