"""Stub Radient hub over REAL HTTP for H-B QA (core #4238 / UI #1977).

The product's own TeamBridgeClient talks to this over loopback via
RADIENT_API_BASE_URL + RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1, so the client's
URL building, params, bearer attachment and HTTP-status -> reason mapping are
all real. Every request is logged (bearer redacted to presence only).
The classification fetch (/messages/{id}/body) answers 418 and logs BOMB.
Answers are keyed by before_seq (card seq + 1), scripted in SCRIPT.
"""
import hashlib, json, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

PORT = int(sys.argv[1]); LOG = sys.argv[2]
CID = "bcv_abcdefgh"
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
lock = threading.Lock()


def row(mid, seq, body=None, **kw):
    r = {"id": mid, "conversation_id": CID, "seq": seq, "body": body,
         "body_hash": H(body) if body is not None else "x" * 64, "sender": {"kind": "human"}}
    r.update(kw); return r


HUB_BODY = "Hub-only message from **Ola**: the `v0.108` notes are in the board."
LOCAL_BODY = "**Hello** from Ola - can you check the _release_ notes before Friday?"
COAL = [f"bcmsg_coal{i:04d}" for i in range(23)]

# before_seq -> (status, json)
SCRIPT = {
    101: (200, {"messages": [row("bcmsg_local001", 100, LOCAL_BODY)], "has_more": False}),
    111: (200, {"messages": [row("bcmsg_hub00001", 110, HUB_BODY)], "has_more": False}),
    121: (200, {"messages": [dict(row("bcmsg_hubbad01", 120, "tampered bytes"), body_hash=H("original bytes"))], "has_more": False}),
    141: (200, {"messages": [row("bcmsg_refu0001", 140, None, body_state={"state": "rejected", "reason": "secret category S7"})], "has_more": False}),
    151: (200, {"messages": [row("bcmsg_nadm0001", 150, None, body_state={"state": "not_admitted"})], "has_more": False}),
    161: (200, {"messages": [row("bcmsg_dele0001", 160, None, deleted_at="2026-11-18T00:00:00Z")], "has_more": False}),
    171: (403, {"error": {"code": "not_member", "message": "qa stub: not a member"}}),
    181: (501, {"error": {"code": "not_implemented", "message": "qa stub"}}),
    191: (500, {"error": {"code": "internal", "message": "qa stub boom"}}),
    323: (200, {"messages": [row(m, 300 + i, f"Coalesced message **{i + 1}** of 23.") for i, m in enumerate(COAL)][-50:], "has_more": False}),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _log(self, status, extra=None):
        u = urlsplit(self.path)
        e = {"method": self.command, "path": u.path, "query": {k: v[0] for k, v in parse_qs(u.query).items()},
             "bearer": bool((self.headers.get("Authorization") or "").startswith("Bearer ")), "status": status}
        if extra: e.update(extra)
        with lock, open(LOG, "a") as f:
            f.write(json.dumps(e) + "\n")

    def _send(self, status, obj):
        b = json.dumps(obj).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlsplit(self.path); q = parse_qs(u.query)
        if re.search(r"/messages/[^/]+/body$", u.path):
            self._log(418, {"BOMB": True}); return self._send(418, {"error": "BOMB classification fetch"})
        m = re.search(r"/messaging/conversations/([^/]+)/messages$", u.path)
        if m:
            bs = int(q.get("before_seq", ["0"])[0])
            status, obj = SCRIPT.get(bs, (200, {"messages": [], "has_more": False}))
            self._log(status, {"before_seq": bs})
            return self._send(status, {"msg": "ok", "result": obj} if status == 200 else obj)
        self._log(404); self._send(404, {"error": "qa stub: unknown route"})

    do_POST = do_PUT = do_DELETE = do_PATCH = lambda self: (self._log(405, {"WRITE": True}), self._send(405, {"error": "qa stub is read-only"}))


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
