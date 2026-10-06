"""Round-1 wire transcript for issue #2016: the SHAPE layer, and the happy path.

Run:
    LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/round1/scripts/mobile_2016_transcript_r1.py" \
      <port> <outdir>

Pure stdlib against the isolated fixture daemon. It pins the round-1 change that
is visible on the wire — malformed identity is refused at the shape layer (422),
exactly as the desktop ``SeenItem`` refuses it — while a well-formed but unknown
pair keeps the per-item ``unknown`` verdict, and the happy path still clears 2 -> 0.

  1. login
  2. GET  /api/attention/unread          the badge BEFORE (count 2, tokens on the wire)
  3. POST /api/attention/seen            "zzz-not-hex" id            -> 422
  4. POST /api/attention/seen            "ABCDEF012345" (uppercase)  -> 422
  5. POST /api/attention/seen            token "not-a-uuid"          -> 422
  6. POST /api/attention/seen            token ""                   -> 422
  7. POST /api/attention/seen            well-formed, unknown id     -> 200 unknown
  8. POST /api/attention/seen            the two rendered pairs      -> 200 read=[2]
  9. GET  /api/attention/unread          AFTER                      -> count 0

Outputs into <outdir>: transcript.log, transcript.json, shape-readings.json.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

SEEDED_IDS = ("c0ffee000016", "c0ffee000017")
DEAD_ID = "deadbeef1234"
BODY_CAP = 40_000


class Wire:
    def __init__(self, port: int) -> None:
        self.port = port
        self.cookie = ""
        self.steps: list[dict] = []

    def call(
        self,
        note: str,
        method: str,
        path: str,
        *,
        body: str | None = None,
        headers: dict | None = None,
    ) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=25)
        hdrs = {"Accept": "application/json"}
        if body is not None:
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        if self.cookie:
            hdrs["Cookie"] = self.cookie
        conn.request(method, path, body=body.encode() if isinstance(body, str) else body, headers=hdrs)
        resp = conn.getresponse()
        status = resp.status
        headers = [list(pair) for pair in resp.getheaders()]
        set_cookie = [
            value.split(";", 1)[0].strip()
            for key, value in resp.getheaders()
            if key.lower() == "set-cookie"
        ]
        raw = resp.read().decode("utf-8", "replace")
        if len(raw) > BODY_CAP:
            raw = raw[:BODY_CAP] + f"\n<… truncated at {BODY_CAP} chars …>"
        resp.close()
        conn.close()
        entry = {
            "note": note,
            "request": f"{method} {path}",
            "status": status,
            "content_type": dict((k.lower(), v) for k, v in headers).get("content-type", ""),
            "set_cookie_names": [p.split("=", 1)[0] for p in set_cookie],
            "request_body": "<password redacted>" if note == "login" else (str(body) if body else None),
            "body": "<password redacted>" if note == "login" else raw,
        }
        self.steps.append(entry)
        if note == "login" and set_cookie:
            self.cookie = "; ".join(set_cookie)
        return entry


def main() -> None:
    port = int(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    password = os.environ.get("LOP_MOBILE_FIXTURE_PASSWORD", "")
    if not password:
        raise SystemExit("set LOP_MOBILE_FIXTURE_PASSWORD to the fixture's per-run password")

    w = Wire(port)
    w.call(
        "login",
        "POST",
        "/login",
        body="password=" + quote(password),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if not w.cookie:
        raise SystemExit("login did not set a cookie — is the fixture up on this port?")

    before = w.call("badge BEFORE (count 2)", "GET", "/api/attention/unread")
    payload = json.loads(before["body"])
    tokens = {row["session_id"]: row.get("completion_token") for row in payload.get("conversations", [])}
    for session_id in SEEDED_IDS:
        if not tokens.get(session_id):
            raise SystemExit(f"the unread read did not enumerate a token for {session_id}")

    def seen(note: str, session_id: str, token: str) -> dict:
        return w.call(
            note,
            "POST",
            "/api/attention/seen",
            body=json.dumps({"items": [{"session_id": session_id, "completion_token": token}]}),
        )

    shape_cases = [
        ("SHAPE: 'zzz-not-hex' session id", "zzz-not-hex", str(uuid.uuid4())),
        ("SHAPE: uppercase hex session id", "ABCDEF012345", str(uuid.uuid4())),
        ("SHAPE: 11-character session id", "aaaaaaaaaaa", str(uuid.uuid4())),
        ("SHAPE: token is not a UUID", "aaaaaaaaaaaa", "not-a-uuid"),
        ("SHAPE: empty token", "aaaaaaaaaaaa", ""),
    ]
    shape = [seen(note, sid, token) for note, sid, token in shape_cases]

    unknown = seen("WELL-FORMED but unknown id -> per-item unknown", DEAD_ID, str(uuid.uuid4()))

    happy = w.call(
        "HAPPY PATH: the two rendered pairs",
        "POST",
        "/api/attention/seen",
        body=json.dumps(
            {
                "items": [
                    {"session_id": sid, "completion_token": tokens[sid]} for sid in SEEDED_IDS
                ]
            }
        ),
    )
    after = w.call("badge AFTER (count 0)", "GET", "/api/attention/unread")

    readings = {
        "count_before": payload.get("count"),
        "count_after": json.loads(after["body"]).get("count"),
        "shape_statuses": {entry["note"]: entry["status"] for entry in shape},
        "shape_error": json.loads(shape[0]["body"]).get("error"),
        "unknown_status": unknown["status"],
        "unknown_body": unknown["body"].strip(),
        "happy_status": happy["status"],
        "happy_body": happy["body"].strip(),
    }

    lines = ["# Raw HTTP transcript — issue #2016 round-1 remediation (shape layer)", ""]
    for index, step in enumerate(w.steps, start=1):
        lines.append(f"=== [{index}] {step['note']} ===")
        lines.append(f"> {step['request']}")
        if step["note"] == "login":
            lines.append("> (request body: password=<redacted>)")
        elif step["request_body"] is not None:
            lines.append(f"> request body: {step['request_body'][:600]}")
        lines.append(f"< {step['status']}  content-type: {step['content_type']}")
        if step["set_cookie_names"]:
            lines.append(f"< set-cookie names: {step['set_cookie_names']} (value redacted)")
        if step["note"] != "login":
            lines.append(f"< body: {step['body'].strip()[:600]}")
        lines.append("")
    lines.append("=== READINGS ===")
    lines.append(json.dumps(readings, indent=2))

    (outdir / "transcript.log").write_text("\n".join(lines) + "\n")
    (outdir / "transcript.json").write_text(json.dumps({"port": port, "steps": w.steps}, indent=2))
    (outdir / "shape-readings.json").write_text(json.dumps(readings, indent=2) + "\n")
    print(json.dumps(readings, indent=2))


if __name__ == "__main__":
    main()
