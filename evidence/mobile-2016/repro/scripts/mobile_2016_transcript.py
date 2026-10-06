"""Raw HTTP transcript for issue #2016 against the isolated fixture daemon.

Run:
    LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/scripts/mobile_2016_transcript.py" \
      <port> <outdir> [seed.json]

Pure stdlib (``http.client``) — it imports no ``local_operator`` module and
touches no store; the only state it reads is the daemon's own wire. It logs in
with the same per-run password the fixture was started with and records,
verbatim, the seven calls that pin the issue:

 1. GET  /api/attention/unread        the badge BEFORE
 2. GET  /api/sessions                the list frame BEFORE (per-row `unseen`)
 3. GET  /api/sessions/<id>/events    the projection frame the phone reads —
                                      the completion_token the /seen handshake
                                      needs (both seeded conversations)
 4. POST /api/attention/seen          the natural bulk sibling of the single
                                      route — expected ABSENT (404)
 5. POST /api/sessions/<id>/seen      ONE session, the token from step 3 → the
                                      receipt semantics on the wire
 6. GET  /api/attention/unread        AFTER — the count must move by exactly one
 7. GET  /api/sessions                AFTER — the marked row is not unseen, the
                                      other still is

Outputs into <outdir>: transcript.log, transcript.json,
attention-unread-{before,after}.json, sessions-{before,after}.json.

The token fallback: if the SSE read yields no frame, the seed file's token is
used and the log SAYS SO (the receiver still validates it — a wrong or stale
token answers 409, so the receipt semantics stay honest either way).
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

SEEDED_IDS = ("c0ffee000016", "c0ffee000017")
MARK_ONE = "c0ffee000016"
BODY_CAP = 60_000


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
        headers: dict[str, str] | None = None,
        sse: bool = False,
    ) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=25)
        hdrs = {"Accept": "application/json"}
        if body is not None:
            hdrs["Content-Type"] = "application/json"
        if self.cookie:
            hdrs["Cookie"] = self.cookie
        if headers:
            hdrs.update(headers)
        conn.request(method, path, body=body.encode() if isinstance(body, str) else body, headers=hdrs)
        resp = conn.getresponse()
        status = resp.status
        resp_headers = [list(pair) for pair in resp.getheaders()]
        set_cookie_pairs = [
            value.split(";", 1)[0].strip()
            for key, value in resp.getheaders()
            if key.lower() == "set-cookie"
        ]
        raw = ""
        sse_head = ""
        if sse:
            lines: list[str] = []
            deadline = time.time() + 12
            try:
                while time.time() < deadline:
                    line = resp.readline()  # HTTPResponse.readline decodes chunked bodies
                    if not line:
                        break
                    text = line.decode("utf-8", "replace").rstrip("\r\n")
                    lines.append(text)
                    if text.startswith("data:"):
                        raw = text[len("data:") :].strip()
                        break
            except (TimeoutError, OSError) as exc:  # keep the attempt in the record
                lines.append(f"<read failed: {exc!r}>")
            sse_head = "\n".join(lines)
            resp.close()
        else:
            raw = resp.read().decode("utf-8", "replace")
            if len(raw) > BODY_CAP:
                raw = raw[:BODY_CAP] + f"\n<… truncated at {BODY_CAP} chars …>"
            resp.close()
        conn.close()
        recorded_body = raw
        if note == "login":
            recorded_body = "<password redacted>"
        recorded_request = None
        if body is not None:
            recorded_request = "<password redacted>" if note == "login" else str(body)
        entry = {
            "note": note,
            "request": f"{method} {path}",
            "status": status,
            "headers": {k: v for k, v in resp_headers if k.lower() != "set-cookie"},
            "set_cookie_names": [p.split("=", 1)[0] for p in set_cookie_pairs],
            "request_body": recorded_request,
            "body": recorded_body,
        }
        if sse:
            entry["sse_head"] = sse_head
        self.steps.append(entry)
        if note == "login" and set_cookie_pairs:
            self.cookie = "; ".join(set_cookie_pairs)
        return entry


def _rows_of(sessions_body: str) -> dict[str, dict]:
    try:
        payload = json.loads(sessions_body)
    except ValueError:
        return {}
    out = {}
    for row in payload.get("sessions", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("session_id") in SEEDED_IDS:
            out[row["session_id"]] = {
                key: row.get(key)
                for key in (
                    "session_id",
                    "conversation_name",
                    "section",
                    "unseen",
                    "completion_kind",
                    "needs_attention",
                    "streaming",
                )
            }
    return out


def main() -> None:
    port = int(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    seed_tokens: dict[str, str] = {}
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        seed = json.loads(Path(sys.argv[3]).read_text())
        seed_tokens = {item["session_id"]: item["completion_token"] for item in seed.get("seed", [])}

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

    before = w.call("badge BEFORE any mark", "GET", "/api/attention/unread")
    sessions_before = w.call("list frame BEFORE any mark", "GET", "/api/sessions")

    tokens: dict[str, str] = {}
    token_source: dict[str, str] = {}
    for session_id in SEEDED_IDS:
        frame = w.call(
            f"projection frame for {session_id} (the token the phone reads)",
            "GET",
            f"/api/sessions/{session_id}/events",
            sse=True,
            headers={"Accept": "text/event-stream"},
        )
        token = ""
        try:
            token = json.loads(frame["body"]).get("attention", {}).get("completion_token") or ""
        except (ValueError, AttributeError):
            token = ""
        if token:
            token_source[session_id] = "daemon SSE projection frame"
        elif session_id in seed_tokens:
            token = seed_tokens[session_id]
            token_source[session_id] = "seed cross-check (SSE yielded no frame)"
        tokens[session_id] = token

    bulk_body = json.dumps(
        {
            "items": [
                {"session_id": session_id, "completion_token": tokens[session_id]}
                for session_id in SEEDED_IDS
            ]
        }
    )
    bulk = w.call(
        "bulk sibling attempt POST /api/attention/seen (expected: route absent)",
        "POST",
        "/api/attention/seen",
        body=bulk_body,
        headers={"Content-Type": "application/json"},
    )

    seen = w.call(
        f"single-session mark POST /api/sessions/{MARK_ONE}/seen",
        "POST",
        f"/api/sessions/{MARK_ONE}/seen",
        body=json.dumps({"completion_token": tokens[MARK_ONE]}),
        headers={"Content-Type": "application/json"},
    )

    after = w.call("badge AFTER the single mark", "GET", "/api/attention/unread")
    sessions_after = w.call("list frame AFTER the single mark", "GET", "/api/sessions")

    for name, entry in (
        ("attention-unread-before.json", before),
        ("attention-unread-after.json", after),
        ("sessions-before.json", sessions_before),
        ("sessions-after.json", sessions_after),
    ):
        (outdir / name).write_text(entry["body"] + "\n")

    readings = {
        "token_source": token_source,
        "unread_before": before["body"].strip(),
        "unread_after": after["body"].strip(),
        "rows_before": _rows_of(sessions_before["body"]),
        "rows_after": _rows_of(sessions_after["body"]),
        "bulk_status": bulk["status"],
        "bulk_body": bulk["body"].strip(),
        "single_status": seen["status"],
        "single_body": seen["body"].strip(),
    }

    log_lines = ["# Raw HTTP transcript — issue #2016 (fixture daemon, isolated)", ""]
    for index, step in enumerate(w.steps, start=1):
        log_lines.append(f"=== [{index}] {step['note']} ===")
        log_lines.append(f"> {step['request']}")
        if step["note"] == "login":
            log_lines.append("> (request body: password=<redacted>)")
        elif step.get("request_body") is not None:
            log_lines.append(f"> request body: {step['request_body'][:600]}")
        log_lines.append(f"< {step['status']}  content-type: {step['headers'].get('content-type', '')}")
        if step["set_cookie_names"]:
            log_lines.append(f"< set-cookie names: {step['set_cookie_names']} (value redacted)")
        if "sse_head" in step:
            log_lines.append("< sse head:")
            log_lines.extend(f"<   {line}" for line in step["sse_head"].splitlines())
        elif step["note"] != "login":
            body_text = step["body"].strip()
            log_lines.append(f"< body: {body_text[:1200]}")
        log_lines.append("")
    log_lines.append("=== READINGS ===")
    log_lines.append(json.dumps(readings, indent=2))
    (outdir / "transcript.log").write_text("\n".join(log_lines) + "\n")
    (outdir / "transcript.json").write_text(json.dumps({"port": port, "steps": w.steps}, indent=2))

    print(
        json.dumps(
            {
                "token_source": token_source,
                "unread_before": readings["unread_before"],
                "unread_after": readings["unread_after"],
                "bulk_status": readings["bulk_status"],
                "bulk_body": readings["bulk_body"],
                "single_status": readings["single_status"],
                "single_body": readings["single_body"],
                "rows_before": readings["rows_before"],
                "rows_after": readings["rows_after"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
