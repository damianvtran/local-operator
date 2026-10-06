"""Post-fix raw HTTP transcript for issue #2016 against the isolated fixture daemon.

Run:
    LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/postfix/scripts/mobile_2016_transcript.py" \
      <port> <outdir> [seed.json]

Pure stdlib (``http.client``) — it imports no ``local_operator`` module for the
wire half and touches no store directly; the store is read only through the
fixture's own ``/fixture/state`` hook. It logs in with the same per-run password
the fixture was started with and records the calls that pin the FIXED contract:

  1. GET  /api/attention/unread        the badge BEFORE (count 2, tokens on the wire)
  2. GET  /api/sessions                the list frame BEFORE (per-row `unseen`)
  3. POST /api/attention/seen          a MIXED batch: one rendered receipt + one
                                       dead id -> read=[one], unknown=[dead],
                                       still 200 (per-item, never per-call)
  4. GET  /api/attention/unread        AFTER the mixed batch (2 -> 1)
  5. POST /fixture/publish             the fixture settles a NEWER completion for
                                       the second conversation (after the render)
  6. POST /api/attention/seen          the STALE token -> superseded=[that id];
                                       NOT A SWEEP: nothing is cleared
  7. GET  /api/attention/unread        unchanged (still 1)
  8. GET  /fixture/state               the SHARED STORE says the conversation is
                                       still unseen, on the NEWER token
  9. POST /api/attention/seen          the CURRENT token -> read; N -> 0
 10. GET  /api/attention/unread        AFTER (0)
 11. GET  /api/sessions                AFTER (both rows unseen:false)
 12. POST /api/attention/seen          a NO-OP batch (all unknown) is still 200
 13. GET  /fixture/state x2            the shared-store read for both, at rest

Outputs into <outdir>: transcript.log, transcript.json,
attention-unread-{before,after}.json, sessions-{before,after}.json,
store-state.json, publish-result.json.
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
KNOWN_ONE = "c0ffee000016"
STALE_TARGET = "c0ffee000017"
DEAD_ID = "deadbeef1234"
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


def _count(body: str) -> object:
    try:
        return json.loads(body).get("count")
    except ValueError:
        return None


def _json_or_text(body: str) -> object:
    try:
        return json.loads(body)
    except ValueError:
        return body.strip()


def main() -> None:
    port = int(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    # argv[3] (the repro phase's seed.json) is accepted for CLI parity only:
    # the tokens now ride the unread read itself, so nothing reads that file.

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
    before_json = json.loads(before["body"])
    tokens = {
        row["session_id"]: row.get("completion_token")
        for row in before_json.get("conversations", [])
    }
    for sid in SEEDED_IDS:
        if not tokens.get(sid):
            raise SystemExit(f"the unread read did not enumerate a token for {sid}")

    mixed = w.call(
        "bulk clear: one rendered receipt + one DEAD id (per-item verdicts)",
        "POST",
        "/api/attention/seen",
        body=json.dumps(
            {
                "items": [
                    {"session_id": KNOWN_ONE, "completion_token": tokens[KNOWN_ONE]},
                    {"session_id": DEAD_ID, "completion_token": str(uuid.uuid4())},
                ]
            }
        ),
    )
    mid = w.call("badge AFTER the mixed batch", "GET", "/api/attention/unread")

    publish = w.call(
        "fixture publishes a NEWER completion for the stale target (lands after the render)",
        "POST",
        "/fixture/publish",
        body=json.dumps({"session_id": STALE_TARGET}),
    )
    newer_token = json.loads(publish["body"])["completion_token"]

    stale = w.call(
        "bulk clear with the STALE token (NOT A SWEEP: nothing may clear)",
        "POST",
        "/api/attention/seen",
        body=json.dumps(
            {"items": [{"session_id": STALE_TARGET, "completion_token": tokens[STALE_TARGET]}]}
        ),
    )
    after_stale = w.call("badge AFTER the stale attempt (unchanged)", "GET", "/api/attention/unread")
    state_stale = w.call(
        "fixture reads the SHARED STORE for the stale target",
        "GET",
        f"/fixture/state?session_id={STALE_TARGET}",
    )

    finish = w.call(
        "bulk clear with the CURRENT token (N -> 0)",
        "POST",
        "/api/attention/seen",
        body=json.dumps({"items": [{"session_id": STALE_TARGET, "completion_token": newer_token}]}),
    )
    after = w.call("badge AFTER the finish", "GET", "/api/attention/unread")
    sessions_after = w.call("list frame AFTER the finish", "GET", "/api/sessions")

    noop = w.call(
        "no-op batch: an all-unknown call is still 200",
        "POST",
        "/api/attention/seen",
        body=json.dumps(
            {"items": [{"session_id": DEAD_ID, "completion_token": str(uuid.uuid4())}]}
        ),
    )
    state_one = w.call(
        "fixture reads the SHARED STORE for the cleared conversation",
        "GET",
        f"/fixture/state?session_id={KNOWN_ONE}",
    )
    state_stale_rest = w.call(
        "fixture reads the SHARED STORE for the finished conversation",
        "GET",
        f"/fixture/state?session_id={STALE_TARGET}",
    )

    for name, entry in (
        ("attention-unread-before.json", before),
        ("attention-unread-after.json", after),
        ("sessions-before.json", sessions_before),
        ("sessions-after.json", sessions_after),
        ("publish-result.json", publish),
    ):
        (outdir / name).write_text(entry["body"] + "\n")
    (outdir / "store-state.json").write_text(
        json.dumps(
            {
                KNOWN_ONE: _json_or_text(state_one["body"]),
                STALE_TARGET: _json_or_text(state_stale_rest["body"]),
                "stale_attempt_read": _json_or_text(state_stale["body"]),
            },
            indent=2,
        )
        + "\n"
    )

    readings = {
        "counts": {
            "before": _count(before["body"]),
            "after_mixed": _count(mid["body"]),
            "after_stale": _count(after_stale["body"]),
            "after_finish": _count(after["body"]),
        },
        "mixed": {"status": mixed["status"], "body": mixed["body"].strip()},
        "stale": {"status": stale["status"], "body": stale["body"].strip()},
        "finish": {"status": finish["status"], "body": finish["body"].strip()},
        "noop": {"status": noop["status"], "body": noop["body"].strip()},
        "state_after_stale_attempt": _json_or_text(state_stale["body"]),
        "rows_before": _rows_of(sessions_before["body"]),
        "rows_after": _rows_of(sessions_after["body"]),
    }

    log_lines = ["# Raw HTTP transcript — issue #2016, post-fix (fixture daemon, isolated)", ""]
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
        if step["note"] != "login":
            body_text = step["body"].strip()
            log_lines.append(f"< body: {body_text[:1200]}")
        log_lines.append("")
    log_lines.append("=== READINGS ===")
    log_lines.append(json.dumps(readings, indent=2))
    (outdir / "transcript.log").write_text("\n".join(log_lines) + "\n")
    (outdir / "transcript.json").write_text(json.dumps({"port": port, "steps": w.steps}, indent=2))

    print(json.dumps(readings, indent=2))


if __name__ == "__main__":
    main()
