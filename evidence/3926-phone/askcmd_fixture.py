"""Phone fixture for PR #3926 (ask suggested commands) — design capture only.

Run from the PR worktree root with PYTHONPATH=<worktree>:
    LOP_MOBILE_FIXTURE_PASSWORD=... python askcmd_fixture.py PORT LANG
Reuses the worktree's own fixture helpers (probe_isolation re-homes HOME and the
config dir and strips CMUX_*), serves ONE live, answerable runtime whose asks
carry `command` / `command_target`, loopback only. LANG ('en', 'ur', 'ru') is
written into the sandbox config so the phone resolves that locale.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import scripts.probe_isolation as iso  # noqa: F401 -- must be first

import os

lang = sys.argv[2] if len(sys.argv) > 2 else "en"
cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
cfg.mkdir(parents=True, exist_ok=True)
(cfg / "config.yml").write_text(f"values:\n  language: {lang}\n", encoding="utf-8")

import uvicorn  # noqa: E402

from local_operator.mobile.daemon import MobileDaemon, build_app  # noqa: E402
from local_operator.mobile.types import SessionProjection, TranscriptEntry  # noqa: E402
from scripts.mobile_overflow_fixture import (  # noqa: E402
    _ask,
    _question,
    required_password,
    seed_ask_index,
    serve_queued_ask_runtime,
)

LONG = (
    "curl -fsSL --retry 3 -H 'Accept: application/vnd.github+json' "
    "https://raw.githubusercontent.com/damianvtran/local-operator/refs/heads/feat/"
    "ask-suggested-commands/scripts/very/long/unbroken/path/segment/that/keeps/going/"
    "well/past/the/edge/of/a/phone/screen/without/a/single/space/install-and-verify-the-"
    "release-artifact-checksums.sh | bash -s -- --target ~/.local/share/lop --verify-sha256 --yes"
)


def _q(qid, text, *, options=None, command=None, target="agent", multi=False):
    q = _question(qid, text, options=options, multi=multi)
    if command:
        q["command"] = command
        q["command_target"] = target
    return q


def _asks():
    yes_no = [("yes", "I'll do it now"), ("no", "skip it for now")]
    return [
        _ask("ac-before", created_s=-600, expires_in_s=840,
             questions=[_q("b1", "Sign in to GitLab first?", options=yes_no)]),
        _ask("ac-short", created_s=-500, expires_in_s=840,
             questions=[_q("s1", "Sign in to GitLab first?", options=yes_no,
                           command="glab auth login", target="user")]),
        _ask("ac-long", created_s=-400, expires_in_s=840,
             questions=[_q("l1", "Run the installer check before I continue?", options=yes_no,
                           command=LONG)]),
        _ask("ac-free", created_s=-300, expires_in_s=840,
             questions=[_q("f1", "Paste the token it prints once you have run this:",
                           command="gh auth token --hostname github.com", target="user")]),
        _ask("ac-multi", created_s=-200, expires_in_s=840, questions=[
            _q("m1", "Which branch should I deploy?",
               options=[("main", "the released head"), ("staging", "the QA head")]),
            _q("m2", "Restart the worker after the migration?", options=yes_no,
               command="systemctl --user restart lop-worker.service"),
        ]),
    ]


def _projection() -> SessionProjection:
    p = SessionProjection(
        session_id="askcmd-fixture-0001",
        pid=990001,
        kind="tui",
        conversation_name="Ask commands fixture",
        streaming=False,
        transcript=[
            TranscriptEntry(id="x-1", kind="user", text="Deploy the worker and ask me what you need."),
            TranscriptEntry(id="x-2", kind="assistant", text="I queued a few questions; some come with a command to run."),
        ],
        version=3,
    )
    p.asks = _asks()
    p.asks_open = len(p.asks)
    return p


async def main() -> None:
    port = int(sys.argv[1])
    password = required_password([])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    live = _projection()
    seed_ask_index([live])
    harness, registrant, dial = await serve_queued_ask_runtime(daemon, live)
    app = build_app(daemon)
    import local_operator

    print(f"Fixture serves tree: {Path(local_operator.__file__).resolve().parents[1]}", flush=True)
    print(f"Fixture sandbox: {iso.SANDBOX} lang={lang}", flush=True)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    try:
        await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).serve()
    finally:
        dial.cancel()
        registrant.close()


if __name__ == "__main__":
    asyncio.run(main())
