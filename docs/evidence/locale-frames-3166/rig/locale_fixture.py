"""Serve the REAL relay SPA bundle with a config naming ONE locale, for
per-locale rendered evidence on #3165.

Run:
    python locale_fixture.py --port 4701 --locale ru --mode populated --password-file <f>

WHAT THIS IS. A minimal sibling of ``scripts/mobile_overflow_fixture.py``: the
same ``MobileDaemon`` + ``build_app`` + synthetic ``SessionEntry`` seeding, cut
down to what a language frame needs. It is scratch scaffolding for one QA round,
not a fixture the repo should carry (QA never edits the repo).

WHY THE REAL DAEMON AND NOT A STUB. The locale the SPA paints comes from the
config stream's baseline frame (``local_operator/i18n/frames.py``), resolved by
the real ``resolve_language`` off the real config file. A stub server would let
the harness "prove" a rendering path the product does not have.

MODES:
  populated — two synthetic sessions, so the list and a session deep link render.
  empty     — no sessions, so the list's own empty state renders.
  deadapi   — the daemon serves the bundle and the config stream, and answers
              503 for every other API route: the "relay reachable, session API
              broken" error state. A plain static host cannot produce it: with no
              config stream the client has no locale at all, so the frame would
              be English and prove nothing about the locale.

NOTHING HERE TOUCHES THE OPERATOR'S DAEMON: ``scripts.probe_isolation`` re-homes
HOME and the config dir on import, and ``dial_registrants=False`` keeps this
daemon off the registrant sockets.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

# The PR worktree (this file lives in the session scratchpad, not the repo).
import os as _os
REPO = Path(_os.environ["LF_REPO"])
sys.path.insert(0, str(REPO))

import scripts.probe_isolation  # noqa: F401,E402  -- must be the first local import

import uvicorn  # noqa: E402

from local_operator.config import ConfigManager  # noqa: E402
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app  # noqa: E402
from local_operator.mobile.types import (  # noqa: E402
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)
from local_operator.paths import config_dir  # noqa: E402


def _pid_for(session_id: str) -> int:
    """A stable synthetic pid per session id (never a real one)."""
    digest = hashlib.sha256(session_id.encode()).hexdigest()
    return 910000 + int(digest[:6], 16) % 9_000


def projection(session_id: str, name: str, *, streaming: bool) -> SessionProjection:
    """One ordinary conversation, enough copy on screen to read a language off."""
    return SessionProjection(
        session_id=session_id,
        pid=_pid_for(session_id),
        kind="tui",
        conversation_name=name,
        streaming=streaming,
        activity="running the locale evidence pass",
        activity_started_s=37,
        transcript=[
            TranscriptEntry(
                id=f"{session_id}-u1",
                kind="user",
                text="Which locale is on screen right now?",
            ),
            TranscriptEntry(
                id=f"{session_id}-a1",
                kind="assistant",
                text=(
                    "This line rides the transcript wire rather than the bundled "
                    "owned copy, so it stays English by design."
                ),
            ),
        ],
        version=3,
    )


def seed(daemon: MobileDaemon, projections: list[SessionProjection]) -> None:
    """Seed live entries AND their session directories.

    The directory is not decoration: a LIVE-ONLY entry is admitted to the list
    only when ``_live_generation_is_user_facing`` finds its session directory
    (``config_dir()/sessions/<id>``) present and not marked as an agent-shell
    origin. Without it the list paints its empty state beside a deep link that
    renders the same session perfectly, which is exactly the confusing pair
    this seeding exists to avoid.
    """
    sessions_root = config_dir() / "sessions"
    for proj in projections:
        (sessions_root / proj.session_id).mkdir(parents=True, exist_ok=True)

        record = SessionRecord(
            pid=proj.pid,
            kind="tui",
            session_id=proj.session_id,
            conversation_name=proj.conversation_name,
            cwd="/synthetic",
            model_label="fixture",
            control_port=1,
            control_key="fixture",
        )
        entry = SessionEntry(record)
        entry.projection = proj
        daemon.session_projections[proj.session_id] = proj
        daemon.table.entries[record.pid] = entry


def set_language(tag: str) -> None:
    """Write the machine's ``language`` value the way a user's write lands.

    Through ``ConfigManager`` rather than by hand: the resolver reads this file
    through the same manager, and a hand-written document it would treat as
    malformed gets moved aside silently — which reads as "the config was
    ignored" and would make a control vacuous.
    """
    manager = ConfigManager(config_dir())
    manager.set_config_value("language", tag)
    resolved = manager.get_config().values.get("language")
    print(f"config language = {resolved!r}", flush=True)


def slow_chunk(app, locale: str, delay_ms: int):
    """Delay ONE locale chunk on the wire, leaving every other byte untouched.

    WHY SERVER-SIDE. The budget control has to make the chunk arrive after
    ``LOCALE_BOOTSTRAP_BUDGET_MS`` (1500 ms) while everything else is normal. A
    CDP request-interception would be the harness delaying its own network; a
    delay here is the same 200 response a cold or congested link produces, and
    the client's behaviour — await inside the budget, else paint English and
    flip when it lands — is the real one either way. Nothing else is slowed, so
    the main bundle still arrives at full speed and the boot is not a
    different scenario.
    """
    needle = f"bundled.{locale}.gen-"

    async def wrapped(scope, receive, send):
        if scope["type"] == "http" and needle in scope.get("path", ""):
            await asyncio.sleep(delay_ms / 1000)
        await app(scope, receive, send)

    return wrapped


def dead_api(app):
    """Wrap the daemon app: keep the bundle, the login and the config stream.

    Everything else under ``/api`` answers 503 so the SPA's own error path runs
    with a locale frame already in hand. The config stream has to stay live —
    it is what names the locale — and the login has to stay live or there is no
    session to paint at all.
    """
    from starlette.responses import PlainTextResponse

    keep = ("/api/config/events",)

    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path.startswith("/api/") and path not in keep:
                response = PlainTextResponse("relay unavailable", status_code=503)
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return wrapped


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--locale", required=True)
    parser.add_argument(
        "--mode", default="populated", choices=("populated", "empty", "deadapi", "loading")
    )
    parser.add_argument("--password-file", required=True)
    parser.add_argument("--slow-chunk", default=None, help="locale whose chunk is held")
    parser.add_argument("--slow-ms", type=int, default=2500)
    args = parser.parse_args()

    password = Path(args.password_file).read_text().strip()
    if not password:
        raise SystemExit(f"{args.port}: empty password file")

    set_language(args.locale)

    daemon = MobileDaemon(port=args.port, password=password, dial_registrants=False)
    if args.mode != "empty":
        seeded = [
            projection("locale-a", "Locale evidence A", streaming=True),
            projection("locale-b", "Locale evidence B", streaming=False),
        ]
        seed(daemon, seeded)
        if args.mode == "loading":
            # The list knows the conversation; its PROJECTION stream never speaks,
            # so the session screen stays on `connecting to session…` — the real
            # loading state of a relay whose runtime is slow to attach, which is
            # what a loading still has to photograph rather than a race.
            for proj in seeded:
                daemon.session_projections.pop(proj.session_id, None)
    app = build_app(daemon)
    if args.mode == "deadapi":
        app = dead_api(app)
    if args.slow_chunk:
        app = slow_chunk(app, args.slow_chunk, args.slow_ms)
    import local_operator.mobile.daemon as _d
    print(f"locale fixture: http://127.0.0.1:{args.port} locale={args.locale} mode={args.mode} daemon={_d.__file__} dist={_d._DIST_DIR} assets={sorted(p.name for p in (_d._DIST_DIR/'assets').glob('bundled.*'))}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
