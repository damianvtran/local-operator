"""Fixture wrapper for issue #2016 — the shared mobile overflow fixture PLUS two
conversations carrying unread completion receipts, PLUS two fixture-only hooks
the post-fix transcript needs.

Run (from YOUR worktree, so ``scripts`` resolves to it):
    cd ~/local-operator-worktrees/mobile-2016-mark-all-read
    PYTHONPATH=. LOP_MOBILE_FIXTURE_PASSWORD=<per-run> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/postfix/scripts/mobile_2016_fixture.py" 4216

Optionally set ``LOP_2016_SEED_OUT=<path>`` to dump the seeded tokens (a
cross-check for the transcript; synthetic ids, loopback only).

WHY A WRAPPER INSTEAD OF AN EDIT. ``scripts/mobile_overflow_fixture.py`` is a
shared rig every session on this machine can run; it seeds NO unread state, and
the surface issue #2016 is about cannot be photographed without it (the
mark-all control only means something against unread rows). The wrapper imports
the fixture as a module and monkeypatches ONE seam (``fx.build_app``) so the
fixture's own ``main()`` runs unmodified and the seed lands right before the app
is built — after every fixture row exists, so the seeded records carry the
newest ``started_at``.

WHAT THE SEED IS, and through which door. For two synthetic 12-hex sessions:
  * a session directory with a real opening turn (``fx._write_opening_turn``,
    the fixture's own writer — a directory with ``transcript.jsonl`` is what the
    durable scan lists; ``mobile_delegating_fixture.py::_seed_unread`` is the
    same two-part shape);
  * ONE completion receipt through the REAL writer, ``AttentionStore.publish``
    — the call a runtime's receipt goes through — so the ``unseen`` the badge
    and the list serve comes from the store, never from a hand-built payload;
  * a live entry carrying the record, exactly as the fixture injects its rows.
The daemon's own merge (``_merge_summaries`` -> ``_attention_states`` via
``AttentionStore.state_many_and_revision``) is what turns the receipt into the
row's ``unseen``; nothing here paints a mark the store does not report.

THE TWO HOOKS (added for the post-fix evidence run; fixture-only, loopback):
  * ``POST /fixture/publish {"session_id"}`` — mint a FRESH completion through
    ``AttentionStore.publish`` mid-run. This is how the not-a-sweep case is
    staged against a real store write: the transcript renders nothing, the
    hook settles a new turn, and the stale receipt must NOT clear it.
  * ``GET /fixture/state?session_id=`` — the store's own ``state`` for a
    conversation, so the evidence can show the SHARED store's read changing
    (the same file the desktop and TUI open).
Both are appended to the built Starlette router; they carry no auth (sandbox
only) and touch nothing outside this run's sandbox root.

ISOLATION. ``scripts.probe_isolation`` (imported FIRST) re-homes HOME and
LOCAL_OPERATOR_CONFIG_DIR to a fresh sandbox, so the receipts land in a sandbox
``attention.db``; every ``LOP_*`` variable is scrubbed before any
``local_operator`` import (the AGENTS.md "Isolating a run" leak — an inherited
``LOP_RUNTIME_ADOPT_SESSION``/provider export once made a child run under the
parent session's identity); and the fixture itself runs
``dial_registrants=False``, so the live fleet is never dialled or touched.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

# READ THE CONTRACT BEFORE THE SCRUB: the per-run password is this run's own
# input (env var, or argv[2] for parity with the shared fixture), not one of the
# inherited LOP_* flags the scrub below exists to drop.
_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"
_password = os.environ.get(_PASSWORD_ENV, "") or (sys.argv[2] if len(sys.argv) > 2 else "")
# Also read the seed-dump path BEFORE the scrub — it is an LOP_* variable too,
# and the scrub is wholesale (the first run of this wrapper proved the point:
# the receipts landed but the dump silently went to "").
_SEED_OUT = os.environ.get("LOP_2016_SEED_OUT", "")
for _key in tuple(os.environ):
    if _key.startswith("LOP_"):
        os.environ.pop(_key)
if _password:
    os.environ[_PASSWORD_ENV] = _password

import scripts.probe_isolation  # noqa: E402,F401  -- must be the first local import
import scripts.mobile_overflow_fixture as fx  # noqa: E402

#: Two distinct 12-hex ids in the product's own session-id shape (lowercase
#: hex, 12 chars). Distinct from every fixture session so the seed can be
#: counted and named unambiguously in the evidence.
_SEED_IDS = ("c0ffee000016", "c0ffee000017")
_SEED_TITLES = {
    "c0ffee000016": "Mobile parity audit",
    "c0ffee000017": "Release notes draft",
}
_SEED_OPENERS = {
    "c0ffee000016": "Mobile parity audit — the phone should clear its unread pile in one gesture.",
    "c0ffee000017": "Release notes draft — one unread completion is waiting here.",
}


def _seed_unread(daemon: "fx.MobileDaemon") -> None:
    """Two conversations with unread receipts, through the supported doors.

    Idempotent: ``build_app`` is a single call site today, and a second call
    must not double-publish (a duplicate publish would mint a second sequence
    for the same conversation and the badge would lie).
    """
    if getattr(daemon, "_seed_2016_done", False):
        return
    daemon._seed_2016_done = True

    from local_operator.paths import config_dir
    from local_operator.session.attention import AttentionStore

    seeds: list[dict[str, str]] = []
    for offset, session_id in enumerate(_SEED_IDS):
        directory = config_dir() / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        fx._write_opening_turn(directory, _SEED_OPENERS[session_id])
        token = str(uuid.uuid4())
        # The real writer, exactly as a runtime's settled turn does it: one
        # completion row, sequence assigned by the store, no receipt.
        AttentionStore().publish(f"session/{session_id}", token, "fixture-completion", "complete")
        record = fx.SessionRecord(
            pid=900901 + offset,
            kind="tui",
            session_id=session_id,
            conversation_name=_SEED_TITLES[session_id],
            cwd="/synthetic",
            model_label="fixture",
            control_port=1,
            control_key="fixture",
            detached=True,
        )
        daemon.table.entries[record.pid] = fx.SessionEntry(record)
        seeds.append(
            {
                "session_id": session_id,
                "conversation_name": _SEED_TITLES[session_id],
                "completion_token": token,
                "anchor_id": "fixture-completion",
            }
        )

    out = _SEED_OUT
    if out:
        Path(out).write_text(json.dumps({"seed": seeds}, indent=2), encoding="utf-8")


_build_app_unseeded = fx.build_app


def _build_app_seeded(daemon: "fx.MobileDaemon"):
    """The one seam: seed, the fixture's own app, then the evidence hooks."""
    _seed_unread(daemon)
    app = _build_app_unseeded(daemon)

    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from local_operator.session.attention import AttentionStore

    async def fixture_publish(request: Request) -> JSONResponse:
        body = await request.json()
        session_id = str((body or {}).get("session_id") or "")
        token = str(uuid.uuid4())
        AttentionStore().publish(f"session/{session_id}", token, "fixture-anchor", "complete")
        return JSONResponse({"session_id": session_id, "completion_token": token})

    async def fixture_state(request: Request) -> JSONResponse:
        session_id = str(request.query_params.get("session_id") or "")
        return JSONResponse(AttentionStore().state(f"session/{session_id}"))

    app.router.routes.append(Route("/fixture/publish", fixture_publish, methods=["POST"]))
    app.router.routes.append(Route("/fixture/state", fixture_state, methods=["GET"]))
    return app


fx.build_app = _build_app_seeded  # type: ignore[assignment]

if __name__ == "__main__":
    import asyncio

    asyncio.run(fx.main())
