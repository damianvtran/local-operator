"""Fixture wrapper for issue #2016 — round-1 remediation evidence.

Extends the post-fix wrapper with two capability hooks the round-1 states need,
both built on the daemon's REAL code paths rather than a mocked response:

  * ``/fixture/degrade?on=1|0`` — makes ``AttentionStore.state_many_and_revision``
    raise ``OSError``, which is exactly the seam the repo's own
    ``tests/unit/mobile/test_attention_unread.py`` injects. The daemon then
    serves the genuine degraded aggregate (``{"degraded": ["attention"]}``, no
    ``conversations`` key) through its own handler, so the frame of the degraded
    receipt is evidence of the real path, not of a hand-built body.
  * ``/fixture/publish`` now ALSO invalidates the summaries cache, so a
    completion minted out of band is visible on the next paint without waiting
    out the cache TTL.

Run (from the PR-head worktree, so ``scripts`` resolves to it):

    cd ~/local-operator-worktrees/mobile-2016-mark-all-read
    PYTHONPATH=. LOP_MOBILE_FIXTURE_PASSWORD=<per-run> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/round1/scripts/mobile_2016_fixture_r1.py" 4216

ISOLATION is unchanged from the round-0 wrapper: ``scripts.probe_isolation``
first (fresh HOME + config root), every ``LOP_*`` variable scrubbed before any
``local_operator`` import, ``dial_registrants=False``, synthetic ids, non-default
port. Nothing here touches the operator's live daemon.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

# Read the run's own inputs BEFORE the wholesale LOP_* scrub below.
_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"
_password = os.environ.get(_PASSWORD_ENV, "") or (sys.argv[2] if len(sys.argv) > 2 else "")
_SEED_OUT = os.environ.get("LOP_2016_SEED_OUT", "")
for _key in tuple(os.environ):
    if _key.startswith("LOP_"):
        os.environ.pop(_key)
if _password:
    os.environ[_PASSWORD_ENV] = _password

import scripts.probe_isolation  # noqa: E402,F401  -- must be the first local import
import scripts.mobile_overflow_fixture as fx  # noqa: E402

_SEED_IDS = ("c0ffee000016", "c0ffee000017")
_SEED_TITLES = {
    "c0ffee000016": "Mobile parity audit",
    "c0ffee000017": "Release notes draft",
}
_SEED_OPENERS = {
    "c0ffee000016": "Mobile parity audit — the phone should clear its unread pile in one gesture.",
    "c0ffee000017": "Release notes draft — one unread completion is waiting here.",
}

#: The degraded-read switch, read by the patched store method below.
_DEGRADE = {"on": False}


def _patch_store() -> None:
    """Make the aggregate's own read seam fail on demand.

    Patched on the CLASS, so every construction of the store (the daemon builds
    a fresh one per read) is covered; the failure is raised only while the
    fixture's switch is on.
    """
    from local_operator.session.attention import AttentionStore

    real = AttentionStore.state_many_and_revision

    def possibly_degraded(self, *args, **kwargs):
        if _DEGRADE["on"]:
            raise OSError("fixture: the unread store could not be read")
        return real(self, *args, **kwargs)

    AttentionStore.state_many_and_revision = possibly_degraded  # type: ignore[assignment]


def _seed_unread(daemon: "fx.MobileDaemon") -> None:
    """Two conversations with unread receipts, through the supported doors."""
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
    """Seed, then the fixture's own app, then the round-1 capability hooks."""
    _seed_unread(daemon)
    _patch_store()
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
        # The out-of-band write must be visible on the next paint, not after the
        # summaries TTL: the same invalidation the daemon's own scan loop does.
        daemon.table.invalidate_summaries_cache()
        return JSONResponse({"session_id": session_id, "completion_token": token})

    async def fixture_state(request: Request) -> JSONResponse:
        session_id = str(request.query_params.get("session_id") or "")
        return JSONResponse(AttentionStore().state(f"session/{session_id}"))

    async def fixture_degrade(request: Request) -> JSONResponse:
        _DEGRADE["on"] = request.query_params.get("on") in ("1", "true", "yes")
        return JSONResponse({"degraded": _DEGRADE["on"]})

    app.router.routes.append(Route("/fixture/publish", fixture_publish, methods=["POST"]))
    app.router.routes.append(Route("/fixture/state", fixture_state, methods=["GET"]))
    app.router.routes.append(Route("/fixture/degrade", fixture_degrade, methods=["GET"]))
    return app


fx.build_app = _build_app_seeded  # type: ignore[assignment]

if __name__ == "__main__":
    import asyncio

    asyncio.run(fx.main())
