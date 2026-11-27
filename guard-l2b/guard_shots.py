"""Radient Guard L2b TUI frames over the REAL OperatorApp (stylesheet loaded).

env -u NO_COLOR TERM=xterm-256color <wt>/.venv/bin/python guard_shots.py <wt> <outdir>
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

WT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WT))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

import local_operator  # noqa: E402

assert str(local_operator.__file__).startswith(str(WT)), local_operator.__file__

from local_operator.guard import consent as C  # noqa: E402
from local_operator.guard import surface  # noqa: E402
from local_operator.guard.reports import GuardReports  # noqa: E402
from local_operator.hub_org.eligibility import OrgRef  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.tui import app as app_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import TurnStarted  # noqa: E402
from local_operator.tui.widgets import guard_view  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.guard_view import GuardScreen  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

TERMS = "sha256:" + "ab" * 32
TEAM = [OrgRef("ten_acme", "Acme Robotics", "member")]
facts: dict[str, object] = {}


def offer(rev: int = 1, reason: str = "new_org", scopes=None, added=None) -> dict:
    return {
        "tenant": "ten_acme",
        "org_display_name": "Acme Robotics",
        "collection_rev": rev,
        "revocation_gen": 0,
        "scopes": scopes or ["decisions", "task_time", "review_cycles", "reports"],
        "added_scopes": added or [],
        "rendered_terms_digest": TERMS,
        "reason": reason,
    }


def fresh_root() -> Path:
    root = Path(config_dir())
    guard = root / "guard"
    if guard.exists():
        for f in guard.iterdir():
            f.unlink()
    return root


def seed_live(root: Path, scopes: list[str]) -> None:
    conn = C.GuardConsentStore(root).connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO consents (tenant, collection_rev, scopes, granted_at, state)"
            " VALUES (?, ?, ?, ?, 'live')",
            ("ten_acme", 1, json.dumps(scopes), time.time()),
        )
    finally:
        conn.close()


async def seeded(pilot, app) -> None:
    await pilot.pause()
    app.query_one(Editor).cursor_blink = False
    for turn in range(1, 4):
        app._append_block(UserBlock(f"Turn {turn}: rotate the staging deploy keys"))
        prose = AssistantBlock()
        prose.update_text(f"Answer {turn}: rotating the keys and re-running the deploy check.")
        app._append_block(prose)
    await pilot.pause()


async def settle(pilot, n: int = 6) -> None:
    for _ in range(n):
        await pilot.pause()
    await asyncio.sleep(0.6)  # past the 500 ms debounce, so the frame is the settled one
    await pilot.pause()


async def run(name: str, setup, action, size=(100, 34)) -> None:
    root = fresh_root()
    setup(root)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await seeded(pilot, app)
        await action(pilot, app, root)
        save_capture(app, str(OUT / f"{name}.svg"))
    print("captured", name)


def main() -> None:
    surface._team_orgs = lambda _root: list(TEAM)  # the cached Team record, in-rig

    def pending_new(root):
        C.GuardConsentHost(root, transport=None, mirror=False).offer(C.ConsentOffer.from_wire(offer()))

    async def notice_during_turn(pilot, app, root):
        app.post_message(TurnStarted())
        await pilot.pause()
        app._poll_guard()
        await settle(pilot)
        facts["notice_during_turn"] = {
            "streaming": bool(app._status._streaming),
            "badge": app._guard_badge,
            "screen": type(app.screen).__name__,
        }

    async def prompt_during_turn(pilot, app, root):
        app.post_message(TurnStarted())
        await pilot.pause()
        app._poll_guard()
        app._cmd_guard()
        await settle(pilot)
        # Type-ahead: Enter right after a fresh render must not answer it.
        screen = app.screen
        screen._armed_at = time.monotonic() + 0.5
        await pilot.press("enter")
        await pilot.pause()
        row = C.GuardConsentHost(root, transport=None, mirror=False).prompt("ten_acme", 1)
        facts["prompt_during_turn"] = {
            "streaming_after_open": bool(app._status._streaming),
            "screen": type(app.screen).__name__,
            "focused": getattr(app.focused, "id", None),
            "row_state_after_typeahead_enter": row.state if row else None,
        }
        await asyncio.sleep(0.6)
        await pilot.pause()

    async def esc_is_later(pilot, app, root):
        app._cmd_guard()
        await settle(pilot)
        await pilot.press("escape")
        await settle(pilot, 3)
        row = C.GuardConsentHost(root, transport=None, mirror=False).prompt("ten_acme", 1)
        facts["esc"] = {"screen": type(app.screen).__name__, "row_state": row.state if row else None}

    def sharing(root):
        seed_live(root, ["decisions", "task_time"])

    async def overview(pilot, app, root):
        app._poll_guard()
        app._cmd_guard()
        await settle(pilot)
        facts["sharing"] = {"badge": app._guard_badge, "focused": getattr(app.focused, "id", None)}

    def widening(root):
        seed_live(root, ["decisions", "task_time"])
        C.GuardConsentHost(root, transport=None, mirror=False).offer(
            C.ConsentOffer.from_wire(
                offer(rev=2, reason="widened", scopes=["decisions", "task_time", "posture"], added=["posture"])
            )
        )

    async def widening_card(pilot, app, root):
        app._poll_guard()
        app._cmd_guard()
        await settle(pilot)
        facts["widening"] = {
            "badge": app._guard_badge,
            "focused": getattr(app.focused, "id", None),
            "state": surface.org_states(root)[0].state,
        }

    def report(root):
        seed_live(root, ["decisions", "reports"])
        GuardReports(root).draft("ten_acme", "credential_exposure")

    async def report_card(pilot, app, root):
        app._cmd_guard()
        await settle(pilot)
        facts["report"] = {"focused": getattr(app.focused, "id", None)}

    def answered(root):
        host = C.GuardConsentHost(root, transport=None, mirror=False)
        host.offer(C.ConsentOffer.from_wire(offer()))

    async def answered_elsewhere(pilot, app, root):
        app._cmd_guard()
        await settle(pilot)
        # The desktop answered the same row (decline): the open card closes with
        # "Answered on the desktop app" at its next re-check.
        C.GuardConsentHost(root, transport=None, mirror=False).decline("ten_acme", 1, surface="desktop")
        await app.screen._recheck()
        await settle(pilot)

    run_list = [
        ("01-notice-badge-during-active-turn", pending_new, notice_during_turn),
        ("02-prompt-card-focus-dont-share-during-turn", pending_new, prompt_during_turn),
        ("03-esc-decide-later-back-to-transcript", pending_new, esc_is_later),
        ("04-guard-overview-sharing-indicator", sharing, overview),
        ("05-widening-card-keep-current-focus", widening, widening_card),
        ("06-report-draft-card", report, report_card),
        ("07-answered-on-desktop", answered, answered_elsewhere),
    ]
    for name, setup, action in run_list:
        asyncio.run(run(name, setup, action))

    # Non-Team: no Team record at all.
    surface._team_orgs = lambda _root: []
    guard_view._has_team = lambda _root: False
    for name, body in (
        ("08-non-team-card-trial-eligible", {"eligible": True, "reason": "", "trial_days": 30}),
        ("09-non-team-card-ineligible", {"eligible": False, "reason": "used", "trial_days": 30}),
    ):
        calls: list[str] = []

        def loader(body=body, calls=calls):
            calls.append("GET /v1/team-trial/eligibility")
            return body

        guard_view.load_trial_eligibility = lambda _root, loader=loader: loader()

        async def card(pilot, app, root, calls=calls, name=name):
            app._poll_guard()
            app._cmd_guard()
            await settle(pilot, 10)
            facts[name] = {
                "calls": list(calls),
                "guard_files": sorted(p.name for p in (root / "guard").glob("*")) if (root / "guard").exists() else [],
                "badge": app._guard_badge,
            }

        asyncio.run(run(name, lambda root: None, card))
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


main()
