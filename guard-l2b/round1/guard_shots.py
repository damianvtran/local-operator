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


SIZES = ((100, 34), (80, 24))
GEOMETRY: dict[str, object] = {}


def _geometry(app) -> dict[str, object]:
    out: dict[str, object] = {"screen": list(app.size)}
    screen = app.screen
    for sel in ("#guard-title", "#guard-body", "#guard-buttons"):
        found = screen.query(sel)
        if found:
            w = found.first()
            out[sel] = {"region": list(w.region), "virtual_size": list(w.virtual_size),
                        "max_height": str(w.styles.max_height)}
    out["buttons"] = {b.id: list(b.region) for b in screen.query("Button")}
    out["focused"] = getattr(app.focused, "id", None)
    return out


async def run(name: str, setup, action, size=(100, 34), env: dict | None = None) -> None:
    import os

    root = fresh_root()
    setup(root)
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await seeded(pilot, app)
            await action(pilot, app, root)
            tag = f"{name}-{size[0]}x{size[1]}"
            save_capture(app, str(OUT / f"{tag}.svg"))
            GEOMETRY[tag] = _geometry(app)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("captured", tag)


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

    def paused_console(root):
        host = C.GuardConsentHost(root, transport=None, mirror=False)
        host.offer(C.ConsentOffer.from_wire(offer()))
        row = host.prompt("ten_acme", 1)
        seed_live(root, ["decisions"])
        host._observe_console_revoke(row)

    def will_ask(root):
        host = C.GuardConsentHost(root, transport=None, mirror=False)
        host.offer(C.ConsentOffer.from_wire(offer()))
        host.decline("ten_acme", 1, surface="tui")
        host.share_again("ten_acme")

    async def stop_confirm(pilot, app, root):
        app._cmd_guard()
        await settle(pilot)
        app.screen._armed_at = 0.0
        app.screen.query_one("#guard-stop").press()
        await settle(pilot)
        facts["stop_confirm"] = {"focused": getattr(app.focused, "id", None)}

    async def badge_only(pilot, app, root):
        app._poll_guard()
        await settle(pilot)
        facts.setdefault("badges", {})[root.name + str(app.size)] = app._guard_badge

    def report_waiting(root):
        seed_live(root, ["decisions", "reports"])
        GuardReports(root).draft("ten_acme", "credential_exposure")

    async def readonly(pilot, app, root):
        app._cmd_guard()
        await settle(pilot)
        facts.setdefault("readonly", {})[str(app.size)] = {
            "buttons": [b.id for b in app.screen.query("Button")],
            "focused": getattr(app.focused, "id", None),
        }

    run_list = [
        ("01-notice-badge-during-active-turn", pending_new, notice_during_turn, None),
        ("02-first-consent-card", pending_new, prompt_during_turn, None),
        ("03-esc-decide-later-back-to-transcript", pending_new, esc_is_later, None),
        ("04-overview-sharing", sharing, overview, None),
        ("05-widening-card", widening, widening_card, None),
        ("06-report-card-focus-decide-later", report, report_card, None),
        ("07-answered-on-desktop", answered, answered_elsewhere, None),
        ("10-overview-paused-console", paused_console, overview, None),
        ("11-overview-will-ask-again", will_ask, overview, None),
        ("12-stop-sharing-confirm", sharing, stop_confirm, None),
        ("13-agent-shell-read-only-card", pending_new, readonly, {"LOCAL_OPERATOR_AGENT_SHELL": "1"}),
        ("14-badge-new-items-waiting", widening, badge_only, None),
        ("15-badge-report-waiting", report_waiting, badge_only, None),
    ]
    for size in SIZES:
        for name, setup, action, env in run_list:
            asyncio.run(run(name, setup, action, size=size, env=env))

    # A Team member whose orgs never raised Guard: the empty overview.
    guard_view._has_team = lambda _root: True
    for size in SIZES:
        asyncio.run(run("16-overview-empty", lambda root: None, overview, size=size))

    # Non-Team: no Team record at all.
    surface._team_orgs = lambda _root: []
    guard_view._has_team = lambda _root: False
    import threading

    hold = threading.Event()
    for size in SIZES:
        hold.clear()
        guard_view.load_trial_eligibility = lambda _root: (hold.wait(10), None)[1]

        async def loading(pilot, app, root):
            app._cmd_guard()
            await settle(pilot, 4)

        asyncio.run(run("17-non-team-card-loading", lambda root: None, loading, size=size))
        hold.set()
    for name, body, size in [
        (n, b, sz)
        for sz in SIZES
        for n, b in (
            ("08-non-team-card-trial-eligible", {"eligible": True, "reason": "", "trial_days": 30}),
            ("09-non-team-card-ineligible", {"eligible": False, "reason": "used", "trial_days": 30}),
        )
    ]:
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

        asyncio.run(run(name, lambda root: None, card, size=size))
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))
    (OUT / "geometry.json").write_text(json.dumps(GEOMETRY, indent=2))
    print(json.dumps(facts, indent=2))


main()
