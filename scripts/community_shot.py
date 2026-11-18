"""Capture the ``/community`` page and ``/orgs`` switcher states (TUI frames T1–T10).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/community_shot.py OUT_DIR [STATE ...]

Each state is opened the way a person opens it — ``/community`` or ``/orgs``
typed into the REAL ``OperatorApp`` composer (so ``local_operator.tcss``
applies) — over a fake community backend built from the client's response
models (``tests/unit/tui/community_fakes.py``). One ``<state>.svg`` per state;
``--list`` prints the state names.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()
os.environ["LOCAL_OPERATOR_COMMUNITY"] = "1"

from local_operator.community import errors as e  # noqa: E402
from local_operator.community.service import PostOutcome  # noqa: E402
from local_operator.tui import community_backend  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.community_view import CommunityView  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.community_fakes import FakeBackend  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeSession,
    _await_session,
    _factory,
)

STATES: dict[str, tuple[tuple[int, int], str]] = {
    "home-80x24": ((80, 24), "/community"),
    "home-60x20-composer": ((60, 20), "/community"),
    "home-150x40": ((150, 40), "/community"),
    "loading": ((80, 24), "/community"),
    "empty": ((80, 24), "/community"),
    "error": ((80, 24), "/community"),
    "page-error": ((80, 24), "/community"),
    "checking": ((80, 24), "/community"),
    "cancelled": ((80, 24), "/community"),
    "too-late": ((80, 24), "/community"),
    "posted": ((80, 24), "/community"),
    "refused": ((80, 24), "/community"),
    "refused-why": ((80, 24), "/community"),
    "rate-limited": ((80, 24), "/community"),
    "moderation-unavailable": ((80, 24), "/community"),
    "read-only": ((80, 24), "/community"),
    "not-joined": ((80, 24), "/community"),
    "join-before": ((80, 24), "/community"),
    "join-after": ((80, 24), "/community"),
    "browse": ((80, 24), "/community"),
    "report": ((80, 24), "/community"),
    "settings-agents": ((80, 24), "/community"),
    "settings-leave": ((80, 24), "/community"),
    "orgs": ((80, 24), "/orgs"),
    "orgs-invited": ((80, 24), "/orgs"),
    "orgs-from-page": ((80, 24), "/community"),
    # Hostile server text (review R1-3): OSC 52 / clear / title payloads in a
    # post and a display name, painted through the page's ingress sanitiser.
    "hostile-text": ((80, 24), "/community"),
}

#: The agent-post approval card and the full-post screen behind it (review
#: R1-5). Raised through the gate the real app installs, as a tool call does.
CARD_STATES: dict[str, tuple[tuple[int, int], str, str]] = {
    "post-card-short-110x34": ((110, 34), "short", ""),
    "post-card-short-80x24": ((80, 24), "short", ""),
    "post-card-long-110x34": ((110, 34), "long", ""),
    "post-card-long-80x24": ((80, 24), "long", ""),
    "post-view-long-80x24": ((80, 24), "long", "y"),
    "post-view-read-80x24": ((80, 24), "long", "y,end"),
}

SHORT_POST = "Doctor fixed it for me too: `lop doctor --fix`, then restart the app."
LONG_POST = "\n\n".join(
    f"Step {i}: "
    + "run the doctor, read what it reports, and paste the summary here so others can compare. " * 3
    for i in range(1, 15)
)


def _backend(state: str) -> FakeBackend:
    backend = FakeBackend(joined=state not in ("not-joined", "join-before", "join-after"))
    if state == "loading":
        backend.submit_gate = asyncio.Event()

        async def slow(*_a: object) -> list[object]:
            await asyncio.Event().wait()
            return []

        backend.history = slow  # type: ignore[method-assign]
    if state == "empty":
        backend.histories["c_help"] = []
    if state == "error":
        backend.history_error["c_help"] = e.CommunityUnavailable()
    if state == "page-error":
        backend.home_error = e.CommunityUnreachable()
    if state in ("checking", "cancelled"):
        backend.submit_gate = asyncio.Event()
    if state == "too-late":
        backend.submit_gate = asyncio.Event()
        backend.cancel_outcome = PostOutcome("too_late", submission_id="k", message_id="m9")
    if state in ("refused", "refused-why"):
        backend.next_outcome = PostOutcome(
            "refused",
            submission_id="k",
            category="harassment_or_abuse",
            reference="MOD-7Q4K-2M9D",
        )
    if state == "rate-limited":
        backend.next_outcome = PostOutcome(
            "rate_limited", submission_id="k", scope="account_burst", retry_after_s=42
        )
    if state == "moderation-unavailable":
        backend.next_outcome = PostOutcome("moderation_unavailable", submission_id="k")
    if state == "hostile-text":
        from tests.unit.tui.community_fakes import HOSTILE_HELP

        backend.histories["c_help"] = list(HOSTILE_HELP)
    if state == "orgs-invited":
        backend.memberships.append(
            {"tenant_id": "t_new", "tenant_name": "Northwind", "status": "invited"}
        )
    return backend


async def _type(pilot: object, text: str) -> None:
    for char in text:
        await pilot.press("space" if char == " " else char)  # type: ignore[attr-defined]


async def capture(state: str, out_dir: Path) -> Path:
    size, command = STATES[state]
    backend = _backend(state)
    community_backend.LiveCommunityBackend = lambda _cfg: backend  # type: ignore[assignment]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await _await_session(app, pilot)
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.load_text(command)
        editor.focus()
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()
        view = app.query_one(CommunityView)
        if state not in (
            "orgs",
            "orgs-invited",
            "page-error",
            "not-joined",
            "join-before",
            "join-after",
            "loading",
        ):
            view.channel_index = next(i for i, c in enumerate(view.channels) if c.name == "help")
            view._open_channel()
            await pilot.pause()
        if state == "loading":
            view.channel_index = 2
            view._open_channel()
        if state in ("home-80x24", "home-150x40"):
            await pilot.press("tab")
            await pilot.press("up")
        if state == "home-60x20-composer":
            await pilot.press("tab", "tab")
        if state in (
            "checking",
            "cancelled",
            "too-late",
            "posted",
            "refused",
            "refused-why",
            "rate-limited",
            "moderation-unavailable",
        ):
            await pilot.press("tab", "tab")
            await _type(pilot, "Will try doctor and report back")
            await pilot.press("enter")
            await pilot.pause()
            if state in ("cancelled", "too-late"):
                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()
            if state == "refused-why":
                await pilot.press("f2")
        if state == "read-only":
            view.channel_index = 0
            view._open_channel()
            await pilot.press("tab", "tab")
        if state in ("join-before", "join-after"):
            await pilot.press("j")
            if state == "join-after":
                await pilot.press("space", "tab")
        if state == "browse":
            await pilot.press("ctrl+k")
            await pilot.pause()
        if state == "report":
            await pilot.press("tab", "up", "up")
            await pilot.press("exclamation_mark")
            await pilot.pause()
            await _type(pilot, "Third time this week")
        if state == "settings-agents":
            await pilot.press("s")
        if state == "settings-leave":
            await pilot.press("s", "tab")
        if state == "orgs-from-page":
            await pilot.press("g")
            for _ in range(3):
                await pilot.pause()
        for _ in range(3):
            await pilot.pause()
        target = out_dir / f"{state}.svg"
        save_capture(app, target)
    return target


async def capture_card(state: str, out_dir: Path) -> Path:
    from local_operator.community.agent_ops import post_card
    from local_operator.harness.approval import COMMUNITY_POST_GATE
    from tests.unit.tui.test_approvals_ux import GatedSession, _gate, _wait_prompt

    size, which, keys = CARD_STATES[state]
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _await_session(app, pilot)
        card = post_card(
            community="Local Operator Community",
            channel="#help",
            text=SHORT_POST if which == "short" else LONG_POST,
            tainted=which == "long",
        )
        pending = asyncio.ensure_future(_gate(session)(COMMUNITY_POST_GATE, card))
        await _wait_prompt(pilot, app)
        for key in filter(None, keys.split(",")):
            await pilot.press(key)
            for _ in range(3):
                await pilot.pause()
        for _ in range(3):
            await pilot.pause()
        target = out_dir / f"{state}.svg"
        save_capture(app, target)
        pending.cancel()
    return target


async def main() -> None:
    if "--list" in sys.argv:
        print("\n".join([*STATES, *CARD_STATES]))
        return
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[2:] or [*STATES, *CARD_STATES]
    for state in wanted:
        print(await (capture_card if state in CARD_STATES else capture)(state, out_dir))


asyncio.run(main())
