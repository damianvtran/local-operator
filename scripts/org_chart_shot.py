"""Capture the `/team chart` org-chart surface for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/org_chart_shot.py OUT.svg SCENARIO [COLSxROWS] [TIER] [EXPAND]

``SCENARIO`` seeds a ``TeamRegistry`` for one permutation of the §7.2 matrix
(1-agent, empty, flat-multi, nested, org-within-org, cycle, count, deep,
missing) and opens the chart at ``TIER`` (0/1/2). ``picker-*`` scenarios instead
capture the `/team ` argument list frames (the collision the user feared) by
typing into the real editor. Modelled on ``scripts/ask_shot.py``: a real
``OperatorApp`` so the stylesheet is applied and the frame is what a user sees.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.teams import (  # noqa: E402
    MAX_ORG_DEPTH,
    TeamEditFields,
    TeamMember,
    TeamRegistry,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def _registry() -> TeamRegistry:
    return TeamRegistry(Path(tempfile.mkdtemp()))


def _seed(scenario: str) -> TeamRegistry:
    reg = _registry()
    if scenario == "one-agent":
        reg.create_team(
            TeamEditFields(name="solo", manager="boss", members=[TeamMember(role="coder")])
        )
    elif scenario == "empty":
        reg.create_team(TeamEditFields(name="empty", manager="boss", members=[]))
    elif scenario == "flat-multi":
        reg.create_team(
            TeamEditFields(
                name="squad",
                manager="lead",
                members=[
                    TeamMember(role="coder"),
                    TeamMember(role="reviewer"),
                    TeamMember(role="designer"),
                    TeamMember(role="architect"),
                ],
            )
        )
    elif scenario in ("nested", "count", "zoom"):
        reg.create_team(
            TeamEditFields(
                name="pod-a",
                manager="mgr-a",
                members=[TeamMember(role="coder"), TeamMember(role="reviewer", count=2)],
            )
        )
        reg.create_team(
            TeamEditFields(name="pod-b", manager="mgr-b", members=[TeamMember(role="scout")])
        )
        reg.create_team(
            TeamEditFields(
                name="org",
                manager="director",
                members=[
                    TeamMember(role="pod-a", kind="team"),
                    TeamMember(role="pod-b", kind="team", count=2),
                ],
            )
        )
    elif scenario == "org-within-org":
        reg.create_team(
            TeamEditFields(name="squad", manager="sl", members=[TeamMember(role="scout")])
        )
        reg.create_team(
            TeamEditFields(
                name="pod", manager="lead", members=[TeamMember(role="squad", kind="team")]
            )
        )
        reg.create_team(
            TeamEditFields(
                name="org", manager="director", members=[TeamMember(role="pod", kind="team")]
            )
        )
    elif scenario == "cycle":
        reg.create_team(
            TeamEditFields(name="a", manager="m-a", members=[TeamMember(role="b", kind="team")])
        )
        reg.create_team(
            TeamEditFields(name="b", manager="m-b", members=[TeamMember(role="a", kind="team")])
        )
    elif scenario == "deep":
        depth = MAX_ORG_DEPTH + 2
        for i in range(depth):
            members = [TeamMember(role=f"t{i + 1}", kind="team")] if i < depth - 1 else []
            reg.create_team(TeamEditFields(name=f"t{i}", manager=f"m{i}", members=members))
    elif scenario == "missing-refs":
        reg.create_team(
            TeamEditFields(
                name="org",
                manager="director",
                members=[
                    TeamMember(role="ghost-team", kind="team"),
                    TeamMember(role="ghost-agent"),
                    TeamMember(role="coder"),
                ],
            )
        )
    elif scenario == "wide":
        # A 12-member flat team: overflows a normal terminal, the U1/U2 case.
        reg.create_team(
            TeamEditFields(
                name="wide",
                manager="boss",
                members=[TeamMember(role=f"m{i}") for i in range(12)],
            )
        )
    else:
        # picker scenarios need a couple of teams incl. one named `chart`.
        reg.create_team(
            TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")])
        )
        reg.create_team(
            TeamEditFields(name="chart", manager="m", members=[TeamMember(role="coder")])
        )
        reg.create_team(
            TeamEditFields(name="channels", manager="m", members=[TeamMember(role="coder")])
        )
    return reg


#: Which top-level team each chart scenario opens.
_ROOT = {
    "one-agent": "solo",
    "empty": "empty",
    "flat-multi": "squad",
    "nested": "org",
    "count": "org",
    "zoom": "org",
    "org-within-org": "org",
    "cycle": "a",
    "deep": "t0",
    "missing-refs": "org",
    "wide": "wide",
}


async def main() -> None:
    out = sys.argv[1]
    scenario = sys.argv[2]
    size = (100, 30)
    if len(sys.argv) > 3 and sys.argv[3]:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))
    tier = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 1
    expand = len(sys.argv) > 5 and sys.argv[5] == "expand"
    legend = len(sys.argv) > 5 and "legend" in sys.argv[5:]

    session = FakeSession()
    session.team_registry = _seed(scenario)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        # Wait for the session (and its team registry) to exist before opening
        # the chart — the resolver reads the registry off the live session.
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        if scenario.startswith("picker-"):
            # Drive the argument list open through REAL key presses. Post-#250
            # the picker's detection is caret-anchored, so setting editor.text
            # directly no longer opens the list — only typing does (the same
            # path the app takes and the tests use).
            app.query_one(Editor).focus()
            buffer = {
                "picker-team": "/team ",
                "picker-ch": "/team ch",
                "picker-chart": "/team chart ",
            }[scenario]
            for char in buffer:
                key = "slash" if char == "/" else ("space" if char == " " else char)
                await pilot.press(key)
            await pilot.pause()
            await pilot.pause()
        else:
            app._open_org_chart_view(_ROOT[scenario])
            await pilot.pause()
            await pilot.pause()  # let the one-shot auto-fit settle first
            view = app._org_chart_view
            assert view is not None
            # Set the tier explicitly AFTER auto-fit so the capture is
            # deterministic regardless of what auto-fit chose on open.
            view._set_tier(tier)
            if expand:
                view.action_toggle_expand()
            if legend:
                view.action_toggle_legend()
            await pilot.pause()
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
