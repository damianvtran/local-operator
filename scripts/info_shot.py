"""Capture the real ``/info`` slash path against a SYNTHETIC snapshot.

Usage: python scripts/info_shot.py OUTDIR 100x30 [scenario]
  scenarios: populated | empty | degraded | nested | shadowed | loading

Every scenario feeds a hand-built :class:`InfoSnapshot` rather than the
operator's real one. That is not convenience — the frames go on a PR, and the
redaction rules apply to a PNG exactly as they do to the clipboard, so a capture
of a live machine would publish real session names, real working directories and
real credential key names.

The isolation import must stay FIRST: it re-homes ``HOME`` and
``LOCAL_OPERATOR_CONFIG_DIR`` before anything under ``local_operator`` can
resolve the operator's real config (see ``scripts/probe_isolation.py`` for the
incident that made it an import-time action rather than a function to remember).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import json  # noqa: E402
from typing import Any  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.info.collect import LiveState  # noqa: E402
from local_operator.info.model import (  # noqa: E402
    AgentsInfo,
    EnvInfo,
    InfoSnapshot,
    InstallInfo,
    ProcessInfo,
    SessionLine,
    SessionsInfo,
    SubagentLine,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.info_panel import InfoScreen  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: A REAL-SHAPED session id: the harness generates ``uuid4().hex[:12]``, so
#: every session directory on a live machine is exactly 12 characters. A longer
#: demo string would manufacture a header crop no operator can actually hit.
SESSION_ID = "a3f9c21b7e40"

#: Synthetic home root for the path values, so the frames show the ``~``
#: collapsing the screen really does without carrying anyone's username.
HOME = str(Path.home())


def _install(**overrides: object) -> InstallInfo:
    base: dict[str, Any] = dict(
        version="0.51.6",
        kind="uv-tool",
        prefix=f"{HOME}/.local/share/uv/tools/local-operator",
        executable=f"{HOME}/.local/share/uv/tools/local-operator/bin/python",
        is_git_snapshot=True,
        source_ref="4311eb653aa9f00d1c2b",
        build_age_s=5_400.0,
        latest_known="0.51.6",
        latest_age_s=7_200.0,
        python_version="3.12.13",
        python_implementation="CPython",
        platform="macOS-26.6.2-arm64-arm-64bit",
        machine="arm64",
        import_path=f"{HOME}/.local/share/uv/tools/local-operator/lib/python3.12/"
        "site-packages/local_operator",
        import_path_foreign=False,
    )
    base.update(overrides)
    return InstallInfo(**base)  # type: ignore[arg-type]


def _process(**overrides: object) -> ProcessInfo:
    base: dict[str, Any] = dict(
        pid=4243,
        session_id=SESSION_ID,
        conversation_name="Investigate request latency",
        cwd=f"{HOME}/workspace/repos/lo-session-sidebar",
        model_label="anthropic/claude-opus-5",
        effective_model="anthropic/claude-opus-5",
        uptime_s=840.0,
        config_dir=f"{HOME}/.local-operator",
        cache_dir=f"{HOME}/.local-operator/cache",
        agent_home=f"{HOME}/local-operator-home",
        log_dir=f"{HOME}/Library/Logs/local-operator",
        control_port=51234,
        protocol=5,
        kind="tui",
    )
    base.update(overrides)
    return ProcessInfo(**base)  # type: ignore[arg-type]


def _session_line(
    pid: int, name: str, *, state: str = "live", is_self: bool = False, **overrides: object
) -> SessionLine:
    base: dict[str, Any] = dict(
        pid=pid,
        # ``daemon`` by default because that is what a live machine actually
        # holds: ``runtime/process.py`` spawns every reattachable runtime with
        # ``kind="daemon"``, and a scan of this host returned daemon for all 12
        # records. A ``tui``-heavy fixture would picture the rare case.
        kind="daemon",
        state=state,
        session_id=f"{pid:012x}",
        conversation_name=name,
        model_label="anthropic/claude-opus-5",
        cwd=f"{HOME}/workspace/repos/lo-{name.split()[0].lower()}",
        uptime_s=840.0,
        heartbeat_age_s=2.0,
        rss_bytes=190 * (1 << 20),
        footprint_bytes=228 * (1 << 20),
        is_self=is_self,
    )
    base.update(overrides)
    return SessionLine(**base)  # type: ignore[arg-type]


def _env(**overrides: object) -> EnvInfo:
    base: dict[str, Any] = dict(
        mcp_configured=3,
        mcp_connected=3,
        theme="dusk",
        approval_mode="ask",
        terminal_size=(100, 30),
        term="xterm-256color",
        colorterm="truecolor",
        multiplexer="cmux",
        is_tty=True,
        browser_backend="extension",
        browser_name="Chrome",
        browser_paired=True,
        mobile_installed=True,
        mobile_healthy=True,
        mobile_port=8420,
        credential_keys=("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RADIENT_API_KEY"),
        guides=2,
        skills=6,
    )
    base.update(overrides)
    return EnvInfo(**base)  # type: ignore[arg-type]


#: Fixed capture stamp so two runs of one scenario are byte-comparable. A
#: wall-clock "captured HH:MM:SS" row would otherwise differ on every capture
#: and defeat the identical-consecutive-frames check.
CAPTURED_AT = 1_788_602_400.0


def snapshot_for(scenario: str) -> InfoSnapshot | None:
    """The synthetic snapshot each named frame renders. ``None`` = loading."""
    if scenario == "loading":
        return None

    if scenario == "empty":
        # The FRESH INSTALL case: one session, no subagents. The case most
        # people see, and the one that must not look broken.
        return InfoSnapshot(
            install=_install(is_git_snapshot=False, source_ref="", build_age_s=None),
            process=_process(),
            sessions=SessionsInfo(
                lines=(_session_line(4243, "Investigate request latency", is_self=True),),
                total=1,
                live=1,
            ),
            agents=AgentsInfo(profiles=0, teams=0),
            env=_env(
                mcp_configured=0,
                mcp_connected=0,
                credential_keys=("ANTHROPIC_API_KEY",),
                mobile_installed=False,
                mobile_healthy=False,
                browser_backend="none",
                browser_name="",
                skills=0,
            ),
            captured_at=CAPTURED_AT,
        )

    if scenario == "shadowed":
        # The ``launch.py`` cwd trap: a uv-tool install whose RUNNING CODE is a
        # checkout on sys.path. Active on this machine, not hypothetical.
        return InfoSnapshot(
            install=_install(
                import_path=f"{HOME}/local-operator/local_operator",
                import_path_foreign=True,
            ),
            process=_process(cwd=f"{HOME}/local-operator"),
            sessions=SessionsInfo(
                lines=(_session_line(4243, "Investigate request latency", is_self=True),),
                total=1,
                live=1,
            ),
            agents=AgentsInfo(profiles=20, teams=3),
            env=_env(),
            captured_at=CAPTURED_AT,
        )

    if scenario == "degraded":
        # One unavailable FIELD and one unavailable SECTION, plus a settled MCP
        # failure and a build skew — the shapes §6 specifies.
        return InfoSnapshot(
            install=_install(
                kind="", prefix="", executable="", latest_known=None, latest_age_s=None
            ),
            process=_process(control_port=None, config_dir_redirected=True),
            sessions=SessionsInfo(available=False),
            agents=AgentsInfo(profiles=20, teams=3),
            env=_env(
                mcp_configured=3,
                mcp_connected=1,
                mcp_failed=2,
                mcp_failures=(
                    ("github", "command not found: gh"),
                    ("linear", "connect ECONNREFUSED 127.0.0.1:5173"),
                ),
                browser_backend="extension (stale)",
                mobile_healthy=False,
            ),
            degraded=(
                ("install.kind", "PackageNotFoundError: local-operator"),
                ("sessions", "OSError: [Errno 5] Input/output error: '~/.local-operator/run'"),
            ),
            captured_at=CAPTURED_AT,
        )

    if scenario == "nested":
        # A depth>=2 tree with the cap exercised, plus a fleet and a build skew.
        return InfoSnapshot(
            install=_install(latest_known="0.52.0", behind=True, latest_age_s=1_200.0),
            process=_process(),
            sessions=SessionsInfo(
                lines=(
                    _session_line(4243, "Investigate request latency", is_self=True, busy=True),
                    _session_line(4244, "Add /move command", pending="approval", version="0.51.5"),
                    _session_line(4245, "OSWorld benchmark", detached=True),
                    _session_line(4246, "Wedged runtime", state="wedged", heartbeat_age_s=310.0),
                ),
                total=4,
                live=3,
                wedged=1,
                busy=1,
                pending=1,
                detached=1,
                build_skew=True,
            ),
            agents=AgentsInfo(
                profiles=20,
                teams=3,
                running=3,
                queued=1,
                settled=7,
                max_running=4,
                at_capacity=False,
                max_depth=4,
                deeper=2,
                tree=(
                    SubagentLine(
                        job_id="j1",
                        label="reviewer",
                        status="running",
                        depth=0,
                        agent_role="reviewer",
                    ),
                    SubagentLine(
                        job_id="j2", label="scout", status="running", depth=1, agent_role="scout"
                    ),
                    SubagentLine(
                        job_id="j3",
                        label="coder",
                        status="completed",
                        depth=2,
                        agent_role="coder",
                    ),
                    SubagentLine(job_id="j4", label="qa-tester", status="queued", depth=0),
                    SubagentLine(job_id="j5", label="designer", status="failed", depth=0),
                    SubagentLine(job_id="j6", label="architect", status="completed", depth=0),
                ),
            ),
            env=_env(),
            captured_at=CAPTURED_AT,
        )

    # populated: the reference frame.
    return InfoSnapshot(
        install=_install(),
        process=_process(),
        sessions=SessionsInfo(
            lines=(
                _session_line(4243, "Investigate request latency", is_self=True),
                _session_line(4244, "Add /move command", busy=True),
                _session_line(4245, "OSWorld benchmark"),
            ),
            total=3,
            live=3,
            busy=1,
        ),
        agents=AgentsInfo(
            profiles=20,
            teams=3,
            running=1,
            queued=0,
            settled=4,
            max_running=4,
            max_depth=1,
            tree=(
                SubagentLine(
                    job_id="j1", label="reviewer", status="running", depth=0, agent_role="reviewer"
                ),
                SubagentLine(job_id="j2", label="scout", status="completed", depth=1),
            ),
        ),
        env=_env(),
        captured_at=CAPTURED_AT,
    )


def live_for(scenario: str) -> LiveState:
    """The pre-yield live capture the screen paints its first frame from."""
    snapshot = snapshot_for(scenario)
    agents = snapshot.agents if snapshot else AgentsInfo()
    return LiveState(
        session_id=SESSION_ID,
        conversation_name="Investigate request latency",
        model_label="anthropic/claude-opus-5",
        effective_model="anthropic/claude-opus-5",
        kind="tui",
        theme="dusk",
        terminal_size=(100, 30),
        approval_mode="ask",
        tree=agents.tree,
        running=agents.running,
        queued=agents.queued,
        settled=agents.settled,
        max_running=agents.max_running,
        max_depth=agents.max_depth,
        deeper=agents.deeper,
    )


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = sys.argv[2].split("x")
    size = (int(cols), int(rows))
    scenario = sys.argv[3] if len(sys.argv) > 3 else "populated"

    session = FakeSession()
    session.set_conversation_name("Investigate request latency")
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # Drive the REAL slash path through the real editor, so the capture
        # exercises registration, dispatch and the push, not just the renderer.
        await _submit(pilot, app, "/info")
        await pilot.pause()
        screen = app.screen
        if not isinstance(screen, InfoScreen):
            raise SystemExit(f"/info did not open InfoScreen (got {type(screen).__name__})")

        # Substitute the synthetic snapshot for whatever the worker found on
        # this machine: a frame that goes on a PR must not carry real data.
        await app.workers.wait_for_complete()
        screen.live = live_for(scenario)
        snapshot = snapshot_for(scenario)
        if snapshot is None:
            screen.snapshot = None
            screen._repaint()
        else:
            screen.set_snapshot(snapshot)
        await pilot.pause()

        save_capture(app, str(out / "opened.svg"))
        await pilot.pause()
        # A SECOND settled frame: identical to the first proves no post-paint
        # reflow and no animation (AGENTS.md §5).
        save_capture(app, str(out / "settled.svg"))

        for page in range(1, 5):
            await pilot.press("pagedown")
            await pilot.pause()
            save_capture(app, str(out / f"page-{page}.svg"))
        await pilot.press("end")
        await pilot.pause()
        save_capture(app, str(out / "bottom.svg"))

        scroll = getattr(screen, "_scroll", None)
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "scenario": scenario,
            "size": list(size),
            "screen": type(screen).__name__,
            "screen_geometry": {
                "size": list(screen.size),
                "virtual_size": list(screen.virtual_size),
                # The screen must NOT scroll: the body does. A screen-level
                # scrollbar costs two cells and reflows the transcript behind.
                "screen_scrolls": list(screen.virtual_size) != list(screen.size),
                "vertical_scrollbar": bool(screen.show_vertical_scrollbar),
            },
            "card_width": screen._card_width(),
            "hint": screen._info_hint(),
            "prompts": session.prompts,
            "scroll": (
                {
                    "size": list(scroll.size),
                    "virtual_size": list(scroll.virtual_size),
                    "max_x": scroll.max_scroll_x,
                    "max_y": scroll.max_scroll_y,
                }
                if scroll
                else None
            ),
        }

        await pilot.press("escape")
        await pilot.pause()
        save_capture(app, str(out / "closed.svg"))
        metrics["composer_focused_after_close"] = app.focused is app.query_one(Editor)
        (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics))


if __name__ == "__main__":
    asyncio.run(main())
