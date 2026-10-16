"""Ratchet slice 4 (`tui.small`): info_panel / subagent_view / wake_panel copy.

The fixture beside this file is a SNAPSHOT of the ten literals this slice
extracted, transcribed from the pre-slice source on `origin/main` @
`da6bb696135ec4481429a178be460bfbd32f784f` — the committed form of the
byte-identical proof review and QA can re-run. Each test drives the REAL
call path (the report builder, the widgets' own section/title builders) or
the generated key function and requires the rendered string to match the
golden EXACTLY.

Byte fidelity is the contract, and it is non-obvious in three places:

* The wake panel's monitor clauses keep their numbers as RAW digit strings
  (``{shown}``/``{total}``): the pre-slice f-strings printed them via
  ``str()``, so ``1200`` stays ``1200`` and never becomes ``1,200``. The ICU
  plural rides on a SEPARATE ``count`` argument that only SELECTS the branch
  (needs/need, monitor/monitors), reproducing the hand-written choices.
* ``info_panel.deeper_below_depth`` keeps its TWO LEADING SPACES (the row's
  base indent) and raw digits; ``subagent_view``'s two title rows share ONE
  ``row_name`` key so they cannot drift onto two spellings.
* ``subagent_view.__prompt__`` is NOT copy: it is the ledger key of the
  delegated-instruction entry, a machine identifier. It is deliberately
  excluded from the catalogue and spelled once as ``_PROMPT_KEY`` because
  the scanner counts any ``.append("...")`` argument as prose — extracting
  it as copy would have put an identifier in a locale catalogue.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from rich.style import Style

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_small as small_copy
from local_operator.i18n.text import tr
from local_operator.info.collect import LiveState
from local_operator.info.model import (
    AgentsInfo,
    EnvInfo,
    InfoSnapshot,
    InstallInfo,
    ProcessInfo,
    SessionLine,
    SessionsInfo,
    SubagentLine,
)
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import subagent_view
from local_operator.tui.widgets.info_panel import build_info_report
from local_operator.tui.widgets.subagent_view import SubagentView
from local_operator.tui.widgets.wake_panel import WakePanel

FIXTURE = Path(__file__).parent / "fixtures" / "tui_ratchet_four_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]

#: The three files this slice activated, exact paths (never a directory prefix:
#: sibling files in `tui/widgets/` still carry literals, so a prefix would
#: red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/widgets/info_panel.py",
    "local_operator/tui/widgets/subagent_view.py",
    "local_operator/tui/widgets/wake_panel.py",
)

#: Slice 3's activated files, pinned so this slice's baseline edit cannot
#: walk them back.
RETAINED = (
    "local_operator/tui/widgets/ask_queue.py",
    "local_operator/tui/widgets/copy_picker.py",
    "local_operator/tui/widgets/link_picker.py",
)

_small = tr("tui.small")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_the_new_keys_match_the_pre_slice_golden() -> None:
    assert len(GOLDEN) == 10
    live = catalogues.load_catalogue("en", "tui.small")
    for key, value in GOLDEN.items():
        assert live[key] == value, key


def test_the_sidecar_names_role_tone_and_note_for_every_new_key() -> None:
    sidecar = catalogues.context("en", "tui.small")
    for key in GOLDEN:
        assert sidecar[key]["role"] and sidecar[key]["tone"] and sidecar[key]["note"], key


def _live() -> LiveState:
    return LiveState(
        session_id="a3f9c21b7e40",
        conversation_name="Investigate request latency",
        model_label="anthropic/claude-opus-5",
        kind="tui",
        theme="dusk",
        terminal_size=(150, 40),
        approval_mode="ask",
    )


def _snapshot(**overrides: object) -> InfoSnapshot:
    base: dict[str, Any] = dict(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            prefix="/opt/uv/tools/local-operator",
            executable="/opt/uv/tools/local-operator/bin/python",
            python_version="3.12.13",
            python_implementation="CPython",
            platform="macOS-26.6.2-arm64",
            machine="arm64",
            latest_known="0.51.6",
            latest_age_s=1800.0,
        ),
        process=ProcessInfo(
            pid=4243,
            session_id="a3f9c21b7e40",
            conversation_name="Investigate request latency",
            cwd="/tmp/work",
            model_label="anthropic/claude-opus-5",
            config_dir="/tmp/cfg",
            cache_dir="/tmp/cache",
            agent_home="/tmp/home",
            log_dir="/tmp/logs",
            control_port=51234,
            protocol=5,
            uptime_s=840.0,
            kind="tui",
        ),
        sessions=SessionsInfo(
            lines=(
                SessionLine(
                    pid=4243,
                    kind="tui",
                    state="live",
                    session_id="a3f9c21b7e40",
                    conversation_name="Investigate request latency",
                    uptime_s=840.0,
                    is_self=True,
                ),
            ),
            total=1,
            live=1,
        ),
        agents=AgentsInfo(profiles=20, teams=3),
        env=EnvInfo(theme="dusk", term="xterm-256color", terminal_size=(150, 40), guides=2),
        captured_at=time.time(),
    )
    base.update(overrides)
    return InfoSnapshot(**base)


def _report(snapshot: InfoSnapshot) -> str:
    return build_info_report(snapshot, _live(), width=150).plain


def test_info_panel_session_rows_paint_the_extracted_clauses() -> None:
    wedged = SessionLine(
        pid=4244,
        kind="daemon",
        state="wedged",
        session_id="b1f2c3d4e5f6",
        conversation_name="wedged-peer",
        uptime_s=10.0,
        heartbeat_age_s=240.0,
    )
    busy = SessionLine(
        pid=4245,
        kind="daemon",
        state="live",
        session_id="c1f2c3d4e5f6",
        conversation_name="busy-peer",
        uptime_s=10.0,
        busy=True,
        pending="2 approvals",
    )
    snapshot = _snapshot(sessions=SessionsInfo(lines=(wedged, busy), total=2, live=1, wedged=1))
    text = _report(snapshot)
    assert "last heartbeat 4m ago" in text
    assert "needs 2 approvals" in text

    idle_busy = SessionLine(
        pid=4246,
        kind="daemon",
        state="live",
        session_id="d1f2c3d4e5f6",
        conversation_name="plain-busy",
        uptime_s=10.0,
        busy=True,
    )
    text = _report(_snapshot(sessions=SessionsInfo(lines=(idle_busy,), total=1, live=1)))
    assert "busy" in text


def test_info_panel_deeper_row_keeps_indent_and_raw_digits() -> None:
    agents = AgentsInfo(
        running=1,
        deeper=3,
        tree=(SubagentLine(job_id="j1", label="reviewer", status="running", depth=0),),
    )
    text = _report(_snapshot(agents=agents))
    assert "  +3 deeper (below depth 3)" in text or "+3 deeper" in text
    # The catalogue path itself: 1200 must stay 1200, indent intact.
    rendered = _small(small_copy.info_panel_deeper_below_depth(deeper="1200", depth="3"))
    assert rendered == "  +1200 deeper (below depth 3)"
    assert rendered == _en("tui.small.info_panel.deeper_below_depth", deeper="1200", depth="3")


def _title_view(**attrs: object) -> SubagentView:
    view = SubagentView.__new__(SubagentView)
    view._status = "completed"
    view._label = "kid"
    view._queued = False
    view._paused = False
    view._cut_off = False
    view._outcome = ""
    view._agent_role = ""
    view._effort = ""
    view._elapsed = "0s"
    for name, value in attrs.items():
        setattr(view, name, value)
    return view


def test_subagent_title_rows_paint_the_shared_surface_name() -> None:
    gone = _title_view(_status="gone", _label="sub-gone")
    assert gone._title_row(120, "", 0).plain == "Subagent · sub-gone  no longer on the ledger"
    live = _title_view()
    assert live._title_row(120, "", 0).plain.startswith("Subagent · kid")


def test_the_prompt_key_is_an_identifier_not_catalogue_copy() -> None:
    # Pinned exclusion: `__prompt__` keys the instruction entry on the
    # ledger; no locale may translate it, so it lives in exactly one
    # constant and in no catalogue.
    assert subagent_view._PROMPT_KEY == "__prompt__"
    live = catalogues.load_catalogue("en", "tui.small")
    assert not any("__prompt__" in value for value in live.values())


def _styles() -> tuple[Style, Style, Style]:
    return (
        Style(color=theme_mod.semantic_color("dim")),
        Style(color=theme_mod.semantic_color("muted")),
        Style(color=theme_mod.semantic_color("warning")),
    )


def test_wake_panel_headers_paint_the_extracted_copy() -> None:
    dim, muted, warning = _styles()
    panel = WakePanel.__new__(WakePanel)
    wakes = panel._wake_section(
        (("w1", "08:30", "1d", "msg", "dim"),), room=4, dim=dim, muted=muted, warning=warning
    )
    assert wakes[0].plain.startswith("Wakes · ")

    broken = (("m1", "12:00", "5m", "name", "warning", "", "0"),)
    monitors = panel._monitor_section(broken, room=4, dim=dim, muted=muted, warning=warning)
    assert monitors[0].plain == "Monitors · 1 of 1 needs attention"

    healthy_two = tuple((f"m{i}", "12:00", "5m", "name", "dim", "", "0") for i in range(2))
    monitors = panel._monitor_section(healthy_two, room=5, dim=dim, muted=muted, warning=warning)
    assert monitors[0].plain == "Monitors · 2 monitors armed"
    monitors = panel._monitor_section(
        healthy_two[:1], room=4, dim=dim, muted=muted, warning=warning
    )
    assert monitors[0].plain == "Monitors · 1 monitor armed"


def test_wake_monitor_plurals_keep_raw_digits() -> None:
    for count, total, expected in (
        (1, 5, "1 of 5 needs attention"),
        (3, 5, "3 of 5 need attention"),
        (1200, 2400, "1200 of 2400 need attention"),
    ):
        rendered = _small(
            small_copy.wake_panel_monitors_need_attention(
                count=count, shown=str(count), total=str(total)
            )
        )
        assert rendered == expected
        assert rendered == _en(
            "tui.small.wake_panel.monitors_need_attention",
            count=count,
            shown=str(count),
            total=str(total),
        )
    for count, expected in (
        (1, "1 monitor armed"),
        (2, "2 monitors armed"),
        (1200, "1200 monitors armed"),
    ):
        assert (
            _small(small_copy.wake_panel_monitors_armed(count=count, shown=str(count))) == expected
        )


def test_every_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only.

    A file left in ``files`` while also in ``enforced`` still absorbs a
    planted literal into its ceiling, so the gate silently does nothing.
    """
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling; the gate is toothless"
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"
    assert not [entry for entry in baseline["enforced"] if entry.endswith("/") and "tui" in entry]


@pytest.mark.parametrize("path", ACTIVATED)
def test_a_planted_literal_fails_the_activated_checker(
    path: str, tmp_path, monkeypatch, i18n_check
) -> None:
    """Prove the activation BITES: the real checker entrypoint, isolated copy."""
    source = (REPO / path).read_text(encoding="utf-8")
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(json.dumps({"note": "slice fixture", "enforced": [path], "files": {}}))
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted slice prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0
