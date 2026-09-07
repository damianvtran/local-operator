"""``/info``'s rendered screen: useful before the worker, honest afterwards.

Assertions are over ``render_lines_for_test()`` — the plain strings a user
actually reads — rather than over the ``Text`` objects, following
``UsagePanel``/``SessionPickerScreen``'s existing test helpers. The pilot tests
at the bottom drive the REAL ``OperatorApp`` (the one that loads
``local_operator.tcss``), because a lightweight host declares no ``CSS_PATH``
and could not show a stylesheet problem at all.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from textual.binding import Binding

from local_operator.info.collect import LiveState, collect_live
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
from local_operator.tui.widgets.info_panel import (
    INFO_COPY_KEY,
    InfoScreen,
    build_info_report,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _boot, _submit


def _live(**overrides: object) -> LiveState:
    base: dict[str, Any] = dict(
        session_id="a3f9c21b7e40",
        conversation_name="Investigate request latency",
        model_label="anthropic/claude-opus-5",
        kind="tui",
        theme="dusk",
        terminal_size=(100, 30),
        approval_mode="ask",
    )
    base.update(overrides)
    return LiveState(**base)


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
                    cwd="/tmp/work",
                    model_label="anthropic/claude-opus-5",
                    uptime_s=840.0,
                    rss_bytes=190_000_000,
                    is_self=True,
                ),
            ),
            total=1,
            live=1,
        ),
        agents=AgentsInfo(profiles=20, teams=3),
        env=EnvInfo(theme="dusk", term="xterm-256color", terminal_size=(100, 30), guides=2),
        captured_at=time.time(),
    )
    base.update(overrides)
    return InfoSnapshot(**base)


def _binding_keys() -> set[str]:
    """Declared keys, however each BINDINGS entry is spelled.

    Textual accepts both ``Binding`` objects and bare tuples in a ``BINDINGS``
    list, so reading ``.key`` off every entry is only correct by accident of
    what this class happens to use today.
    """
    keys: set[str] = set()
    for binding in InfoScreen.BINDINGS:
        keys.add(binding.key if isinstance(binding, Binding) else binding[0])
    return keys


def _lines(snapshot: InfoSnapshot | None, live: LiveState | None = None, width: int = 83):
    return build_info_report(snapshot, live or _live(), width).plain.split("\n")


def _text(snapshot: InfoSnapshot | None, live: LiveState | None = None, width: int = 83) -> str:
    return "\n".join(_lines(snapshot, live, width))


# -- the loading frame --------------------------------------------------------


def test_loading_frame_is_already_useful() -> None:
    """Invariant #15: the live half paints before the ~900 ms probe returns.

    This is what makes push-screen-then-fill correct rather than merely fast:
    the frame the user gets immediately already answers "which session am I in
    and on what model".
    """
    body = _text(None)
    assert "checking…" in body
    assert "a3f9c21b7e40" in body
    assert "anthropic/claude-opus-5" in body
    # Every section header is present from frame one; a missing section reads as
    # a rendering bug rather than as a pending read.
    for header in ("Install", "This session", "Agents and subagents", "Environment"):
        assert header in body


def test_all_default_snapshot_renders_without_raising() -> None:
    """Invariant #17: an all-defaults snapshot is a screen, not a traceback."""
    body = _text(InfoSnapshot())
    assert "Install" in body and "Environment" in body
    assert "unavailable" in body


# -- the populated frame ------------------------------------------------------


def test_populated_frame_carries_the_identifying_facts() -> None:
    body = _text(_snapshot())
    assert "0.51.6" in body
    assert "uv tool" in body or "uv-tool" in body
    assert "Install path" in body
    assert "1 live · 1 total" in body
    assert "this session" in body


def test_the_self_row_says_so() -> None:
    """A one-row list otherwise raises "is that me, or someone else?"."""
    assert "this session" in _text(_snapshot())


def test_config_and_cache_directories_are_shown_side_by_side() -> None:
    """The divergence AGENTS.md documents is only visible if both are drawn."""
    body = _text(_snapshot())
    assert "Config dir" in body and "Cache dir" in body


def test_the_version_row_never_claims_up_to_date_from_an_unknown() -> None:
    """Three distinct states, never collapsed into two."""
    unknown = _snapshot(
        install=InstallInfo(version="0.51.6", kind="pip", latest_known=None, latest_age_s=None)
    )
    assert "never checked" in _text(unknown)

    stale = _snapshot(
        install=InstallInfo(
            version="0.51.6", kind="pip", latest_known="0.51.6", latest_age_s=9 * 60 * 60
        )
    )
    assert "stale" in _text(stale)


def test_an_available_update_renders_the_remedy_whole() -> None:
    """``— /update`` is dropped WHOLE when it does not fit: ``/upd…`` is an
    instruction nobody can follow."""
    behind = _snapshot(
        install=InstallInfo(
            version="0.51.6", kind="uv-tool", latest_known="0.52.0", behind=True, latest_age_s=60.0
        )
    )
    wide = _text(behind, width=120)
    assert "latest is v0.52.0 — /update" in wide

    narrow = _text(behind, width=40)
    assert "/upd…" not in narrow
    assert "v0.52.0" in narrow


# -- the subagent tree --------------------------------------------------------


def test_tree_renders_nested_labels_indented_by_depth() -> None:
    """Invariant #16, and the ``└``-only alphabet §5.2 specifies."""
    agents = AgentsInfo(
        running=2,
        max_depth=2,
        tree=(
            SubagentLine(job_id="j1", label="reviewer", status="running", depth=0),
            SubagentLine(job_id="j2", label="scout", status="running", depth=1),
            SubagentLine(job_id="j3", label="coder", status="completed", depth=2),
        ),
    )
    lines = _lines(_snapshot(agents=agents))
    reviewer = next(line for line in lines if "reviewer" in line)
    scout = next(line for line in lines if "scout" in line)
    coder = next(line for line in lines if "coder" in line)

    assert reviewer.index("reviewer") < scout.index("scout") < coder.index("coder")
    assert "└" in scout and "└" in coder
    # No ├ and no │ continuation runs, at any depth: that is a match to how
    # /analytics and /session already draw their trees.
    assert "├" not in "\n".join(lines)
    assert "│" not in "\n".join(lines)


def test_tree_depth_overflow_is_summarised_not_indented_off_the_card() -> None:
    agents = AgentsInfo(
        running=1,
        max_depth=5,
        deeper=3,
        tree=(SubagentLine(job_id="j1", label="reviewer", status="running", depth=0),),
    )
    assert "+3 deeper" in _text(_snapshot(agents=agents))


def test_empty_tree_reads_as_a_fact_not_as_a_broken_screen() -> None:
    """Invariant: the header is present, the meta is a WORD, the body explains."""
    body = _text(_snapshot(agents=AgentsInfo(profiles=20, teams=3)))
    assert "Agents and subagents" in body
    assert "none running" in body
    assert "0 running" not in body  # a zero in a count column reads as a failure
    assert "No subagents have been launched in this session." in body


def test_the_tree_section_states_the_cross_session_boundary() -> None:
    """Invariant #19: a tree on screen must not read as a fleet-wide view."""
    agents = AgentsInfo(
        running=1, tree=(SubagentLine(job_id="j", label="reviewer", status="running"),)
    )
    body = _text(_snapshot(agents=agents))
    assert "only this session's is visible" in body


def test_settled_is_labelled_as_retained() -> None:
    """A count that silently under-reports must say what it counts."""
    agents = AgentsInfo(running=1, queued=0, settled=4)
    assert "settled (retained)" in _text(_snapshot(agents=agents), width=120)


def test_at_capacity_is_surfaced() -> None:
    agents = AgentsInfo(running=4, max_running=4, at_capacity=True)
    assert "at capacity" in _text(_snapshot(agents=agents), width=120)


# -- degraded states ----------------------------------------------------------


def test_an_unavailable_section_replaces_its_header() -> None:
    """``/session``'s proven ``Ledger unavailable`` shape."""
    body = _text(_snapshot(sessions=SessionsInfo(available=False)))
    assert "Sessions unavailable" in body
    assert "Could not scan the session registry" in body


def test_degraded_probes_are_named_with_their_reason() -> None:
    snapshot = _snapshot(degraded=(("env.browser", "OSError: no such file"),))
    body = _text(snapshot)
    assert "Could not read" in body
    assert "env.browser" in body
    assert "OSError" in body


def test_a_field_that_could_not_be_read_says_unavailable_not_blank() -> None:
    body = _text(_snapshot(install=InstallInfo()))
    assert "unavailable" in body
    # Never these: a missing row is worse than an unavailable one, because the
    # reader cannot tell whether the field does not apply or the screen broke.
    assert "None" not in body
    assert "N/A" not in body


def test_mcp_settling_says_connecting_and_not_a_failure_tally() -> None:
    """Invariant #18: naming a server failed mid-handshake makes a false bug."""
    env = EnvInfo(mcp_configured=3, mcp_connected=1, mcp_failed=2, mcp_settling=True)
    body = _text(_snapshot(env=env), width=120)
    assert "still connecting" in body
    assert "2 failed" not in body


def test_settled_mcp_failures_are_named_with_their_message() -> None:
    env = EnvInfo(
        mcp_configured=2,
        mcp_connected=1,
        mcp_failed=1,
        mcp_settling=False,
        mcp_failures=(("github", "command not found: gh"),),
    )
    body = _text(_snapshot(env=env), width=120)
    assert "1 failed" in body
    assert "github: command not found: gh" in body


def test_build_skew_across_live_sessions_is_called_out() -> None:
    sessions = SessionsInfo(
        lines=(SessionLine(pid=1, state="live", conversation_name="a"),),
        total=1,
        live=1,
        build_skew=True,
    )
    assert "more than one build" in _text(_snapshot(sessions=sessions))


# -- width behaviour ----------------------------------------------------------


@pytest.mark.parametrize("width", [38, 50, 65, 83, 128, 160])
def test_no_line_exceeds_the_card_at_any_width(width: int) -> None:
    """The acceptance criterion for the layout: nothing overflows the card.

    A row that overflows folds onto a second unindented line, which reads as a
    separate record — the fault every shed rule on these screens exists to
    avoid.
    """
    from rich.cells import cell_len

    for line in _lines(_snapshot(), width=width):
        assert cell_len(line) <= max(38, width), repr(line)


@pytest.mark.parametrize("width", [60, 65, 70, 83, 100, 128])
def test_a_session_rows_meta_never_crops_mid_item(width: int) -> None:
    """The acceptance criterion for §3.4, and a real defect found by a frame.

    A session meta is a ``·``-joined list, so a plain crop lands MID-ITEM: at 65
    cells the self row ended ``… · b``, half of ``busy``. That is the ``8.8k
    thin`` fault the existing ``/session`` screen has, and the whole reason
    ``/info`` uses a shed ladder rather than a truncate. Dropping a whole item
    is legible; half of one is not.
    """
    sessions = SessionsInfo(
        lines=(
            SessionLine(
                pid=4243,
                state="live",
                kind="tui",
                conversation_name="Investigate request latency",
                uptime_s=840.0,
                rss_bytes=190_000_000,
                busy=True,
                is_self=True,
            ),
        ),
        total=1,
        live=1,
    )
    row = next(
        line for line in _lines(_snapshot(sessions=sessions), width=width) if "Investigate" in line
    )
    tail = row.rstrip()
    # Every rendered item is complete: no trailing separator, and no partial
    # word after the last one.
    assert not tail.endswith("·")
    for fragment in ("this session", "228 MB", "190 MB", "busy", "14m"):
        # A fragment either appears whole or not at all — never a prefix of it.
        for cut in range(1, len(fragment)):
            partial = fragment[:cut]
            if tail.endswith(" " + partial) or tail.endswith("· " + partial):
                raise AssertionError(f"{fragment!r} cropped to {partial!r} at width {width}")


def test_narrow_frames_shed_notes_rather_than_cropping_them_mid_word() -> None:
    """The existing ``/session`` screen crops ``8.8k thin`` at 80 columns.

    ``/info`` must not inherit that: below ``_NOTE_MIN`` a plain note is shed
    whole and a laddered note falls to its shortest rung.
    """
    narrow = _text(_snapshot(), width=45)
    assert "non-editable; `lop-update` rebuilds it" not in narrow
    # The FACT survives at every width even when its qualifier does not.
    assert "0.51.6" in narrow


# -- the screen ---------------------------------------------------------------


def test_screen_render_lines_for_test_matches_the_builder() -> None:
    screen = InfoScreen(_live(), _snapshot())
    assert any("0.51.6" in line for line in screen.render_lines_for_test())


def test_copy_key_is_the_shared_constant_not_a_second_literal() -> None:
    """One gesture for "copy this surface out" across the app."""
    from local_operator.tui.widgets.aside_panel import ASIDE_COPY_KEY

    assert INFO_COPY_KEY == ASIDE_COPY_KEY == "ctrl+r"
    assert INFO_COPY_KEY in _binding_keys()


def test_screen_has_no_metric_toggle() -> None:
    """``/info`` has nothing to plot, so ``t`` must not be advertised or bound."""
    assert "t" not in _binding_keys()


# -- pilot: the real app ------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_info_opens_and_closes_the_screen() -> None:
    """Invariant #20, on the real ``OperatorApp`` and through the real editor."""
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/info")
        screen = app.screen
        assert isinstance(screen, InfoScreen)

        # The frame is useful BEFORE the worker lands.
        assert any("Install" in line for line in screen.render_lines_for_test())

        await app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.snapshot is not None

        await pilot.press("escape")
        await pilot.pause()
        assert type(app.screen).__name__ != "InfoScreen"


@pytest.mark.asyncio
async def test_slash_info_with_an_argument_is_refused_without_a_screen() -> None:
    """The ``_cmd_analytics``/``_cmd_session`` rejection rule: nothing ran, so
    the boot composition must survive it."""
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/info extra")
        assert type(app.screen).__name__ != "InfoScreen"
        notices = [
            block._text
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        assert any("takes no arguments" in text for text in notices)


@pytest.mark.asyncio
async def test_the_screen_does_not_make_the_app_screen_scrollable() -> None:
    """Design acceptance #6: the BODY scrolls, the screen does not.

    A screen-level scrollbar costs two cells of width and reflows the transcript
    behind the overlay — a pre-existing fault a previous round found this way.
    """
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/info")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.virtual_size == app.screen.size


@pytest.mark.asyncio
async def test_two_consecutive_settled_frames_are_identical() -> None:
    """Design acceptance #5: no post-paint reflow and no animation.

    A first frame that differs from the settled frame is motion the user sees,
    and it would make two screenshots of one state disagree.
    """
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/info")
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, InfoScreen)
        first = screen.render_lines_for_test()
        await pilot.pause()
        await pilot.pause()
        assert screen.render_lines_for_test() == first


@pytest.mark.asyncio
async def test_copy_puts_the_redacted_export_on_the_clipboard() -> None:
    from pathlib import Path

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/info")
        await app.workers.wait_for_complete()
        await pilot.pause()

        copied: list[str] = []
        app.copy_to_clipboard = lambda text: copied.append(text)  # type: ignore[method-assign]
        await pilot.press(INFO_COPY_KEY)
        await pilot.pause()

        assert copied, "ctrl+r must write to the clipboard, not silently no-op"
        payload = copied[0]
        assert "local-operator" in payload
        assert "## Install" in payload
        assert str(Path.home()) not in payload


@pytest.mark.asyncio
async def test_live_capture_reads_the_session_subagent_graph() -> None:
    """The tree comes from the LIVE capture, so it is on the first frame."""

    class _Node:
        job_id = "j1"
        label = "reviewer"
        parent_job_id = None
        status = "running"
        agent_role = "reviewer"
        effort = ""
        session_id = None
        live = True

    class _Comms:
        def nodes(self):
            return [_Node()]

    session = FakeSession()
    session.subagent_comms = _Comms()  # type: ignore[attr-defined]
    live = collect_live(session, theme="dusk", size=(100, 30))
    assert live.running == 1
    assert [node.label for node in live.tree] == ["reviewer"]
    assert "reviewer" in _text(None, live)


# -- the resolved import path -------------------------------------------------


def test_a_matching_import_path_is_shown_without_a_warning() -> None:
    healthy = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            prefix="/opt/uv/tools/local-operator",
            import_path="/opt/uv/tools/local-operator/lib/python3.12/"
            "site-packages/local_operator",
            import_path_foreign=False,
        )
    )
    body = _text(healthy, width=120)
    assert "Running code" in body
    assert "shadowing" not in body
    assert "not under the install path" not in body


def test_a_shadowed_install_is_flagged_as_a_warning_not_an_unavailable() -> None:
    """The ``launch.py:287`` trap made visible.

    ``/reload`` does not fix it and every other field on the screen still
    describes the install, so this must be loud rather than dim: the reader is
    being told that the code executing is not the code they think it is.
    """
    shadowed = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            prefix="/opt/uv/tools/local-operator",
            import_path="/Users/x/repos/local-operator/local_operator",
            import_path_foreign=True,
        )
    )
    body = _text(shadowed, width=120)
    assert "not under the install path" in body
    assert "shadowing the install" in body
    # Not treated as a failed read: nothing failed, and calling it unavailable
    # would understate a state the user has to act on.
    assert "Running code           unavailable" not in body


def test_an_editable_checkout_is_named_as_expected_rather_than_alarming() -> None:
    """The one legitimate divergence: an editable install IS the checkout."""
    editable = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="editable",
            prefix="/Users/x/repos/local-operator/.venv",
            import_path="/Users/x/repos/local-operator/local_operator",
            import_path_foreign=True,
        )
    )
    body = _text(editable, width=120)
    assert "expected for an editable checkout" in body
    assert "shadowing" not in body


# -- inherited invariant: an absent measurement is never a measured zero ------


def test_an_absent_measurement_is_never_rendered_as_a_measured_zero() -> None:
    """``/session``'s standing invariant, inherited explicitly.

    ``session_panel``'s ``_timing_rows`` / ``_gauge_row`` / ``_group_rows`` all
    encode one rule — "an absent sample is not a fast one" — and it is exactly
    as load-bearing here, because most of this screen's fields are read from
    subsystems that legitimately answer "I do not know":

    * a cached-only PyPI read on a COLD cache is ``None``, and must say
      ``unknown``; printing "latest" from it tells a user on a broken network
      they are up to date;
    * an unreadable ``.lop-source`` is ``None``, and must not become an empty
      source line that reads as a PyPI wheel;
    * an unmeasurable build age is ``None``, and must not render as ``0s ago``,
      which claims the install was written this second.

    The failure mode this guards is a PLAUSIBLE wrong answer, which is worse
    than a visible gap: a reader cannot tell a real zero from a failed read.
    """
    unknown = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            latest_known=None,
            latest_age_s=None,
            build_age_s=None,
            is_git_snapshot=True,
            source_ref="",
        )
    )
    body = _text(unknown, width=120)

    assert "never checked" in body
    # None of the zero-shaped spellings of "we did not measure it".
    for forbidden in ("0s ago", "checked 0s ago", "0m ago", "latest · checked"):
        assert forbidden not in body, forbidden
    # The build-age row is OMITTED rather than rendered as a zero age, and the
    # source row still names the snapshot without inventing a ref.
    assert "Built" not in body
    assert "git snapshot" in body
    assert "@ " not in body  # no empty ref stub


def test_a_zero_that_was_actually_measured_still_renders() -> None:
    """The invariant is about ABSENCE, not about suppressing real zeros.

    A machine with genuinely no MCP servers configured, or a genuinely fresh
    build, must still say so — otherwise the rule above would turn into a second
    way of hiding facts.
    """
    measured = _snapshot(
        install=InstallInfo(version="0.51.6", kind="pip", build_age_s=0.0),
        env=EnvInfo(mcp_configured=0),
    )
    body = _text(measured, width=120)
    assert "Built" in body
    assert "none configured" in body


def test_unknown_uptime_is_not_rendered_as_zero_elapsed() -> None:
    unknown = _snapshot(process=_snapshot().process.__class__(session_id="x", uptime_s=None))
    assert "0s ago" not in _text(unknown, width=120)


def test_the_shadowed_row_explains_why_a_passive_row_earns_its_space() -> None:
    """The failure is SILENT and TOTAL, so the row has to say so.

    Confirmed live rather than hypothesised: several sessions on this machine
    run with a cwd inside a checkout of this repo, and
    ``session/runtime/launch.py`` spawns with ``-m`` and no ``cwd=``.
    """
    shadowed = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            prefix="/opt/uv/tools/local-operator",
            import_path="/Users/x/repos/local-operator/local_operator",
            import_path_foreign=True,
        )
    )
    body = _text(shadowed, width=120)
    assert "Nothing will error" in body
    assert "/reload does not" in body


def test_an_editable_checkout_gets_no_alarm_row() -> None:
    """The benign divergence keeps its explanatory note and loses the warning."""
    editable = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="editable",
            prefix="/Users/x/repos/local-operator/.venv",
            import_path="/Users/x/repos/local-operator/local_operator",
            import_path_foreign=True,
        )
    )
    body = _text(editable, width=120)
    assert "expected for an editable checkout" in body
    assert "not under the install path above" not in body
    assert "Nothing will error" not in body


class _StubApp:
    """The app surface these actions touch. Not an `OperatorApp`.

    The pilot tests below drive the real app; these three assert a decision the
    action makes before it reaches the app, so a stub keeps them fast and makes
    the assertion about the action rather than about the frame.
    """

    def __init__(
        self,
        refreshes: list[object],
        notices: list[tuple[str, str]] | None = None,
    ) -> None:
        self._refreshes = refreshes
        self._notices = notices if notices is not None else []

    def refresh_info_screen(self, screen: object) -> None:
        self._refreshes.append(screen)

    def _system_notice(self, message: str, kind: str = "info") -> None:
        self._notices.append((message, kind))

    def bell(self) -> None:
        pass

    def _put_on_clipboard(self, text: str, owner: object | None = None) -> None:
        pass


def test_currency_is_never_claimed_from_an_unreadable_installed_version() -> None:
    """M3: the unknown moved to the INSTALLED side and the lie came back.

    `is_behind("", latest)` is False by design, so a failed version probe beside
    a cached NEWER release reported currency on both surfaces. The three-state
    rule is about not collapsing "unknown" into "up to date" — it does not
    matter which half of the comparison is the unknown one, and both halves fail
    together on a broken install, which is the state /info is opened in.
    """
    from local_operator.info.render import build_export

    broken = _snapshot(
        install=InstallInfo(version="", kind="uv-tool", latest_known="0.52.0", latest_age_s=60.0)
    )
    body = _text(broken, width=120)
    export = build_export(broken)

    assert "unknown" in body
    assert "unknown" in export
    for surface, name in ((body, "screen"), (export, "export")):
        assert "latest ·" not in surface, name
        assert "up to date" not in surface, name


@pytest.mark.asyncio
async def test_refresh_returns_worker_filled_fields_to_the_loading_state() -> None:
    """U2: `r` must acknowledge within a frame, not after the ~4 s probe.

    Driven through the real app and the real binding: `Screen.app` is a
    read-only property, so this cannot be asserted against a stub, and the
    keypress is the thing under test anyway.
    """
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/info")
        screen = app.screen
        assert isinstance(screen, InfoScreen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.snapshot is not None

        await pilot.press("r")
        # Immediately after the keypress, BEFORE the worker can land.
        assert screen.snapshot is None, "the screen must drop to `checking…` at once"
        assert any("checking…" in line for line in screen.render_lines_for_test())

        await app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.snapshot is not None, "and refill when the probe returns"


@pytest.mark.asyncio
async def test_copying_before_the_probe_lands_says_so_rather_than_only_belling() -> None:
    """U3: on a terminal with the bell off, the refusal was completely silent."""
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/info")
        screen = app.screen
        assert isinstance(screen, InfoScreen)
        screen.snapshot = None  # the loading window, deterministically

        copied: list[str] = []
        app.copy_to_clipboard = lambda text: copied.append(text)  # type: ignore[assignment]
        notices: list[str] = []
        original = app._system_notice

        def _record(message: str, kind: str = "info") -> None:
            notices.append(message)

        app._system_notice = _record  # type: ignore[assignment,method-assign]
        try:
            await pilot.press("ctrl+r")
            await pilot.pause()
        finally:
            app._system_notice = original  # type: ignore[assignment,misc]

        assert not copied, "nothing to copy yet"
        assert notices, "a silent refusal lets the user paste stale clipboard content"
        assert "again" in notices[0]


def test_the_shadow_label_sheds_whole_phrases_at_narrow_widths() -> None:
    """U5: `! not und…` at 45 cells communicated nothing at all."""
    shadowed = _snapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            prefix="/opt/uv/tools/local-operator",
            import_path="/Users/x/repos/local-operator/local_operator",
            import_path_foreign=True,
        )
    )
    for width in (45, 55, 65, 83, 120):
        lines = _lines(shadowed, width=width)
        row = next((line for line in lines if line.lstrip().startswith("!")), "")
        if row:
            # Whatever survives must be a WHOLE phrase, never a fragment.
            assert "…" not in row, f"cropped mid-phrase at {width}: {row!r}"
            assert "not" in row and ("install path" in row or "installed tree" in row), row
        else:
            # Below the shortest rung the row is dropped rather than cropped —
            # `! not the…` says nothing. The consequence must still be on the
            # screen, which is what makes dropping it acceptable rather than a
            # silent loss of the warning.
            # Measured threshold: the row survives at 70 (shorter rung) and 83
            # (full phrase) and is dropped below 70, where the widest meta's
            # reservation leaves the label under its shortest rung.
            assert width < 70, f"the row should still fit at {width}"
        body = "\n".join(lines)
        assert "Nothing will error" in body, f"the consequence vanished at {width}"


def test_the_terminal_row_says_checking_rather_than_unavailable_while_loading() -> None:
    """U6: `—` means "we looked and could not tell" everywhere else here."""
    loading = _text(None, width=100)
    terminal_row = next(line for line in loading.split("\n") if "Terminal" in line)
    assert "checking…" in terminal_row
    assert "—" not in terminal_row


def test_the_unavailable_sessions_advice_names_the_key_that_helps() -> None:
    """U7: `r` re-runs these probes and is advertised two rows below."""
    body = _text(_snapshot(sessions=SessionsInfo(available=False)), width=100)
    assert "Press r to try again" in body
    assert "Close and reopen" not in body
