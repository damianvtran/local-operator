"""Remappable hotkeys, driven through the REAL :class:`OperatorApp`.

The real app rather than a lightweight host, for two reasons beyond the
stylesheet: only the real app declares the bindings under test, and only the
real app can disarm them (``check_action``), which is the whole mechanism
capture rests on.

Every assertion here is STRUCTURAL — which key fired, what is bound, whether a
flag is clear. No timing bound appears anywhere in this file, and the
propagation test drives ``watcher.poll_now()`` directly rather than sleeping,
so the property under test is "the callback rebinds" rather than "it rebinds
within N ms" (AGENTS.md: wait on the event, never on the clock — here by
removing the wait entirely).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import keymap, settings_io
from local_operator.config import ConfigManager
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.settings_view import SettingsView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never the developer's own config: these tests write hotkeys."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_reload

    settings_reload()
    return tmp_path


def _spy(app: OperatorApp) -> list[Any]:
    """Record the two commands the hotkeys fire, without running them.

    ``_cmd_new`` rebuilds the session and ``_cmd_resume`` opens a picker
    screen; neither is what these tests are about, and both would make the
    assertions depend on session machinery rather than on dispatch.
    """
    fired: list[Any] = []

    def _new(notice: Any) -> None:
        fired.append("new")

    def _resume(arg: str, notice: Any) -> None:
        fired.append(("resume", arg))

    app._cmd_new = _new  # type: ignore[method-assign]
    app._cmd_resume = _resume  # type: ignore[method-assign]
    return fired


def _select(view: SettingsView, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return
    raise AssertionError(f"no row for {key}")


@pytest.mark.asyncio
async def test_default_chords_fire_past_a_focused_composer_holding_a_draft() -> None:
    """The measurement shape the aside chords established: the action fires AND
    the draft is byte-identical afterwards.

    The composer holds focus in the common case, so a hotkey that only works
    when focus is elsewhere is a hotkey that never works.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired = _spy(app)
        editor = app._editor()
        editor.focus()
        editor.load_text("half a thought")
        await pilot.pause()

        await pilot.press("ctrl+n")
        assert fired == ["new"]
        assert editor.text == "half a thought"

        fired.clear()
        await pilot.press("ctrl+s")
        assert fired == [("resume", "")]
        assert editor.text == "half a thought"


@pytest.mark.asyncio
async def test_a_remap_moves_the_action_AND_retires_the_old_key() -> None:
    """The negative half is the one that regresses silently.

    A remap that bound the new key while leaving the old one live would pass
    any "does my new key work?" check and leave the user with two keys for one
    action — and the config page saying otherwise.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired = _spy(app)
        app._editor().focus()
        await pilot.pause()

        app._apply_keymap({"keymap.new_session": "ctrl+g"}, announce=False)
        await pilot.pause()

        await pilot.press("ctrl+g")
        assert fired == ["new"]

        fired.clear()
        await pilot.press("ctrl+n")
        assert fired == [], "the OLD key still fires — the remap only added a binding"


@pytest.mark.asyncio
async def test_an_unusable_key_on_disk_leaves_the_default_live() -> None:
    """A config carrying garbage must not disarm the action.

    ``tui/settings.py``'s rule — a missing or unreadable config never breaks
    the TUI — matters more here than anywhere: a hotkey config that disarmed
    itself would take away one of the routes the user has for fixing it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired = _spy(app)
        app._editor().focus()
        await pilot.pause()

        app._apply_keymap({"keymap.new_session": "banana"}, announce=False)
        await pilot.pause()

        await pilot.press("ctrl+n")
        assert fired == ["new"], "a bogus config value left the action unreachable"


@pytest.mark.asyncio
async def test_capture_mode_disarms_app_actions_in_both_directions() -> None:
    """With the gate ON a priority binding must not fire; with it OFF it must.

    Both directions, because a test of the ON half alone passes just as well
    against an app whose bindings never work at all.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert "shift+tab" in app.screen.active_bindings
        assert "ctrl+t" in app.screen.active_bindings

        app._set_keymap_capture(True)
        await pilot.pause()
        assert "shift+tab" not in app.screen.active_bindings
        assert "ctrl+t" not in app.screen.active_bindings

        app._set_keymap_capture(False)
        await pilot.pause()
        assert "shift+tab" in app.screen.active_bindings
        assert "ctrl+t" in app.screen.active_bindings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    ["escape", "commit", "cursor-move", "leave-page", "close-mode", "bare-unmount"],
)
async def test_every_exit_route_clears_capture_mode(route: str) -> None:
    """RISK R1, one case per route.

    A stuck flag does not merely break this feature — it removes every hotkey
    in the app, ``ctrl+c`` included, with nothing on screen saying why. So each
    way out of capture is asserted separately rather than trusting the one the
    happy path takes.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.press("enter")
        await pilot.pause()
        assert app._keymap_capture, "capture did not arm"

        if route == "escape":
            await pilot.press("escape")
        elif route == "commit":
            await pilot.press("ctrl+g")
            await pilot.press("enter")
        elif route == "cursor-move":
            # NOT an arrow key: while capturing, the arrows are KEYS TO BIND
            # like any other (`down` is a legitimate, if unwise, binding), so
            # the gesture that moves the cursor out of a capture is a click on
            # another row or a programmatic move. `action_move` is what both
            # reach, and it routes through `_settle_row`.
            view.action_move(1)
        elif route == "leave-page":
            view._leave()
        elif route == "close-mode":
            # The route a session swap or a `/clear` takes. The page never
            # hears about it, and a message posted from a removed widget is
            # never delivered — so the app has to clear the flag itself.
            app._close_settings_view()
        elif route == "bare-unmount":
            # The widget removed WITHOUT the app's help. No caller does this
            # today — every teardown path goes through `_close_settings_view`
            # — so this asserts the backstop `on_unmount` claims to be, rather
            # than a reachable user gesture. It failed before round 1's M3
            # fix: `on_unmount` cleared the page's own field and left
            # `app._keymap_capture` True, disarming ctrl+c for the rest of the
            # process.
            await view.remove()
        await pilot.pause()

        assert not app._keymap_capture, f"capture leaked out of the {route} route"
        assert "ctrl+t" in app.screen.active_bindings, "app hotkeys stayed disarmed"


@pytest.mark.asyncio
async def test_capture_writes_only_on_the_second_enter(tmp_path: Path) -> None:
    """The page's #440 contract, applied to the new kind: enter opens, enter
    accepts, and the detected key is echoed in between so the accept is made on
    something the user has seen."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+g")
        await pilot.pause()
        # A FRESH manager per read: `ConfigManager` snapshots the file, so a
        # held one would answer from before the write either way and the
        # assertion would pass without the behaviour.
        assert (
            "keymap.new_session" not in ConfigManager(tmp_path).get_config().values
        ), "the first enter wrote config.yml"
        assert "ctrl+g" in view._detail_text.plain

        await pilot.press("enter")
        await pilot.pause()
        setting = settings_io.BY_KEY["keymap.new_session"]
        assert settings_io.read_setting(ConfigManager(tmp_path), setting) == "ctrl+g"


@pytest.mark.asyncio
async def test_the_footer_states_what_the_keys_do_while_capturing() -> None:
    """The D7 class this page already forbids: the footer must not advertise a
    meaning a key does not have.

    While capturing, `esc` cancels rather than leaving the page, the arrows are
    keys to BIND rather than the pane switch, and `r` would capture `r` rather
    than reset — so a footer left reading `move · default · panes · back to
    conversation` would be wrong about four keys at once, in the one state
    where the way out matters most (every app binding is disarmed).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        resting = view.rendered_hints()
        assert "back to conversation" in resting

        await pilot.press("enter")
        await pilot.pause()
        hints = view.rendered_hints()
        assert "esc  cancel" in hints.replace("esc cancel", "esc  cancel")
        assert "back to conversation" not in hints
        assert "panes" not in hints
        assert "default" not in hints

        await pilot.press("escape")
        await pilot.pause()
        assert view.rendered_hints() == resting, "the footer did not return to rest"


@pytest.mark.asyncio
async def test_a_second_key_while_pending_replaces_the_first() -> None:
    """A user who fumbles a chord should press the right one, not cancel and
    re-enter. Correcting is the common case by a wide margin, which is also why
    capture takes exactly one key rather than accumulating alternates."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+g")
        await pilot.press("f5")
        await pilot.pause()
        assert view._capture is not None and view._capture.key == "f5"


@pytest.mark.asyncio
async def test_a_reserved_key_is_refused_with_its_reason_and_capture_stays_open() -> None:
    """Refusal must be VISIBLE. A key that appears to do nothing in capture
    mode is indistinguishable from a terminal that failed to send it — and
    while capture is live the app's own `ctrl+c` is disarmed, so this handler
    is the only thing that can say so."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert view._capture is not None and not view._capture.key
        assert "cannot be bound" in view._detail_text.plain
        assert app._keymap_capture, "a refusal must not end the capture"


@pytest.mark.asyncio
async def test_a_conflicting_key_names_its_victim_and_is_still_allowed() -> None:
    """Warn-and-allow, with the cost stated at the moment of the decision.

    Read from the DECLARED binding map rather than ``active_bindings``: capture
    mode empties the latter by design, so a lookup through it silently found
    nothing and the warning never fired (observed against the real app).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+t")  # the app's own "expand/collapse todos"
        await pilot.pause()
        detail = view._detail_text.plain
        assert "takes ctrl+t" in detail and "todos" in detail.lower()
        assert "enter confirms" in detail, "a soft conflict must still be committable"


@pytest.mark.asyncio
async def test_a_composer_key_warns_even_though_textual_reports_no_clash() -> None:
    """The class Textual cannot see. Its clash detection covers only bindings
    in the same BindingsMap, so a remap onto a TextArea key reports nothing and
    then silently loses to the composer — measured. Without this warning the
    user binds ctrl+u, sees nothing happen, and concludes the feature is
    broken."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+u")
        await pilot.pause()
        assert "composer" in view._detail_text.plain


@pytest.mark.asyncio
async def test_a_settings_page_write_rebinds_THIS_process(tmp_path: Path) -> None:
    """THE local-branch trap, re-run.

    ``_on_config_change`` returns early on ``source == "local"``, and a hotkey
    write from the page in this process is applied by NOTHING else — the page
    stores config, the app holds the BindingsMap. This is verbatim the
    ``tool_approval_mode`` defect that method documents at length: the section
    is labelled LIVE, the page paints that claim, and the pane goes on
    answering the old key. The apply is on both branches; this asserts the one
    that is easy to omit.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired = _spy(app)
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+g")
        await pilot.press("enter")
        await pilot.pause()
        app._close_settings_view()
        app._editor().focus()
        await pilot.pause()

        await pilot.press("ctrl+g")
        assert fired == ["new"], "the page wrote the key and this pane did not rebind"
        fired.clear()
        await pilot.press("ctrl+n")
        assert fired == []


@pytest.mark.asyncio
async def test_a_write_from_another_process_propagates_through_the_watcher(
    tmp_path: Path,
) -> None:
    """Scope LIVE, delivered by the mechanism that already exists.

    Driven through ``poll_now()`` rather than a sleep: the property that
    matters is that the listener rebinds, and polling directly makes it a fact
    about the callback rather than a bet on an interval.
    """
    from local_operator.config_watch import ConfigWatcher

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired = _spy(app)
        app._editor().focus()
        await pilot.pause()

        watcher = ConfigWatcher(tmp_path)
        watcher.subscribe(app._on_config_change)

        # The "other process": a manager of its own writing through the same
        # facade, exactly as a second pane or `lop config edit` would.
        setting = settings_io.BY_KEY["keymap.new_session"]
        settings_io.write_setting(ConfigManager(tmp_path), setting, "ctrl+g")

        change = watcher.poll_now()
        assert change is not None and "keymap.new_session" in change.changed_keys
        await pilot.pause()

        await pilot.press("ctrl+g")
        assert fired == ["new"]
        fired.clear()
        await pilot.press("ctrl+n")
        assert fired == []


@pytest.mark.asyncio
async def test_the_splash_tip_names_the_key_that_is_actually_bound(
    tmp_path: Path,
) -> None:
    """A hardcoded chord in the tip becomes a lie the moment it is remapped,
    on the one screen a first-run user reads word for word. Resolved from the
    PERSISTED value, because ``Binding.key`` still reports the old key."""
    from local_operator.config_watch import ConfigWatcher
    from local_operator.tui.widgets import welcome

    setting = settings_io.BY_KEY["keymap.new_session"]
    settings_io.write_setting(ConfigManager(tmp_path), setting, "ctrl+g")
    ConfigWatcher(tmp_path).poll_now()

    template = keymap.BY_ID["keymap.new_session"].tip
    assert welcome._resolve_tip(template).startswith("ctrl+g")


def test_tip_min_width_has_not_risen_above_its_measured_bound() -> None:
    """ANTI-DRIFT, not a re-measurement.

    ``TIP_MIN_WIDTH`` decides whether the splash draws a tip row AT ALL, and
    the row's presence has to be a function of terminal width alone — a
    threshold that moved with a user's configuration would make the row appear
    and disappear as the reel rotated, shoving the whole splash up and down.
    The keyed templates are budgeted at a worst-case key for that reason, and
    59 is the value the non-keyed pool already set. A future template that
    pushes this up should be SHORTENED rather than the bound raised.
    """
    from local_operator.tui.widgets import welcome

    assert welcome.TIP_MIN_WIDTH == 59


def test_keyed_tip_templates_stay_within_their_width_budget() -> None:
    """The property that keeps the assertion above true as tips are added.

    Only templates that RESERVE a key are budgeted at 32 cells: they are
    measured against a 25-cell worst-case key on top of their own text, so
    they are the only ones that can push the threshold. A keyless entry is
    bounded by the pool's ordinary maximum like any other tip.
    """
    from rich.cells import cell_len

    from local_operator.tui.widgets import welcome

    for template, binding_id in welcome.KEYED_TIPS:
        if binding_id is None:
            assert cell_len(f"{welcome.TIP_GLYPH} {template}") <= welcome.TIP_MIN_WIDTH, template
            continue
        assert cell_len(template.replace("{key}", "")) <= 32, template


def test_composer_keys_has_not_drifted_from_the_editor_bindings() -> None:
    """``COMPOSER_KEYS`` is a hand-maintained mirror; this is what catches drift.

    ``keymap.py`` deliberately does not import the TUI — it is loaded by the
    CLI and the server, neither of which should pay for Textual — so the
    composer table cannot be derived at runtime and is copied by hand. That
    trade is sound, but it leaves nothing failing when ``Editor.BINDINGS``
    gains a key: the new key silently stops producing its "the composer uses
    this" warning, and a user remaps onto it and finds the hotkey dead while
    they are typing, which is the one place a non-priority binding loses
    (review round 1, m3).

    The import cost argument does not apply to a test in ``tests/unit/tui``,
    which already imports the whole app. Only keys that are BINDABLE are
    asserted: the composer also binds `enter`, `escape` and friends, which the
    reserved set refuses outright, so a warning about them is unreachable.
    """
    from local_operator.tui.widgets.editor import Editor

    declared: set[str] = set()
    for binding in Editor.BINDINGS:
        raw = getattr(binding, "key", None) or (binding[0] if isinstance(binding, tuple) else None)
        if not raw:
            continue
        for part in str(raw).split(","):
            declared.add(part.strip())

    bindable = {key for key in declared if key and keymap.validate_key(key) is None}
    missing = sorted(bindable - keymap.COMPOSER_KEYS)
    assert not missing, (
        f"Editor.BINDINGS gained bindable keys that COMPOSER_KEYS does not mirror: {missing}. "
        "Add them there so a remap onto one still warns."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (80, 24), (60, 20), (46, 18)])
@pytest.mark.parametrize("scenario", ["empty", "pending", "conflict", "refused"])
async def test_the_capture_line_always_states_the_way_out(
    size: tuple[int, int], scenario: str
) -> None:
    """The exit survives every width, and a cut carries a mark.

    While capture is live ``check_action`` has disarmed every app binding, so
    this line is the ONLY correct statement of how to leave. It used to be
    appended straight into a `height: 2`, no-wrap widget and clipped without a
    mark: at 80x24 `esc cancels` was cut to `esc`, and at 60x20 the contract
    was gone entirely and the line ended on a dangling `·` (design round 1,
    D1).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.press("enter")
        await pilot.pause()
        if scenario == "pending":
            await pilot.press("f5")
        elif scenario == "conflict":
            await pilot.press("ctrl+t")
        elif scenario == "refused":
            await pilot.press("ctrl+c")
        await pilot.pause()

        painted = view._detail_text.plain
        assert "esc cancels" in painted, f"{scenario} at {size} lost the exit: {painted!r}"
        # A shed must never leave the separator that joined the dropped part.
        assert not painted.rstrip().endswith("\u00b7"), f"dangling separator: {painted!r}"
        assert len(painted) <= view._detail_width() + 1, f"overflowed its width: {painted!r}"


@pytest.mark.asyncio
async def test_the_footer_offers_enter_only_once_a_key_is_pending() -> None:
    """A lit hint whose key does nothing, and here it actively errors.

    ``enter`` is reserved, so pressing it while capture is empty is REFUSED
    and repaints the detail line in danger ink. Advertising it in that phase
    named a key, had the user obey, and gave them an error (design round 1,
    D2). Once a key IS pending, ``enter`` is the last lead shed, so the only
    committing gesture stays advertised down to 46x18 (D4).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(46, 18)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.press("enter")
        await pilot.pause()
        assert not view._enter_hint.display, "enter advertised before a key was pressed"

        await pilot.press("f5")
        await pilot.pause()
        assert view._enter_hint.display, "the commit key was shed at 46x18 while pending"


@pytest.mark.asyncio
async def test_a_pending_key_is_marked_without_relying_on_colour() -> None:
    """Pending and saved differed only by hue (design round 1, D5).

    The other cue was the ABSENCE of the `▸` affordance, and an absence reads
    as "less" rather than as "uncommitted" — neither survives a monochrome
    terminal.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.pause()

        def row() -> str:
            return next(line for line in view.render_lines_for_test() if "New session" in line)

        assert "?" not in row()
        await pilot.press("enter")
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert "?" in row(), "a pending key carries no non-colour mark"


@pytest.mark.asyncio
async def test_a_live_capture_still_names_its_exit_when_the_page_is_too_short() -> None:
    """A modal state with no visible representation must still say `esc`.

    Shrinking while capture is open hides the body but leaves capture live and
    every app binding disarmed, with nothing on screen naming the way out
    (design round 1, D6).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.press("enter")
        await pilot.pause()

        await pilot.resize_terminal(80, 10)
        await pilot.pause()
        assert view._too_short and view._capture is not None
        assert "esc" in view._detail_text.plain, "no exit advertised while capture is hidden"
