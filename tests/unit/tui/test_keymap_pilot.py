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
from tests.unit.tui.conftest import painted_row
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

    def _new(arg: str, notice: Any) -> None:
        # The ARGUMENT is part of the signature since `/new` learned to read it
        # (`/new remote <peer>`); this spy replaces the handler, so it has to keep
        # the call shape the dispatcher uses or the chord's test fails on a
        # signature rather than on the dispatch it is about.
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

        app._apply_keymap({"keymap.new_session": "f3"}, announce=False)
        await pilot.pause()

        await pilot.press("f3")
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
async def test_cleared_desktop_hotkeys_do_not_raise_or_warn() -> None:
    """`none` stored for the dictation rows tolerates `set_keymap`.

    Cleared is distinct from unusable: the resolver keeps `none` and never
    lists the row as rejected, so no "ignoring unusable keymap" notice
    appears and `set_keymap` does not raise (desktop ids have no Binding).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        notices: list[str] = []
        original = app._system_notice

        def _capture(body: str, kind: Any = "info") -> None:
            notices.append(body)
            original(body, kind)

        app._system_notice = _capture  # type: ignore[method-assign]
        resolved, rejected = keymap.resolved_keymap(
            {
                "keymap.push_to_talk": "none",
                "keymap.start_recording": "none",
            }
        )
        assert resolved == {
            "keymap.push_to_talk": "none",
            "keymap.start_recording": "none",
        }
        assert rejected == []
        app.set_keymap(resolved)  # must not raise
        app._apply_keymap(
            {
                "keymap.push_to_talk": "none",
                "keymap.start_recording": "none",
            },
            announce=True,
        )
        await pilot.pause()
        assert not any("ignoring unusable" in notice for notice in notices), notices


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
            await pilot.press("f3")
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
        await pilot.press("f3")
        await pilot.pause()
        # A FRESH manager per read: `ConfigManager` snapshots the file, so a
        # held one would answer from before the write either way and the
        # assertion would pass without the behaviour.
        assert (
            "keymap.new_session" not in ConfigManager(tmp_path).get_config().values
        ), "the first enter wrote config.yml"
        assert "f3" in view._detail_text.plain

        await pilot.press("enter")
        await pilot.pause()
        setting = settings_io.BY_KEY["keymap.new_session"]
        assert settings_io.read_setting(ConfigManager(tmp_path), setting) == "f3"


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
        await pilot.press("f3")
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
@pytest.mark.parametrize("action_id", ["keymap.quick_send", "keymap.start_recording"])
async def test_a_desktop_row_captures_and_validates_by_its_own_rules(
    tmp_path: Path, action_id: str
) -> None:
    """The capture widget dispatches on the ROW's scope, not one global rule.

    The refusal SENTENCE is what proves which grammar ran: the app scope's
    bare-key refusal names the composer, the desktop one names the missing
    modifier. A desktop row judged by the app rules would also refuse its own
    shipped default (`primary+alt+shift+space` is not a key a terminal can send), so
    a regression here would read as "this row can never be edited from the
    terminal". The committed value is read back through a FRESH manager, the
    same reason the other capture test does.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, action_id)
        await pilot.pause()
        if action_id == "keymap.start_recording":
            assert view._detail_text.plain.startswith(
                "Desktop speech-to-text shortcut — set it here or in the app."
            )
        else:
            assert "Global shortcut for the desktop app" in view._detail_text.plain

        await pilot.press("enter")
        await pilot.pause()
        assert app._keymap_capture, "capture did not arm on the desktop row"

        # A bare printable is refused with the DESKTOP sentence, and the
        # capture stays open — a refusal that ended the mode would read as the
        # key having worked.
        await pilot.press("n")
        await pilot.pause()
        detail = view._detail_text.plain
        assert "modifier" in detail and "composer" not in detail, detail
        assert app._keymap_capture, "a refusal must not end the capture"

        # A chord a terminal CAN send is accepted and normalized in the
        # desktop grammar, and the confirming enter writes it.
        await pilot.press("ctrl+alt+space")
        await pilot.pause()
        assert view._capture is not None and view._capture.key == "ctrl+alt+space"
        await pilot.press("enter")
        await pilot.pause()

        setting = settings_io.BY_KEY[action_id]
        assert settings_io.read_setting(ConfigManager(tmp_path), setting) == "ctrl+alt+space"


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [100, 200])
async def test_the_recording_sentence_is_never_painted_faint(width: int) -> None:
    """The recording row's required sentence is never painted `faint`.

    At >=120 cols the shed ladder keeps the help beside the clause and the key
    path, and there it used to take the `faint` rung: #4a4539 on #14110c
    measures 1.97:1 dark / 2.08:1 light, both under the 4.5:1 normal-text
    floor — and that sentence cannot be inferred from anything else on the row.
    Both ends are asserted: at 200 cols the sentence rides the help-beside-key
    rung and must resolve `muted`; at 100 cols it already does, and the width
    where it drops to the help-alone `dim` rung is a later reordering's
    problem, not a silent return to `faint`.
    """
    from rich.color import Color as RichColor

    from local_operator.tui import theme as theme_mod
    from local_operator.tui.widgets.settings_view import _recording_hotkey_detail

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.start_recording")
        await pilot.pause()

        plain = view._detail_text.plain
        # SENTENCE FIRST on a desktop row, and never clipped: the whole string
        # is 60 cells, so even the narrow band holds it.
        assert plain.startswith(_recording_hotkey_detail()), plain

        faint = RichColor.parse(str(theme_mod.semantic_color("faint")))
        muted = RichColor.parse(str(theme_mod.semantic_color("muted")))
        sentence = next(span for span in view._detail_text.spans if span.start == 0)
        assert getattr(sentence.style, "color", None) == muted, view._detail_text.spans
        # No span OVERLAPPING the sentence may carry `faint`: the sentence is
        # spans[0] today, and this still holds if a later change splits it.
        sentence_end = len(_recording_hotkey_detail())
        for span in view._detail_text.spans:
            if span.start >= sentence_end or span.end <= 0:
                continue
            assert (
                getattr(span.style, "color", None) != faint
            ), f"the required sentence carries faint ink: {span}"


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
        await pilot.press("ctrl+home")  # the app's own "Transcript top"
        await pilot.pause()
        detail = view._detail_text.plain
        # The registry's own soft warning owns this class (R4): `ctrl+home` is in
        # `FIXED_APP_KEYS`. It names the VICTIM's job (M3), because there is no
        # setting to rebind a fixed chord by — "pick another" is the only route.
        assert "ctrl+home is fixed for Transcript top" in detail, detail
        # The detail column truncates this sentence at 120 columns ("pick anoth…"):
        # a pre-existing budget on every long conflict note, not this wording's,
        # so the test asserts the victim's job — the part M3 restored.
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

    ``_on_config_change``'s local branch applies a ``source == "local"`` write
    and nothing else does — the page stores config, the app holds the
    BindingsMap — so a hotkey written from the page in this process is applied
    by nothing but that branch. This is verbatim the ``tool_approval_mode``
    defect that method documents at length: the section is labelled LIVE, the
    page paints that claim, and the pane goes on answering the old key. The
    apply is on both branches; this asserts the one that is easy to omit.

    That branch carries a second duty since #1282 — ``"local"`` is also the one
    delivery that may LOOSEN the approval gate, in the process that owns it — so
    "the fast path can be skipped" would now cost an operator action as well as
    a stale hotkey.
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
        await pilot.press("f3")
        await pilot.press("enter")
        await pilot.pause()
        app._close_settings_view()
        app._editor().focus()
        await pilot.pause()

        await pilot.press("f3")
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
        settings_io.write_setting(ConfigManager(tmp_path), setting, "f3")

        change = watcher.poll_now()
        assert change is not None and "keymap.new_session" in change.changed_keys
        await pilot.pause()

        await pilot.press("f3")
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
    settings_io.write_setting(ConfigManager(tmp_path), setting, "f3")
    ConfigWatcher(tmp_path).poll_now()

    template = keymap.BY_ID["keymap.new_session"].tip
    assert welcome._resolve_tip(template).startswith("f3")


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
        # Asserted on the PAINTED row as well as on `display`. The display flag
        # alone is what this test checked in round 1, and it is satisfiable
        # while the row renders wrongly — the seam defect (M5/D8) left this
        # green with the footer reading `any keyenter` (review round 2, M5).
        assert not view._enter_hint.display, "enter advertised before a key was pressed"
        assert "confirm" not in painted_row(app, view._hints)

        await pilot.press("f5")
        await pilot.pause()
        assert view._enter_hint.display, "the commit key was shed at 46x18 while pending"
        assert "confirm" in painted_row(app, view._hints)


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
        await pilot.press("f3")
        await pilot.pause()
        assert "?" in row(), "a pending key carries no non-colour mark"


@pytest.mark.asyncio
async def test_a_capture_does_not_survive_the_page_going_too_short() -> None:
    """The round-1 D6 guard, REWRITTEN to the contract that replaced it.

    D6 asked for a modal state with no visible representation to still name
    `esc`, and round 1 answered by appending the clause to the too-short line.
    Round 2 (D9) showed that clause being clipped away unmarked at exactly the
    widths where it was the only thing on screen, and D10 showed the row not
    painting at all at height 10 — so there is no width at which a string is a
    general answer.

    The state is now impossible rather than advertised, so this asserts the
    stronger property: shrinking ENDS the capture, which is what makes the
    absence of an `esc` clause correct rather than a regression of D6.
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
        assert view._capture is not None

        await pilot.resize_terminal(80, 10)
        await pilot.pause()
        assert view._too_short
        assert view._capture is None, "a capture survived into a page that cannot show it"
        assert app.check_action("interrupt", ()) is not False, "ctrl+c left disarmed"


# ---------------------------------------------------------------------------
# RENDERED-FRAME GUARDS
#
# Everything below asserts on the PAINTED row (`painted_row`, which reads the
# compositor strips) and not on the widget's model `Text`. That distinction is
# the whole point of this block and it is not stylistic: three defects in this
# family have shipped on these two rows, and in every one the model string was
# correct while the frame was wrong, so every substring assertion passed. See
# `painted_row`'s docstring in conftest for the list.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(150, 40), (100, 30), (80, 24), (60, 20), (52, 18), (46, 18)])
async def test_painted_footer_keeps_a_seam_between_every_hint(size: tuple[int, int]) -> None:
    """Hints must never render as one word (design round 2, D8 / review M5).

    Widths span 150x40 down to 46x18 because the reviewer measured the defect
    across that whole range, not only at the three the design round sampled —
    a guard narrower than the defect is a guard that reports a fixed bug.

    The footer paints in DOM order while `plan` carries the SHED order, and
    capture inverts the two (`enter` leads so it is shed last). Suppressing the
    seam by plan position therefore silenced it on a hint that paints second,
    and the row rendered `any keyenter`.

    Asserted on the painted strip because `rendered_hints()` returns the
    concatenation of the hint models, in which the missing seam is invisible.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.press("enter")
        await pilot.press("f5")
        await pilot.pause()

        row = painted_row(app, view._hints)
        assert row, "the footer painted nothing while a key was pending"
        # Every visible hint's label must be preceded by whitespace or a seam,
        # never by another hint's last letter. Checked against the labels
        # actually on screen rather than a fixed list, so the guard survives a
        # future state offering different keys.
        for hint in (view._enter_hint, view._exit_hint, view._reset_hint, view._pane_hint):
            if not hint.display:
                continue
            label = hint._key
            index = row.find(label)
            if index <= 0:
                continue
            preceding = row[index - 1]
            assert (
                not preceding.isalnum()
            ), f"{label!r} is jammed onto the previous hint at {size}: {row!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (80, 24), (60, 20), (46, 18)])
@pytest.mark.parametrize("scenario", ["empty", "pending", "refused"])
async def test_the_painted_capture_line_carries_its_exit_or_an_ellipsis(
    size: tuple[int, int], scenario: str
) -> None:
    """The rendered detail row states the way out, or marks that it was cut.

    The round-1 version of this guard read `_detail_text`, which is the model.
    `.settings-view-detail` is `height: 2` and no-wrap, so the widget clips
    what the model carried — the round-2 blocker (D9) was exactly a clause
    present in the model and absent from the frame at every width where it
    mattered.
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
        elif scenario == "refused":
            await pilot.press("ctrl+c")
        await pilot.pause()

        row = painted_row(app, view._detail)
        assert row, f"the capture line painted nothing at {size}"
        assert "esc cancels" in row, f"{scenario} at {size} painted no exit: {row!r}"
        assert not row.rstrip().endswith("\u00b7"), f"dangling separator: {row!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 12), (64, 12), (60, 12), (46, 12), (46, 10)])
async def test_shrinking_below_the_body_floor_never_strands_a_capture(
    size: tuple[int, int],
) -> None:
    """R1, reached by RESIZE rather than by a key (design round 2, D9/D10).

    Capture disarms every app binding including ctrl+c, so it may only be live
    while the page can say so. Below the body floor it cannot: the row is
    hidden, the detail line is ~42 cells, and at height 10 the detail widget
    paints nothing at all — so the escape clause has no guaranteed home and the
    capture is ended on the way in instead.

    The capture must be entered at a workable size and the terminal shrunk
    AFTER: the shot helpers press `enter` at the target size, where a too-short
    page never enters capture and the frame looks fine.
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
        assert app._keymap_capture, "capture did not arm at a workable size"

        await pilot.resize_terminal(*size)
        await pilot.pause()

        assert view._too_short, f"{size} was expected to be below the body floor"
        visible_exit = painted_row(app, view._detail)
        assert (
            not app._keymap_capture or "esc cancels" in visible_exit
        ), f"capture disarmed the app with no PAINTED exit at {size}: {visible_exit!r}"
        assert view._capture is None, f"a capture survived into too-short at {size}"
        assert not app._keymap_capture, f"the app flag survived into too-short at {size}"
        assert (
            app.check_action("interrupt", ()) is not False
        ), f"ctrl+c is disarmed with no way to say so at {size}"


@pytest.mark.asyncio
async def test_capture_repaints_selected_strip_without_flushing_the_list() -> None:
    """#779's row cache must show every capture phase without global invalidation."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")
        await pilot.pause()
        view._scroll_to_selection(immediate=True)
        await pilot.pause()
        token = view._paint_token()

        def painted_value() -> str:
            # Read the compositor, not `_list_text`, which can rebuild model
            # rows and conceal stale rasterized strips.
            return next(
                strip.text
                for strip in app.screen._compositor.render_strips()
                if "New session" in strip.text
            )

        assert "ctrl+n" in painted_value()
        await pilot.press("enter")
        await pilot.pause()
        assert "press a key" in painted_value()
        await pilot.press("f5")
        await pilot.pause()
        assert "f5 ?" in painted_value()
        await pilot.press("f2")
        await pilot.pause()
        assert "f2 ?" in painted_value() and "f5" not in painted_value()
        assert view._paint_token() == token, "capture flushed the entire list cache"
        await pilot.press("escape")
        await pilot.pause()
        assert "ctrl+n" in painted_value() and "?" not in painted_value()
        await pilot.press("enter", "f5", "enter")
        await pilot.pause()
        assert "f5" in painted_value() and "?" not in painted_value()
        assert not app._keymap_capture


@pytest.mark.asyncio
async def test_a_hold_row_displays_and_refuses_to_arm_capture() -> None:
    """The push-to-talk row: displayed with its platform-resolved hold value,
    and `enter` refuses the capture gesture instead of arming it.

    A hold has no terminal representation at all, so an armed frame could
    never resolve — every completion the validator refuses. The press must
    state the constraint ("set it in the desktop app") on the SAME surface a
    refused write uses, and that surface clears once the cursor moves on.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.push_to_talk")
        # The pilot `_select` deliberately does not scroll; the row is deep in
        # the list, so bring it on screen before reading painted strips.
        view._scroll_to_selection(immediate=True)
        await pilot.pause()

        # The row PAINTS with the platform-resolved hold display.
        expected = keymap.display_key("alt-right-hold", scope="desktop")
        assert expected.endswith("(hold)")
        painted = next(
            strip.text
            for strip in app.screen._compositor.render_strips()
            if "Push to talk" in strip.text
        )
        assert expected in painted, painted

        # `enter` must NOT arm — and it says why, on the detail row.
        await pilot.press("enter")
        await pilot.pause()
        assert not app._keymap_capture, "a hold row must never arm capture"
        assert view._capture is None
        detail = painted_row(app, view._detail)
        assert "desktop app" in detail, detail
        assert "cannot capture" in detail, detail

        # The refusal has `_error`'s lifetime: moving on clears it.
        await pilot.press("down")
        await pilot.pause()
        assert "cannot capture" not in painted_row(app, view._detail)


@pytest.mark.asyncio
async def test_the_palette_opens_on_its_live_key_and_ctrl_p_stops_opening_it() -> None:
    """`ctrl+p` was the one app chord nobody could remap.

    Textual injects its own priority binding on it unless a binding carries
    ``command_palette``; the app declares that action now. Driven with
    ``App.simulate_key`` rather than ``pilot.press``: the pilot synthesises a
    PRINTABLE character for a ctrl+letter (`ctrl+p` arrives with character
    `"p"`), which a focused TextArea's ``check_consume_key`` legitimately
    claims — so a `press`-driven negative half passes even when the binding is
    still live. ``simulate_key`` sends the control key a terminal sends
    (character ``None``).

    The default app is asserted FIRST: without that control, "ctrl+p no longer
    opens it" proves nothing about the remap.
    """
    from local_operator.tui.command_palette import palette_is_open

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        palette_bindings = [
            binding
            for _key, binding in app._bindings
            if getattr(binding, "action", "") == "command_palette"
        ]
        assert len(palette_bindings) == 1, "Textual injected a second palette binding"
        assert palette_bindings[0].key == "ctrl+p"

        app.simulate_key("ctrl+p")
        await pilot.pause()
        assert palette_is_open(app), "the shipped key does not open the palette"
        await pilot.press("escape")
        await pilot.pause()
        assert not palette_is_open(app)

        app._apply_keymap({"keymap.command_palette": "ctrl+o"}, announce=False)
        await pilot.pause()

        app.simulate_key("ctrl+p")
        await pilot.pause()
        assert not palette_is_open(app), "ctrl+p still opens the palette after the remap"

        app.simulate_key("ctrl+o")
        await pilot.pause()
        assert palette_is_open(app), "the remapped key does not open the palette"


@pytest.mark.asyncio
async def test_a_non_priority_app_row_moves_and_retires_its_old_key() -> None:
    """f9/b/f10-class rows are ordinary bubbling bindings; one live remap
    proves the derivation reaches them and the old chord is gone."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired: list[str] = []
        app.action_toggle_sidebar = lambda: fired.append("sidebar")  # type: ignore[method-assign]
        app._editor().focus()
        await pilot.pause()

        app._apply_keymap({"keymap.toggle_sidebar": "f3"}, announce=False)
        await pilot.pause()

        await pilot.press("f3")
        assert fired == ["sidebar"]

        fired.clear()
        await pilot.press("ctrl+b")
        assert fired == [], "ctrl+b still toggles the sidebar after the remap"


@pytest.mark.asyncio
async def test_a_priority_app_row_moves_with_its_precedence_intact() -> None:
    """`shift+tab` is priority because Textual's Screen owns it.

    The remap must keep firing PAST a focused composer (the flag travels with
    the row, not with the literal key), and the shipped chord must stop.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        fired: list[str] = []
        app.action_cycle_effort = lambda: fired.append("effort")  # type: ignore[method-assign]
        editor = app._editor()
        editor.focus()
        editor.load_text("half a thought")
        await pilot.pause()

        app._apply_keymap({"keymap.cycle_effort": "f12"}, announce=False)
        await pilot.pause()

        await pilot.press("f12")
        assert fired == ["effort"]
        assert editor.text == "half a thought"

        fired.clear()
        await pilot.press("shift+tab")
        assert fired == [], "shift+tab still cycles effort after the remap"


@pytest.mark.asyncio
async def test_a_second_remappable_row_holding_the_key_is_a_refusal_not_a_warning() -> None:
    """Design §C.4 at the page: two ids on one key has no reading under which
    it is intended, so the capture refuses and the row keeps its default.

    The soft warning next door covers a chord an UNREMAPPABLE binding holds;
    here both victims are inside the feature, so the group check must win.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "keymap.new_session")

        await pilot.press("enter")
        await pilot.press("ctrl+t")  # keymap.toggle_todos's shipped default
        await pilot.pause()
        detail = view._detail_text.plain
        assert "already uses ctrl+t" in detail, detail
        assert view._capture is not None, "a refusal must not close the capture"

        await pilot.press("enter")
        await pilot.pause()
        assert view._capture is not None, "enter committed a refused key"
        import os

        manager = ConfigManager(Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]))
        assert (
            settings_io.read_setting(manager, settings_io.BY_KEY["keymap.new_session"]) == "ctrl+n"
        )


@pytest.mark.asyncio
async def test_a_live_remap_repaints_every_mounted_hint_that_names_a_chord() -> None:
    """R1: `set_keymap` alone left the old chord painted (review round 1, major).

    Each of these surfaces either caches its text behind an equality guard (the
    dock header/summary, the todo affordance), holds it in a `Static` (the
    legend, the sidebar chip) or budgets it by width (the ask bar tail), so the
    repaint has to be triggered by the app rather than by the next unrelated
    event.
    """
    from local_operator.tui.widgets.ask_queue import AskBar, ask_rows
    from local_operator.tui.widgets.keys_legend import KeysLegend
    from local_operator.tui.widgets.subagent_panel import SubagentPanel
    from local_operator.tui.widgets.todo_panel import TodoPanel
    from tests.unit.tui.test_ask_queue_surface import _row
    from tests.unit.tui.test_session_sidebar import _plain, _sidebar_with

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        sidebar = await _sidebar_with(pilot, app, [_plain("a", active=True)])
        resting = sidebar.render().plain
        assert "f9" in resting and "ctrl+b" in resting, resting

        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await pilot.pause()
        bar = app.query_one(AskBar)
        calls: list[str] = []
        original_hook = bar.keymap_hints_changed

        def _spy_hook() -> None:
            calls.append("ask-bar")
            original_hook()

        bar.keymap_hints_changed = _spy_hook  # type: ignore[method-assign]

        app.action_toggle_keys_legend()
        await pilot.pause()
        legend = app.query_one(KeysLegend)
        # The FULL composition, not the window: the dock's rows shrink the card's
        # budget, so the todo row can be scrolled out of the painted window while
        # still being what the card would paint.
        assert any("ctrl+t" in row.plain for row in legend._compose_content())
        original_legend_hook = legend.keymap_hints_changed

        def _spy_legend() -> None:
            calls.append("keys-legend")
            original_legend_hook()

        legend.keymap_hints_changed = _spy_legend  # type: ignore[method-assign]

        dock = app.query_one(SubagentPanel)
        todo = app.query_one(TodoPanel)
        # The paint-once guards, set the way a mounted frame leaves them.
        dock._header_shown = True
        dock._summary_key = ("painted",)  # type: ignore[assignment]
        assert "ctrl+t" in todo._affordance_row(0, 0).plain

        app._apply_keymap(
            {
                "keymap.focus_sidebar": "f11",
                "keymap.toggle_sidebar": "ctrl+o",
                "keymap.toggle_asks": "ctrl+y",
                "keymap.toggle_todos": "f12",
                "keymap.toggle_subagents": "f13",
            },
            announce=False,
        )
        await pilot.pause()

        after = sidebar.render().plain
        assert "f11" in after and "ctrl+o" in after, after
        assert "f9 focus" not in after and "ctrl+b" not in after, after
        assert calls == [
            "ask-bar",
            "keys-legend",
        ], "the app never ran the mounted repaint hooks"
        assert keymap.painted_key("keymap.toggle_asks", app._keymap) == "ctrl+y"
        lines = [row.plain for row in legend._compose_content()]
        assert any("f12" in line for line in lines), lines
        assert not any("ctrl+t" in line for line in lines), lines
        assert dock._header_shown is False and dock._summary_key is None
        assert dock._toggle_key() == "f13"
        assert "f12" in todo._affordance_row(0, 0).plain


@pytest.mark.asyncio
async def test_the_fixed_hotkey_sentences_follow_a_live_remap() -> None:
    """R2/Q3: the sentences that named a shipped chord, not a cached Text.

    Each of these was a hardcoded string: the credential refusal's `(f4)`, the
    `/effort` listing's `shift+tab cycles`, the fork-with-no-aside notice, the
    aside card's footer and the slash rows (`/` picker + `/help` table). They
    are asserted at the SAME default spelling first, so a remap is the only
    thing that can change them.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        assert "(f4)" in app._ask_secret_refusal_text()
        assert "shift+tab cycles" in app._effort_listing(("low", "high"), "low")
        effort_row = next(row for row in app._offered_commands() if row.name == "effort")
        assert "shift+tab cycles" in effort_row.description

        notices: list[str] = []

        def _capture(body: str, kind: Any = "info") -> None:
            notices.append(body)

        app._system_notice = _capture  # type: ignore[method-assign]
        app.action_fork_aside()
        assert notices and "ctrl+f" in notices[-1] and "f8" in notices[-1].lower(), notices

        app._apply_keymap(
            {
                "keymap.toggle_asks": "ctrl+y",
                "keymap.cycle_effort": "f12",
                "keymap.fork_aside": "ctrl+o",
                "keymap.aside": "f11",
            },
            announce=False,
        )
        await pilot.pause()
        assert "ctrl+y" in app._ask_secret_refusal_text()
        assert "(f4)" not in app._ask_secret_refusal_text()
        assert "f12 cycles" in app._effort_listing(("low", "high"), "low")
        remapped_effort = next(row for row in app._offered_commands() if row.name == "effort")
        assert "f12 cycles" in remapped_effort.description
        notices.clear()
        app.action_fork_aside()
        assert "ctrl+o" in notices[-1] and "f11" in notices[-1], notices


@pytest.mark.asyncio
async def test_the_aside_footer_names_the_live_fork_key() -> None:
    """R2: the card taught `^f`; the row's default is `ctrl+f` and can move.

    The footer is composed per repaint from `_hint_row`, so this drives the
    composition directly with the fork hint enabled — the state a completed
    exchange leaves — rather than replaying a whole aside turn.
    """
    from local_operator.tui.widgets.aside_panel import AsidePanel, AsideTurn

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app.action_aside()
        await pilot.pause()
        card = app.query_one(AsidePanel)
        # A completed exchange is what enables every hint.
        card._turns = [AsideTurn(question="q?", answer="a.", state="done")]
        card._can_fork = True
        # The state the card is left in after a completed exchange; a real turn
        # arrives with a repaint, and this fixture sets the fields directly.
        card._repaint()
        calls: list[str] = []
        original = card.keymap_hints_changed

        def _spy() -> None:
            calls.append("aside")
            original()

        card.keymap_hints_changed = _spy  # type: ignore[method-assign]
        painted = str(card.content)  # Static.content; str() drops the styles
        assert "ctrl+f" in painted, painted

        app._apply_keymap({"keymap.fork_aside": "ctrl+o"}, announce=False)
        await pilot.pause()
        # Q12: the app's own hook repaints the OPEN card, so the stored
        # (painted) content names the new key — not `_hint_row()` recomposed on
        # demand, which would pass even if nothing ever triggered a repaint.
        assert calls == ["aside"], calls
        painted = str(card.content)
        assert "ctrl+o" in painted, painted
        assert "^f" not in painted


@pytest.mark.asyncio
async def test_a_remapped_cycle_effort_stops_delegating_shift_tab_to_forms() -> None:
    """R5: the form delegation was unconditional, so a remap to `f12` kept it.

    `shift+tab` is delegated to a focused form only BECAUSE the app holds it as
    a priority binding and would otherwise shadow the form's own one. Remapped
    away, the app no longer holds it — the delegation would steal the chord the
    form regained, and `f12` would then have to move fields instead of cycling
    effort. Default behaviour is byte-identical: the delegation still runs.
    """

    class _Form:
        def __init__(self) -> None:
            self.previous = 0

        def action_previous_field(self) -> None:
            self.previous += 1

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        form = _Form()
        app._team_form = form
        app.action_cycle_effort()
        assert form.previous == 1, "the default chord must still reach the form"

        app._apply_keymap({"keymap.cycle_effort": "f12"}, announce=False)
        app._session = None  # the effort path stops at "still starting", quietly
        app.action_cycle_effort()
        assert form.previous == 1, "a remapped chord must not be delegated"

        app._apply_keymap({"keymap.cycle_effort": "f12,shift+tab"}, announce=False)
        app.action_cycle_effort()
        assert form.previous == 2, "an alternate that still holds shift+tab delegates"


@pytest.mark.asyncio
async def test_the_slash_picker_paints_the_live_chord_after_remap_and_at_boot(
    monkeypatch, tmp_path
) -> None:
    """Q10: the picker holds a COPY of the registry rows, so a remap left the
    painted row saying `shift+tab` while `_offered_commands()` returned the live
    text. The boot half is the same seam: the config override is applied by
    `_apply_keymap`, after the editor was composed, so boot must refresh too.
    """
    from local_operator.config_watch import _reset_for_tests
    from local_operator.settings_io import Setting
    from tests.unit.tui.test_app_pilot import _set_editor_line

    def _painted_effort(app: OperatorApp) -> str:
        # `render_rows` is what the picker PAINTS (the widget's own row builder),
        # not the list it was handed: a stale copy would pass the latter.
        picker = app._editor()._picker
        assert picker.is_open(), "the picker never opened for /eff"
        rows = [row.plain for row in picker.render_rows(70)]
        return next(row for row in rows if "/effort" in row)

    from local_operator import settings_io
    from local_operator.config import ConfigManager

    _reset_for_tests()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    target = settings_io.resolve_key("keymap.cycle_effort")
    assert isinstance(target, Setting)
    settings_io.write_setting(ConfigManager(tmp_path), target, "f5")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        for _ in range(200):
            await pilot.pause()
            if app._keymap:
                break
        assert app._keymap.get("keymap.cycle_effort") == "f5", app._keymap
        app._editor().focus()
        await pilot.pause()
        _set_editor_line(app._editor(), "/eff")
        await pilot.pause()
        assert "f5 cycles" in _painted_effort(app)

        app._apply_keymap({"keymap.cycle_effort": "f7"}, announce=False)
        await pilot.pause()
        assert "f7 cycles" in _painted_effort(app)
        assert "shift+tab cycles" not in _painted_effort(app)
