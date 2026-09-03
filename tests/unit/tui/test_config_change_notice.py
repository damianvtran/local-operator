"""``OperatorApp`` reacts to a ``config.yml`` change from another process.

The session applies the values (``tests/unit/session/test_config_live.py``);
the app's job is the user-facing half — say what changed and why behaviour
just moved — plus the two groups only the TUI owns, ``display.*`` and
``tui.theme``, when the write came from ANOTHER process. A write from this
process is silent: the page already told the user.

The watcher is driven by ``poll_now()`` here rather than by its timer, so the
tests are bound by loop turns, never by the 2 s cadence.
"""

from __future__ import annotations

from unittest import mock

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _write_elsewhere(config_dir, key: str, value) -> None:
    """A write shaped like another process's: below the notify hook."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def _notices(app) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


async def _adopted(app, pilot) -> None:
    for _ in range(200):
        if app._session is not None and app._unsubscribe_config_watch is not None:
            return
        await pilot.pause()
    raise AssertionError("the app never adopted a session / subscribed to config")


@pytest.mark.asyncio
async def test_a_change_from_another_process_is_announced_once_with_its_keys(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        watcher = process_watcher(tmp_path)
        _write_elsewhere(tmp_path, "compaction.threshold_percent", 0.5)
        _write_elsewhere(tmp_path, "retry.maxRetries", 3)
        watcher.poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == [
            "config.yml changed: applied: compaction.threshold_percent, retry.maxRetries"
        ]


@pytest.mark.asyncio
async def test_a_non_live_key_is_named_as_taking_effect_on_new(monkeypatch, tmp_path) -> None:
    """``tool_approval_mode`` is deliberately build-time; the notice must not
    claim it applied."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "tool_approval_mode", "auto")
        _write_elsewhere(tmp_path, "compaction.enabled", False)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == [
            "config.yml changed: applied: compaction.enabled; "
            "tool_approval_mode takes effect on /new; "
            "tool approvals now auto — every tool runs without asking"
        ]


@pytest.mark.asyncio
async def test_a_lone_approval_mode_change_from_another_pane_is_announced(
    monkeypatch, tmp_path
) -> None:
    """The one setting where silence is a safety problem (review round 1, M2).

    ``tool_approval_mode`` used to be dropped from the notice whenever it was
    the ONLY changed key, on the theory that `/approvals default` had already
    printed its own receipt. But that write bypassed ``settings_io``, so it
    arrived as ``source="disk"`` — indistinguishable from another pane's edit —
    and the suppression silenced the genuine cross-process case too: flip the
    approval default in pane A, hear nothing in pane B. Batched changes still
    announced it, which is the tell that the rule was about the wrong thing.

    ``_save_approvals_default`` now writes through the facade, so a local write
    is silenced by ``source="local"`` like every other local write, and this
    case is free to speak.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "tool_approval_mode", "auto")
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == [
            "config.yml changed: tool_approval_mode takes effect on /new; "
            "tool approvals now auto — every tool runs without asking"
        ]


@pytest.mark.asyncio
async def test_saving_the_approvals_default_here_does_not_announce_itself(
    monkeypatch, tmp_path
) -> None:
    """The other half of M2: the LOCAL write must still be silent.

    Drives the real ``/approvals default auto`` handler rather than calling the
    writer directly, because what is being asserted is that the facade route
    reaches the watcher as ``source="local"`` on the path a user actually
    takes. The command's own receipt is what tells them; a harness notice on
    top would be the same news twice.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        saved_to, problem = app._save_approvals_default(True)
        await pilot.pause()
        assert not problem, problem
        assert saved_to, "the write reported no destination"
        # It really landed on disk, through the facade.
        assert ConfigManager(tmp_path).get_config_value("tool_approval_mode") == "auto"
        # And produced no harness notice, and no pending change for the poll.
        assert not [n for n in _notices(app) if "config.yml changed" in n]
        assert process_watcher(tmp_path).poll_now() is None


@pytest.mark.asyncio
async def test_a_write_from_this_process_is_silent(monkeypatch, tmp_path) -> None:
    """The page or command here already showed its result; a second line
    would be the same news twice."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        setting = settings_io.resolve_key("compaction.enabled")
        assert setting is not None
        settings_io.write_setting(ConfigManager(tmp_path), setting, False)
        await pilot.pause()
        assert not [n for n in _notices(app) if "config.yml changed" in n]
        # And the fingerprint was recorded: the next tick has nothing to say.
        assert process_watcher(tmp_path).poll_now() is None


@pytest.mark.asyncio
async def test_a_metadata_only_rewrite_produces_no_line(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        ConfigManager(tmp_path).update_config({}, write=True)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert not [n for n in _notices(app) if "config.yml changed" in n]


@pytest.mark.asyncio
async def test_a_theme_written_by_another_process_is_applied_here(monkeypatch, tmp_path) -> None:
    """Closes the cross-process gap for the one LIVE group the session does
    not own: ``/theme`` in pane A repaints pane B."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        before = theme_mod.current_theme()
        target = next(name for name in theme_mod.available_themes() if name != before)
        _write_elsewhere(tmp_path, "tui.theme", target)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert theme_mod.current_theme() == target
        assert any("applied: tui.theme" in n for n in _notices(app))
    theme_mod.set_theme(before)


@pytest.mark.asyncio
async def test_an_unknown_theme_on_disk_is_reported_not_raised(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        before = theme_mod.current_theme()
        # Below the validating facade on purpose: a hand edit is exactly how
        # an unknown name reaches the file.
        manager = ConfigManager(tmp_path)
        manager.set_config_value("tui", {"theme": "no-such-theme"})
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert theme_mod.current_theme() == before
        assert any("unknown theme" in n for n in _notices(app))


@pytest.mark.asyncio
async def test_a_display_flag_from_another_process_drops_the_paint_cache(
    monkeypatch, tmp_path
) -> None:
    from local_operator.tui import settings as tui_settings

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        tui_settings.settings_reload()
        assert tui_settings.settings_get("display.terminal_title") is True  # primes the cache
        _write_elsewhere(tmp_path, "display.terminal_title", False)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert tui_settings.settings_get("display.terminal_title") is False
    tui_settings.settings_reload()


@pytest.mark.asyncio
async def test_unmount_unsubscribes_the_app(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        watcher = process_watcher(tmp_path)
        assert len(watcher._listeners) == 1
    assert watcher._listeners == []
    assert app._unsubscribe_config_watch is None


@pytest.mark.asyncio
async def test_a_new_launch_key_asks_for_a_relaunch_not_a_new(monkeypatch, tmp_path) -> None:
    """Design review round 1, D1: `/new` does not re-exec, so it cannot adopt
    a NEW_LAUNCH key. Saying `/new` sent the user to do something that would
    not work and gave them no second message when it did not."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "hosting", "openrouter")
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == ["config.yml changed: hosting needs a relaunch"]


@pytest.mark.asyncio
async def test_a_retired_key_is_not_promised_to_take_effect(monkeypatch, tmp_path) -> None:
    """Design review round 1, D1, sharpest edge: the `retired` section is
    documented as "read but no longer do anything", and the notice was telling
    the user one of those keys would take effect on `/new`."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    retired = [s for s in settings_io.SETTINGS if s.section == "retired"]
    assert retired, "the retired section is empty; this test has nothing to pin"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, retired[0].key, 4321)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == [f"config.yml changed: {retired[0].key} is retired and does nothing"]
        assert "takes effect" not in notices[0]


@pytest.mark.asyncio
async def test_the_approval_mode_notice_names_the_mode_and_warns(monkeypatch, tmp_path) -> None:
    """Design review round 1, D2.

    Asserts the two things the frame made obvious: the line says WHICH way the
    switch went, and it renders at the same severity as the local receipt for
    the identical transition rather than as dim grey routine info.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "tool_approval_mode", "auto")
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        blocks = [b for b in app.query(NoticeBlock) if "config.yml changed" in (b.text() or "")]
        assert blocks, "no config-change notice was emitted"
        text = blocks[-1].text() or ""
        assert "now auto" in text and "without asking" in text, text
        # Asserted on the rendered token, not a bespoke attribute: this is
        # what actually decides the ink, and `info` maps to "dim".
        assert blocks[-1]._token == "warning", f"rendered in {blocks[-1]._token!r} ink, not warning"

        # And the cached default follows, so a later bare `/approvals` does not
        # report a value a new session would not boot with (QA round 2, Q1).
        assert app._approvals_default_auto is True


@pytest.mark.asyncio
async def test_new_reloads_the_launch_config_so_the_notice_promise_is_true(
    monkeypatch, tmp_path
) -> None:
    """`/new` must adopt a NEW_SESSIONS key written by another process.

    Design review round 1 (D1) and QA round 2 (Q1) both drove this path and
    found the session rebuilt from the launch-time `ConfigManager` the factory
    closed over, so the value the notice promised would "take effect on /new"
    did not. The notice is the reason this matters: the harness tells the user
    to run `/new`, and `tool_approval_mode` is among the keys it says that
    about.

    Built the way `cli.py:3004` builds it — `on_config_changed=<manager>.reload`
    with the factory closing over that same instance — because the defect only
    exists in that arrangement; a factory that re-reads config per call would
    pass while production still failed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    launch_manager = ConfigManager(tmp_path)
    launch_manager.set_config_value("hosting", "anthropic")
    built: list[str] = []

    async def resume_factory(resume_id):
        built.append(str(launch_manager.get_config_value("hosting")))
        return await _factory(FakeSession())

    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=resume_factory,
        on_config_changed=launch_manager.reload,
    )
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "hosting", "openrouter")

        app._cmd_new(lambda body, kind="info": None)
        for _ in range(400):
            if built:
                break
            await pilot.pause()
        await pilot.pause()

    assert built, "/new never rebuilt the session"
    assert built[-1] == "openrouter", (
        f"/new built with {built[-1]!r}: the launch-time config manager was not reloaded, "
        "so the value the notice promised would take effect did not"
    )


#: The five shapes `ConfigManager._load_config` mishandles: the first three
#: RAISE, the last two are the destructive ones — it moves the user's file
#: aside to `config.yml.bad.<ts>` and continues from defaults.
_MALFORMED_CONFIGS = {
    "scalar": "values: 3\n",
    "list": "values:\n - a\n",
    "null": "values:\n",
    "broken-yaml": "values:\n  bad: : :\n",
    "non-mapping-top": "- a\n- b\n",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_MALFORMED_CONFIGS))
async def test_new_survives_a_malformed_config_without_touching_the_file(
    monkeypatch, tmp_path, shape: str
) -> None:
    """`/new` must not crash on, or destroy, a config it cannot parse.

    Review round 3, B1 — a regression the `/new` reload introduced. In
    production `_on_config_changed` is `ConfigManager.reload`, which raises on
    a non-mapping `values:` and MOVES `config.yml` aside on a YAML error. The
    story this PR makes likelier: the watcher deliberately holds a broken file
    in silence, so the user's first hint of a fat-fingered edit is the `/new`
    the notice sent them to — which either died or replaced every setting they
    had with defaults.

    Asserts all four properties per shape, because the fix has to deliver all
    four: no exception, the session still builds, the file is untouched, and no
    `.bad` file appears. The last two are the ones that matter most; a crash is
    recoverable, a silently discarded config is not.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    launch_manager = ConfigManager(tmp_path)
    launch_manager.set_config_value("hosting", "anthropic")
    built: list[str] = []

    async def resume_factory(resume_id):
        built.append(str(launch_manager.get_config_value("hosting")))
        return await _factory(FakeSession())

    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=resume_factory,
        on_config_changed=launch_manager.reload,
    )
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        body = _MALFORMED_CONFIGS[shape]
        (tmp_path / "config.yml").write_text(body)

        app._cmd_new(lambda body, kind="info": None)  # must not raise
        for _ in range(400):
            if built:
                break
            await pilot.pause()

    assert built, f"{shape}: /new did not build a session"
    assert (
        tmp_path / "config.yml"
    ).read_text() == body, f"{shape}: the user's config was rewritten"
    assert not list(tmp_path.glob("*.bad*")), f"{shape}: config.yml was moved aside by /new"


@pytest.mark.asyncio
async def test_new_adopts_the_approval_mode_into_the_real_gate(monkeypatch, tmp_path) -> None:
    """The fourth NEW_SESSIONS key, which the round-2 fix could not reach.

    QA round 3, Q3. `_cmd_new` → `_on_config_changed` repairs the session
    FACTORY path, but the TUI's approval gate is process state written only by
    `_load_approvals_default` at mount — so `tool_approval_mode` was the one
    key of four whose "takes effect on /new" promise was not kept, and the one
    where being wrong is a safety problem.

    Driven in the tightening direction (`auto → ask`), which is the dangerous
    one: the notice promises prompts and the old behaviour kept auto-approving.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    launch_manager = ConfigManager(tmp_path)
    launch_manager.set_config_value("hosting", "anthropic")
    _write_elsewhere(tmp_path, "tool_approval_mode", "auto")

    async def resume_factory(resume_id):
        return await _factory(FakeSession())

    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=resume_factory,
        on_config_changed=launch_manager.reload,
    )
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        assert app._approve_all is True, "the app did not boot on the saved auto default"

        _write_elsewhere(tmp_path, "tool_approval_mode", "ask")
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        # The RUNNING session's gate must not move — the key is NEW_SESSIONS.
        assert app._approve_all is True

        app._cmd_new(lambda body, kind="info": None)
        for _ in range(400):
            await pilot.pause()
            if app._approve_all is False:
                break
        assert app._approve_all is False, (
            "the notice promised tool_approval_mode takes effect on /new, but the "
            "approval gate still auto-approves"
        )


@pytest.mark.asyncio
async def test_multi_key_clauses_agree_in_number(monkeypatch, tmp_path) -> None:
    """Plural key lists take plural verbs (design review round 1, D7).

    Pinned with MULTIPLE keys per clause on purpose: every other pin in this
    file asserts a single-key string, which is exactly why three ungrammatical
    clauses shipped — `hosting, model_name needs a relaunch` and, worse,
    `a, b, c is retired and does nothing`. The two-key relaunch form is the
    common case, not an edge: those are the keys a user changes together when
    switching models.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    retired = [s.key for s in settings_io.SETTINGS if s.section == "retired"][:3]
    assert len(retired) >= 2, "need two retired keys to pin the plural form"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(200, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "hosting", "openrouter")
        _write_elsewhere(tmp_path, "model_name", "some/model")
        for key in retired:
            _write_elsewhere(tmp_path, key, 4321)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()

        notice = [n for n in _notices(app) if "config.yml changed" in n][-1]

    assert "hosting, model_name need a relaunch" in notice, notice
    assert f"{', '.join(sorted(retired))} are retired and do nothing" in notice, notice
    # And the singular forms must survive the change.
    assert " needs a relaunch" not in notice
    assert " is retired and does nothing" not in notice


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_MALFORMED_CONFIGS))
async def test_a_display_repaint_after_a_fan_out_never_destroys_the_config(
    monkeypatch, tmp_path, shape: str
) -> None:
    """The paint path must not move the user's `config.yml` aside.

    Review round 4, B3 — the same defect class as B1, reached without any user
    action. `tui/settings.py` caches the display flags and this PR added the
    first invalidator that fires on ANOTHER process's write: before it, the
    cache was dropped only by a write this process had just made, so the file
    was well-formed by construction.

    The story needs no `/new`: a display flag changes in another pane (LIVE and
    encouraged), the cache drops, the user's next edit is mis-indented — the
    state the watcher deliberately holds in silence — and the next shimmer or
    glyph read repaints. `settings_get` is called from five paint paths rather
    than every frame, so the window between the invalidation and the
    repopulating read is unbounded in wall time.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_get, settings_reload

    ConfigManager(tmp_path).set_config_value("hosting", "")
    watcher = process_watcher(tmp_path)
    watcher.poll_now()

    # Another pane flips a display flag; the listener drops this cache.
    _write_elsewhere(tmp_path, "display.shimmer", False)
    watcher.poll_now()
    settings_reload()
    assert settings_get("display.shimmer") is False
    settings_reload()  # empty again, as it is between paints

    # The user's NEXT edit is malformed. The watcher holds it silently.
    body = _MALFORMED_CONFIGS[shape]
    (tmp_path / "config.yml").write_text(body)
    assert watcher.poll_now() is None, f"{shape}: the watcher adopted a malformed file"

    settings_get("display.shimmer")  # the repaint

    assert (
        tmp_path / "config.yml"
    ).read_text() == body, f"{shape}: a repaint rewrote the user's config"
    assert not list(tmp_path.glob("*.bad*")), f"{shape}: a repaint moved config.yml aside"


@pytest.mark.asyncio
async def test_no_config_watch_listener_path_constructs_a_config_manager(
    monkeypatch, tmp_path
) -> None:
    """The INVARIANT behind B1 and B3, rather than a third point fix.

    Every blocker in this change's history has been one defect: a caller
    reachable from a config-watch fan-out constructs a `ConfigManager`, whose
    `_load_config` MOVES a malformed `config.yml` aside and continues from
    defaults. Point-fixing each site leaves the generator intact — the class
    returns the next time someone adds a consumer to the fan-out, and the
    failure is silent, destructive, and found by a user rather than a test.

    So this asserts the property directly, against the REAL listener
    (`OperatorApp._on_config_change`) and the real repaint that follows it,
    rather than a stand-in registered by the test — a stand-in would only ever
    prove things about itself, and the whole point is to catch a consumer
    nobody has written yet.

    Deliberately allows the FALLBACK in `tui/settings._load`: that branch runs
    only when no watcher exists, which is exactly when there is no
    cross-process invalidation to race with, and removing it would break every
    CLI process that never starts a watcher.
    """
    import local_operator.config as config_mod
    from local_operator.tui import settings as tui_settings

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    real_manager = config_mod.ConfigManager
    real_manager(tmp_path).set_config_value("hosting", "")

    constructed: list[str] = []

    class TattlingConfigManager(real_manager):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            constructed.append("ConfigManager")
            super().__init__(*args, **kwargs)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        # A display key, because that is the branch with a cache to drop; the
        # assertion covers whatever else the listener does on the same change.
        _write_elsewhere(tmp_path, "display.shimmer", False)
        constructed.clear()

        with mock.patch.object(config_mod, "ConfigManager", TattlingConfigManager):
            change = process_watcher(tmp_path).poll_now()
            assert change is not None, "the fan-out did not fire; this proved nothing"
            await pilot.pause()
            # The repaint that follows the invalidation.
            tui_settings.settings_get("display.shimmer")

    assert not constructed, (
        "a config-watch listener path constructed a ConfigManager. Its _load_config "
        "moves a malformed config.yml aside and continues from defaults, so this is "
        "the B1/B3 defect class reopening. Read the watcher's validated snapshot "
        "(existing_watcher().values) instead."
    )
