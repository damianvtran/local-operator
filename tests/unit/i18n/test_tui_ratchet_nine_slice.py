"""Ratchet slice 9: `tui/app.py` (tui.app).

`app.py` is the app shell, so its copy is the shell's own: sidebar reasons,
the skills help blocks, config-change receipts, the parked-summary, the
stop-all qualifiers, the browser sign-in flow, the help-table notes, the
notifications receipt and the provider-status line. Four per-feature
namespaces already exist for the widgets this file HOSTS (`tui.projects`,
`tui.status`, `tui.welcome`, ...); this namespace is for what the shell
itself says, so one file's copy stays in one catalogue.

**Grammar moved into the message.** Six of these keys carry ICU plurals, and
they are the interesting ones: `takes/take`, `needs/need`, the `it/they`
pronoun in the stop-all qualifier, and the member/wake/monitor nouns were all
`x if n == 1 else y` expressions in code — English grammar encoded in Python,
which no other locale can correct. In the catalogue a translator can inflect
them. The counts still render RAW (`shown` is the stringified digit), never
`#`, because ICU's `#` applies locale grouping and `1200` must not become
`1,200` (the trap QA recorded on slice 8).

**What is still English in this file, stated rather than implied:** the
`" and "` list joiner in the parked summary, and the list separators inside
the other `.join(...)` calls, are passed in as already-joined strings — list
formatting has no message shape yet. The words around them DO translate.
Also untranslated by design: command names, env vars and key paths inside the
help notes (`/agent`, `LOCAL_OPERATOR_NO_TERMINAL_TITLE`, `config.yml`,
`display.terminal_title`), which the style guide lists as kept-English.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.i18n.keys import tui_app as _x
from local_operator.i18n.text import tr

REPO = Path(__file__).resolve().parents[3]
_app = tr("tui.app")

ACTIVATED = "local_operator/tui/app.py"

RETAINED = (
    "local_operator/tui/widgets/settings_view.py",
    "local_operator/tui/widgets/usage_panel.py",
    "local_operator/tui/projects_render.py",
)

#: The exact bytes each site carried before extraction, keyed by catalogue key.
#: A dict rather than 46 separate asserts so the whole namespace is pinned in
#: one place and a re-word shows up as a single diff hunk.
BASE = {
    "tui.app.sidebar.why_retry_budget": "retry budget exhausted",
    "tui.app.sidebar.why_retired": "source retired",
    "tui.app.sidebar.why_left_screen": "source left the screen",
    "tui.app.skills.teams_heading": "teams",
    "tui.app.skills.agents_heading": "agents",
    "tui.app.skills.led_by": (
        "Led by {manager} · {count, plural, one {{shown} member} other {{shown} members}}"
    ),
    "tui.app.skills.send_agent": "Send: /agent <name> <message>",
    "tui.app.skills.detach_agent": "Detach: /agent clear",
    "tui.app.goals.settled_heading": "settled goals",
    "tui.app.config.applied": "applied: {keys}",
    "tui.app.config.effect_new": "{names} {count, plural, one {takes} other {take}} effect on /new",
    "tui.app.config.needs_relaunch": "{names} {count, plural, one {needs} other {need}} a relaunch",
    "tui.app.config.toggle_on": "{label} on — takes effect from your next turn",
    "tui.app.config.toggle_off": "{label} off — calls will be refused until it is turned back on",
    "tui.app.config.label_web_search": "web search",
    "tui.app.config.label_web_fetch": "web fetch",
    "tui.app.parked.wakes": "{count, plural, one {{shown} wake} other {{shown} wakes}}",
    "tui.app.parked.monitors": "{count, plural, one {{shown} monitor} other {{shown} monitors}}",
    "tui.app.parked.dormant": " — {parked} dormant until then",
    "tui.app.stop_all.already_leaving": "{shown} already leaving — asked again, then left alone",
    "tui.app.stop_all.will_be_stopped": (
        "{count, plural, one {{shown} leaving, but it will be stopped, not left to finish} "
        "other {{shown} leaving, but they will be stopped, not left to finish}}"
    ),
    "tui.app.stop_all.repeat_hint": "repeat /stop all within {seconds}s to stop them all",
    "tui.app.aida.next_checkin": "next check-in {when}",
    "tui.app.aida.extras_used": "extra check-ins used today: {used} of {maximum}",
    "tui.app.signin.opening": "Opening your browser to sign in to {label}…",
    "tui.app.signin.finished_hint": (
        "Finished in the browser? This window continues on its own · ctrl+c cancels"
    ),
    "tui.app.signin.full_link": "Full link: {url}",
    "tui.app.help.paste_note": "attaches an image or file from the system clipboard",
    "tui.app.help.copy_note": "copies the highlighted range in the composer",
    "tui.app.help.copy_note_more": "move the caret to hand the key back; esc interrupts",
    "tui.app.help.skill_note": "runs a named skill on the rest of the line",
    "tui.app.help.model_saved_note": "switches back to the config.yml default",
    "tui.app.help.cmd_paste_note": "same, where the terminal forwards it; not Terminal.app",
    "tui.app.help.title_note": (
        "shows session state outside the app; shell: `lop config edit display.terminal_title "
        "false`"
    ),
    "tui.app.help.title_note_more": (
        "or set `LOCAL_OPERATOR_NO_TERMINAL_TITLE=1`; legend: `lo ›` idle, `lo ⣾` running, `lo "
        "!` waiting for approval"
    ),
    "tui.app.help.notify_note": (
        "desktop toast when a turn finishes here, or when a detached session needs you; shell: "
        "`lop config edit display.notifications false`"
    ),
    "tui.app.help.notify_note_more": (
        "or set `LOCAL_OPERATOR_NO_NOTIFICATIONS=1`; in-band ones wait for the terminal to be "
        "unfocused, detached ones click to reopen"
    ),
    "tui.app.notifications.more": "  …{shown} more",
    "tui.app.notifications.sidebar_hint": "{bound} — ctrl+b shows or hides the sidebar",
    "tui.app.notifications.mark_all_read": "/notifications read marks all {count} read",
    "tui.app.notifications.mark_these_read": "/notifications read marks these read",
    "tui.app.providers.none": "(none)",
    "tui.app.providers.stale_all": "stale list: all providers",
    "tui.app.providers.stale_named": "stale list: {names}",
    "tui.app.providers.stale_all_error": "stale list: all providers — {error}",
    "tui.app.providers.no_live_list": "no live list: {names}",
}


def test_the_namespace_is_exactly_the_base_bytes() -> None:
    """Every key, pinned to the bytes its site carried before extraction."""
    live = catalogues.load_catalogue("en", "tui.app")
    assert live == BASE
    sidecar = catalogues.context("en", "tui.app")
    assert set(sidecar) == set(BASE)
    for key, entry in sidecar.items():
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_plurals_render_the_base_sentence_and_keep_raw_digits() -> None:
    # Was: f"{wakes} wake{'s' if wakes != 1 else ''}" and the monitor twin.
    assert _app(_x.parked_wakes(count=1, shown="1")) == "1 wake"
    assert _app(_x.parked_wakes(count=3, shown="3")) == "3 wakes"
    assert _app(_x.parked_monitors(count=1, shown="1")) == "1 monitor"
    assert _app(_x.parked_monitors(count=2, shown="2")) == "2 monitors"
    # Was: f"Led by {team.manager} · {slots} {member_word}"
    assert _app(_x.skills_led_by(manager="ada", count=1, shown="1")) == "Led by ada · 1 member"
    assert _app(_x.skills_led_by(manager="ada", count=4, shown="4")) == "Led by ada · 4 members"
    # Was: verb = "takes" if len(new_sessions) == 1 else "take"
    assert _app(_x.config_effect_new(names="tui.color", count=1)) == (
        "tui.color takes effect on /new"
    )
    assert _app(_x.config_effect_new(names="a, b", count=2)) == "a, b take effect on /new"
    # Was: verb = "needs" if len(relaunch) == 1 else "need"
    assert _app(_x.config_needs_relaunch(names="model.name", count=1)) == (
        "model.name needs a relaunch"
    )
    assert _app(_x.config_needs_relaunch(names="a, b", count=2)) == "a, b need a relaunch"
    # Was: f"{n} leaving, but {'it' if n == 1 else 'they'} will be stopped, ..."
    assert _app(_x.stop_all_will_be_stopped(count=1, shown="1")) == (
        "1 leaving, but it will be stopped, not left to finish"
    )
    assert _app(_x.stop_all_will_be_stopped(count=2, shown="2")) == (
        "2 leaving, but they will be stopped, not left to finish"
    )
    # Grouping discipline: ICU's `#` would group these; `shown` must not.
    for count in (1200, 10000):
        rendered = _app(_x.parked_wakes(count=count, shown=str(count)))
        assert rendered == f"{count} wakes"
        assert "," not in rendered


def test_catalogue_status_keeps_its_bytes_end_to_end() -> None:
    """Drive the real function, not just the keys around it."""
    from local_operator.tui.app import _catalogue_status

    # Was: f"{STALE_LIST_LABEL} all providers" when every listed provider is stale.
    # "all providers" needs every LISTED provider stale: "cached" still counts as
    # listed, so a cached leg takes the named branch.
    assert _catalogue_status({"xai": "stale", "anthropic": "stale"}) == "stale list: all providers"
    assert _catalogue_status({"xai": "stale", "anthropic": "stale", "openrouter": "cached"}) == (
        "stale list: anthropic, xai"
    )
    assert _catalogue_status({"xai": "stale", "openrouter": "cached"}) == "stale list: xai"
    assert _catalogue_status({"xai": "empty"}) == "no live list: xai"
    assert _catalogue_status({"anthropic": "cached", "openrouter": "ok"}) == ""
    assert _catalogue_status({"xai": "stale", "y": "empty"}) == (
        "stale list: xai · no live list: y"
    )


def test_the_activated_file_lost_its_ceiling() -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    assert ACTIVATED in baseline["enforced"], f"{ACTIVATED} is not activated"
    assert ACTIVATED not in baseline["files"], f"{ACTIVATED} kept its ceiling"
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"
    assert not [e for e in baseline["enforced"] if e.endswith("/") and "/tui/" in e]


def test_a_planted_literal_fails_the_activated_checker(tmp_path, monkeypatch, i18n_check) -> None:
    source = (REPO / ACTIVATED).read_text(encoding="utf-8")
    target = tmp_path / ACTIVATED
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(json.dumps({"note": "slice fixture", "enforced": [ACTIVATED], "files": {}}))
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    # A SINK-shaped plant, not an assignment: the collector reads call
    # arguments, so an assignment would prove nothing (that blind class is
    # documented in the module docstring and is why this control uses print).
    target.write_text(source + '\nprint("planted slice prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


@pytest.fixture
def fr_catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    for locale, values in (("en", BASE), ("fr", {k: "FR " + v for k, v in BASE.items()})):
        folder = tmp_path / locale
        folder.mkdir()
        (folder / "tui.app.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


def test_the_shell_copy_follows_a_live_language_change(fr_catalogue) -> None:
    """`app.py` holds no import-time copy, so this proves it by DRIVING it.

    A function written for a backends-locale switch reads the catalogue at each
    call; a constant would have kept English. The synthetic fr catalogue is
    never eligible outside this fixture.
    """
    from local_operator.tui.app import _catalogue_status

    assert _catalogue_status({"xai": "empty"}) == "no live list: xai"
    assert _app(_x.sidebar_why_retired()) == "source retired"
    assert _app(_x.skills_teams_heading()) == "teams"
    text.set_active_locale("fr")
    assert _catalogue_status({"xai": "empty"}) == "FR no live list: xai"
    assert _app(_x.sidebar_why_retired()) == "FR source retired"
    assert _app(_x.skills_teams_heading()) == "FR teams"
