"""The ``/settings`` page's remaining chrome, keyed under ``tui.settings_view.chrome``.

WHY THIS FILE EXISTS. #4361 keyed the page's first residual chrome and the
operator's goal is still that switching language leaves no English on the
page. What remained was invisible to ``scripts/i18n/check.py`` (the file sits
at a zero baseline): footer hint labels built as ``" move"`` tuples, the value
vocabulary inside ``_render_value``, the cascade empty states, the delete and
capture asks, the detail-line state clauses, module-level hotkey sentences and
the page's own notices/refusals assigned to ``self._error``/``self._notice``.

Claims, each driven through the page's REAL paint or event paths (never by
reading the catalogue back):

1. English is byte-identical to the pre-extraction literals, spelled out
   literally so a wording change cannot pass as "still equal".
2. With a pseudo-locale (``fr`` = ``"FR " + English``) carrying only this
   namespace plus the two reused ``tui.small`` bool words, EVERY surface paints
   the pseudo text after a LIVE language switch — nothing cached across it —
   while key names (``enter``, ``esc``, ``r``, ``d``) and commands survive.
3. A mutation control: restoring one literal makes claim 2 fail.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator import keymap as keymap_mod
from local_operator import settings_io
from local_operator.i18n import catalogues, resolve, text
from local_operator.settings_io import Kind
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import settings_view as sv_mod
from local_operator.tui.widgets.settings_view import SettingsView, _Capture, _Row
from tests.unit.tui.test_app_pilot import FakeSession, _factory

NS = "tui.settings_view.chrome"
EN_ROOT = Path(catalogues.__file__).resolve().parent / "catalogues"
BASE: dict[str, str] = json.loads((EN_ROOT / "en" / f"{NS}.json").read_text("utf-8"))
SMALL_BOOL = ("tui.small.settings_view.bool_on", "tui.small.settings_view.bool_off")


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_config(tmp_path, monkeypatch)


def _seed_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The page writes on reset/commit; never against the developer's config.
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    # A malformed cascade is only reachable from OUTSIDE the app (a hand edit
    # or a pre-#440-fix install), so it is written to the file directly, the
    # way `tests/unit/tui/test_settings_view.py::_corrupt_the_cascade` does.
    document = {"values": {"retry": {"fallbackChains": "{'default': ['x/y']}"}}}
    (config / "config.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    from local_operator.tui.settings import settings_reload

    settings_reload()


def _select(view: SettingsView, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return
    raise AssertionError(f"no row for {key}")


def _hints(view: SettingsView, monkeypatch: pytest.MonkeyPatch, **state: Any) -> str:
    """The footer for one page state, painted by the real ``_paint_hints``."""
    with monkeypatch.context() as patch:
        for name, value in state.items():
            patch.setattr(view, name, value)
        view._paint_hints()
        return view.rendered_hints()


def _first_of(kind: Kind, **attrs: Any) -> settings_io.Setting:
    for setting in settings_io.SETTINGS:
        if setting.kind is kind and all(getattr(setting, k) == v for k, v in attrs.items()):
            return setting
    raise AssertionError(f"no {kind} setting with {attrs}")


async def _surfaces(
    pilot: Any, app: OperatorApp, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    app._open_settings_view()
    view = app.query_one(SettingsView)
    await pilot.pause()
    out: dict[str, Any] = {}

    # Footer, one row per state the page distinguishes.
    rest = {"_reset_applies": lambda: True, "_current_is_readonly": lambda: False}
    out["foot_rest"] = _hints(view, monkeypatch, **rest)
    widest = sv_mod._t_chrome(sv_mod._svc.footer_back_to_conversation())
    out["foot_narrow"] = _hints(
        view,
        monkeypatch,
        **rest,
        # Only the widest `esc` label overflows: the bar falls to its narrow word.
        _measure_hints=lambda plan, esc: 10**6 if esc == widest else 0,
    )
    out["foot_edit"] = _hints(
        view, monkeypatch, _editing="retry.baseDelayMs", _suggest_rows=lambda: []
    )
    out["foot_suggest"] = _hints(
        view, monkeypatch, _editing="model_name", _suggest_rows=lambda: [object()]
    )
    out["foot_choose"] = _hints(view, monkeypatch, _expanded="tui.theme")
    out["foot_listen"] = _hints(view, monkeypatch, _capture=_Capture(setting_key="k"))
    out["foot_pending"] = _hints(view, monkeypatch, _capture=_Capture(setting_key="k", key="f5"))

    # Value vocabulary, as the value column paints it.
    out["values"] = [
        sv_mod._render_value(None),
        sv_mod._render_value(True),
        sv_mod._render_value(False),
        sv_mod._render_value({"a": []}),
        sv_mod._render_value({"a": [], "b": []}),
    ]

    # Cascade empty states (painted rows) and the cascade notices/errors.
    cascade = _first_of(Kind.CASCADE)
    with monkeypatch.context() as patch:
        patch.setattr(view, "_cascade_is_malformed", lambda _s: False)
        out["cascade_empty"] = [r.text for r in view._cascade_rows(cascade) if r.kind == "empty"]
    with monkeypatch.context() as patch:
        patch.setattr(view, "_cascade_is_malformed", lambda _s: True)
        out["cascade_malformed"] = [
            r.text for r in view._cascade_rows(cascade) if r.kind == "empty"
        ]
    # The fixture stored the #440 wreckage (a non-mapping cascade) on disk:
    # `r` clears it and says so. Then a healthy cascade refuses `r`.
    _select(view, cascade.key)
    view.action_reset()
    out["cascade_cleared"] = view.notice_text
    settings_io.write_chains(view._manager, {"default": ["openai/gpt-4o"]}, base={})
    view._repaint()
    _select(view, cascade.key)
    view.action_reset()
    out["cascade_reset_refused"] = view.notice_text
    view._editing, view._buffer = "chainadd", ""
    view._commit_chain_add()
    out["cascade_expected"] = view.error_text
    view._editing, view._buffer = "chainadd", "fast nonsense"
    view._commit_chain_add()
    out["cascade_first_hop"] = view.error_text
    view._editing = None

    # Save failures, through the real `_save` wrapper.
    def boom(exc: BaseException):
        def action() -> None:
            raise exc

        return action

    view._save(boom(settings_io.ConfigUnreadableError("bad yaml")))
    out["err_unreadable"] = view.error_text
    view._save(boom(TypeError("oops")))
    out["err_structure"] = view.error_text.replace(str(view._config_path()), "<path>")
    view._save(boom(OSError("disk full")))
    out["err_save"] = view.error_text

    # Retired / not-text refusals and the discard notice.
    _select(view, "conversation_length")
    view.action_activate()
    out["err_retired"] = view.error_text
    view._begin_edit(_Row(kind="setting", setting=cascade))
    out["err_not_text"] = view.error_text
    view._editing, view._buffer, view._edit_seed, view._discard_announced = (
        "retry.baseDelayMs",
        "9",
        "1",
        False,
    )
    view._settle_row()
    out["notice_discarded"] = view.notice_text

    # Delete confirmation, both contract rungs.
    view._confirm_delete = "default"
    with monkeypatch.context() as patch:
        patch.setattr(view, "_detail_width", lambda: 200)
        out["confirm_long"] = list(view._confirm_parts())
        patch.setattr(view, "_detail_width", lambda: 30)
        out["confirm_short"] = list(view._confirm_parts())
    view._confirm_delete = None

    # Capture: listening, refused, pending with a stolen binding, saved.
    hotkey = next(
        s.key
        for s in settings_io.SETTINGS
        if s.kind is Kind.HOTKEY and keymap_mod.capturable_in_terminal(s.key)
    )
    view._error = view._notice = ""
    view._capture = _Capture(setting_key=hotkey)
    view._paint_detail()
    out["capture_listen"] = view._detail_text.plain
    view._capture = _Capture(setting_key=hotkey, refused="REFUSED")
    view._paint_detail()
    out["capture_refused"] = view._detail_text.plain
    with monkeypatch.context() as patch:
        patch.setattr(keymap_mod, "conflict_note", lambda *_a: "")
        patch.setattr(view, "_app_binding_victim", lambda _k: "Quit")
        view._capture = _Capture(setting_key=hotkey, key="f5")
        out["capture_conflict"] = view._capture_conflict()
        view._paint_detail()
        out["capture_pending"] = view._detail_text.plain
    view._capture = None
    with monkeypatch.context() as patch:
        patch.setattr(view, "_write", lambda *_a, **_k: True)
        view._capture = _Capture(setting_key=hotkey, key="f5")
        view._commit_capture()
        assert not view.error_text, view.error_text
        out["capture_saved"] = view.notice_text
    view._capture = None

    # Detail-line state clauses.
    expandable = _first_of(Kind.BOOL)
    view._expanded = expandable.key
    out["clause_not_saved"] = view._detail_clause(_Row(kind="setting", setting=expandable))
    view._expanded = None
    child = next(s for s in settings_io.SETTINGS if s.gated_by is not None)
    assert child.gated_by is not None
    # The master's label is registry copy (another namespace); masked so the
    # assertion is about this clause's own words in any language.
    master_label = settings_io.BY_KEY[child.gated_by].label
    with monkeypatch.context() as patch:
        patch.setattr(view, "_gated_off", lambda _s: True)
        out["clause_inert"] = view._gate_clause(child).replace(master_label, "<master>")
    master = settings_io.BY_KEY["session.cleanup.enabled"]
    with monkeypatch.context() as patch:
        patch.setattr(settings_io, "read_setting", lambda _m, _s: True)
        patch.setattr(view, "_cleanup_preview_count", lambda: None)
        out["clause_no_limits"] = view._gate_clause(master)
        patch.setattr(view, "_cleanup_preview_count", lambda: 3)
        out["clause_would_remove"] = view._gate_clause(master)
    with monkeypatch.context() as patch:
        patch.setattr(settings_io, "is_default", lambda _m, _s: False)
        patch.setattr(view, "_gate_clause", lambda _s: "")
        bool_row = _Row(kind="setting", setting=expandable)
        out["clause_default"] = view._detail_clause(bool_row)

    # Hotkey sentences: the hold-to-talk row's detail line (painted) and its
    # `enter` refusal (the real action), plus the two sibling sentences.
    view._error = view._notice = ""
    _select(view, "keymap.push_to_talk")
    out["hotkey_hold_painted"] = view._detail_text.plain
    view.action_activate()
    out["hotkey_refusal_raised"] = view.error_text
    view._error = ""
    out["hotkey"] = [
        sv_mod._desktop_hotkey_detail(),
        sv_mod._hold_hotkey_detail(),
        sv_mod._recording_hotkey_detail(),
        sv_mod._hold_hotkey_refusal(),
    ]
    # The gate-kept alert, through the real shed ladder at three widths.
    rungs = []
    for width in (200, 50, 30):
        with monkeypatch.context() as patch:
            patch.setattr(view, "_detail_width", lambda w=width: w)
            rungs.append(view._alert_text())
    out["gate_kept"] = rungs
    return out


ENGLISH: dict[str, Any] = {
    "capture_conflict": "takes f5 from “Quit”",
    "capture_listen": "listening for a key · esc cancels",
    "capture_pending": "f5 · takes f5 from “Quit” · enter confirms · esc cancels",
    "capture_refused": "REFUSED · press another key · esc cancels",
    "capture_saved": "f5 saved",
    "cascade_cleared": "cleared a malformed cascade",
    "cascade_empty": [
        "no cascade configured",
    ],
    "cascade_expected": "expected: <chain-key> <provider>/<model>",
    "cascade_first_hop": (
        "chain needs a first hop — expected provider/model"
        " (e.g. openrouter/deepseek/deepseek-chat)"
    ),
    "cascade_malformed": [
        "malformed cascade — press r to clear it",
    ],
    "cascade_reset_refused": "r resets one setting; delete a chain with d on its row",
    "clause_inert": "inert: <master> is off",
    "clause_no_limits": "on, but no limits set below",
    "clause_not_saved": "not saved until enter",
    "clause_would_remove": (
        "would remove 3 at next launch · preview: lop sessions cleanup --dry-run"
    ),
    "confirm_long": [
        "delete chain default and its 1 hop?",
        "press d again to confirm · esc cancels",
    ],
    "confirm_short": [
        "delete chain default and its 1 hop?",
        "d confirms · esc cancels",
    ],
    "err_not_text": "this setting is not edited as text",
    "err_retired": "this setting is retired and cannot be changed",
    "err_save": "could not save: disk full",
    "err_structure": (
        "could not save — <path> has an unexpected structure and may need repairing (oops)"
    ),
    "err_unreadable": "config.yml is unreadable, nothing was written — bad yaml",
    "foot_choose": "↑↓  move · enter  choose · ←→  panes · esc cancel",
    "foot_edit": "↑↓  move cancel · enter  save · ←→  panes · esc cancel",
    "foot_listen": "↑↓  any key · esc cancel",
    "foot_narrow": "↑↓  move · enter  change · r  default · ←→  panes · esc back",
    "foot_pending": "↑↓  any key · enter  confirm · esc cancel",
    "foot_rest": "↑↓  move · enter  change · r  default · ←→  panes · esc back to conversation",
    "foot_suggest": "↑↓  suggestions · tab  complete · enter  save · esc cancel",
    "gate_kept": [
        "saved; this session keeps asking — /approvals auto loosens it",
        "keeps asking — /approvals auto loosens it",
        "/approvals auto loosens it",
    ],
    "hotkey": [
        "Global shortcut for the desktop app — press keys the terminal can send,"
        " or set it in the app.",
        "Global shortcut for the desktop app — set it in the app.",
        "Desktop speech-to-text shortcut — set it here or in the app.",
        "hold keys are set in the desktop app — a terminal cannot capture a held modifier",
    ],
    "hotkey_hold_painted": (
        "Global shortcut for the desktop app — set it in the app.   keymap.push_to_talk"
    ),
    "hotkey_refusal_raised": (
        "hold keys are set in the desktop app — a terminal cannot capture a held modifier"
    ),
    "notice_discarded": "discarded — enter saves, esc or moving away cancels",
    "values": [
        "auto",
        "on",
        "off",
        "1 chain",
        "2 chains",
    ],
}


@pytest.mark.asyncio
async def test_english_bytes_match_the_pre_extraction_literals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        got = await _surfaces(pilot, app, monkeypatch)
    default = _first_of(Kind.BOOL).default
    expected = dict(ENGLISH, clause_default=f"default: {'on' if default else 'off'}")
    mismatched = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    assert not mismatched, mismatched


@pytest.fixture
def pseudo_fr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production.

    COPIED, not symlinked (see the #4361 slice test): a symlinked `en` escapes
    the catalogue root and every OTHER namespace degrades to its raw key.
    """
    root = tmp_path / "catalogues"
    shutil.copytree(EN_ROOT / "en", root / "en")
    (root / "fr").mkdir()
    (root / "fr" / f"{NS}.json").write_text(
        json.dumps({k: "FR " + v for k, v in BASE.items()}, ensure_ascii=False), encoding="utf-8"
    )
    small = json.loads((EN_ROOT / "en" / "tui.small.json").read_text("utf-8"))
    (root / "fr" / "tui.small.json").write_text(
        json.dumps({k: "FR " + small[k] for k in SMALL_BOOL}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


async def _pseudo_surfaces(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        # A LIVE switch: the app is already up in English when fr arrives.
        text.set_active_locale("fr")
        return await _surfaces(pilot, app, monkeypatch)


#: How many hint labels each footer state paints; each label is one key.
_FOOTER_WORDS = {
    "foot_rest": 5,
    "foot_narrow": 5,
    "foot_edit": 4,
    "foot_suggest": 4,
    "foot_choose": 4,
    "foot_listen": 2,
    "foot_pending": 3,
}


def _unkeyed(got: dict[str, Any]) -> list[str]:
    """Every painted string that does not carry the pseudo-locale marker."""
    misses: list[str] = []
    footer_words = _FOOTER_WORDS
    for name, count in footer_words.items():
        # Each visible hint's label is keyed: one FR per hint.
        if got[name].count("FR ") != count:
            misses.append(f"{name}: {got[name]!r}")
    for name, value in got.items():
        if name in footer_words:
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        misses.extend(f"{name}: {item!r}" for item in items if "FR " not in item)
    return misses


@pytest.mark.asyncio
async def test_every_surface_is_keyed_and_follows_a_live_language_change(
    pseudo_fr, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = await _pseudo_surfaces(monkeypatch)
    assert _unkeyed(got) == [], _unkeyed(got)
    # Technical tokens survive verbatim inside the translated copy.
    assert got["foot_rest"].startswith("↑↓  FR move · enter  FR change · r  FR default"), got
    assert got["cascade_malformed"] == ["FR malformed cascade — press r to clear it"]
    assert got["gate_kept"][-1] == "FR /approvals auto loosens it"
    assert got["clause_would_remove"].endswith("preview: lop sessions cleanup --dry-run")
    # The inner value word is projected once, inside the outer sentence.
    assert got["clause_default"].startswith("FR default: FR o"), got["clause_default"]
    assert got["values"][3:] == ["FR 1 chain", "FR 2 chains"]


@pytest.mark.asyncio
async def test_mutation_control_a_restored_literal_fails_the_pseudo_check(
    pseudo_fr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring ONE literal (the footer's `move` label) must break claim 2.

    Simulated at the resolver: the key paints its English literal instead of
    resolving, exactly what reverting that call site would do.
    """
    real = sv_mod._t_chrome

    def restored(message):
        if message.code == f"{NS}.footer.move":
            return "move"
        return real(message)

    monkeypatch.setattr(sv_mod, "_t_chrome", restored)
    got = await _pseudo_surfaces(monkeypatch)
    misses = _unkeyed(got)
    assert any(miss.startswith("foot_rest:") for miss in misses), misses


@pytest.fixture
def synthetic_ur(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real shipped `ur` tree plus a SYNTHETIC translation of this namespace.

    Without a `ur` translation the copy falls back to English and the RTL
    projection never runs, so a fallback-only test proves nothing. Each value is
    `"اردو " + English`: Urdu script around Latin prose is the hardest case for
    the projection, because a one-letter key (`r`, `d`) also occurs INSIDE the
    Latin words, and the display boundary refuses an island starting mid-word.
    """
    root = tmp_path / "catalogues"
    shutil.copytree(EN_ROOT / "en", root / "en")
    shutil.copytree(EN_ROOT / "ur", root / "ur")
    (root / "ur" / f"{NS}.json").write_text(
        json.dumps({k: "اردو " + v for k, v in BASE.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


@pytest.mark.asyncio
async def test_urdu_projection_paints_every_surface(
    synthetic_ur, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every surface survives the RTL display projection (#4361 R1/R2 class).

    Also feeds the free-text params hostile input — an exception carrying a tab
    and a bidi override — which the boundary would refuse unsanitised.
    """
    assert "ur" in resolve.shipped_locales()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale("ur")
        got = await _surfaces(pilot, app, monkeypatch)
        view = app.query_one(SettingsView)

        def boom() -> None:
            raise OSError("disk\tfull \u202e reversed")

        view._save(boom)
        hostile = view.error_text
    for name, value in got.items():
        items = value if isinstance(value, (list, tuple)) else [value]
        if name.startswith("foot_") or name == "values":
            continue  # footer: joined with key glyphs; values: checked below
        assert all("اردو" in item for item in items), (name, value)
    # on/off are the REAL shipped `ur` words (reused `tui.small` keys).
    assert got["values"] == [
        "\u2067اردو auto\u2069",
        "\u2067آن\u2069",
        "\u2067آف\u2069",
        "\u2067اردو 1 chain\u2069",
        "\u2067اردو 2 chains\u2069",
    ], got["values"]
    assert all(got[name].count("اردو") == count for name, count in _FOOTER_WORDS.items()), got
    # The one-letter key is ONE isolated island, not a split word.
    assert "\u2066r\u2069" in got["cascade_malformed"][0], got["cascade_malformed"]
    assert "\u2066d\u2069" in got["confirm_short"][1], got["confirm_short"]
    assert "disk full   reversed" in hostile, hostile
