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
from rich.cells import cell_len

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
        # 64 cells: the long contract overruns, the short one fits whole with
        # the full name. Narrower widths shed further (see the 60-column test).
        patch.setattr(view, "_detail_width", lambda: 64)
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


@pytest.fixture
def shipped_tree(monkeypatch: pytest.MonkeyPatch):
    """The REAL shipped catalogue tree and shipped set — no synthetic locale.

    The pseudo/synthetic fixtures prove the wiring; this one proves each audited
    translation reaches the painted frame. A locale missing its ledger rows
    would drop out of ``shipped.json`` and the page would silently paint
    English, and a real RTL/Devanagari value the display boundary refuses would
    only crash at paint — neither is visible to the synthetic fixtures.
    """
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


#: Surfaces whose copy carries no technical placeholder, so the catalogue value
#: must appear verbatim in the painted string (RTL isolates wrap, never split, it).
_PLAIN_SURFACES = {"err_retired": "error.retired", "err_not_text": "error.not_text"}


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["fr", "es", "zh-CN", "ru", "vi", "hi", "ur"])
async def test_shipped_catalogue_paints_every_surface(
    shipped_tree, locale: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert locale in resolve.shipped_locales()
    shipped = json.loads((EN_ROOT / locale / f"{NS}.json").read_text(encoding="utf-8"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale(locale)
        got = await _surfaces(pilot, app, monkeypatch)
    for surface, key in _PLAIN_SURFACES.items():
        assert shipped[f"{NS}.{key}"] in got[surface], (surface, got[surface])
    left = {
        k: v
        for k, v in got.items()
        if k in ("foot_rest", "notice_discarded", *_PLAIN_SURFACES) and v == ENGLISH[k]
    }
    assert not left, left


# -- PR #4405 round-1 remediation ------------------------------------------


def _boom(exc: BaseException):
    def action() -> None:
        raise exc

    return action


@pytest.mark.asyncio
async def test_stored_error_and_notice_follow_a_live_language_switch(
    pseudo_fr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: a message already ON SCREEN re-resolves when the language moves.

    Negative control: storing the rendered string (the round-1 head) keeps
    `could not save: E123` in English after the switch and fails here.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view._save(_boom(OSError("E123")))
        assert view.error_text == "could not save: E123"
        text.set_active_locale("fr")
        view._paint_detail()
        assert view.error_text == "FR could not save: E123", view.error_text
        assert "FR could not save: E123" in view._detail_text.plain
        text.set_active_locale("en")
        view._error = ""
        settings_io.write_chains(view._manager, {"default": ["openai/gpt-4o"]}, base={})
        view._repaint()
        _select(view, _first_of(Kind.CASCADE).key)
        view.action_reset()
        refused = "r resets one setting; delete a chain with d on its row"
        assert view.notice_text == refused
        text.set_active_locale("fr")
        view._paint_detail()
        assert view.notice_text == "FR " + refused, view.notice_text
        assert ("FR " + refused) in view._detail_text.plain


def test_detail_param_keeps_zwnj_and_zwj_and_blanks_only_forbidden_controls() -> None:
    """R2: only what the display boundary refuses is blanked.

    Negative control: the round-1 head blanked every `Cf`, turning the Urdu
    ZWNJ and the emoji ZWJ below into spaces.
    """
    urdu = "نی‌لا"  # ZWNJ inside one word
    emoji = "👩‍💻"  # woman + ZWJ + laptop
    hostile = "\t\u202e\u2066\u2028x"
    got = sv_mod._detail_param(f"{urdu} {emoji} {hostile}")
    assert got == f"{urdu} {emoji}     x", repr(got)


@pytest.mark.asyncio
async def test_urdu_save_error_keeps_zwnj_and_zwj_on_the_painted_row(
    shipped_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2 end to end: the real `ur` projection admits ZWNJ/ZWJ, so they survive."""
    urdu = "نی‌لا"
    emoji = "👩‍💻"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale("ur")
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view._save(_boom(OSError(f"{urdu} {emoji}")))
        view._paint_detail()
    assert urdu in view.error_text and emoji in view.error_text, repr(view.error_text)


@pytest.mark.asyncio
async def test_urdu_save_against_an_unreadable_config_reports_instead_of_crashing(
    shipped_tree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1: the REAL `ConfigUnreadableError` names `…/config/config.yml`.

    With `{file}` declared technical, its span matched `config.yml` inside that
    path too, after a `/`, and the boundary raised `technical status span must
    include its whole prefix` from the handler reporting the failure.
    Negative control: re-adding `file` to `technical_params` raises here.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale("ur")
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        config = tmp_path / "config" / "config.yml"
        config.write_text("values: [unclosed\n", encoding="utf-8")
        setting = _first_of(Kind.BOOL)
        assert view._save(lambda: settings_io.reset_setting(view._manager, setting)) is False
        view._paint_detail()
    shipped = json.loads((EN_ROOT / "ur" / f"{NS}.json").read_text(encoding="utf-8"))
    template = shipped[f"{NS}.error.config_unreadable"]
    lead = template.split("{error}")[0].replace("{file}", "config.yml")
    assert lead in view.error_text, view.error_text
    assert str(config) in view.error_text, view.error_text


_SHIPPED_PLUS_EN = ["en", "fr", "es", "zh-CN", "ru", "vi", "hi", "ur"]
_STRIP_TAIL = "\u2066\u2067\u2068\u2069 "


def _assert_balanced(plain: str, what: str) -> None:
    # A cut or shed `ur` line must close every isolate it opens, or the
    # direction leaks into whatever the terminal paints next.
    assert sv_mod._open_isolates(plain) == 0, (what, plain)
    assert plain.count("\u2069") == sum(plain.count(c) for c in "\u2066\u2067\u2068"), (
        what,
        plain,
    )


def _assert_one_row(view: SettingsView, plain: str, whole: str, what: str) -> None:
    width = view._detail_width()
    # The budget IS the painted row: no hidden second line can exist.
    assert cell_len(plain) <= width, (what, cell_len(plain), width, plain)
    _assert_balanced(plain, what)
    tail = plain.rstrip(_STRIP_TAIL)
    assert not tail.rstrip("…").rstrip(_STRIP_TAIL).endswith("·"), (what, plain)
    if cell_len(whole) > width:
        assert "…" in plain, (what, plain)
    else:
        assert plain == whole, (what, plain, whole)


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", _SHIPPED_PLUS_EN)
async def test_detail_band_never_hides_a_wrapped_line_at_60_columns(
    shipped_tree, locale: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1/Q2: at 60x20 the one-row band cuts with `…`, never wraps out of sight.

    Negative control: on the round-1 head the fr/ru discarded notice and the
    ru delete ask exceed the 58-cell row with no `…`, and the ask keeps a
    trailing `·` once the frame clips it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        text.set_active_locale(locale)
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        # The budget never exceeds the widget's real content width (geometry,
        # not belief): a line within it cannot wrap to a hidden second row.
        assert view._detail_width() <= view._detail.content_region.width, (
            view._detail_width(),
            view._detail.content_region.width,
        )
        # Discarded notice, through the real settle path.
        view._editing, view._buffer, view._edit_seed, view._discard_announced = (
            "retry.baseDelayMs",
            "9",
            "1",
            False,
        )
        view._settle_row()
        view._paint_detail()
        _assert_one_row(view, view._detail_text.plain, view._notice, "discarded")
        # Reset refused on a healthy cascade, through the real `r` action.
        view._notice = view._error = ""
        settings_io.write_chains(view._manager, {"default": ["openai/gpt-4o"]}, base={})
        view._repaint()
        _select(view, _first_of(Kind.CASCADE).key)
        view.action_reset()
        view._paint_detail()
        _assert_one_row(view, view._detail_text.plain, view._notice, "reset_refused")
        # The delete ask, short name and a long one.
        view._notice = ""
        for chain in ("default", "openrouter-budget-fallback-chain"):
            settings_io.write_chains(view._manager, {chain: ["openai/gpt-4o"]}, base={})
            view._confirm_delete = chain
            view._paint_detail()
            plain = view._detail_text.plain
            assert cell_len(plain) <= view._detail_width(), (chain, plain)
            _assert_balanced(plain, chain)
            assert not plain.rstrip(_STRIP_TAIL).endswith("·"), (chain, plain)
            # The confirming key is never shed.
            contract = view._confirm_parts()[1]
            assert plain.endswith(contract) and "d" in contract, (plain, contract)
            view._confirm_delete = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "contract"),
    [
        ((120, 40), "press d again to confirm · esc cancels"),
        ((80, 24), "d confirms · esc cancels"),
    ],
)
async def test_english_detail_band_is_unchanged_at_120_and_80(
    shipped_tree, size: tuple[int, int], contract: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """English frames at 120/80 are byte-identical to the pre-remediation paint."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view._editing, view._buffer, view._edit_seed, view._discard_announced = (
            "retry.baseDelayMs",
            "9",
            "1",
            False,
        )
        view._settle_row()
        view._paint_detail()
        assert view._detail_text.plain == "discarded — enter saves, esc or moving away cancels"
        view._notice = ""
        settings_io.write_chains(view._manager, {"default": ["openai/gpt-4o"]}, base={})
        view._repaint()
        _select(view, _first_of(Kind.CASCADE).key)
        view.action_reset()
        view._paint_detail()
        assert view._detail_text.plain == "r resets one setting; delete a chain with d on its row"
        view._notice = ""
        view._confirm_delete = "default"
        view._paint_detail()
        assert view._detail_text.plain == (f"▸ delete chain default and its 1 hop?  {contract}")


_ENUM_LABEL_CASES = (
    ("speech.voice.tone", "warm"),
    ("speech.voice.expressiveness", "medium"),
    ("providers.openai.api", "responses"),
    ("tool_approval_mode", "ask"),
    ("tool_approval_mode", "auto"),
    ("retry.accountSelectionPolicy", "reset-soonest"),
    ("display.dock", "full"),
    ("display.nerd_icons", None),
    ("display.nerd_icons", True),
    ("display.nerd_icons", False),
)


@pytest.mark.parametrize("locale", ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"))
def test_enum_summary_uses_registered_label_in_every_locale(locale: str) -> None:
    """Actual shipped copy, not synthetic parity: picker, summary and raw slot agree."""
    from local_operator.i18n.messages import render_record
    from local_operator.tui.i18n_status import localized_display

    previous = text.active_locale()
    try:
        text.set_active_locale(locale)
        assert text.active_locale() == locale
        for key, value in _ENUM_LABEL_CASES:
            setting = settings_io.BY_KEY[key]
            choice = next(
                c
                for c in setting.resolved_choices
                if type(c.value) is type(value) and c.value == value
            )
            message = choice.label.message
            record = render_record(message.code, message.params, locale=locale)
            expected = localized_display(record.text, source_locale=record.source_locale).plain
            assert sv_mod._render_value(value, setting=setting) == expected, (locale, key)
            assert sv_mod._choice_label(setting, choice) == expected, (locale, key)
            assert sv_mod._render_value(value, setting=setting, raw=True) == record.text
            assert not any(c in record.text for c in "\u2066\u2067\u2068\u2069")
            assert sv_mod._edit_seed(value) == ("" if value is None else str(value))
    finally:
        text.set_active_locale(previous)


def test_enum_label_technical_and_non_enum_values_stay_unchanged() -> None:
    from local_operator.tui.i18n_status import technical_display

    previous = text.active_locale()
    try:
        for locale in ("fr", "zh-CN", "ur"):
            text.set_active_locale(locale)
            language = settings_io.BY_KEY["language"]
            native = next(c for c in language.resolved_choices if c.value == "fr")
            expected = technical_display(native.label).plain
            assert sv_mod._render_value("fr", setting=language) == expected
            theme = settings_io.BY_KEY["tui.theme"]
            choice = theme.resolved_choices[0]
            assert sv_mod._render_value(choice.value, setting=theme) == choice.label
            boolean = _first_of(Kind.BOOL)
            for value in (True, False, None, ["warm", "auto"], "/warm/auto", "ctrl+s"):
                assert sv_mod._render_value(value, setting=boolean) == sv_mod._render_value(value)
            assert sv_mod._edit_seed(["warm", "auto"]) == "warm, auto"
    finally:
        text.set_active_locale(previous)


def test_enum_label_repaint_resolves_only_match_without_catalogue_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting = settings_io.BY_KEY["speech.voice.tone"]
    sv_mod._render_value("warm", setting=setting)  # warm the shared namespace caches
    real = sv_mod.tr_record
    calls = []

    def resolve(namespace):
        calls.append(namespace)
        return real(namespace)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("a warm repaint must not reread catalogue files")

    monkeypatch.setattr(sv_mod, "tr_record", resolve)
    monkeypatch.setattr(catalogues, "load_catalogue", unexpected_read)
    for _ in range(3):
        sv_mod._render_value("warm", setting=setting)
    assert calls == ["wire.settings"] * 3
    calls.clear()
    assert sv_mod._render_value("removed-choice", setting=setting) == "removed-choice"
    assert calls == []
    assert sv_mod._render_value(["warm"], setting=setting) == "warm"
    assert sv_mod._render_value("/warm/auto") == "/warm/auto"


def test_enum_label_urdu_and_english_fallback_keep_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A translated Urdu label projects once; missing Urdu copy stays English."""
    from local_operator.i18n.keys import wire_settings as keys
    from local_operator.tui.i18n_status import localized_display

    message = keys.speech_voice_tone_choice_warm_label()
    root = tmp_path / "label-catalogues"
    for locale, source in (("en", "warm"), ("ur", "وارم")):
        (root / locale).mkdir(parents=True)
        (root / locale / "wire.settings.json").write_text(
            json.dumps({message.code: source}), encoding="utf-8"
        )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(text, "_active_locale", "ur")
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ur"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    setting = settings_io.BY_KEY["speech.voice.tone"]
    choice = settings_io.Choice("warm", settings_io._LocalizedText("stale", message))
    assert sv_mod._choice_label(setting, choice, raw=True) == "وارم"
    expected = localized_display("وارم", source_locale="ur").plain
    assert sv_mod._choice_label(setting, choice) == expected
    # A separate immutable root avoids invalidating caches to manufacture a pass.
    fallback = tmp_path / "fallback-catalogues"
    (fallback / "en").mkdir(parents=True)
    (fallback / "en" / "wire.settings.json").write_text(
        json.dumps({message.code: "warm"}), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", fallback)
    assert sv_mod._choice_label(setting, choice) == "warm"
    assert sv_mod._choice_label(setting, choice, raw=True) == "warm"


@pytest.mark.asyncio
async def test_enum_summary_and_default_follow_live_locale() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    previous = text.active_locale()
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._open_settings_view()
            view = app.query_one(SettingsView)
            setting = settings_io.BY_KEY["speech.voice.tone"]
            settings_io.write_setting(view._manager, setting, "neutral")
            _select(view, setting.key)
            row = view._rows[view._selected]
            for locale in ("fr", "zh-CN", "ur"):
                text.set_active_locale(locale)
                view._repaint()
                choices = sv_mod._choices_for(setting)
                selected = next(c for c in choices if c.value == "neutral")
                default = next(c for c in choices if c.value == setting.default)
                label = sv_mod._choice_label(setting, selected)
                assert label in view._row_text(row, view._selected, 96).plain
                expected = sv_mod._t_chrome(
                    sv_mod._svc.clause_default(
                        value=sv_mod._choice_label(setting, default, raw=True)
                    )
                )
                assert view._detail_clause(row) == expected
                view.action_activate()
                choice_row = next(
                    r for r in view._rows if r.kind == "choice" and r.choice.value == "neutral"
                )
                assert label in view._row_text(choice_row, view._rows.index(choice_row), 96).plain
                view._expanded = None
                view._repaint()
                _select(view, setting.key)
    finally:
        text.set_active_locale(previous)
