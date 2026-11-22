"""Keep display localization separate from machine selectors and legacy adapters."""

from types import SimpleNamespace
from typing import cast

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.i18n.keys import tui_model_persistence as keys
from local_operator.i18n.text import active_locale, set_active_locale
from local_operator.slash_commands import PERSIST_HINT
from local_operator.tui.app import OperatorApp, _status_line
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.model_picker import (
    PERSIST_CLAUSE_MARKER,
    PERSIST_HINT_PREFIX,
    ModelPicker,
)
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptView,
    _break_word,
    _wrap_notice_atoms,
    wrap_cells,
)
from tests.unit.tui.conftest import await_session_boot
from tests.unit.tui.test_app_pilot import (
    FakeSession,
    _AccessController,
    _factory,
    _open_model_picker,
)


class _CountingAccessController(_AccessController):
    """The pilot's catalogue fake, counting the reads a repaint must not make.

    A locale move is supposed to cost one paint of state the widget already
    holds, so the three reads that would mean it went back to the app's populate
    path — the static catalogue, the live fetch and the credential store — are
    counted here rather than inferred from timing.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003 - forwarded
        super().__init__(*args, **kwargs)
        self.catalogue_reads = 0
        self.live_fetches = 0
        self.credential_reads = 0

    def static_catalogue(self):
        self.catalogue_reads += 1
        return super().static_catalogue()

    async def live_catalogue(self, *, ttl_s=None):
        self.live_fetches += 1
        return await super().live_catalogue(ttl_s=ttl_s)

    def has_any_credential(self, provider):
        self.credential_reads += 1
        return super().has_any_credential(provider)


def _notice(*, setup: str | None = None, missing: bool = False, label: str = "") -> str:
    # This adapter supplies only the state the production method reads. Real
    # picker opening, styled wrapping and pinned-tail dedup remain pilot tests.
    state = SimpleNamespace(
        _session=SimpleNamespace(model_label=label),
        _setup_state=setup,
        _model_missing_for=missing,
    )
    # The adapter deliberately models only this synchronous method's state;
    # casting the receiver does not claim that it is a runnable Textual app.
    return OperatorApp._persist_hint_notice(cast(OperatorApp, state))


@pytest.fixture(autouse=True)
def restore_locale():
    previous = active_locale()
    yield
    set_active_locale(previous)


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
@pytest.mark.parametrize(
    ("setup", "missing", "routes"),
    [
        (None, False, ("persist", "saved", "settings")),
        ("missing_model", True, ("persist", "settings")),
        ("no_hosting", False, ("settings",)),
        ("unknown_hosting", False, ("settings",)),
    ],
)
def test_notice_resolves_each_semantic_branch_at_current_locale(locale, setup, missing, routes):
    set_active_locale(locale)
    expected_routes = " · ".join(
        getattr(keys, route)().envelope(locale=locale)["text"] for route in routes
    )
    # Selector syntax and labels are opaque input, even when they look like
    # markup. Changing either here would silently break copy/paste workflows.
    label = "vendor/[model]/v1:exact"
    expected = keys.with_model(label=label, routes=expected_routes).envelope(locale=locale)["text"]
    assert _notice(setup=setup, missing=missing, label=label) == expected
    assert _notice(setup=setup, missing=missing) == expected_routes
    assert label in expected
    for route in routes:
        command = {"persist": "/model default", "saved": "/model saved", "settings": "/settings"}[
            route
        ]
        assert command in expected


def test_notice_follows_live_locale_without_rewriting_previous_text():
    set_active_locale("en")
    english = _notice(label="opaque/model")
    assert english == (
        f"model: opaque/model — {PERSIST_HINT} · "
        "switch to the boot default with /model saved · /settings edits the boot default too"
    )
    set_active_locale("fr")
    french = _notice(label="opaque/model")
    assert french != english
    assert "saves this for new sessions" not in french
    assert "edits the boot default too" not in french
    assert keys.persist().envelope(locale="en")["text"] == PERSIST_HINT
    set_active_locale("en")
    assert _notice(label="opaque/model") == english


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (80, 24)])
@pytest.mark.parametrize("locale", ["zh-CN", "en"])
async def test_opened_notice_keeps_commands_on_one_rendered_row(monkeypatch, size, locale):
    """Exercise the opening path, not a constructed widget or normalized copy."""
    monkeypatch.setenv("LOP_LANG", locale)

    class OpaqueSession(FakeSession):
        @property
        def model_label(self) -> str:
            return "vendor/[model]/v1:exact"

    session = OpaqueSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await await_session_boot(app)
        app.query_one(Editor).focus()
        await pilot.press("slash", "m", "o", "d", "e", "l", "space")
        await pilot.pause()
        notices = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, NoticeBlock)]
        assert len(notices) == 1
        assert notices[0]._wrap_atoms == (
            "vendor/[model]/v1:exact",
            "/model default",
            "/model saved",
            "/settings",
        )
        renderable = notices[0].renderable
        assert isinstance(renderable, Text)
        rendered = renderable.plain
        assert active_locale() == locale
        for command in ("/model default", "/model saved", "/settings"):
            assert command in rendered, rendered


@pytest.mark.parametrize("width", [-1, 0, 1, 2, 8, 14, 30])
@pytest.mark.parametrize("text", ["", "ordinary prose", "  a   b  ", "a\nb", "模型甲乙丙"])
def test_default_atom_path_is_legacy(text, width):
    assert _wrap_notice_atoms(text, width, ()) == wrap_cells(text, width)
    assert _wrap_notice_atoms(text, width, ("not present",)) == wrap_cells(text, width)
    if width <= 0:
        assert _wrap_notice_atoms(text, width, (text,)) == [text]


@pytest.mark.parametrize("command", ["/model default", "/model saved", "/settings"])
def test_atoms_adjacent_to_prose_add_no_characters(command):
    text = f"前{command}后{command}{command}尾"
    rows = _wrap_notice_atoms(text, len(command), (command,))
    assert "".join(rows) == text
    assert sum(command in row for row in rows) == 3


@pytest.mark.parametrize("width", [1, 2, 8, 13, 14, 15])
def test_oversized_atom_uses_existing_cell_fallback(width):
    atom = "/model default"
    rows = _wrap_notice_atoms(atom, width, (atom,))
    assert "".join(rows) == atom
    assert all(cell_len(row) <= width for row in rows)
    if width >= cell_len(atom):
        assert rows == [atom]
    else:
        expected: list[str] = []
        tail = _break_word(atom, width, expected, cell_len(atom))
        if tail:
            expected.append(tail)
        assert rows == expected


@pytest.mark.parametrize(
    "label", ["vendor/[model]/v1:exact", "vendor/model default:2026/[opaque]", "x" * 120]
)
def test_longest_opaque_literal_wins(label):
    rows = _wrap_notice_atoms(label, len(label), ("/model default", "/settings", label))
    assert rows == [label]
    narrow = _wrap_notice_atoms(label, 8, ("/model default", label))
    expected: list[str] = []
    tail = _break_word(label, 8, expected, cell_len(label))
    if tail:
        expected.append(tail)
    assert narrow == expected
    assert "".join(narrow) == label


def test_notice_authored_newlines_indent_and_copy_stay_intact():
    text = "\n  前/model default后\n\n    /settings尾\n"
    block = NoticeBlock(text, wrap_atoms=("/model default", "/settings"))
    rows = block._rows(18)
    assert rows == ["  前/model default", "  后", "", "    /settings尾"]
    assert block.text() == text


# -- the picker FOOTER (a fixed-width row, not a wrapping notice) -------------
#
# The footer was the second surface this slice left in English: its clause was
# the import-time ``PERSIST_HINT`` the app composed into the picker's status
# string, and that string is composed when the app POPULATES the list while the
# footer is repainted later by the WIDGET (a resize, a scroll, a key, a locale
# move). A sentence baked into the status therefore kept painting English after
# a language switch. The clause now travels as ``PERSIST_CLAUSE_MARKER`` and the
# widget resolves the audited key at paint time, so these tests pin BOTH halves:
# the painted row follows the active locale, and the legacy English adapter keeps
# behaving byte-for-byte for every caller that still hands it over.


def _picker(status: str) -> ModelPicker:
    picker = ModelPicker(lambda row: None)
    picker.set_rows([], status=status)
    return picker


def _footer_rows(picker: ModelPicker, width: int = 90) -> list[str]:
    return [row.plain for row in picker._footer_rows(width)]


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
def test_footer_paints_the_persist_clause_in_the_active_locale(locale):
    """The protected footer row IS the audited key, resolved when it is painted."""
    set_active_locale(locale)
    clause = keys.persist().envelope(locale=locale)["text"]
    rows = _footer_rows(
        _picker(_status_line("cached: anthropic, openrouter", PERSIST_CLAUSE_MARKER))
    )
    # Last row, because the clause keeps its own protected row rather than losing
    # the command behind a longer status clause.
    assert rows[-1].strip() == clause, rows[-1]
    assert PERSIST_CLAUSE_MARKER not in "\n".join(rows), rows
    if locale == "en":
        assert rows[-1].strip() == PERSIST_HINT


def test_a_legacy_english_footer_status_still_renders_byte_for_byte():
    """The prefix branch is unchanged for any caller that still composes it.

    ``PERSIST_HINT`` remains the English adapter the catalogue note names for
    CLI/help/receipt consumers, and the picker still recognises a status that
    carries it. That path must stay byte-for-byte and locale-blind: it renders
    what the caller handed over, in the caller's language.
    """
    set_active_locale("en")
    english = _footer_rows(_picker(_status_line("cached: anthropic", PERSIST_HINT)))[-1]
    set_active_locale("fr")
    french = _footer_rows(_picker(_status_line("cached: anthropic", PERSIST_HINT)))[-1]
    assert english.strip() == PERSIST_HINT
    assert french == english
    assert keys.persist().envelope(locale="fr")["text"] not in french


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
def test_the_marker_is_never_prefix_parsed_and_the_status_stays_locale_free(locale):
    """The parser boundary, both directions.

    The marker is not a legacy lead (so the English prefix parser cannot claim
    it), and the app hands over the marker — never the sentence — so translated
    bytes never enter the status the widget stores. That matters because a French
    clause DOES open with the same command and would be captured by the legacy
    prefix; it would then be rendered byte-for-byte and would stop following the
    locale on the next paint.
    """
    set_active_locale(locale)
    clause = keys.persist().envelope(locale=locale)["text"]
    assert not PERSIST_CLAUSE_MARKER.startswith(PERSIST_HINT_PREFIX)
    assert clause.startswith(PERSIST_HINT_PREFIX)
    # What the app composes, in every locale, is the marker.
    state = SimpleNamespace(_setup_state=None, _model_missing_for=None)
    assert OperatorApp._footer_persist_hint(cast(OperatorApp, state)) == PERSIST_CLAUSE_MARKER
    # Painting resolves the marker and STORES nothing: the widget's status is
    # still the marker afterwards, which is what lets the next paint re-resolve.
    picker = _picker(PERSIST_CLAUSE_MARKER)
    assert _footer_rows(picker)[-1].strip() == clause
    assert picker._status == PERSIST_CLAUSE_MARKER
    assert clause not in picker._status


@pytest.mark.asyncio
async def test_open_picker_paints_a_footer_in_the_configured_language(monkeypatch):
    """Real open, real frame: the footer row is the French clause, command exact."""
    monkeypatch.setenv("LOP_LANG", "fr")
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=_AccessController())
    async with app.run_test(size=(100, 30)) as pilot:
        await await_session_boot(app)
        assert active_locale() == "fr"
        picker = await _open_model_picker(app, pilot)
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        row = picker.render_text(100).plain.split("\n")[-1]
    clause = keys.persist().envelope(locale="fr")["text"]
    # The command is an ASCII literal in the translated clause: byte-exact and in
    # LTR reading order, so it stays copy-pasteable.
    assert row.strip().startswith("/model default "), row
    assert row.strip() == clause, row
    assert not {"\u202a", "\u202b", "\u202c", "\u2066", "\u2067", "\u2068", "\u2069"} & set(row)
    # …and it reached the FRAME, not just the widget's return value.
    assert any(line.strip() == clause for line in painted), painted
    assert not any(PERSIST_HINT in line for line in painted), painted


@pytest.mark.asyncio
async def test_live_locale_move_repaints_the_open_footer_without_reloading(monkeypatch):
    """A language switch repaints the opened list and reads nothing again.

    The clause is resolved at paint time, so the app has to ask for that paint —
    and the paint must be all it costs: no credential read, no catalogue
    repopulation, no provider worker.
    """
    monkeypatch.setenv("LOP_LANG", "en")
    ctrl = _CountingAccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await await_session_boot(app)
        await _open_model_picker(app, pilot)
        before_paint = [strip.text for strip in app.screen._compositor.render_strips()]
        before_reads = (ctrl.catalogue_reads, ctrl.live_fetches, ctrl.credential_reads)
        app._apply_language("fr")
        await pilot.pause()
        await pilot.pause()
        after_paint = [strip.text for strip in app.screen._compositor.render_strips()]
    french = keys.persist().envelope(locale="fr")["text"]
    # The transcript's already-printed notice keeps the language it was printed
    # in — historical blocks are never rewritten — so the proof is about the
    # FOOTER row: it carried the whole English hint before the move and the whole
    # French clause after it. Those are the only full-length rows of either.
    assert any(line.strip() == PERSIST_HINT for line in before_paint), before_paint
    assert not any(line.strip() == PERSIST_HINT for line in after_paint), after_paint
    assert any(line.strip() == french for line in after_paint), after_paint
    assert (ctrl.catalogue_reads, ctrl.live_fetches, ctrl.credential_reads) == before_reads
