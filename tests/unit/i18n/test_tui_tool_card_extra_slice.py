"""The tool card's remaining words, keyed under ``tui.tool_card.extra``.

WHY THIS FILE EXISTS. After the card's chrome (``tui.tool_card``) and status
words (``tui.tool_card.status``) were keyed, a language switch still left
English on the card: the web_fetch ``Rendered:`` row's ``N lines`` / ``N
attempts``, the never-sent record (``never sent · N composed`` / ``nothing
composed``) and the live ``composing…`` label, the send row's mode markers
(``wake`` / ``quiet`` / ``now``) and model-switch ``model`` marker, the
web_search ``Provider:`` / ``Sources:`` rows and ``Untitled result``, and the
``not delivered: <cause>`` joiner. They live in a third child namespace so
neither audited parent's bytes move.

Keying the ``Fetched:``/``Rendered:``/``Provider:`` rows had a second cost the
earlier slice paid silently: the body painters picked each row's ink by
matching its ENGLISH lead, so a translated row lost its dim/link treatment.
The painters now style by the kind the row builder records
(``_HeaderRow``), and this file proves it in a non-English locale.

Claims, each driven through the REAL paint paths (``_build_row``,
``_build_content``, ``_status_runs``), never the catalogue:

1. English is byte-identical to the pre-extraction literals, spelled out here.
2. With a synthetic ``fr`` carrying only this namespace, every surface paints
   the fr word after a LIVE language change, settled cards included.
3. ICU plurals select ``one``/``other`` from the count.
4. The failed-send joiner is the locale's: a full-width colon survives the
   word-first cap, and the cause is what gets cut.
5. In fr the fetch/search rows keep their styling (URL and attempt count in
   ``signal``, labels and metadata in ``dim``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from rich.style import Style
from rich.text import Text

from local_operator.i18n import catalogues, resolve, text
from local_operator.tui import bindings
from local_operator.tui.widgets.tool_card import ToolCard

NS = "tui.tool_card.extra"
EN_ROOT = Path(catalogues.__file__).resolve().parent / "catalogues"
BASE: dict[str, str] = json.loads((EN_ROOT / "en" / f"{NS}.json").read_text("utf-8"))

URL = "https://example.com/page"
FETCH_DETAILS: dict[str, object] = {
    "url": URL,
    "final_url": URL,
    "status": 200,
    "content_type": "text/html",
    "render_method": "markdownify",
    "lines": 800,
    "bytes": 53_000,
    "attempts": 2,
    "profiles": ["default", "browser"],
    "cache": "miss",
}
SEARCH_DETAILS: dict[str, object] = {
    "provider": "brave",
    "auth_mode": "api_key",
    "sources": [{"title": "", "url": "https://a.example", "snippet": "a snippet"}],
}


def _fetch_card(**overrides: object) -> ToolCard:
    card = ToolCard("f", "web_fetch", {"url": URL})
    card.mark_done(
        "[200] " + URL + "\nmarkdownify · text/html · cache miss\n\npage body",
        {**FETCH_DETAILS, **overrides},
    )
    card.toggle_expanded()
    return card


def _search_card() -> ToolCard:
    card = ToolCard("s", "web_search", {"query": "q"})
    card.mark_done("results", dict(SEARCH_DETAILS))
    card.toggle_expanded()
    return card


def _never_sent(argument_bytes: int) -> ToolCard:
    card = ToolCard("n", "bash", {"command": "x"})
    card.set_composing(argument_bytes, "bash")
    card.mark_interrupted()
    return card


def _send(args: dict[str, object]) -> ToolCard:
    return ToolCard("p", "send", {"target": "release-owner", "message": "hi", **args})


def _surfaces() -> dict[str, str]:
    """Every keyed surface, read from what the card actually paints."""
    out: dict[str, str] = {}
    out["fetch"] = _fetch_card()._build_content(140).plain
    out["fetch_one"] = (
        _fetch_card(lines=1, attempts=3, profiles=["default"])._build_content(140).plain
    )
    out["search"] = _search_card()._build_content(140).plain
    out["never_sent"] = _never_sent(1300)._build_row(120).plain
    out["nothing"] = _never_sent(0)._build_row(120).plain
    composing = ToolCard("c", "bash", {"command": "x"})
    composing.set_composing(1300, "bash")
    out["composing"] = composing._build_row(120).plain
    out["wake"] = _send({})._build_row(120).plain
    out["quiet"] = _send({"wake": False})._build_row(120).plain
    out["now"] = _send({"now": True})._build_row(120).plain
    out["model"] = ToolCard("m", "send", {"pid": 48213, "model": "x/y"})._build_row(120).plain
    delivered = _send({"wake": False})
    delivered.mark_done("ok", {"delivery": {"state": "delivered"}})
    out["settled_quiet"] = delivered._build_row(120).plain
    lost = _send({})
    lost.mark_failed(
        "could not deliver",
        details={"delivery": {"state": "failed", "reason": "this session cannot receive"}},
    )
    out["not_delivered"] = lost._build_row(200).plain
    return out


def test_english_bytes_match_the_pre_extraction_literals() -> None:
    got = _surfaces()
    assert (
        "Rendered: markdownify · 2 attempts (default, browser-profile) · 800 lines · 51.8 KB"
        in got["fetch"]
    )
    # Single identity: no parenthetical; one line: the singular.
    assert "Rendered: markdownify · 3 attempts · 1 line · 51.8 KB" in got["fetch_one"]
    assert "Provider: brave (api_key)" in got["search"]
    assert "Sources:" in got["search"]
    assert "1. Untitled result" in got["search"]
    assert "never sent · 1.3 KB composed" in got["never_sent"]
    assert "never sent · nothing composed" in got["nothing"]
    assert "composing… 1.3 KB" in got["composing"]
    assert "wake · release-owner · hi" in got["wake"]
    assert "quiet · release-owner · hi" in got["quiet"]
    assert "now · release-owner · hi" in got["now"]
    assert "model · pid 48213 · → x/y" in got["model"]
    assert "quiet · release-owner · hi" in got["settled_quiet"]
    assert "not delivered: this session cannot receive" in got["not_delivered"]


def _pseudo(namespace: str) -> dict[str, str]:
    source = json.loads((EN_ROOT / "en" / f"{namespace}.json").read_text("utf-8"))
    return {k: f"FR {v}" for k, v in source.items()}


def _install_fr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    # A COPY, not a symlink: `catalogues._file` refuses a path that resolves out
    # of the root, and a lost `en` would paint raw key IDs instead of English.
    shutil.copytree(
        EN_ROOT / "en", tmp_path / "en", ignore=shutil.ignore_patterns("*.context.json")
    )
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / f"{NS}.json").write_text(json.dumps(values, ensure_ascii=False), "utf-8")
    # The rows the styling claim is about carry PARENT-namespace leads too
    # (``Fetched:`` in tui.tool_card, ``Rendered:`` in tui.small), so those are
    # pseudo-translated as well: a lead that stayed English would let a painter
    # that still matched English text pass.
    for parent in ("tui.tool_card", "tui.small"):
        (tmp_path / "fr" / f"{parent}.json").write_text(
            json.dumps(_pseudo(parent), ensure_ascii=False), "utf-8"
        )
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()


@pytest.fixture
def fr_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pseudo-locale: every value prefixed ``FR ``, the joiner a full-width colon.

    Only this fixture may make a synthetic locale eligible, never production.
    """
    values = {k: f"FR {v}" for k, v in BASE.items()}
    values[f"{NS}.send.failed_reason"] = "{word}：{cause}"
    _install_fr(tmp_path, monkeypatch, values)
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


def test_every_surface_follows_a_live_language_change(fr_extra) -> None:
    english = _surfaces()  # built AND painted in English first
    text.set_active_locale("fr")
    got = _surfaces()
    assert "FR 2 attempts (default, browser-profile)" in got["fetch"]
    assert "FR 800 lines" in got["fetch"]
    assert "FR 3 attempts · FR 1 line" in got["fetch_one"]
    assert "FR Provider: brave (api_key)" in got["search"]
    assert "FR Sources:" in got["search"]
    assert "1. FR Untitled result" in got["search"]
    assert "FR never sent · FR 1.3 KB composed" in got["never_sent"]
    assert "FR never sent · FR nothing composed" in got["nothing"]
    assert "FR composing… 1.3 KB" in got["composing"]
    assert "FR wake · release-owner · hi" in got["wake"]
    assert "FR quiet · release-owner · hi" in got["quiet"]
    assert "FR now · release-owner · hi" in got["now"]
    assert "FR model · pid 48213 · → x/y" in got["model"]
    assert "FR quiet · release-owner · hi" in got["settled_quiet"]
    assert "not delivered：this session cannot receive" in got["not_delivered"]
    assert got != english


def test_settled_cards_repaint_on_a_language_change(fr_extra) -> None:
    """Built and settled in English, painted again after ``/language fr``."""
    fetch, search = _fetch_card(), _search_card()
    never, nothing = _never_sent(1300), _never_sent(0)
    delivered = _send({"wake": False})
    delivered.mark_done("ok", {"delivery": {"state": "delivered"}})
    switched = ToolCard("m", "send", {"pid": 48213, "model": "x/y"})
    switched.mark_done("ok", {"outcome": "nonsense-outcome"})  # keeps the marker
    for card in (fetch, search):
        assert (
            "Rendered:" in card._build_content(140).plain
            or "Provider:" in card._build_content(140).plain
        )
    assert "never sent · 1.3 KB composed" in never._build_row(120).plain
    assert "model · pid 48213" in switched._build_row(120).plain
    text.set_active_locale("fr")
    assert "FR 2 attempts" in fetch._build_content(140).plain
    assert "FR Sources:" in search._build_content(140).plain
    assert "FR never sent · FR 1.3 KB composed" in never._build_row(120).plain
    assert "FR never sent · FR nothing composed" in nothing._build_row(120).plain
    assert "FR quiet · release-owner" in delivered._build_row(120).plain
    assert "FR model · pid 48213" in switched._build_row(120).plain
    text.set_active_locale("en")
    assert "never sent · 1.3 KB composed" in never._build_row(120).plain
    assert "FR" not in never._build_row(120).plain


def test_failed_send_cap_keeps_the_word_and_the_locale_joiner(fr_extra) -> None:
    """The cap composes from parts; it no longer splits on an English ``": "``."""
    text.set_active_locale("fr")
    lost = _send({})
    lost.mark_failed(
        "could not deliver",
        details={"delivery": {"state": "failed", "reason": "this session cannot receive"}},
    )
    word = "not delivered"  # tui.tool_card.status is not in this fr catalogue
    for cap in (40, 34, 30):
        reason = "".join(run for run, _style in lost._status_runs(cap))
        assert f"{word}：" in reason, (cap, reason)
        assert "…" in reason, (cap, reason)  # the CAUSE gave way, not the word


def _styles_over(content: Text, needle: str, after: str = "") -> set[Style | str]:
    """Every style a span applies inside ``needle``'s first occurrence at or
    after ``after`` (the summary row above the body repeats the URL)."""
    start = content.plain.index(needle, content.plain.index(after) if after else 0)
    end = start + len(needle)
    return {span.style for span in content.spans if span.start < end and span.end > start}


def test_translated_fetch_and_search_rows_keep_their_styling(fr_extra) -> None:
    text.set_active_locale("fr")
    signal, dim = bindings.style("tool.fetch.signal"), bindings.style("tool.fetch.dim")
    muted = bindings.style("tool.fetch.snippet")
    fetch = _fetch_card()._build_content(140)
    # The URL token on the Fetched row is the anchor ink, its label is dim. The
    # pre-fix painter matched ``Fetched:`` and painted this row ``muted`` whole.
    assert _styles_over(fetch, URL, after="FR Fetched:") == {signal}
    assert _styles_over(fetch, "FR Fetched:") == {dim}
    assert muted not in _styles_over(fetch, "markdownify")
    assert _styles_over(fetch, "FR 2 attempts (default, browser-profile)") == {signal}
    assert _styles_over(fetch, "FR 800 lines") == {dim}

    search_dim = bindings.style("tool.search.dim")
    search = _search_card()._build_content(140)
    assert _styles_over(search, "FR Provider: brave (api_key)") == {search_dim}
    assert _styles_over(search, "FR Sources:") == {search_dim}
