"""The tool card's own chrome, keyed under ``tui.tool_card``.

WHY THIS FILE EXISTS. The operator's goal is that switching language leaves no
English in the interface, and every tool call in the transcript still painted
English chrome: the ``⟨expand⟩``/``⟨collapse⟩`` affordances, the live block's
``⋯ running · no output yet`` header, the ``… N more lines`` family of elision
markers, the image-generation ``⟨ctrl+c to interrupt⟩`` hint and queue line,
the ``Fetched:`` row, the ``stop unconfirmed`` status and the ``you:`` chip.
They reached the screen through ``truncate_cells``, module constants and
f-strings the i18n ratchet cannot see, so its zero count proved nothing here;
this file is the guard instead.

Two claims, each driven through the REAL paint paths (``_build_row`` and
``_build_content``), never the catalogue:

1. English is byte-identical to the pre-extraction literals. The expected
   strings are spelled out literally on purpose — reading them back from the
   catalogue would let a wording change slip through as "still equal".
2. With a synthetic ``fr`` carrying only this namespace, every keyed surface
   paints the fr text after a LIVE language change, while the machine tokens
   (``ctrl+c``, URLs, the ``miss`` cache token, the ``…``/``⋯``/``⟨⟩`` glyphs)
   stay verbatim. The fixture COPIES the real English tree (the loader refuses
   symlinked catalogue paths as escaping the root, so a symlinked ``en`` would
   paint raw key IDs) so every other namespace keeps rendering; only this
   fixture makes the synthetic locale eligible.
3. With the SHIPPED fr catalogue (no fixture tree), the audited translation
   reaches the painted card.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.tui.widgets import tool_card
from local_operator.tui.widgets.tool_card import (
    EXPAND_MAX_LINES,
    LIVE_MAX_LINES,
    ToolCard,
)

NS = "tui.tool_card"
EN_ROOT = Path(catalogues.__file__).resolve().parent / "catalogues"
BASE: dict[str, str] = json.loads((EN_ROOT / "en" / f"{NS}.json").read_text("utf-8"))


def _fr(value: str) -> str:
    """Prefix every literal run but keep ICU syntax intact (plural branches too)."""
    if value.startswith("{count, plural"):
        return value.replace("one {", "one {FR ").replace("other {", "other {FR ")
    return "FR " + value


def _surfaces() -> dict[str, str]:
    """Every keyed surface, read from what the card actually paints."""
    out: dict[str, str] = {}

    search = ToolCard("s", "web_search", {"query": "q"})
    search.mark_done("\n".join(f"hit {i}" for i in range(EXPAND_MAX_LINES + 5)))
    out["expand"] = search._build_row(120).plain
    search.toggle_expanded()
    out["collapse"] = search._build_row(120).plain
    out["search_more"] = search._build_content(120).plain.splitlines()[-1].strip()

    read = ToolCard("r", "read", {"path": "big.txt"})
    read.mark_done("\n".join(f"line {i}" for i in range(EXPAND_MAX_LINES + 1)))
    read.toggle_expanded()
    out["more_one"] = read._build_content(80).plain.splitlines()[-1].strip()

    diff = ["--- ", "+++ ", "@@ -0,0 +1,120 @@", *(f"+line {n}" for n in range(120))]
    write = ToolCard("w", "write", {"path": "a.py"})
    write.mark_done("Wrote a.py", {"path": "a.py", "added": 120, "removed": 0, "diff": diff})
    write.diff_call_id = ""
    write.toggle_expanded()
    out["diff_more"] = write._build_content(100).plain.splitlines()[-1].strip()

    idle = ToolCard("b", "bash", {"command": "sleep 3"})
    idle._expanded = True
    out["live_pending"] = idle._build_content(80).plain.splitlines()[-1].strip()

    tail = ToolCard("t", "bash", {"command": "seq 500"})
    tail._expanded = True
    tail.set_partial_detail("\n".join(str(n) for n in range(1, 501)))
    out["live_dropped"] = tail._build_content(80).plain

    elided = ToolCard("e", "bash", {"command": "seq 1000000"})
    elided._expanded = True
    elided.set_partial_detail("\n".join(str(n) for n in range(1, 1_000_001)))
    out["live_elided"] = elided._build_content(80).plain

    image = ToolCard("g0", "generate_image", {"prompt": "a cat"})
    image.begin_running("generate_image", {"prompt": "a cat"}, None)
    image._expanded = True
    image.set_live_details({"stage": "running", "queue_position": 1})
    out["image_row"] = image._build_row(100).plain
    image.set_live_details({"stage": "running", "queue_position": 3})
    out["image_body"] = image._build_content(100).plain

    fetch = ToolCard("f", "web_fetch", {"url": "https://example.com/docs"})
    fetch.mark_done(
        "[200] https://example.com/docs\n\n# Docs\n\nBody.",
        {
            "url": "https://example.com/docs",
            "final_url": "https://example.com/docs/",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 1234,
            "lines": 40,
        },
    )
    fetch.toggle_expanded()
    out["fetched"] = next(
        line.strip()
        for line in fetch._build_content(160).plain.splitlines()
        if "https://example.com/docs/" in line
    )

    stop = ToolCard("p", "sessions", {"session": "payments-audit"})
    stop.mark_done("stop accepted, not exited yet.", {"op": "stop", "method": "pending"})
    out["stop"] = stop._build_row(80).plain

    mine = ToolCard("u", "bash", {"command": "git status"}, user_run=True)
    mine.mark_done("x")
    out["you"] = mine._build_row(80).plain
    return out


def test_english_bytes_match_the_pre_extraction_literals() -> None:
    got = _surfaces()
    assert "⟨expand⟩ ✓" in got["expand"]
    assert "⟨collapse⟩ ✓" in got["collapse"]
    assert got["search_more"] == "… 5 more search lines"
    assert got["more_one"] == "… 1 more line"
    assert got["diff_more"] == f"… {121 - EXPAND_MAX_LINES} more diff lines ⟨expand⟩"
    assert got["live_pending"] == "⋯ running · 0s · no output yet"
    assert f"\n  … {500 - LIVE_MAX_LINES} earlier lines\n" in got["live_dropped"]
    assert "\n  … earlier output not shown\n" in got["live_elided"]
    assert "⟨ctrl+c to interrupt⟩" in got["image_row"]
    assert "\n  3 requests ahead" in got["image_body"]
    assert got["fetched"].startswith(
        "Fetched: https://example.com/docs  (final: https://example.com/docs/)  ·  "
    )
    assert got["fetched"].endswith(" · cache miss")
    assert "stop unconfirmed" in got["stop"]
    assert " you: git status " in got["you"]
    # The legacy constant names other modules import still give the English.
    assert tool_card.EXPAND_HINT == "⟨expand⟩"
    assert tool_card.COLLAPSE_HINT == "⟨collapse⟩"
    assert tool_card.IMAGE_INTERRUPT_HINT == "⟨ctrl+c to interrupt⟩"
    assert tool_card.LIVE_HEADER_RUNNING == "⋯ running"
    assert tool_card.LIVE_HEADER_PENDING == "no output yet"


def test_singular_branches_are_english_singular() -> None:
    """ICU `one` branches: the counts the plural rewrite could get wrong."""
    image = ToolCard("g1", "generate_image", {"prompt": "a cat"})
    image.begin_running("generate_image", {"prompt": "a cat"}, None)
    image._expanded = True
    image.set_live_details({"stage": "running", "queue_position": 1})
    assert "\n  1 request ahead" in image._build_content(100).plain
    tail = ToolCard("t1", "bash", {"command": "seq"})
    tail._expanded = True
    tail.set_partial_detail("\n".join(str(n) for n in range(1, LIVE_MAX_LINES + 2)))
    assert "\n  … 1 earlier line\n" in tail._build_content(80).plain


@pytest.fixture
def fr_tool_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    # Copy, never symlink: `load_catalogue` refuses a symlinked path as escaping
    # the catalogue root, and English lookups would then paint raw key IDs.
    shutil.copytree(EN_ROOT / "en", tmp_path / "en")
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / f"{NS}.json").write_text(
        json.dumps({k: _fr(v) for k, v in BASE.items()}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


def test_every_surface_follows_a_live_language_change(fr_tool_card) -> None:
    # Under the fixture tree English must still be real English, not key IDs:
    # this is what a symlinked `en` silently broke.
    before = _surfaces()
    assert "⟨expand⟩ ✓" in before["expand"]
    assert before["live_pending"] == "⋯ running · 0s · no output yet"
    assert not any(NS in value for value in before.values()), before
    text.set_active_locale("fr")
    got = _surfaces()
    assert "⟨FR expand⟩ ✓" in got["expand"]
    assert "⟨FR collapse⟩ ✓" in got["collapse"]
    assert got["search_more"] == "… FR 5 more search lines"
    assert got["more_one"] == "… FR 1 more line"
    assert got["diff_more"] == f"… FR {121 - EXPAND_MAX_LINES} more diff lines ⟨FR expand⟩"
    assert got["live_pending"] == "⋯ FR running · 0s · FR no output yet"
    assert f"\n  … FR {500 - LIVE_MAX_LINES} earlier lines\n" in got["live_dropped"]
    assert "\n  … FR earlier output not shown\n" in got["live_elided"]
    # The chord is a placeholder: the words translate, `ctrl+c` never does.
    assert "⟨FR ctrl+c to interrupt⟩" in got["image_row"]
    # The image body's `⋯ running` state line is `imagegen.py`'s own word
    # (`STATE_RUNNING`), a separate slice; only this card's copy is pinned here.
    assert "\n  FR 3 requests ahead" in got["image_body"]
    assert got["fetched"].startswith(
        "FR Fetched: https://example.com/docs  FR (final: https://example.com/docs/)  ·  "
    )
    assert got["fetched"].endswith(" · FR cache miss")
    assert "FR stop unconf" in got["stop"]
    assert " FR you: git status" in got["you"]
    assert tool_card.EXPAND_HINT == "⟨FR expand⟩"  # legacy names resolve live too


@pytest.fixture
def shipped_fr(monkeypatch: pytest.MonkeyPatch):
    """The REAL shipped catalogue tree and shipped set — no synthetic locale.

    The synthetic fixture proves the wiring; this proves the audited fr
    translation reaches the painted card once fr is eligible through the
    generated ``shipped.json`` (a missing ledger row would un-ship fr and the
    card would silently fall back to English).
    """
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


def test_shipped_fr_catalogue_paints_the_tool_card(shipped_fr) -> None:
    assert "fr" in resolve.shipped_locales()
    text.set_active_locale("fr")
    got = _surfaces()
    assert "⟨déplier⟩" in got["expand"], got["expand"]
    # `diff` is not sanctioned English in fr (audit round 1, F1): the marker
    # must carry the translated noun.
    assert "comparaison" in got["diff_more"], got["diff_more"]
