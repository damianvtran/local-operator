"""Rules and invariants of ``local_operator.tools.result_normalize``.

The invariants matter more than any single rule: the normalized bytes are
persisted and re-sent from the prompt cache, so the transform must be
deterministic and idempotent, must keep JSON valid with URL fields byte-equal,
must never touch code, and ``off`` must be the identity.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from local_operator.tools.result_normalize import (
    RESULT_NORMALIZATION_MODES,
    normalize_markdown,
    normalize_tool_text,
)

PROSE = (
    "Mark Sutcliffe is the 59th mayor of Ottawa and was born July 14, 1968, according to "
    '[the CBC profile](https://www.cbc.ca/news/sutcliffe "CBC") published that week.'
)

PAGE = "\n".join(
    [
        "# Mark Sutcliffe",
        "",
        "![](https://px.example/1x1.gif)",
        "![Portrait of Mark Sutcliffe](https://img.example/p.jpg)",
        "* [Home](/)",
        "* [News](https://example.com/news)",
        "- [![](https://icons.example/fb.svg)](https://facebook.com/share)",
        "",
        "",
        "",
        PROSE + "[[24]](#cite_note-24)",
        'See [Mayor of Ottawa](/wiki/Mayor_of_Ottawa "Mayor of Ottawa") and [top](#top).',
        "Short [CBC](https://www.cbc.ca/x) line, [proto](//cdn.example/x), "
        "[js](javascript:void(0)), [mail](mailto:press@example.com).",
        "",
        "```python",
        "items = list[0]  # [a](/b)  ![x](y)",
        "```",
        "",
        "    indented [code](/kept)   verbatim",
        "",
        "Use `[a](/b)` inline   and  [[n]] text [7] stays.",
    ]
)


def test_images_become_alt_text_and_empty_alt_is_dropped() -> None:
    out = normalize_markdown(PAGE)
    assert "Portrait of Mark Sutcliffe" in out
    assert "img.example" not in out
    assert "px.example" not in out


def test_on_site_nav_bullets_are_dropped_and_absolute_bullets_keep_their_link() -> None:
    out = normalize_markdown(PAGE)
    assert "Home" not in out  # ``* [Home](/)``: on-site navigation chrome
    # An absolute one-link bullet is kept whole in prose-links (the default) and
    # reduced to its text in strip-links (R3 + round-1 addendum).
    assert "* [News](https://example.com/news)" in out
    assert "* News\n" in normalize_markdown(PAGE, keep_prose_links=False)
    assert "example.com/news" not in normalize_markdown(PAGE, keep_prose_links=False)
    assert "facebook.com" not in out  # an icon-only bullet has no text left


def test_cite_markers_dropped_but_bare_brackets_kept() -> None:
    out = normalize_markdown(PAGE)
    assert "cite_note" not in out
    assert "[[24]]" not in out
    assert "[[n]] text [7] stays." in out


def test_relative_fragment_protocol_relative_and_js_targets_become_anchor_text() -> None:
    out = normalize_markdown(PAGE)
    assert "See Mayor of Ottawa and top." in out
    assert "/wiki/" not in out
    assert "cdn.example" not in out
    assert "javascript:" not in out
    assert " proto," in out and " js," in out


def test_other_schemes_are_kept_because_the_target_may_be_the_datum() -> None:
    assert "[mail](mailto:press@example.com)" in normalize_markdown(PAGE)


def test_prose_links_mode_keeps_absolute_links_only_on_prose_lines() -> None:
    out = normalize_markdown(PAGE, keep_prose_links=True)
    assert "[the CBC profile](https://www.cbc.ca/news/sutcliffe)" in out  # hover title dropped
    assert '"CBC"' not in out
    assert "Short CBC line" in out  # short line: target dropped
    assert "cbc.ca/x" not in out


def test_strip_links_mode_drops_every_http_target() -> None:
    out = normalize_markdown(PAGE, keep_prose_links=False)
    assert "the CBC profile published" in out
    assert "cbc.ca" not in out


def test_empty_anchor_link_is_never_kept_even_in_prose() -> None:
    line = "x" * 90 + " [](https://example.com/empty) tail"
    assert "example.com" not in normalize_markdown(line)


def test_code_is_verbatim() -> None:
    out = normalize_markdown(PAGE)
    assert "items = list[0]  # [a](/b)  ![x](y)" in out
    assert "    indented [code](/kept)   verbatim" in out
    assert "`[a](/b)`" in out


def test_unclosed_fence_protects_the_rest() -> None:
    text = "intro ![](https://px/a.gif)\n```\n[a](/b)   x"
    assert normalize_markdown(text) == "intro\n```\n[a](/b)   x"


def test_list_continuation_is_not_mistaken_for_code() -> None:
    text = "* item ![](https://px/a.gif)\n\n    continued [rel](/x)"
    assert normalize_markdown(text) == "* item\n\n    continued rel"


def test_blank_runs_shrink_to_two_and_intra_line_spacing_is_kept() -> None:
    out = normalize_markdown(PAGE)
    assert "\n\n\n\n" not in out  # the 3-blank run became 2
    assert "inline   and  [[n]]" in out  # an unchanged line keeps its spacing


def test_plain_text_without_web_markup_is_returned_unchanged() -> None:
    text = "Line one.\nLine two has  two spaces.   \n\n\n\nEnd [a](/b).\n"
    assert normalize_markdown(text) is text
    assert normalize_tool_text(text) is text


def test_many_unmatched_backticks_and_brackets_stay_linear() -> None:
    # A quadratic scan would take minutes here; the bounded/paired scans do not.
    hostile = ("`a [b " * 40_000) + "\n" + ("[" * 100_000)
    assert normalize_markdown(hostile)


# ---- normalize_tool_text: JSON routing ---------------------------------------------


def _fetch_payload() -> dict[str, Any]:
    return {
        "url": "https://en.wikipedia.org/wiki/Mark_Sutcliffe_(politician)",
        "title": "Mark Sutcliffe - Wikipedia",
        "offset": 0,
        "total_chars": 51234,
        "results": [
            {"url": "https://example.com/a?x=1&y=2", "snippet": "short [x](/y)", "score": 0.5},
            {"url": "https://example.com/b", "content": PAGE},
        ],
        "content": PAGE,
    }


def test_json_long_leaves_normalized_short_values_and_urls_untouched() -> None:
    raw = json.dumps(_fetch_payload(), indent=2)
    out = normalize_tool_text(raw)
    parsed = json.loads(out)
    original = _fetch_payload()
    assert list(parsed) == list(original)  # key order preserved
    for key in ("url", "title", "offset", "total_chars"):
        assert parsed[key] == original[key]
    assert parsed["results"][0] == original["results"][0]  # short leaves byte-equal
    assert parsed["results"][1]["url"] == "https://example.com/b"
    assert parsed["content"] == normalize_markdown(PAGE)
    assert out == json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def test_json_url_keyed_leaves_are_never_normalized_even_when_long() -> None:
    long_url_text = "see [x](https://e.example/y) " * 30
    raw = json.dumps({"source_url": long_url_text, "body": long_url_text})
    parsed = json.loads(normalize_tool_text(raw, "strip-links"))
    assert parsed["source_url"] == long_url_text
    assert parsed["body"] != long_url_text


def test_json_non_ascii_is_emitted_raw() -> None:
    leaf = "Montréal " * 30 + "![](https://a.example/b.png)"
    out = normalize_tool_text(json.dumps({"name": "Montréal", "t": leaf}))
    assert out == '{"name":"Montréal","t":"' + "Montréal " * 29 + 'Montréal"}'


def test_json_with_duplicate_keys_is_returned_unchanged() -> None:
    raw = '{"a": 1, "a": 2}'
    assert normalize_tool_text(raw) == raw


def test_invalid_json_takes_the_markdown_path() -> None:
    assert normalize_tool_text("{not json  [a](/b) ![](https://p/x.gif)") == "{not json  a"


def test_scalar_json_takes_the_markdown_path() -> None:
    assert normalize_tool_text("42") == "42"


def test_off_is_identity_byte_for_byte() -> None:
    for text in (PAGE, json.dumps(_fetch_payload(), indent=2), "  x  \n\n\n", ""):
        assert normalize_tool_text(text, "off") is text


def test_unknown_mode_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        normalize_tool_text("x", "everything")


# ---- review/QA round 1 regressions (PR #4514: R2-R6, Q1-Q3) ------------------------

# Q1/R2: exact reproductions. None carries web markup, so every one must come
# back byte-identical -- code, regexes and column layouts are evidence too.
CODE_AND_PLAIN = [
    "def dispatch(handlers, name, arg):\n    return handlers[name](arg)\n",
    "const out = callbacks[i](event);\nconst y = arr[0](x);",
    "const fn = handlers[type](event);   // dispatch table\nreturn callbacks[0](fn);\n",
    'ID = re.compile(r"[A-Z](\\d+)")\n\n\n\ndef f(self, name, *args):\n'
    "    return self.callbacks[name](*args)\n",
    "match [a-z](foo|bar) here",
    'import os\n\n\nclass A:\n    pass\n\n\n\ntable  =  {"k":  1}\n',
    "NAME      PID    CPU\nlop       123    0.5\n",
]


@pytest.mark.parametrize("mode", ("prose-links", "strip-links"))
@pytest.mark.parametrize("text", CODE_AND_PLAIN)
def test_code_and_plain_text_are_byte_identical(text: str, mode: str) -> None:
    assert normalize_tool_text(text, mode) == text
    # ... and inside a long JSON leaf (a GitHub-style ``{"path", "content"}``).
    raw = json.dumps({"path": "src/x.py", "content": text * 8})
    assert normalize_tool_text(raw, mode) == raw


def test_index_or_call_syntax_is_never_a_link_even_in_web_content() -> None:
    # Web content passes the gate, yet ``name[x](y)`` and non-URL targets stay.
    text = (
        "![](https://px.example/a.gif)\n"
        "Call handlers[name](arg) or callbacks[0](fn) or self.cb[k](*args); "
        "regex [a-z](foo|bar) and [A-Z](\\d+) stay; [Docs](/docs/a) goes."
    )
    out = normalize_markdown(text)
    assert out == (
        "Call handlers[name](arg) or callbacks[0](fn) or self.cb[k](*args); "
        "regex [a-z](foo|bar) and [A-Z](\\d+) stay; Docs goes."
    )


def test_trailing_newline_and_line_count_are_preserved() -> None:
    text = "# T ![](https://px/a.gif)\n\nbody [a](/b)\n\n\n\nend\n"
    out = normalize_markdown(text)
    assert out == "# T\n\nbody a\n\n\nend\n"


# Q2/R3: exact reproductions. A result that IS a list of links keeps every title.
SEARCH_RESULTS = (
    "Search results:\n\n"
    "1. [Mark Sutcliffe - Wikipedia](https://en.wikipedia.org/wiki/Mark_Sutcliffe)\n"
    "2. [Mayor of Ottawa | City of Ottawa](https://ottawa.ca/en/mayor)\n"
    "3. [Mark Sutcliffe (@marksutcliffe) / X](https://x.com/marksutcliffe)\n"
)
BOARD_DECKS = (
    "- [Q3 board deck](https://drive.example.com/q3)\n"
    "- [Q4 board deck](https://drive.example.com/q4)"
)
README = (
    "# Project\n\n"
    "Read the guide below. ![build](https://ci.example/badge.svg)\n\n"
    "## Docs\n\n"
    "- [Install](docs/INSTALL.md)\n"
    "- [Usage](docs/USAGE.md)\n"
    "- [Configuration](docs/CONFIG.md)\n"
    "- [Changelog](https://github.com/acme/project/blob/main/CHANGELOG.md)\n"
)


def test_link_lists_keep_their_urls_in_prose_links_mode() -> None:
    # A search-result list is often the only place the model learns which URL
    # to fetch or cite: the default keeps every ``[title](url)`` verbatim.
    assert normalize_tool_text(SEARCH_RESULTS, "prose-links") == SEARCH_RESULTS
    assert normalize_tool_text(BOARD_DECKS, "prose-links") == BOARD_DECKS


def test_link_lists_keep_their_titles_in_strip_links_mode() -> None:
    out = normalize_tool_text(SEARCH_RESULTS, "strip-links")
    assert out == (
        "Search results:\n\n"
        "1. Mark Sutcliffe - Wikipedia\n"
        "2. Mayor of Ottawa | City of Ottawa\n"
        "3. Mark Sutcliffe (@marksutcliffe) / X\n"
    )
    assert "https://" not in out
    assert normalize_tool_text(BOARD_DECKS, "strip-links") == "- Q3 board deck\n- Q4 board deck"


def test_single_absolute_link_on_a_short_line_is_kept_but_nav_bars_are_not() -> None:
    text = (
        "![](https://px.example/a.gif)\n"
        "Source: [CBC News](https://www.cbc.ca/news/x)\n"
        "[Home](https://e.example/) | [About](https://e.example/about)\n"
        "Icon [](https://e.example/icon)\n"
    )
    out = normalize_tool_text(text, "prose-links")
    assert out == "Source: [CBC News](https://www.cbc.ca/news/x)\nHome | About\nIcon\n"


def test_nav_bullets_that_are_most_of_the_result_are_kept_as_text() -> None:
    # 3 of the 8 non-blank lines are on-site one-link bullets: a minority, so
    # they go. Add more and they become the content, so they stay as text.
    site_nav = README.replace("](docs/", "](/docs/")
    out = normalize_tool_text(site_nav)
    assert "Install" not in out and "Changelog" in out
    docs_index = "".join(f"- [Page {i}](/docs/{i})\n" for i in range(6)) + "![](https://p/x.gif)\n"
    assert normalize_tool_text(docs_index) == "".join(f"- Page {i}\n" for i in range(6))


# Q4 (QA round 2): file-relative targets name the file that holds the content.
# QA's fixture, and the ``## Docs`` list of the real openai/codex README.
QA_README_DOCS = (
    "# widget\n\n"
    "![build](https://img.shields.io/badge/build-passing-green.svg)\n\n"
    "A tiny widget library for parsing config files.\n\n"
    "## Docs\n\n"
    "- [Install guide](docs/install.md)\n"
    "- [Configuration](docs/config.md)\n"
    "- [API reference](docs/api.md)\n"
    "- [FAQ](docs/faq.md)\n"
    "- [Changelog](https://github.com/acme/widget/blob/main/CHANGELOG.md)\n\n"
    "## License\n\nMIT\n"
)
CODEX_README_DOCS = (
    "## Docs\n\n"
    "- [**Codex Documentation**](https://developers.openai.com/codex)\n"
    "- [**Contributing**](./docs/contributing.md)\n"
    "- [**Installing & building**](./docs/install.md)\n"
    "- [**Open source fund**](./docs/open-source-fund.md)\n\n"
    "This repository is licensed under the [Apache-2.0 License](LICENSE).\n"
)


def test_file_relative_link_items_keep_title_and_path_in_prose_links() -> None:
    out = normalize_tool_text(QA_README_DOCS, "prose-links")
    for title, path in (
        ("Install guide", "docs/install.md"),
        ("Configuration", "docs/config.md"),
        ("API reference", "docs/api.md"),
        ("FAQ", "docs/faq.md"),
    ):
        assert f"- [{title}]({path})\n" in out
    assert out == QA_README_DOCS.replace(
        "![build](https://img.shields.io/badge/build-passing-green.svg)", "build"
    )
    # No image or cite marker: the codex snippet passes through byte for byte.
    assert normalize_tool_text(CODEX_README_DOCS, "prose-links") == CODEX_README_DOCS


def test_file_relative_link_items_keep_their_path_as_text_in_strip_links() -> None:
    out = normalize_tool_text(QA_README_DOCS, "strip-links")
    assert "- Install guide (docs/install.md)\n- Configuration (docs/config.md)\n" in out
    assert "- FAQ (docs/faq.md)\n- Changelog\n" in out
    codex = normalize_tool_text(CODEX_README_DOCS, "strip-links")
    assert codex == (
        "## Docs\n\n"
        "- **Codex Documentation**\n"
        "- **Contributing** (./docs/contributing.md)\n"
        "- **Installing & building** (./docs/install.md)\n"
        "- **Open source fund** (./docs/open-source-fund.md)\n\n"
        "This repository is licensed under the [Apache-2.0 License](LICENSE).\n"
    )


def test_only_root_fragment_and_js_one_link_items_are_nav() -> None:
    signal = "![](https://p/x.gif)\n" + "Body paragraph.\n" * 12
    items = (
        "- [Home](/)\n- [Top](#top)\n- [Menu](javascript:void(0))\n"
        "- [Up](../CONTRIBUTING.md)\n- [Guide](guide/intro.md)\n"
        "- [Upload](//commons.example.org/upload)\n- [Next](?page=2)\n"
    )
    out = normalize_tool_text(signal + items, "prose-links")
    assert "Home" not in out and "Top" not in out and "Menu" not in out
    assert "- [Up](../CONTRIBUTING.md)\n- [Guide](guide/intro.md)\n" in out
    assert "- [Upload](//commons.example.org/upload)\n" in out  # another host: absolute
    assert "- Next\n" in out  # on-site query: never dropped, but its target goes


def test_adjacent_links_both_convert_but_call_syntax_stays() -> None:
    text = "![](https://p/x.gif)\nSee [a](/x)[b](/y) and f(x)[i](y/z) here."
    assert normalize_tool_text(text) == "See ab and f(x)[i](y/z) here."


def test_dropped_leading_chrome_leaves_no_leading_blank_line() -> None:
    text = "* [Home](/)\n\nText and [x](https://e.com/) here"
    assert normalize_markdown(text) == "Text and [x](https://e.com/) here"
    # A blank line the server itself sent first is content and stays.
    assert normalize_markdown("\n" + text) == "\nText and [x](https://e.com/) here"


def test_long_on_site_anchor_is_content_not_nav() -> None:
    title = "A long article title that is clearly content, not chrome"
    text = f"intro ![](https://p/x.gif)\n\n* [{title}](/articles/1)\n\n" + "para\n" * 5
    assert f"* {title}" in normalize_tool_text(text)


# R4: a lone surrogate escape must not reach the provider as a raw surrogate.
def test_json_lone_surrogate_stays_escaped_and_encodable() -> None:
    raw = json.dumps({"t": "x " * 120 + "![](https://a.example/b.png) end"})[:-2] + '\\ud83d"}'
    out = normalize_tool_text(raw)
    out.encode("utf-8")  # raises UnicodeEncodeError if a raw surrogate leaked
    assert "\\ud83d" in out and "a.example" not in out
    assert json.loads(out)["t"].endswith("end\ud83d")


def test_unencodable_markdown_output_falls_back_to_the_input() -> None:
    raw = "x ![](https://a.example/b.png) \ud83d"  # already unencodable from the server
    assert normalize_tool_text(raw) is raw


def test_json_with_nothing_to_normalize_keeps_the_server_bytes() -> None:
    raw = json.dumps({"t": "plain words " * 40, "n": 1.50, "u": "https://e/x"}, indent=2)
    assert normalize_tool_text(raw) is raw


# R5: numbers that would change value on re-emit leave the payload untouched.
@pytest.mark.parametrize("number", ("1e400", "0.1234567890123456789", "NaN", "-Infinity"))
def test_json_numbers_that_would_change_keep_the_original(number: str) -> None:
    leaf = json.dumps("x " * 120 + "![](https://a.example/b.png)")
    raw = '{"a": ' + number + ', "t": ' + leaf + "}"
    assert normalize_tool_text(raw) is raw


def test_json_number_spelling_may_change_but_not_value() -> None:
    leaf = json.dumps("x " * 120 + "![](https://a.example/b.png)")
    out = normalize_tool_text('{"a": 1.50, "b": 12345678901234567890, "t": ' + leaf + "}")
    assert out.startswith('{"a":1.5,"b":12345678901234567890,')


# R6/Q3: inputs that needed a second pass before; f(f(x)) == f(x) must hold.
IDEMPOTENCE_EDGE_CASES = [
    "[[a](/b)](/c) ![](https://x.example/y.png)",
    "[[a](https://x.example/b)](https://x.example/c) " + "w " * 50,
    "* [N](/w)\n[ ]",
    " ![](y)```",
    " ![](https://y.example/z.png)```\n[a](/b)",
    "![](https://p/x.gif)\n[1, 2]",
    "- [![](https://i/x.svg)](/home)\n- ![](https://i/y.svg)\n\n\n\ntext",
]


_FIXTURES = [
    PAGE,
    PROSE,
    json.dumps(_fetch_payload(), indent=2),
    json.dumps([_fetch_payload(), {"k": PAGE}]),
    "* item\n\n    continued [rel](/x)\n\n    more",
    "para\n\n* [nav](/x)\n\n    after nav [r](/y)",
    "![](a) * [b](/c)\n![alt](d)",
    SEARCH_RESULTS,
    BOARD_DECKS,
    README,
    QA_README_DOCS,
    CODEX_README_DOCS,
    "![](https://p/x.gif)\n[a](/x)[b](/y)[c](docs/c.md)",
    *CODE_AND_PLAIN,
    *IDEMPOTENCE_EDGE_CASES,
]


@pytest.mark.parametrize("mode", RESULT_NORMALIZATION_MODES)
@pytest.mark.parametrize("text", _FIXTURES)
def test_idempotent(text: str, mode: str) -> None:
    once = normalize_tool_text(text, mode)
    assert normalize_tool_text(once, mode) == once


def test_deterministic_across_processes() -> None:
    # A fresh interpreter has a different hash seed; nothing may depend on it.
    code = (
        "import sys;"
        "from local_operator.tools.result_normalize import normalize_tool_text as n;"
        "sys.stdout.write(n(sys.stdin.read()))"
    )
    payload = json.dumps(_fetch_payload(), indent=2)
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": ":".join(sys.path)},
        ).stdout
        for seed in ("1", "2")
    }
    assert runs == {normalize_tool_text(payload)}


# --- search-hit snippet clipping (``mcp.search_snippet_chars``) -------------

#: A provider snippet as long as the replayed run's average (1,098 chars), with
#: the line breaks a page extract carries.
_LONG_SNIPPET = ("Mark Sutcliffe was elected the 59th mayor of Ottawa in 2022.\n" * 18).strip()


def _hit(n: int, **extra: Any) -> dict[str, Any]:
    return {
        "title": f"Mark Sutcliffe - result {n}",
        "url": f"https://example.com/sutcliffe/{n}?utm_source=x&id={n}",
        "snippet": _LONG_SNIPPET,
        "published_date": None,
        **extra,
    }


#: The Pergamon Forge ``search`` shape (Tavily basic):
#: ``{query, results: [{title, url, snippet, published_date}]}``.
_TAVILY_SHAPE = json.dumps({"query": "Mark Sutcliffe Ottawa", "results": [_hit(1), _hit(2)]})


def _clip(text: str, tool_name: str = "search", chars: int = 400, mode: str = "prose-links") -> str:
    return normalize_tool_text(text, mode, tool_name=tool_name, search_snippet_chars=chars)


@pytest.mark.parametrize("mode", ["prose-links", "strip-links"])
def test_search_hit_snippets_are_clipped_and_null_fields_dropped(mode: str) -> None:
    out = json.loads(_clip(_TAVILY_SHAPE, mode=mode))
    raw = json.loads(_TAVILY_SHAPE)
    assert out["query"] == raw["query"]
    for got, sent in zip(out["results"], raw["results"], strict=True):
        assert got["url"] == sent["url"]  # the citation key, byte-equal
        assert got["title"] == sent["title"]
        assert "published_date" not in got  # null field dropped
        assert len(got["snippet"]) <= 400 and got["snippet"].endswith("…")
        assert got["snippet"].startswith("Mark Sutcliffe was elected the 59th mayor")
        assert "\n" not in got["snippet"]


def test_top_level_hit_list_is_clipped() -> None:
    out = json.loads(_clip(json.dumps([_hit(1), _hit(2, content=_LONG_SNIPPET)])))
    assert all(len(h["snippet"]) <= 400 for h in out)
    assert len(out[1]["content"]) <= 400


def test_snippet_clip_is_the_web_search_clipper() -> None:
    """One clipper: built-in ``web_search`` (320) and MCP search (400) agree."""
    from local_operator.tools.result_normalize import clip_snippet
    from local_operator.web_search import tool as web_search_tool

    assert web_search_tool._clip is clip_snippet
    clipped = json.loads(_clip(_TAVILY_SHAPE, chars=320))["results"][0]["snippet"]
    assert clipped == clip_snippet(_LONG_SNIPPET, 320)


def test_short_snippet_keeps_its_exact_bytes() -> None:
    short = "Ottawa mayor.\n  Elected 2022."
    out = json.loads(_clip(json.dumps([_hit(1, snippet=short), _hit(2, snippet=short)])))
    assert [h["snippet"] for h in out] == [short, short]


def test_non_search_tool_is_untouched() -> None:
    """A GitHub-issues-shaped list from a non-search tool keeps every byte."""
    body = "Steps to reproduce:\n" + "the widget fails when resized. " * 40
    issues = json.dumps(
        [
            {"title": "Crash", "html_url": "https://github.com/o/r/issues/1", "body": body},
            {"title": "Hang", "html_url": "https://github.com/o/r/issues/2", "body": None},
        ]
    )
    assert _clip(issues, tool_name="list_issues") == issues
    # The Tavily shape itself is only clipped when a search returned it.
    assert _clip(_TAVILY_SHAPE, tool_name="fetch") == _TAVILY_SHAPE


def test_fetch_payload_is_untouched_even_from_a_search_tool() -> None:
    """A page (one object, not a hit list) is never clipped."""
    page = json.dumps({"url": "https://example.com/p", "content": _LONG_SNIPPET * 5})
    assert _clip(page, tool_name="search") == normalize_tool_text(page, "prose-links")
    assert len(json.loads(_clip(page))["content"]) > 4_000


@pytest.mark.parametrize(
    "payload",
    [
        # no ``title``: links, not ranked hits
        json.dumps([{"url": "https://a.example", "snippet": _LONG_SNIPPET}] * 2),
        # mixed list: one item is not a hit
        json.dumps([_hit(1), {"note": _LONG_SNIPPET}]),
        # hits with no discovery text at all
        json.dumps([{"title": "A", "url": "https://a.example", "score": None}]),
        # plain-text search output
        "1. Mark Sutcliffe - https://example.com/1\n" + _LONG_SNIPPET,
    ],
    ids=["no-title", "mixed", "no-snippet", "plain-text"],
)
def test_non_hit_shapes_are_untouched(payload: str) -> None:
    assert _clip(payload) == normalize_tool_text(payload, "prose-links")


def test_search_clip_is_idempotent() -> None:
    once = _clip(_TAVILY_SHAPE)
    assert once != _TAVILY_SHAPE
    assert _clip(once) == once


def test_zero_chars_is_byte_identical_to_plain_normalization() -> None:
    for mode in ("prose-links", "strip-links"):
        assert _clip(_TAVILY_SHAPE, chars=0, mode=mode) == normalize_tool_text(_TAVILY_SHAPE, mode)
    assert _clip(_TAVILY_SHAPE, chars=0) == _TAVILY_SHAPE  # nothing else to normalize here


def test_off_mode_is_byte_identical_even_for_a_search() -> None:
    assert _clip(_TAVILY_SHAPE, mode="off") == _TAVILY_SHAPE


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("search", True),
        ("web_search_exa", True),
        ("SearchDocs", True),
        ("web_search", True),
        ("tavily-search", True),
        ("searchWeb", True),
        ("brave_web_search", True),
        ("docs.search", True),
        ("search2", True),
        ("HTTPSearch", True),
        ("fetch", False),
        # Substring-only matches are NOT searches (review R1 / QA Q1): their
        # text fields are the answer, not discovery snippets.
        ("research_fetch", False),
        ("deep_researcher_check", False),
        ("perplexity_research", False),
        ("company_research_exa", False),
        ("elasticsearch_get_document", False),
        ("researchSearchless", False),
    ],
)
def test_is_search_tool(name: str, expected: bool) -> None:
    from local_operator.tools.result_normalize import is_search_tool

    assert is_search_tool(name) is expected
