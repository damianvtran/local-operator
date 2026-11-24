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


# --- result shaping v2: glued/truncated link leak (structural-economy §6) ----


@pytest.mark.parametrize("mode", ["prose-links", "strip-links"])
def test_glued_absolute_links_become_their_anchor(mode: str) -> None:
    """Infobox cells glue a label to its link (``Preceded by[Carolyn
    Parrish](…)``): the target is absolute, so it is a link, not an index."""
    line = (
        "Preceded by[Carolyn Parrish](https://en.wikipedia.org/wiki/Carolyn_Parrish) "
        "p[Canada](https://en.wikipedia.org/wiki/Canada)"
    )
    out = normalize_markdown(line, keep_prose_links=mode == "prose-links")
    assert out == "Preceded byCarolyn Parrish pCanada"


def test_glued_absolute_link_after_a_close_paren_converts() -> None:
    out = normalize_markdown("Born (1957)[Hong Kong](https://en.wikipedia.org/wiki/HK) x [y](/z)")
    assert out == "Born (1957)Hong Kong x y"


def test_glued_code_stays_byte_identical_beside_glued_links() -> None:
    """Regression: the glued-link allowance is for ABSOLUTE targets only.
    ``handlers[name](arg)`` and friends are code and keep every byte, even
    on a line that also carries a glued absolute link."""
    code = (
        "Call handlers[name](arg) or callbacks[0](fn) or f(x)[i](y) or "
        "self.cb[k](*args) or docs[i](./a.md) or m[k](/root)"
    )
    page = (
        "![](https://px.example/a.gif)\n" + code + "\nTitle[Canada](https://x.example/Canada) end"
    )
    out = normalize_markdown(page, keep_prose_links=False)
    assert out == code + "\nTitleCanada end"


@pytest.mark.parametrize("mode", ["prose-links", "strip-links"])
@pytest.mark.parametrize("terminator", ["", "\n"])
def test_trailing_unterminated_link_is_cut_to_its_anchor(mode: str, terminator: str) -> None:
    """A server chunk boundary that lands inside a link target leaves
    ``[anchor](http…`` with no ``)``; the anchor is the content."""
    text = (
        "Olivia Chow is the 66th mayor of Toronto and was preceded in office by "
        "[John Tory](https://en.wikipedia.org/wiki/John_Tory) and earlier by "
        "[Rob Ford](https://en.wikipedia.org/wiki/Rob_(polit" + terminator
    )
    out = normalize_markdown(text, keep_prose_links=mode == "prose-links")
    assert out.endswith("earlier by Rob Ford" + terminator)
    assert "](https://en.wikipedia.org/wiki/Rob_(polit" not in out


@pytest.mark.parametrize("cut", ["h", "http", "https:", "https:/", "HTTP:/"])
def test_link_cut_inside_its_scheme_is_cut_to_its_anchor(cut: str) -> None:
    """Run 2, rec-006: ``... trailed [Pierre Poilievre](http`` ended a chunk."""
    text = (
        "[a](https://x.example/a) and [b](https://x.example/b)\ntrailed [Pierre Poilievre](" + cut
    )
    assert normalize_markdown(text, keep_prose_links=False) == "a and b\ntrailed Pierre Poilievre"


def test_scheme_cut_tail_is_cut_even_with_no_other_link() -> None:
    """Regression (replay): the text had no ``//`` link left to pass the gate."""
    text = "By late 2024 the government trailed [Pierre Poilievre](http"
    assert normalize_markdown(text) == "By late 2024 the government trailed Pierre Poilievre"
    assert normalize_markdown("plain text, no links\n") == "plain text, no links\n"


def test_trailing_short_non_url_target_keeps_its_bytes() -> None:
    for tail in ("handlers[name](help", "[x](ftp", "[x](/", "[x]("):
        text = "[a](https://x.example/a) and [b](https://x.example/b)\n" + tail
        assert normalize_markdown(text, keep_prose_links=False).endswith(tail)


def test_trailing_unterminated_image_is_cut_to_its_alt() -> None:
    text = (
        "[a](https://x.example/a) and [b](https://x.example/b)\n![Portrait of Chow](https://img.ex"
    )
    assert normalize_markdown(text, keep_prose_links=False) == "a and b\nPortrait of Chow"


@pytest.mark.parametrize(
    "text",
    [
        # unterminated but NOT at the end of the text block
        "see [John Tory](https://en.wikipedia.org/wiki/John_T and more text "
        "[a](https://x.example/a)",
        # code: unterminated call with a non-URL argument at the end
        "[a](https://x.example/a)\nresult = handlers[name](arg",
        # inside a fenced code block at the end
        "[a](https://x.example/a)\n```\nx = links[0](https://example.com/cut",
        # inside an inline code span at the end
        "[a](https://x.example/a)\nuse `[x](https://example.com/cut`",
    ],
    ids=["mid-text", "code-call", "fenced", "inline-code"],
)
def test_unterminated_link_elsewhere_keeps_its_bytes(text: str) -> None:
    out = normalize_markdown(text, keep_prose_links=False)
    # The unterminated ``[..](..`` keeps its bytes wherever it is not a cut tail.
    tail = text.split("\n")[-1]
    kept = tail[: tail.index("https://x.example/a")] if text.startswith("see") else tail
    assert kept.split(" and more")[0] in out


def test_trailing_unterminated_link_inside_a_json_page_leaf() -> None:
    content = (
        "Olivia Chow is the mayor of Toronto. " * 8
    ) + "Preceded by [John Tory](https://en.wik"
    raw = json.dumps({"url": "https://www.toronto.ca/mayor", "content": content})
    out = json.loads(normalize_tool_text(raw, "prose-links"))
    assert out["content"].endswith("Preceded by John Tory")
    assert out["url"] == "https://www.toronto.ca/mayor"


# --- result shaping v2: seen-URL snippet elision (structural-economy §2) -----


def test_seen_url_key_strips_only_fragment_and_trailing_slash() -> None:
    from local_operator.tools.result_normalize import seen_url_key

    assert seen_url_key("https://Example.com/a/#top") == "https://Example.com/a"
    assert seen_url_key("https://example.com/a?x=1") == "https://example.com/a?x=1"
    assert seen_url_key("http://www.example.com/a") == "http://www.example.com/a"


def _elide(text: str, seen: frozenset[str], tool_name: str = "search", **kw: Any) -> str:
    return normalize_tool_text(
        text, "prose-links", tool_name=tool_name, search_snippet_chars=400, seen_urls=seen, **kw
    )


def test_seen_hit_keeps_title_and_url_but_drops_its_snippet() -> None:
    from local_operator.tools.result_normalize import seen_url_key

    raw = json.loads(_TAVILY_SHAPE)
    # The ledger holds keys; a fragment/trailing-slash variant still matches.
    seen = frozenset({seen_url_key(raw["results"][0]["url"] + "#section")})
    out = json.loads(_elide(_TAVILY_SHAPE, seen))
    first, second = out["results"]
    assert first == {"title": raw["results"][0]["title"], "url": raw["results"][0]["url"]}
    assert second["url"] == raw["results"][1]["url"]
    assert second["snippet"].startswith("Mark Sutcliffe was elected")


def test_seen_elision_drops_every_snippet_like_field() -> None:
    hit = {
        "title": "T",
        "url": "https://a.example/x",
        "content": "c " * 50,
        "summary": "s",
        "score": 0.5,
    }
    out = json.loads(
        _elide(json.dumps({"query": "q", "results": [hit]}), frozenset({"https://a.example/x"}))
    )
    assert out["results"] == [{"title": "T", "url": "https://a.example/x", "score": 0.5}]


def test_seen_elision_never_touches_non_search_tools_or_off_mode() -> None:
    seen = frozenset({json.loads(_TAVILY_SHAPE)["results"][0]["url"]})
    assert _elide(_TAVILY_SHAPE, seen, tool_name="fetch") == _TAVILY_SHAPE
    off = normalize_tool_text(_TAVILY_SHAPE, "off", tool_name="search", seen_urls=seen)
    assert off == _TAVILY_SHAPE


def test_empty_seen_set_is_byte_identical_to_no_elision() -> None:
    assert _elide(_TAVILY_SHAPE, frozenset()) == _clip(_TAVILY_SHAPE)


def test_seen_elision_is_idempotent_for_a_fixed_ledger() -> None:
    seen = frozenset({json.loads(_TAVILY_SHAPE)["results"][1]["url"]})
    once = _elide(_TAVILY_SHAPE, seen)
    assert _elide(once, seen) == once


def test_shown_urls_names_hit_and_page_urls_in_order() -> None:
    from local_operator.tools.result_normalize import shown_urls

    hits = json.loads(_TAVILY_SHAPE)["results"]
    assert shown_urls(_TAVILY_SHAPE, tool_name="search") == [h["url"] for h in hits]
    # A non-search tool's hit-shaped list is not a search; nothing is recorded.
    assert shown_urls(_TAVILY_SHAPE, tool_name="list_issues") == []
    page = json.dumps({"url": "https://example.com/p", "content": "body", "offset": 0})
    assert shown_urls(page, tool_name="fetch") == ["https://example.com/p"]
    assert shown_urls("not json", tool_name="search") == []


# --- result shaping v2: JSON -> text rendering (structural-economy §2) -------


def _decode_scalar(value: str) -> Any:
    """Inverse of ``result_normalize._scalar``: JSON when it parses, else the raw string."""
    try:
        return json.loads(value)
    except ValueError:
        return value


def _decode_hits(lines: list[str]) -> list[dict[str, Any]]:
    """Reference decoder for the rendered hit list (see ``render_result_text``)."""
    import re

    field = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*): (.*)\Z")
    hits: list[dict[str, Any]] = []
    slot = 0
    for line in lines:
        head = re.match(r"(\d+)\. (.*)\Z", line)
        if head:
            assert int(head.group(1)) == len(hits) + 1
            hits.append({"title": _decode_scalar(head.group(2))})
            slot = 0
            continue
        assert line.startswith("   "), line
        body = line[3:]
        if slot == 0:
            hits[-1]["url"] = _decode_scalar(body)
            slot = 1
            continue
        match = field.match(body)
        if slot == 1 and not match:
            hits[-1]["snippet"] = _decode_scalar(body)
            slot = 2
            continue
        assert match, body
        hits[-1][match.group(1)] = _decode_scalar(match.group(2))
        slot = 2
    return hits


def _decode_search(text: str) -> Any:
    lines = text.split("\n")
    if lines[0][:1].isdigit():
        return _decode_hits(lines)
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        key, sep, value = lines[i].partition(":")
        assert sep, lines[i]
        if value == "":
            j = i + 1
            while j < len(lines) and (lines[j][:1].isdigit() or lines[j].startswith("   ")):
                j += 1
            out[key] = _decode_hits(lines[i + 1 : j])
            i = j
            continue
        assert value.startswith(" ")
        out[key] = _decode_scalar(value[1:])
        i += 1
    return out


def _decode_page(text: str) -> dict[str, Any]:
    header, sep, content = text.partition("\n\n")
    assert sep
    out: dict[str, Any] = {}
    for line in header.split("\n"):
        key, _, value = line.partition(": ")
        out[key] = _decode_scalar(value)
    out["content"] = content
    return out


def _render(text: str, tool_name: str = "search", mode: str = "prose-links", **kw: Any) -> str:
    return normalize_tool_text(
        text, mode, tool_name=tool_name, search_snippet_chars=400, text_rendering=True, **kw
    )


def test_search_hit_list_renders_as_numbered_text_and_decodes_losslessly() -> None:
    out = _render(_TAVILY_SHAPE)
    lines = out.split("\n")
    assert lines[0] == "query: Mark Sutcliffe Ottawa"
    assert lines[1] == "results:"
    assert lines[2] == "1. Mark Sutcliffe - result 1"
    assert lines[3] == "   https://example.com/sutcliffe/1?utm_source=x&id=1"  # url byte-equal
    assert lines[4].startswith("   Mark Sutcliffe was elected")
    # Lossless against the JSON this would otherwise have sent.
    assert _decode_search(out) == json.loads(_clip(_TAVILY_SHAPE))


def test_fetch_payload_renders_as_header_plus_raw_content() -> None:
    content = (
        '# Olivia Chow\n\nOlivia Chow (born March 24, 1957) is the mayor.\n\n"quoted" \\ tab\t.'
    )
    raw = json.dumps(
        {
            "url": "https://x.example/chow",
            "content": content,
            "offset": 0,
            "next_offset": None,
            "total_chars": 81,
        },
        indent=2,
    )
    out = _render(raw, tool_name="fetch")
    assert out.startswith(
        "url: https://x.example/chow\noffset: 0\nnext_offset: null\ntotal_chars: 81\n\n"
    )
    assert out.endswith(content)  # raw markdown, no JSON escaping
    assert _decode_page(out) == json.loads(raw)


@pytest.mark.parametrize(
    "hit",
    [
        {"title": "Line\nbreak", "url": "https://a.example/1", "snippet": "Note: a colon prefix"},
        {"title": "", "url": "https://a.example/2", "snippet": "0"},
        {"title": "  padded ", "url": "https://a.example/3", "snippet": "true", "score": 0.91},
        {"title": "T", "url": "https://a.example/4", "score": 1, "rank": "2"},
        {
            "url": "https://a.example/5",
            "title": "Order",
            "snippet": "s",
            "tags": ["a", {"b": None}],
        },
        {"title": "Lone \ud83d half", "url": "https://a.example/6", "snippet": "x"},
    ],
    ids=[
        "newline+field-like",
        "empty+numeric",
        "padded+literal",
        "extras-only",
        "nested",
        "surrogate",
    ],
)
def test_rendering_is_lossless_for_awkward_values(hit: dict[str, Any]) -> None:
    raw = json.dumps(
        {"query": "q", "results": [hit, {"title": "B", "url": "https://b.example", "snippet": "s"}]}
    )
    out = _render(raw)
    out.encode("utf-8")
    assert _decode_search(out) == json.loads(_clip(raw))


def test_top_level_hit_list_renders_without_a_header() -> None:
    raw = json.dumps([_hit(1), _hit(2)])
    out = _render(raw)
    assert out.startswith("1. Mark Sutcliffe - result 1\n")
    assert _decode_search(out) == json.loads(_clip(raw))


@pytest.mark.parametrize(
    ("payload", "tool_name"),
    [
        (_TAVILY_SHAPE, "list_issues"),  # a hit-shaped list from a NON-search tool
        (json.dumps({"web": {"results": [_hit(1)]}}), "search"),  # nested hit list
        (json.dumps({"q": "x", "a": [_hit(1)], "b": [_hit(2)]}), "search"),  # two hit lists
        (json.dumps([{"id": 1, "body": "x " * 200}]), "list_issues"),  # arbitrary JSON
        (json.dumps({"url": "https://x.example", "content": "c", "bad key": 1}), "fetch"),
        (json.dumps({"url": "https://x.example", "content": "c\ud800"}), "fetch"),
        (json.dumps({"content": "no url"}), "fetch"),
    ],
    ids=[
        "non-search",
        "nested",
        "two-lists",
        "arbitrary",
        "bad-key",
        "surrogate-content",
        "no-url",
    ],
)
def test_other_shapes_are_never_rendered(payload: str, tool_name: str) -> None:
    out = _render(payload, tool_name=tool_name)
    assert out == normalize_tool_text(
        payload, "prose-links", tool_name=tool_name, search_snippet_chars=400
    )


def test_rendering_off_keeps_json() -> None:
    assert _clip(_TAVILY_SHAPE) == normalize_tool_text(
        _TAVILY_SHAPE,
        "prose-links",
        tool_name="search",
        search_snippet_chars=400,
        text_rendering=False,
    )
    assert (
        normalize_tool_text(_TAVILY_SHAPE, "off", tool_name="search", text_rendering=True)
        == _TAVILY_SHAPE
    )


def test_rendered_text_that_markdown_would_rewrite_falls_back_to_json() -> None:
    """Idempotence by construction: a rendering the markdown pass would change
    on a second trip (a short title still carrying image markup) is not used."""
    hit = {
        "title": "Logo ![](https://px.example/1.gif) Chow",
        "url": "https://a.example",
        "snippet": "s",
    }
    raw = json.dumps({"query": "q", "results": [hit]})
    out = _render(raw)
    assert json.loads(out)["results"][0]["title"] == hit["title"]


def test_all_seen_search_still_renders_as_text() -> None:
    """Regression (real-path run): a repeat search whose every hit was seen
    lost every snippet and then fell back to JSON instead of rendering."""
    from local_operator.tools.result_normalize import seen_url_key

    seen = frozenset(seen_url_key(h["url"]) for h in json.loads(_TAVILY_SHAPE)["results"])
    out = _render(_TAVILY_SHAPE, seen_urls=seen)
    assert out.split("\n")[:4] == [
        "query: Mark Sutcliffe Ottawa",
        "results:",
        "1. Mark Sutcliffe - result 1",
        "   https://example.com/sutcliffe/1?utm_source=x&id=1",
    ]
    assert _decode_search(out) == json.loads(_elide(_TAVILY_SHAPE, seen))
    assert _render(out, seen_urls=seen) == out


@pytest.mark.parametrize("mode", ["prose-links", "strip-links"])
def test_rendering_is_idempotent(mode: str) -> None:
    page = json.dumps({"url": "https://x.example/p", "content": PAGE, "offset": 0})
    for raw, tool in ((_TAVILY_SHAPE, "search"), (page, "fetch")):
        once = _render(raw, tool_name=tool, mode=mode)
        assert once != raw
        assert _render(once, tool_name=tool, mode=mode) == once


# --- result shaping v2: boilerplate section elision (structural-economy §5) --

_WIKI = "\n".join(
    [
        "# Olivia Chow",
        "",
        "Olivia Chow (born 24 March 1957) is the 66th mayor of Toronto.",
        "",
        "## Career",
        "",
        "She served on Toronto City Council from 1991 to 2005.",
        "",
        "### Notes",
        "",
        "1. A footnote that is not a cited fact.",
        "",
        "### Electoral record",
        "",
        "Won the 2023 by-election.",
        "",
        "## See also",
        "",
        "* Jack Layton",
        "",
        "## References",
        "",
        "1. Smith, J. (2023). *Chow wins*.",
        "",
        "```",
        "## Not a heading inside code",
        "```",
        "",
        "## External links",
        "",
        "* Official site",
        "",
        "## Personal life",
        "",
        "Chow married Jack Layton in 1988.",
        "",
        "## Further reading",
        "",
        "* A book",
    ]
)


def _page(content: str, url: str = "https://en.wikipedia.org/wiki/Olivia_Chow") -> str:
    return json.dumps({"url": url, "content": content, "offset": 0, "total_chars": len(content)})


@pytest.mark.parametrize("mode", ["prose-links", "strip-links"])
def test_boilerplate_sections_are_dropped_from_page_content(mode: str) -> None:
    content = json.loads(normalize_tool_text(_page(_WIKI), mode))["content"]
    assert content == "\n".join(
        [
            "# Olivia Chow",
            "",
            "Olivia Chow (born 24 March 1957) is the 66th mayor of Toronto.",
            "",
            "## Career",
            "",
            "She served on Toronto City Council from 1991 to 2005.",
            "",
            "### Electoral record",
            "",
            "Won the 2023 by-election.",
            "",
            "## Personal life",
            "",
            "Chow married Jack Layton in 1988.",
        ]
    )


def test_section_elision_is_only_for_page_payloads() -> None:
    # Plain markdown text (a README read, a docs tool) keeps its Notes section.
    plain = _WIKI + "\n[x](https://x.example)"
    assert "### Notes" in normalize_tool_text(plain, "prose-links")
    # A JSON leaf that is not a fetched page keeps it too.
    issue = json.dumps({"title": "Bug", "body": _WIKI + "\n" + "detail " * 40})
    assert "## References" in json.loads(normalize_tool_text(issue, "prose-links"))["body"]
    # A relative/non-web ``url`` is not a fetched page.
    relative = normalize_tool_text(_page(_WIKI, url="docs/a.md"), "prose-links")
    assert "## References" in json.loads(relative)["content"]
    # ``off`` is the identity.
    assert normalize_tool_text(_page(_WIKI), "off") == _page(_WIKI)


def test_section_elision_keeps_nav_chrome_lines_and_is_idempotent() -> None:
    """Rejected lever (lost a cited fact): short chrome-looking lines stay."""
    content = "Minister of Foreign Affairs\nShare\nEdit links\n\n## References\n\n* ref"
    once = normalize_tool_text(_page(content), "prose-links")
    assert json.loads(once)["content"] == "Minister of Foreign Affairs\nShare\nEdit links"
    assert normalize_tool_text(once, "prose-links") == once
