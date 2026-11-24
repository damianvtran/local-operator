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


def test_one_link_list_items_are_dropped() -> None:
    out = normalize_markdown(PAGE)
    assert "Home" not in out
    assert "example.com/news" not in out
    assert "facebook.com" not in out  # an icon-only bullet is one link too


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
    text = "intro  text\n```\n[a](/b)   x"
    assert normalize_markdown(text) == "intro text\n```\n[a](/b)   x"


def test_list_continuation_is_not_mistaken_for_code() -> None:
    text = "* item\n\n    continued [rel](/x)"
    assert normalize_markdown(text) == "* item\n\n    continued rel"


def test_whitespace_collapse() -> None:
    out = normalize_markdown(PAGE)
    assert "\n\n\n" not in out
    assert "inline and [[n]]" in out
    assert not any(line != line.rstrip() for line in out.split("\n"))


def test_plain_text_passes_through_except_whitespace() -> None:
    text = "Line one.\nLine two has  two spaces.   \n\n\n\nEnd."
    assert normalize_markdown(text) == "Line one.\nLine two has two spaces.\n\nEnd."


def test_many_unmatched_backticks_and_brackets_stay_linear() -> None:
    # A quadratic scan would take minutes here; the bounded/paired scans do not.
    hostile = ("`a [b " * 40_000) + "\n" + ("[" * 100_000)
    assert normalize_markdown(hostile)


# ---- normalize_tool_text: JSON routing ---------------------------------------------


def _fetch_payload() -> dict[str, object]:
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
    long_url_text = "see [x](/y) " * 30
    raw = json.dumps({"source_url": long_url_text, "body": long_url_text})
    parsed = json.loads(normalize_tool_text(raw))
    assert parsed["source_url"] == long_url_text
    assert parsed["body"] != long_url_text


def test_json_non_ascii_is_emitted_raw() -> None:
    out = normalize_tool_text(json.dumps({"name": "Montréal"}))
    assert out == '{"name":"Montréal"}'


def test_json_with_duplicate_keys_is_returned_unchanged() -> None:
    raw = '{"a": 1, "a": 2}'
    assert normalize_tool_text(raw) == raw


def test_invalid_json_takes_the_markdown_path() -> None:
    assert normalize_tool_text("{not json  [a](/b)") == "{not json a"


def test_scalar_json_takes_the_markdown_path() -> None:
    assert normalize_tool_text("42") == "42"


def test_off_is_identity_byte_for_byte() -> None:
    for text in (PAGE, json.dumps(_fetch_payload(), indent=2), "  x  \n\n\n", ""):
        assert normalize_tool_text(text, "off") is text


def test_unknown_mode_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        normalize_tool_text("x", "everything")


_FIXTURES = [
    PAGE,
    PROSE,
    json.dumps(_fetch_payload(), indent=2),
    json.dumps([_fetch_payload(), {"k": PAGE}]),
    "* item\n\n    continued [rel](/x)\n\n    more",
    "para\n\n* [nav](/x)\n\n    after nav [r](/y)",
    "![](a) * [b](/c)\n![alt](d)",
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
