"""Render-pipeline unit tests: both HTML backends, JSON, text, binary, quality."""

from __future__ import annotations

import builtins

import pytest

from local_operator.web_fetch import render


def test_markdownify_backend_renders_headings_and_links() -> None:
    html = (
        "<html><body><h1>Title</h1><p>Hello <a href='https://x.example'>link</a></p></body></html>"
    )
    text, method = render.render_html(html)
    assert method == "markdownify"
    assert "# Title" in text
    assert "[link](https://x.example)" in text


def test_stdlib_fallback_when_forced() -> None:
    html = "<html><body><h1>Heading</h1><p>Body text here that is long enough.</p></body></html>"
    text, method = render.render_html(html, force_stdlib=True)
    assert method == "stdlib"
    assert "# Heading" in text
    assert "Body text here" in text


def test_stdlib_fallback_when_markdownify_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare install (no markdownify/bs4) must still render HTML, via stdlib.

    The import is forced to fail so the degradation path is exercised for real
    rather than assumed — this is the contract that keeps a bare
    ``pip install local-operator`` fully functional (design test 15).
    """
    real_import = builtins.__import__

    def _no_markdownify(name: str, *args: object, **kwargs: object) -> object:
        if name in ("markdownify", "bs4"):
            raise ImportError(f"forced: {name} unavailable")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_markdownify)
    assert render.html_backend_available() is False
    html = "<html><body><h1>Heading</h1><p>Some readable body text goes here.</p></body></html>"
    text, method = render.render_html(html)
    assert method == "stdlib"
    assert "Heading" in text
    assert "readable body text" in text


def test_strip_tags_drop_navigation_chrome() -> None:
    html = (
        "<html><body><nav>Home About Contact</nav>"
        "<script>var x=1;</script>"
        "<p>The real content that matters is right here in the paragraph.</p>"
        "</body></html>"
    )
    text, _ = render.render_html(html, force_stdlib=True)
    assert "var x" not in text
    assert "real content" in text


def test_render_json_pretty_prints() -> None:
    text, method = render.render_json('{"b":2,"a":1}')
    assert method == "json"
    assert '"b": 2' in text
    assert text.count("\n") >= 1  # indented, not one line


def test_render_json_malformed_falls_back_to_text() -> None:
    text, method = render.render_json("{not valid json")
    assert method == "text"
    assert "not valid json" in text


def test_binary_notice_reports_type_and_size_without_inlining() -> None:
    text, method = render.binary_notice("application/pdf", 2 * 1024 * 1024, "https://x/y.pdf")
    assert method == "binary"
    assert "application/pdf" in text
    assert "2.0 MB" in text
    assert "https://x/y.pdf" in text
    assert "not inlined" in text


def test_low_quality_flags_short_output() -> None:
    assert render.is_low_quality("tiny") is True


def test_low_quality_flags_js_gate() -> None:
    assert render.is_low_quality("Please enable JavaScript to view this site." * 5) is True


def test_low_quality_flags_navigation_heavy() -> None:
    menu = "\n".join(["Home", "About", "Blog", "Docs", "Contact", "Login", "Signup"])
    assert render.is_low_quality(menu) is True


def test_low_quality_passes_real_prose() -> None:
    prose = (
        "This is a paragraph of genuine article content that runs on for a while "
        "with several sentences of real information. It explains something at "
        "length and does not look like a navigation menu at all, so the quality "
        "gate should leave it alone and not suggest reaching for the browser."
    )
    assert render.is_low_quality(prose) is False
