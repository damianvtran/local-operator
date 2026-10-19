from __future__ import annotations

import pytest

from local_operator.library.classify import (
    CATEGORIES,
    KINDS,
    category_of,
    classify,
    content_type,
    is_known_extension,
)


@pytest.mark.parametrize(
    ("name", "kind", "category"),
    [
        ("/tmp/a.png", "image", "image"),
        ("/tmp/a.JPG", "image", "image"),
        ("/tmp/loop.gif", "image", "image"),
        ("/tmp/voice.mp3", "audio", "sound"),
        ("/tmp/take.m4a", "audio", "sound"),
        ("/tmp/clip.mp4", "video", "video"),
        ("/tmp/paper.pdf", "pdf", "document"),
        ("/tmp/deck.pptx", "pptx", "presentation"),
        ("/tmp/letter.docx", "docx", "document"),
        ("/tmp/notes.md", "markdown", "document"),
        ("/tmp/page.html", "html", "document"),
        ("/tmp/rows.csv", "spreadsheet", "spreadsheet"),
        ("/tmp/sheet.xlsx", "spreadsheet", "spreadsheet"),
        ("/tmp/app.py", "code", "code"),
        ("/tmp/Dockerfile", "code", "code"),
        ("/tmp/readme.txt", "text", "document"),
        ("/tmp/blob", "other", "other"),
    ],
)
def test_representative_paths(name: str, kind: str, category: str) -> None:
    answer = classify(name)
    assert (answer.kind, answer.category) == (kind, category)
    assert answer.kind in KINDS and answer.category in CATEGORIES


def test_mime_is_the_fallback_and_never_overrides_a_suffix() -> None:
    assert classify("mystery", "audio/mpeg").kind == "audio"
    assert classify("styled.md", "text/plain").kind == "markdown"
    assert classify("art", "image/svg+xml").kind == "image"


def test_content_type_prefers_a_supplied_mime() -> None:
    assert content_type("x.png") == "image/png"
    assert content_type("x.bin", "text/plain; charset=utf-8") == "text/plain"
    assert content_type("unknown.bin") == "application/octet-stream"


def test_category_of_tolerates_an_unknown_kind() -> None:
    assert category_of("audio") == "sound"
    assert category_of("nope") == "other"


def test_known_extensions_gate_the_mention_pass() -> None:
    assert is_known_extension("report.pdf")
    assert is_known_extension("Dockerfile")
    assert not is_known_extension("no-suffix")
    assert not is_known_extension("data.unknownext")
