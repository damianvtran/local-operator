"""S-1 residual: chunk only partial-mask cleanup, never the credential scanner.

Compare byte-for-byte to the pre-remediation whole-text cleanup. In particular,
no arbitrary cut may separate credential material from a marker: only LF cuts
are safe for these two patterns, and a long unbroken line remains whole.
"""

from __future__ import annotations

import pytest

from local_operator import redaction_shapes as shapes


def _whole(text: str) -> str:
    for _ in range(3):
        closed = shapes._INCOMPLETE_MASK_RE.sub(shapes.REDACTION_MARKER, text)
        closed = shapes._INCOMPLETE_MASK_LEFT_RE.sub(shapes.REDACTION_MARKER, closed)
        if closed == text:
            break
        text = closed
    return text


@pytest.mark.parametrize("offset", [65500, 65535, 65536, 65537, 65580])
@pytest.mark.parametrize(
    "fragment",
    [
        "prefix'" + shapes.REDACTION_MARKER + "'tail",
        'password: "' + shapes.REDACTION_MARKER + '"\nnext line',
        "AWS_ACCESS_KEY_ID='" + shapes.REDACTION_MARKER + "'",
        "x\r\n" + shapes.REDACTION_MARKER + "'tail\n",
        "x\n'" + shapes.REDACTION_MARKER + "'tail\n",
        "(prefix'" + shapes.REDACTION_MARKER + "'tail),suffix",
    ],
)
def test_mask_cleanup_lf_chunks_equal_whole_text(offset: int, fragment: str) -> None:
    prefix = ("ordinary source text\n" * 4000)[:offset]
    text = prefix + fragment + "\n" + "more ordinary text\n" * 8000
    assert shapes._close_partial_masks(text) == _whole(text)


def test_marker_free_paste_never_enters_partial_mask_regex(monkeypatch) -> None:
    class RefuseScan:
        def sub(self, *_args):
            raise AssertionError("a marker-free paste cannot have an incomplete mask")

    monkeypatch.setattr(shapes, "_INCOMPLETE_MASK_RE", RefuseScan())
    monkeypatch.setattr(shapes, "_INCOMPLETE_MASK_LEFT_RE", RefuseScan())
    paste = "ordinary source text\n" * 60000
    assert shapes._close_partial_masks(paste) is paste


def test_a_long_unbroken_credential_is_not_split() -> None:
    text = "prefix" * 20000 + "'" + shapes.REDACTION_MARKER + "'tail"
    assert shapes._close_partial_masks(text) == _whole(text)
    assert "prefix" not in shapes._close_partial_masks(text)


@pytest.mark.parametrize(
    "anchor",
    ["PRIVATE KEY", "private_key", "private-key", "privatekey", "prİvate_key", "prıvatekey"],
)
@pytest.mark.parametrize("offset", [65525, 65530, 65535, 65536, 65537])
def test_multiline_anchor_scan_carries_window_overlap(anchor: str, offset: int) -> None:
    assert shapes._has_multiline_shape_anchor("x" * offset + anchor + "x" * 70000)


def test_multiline_gate_is_identical_to_ungated_scan_over_credential_corpus(monkeypatch) -> None:
    from tests.unit.secrets.credential_shape_corpus import (
        NEGATIVE_CASES,
        POSITIVE_CASES,
    )

    cases = [case.text for case in (*POSITIVE_CASES, *NEGATIVE_CASES)]
    # Include a real multiline key at the bounded-scan cut, with unchanged
    # context on both sides. Compare outputs AND hit order, which determines
    # what the session's redaction registry learns.
    cases.extend(
        "ordinary\n" * 7281 + text
        for text in tuple(cases)
        if shapes._has_multiline_shape_anchor(text)
    )
    for text in cases:
        optimized = shapes.scrub_shapes_with_hits(text)
        with monkeypatch.context() as patch:
            patch.setattr(shapes, "_has_multiline_shape_anchor", lambda text: True)
            patch.setattr(shapes, "_close_partial_masks", _whole)
            original = shapes.scrub_shapes_with_hits(text)
        assert optimized == original
