"""QA round 7: the reducer's own renderings the verifier still mis-recomputed.

Each refused ONE epoch, and adoption is all-or-nothing per response, so each
voided a whole session on QA's independent corpus (8/12 adopted, 45.8%). Every
fixture here is accepted by the fixed verifier and refused with its fix deleted
(the mutation rows on PR #3078 name the deletion for each).
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.saver.validate import HEADER, SaverRejected, check_provenance

_PAD = " the deployment finished and the log line says everything is fine" * 6


def _filler(count: int = 6) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": f"filler line {n} " + _PAD}]}
        for n in range(count)
    ]


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def _result(call_id: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [{"type": "text", "text": text}],
        **extra,
    }


def test_a_group_verb_is_the_calls_own_verb_not_a_substring() -> None:
    """R7-A: `sed` must not claim a call whose OUTPUT says `Raised`.

    The reducer names each grouped read by the first word of its command
    (`reduce_v2.call` -> `shell_read_only`). The verifier matched the verb as a
    substring of the whole record, merged result included. Here the real `sed`
    call is already spent, so the substring rule walked on and claimed the
    grep call whose output says `Raised` — the call the NEXT record needs.
    Matching the derived verb leaves that grep call free.
    """
    rows = [
        _call("c1", "bash", '{"command": "sed -n 1,5p src/app.py"}'),
        _result("c1", "line one"),
        _call("c2", "bash", '{"command": "grep -n ScriptedStream src/stream.py"}'),
        _result("c2", "40: Raised in class ScriptedStream"),
    ] + _filler()
    check_provenance(
        HEADER
        + "\ncall bash: sed -n 1,5p src/app.py"
        + "\ncall: [radient:1 read-only commands, all ok: sed]"
        + "\ncall bash: grep -n ScriptedStream src/stream.py",
        rows,
    )


def test_a_failed_result_is_matched_in_the_reducers_order() -> None:
    """R7-B: `exit N` comes before the body, so the source must too.

    `reduce_v2.result` renders a failed result as `exit -9` then its body. The
    verifier's record held the raw text first and the `exit N <body>`
    rendering last, so the in-order cursor spent the last line on the status
    piece and the body piece had nothing left.
    """
    body = "SCRATCHPAD CAP EXCEEDED: the scratchpad crossed its size limit"
    rows = [
        _call("c1", "bash", '{"command": "python3 build.py"}'),
        _result("c1", f"exit code: -9\n{body}", is_error=True),
    ] + _filler()
    check_provenance(
        HEADER + "\ncall bash: python3 build.py\nerror: exit -9 [radient:omitted 1 lines, 62 chars]"
        "\n\t" + body,
        rows,
    )


def _image_rows() -> list[dict[str, Any]]:
    image = {
        "type": "image_ref",
        "sha256": "f502f7728422c37807376dcad5758051046d37c1638aa2e800fe774dfde35846",
        "bytes": 18_400,
        "width": 400,
        "height": 200,
        "mime": "image/png",
    }
    return [
        {"role": "user", "content": [{"type": "text", "text": "look at this frame " + _PAD}]},
        {"role": "user", "content": [image]},
    ] + _filler()


_RECEIPT = (
    "[radient:image omitted — image/png 400x200 18.4 kB sha256=f502f7728422; it was visible "
    "when first sent and is not visible now; describe it only from what was said about it]"
)


def test_an_image_receipt_from_the_reducer_is_accepted() -> None:
    """R7-C: every image receipt was refused, so every multimodal session was.

    Four defects, each needed: the `sha256=` key was never split from its
    value, the media digest set hashed the row instead of carrying the
    image_ref's own digest, the receipt's mime/dimensions slots were checked
    against a record with no text, and its size against the record's own
    CHARACTER count instead of the attachment's bytes. The size here (18.4 kB)
    is larger than every row's text, so no other candidate's ceiling can admit
    it: only the image_ref's own byte count can.
    """
    check_provenance(HEADER + "\nuser: look at this frame\nuser: " + _RECEIPT, _image_rows())


@pytest.mark.parametrize(
    "forged",
    [
        _RECEIPT.replace("f502f7728422", "deadbeef0123"),
        _RECEIPT.replace("400x200", "401x200"),
        _RECEIPT.replace("18.4 kB", "9.9 MB"),
        _RECEIPT.replace("image/png", "text/html"),
    ],
)
def test_an_image_receipt_slot_must_be_the_attachments_own(forged: str) -> None:
    """The receipt's slots are DERIVED from the image_ref, so a forged one fails."""
    with pytest.raises(SaverRejected, match="provenance"):
        check_provenance(HEADER + "\nuser: look at this frame\nuser: " + forged, _image_rows())


def test_a_call_carries_one_canonical_rendering_not_two() -> None:
    """S-21: the cleaned rendering REPLACES the plain canonical line.

    Appending it gave every dict-argument call three matchable copies of its
    arguments (raw, canonical, cleaned), so the structural rules alone admitted
    three emitted copies (2.664x with the ruler disabled).
    """
    from local_operator.saver.provenance import source_records

    rows = [_call("c1", "eval", '{"code": "x = 1\\n", "i": "set x", "timeout": 90}')]
    records, _, _ = source_records(rows)
    text = records["call"][0].text
    assert '{"code":"x = 1\\n","i":"set x","timeout":90}' not in text
    assert '{"code":"x = 1","timeout":90}' in text


# ---------------------------------------------------------------------------
# Round 8 (security S-27, S-24, S-25): the multimodal denominator, and slots
# bound to the ref that printed them.
# ---------------------------------------------------------------------------

_TAIL = (
    "frame set captured. The composer settled, the banner is aligned with the "
    "grid, and the status chip still reads connected in the corner of every "
    "frame. "
) * 4


def _multimodal_rows(n: int, tail: str = _TAIL) -> list[dict[str, Any]]:
    """`n` image-only rows plus one text row the epoch deletes."""
    rows: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_ref",
                    "sha256": f"{i:02x}" + "a" * 62,
                    "bytes": 1125 + i,
                    "width": 400 + 100 * i,
                    "height": 200 + 100 * i,
                    "mime": "image/png" if i % 2 == 0 else "image/jpeg",
                }
            ],
        }
        for i in range(n)
    ]
    return rows + [{"role": "user", "content": [{"type": "text", "text": tail}]}]


def _receipt_epoch(rows: list[dict[str, Any]]) -> str:
    from local_operator.saver.provenance import _media_receipts

    return HEADER + "\n" + "\n".join(f"user: {receipt}" for receipt in _media_receipts(rows))


def test_a_multimodal_epoch_is_measured_against_the_spans_own_receipts() -> None:
    """S-27: the denominator must count the receipt an image row costs.

    `parts` holds TEXT blocks, so an image row added nothing to the span side
    while the epoch paid ~49 cl100k tokens of receipt for it: an honest
    four-image epoch measured 3.98x on the text-only denominator — refused —
    and 0.958x once the span's own receipts are counted. A refusal is charged
    at full, so that was a bill for an honest multimodal turn.
    """
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.saver.provenance import _media_receipts, source_records

    rows = _multimodal_rows(4)
    text = _receipt_epoch(rows)
    parts = source_records(rows)[1]
    text_only = count_text_tokens("\n".join(parts))
    with_receipts = count_text_tokens("\n".join(list(parts) + _media_receipts(rows)))
    measured = count_text_tokens(text)
    # The fixture is only a guard if the OLD denominator refused it.
    assert measured / text_only > 1.0, "fixture must amplify on the text-only denominator"
    assert measured / with_receipts < 0.9, "and be cheaper with the span's receipts counted"
    check_provenance(text, rows)


def test_a_receipt_slot_belongs_to_the_ref_that_printed_it() -> None:
    """S-24: image B's mime and dimensions are not admissible for image A.

    Asserted on `_check_slots` directly, because the grammar layer can also
    catch a transplanted slot and would then hide whether the BINDING is doing
    any work. Every value here exists in the span; only the pairing is wrong,
    which is exactly what span-wide admission (`token in media`) accepted.
    """
    from local_operator.saver.provenance import (
        _check_slots,
        _derived_digests,
        _elements,
        _media_receipts,
        source_records,
    )

    rows = _multimodal_rows(2, tail=_TAIL)
    recs, parts, media = source_records(rows)
    derived = _derived_digests(parts)
    images = [rec for rec in recs["user"] if "image_ref" in rec.block_types]
    assert len(images) == 2, images
    first, second = images
    assert first.media_slots and second.media_slots
    assert first.media_slots != second.media_slots

    own = _elements(_media_receipts(rows)[0])
    other = _elements(_media_receipts(rows)[1])
    # Its own receipt passes for the record that printed it...
    _check_slots(own, first, media, derived, [99], 10**9)
    _check_slots(other, second, media, derived, [99], 10**9)
    # ...and the other ref's does not.
    with pytest.raises(SaverRejected):
        _check_slots(other, first, media, derived, [99], 10**9)


def test_a_receipt_size_is_the_attachments_exact_byte_rendering() -> None:
    """S-25: the old rule accepted any size within `ceiling * 1.05 + 64`."""
    rows = _multimodal_rows(1, tail=_TAIL)
    receipts = _receipt_epoch(rows).split("\n")
    forged = receipts[1].replace("1.1 kB", "1.0 kB")
    with pytest.raises(SaverRejected):
        check_provenance(HEADER + "\n" + forged, rows)
