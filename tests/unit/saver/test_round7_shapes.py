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
