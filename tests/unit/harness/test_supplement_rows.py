"""``rows.supplement_row_details`` — the one reader for the two row shapes.

The decision under it (which disposition a row paints as) is
``supplements.contract.reader_disposition``; this reader only answers "is this
row one of ours, and what does it say". Both shapes a surface holds are pinned
here because a divergence between them is invisible until a surface shows
nothing for a row the journal holds — the defect class this module exists to
prevent (see ``harness/rows.py``'s docstring).
"""

from __future__ import annotations

from local_operator.harness.rows import supplement_row_details


def test_reads_a_bare_payload_mapping() -> None:
    payload = {"custom_type": "supplement_v1", "details": {"anchor": "m1", "state": "done"}}
    assert supplement_row_details(payload) == {"anchor": "m1", "state": "done"}


def test_reads_a_wrapped_entry() -> None:
    entry = {
        "id": "e1",
        "type": "custom",
        "payload": {"custom_type": "supplement_v1", "details": {"anchor": "m2"}},
    }
    assert supplement_row_details(entry) == {"anchor": "m2"}


def test_refuses_everything_else() -> None:
    assert supplement_row_details(None) is None
    assert supplement_row_details("not a row") is None
    assert supplement_row_details({"custom_type": "attention_v1", "details": {"a": 1}}) is None
    # A matching type with a malformed details body paints nothing rather than
    # raising on a render path.
    assert supplement_row_details({"custom_type": "supplement_v1", "details": []}) is None
    assert supplement_row_details({"custom_type": "supplement_v1"}) is None
