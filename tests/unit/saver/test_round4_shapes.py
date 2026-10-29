"""Round-4 findings: the injective cursor, and every case it must refuse.

R4-1 found the round-4 verifier *documented* an injective, in-order mapping and
then matched fragments as substrings: `abcabc` emitted as `bc` + `ab` was
accepted at 46 o200k tokens against a 5-token span (9.2x). These fixtures pin
the three properties that fix it — the cursor only moves forward (order), a
matched line is never reconsidered (distinctness), and `MIN_CUT` decides whether
a short prefix or suffix is a cut at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.saver.validate import HEADER, SaverRejected, check_provenance


def _span(first: str, lines: int = 6) -> list[dict[str, Any]]:
    """One real source row plus filler, so the epoch has something to delete."""
    return [{"role": "user", "content": [{"type": "text", "text": first}]}] + [
        {"role": "user", "content": [{"type": "text", "text": f"filler line {n}"}]}
        for n in range(lines)
    ]


def _reject(rows: list[dict[str, Any]], record: str) -> None:
    with pytest.raises(SaverRejected):
        check_provenance(HEADER + "\n" + record, rows)


def test_r4_1_a_mid_line_split_into_short_pieces_is_refused() -> None:
    """The reviewer's 9.2x case: `abcabc` emitted as `bc` then `ab`.

    Guarded by the LINE CURSOR, not by ``MIN_CUT``: both pieces are two
    characters, so both count as cuts, and it is one source line answering two
    emitted lines that refuses the epoch. The mutation matrix in the PR comment
    records that attribution; `MIN_CUT`'s own isolated fixture is
    `test_a_one_character_fragment_must_be_a_whole_word` in
    `test_round5_units.py`.
    """
    _reject(_span("abcabc"), "user: bc\n\tab")


def test_r4_1_the_cursor_alone_refuses_two_long_pieces_of_one_source_line() -> None:
    """Distinctness, isolated: both pieces are >= MIN_CUT, so only the cursor fires.

    `cdef` is a suffix of `abcdefabcdef` and `abcd` is a prefix of it, so with
    reuse allowed the epoch is accepted; `cursor = found + 1` is what refuses it.
    """
    _reject(_span("abcdefabcdef"), "user: cdef\n\tabcd")


# The word-fragmentation ATTACK (a 5.08x class) needs few source lines and many
# emitted ones, which the epoch line budget refuses — measured in the matrix.
# With a span long enough to satisfy the budget the same shape is NOT an attack:
# 400 whole source words out of a 500-line span costs fewer tokens than the span,
# so there is nothing to refuse. Recorded here so the guard attribution is honest.


def test_r4_1_one_source_line_cannot_answer_two_emitted_lines() -> None:
    """Distinctness: the second `alpha` has no second source line to spend."""
    rows = [{"role": "user", "content": [{"type": "text", "text": "alpha\nbeta"}]}] + _span(
        "filler", 6
    )[1:]
    _reject(rows, "user: alpha\n\talpha")


def test_r4_1_word_fragmentation_is_refused_with_a_token_ratio_above_one() -> None:
    """The 5.08x class: a repeated-word span shredded one word per emitted line.

    Measured here rather than asserted by name: the epoch is refused, and the
    o200k ratio of the would-be epoch against its span is above 1.
    """
    source = "word " * 4000
    rows = _span(source.strip())
    record = "user: " + "\n\t".join(["word"] * 4000)
    _reject(rows, record)
    tokenizer = pytest.importorskip(
        "radient_context_reduction.tokenize", reason="reference tokenizer not on PYTHONPATH"
    )
    span_tokens = tokenizer.token_count_text(source)
    epoch_tokens = tokenizer.token_count_text(record)
    assert epoch_tokens / span_tokens > 1.0, (epoch_tokens, span_tokens)


@pytest.mark.parametrize(
    "source",
    [
        "the quick brown fox jumps over the lazy dog " * 400,  # QA's prose span
        "internationalization " * 750,  # QA's repeated-word span
    ],
    ids=["prose", "repeated-word"],
)
def test_r4_1_qa_fragmentation_spans_are_refused_or_cost_less_than_their_span(
    source: str,
) -> None:
    """QA's 1.70x / 3.95x spans: refused here, so the ratio question is moot."""
    characters = [ch for ch in source if ch.strip()]
    record = "user: " + characters[0] + "\n\t" + "\n\t".join(characters[1:2000])
    _reject(_span(source.strip()), record)


def test_a_normal_prefix_cut_is_still_accepted() -> None:
    """`MIN_CUT` must not reject honest output: a cut line is a prefix."""
    source = "the quick brown fox jumps over the lazy dog and then keeps running " * 12
    check_provenance(HEADER + "\nuser: the quick brown fox jumps over the lazy dog", _span(source))


def test_s16_the_cursor_refuses_one_source_line_backstopping_many_emitted_lines() -> None:
    """S-16 / the 5.08x class, with the cursor as the ONLY guard.

    60 source lines, 50 emitted copies of line 0. The epoch is a strict line
    reduction (50 < 60) and every emitted line is a whole source line, so the
    line budget and `MIN_CUT` are both satisfied; only distinctness refuses it.
    Without `cursor = found + 1` the epoch is accepted at ~28x its span tokens,
    which the mutation matrix in the PR comment records.
    """
    sentence = "the deployment finished and the log line says everything is fine " * 6
    rows = [{"role": "user", "content": [{"type": "text", "text": sentence.splitlines()[0]}]}] + [
        {"role": "user", "content": [{"type": "text", "text": "x"}]} for _ in range(59)
    ]
    record = "user: " + "\n\t".join([sentence.splitlines()[0]] * 50)
    _reject(rows, record)
    tokenizer = pytest.importorskip(
        "radient_context_reduction.tokenize", reason="reference tokenizer not on PYTHONPATH"
    )
    span_tokens = tokenizer.token_count_text(sentence.splitlines()[0] + "x" * 59)
    epoch_tokens = tokenizer.token_count_text(record)
    assert epoch_tokens > span_tokens, (epoch_tokens, span_tokens)


def test_the_marker_cap_refuses_stuffing_that_the_line_budget_cannot_see() -> None:
    """The cap is the only guard here: one emitted line, 50 markers on it.

    The epoch is one line against a seven-line span, so the line budget passes,
    and the carried content is a whole source line, so the matcher passes.
    Deleting `_check_bounds` accepts this, which the mutation matrix records.
    """
    record = "user: a source line with ordinary words in it " + (
        "[radient:omitted 1 lines, 40 chars] " * 50
    )
    _reject(_span("a source line with ordinary words in it"), record)


def test_the_note_cap_refuses_stuffing_that_the_line_budget_cannot_see() -> None:
    """Same isolation for the reducer's notes, sized so no slot is a free token."""
    record = "user: a source line with ordinary words in it " + ("[40 chars] " * 50)
    _reject(_span("a source line with ordinary words in it"), record)
