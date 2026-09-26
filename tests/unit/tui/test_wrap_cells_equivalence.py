"""``wrap_cells`` breaks an oversized token in linear work, at exactly the same places.

The break loop used to re-measure the whole SHRINKING remainder on every pass
(``while cell_len(word) > width``) while each pass removes only ``width`` cells, so
the work was quadratic in the token's length. Instrumented ``cell_len`` — every call
counted and every character it scanned summed — measured 41,419 calls over 10,140,339
characters for a 40,000-character token: 253x the token's own length, and 194.86 ms
for 80,000 characters **on the TUI main thread**, where one pasted base64 blob or
minified log line stalls a frame.

The rewrite answers that loop condition by arithmetic (``remaining -= consumed``)
for the words where ``cell_len`` is additive over characters, and keeps the original
measured loop for the ones where it is not. So the property this module pins is not
"it got faster": it is that both paths still return exactly what the old loop
returned, on every input whose shape can reach them.

**The oracle is the old implementation, verbatim, at the bottom of this file.** It is
deliberately a copy rather than a re-derivation: a re-derivation would encode this
author's understanding of the old behaviour, and the whole risk in this change is
that the old behaviour is not what this author thinks it is. Equivalence is asserted
on the OUTPUT (the rows) over a generated corpus that crosses the widths, the lengths,
the exact multiples, and every codepoint class that makes ``cell_len`` non-additive.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator

import pytest
from rich.cells import cell_len as rich_cell_len

from local_operator.tui.widgets import transcript
from local_operator.tui.widgets.transcript import wrap_cells

#: The characters whose presence makes ``cell_len`` non-additive over a word's
#: characters: a keycap cluster measures 2 where its characters sum to 1, and a
#: joiner swallows the character after it. Every one of them is in the corpus.
KEYCAP = "1\ufe0f\u20e3"
JOINER = "a\u200db"
EMOJI_SEQUENCE = "\U0001f469\u200d\U0001f4bb"
COMBINING = "e\u0301"
CJK = "\u65e5\u672c\u8a9e"

#: A small alphabet over which EVERY string up to ``_EXHAUSTIVE_LENGTH`` is wrapped
#: at every width in ``_EXHAUSTIVE_WIDTHS``. Exhaustive-and-small beats
#: sampled-and-large for a boundary bug: the failure mode here is an off-by-one that
#: needs a cluster to straddle a chunk boundary, and only enumeration is guaranteed
#: to place one there.
_EXHAUSTIVE_ALPHABET = ("a", "\u4e2d", "\u200d", "\ufe0f", "e\u0301", "\u200b")
_EXHAUSTIVE_LENGTH = 4
_EXHAUSTIVE_WIDTHS = (1, 2, 3, 5, 8)

#: Widths the family and fuzz corpora are wrapped at. 1-5 are widths a single
#: character cannot fit in; 80 is the app's own column; 200 is past any real frame,
#: where a chunk is a large fraction of the token and the arithmetic has the most
#: room to drift.
_CORPUS_WIDTHS = (1, 2, 3, 5, 7, 10, 20, 33, 80, 81, 199, 200)

#: Inputs with no space in them, in families that reach every branch: plain runs,
#: CJK (2 cells per character), keycap clusters (the non-additive path), joiner
#: sequences, combining marks, and zero-width runs. The lengths hit exact multiples
#: of the widths in ``_CORPUS_WIDTHS`` as well as one either side of them.
_FAMILIES = (
    "x",
    "0123456789abcdef",
    "\u4e2d",
    CJK,
    KEYCAP,
    JOINER,
    EMOJI_SEQUENCE,
    COMBINING,
    "\u200b",
    "\u4e2d" + KEYCAP,
)
_FAMILY_LENGTHS = (1, 2, 3, 4, 5, 7, 8, 16, 39, 40, 41, 79, 80, 81, 160, 200, 201)


def _family_corpus() -> Iterator[tuple[str, int]]:
    for unit in _FAMILIES:
        for length in _FAMILY_LENGTHS:
            text = unit * length
            for width in _CORPUS_WIDTHS:
                yield text, width


def _exhaustive_corpus() -> Iterator[tuple[str, int]]:
    for length in range(_EXHAUSTIVE_LENGTH + 1):
        for letters in itertools.product(_EXHAUSTIVE_ALPHABET, repeat=length):
            text = "".join(letters)
            for width in _EXHAUSTIVE_WIDTHS:
                yield text, width


def _fuzz_corpus() -> Iterator[tuple[str, int]]:
    """Seeded, so a failure is reproducible, over the whole mixed alphabet.

    Token-shaped, not prose-shaped: the long-word branch is the one under test, and a
    space mostly takes the cheap path. The rarely-reached arms are seeded in
    explicitly — a leading, trailing or repeated space, a tab, a newline — because
    ``split(" ")`` makes each of them an ordinary (empty) word, and the empty word is
    the state that produces an empty row.
    """
    alphabet = [
        "a",
        "z",
        "9",
        " ",
        "\t",
        "\n",
        "\u4e2d",
        "\ufe0f",
        "\u200d",
        "\u20e3",
        "\u0301",
        "\u200b",
        "\U0001f600",
    ]
    generator = random.Random(20260925)
    for _ in range(400):
        text = "".join(generator.choice(alphabet) for _ in range(generator.randint(0, 90)))
        for width in (1, 2, 3, 5, 13, 40, 200):
            yield text, width
    for edge in ("", " ", "  ", "   x", "x   ", " x ", "\t", "\n", "\n \n", "a\tb", "\u200b"):
        for width in (1, 2, 3, 5, 13, 40, 200):
            yield edge, width


#: Built once: the fuzz member is seeded, so this is deterministic, and the
#: equivalence test would otherwise re-generate it for its failure message alone.
_CASES: tuple[tuple[str, int], ...] = (
    tuple(_exhaustive_corpus()) + tuple(_family_corpus()) + tuple(_fuzz_corpus())
)


def test_the_corpus_is_worth_asserting_on() -> None:
    """A corpus that quietly shrank to three cases would pass every test below.

    The counts are floors, not measurements: they exist so that deleting a family —
    or an import that silently empties one — fails here instead of making the
    equivalence assertions vacuous.
    """
    assert len(_CASES) > 5000, len(_CASES)
    texts = {text for text, _ in _CASES}
    assert "" in texts, "the empty input is not optional"
    assert " " in texts, "a bare separator is not optional"
    for unit in (KEYCAP, JOINER, EMOJI_SEQUENCE, COMBINING, CJK, "\u200b"):
        assert any(unit in text for text in texts), unit
    assert len({width for _, width in _CASES}) >= 15


def test_the_corpus_reaches_both_paths() -> None:
    """The arithmetic path and the measured path must BOTH be exercised.

    The rewrite is one function with two branches, chosen by the word's characters. A
    corpus of plain ASCII would pin the arithmetic path and never execute the measured
    one, where all the historical behaviour lives — so the non-additive members are
    asserted present by the same test that reads the branch, not merely present in a
    list somewhere.
    """
    additive = [text for text, _ in _CASES if "\u200d" not in text and "\ufe0f" not in text]
    measured = [text for text, _ in _CASES if "\u200d" in text or "\ufe0f" in text]
    assert len(additive) > 2000, len(additive)
    assert len(measured) > 2000, len(measured)


def test_every_case_matches_the_old_implementation() -> None:
    """The lane's backbone: byte-identical rows, over the whole corpus."""
    mismatches = [
        (text, width, wrap_cells_measured(text, width), wrap_cells(text, width))
        for text, width in _CASES
        if wrap_cells(text, width) != wrap_cells_measured(text, width)
    ]
    assert not mismatches, f"{len(mismatches)} of {len(_CASES)} cases differ: {mismatches[:5]!r}"


def test_the_long_token_cases_match_the_old_implementation() -> None:
    """The inputs the lane exists for, at the sizes the lane was measured at.

    The oracle is quadratic by construction, so the sizes are graduated: the additive
    families — which is what a real oversized token is, base64 or a minified line — are
    taken to the 80,000 characters this lane was measured at, and the two cluster
    families, whose scan is the slow table walk rather than a plain sum, to 2,000. The
    cluster families are also in the corpus above up to 201 units each, so their branch
    is pinned at every width; this is about length, not about the branch.
    """
    for unit in ("x", "\u4e2d"):
        for length in (5_000, 40_000, 80_000):
            text = unit * length
            for width in (80, 200):
                expected = wrap_cells_measured(text, width)
                assert wrap_cells(text, width) == expected, (unit, length, width)
    for unit in (KEYCAP, JOINER):
        for length in (500, 2_000):
            text = unit * length
            for width in (80, 200):
                expected = wrap_cells_measured(text, width)
                assert wrap_cells(text, width) == expected, (unit, length, width)


@pytest.mark.parametrize("text", ["", " ", "   ", "a", "x" * 80, KEYCAP, "a b c", "a  b"])
def test_both_paths_agree_on_the_shapes_a_reader_would_write_down(text: str) -> None:
    """A handful of hand-written cases beside the corpus, as a readable contract."""
    for width in (1, 2, 3, 5, 10, 80):
        expected = wrap_cells_measured(text, width)
        assert wrap_cells(text, width) == expected, (text, width)


def test_no_row_exceeds_the_width_except_a_character_wider_than_it() -> None:
    """Every row fits, or is a single character the frame cannot hold at all.

    This is the function's whole contract, and it is asserted over the same corpus as
    the equivalence: a row that overhangs its frame is what the break loop exists to
    prevent, and the one-character exception is the documented outcome for a width
    below a CJK ideograph's own two cells.
    """
    overhanging = [
        (text, width, row)
        for text, width in _CASES
        for row in wrap_cells(text, width)
        if rich_cell_len(row) > width and len(row) != 1
    ]
    assert not overhanging, overhanging[:5]


def test_a_token_with_no_separator_survives_the_break_whole() -> None:
    """Concatenating the rows reproduces the token, for every space-free input.

    The break must not DROP a character, and a dropped tail is invisible to a per-row
    width assertion: an implementation that stopped one chunk early would still
    produce rows that fit. Assembling the input back is what catches it.
    """
    for text, width in _CASES:
        if " " in text:
            continue
        assert "".join(wrap_cells(text, width)) == text, (text, width)


def _measured_work(
    text: str, width: int, monkeypatch: pytest.MonkeyPatch, *, whole: bool = True
) -> tuple[int, int]:
    """Wrap ``text`` counting ``cell_len`` calls and the characters each one scans.

    ``whole`` asserts the reassembly invariant for the callers whose input has no
    separator in it: it is the cheapest proof that the wrapper is measuring the
    input the caller thinks it is.
    """
    calls: list[int] = []
    real = rich_cell_len

    def counting(candidate: str) -> int:
        calls.append(len(candidate))
        return real(candidate)

    monkeypatch.setattr(transcript, "cell_len", counting)
    rows = wrap_cells(text, width)
    if whole:
        assert "".join(rows) == text
    return len(calls), sum(calls)


def test_the_work_is_linear_in_the_token_not_quadratic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Work done, never wall time: characters scanned must not grow per character.

    The old loop's signature is that characters-measured per token character DOUBLES
    each time the token doubles (34.7 -> 66.0 -> 128.5 -> 253.5 over 5k..40k), because
    every pass re-scans the whole remainder. The assertion is on that ratio: bounded by
    a small constant at every length, and flat between the lengths. A timing assertion
    would say the same thing about this host; this one says it about the function.
    """
    ratios = {}
    for length in (2_500, 5_000, 10_000, 20_000, 40_000, 80_000):
        _, characters = _measured_work("x" * length, 80, monkeypatch)
        ratios[length] = characters / length
        assert ratios[length] <= 4, (length, ratios[length])

    # 80k is 32x the 2.5k token; a quadratic tail would put its ratio ~32x higher.
    assert ratios[80_000] <= ratios[2_500] * 1.5, ratios


def test_a_token_after_a_row_in_progress_is_not_re_measured_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other entry into the break loop: a word that arrives with a row open.

    ``current`` is non-empty here, so the word's own cell count is a second
    measurement — the fit test measured the candidate, which includes the row. The
    point is that it stays a small constant per token, not that the token is measured
    exactly once.
    """
    token = "x" * 40_000
    calls, characters = _measured_work(f"log: {token}", 80, monkeypatch, whole=False)
    assert characters <= 4 * len(token), (calls, characters)


def test_the_non_additive_path_is_still_measured_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is real, and it is the branch that keeps the rare cases honest.

    A joiner-bearing word cannot use the arithmetic — that is the entire reason the
    branch exists — so it must still pay a measurement per chunk. Asserted as work
    done, and asserted as a RELATION between two same-length tokens: if a later edit
    "optimises" the measured path into the arithmetic one, this goes red rather than
    the equivalence corpus quietly absorbing it.
    """
    plain = "x" * 4_000
    joined = JOINER * 1_333
    _, plain_characters = _measured_work(plain, 80, monkeypatch)
    _, joined_characters = _measured_work(joined, 80, monkeypatch)
    assert plain_characters <= 2 * len(plain)
    # Per character, so the two tokens need not be the same length: the joiner one is
    # re-measured once per chunk, the plain one is not measured again at all.
    assert joined_characters / len(joined) > plain_characters / len(plain), (
        plain_characters,
        joined_characters,
    )


def wrap_cells_measured(text: str, width: int) -> list[str]:
    """Verb-for-verb the implementation this lane replaced. THE ORACLE.

    Copied from ``transcript.py`` at ``baeb8089d`` and kept here on purpose: the
    production copy is gone, and an oracle that is *derived* from the new code cannot
    disagree with it. Nothing in this file asserts that this function is fast, or
    correct in isolation — only that the shipped one still agrees with it.
    """
    if width <= 0:
        return [text]
    rows: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if rich_cell_len(candidate) <= width:
            current = candidate
            continue
        if current:
            rows.append(current)
            current = ""
        while rich_cell_len(word) > width:
            head = ""
            used = 0
            for char in word:
                size = rich_cell_len(char)
                if used + size > width:
                    break
                head += char
                used += size
            while head and rich_cell_len(head) > width:
                head = head[:-1]
            if not head:
                head = word[0]
            rows.append(head)
            word = word[len(head) :]
        current = word
    if current or not rows:
        rows.append(current)
    return rows
