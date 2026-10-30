"""Title quality for the ``notify`` tool (docs/design/notifications-inbox.md §3.1).

The inbox list shows the title ALONE, so a title that identifies no event is a
row the operator must read and dismiss to learn nothing. This module holds the
boundary: what counts as degenerate, and — just as important — what a short but
real title must still be allowed to say.

Both sides of the table matter. Over-refusing a legitimate title is a
regression: it blocks a real notification and pushes the model towards saying
nothing at all.
"""

from __future__ import annotations

import pytest

from local_operator.notifications.title_quality import _PLACEHOLDER_TITLES, title_defect

#: Titles that must PASS. Written as the operator would see them.
ACCEPTED = [
    "PR #3116 merged — team bridge receiver",
    "CI green",
    "PR #9",
    "Release v0.92.2 shipped",
    "3 fraud attempts blocked",
    "修复完成",
    "Ошибка деплоя",
    # A character is a word or a morpheme in these scripts, so two of them is a
    # title: the guard counts meaning, not characters (docs §3.1).
    "部署",
    "失败",
    "完了",
    "완료",
    "تم",
    # A compatibility form is the same word to the reader, so NFKC folds it and
    # a full-width title is judged as the ASCII title it renders as.
    "Ｒｅｌｅａｓｅ ｖ０.９２.２",
    # Same words, real information: whole-title equality only, never a prefix
    # or substring match on the placeholder vocabulary.
    "Update: 3 PRs merged",
    "Tests failing on shard 3",
    "WIP: migration half applied, do not deploy",
    "Done with the 14 table migration",
    "ok, but the deploy is still red",
]

#: Titles that must be REFUSED, each for the reported reason.
REFUSED = [
    "x",
    "y",
    "z",
    "noop",
    "no-op",
    "n/a",
    "test",
    "todo",
    "tbd",
    "update",
    "unused",
    "placeholder",
    "wip",
    "done",
    "ok",
    # Full-width forms are the same words, so NFKC must fold them into the
    # vocabulary; unfolded they are unmatched "letters" that write a junk row.
    "ＮＯＯＰ",
    "ＴＥＳＴ",
    "ＮＯ－ＯＰ",
    "Ｎ／Ａ",
    # Full-width characters count as the Latin letters they render as.
    "ｘｘ",
    # Below the floor in every script: one character is a word, not a title.
    "修",
    "完",
    "",
    "   ",
    "...",
    "🚀",
]


@pytest.mark.parametrize("title", ACCEPTED)
def test_a_title_that_identifies_an_event_is_accepted(title: str) -> None:
    assert title_defect(title) is None, f"{title!r} carries real information"


@pytest.mark.parametrize("title", REFUSED)
def test_a_title_that_identifies_nothing_is_refused(title: str) -> None:
    defect = title_defect(title)
    assert defect is not None, f"{title!r} identifies no event"
    assert isinstance(defect, str) and defect


def test_the_placeholder_check_ignores_case_and_separators() -> None:
    """``No-Op`` and ``N A`` are the same null row as ``noop``."""
    assert title_defect("NO-OP") is not None
    assert title_defect("  No Op  ") is not None
    assert title_defect("n/a") is not None
    assert title_defect("N.A.") is not None


def test_the_information_floor_counts_alphanumerics_only() -> None:
    """Two letters is a placeholder; punctuation and emoji add no information."""
    assert title_defect("ab") is not None
    assert title_defect("--") is not None
    assert title_defect("!!!") is not None
    # One alphanumeric character is still one character, however it is dressed.
    assert title_defect("x --- y") is not None
    # Three is the floor, and a real three-character title is admitted.
    assert title_defect("PR9") is None


def test_the_floor_is_measured_in_information_not_characters() -> None:
    """One character carries a word in some scripts, so two of them is a title.

    ``部署``/``失败``/``完了``/``완료``/``تم`` are complete statements of two
    characters; holding them to three refused exactly the short-but-real titles
    §3.1 says never to judge. The Latin floor does not move, and one character
    of any script is still refused.
    """
    assert title_defect("部署") is None
    assert title_defect("失败") is None
    assert title_defect("完了") is None
    assert title_defect("완료") is None
    assert title_defect("تم") is None
    assert title_defect("ab") is not None
    assert title_defect("ok") is not None
    assert title_defect("修") is not None
    assert title_defect("完") is not None


def test_only_a_letter_earns_the_dense_floor() -> None:
    """A non-ASCII letter carries a word; a non-ASCII digit carries a digit.

    ``١٢`` is "12" in Arabic-Indic digits and says exactly what ``12`` says, so
    it must be refused at the same floor — the dense floor is granted for a
    script that packs a word into a character, never for a numeral in that
    script. ``１２`` is the same case one fold away: NFKC turns it into ``12``.
    """
    # A named local, not the call inline: ``title_defect`` returns ``str | None``,
    # and pyright refuses ``in`` against an Optional (the CI type-check caught it).
    arabic_two = title_defect("١٢")
    assert arabic_two is not None
    assert "at least 3 are needed" in arabic_two
    arabic_two_other = title_defect("٣٤")
    assert arabic_two_other is not None
    assert "at least 3 are needed" in arabic_two_other
    assert title_defect("１２") is not None
    # Three digits are admitted in either spelling: the floor is a count, not a
    # script ban, and the ASCII arm was never in question.
    assert title_defect("123") is None
    assert title_defect("١٢٣") is None
    # The letters the dense floor exists for are untouched by that change.
    assert title_defect("部署") is None
    assert title_defect("完了") is None
    assert title_defect("تم") is None
    assert title_defect("修") is not None


def test_a_compatibility_form_is_the_same_word() -> None:
    """NFKC before the vocabulary, or the guard is an IME keystroke from bypassed."""
    assert title_defect("ＮＯＯＰ") is not None
    assert title_defect("ＴＥＳＴ") is not None
    assert title_defect("Ｎ／Ａ") is not None
    # The fold is a compatibility one, not a rewrite: a real full-width title stays.
    assert title_defect("Ｒｅｌｅａｓｅ ｖ０.９２.２") is None


def test_a_substantial_body_cannot_rescue_a_degenerate_title() -> None:
    """The body is not on the list row, so it is not what the operator sees."""
    assert (
        title_defect("x", body="Migrated 14 tables in production, backfilled 2400 rows.")
        is not None
    )


def test_a_whitespace_body_does_not_change_a_good_title() -> None:
    assert title_defect("CI green", body="") is None


def test_the_reason_names_the_defect_for_the_model_to_correct() -> None:
    """A refusal the model cannot act on is a dead end, so each arm says which."""
    placeholder = title_defect("noop") or ""
    floor = title_defect("ab") or ""
    empty = title_defect("   ") or ""
    assert "placeholder" in placeholder.lower()
    assert "noop" in placeholder
    assert "letter" in floor.lower() or "digit" in floor.lower() or "character" in floor.lower()
    assert "nothing" in empty.lower() or "empty" in empty.lower() or "punctuation" in empty.lower()
    assert len({placeholder, floor, empty}) == 3, "each defect arm says something different"


def test_the_floor_reason_pluralises_both_nouns() -> None:
    """The sentence is the correction instruction, so it has to read."""
    assert "its 2 letters or digits cannot" in (title_defect("ab") or "")
    assert "its 1 letter or digit cannot" in (title_defect("修") or "")


def test_every_placeholder_entry_is_refused_as_a_placeholder() -> None:
    """The whole-title match is the precise diagnosis, so it is checked first.

    Five entries (``x``, ``y``, ``z``, ``ok``, ``n/a``) are also below the floor;
    with the floor first they would be unreachable, which read as dead vocabulary
    and would silently reactivate if the floor ever moved. Every member must
    therefore report itself as the placeholder it is.
    """
    assert _PLACEHOLDER_TITLES, "the vocabulary is not empty"
    for title in sorted(_PLACEHOLDER_TITLES):
        defect = title_defect(title)
        assert defect is not None, f"{title!r} is in the placeholder set"
        assert "placeholder" in defect, f"{title!r} was refused as {defect!r}"
