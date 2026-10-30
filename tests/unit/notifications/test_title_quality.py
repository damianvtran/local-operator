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

from local_operator.notifications.title_quality import title_defect

#: Titles that must PASS. Written as the operator would see them.
ACCEPTED = [
    "PR #3116 merged — team bridge receiver",
    "CI green",
    "PR #9",
    "Release v0.92.2 shipped",
    "3 fraud attempts blocked",
    "修复完成",
    "Ошибка деплоя",
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
    floor = title_defect("x") or ""
    empty = title_defect("   ") or ""
    assert "placeholder" in placeholder.lower()
    assert "noop" in placeholder
    assert "letter" in floor.lower() or "digit" in floor.lower() or "character" in floor.lower()
    assert "nothing" in empty.lower() or "empty" in empty.lower() or "punctuation" in empty.lower()
    assert len({placeholder, floor, empty}) == 3, "each defect arm says something different"
