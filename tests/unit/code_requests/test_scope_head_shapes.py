"""``Scope`` range shapes: the reviewed head is the RANGE HEAD, never the base.

Failing-first against the frozen ``rounds.py`` (blob ``75779ccda9e5`` at commit
``f4a2b027f4cc``). Rows marked ``RED`` assert the head a review actually closed on
and are answered differently today; the rest pin behaviour that must not move.

The provenance of the headline case is the verbatim ``Scope`` line of
``damianvtran/local-operator#2345`` comment ``6152890889`` — a round whose head was
the range END, rendered as ``stale — reviewed 82adb04`` because the range never
matched and a first-hash fallback returned the BASE.

The ``never fresh`` table is the other half of the contract, and the reason the fix
is not "take the last range": an ambiguous or malformed scope must degrade to
``unknown``. A head that is wrong but happens to sit under the live head reports a
stale round as FRESH, which is the failure this field exists to prevent.

The remediation round adds the reviewers' own cases: the ``Head:`` field's
``"X" (was "Y")`` spelling (F1), one commit written at two prefix lengths (F2), a fold
chain written in the other order (F3), the symbolic-range-plus-parenthesised-head
spelling the fleet's release PRs use (Q2 recall), and a bounded-time guard for the
long-Scope-line stall (Q1).
"""

from __future__ import annotations

import time

import pytest

from local_operator.code_requests.rounds import (
    FRESHNESS_FRESH,
    FRESHNESS_UNKNOWN,
    Comment,
    _reviewed_head,
    freshness,
    parse,
)

#: #2345's live head, abbreviated to its first 16 characters for the fixture.
LIVE_HEAD = "3ea522cc21f2a4d5"

#: The real ``Scope`` value: a symbolic range, then the explicit one in parentheses.
PARENTHESIZED = "`main..release-next-0-70-20` (`82adb042eb`..`3ea522cc21`)"

_HEAD_40 = "0123456789abcdef0123456789abcdef01234567"
_TAIL_40 = "89abcdef0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize(
    "scope,expected",
    [
        # -- must not regress (green on the frozen module) -----------------------------
        ("`2d47e1fd30..8dc2b9797f`", "8dc2b9797f"),  # whole-range backticks
        ("`main..72bea95`", "72bea95"),  # symbolic base, hex head
        ("`abc1234..def5678` + `abc1234..def5678`", "def5678"),  # duplicate range
        # The repo's own assertions on this field (tests/unit/code_requests/test_rounds.py):
        # pinned here so the no-regression claim is exercised rather than assumed.
        ("`46b12d2ff9..039476dff3`", "039476dff3"),
        ("`aaaaaaa..bbbbbbb`", "bbbbbbb"),
        # -- RED: the per-token backtick spelling -------------------------------------
        (PARENTHESIZED, "3ea522cc21"),
        ("(`82adb042eb`..`3ea522cc21`)", "3ea522cc21"),
        ("`a1b2c3d`..`e4f5a6b`", "e4f5a6b"),
        (f"`{_HEAD_40}`..`{_TAIL_40}`", _TAIL_40),
        # -- RED: several heads, decided by the chain rule ----------------------------
        ("`04f8b3e080..02eb44ef9e` + fold `02eb44ef9e..fd0683986b`", "fd0683986b"),
        ("`a1b2c3d..0123456789ab` + `0123456789ab..f0e1d2c3b4a5`", "f0e1d2c3b4a5"),
        # -- mixed prefix lengths ------------------------------------------------
        ("`a1b2c3d..0123456789ab`", "0123456789ab"),
        # -- the reviewers' recall cases (F2, F3, Q2) ---------------------------------
        # The same commit written at two prefix lengths is one head, not two.
        ("`aaaaaaa..ec9a5e628f4` and `bbbbbbb..ec9a5e628f4d0e1`", "ec9a5e628f4d0e1"),
        # A fold chain written in the other order still ends at the final head.
        ("`02eb44ef9e..fd0683986b` + fold `04f8b3e080..02eb44ef9e`", "fd0683986b"),
        # A symbolic range whose head is named beside it: the fleet's release spelling.
        ("`main..release-next-0-70-20` (head `3ea522cc21`)", "3ea522cc21"),
        # One bare hash and no hex range: the only head the comment names.
        ("`..def5678`", "def5678"),
        ("`main... 3ea522cc21`", "3ea522cc21"),
    ],
)
def test_the_reviewed_head_is_a_ranges_head(scope: str, expected: str) -> None:
    assert _reviewed_head(scope, "") == expected


@pytest.mark.parametrize(
    "scope",
    [
        # Two ranges that neither agree nor chain: which head was reviewed is unstated.
        "`aaaaaaaa..bbbbbbbb` + `cccccccc..dddddddd`",
        # Range syntax with nothing well-formed in it.
        "`abc..`",
        "`abcdef1..` + `def5678`",
        # A range that returns to its own start is a cycle, not a chain.
        "`a1b2c3d..b2c3d4e` + `b2c3d4e..a1b2c3d`",
        # A prose ellipsis is not a range, so nothing decides between two hashes.
        "`82adb042eb... 3ea522cc21`",
        # No range at all, and two bare hashes to choose from.
        "`82adb042eb` and `3ea522cc21`",
        # A symbolic range names no hash; freshness is unknown, never stale.
        "main..release-next-0-70-20",
        "",
    ],
)
def test_an_ambiguous_scope_is_unknown_and_never_fresh(scope: str) -> None:
    reviewed = _reviewed_head(scope, "")
    assert reviewed is None, f"{scope!r} produced a guessed head {reviewed!r}"
    assert freshness(reviewed, LIVE_HEAD) == FRESHNESS_UNKNOWN


@pytest.mark.parametrize(
    "scope,head_field,expected",
    [
        # The Head field is the last resort, and still resolves a bare hash.
        ("main..release-next-0-70-20", "3ea522cc21f2", "3ea522cc21f2"),
        ("", "3ea522cc21f2", "3ea522cc21f2"),
        # A malformed scope falls through to Head rather than mining the scope line.
        ("`abc..`", "3ea522cc21f2", "3ea522cc21f2"),
        # One bare hash in Scope needs no Head field at all.
        ("`82adb042eb`", "", "82adb042eb"),
        # The Head field's standard spelling: the head first, the earlier one after it.
        ("", "`3ea522cc21` (was `82adb042eb`)", "3ea522cc21"),
        ("", "`3ea522cc21f2` (one commit on top of `82adb042eb`)", "3ea522cc21f2"),
    ],
)
def test_the_head_field_still_resolves(scope: str, head_field: str, expected: str) -> None:
    assert _reviewed_head(scope, head_field) == expected


_REAL_2345 = (
    "### Agent review — round 1\n\n"
    "**Reviewer:** reviewer (Local Operator subagent) — model `radient/auto`\n"
    "**Scope:** `main..release-next-0-70-20` (`82adb042eb`..`3ea522cc21`)\n\n"
    "#### 1. Diff is exactly one line in one file — PASS\n"
)


def test_the_real_2345_review_reads_fresh() -> None:
    """RED on the frozen module: the round is reported against the range BASE.

    This is the observed defect — the shipped tool rendered
    ``terminal, stale — reviewed 82adb04, head 3ea522cc21`` for a round that closed
    on the head the PR still had.
    """
    (item,) = parse(
        [Comment(id="6152890889", created_at="2026-10-13T22:00:00Z", body=_REAL_2345)]
    ).passes
    assert item.reviewed_head == "3ea522cc21", "the base must never be the reviewed head"
    assert freshness(item.reviewed_head, LIVE_HEAD) == FRESHNESS_FRESH


def test_a_long_scope_line_parses_in_bounded_time() -> None:
    """Q1: without the leading ``\\b`` the range pattern backtracked combinatorially.

    A forge commenter could put a 32 kB ``Scope:`` line on a tracked PR and stall the
    event loop for 11.4 s (measured); with the anchor the same line takes milliseconds.
    The bound is deliberately loose — it flags the quadratic shape, not a busy host.
    """
    body = "### Agent review — round 1\n\nReviewer: r\nScope: `" + "a1b2c3d.." + "z" * 32000 + "`\n"
    start = time.perf_counter()
    parse([Comment(id="1", created_at="2026-10-14T00:00:00Z", body=body)])
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"a 32 kB Scope line took {elapsed:.1f}s"
