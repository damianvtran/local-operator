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

from local_operator.code_requests import rounds
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
        # A bold label is still a label: the base it names is not the head (F17).
        ("`3ea522cc21` (**Base:** `82adb042eb`)", "3ea522cc21"),
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
        # -- review round 2: three-dot ranges, labelled companions, exact chains -------
        # Git's three-dot range names the same head on its right-hand side (F7).
        ("`1c2d3e4...5f6a7b8`", "5f6a7b8"),
        # A labelled base/merge hash is not the head, so the head is the only candidate (Q4).
        ("`main..release-next` (head `3ea522cc21`) (base `82adb042eb`)", "3ea522cc21"),
        ("`main..release-next` (merge `5f6a7b8`) (head `3ea522cc21`)", "3ea522cc21"),
        # A base that EQUALS the previous head still chains; only prefixes are restricted.
        ("`b..b1a2c3d4` + `b1a2c3d4..f0e1d2c3`", "f0e1d2c3"),
        # -- review round 3: labels whose window contains hex LETTERS ------------------
        # A cited report is not the reviewed head.
        ("`3ea522cc21` (Round-1 report: `82adb042eb`)", "3ea522cc21"),
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
        # A cycle that never returns to its own root is still a cycle (F8).
        "`aaaaaaa..bbbbbbb` + `bbbbbbb..ccccccc` + `ccccccc..bbbbbbb`",
        # A prose ellipsis with a word inside is not a range: nothing decides here.
        "`82adb042eb and 3ea522cc21`",
        # F10, the direction that matters most: a prose ellipsis must NOT become a
        # range. With the loose separator this read FRESH against a live head of
        # def5678abc — a head the review never named.
        "`abc1234` — rebased onto main... `def5678abc`",
        # A three-dot range has to be TIGHT to count as one. With room around the dots
        # it is prose, and two named hashes with no range leave nothing to decide on —
        # which is the safe answer, not the looser one (F10).
        "`82adb042eb... 3ea522cc21`",
        # Four or more dots are never a range: the three-dot arm lost its non-dot guards
        # once and these read FRESH (review round 4, F16; QA round 4, Q9).
        "`abc1234....def5678abc`",
        "`abc1234` — rebased onto main....`def5678abc`",
        "waited.....`def5678abc` and `abc1234`",
        # The accepted cost of the narrow label window (F12). ``Base `main` @ `d1fd63d` ``
        # needs a window wide enough to cross a quoted token, and that same width
        # mislabelled real heads on #2223/#2114 — a wrong head can read FRESH, a lost
        # one reads unknown, so this shape stays unknown by decision.
        "`main..release` (base `main` @ `d1fd63d`) (head `3ea522cc21`)",
        # The deliberate exclusion, pinned: ``on`` is ordinary prose, so a hash it
        # introduces stays a candidate and an ambiguous scope stays unknown (Q7).
        "`main..release` (on `82adb042eb`) and `3ea522cc21`",
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
        # Review round 2: a Head field quoting a range takes that range's head (F6), and
        # an indecisive Scope now falls through to Head instead of stopping (Q5).
        ("", "`82adb042eb..3ea522cc21`", "3ea522cc21"),
        ("`82adb042eb` and `3ea522cc21`", "f0e1d2c3b4a5", "f0e1d2c3b4a5"),
        # A range mentioned AFTER the head is a description of a delta, not the head
        # (F11): the leading hash wins.
        (
            "",
            "`fd0683986b` (merge of `02eb44ef9e` into `b011d09a0b`) … delta `04f8b3e0..02eb44ef9e`",
            "fd0683986b",
        ),
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
    """F5/Q6/Q8: the pattern must stay linear, and the guard must be able to fail.

    Two rounds of review found quadratic shapes here: a ``z`` run (11.4 s at 32 kB) and
    ``a-a-``/``a.``/``a/`` runs (18-73 s at 32-64 kB), on a parse that runs inline on the
    event loop over forge-supplied text. Round 3 then found this guard proved nothing —
    its payloads were past the 4 kB ``_MAX_FIELD`` cap, so they never reached the
    pattern, and a 4 kB payload is too small for a loose bound to notice either.

    So the sensitive check drives the PATTERN directly on 32 kB payloads, where the
    round-2 build took 18 s and this head takes ~0.2 ms: the 0.5 s bound carries ~2500x
    headroom on a correct build and is 36x short of a quadratic one. A second loop keeps
    the end-to-end field path honest under the cap, and the cap's own path is asserted
    to stay cheap. (A 32 kB run of *spaces* costs ~2.5 s and costs ``main`` the same:
    that is ``_FIELD``, outside this pattern.)
    """
    hostile = {
        "z run": "z" * 32000,
        "a-a- run": "a-a-" * 8000,
        "a. run": "a." * 16000,
        "a/ run": "a/" * 16000,
        "dotted words": "abc.def.ghi " * 2600,
        "backtick run": "`" * 32000,
        "tab run": "\t" * 32000,
        "space run": " " * 32000,
    }
    for label, payload in hostile.items():
        start = time.perf_counter()
        rounds._SHA_RANGE.findall(payload)  # the pattern itself: no field cap here
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"{label} in the pattern took {elapsed:.2f}s — quadratic?"

    # Under the cap, the whole field path — collapse, scan, collect, decide.
    for label, payload in hostile.items():
        assert len(payload) > 4096, label
        start = time.perf_counter()
        assert rounds._reviewed_head(payload, "") is None, label
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"{label} through _reviewed_head took {elapsed:.2f}s"

    body = "### Agent review — round 1\n\nReviewer: r\nScope: `" + "a-a-" * 1000 + "`\n"
    start = time.perf_counter()
    parse([Comment(id="1", created_at="2026-10-14T00:00:00Z", body=body)])
    assert time.perf_counter() - start < 1.0, "the end-to-end parse path regressed"


@pytest.mark.parametrize(
    "left,right,expected",
    [
        # One commit at two prefix lengths is the same commit.
        ("ec9a5e628f4", "ec9a5e628f4d0e1", True),
        ("main", "main", True),
        # Prefix agreement is for hex on BOTH sides only: a branch named ``b`` is not a
        # head starting with ``b`` (F9), and two different words never agree.
        ("b", "b1a2c3d4", False),
        ("main", "mainline", False),
    ],
)
def test_prefix_agreement_is_hex_only(left: str, right: str, expected: bool) -> None:
    # Reached through the module, not a direct import: the pre-fix module has no such
    # helper, and a direct import would turn the failing-first run into a collection
    # error instead of the failures it is meant to show.
    assert rounds._same_sha(left, right) is expected
