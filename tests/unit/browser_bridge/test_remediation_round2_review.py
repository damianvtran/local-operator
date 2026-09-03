"""Regression test for the round-2 QA remediation on PR #563 (Q5).

Round 1 stopped a redacted handle from FORKING a phantom record. Q5 is the
milder shape that survived it: a redacted handle still could not name a tab, so
a handle-less command result refreshed the most recently updated record — which
is not necessarily the tab the extension was describing. This test fails on the
round-2 head (`aebea0c5`), where `note_driven` fell straight through to recency.
"""

from __future__ import annotations

import time

from local_operator.browser_bridge.daemon import ExtensionLink


def test_redacted_handle_refreshes_its_own_record_not_the_most_recent() -> None:
    """Q5: the two sides count "most recent" on different clocks.

    `nav.ts` resolves a handle-less command against the most recently USED
    surface (bumped by every tab-scoped command), while the daemon's recency
    fallback sees the most recently UPDATED record. When those disagree — tab A
    used last, tab B updated last — a handle-less `status` describing A used to
    overwrite B's record with A's URL, so `status` and the popup misreported
    which tab held which page. A redacted token cannot KEY a record, but it
    carries enough nonce to RECOGNISE one, so it must be matched rather than
    guessed at.
    """
    link = ExtensionLink()
    link.note_driven("bridge:11:aaaaaa0123456789", "https://site-a.example/a", "A")
    # A distinct timestamp is what makes B the most recently UPDATED record;
    # without it the recency fallback is a coin flip rather than a wrong answer.
    time.sleep(0.01)
    link.note_driven("bridge:22:bbbbbb0123456789", "https://site-b.example/b", "B")

    # A is most recently USED on the extension clock, so a handle-less `status`
    # reports A — with its handle redacted on the way out.
    link.note_driven("bridge:11:aaaaaa\u2026", "https://site-a.example/a", "A")

    assert link.driven["bridge:22:bbbbbb0123456789"].url == "https://site-b.example/b"
    assert link.driven["bridge:11:aaaaaa0123456789"].url == "https://site-a.example/a"
    assert len(link.driven) == 2, "matching a redacted handle must not fork a record"


def test_unrecognised_redacted_handle_still_falls_back_to_recency() -> None:
    """The fallback stays reachable for a handle we do not track.

    Matching is an improvement on the guess, not a replacement for it: a peer
    can report a surface the daemon has no record of (it reconnected, or the
    record was repaired away). That must still refresh rather than fork, which
    is what keeps a mixed-version pair reporting one driven tab.
    """
    link = ExtensionLink()
    link.note_driven("bridge:11:aaaaaa0123456789", "https://site-a.example", "A")
    time.sleep(0.01)
    link.note_driven("bridge:22:bbbbbb0123456789", "https://site-b.example", "B")

    link.note_driven("bridge:99:zzzzzz\u2026", "https://site-z.example", "Z")

    assert len(link.driven) == 2
    assert link.driven["bridge:22:bbbbbb0123456789"].url == "https://site-z.example"


def test_handle_less_update_still_collapses_onto_the_unkeyed_record() -> None:
    """An old build sends no handle at all, and must not self-match on "".

    The unkeyed record is keyed by the empty string, so a matcher that accepted
    an empty token would prefix-match it against every key. A genuinely
    handle-less update therefore stays on the recency path, which is what makes
    three navigations from a 0.1.5 extension one record instead of three.
    """
    link = ExtensionLink()
    for url in ("https://legacy.example/1", "https://legacy.example/2"):
        link.note_driven("", url, "Legacy")

    assert list(link.driven) == [""]
    assert link.current_url == "https://legacy.example/2"
