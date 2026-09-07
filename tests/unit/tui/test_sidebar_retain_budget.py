"""The presentation cache must actually cache, and a refusal must not loop.

Two failures, one visible symptom. The operator reported "lag around the UI in
general with the sidebar open, when I close the sidebar it's fine", plus session
switches that stayed slow for long conversations. Both came from
``SessionPresentation.retainable()`` refusing presentations it should admit:

* the string term charged ``len(value) * 4`` (a worst-case UTF-32 estimate)
  against a 1 MiB budget, so the effective budget was 256 KiB while
  ``DISPLAY_HISTORY_BYTES`` lets the window layer hand it 512 KiB. Measured on
  the operator's real transcripts, 4 of 8 were refused and 2 of those were
  refused *purely* on the over-charge;
* a refused presentation is never inserted into ``_sidebar_presentations``, so
  it never stops matching the prewarm candidate filter. Every 2 s poll
  re-selected it and paid a full prepare — connect, window, replay, MOUNT,
  layout wait, teardown — on the event loop, for a result guaranteed to be
  discarded.

The assertions here are STRUCTURAL — retained/refused booleans and prepare call
counts — not wall-clock bounds, per AGENTS.md §"Timing, flakes": the defect is
"how many times did this run", which cannot flake, and this repo has abandoned
numeric timing bounds as unportable.

Each test asserts its PRECONDITION before its claim. A previous PR in this
series shipped three vacuous tests, one of which passed *with* the defect
present, so "the fixture is genuinely near the boundary" is checked rather than
assumed.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from local_operator.tui.session_presentation import (
    RETAIN_TEXT_BYTES,
    PreparedReplay,
    SessionPresentation,
)
from local_operator.tui.widgets.transcript import NoticeBlock


@pytest.fixture(autouse=True)
def isolated_retain(tmp_path, monkeypatch):
    # Headless tests must never touch the operator's real config, cache, or
    # multiplexer workspace. HOME as well as the config dir: the config var
    # alone does not redirect the catalogue cache.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")


def _presentation(text: str) -> SessionPresentation:
    """A presentation holding exactly ``text`` in one retainable block.

    ``NoticeBlock`` is used because it retains through the base
    ``retained_payloads`` -> ``text()`` path and stores the string verbatim, so
    the walk reaches exactly the bytes under test and nothing else is being
    measured. (``AssistantBlock`` would re-render megabytes of Markdown per
    fixture to hold the same string.)
    """
    replay = PreparedReplay()
    block = NoticeBlock("payload", "note")
    # The payload is installed AFTER construction on purpose. `retainable()`
    # walks `retained_payloads()` -> `text()` -> `_text`, which is exactly what
    # a real block exposes; constructing with the full string instead would
    # make every fixture render a megabyte of Rich content the predicate never
    # looks at, for no additional coverage.
    block._text = text
    replay.blocks.append(block)
    replay.view._blocks.append(block)
    return SessionPresentation(replay)


def test_a_window_sized_payload_is_retainable():
    """The retain budget must admit what the window layer is allowed to build.

    ``DISPLAY_HISTORY_BYTES`` is 512 KiB and this budget is 1 MiB, so a
    half-megabyte of ASCII prose — the largest window the layer above can
    legitimately hand over — has to fit with room to spare. Under the
    ``len * 4`` over-charge it did not: 512 KiB was charged as 2 MiB against a
    1 MiB budget and refused.
    """
    from local_operator.session.history_window import DISPLAY_HISTORY_BYTES

    text = "x" * DISPLAY_HISTORY_BYTES
    presentation = _presentation(text)

    # PRECONDITION: this fixture is genuinely the size being claimed, and the
    # budget really is the one the window layer must fit inside. Without these
    # the assertion below would pass on an empty presentation.
    assert len(text) == 512 * 1024
    assert RETAIN_TEXT_BYTES >= DISPLAY_HISTORY_BYTES
    # PRECONDITION: the string is ASCII, so its resident cost is ~1 byte per
    # character and the old estimate over-charged it by exactly 4x. If this
    # ever fails the test is measuring a different thing than it claims.
    assert sys.getsizeof(text) < len(text) * 1.01

    assert presentation.retainable(), "a legal display window must be cacheable"


def test_the_budget_still_refuses_a_payload_above_it():
    """Removing the protection is not the fix; the bound must still bite.

    One cached view must not be able to pin an arbitrary slice of a 200 MB
    journal, and this host has hit macOS "out of application memory". A
    presentation whose resident text exceeds the budget is still refused.
    """
    text = "x" * (RETAIN_TEXT_BYTES + 64 * 1024)
    presentation = _presentation(text)

    # PRECONDITION: over the budget by resident bytes, which is the unit the
    # predicate now charges in — not by a character count that happens to agree.
    assert sys.getsizeof(text) > RETAIN_TEXT_BYTES

    assert not presentation.retainable(), "the memory bound must still refuse"


def test_the_cost_estimate_is_resident_bytes_not_a_character_multiplier():
    """Charge what the string actually costs, whatever its widest code point.

    CPython stores ``str`` in PEP 393 compact form (1, 2 or 4 bytes per
    character), so no constant multiplier is correct for all content. The guard
    that matters in both directions: ASCII at 60% of budget must be admitted,
    and the SAME character count in astral plane — which really does cost ~4x —
    must be refused.
    """
    count = int(RETAIN_TEXT_BYTES * 0.6)
    ascii_text = "x" * count
    astral_text = "\U0001f642" * count

    # PRECONDITION: identical character counts, different resident costs. This
    # is the whole point — a `len()`-based charge cannot tell these apart, and
    # a `len * 4` charge cannot tell them apart either.
    assert len(ascii_text) == len(astral_text) == count
    assert sys.getsizeof(ascii_text) < RETAIN_TEXT_BYTES
    assert sys.getsizeof(astral_text) > RETAIN_TEXT_BYTES

    assert _presentation(ascii_text).retainable(), "ASCII under budget must be admitted"
    assert not _presentation(astral_text).retainable(), "4-byte chars must be charged 4 bytes"


class _Source(SimpleNamespace):
    """The two attributes the admission path reads off a ``SessionInteraction``."""

    def __init__(self, session_id: str, revision: int = 0) -> None:
        super().__init__(session_id=session_id, presentation_revision=revision)


def _admission_host():
    """An object carrying only the admission state, bound to the real methods.

    The helpers under test read `_sidebar_unretainable` and `_sidebar_sources`
    and nothing else, so binding them to a bare namespace exercises the
    production code without booting a full app — and keeps the assertion about
    admission rather than about Textual.
    """
    from local_operator.tui.app import OperatorApp

    host = SimpleNamespace(_sidebar_unretainable={}, _sidebar_sources={})
    host._admit_sidebar_presentation = OperatorApp._admit_sidebar_presentation.__get__(host)
    host._sidebar_prewarm_refused = OperatorApp._sidebar_prewarm_refused.__get__(host)
    return host


def test_prewarm_stops_re_preparing_a_refused_session():
    """A refusal must be remembered, or prewarm loops on it forever.

    This is the load-bearing half. Some session will exceed any finite budget
    (the reporting operator holds a 218 MB journal), so a correct budget alone
    does not stop the loop — the next oversized conversation reproduces it.

    The assertion is a COUNT of admission attempts across simulated polls, not
    a duration.
    """
    host = _admission_host()
    source = _Source("huge-session")
    host._sidebar_sources["huge-session"] = source
    oversized = _presentation("x" * (RETAIN_TEXT_BYTES + 64 * 1024))

    # PRECONDITION: this presentation really is refused, so the loop under test
    # is the one being described.
    assert not oversized.retainable()

    attempts = 0
    for _ in range(10):  # ten 2-second sidebar polls
        if host._sidebar_prewarm_refused("huge-session"):
            continue
        attempts += 1
        host._admit_sidebar_presentation("huge-session", source, oversized)

    assert attempts == 1, f"prewarm re-prepared a refused session {attempts} times over 10 polls"


def test_a_changed_session_is_reconsidered_exactly_once_more():
    """A refusal is evidence about content, and content changes.

    Keyed on ``presentation_revision`` — the same staleness stamp the cache-hit
    path uses — so a session that has moved on since it was refused gets one
    fresh attempt instead of a permanent blacklist. The over-fix guard is that
    it is ONE attempt, not a re-opened loop.
    """
    host = _admission_host()
    source = _Source("grown-session", revision=1)
    host._sidebar_sources["grown-session"] = source
    oversized = _presentation("x" * (RETAIN_TEXT_BYTES + 64 * 1024))

    assert not host._admit_sidebar_presentation("grown-session", source, oversized)
    # PRECONDITION: it is currently suppressed, so the next assertion is about
    # the revision bump and not about a set that was never populated.
    assert host._sidebar_prewarm_refused("grown-session")

    source.presentation_revision = 2
    assert not host._sidebar_prewarm_refused(
        "grown-session"
    ), "a changed session must be reconsidered"

    # And re-refusing at the new revision suppresses it again — one retry per
    # change, not an unbounded retry loop.
    host._admit_sidebar_presentation("grown-session", source, oversized)
    assert host._sidebar_prewarm_refused("grown-session")


def test_admission_clears_a_stale_refusal():
    """A session that fits now must not carry a refusal that suppresses it.

    Otherwise a conversation that shrank below the budget (compaction, a
    cleared transcript) would be admitted to the cache while still being
    skipped by prewarm — a split state that is hard to see and easy to keep.
    """
    host = _admission_host()
    source = _Source("shrinking-session")
    host._sidebar_sources["shrinking-session"] = source

    assert not host._admit_sidebar_presentation(
        "shrinking-session", source, _presentation("x" * (RETAIN_TEXT_BYTES + 64 * 1024))
    )
    # PRECONDITION: the refusal was actually recorded.
    assert "shrinking-session" in host._sidebar_unretainable

    assert host._admit_sidebar_presentation("shrinking-session", source, _presentation("small"))
    assert "shrinking-session" not in host._sidebar_unretainable
    assert not host._sidebar_prewarm_refused("shrinking-session")
