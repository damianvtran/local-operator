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


def _presentation_of(texts: list[str]) -> SessionPresentation:
    """The same fixture over MANY blocks, one payload each.

    Separate from :func:`_presentation` because block COUNT is the variable in
    the address-reuse case: one payload tuple is built and freed per block, so
    the hazard only appears once several of them are allocated in sequence.
    """
    replay = PreparedReplay()
    for text in texts:
        block = NoticeBlock("payload", "note")
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
    # The bound must be ABSOLUTE, not merely self-consistent. Every fixture
    # below is derived from `RETAIN_TEXT_BYTES`, so they all scale with it and
    # stay green if the constant is raised 100x — which is exactly the
    # mis-tuning this file exists to catch, and it has already happened twice
    # in the same direction. `RETAINED_PRESENTATIONS` (4) parked views each get
    # this budget, so the retained-text ceiling is 4x it; on a host that has
    # hit macOS "out of application memory", 4 MiB of retained text is the most
    # that can be justified without a fresh RSS measurement.
    assert RETAIN_TEXT_BYTES <= 4 * 1024 * 1024, (
        "the retain budget grew without a measurement: N x this bounds retained "
        "text, and this host has hit 'out of application memory'"
    )

    text = "x" * (RETAIN_TEXT_BYTES + 64 * 1024)
    presentation = _presentation(text)

    # PRECONDITION: over the budget by resident bytes, which is the unit the
    # predicate now charges in — not by a character count that happens to agree.
    assert sys.getsizeof(text) > RETAIN_TEXT_BYTES

    assert not presentation.retainable(), "the memory bound must still refuse"


def test_a_shared_string_is_charged_once_because_it_is_one_copy_of_ram():
    """The charge is per OBJECT, so aliasing must not multiply it.

    ``getsizeof`` is honest about one string; charging it above the identity
    check made it dishonest about N references to that string, so a transcript
    with repeated identical tool output was refused at N x its true cost. That
    is the OVER-charging direction — the one that produced both previous
    mis-tunings and the re-preparation loop.

    A ``Mapping`` root is used rather than N blocks on purpose: each block
    builds a fresh payload tuple, and freeing those is the separate address-
    reuse hazard the walk's keepalive covers. One live container isolates
    aliasing as the single variable.
    """
    shared = "y" * 300_000
    presentation = SessionPresentation(PreparedReplay())
    presentation.tool_cards = {f"call-{index}": shared for index in range(5)}

    # PRECONDITIONS: genuinely ONE object, whose single copy fits and whose
    # five-fold charge does not. Without these the assertion could pass on a
    # fixture that was simply small.
    assert len({id(value) for value in presentation.tool_cards.values()}) == 1
    assert sys.getsizeof(shared) < RETAIN_TEXT_BYTES
    assert 5 * sys.getsizeof(shared) > RETAIN_TEXT_BYTES

    assert presentation.retainable(), "one string aliased N times is one copy of RAM"


def test_distinct_blocks_are_all_charged_despite_recycled_payload_addresses():
    """``seen`` holds addresses, which only identify objects that are ALIVE.

    ``retained_payloads()`` returns a FRESH tuple per call. The walk pops it,
    charges it and drops it — so CPython frees it and hands the same address to
    the next block's tuple, which ``seen`` then skips as already-visited,
    silently un-charging that block's text. Measured before the keepalive: 62
    of 64 payload tuples were skipped and 0.13 MiB was charged against 4.00 MiB
    actually held, and the presentation was admitted.

    This is the UNDER-charging direction. It is the one that breaks the PR's
    headline claim that memory is bounded, so it is asserted separately from
    the aliasing case above rather than being folded into it.
    """
    texts = [chr(97 + index % 26) * 65536 for index in range(64)]
    presentation = _presentation_of(texts)

    # PRECONDITION: the strings really are distinct objects, so a correct walk
    # has to charge all 64. Proven, not assumed — equal-valued literals would
    # be interned and legitimately charged once.
    assert len({id(text) for text in texts}) == 64
    # PRECONDITION: genuinely over budget when every block is charged, and
    # genuinely under it if only a couple are. The gap between these is what
    # the defect lived in.
    assert sum(sys.getsizeof(text) for text in texts) > RETAIN_TEXT_BYTES
    assert 2 * sys.getsizeof(texts[0]) < RETAIN_TEXT_BYTES

    assert not presentation.retainable(), (
        "64 distinct blocks holding 4 MiB were admitted against a 1 MiB budget: "
        "freed payload tuples recycled their addresses into `seen`"
    )


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
    """What the admission path reads off a ``SessionInteraction``.

    ``presentation_revision`` counts EVENTS and ``session`` carries the two
    CONTENT stamps. Both are present because the distinction is the point: an
    earlier revision of this fix keyed refusals on `presentation_revision`, and
    a fixture that never moved it hid the resulting loop entirely.
    """

    def __init__(
        self,
        session_id: str,
        revision: int = 0,
        *,
        replay_revision: int = 0,
        history_size: int = 100,
    ) -> None:
        super().__init__(
            session_id=session_id,
            presentation_revision=revision,
            session=SimpleNamespace(
                display_history_revision=replay_revision,
                history_message_count=history_size,
            ),
        )

    def stream_events(self, count: int = 37) -> None:
        """What a background turn does: bump the EVENT counter, nothing else.

        `_on_message` bumps `presentation_revision` on any ``SessionEvent`` on
        a hidden session, so a streaming turn moves it dozens of times per poll
        while the retained cost is unchanged.
        """
        self.presentation_revision += count

    def append_row(self) -> None:
        """A durable row lands: the conversation GREW, so a refusal still holds."""
        self.presentation_revision += 1
        self.session.history_message_count += 1

    def rewrite_history(self) -> None:
        """Compaction/prune/recovery: the only change that can REVERSE a refusal."""
        self.presentation_revision += 1
        self.session.display_history_revision += 1


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
    # A staticmethod, like `_sidebar_source_stamp`: taken UNBOUND so the host
    # calls it exactly as the production methods above do.
    host._sidebar_refusal_stamp = OperatorApp._sidebar_refusal_stamp
    return host


def test_prewarm_stops_re_preparing_a_refused_session():
    """A refusal must be remembered, or prewarm loops on it forever.

    This is the load-bearing half. Some session will exceed any finite budget
    (the reporting operator holds a 218 MB journal), so a correct budget alone
    does not stop the loop — the next oversized conversation reproduces it.

    The assertion is a COUNT of admission attempts across simulated polls, not
    a duration.

    The BUSY case is the one that matters and the one an earlier revision of
    this test missed. Its source never emitted an event, so it only ever
    exercised an idle session — while the refusal was keyed on
    `presentation_revision`, which every streamed event bumps. A refused
    session that was also streaming therefore re-prepared on all ten polls with
    this test green: 8-12 concurrent sessions with turns running is the
    reporting operator's normal state, so the vacuous half was precisely the
    population the change exists for.
    """
    oversized = _presentation("x" * (RETAIN_TEXT_BYTES + 64 * 1024))
    # PRECONDITION: this presentation really is refused, so the loop under test
    # is the one being described.
    assert not oversized.retainable()

    def polls(session_id: str, between_polls) -> int:
        host = _admission_host()
        source = _Source(session_id)
        host._sidebar_sources[session_id] = source
        attempts = 0
        for _ in range(10):  # ten 2-second sidebar polls
            between_polls(source)
            if host._sidebar_prewarm_refused(session_id):
                continue
            attempts += 1
            host._admit_sidebar_presentation(session_id, source, oversized)
        return attempts

    idle = polls("idle-session", lambda _source: None)
    busy = polls("busy-session", lambda source: source.stream_events())
    growing = polls("growing-session", lambda source: source.append_row())

    assert idle == 1, f"prewarm re-prepared an idle refused session {idle} times over 10 polls"
    assert busy == 1, (
        f"prewarm re-prepared a BUSY refused session {busy} times over 10 polls: "
        "the refusal expires on streamed events rather than on content change"
    )
    # Growth cannot reverse a refusal — every refusal cause is monotone in
    # content — so appending rows must not reopen the attempt either.
    assert (
        growing == 1
    ), f"prewarm re-prepared a GROWING refused session {growing} times over 10 polls"


def test_a_rewritten_session_is_reconsidered_exactly_once_more():
    """A refusal is evidence about content, and content can be REWRITTEN.

    Compaction, prune and recovery bump ``display_history_revision`` and are
    the changes that can make an over-budget window smaller — so they are what
    reopens the attempt. The over-fix guard is that it is ONE attempt, not a
    re-opened loop.
    """
    host = _admission_host()
    source = _Source("compacted-session", replay_revision=1)
    host._sidebar_sources["compacted-session"] = source
    oversized = _presentation("x" * (RETAIN_TEXT_BYTES + 64 * 1024))

    assert not host._admit_sidebar_presentation("compacted-session", source, oversized)
    # PRECONDITION: it is currently suppressed, so the next assertion is about
    # the rewrite and not about a set that was never populated.
    assert host._sidebar_prewarm_refused("compacted-session")

    source.rewrite_history()
    assert not host._sidebar_prewarm_refused(
        "compacted-session"
    ), "a rewritten session must be reconsidered"

    # And re-refusing at the new stamp suppresses it again — one retry per
    # rewrite, not an unbounded retry loop.
    host._admit_sidebar_presentation("compacted-session", source, oversized)
    assert host._sidebar_prewarm_refused("compacted-session")


def test_a_shrunken_session_is_reconsidered():
    """Fewer durable rows can genuinely fit where more did not.

    The counterpart to the growth case: `history_message_count` falling is a
    content change that can reverse a refusal, so it must reopen the attempt
    even though `display_history_revision` is unchanged.
    """
    host = _admission_host()
    source = _Source("shrunken-session", history_size=100)
    host._sidebar_sources["shrunken-session"] = source
    oversized = _presentation("x" * (RETAIN_TEXT_BYTES + 64 * 1024))

    assert not host._admit_sidebar_presentation("shrunken-session", source, oversized)
    assert host._sidebar_prewarm_refused("shrunken-session")

    source.session.history_message_count = 40
    assert not host._sidebar_prewarm_refused(
        "shrunken-session"
    ), "a session that lost rows must be reconsidered"


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


@pytest.mark.asyncio
async def test_prewarm_filter_stops_selecting_a_refused_session():
    """The FILTER must skip a refused session, not merely the helper it calls.

    The tests above drive `_sidebar_prewarm_refused` through a hand-written
    poll loop, which proves the helper answers correctly but says nothing about
    the one line that actually breaks the 2 s re-preparation loop — the
    `and not self._sidebar_prewarm_refused(entry.id)` clause in
    `_prewarm_sidebar`'s candidate comprehension. Deleting that clause left the
    rest of this file entirely green while restoring `main`'s exact behaviour:
    every poll re-selecting the same refused sessions and paying a full prepare
    for a result guaranteed to be discarded.

    So this drives the REAL `_prewarm_sidebar` over ten polls and counts
    prepares. Only the prepare seam itself is stubbed — leasing reaches for an
    owner record on disk and mounting needs a laid-out frame; the candidate
    filter, the admission call and the refusal cache are all production code,
    which is where the defect lived.
    """
    from local_operator.resume import SessionRow
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.session_catalog import CatalogEntry
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    oversized = "x" * (RETAIN_TEXT_BYTES + 64 * 1024)
    # PRECONDITION: the fixture is genuinely refused and the small one is
    # genuinely admitted, so a count of 1 below means "stopped" rather than
    # "never started", and the admissible session proves the filter has not
    # simply been switched off for everyone.
    assert not _presentation(oversized).retainable()
    assert _presentation("small").retainable()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._session_sidebar.display = True
        # The timer would poll on its own schedule; this test supplies the
        # polls explicitly so the count is the test's, not the clock's.
        if app._sidebar_timer is not None:
            app._sidebar_timer.pause()

        prepares: list[str] = []

        async def prepare(session_id: str, *, speculative: bool = False):
            prepares.append(session_id)
            source = _Source(session_id)
            # `_Source` carries exactly the attributes the admission path reads
            # (`presentation_revision`, and `session`'s two content stamps); a
            # real `SessionInteraction` would need an owner record on disk.
            app._sidebar_sources[session_id] = source  # type: ignore[assignment]
            text = "small" if session_id == "small-session" else oversized
            return source, _presentation(text)

        app._prepare_sidebar_session = prepare  # type: ignore[method-assign]

        async def release(prepared) -> None:
            return None

        app._release_sidebar_preparation = release  # type: ignore[method-assign]

        entries = [
            CatalogEntry(SessionRow("huge-session", 100, "Huge", live_state="busy")),
            CatalogEntry(SessionRow("small-session", 50, "Small", live_state="busy")),
        ]

        for _ in range(10):  # ten 2-second sidebar polls
            # A background turn streams between polls: this is what bumped
            # `presentation_revision` and expired the refusal on the operator's
            # actual population. Only the synthetic sources are bumped — the
            # app's own current session is a real `SessionInteraction` in this
            # dict and is not what this test drives.
            for source in app._sidebar_sources.values():
                if isinstance(source, _Source):
                    source.stream_events()
            app._prewarm_sidebar(entries)
            worker = app._sidebar_prefetch
            if worker is not None:
                await worker.wait()
            await pilot.pause()

        huge = prepares.count("huge-session")
        small = prepares.count("small-session")

    assert huge == 1, (
        f"_prewarm_sidebar re-prepared a refused session {huge} times over 10 polls: "
        "the candidate filter does not consult the refusal cache"
    )
    # The admissible session is prepared once and then cached, which is what
    # distinguishes "the filter works" from "prewarm stopped running".
    assert small == 1, f"an admissible session was prepared {small} times instead of once"
    assert "small-session" in app._sidebar_presentations
    assert "huge-session" not in app._sidebar_presentations
