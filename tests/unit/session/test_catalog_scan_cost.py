"""The sidebar catalog poll's cost, and the contract that cost may not buy.

The sidebar re-runs ``load_catalog`` every 2 seconds while it is open. That
scan asked "when was this last active" — two stats — for every directory in
the store before asking the far more selective question "is this session even
user-visible", and it located an unrelated marker file with a ``glob`` whose
wildcard is a directory component, which opens and enumerates every session
directory. On the reporting machine (1,946 directories, 92% of them subagent
sessions the picker never lists) one poll issued 9,773 syscalls and cost
126 ms, permanently, per TUI process.

What changed is the CONSTANT, and these tests are named for that. The poll
went from ~5.8 to ~2.2 syscalls per directory (9,773 -> 4,350 on that store,
126 ms -> 25 ms), but it is still ``O(every session directory ever created)``
— two unconditional per-directory stats remain, the origin marker here and the
desktop marker in ``load_catalog``. The store growing large enough still
degrades the poll; it re-reaches the old cost at roughly 8,000 directories.
Reaching ``O(the user's own sessions)`` needs an index or a persistent memo,
which this change deliberately does not attempt.

The fix is entirely a reordering and a call-shape change: no caching, no new
source of truth, and above all **no second ranking clock**. These tests pin
both halves — that the answer is unchanged, and that the cost model is the new
one — because an optimisation here is only worth having if the listing is
byte-identical, and the failure mode of getting it wrong is the retention
policy deleting rows the picker is still showing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import local_operator.session.retention as retention
from local_operator.resume import _recent_sessions_with_origin
from local_operator.session.catalog import load_catalog
from local_operator.session.retention import session_activity, session_activity_path


def _session(
    root: Path,
    session_id: str,
    *,
    transcript: str | None = "{}\n",
    inbox: str | None = None,
    origin: str | None = None,
    stamp: float | None = None,
) -> Path:
    """One session directory in whichever awkward shape a test needs."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if transcript is not None:
        (directory / "transcript.jsonl").write_text(transcript, encoding="utf-8")
    if inbox is not None:
        (directory / "inbox.jsonl").write_text(inbox, encoding="utf-8")
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}), encoding="utf-8")
    if stamp is not None:
        for name in ("transcript.jsonl", "inbox.jsonl"):
            if (directory / name).exists():
                os.utime(directory / name, (stamp, stamp))
    return directory


def _counting(names: tuple[str, ...] = ("stat", "lstat", "scandir")) -> Any:
    """Count ``os`` filesystem calls made inside the context."""

    class Counter:
        def __init__(self) -> None:
            self.counts: dict[str, int] = {name: 0 for name in names}
            self._originals: dict[str, Callable[..., Any]] = {}

        def __enter__(self) -> "Counter":
            for name in names:
                real = getattr(os, name)
                self._originals[name] = real

                def wrap(real: Callable[..., Any] = real, name: str = name) -> Any:
                    def counting(*args: Any, **kwargs: Any) -> Any:
                        self.counts[name] += 1
                        return real(*args, **kwargs)

                    return counting

                setattr(os, name, wrap())
            return self

        def __exit__(self, *exc: Any) -> None:
            for name, real in self._originals.items():
                setattr(os, name, real)

        @property
        def total(self) -> int:
            return sum(self.counts.values())

    return Counter()


# ---------------------------------------------------------------------------
# The clock stays single-sourced
# ---------------------------------------------------------------------------


class TestTheRankingClockIsNotForked:
    """``session_activity`` is shared with ``session.cleanup`` on purpose: the
    picker's "most recent" and the policy's "most recent" must be the same
    directories, or the policy deletes rows the picker shows (QA round 1 Q2,
    UX round 2 U11). The fast path may change the CALL SHAPE and never the
    rule."""

    def test_the_path_form_is_the_only_body(self) -> None:
        """``session_activity`` delegates rather than duplicating the loop, so
        the two answers cannot drift apart by editing one of them."""
        import inspect

        source = inspect.getsource(session_activity)
        assert "session_activity_path" in source
        # The rule itself — iterating the activity files — lives in ONE place.
        assert "_ACTIVITY_FILES" not in source

    def test_both_forms_agree_on_every_shape(self, tmp_path: Path) -> None:
        shapes = {
            "both": _session(tmp_path, "a" * 12, transcript="{}\n", inbox="{}\n"),
            "transcript only": _session(tmp_path, "b" * 12, transcript="{}\n"),
            "inbox only": _session(tmp_path, "c" * 12, transcript=None, inbox="{}\n"),
            "neither": _session(tmp_path, "d" * 12, transcript=None),
            "absent": tmp_path / "sessions" / "nope",
        }
        for label, directory in shapes.items():
            assert session_activity(directory) == session_activity_path(
                str(directory)
            ), f"forms disagree for {label}"

    def test_the_clock_is_still_only_the_transcript_and_the_spool(self, tmp_path: Path) -> None:
        """Bookkeeping the harness writes must not restamp "last activity", and
        the DIRECTORY mtime is never the clock — that was the U11 defect."""
        directory = _session(tmp_path, "e" * 12, stamp=1000.0)
        (directory / "title-scan.json").write_text("{}", encoding="utf-8")
        os.utime(directory, (time.time(), time.time()))
        assert session_activity_path(str(directory)) == 1000.0

    def test_the_scan_and_the_policy_rank_the_same_directories(self, tmp_path: Path) -> None:
        """The property the shared clock exists to guarantee, end to end."""
        for index in range(6):
            _session(tmp_path, f"{index:012x}", stamp=1_000_000.0 + index)
        listed = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
        ranked = sorted(
            (p.name for p in (tmp_path / "sessions").iterdir()),
            key=lambda name: -(retention.session_activity(tmp_path / "sessions" / name) or 0),
        )
        assert listed == ranked


# ---------------------------------------------------------------------------
# Invalidation: new activity must never be missed
# ---------------------------------------------------------------------------


class TestActivityBetweenPollsIsSeenImmediately:
    """A listing that misses new activity is a worse bug than a slow one, so
    every one of these asserts the NEXT poll — not an eventual one.

    These are the regression guard for a tempting optimisation this change
    deliberately does NOT make: gating the activity stat on the directory's
    own mtime. Appending to an existing file does not change its directory's
    mtime (POSIX only requires that for adding or removing entries; measured
    on APFS), and ``mark_session_origin`` explicitly restores the directory
    mtime after writing a marker — so a dir-mtime gate would silently miss
    both."""

    def test_an_appended_transcript_reranks_on_the_very_next_poll(self, tmp_path: Path) -> None:
        older = _session(tmp_path, "a" * 12, stamp=1000.0)
        _session(tmp_path, "b" * 12, stamp=2000.0)
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == [
            "b" * 12,
            "a" * 12,
        ]

        with (older / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        os.utime(older / "transcript.jsonl", (3000.0, 3000.0))

        rows = _recent_sessions_with_origin(tmp_path)
        assert [row[0] for row in rows] == ["a" * 12, "b" * 12]
        assert rows[0][1] == 3000.0

    def test_a_brand_new_session_appears_on_the_very_next_poll(self, tmp_path: Path) -> None:
        _session(tmp_path, "a" * 12, stamp=1000.0)
        assert len(_recent_sessions_with_origin(tmp_path)) == 1
        _session(tmp_path, "b" * 12, stamp=2000.0)
        assert len(_recent_sessions_with_origin(tmp_path)) == 2

    def test_a_first_inbox_message_makes_a_silent_directory_rank(self, tmp_path: Path) -> None:
        """Unread mail is activity waiting for a person, so a directory with
        only a spool is a row."""
        directory = _session(tmp_path, "a" * 12, transcript=None)
        assert _recent_sessions_with_origin(tmp_path) == []
        (directory / "inbox.jsonl").write_text("{}\n", encoding="utf-8")
        os.utime(directory / "inbox.jsonl", (5000.0, 5000.0))
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]

    def test_a_marker_written_after_a_listing_hides_the_session_next_poll(
        self, tmp_path: Path
    ) -> None:
        """The origin backfill stamps directories later; the gate must notice.

        ``mark_session_origin`` restores the directory's mtime after writing,
        which is exactly why the gate reads the MARKER's stat and not the
        directory's.
        """
        from local_operator.resume import mark_session_origin

        directory = _session(tmp_path, "a" * 12, stamp=1000.0)
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]
        before = directory.stat().st_mtime
        mark_session_origin(directory, "subagent")
        # The writer deliberately preserves the directory mtime, so this
        # transition is invisible to anything keyed on it.
        assert directory.stat().st_mtime == before
        assert _recent_sessions_with_origin(tmp_path) == []


# ---------------------------------------------------------------------------
# Visibility and ordering are unchanged
# ---------------------------------------------------------------------------


class TestTheListingIsUnchanged:
    """Reordering the gates is only sound because they are a CONJUNCTION: a row
    needs activity AND user-visibility, so asking the selective question first
    cannot change the answer."""

    def test_every_awkward_shape_lands_on_the_right_side(self, tmp_path: Path) -> None:
        _session(tmp_path, "a" * 12, stamp=5000.0)  # plain user session
        _session(tmp_path, "b" * 12, origin="fork", stamp=4000.0)  # visible
        _session(tmp_path, "c" * 12, origin="subagent", stamp=9000.0)  # hidden
        _session(tmp_path, "d" * 12, transcript=None)  # no activity
        _session(tmp_path, "e" * 12, transcript=None, inbox="{}\n", stamp=3000.0)
        # A corrupt marker reads as the user's own session rather than
        # vanishing: existence gates the READ, never the verdict.
        corrupt = _session(tmp_path, "f" * 12, stamp=2000.0)
        (corrupt / "origin.json").write_text("{not json", encoding="utf-8")

        rows = _recent_sessions_with_origin(tmp_path)
        assert [row[0] for row in rows] == ["a" * 12, "b" * 12, "e" * 12, "f" * 12]
        assert dict((row[0], row[2]) for row in rows)["b" * 12] == "fork"

    def test_equal_stamps_break_on_the_id_ascending(self, tmp_path: Path) -> None:
        """A stable tie order is load-bearing: with an unstable one the policy's
        first page and the picker's disagreed (QA round 2, Q10)."""
        for name in ("c", "a", "b"):
            _session(tmp_path, name * 12, stamp=7000.0)
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == [
            "a" * 12,
            "b" * 12,
            "c" * 12,
        ]

    def test_the_verdict_cache_still_answers_identically(self, tmp_path: Path) -> None:
        """Two consecutive scans agree; the second is served from the cache."""
        _session(tmp_path, "a" * 12, stamp=5000.0)
        _session(tmp_path, "b" * 12, origin="subagent", stamp=6000.0)
        _session(tmp_path, "c" * 12, origin="fork", stamp=7000.0)
        assert _recent_sessions_with_origin(tmp_path) == _recent_sessions_with_origin(tmp_path)


# ---------------------------------------------------------------------------
# The cost model
# ---------------------------------------------------------------------------


class TestThePollsPerDirectoryCostIsBounded:
    """These pin the PER-DIRECTORY cost, which is what this change reduces —
    not the scaling, which it does not change.

    The poll remains O(total session directories): the origin-marker stat runs
    for every entry and ``load_catalog``'s desktop-marker probe for every
    unlisted one. What the reordering buys is the constant — a hidden directory
    costs one stat where it cost three. Name these tests for that bound, so a
    later reader does not take a green suite as proof the store no longer
    matters; measured flat at ~2.0-2.2 syscalls/dir from 150 to 4,050
    directories in agent review / QA round 1.

    Asserted as syscall counts, not wall-clock: timings on a loaded machine
    measure contention, syscall counts measure the algorithm."""

    def test_a_hidden_session_costs_one_stat_not_three(self, tmp_path: Path) -> None:
        """The origin gate runs FIRST, so a subagent directory never pays for
        the two activity stats whose answer would be discarded."""
        for index in range(50):
            _session(tmp_path, f"{index:012x}", origin="subagent", stamp=1000.0)
        _recent_sessions_with_origin(tmp_path)  # warm the verdict cache

        with _counting() as counter:
            _recent_sessions_with_origin(tmp_path)
        # One scandir for the store, then exactly one marker stat per hidden
        # directory. Three per directory was the old shape.
        assert counter.counts["scandir"] == 1
        assert counter.counts["stat"] <= 50 + 5

    def test_the_scan_does_not_open_every_session_directory(self, tmp_path: Path) -> None:
        """``load_catalog`` located ``desktop.json`` with ``glob("*/desktop.json")``,
        whose directory-component wildcard enumerates every session directory —
        1,946 ``scandir`` calls per poll on the reporting machine to find one
        file. A stat answers the same question."""
        for index in range(40):
            _session(tmp_path, f"{index:012x}", origin="subagent", stamp=1000.0)
        _session(tmp_path, "f" * 12, stamp=2000.0)
        load_catalog(tmp_path)  # warm every cache

        with _counting() as counter:
            load_catalog(tmp_path)
        # A small constant: the sessions directory itself, plus whatever the
        # registry and attention stores read. Emphatically not one per session.
        assert counter.counts["scandir"] < 10

    def test_a_hidden_session_adds_one_stat_not_three(self, tmp_path: Path) -> None:
        """Each added subagent directory costs ONE stat, down from three.

        This is deliberately NOT "hidden sessions are free": the bound below is
        linear in the directories added (200 dirs -> <=200 stats), because that
        is the property that actually ships. It fails on the previous
        implementation, which paid three. Asserted as a DELTA against a
        measured baseline rather than an absolute ceiling, so it pins the
        per-directory slope instead of encoding one machine's number."""
        for index in range(10):
            _session(tmp_path, f"{index:012x}", stamp=1000.0 + index)
        _recent_sessions_with_origin(tmp_path)
        with _counting() as small:
            _recent_sessions_with_origin(tmp_path)

        for index in range(200):
            _session(tmp_path, f"{index + 1000:012x}", origin="subagent", stamp=500.0)
        _recent_sessions_with_origin(tmp_path)
        with _counting() as large:
            _recent_sessions_with_origin(tmp_path)

        # Each added hidden directory costs ONE stat. The old code paid three,
        # so this bound fails on the previous implementation.
        assert large.total - small.total <= 200 + 10
