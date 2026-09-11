"""The sidebar catalog poll's cost, and the contract that cost may not buy.

The sidebar re-runs ``load_catalog`` every 2 seconds while it is open. That
scan asked "when was this last active" — two stats — for every directory in
the store before asking the far more selective question "is this session even
user-visible", and it located an unrelated marker file with a ``glob`` whose
wildcard is a directory component, which opens and enumerates every session
directory. On the reporting machine (1,946 directories, 92% of them subagent
sessions the picker never lists) one poll issued 9,773 syscalls and cost
126 ms, permanently, per TUI process.

The first fix (#867) reordered the two gates, which improved the CONSTANT —
~5.8 to ~2.2 syscalls per directory — while leaving the poll
``O(every session directory ever created)``: two unconditional per-directory
stats remained, the origin marker in the scan and the desktop marker in
``load_catalog``.

The second fix removed the scaling for the population that dominates a real
store. A directory already known to be hidden is now skipped whole, for ZERO
syscalls, because both facts that decision needs — the name and the inode —
come free from the ``readdir`` batch ``scandir`` already paid for. Per-directory
cost is therefore **O(user sessions + directories that are neither listed nor
cached-hidden)**: measured flat at 266 syscalls per poll while the store grew
from 100 to 8,000 directories with users held at 50, against 366 -> 16,166
before.

That middle term is not a rounding error and is stated rather than elided. A
directory with neither an origin marker nor any activity is in neither the
listing nor the hidden set, so it never arms the skip and pays ~4 syscalls per
poll forever while never being listable — 266 / 2,266 / 8,266 / 32,266 over
0 / 500 / 2,000 / 8,000 of them with both other populations fixed
(``bench_catalog_scan.py unmarked-axis``; agent review round 1, R1 measured the
same 4.0 slope on a counter with a different constant). "Tracks the user's own
sessions" is true of the hidden population
and false of this one, and
``test_the_cost_is_linear_in_directories_that_are_neither_listed_nor_hidden``
pins the limit so it cannot quietly detach from the claim again.

What remains O(total entries) is the single batched ``scandir`` — the poll
still touches the store, it just stops stat-ing it — and the verdict cache's
own parse (704 KB / 5.6 ms at 7,950 markers). Those and the never-active
population are the dominant remaining terms.

Neither fix is allowed to buy that with a different answer: no second source of
truth, and above all **no second ranking clock**. These tests pin both halves —
that the listing is byte-identical, and that the cost model is the new one —
because the failure mode of getting the first wrong is the retention policy
deleting rows the picker is still showing, and the failure mode of getting the
skip wrong is a real session invisible in the picker forever.
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
    """These pin the PER-DIRECTORY cost, which is now O(the user's own
    sessions) rather than O(the store).

    A directory already known to be hidden is skipped before its marker is
    stat'd, for zero syscalls. What remains O(total entries) is the single
    batched ``scandir`` — one call, not a stat each — so these tests assert a
    FLAT syscall count as the hidden population grows, which is precisely the
    assertion the previous round lacked and which would have caught #867's
    original overstatement.

    Asserted as syscall counts, not wall-clock: timings on a loaded machine
    measure contention, syscall counts measure the algorithm."""

    def test_a_hidden_session_costs_no_syscalls_at_all(self, tmp_path: Path) -> None:
        """The steady-state poll skips a known-hidden directory whole.

        One ``scandir`` for the store and nothing else: ``DirEntry.inode()``
        and ``is_dir()`` are both answered from the ``readdir`` batch that call
        already paid for.
        """
        for index in range(50):
            _session(tmp_path, f"{index:012x}", origin="subagent", stamp=1000.0)
        # Two warm-ups: the first builds the verdict cache AND is the cold-start
        # revalidating scan, so the second is the first poll able to use it.
        _recent_sessions_with_origin(tmp_path)
        _recent_sessions_with_origin(tmp_path)

        with _counting() as counter:
            _recent_sessions_with_origin(tmp_path)
        assert counter.counts["scandir"] == 1
        assert counter.counts["stat"] == 0
        assert counter.total == 1

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

    def test_the_cost_is_flat_as_the_hidden_population_grows(self, tmp_path: Path) -> None:
        """THE DECISIVE ASSERTION: users held fixed, hidden directories varied.

        This is the shape of the experiment, run in miniature. A cost that
        tracks the STORE rises with each batch; a cost that tracks the USER's
        sessions does not move at all. The previous implementation adds one stat
        per directory here and fails on the first batch, which is exactly the
        guard the earlier round lacked — a ladder that grows both populations
        together cannot tell the two hypotheses apart, and that is how #867
        shipped a claim bigger than its measurement.

        Asserted as EQUALITY, not a bound: 'flat' is the property, and a
        tolerance would let the slope creep back in unnoticed.
        """
        for index in range(10):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)

        measurements = []
        for batch in range(3):
            for index in range(200):
                _session(
                    tmp_path,
                    f"sub{batch:03x}{index:06x}",
                    origin="subagent",
                    stamp=500.0,
                )
            # Two scans: the first discovers the new directories, the second is
            # the steady state the sidebar actually pays.
            _recent_sessions_with_origin(tmp_path)
            _recent_sessions_with_origin(tmp_path)
            with _counting() as counter:
                rows = _recent_sessions_with_origin(tmp_path)
            measurements.append(counter.total)
            assert len(rows) == 10, "the listing must not move while cost is measured"

        assert measurements[0] == measurements[1] == measurements[2], measurements

    def test_the_cost_is_linear_in_directories_that_are_neither_listed_nor_hidden(
        self, tmp_path: Path
    ) -> None:
        """WHERE THE FLATNESS ABOVE STOPS — recorded, not left to be discovered.

        A directory with neither an origin marker nor any activity is in
        neither ``rows`` nor ``hidden_names``. It therefore can never arm the
        zero-syscall skip, and pays its origin stat on every poll forever while
        never being listable. The test above grows the store with HIDDEN
        directories, which is exactly the population the skip handles, so it
        reports flat while this cost rises — the same blind spot that let #867
        present a true statement about one axis as a general scaling property
        (agent review round 1, R1).

        This is deliberately a POSITIVE assertion of the limit rather than a
        bound to be improved: the honest claim is O(user sessions + directories
        that are neither listed nor cached-hidden), and a test that merely
        capped the cost would let the qualifier quietly detach from the number.
        Fixing the cost is refused on purpose — caching "unmarked" would serve a
        stale verdict for a directory the backfill stamps later — so this pins
        the consequence of that refusal.

        Asserted as a strict rise per batch rather than an exact slope: the
        per-directory constant is an implementation detail (~4 syscalls today,
        origin stat plus desktop probe), while "not flat" is the property.
        """
        for index in range(10):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(50):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0)

        measurements = []
        for batch in range(3):
            for index in range(100):
                # No transcript and no origin marker: the shape an idle
                # open-and-quit launch leaves behind.
                (tmp_path / "sessions" / f"idle{batch:03x}{index:06x}").mkdir(parents=True)
            _recent_sessions_with_origin(tmp_path)
            _recent_sessions_with_origin(tmp_path)
            with _counting() as counter:
                rows = _recent_sessions_with_origin(tmp_path)
            measurements.append(counter.total)
            assert len(rows) == 10, "a never-active directory is never listable"

        assert measurements[0] < measurements[1] < measurements[2], measurements
        first_step = measurements[1] - measurements[0]
        second_step = measurements[2] - measurements[1]
        assert first_step == second_step, (measurements, "linear, not accelerating")
        assert first_step >= 100, "at least one syscall per never-active directory"

    def test_the_desktop_probe_skips_known_hidden_directories(self, tmp_path: Path) -> None:
        """``load_catalog``'s SECOND O(store) stat, removed the same way.

        It probed ``desktop.json`` in every directory the listing did not
        return, which is the ~91% hidden population. A hidden directory cannot
        carry a desktop marker — ``DesktopSessions.create`` is the only writer
        and mints a fresh directory it never marks — so the probe asked a
        question already answered. This is half the total saving, so it gets its
        own pin.
        """
        for index in range(60):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=1000.0)
        _session(tmp_path, "f" * 12, stamp=2000.0)
        load_catalog(tmp_path)
        load_catalog(tmp_path)

        with _counting() as counter:
            load_catalog(tmp_path)
        # Well under one per hidden directory. Left as a bound rather than an
        # equality because ``load_catalog`` also reads the registry and the
        # attention store, whose constant is not this test's subject.
        assert counter.counts["stat"] < 60

    def test_a_directory_carrying_both_markers_resolves_to_the_origin_verdict(
        self, tmp_path: Path
    ) -> None:
        """PINS THE CROSS-MODULE INVARIANT THE PROBE SKIP RESTS ON.

        The skip above is sound because ``desktop.json`` has exactly one writer
        (``DesktopSessions.create``), which mints a fresh directory and never
        writes an origin marker into it — so a directory cannot carry both.
        That is correct today, but it is a coupling between two modules
        enforced only by prose: a future writer in ``desktop_sessions.py`` that
        stamped a desktop marker into an existing session directory would
        silently drop that row, with nothing failing (agent review round 1, R4).

        This asserts what the code does today so that change trips HERE, next to
        the reasoning, rather than as a missing sidebar row somebody bisects
        later. The behaviour pinned is deliberately the current one, not a
        wished-for one: the origin verdict WINS, the both-marker directory is
        hidden and its draft is not offered, and a legitimate desktop draft
        beside it is unaffected. If the single-writer premise is ever
        intentionally broken, this test is the place to state the new rule.
        """
        both = tmp_path / "sessions" / ("b" * 12)
        both.mkdir(parents=True)
        (both / "origin.json").write_text(json.dumps({"origin": "subagent"}), encoding="utf-8")
        (both / "desktop.json").write_text(json.dumps({"title": "draft"}), encoding="utf-8")
        # The shape the probe legitimately exists to find, as a control: if the
        # skip ever swallowed drafts wholesale this row would vanish too.
        draft = tmp_path / "sessions" / ("c" * 12)
        draft.mkdir(parents=True)
        (draft / "desktop.json").write_text(json.dumps({"title": "real"}), encoding="utf-8")

        # Across an armed poll as well as the cold one: the skip is what makes
        # the second cheap, so the verdict must not differ between them.
        for poll in range(3):
            rows = _recent_sessions_with_origin(tmp_path)
            catalog = [entry.id for entry in load_catalog(tmp_path)]
            assert [row[0] for row in rows] == [], f"poll {poll}: origin marker hides the row"
            assert catalog == ["c" * 12], f"poll {poll}: the legitimate draft is still offered"


# ---------------------------------------------------------------------------
# The hidden-skip fast path: what makes skipping a directory safe
# ---------------------------------------------------------------------------


class TestTheHiddenSkipCannotHideRealWork:
    """The fast path returns an answer WITHOUT looking at the directory, so
    every way that answer could be wrong is a real session missing from the
    picker — the severe failure this design exists to avoid.

    Each test here corresponds to one of the repairs the design carries: the
    inode qualification (id reuse), the revalidation epoch and the cold start
    (a hand-deleted marker), and the fail-safe that a corrupt marker is never
    memoised as hidden in the first place."""

    def test_a_recreated_id_is_visible_rather_than_serving_a_dead_verdict(
        self, tmp_path: Path
    ) -> None:
        """ID REUSE — the case that motivates the inode qualification.

        Delete a subagent directory, then create a REAL session under the same
        12-hex id. Keyed on the name alone the dead directory's hidden verdict
        is served for the live session, and the ONLY thing that ever repairs it
        is the epoch.

        The inode is what usually repairs it sooner, and how much sooner is a
        property of the FILESYSTEM, which is why this test asserts two
        different bars:

        * where the inode is REALLOCATED on recreate (measured on APFS:
          912841799 -> 912841800), the recreated id misses the cache and is
          visible on the VERY NEXT poll;
        * where it is RECYCLED immediately (measured on ext4 in
          ``python:3.12-slim``: 67634 -> 67634), the skip still fires and the
          session is hidden until the epoch — the documented degradation, since
          the inode is a hint and never a source of truth.

        Both are asserted rather than one being assumed, because the CI matrix
        runs both filesystems and an assertion written for APFS alone is how a
        Linux-only failure ships. What is NOT allowed on either is the session
        staying invisible forever, so the revalidation bar is checked
        unconditionally at the end.
        """
        import shutil

        reused = "a" * 12
        directory = _session(tmp_path, reused, origin="subagent", stamp=1000.0)
        before_ino = directory.stat().st_ino
        _session(tmp_path, "b" * 12, stamp=2000.0)
        _recent_sessions_with_origin(tmp_path)
        # Arm the fast path: this is the poll that would serve the stale answer.
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["b" * 12]

        shutil.rmtree(tmp_path / "sessions" / reused)
        recreated = _session(tmp_path, reused, stamp=3000.0)
        reallocated = recreated.stat().st_ino != before_ino

        listed = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
        if reallocated:
            assert listed == [reused, "b" * 12], "a moved inode must miss the cache at once"
        else:
            assert listed == ["b" * 12], "a recycled inode degrades to the epoch, by design"

        # The bar that holds on EVERY filesystem: never invisible for good.
        from local_operator import resume as resume_mod

        resume_mod._SCAN_COUNT[str(tmp_path)] = resume_mod.REVALIDATE_EVERY
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == [reused, "b" * 12]

    def test_a_missing_inode_degrades_to_the_slow_path_not_to_a_guess(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Inode behaviour was measured on APFS; a filesystem that cannot
        supply one must stay CORRECT, merely slower.

        Simulated by making ``DirEntry.inode()`` raise, which is the shape of
        the failure on a filesystem that does not answer it. The listing must
        be unchanged and the marker must be re-read — never a cached verdict
        accepted on a key that could not be qualified.
        """
        _session(tmp_path, "a" * 12, origin="subagent", stamp=1000.0)
        _session(tmp_path, "b" * 12, stamp=2000.0)
        with_inode = _recent_sessions_with_origin(tmp_path)
        _recent_sessions_with_origin(tmp_path)

        real_scandir = os.scandir

        class _NoInode:
            """A DirEntry whose inode is unavailable, delegating everything else."""

            def __init__(self, entry: Any) -> None:
                self._entry = entry
                self.name = entry.name
                self.path = entry.path

            def inode(self) -> int:
                raise OSError("inode unavailable on this filesystem")

            def __getattr__(self, item: str) -> Any:
                return getattr(self._entry, item)

        class _Scan:
            """Both a context manager and an iterator, exactly as
            ``os.scandir`` returns — the scan uses it as both."""

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __iter__(self) -> Any:
                return (_NoInode(entry) for entry in self._inner)

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: Any) -> None:
                self._inner.close()

        monkeypatch.setattr(os, "scandir", lambda path: _Scan(real_scandir(path)))
        assert _recent_sessions_with_origin(tmp_path) == with_inode

    def test_a_hand_deleted_marker_is_repaired_by_revalidation(self, tmp_path: Path) -> None:
        """Deleting ``origin.json`` is a SUPPORTED un-hide gesture.

        The origin backfill refuses to re-stamp an existing marker precisely so
        that a marker a human removed is not silently written back, so a
        permanent skip would answer that gesture with a session hidden forever.
        The epoch bounds it: while the fast path is armed the session stays
        hidden, and the revalidating poll finds it.
        """
        from local_operator import resume as resume_mod

        directory = _session(tmp_path, "a" * 12, origin="subagent", stamp=1000.0)
        _recent_sessions_with_origin(tmp_path)
        assert _recent_sessions_with_origin(tmp_path) == []

        (directory / "origin.json").unlink()
        assert _recent_sessions_with_origin(tmp_path) == [], "stale while armed, by design"

        # Drive the counter to the epoch rather than waiting 150 polls: the
        # arithmetic is the subject, not the wall time.
        resume_mod._SCAN_COUNT[str(tmp_path)] = resume_mod.REVALIDATE_EVERY
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]

    def test_a_cold_start_always_revalidates(self, tmp_path: Path) -> None:
        """Restarting ``lop`` repairs a stale verdict immediately.

        The counter starts at 0 for a fresh process, so ``0 % REVALIDATE_EVERY``
        is 0 and the first scan of every process is a revalidating one. That is
        the second of the three independent repairs, and the one an operator
        reaches for without being told about epochs.
        """
        from local_operator import resume as resume_mod

        directory = _session(tmp_path, "a" * 12, origin="subagent", stamp=1000.0)
        _recent_sessions_with_origin(tmp_path)
        _recent_sessions_with_origin(tmp_path)
        (directory / "origin.json").unlink()
        assert _recent_sessions_with_origin(tmp_path) == []

        resume_mod._SCAN_COUNT.pop(str(tmp_path), None)  # a fresh process
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]

    def test_a_corrupt_marker_stays_visible_on_every_poll(self, tmp_path: Path) -> None:
        """The fail-safe the fast path must not invert.

        A truncated or hand-edited marker parses to ``""`` — the user's own
        session — so it is never a HIDDEN verdict and can never arm the skip.
        Asserted over five consecutive polls because the bug this guards
        against is one that appears only once the cache is warm.
        """
        corrupt = _session(tmp_path, "a" * 12, stamp=1000.0)
        (corrupt / "origin.json").write_text('{"origin": "suba', encoding="utf-8")
        # Cut inside a multi-byte character: the read must not raise either.
        truncated = _session(tmp_path, "b" * 12, stamp=900.0)
        (truncated / "origin.json").write_bytes('{"origin": "sübagent'.encode("utf-8")[:-3])

        for poll in range(5):
            listed = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
            assert listed == ["a" * 12, "b" * 12], f"vanished on poll {poll}"

    def test_an_unreadable_marker_is_never_memoised_as_a_verdict(self, tmp_path: Path) -> None:
        """A transient EMFILE describes the MOMENT, not the file.

        The scan creates descriptor pressure itself, so a read failure must
        fall through and be retried on the next scan rather than being frozen
        into the cache for the life of the marker. The session stays visible
        (fail-safe) and the verdict is re-derived once the read succeeds.
        """
        from local_operator import resume as resume_mod

        directory = _session(tmp_path, "a" * 12, origin="subagent", stamp=1000.0)
        real_read = resume_mod._session_origin_read
        calls: list[Path] = []

        def failing(path: Path) -> tuple[str, bool]:
            calls.append(path)
            return "", False

        resume_mod._session_origin_read = failing  # type: ignore[assignment]
        try:
            assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]
            assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]
            assert len(calls) == 2, "an unreadable marker must be re-read, not memoised"
        finally:
            resume_mod._session_origin_read = real_read  # type: ignore[assignment]

        assert directory.is_dir()
        # Once the read succeeds the verdict is derived normally and the
        # session hides, proving nothing wrong was cached in the meantime.
        assert _recent_sessions_with_origin(tmp_path) == []

    def test_a_new_subagent_hides_immediately_while_the_fast_path_is_armed(
        self, tmp_path: Path
    ) -> None:
        """The skip only ever SUPPRESSES work for a directory already known
        hidden; an unknown directory always takes the slow path, so a subagent
        created mid-poll-cycle is hidden on the very next poll rather than
        waiting for an epoch."""
        _session(tmp_path, "a" * 12, stamp=1000.0)
        _recent_sessions_with_origin(tmp_path)
        _recent_sessions_with_origin(tmp_path)

        _session(tmp_path, "b" * 12, origin="subagent", stamp=2000.0)
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]

    def test_the_skip_does_not_change_the_listing_on_any_awkward_shape(
        self, tmp_path: Path
    ) -> None:
        """Every shape at once, over enough polls that the fast path is armed
        for all of them. The listing is the contract; the cost is not allowed
        to buy a different one."""
        _session(tmp_path, "a" * 12, stamp=5000.0)
        _session(tmp_path, "b" * 12, origin="fork", stamp=4000.0)
        _session(tmp_path, "c" * 12, origin="subagent", stamp=9000.0)
        _session(tmp_path, "d" * 12, transcript=None)
        _session(tmp_path, "e" * 12, transcript=None, inbox="{}\n", stamp=3000.0)
        _session(tmp_path, "f" * 12, origin="subagent", transcript=None)

        expected = ["a" * 12, "b" * 12, "e" * 12]
        for poll in range(4):
            assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == expected, poll

    def test_a_symlinked_session_directory_agrees_across_polls(self, tmp_path: Path) -> None:
        """A store may contain a symlink to a session directory. Whatever the
        first poll decides, the warm polls must decide identically — the point
        being that the fast path introduces no divergence of its own."""
        target = _session(tmp_path, "a" * 12, stamp=1000.0)
        _session(tmp_path, "b" * 12, origin="subagent", stamp=2000.0)
        link = tmp_path / "sessions" / ("c" * 12)
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:  # pragma: no cover - platforms without symlink permission
            import pytest

            pytest.skip("symlinks unavailable")

        first = _recent_sessions_with_origin(tmp_path)
        for _ in range(3):
            assert _recent_sessions_with_origin(tmp_path) == first

    def test_an_unknown_cache_version_rebuilds_rather_than_misreading(self, tmp_path: Path) -> None:
        """The migration path for the added ``ino`` field.

        A cache written by an older version has no inode to qualify anything
        with, so it is discarded wholesale — which the loader already did for
        any unknown version. Asserted end to end: the listing is right and the
        rewritten file carries the current version.
        """
        from local_operator import resume as resume_mod

        _session(tmp_path, "a" * 12, stamp=1000.0)
        _session(tmp_path, "b" * 12, origin="subagent", stamp=2000.0)
        _recent_sessions_with_origin(tmp_path)

        cache = resume_mod.origin_cache_path(tmp_path)
        stale = json.loads(cache.read_text(encoding="utf-8"))
        stale["version"] = resume_mod.ORIGIN_CACHE_VERSION - 1
        for entry in stale["entries"].values():
            entry.pop("ino", None)
        cache.write_text(json.dumps(stale), encoding="utf-8")

        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["a" * 12]
        rebuilt = json.loads(cache.read_text(encoding="utf-8"))
        assert rebuilt["version"] == resume_mod.ORIGIN_CACHE_VERSION
        assert all("ino" in entry for entry in rebuilt["entries"].values())

    def test_a_store_past_the_scan_limit_still_lists_identically(self, tmp_path: Path) -> None:
        """``CATALOG_SCAN_LIMIT`` caps the PAGE, never the scan.

        With more user sessions than the cap, the scan must still return all of
        them (``/resume`` shows the whole store) while ``load_catalog`` shows a
        capped, correctly ordered prefix — and both must be stable once the fast
        path is armed.
        """
        from local_operator.session.catalog import CATALOG_SCAN_LIMIT

        users = CATALOG_SCAN_LIMIT + 50
        for index in range(users):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(300):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0)

        first = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
        assert len(first) == users
        warm = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
        assert warm == first
        page = load_catalog(tmp_path)
        assert len(page) == CATALOG_SCAN_LIMIT
        assert [entry.id for entry in load_catalog(tmp_path)] == [entry.id for entry in page]

    def test_resume_and_the_sidebar_select_the_same_sessions(self, tmp_path: Path) -> None:
        """The invariant the whole change is measured against: both surfaces
        must SELECT the same sessions, because they share one scan.

        Set equality, not order equality — the two deliberately order
        differently and always have. ``_recent_sessions_with_origin`` ranks by
        the activity clock, while the sidebar re-ranks by ``CatalogEntry.rank``
        (``tier, wake_rank, -birth, id``), which is birth order inside a tier.
        Asserting a prefix relationship here would pin a coincidence rather
        than the contract, and would fail on any store where creation order and
        activity order differ.

        What must never happen is the two ranging over different SETS: the
        hidden-skip changes visibility, and a session the picker offers but the
        sidebar cannot show (or the reverse) is the divergence this pins."""
        for index in range(20):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(40):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0)
        _recent_sessions_with_origin(tmp_path)
        _recent_sessions_with_origin(tmp_path)

        picker = {row[0] for row in _recent_sessions_with_origin(tmp_path)}
        sidebar = {entry.id for entry in load_catalog(tmp_path)}
        assert sidebar == picker
        assert len(picker) == 20, "every subagent session must stay hidden from both"


def test_a_live_exec_record_labels_its_row_as_an_exec_run(tmp_path) -> None:
    """``kind`` reaches the row from the RECORD, through the real scan.

    The picker's tag is only as honest as this plumbing: it reads
    ``SessionRow.kind``, which nothing but ``decorate_rows`` sets. Asserted
    against a published record and the real ``registry.scan`` rather than a
    hand-built row, because the defect this guards is the two disagreeing.
    """
    from local_operator.resume import SessionRow
    from local_operator.session.catalog import decorate_rows
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    (tmp_path / "run" / "mobile").mkdir(parents=True)
    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="exec",
            session_id="bbbbbbbbbbbb",
            conversation_name="nightly audit",
            cwd=str(tmp_path),
            model_label="p/m",
            control_port=1234,
            control_key="k",
            detached=True,
        ),
        tmp_path,
    )
    rows = [
        SessionRow(id="bbbbbbbbbbbb", mtime=0.0, name="nightly audit"),
        SessionRow(id="cccccccccccc", mtime=0.0, name="no record at all"),
    ]
    by_id = {row.id: row for row in decorate_rows(tmp_path, rows)}
    assert by_id["bbbbbbbbbbbb"].kind == "exec"
    assert by_id["bbbbbbbbbbbb"].live_state == "idle"
    # A row with no live record must not inherit a kind from anywhere.
    assert by_id["cccccccccccc"].kind == ""


def test_an_idle_exec_row_says_what_it_is_instead_of_ready(tmp_path) -> None:
    """The words and the glyph may not disagree.

    ``CatalogEntry.status`` mirrors ``row_state_mark``'s precedence exactly, so
    the kind is allowed to speak only at the ``idle`` rung — a BUSY exec run
    still reports "Working", because what it is doing outranks what kind it is.
    """
    from local_operator.resume import SessionRow
    from local_operator.session.catalog import CatalogEntry

    idle = CatalogEntry(SessionRow("b", 0.0, "audit", live_state="idle", kind="exec"))
    busy = CatalogEntry(SessionRow("c", 0.0, "audit", live_state="busy", kind="exec"))
    mine = CatalogEntry(SessionRow("d", 0.0, "mine", live_state="idle"))
    assert idle.status == "Running headless (exec)"
    assert busy.status == "Working"
    assert mine.status == "Ready"


class TestTheSubagentLayerIsOptIn:
    """The sidebar's ⌥ layer: hidden subagent runs, listed without becoming rows.

    Two properties are load-bearing here and both were measured before this
    layer was written.

    SUB ROWS MUST NOT BE ORDINARY ROWS. ``CatalogEntry.active`` is
    ``pending or unseen or live_state``, and the attention store keys on
    conversation identity — which answers for subagent ids too, with 45% of real
    subagent directories carrying an unseen receipt. A sub row routed through
    ``decorate_rows`` and the attention comprehension therefore comes out
    ``active=True`` and sections into Active Sessions, ABOVE the user's own
    work. There is no filter inside that path that avoids it, so sub rows are
    built by hand outside ``rows`` and rejoin at the single ``rank_entries``
    call. ``test_a_sub_row_is_never_active`` is that guard.

    MAINS AND SUBS HYDRATE IN ONE CALL. ``cached_session_rows`` ends
    ``_ROW_CACHE.clear(); _ROW_CACHE.update(fresh)``, so a second call evicts
    the first call's rows — a warm 2.0 ms poll becomes 12.2 ms, a 6x regression
    on the hottest path the sidebar has.
    ``test_the_hydration_cache_is_not_evicted_by_the_layer`` is that guard.

    And the whole layer is opt-in: with the flag off this function must issue
    exactly the syscalls it issued before it existed.
    """

    def test_the_flag_off_is_byte_identical_to_today(self, tmp_path: Path) -> None:
        """The default path may not pay for a feature it is not using."""
        for index in range(6):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(20):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)
        load_catalog(tmp_path)  # warm every cache

        with _counting() as implicit:
            bare = [entry.id for entry in load_catalog(tmp_path)]
        with _counting() as explicit:
            flagged = [entry.id for entry in load_catalog(tmp_path, include_subagents=False)]

        assert bare == flagged
        assert explicit.total == implicit.total

    def test_the_layer_appends_hidden_rows_when_it_is_on(self, tmp_path: Path) -> None:
        for index in range(3):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(5):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)

        entries = load_catalog(tmp_path, include_subagents=True)
        assert len(entries) == 8
        assert sum(1 for entry in entries if entry.subagent) == 5
        assert {entry.id for entry in entries if not entry.subagent} == {
            f"user{index:08x}" for index in range(3)
        }

    def test_the_layer_is_capped(self, tmp_path: Path) -> None:
        """The layer answers "what just ran", so it is capped rather than paged."""
        from local_operator.session.catalog import SUBAGENT_LAYER_CAP

        _session(tmp_path, "user00000000", stamp=1000.0)
        for index in range(SUBAGENT_LAYER_CAP + 10):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)

        entries = load_catalog(tmp_path, include_subagents=True)
        assert sum(1 for entry in entries if entry.subagent) == SUBAGENT_LAYER_CAP

    def test_the_layer_is_newest_first(self, tmp_path: Path) -> None:
        """``rank``'s third key is ``-created_at``, so every sub row must have
        its creation time STAMPED. A row left at the 0.0 default ties with every
        other one and falls through to the id tie-break, silently reversing the
        order the user is promised."""
        from local_operator.session.catalog import SUBAGENT_LAYER_CAP

        for index in range(SUBAGENT_LAYER_CAP + 5):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)

        entries = [e for e in load_catalog(tmp_path, include_subagents=True) if e.subagent]
        stamps = [entry.row.created_at for entry in entries]
        assert stamps == sorted(stamps, reverse=True)
        newest = f"sub{SUBAGENT_LAYER_CAP + 4:09x}"
        assert newest in {entry.id for entry in entries}
        assert f"sub{0:09x}" not in {entry.id for entry in entries}

    def test_a_sub_row_is_never_active(self, tmp_path: Path) -> None:
        """THE SECTIONING GUARD. This directory looks like 45% of the real
        subagent population: it carries an unseen attention receipt. Routed
        through the ordinary row path it would come out ``active=True`` and sort
        above the user's own sessions, which is the blocker this layer is shaped
        around."""
        import uuid

        from local_operator.session.attention import (
            AttentionStore,
            conversation_identity,
        )

        _session(tmp_path, "user00000000", stamp=1000.0)
        directory = _session(tmp_path, "sub000000001", origin="subagent", stamp=900.0)
        store = AttentionStore(tmp_path / "attention.db")
        # A real token: the store parses it as a UUID.
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "anchor-1", "complete")
        # The receipt really is unseen, or this test proves nothing.
        assert store.state(conversation_identity(directory))["unseen"] is True

        entry = next(
            e for e in load_catalog(tmp_path, include_subagents=True) if e.id == "sub000000001"
        )
        assert entry.subagent is True
        assert entry.active is False
        assert entry.unseen is False
        assert entry.completion_kind == ""
        assert entry.completion_token == ""
        assert entry.anchor_id == ""
        assert entry.row.live_state == ""
        assert entry.row.pending is None
        assert entry.row.wakes == 0

    def test_a_sub_row_carries_its_role_and_label(self, tmp_path: Path) -> None:
        directory = tmp_path / "sessions" / "sub000000001"
        _session(tmp_path, "sub000000001", origin="subagent", stamp=900.0)
        (directory / "origin.json").write_text(
            json.dumps({"origin": "subagent", "agent": "reviewer", "label": "round 2"}),
            encoding="utf-8",
        )

        entry = next(e for e in load_catalog(tmp_path, include_subagents=True) if e.subagent)
        assert entry.agent == "reviewer"
        assert entry.label == "round 2"
        assert entry.sub_title == "round 2 · reviewer"

    def test_a_sub_row_keeps_its_marker_fields_through_hydration(self, tmp_path: Path) -> None:
        """The hydration pass rebuilds ``entry.row``; it must not drop the
        entry's own fields on the way."""
        from local_operator.resume import write_session_title

        directory = _session(tmp_path, "sub000000001", origin="subagent", stamp=900.0)
        (directory / "origin.json").write_text(
            json.dumps({"origin": "subagent", "agent": "qa", "label": "smoke"}),
            encoding="utf-8",
        )
        # Through the real naming path, so the name resolves the way the picker
        # and the sidebar resolve it rather than through a hand-written sidecar.
        write_session_title(directory, "a real name", user_set=False, past_names=[])

        entry = next(e for e in load_catalog(tmp_path, include_subagents=True) if e.subagent)
        # A name proves it went through `cached_session_rows` rather than
        # keeping the empty placeholder its hand-built row started with.
        assert entry.row.name == "a real name"
        assert entry.subagent is True
        assert entry.agent == "qa"
        assert entry.label == "smoke"

    def test_a_corrupt_origin_marker_degrades_to_the_session_name(self, tmp_path: Path) -> None:
        """A marker is best-effort: it may cost the row its role and label,
        never the row."""
        broken = _session(tmp_path, "sub000000001", origin="subagent", stamp=900.0)
        (broken / "origin.json").write_text("{not json", encoding="utf-8")
        empty = _session(tmp_path, "sub000000002", origin="subagent", stamp=800.0)
        (empty / "origin.json").write_text("{}", encoding="utf-8")

        entries = {entry.id: entry for entry in load_catalog(tmp_path, include_subagents=True)}
        assert {"sub000000001", "sub000000002"} <= set(entries)
        for session_id in ("sub000000001", "sub000000002"):
            entry = entries[session_id]
            assert entry.agent == ""
            assert entry.label == ""
            assert entry.sub_title == entry.row.name

    def test_a_pinned_hidden_id_is_hydrated_with_the_layer_off(self, tmp_path: Path) -> None:
        """A pin outranks the layer switch: pinning a subagent run and then
        turning the layer off must not make the pin render as nothing."""
        _session(tmp_path, "user00000000", stamp=1000.0)
        for index in range(3):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)

        entries = load_catalog(tmp_path, pinned_hidden_ids=["sub000000001"])
        by_id = {entry.id: entry for entry in entries}
        assert by_id["sub000000001"].subagent is True
        assert "sub000000000" not in by_id
        assert "sub000000002" not in by_id

    def test_a_pinned_hidden_id_beyond_the_cap_is_still_hydrated(self, tmp_path: Path) -> None:
        """Pins are exempt from the cap, or a pin to an old run would resolve to
        nothing the moment 40 newer runs existed."""
        from local_operator.session.catalog import SUBAGENT_LAYER_CAP

        for index in range(SUBAGENT_LAYER_CAP + 5):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)
        oldest = f"sub{0:09x}"

        entries = load_catalog(tmp_path, include_subagents=True, pinned_hidden_ids=[oldest])
        subs = [entry for entry in entries if entry.subagent]
        assert oldest in {entry.id for entry in subs}
        assert len(subs) == SUBAGENT_LAYER_CAP + 1

    def test_a_pinned_id_that_is_not_hidden_costs_nothing(self, tmp_path: Path) -> None:
        """A visible session is already in the listing; pinning it must not add
        a second copy through the hidden path."""
        for index in range(3):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        load_catalog(tmp_path)

        plain = [entry.id for entry in load_catalog(tmp_path)]
        pinned = [entry.id for entry in load_catalog(tmp_path, pinned_hidden_ids=["user00000001"])]
        assert pinned == plain
        assert len(pinned) == len(set(pinned))

    def test_the_hydration_cache_is_not_evicted_by_the_layer(self, tmp_path: Path) -> None:
        """THE CACHE GUARD. ``cached_session_rows`` ends with
        ``_ROW_CACHE.clear(); _ROW_CACHE.update(fresh)``, so hydrating mains and
        subs in TWO calls leaves only the second call's rows cached and the
        next poll rebuilds the rest from disk — measured 2.0 ms warm against
        12.2 ms cache-wiped. Both populations must go through ONE call with a
        concatenated candidate list.

        Every fixture directory here carries a transcript deliberately, not
        decoratively: ``_ROW_CACHE`` is written only when ``_row_stat_key``
        succeeds, and the layer itself drops a hidden name whose transcript stat
        raises. Bare directories would produce a cache of 10 and zero sub rows —
        a fixture failure wearing the costume of the bug this guards. The sub
        count is asserted alongside so the two are distinguishable at a glance.
        """
        from local_operator.session.catalog import _ROW_CACHE

        for index in range(10):
            _session(tmp_path, f"user{index:08x}", transcript="{}\n", stamp=1000.0 + index)
        for index in range(10):
            _session(
                tmp_path,
                f"sub{index:09x}",
                transcript="{}\n",
                origin="subagent",
                stamp=500.0 + index,
            )

        entries = load_catalog(tmp_path, include_subagents=True)
        assert sum(1 for entry in entries if entry.subagent) == 10, "fixture: no sub rows built"
        assert len(_ROW_CACHE) == 20, "a second cached_session_rows call evicted the first's rows"

        # And the warm poll agrees with the cold one.
        assert [entry.id for entry in load_catalog(tmp_path, include_subagents=True)] == [
            entry.id for entry in entries
        ]

    def test_the_layer_competes_for_the_page_on_a_full_store(self, tmp_path: Path) -> None:
        """A KNOWN, ACCEPTED trade-off, pinned so the next reader can tell it was
        decided rather than missed.

        ``[:limit]`` applies to the COMBINED list, so a store with more than
        ~160 visible sessions cannot fit both populations in
        ``CATALOG_SCAN_LIMIT``. Sub rows do not lose that race: their
        ``session_category`` inputs are all falsy by construction, so they share
        a tier with a cold visible row and the tie-break is ``-created_at``,
        where recent subagent runs win. What gets pushed past the slice is
        therefore the user's OLDEST COLD sessions — never an active row, which
        ranks above the whole contest.

        Raising the limit would restore the per-poll row-building cost
        ``CATALOG_SCAN_LIMIT``'s comment exists to document removing, so the
        trade is taken deliberately: the layer is opt-in, the rows at stake are
        past rank 160, and ``/resume`` — the surface for finding an old session
        — does not go through ``load_catalog`` at all.
        """
        from local_operator.session.catalog import (
            CATALOG_SCAN_LIMIT,
            SUBAGENT_LAYER_CAP,
        )

        # Just under the cap, so the store fits ENTIRELY with the layer off and
        # the rows that vanish can only be the ones the layer displaced.
        visible = CATALOG_SCAN_LIMIT - 10
        # Visible sessions carry the OLDER creation stamps; `created_at` is
        # written explicitly rather than left to the filesystem birthtime, which
        # macOS has and Linux does not — on CI every row would otherwise tie at
        # 0.0 and rank by session id, testing the tie-break instead of the order.
        for index in range(visible):
            directory = _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
            (directory / "created_at.json").write_text(str(1000.0 + index), encoding="utf-8")
        oldest_visible = "user00000000"
        for index in range(SUBAGENT_LAYER_CAP):
            directory = _session(
                tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index
            )
            # Subagent runs are RECENT, which is what wins them the tie-break.
            (directory / "created_at.json").write_text(str(9000.0 + index), encoding="utf-8")

        # With the layer off the whole store fits and nothing is displaced.
        assert oldest_visible in {entry.id for entry in load_catalog(tmp_path)}

        entries = load_catalog(tmp_path, include_subagents=True)
        assert len(entries) == CATALOG_SCAN_LIMIT
        subs = [entry for entry in entries if entry.subagent]
        assert len(subs) == SUBAGENT_LAYER_CAP, "sub rows must not be what falls off the end"
        assert len(entries) - len(subs) == CATALOG_SCAN_LIMIT - SUBAGENT_LAYER_CAP
        # The user's oldest sessions are precisely what the layer displaced.
        surviving = {entry.id for entry in entries}
        assert oldest_visible not in surviving
        displaced = visible + SUBAGENT_LAYER_CAP - CATALOG_SCAN_LIMIT
        for index in range(displaced):
            assert f"user{index:08x}" not in surviving
        assert f"user{displaced:08x}" in surviving

    def test_subagent_population_counts_the_hidden_store(self, tmp_path: Path) -> None:
        from local_operator.session.catalog import subagent_population

        for index in range(4):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(7):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)

        assert subagent_population(tmp_path) == 7

    def test_the_layer_does_not_reach_the_deletion_authority(self, tmp_path: Path) -> None:
        """THE STRUCTURAL GUARANTEE. ``session.cleanup`` DELETES, and it decides
        what to protect from ``recent_sessions(..., revalidate=True)`` rather
        than from ``load_catalog``. It takes no layer flag and cannot be given
        one, so sub rows are structurally unable to reach it — asserted here so
        a future refactor that routes cleanup through the catalog fails loudly
        instead of quietly counting subagent runs against the user's guard.
        """
        from local_operator.session.cleanup import _picker_rows

        for index in range(2):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(5):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0 + index)

        assert set(_picker_rows(tmp_path)) == {"user00000000", "user00000001"}
