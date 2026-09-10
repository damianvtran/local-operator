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

The second fix removed the scaling itself. A directory already known to be
hidden is now skipped whole, for ZERO syscalls, because both facts that
decision needs — the name and the inode — come free from the ``readdir`` batch
``scandir`` already paid for. Per-directory cost therefore tracks the USER's
own sessions: measured flat at 266 syscalls per poll while the store grew from
100 to 8,000 directories with users held at 50, against 366 -> 16,166 before.
What remains O(total entries) is the single batched ``scandir`` — the poll
still touches the store, it just stops stat-ing it — and the verdict cache's
own parse (704 KB / 5.6 ms at 7,950 markers), which is now the dominant
remaining term.

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
        """ID REUSE — the case that fails a name-keyed cache.

        Delete a subagent directory, then create a REAL session under the same
        12-hex id. Keyed on the name alone the dead directory's hidden verdict
        is served for the live session and it never appears. The inode closes
        it: a recreated directory gets a different inode, so it misses the
        cache and takes the slow path.

        This is why the inode qualification is load-bearing rather than a
        nicety, and it is the one assertion that distinguishes the shipped
        design from the cheaper one.
        """
        import shutil

        reused = "a" * 12
        _session(tmp_path, reused, origin="subagent", stamp=1000.0)
        _session(tmp_path, "b" * 12, stamp=2000.0)
        _recent_sessions_with_origin(tmp_path)
        # Arm the fast path: this is the poll that would serve the stale answer.
        assert [row[0] for row in _recent_sessions_with_origin(tmp_path)] == ["b" * 12]

        shutil.rmtree(tmp_path / "sessions" / reused)
        _session(tmp_path, reused, stamp=3000.0)

        listed = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
        assert reused in listed, "a session recreated under a reused id must be visible"
        assert listed == [reused, "b" * 12]

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

    def test_resume_and_the_sidebar_list_the_same_set_in_the_same_order(
        self, tmp_path: Path
    ) -> None:
        """The invariant the whole change is measured against. Both surfaces
        share one scan, so they cannot diverge — pinned here so a future
        'optimisation' that gives the sidebar its own path fails loudly."""
        for index in range(20):
            _session(tmp_path, f"user{index:08x}", stamp=1000.0 + index)
        for index in range(40):
            _session(tmp_path, f"sub{index:09x}", origin="subagent", stamp=500.0)
        _recent_sessions_with_origin(tmp_path)
        _recent_sessions_with_origin(tmp_path)

        picker = [row[0] for row in _recent_sessions_with_origin(tmp_path)]
        sidebar = [entry.id for entry in load_catalog(tmp_path)]
        assert sidebar == picker[: len(sidebar)]
