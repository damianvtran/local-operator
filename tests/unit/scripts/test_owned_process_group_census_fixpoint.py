"""strict_linux reconciles /proc churn to a fixpoint instead of exhausting.

Why this file exists: PR3269's synthetic merge (CI run 41657372783, job
125993075099) refused a finally-refresh with ``Unknown: unstable /proc census``
because each of strict_linux's six attempts demanded a machine-wide zero delta
between its opening and closing /proc listings -- any unrelated fork or exit on
the runner broke every attempt. The fix (architect ruling 641ae72fe5d1) classifies
every PID the closing listing adds with the same proof the opening scan uses and
relists until nothing unclassified remains. These tests pin both halves: churn
that CAN be classified completes, and anything that cannot stays Unknown.
"""

from __future__ import annotations

import errno
from typing import Any

import pytest

from local_operator import owned_process_group as owned

OURS = 10


def _row(pid, group):
    return f"{pid} (synthetic) S 1 {group}"


class World:
    """Scripted /proc: successive listings plus per-PID stat outcomes.

    ``listings`` is consumed one per listdir call; the last entry repeats so a
    settled namespace stays settled. ``rows`` maps PID -> group int, or an
    exception instance to raise from the stat-row read, or a raw string row.
    """

    def __init__(self, listings, rows, gone=()):
        self.listings = [list(map(str, listing)) for listing in listings]
        self.rows = rows
        self.gone = {str(pid) for pid in gone}
        self.lists = 0
        self.reads: list[str] = []

    def listdir(self, path):
        assert path == "/proc"
        listing = self.listings[min(self.lists, len(self.listings) - 1)]
        self.lists += 1
        return listing

    def read(self, path):
        pid = path.split("/")[2]
        self.reads.append(pid)
        if pid in self.gone:
            raise FileNotFoundError(errno.ENOENT, "gone")
        row = self.rows[int(pid)]
        if isinstance(row, BaseException):
            raise row
        if isinstance(row, str):
            return row
        return _row(pid, row)

    def stat(self, path):
        if path.split("/")[2] in self.gone:
            raise FileNotFoundError(errno.ENOENT, "gone")
        return object()

    def census(self, diagnostics=None, clock=lambda: 0, deadline=5):
        return owned.strict_linux(
            OURS,
            deadline,
            listdir=self.listdir,
            read=self.read,
            stat=self.stat,
            clock=clock,
            diagnostics=diagnostics,
        )


def test_foreign_arrival_between_listings_completes_and_is_excluded():
    # The PR3269 shape: an unrelated runner process forks mid-scan.
    world = World([[10, 11], [10, 11, 500]], {10: OURS, 11: OURS, 500: 77})
    detail: dict[str, Any] = {}
    assert world.census(detail) == ([10, 11], True, True)
    assert detail["phase"] == "complete"
    assert detail["attempt_count"] == 1
    attempt = detail["attempts"][0]
    assert attempt["relist_count"] == 1
    assert attempt["reconciled"] == {
        "different_group": 1,
        "same_group": 0,
        "gone": 0,
        "departed": 0,
    }
    assert attempt["unclassified_count"] is None


def test_arrival_in_our_group_joins_the_members():
    world = World([[10], [10, 12]], {10: OURS, 12: OURS})
    detail: dict[str, Any] = {}
    assert world.census(detail) == ([10, 12], True, True)
    assert detail["attempts"][0]["reconciled"]["same_group"] == 1


def test_foreign_removal_needs_no_reread():
    world = World([[10, 20], [10]], {10: OURS, 20: 90})
    assert world.census() == ([10], True, True)
    assert sorted(world.reads) == ["10", "20"]


def test_own_member_that_exits_mid_census_is_excluded_and_complete():
    # Review round 1, R1 repro shape: PID 20 in our pgid is read, then gone
    # (row and directory ENOENT) at the closing listing and every later one.
    # The leader 10 is not listed, so the census must end ([], True, False):
    # a corpse returned as a member would make _terminal_absence refuse.
    class Exits(World):
        def read(self, path):
            if self.lists > 1:
                self.gone.add("20")
            return super().read(path)

        def stat(self, path):
            if self.lists > 1:
                self.gone.add("20")
            return super().stat(path)

    world = Exits([[20], []], {20: OURS})
    detail: dict[str, Any] = {}
    assert world.census(detail) == ([], True, False)
    attempt = detail["attempts"][0]
    assert attempt["reconciled"]["departed"] == 1
    assert attempt["relist_count"] == 1
    assert world.lists == 3


def test_own_member_readable_while_unlisted_is_a_contradiction():
    # 20 reads as ours but the listings keep omitting it: not proven gone.
    # Every attempt opens on [10, 20] and closes on [10].
    world = World([[10, 20], [10]] * 40, {10: OURS, 20: OURS})
    detail: dict[str, Any] = {}
    with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
        world.census(detail)
    assert detail["attempt_count"] == 6
    assert all(a["phase"] == "reconcile-contradiction" for a in detail["attempts"])


def test_own_member_unreadable_after_leaving_is_incomplete():
    class Denied(World):
        def read(self, path):
            # Closing listings are the even-numbered ones; only there is 20
            # unlisted and its re-read denied.
            if self.lists % 2 == 0 and path == "/proc/20/stat":
                raise PermissionError(errno.EACCES, "hidepid")
            return super().read(path)

    world = Denied([[10, 20], [10]] * 40, {10: OURS, 20: OURS})
    with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
        world.census()


def test_arrival_proven_gone_with_enoent_is_harmless():
    world = World([[10], [10, 600], [10]], {10: OURS}, gone={600})
    detail: dict[str, Any] = {}
    assert world.census(detail) == ([10], True, True)
    assert detail["attempts"][0]["reconciled"]["gone"] == 1


@pytest.mark.parametrize(
    "row",
    [
        PermissionError(errno.EACCES, "hidepid"),
        OSError(errno.EIO, "io"),
        "700 (short) S 1",
        "garbage",
    ],
    ids=["eacces", "eio", "short", "unparseable"],
)
def test_unclassifiable_arrival_keeps_every_attempt_incomplete(row):
    # Odd listings omit 700, even listings add it, so every attempt's closing
    # list adds the unreadable PID.
    world = World([[10], [10, 700]] * 20, {10: OURS, 700: row})
    detail: dict[str, Any] = {}
    with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
        world.census(detail)
    assert detail["attempt_count"] == 6
    assert all(a["unclassified_count"] == 1 for a in detail["attempts"])
    assert all(a["phase"] == "reconcile-unclassified" for a in detail["attempts"])


def test_arrival_whose_stat_row_vanishes_beside_a_live_directory_is_not_harmless():
    # ENOENT on the row but the directory still stats: not proof of absence.
    world = World([[10], [10, 800]] * 20, {10: OURS, 800: FileNotFoundError(errno.ENOENT, "x")})
    with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
        world.census()


def test_fixpoint_needing_several_relists_completes():
    world = World(
        [[10], [10, 501], [10, 501, 502], [10, 502, 12], [10, 502, 12]],
        {10: OURS, 501: 77, 502: 78, 12: OURS},
    )
    detail: dict[str, Any] = {}
    assert world.census(detail) == ([10, 12], True, True)
    assert detail["attempt_count"] == 1
    assert detail["attempts"][0]["relist_count"] == 3
    # m3: 501 arrived in an intermediate relist and left; it still counts.
    assert detail["attempts"][0]["added_count"] == 3
    assert detail["attempts"][0]["reconciled"] == {
        "different_group": 2,
        "same_group": 1,
        "gone": 0,
        "departed": 0,
    }


def test_churn_that_never_settles_is_unknown_within_six_attempts():
    # Every listing adds a brand-new foreign PID, so no closing list settles.
    class Endless(World):
        def listdir(self, path):
            self.lists += 1
            return ["10", str(1000 + self.lists)]

    world = Endless([[10]], {10: OURS})
    world.rows = {10: OURS, **{1000 + n: 77 for n in range(1, 500)}}
    detail: dict[str, Any] = {}
    with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
        world.census(detail)
    assert detail["attempt_count"] == 6
    relists = [a["relist_count"] for a in detail["attempts"]]
    assert relists == [owned._LINUX_CENSUS_RELISTS] * 6
    # One opening + (cap + 1) closing listings per attempt, never more.
    assert world.lists == 6 * (owned._LINUX_CENSUS_RELISTS + 2)


def test_churn_past_the_deadline_is_unknown_without_extra_attempts():
    class Endless(World):
        def listdir(self, path):
            self.lists += 1
            return ["10", str(1000 + self.lists)]

    world = Endless([[10]], {10: OURS})
    world.rows = {10: OURS, **{1000 + n: 77 for n in range(1, 500)}}
    ticks = iter(range(10_000))
    detail: dict[str, Any] = {}
    with pytest.raises(owned.Unknown, match="deadline") as raised:
        world.census(detail, clock=lambda: next(ticks), deadline=6)
    failure = raised.value.census_failure
    assert failure is not None
    assert failure["attempt_count"] == 1
    assert world.lists <= 4
