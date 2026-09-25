"""``@latest`` is chosen by the picker's own scan, and the walk is the check.

``--resume @latest`` answered "which conversation reopens" with a private
``sessions.glob("*")`` walk: two ``stat`` calls and an ``origin.json`` READ per
directory, ~358 ms CPU on the operator's real store and re-paid on every
``lop --resume``. It now takes the first row of ``_scan_sessions`` — the one
scan every listing already goes through, whose known-hidden skip makes ~93% of
a real store cost zero syscalls.

Two things are pinned here, and the first is not negotiable:

* IDENTITY. The selected id must equal the id the old rule selected, on every
  store shape the rule can see, including the ones where a plausible-looking
  rewrite differs: ties, archived sessions, a subagent-only store, an inbox-only
  session, a corrupt marker, an unreadable marker and a session whose title
  sidecar is missing. The old implementation is kept VERBATIM as an oracle
  rather than re-described, because "same session" is the whole contract and a
  paraphrase is what lets a rewrite differ in the corner it forgot.
* WORK DONE, asserted as syscall counts and never as wall time: a second
  resolution must issue nothing beyond the invalidation check — one ``scandir``
  (how a new session directory is noticed at all) plus the per-directory checks
  the scan cannot answer from its cache — and the hidden population's
  contribution to that must be exactly ZERO, asserted both as an exact count and
  as flatness while that population grows by hundreds of directories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator import resume as resume_mod
from local_operator.resume import (
    ORIGIN_NAME,
    RESUME_LATEST,
    TRANSCRIPT_NAME,
    ResumeNotFound,
    _scan_sessions,
    resume_dir,
)
from local_operator.session.runtime.inbox import INBOX_NAME

TITLE_SIDECAR = "title.json"


# ---------------------------------------------------------------------------
# The fixture store and the oracle
# ---------------------------------------------------------------------------


def _session(
    root: Path,
    session_id: str,
    *,
    transcript: bool = True,
    inbox: str | None = None,
    origin: str | None = None,
    marker_bytes: bytes | None = None,
    sidecar: dict[str, Any] | None = None,
    stamp: float | None = None,
) -> Path:
    """One session directory, in whichever awkward shape a case needs.

    ``stamp`` pins the mtime of every activity file, which is the ONE clock the
    ranking reads: leaving it unset means "now", which would make every fixture
    a tie and hide the ordering under test.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if transcript:
        (directory / TRANSCRIPT_NAME).write_text("{}\n", encoding="utf-8")
    if inbox is not None:
        (directory / INBOX_NAME).write_text(inbox, encoding="utf-8")
    if origin is not None:
        (directory / ORIGIN_NAME).write_text(json.dumps({"origin": origin}), encoding="utf-8")
    if marker_bytes is not None:
        (directory / ORIGIN_NAME).write_bytes(marker_bytes)
    if sidecar is not None:
        (directory / TITLE_SIDECAR).write_text(json.dumps(sidecar), encoding="utf-8")
    if stamp is not None:
        for entry in directory.iterdir():
            os.utime(entry, (stamp, stamp))
    return directory


def _oracle_latest(config_dir: Path) -> str | None:
    """``resume_dir``'s pre-change rule, verbatim, as the identity oracle.

    Deliberately a COPY of the removed implementation (glob, the activity clock,
    ``is_user_session``, and ``max`` over ``(activity, _reverse_name(name))``)
    rather than a description of it. If the new rule and this disagree on any
    store below, the lane has changed which conversation reopens.
    """
    from local_operator.resume import _reverse_name, is_user_session
    from local_operator.session.retention import session_activity

    sessions = config_dir / "sessions"
    candidates = [
        (activity, path)
        for path in sessions.glob("*")
        if (activity := session_activity(path)) is not None and is_user_session(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], _reverse_name(item[1].name)))[1].name


def _selected(config_dir: Path) -> str | None:
    """The id ``@latest`` resolves to, or ``None`` where the oracle found none."""
    try:
        return resume_dir(config_dir, RESUME_LATEST).name
    except ResumeNotFound:
        return None


def _assert_same_as_oracle(config_dir: Path) -> str | None:
    oracle = _oracle_latest(config_dir)
    chosen = _selected(config_dir)
    assert chosen == oracle, f"@latest picked {chosen!r}, the old rule picked {oracle!r}"
    if oracle is None:
        raise AssertionError("fixture: the store has no latest session to compare")
    return chosen


def _plain(root: Path) -> None:
    _session(root, "a" * 12, stamp=1_000.0)
    _session(root, "b" * 12, stamp=2_000.0)


def _ties(root: Path) -> None:
    """Equal stamps: the ascending id wins, which ``max`` gets from
    ``_reverse_name`` and the scan from its ``(-activity, name)`` sort."""
    for name in ("c" * 12, "a" * 12, "b" * 12):
        _session(root, name, stamp=5_000.0)


def _archived_newest(root: Path) -> None:
    """The archived axis, where a "tidy" rewrite would differ.

    The glob never consulted the archive index, so an archived session has
    always been eligible for ``@latest`` even though the picker does not draw
    it. Identity here means the archived session still wins.
    """
    from local_operator.session.archived import ARCHIVED_FILE

    _session(root, "a" * 12, stamp=1_000.0)
    newest = _session(root, "b" * 12, stamp=9_000.0)
    (root / ARCHIVED_FILE).write_text(json.dumps(["b" * 12]), encoding="utf-8")
    assert newest.is_dir()


def _hidden_newest(root: Path) -> None:
    """A delegated run finishing after the parent's last turn must not steal
    the reopen — the defect the ``@latest`` rule exists for."""
    _session(root, "a" * 12, stamp=1_000.0)
    _session(root, "b" * 12, origin="subagent", stamp=9_000.0)
    _session(root, "c" * 12, origin="agent-shell", stamp=8_000.0)


def _hidden_only(root: Path) -> None:
    for index in range(4):
        _session(root, f"sub{index:09x}", origin="subagent", stamp=9_000.0 + index)


def _inbox_only(root: Path) -> None:
    """A spooled peer message is a reason to come back: the session has no
    transcript at all and must still be the newest row (R3-5)."""
    _session(root, "a" * 12, stamp=1_000.0)
    _session(root, "b" * 12, transcript=False, inbox='{"from": "peer"}\n', stamp=9_000.0)


def _missing_title_sidecar(root: Path) -> None:
    """Every session here predates the title sidecar, so ``stored_session_title``
    has to fall back to the opener — and the NEWEST one has no sidecar while an
    older one does, which is the shape a sidecar-keyed rewrite would reorder."""
    _session(root, "a" * 12, stamp=1_000.0, sidecar={"text": "named"})
    _session(root, "b" * 12, stamp=9_000.0)
    assert not (root / "sessions" / ("b" * 12) / TITLE_SIDECAR).exists()


def _corrupt_and_unreadable_markers(root: Path) -> None:
    """Both fail-safes at once: a truncated marker parses to ``""`` (the user's
    own session, deliberately) and sits newest, beside one cut inside a
    multi-byte character and one cut mid-token."""
    _session(root, "a" * 12, stamp=1_000.0)
    _session(root, "b" * 12, marker_bytes=b'{"origin": "suba', stamp=9_000.0)
    _session(root, "c" * 12, marker_bytes=b'{"origin": "subagent", "l": "caf\xc3', stamp=8_000.0)


def _visible_origins(root: Path) -> None:
    """Fork and agent-workstream are the user's OWN work, so they rank like any
    other conversation rather than being skipped."""
    _session(root, "a" * 12, origin="fork", stamp=1_000.0)
    _session(root, "b" * 12, origin="agent-workstream", stamp=9_000.0)
    _session(root, "c" * 12, origin="subagent", stamp=10_000.0)


def _never_active(root: Path) -> None:
    """Directories that are neither a session nor a delegated run: no activity
    file at all, with and without a marker, newest by name."""
    _session(root, "a" * 12, stamp=1_000.0)
    _session(root, "zzz-empty" + "0" * 3, transcript=False)
    _session(root, "zzz-marked" + "0" * 2, transcript=False, origin="subagent")


def _dot_named(root: Path) -> None:
    """A dot-named directory holding a transcript. ``scandir`` always lists it,
    and it is the same rule the picker and the scan use."""
    _session(root, "a" * 12, stamp=1_000.0)
    _session(root, ".hidden-session", stamp=9_000.0)


def _symlink_to_a_session(root: Path) -> None:
    target = _session(root, "a" * 12, stamp=1_000.0)
    link = root / "sessions" / ("b" * 12)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:  # pragma: no cover - platforms without symlink permission
        pytest.skip("symlinks unavailable")


SHAPES: dict[str, Callable[[Path], None]] = {
    "plain": _plain,
    "ties": _ties,
    "archived-newest": _archived_newest,
    "hidden-newest": _hidden_newest,
    "inbox-only": _inbox_only,
    "missing-title-sidecar": _missing_title_sidecar,
    "corrupt-markers": _corrupt_and_unreadable_markers,
    "visible-origins": _visible_origins,
    "never-active": _never_active,
    "dot-named": _dot_named,
    "symlinked": _symlink_to_a_session,
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_latest_selects_the_same_session_as_the_rule_it_replaced(
    tmp_path: Path, shape: str
) -> None:
    """THE IDENTITY CONTRACT, shape by shape, against the verbatim old rule."""
    SHAPES[shape](tmp_path)
    _assert_same_as_oracle(tmp_path)


def test_a_hidden_only_store_has_nothing_to_resume(tmp_path: Path) -> None:
    """A delegated run is not a conversation the user can come back to: the
    store is non-empty on disk and ``@latest`` must still refuse rather than
    reopening the reviewer."""
    _hidden_only(tmp_path)
    assert _selected(tmp_path) is None
    with pytest.raises(ResumeNotFound, match="no previous session to resume"):
        resume_dir(tmp_path, RESUME_LATEST)


def test_a_store_that_is_not_there_is_an_empty_answer_not_an_error(tmp_path: Path) -> None:
    """A fresh install, a config dir a probe invented: the same answer the
    explicit-id path gives for the same condition."""
    with pytest.raises(ResumeNotFound):
        resume_dir(tmp_path / "absent", RESUME_LATEST)


def test_latest_is_the_pickers_first_row_on_a_store_it_draws(tmp_path: Path) -> None:
    """The originating parity claim (R3-5): on a store with nothing archived or
    hidden, ``@latest`` IS ``recent_session_rows``' first row — one clock, one
    tie-break, one answer."""
    _ties(tmp_path)
    _session(tmp_path, "d" * 12, origin="subagent", stamp=9_000.0)
    picked = _assert_same_as_oracle(tmp_path)
    assert picked == "a" * 12
    rows = resume_mod.recent_sessions(tmp_path)
    assert [name for name, _ in rows][0] == picked


def test_a_repeated_resolution_does_not_move_the_answer(tmp_path: Path) -> None:
    """The verdict cache is warm after the first call; a warm call must return
    the same session as the cold one, or the cache has a second opinion."""
    _hidden_newest(tmp_path)
    first = _selected(tmp_path)
    for _ in range(4):
        assert _selected(tmp_path) == first


# ---------------------------------------------------------------------------
# Work done: the walk is the invalidation check, and nothing else
# ---------------------------------------------------------------------------


def _counting(names: tuple[str, ...] = ("stat", "lstat", "scandir")) -> Any:
    """Count ``os`` filesystem calls made inside the context (audit hook)."""

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

    return Counter()


def _store(root: Path, *, users: int, hidden: int) -> None:
    for index in range(users):
        _session(root, f"user{index:08x}", stamp=1_000.0 + index)
    for index in range(hidden):
        _session(root, f"sub{index:09x}", origin="subagent", stamp=500.0)


def _warm(root: Path) -> None:
    """Two resolutions: the first builds the verdict cache and is the cold
    start's revalidating scan, the second is the poll the claim is about."""
    _selected(root)
    _selected(root)


class TestTheWalkIsTheInvalidationCheck:
    def test_a_warm_resolution_issues_only_the_check_and_never_reads_a_marker(
        self, tmp_path: Path
    ) -> None:
        """THE STRUCTURAL CLAIM, as an exact count.

        The terms are: one ``scandir`` (the store enumeration — that IS how a
        session directory added since the last scan is noticed), per
        USER-VISIBLE session one marker stat (the verdict the scan cannot
        memoise for an unmarked directory) and two activity stats (the ranking
        clock deciding whether it is a session at all), and the archive index.
        The 40 hidden directories contribute NOTHING, and no ``origin.json`` is
        re-read, because the verdict cache is warm.

        The old implementation's counts on this fixture were 46 ``stat`` per
        resolution and one marker READ per directory, from a ``glob`` that also
        enumerated the store a second time.
        """
        _store(tmp_path, users=6, hidden=40)
        _warm(tmp_path)

        reads: list[Path] = []
        real_read = resume_mod._session_origin_read

        def counting_read(session_dir: Path) -> tuple[str, bool]:
            reads.append(session_dir)
            return real_read(session_dir)

        resume_mod._session_origin_read = counting_read  # type: ignore[assignment]
        try:
            with _counting() as counter:
                chosen = resume_dir(tmp_path, RESUME_LATEST)
        finally:
            resume_mod._session_origin_read = real_read  # type: ignore[assignment]

        assert chosen.name == "user00000005"
        assert reads == [], "a warm resolution re-read an origin marker"
        assert counter.counts["scandir"] == 1, "the glob's second enumeration is back"
        assert counter.counts["lstat"] == 0
        # 6 marker checks + 2 activity files each + the archive index.
        assert counter.counts["stat"] == 6 + 2 * 6 + 1, counter.counts

    def test_the_projection_adds_no_call_of_its_own(self, tmp_path: Path) -> None:
        """ZERO BEYOND THE CHECK, stated literally: resolving ``@latest`` costs
        exactly what the invalidation check — a bare ``_scan_sessions`` call —
        costs, so nothing in this path is a second walk wearing a cache."""
        _store(tmp_path, users=8, hidden=60)
        _warm(tmp_path)
        with _counting() as check:
            _scan_sessions(tmp_path, 1, include_archived=True)
        with _counting() as latest:
            resume_dir(tmp_path, RESUME_LATEST)
        assert latest.counts == check.counts, (latest.counts, check.counts)
        assert latest.counts["scandir"] == 1

    def test_the_hidden_population_costs_zero_however_large_it_gets(self, tmp_path: Path) -> None:
        """FLATNESS, the property that separates "tracks the user's sessions"
        from "tracks the store": 300 more delegated runs must not move the
        second resolution's syscall counts AT ALL.

        Asserted as equality of the whole count map rather than a bound on the
        total, because a bound would let the per-directory slope creep back in
        under a constant that grows with it.
        """
        _store(tmp_path, users=5, hidden=20)
        _warm(tmp_path)
        with _counting() as before:
            resume_dir(tmp_path, RESUME_LATEST)

        for index in range(300):
            _session(
                tmp_path,
                f"grow{index:08x}",
                origin="subagent",
                stamp=400.0,
            )
        _selected(tmp_path)  # the new directories are discovered here
        _selected(tmp_path)
        with _counting() as after:
            chosen = resume_dir(tmp_path, RESUME_LATEST)

        assert chosen.name == "user00000004", "the listing moved while cost was measured"
        assert after.counts == before.counts, (before.counts, after.counts)

    def test_a_store_of_nothing_but_delegated_runs_issues_no_stat_at_all(
        self, tmp_path: Path
    ) -> None:
        """The extreme the claim implies: with no user session to re-validate,
        resolution is one ``scandir`` and ZERO stats — every directory is
        skipped whole from the readdir batch alone."""
        _store(tmp_path, users=0, hidden=300)
        _warm(tmp_path)
        with _counting() as counter:
            with pytest.raises(ResumeNotFound):
                resume_dir(tmp_path, RESUME_LATEST)
        assert counter.counts == {"stat": 0, "lstat": 0, "scandir": 1}, counter.counts

    def test_the_newest_marker_is_never_served_from_the_cache(self, tmp_path: Path) -> None:
        """The one cache this path trusts may not hide a session that appeared
        since the last scan: absence is deliberately NOT memoised, so a
        delegated run created mid-cycle is skipped on this very resolution
        rather than after an epoch."""
        _store(tmp_path, users=3, hidden=3)
        _warm(tmp_path)
        _session(tmp_path, "late00000000", origin="subagent", stamp=99_000.0)
        assert _selected(tmp_path) == "user00000002"
