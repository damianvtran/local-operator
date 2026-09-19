"""Bounds on the ``@`` picker's filesystem scan.

``scan_directory`` runs from ``_sync_picker`` on EVERY keystroke, with no
debounce, no cancellation and no generation counter. What makes that safe is
not that the function is fast on this machine — it is that the function scans
exactly ONE directory, which is a fact about the algorithm rather than about
the hardware. So the guard below counts syscalls instead of measuring seconds;
see ``test_exactly_one_directory_is_scanned_per_keystroke``.
"""

from __future__ import annotations

import os
from typing import Any, Callable
from unittest import mock

from local_operator import references
from local_operator.references import (
    SCAN_CANDIDATE_LIMIT,
    scan_directory,
    scan_directory_report,
)


def _counting(names: tuple[str, ...] = ("stat", "lstat", "scandir")) -> Any:
    """Count ``os`` filesystem calls made inside the context.

    Lifted from ``tests/unit/session/test_catalog_scan_cost.py:91-122``
    deliberately: one counting idiom for scan-cost guards means a reader who
    has understood that one has understood this one.
    """

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


def _names(choices) -> list[str]:
    return [choice.name for choice in choices]


def test_scan_lists_one_directory_level_only(tmp_path):
    (tmp_path / "top.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("x", encoding="utf-8")

    names = _names(scan_directory("", str(tmp_path)))

    assert "top.txt" in names
    assert "sub/" in names
    assert "deep.txt" not in names


def test_scan_skips_prune_dirs_and_dotfiles(tmp_path):
    """The walker's own exclusion vocabulary, so ``@`` agrees with ``grep`` and
    ``glob`` about what is worth showing."""
    for name in ("node_modules", ".git", "__pycache__", "dist", "build", ".venv"):
        (tmp_path / name).mkdir()
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    (tmp_path / "visible.py").write_text("x", encoding="utf-8")

    names = _names(scan_directory("", str(tmp_path)))

    assert names == ["visible.py"]


def test_scan_honours_gitignore_for_the_listed_directory(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    (tmp_path / "ignored.log").write_text("x", encoding="utf-8")
    (tmp_path / "kept.log").write_text("x", encoding="utf-8")

    names = _names(scan_directory("", str(tmp_path)))

    assert "kept.log" in names
    assert "ignored.log" not in names


def test_scan_returns_empty_on_oserror(tmp_path, monkeypatch):
    """Textual turns an escaped error into a full-screen crash, and this runs on
    the keystroke path — so an unreadable directory costs an empty list."""

    def boom(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(os, "scandir", boom)

    assert scan_directory("", str(tmp_path)) == []


def test_scan_caps_the_candidate_list(tmp_path):
    """A pathological directory must not make the SORT the cost."""
    for index in range(SCAN_CANDIDATE_LIMIT + 50):
        (tmp_path / f"entry_{index:05d}.txt").write_text("x", encoding="utf-8")

    assert len(scan_directory("", str(tmp_path))) == SCAN_CANDIDATE_LIMIT


def test_a_directory_entry_is_marked_with_a_trailing_slash(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    names = _names(scan_directory("", str(tmp_path)))

    assert "folder/" in names
    assert "file.txt" in names


def test_exactly_one_directory_is_scanned_per_keystroke(tmp_path):
    """The real guarantee behind the keystroke budget, asserted STRUCTURALLY.

    A wall-clock ceiling here would be a bet on machine load — and a bet
    calibrated on a laptop is worthless on a loaded CI box, which is why
    ``AGENTS.md``'s "prefer a structural invariant to a numeric one" is
    binding. ``counts["scandir"] == 1`` is a fact about the algorithm instead:
    it cannot flake, and it fails deterministically the moment someone reaches
    for a walk. The measured stakes: one ``scandir`` is 0.04 ms, an
    ignore-aware walk of this repo is 68 ms and of a real workspace 8556 ms —
    per keystroke, with no debounce to hide behind.
    """
    deep = tmp_path
    for level in range(6):
        deep = deep / f"level_{level}"
        deep.mkdir()
        for index in range(20):
            (deep / f"file_{index}.txt").write_text("x", encoding="utf-8")
    for index in range(20):
        (tmp_path / f"top_{index}.txt").write_text("x", encoding="utf-8")

    with _counting() as counter:
        choices = scan_directory("", str(tmp_path))

    assert choices, "the fixture must produce rows, or the count below is vacuous"
    assert counter.counts["scandir"] == 1


def test_the_cap_bounds_the_per_entry_work_not_just_the_row_count(tmp_path):
    """The cap must bound the EXPENSIVE work, asserted structurally.

    ``scan_directory`` used to sort every entry and then ``break`` at the cap,
    so ``SCAN_CANDIDATE_LIMIT`` bounded the returned list while the per-entry
    work still scaled with the directory. Measured at 10,000 entries: 20-22 ms
    per keystroke against a 16.7 ms frame budget, with a 40-entry directory at
    0.33 ms.

    THE BOUND IS COUNTED, NOT TIMED, and that is deliberate — ``AGENTS.md``'s
    "prefer a structural invariant to a numeric one" is binding here, and the
    sibling test ``test_exactly_one_directory_is_scanned_per_keystroke`` already
    counts ``scandir`` for the same reason. A wall-clock ceiling on a laptop is
    worthless on a loaded CI box; "at most one ``stat`` per RETURNED row" is a
    fact about the algorithm that cannot flake.

    ``_entry_detail`` is the right thing to count: it is the per-row builder,
    and it makes a ``DirEntry.stat()`` syscall for the size and mtime columns.
    Before the fix it ran once per ENTRY; now it runs only for the rows that
    survive the cap.

    NOTE it is counted by wrapping the function rather than through
    ``_counting()``: that helper patches ``os.stat``/``os.lstat``, while
    ``_entry_detail`` calls the ``stat`` METHOD on the ``DirEntry`` the kernel
    already returned. Asserting on ``counter.counts["stat"]`` here measures
    nothing — it was 4 against a limit of 2000, true no matter what the code
    does — which is exactly the vacuous-guard shape ``AGENTS.md`` warns about.
    """
    for index in range(SCAN_CANDIDATE_LIMIT * 5):
        (tmp_path / f"entry_{index:06d}.txt").write_text("x", encoding="utf-8")
    calls = 0
    real_detail = references._entry_detail

    def counting_detail(entry, is_dir):
        nonlocal calls
        calls += 1
        return real_detail(entry, is_dir)

    with _counting() as counter:
        with mock.patch.object(references, "_entry_detail", counting_detail):
            choices = scan_directory("", str(tmp_path))

    assert len(choices) == SCAN_CANDIDATE_LIMIT
    # The whole point: 10,000 entries on disk, at most 2,000 rows BUILT.
    assert calls <= SCAN_CANDIDATE_LIMIT
    # And the one-directory guarantee still holds alongside it.
    assert counter.counts["scandir"] == 1


def test_the_cap_keeps_the_alphabetically_first_rows(tmp_path):
    """Capping before the sort must not change WHICH rows are returned.

    ``heapq.nsmallest`` replaced sort-then-truncate, so this pins that the two
    agree: the picker's ordering is a user-visible contract, and a cap that
    returned an arbitrary 2,000 would be a different feature.
    """
    for index in range(SCAN_CANDIDATE_LIMIT + 500):
        (tmp_path / f"entry_{index:06d}.txt").write_text("x", encoding="utf-8")

    names = _names(scan_directory("", str(tmp_path)))

    assert names == sorted(names)
    assert names[0] == "entry_000000.txt"
    assert names[-1] == f"entry_{SCAN_CANDIDATE_LIMIT - 1:06d}.txt"


def test_a_dotenv_entry_is_flagged_by_name_not_only_by_directory(tmp_path):
    """``alert`` marks a row the operator should look twice at.

    The same prefix-only gap the resolver had: a name ENDING in ``.env`` was
    not flagged. Pinned here too because the picker's paint and the resolver's
    approval gate must agree about what counts as sensitive — a row the picker
    shows as ordinary and the resolver then prompts for is a confusing pair.
    """
    for name in ("workspace.env", "prod.env", ".env", ".env.local"):
        (tmp_path / name).write_text("TOKEN=placeholder\n", encoding="utf-8")
    (tmp_path / "ordinary.txt").write_text("x", encoding="utf-8")

    alerts = {choice.name: choice.alert for choice in scan_directory("", str(tmp_path))}

    assert alerts["workspace.env"] is True
    assert alerts["prod.env"] is True
    assert alerts["ordinary.txt"] is False
    # Dotfiles are excluded from the picker entirely, so `.env` itself is not a
    # row here — the resolver's gate is what covers a hand-typed `@.env`.
    assert ".env" not in alerts


def test_the_scan_reports_what_its_own_cap_kept_out(tmp_path):
    """D6: the count is of the DIRECTORY, and the cap was hiding 50 of its entries.

    `scan_directory` dropped the fact that it had refused entries, so the
    picker's overflow row described the capped set as if it were the whole
    directory: a 2500-entry directory read ``… 1992 more`` and never mentioned
    the 500 it had not looked at. The number is exact rather than an estimate and
    costs no extra syscall — the cap is applied after the cheap pass has already
    enumerated every listable entry, so it was a number this function had in hand
    and threw away.
    """
    for index in range(SCAN_CANDIDATE_LIMIT + 50):
        (tmp_path / f"entry_{index:05d}.txt").write_text("x", encoding="utf-8")

    rows, unlisted = scan_directory_report("", str(tmp_path))

    assert len(rows) == SCAN_CANDIDATE_LIMIT, "premise: the cap is still the cap"
    assert unlisted == 50, "the entries the cap refused are not accounted for"

    # Nothing under the cap over-reports, and the plain call is unchanged: its
    # forty existing callers must not have had to learn about a second return.
    # One more file, sorting LAST, so the kept 2000 are the same 2000 and the
    # refusal count moves from 50 to 51 — which is the point of counting here
    # rather than inside the picker, where the extra entry never arrives.
    (tmp_path / "zzz_extra.txt").write_text("x", encoding="utf-8")
    assert scan_directory("", str(tmp_path)) == rows
    assert scan_directory_report("", str(tmp_path)) == (rows, 51)
