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

from local_operator.references import SCAN_CANDIDATE_LIMIT, scan_directory


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
