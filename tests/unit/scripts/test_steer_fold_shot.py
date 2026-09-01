"""The shot script must clear only what it seeded, never the operator's data.

``--workdir`` is a path the operator types, and this script runs with their full
permissions, so an unguarded recursive delete of that path is a data-loss bug
rather than a tidiness question: a mistyped ``--workdir .`` used to remove the
directory's whole contents before seeding. These tests pin the guard that keeps
the blast radius equal to the one subdirectory the script actually creates.
"""

import pytest

from scripts.steer_fold_shot import SEED_DIRNAME, WorkdirRefused, prepare_seed_dir


def test_sibling_data_in_the_workdir_survives_seeding(tmp_path):
    """Only ``<workdir>/child`` is cleared -- siblings are none of our business."""
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "data.txt").write_text("irreplaceable", encoding="utf-8")

    seed_dir = prepare_seed_dir(tmp_path)

    assert seed_dir == tmp_path / SEED_DIRNAME
    assert (precious / "data.txt").read_text(encoding="utf-8") == "irreplaceable"


def test_a_stale_seed_dir_of_ours_is_cleared(tmp_path):
    """Repeated runs must not stack generations, so our own artifacts go."""
    seed_dir = tmp_path / SEED_DIRNAME
    seed_dir.mkdir()
    (seed_dir / "transcript.jsonl").write_text("stale row\n", encoding="utf-8")

    prepare_seed_dir(tmp_path)

    assert not seed_dir.exists()


def test_foreign_files_in_the_seed_dir_are_refused(tmp_path):
    """A seed target we did not write is the signal that --workdir is wrong."""
    seed_dir = tmp_path / SEED_DIRNAME
    seed_dir.mkdir()
    (seed_dir / "notes.txt").write_text("someone's work", encoding="utf-8")

    with pytest.raises(WorkdirRefused, match="notes.txt"):
        prepare_seed_dir(tmp_path)

    assert (seed_dir / "notes.txt").read_text(encoding="utf-8") == "someone's work"


def test_force_clears_a_seed_dir_holding_foreign_files(tmp_path):
    """The escape hatch stays available, but only when asked for explicitly."""
    seed_dir = tmp_path / SEED_DIRNAME
    seed_dir.mkdir()
    (seed_dir / "notes.txt").write_text("someone's work", encoding="utf-8")

    prepare_seed_dir(tmp_path, force=True)

    assert not seed_dir.exists()


def test_a_seed_path_that_is_a_file_is_refused(tmp_path):
    """Never unlink a non-directory sitting where the seed dir belongs."""
    seed_path = tmp_path / SEED_DIRNAME
    seed_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(WorkdirRefused, match="not a directory"):
        prepare_seed_dir(tmp_path)

    assert seed_path.read_text(encoding="utf-8") == "not a directory"


def test_a_missing_workdir_is_created(tmp_path):
    """Seeding into a fresh path stays a one-liner for the caller."""
    workdir = tmp_path / "nested" / "workdir"

    seed_dir = prepare_seed_dir(workdir)

    assert workdir.is_dir()
    assert not seed_dir.exists()
