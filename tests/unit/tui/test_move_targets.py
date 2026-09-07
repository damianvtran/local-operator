"""Suggestion assembly, path expansion and validation behind ``/move``.

The ordering tests are the load-bearing ones. The tier order IS the feature —
a picker that led with `/tmp` and buried the directory the user's other session
is in would technically list everything and help with nothing — so the order is
pinned rather than left to the order the sources happen to be concatenated in.

Everything here runs against a real temporary filesystem rather than mocks:
these functions exist to answer questions about directories (does it exist, can
it be entered, what is inside it), and a mocked filesystem would assert that
the code calls the functions it calls rather than that it gives right answers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.tui.move_targets import (
    RECENTS_FILE,
    RECENTS_LIMIT,
    MoveError,
    child_dirs,
    complete_path,
    expand_path,
    filter_targets,
    format_label,
    looks_like_path,
    read_recents,
    remember_recent,
    suggest_targets,
    validate_target,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small directory tree: a home, a project with children, and a config."""
    (tmp_path / "home").mkdir()
    project = tmp_path / "home" / "project"
    project.mkdir()
    for name in ("alpha", "beta", ".hidden"):
        (project / name).mkdir()
    (project / "readme.md").write_text("not a directory")
    (tmp_path / "config").mkdir()
    return tmp_path


# -- labels and expansion ----------------------------------------------------


def test_a_directory_inside_home_is_labelled_relative_to_it(tree: Path) -> None:
    home = tree / "home"
    assert format_label(home / "project", home=home) == "~/project"
    assert format_label(home, home=home) == "~"


def test_a_directory_outside_home_keeps_its_absolute_path(tree: Path) -> None:
    """The absolute path is already the shortest honest rendering there."""
    assert format_label(tree / "config", home=tree / "home") == str(tree / "config")


def test_a_tilde_path_expands_to_the_users_home(monkeypatch, tree: Path) -> None:
    monkeypatch.setenv("HOME", str(tree / "home"))
    assert expand_path("~/project", cwd="/nowhere") == tree / "home" / "project"


def test_a_relative_path_resolves_against_the_SESSIONS_directory(tree: Path) -> None:
    """Not the process's cwd: a resumed session routinely differs from it, and
    `/move ../sibling` has to mean the sibling of what the band is showing."""
    project = tree / "home" / "project"
    assert expand_path("alpha", cwd=project) == project / "alpha"
    assert expand_path("..", cwd=project) == tree / "home"


def test_expansion_does_not_resolve_symlinks(tree: Path) -> None:
    """A user who moves into a symlink means the symlink; printing its target
    back on the band would read as the move having gone somewhere else."""
    link = tree / "home" / "current"
    link.symlink_to(tree / "home" / "project")
    assert expand_path(str(link), cwd=tree) == link


# -- validation --------------------------------------------------------------


def test_a_missing_directory_is_refused_by_name(tree: Path) -> None:
    with pytest.raises(MoveError, match="no such directory"):
        validate_target(tree / "nope")


def test_a_file_is_refused_as_not_a_directory(tree: Path) -> None:
    """A distinct sentence from "missing": the fix is different."""
    with pytest.raises(MoveError, match="not a directory"):
        validate_target(tree / "home" / "project" / "readme.md")


def test_an_unenterable_directory_is_refused_before_the_move(tree: Path) -> None:
    """Checked as well as existence: a directory that cannot be entered would
    let the move "succeed" and then fail every tool call made inside it."""
    locked = tree / "locked"
    locked.mkdir(mode=0o000)
    try:
        if os.access(locked, os.R_OK | os.X_OK):  # pragma: no cover - running as root
            pytest.skip("this user can enter a 0o000 directory")
        with pytest.raises(MoveError, match="permission denied"):
            validate_target(locked)
    finally:
        locked.chmod(0o755)


def test_a_usable_directory_validates_to_itself(tree: Path) -> None:
    assert validate_target(tree / "home") == tree / "home"


# -- recents -----------------------------------------------------------------


def test_recents_start_empty_and_round_trip(tree: Path) -> None:
    config = tree / "config"
    assert read_recents(config) == []
    remember_recent(config, "/one")
    remember_recent(config, "/two")
    assert read_recents(config) == ["/two", "/one"]


def test_remembering_a_directory_again_moves_it_to_the_front(tree: Path) -> None:
    config = tree / "config"
    for path in ("/one", "/two", "/one"):
        remember_recent(config, path)
    assert read_recents(config) == ["/one", "/two"]


def test_recents_are_capped(tree: Path) -> None:
    config = tree / "config"
    for index in range(RECENTS_LIMIT + 5):
        remember_recent(config, f"/dir{index}")
    assert len(read_recents(config)) == RECENTS_LIMIT


@pytest.mark.parametrize("body", ["not json at all", '{"not": "a list"}', "[1, 2, 3]"])
def test_a_corrupt_recents_file_reads_as_empty(tree: Path, body: str) -> None:
    """Best-effort by contract: this list has two live sources besides it, so
    every failure degrades to "nothing remembered" rather than to an error."""
    config = tree / "config"
    (config / RECENTS_FILE).write_text(body)
    assert read_recents(config) == []


def test_recents_are_written_atomically(tree: Path) -> None:
    """Same-directory temp then replace: a torn read would silently empty the
    list. Asserted by proving no temp file survives a successful write."""
    config = tree / "config"
    remember_recent(config, "/one")
    leftovers = [p.name for p in config.iterdir() if p.name != RECENTS_FILE]
    assert leftovers == []
    assert json.loads((config / RECENTS_FILE).read_text()) == ["/one"]


def test_an_unwritable_config_dir_costs_the_memory_not_the_move(tree: Path) -> None:
    """This runs after the user has been told the move succeeded."""
    config = tree / "readonly"
    config.mkdir(mode=0o500)
    try:
        remember_recent(config, "/one")  # must not raise
    finally:
        config.chmod(0o755)


# -- children ----------------------------------------------------------------


def test_children_are_directories_only_alphabetical_and_unhidden(tree: Path) -> None:
    """Hidden ones would otherwise crowd out every real child on a checkout."""
    project = tree / "home" / "project"
    assert [Path(p).name for p in child_dirs(project)] == ["alpha", "beta"]


def test_children_are_bounded(tree: Path) -> None:
    wide = tree / "wide"
    wide.mkdir()
    for index in range(40):
        (wide / f"d{index:02d}").mkdir()
    assert len(child_dirs(wide, limit=5)) == 5


def test_an_unreadable_directory_yields_no_children(tree: Path) -> None:
    assert child_dirs(tree / "nope") == []


# -- suggestion assembly -----------------------------------------------------


def test_the_current_directory_always_leads_and_is_marked(tree: Path) -> None:
    """A picker that does not show where you are makes "did that work?"
    unanswerable."""
    project = tree / "home" / "project"
    rows = suggest_targets(project, config_dir=tree / "config", home=tree / "home")
    assert rows[0].path == str(project)
    assert rows[0].kind == "current"
    assert rows[0].detail == "current"


def test_the_tiers_are_offered_in_order(tree: Path) -> None:
    """current -> recent -> home -> tmp -> parent -> children. The order IS the
    feature: places the user has worked beat structural fallbacks."""
    config = tree / "config"
    remember_recent(config, str(tree / "config"))
    rows = suggest_targets(tree / "home" / "project", config_dir=config, home=tree / "home")
    kinds = [row.kind for row in rows]
    assert kinds[0] == "current"
    assert "recent" in kinds
    # Every tier appears no earlier than the one before it.
    order = ["current", "recent", "home", "tmp", "parent", "child"]
    positions = [order.index(kind) for kind in kinds]
    assert positions == sorted(positions), kinds


def test_a_directory_offered_by_two_tiers_appears_once_in_the_higher_one(
    tree: Path,
) -> None:
    home = tree / "home"
    remember_recent(tree / "config", str(home))
    rows = suggest_targets(home / "project", config_dir=tree / "config", home=home)
    matching = [row for row in rows if row.path == str(home)]
    assert len(matching) == 1
    assert matching[0].kind == "recent"


def test_dedup_is_by_RESOLVED_path_not_by_spelling(tree: Path) -> None:
    """`/tmp` is a symlink to `/private/tmp` on macOS, so a running session
    reports the resolved form while the built-in row does not — a string dedup
    offered the same directory twice, in two spellings (seen on the real store).
    """
    real = tree / "home" / "project"
    link = tree / "home" / "alias"
    link.symlink_to(real)
    remember_recent(tree / "config", str(link))
    rows = suggest_targets(real, config_dir=tree / "config", home=tree / "home")
    resolved = [os.path.realpath(row.path) for row in rows]
    assert len(resolved) == len(set(resolved))


def test_a_remembered_directory_that_is_gone_is_not_offered(tree: Path) -> None:
    """A suggestion that fails validation the moment it is chosen is worse
    than no suggestion."""
    remember_recent(tree / "config", str(tree / "deleted"))
    rows = suggest_targets(tree / "home", config_dir=tree / "config", home=tree / "home")
    assert all(row.path != str(tree / "deleted") for row in rows)


def test_the_current_directory_is_offered_even_when_it_is_gone(tree: Path) -> None:
    """The one exception, and the case a user most needs `/move` for: hiding
    the row would leave the picker silently disagreeing with the band."""
    missing = tree / "home" / "vanished"
    rows = suggest_targets(missing, config_dir=tree / "config", home=tree / "home")
    assert rows[0].path == str(missing)


def test_suggestions_are_bounded(tree: Path) -> None:
    wide = tree / "wide"
    wide.mkdir()
    for index in range(50):
        (wide / f"d{index:02d}").mkdir()
    rows = suggest_targets(wide, config_dir=tree / "config", home=tree / "home", limit=6)
    assert len(rows) == 6


# -- filtering and completion ------------------------------------------------


def test_filtering_narrows_without_reordering(tree: Path) -> None:
    """A fixed query must not move a row out from under the cursor."""
    rows = suggest_targets(
        tree / "home" / "project", config_dir=tree / "config", home=tree / "home"
    )
    filtered = filter_targets(rows, "a")
    assert filtered == [row for row in rows if "a" in row.label.lower() or "a" in row.path.lower()]


def test_an_empty_filter_admits_everything(tree: Path) -> None:
    rows = suggest_targets(tree / "home", config_dir=tree / "config", home=tree / "home")
    assert filter_targets(rows, "   ") == rows


@pytest.mark.parametrize(
    "text,expected",
    [
        ("~/x", True),
        ("/etc", True),
        ("./here", True),
        ("../up", True),
        ("a/b", True),
        ("repos", False),
        ("", False),
    ],
)
def test_the_path_and_filter_modes_split_predictably(text: str, expected: bool) -> None:
    """One input, two jobs — so the rule has to be one a user can predict."""
    assert looks_like_path(text) is expected


def test_completion_lists_directories_under_a_typed_prefix(tree: Path) -> None:
    project = tree / "home" / "project"
    rows = complete_path(f"{project}/al", cwd=tree, home=tree / "home")
    assert [row.path for row in rows] == [str(project / "alpha")]


def test_a_trailing_separator_lists_inside_the_directory(tree: Path) -> None:
    """What makes a second tab descend rather than re-match siblings."""
    project = tree / "home" / "project"
    rows = complete_path(f"{project}/", cwd=tree, home=tree / "home")
    assert [Path(row.path).name for row in rows] == ["alpha", "beta"]


def test_completion_matches_case_insensitively(tree: Path) -> None:
    """A case-sensitive filter over a case-insensitive filesystem reports "no
    matches" for a directory the user can see."""
    project = tree / "home" / "project"
    rows = complete_path(f"{project}/AL", cwd=tree, home=tree / "home")
    assert [Path(row.path).name for row in rows] == ["alpha"]


def test_completion_hides_dotted_directories_unless_asked_for(tree: Path) -> None:
    project = tree / "home" / "project"
    assert not [r for r in complete_path(f"{project}/", cwd=tree) if ".hidden" in r.path]
    dotted = complete_path(f"{project}/.h", cwd=tree)
    assert [Path(row.path).name for row in dotted] == [".hidden"]


def test_completion_never_offers_a_file(tree: Path) -> None:
    project = tree / "home" / "project"
    rows = complete_path(f"{project}/read", cwd=tree, home=tree / "home")
    assert rows == []


def test_completing_a_missing_directory_is_empty_not_an_error(tree: Path) -> None:
    assert complete_path(f"{tree}/nowhere/x", cwd=tree) == []


def test_a_bare_word_completes_inside_the_sessions_directory(tree: Path) -> None:
    """Rather than being treated as a filesystem root."""
    project = tree / "home" / "project"
    rows = complete_path("alp", cwd=project, home=tree / "home")
    assert [row.path for row in rows] == [str(project / "alpha")]
