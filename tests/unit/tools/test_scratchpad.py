"""The ``scratchpad://`` URL grammar, one assertion per row.

The grammar is a security boundary as much as a convenience: a scratchpad URL is
resolved by ``read``/``write``/``edit`` against a directory OUTSIDE the working
directory, so the containment rule here is the only thing standing between an
agent's scratch folder and the rest of the disk. Each row of the table below
therefore asserts its own sentence, not just "it raises".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import scratchpad as scratchpad_module
from local_operator.scratchpad import (
    SCRATCHPAD_DIRNAME,
    SCRATCHPAD_NAMESPACE,
    SCRATCHPAD_SCHEME,
    SCRATCHPAD_UNAVAILABLE,
    ScratchpadPathError,
    parse_scratchpad_url,
    scratchpad_root,
)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A real scratchpad root, because the parser resolves against disk."""
    session_dir = tmp_path / "sessions" / "abc123"
    scratchpad = session_dir / SCRATCHPAD_DIRNAME
    scratchpad.mkdir(parents=True)
    (scratchpad / "runs").mkdir()
    (scratchpad / "perf.md").write_text("1\n", encoding="utf-8")
    (scratchpad / "runs" / "deep.csv").write_text("a,b\n", encoding="utf-8")
    return scratchpad


def test_the_name_lives_in_exactly_one_constant() -> None:
    """One token names the protocol; the URL prefix and the on-disk folder are
    both derived from it, so a rename cannot move the address without moving the
    directory with it. This assertion is what makes that coupling load-bearing.
    """
    assert SCRATCHPAD_SCHEME == f"{SCRATCHPAD_NAMESPACE}://"
    assert SCRATCHPAD_DIRNAME == SCRATCHPAD_NAMESPACE
    assert SCRATCHPAD_UNAVAILABLE.startswith(SCRATCHPAD_SCHEME)


def test_the_parser_reads_the_scheme_from_the_constant(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The reviewer's own probe for the partial-rename defect: set the constants
    to a new name and check the parser follows THEM rather than a hardcoded
    literal. A literal in the parser leaves every URL rejected by a rename, with
    an error that reads as nonsense ("not a scratchpad:// URL" printed beside a
    ``scratchpad://`` URL).
    """
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_NAMESPACE", "pad")
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_SCHEME", "pad://")

    renamed = parse_scratchpad_url("pad://perf.md", root)
    assert renamed.path == root / "perf.md"

    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("scratchpad://perf.md", root)
    assert "not a pad:// URL" in str(caught.value)


@pytest.mark.parametrize(
    ("url", "relative", "directory"),
    [
        # The bare root and its explicit no-op spelling: both list.
        ("scratchpad://", "", True),
        ("scratchpad://.", "", True),
        # Trailing slash is directory syntax, whether or not the directory exists.
        ("scratchpad://runs/", "runs", True),
        ("scratchpad://runs/deeper/", "runs/deeper", True),
        # No trailing slash: a file if it exists, a listing if it is a dir.
        ("scratchpad://perf.md", "perf.md", False),
        ("scratchpad://runs", "runs", False),
        ("scratchpad://runs/deep.csv", "runs/deep.csv", False),
        # '//' and './' segments are no-ops, not escapes.
        ("scratchpad://./perf.md", "perf.md", False),
        ("scratchpad://runs//deep.csv", "runs/deep.csv", False),
        # Unquoted BEFORE splitting, so an encoded separator becomes a segment
        # boundary and is validated as one.
        ("scratchpad://runs%2Fdeep.csv", "runs/deep.csv", False),
        # An encoded '?' names a file that contains one; the raw one is refused.
        ("scratchpad://a%3Fb.md", "a?b.md", False),
        ("scratchpad://a%23b.md", "a#b.md", False),
        # On POSIX a backslash is an ordinary filename character; on Windows
        # pathlib splits on it and containment still governs.
        ("scratchpad://runs\\deep.csv", "runs\\deep.csv", False),
    ],
)
def test_accepted_urls_resolve_under_the_root(
    root: Path, url: str, relative: str, directory: bool
) -> None:
    target = parse_scratchpad_url(url, root)
    assert target.path == (root / relative if relative else root)
    assert target.directory is directory


@pytest.mark.parametrize(
    ("url", "sentence"),
    [
        # urlsplit puts '/abs' in the path with an empty netloc, so the joined
        # string starts with '/' and is refused as absolute.
        ("scratchpad:///etc/passwd", "absolute paths are not allowed"),
        ("scratchpad://..", "'..' segments are not allowed"),
        ("scratchpad://../escape.md", "'..' segments are not allowed"),
        ("scratchpad://runs/../../escape.md", "'..' segments are not allowed"),
        ("scratchpad://.hidden.md", "dotfiles are not listed and cannot be read"),
        ("scratchpad://runs/.env", "dotfiles are not listed and cannot be read"),
        # A query or fragment would silently truncate the name, so the file the
        # agent asked for and the file it got would differ with no signal.
        ("scratchpad://perf.md?x=1", "'?' and '#' open a query or fragment"),
        ("scratchpad://perf.md#top", "'?' and '#' open a query or fragment"),
    ],
)
def test_refused_urls_say_what_is_wrong(root: Path, url: str, sentence: str) -> None:
    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url(url, root)
    assert sentence in str(caught.value)
    assert url in str(caught.value)


def test_a_symlink_out_of_the_root_is_refused(root: Path, tmp_path: Path) -> None:
    """The containment check is the authority, not the segment rules: a link
    inside the folder is invisible to the grammar and must still lose."""
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret\n", encoding="utf-8")
    (root / "leak.md").symlink_to(secret)

    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("scratchpad://leak.md", root)
    assert "escapes the scratchpad directory" in str(caught.value)


def test_a_symlink_to_another_scratchpad_file_is_accepted(root: Path) -> None:
    """The containment rule is about WHERE the target is, not how it got there:
    a link between two files inside the folder resolves."""
    (root / "alias.md").symlink_to(root / "perf.md")

    target = parse_scratchpad_url("scratchpad://alias.md", root)
    assert target.path == (root / "perf.md").resolve()


def test_a_directory_symlink_out_of_the_root_is_refused_for_its_child(
    root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.md").write_text("x\n", encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("scratchpad://link/x.md", root)
    assert "escapes the scratchpad directory" in str(caught.value)


def test_another_scheme_is_refused_by_the_parser_itself(root: Path) -> None:
    """Callers check the prefix first; this is the belt to that braces, so a
    direct caller cannot address the filesystem through another scheme."""
    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("skill://demo", root)
    assert f"not a {SCRATCHPAD_SCHEME} URL" in str(caught.value)


# ---------------------------------------------------------------------------
# scratchpad_root: which directories may hold scratch at all
# ---------------------------------------------------------------------------


def test_scratchpad_root_is_the_folder_under_a_session_directory(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "abc123"
    assert scratchpad_root(session_dir) == session_dir / SCRATCHPAD_DIRNAME
    # A string path is accepted: the ToolContext field is a string.
    assert scratchpad_root(str(session_dir)) == session_dir / SCRATCHPAD_DIRNAME


def test_scratchpad_root_is_none_for_an_agent_directory(tmp_path: Path) -> None:
    """``--train`` keeps its transcript in ``<config_dir>/agents/<id>/``, and
    ``AgentRegistry.export_agent`` publishes that directory whole — a scratch
    folder there would ship to strangers."""
    assert scratchpad_root(tmp_path / "agents" / "abc123") is None


@pytest.mark.parametrize(
    "value",
    [None, "", "abc123", "/tmp/not-a-session-dir", Path("relative/dir")],
)
def test_scratchpad_root_is_none_for_anything_that_is_not_a_session_dir(value: object) -> None:
    assert scratchpad_root(value) is None  # type: ignore[arg-type]


def test_scratchpad_root_tolerates_a_non_path_argument() -> None:
    """A host that passed something path-like-but-not must lose its scratch
    area, not raise on the path every turn walks."""
    assert scratchpad_root(1234) is None  # type: ignore[arg-type]
