import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from local_operator.tools import _get_git_ignored_files, get_current_directory_info


@pytest.fixture
def mock_file_system():
    """Mock file system with various file types and sizes"""
    mock_files = {
        ".": [("test.py", 100), ("doc.md", 200), ("image.png", 300), ("other.bin", 400)],
        "./subdir": [("code.js", 500), ("readme.txt", 600)],
    }

    def fake_join(*args):
        if len(args) == 2:
            # Mimic the original behavior: if the first arg is ".", return the second;
            # otherwise join with "/"
            return f"{args[0]}/{args[1]}" if args[0] != "." else args[1]
        return "/".join(args)

    def mock_stat_size(path, *args, **kwargs):
        file_name = os.path.basename(str(path))
        for dir_files in mock_files.values():
            for fname, size in dir_files:
                if fname == file_name:
                    m = MagicMock()
                    m.st_size = size
                    return m
        raise FileNotFoundError()

    def fake_path_stat(self, *args, **kwargs):
        # Use the instance (self) to derive the file path for stat simulation.
        return mock_stat_size(self, *args, **kwargs)

    # Patch os.walk, os.stat, and pathlib.Path.stat as before,
    # but only patch os.path.join inside local_operator.tools so that other patches
    # (e.g. for open) are not affected.
    with (
        patch(
            "os.walk",
            return_value=[
                (".", ["subdir", ".git"], ["test.py", "doc.md", "image.png", "other.bin"]),
                ("./subdir", [], ["code.js", "readme.txt"]),
            ],
        ),
        patch("os.stat", side_effect=mock_stat_size),
        patch("pathlib.Path.stat", new=fake_path_stat),
        patch("local_operator.tools.os.path.join", side_effect=fake_join),
    ):
        yield mock_files


def test_get_git_ignored_files_in_repo():
    """Test getting ignored files from a .gitignore file in a repo context"""
    m = mock_open(read_data="ignored1.txt\nsubdir/ignored2.txt\n")
    with patch("builtins.open", m):
        ignored = _get_git_ignored_files(".gitignore")
    assert ignored == {"ignored1.txt", "subdir/ignored2.txt"}


def test_get_git_ignored_files_no_repo():
    """Test getting ignored files when .gitignore is not present"""
    with patch("builtins.open", side_effect=FileNotFoundError()):
        ignored = _get_git_ignored_files(".gitignore")
    assert ignored == set()


def test_get_current_directory_info(mock_file_system):
    """Test indexing directory with various file types when no .gitignore is present"""
    # Simulate a non-git environment by having .gitignore not found.
    with patch("builtins.open", side_effect=FileNotFoundError()):
        index = get_current_directory_info()

    assert len(index) == 2  # Root and subdir

    # Check root directory
    assert sorted(index["."]) == [
        ("doc.md", "doc", 200),
        ("image.png", "image", 300),
        ("other.bin", "other", 400),
        ("test.py", "code", 100),
    ]

    # Check subdirectory (normalized key)
    assert sorted(index["subdir"]) == [("code.js", "code", 500), ("readme.txt", "doc", 600)]


def test_index_current_directory_with_git_ignored(mock_file_system):
    """Test indexing directory respects .gitignore glob patterns"""
    # Simulate a .gitignore that ignores files matching 'ignored1.txt' in the root
    # and 'subdir/ignored2.txt' in the subdirectory by patching open and os.walk.
    with (
        patch("builtins.open", new=mock_open(read_data="ignored1.txt\nsubdir/ignored2.txt\n")),
        patch(
            "os.walk",
            return_value=[
                (
                    ".",
                    ["subdir", ".git"],
                    ["test.py", "doc.md", "image.png", "other.bin", "ignored1.txt"],
                ),
                ("subdir", [], ["code.js", "readme.txt", "ignored2.txt"]),
            ],
        ),
    ):
        index = get_current_directory_info()

    # Verify ignored files are not included
    all_files = []
    for files in index.values():
        all_files.extend(f[0] for f in files)

    assert "ignored1.txt" not in all_files
    assert "ignored2.txt" not in all_files


def test_get_current_directory_info_empty_directory(tmp_path, monkeypatch):
    """Test indexing an empty directory returns an empty dictionary."""
    # Change the current working directory to a new, empty temporary directory.
    monkeypatch.chdir(tmp_path)
    # Simulate no .gitignore file present.
    with patch("builtins.open", side_effect=FileNotFoundError()):
        index = get_current_directory_info()
    assert index == {}
