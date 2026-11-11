"""Unit tests for :mod:`local_operator.harness.tool_names`.

The pure matcher is the whole ballgame for dispatch hints; the table here
pins the cases the architect specified, including the longest-prefix tie
and the exact-name negative.
"""

from __future__ import annotations

import re

import pytest

from local_operator.harness.tool_names import near_miss_tool

CANDIDATES = ["bash", "read", "write", "grep"]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("bashLecture", "bash"),
        ("bashLectureLecture", "bash"),
        ("bashLectureLectureLecture", "bash"),
        ("bash-x", "bash"),
        ("bashFile", "bash"),
        ("readFile", "read"),
        ("writeFile", "write"),
        ("grepTool", "grep"),
    ],
)
def test_near_miss_matches_garbled_names(name: str, expected: str) -> None:
    assert near_miss_tool(name, CANDIDATES) == expected


@pytest.mark.parametrize(
    "name",
    [
        "bash",
        "read",
        "write",
        "grep",
        "bash_file",
        "read_file",
        "bash2",
        "mcp__linear_create_issue",
        "mcp__bashLecture",
        "ba",
        "x",
        "unknown_tool",
        "ghost",
        "",
    ],
)
def test_near_miss_rejects_exact_and_non_garbled(name: str) -> None:
    assert near_miss_tool(name, CANDIDATES) is None


def test_near_miss_longest_prefix_wins() -> None:
    candidates = ["bash", "bash_file", "read"]
    assert near_miss_tool("bash_fileLecture", candidates) == "bash_file"


def test_near_miss_rejects_remainder_starting_with_lowercase_or_digit() -> None:
    candidates = ["bash", "bash_file"]
    assert near_miss_tool("bash_file2", candidates) is None
    assert near_miss_tool("bash_filex", candidates) is None


def test_near_miss_ignores_short_and_mcp_candidates() -> None:
    candidates = ["ab", "mcp__linear", "bash"]
    assert near_miss_tool("abLecture", candidates) is None
    assert near_miss_tool("mcp__linearLecture", candidates) is None


def test_near_miss_filters_candidate_names_by_pattern() -> None:
    candidates = ["Bash", "bash-tool", "bash"]
    assert near_miss_tool("bashLecture", candidates) == "bash"
    assert near_miss_tool("BashLecture", candidates) is None


def test_candidate_pattern_matches_spec() -> None:
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    assert pattern.match("bash")
    assert pattern.match("read_file")
    assert not pattern.match("Bash")
    assert not pattern.match("bash-x")
    assert not pattern.match("_bash")
