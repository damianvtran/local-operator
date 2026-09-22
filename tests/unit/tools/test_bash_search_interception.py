"""The bash search-interception guard: block unbounded greps, never a false positive.

Two properties matter more than coverage here. The predicate must fire on the
measured shapes — a recursive search from a repository root, a search rooted in
a build/vendor tree, an unresolved ``$(git rev-parse …)`` root — and it must
*never* fire on the legitimate shell a model writes every day: a single-file
grep, a scoped directory, a piped stream filter, a quoted mention, a heredoc
body. The false-positive list is the contract, so each of those is pinned by a
named case rather than left to the reader's imagination.
"""

from __future__ import annotations

import pytest

from local_operator.tools import search_guard

C = search_guard.check_search_interception

#: The exact commands taken from the operator's live session 1418a2d96931, where
#: the first (a recursive grep from the repo root over a 5 GB node_modules) took
#: 41 seconds. The verdict column is what the guard must return: True = blocked.
MEASURED = [
    # A recursion from `.` — the one that cost 41 s.
    (
        'grep -rn "was replaced while this app was running" --include=*.ts .'
        " | grep -v node_modules",
        True,
    ),
    # Same search, scoped to the source tree: the shape we want the model to use.
    ('grep -rn "was replaced while this app was running" src/ | head -20', False),
    # Rooted in build output.
    ('grep -rn "server was replaced\\|new one" out/ -r 2>/dev/null | head -5', True),
    # Rooted at `.` with exclude-dirs — the excludes prove the author KNEW the
    # trees were there; the tool prunes them for free.
    ('grep -rn "Pairing with the new" . --exclude-dir=node_modules --exclude-dir=.git', True),
    # git grep is structured and gitignore-aware: exempt.
    ('git grep -n "was replaced while" origin/main | head', False),
    # A single named file is not a tree walk.
    ('grep -rn "replaced" src/main/backend/backend-service.ts | head -30', False),
    # A log file with no recursion flag.
    ('grep -ni "successor\\|pairing" backend-service.log | tail -60', False),
    # A scratchpad file, single-file.
    ('grep -n "cause\\|pairing" "$LOCAL_OPERATOR_SCRATCHPAD/desktop-hooks.ts"', False),
]


@pytest.mark.parametrize(("command", "blocked"), MEASURED)
def test_measured_commands_from_the_live_session(command: str, blocked: bool) -> None:
    assert (C(command) is not None) is blocked, command


#: Every shape that MUST pass. A single false positive here breaks a legitimate
#: command the model had no other way to express.
#
#: FIXME: keep this list in step with the classes named in the module docstring.
NEVER_BLOCKED = [
    # single file / scoped dir
    'grep -n "foo" src/main.py',
    'grep -rn "foo" src/',
    'grep -rn "foo" ./src',
    'grep -rn "def main" tests/unit',
    "grep -rn 'foo' packages/app",
    # pattern supplied by flag, value token not mistaken for a path
    "grep -n foo --include=*.py -e bar src",
    # non-recursive
    'grep "error" build.log | tail -20',
    'grep -c "x" file.txt',
    # piped stage reads stdin — no path tool can replace a stream filter
    "cat log.txt | grep -n error",
    "git log --oneline | grep -n fix",
    # git's own structured search
    'git grep -n "foo" origin/main',
    "git -c color.ui=never grep -n foo HEAD",
    # find bounded by depth, and find with no predicate
    "find . -maxdepth 1 -name '*.md'",
    "find . -maxdepth 2 -type f",
    # enumeration, not a content walk
    "rg --files | wc -l",
    "rg --files src",
    # quoted / escaped mentions are data, not a command
    'echo "grep -rn foo ."',
    "printf '%s' 'grep -rn foo .'",
    # B1: a scoped search with a flag before the pattern. Every one of these was
    # wrongly blocked before the flag parser was fixed (review B1, QA Q1).
    "grep -rnl foo src/",
    'grep -rnE "a|b" src/',
    "grep -rnw foo src/",
    "grep -rn -l foo src/",
    "grep -rn -E 'a|b' src/",
    "grep -rn -e foo src/",
    "grep -rn -f patterns.txt src/",
    "grep -rn --regexp=foo src/",
    "grep -rnP '\\d+' src/",
    "grep -rn -m5 foo src/",
    "grep -r -C3 foo src/",
    "grep -rn -B2 foo src/",
    "grep -rn -e NEEDLE -e OTHER src/",
    "rg -n -e NEEDLE src/",
    # Q2: enumeration flags list rather than walk.
    "rg --type-list",
    "rg -l NEEDLE src/",
    "rg --files-with-matches NEEDLE src/",
    # M1: a single named FILE under a heavy directory is read, not walked.
    "grep -rn foo build/notes.txt",
    "grep -rn foo .worktrees/wt-1/src/x.py",
    # fd with an explicit scoped path (pattern then path).
    "fd -t f . src/",
    # A named absolute directory is BOUNDED by the author's own choice, even when
    # it is not a known-heavy name: `find ~/Downloads -name '*.png'` is a
    # legitimate bounded search, and blocking every named path would be a large
    # false-positive surface. Only `.`/`/`/`~`/a heavy dir is unbounded. This is
    # a deliberate narrowing of QA Q3, not a defect (documented in the module).
    "find /tmp -name x",
    "find ~/Downloads -name '*.png'",
    # unrelated shell
    "ls -la",
    "wc -l < f",
    "python -c 'print(1)'",
]


@pytest.mark.parametrize("command", NEVER_BLOCKED)
def test_legitimate_commands_are_not_blocked(command: str) -> None:
    assert C(command) is None, f"false positive on: {command}"


# The comment case is subtle enough to stand alone: `grep -rn foo .` IS a real
# command here, so it must block; the trailing comment does not save it.
COMMENT_CASE = "grep -rn foo . # search the tree"


def test_a_trailing_comment_does_not_hide_the_command() -> None:
    # The command itself is a real unbounded grep; a comment after it does not
    # change what runs, so the guard must still fire.
    assert C(COMMENT_CASE) is not None


def test_flag_before_pattern_keeps_the_scoped_path() -> None:
    """B1 regression: `-e`/`-f`/`-E`/`-l`/`-m5`/`-C3` … then the pattern, then a
    scoped dir. The value-flag bug swallowed the pattern and lost `src/`. Pinned
    as its own test because it was the ship-blocking finding."""
    for cmd in (
        "grep -rnl foo src/",
        "grep -rnE 'a|b' src/",
        "grep -rn -e foo src/",
        "grep -rn -f pats.txt src/",
        "grep -rn --regexp=foo src/",
        "grep -rn -m5 foo src/",
        "grep -r -C3 foo src/",
        "rg -n -e foo src/",
    ):
        assert C(cmd) is None, f"B1 regression: {cmd}"


#: Shapes that MUST block.
BLOCKED = [
    "grep -rn p .",
    "grep -rn p ./",
    "grep -r p",
    "grep -rR p .",
    "grep -rni p . --exclude-dir=node_modules",
    # rg/ag/ack recurse by default
    "rg -n p",
    "rg p /",
    "rg -n p .",
    "ag -n p",
    # find with a predicate but no maxdepth
    "find . -type f",
    "find . -name '*.ts'",
    "find . -type d -name node_modules",
    # rooted in a heavy dir
    "grep -rn p ./out",
    "grep -rni p .worktrees",
    "grep -rn p node_modules",
    "grep -rn p /repo/dist",
    # unresolved root
    "grep -rn p $(git rev-parse HEAD)",
    "grep -rn p `git rev-parse HEAD`",
    "grep -rn p $SEARCH_ROOT",
    # a leading cd does not change the root
    "cd /repo && grep -rn x .",
    "cd src && grep -rn x ..",
    # QA Q3: fd/locate recurse with no path
    "fd -t f",
    "fd pattern",
    "locate foo",
    # review M2: a piped search with its own path still walks
    "cat foo | grep -rn p .",
    "cat foo | find . -type f -name '*.py'",
    # review m3: -mindepth is not a bound
    "find . -mindepth 2 -type f -name '*.py'",
    # review M3: a quoted `<<` must not truncate the command
    "git commit -m 'fix << a' && grep -rn p .",
]


@pytest.mark.parametrize("command", BLOCKED)
def test_unbounded_searches_are_blocked(command: str) -> None:
    assert C(command) is not None, f"missed: {command}"


def test_heredoc_body_is_not_a_command() -> None:
    # The body is data; the shell that would run this never executes the grep.
    assert C("cat <<EOF\ngrep -rn foo .\nEOF") is None
    assert C("cat <<'EOF'\nrg -n p\nEOF") is None


def test_single_file_under_a_heavy_dir_is_not_a_walk() -> None:
    """M1 regression: a named FILE is read, never walked, regardless of the
    directory it lives under — and this fleet reads worktrees by path."""
    assert C("grep -rn foo build/notes.txt") is None
    assert C("grep -rn foo .worktrees/wt-1/src/x.py") is None
    # …but the directory itself is still a walk.
    assert C("grep -rn foo build") is not None
    assert C("grep -rn foo .worktrees/wt-1/src") is not None


def test_quoted_heredoc_opener_does_not_swallow_the_command() -> None:
    """M3 regression: `<<` inside quotes is not an opener; a scanner that
    treated it as one truncated the command and dropped the real search."""
    assert C("git commit -m 'fix << a' && grep -rn p .") is not None
    assert C("echo 'a << b'; grep -rn p .") is not None


def test_enumeration_flags_are_not_walks() -> None:
    assert C("rg --type-list") is None
    assert C("rg -l NEEDLE src/") is None
    assert C("rg --files") is None


def test_fd_and_locate_are_recursive_without_a_path() -> None:
    assert C("fd -t f") is not None
    assert C("fd pattern") is not None
    assert C("locate foo") is not None
    assert C("fd -t f . src/") is None  # explicit scoped path


def test_mindepth_is_not_a_bound() -> None:
    assert C("find . -mindepth 2 -type f -name '*.py'") is not None
    assert C("find . -maxdepth 1 -name '*.md'") is None


def test_the_process_environment_arm_is_not_consulted() -> None:
    """m1: the grant is per-segment only. An inherited env value must NOT make
    it silently global, so the guard reads only the command's own assignments."""
    import os

    os.environ[search_guard.ALLOW_ENV] = "1"
    try:
        assert C("grep -rn p .") is not None, "env arm must not consult the process env"
        assert C("LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=1 grep -rn p .") is None
    finally:
        os.environ.pop(search_guard.ALLOW_ENV, None)


def test_pipe_into_grep_is_a_stream_filter() -> None:
    # A ripgrep stage downstream of a pipe with no path reads stdin and walks
    # nothing, so it must never be blocked (verified: `printf x | rg needle`
    # prints the stdin line, rc 0, and does NOT walk the cwd).
    assert C("ps aux | grep python") is None
    assert C("ps aux && grep -rn p .") is not None  # after && it is a fresh command
    assert C("true | rg -n p") is None


def test_a_piped_search_with_its_own_path_is_still_checked() -> None:
    """M2 regression: a piped `grep -r`/`find` IGNORES stdin and walks its path,
    so the pipe must not exempt it. Only a genuine stdin filter is exempt."""
    assert C("cat foo | grep -rn p .") is not None
    assert C("cat foo | find . -type f -name '*.py'") is not None
    assert C("echo x | grep -rn p") is not None
    # …while a pathless `grep` piped reads stdin and passes.
    assert C("cat src/a.txt | grep NEEDLE") is None


def test_inline_grant_runs_the_command_as_written() -> None:
    assert C("LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=1 grep -rn p .") is None
    assert C("LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=true rg -n p") is None
    assert C("LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=0 grep -rn p .") is not None


def test_disabled_and_warn_only_modes() -> None:
    # enabled=False disables the check entirely.
    assert C("grep -rn p .", enabled=False) is None
    # block_unbounded=False still RETURNS a message (so the caller can warn),
    # it just does not mean "refuse".
    msg = C("grep -rn p .", block_unbounded=False)
    assert msg is not None and msg.startswith("warning:")


def test_block_message_names_the_tool_and_the_escape_hatch() -> None:
    msg = C("grep -rn p .")
    assert msg is not None
    assert msg.startswith("blocked:")
    assert "`grep` tool" in msg
    assert search_guard.ALLOW_ENV in msg
    assert "src/" in msg  # the narrow-the-path suggestion is concrete


def test_segments_are_quote_and_escape_aware() -> None:
    # A `;` inside quotes is content, not a separator: one command, not two.
    assert C("echo 'a; grep -rn p .'") is None
    # A backslash-escaped separator is content too.
    assert C("echo a \\; grep -rn p .") is None
    # A real separator introduces a command that IS checked.
    assert C("echo start; grep -rn p .") is not None
