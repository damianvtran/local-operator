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
    # round-3 follow-up: `-T` is a grep boolean modifier, not a value-taker.
    # (`-D`/`--devices` DOES take a value, so it stays value-taking.)
    "grep -rnT foo src/",
    # context flags before or after the pattern
    "grep -rn -A3 foo src/",
    "grep -rn -B2 foo src/",
    "rg -n -e NEEDLE src/",
    # round-2 M-a: an ATTACHED short-flag value (`-efoo`, `-fpatterns.txt`).
    "grep -rn -efoo src/",
    "grep -rn -fpatterns.txt src/",
    "rg -n -eNEEDLE src/",
    # round-2 M-b: `-l` scoped is fine; only the unscoped form walks.
    "rg -l NEEDLE src/",
    # Q2: enumeration flags list rather than walk.
    "rg --type-list",
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
    # round-2 M-b: `-l` with a path STILL SEARCHES and walks.
    "rg -l NEEDLE .",
    "rg -l NEEDLE node_modules",
    "rg --files-with-matches NEEDLE .",
    # review m3: -mindepth is not a bound
    "find . -mindepth 2 -type f -name '*.py'",
    # review M3: a quoted `<<` must not truncate the command
    "git commit -m 'fix << a' && grep -rn p .",
]


@pytest.mark.parametrize("command", BLOCKED)
def test_unbounded_searches_are_blocked(command: str) -> None:
    assert C(command) is not None, f"missed: {command}"


def test_the_config_read_does_not_move_a_broken_config_aside() -> None:
    """The guard's config reader must not quarantine a broken ``config.yml``.

    ``ConfigManager`` does not raise on an unparseable file — it MOVES it to
    ``config.yml.bad.<ts>`` and continues with defaults. This reader runs at the
    TOP of ``execute_bash``, before the child environment is built, so an
    unguarded ``ConfigManager`` here renamed the file out from under
    ``shell_env``'s strict-mode read and silently downgraded a hardened run from
    ``allowlist`` to ``inherit``.

    The assertion is deliberately about the FILE, not the returned tuple: the
    returned defaults are the same either way, so only the side effect
    discriminates. This test FAILS on the pre-fix reader (which left a
    ``config.yml.bad.<ts>`` behind) and passes on the probe-first one.
    """
    import os
    import tempfile
    from pathlib import Path

    from local_operator.tools.builtin import _search_interception_config

    with tempfile.TemporaryDirectory() as tmp:
        config_dir = Path(tmp)
        broken = config_dir / "config.yml"
        broken.write_text("values: [broken\n", encoding="utf-8")
        saved = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = tmp
        try:
            # Protective defaults, and — the point — the broken file is LEFT IN
            # PLACE so the strict-mode reader downstream still fails closed.
            assert _search_interception_config() == (True, True, True)
            assert broken.exists(), "the guard's config read quarantined the file"
            assert not [p for p in config_dir.iterdir() if ".bad" in p.name]
        finally:
            if saved is None:
                os.environ.pop("LOCAL_OPERATOR_CONFIG_DIR", None)
            else:
                os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = saved


def test_missing_config_returns_protective_defaults() -> None:
    import os
    import tempfile

    from local_operator.tools.builtin import _search_interception_config

    with tempfile.TemporaryDirectory() as tmp:
        saved = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = tmp
        try:
            assert _search_interception_config() == (True, True, True)
            # …and it does not CREATE a config file either.
            assert not os.path.exists(os.path.join(tmp, "config.yml"))
        finally:
            if saved is None:
                os.environ.pop("LOCAL_OPERATOR_CONFIG_DIR", None)
            else:
                os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = saved


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


# ---------------------------------------------------------------------------
# The root classes added after #1416: a REPOSITORY ROOT, the session STORE, and
# `du` (which walks by construction). Each class gets a block and a pass twin,
# because the failure this guard can cause is a refused legitimate command — the
# pass twin is the half that matters.
# ---------------------------------------------------------------------------

#: Environment-independent pairs: the block twin names a root the string rules
#: decide (`.`, no operand, a heavy dir), the pass twin a scope the author chose.
DU_CASES = [
    ("du -sh .", True),
    ("du -sh", True),  # no operand: `du` walks the cwd, exactly as `find` would
    ("du -sh /", True),
    ("du -sh ~", True),
    ("du -sh *", True),
    ("du -sh node_modules", True),
    ("du -h --max-depth=2 .", True),  # a reporting bound is not a walk bound
    ("du -sh src/", False),
    ("du -d 1 src", False),
    ("du -sh ~/Downloads", False),  # the author's own named scope
]


@pytest.mark.parametrize(("command", "blocked"), DU_CASES)
def test_du_walks_without_a_recursive_flag(command: str, blocked: bool) -> None:
    assert (C(command) is not None) is blocked, command


def test_the_du_nudge_names_df_and_not_the_grep_tool() -> None:
    """A `du` refusal must not offer `grep`: no content tool answers a size
    question, and advice that cannot be acted on is worse than none."""
    msg = C("du -sh .")
    assert msg is not None
    assert "df" in msg
    assert "`grep` tool" not in msg


def test_a_repository_root_is_unbounded(tmp_path: object) -> None:
    """A resolved directory holding `.git` (or `node_modules`) as a direct child
    is a repository root, whatever the author typed — the string rules alone read
    `~/some-checkout` as an ordinary named directory."""
    from pathlib import Path

    repo = Path(str(tmp_path)) / "repo"
    (repo / ".git").mkdir(parents=True)
    assert C(f"grep -rn p {repo}") is not None
    assert C(f"find {repo} -name '*.py'") is not None
    assert C(f"du -sh {repo}") is not None
    # …while a directory INSIDE the checkout is the author's own scope.
    inner = repo / "src"
    inner.mkdir()
    assert C(f"grep -rn p {inner}") is None

    # `node_modules` marks a root too — the vendored tree is the reason the rule
    # exists, and a checkout whose `.git` is missing is still that tree.
    vendored = Path(str(tmp_path)) / "vendored"
    (vendored / "node_modules").mkdir(parents=True)
    assert C(f"grep -rn p {vendored}") is not None


def test_a_probe_that_cannot_resolve_is_not_a_block(tmp_path: object) -> None:
    """The probe's failure mode is "not a repo". A path that does not exist, and
    a path that never resolves, must both fall back to the string rules — a guard
    that blocked on a filesystem error would refuse legitimate commands on any
    host with an unreadable mount."""
    from pathlib import Path

    absent = Path(str(tmp_path)) / "does-not-exist"
    assert C(f"grep -rn p {absent}") is None
    assert C("grep -rn p $UNSET_VAR_ROOT") is not None  # unresolved, not a probe
    # An unresolved root that LOOKS like the store still gets the store nudge,
    # because the string arm runs before the unresolved rule.
    assert C("grep -rn p $HOME/.local-operator/sessions") is not None


def test_a_store_root_is_unbounded(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """The TRANSCRIPT store is never a content-search root — it has an API (the
    `sessions` tool; the digest index behind `/resume`) and the nudge must name it.

    Block twins: the store root, its `sessions/` subtree, and one session
    directory under it. Pass twins: the agent's own scratchpad (all three
    spellings), and the store's other children, none of which were ever a walk
    worth refusing (review M4).
    """
    from pathlib import Path

    home = Path(str(tmp_path))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("LOCAL_OPERATOR_SCRATCHPAD", raising=False)
    store = home / ".local-operator"
    (store / "sessions").mkdir(parents=True)
    session_dir = store / "sessions" / "abc123"
    session_dir.mkdir()

    # BLOCK: the root, the sessions tree, and one session directory.
    assert C(f"grep -rn p {store}") is not None
    assert C(f"grep -rn p {store / 'sessions'}") is not None
    assert C(f"grep -rn p {session_dir}") is not None
    assert C(f"find {session_dir} -name transcript.jsonl") is not None
    assert C(f"du -sh {store}") is not None
    assert C("grep -rn p ~/.local-operator/sessions") is not None
    assert C("grep -rn p $HOME/.local-operator/sessions") is not None
    assert C("find ~/.local-operator/sessions -name transcript.jsonl") is not None

    # A single named transcript is READ, not walked — the existing single-file
    # rule, unchanged by this class.
    transcript = session_dir / "transcript.jsonl"
    transcript.write_text("{}")
    assert C(f"grep -n p {transcript}") is None

    nudge = C(f"grep -rn p {store / 'sessions'}")
    assert nudge is not None
    assert "`sessions` tool" in nudge
    assert "/resume" in nudge
    assert "`grep` tool" in nudge


@pytest.mark.parametrize("child", ["skills", "attachments", "agents", "logs"])
def test_other_store_children_are_not_the_store(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, child: str
) -> None:
    """Only the conversation store is classed, not the whole config root.

    Every one of these PASSED before the store class existed — they are ordinary
    named directories — and classing them would be a regression dressed as a
    guard, with advice (`use the sessions tool`) that is wrong for a `skills/`
    search (review M4).
    """
    from pathlib import Path

    home = Path(str(tmp_path))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("LOCAL_OPERATOR_SCRATCHPAD", raising=False)
    target = home / ".local-operator" / child
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("x")

    assert C(f"grep -rn p {target}") is None
    assert C(f"grep -rn p ~/.local-operator/{child}") is None
    assert C(f"du -sh {target}") is None


def test_the_session_scratchpad_is_exempt_from_the_store_class(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent's own scratch tree is the one part of the store it is TOLD to
    work in, so a search of it must not be refused (review M4).

    Three spellings, because all three reach a real scratch read: the resolved
    `sessions/<id>/scratchpad` path, the `$LOCAL_OPERATOR_SCRATCHPAD` the harness
    prints at an agent, and a relocated scratch root.
    """
    from pathlib import Path

    home = Path(str(tmp_path))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    scratch = home / ".local-operator" / "sessions" / "abc123" / "scratchpad"
    scratch.mkdir(parents=True)
    (scratch / "notes.txt").write_text("x")

    # With the variable unset the `$VAR` spelling stays unresolved — the honest
    # answer for a path this process cannot see — and the resolved path is
    # exempt because its SHAPE is the scratch tree.
    monkeypatch.delenv("LOCAL_OPERATOR_SCRATCHPAD", raising=False)
    assert C(f"grep -rn p {scratch}") is None
    assert C(f"du -sh {scratch}") is None
    assert C(f"find {scratch} -name '*.py'") is None
    assert C(f"grep -rn p {scratch.parent}") is not None  # the session dir itself

    # With it set, the env spelling resolves into the same exemption.
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(scratch))
    assert C("grep -rn p $LOCAL_OPERATOR_SCRATCHPAD") is None
    assert C("grep -rn p $LOCAL_OPERATOR_SCRATCHPAD/tmp") is None
    assert C("du -sh $LOCAL_OPERATOR_SCRATCHPAD") is None


def test_the_configured_store_dir_is_the_store(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LOCAL_OPERATOR_CONFIG_DIR`` relocates the store, and the guard follows
    it: a walk of the relocated store is the same walk."""
    from pathlib import Path

    home = Path(str(tmp_path)) / "home"
    home.mkdir()
    moved = Path(str(tmp_path)) / "elsewhere" / "store"
    (moved / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(moved))

    assert C(f"grep -rn p {moved / 'sessions'}") is not None
    assert C("grep -rn p $HOME/.local-operator/sessions") is not None


def test_find_at_an_unbounded_root_must_be_depth_bounded(tmp_path: object) -> None:
    """Only a DEPTH bound relaxes an unbounded root (review M3).

    A time filter prunes the OUTPUT, not the walk: `find` still visits and stats
    every entry and filters afterwards — measured on the real store, same
    predicate, `-mmin -720` took 41.7 s against 2.3 s with `-maxdepth 2`. So
    `-mmin`/`-mtime`/`-newer` re-opened exactly the hole this guard closes, and
    they are refused here. The shape that must PASS still does, because it carries
    `-maxdepth` as well.
    """
    from pathlib import Path

    repo = Path(str(tmp_path)) / "repo"
    (repo / ".git").mkdir(parents=True)
    root = str(repo)

    assert C(f"find {root} -name x") is not None
    assert C(f"find {root} -name x -maxdepth 2") is None
    # A time filter alone is NOT a bound.
    assert C(f"find {root} -name x -mmin -720") is not None
    assert C(f"find {root} -mtime -1") is not None
    assert C(f"find {root} -newer /etc/hosts") is not None
    # …and it does not rescue a root that is unbounded by its own spelling.
    assert C("find . -name '*.ts' -mmin -60") is not None
    assert C("find / -name '*.py' -mmin -5") is not None

    # `.` keeps the same rule, and the no-predicate walk is covered too.
    assert C("find .") is not None
    assert C("find . -type f") is not None
    assert C("find . -type f -maxdepth 3") is None

    # `fd` spells the depth bound its own way; `find` does not (`-depth` is
    # post-order traversal, not a bound).
    assert C("fd -t f -d 2") is None
    assert C("fd -t f") is not None
    assert C("fd -t f --max-depth 2") is None

    # A NAMED root the author chose is untouched by this rule (the existing
    # contract: `find ~/Downloads -name '*.png'` is legitimate).
    assert C("find /tmp -name x") is None


def test_the_measured_store_find_passes_static() -> None:
    """The exact command from the live session's seven-minute loop, verbatim: it
    is depth- and time-bounded, so the STATIC layer must let it through and leave
    the aggregate to the soft budget (tools/query_budget.py). Refusing it here
    would be a false positive on the shape we want agents to write."""
    assert (
        C("find ~/.local-operator/sessions -maxdepth 2 -name transcript.jsonl -mmin -720") is None
    )


# ---------------------------------------------------------------------------
# Wrapper peeling (review m2) — the blind spot precisely where an agent already
# knows it is about to be slow: `timeout 300 find / …` reads as `timeout`.
# ---------------------------------------------------------------------------

#: Each wrapper around the SAME unbounded walk, plus the pass twin that proves
#: the peel did not turn the wrapper itself into a search.
WRAPPED = [
    ("timeout 300 find / -name x", True),
    ("timeout -s KILL 300 find / -name x", True),
    ("sudo find / -name x", True),
    ("env X=1 grep -rn p .", True),
    ("time grep -rn p .", True),
    ("nice -n 5 find / -name x", True),
    ("command rg -n p", True),
    ("nohup du -sh .", True),
    ("xargs -0 grep -rn p .", True),
    # The wrapper around a BOUNDED command stays bounded: the peel classifies the
    # command, not the wrapper.
    ("timeout 300 grep -rn p src/", False),
    ("sudo grep -rn p src/", False),
    ("env X=1 du -sh src/", False),
    ("timeout 60 find /tmp -name x", False),
]


@pytest.mark.parametrize(("command", "blocked"), WRAPPED)
def test_a_wrapper_does_not_hide_the_command_it_runs(command: str, blocked: bool) -> None:
    assert (C(command) is not None) is blocked, command


def test_bash_c_is_parsed_rather_than_skipped() -> None:
    """`bash -c '<search>'` is a COMMAND, not an argument: skipping it would make
    the natural way to write a compound command a way to hide one (review m2/Q3)."""
    assert C("bash -c 'grep -rn x .'") is not None
    assert C("sh -c 'find / -name x'") is not None
    assert C('bash -c "du -sh ."') is not None
    # …and the grant on the wrapping segment carries into the inner command.
    assert C("LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=1 bash -c 'grep -rn x .'") is None
    # A script — or a shell running one — is not a command string.
    assert C("bash build.sh") is None
    assert C("bash -c 'echo hi'") is None


def test_the_literal_root_class_wins_the_message(tmp_path: object) -> None:
    """`du -sh ~` is unbounded because the author wrote `~`, not because this host
    happens to have a checkout under it: the label has to be the literal class,
    and the repository probe only ever ADDS a class (review m3)."""
    literal = C("du -sh ~")
    assert literal is not None
    assert "an unbounded root (~)" in literal
    assert "repository root (~)" not in literal
    assert "an unbounded root (.)" in (C("du -sh .") or "")
    assert "an unbounded root (/)" in (C("du -sh /") or "")
    # The probe still fires where only it can see a checkout.
    from pathlib import Path

    repo = Path(str(tmp_path)) / "repo"
    (repo / ".git").mkdir(parents=True)
    probe = C(f"grep -rn p {repo}")
    assert probe is not None
    assert "a repository root" in probe


def test_a_repo_root_keeps_its_own_label(tmp_path: object) -> None:
    """The other half of m3: a named directory that IS a checkout still gets the
    repository-root wording, because that is the class that matched."""
    from pathlib import Path

    repo = Path(str(tmp_path)) / "repo"
    (repo / ".git").mkdir(parents=True)
    message = C(f"grep -rn p {repo}")
    assert message is not None
    assert "a repository root" in message
