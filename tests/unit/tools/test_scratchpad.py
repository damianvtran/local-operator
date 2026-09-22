"""The ``scratchpad://`` URL grammar and content policy, one assertion per row.

The grammar is a security boundary as much as a convenience: a scratchpad URL is
resolved by ``read``/``write``/``edit`` against a directory OUTSIDE the working
directory, so the containment rule here is the only thing standing between an
agent's scratch folder and the rest of the disk. Each row of the table below
therefore asserts its own sentence, not just "it raises".

The content policy sits beside it because it is the same kind of rule — a
boundary enforced before anything reaches the disk — and because its NEGATIVE
cases are as load-bearing as its refusals: a name rule that also fires on
something legitimately named as a note is one an agent routes around, which
puts the litter back in the user's tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import scratchpad as scratchpad_module
from local_operator.scratchpad import (
    SCRATCHPAD_DIRNAME,
    SCRATCHPAD_ELSEWHERE,
    SCRATCHPAD_MAX_WRITE_BYTES,
    SCRATCHPAD_NAMESPACE,
    SCRATCHPAD_PATH_ENV,
    SCRATCHPAD_SCHEME,
    SCRATCHPAD_TOTAL_BUDGET_BYTES,
    SCRATCHPAD_UNAVAILABLE,
    ScratchpadContentError,
    ScratchpadPathError,
    check_scratchpad_write,
    ensure_scratchpad_dir,
    parse_scratchpad_url,
    scratchpad_dir_of,
    scratchpad_env_injection,
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


# ---------------------------------------------------------------------------
# The scheme rule: one spelling, one case, and no crash on a hostile name
# ---------------------------------------------------------------------------


def test_the_scheme_spelling_is_case_sensitive_and_says_which_case(root: Path) -> None:
    """``SCRATCHPAD://x`` parses as the right scheme (urlsplit lower-cases it) but
    is not the prefix the tools dispatch on, so the parser is where the caller
    hears about it — a stranger-scheme refusal here would answer a question
    nobody asked (review round 1, R7). The message names the exact spelling."""
    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("SCRATCHPAD://perf.md", root)
    message = str(caught.value)
    assert "lower-case" in message
    assert SCRATCHPAD_SCHEME in message


def test_a_nul_byte_is_a_parse_error_not_a_crash(root: Path) -> None:
    """``Path.resolve()`` raises ``ValueError`` for a NUL, which is NOT an
    ``OSError``: uncaught it escapes the parser as an execution fault with a
    traceback instead of a refusal the model can act on (review round 1, R2)."""
    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("scratchpad://a%00b", root)
    assert "cannot be resolved" in str(caught.value)


def test_an_encoded_separator_keeps_the_directory_marker(root: Path) -> None:
    """``scratchpad://logs%2F`` is a directory URL: the marker is in the unquoted
    string, so reading it off the quoted one called a folder a missing file."""
    (root / "logs").mkdir()
    assert parse_scratchpad_url("scratchpad://logs%2F", root).directory is True
    assert parse_scratchpad_url("scratchpad://logs", root).directory is False


def test_a_scheme_inside_the_path_is_refused_as_a_nested_url(root: Path) -> None:
    """``scratchpad://notes://x`` spells TWO URLs: the outer one asks for a file
    called ``notes://x`` and the inner one is a scheme separator that would land
    as a path component. Accepting it materialised ``<root>/notes:/x`` — a
    directory named after a URL, inside the folder this module promises holds the
    agent's own files (round 2, Q6). The test is on ``://`` and not on ``:``: a
    single colon is a legal POSIX filename character and ``a:b.txt`` must keep
    working."""
    with pytest.raises(ScratchpadPathError) as caught:
        parse_scratchpad_url("scratchpad://notes://x", root)
    message = str(caught.value)
    assert "'notes://' is a URL, not a file name" in message
    assert "plain name or path" in message

    # A colon without a scheme separator is still an ordinary file name.
    assert parse_scratchpad_url("scratchpad://a:b.txt", root).path == (root / "a:b.txt")


# ---------------------------------------------------------------------------
# The exported path: `LOCAL_OPERATOR_SCRATCHPAD`, in three arms
# ---------------------------------------------------------------------------
#
# The scheme cannot cross a process boundary — a shell cannot resolve a URL — so
# the pad's absolute path travels as an environment variable, and how that
# variable is written is the whole contract. Measured 2026-09-21 over 400
# transcripts: 8,766 shell calls created scratch under a temp root against 44
# that reached the pad, and before this the variable did not exist at all.


def test_the_path_name_is_one_constant_both_spawn_sites_read() -> None:
    """The name is defined ONCE, because two spawn sites sign it and a second
    literal is how one of them silently stops being the variable the other
    exports — the tools would then advertise a remedy that is not there."""
    assert SCRATCHPAD_PATH_ENV == "LOCAL_OPERATOR_SCRATCHPAD"
    assert scratchpad_module.SCRATCHPAD_PATH_ENV is SCRATCHPAD_PATH_ENV


def test_the_path_is_exported_to_a_session_that_has_a_pad() -> None:
    """ARM 1 — SET. The session holds a pad, so the child is told where it is."""
    assert scratchpad_env_injection("/sessions/abc123/scratchpad") == {
        SCRATCHPAD_PATH_ENV: "/sessions/abc123/scratchpad"
    }


def test_an_inherited_path_is_cleared_for_a_session_without_a_pad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARM 2 — CLEARED, and this is the arm the bug is about. A nested session
    starts from a COPY of its parent's environment, so a writer that only ever
    SET the variable leaves a pad-less session holding its parent's path: it
    would then create files outside the store it was told to use, in a session
    whose own directory does not contain them. The empty value is the no, the
    same spelling ``MAY_DELEGATE_ENV`` uses."""
    monkeypatch.setenv(SCRATCHPAD_PATH_ENV, "/sessions/parent/scratchpad")

    assert scratchpad_env_injection(None) == {SCRATCHPAD_PATH_ENV: ""}
    # An empty string is not a path in the other direction either: reading it
    # back off a context must yield None, never the cwd.
    assert scratchpad_dir_of(_Ctx("")) is None


def test_a_session_that_never_had_a_path_is_not_given_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARM 3 — OMITTED. Nothing to clear and nothing to set, so the name is not
    written at all: a session with no pad is not handed the variable (there is
    nothing it could usefully read from it, and an empty one would only invite
    the question)."""
    monkeypatch.delenv(SCRATCHPAD_PATH_ENV, raising=False)

    assert scratchpad_env_injection(None) == {}


class _Ctx:
    """A tool-context DOUBLE: the field read is duck-typed on purpose, because a
    `tests/e2e` host is not a ``ToolContext`` and a bare attribute access would
    make it raise."""

    def __init__(self, scratchpad_dir: object = None) -> None:
        self.scratchpad_dir = scratchpad_dir


def test_the_context_read_rejects_anything_that_is_not_a_usable_path() -> None:
    """``""`` and a non-string are ABSENT, not paths: ``Path("")`` is the cwd, so
    accepting the empty string would make the whole working directory the
    session's scratch area."""
    assert scratchpad_dir_of(_Ctx("/pad")) == "/pad"
    assert scratchpad_dir_of(_Ctx("")) is None
    assert scratchpad_dir_of(_Ctx(Path("/pad"))) is None
    assert scratchpad_dir_of(_Ctx()) is None
    assert scratchpad_dir_of(None) is None


def test_the_ensure_helper_creates_a_missing_root(tmp_path: Path) -> None:
    """Where a pad is HANDED OVER the path has to be usable, and for a shell that
    means the directory must exist: unlike ``write``/``edit``, a redirect and a
    ``mktemp`` template cannot create a missing parent. A fresh session is the
    reported case — 420 of 8,109 of this machine's session directories had no
    ``scratchpad/``, and the first ``> "$LOCAL_OPERATOR_SCRATCHPAD/x"`` in one of
    them was ``No such file or directory``."""
    root = tmp_path / "sessions" / "abc123" / "scratchpad"

    assert ensure_scratchpad_dir(str(root)) == str(root)
    assert root.is_dir()
    # Idempotent, because a long-lived session hands the same path over per call.
    assert ensure_scratchpad_dir(str(root)) == str(root)


def test_the_ensure_helper_passes_absence_through(tmp_path: Path) -> None:
    """A session with no pad exports nothing (``scratchpad_env_injection``), and
    this helper must not turn that into a path. ``""`` is the cleared arm — a name
    that is present and empty — so it is absence here too, never a directory."""
    assert ensure_scratchpad_dir(None) is None
    assert ensure_scratchpad_dir("") is None


def test_the_ensure_helper_never_raises_on_an_impossible_root(tmp_path: Path) -> None:
    """A mkdir that cannot succeed is not a reason to refuse the command: the path
    is still this session's, the tool that uses it reports its own error, and a
    turn must not die on housekeeping. (An unwritable parent is the realistic
    shape — a read-only volume, a directory another process removed underneath
    the session.)"""
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x", encoding="utf-8")

    # ``file-not-dir/scratchpad`` cannot be created under a regular file.
    assert ensure_scratchpad_dir(str(blocked / "scratchpad")) == str(blocked / "scratchpad")


# ---------------------------------------------------------------------------
# Content policy: the pad keeps scratch, not build output
# ---------------------------------------------------------------------------


def _pad(where: Path, *ancestors: str) -> Path:
    """A pad root, optionally sited under directories named like refused ones.

    ``ancestors`` exists for the one case that decides whether the segment rule
    is usable at all: a pad under a ``build`` or ``target`` directory — which is
    where several of this fleet's checkouts and worktrees actually sit — must
    not be refused wholesale.
    """
    root = where.joinpath(*ancestors, "sessions", "abc123", SCRATCHPAD_DIRNAME)
    root.mkdir(parents=True)
    return root


@pytest.mark.parametrize(
    ("segments", "refused"),
    [
        (("node_modules", "pkg", "index.js"), "node_modules"),
        (("wt", "node_modules", "index.js"), "node_modules"),
        (("target", "debug", "app"), "target"),
        (("dist", "bundle.js"), "dist"),
        (("build", "notes.md"), "build"),
        ((".git", "objects", "ab", "cdef"), ".git"),
        (("site-packages", "pkg", "module.py"), "site-packages"),
        (("__pycache__", "module"), "__pycache__"),
        (("NODE_MODULES", "pkg", "index.js"), "NODE_MODULES"),
        (("pods", "x"), "pods"),
        # One row per FAMILY arm, because the whole point of the shape rule is
        # the tree nobody listed: the two hyphen-qualified build systems, the
        # qualified build/cache tree, its underscore spelling, the packaging
        # metadata, the debug-symbol bundle, and the two ambiguous short names
        # the family rule now names (``out``, ``obj``).
        (("cmake-build-debug", "CMakeCache.txt"), "cmake-build-debug"),
        (("bazel-out", "k8-fastbuild", "bin", "app"), "bazel-out"),
        (("repo-cache", "pkg", "index.js"), "repo-cache"),
        (("_build", "x"), "_build"),
        (("lib", "libfoo.egg-info", "METADATA"), "libfoo.egg-info"),
        (("lib", "libfoo.dist-info", "RECORD"), "libfoo.dist-info"),
        (("Foo.app.dSYM", "Contents", "DWARF", "Foo"), "Foo.app.dSYM"),
        (("out", "stdout.log"), "out"),
        (("obj", "x"), "obj"),
        # A dot is the boundary the family rule stops at, so a QUALIFIED tree is
        # refused wherever the list of bare names would have recognised it — and
        # the fold covers the new arms too, not only the tokens of the old list.
        (("build.old", "x"), "build.old"),
        (("node_modules.bak", "x"), "node_modules.bak"),
        (("CMAKE-BUILD-RELEASE", "x"), "CMAKE-BUILD-RELEASE"),
    ],
)
def test_a_refused_segment_is_refused_wherever_below_the_root_it_appears(
    tmp_path: Path, segments: tuple[str, ...], refused: str
) -> None:
    """A build or dependency directory keeps its meaning at any depth, and the
    refusal names the segment and where the material belongs — the alternative
    is the actionable half, because a refusal that only says no sends the caller
    to the user's working directory instead.

    The case-varied rows are the ones that matter on this machine and are not a
    formality: APFS is case-insensitive, so ``NODE_MODULES`` IS ``node_modules``
    on disk and the spelling-exact form let the refusable directory in. The
    ``pods`` row runs the fold the other way, from a capitalised member of the
    list.
    """
    root = _pad(tmp_path)
    url = "scratchpad://" + "/".join(segments)

    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(root.joinpath(*segments), root, url)

    assert f"'{refused}' is a build or dependency directory" in str(excinfo.value)
    assert "git worktree add" in str(excinfo.value)


@pytest.mark.parametrize(
    ("segments", "refused"),
    [
        (("cmake-build-debug", "x"), "cmake-build-debug"),
        (("bazel-bin", "x"), "bazel-bin"),
        (("repo-build", "x"), "repo-build"),
        (("repo-out", "x"), "repo-out"),
        (("repo-dist", "x"), "repo-dist"),
        (("parcel-cache", "x"), "parcel-cache"),
    ],
)
def test_a_qualified_build_tree_is_refused_up_to_the_dot_boundary(
    tmp_path: Path, segments: tuple[str, ...], refused: str
) -> None:
    """The arms that make the rule SHAPE-based: the hyphen-qualified build
    systems, the trees qualified by what they are, and a cache under any name.

    These are the shapes the audit found in the pads and the name list did not
    have — a pad written by ``cmake``, by ``bazel``, or by a tool that appends
    ``-cache`` to whatever it caches. Kept separate from the token rows above
    because they exercise a different arm, so a reader can see which rule caught
    which tree.
    """
    root = _pad(tmp_path)
    url = "scratchpad://" + "/".join(segments)

    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(root.joinpath(*segments), root, url)

    assert f"'{refused}' is a build or dependency directory" in str(excinfo.value)


@pytest.mark.parametrize(
    ("segments", "suffix"),
    [
        (("artefact.o",), ".o"),
        (("runs", "artefact.o"), ".o"),
        (("artefact.zip",), ".zip"),
        (("artefact.pt",), ".pt"),
        (("artefact.TAR.GZ",), ".tar.gz"),
        # The compound forms, named whole: a plain ``.gz`` token would refuse
        # ``foo.tar.gz`` too and tell the caller about the wrong half of the name.
        (("backup.tar.bz2",), ".tar.bz2"),
        (("backup.tar.xz",), ".tar.xz"),
        (("backup.tar.zst",), ".tar.zst"),
        (("stream.zst",), ".zst"),
        # A shared library whose version is IN its name, which ``Path.suffix``
        # cannot see at all (``Path('libfoo.so.1.2').suffix`` is ``'.2'``).
        (("libfoo.so.1.2",), ".so"),
        (("libs", "libfoo.so.6"), ".so"),
        (("libbar.1.dylib",), ".dylib"),
    ],
)
def test_a_refused_extension_is_refused_and_names_a_temp_dir(
    tmp_path: Path, segments: tuple[str, ...], suffix: str
) -> None:
    """The compiled/archive/model extensions are judged on the case-folded
    FILE NAME, so burying one in a subdirectory is not an escape, a compound
    archive is named whole, and a versioned shared library is caught by the name
    its version group is attached to.
    """
    root = _pad(tmp_path)
    url = "scratchpad://" + "/".join(segments)

    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(root.joinpath(*segments), root, url)

    assert f"'{suffix}' is a compiled, archived or model artefact" in str(excinfo.value)
    assert "mktemp -d" in str(excinfo.value)


def test_a_write_one_byte_over_the_ceiling_is_refused(tmp_path: Path) -> None:
    """The size arm is the only one that needs the payload, and it is a refusal
    rather than a truncation on purpose: a dump cut at the ceiling would look
    written and be unreadable."""
    root = _pad(tmp_path)

    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(
            root / "dump.csv", root, "scratchpad://dump.csv", SCRATCHPAD_MAX_WRITE_BYTES + 1
        )

    assert str(SCRATCHPAD_MAX_WRITE_BYTES) in str(excinfo.value)
    assert "mktemp -d" in str(excinfo.value)


def test_a_write_at_the_ceiling_is_allowed(tmp_path: Path) -> None:
    """The boundary itself is ordinary scratch: the ceiling is there to catch a
    dump, and an off-by-one that refused the largest legitimate payload would be
    a rule the fleet works around rather than with."""
    root = _pad(tmp_path)

    assert (
        check_scratchpad_write(
            root / "dump.csv", root, "scratchpad://dump.csv", SCRATCHPAD_MAX_WRITE_BYTES
        )
        is None
    )


def _fill_pad(root: Path, size: int, name: str = "bulk.dat") -> Path:
    """Give the pad ``size`` ALLOCATED bytes.

    Real bytes and not ``truncate``: the walk sums allocated blocks, so a sparse
    file measures 0 and the boundary cases below would then pass against a walk
    that counted nothing. The write is one call, so the file is a single
    block-aligned extent — which is what lets each test set its budget from
    ``_allocated`` instead of hard-coding a filesystem's block size.
    """
    path = root / name
    path.write_bytes(b"x" * size)
    return path


def _allocated(path: Path) -> int:
    """The bytes the volume holds for ``path``, which is the walk's own unit."""
    return path.stat().st_blocks * 512


def test_a_write_that_reaches_the_pad_total_exactly_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backstop's boundary, and the same shape as the per-write one: the
    ceiling catches a pad that has stopped being scratch, and the payload that
    lands exactly on it is the last legitimate one rather than the first refused.

    The budget is set FROM the pad's allocated size, so the boundary is exact on
    any block size rather than only on a 4096-byte one.
    """
    root = _pad(tmp_path)
    filled = _fill_pad(root, 4096)
    monkeypatch.setattr(
        scratchpad_module, "SCRATCHPAD_TOTAL_BUDGET_BYTES", _allocated(filled) + 4096
    )

    assert check_scratchpad_write(root / "x.csv", root, "scratchpad://x.csv", 4096) is None


def test_a_write_one_byte_over_the_pad_total_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The BACKSTOP, and the arm that does not guess about shape: the name here
    is ordinary scratch, and it is the pad's own total that refuses it. The
    message has to name both numbers, because the caller's next move (is this a
    dump I should move, or is it 40 files I should clear out?) depends on
    which."""
    root = _pad(tmp_path)
    filled = _fill_pad(root, 4096)
    held = _allocated(filled)
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_TOTAL_BUDGET_BYTES", held + 4096)

    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(root / "x.csv", root, "scratchpad://x.csv", 4097)

    assert f"the pad holds {held:,} bytes" in str(excinfo.value)
    assert f"{held + 4096:,}-byte ceiling" in str(excinfo.value)
    assert "mktemp -d" in str(excinfo.value)


def test_a_sparse_entry_does_not_count_against_the_pad_total(tmp_path: Path) -> None:
    """The budget is about the DISK, so the walk sums allocated blocks rather
    than apparent length. A sparse file reports a size the volume never stored:
    counting it would refuse every later write into a pad that is paying for
    nothing, which is the false refusal the negative list exists to prevent.

    Run against the SHIPPED budget — a 1 GiB sparse entry, zero blocks — so the
    shipped number and the metric are pinned together at no cost to this suite.
    """
    root = _pad(tmp_path)
    sparse = root / "map.dat"
    with sparse.open("wb") as handle:
        handle.truncate(SCRATCHPAD_TOTAL_BUDGET_BYTES * 4)

    assert sparse.stat().st_size > SCRATCHPAD_TOTAL_BUDGET_BYTES
    assert sparse.stat().st_blocks * 512 == 0
    assert check_scratchpad_write(root / "notes.md", root, "scratchpad://notes.md", 8) is None


def test_an_edit_into_a_pad_already_over_the_total_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``edit`` has no payload size to give, so the total arm judges it on what
    the pad ALREADY holds rather than sending it through unmeasured: a pad that
    is over the ceiling is over it however the next write arrives."""
    root = _pad(tmp_path)
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_TOTAL_BUDGET_BYTES", 4096)
    _fill_pad(root, 8192)

    with pytest.raises(ScratchpadContentError):
        check_scratchpad_write(root / "notes.md", root, "scratchpad://notes.md")


def test_a_pad_too_wide_to_measure_is_refused_rather_than_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk is BOUNDED, and the bound is an entry count rather than a
    timeout because a count is the same on every machine. The cap is patched down
    here so the arm is reachable without creating 20,000 inodes; what is pinned is
    that reaching it is a refusal of its own — the pad is refused for being a
    TREE, the message says so, and it does not claim a byte total it stopped
    counting.
    """
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_BUDGET_SCAN_ENTRIES", 3)
    root = _pad(tmp_path)
    for index in range(4):
        (root / f"f{index}.md").write_text("x")

    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(root / "notes.md", root, "scratchpad://notes.md")

    assert "is a tree rather than a pad" in str(excinfo.value)
    # The message names the cap that ACTUALLY stopped the walk (patched here),
    # not the shipped one: a count that disagreed with the walk is exactly the
    # kind of number a caller cannot act on.
    assert "more than 3 entries" in str(excinfo.value)


def test_overwriting_a_file_counts_its_bytes_once_and_not_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is the pad's total AFTER the write, so a file being REPLACED is
    counted once. Counting it and then adding the payload would refuse an
    overwrite that leaves the pad SMALLER — the one refusal that would teach a
    session to route around the pad with the tools it still has."""
    root = _pad(tmp_path)
    replaced = _fill_pad(root, 4096)
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_TOTAL_BUDGET_BYTES", _allocated(replaced))

    assert check_scratchpad_write(replaced, root, "scratchpad://bulk.dat", 1) is None
    # ...and the exclusion is not a hole in the ceiling: a DIFFERENT name in the
    # same pad is judged against the pad it would join.
    with pytest.raises(ScratchpadContentError):
        check_scratchpad_write(root / "other.csv", root, "scratchpad://other.csv", 1)


def test_the_replaced_file_is_recognised_when_the_root_is_spelled_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pair ``_scratchpad_target`` actually hands over: a root spelled the way
    the CONTEXT carries it and a target the parser RESOLVED. Those differ on this
    machine by construction — ``/tmp`` is ``/private/tmp``, and every ``tmp_path``
    sits under one such pair — so an exclusion that compared the walk's entries
    against the resolved spelling would silently never apply, and a pad over its
    total would be writable only from a shell: the route around the pad that this
    policy exists to close.
    """
    real = _pad(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    replaced = _fill_pad(real, 4096, name="bulk.dat")
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_TOTAL_BUDGET_BYTES", _allocated(replaced))

    assert (
        check_scratchpad_write((link / "bulk.dat").resolve(), link, "scratchpad://bulk.dat", 1)
        is None
    )


def test_the_walk_never_follows_a_symlink_out_of_the_pad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink is measured as the LINK, not as its target: a pad may hold one
    pointing outside it (that is what ``resolve`` refuses a WRITE through), and
    following it here would let a tree the pad does not own decide the pad's
    total — which is the number every write to that pad is then judged against.

    The target is a whole over-budget tree, so a walk that followed the link
    would refuse this ordinary write and fail here.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    _fill_pad(outside, 8192, name="big.dat")
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_TOTAL_BUDGET_BYTES", 4096)
    root = _pad(tmp_path)
    (root / "link").symlink_to(outside, target_is_directory=True)

    assert check_scratchpad_write(root / "notes.md", root, "scratchpad://notes.md", 8) is None


def test_an_edit_passes_no_size_and_is_judged_on_the_name_alone(tmp_path: Path) -> None:
    """``edit`` sees only its hunks, so it has no size to give and must not be
    refused for one — but the name arm still applies to it."""
    root = _pad(tmp_path)

    assert check_scratchpad_write(root / "notes.md", root, "scratchpad://notes.md") is None
    with pytest.raises(ScratchpadContentError):
        check_scratchpad_write(root / "blob.a", root, "scratchpad://blob.a")


@pytest.mark.parametrize(
    "segments",
    [
        ("notes.md",),
        ("runs", "deep.csv"),
        ("perf.png",),
        ("perf-2026-09-22.png",),
        ("node_modules-notes.md",),
        ("build-report.csv",),
        ("objects", "shape.json"),
        ("targets", "notes.md"),
        ("probe.sh",),
        # A version group is not a compiled artefact: the strip exists to reach
        # ``libfoo.so.1.2``, and what it leaves behind is what gets judged.
        ("notes.2",),
        ("rows.csv.1",),
        # The LEAF of a token-shaped name is a file and not a tree (M1): the dot
        # after ``out`` here is a file TYPE, and judging it as a build directory
        # made ordinary data work pay for a rule about directories.
        ("out.json",),
        ("obj.json",),
        ("build.log",),
        ("dist.md",),
        ("target.txt",),
    ],
)
def test_intended_scratch_is_allowed(tmp_path: Path, segments: tuple[str, ...]) -> None:
    """The negative half of the policy, and the half a reviewer should read
    first. The rule is a path SEGMENT and never a substring, so a name that
    merely contains a refused token is scratch; the ambiguous directory names
    (``objects``, ``targets``) stay out of the list, because a false refusal
    teaches the caller to route around the pad and put the litter back in the
    user's tree.
    """
    root = _pad(tmp_path)
    url = "scratchpad://" + "/".join(segments)

    assert check_scratchpad_write(root.joinpath(*segments), root, url) is None


@pytest.mark.parametrize("name", ["out", "obj", "build", "dist", "target", "node_modules"])
def test_a_token_shaped_leaf_is_a_file_while_the_directory_of_that_name_is_not(
    tmp_path: Path, name: str
) -> None:
    """The leaf/parent pair, pinned together so the narrowing cannot be
    implemented by deleting the arm. M1: the segment arms judge the PARENT parts,
    because the same spelling is a build TREE as a directory and a file name as a
    leaf — and a leaf called ``out.json`` is data work, not a build tree.

    ``node_modules`` is the case the leaf rule is for: a file of that name is
    writeable, and everything written THROUGH it is refused one level earlier.
    """
    root = _pad(tmp_path)

    assert check_scratchpad_write(root / f"{name}.json", root, f"scratchpad://{name}.json") is None
    assert check_scratchpad_write(root / name, root, f"scratchpad://{name}") is None
    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(root / name / "x.md", root, f"scratchpad://{name}/x.md")

    assert f"'{name}' is a build or dependency directory" in str(excinfo.value)


def test_a_refused_name_in_an_ancestor_of_the_root_does_not_refuse_the_write(
    tmp_path: Path,
) -> None:
    """Only what is BELOW the root is judged, and this is the case that makes
    that necessary rather than tidy: the path is absolute, and this fleet's own
    checkouts sit under worktrees whose ancestors are named like refused ones.
    Judging every segment of the absolute path would make every write in such a
    pad a refusal.
    """
    root = _pad(tmp_path, "build", "target")

    assert check_scratchpad_write(root / "notes.md", root, "scratchpad://notes.md") is None
    # ...and the rule is not disarmed for that root: a refused segment BELOW it
    # is still refused, which is what tells the two halves apart.
    with pytest.raises(ScratchpadContentError):
        check_scratchpad_write(
            root / "node_modules" / "x.js", root, "scratchpad://node_modules/x.js"
        )


def test_the_segment_rule_survives_a_symlinked_root(tmp_path: Path) -> None:
    """``_scratchpad_root`` hands back the context's string as a plain ``Path``
    while the parsed target is ``Path.resolve()``d, so a pad reached through a
    symlink is the ordinary case rather than the exotic one — ``/tmp`` is
    ``/private/tmp`` on macOS, and every pytest ``tmp_path`` sits under one of
    those. The two spellings share no textual prefix, so it is the resolved
    second attempt that relates them; without it the pair would be unplaceable,
    which is now a refusal too, but for the wrong reason and at the cost of
    every write into a symlinked pad.
    """
    real = _pad(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ScratchpadContentError):
        check_scratchpad_write(
            (link / "node_modules" / "x.js").resolve(), link, "scratchpad://node_modules/x.js"
        )


@pytest.mark.parametrize(
    ("target", "root"),
    [
        (Path("/c/node_modules/x.js"), Path("/a/b/scratchpad")),
        (Path("/a/b/scratchpad/node_modules/x.js"), Path("/a/b/other")),
        (
            Path("/tmp/probe2/sessions/s1/scratchpad/node_modules/x.js"),
            Path("sessions/s1/scratchpad"),
        ),
    ],
)
def test_a_path_that_cannot_be_placed_inside_the_pad_is_refused(target: Path, root: Path) -> None:
    """The spellings that reach the branch where neither containment attempt
    relates the target to the root: a mismatched absolute pair, and a relative
    root against an absolute target.

    This is the arm the reviewer flagged as failing OPEN — judging the bare file
    name there would ALLOW ``node_modules/x.js`` in exactly the case where the
    name cannot be trusted, which is the outcome the helper exists to prevent.
    It is unreachable through the tools (``parse_scratchpad_url`` has already
    proven containment), so this is the function-level pin for the defence in
    depth, and it is the honest place to prove it.
    """
    with pytest.raises(ScratchpadContentError) as excinfo:
        check_scratchpad_write(target, root, "scratchpad://node_modules/x.js")

    assert "could not be placed inside the pad" in str(excinfo.value)


def test_every_content_refusal_ends_with_the_same_elsewhere_sentence(tmp_path: Path) -> None:
    """The constant claims to be the sentence EVERY content refusal ends with,
    and the reviewer's probe was literally ``endswith`` — so all four arms are
    pinned to it here.

    The size arm used to restate the alternatives inline with "build output
    belongs in a git worktree", which is wrong for the caller it addressed: an
    oversized shaped extract is not build output, and a second copy of the
    advice is a second copy that drifts from the first.
    """
    root = _pad(tmp_path)
    attempts = [
        (root / "node_modules" / "x.js", root, "scratchpad://node_modules/x.js", None),
        (root / "artefact.o", root, "scratchpad://artefact.o", None),
        (root / "dump.csv", root, "scratchpad://dump.csv", SCRATCHPAD_MAX_WRITE_BYTES + 1),
        (
            Path("/c/node_modules/x.js"),
            Path("/a/b/scratchpad"),
            "scratchpad://node_modules/x.js",
            None,
        ),
    ]
    for target, where, url, size in attempts:
        with pytest.raises(ScratchpadContentError) as excinfo:
            check_scratchpad_write(target, where, url, size)

        assert str(excinfo.value).endswith(SCRATCHPAD_ELSEWHERE), url
