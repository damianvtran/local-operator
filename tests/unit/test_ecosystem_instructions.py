"""Imported user-scope instructions (``~/.agents/AGENTS.md``).

The loader's job is small; the properties that matter are the ones a
regression would make invisible — that lop never WRITES these files, that a
duplicate of ``system_prompt.md`` is not paid for twice on every cached
request, and that a bad file costs no session.

``HOME`` is redirected to ``tmp_path`` by the autouse ``isolate_environment``
fixture, so these tests reach a scratch home rather than the developer's real
``~/.agents`` — which on a machine that has one would otherwise inject 7k of
the developer's own preferences into every assertion here.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator import ecosystem_instructions as eco


def _write_agents_md(home: Path, text: str) -> Path:
    path = home / ".agents" / "AGENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_the_shared_agents_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The feature in one line: instructions kept for other agent tools reach
    lop without being copied into a lop-specific filename."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "- Use conventional commits.")

    assert eco.load_ecosystem_instructions() == "- Use conventional commits."


def test_no_shared_file_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert eco.load_ecosystem_instructions() == ""
    assert eco.ecosystem_instruction_files() == []


def test_the_env_override_replaces_the_default_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirectable, not merely switchable — the shape
    ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` established."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "- DEFAULT PATH")
    elsewhere = tmp_path / "dotfiles" / "shared.md"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("- OVERRIDE PATH", encoding="utf-8")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, str(elsewhere))

    out = eco.load_ecosystem_instructions()

    assert out == "- OVERRIDE PATH"
    assert "DEFAULT PATH" not in out, "the override REPLACES the set, never extends it"


def test_the_env_override_reads_several_paths_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("- FIRST", encoding="utf-8")
    second.write_text("- SECOND", encoding="utf-8")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, f"{first}:{second}")

    assert eco.load_ecosystem_instructions() == "- FIRST\n\n- SECOND"


def test_an_empty_override_disables_the_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who wants ONLY ``system_prompt.md`` gets exactly that,
    without needing a second on/off variable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "- Shared rule.")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, "")

    assert eco.load_ecosystem_instructions() == ""


def test_a_tilde_in_the_override_is_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "- Shared rule.")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, "~/.agents/AGENTS.md")

    assert eco.load_ecosystem_instructions() == "- Shared rule."


def test_content_identical_to_the_native_file_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync-script case, which is the CURRENT workaround this replaces.
    Both copies surviving would double the cost of the thing being fixed, on
    every cached request of every session."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "- Shared rule.\n")

    out = eco.load_ecosystem_instructions(
        skip_digests=frozenset({eco.content_digest("- Shared rule.")})
    )

    assert out == "", "trailing-whitespace drift must not defeat the dedup"


def test_different_content_survives_the_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "- Shared rule.")

    out = eco.load_ecosystem_instructions(skip_digests=frozenset({eco.content_digest("- Other.")}))

    assert out == "- Shared rule."


def test_duplicate_override_paths_collapse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two entries whose content matches contribute once — the same rule the
    native comparison uses, applied within the set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("- Same rule.", encoding="utf-8")
    second.write_text("- Same rule.", encoding="utf-8")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, f"{first}:{second}")

    assert eco.load_ecosystem_instructions() == "- Same rule."


def test_a_symlinked_file_is_followed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberately the OPPOSITE of ``context_files``' O_NOFOLLOW policy: this
    path is in the operator's own home, and pointing it at a dotfiles checkout
    is the main reason to want shared instructions at all."""
    monkeypatch.setenv("HOME", str(tmp_path))
    real = tmp_path / "dotfiles" / "AGENTS.md"
    real.parent.mkdir(parents=True)
    real.write_text("- Versioned in dotfiles.", encoding="utf-8")
    link = tmp_path / ".agents" / "AGENTS.md"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)

    assert eco.load_ecosystem_instructions() == "- Versioned in dotfiles."


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX alarm only")
def test_a_fifo_reaching_the_reader_is_refused_instead_of_blocking(tmp_path: Path) -> None:
    """The guard ``_read_bounded`` documents, exercised where it actually runs.

    ``ecosystem_instruction_files`` filters fifos out via ``is_file()``, so the
    ordinary arrangement never reaches this code — which is exactly why the
    guard needs its own test rather than riding on the loader-level one below.
    The case it defends is the TOCTOU window: a regular file swapped for a fifo
    after that listing, reproduced here by calling the reader directly.

    Guarded by ``SIGALRM``, matching ``test_a_fifo_is_refused_without_being_opened``
    in the composer's suite, because the regression is an INFINITE BLOCK inside
    a synchronous ``os.open`` on the session-construction path, which has no
    timeout above it. Without the alarm, dropping ``O_NONBLOCK`` would not fail
    this test — it would hang CI with no report and nothing to read.
    """
    fifo = tmp_path / "AGENTS.md"
    os.mkfifo(fifo)

    def _blocked(signum, frame):  # noqa: ANN001 - signal handler signature
        raise AssertionError("the reader blocked on a fifo instead of refusing it")

    previous = signal.signal(signal.SIGALRM, _blocked)
    signal.alarm(10)
    try:
        with pytest.raises(OSError, match="not a regular file"):
            eco._read_bounded(fifo)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX alarm only")
def test_a_fifo_at_the_path_costs_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-loader view of the same hazard: a named pipe left at the
    shared path degrades to no imported instructions, and startup proceeds.

    Alarmed for the same reason as the test above — the failure being pinned is
    a hang, not a wrong return value.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".agents").mkdir(parents=True)
    os.mkfifo(tmp_path / ".agents" / "AGENTS.md")

    def _blocked(signum, frame):  # noqa: ANN001 - signal handler signature
        raise AssertionError("load_ecosystem_instructions blocked on a fifo")

    previous = signal.signal(signal.SIGALRM, _blocked)
    signal.alarm(10)
    try:
        assert eco.load_ecosystem_instructions() == ""
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_the_nonblocking_open_does_not_short_read_a_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``O_NONBLOCK`` is set for the fifo guard above, so pin that it costs the
    ordinary case nothing. It has no meaning for regular files, but a partial
    read here would silently truncate everyone's instructions rather than
    fail — the kind of regression a fifo test would never catch."""
    monkeypatch.setenv("HOME", str(tmp_path))
    body = "".join(f"- Rule {n}.\n" for n in range(4000))
    assert len(body.encode("utf-8")) > 32 * 1024, "too small to exercise a multi-chunk read"
    _write_agents_md(tmp_path, body)

    assert eco.load_ecosystem_instructions() == body.strip()


def test_a_directory_at_the_path_degrades_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".agents" / "AGENTS.md").mkdir(parents=True)

    assert eco.load_ecosystem_instructions() == ""


def test_an_unreadable_file_degrades_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad file must never cost the operator their session."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _write_agents_md(tmp_path, "- Shared rule.")
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):  # pragma: no cover — root ignores the mode
            pytest.skip("running as root; the mode is not enforced")
        assert eco.load_ecosystem_instructions() == ""
    finally:
        path.chmod(0o644)


def test_undecodable_bytes_cost_a_glyph_not_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".agents" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"- Keep \xff this rule.")

    assert "Keep" in eco.load_ecosystem_instructions()
    assert "this rule." in eco.load_ecosystem_instructions()


def test_a_bom_never_reaches_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``utf-8-sig`` the ``\\ufeff`` survives ahead of the first rule."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".agents" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xef\xbb\xbf- First rule.")

    assert eco.load_ecosystem_instructions() == "- First rule."


def test_a_whitespace_only_file_contributes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "\n\n   \n")

    assert eco.load_ecosystem_instructions() == ""


def test_one_pathological_file_is_bounded_at_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_agents_md(tmp_path, "z" * (eco.MAX_FILE_BYTES * 4))

    assert len(eco.load_ecosystem_instructions()) == eco.MAX_FILE_BYTES


def test_the_loader_never_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-only is the guarantee that keeps ``system_prompt.md`` the single
    write target of Settings and ``PATCH /v1/config/system-prompt``. Asserted
    against the filesystem, because "we simply do not call write" is exactly
    the kind of claim a later refactor invalidates silently."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _write_agents_md(tmp_path, "- Shared rule.")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    listing_before = sorted(p.name for p in (tmp_path / ".agents").iterdir())

    eco.load_ecosystem_instructions()

    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in (tmp_path / ".agents").iterdir()) == listing_before


def test_the_default_path_sits_beside_the_skills_directory_lop_already_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the CHOICE of directory, which is the whole argument of the issue:
    ``~/.agents/skills`` is already an ecosystem skill root, so the AGENTS.md
    beside it needs no new relationship with a path lop does not know."""
    from local_operator.skills.api import _ECOSYSTEM_SKILL_SUBDIRS

    assert eco.ECOSYSTEM_INSTRUCTION_PATHS == (Path(".agents") / "AGENTS.md",)
    assert Path(".agents") / "skills" in _ECOSYSTEM_SKILL_SUBDIRS


def test_the_module_stays_stdlib_only() -> None:
    """It is imported at module level by ``session_factory``, the composition
    root every host funnels through, so a third-party import here lands on the
    startup path of every invocation.

    Asserted as the PROPERTY — what actually landed in ``sys.modules`` — rather
    than by scanning the source for known spellings, which would pass for any
    third-party package not named in the scan and for anything reached
    indirectly. Run in a FRESH SUBPROCESS for the reason
    ``tests/unit/test_import_graph.py`` documents: by the time a test body runs
    pytest has imported half the tree, so an in-process check would find a
    wrongly-imported module already present and report nothing.
    """
    probe = (
        "import json, sys;"
        "before = set(sys.modules);"
        "__import__('local_operator.ecosystem_instructions');"
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(eco.__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-3000:]}"

    added = json.loads(proc.stdout.strip().splitlines()[-1])
    # Top-level names only: ``sys.stdlib_module_names`` lists roots, not
    # submodules, so ``collections.abc`` has to be tested as ``collections``.
    # ``local_operator`` itself is the package being imported, not a dependency.
    top_level = {name.split(".")[0] for name in added} - {"local_operator"}
    offenders = sorted(top_level - sys.stdlib_module_names)

    assert not offenders, f"non-stdlib imports on the startup path: {offenders}"
    assert "local_operator.ecosystem_instructions" in added, "probe never imported the module"
    intra = sorted(n for n in added if n.startswith("local_operator."))
    assert intra == [
        "local_operator.ecosystem_instructions"
    ], f"no intra-package imports on this path; saw {intra}"
