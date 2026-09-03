"""Fail the evidence suite when a test abandons a temporary directory.

WHY THIS EXISTS. Three tests in this package created bundle roots with a bare
``tempfile.mkdtemp()`` and never removed them. Because they run the writer
inside an embedded ``script = r\"\"\"...\"\"\"`` subprocess, pytest's ``tmp_path``
was never in play and nothing reclaimed the directories. Each run of this
package leaked three bundles, one of them 64 MiB (the streaming-redaction RSS
test publishes a 64 MiB artifact). Measured on the operator's machine: 43,627
abandoned bundle directories totalling ~30 GB, growing ~194 per 10 minutes
while suites ran, until the disk hit 94% full. The leak was invisible because
every test passed the whole time.

Fixing the three call sites removes today's leak. This fixture is what stops
it silently coming back: a new bundle test that forgets cleanup fails here
instead of quietly costing a gigabyte a day.

WHY A PRIVATE TEMP DIR PER TEST RATHER THAN A SESSION-WIDE SNAPSHOT. The
obvious shape — record the temp dir before the session, diff it after — cannot
be made reliable here. This suite runs under ``xdist`` (``-n auto`` via
``addopts``), so sibling workers, sibling worktrees, and unrelated processes on
the developer's machine all write into the same shared ``TMPDIR`` concurrently.
A diff would attribute their entries to this run and flake, and the usual
answer to that (ignore anything you did not recognise) is exactly what let the
original leak hide.

Redirecting ``tempfile.tempdir`` to a per-test directory removes the ambiguity
instead of tolerating it: the directory is created for one test, is visible to
nothing else, and inherits into subprocesses through ``TMPDIR`` so the embedded
scripts are covered too. Anything left in it afterwards was leaked by that
test, with no shared-state reasoning required. The check is therefore exact,
and it names the offending test rather than the session.

WHY IT IS SCOPED TO THIS PACKAGE. A tree-wide version would be a much larger
behavioural change — some suites cache under ``gettempdir()`` deliberately, and
code that reads the temp dir at import time would not see the redirect — for no
extra safety on the leak that actually hurt. Evidence bundles are the case that
matters: they are unbounded in size (a bundle holds published artifacts) and
they are built by subprocess scripts where ``tmp_path`` does not reach. A
reliable narrow guard beats a broad flaky one. If bundle-writing tests appear
elsewhere, import ``private_tempdir`` there rather than widening this.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

#: Entries that are legitimately shared process-wide caches rather than
#: per-test scratch, and so are not leaks if they appear. ``data-gym-cache`` is
#: tiktoken's BPE table, which ``tests/conftest.py`` warms deliberately and
#: which is keyed off ``gettempdir()``; it is cheap to re-download but pointless
#: to flag. Keep this list short — every entry is a hole in the guard.
_SHARED_CACHE_NAMES = frozenset({"data-gym-cache"})


@pytest.fixture(autouse=True)
def private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point this test's temp allocations at a private dir and assert it drains.

    Both ``tempfile.tempdir`` and ``TMPDIR`` are set: the former covers
    in-process ``mkdtemp``/``NamedTemporaryFile`` calls, the latter covers
    subprocesses, which is where this suite's leaks actually were. ``tmp_path``
    itself is unaffected — pytest resolves its basetemp once at session start,
    before this fixture runs — so tests that take ``tmp_path`` keep working.
    """
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()

    # tempfile caches the resolved directory in a module global, so setting the
    # env var alone would not redirect in-process calls in an interpreter that
    # has already called gettempdir(). Set both.
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setenv("TMPDIR", str(scratch))

    yield

    leaked = sorted(
        entry.name for entry in scratch.iterdir() if entry.name not in _SHARED_CACHE_NAMES
    )
    if not leaked:
        return

    # Report the shape, not just the names: a bundle-shaped leak (state.json
    # inside) is the regression this guard exists for, and the byte count is
    # what makes "so what?" answerable for a reader seeing this fail.
    details = []
    for name in leaked:
        entry = scratch / name
        total = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        shape = "bundle" if (entry / "bundle" / "state.json").exists() else "dir"
        details.append(f"  {name} ({shape}, {total} bytes)")

    raise AssertionError(
        "This test left "
        f"{len(leaked)} entr{'y' if len(leaked) == 1 else 'ies'} in its temporary "
        "directory:\n"
        + "\n".join(details)
        + "\n\nEvidence bundles are unbounded on disk and nothing else reclaims them "
        "(see this file's module docstring: ~30 GB accumulated this way). Wrap the "
        "allocation in `tempfile.TemporaryDirectory()` or use the `tmp_path` fixture. "
        "If the directory is created inside an embedded subprocess script, the SCRIPT "
        "must clean up in a `finally`/context manager \u2014 `tmp_path` cannot reach it."
    )


def pytest_configure(config: pytest.Config) -> None:
    """Fail loudly if the guard is disabled by an ambient TMPDIR override.

    ``monkeypatch.setenv`` in the fixture wins for the code under test, so this
    is only a sanity check that the temp dir is writable at all; a read-only or
    missing ``TMPDIR`` would otherwise surface as an unrelated error inside
    whichever test allocated first.
    """
    probe = os.environ.get("TMPDIR")
    if probe and not os.path.isdir(probe):
        raise pytest.UsageError(
            f"TMPDIR={probe!r} is not a directory; the evidence suite needs one"
        )
