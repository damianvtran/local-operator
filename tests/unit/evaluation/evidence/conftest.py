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

WHY IT IS SCOPED TO THIS PACKAGE, AND WHAT WAS MEASURED BEFORE DECIDING. The
obvious follow-up is "apply this to the whole tree". Both tree-wide shapes were
built and measured, and both are worse than the narrow guard:

* **Redirecting ``tempfile.tempdir`` tree-wide breaks AF_UNIX sockets.** A
  redirected temp dir sits under pytest's basetemp, which is deep: a realistic
  socket path measured 138 bytes against darwin's 104-byte ``sun_path`` limit,
  and binding one fails with ``AF_UNIX path too long``. Nothing binds a socket
  under the temp dir today, so this is latent rather than active — but it makes
  a tree-wide redirect a trap for the next person who does.
* **A read-only snapshot detector flakes.** Not redirecting, and instead
  diffing the shared temp dir per test, false-positives on sibling activity:
  with one other process writing into the same ``TMPDIR`` (what concurrent
  worktrees do here routinely), a probe flagged 3 of 3 deliberately clean
  tests. That is the broad-but-flaky guard that gets disabled a week later.

So the leaks outside this package were fixed at their call sites instead — they
were ordinary in-process ``mkdtemp()`` calls with a ``tmp_path`` fixture already
available — and the guard stays where the hazard is structural: evidence
bundles are unbounded in size (a bundle holds published artifacts) and are
built by subprocess scripts where ``tmp_path`` cannot reach. A reliable narrow
guard beats a broad flaky one. If bundle-writing tests appear elsewhere, import
``private_tempdir`` there rather than widening this.
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

#: Carries the teardown leak verdict from the fixture to the report hook.
_LEAK_REPORT: pytest.StashKey[str] = pytest.StashKey[str]()


@pytest.fixture(autouse=True)
def private_tempdir(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Point this test's temp allocations at a private dir and assert it drains.

    Both ``tempfile.tempdir`` and ``TMPDIR`` are set: the former covers
    in-process ``mkdtemp``/``NamedTemporaryFile`` calls, the latter covers
    subprocesses, which is where this suite's leaks actually were. ``tmp_path``
    itself is unaffected — pytest resolves its basetemp once at session start,
    before this fixture runs — so tests that take ``tmp_path`` keep working.

    The verdict is stashed rather than raised here. An exception from fixture
    teardown is reported as an ERROR on an otherwise-passing test, which reads
    like infrastructure trouble; ``pytest_runtest_makereport`` below turns it
    into a FAILURE on the test that actually leaked.
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

    request.node.stash[_LEAK_REPORT] = (
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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Report a leak as a FAILURE of the test, not a teardown ERROR.

    The check can only run in teardown (the directory has to be observed after
    the test is finished with it), but an exception raised there is classified
    as an error, so a leaking run reads ``N passed, 1 error`` — which looks
    like a broken fixture rather than "this test leaked". Rewriting the
    teardown report to a failure keeps the exit code non-zero (it already was)
    while making the summary say the true thing.

    A genuine teardown error is left alone: only a clean teardown carrying a
    stashed leak verdict is rewritten.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "teardown" or report.failed:
        return
    message = item.stash.get(_LEAK_REPORT, None)
    if message is None:
        return
    report.outcome = "failed"
    report.longrepr = message
    # Consumed by pytest_report_teststatus below to relabel the category.
    report.leaked_tempdir = True  # type: ignore[attr-defined]


def pytest_report_teststatus(report: pytest.TestReport, config: pytest.Config):
    """Count a leak as a failure in the summary line, not an error.

    ``_pytest.runner`` maps ANY failed setup/teardown report to the "error"
    category unconditionally, so rewriting the report's outcome alone still
    prints ``N passed, 1 error``. Claiming the category here is the only way
    to make the summary read ``1 failed`` — which is what a leak is: the test
    did its job and then left the disk dirty, not a broken fixture.
    """
    if getattr(report, "leaked_tempdir", False):
        return "failed", "F", "FAILED"
    return None


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
