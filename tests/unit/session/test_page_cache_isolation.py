"""The between-tests reset for the process-wide page cache.

``tests/conftest.py::reset_page_cache_between_tests`` resets
``local_operator.session.page_cache`` before and after every test, because the
cache keys pages on ``(directory, st_ino, st_size)`` and a test that rewrites a
journal at an unchanged identity serves the previous reader's page (CI run
38037554993 shard 3, job 114171921826 — the fixture's docstring carries the
story). Without the reset that is cross-test state, and the earlier test's page
answers the later test's read.

Both cases drive the boundary INSIDE one test, through the same
``reset_page_cache()`` the autouse fixture calls. A two-test pair cannot do
this: pytest-xdist dispatches tests to workers individually and ``-n auto
--dist worksteal`` is this repo's own addopts, so the pair splits across
workers and passes on the unfixed tree — the same reason
``tests/unit/test_root_logger_filter_isolation.py`` crosses its boundary
inside one test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.session.page_cache import (
    load_transcript_page,
    page_cache,
    page_key,
    reset_page_cache,
)
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
    TranscriptPage,
)


def _write_row(directory: Path, content: str) -> int:
    """One journal row; returns the file's size after the write.

    ``write_text`` truncates and rewrites IN PLACE, so a second call keeps the
    inode — which is half of what the collision below needs, the other half
    being equal byte length (``"alpha"`` and ``"bravo"`` are twins).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRANSCRIPT_FILENAME
    path.write_text(
        TranscriptEntry("row-1", 1.0, ENTRY_MESSAGE, {"role": "user", "content": content}).to_json()
        + "\n"
    )
    return path.stat().st_size


def _served(page: TranscriptPage) -> str:
    return page.entries[0].payload["content"]


@pytest.mark.asyncio
async def test_a_same_identity_rewrite_is_stale_until_the_boundary_reset(tmp_path: Path) -> None:
    """The CI collision, deterministic, crossed inside one test.

    The rewrite moves neither half of the key, so the cached ``alpha`` page
    answers a read of a file that now holds ``bravo``. That stale serve is
    asserted as the live poison — it is the residual the page-cache module
    docstring names, and if it ever stops happening this pin must be re-derived
    rather than pass vacuously — and only the boundary reset turns the next
    read fresh.
    """
    directory = tmp_path / "sess"
    size_a = _write_row(directory, "alpha")
    key = page_key(directory, limit=10)
    assert key is not None, "precondition: the journal must exist"
    assert _served(await load_transcript_page(directory, limit=10)) == "alpha"
    assert page_cache().get(key) is not None, "precondition: the first read was cached"

    size_b = _write_row(directory, "bravo")
    assert size_b == size_a, "precondition: the rewrite must not move the size"
    assert page_key(directory, limit=10) == key, "precondition: the identity must not move"

    stale = await load_transcript_page(directory, limit=10)
    assert (
        _served(stale) == "alpha"
    ), "precondition: the poison must be live, or the final read proves nothing"

    # The boundary: exactly the call `reset_page_cache_between_tests` makes
    # between two tests.
    reset_page_cache()
    assert page_cache().entry_count == 0

    fresh = await load_transcript_page(directory, limit=10)
    assert _served(fresh) == "bravo", "the reset must not let the old page answer again"


def test_the_between_tests_reset_is_registered_and_autouse(
    request: pytest.FixtureRequest,
) -> None:
    """The conftest fixture must be part of every test's closure.

    This is the half the case above cannot see: if the fixture is removed or
    renamed (or stops being autouse), that case still passes, because it resets
    explicitly. This one goes red instead of letting the suite reacquire a
    cross-test flake generator.
    """
    assert (
        "reset_page_cache_between_tests" in request.fixturenames
    ), "tests/conftest.py's autouse page-cache reset is missing from this test's fixtures"
