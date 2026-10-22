"""Shared isolation for the Library unit tests.

The indexer is a process-global singleton with a daemon thread and a filesystem
observer. A test that reads through the service starts one lazily, and a worker
that outlives its test keeps watching a deleted ``tmp_path`` — so every test
starts and ends with none.
"""

from __future__ import annotations

import pytest

from local_operator.library import indexer


@pytest.fixture(autouse=True)
def _no_leaked_indexer():
    indexer.reset_for_tests()
    yield
    indexer.reset_for_tests()
