"""Per-test isolation for state the bridge client keeps for the WHOLE process."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from local_operator.team_bridge import client as tb_client


@pytest.fixture(autouse=True)
def _empty_directory_cache() -> Iterator[None]:
    """Every test starts and ends with no cached directory answer.

    ``TeamBridgeClient.directory_cached`` holds answers process-wide (the tool
    builds a client per call, so a per-instance cache would never be hit), and
    whether an answer came from that cache decides whether a MISS re-reads the
    directory. A test that leaves an entry behind — or finds one — therefore
    changes the request count the NEXT test observes, which is exactly how the
    h5d name-miss test became order-dependent.
    """

    tb_client._DIRECTORY_CACHE.clear()
    yield
    tb_client._DIRECTORY_CACHE.clear()
