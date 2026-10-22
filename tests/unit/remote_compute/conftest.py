"""Test-context guards for the remote-compute slice, applied to every test here.

WHY A CONSTRUCTION GUARD. The ``no_real_network`` fixtures in the test modules
refuse a request at the TRANSPORT, which only fires once something has already
built a client with a transport that can reach Radient. A review round still
reached the real API from a test that built ``httpx.AsyncClient()`` with no
transport at all — nothing in that module's fixtures had inspected the client it
handed to the code under test, and httpx's default transport is a real one.

So this makes the CONSTRUCTION the failure instead: under this directory a test
may only build an ``AsyncClient`` that cannot leave the machine, and a client
built without an injected transport fails the test that built it rather than
making a live call. It is autouse and directory-scoped on purpose — a guard a
new test file has to remember to import is a guard that gets left out.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture(autouse=True)
def require_injected_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that builds ``httpx.AsyncClient`` without a transport."""
    original = httpx.AsyncClient.__init__

    def guarded(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        if kwargs.get("transport") is None:
            raise AssertionError(
                "a remote-compute test built httpx.AsyncClient without an injected "
                "transport, so its requests would reach the real API; pass "
                "MockTransport (or patch the module seam) instead"
            )
        original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", guarded)
