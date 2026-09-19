"""Shared fixtures for the unit suite.

Most of this file is the API-client helpers; the two capability fixtures at the
bottom belong to the approval gate's authority model (issue #1310) and are here
rather than in a runtime-local conftest because the tests that need them are
spread across ``tests/unit/session``, ``tests/unit/tui`` and ``tests/unit/server``
— they all build an in-process registrant and then dial it as a follower.
"""

import os
from collections.abc import Iterator
from typing import Callable

import pytest
import requests

from local_operator.harness.approval import (
    mint_operator_cap,
    remember_operator_cap,
    reset_operator_caps_for_tests,
)


@pytest.fixture(autouse=True)
def _isolated_operator_caps() -> Iterator[None]:
    """The operator capability table is process-global (issue #1310).

    Autouse, and reset at BOTH ends, because the failure mode of a leak is a
    later test whose ``AttachClient`` presents a capability the runtime it dials
    never received: that reads as a flake in whichever test happened to run
    next, not in the one that leaked. One entry per spawned runtime is the
    production shape, so nothing legitimate depends on surviving a test.
    """
    reset_operator_caps_for_tests()
    yield
    reset_operator_caps_for_tests()


@pytest.fixture
def operator_cap() -> Iterator[bytes]:
    """The capability for an IN-PROCESS registrant, registered as this process's.

    A test that builds ``RuntimeServer(handle, kind="tui")`` directly is both
    the runtime and the console, and its record carries ``os.getpid()`` — the
    same key ``AttachClient.connect`` resolves against — so wiring the value
    under this process's pid is what makes a follower in the test capable of an
    authority-increasing op, exactly as the process that spawned a detached
    runtime is. Pass the returned value to the registrant:

    ``RuntimeServer(handle, kind="tui", operator_cap=operator_cap)``

    Tests that do NOT take this fixture keep the capability-free registrant, and
    are the ones that pin the refusal.
    """
    capability = mint_operator_cap()
    remember_operator_cap(os.getpid(), capability)
    yield capability


@pytest.fixture
def real_response() -> Callable[[int, bytes], requests.Response]:
    """Build a real ``requests.Response``, as the wire would hand it over.

    The callers of these helpers are *responses*, not mocks, and that difference
    is load-bearing rather than stylistic: ``requests.Response.__bool__`` returns
    ``response.ok``, so a 4xx/5xx response is falsy while a ``MagicMock`` is
    always truthy. A mock therefore cannot exercise the falsy-response path --
    which is exactly the path that made every failed call report "No response
    body" while the server was sending one, and why the pre-existing mocked
    tests stayed green through it.

    Returns:
        Callable[[int, bytes], requests.Response]: ``(status, body) -> response``.
    """

    def _build(status_code: int, body: bytes) -> requests.Response:
        response = requests.Response()
        response.status_code = status_code
        response.url = "https://api.radienthq.com/v1/tools/transcriptions"
        # `_content` false and `_content_consumed` true is what Response.content
        # reads after a non-streamed request completed.
        response._content = body
        response._content_consumed = True
        return response

    return _build
