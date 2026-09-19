"""Shared fixtures for the API client tests."""

from typing import Callable

import pytest
import requests


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
