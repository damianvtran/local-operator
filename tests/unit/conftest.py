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


@pytest.fixture(autouse=True)
def _connector_never_reaches_launchd(monkeypatch: pytest.MonkeyPatch) -> None:
    """No unit test may reach the real launchd through the connector (D2a).

    D2a makes the connector's service lifecycle run on ordinary events — an
    enrolment commit, a ``team_bridge.enabled`` write, an app-start boundary —
    and those events are reachable from ``tests/unit/server`` (the serve
    lifespan, the status route, settings edits) and from anything that reads a
    status boundary, not just from ``tests/unit/team_bridge``. So the seam is
    guarded HERE, at the package-wide root, and the guard is a hard failure
    rather than a no-op: ``launchctl`` has no sandbox and always addresses the
    calling user's live ``gui/<uid>`` domain, whatever ``HOME`` says.

    This is the SECOND layer. The first is the real product guard
    (``launchd.shared_label_refusal``), which refuses every store the suite can
    construct; its own behaviour is pinned in ``test_launchd.py`` and the wakes
    installer tests. A test that deliberately opens that guard (as the D2a
    lifecycle tests do) still cannot reach launchd through this.

    ``raising=False`` keeps the fixture from being itself the failure on a tree
    that predates the seam. The import is function-local anyway, so this module
    stays importable from ``tests/unit`` without the bridge stack at module
    scope; because the fixture is autouse, the FIRST test of every worker pays
    it once and the rest read ``sys.modules``.
    """

    from local_operator.team_bridge.connector import install

    def forbidden(*args: str):
        raise AssertionError(
            "a unit test reached the connector's launchctl seam; inject "
            "install._launchctl or leave the shared-label guard closed"
        )

    monkeypatch.setattr(install, "_launchctl", forbidden, raising=False)
