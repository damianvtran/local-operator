"""``GET /v1/capabilities`` and the desktop document route for turn supplements.

Lives under ``tests/unit/server`` because the ``test_app_client`` fixture does. The owner
record, the attach gate and the relay routes are in ``tests/unit/supplements/test_surfaces.py``;
this file owns the desktop route's contract: status codes, headers verbatim, digest-only
addressing, and the served document's sandbox pairing (CSP meta + JSON, never ``text/html``).
"""

from __future__ import annotations

import pytest

from local_operator.session.attachments import AttachmentStore

DIGEST = "04c2d29b140086cc637ad19ef34012c5"
SESSION = "a1b2c3d4e5f6"

#: A stored component body: exactly what ``SupplementRunner._store`` puts under the
#: digest (the generated body, no prelude, no wrapper).
COMPONENT = '<p class="lead">Q3 looks solid.</p>\n<div id="chart"></div>\n'


@pytest.mark.asyncio
async def test_v1_capabilities_advertises_supplements(test_app_client) -> None:
    response = await test_app_client.get("/v1/capabilities")
    assert response.status_code == 200
    features = response.json()["result"]["features"]
    # `>= 1` is the comparison clients make (`desktopFeatureState`); 0 reads as absent
    assert features.get("supplements", 0) >= 1


def _desktop_client(tmp_path, monkeypatch):
    """The real app behind the desktop bearer boundary (the shape the search-route tests use).

    ``/v1/desktop/`` answers 503 until the desktop token is configured, so a bare
    ``test_app_client`` would test the boundary and never reach the route.
    """
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from local_operator.server.app import app
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    return TestClient(app), {"Authorization": "Bearer token"}


def _session(tmp_path, session_id: str = SESSION):
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    return directory


def _store(tmp_path, blob: str = COMPONENT) -> str:
    ref = AttachmentStore(root=tmp_path / "attachments").put_bytes(
        blob.encode("utf-8"), "text/html"
    )
    assert ref is not None
    return ref.digest


def test_the_desktop_document_route_serves_the_assembled_document(tmp_path, monkeypatch) -> None:
    """The happy path, with the executing-document family's headers verbatim and the
    served bytes byte-equal to the one assembler's output."""
    from local_operator.supplements.document import assemble_stored_component

    digest = _store(tmp_path)
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
    assert response.status_code == 200
    # JSON, never a navigable document: no host may point a frame at a URL (memo §4.1).
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    html = response.json()["html"]
    assert html == assemble_stored_component(COMPONENT)
    # The sandbox pairing: the document carries its own policy (the tunnel strips
    # response headers, so the CSP must ride inside -- memo §2.7), network closed.
    assert '<meta http-equiv="Content-Security-Policy"' in html
    assert "default-src 'none'" in html and "connect-src 'none'" in html


def test_the_desktop_document_route_answers_404_for_a_missing_blob(tmp_path, monkeypatch) -> None:
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{DIGEST}/document", headers=headers
        )
    assert response.status_code == 404


def test_the_desktop_document_route_refuses_a_non_component_blob(tmp_path, monkeypatch) -> None:
    """Bytes that cannot be assembled (a malformed ``<data>`` block) are an ordinary
    404 -- the reader's remedy is the frame-level fallback line, not an error page."""
    digest = _store(tmp_path, "<data>{oops</data><p>tail</p>")
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
    assert response.status_code == 404


def test_the_desktop_document_route_keeps_the_digest_gate(tmp_path, monkeypatch) -> None:
    """A non-digest path never reaches the handler: 422, the traversal gate."""
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            "/v1/desktop/sessions/abc/supplements/not-a-digest/document", headers=headers
        )
    assert response.status_code == 422


def test_the_desktop_document_route_carries_the_policy_on_errors(tmp_path, monkeypatch) -> None:
    """D5's pair rides the family's error responses too (#2134 R4).

    The ladder's 404 and the digest gate's 422 both carry ``nosniff`` +
    ``no-store``, verbatim like the 200 (the unauthorised response is pinned in the
    bearer test below).
    """
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        missing = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{DIGEST}/document", headers=headers
        )
        non_digest = client.get(
            "/v1/desktop/sessions/abc/supplements/not-a-digest/document", headers=headers
        )
    assert missing.status_code == 404
    assert non_digest.status_code == 422
    for response in (missing, non_digest):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"


def test_the_desktop_document_route_has_no_caller_supplied_path(tmp_path, monkeypatch) -> None:
    """There is no path parameter to supply, and a smuggled one changes nothing: the
    document served is the digest's, the same as the clean request."""
    digest = _store(tmp_path)
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        clean = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
        smuggled = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document?path=/etc/passwd",
            headers=headers,
        )
    assert clean.status_code == smuggled.status_code == 200
    assert clean.json()["html"] == smuggled.json()["html"]


def test_the_desktop_document_route_is_behind_the_bearer(tmp_path, monkeypatch) -> None:
    client, _headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(f"/v1/desktop/sessions/abc/supplements/{DIGEST}/document")
    assert response.status_code in (401, 403)
    # The family policy rides the unauthorised response too (#2134 R4, D5).
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
