"""The bundle carrier's client half: raw upload, streamed download (design §8).

What is pinned here is the SHAPE of each call -- route, target params, raw body,
``allow_redirects=False`` -- and the two properties a caller depends on: the
download bound is enforced on bytes RECEIVED (not on a header) and a 304 is a
success with no body. The archive's own validation is covered in
``tests/unit/hub_sync/test_skill_bundle.py``.
"""

from __future__ import annotations

import io
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients._http import APIError
from local_operator.clients.radient import RadientClient, SkillBundleDownload


@pytest.fixture
def base_url() -> str:
    return "https://api.test.radient.com"


@pytest.fixture
def radient_client(base_url: str) -> RadientClient:
    return RadientClient(api_key=SecretStr("test_api_key"), base_url=base_url)


def _envelope(result: Any, *, status: int = 201) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {"msg": "ok", "result": result}
    return response


def _stream(status: int, chunks: list[bytes], headers: Dict[str, str] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = dict(headers or {})
    response.iter_content.return_value = iter(chunks)
    response.raise_for_status.return_value = None
    return response


def test_upload_posts_the_raw_archive_to_the_bundle_route(
    radient_client: RadientClient, base_url: str
) -> None:
    accepted = {
        "hash": "a" * 64,
        "size": 12,
        "uncompressed_size": 40,
        "file_count": 2,
        "entrypoint": "SKILL.md",
        "tree_hash": "b" * 64,
        "name": "pdf-forms",
        "description": "Fill and validate PDF forms.",
    }
    with patch(
        "local_operator.clients.radient.requests.post", return_value=_envelope(accepted)
    ) as post:
        result = radient_client.upload_skill_bundle(
            b"PK\x03\x04payload", visibility="org", tenant_id="t1"
        )

    url = post.call_args.args[0]
    call = post.call_args.kwargs
    assert url == f"{base_url}/skills/bundles"
    assert call["data"] == b"PK\x03\x04payload"
    assert call["headers"]["Content-Type"] == "application/zip"
    assert call["params"] == {"visibility": "org", "tenant_id": "t1"}
    assert call["allow_redirects"] is False
    assert result == accepted


def test_upload_without_a_target_is_the_public_hub(
    radient_client: RadientClient, base_url: str
) -> None:
    with patch("local_operator.clients.radient.requests.post", return_value=_envelope({})) as post:
        radient_client.upload_skill_bundle(b"PK\x03\x04payload")
    assert post.call_args.kwargs["params"] == {}


def test_upload_refuses_oversize_bytes_before_sending(radient_client: RadientClient) -> None:
    from local_operator.hub_sync.skill_bundle import MAX_ARCHIVE_BYTES

    with patch("local_operator.clients.radient.requests.post") as post:
        with pytest.raises(ValueError):
            radient_client.upload_skill_bundle(b"x" * (MAX_ARCHIVE_BYTES + 1))
    post.assert_not_called()


def test_upload_refuses_empty_bytes_before_sending(radient_client: RadientClient) -> None:
    with patch("local_operator.clients.radient.requests.post") as post:
        with pytest.raises(ValueError):
            radient_client.upload_skill_bundle(b"")
    post.assert_not_called()


def test_download_streams_the_archive_and_reports_its_headers(
    radient_client: RadientClient, base_url: str
) -> None:
    response = _stream(
        200,
        [b"PK\x03\x04", b"body"],
        headers={"ETag": '"rev-7"', "X-Skill-Bundle-Sha256": "c" * 64},
    )
    with patch("local_operator.clients.radient.requests.get", return_value=response) as get:
        result = radient_client.download_skill_bundle("sk-1", expected_revision='"rev-6"')

    call = get.call_args.kwargs
    assert get.call_args.args[0] == f"{base_url}/skills/sk-1/bundle"
    assert call["headers"]["If-None-Match"] == '"rev-6"'
    assert call["headers"]["Authorization"] == "Bearer test_api_key"
    assert call["stream"] is True
    assert call["allow_redirects"] is False
    assert result == SkillBundleDownload(
        data=b"PK\x03\x04body", etag='"rev-7"', server_hash="c" * 64
    )
    response.close.assert_called_once()


def test_download_can_stream_into_a_caller_file_object(
    radient_client: RadientClient,
) -> None:
    response = _stream(200, [b"PK\x03\x04", b"body"])
    destination = io.BytesIO()
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        result = radient_client.download_skill_bundle("sk-1", dest=destination)

    assert result.data is None
    assert destination.getvalue() == b"PK\x03\x04body"


def test_download_enforces_its_cap_on_received_bytes(radient_client: RadientClient) -> None:
    """A lying ``Content-Length`` cannot get a body past the bound."""

    response = _stream(200, [b"x" * 6, b"x" * 6])
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        with pytest.raises(APIError) as caught:
            radient_client.download_skill_bundle("sk-1", max_bytes=10)

    assert (caught.value.status_code, caught.value.code) == (413, "payload_too_large")
    assert caught.value.details == {"limit_bytes": 10}
    response.close.assert_called_once()


def test_download_refuses_a_bound_above_the_carrier(radient_client: RadientClient) -> None:
    from local_operator.hub_sync.skill_bundle import MAX_ARCHIVE_BYTES

    with patch("local_operator.clients.radient.requests.get") as get:
        with pytest.raises(ValueError):
            radient_client.download_skill_bundle("sk-1", max_bytes=MAX_ARCHIVE_BYTES + 1)
    get.assert_not_called()


def test_download_304_is_a_success_with_no_body(radient_client: RadientClient) -> None:
    response = _stream(304, [], headers={"ETag": '"rev-6"'})
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        result = radient_client.download_skill_bundle("sk-1", expected_revision='"rev-6"')

    assert result.not_modified is True
    assert result.data is None and result.etag == '"rev-6"'
    response.iter_content.assert_not_called()


def test_download_refuses_a_redirect(radient_client: RadientClient) -> None:
    response = _stream(302, [])
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        with pytest.raises(APIError) as caught:
            radient_client.download_skill_bundle("sk-1")

    assert caught.value.code == "unexpected_redirect"


def test_download_without_the_bearer_for_a_public_row(
    radient_client: RadientClient,
) -> None:
    response = _stream(200, [b"PK\x03\x04"])
    with patch("local_operator.clients.radient.requests.get", return_value=response) as get:
        radient_client.download_skill_bundle("sk-1", bearer=False)
    assert "Authorization" not in get.call_args.kwargs["headers"]


def test_download_maps_a_refusal_to_its_machine_readable_half(
    radient_client: RadientClient,
) -> None:
    response = MagicMock()
    response.status_code = 404
    response.content = b'{"error":"skill not found","code":"skill_not_found"}'
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error", response=response
    )
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        with pytest.raises(APIError) as caught:
            radient_client.download_skill_bundle("sk-1")

    assert (caught.value.status_code, caught.value.code) == (404, "skill_not_found")


# -- capability probe ------------------------------------------------------------------


def test_probe_reads_the_contract_from_one_skill_row(
    radient_client: RadientClient, base_url: str
) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "msg": "ok",
        "result": {
            "skills": [],
            "sync_contract": {"revision_cas": 1, "skills": 1, "skill_bundles": 1},
        },
    }
    response.raise_for_status.return_value = None
    with patch("local_operator.clients.radient.requests.get", return_value=response) as get:
        observed = radient_client.probe_skill_capabilities()

    assert observed == {"revision_cas": 1, "skills": 1, "skill_bundles": 1}
    args, kwargs = get.call_args
    assert args[0] == f"{base_url}/skills"
    assert kwargs["params"] == {"per_page": 1}
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"]["Authorization"] == "Bearer test_api_key"


def test_probe_tolerates_a_top_level_contract(radient_client: RadientClient) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "sync_contract": {"skill_bundles": 1},
        "result": {"skills": []},
    }
    response.raise_for_status.return_value = None
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        observed = radient_client.probe_skill_capabilities()

    assert observed == {"skill_bundles": 1}


def test_probe_returns_empty_when_no_contract_was_sent(radient_client: RadientClient) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"msg": "ok", "result": {"skills": []}}
    response.raise_for_status.return_value = None
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        assert radient_client.probe_skill_capabilities() == {}


def test_probe_refuses_a_redirect(radient_client: RadientClient) -> None:
    response = MagicMock()
    response.status_code = 302
    response.raise_for_status.return_value = None
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        with pytest.raises(APIError) as caught:
            radient_client.probe_skill_capabilities()

    assert getattr(caught.value, "code", None) == "unexpected_redirect"


def test_probe_refuses_a_non_json_body_without_echoing_it(
    radient_client: RadientClient,
) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    response.raise_for_status.return_value = None
    with patch("local_operator.clients.radient.requests.get", return_value=response):
        with pytest.raises(APIError) as caught:
            radient_client.probe_skill_capabilities()

    assert "unrecognised" in str(caught.value)
