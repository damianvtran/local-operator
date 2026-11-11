"""Music's narrow authenticated Radient transport, with no upstream vendor key.

No generic paid retries: submit carries a durable idempotency key and a successor
can replay the same request after a lost response without buying another song.
"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urlparse

import httpx

from local_operator.clients._http import APIError
from local_operator.env import resolve_radient_api_base_url
from local_operator.imagegen.media import download_asset
from local_operator.providers.radient_credentials import resolve_radient_credential
from local_operator.stt.errors import api_error_from_httpx_response


class MusicClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        config_dir: Any = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = resolve_radient_api_base_url(base_url).rstrip("/")
        self.config_dir = config_dir
        self.http = client or httpx.AsyncClient(follow_redirects=False)
        self._owns_http = client is None

    async def close(self) -> None:
        if self._owns_http:
            await self.http.aclose()

    async def request(
        self,
        action: str,
        *,
        body: dict[str, Any] | None = None,
        request_id: str | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        credential = await resolve_radient_credential(self.config_dir, self.base_url)
        token = credential.get_secret_value()
        if not token:
            raise APIError("radient_auth_required", status_code=401)
        headers = {"Authorization": f"Bearer {token}"}
        if key:
            headers["Idempotency-Key"] = key
        method = "POST" if action in {"generate", "cancel"} else "GET"
        response = await self.http.request(
            method,
            f"{self.base_url}/tools/music/{action}",
            headers=headers,
            json=body if method == "POST" else None,
            params={"request_id": request_id} if request_id else None,
            timeout=60.0,
        )
        if response.status_code >= 400:
            raise api_error_from_httpx_response(
                response, fallback_message="music_request_failed", secrets=(token,)
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise APIError("invalid_music_response", status_code=None)
        return payload

    async def download(self, artifact: dict[str, Any]) -> Any:
        url = artifact.get("url")
        # Validate the actual URL before parsing so downstream downloads receive
        # a string, not an unchecked optional value from the provider payload.
        if not isinstance(url, str):
            raise APIError("invalid_music_artifact_origin", status_code=None)
        origin = urlparse(self.base_url)
        parsed = urlparse(url)
        # A signed playback grant is not permission to fetch arbitrary origins.
        # Bearer credentials are never sent to the artifact URL, even on-origin.
        if (
            parsed is None
            or parsed.scheme != origin.scheme
            or parsed.netloc != origin.netloc
            or parsed.username
            or not parsed.path.startswith(origin.path.rstrip("/") + "/tools/audio/artifacts/")
        ):
            raise APIError("invalid_music_artifact_origin", status_code=None)
        asset = await download_asset(
            url,
            client=self.http,
            fallback_content_type="audio/mpeg",
            duration_s=artifact.get("duration_seconds"),
        )
        expected = artifact.get("sha256")
        if not isinstance(expected, str) or hashlib.sha256(asset.data).hexdigest() != expected:
            raise APIError("music_artifact_hash_mismatch", status_code=None)
        if artifact.get("size_bytes") != len(asset.data):
            raise APIError("music_artifact_size_mismatch", status_code=None)
        return asset
