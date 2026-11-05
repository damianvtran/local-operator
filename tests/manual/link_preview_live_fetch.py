"""Live evidence for the link-preview resolver: real DNS, real sockets, real bytes.

RUN IT BY HAND, on a machine with network access:

    PYTHONPATH=<repo> .venv/bin/python tests/manual/link_preview_live_fetch.py

WHY IT IS NOT A UNIT TEST. It fetches two public pages over the open internet
through :class:`~local_operator.web_fetch.service.WebFetchService`'s REAL transport.
Nothing here is mocked except the request counter, which wraps the real transport
rather than replacing it. A CI job that depended on two third-party sites would
fail for reasons that have nothing to do with this code, so this file lives outside
the suite and is run deliberately.

WHAT IT SHOWS, in the order the surface does it:

1. Two public URLs resolve end to end — one that publishes OpenGraph metadata and
   one more that carries an icon — printing the resolution kind, whether a title
   and a site name came back, and for each artifact its mime, byte length and
   sha256.
2. A second resolve of the same URL is a CACHE HIT: the origin request count does
   not move, because the answer is reused inside its positive TTL.
3. The artifact bytes the artifact route would serve hash to the same sha256 as the
   bytes that were stored — the served bytes ARE the cached bytes.
4. ``stats()`` reports the entries and resident bytes the cache is holding.

WHAT IT PRINTS. The URL's scheme, host and path — never a query string or a token,
and this script refuses to run against a URL that carries one. Nothing is written
to a file and nothing is logged: stdout is the whole record.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path

import httpx

from local_operator.server.routes import desktop_link_preview
from local_operator.team_bridge.link_preview.cache import (
    PreviewScope,
    get_preview_cache,
)
from local_operator.team_bridge.link_preview.models import LinkPreviewResolution
from local_operator.team_bridge.link_preview.service import (
    preview_fetcher,
    resolve_preview,
)
from local_operator.team_bridge.link_preview.store import PreviewStore
from local_operator.web_fetch.models import WebFetchSettings

#: Deliberately boring, deliberately public. ``ogp.me`` is the OpenGraph protocol's
#: own page (title, description, site name and an image), and ``python.org`` is a
#: large site whose icon is a real raster. Neither needs a login and neither is
#: behind a challenge, so a failure here is this code's failure and not a WAF's.
URLS = ("https://ogp.me/", "https://www.python.org/")

#: The account the run pretends to be signed in as. A real run has a real one; the
#: point of the scope here is that the ids and the cache entries are scoped, not
#: that this particular string means anything.
ACCOUNT = "acc_0000000000000001"


class _CountingTransport(httpx.AsyncBaseTransport):
    """The REAL httpx transport with a request counter in front of it.

    A wrapper, not a substitute: every request still goes out over TCP through
    ``httpx.AsyncHTTPTransport``, which is what makes the fetch count evidence
    rather than a claim.
    """

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _public(url: str) -> str:
    """The URL as it may be printed: scheme, host and path only."""

    parts = httpx.URL(url)
    return f"{parts.scheme}://{parts.host}{parts.path}"


def _hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _describe(resolution: LinkPreviewResolution) -> None:
    print(f"  kind          : {resolution.kind}")
    if resolution.kind != "ready" or resolution.preview is None:
        print(f"  reason        : {resolution.reason}  retryable={resolution.retryable}")
        return
    preview = resolution.preview
    print(f"  title         : {'yes' if preview.title else 'no'}  ({preview.title[:60]!r})")
    print(f"  description   : {'yes' if preview.description else 'no'}")
    print(f"  siteName      : {'yes' if preview.site_name else 'no'} ({preview.site_name!r})")
    print(f"  resolved url  : {_public(preview.url)}")
    for label, artifact in (("favicon", preview.favicon), ("image", preview.image)):
        if artifact is None:
            print(f"  {label:<13} : absent")
            continue
        print(f"  {label:<13} : id={artifact.id} mime={artifact.mime}")
    # The byte-level facts come from the store, through the route's own read path.


async def main() -> int:
    settings = WebFetchSettings(
        enrich=False,
        max_attempts=1,
        timeout_seconds=15.0,
        max_bytes=2 * 1024 * 1024,
        cache_ttl_seconds=0,  # the preview cache is the cache under test, not web_fetch's
    )
    transport = _CountingTransport()
    # ONE process-wide cache, exactly as the route reaches it — not a private one,
    # or the artifact step below would be checking somebody else's entries.
    cache = get_preview_cache()
    scope = PreviewScope(account_id=ACCOUNT)
    config_dir = Path(tempfile.mkdtemp(prefix="link-preview-live-"))
    store = PreviewStore.from_config(config_dir)
    print(f"artifact store: {store.root}")
    print(f"scope         : account={scope.account_id} tenant={scope.tenant_id!r}\n")

    failures: list[str] = []
    fetcher = preview_fetcher(settings, transport=transport)
    try:
        for url in URLS:
            if "?" in url:
                raise SystemExit("this script will not print a URL carrying a query string")
            before = transport.requests
            resolution = await resolve_preview(
                url=url,
                index=0,
                scope=scope,
                settings=settings,
                store=store,
                cache=cache,
                fetcher=fetcher,
            )
            print(f"{_public(url)}")
            _describe(resolution)
            print(f"  requests      : {transport.requests - before}")
            print(f"  resolutions   : {cache.stats().resolutions}")
            if resolution.kind != "ready":
                failures.append(f"{_public(url)} resolved {resolution.kind}/{resolution.reason}")
                print()
                continue

            # --- the artifact route's own read path, for each artifact ---------
            for label, artifact in (
                ("favicon", resolution.preview.favicon if resolution.preview else None),
                ("image", resolution.preview.image if resolution.preview else None),
            ):
                if artifact is None:
                    continue
                served = await desktop_link_preview._read_owned_artifact(
                    config_dir, scope, artifact.id
                )
                if served is None:
                    failures.append(f"{_public(url)}: {label} {artifact.id} was not served")
                    print(f"  {label:<13} : NOT SERVED (route would 404)")
                    continue
                body, mime = served
                cached = store.get(artifact.id) or b""
                stored_hash = _hash(cached)
                print(f"  {label:<13} : bytes={len(body)} sha256={_hash(body)} mime={mime}")
                print(f"  {label+' (store)':<13}: bytes={len(cached)} sha256={stored_hash}")
                if _hash(body) != stored_hash:
                    failures.append(f"{_public(url)}: {label} served bytes differ from the store")

            # --- the same URL again: the origin must not be contacted ----------
            requests_before_repeat = transport.requests
            repeat = await resolve_preview(
                url=url,
                index=0,
                scope=scope,
                settings=settings,
                store=store,
                cache=cache,
                fetcher=fetcher,
            )
            new_requests = transport.requests - requests_before_repeat
            print(f"  repeat        : kind={repeat.kind} requests={new_requests} (cache hit)")
            if new_requests:
                failures.append(f"{_public(url)}: the repeat fetched {new_requests} more times")
            print()

        stats = cache.stats()
        print(
            f"cache: entries={stats.entries} resident_bytes={stats.resident_bytes} "
            f"hits={stats.hits} misses={stats.misses} resolutions={stats.resolutions}"
        )
        print(f"keys: {[key.canonical_url for key in cache.keys()]}")
    finally:
        await transport.aclose()

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK: both URLs resolved over the real transport, cached, and served back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
