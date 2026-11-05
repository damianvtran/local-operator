"""Resolve one URL into a card, under ONE wall-clock deadline for the whole call.

THE DEADLINE IS THE POINT. The renderer holds a promise while it waits, so the
worst case cannot be "the page's timeout plus the icon's plus the image's": a
slow-drip origin would then hold the card for as long as it liked and the UI
would have to invent its own abort. One deadline is computed here, every hop and
every retry of every sub-fetch spends from it, and the resolution answers by it
whether or not the origin cooperated.

THE FETCHER IS NOT OURS. Every request goes through
:class:`~local_operator.web_fetch.service.WebFetchService` — the existing
hardened fetcher, with its per-hop ``validate_public_url`` re-validation and its
address pinning. Nothing here re-implements transport, and nothing here can
bypass the address check; the only thing this module does with a refusal is name
it.

NOTHING LOGGED CAN IDENTIFY WHAT WAS READ. A log line carrying the URL would put
a correspondent's links (and their query strings, which routinely carry tokens)
into a file that outlives the message. The only identifiers here are coarse
outcome codes and the content-addressed artifact id, which names bytes the origin
already served publicly. (One caveat, not this module's to fix: httpx writes its
own ``HTTP Request: GET <url>`` line on the ``httpx`` logger, which
``logger.quiet_wire_clients`` pins shut at the daemon's level — ``LOG_LEVEL``
defaults to WARNING — and which every other outbound request in the process
already writes at INFO or below. Raising ``LOG_LEVEL`` therefore exposes every
preview URL, exactly as it exposes every provider call.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Final
from urllib.parse import urlsplit

from local_operator.team_bridge.link_preview.extract import (
    Raster,
    decode_markup,
    extract_metadata,
    is_html_type,
    looks_like_markup,
    sniff_raster,
)
from local_operator.team_bridge.link_preview.models import (
    AbsentReason,
    FaviconArtifact,
    ImageArtifact,
    LinkPreview,
    LinkPreviewResolution,
)
from local_operator.team_bridge.link_preview.store import PreviewStore
from local_operator.web_fetch.models import FetchBytes, WebFetchSettings
from local_operator.web_fetch.service import FetchError, PolicyRefusal, WebFetchService

logger = logging.getLogger(__name__)

#: The whole resolution's budget, all hops included. Sized to sit under a UI
#: promise rather than to give the network a fair chance: a card that has not
#: arrived in eight seconds is worth less than the conversation it interrupts.
TOTAL_DEADLINE_S: Final = 8.0

#: The page may spend at most this much of the total, because the media that make
#: a card worth drawing are fetched AFTER it. An origin that stalls on HTML must
#: not consume the budget its own favicon needs.
PAGE_DEADLINE_S: Final = 4.0

#: One media artifact's slice of what remains. Not a per-fetch guarantee: whatever
#: the page left is the ceiling for BOTH, and this keeps the second one from
#: spending all of the remainder.
MEDIA_DEADLINE_S: Final = 3.0

#: Enough markup for any real ``<head>``. What we cannot find in this much of a
#: page is not what its author wanted a card to say.
PAGE_MAX_BYTES: Final = 512 * 1024

#: A page that DECLARES more than this is not a link-card target. Refusing on the
#: declared length is the honest reading of ``too-large`` at this layer: we would
#: otherwise spend the budget reading a prefix of something no card can describe.
PAGE_REFUSE_BYTES: Final = 8 * 1024 * 1024

FAVICON_MAX_BYTES: Final = 256 * 1024
IMAGE_MAX_BYTES: Final = 2 * 1024 * 1024

#: Previews per message, enforced HERE rather than trusted from the caller. The
#: renderer asks for one per link it finds; a message with forty links would
#: otherwise become forty outbound fetches from one paste, and the cap has to be
#: server-side because a modified renderer is not a threat model we get to assume
#: away. ``index`` is the link's position among its message's links, so ``3`` and
#: above answer ``cap`` without a single request.
PER_MESSAGE_CAP: Final = 3

#: Below this much remaining budget a fetch cannot finish, so starting one only
#: burns the deadline it is measured against.
_MIN_SLICE_S: Final = 0.05


class _Expired(Exception):
    """The total deadline is spent. Internal: it never leaves this module."""


async def resolve_preview(
    *,
    url: str,
    index: int,
    settings: WebFetchSettings,
    store: PreviewStore,
    fetcher: WebFetchService | None = None,
) -> LinkPreviewResolution:
    """Resolve ``url`` into a card, or say in one code why there is none.

    ``fetcher`` exists for tests that must drive a fixture origin without a real
    socket; production passes ``None`` and the settings' own service is built.
    Either way the transport is :class:`WebFetchService`.
    """
    # The cap is checked FIRST and returns before any parse, any DNS and any
    # request: an over-cap index must cost nothing at all.
    if index < 0 or index >= PER_MESSAGE_CAP:
        return _absent("cap")

    target = url.strip()
    scheme_refusal = _scheme_refusal(target)
    if scheme_refusal is not None:
        return _absent(scheme_refusal)

    service = fetcher or WebFetchService(settings)
    deadline = time.monotonic() + TOTAL_DEADLINE_S

    try:
        page = await _fetch(
            service,
            target,
            ceiling=PAGE_MAX_BYTES,
            deadline=deadline,
            slice_seconds=PAGE_DEADLINE_S,
        )
    except _Expired:
        return _error(retryable=True)
    except PolicyRefusal as refusal:
        return _absent(_policy_reason(refusal.reason))
    except FetchError as error:
        return _error(retryable=_retryable(error))

    # A refusal or a broken origin is not a page. 4xx can be a WAF challenge and
    # 5xx can be a rolling deploy, so the class decides whether another try could
    # help rather than the preview guessing.
    if page.status >= 400:
        return _error(retryable=page.status >= 500 or page.status in (408, 429))

    declared = _declared_length(page.headers)
    if declared is not None and declared > PAGE_REFUSE_BYTES:
        return _absent("too-large")

    content_type = page.headers.get("content-type", "")
    if content_type and not is_html_type(content_type):
        return _absent("content-type")
    if not looks_like_markup(page.body):
        # Declared HTML (or said nothing) and carries no markup at all: a PDF, a
        # plain-text README, a binary. There is nothing to read, and rendering it
        # as markup would invent tags that were never there.
        return _absent("not-html")

    markup = decode_markup(page.body, content_type)
    metadata = extract_metadata(markup, base_url=page.final_url)
    if metadata.is_empty:
        return _absent("no-metadata")

    favicon = await _favicon(service, metadata.icon_url, deadline=deadline, store=store)
    image = await _image(service, metadata.image_url, deadline=deadline, store=store)

    logger.debug(
        "link preview resolved: title=%s description=%s favicon=%s image=%s",
        bool(metadata.title),
        bool(metadata.description),
        "yes" if favicon else "no",
        "yes" if image else "no",
    )
    return LinkPreviewResolution(
        kind="ready",
        preview=LinkPreview(
            url=page.final_url,
            title=metadata.title,
            description=metadata.description,
            site_name=metadata.site_name,
            favicon=favicon,
            image=image,
        ),
    )


async def _fetch(
    service: WebFetchService,
    url: str,
    *,
    ceiling: int,
    deadline: float,
    slice_seconds: float,
) -> FetchBytes:
    """One sub-fetch, capped by the call's remaining budget and its own slice.

    THE OUTER ``wait_for`` IS LOAD-BEARING, and it is not redundant with
    ``timeout_seconds`` below. The fetcher's timeout is per OPERATION (it lands on
    httpx's ``read``/``write``/``connect``), so a slow drip — headers at once,
    then a byte every 100 ms — resets it forever and the call would run as long as
    the origin kept dripping. Measured here on a real socket before this landed: a
    0.5 s budget answered in 15.3 s. The whole point of a preview is that a
    promise the renderer is holding cannot be held open by a stranger, so the
    total is enforced from OUTSIDE, by cancelling the fetch at the deadline. The
    cancellation closes the connection and the fetcher's own ``finally`` closes
    its pool, so nothing is leaked by the abort.
    """
    remaining = min(deadline - time.monotonic(), slice_seconds)
    if remaining < _MIN_SLICE_S:
        raise _Expired
    try:
        return await asyncio.wait_for(
            service.fetch_bytes(url, max_bytes=ceiling, timeout_seconds=remaining),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        raise _Expired from None


async def _favicon(
    service: WebFetchService, url: str | None, *, deadline: float, store: PreviewStore
) -> FaviconArtifact | None:
    if not url:
        return None
    stored = await _store_artifact(
        service, url, ceiling=FAVICON_MAX_BYTES, deadline=deadline, store=store
    )
    if stored is None:
        return None
    artifact_id, raster = stored
    return FaviconArtifact(id=artifact_id, mime=raster.mime)


async def _image(
    service: WebFetchService, url: str | None, *, deadline: float, store: PreviewStore
) -> ImageArtifact | None:
    if not url:
        return None
    stored = await _store_artifact(
        service, url, ceiling=IMAGE_MAX_BYTES, deadline=deadline, store=store
    )
    if stored is None:
        return None
    artifact_id, raster = stored
    return ImageArtifact(id=artifact_id, mime=raster.mime, width=raster.width, height=raster.height)


async def _store_artifact(
    service: WebFetchService,
    url: str,
    *,
    ceiling: int,
    deadline: float,
    store: PreviewStore,
) -> tuple[str, Raster] | None:
    """Fetch one image and store it, or ``None`` — an optional artifact never fails a card.

    Every refusal here is silent by design: a page whose og:image is an SVG, a
    mislabelled HTML error page, a 4 MB photo or a dead CDN still has a title and
    a description worth drawing, and turning any of those into an error would
    throw away the whole card over its decoration.
    """
    try:
        fetched = await _fetch(
            service, url, ceiling=ceiling, deadline=deadline, slice_seconds=MEDIA_DEADLINE_S
        )
    except (_Expired, FetchError):
        return None
    if fetched.status >= 400 or not fetched.body:
        return None
    if not fetched.complete:
        # The body hit the ceiling, so the bytes are a PREFIX of an image: every
        # format's header survives, which is exactly why storing it would be a
        # silent lie (a card would draw a half-pixel strip and never notice).
        return None
    raster = sniff_raster(fetched.body)
    if raster is None:
        return None
    return store.put(fetched.body), raster


def _scheme_refusal(target: str) -> AbsentReason | None:
    """Refuse a non-http(s) scheme before anything resolves it.

    A bare host has no scheme and is NOT refused: the fetcher normalises it to
    https, and a link pasted without ``https://`` is the commonest case there is.
    """
    if not target:
        return "no-metadata"
    scheme = urlsplit(target).scheme.lower()
    if scheme and scheme not in ("http", "https"):
        return "unsupported-scheme"
    return None


def _policy_reason(reason: str) -> AbsentReason:
    """The fetcher's refusal code as the card's own vocabulary.

    ``private`` covers a redirect hop as well as the first URL: the guard runs on
    every hop, and the renderer's copy — "this link goes somewhere local" — is
    the same sentence for both.
    """
    if reason == "scheme":
        return "unsupported-scheme"
    if reason == "private":
        return "local-or-private"
    return "blocked-host"


def _retryable(error: FetchError) -> bool:
    """Whether another try could help, from the fetcher's own classification.

    An unclassified ``FetchError`` (a redirect chain that never landed, a host
    with no address) is treated as retryable: the card did not resolve, and the
    honest answer to "could this work later" is that we do not know.
    """
    failure = getattr(error, "failure", None)
    if failure is None:
        return True
    return bool(getattr(failure, "retryable", False))


def _declared_length(headers: dict[str, str]) -> int | None:
    """The origin's declared ``Content-Length``, or ``None`` when it is unusable."""
    raw = headers.get("content-length")
    if not raw:
        return None
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _absent(reason: AbsentReason) -> LinkPreviewResolution:
    return LinkPreviewResolution(kind="absent", reason=reason)


def _error(*, retryable: bool) -> LinkPreviewResolution:
    return LinkPreviewResolution(kind="error", retryable=retryable)
