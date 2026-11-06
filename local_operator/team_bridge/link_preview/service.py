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
it. TWO OF THAT FETCHER'S OPTIONS ARE FORCED OFF HERE (:func:`preview_fetcher`):
``allow_private``, because a pasted link is not a link the reader chose to fetch,
and ``trust_env``, because an inherited ``HTTP_PROXY`` would route a stranger's
URL somewhere this process does not control.

WHAT ONE RESOLUTION REMEMBERS, AND FOR WHOM. The answer is cached under
``(account, tenant, canonical URL)`` and the artifact ids inside it are derived
from that same triple, so two accounts previewing one link hold two entries and
two files. Resolving is a cache read when it can be, one shared origin pass when
several renderers ask at once, and the bounded fetch below only on a miss.

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
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from local_operator.team_bridge.link_preview.cache import PreviewCache, PreviewScope
from local_operator.team_bridge.link_preview.cache import artifact_id as artifact_id_for
from local_operator.team_bridge.link_preview.cache import key_for
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
from local_operator.team_bridge.link_preview.throttle import PreviewThrottled, ResolveThrottle
from local_operator.web_fetch.models import FetchBytes, WebFetchSettings
from local_operator.web_fetch.service import (
    FetchError,
    PolicyRefusal,
    WebFetchService,
    url_has_credentials,
)

if TYPE_CHECKING:
    # Annotation only: this module rides ``server.app``'s import path, and the
    # transport it names is already imported by ``web_fetch.service``.
    import httpx

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
    scope: PreviewScope,
    settings: WebFetchSettings,
    store: PreviewStore,
    cache: PreviewCache,
    fetcher: WebFetchService | None = None,
    throttle: ResolveThrottle | None = None,
) -> LinkPreviewResolution:
    """Resolve ``url`` into a card for ``scope``, or say in one code why there is none.

    THE CACHE IS CONSULTED BEFORE ANYTHING COSTS A REQUEST. A fresh entry answers
    from memory; concurrent callers of one key share ONE origin pass
    (:meth:`~local_operator.team_bridge.link_preview.cache.PreviewCache.singleflight);
    and only a miss reaches :func:`_resolve_origin`, the deadline-bounded resolver
    every hop of which runs under.

    ``scope`` is load-bearing, never decoration: the cached answer, the artifact
    ids inside it and the blob behind each id are all keyed by it, so the same URL
    for two accounts is two entries and two files. ``fetcher`` exists for tests
    that must drive a fixture origin without a real socket; production passes
    ``None`` and :func:`preview_fetcher` builds one.
    """
    # THE KILL SWITCH, FIRST — before the cap, before the URL is even looked at.
    # ``web_fetch.enabled: false`` is what a privacy- or airgap-minded operator
    # flips, and the setting's own documentation promises that "no fetching" means
    # no fetching through ANY spelling (``web_fetch/tool.py`` re-checks it per
    # call for exactly that reason). A preview is one of those spellings, so with
    # the switch off it answers ``absent/disabled`` and issues ZERO requests — the
    # round-1 security finding S-1, measured as two outbound fetches from a
    # stranger's link on a switched-off install.
    if not settings.enabled:
        return _absent("disabled")

    # The cap is checked next and returns before any parse, any DNS, any cache
    # read and any request: an over-cap index must cost nothing at all.
    if index < 0 or index >= PER_MESSAGE_CAP:
        return _absent("cap")

    # The URL that is FETCHED and RETURNED stays the caller's; only the key is
    # normalised (see the cache module). A bare host keeps its scheme-less form
    # here and is absolutised by the fetcher, which is why the card reports the
    # URL that actually answered rather than the string it was handed.
    target = url.strip()
    scheme_refusal = _scheme_refusal(target)
    if scheme_refusal is not None:
        return _absent(scheme_refusal)

    # A URL CARRYING CREDENTIALS IS REFUSED HERE, ON THE RAW SPELLING, BEFORE THE
    # KEY IS BUILT (QA Q2-1). ``canonical_url`` DROPS userinfo — that is what
    # keeps a credential out of a key that outlives the request — so
    # ``https://user:secret@host/x`` and ``https://host/x`` are ONE key. Refusing
    # only in the fetcher therefore has two holes this early return closes, and
    # both were measured:
    #
    # * the credentialed URL could be answered ``ready`` from the plain spelling's
    #   already-cached card, because a hit never reaches the fetcher at all — a
    #   card for a URL this contract refuses to resolve;
    # * worse, the credentialed URL's ``blocked-host`` was written back under that
    #   SHARED key and then served to the legitimate plain URL as a negative
    #   answer for the full half-hour ``NEGATIVE_TTL_S``, with no fetch — one
    #   sender's malformed link denying the card everyone else is entitled to.
    #
    # The refusal costs a parse and no request, and it is deliberately NOT cached:
    # nothing is learned here that a later caller may not learn on its own terms.
    # The engine-level refusal in ``validate_public_url`` stays as the depth
    # defence — it re-checks every redirect hop, which this parse never sees.
    if url_has_credentials(target):
        return _absent("blocked-host")

    key = key_for(scope, target)
    cached = cache.get(key)
    if cached is not None:
        return cached

    # THE TOKEN IS SPENT HERE, and only here — after the cache miss, before the
    # flight. Charging at the HTTP boundary instead made a re-render of already
    # cached links spend tokens it never used: seven messages × three links is
    # twenty-one resolves on a purely WARM screen, which exhausted a burst of twenty
    # and started refusing real work (code review round 2, minor). A cap refusal
    # above returns before this line, so it costs nothing either. A stampede of N
    # identical COLD requests does spend N tokens for one collapsed fetch — that is
    # deliberate: a client hammering one URL is precisely what the bound exists for.
    if throttle is not None and not throttle.allow(scope.account_id):
        raise PreviewThrottled(throttle.retry_after_s(scope.account_id))

    async def _resolve_and_remember() -> LinkPreviewResolution:
        resolution = await _resolve_origin(
            target=target,
            settings=settings,
            store=store,
            scope=scope,
            canonical=key.canonical_url,
            fetcher=fetcher,
        )
        cache.put(key, resolution, _retained_bytes(store, resolution))
        return resolution

    return await cache.singleflight(key, _resolve_and_remember)


async def _resolve_origin(
    *,
    target: str,
    settings: WebFetchSettings,
    store: PreviewStore,
    scope: PreviewScope,
    canonical: str,
    fetcher: WebFetchService | None,
) -> LinkPreviewResolution:
    """One pass at the origin: page, then favicon, then image, under one deadline.

    This is what a cache miss costs. ``canonical`` never reaches the network — it
    only names the artifact slots the fetcher's bytes are stored under, so a
    refetched card overwrites its own blobs instead of piling up new ones.
    """
    service = fetcher or preview_fetcher(settings)
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
    # Relative hrefs resolve against the URL that ANSWERED; using the caller's
    # spelling would resolve an icon against a <base> the origin never served.
    metadata = extract_metadata(markup, base_url=page.final_url)
    if metadata.is_empty:
        return _absent("no-metadata")

    favicon = await _favicon(
        service,
        metadata.icon_url,
        deadline=deadline,
        store=store,
        artifact_id=artifact_id_for(scope, canonical, "favicon"),
    )
    image = await _image(
        service,
        metadata.image_url,
        deadline=deadline,
        store=store,
        artifact_id=artifact_id_for(scope, canonical, "image"),
    )

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
            # ``or None``, not ``or ""``: an absent display string is a wire shape
            # (``string | null``, omitted when None), while the extractor keeps
            # empty strings internally because that is how its parsers accumulate.
            title=metadata.title or None,
            description=metadata.description or None,
            site_name=metadata.site_name or None,
            favicon=favicon,
            image=image,
        ),
    )


def preview_fetcher(
    settings: WebFetchSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> WebFetchService:
    """The preview's ONE fetcher: public targets only, and no proxy from the environment.

    TWO OPTIONS ARE REFUSED RATHER THAN HONOURED, for the same reason: a preview is
    driven by whatever a correspondent pasted into a message, so it is the one
    fetch in this process whose target the READER never chose.

    * ``allow_private`` is the tool's escape hatch for a local dev server. Here it
      would let a stranger's link reach 127.0.0.1 or 169.254.169.254 from a
      session that holds credentials, so the preview pins the guard's strict mode
      whatever the setting says and whatever the user previously allowed.
    * ``trust_env`` defaults ON in httpx, which means ``HTTP(S)_PROXY`` and
      ``NO_PROXY`` decide where the request goes. Inheriting that on this path
      hands the routing of an attacker-chosen URL to whatever the surrounding
      shell exported — including a proxy named in ``NO_PROXY``, which is how a
      private target would slip past the guard's judgement entirely.

    ``transport`` is the test seam (an httpx ``MockTransport``); production leaves
    it ``None`` and httpx opens real connections.
    """
    return WebFetchService(_public_only(settings), transport=transport, trust_env=False)


def _public_only(settings: WebFetchSettings) -> WebFetchSettings:
    """``settings`` with ``allow_private`` forced off (see :func:`preview_fetcher`)."""
    if not settings.allow_private:
        return settings
    return settings.model_copy(update={"allow_private": False})


def _retained_bytes(store: PreviewStore, resolution: LinkPreviewResolution) -> dict[str, int]:
    """The bytes each artifact of ``resolution`` costs the cache while it holds it.

    Measured from the store rather than from the response: the cache's resident
    accounting must describe what is actually on disk behind the id, and an
    artifact that failed to write is then counted as the zero bytes it is.
    """
    preview = resolution.preview
    if preview is None:
        return {}
    sizes: dict[str, int] = {}
    for artifact in (preview.favicon, preview.image):
        if artifact is not None:
            sizes[artifact.id] = store.size_of(artifact.id)
    return sizes


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
    service: WebFetchService,
    url: str | None,
    *,
    deadline: float,
    store: PreviewStore,
    artifact_id: str,
) -> FaviconArtifact | None:
    if not url:
        return None
    stored = await _store_artifact(
        service,
        url,
        ceiling=FAVICON_MAX_BYTES,
        deadline=deadline,
        store=store,
        artifact_id=artifact_id,
    )
    if stored is None:
        return None
    stored_id, raster = stored
    return FaviconArtifact(id=stored_id, mime=raster.mime)


async def _image(
    service: WebFetchService,
    url: str | None,
    *,
    deadline: float,
    store: PreviewStore,
    artifact_id: str,
) -> ImageArtifact | None:
    if not url:
        return None
    stored = await _store_artifact(
        service,
        url,
        ceiling=IMAGE_MAX_BYTES,
        deadline=deadline,
        store=store,
        artifact_id=artifact_id,
    )
    if stored is None:
        return None
    stored_id, raster = stored
    return ImageArtifact(id=stored_id, mime=raster.mime, width=raster.width, height=raster.height)


async def _store_artifact(
    service: WebFetchService,
    url: str,
    *,
    ceiling: int,
    deadline: float,
    store: PreviewStore,
    artifact_id: str,
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
    raster = await asyncio.to_thread(sniff_raster, fetched.body)
    if raster is None:
        return None
    # OFF THE EVENT LOOP, deliberately. ``PreviewStore.put`` writes a temporary
    # file, ``os.replace``s it and then sweeps the directory — measured at 30-60 ms
    # per artifact — and this call sits inside the request path, where blocking the
    # loop stalls every other renderer's resolve. The store itself stays synchronous
    # and single-purpose: it is a directory of content-addressed blobs, it owns no
    # loop, and its own unit tests drive it as plain calls; the await belongs to the
    # caller that has a loop (the same split ``link_preview_artifact`` uses for
    # ``store.get``).
    stored = await asyncio.to_thread(store.put, fetched.body, artifact_id=artifact_id)
    return stored, raster


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

    ``host`` is three refusals sharing one code, and each is a dead end rather
    than a blip: no host to judge, credentials carried in the URL (never
    forwarded as an ``Authorization`` header), and a name that does not EXIST.
    All three are ``blocked-host``, which is the one code the renderer draws as
    "this link could not be used" instead of offering a retry.

    A lookup that failed rather than found nothing is deliberately NOT in that
    list: ``EAI_AGAIN`` and the other transient codes arrive as an ordinary
    :class:`~local_operator.web_fetch.service.FetchError` classified retryable
    (review R2-1), because a resolver under load is a blip and answering it with
    this code would park a live host behind the half-hour negative cache.
    """
    if reason == "scheme":
        return "unsupported-scheme"
    if reason == "private":
        return "local-or-private"
    return "blocked-host"


def _retryable(error: FetchError) -> bool:
    """Whether another try could help, from the fetcher's own classification.

    An unclassified ``FetchError`` (a redirect chain that never landed) is
    treated as retryable: the card did not resolve, and the honest answer to
    "could this work later" is that we do not know. A host that does not EXIST
    does not arrive here — the guard refuses it as ``reason="host"``, which
    becomes ``blocked-host`` and is cached as a negative answer. A host whose
    lookup merely FAILED does, and answers retryable (review R2-1): the fetcher
    classifies ``EAI_AGAIN`` and its siblings as a transport failure, so the one
    bad second is never cached as a half-hour verdict on the host.
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
