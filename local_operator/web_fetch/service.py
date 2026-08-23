"""Web-fetch configuration, SSRF policy, and the fetch→render→cache engine.

This module owns everything that touches the network and everything that could
be got dangerously wrong:

- **SSRF policy** (:func:`validate_public_url`) — a scheme allowlist plus a
  resolve-then-check on the hostname's IPs, re-run on EVERY redirect hop. A
  scheme-only gate lets a public URL 302 straight into ``169.254.169.254``; the
  manual redirect loop in :meth:`WebFetchService.fetch` is what closes that.
- **Streaming with a hard byte cap** enforced DURING the read, never a
  ``resp.text`` that buffers an unbounded body first.
- **A metadata-only URL cache** under ``config_dir()/web_fetch_cache/`` that
  points INTO the spill store and self-checks against it, so a cache hit can
  never hand back a handle the spill store has already evicted.

Content lives ONLY in the spill store (bounded, LRU). The cache index holds no
content — just a pointer, count-bounded — so this feature cannot be the reason a
disk fills (the incident that shaped :mod:`local_operator.tools.spill`).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from pydantic import ValidationError

from local_operator.config import ConfigManager
from local_operator.paths import config_dir
from local_operator.web_fetch.models import (
    DEFAULT_WEB_FETCH_CONFIG,
    FetchResult,
    WebFetchSettings,
)
from local_operator.web_fetch.render import (
    binary_notice,
    is_low_quality,
    render_html,
    render_json,
    render_text,
)

#: Sidecar directory for the URL→spill cache index. A sibling of the spill store
#: under the config dir (honouring ``LOCAL_OPERATOR_CONFIG_DIR``) so a test or
#: isolated run touches exactly one tree, the same promise spill relies on.
CACHE_DIRNAME = "web_fetch_cache"

#: Keep the newest N sidecars; prune the oldest on write. Sidecars are ~250 B of
#: metadata each, so even 500 is ~125 KB — the bound exists to stop unbounded
#: directory growth, not because the bytes matter.
CACHE_MAX_ENTRIES = 500

#: Anonymous, self-identifying agent. Deliberately carries NO ambient auth: a
#: web_fetch is not the user's logged-in session (that is what ``browser`` is
#: for), so it must not leak cookies or credentials to an agent-chosen host.
USER_AGENT = "local-operator/web_fetch (+https://github.com/damianvtran/local-operator)"

#: One read chunk. Small enough that the max_bytes cap trips within a chunk of
#: the ceiling rather than after a large over-read.
_CHUNK_BYTES = 64 * 1024

#: Bound the settings the same way search bounds its timeout: a malformed config
#: value must not let a fetch hang forever or download the world.
_TIMEOUT_MIN, _TIMEOUT_MAX = 1.0, 300.0
_MAX_BYTES_FLOOR = 1024
_MAX_REDIRECTS_CEIL = 20


class FetchError(Exception):
    """A fetch that failed for a reason the model should see verbatim.

    Raised for SSRF refusals, redirect-limit breaches, and transport errors so
    the tool layer can turn one exception into one clean error result rather
    than leaking a stack trace or an httpx internal into model context.
    """


def coerce_fetch_settings(raw: object) -> WebFetchSettings:
    """Validate loose YAML while preserving safe defaults for malformed fields.

    Mirrors ``coerce_search_settings``: a single bad field must not blow away
    the whole config. Numeric knobs are clamped to sane bounds after validation
    so a typo (``timeout_seconds: 0``, a negative ceiling) degrades to a working
    value rather than a hang or an empty download.
    """
    merged = dict(DEFAULT_WEB_FETCH_CONFIG)
    if isinstance(raw, Mapping):
        merged.update(raw)
    try:
        settings = WebFetchSettings.model_validate(merged)
    except ValidationError:
        settings = WebFetchSettings.model_validate(DEFAULT_WEB_FETCH_CONFIG)
    settings.timeout_seconds = min(max(settings.timeout_seconds, _TIMEOUT_MIN), _TIMEOUT_MAX)
    settings.max_bytes = max(settings.max_bytes, _MAX_BYTES_FLOOR)
    settings.max_redirects = min(max(settings.max_redirects, 0), _MAX_REDIRECTS_CEIL)
    settings.cache_ttl_seconds = max(settings.cache_ttl_seconds, 0)
    return settings


def load_fetch_settings(manager: ConfigManager) -> WebFetchSettings:
    """Read the current fetch mapping from a configuration manager."""
    return coerce_fetch_settings(manager.get_config_value("web_fetch", None))


def save_fetch_settings(manager: ConfigManager, settings: WebFetchSettings) -> None:
    """Persist the stable public fetch fields under ``values.web_fetch``."""
    manager.set_config_value("web_fetch", settings.model_dump(mode="json"))


def set_fetch_enabled(manager: ConfigManager, enabled: bool) -> WebFetchSettings:
    settings = load_fetch_settings(manager)
    settings.enabled = enabled
    save_fetch_settings(manager, settings)
    return settings


def set_allow_private(manager: ConfigManager, allow: bool) -> WebFetchSettings:
    settings = load_fetch_settings(manager)
    settings.allow_private = allow
    save_fetch_settings(manager, settings)
    return settings


def set_cache_ttl(manager: ConfigManager, ttl_seconds: int) -> WebFetchSettings:
    settings = load_fetch_settings(manager)
    settings.cache_ttl_seconds = max(ttl_seconds, 0)
    save_fetch_settings(manager, settings)
    return settings


def set_render_backend(manager: ConfigManager, backend: str) -> WebFetchSettings:
    settings = load_fetch_settings(manager)
    settings.render_backend = "stdlib" if backend == "stdlib" else "auto"
    save_fetch_settings(manager, settings)
    return settings


# ---------------------------------------------------------------------------
# URL normalization & SSRF policy
# ---------------------------------------------------------------------------


def normalize_url(raw: str) -> str:
    """Repair a cheap set of common URL defects, or raise :class:`FetchError`.

    Handles the three shapes a model or a ``read`` sugar realistically hands us:
    a bare host (``example.com`` → ``https://example.com``), a collapsed scheme
    (``https:/host`` → ``https://host``), and surrounding whitespace. Anything
    that still is not an ``http(s)`` absolute URL is rejected here so the SSRF
    checks below always see a well-formed target.
    """
    url = raw.strip()
    if not url:
        raise FetchError("web_fetch requires a URL")
    # A single-slash scheme (``https:/host/x``) survives urlparse as a path; fix
    # it before parsing rather than teaching every downstream step about it.
    for scheme in ("https:", "http:"):
        if url.lower().startswith(scheme) and not url.lower().startswith(scheme + "//"):
            url = scheme + "//" + url[len(scheme) :].lstrip("/")
            break
    parsed = urlparse(url)
    if not parsed.scheme:
        # A bare host defaults to https — the safe scheme, and the one a docs
        # link almost always resolves to anyway.
        parsed = urlparse("https://" + url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise FetchError(
            f"refusing {raw!r}: only http:// and https:// URLs can be fetched "
            "(use `browser` for anything a real session must open)"
        )
    if not parsed.hostname:
        raise FetchError(f"refusing {raw!r}: no host in URL")
    return urlunparse(parsed)


def _ip_is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``ip`` is a target SSRF must refuse unless ``allow_private``.

    Covers the full private/loopback/link-local/multicast/reserved surface plus
    two cases the stdlib flags do not fold together on their own: the cloud
    metadata address ``169.254.169.254`` (it IS link-local, but naming it makes
    the intent unmissable to a reviewer) and an IPv4-mapped IPv6 address, whose
    embedded v4 must be re-checked or ``::ffff:169.254.169.254`` walks through.
    """
    # An IPv4-mapped IPv6 (``::ffff:a.b.c.d``) hides a v4 target behind a v6
    # literal; unwrap and judge the real address, not the wrapper.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    # Explicit, even though it is link-local above: the cloud metadata endpoint
    # is the single most important thing this policy exists to block.
    if isinstance(ip, ipaddress.IPv4Address) and ip == ipaddress.IPv4Address("169.254.169.254"):
        return True
    return False


def _resolve_host_ips(hostname: str) -> list[str]:
    """Resolve ``hostname`` to every A/AAAA address, or raise :class:`FetchError`.

    ALL addresses are returned and checked, not just the first: a host that
    resolves to one public and one private address must be refused, because
    httpx may connect to either.
    """
    # A hostname that is already an IP literal short-circuits DNS: getaddrinfo
    # would echo it back, but calling it is pointless and a literal is exactly
    # the SSRF vector we most want to judge directly.
    try:
        ipaddress.ip_address(hostname)
        return [hostname]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve host {hostname!r}: {exc}") from exc
    # info[4] is the sockaddr; its first element is the address string for both
    # AF_INET and AF_INET6. str() guards the type checker against the union the
    # stdlib stubs give sockaddr elements.
    return list({str(info[4][0]) for info in infos})


def validate_public_url(url: str, *, allow_private: bool) -> None:
    """Raise :class:`FetchError` if ``url`` points at a non-public target.

    Called on the initial URL AND after every redirect. ``allow_private`` is the
    single, deliberate escape hatch for local dev (``http://localhost:3000``);
    default-deny is the safe posture and the switch is flipped knowingly.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise FetchError(f"refusing {url!r}: only http:// and https:// are allowed")
    hostname = parsed.hostname
    if not hostname:
        raise FetchError(f"refusing {url!r}: no host in URL")
    if allow_private:
        return
    for ip_str in _resolve_host_ips(hostname):
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _ip_is_forbidden(ip):
            raise FetchError(
                f"refusing {url!r}: host resolves to a private/loopback/reserved "
                f"address ({ip_str}). Set web_fetch.allow_private to fetch local targets."
            )


# ---------------------------------------------------------------------------
# Cache index (metadata only; content lives in the spill store)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """One URL→spill pointer. Holds NO content — just where to find it."""

    url: str
    final_url: str
    spill_handle: str
    fetched_at_ms: int
    status: int
    content_type: str
    render_method: str
    complete: bool
    low_quality: bool


def cache_dir() -> Path:
    """Directory holding the cache sidecars. Resolved per call (see spill_dir)."""
    return config_dir() / CACHE_DIRNAME


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def _cache_path(url: str) -> Path:
    return cache_dir() / f"{_url_hash(url)}.json"


def read_cache_entry(url: str) -> CacheEntry | None:
    """Load the sidecar for ``url``, or ``None`` when absent/corrupt."""
    path = _cache_path(url)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return CacheEntry(
            url=str(raw["url"]),
            final_url=str(raw.get("final_url", raw["url"])),
            spill_handle=str(raw["spill_handle"]),
            fetched_at_ms=int(raw["fetched_at_ms"]),
            status=int(raw.get("status", 0)),
            content_type=str(raw.get("content_type", "")),
            render_method=str(raw.get("render_method", "")),
            complete=bool(raw.get("complete", True)),
            low_quality=bool(raw.get("low_quality", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_cache_entry(entry: CacheEntry) -> None:
    """Persist ``entry`` and prune the oldest sidecars past the count bound.

    Best-effort: a read-only or full config dir degrades to "no caching", never
    a failed fetch — the same contract the spill store keeps for its writes.
    """
    directory = cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = _cache_path(entry.url)
        payload = {
            "url": entry.url,
            "final_url": entry.final_url,
            "spill_handle": entry.spill_handle,
            "fetched_at_ms": entry.fetched_at_ms,
            "status": entry.status,
            "content_type": entry.content_type,
            "render_method": entry.render_method,
            "complete": entry.complete,
            "low_quality": entry.low_quality,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        _prune_cache(directory)
    except OSError:
        return


def _prune_cache(directory: Path) -> None:
    """Keep the newest CACHE_MAX_ENTRIES sidecars; delete the rest by mtime."""
    try:
        sidecars = [p for p in directory.iterdir() if p.suffix == ".json"]
    except OSError:
        return
    if len(sidecars) <= CACHE_MAX_ENTRIES:
        return
    sidecars.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in sidecars[CACHE_MAX_ENTRIES:]:
        try:
            stale.unlink()
        except OSError:
            continue


# ---------------------------------------------------------------------------
# The fetch engine
# ---------------------------------------------------------------------------

# Content-Type families. Classification is by header first with a small body
# sniff as the tiebreak (never extension), matching read's content ethos.
_HTML_TYPES = ("text/html", "application/xhtml")
_JSON_TYPES = ("application/json", "text/json", "+json")
_TEXT_TYPES = ("text/", "application/xml", "text/markdown")


class WebFetchService:
    """Resolve settings and run one SSRF-guarded, bounded, cached fetch.

    ``transport`` is injectable purely for tests (httpx ``MockTransport``); in
    production it is ``None`` and httpx opens real connections. The service is
    cheap to construct per call — it holds no connection state — so the tool
    builds a fresh one each invocation, mirroring ``WebSearchService``.
    """

    def __init__(
        self,
        settings: WebFetchSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def fetch(
        self,
        url: str,
        *,
        raw: bool = False,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
    ) -> FetchResult:
        """Fetch one URL, with optional enrichment, through the SSRF-guarded loop.

        Returns a :class:`FetchResult` with the FULL rendered content; spilling
        and preview shaping happen in the tool layer so the sugar and the tool
        share one output shape. Raises :class:`FetchError` for refusals,
        redirect-limit breaches, and transport failures.
        """
        normalized = normalize_url(url)
        ceiling = max(max_bytes if max_bytes is not None else self.settings.max_bytes, 1)
        timeout = timeout_seconds if timeout_seconds is not None else self.settings.timeout_seconds

        async with httpx.AsyncClient(
            transport=self.transport,
            follow_redirects=False,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            # Enrichment (design §6 step 3): before scraping HTML, try the cheap,
            # high-value candidates that many docs sites expose — a ``.md`` twin
            # and content negotiation. If one yields substantial non-HTML text we
            # use it; otherwise fall through to the plain page. Gated by the
            # ``enrich`` switch and skipped for ``raw`` (the caller asked for the
            # source verbatim, not a cleaner rendition).
            if self.settings.enrich and not raw:
                enriched = await self._try_enrichment(client, normalized, ceiling)
                if enriched is not None:
                    return enriched
            final_url, status, headers, body, complete = await self._follow(
                client, normalized, ceiling
            )
            return self._render(normalized, final_url, status, headers, body, complete, raw)

    async def _follow(
        self, client: httpx.AsyncClient, start_url: str, ceiling: int
    ) -> tuple[str, int, dict[str, str], bytes, bool]:
        """Drive the manual, re-validating redirect loop and return the final hop.

        httpx auto-redirect is DISABLED so each hop's destination is re-validated
        BEFORE we connect to it. This is the SSRF-via-redirect defence — a public
        URL that 302s to the metadata endpoint is refused at the hop, never
        followed. Shared by the primary fetch and every enrichment attempt so the
        policy cannot be bypassed through a side door.
        """
        current = start_url
        for _hop in range(self.settings.max_redirects + 1):
            validate_public_url(current, allow_private=self.settings.allow_private)
            status, headers, body, complete = await self._stream_once(client, current, ceiling)
            location = headers.get("location")
            if status in (301, 302, 303, 307, 308) and location:
                current = urljoin(current, location)
                continue
            return current, status, headers, body, complete
        raise FetchError(
            f"too many redirects (> {self.settings.max_redirects}) starting from {start_url!r}"
        )

    async def _try_enrichment(
        self, client: httpx.AsyncClient, url: str, ceiling: int
    ) -> FetchResult | None:
        """Attempt the cheap enrichment candidates; return a result or ``None``.

        Each candidate rides the same SSRF-guarded :meth:`_follow`, so an
        enrichment fetch is exactly as safe as the primary one. A candidate wins
        only when it returns 2xx with substantial, non-HTML text — otherwise fall
        through so a site without these affordances pays nothing but the bounded,
        best-effort probe. Network errors on a probe are swallowed: enrichment is
        an optimization, never a reason to fail the real fetch.
        """
        for candidate in _enrichment_candidates(url):
            try:
                final_url, status, headers, body, complete = await self._follow(
                    client, candidate, ceiling
                )
            except FetchError:
                # A blocked/failed probe must not sink the primary fetch; a
                # candidate that resolves to a private address is simply skipped.
                continue
            if status < 200 or status >= 300 or not body:
                continue
            content_type = headers.get("content-type", "").lower()
            decoded = body.decode("utf-8", errors="replace")
            # Only accept a candidate that is genuinely non-HTML text of
            # substance; an HTML 200 (a soft-404 shell) is no improvement over
            # rendering the real page.
            if _looks_html(decoded, content_type) or _is_binary(content_type, body):
                continue
            if len(decoded.strip()) <= 100:
                continue
            return self._render(url, final_url, status, headers, body, complete, raw=False)
        return None

    async def _stream_once(
        self, client: httpx.AsyncClient, url: str, ceiling: int
    ) -> tuple[int, dict[str, str], bytes, bool]:
        """One request, reading at most ``ceiling`` bytes off the wire.

        The cap is enforced DURING streaming: we stop pulling chunks once the cap
        is reached and flag the body truncated, so a hostile or huge endpoint can
        never balloon memory (the ``resp.text`` trap the design calls out).
        """
        buffer = bytearray()
        complete = True
        try:
            async with client.stream("GET", url) as response:
                headers = {k.lower(): v for k, v in response.headers.items()}
                status = response.status_code
                # A redirect body is irrelevant — we only need the Location — so
                # do not spend the byte budget draining it.
                if status in (301, 302, 303, 307, 308) and "location" in headers:
                    return status, headers, b"", True
                async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                    buffer.extend(chunk)
                    if len(buffer) >= ceiling:
                        del buffer[ceiling:]
                        complete = False
                        break
        except httpx.TimeoutException as exc:
            raise FetchError(f"timed out fetching {url!r}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"could not fetch {url!r}: {exc}") from exc
        return status, headers, bytes(buffer), complete

    def _render(
        self,
        request_url: str,
        final_url: str,
        status: int,
        headers: dict[str, str],
        body: bytes,
        complete: bool,
        raw: bool,
    ) -> FetchResult:
        """Classify the body and render it, producing the full FetchResult."""
        content_type = headers.get("content-type", "").lower()
        byte_count = len(body)

        if _is_binary(content_type, body):
            text, method = binary_notice(content_type, byte_count, final_url)
            return FetchResult(
                url=request_url,
                final_url=final_url,
                status=status,
                content_type=content_type or "application/octet-stream",
                render_method=method,
                content=text,
                bytes=byte_count,
                complete=complete,
            )

        decoded = body.decode("utf-8", errors="replace")
        if raw:
            # Verbatim source: no HTML→markdown, no JSON reflow. Still classified
            # for the card's method field, but the bytes pass through untouched.
            method = "text"
            content = decoded
        elif _matches(content_type, _JSON_TYPES) or _looks_json(decoded, content_type):
            content, method = render_json(decoded)
        elif _matches(content_type, _HTML_TYPES) or _looks_html(decoded, content_type):
            force_stdlib = self.settings.render_backend == "stdlib"
            content, method = render_html(decoded, force_stdlib=force_stdlib)
        elif _matches(content_type, _TEXT_TYPES) or not content_type:
            content, method = render_text(decoded)
        else:
            content, method = render_text(decoded)

        low_quality = method in ("markdownify", "stdlib") and is_low_quality(content)
        return FetchResult(
            url=request_url,
            final_url=final_url,
            status=status,
            content_type=content_type or "text/plain",
            render_method=method,
            content=content,
            bytes=byte_count,
            complete=complete,
            low_quality=low_quality,
        )


def _enrichment_candidates(url: str) -> list[str]:
    """Cheap, high-value alternate URLs to try before scraping the HTML page.

    A proportionate subset of omp's enrichment chain (design §6 step 3): the
    ``.md`` suffix trick that GitHub and many docs generators honour, plus a
    ``/llms.txt`` probe at the site root. Deliberately small — this is meant to
    catch the common docs-site wins, not to become a scraper zoo. Query strings
    and fragments are dropped from the candidate since a ``.md`` twin is a path
    concept. Only http(s) with a path get a ``.md`` candidate.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        return []
    candidates: list[str] = []
    path = parsed.path
    # ``page`` → ``page.md``; a directory path (``/docs/``) has no obvious twin,
    # so only a concrete file-ish path (has a last segment, no suffix) qualifies.
    if path and not path.endswith("/"):
        last = path.rsplit("/", 1)[-1]
        if "." not in last:
            candidates.append(urlunparse(parsed._replace(path=path + ".md", query="", fragment="")))
    # Site-root llms.txt — a growing convention for LLM-friendly docs indexes.
    root = f"{parsed.scheme}://{parsed.netloc}"
    candidates.append(urljoin(root, "/llms.txt"))
    return candidates


def _matches(content_type: str, families: tuple[str, ...]) -> bool:
    return any(family in content_type for family in families)


def _looks_json(body: str, content_type: str) -> bool:
    """A JSON body served with a wrong/absent Content-Type still pretty-prints."""
    if content_type and not _matches(content_type, _TEXT_TYPES + ("application/",)):
        return False
    stripped = body.lstrip()
    return stripped[:1] in ("{", "[")


def _looks_html(body: str, content_type: str) -> bool:
    head = body[:512].lstrip().lower()
    return head.startswith(("<!doctype html", "<html")) or "<body" in head


def _is_binary(content_type: str, body: bytes) -> bool:
    """Whether the body is binary (PDF/image/other) and must not be inlined.

    Header first, then a NUL-byte sniff of the head: a mislabeled octet-stream
    that is really text still renders, and a text/* header that is really binary
    (rare, but possible) is caught by the sniff.
    """
    primary = content_type.split(";", 1)[0].strip()
    if primary.startswith(("image/", "audio/", "video/", "font/")):
        return True
    if primary in ("application/pdf", "application/octet-stream", "application/zip"):
        return True
    if primary.startswith("text/") or _matches(content_type, _JSON_TYPES):
        return False
    # No decisive header: a NUL byte in the head is the classic binary tell.
    return b"\x00" in body[:1024]
