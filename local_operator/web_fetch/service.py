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

import asyncio
import hashlib
import ipaddress
import json
import random
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from pydantic import ValidationError

from local_operator.config import ConfigManager
from local_operator.paths import config_dir
from local_operator.web_fetch.failure import (
    MARKER_SCAN_BYTES,
    FetchFailure,
    classify_exception,
    classify_response,
    describe,
)
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
from local_operator.web_search.io import WebReadIO

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

#: A COMPATIBILITY PROFILE, sent on exactly one retry after the origin has
#: already refused the honest request above (design §4). It is NOT the default
#: posture and must not be used anywhere else: the honest UA is truthful, seven
#: major sites were measured to serve it fine, and a site owner who wants to
#: allow or deny us should be able to identify us.
#:
#: This exact set was measured to flip ``https://medium.com/`` from 403 (5,507
#: bytes, ``cf-mitigated: challenge``) to 200 (~53 KB of real page), 3/3
#: reproductions, INCLUDING through the SSRF-pinned path with ``Host``/SNI
#: preserved — so the escalation rides the existing guarded request and needs no
#: second client.
#:
#: Deliberately absent: ``sec-ch-ua*`` client hints (Chrome-version-coupled and
#: measured not to matter), cookies (``web_search/io.py`` refuses them on
#: purpose — a fetch is not the user's session), and any ``Referer``, since
#: inventing a referrer is a fabrication rather than a profile. It sends no
#: credentials, so it impersonates a generic browser, never a user.
BROWSER_PROFILE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif," "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Upgrade-Insecure-Requests": "1",
}

#: Backoff shape: ``min(BASE * 2**(n-1), CAP)`` with ±JITTER. So retry 1 waits
#: ~0.5 s and retry 2 ~1.0 s — ~1.9 s of added sleep in the worst
#: non-rate-limited case, all of it inside the call's own deadline.
#:
#: The jitter is not decoration: a turn can issue several ``web_fetch`` calls in
#: parallel (the tool description encourages it) and they frequently hit the same
#: origin. Un-jittered backoff synchronises those retries into a burst, which is
#: exactly what a struggling origin does not need. Same family as the TUI
#: sidebar's backoff.
_RETRY_BASE_S = 0.5
_RETRY_CAP_S = 4.0
_RETRY_JITTER = 0.25

#: A backoff is only worth sleeping if the attempt after it can still run. This
#: is the margin left for that attempt; below it we return the failure now
#: rather than sleeping into the deadline and reporting a timeout we caused.
_RETRY_MIN_MARGIN_S = 0.25

#: ``Retry-After`` is honoured only when it fits inside the turn. Beyond this we
#: report the number instead of sleeping on it: blocking a live turn for two
#: minutes to obey a header is worse for the user than telling the agent to come
#: back later, and the agent can act on the number.
_RETRY_AFTER_MAX_SLEEP_S = 10.0

#: NOTE on attempt budgeting — deliberately absent, and the reasoning is a
#: decision recorded in ``docs/design/web_fetch_robustness.md`` §3.3.
#:
#: An earlier revision of this change capped the first attempt of a hop at
#: ``0.6 * timeout`` while a fallback remained, to leave the browser-shaped
#: probe something to spend on a black-holing origin. That cap silently turned a
#: slow-but-working origin into a failure: ``httpx``'s read timeout bounds the
#: wait for response HEADERS too, so an origin whose time-to-first-byte exceeds
#: 0.6·T (12 s of the default 20 s) was cut off on its first, honest attempt and
#: then handed a browser-profile retry that had 0.4·T left — reported to the
#: model as a "silent block" naming a vendor we never saw. Re-measured against
#: the same fake-clock harness that found it: a 13 s-first-byte origin succeeds
#: on the pre-change engine and failed as ``stall`` under the cap.
#:
#: So there is no first-attempt cap: **every attempt gets the full remaining
#: budget**, and the escalation is opportunistic — funded only by whatever the
#: hop's own outcome left behind. A fast refusal (403) still leaves ~all of T for
#: the probe, which is where the measured win lives; a stall that consumes the
#: budget simply reports a classified, readable terminal message instead of
#: pretending to have diagnosed a vendor it never reached.

#: Connect is capped independently of read: a handshake that has not completed
#: in 5 s is not going to, and spending the whole budget on it would leave
#: nothing for the body.
_CONNECT_CAP_S = 5.0

#: Fraction of the whole-call budget that ALL enrichment probes together may
#: spend. Necessary once probes share the call's deadline rather than each
#: getting a fresh copy of the timeout: on a black-holing origin a stalling
#: ``.md`` probe would otherwise consume the budget the real fetch — and its
#: escalation — needs, so a speculative optimisation would decide the outcome of
#: the request the user actually made. Measured: without this cap a 3 s fetch of
#: a silent origin spent 1.8 s on the probe and had nothing left to escalate
#: with.
_ENRICHMENT_BUDGET_FRACTION = 0.25

#: Attempts are bounded on both sides. 1 is "no retries" (the pre-change
#: behaviour, and a legitimate choice); 5 is where a struggling origin is being
#: hammered rather than helped.
_MAX_ATTEMPTS_FLOOR, _MAX_ATTEMPTS_CEIL = 1, 5

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

    ``failure`` carries the classification when there is one (a transport-level
    throw), and is ``None`` for the policy refusals that never reach the network
    — an SSRF refusal is not a "failure class", it is the guard working. It is
    ADDITIVE: every existing ``str(error)`` call site keeps working unchanged.

    ``attempts``/``profiles`` carry the same retry facts the message already
    states in prose, so a caller that builds ``details`` from the exception does
    not have to parse the sentence to find out how many requests were spent. The
    pair is why the terminal path's ``details`` can finally agree with its own
    preview text (design §5.3: ``attempts`` is a key, excluded only for a cache
    hit) — before this the model read "2 attempts" while the structured payload
    said nothing.
    """

    def __init__(
        self,
        message: str,
        *,
        failure: FetchFailure | None = None,
        attempts: int = 0,
        profiles: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.attempts = attempts
        self.profiles = list(profiles)


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
    settings.max_attempts = min(max(settings.max_attempts, _MAX_ATTEMPTS_FLOOR), _MAX_ATTEMPTS_CEIL)
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


def set_max_attempts(manager: ConfigManager, attempts: int) -> WebFetchSettings:
    """Set attempts per hop, clamped. 1 disables retries entirely."""
    settings = load_fetch_settings(manager)
    settings.max_attempts = min(max(attempts, _MAX_ATTEMPTS_FLOOR), _MAX_ATTEMPTS_CEIL)
    save_fetch_settings(manager, settings)
    return settings


def set_blocked_retry(manager: ConfigManager, enabled: bool) -> WebFetchSettings:
    """Toggle the one browser-shaped retry paid after a refusal (§4)."""
    settings = load_fetch_settings(manager)
    settings.blocked_retry = enabled
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


def _pin_request(url: str, pinned_ip: str | None) -> tuple[str, dict[str, str], dict[str, object]]:
    """Rewrite ``url`` to connect to ``pinned_ip`` while keeping the hostname.

    Returns ``(request_url, headers, extensions)``. When ``pinned_ip`` is set the
    URL host is swapped to the vetted IP literal so httpx opens the socket to that
    exact address (M1 anti-rebinding), while:

    - the ``Host`` header keeps the original hostname, so name-based virtual hosts
      and the server's routing still see the site they expect; and
    - the ``sni_hostname`` request extension keeps the original hostname, so the
      TLS handshake sends the right SNI and the certificate is verified against
      the hostname — verification is never turned off, we only redirect which IP
      the bytes go to.

    ``None`` (bypass or literal host) returns the URL untouched with empty
    overrides, so the normal httpx resolution path is used.
    """
    if not pinned_ip:
        return url, {}, {}
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    # httpx.URL.copy_with(host=...) brackets IPv6 literals automatically and
    # preserves scheme/port/path/query, so the request targets the IP verbatim.
    pinned_url = str(httpx.URL(url).copy_with(host=pinned_ip))
    # Host carries the port when the original URL did, matching what the server
    # would have seen without pinning.
    host_header = hostname
    if parsed.port is not None:
        host_header = f"{hostname}:{parsed.port}"
    return pinned_url, {"Host": host_header}, {"sni_hostname": hostname}


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


def validate_public_url(url: str, *, allow_private: bool) -> str | None:
    """Validate ``url``'s target and return the vetted IP to PIN the connection to.

    Called on the initial URL AND after every redirect. ``allow_private`` is the
    single, deliberate escape hatch for local dev (``http://localhost:3000``);
    default-deny is the safe posture and the switch is flipped knowingly.

    Returns the one vetted IP literal the caller MUST connect to (or ``None`` when
    ``allow_private`` bypasses the check, or when the host is already an IP
    literal that needs no pinning beyond itself). Returning the address — rather
    than only raising — is what closes the DNS-rebinding / TOCTOU window (M1): the
    validator resolves the hostname ONCE, judges EVERY returned address, and hands
    back the specific vetted IP so the connection is pinned to it instead of
    letting httpx re-resolve independently at connect time (where a short-TTL or
    round-robin record could swap in an internal address that never passed this
    gate). ALL resolved addresses are checked, so a mixed public/private record is
    refused outright; the first address is the one returned to pin to, and since
    every address passed, any of them is safe.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise FetchError(f"refusing {url!r}: only http:// and https:// are allowed")
    hostname = parsed.hostname
    if not hostname:
        raise FetchError(f"refusing {url!r}: no host in URL")
    if allow_private:
        # The escape hatch turns off IP judgement entirely, so there is nothing
        # to pin to and no rebinding threat model to defend (the operator opted
        # into local targets knowingly). Let httpx resolve as usual.
        return None
    resolved = _resolve_host_ips(hostname)
    vetted: str | None = None
    for ip_str in resolved:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _ip_is_forbidden(ip):
            raise FetchError(
                f"refusing {url!r}: host resolves to a private/loopback/reserved "
                f"address ({ip_str}). Set web_fetch.allow_private to fetch local targets."
            )
        if vetted is None:
            vetted = ip_str
    return vetted


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


def cache_variant(*, raw: bool, max_bytes: int | None) -> str:
    """The render-shape discriminator folded into the cache key (M3).

    Two requests for the same URL that would render DIFFERENTLY must not share a
    cache entry: a ``raw=True`` fetch stores verbatim source, and a later default
    (rendered) fetch of the same URL would otherwise get a hit and receive raw
    HTML. ``max_bytes`` changes where the body is truncated, so a small-capped
    entry must not satisfy a request for the full ceiling. Encoding both here
    keeps the discriminator in one place both the read and the write agree on.
    """
    return f"raw={int(raw)}|max_bytes={max_bytes if max_bytes is not None else 'default'}"


def _url_hash(url: str, variant: str = "") -> str:
    return hashlib.sha256(f"{url}\x00{variant}".encode("utf-8")).hexdigest()[:32]


def _cache_path(url: str, variant: str = "") -> Path:
    return cache_dir() / f"{_url_hash(url, variant)}.json"


def read_cache_entry(url: str, variant: str = "") -> CacheEntry | None:
    """Load the sidecar for ``url`` + render ``variant``, or ``None`` if absent."""
    path = _cache_path(url, variant)
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


def write_cache_entry(entry: CacheEntry, variant: str = "") -> None:
    """Persist ``entry`` under ``url`` + render ``variant`` and prune the excess.

    Best-effort: a read-only or full config dir degrades to "no caching", never
    a failed fetch — the same contract the spill store keeps for its writes. The
    ``variant`` must match the one :func:`read_cache_entry` will look up, or the
    write lands under a key no read reaches (M3).
    """
    directory = cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = _cache_path(entry.url, variant)
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


def _now() -> float:
    """Monotonic clock for the deadline. Monotonic specifically: a wall-clock
    jump (NTP step, laptop wake) must not extend or collapse a fetch budget."""
    return time.monotonic()


@dataclass
class _Budget:
    """The whole call's time budget, shared by every attempt.

    Not a per-request timeout: the point of the deadline is that retries divide
    one budget rather than each getting a fresh copy of it, so a
    ``timeout_seconds=20`` fetch takes ~20 s whether it makes one attempt or
    four.

    ``escalation_available`` lives here rather than on the hop because the
    escalation is one-per-FETCH: a redirect chain that fails at three hops must
    spend one browser-shaped attempt in total, not three.
    """

    deadline: float
    total: float
    max_attempts: int
    escalation_available: bool

    def remaining(self) -> float:
        return self.deadline - _now()


@dataclass
class _Telemetry:
    """What the attempts actually did, carried out to the result.

    Mutable and threaded through the hop loop because the reported outcome has
    to name the attempt count and the identities used even when the final
    response came from the escalated attempt — the reader needs to tell "asked
    once" from "asked twice, the second time wearing a browser's headers".
    """

    attempts: int = 0
    profiles: list[str] = field(default_factory=list)
    failure: FetchFailure | None = None
    retry_after_s: float | None = None

    @property
    def profile(self) -> str:
        """The identity that produced the reported outcome."""
        return self.profiles[-1] if self.profiles else "default"


def _backoff_delay(attempt: int, retry_after_s: float | None) -> float | None:
    """Seconds to wait before the next attempt, or ``None`` to stop retrying.

    When the origin named its own interval (``Retry-After`` on a 429 or a 503),
    honouring it is both the polite and the effective behaviour — but only while
    it fits inside a live turn. A two-minute wait returns ``None`` so the caller
    reports the number instead of blocking the user on it.

    The honoured interval carries the SAME ±25 % jitter as our own backoff curve
    (design §3.4), and it matters most here: the tool description invites parallel
    ``web_fetch`` calls, so several calls refused by one rate-limited origin would
    otherwise sleep the identical interval and retry in lockstep — the exact burst
    the jitter exists to spread out. The jitter is applied BEFORE the cap is
    consulted so a value nudged over ``_RETRY_AFTER_MAX_SLEEP_S`` is reported
    rather than slept.
    """
    if retry_after_s is not None:
        delay = retry_after_s * (1 + random.uniform(-_RETRY_JITTER, _RETRY_JITTER))
        if retry_after_s > _RETRY_AFTER_MAX_SLEEP_S or delay > _RETRY_AFTER_MAX_SLEEP_S:
            return None
        return delay
    delay = min(_RETRY_BASE_S * (2 ** (attempt - 1)), _RETRY_CAP_S)
    return delay * (1 + random.uniform(-_RETRY_JITTER, _RETRY_JITTER))


async def _backoff_sleep(delay: float) -> None:
    """A plain ``asyncio.sleep`` — deliberately, and it is load-bearing.

    ``tool.py::_fetch_or_abort`` races the whole fetch coroutine against the
    abort signal, so a cancellation lands on this sleep immediately and the task
    is reaped. Anything that blocked the loop here (``time.sleep``, a shielded
    wait) would make an abort during a backoff wait out the full delay.
    """
    await asyncio.sleep(delay)


class WebFetchService:
    """Resolve settings and run one SSRF-guarded, bounded, cached fetch.

    ``transport`` is injectable purely for tests (httpx ``MockTransport``); in
    production it is ``None`` and httpx opens real connections. Services remain
    cheap per-call views of settings; the session-owned ``io`` retains bounded
    per-origin pools. Embedders without an owner use a temporary pool that is
    closed before returning, so no event-loop or connection lifetime is leaked.
    """

    def __init__(
        self,
        settings: WebFetchSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        io: WebReadIO | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.io = io

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

        owner = self.io or WebReadIO()
        try:
            return await self._fetch_owned(owner, normalized, ceiling, timeout, raw)
        finally:
            if self.io is None:
                await owner.aclose()

    async def _fetch_owned(
        self, owner: WebReadIO, normalized: str, ceiling: int, timeout: float, raw: bool
    ) -> FetchResult:
        # The whole call gets ONE deadline, computed here and never extended.
        # This is the hard invariant retries rest on: ``timeout`` used to be the
        # per-REQUEST timeout, so N attempts could have taken N×T. It is now a
        # budget the attempts share, which is why no configuration of retries
        # can make a fetch exceed its stated timeout.
        budget = _Budget(
            deadline=_now() + timeout,
            total=timeout,
            max_attempts=self.settings.max_attempts,
            # One escalation per FETCH, not per hop — a redirect chain must not
            # be able to multiply it.
            escalation_available=self.settings.blocked_retry,
        )

        # Enrichment (design §6 step 3): before scraping HTML, try the cheap,
        # high-value candidates that many docs sites expose — a ``.md`` twin
        # and content negotiation. If one yields substantial non-HTML text we
        # use it; otherwise fall through to the plain page. Gated by the
        # ``enrich`` switch and skipped for ``raw`` (the caller asked for the
        # source verbatim, not a cleaner rendition).
        if self.settings.enrich and not raw:
            enriched = await self._try_enrichment(owner, normalized, ceiling, budget)
            if enriched is not None:
                return enriched
            # Any budget the probes did not spend stays with the real fetch;
            # what they DID spend is already gone from the shared deadline.

        telemetry = _Telemetry()
        final_url, status, headers, body, complete = await self._follow(
            owner,
            normalized,
            ceiling,
            budget,
            telemetry,
            max_attempts=budget.max_attempts,
            allow_escalation=True,
        )
        return self._render(
            normalized, final_url, status, headers, body, complete, raw, telemetry=telemetry
        )

    async def _follow(
        self,
        owner: WebReadIO,
        start_url: str,
        ceiling: int,
        budget: _Budget,
        telemetry: _Telemetry,
        *,
        max_attempts: int,
        allow_escalation: bool,
    ) -> tuple[str, int, dict[str, str], bytes, bool]:
        """Drive the manual, re-validating redirect loop and return the final hop.

        httpx auto-redirect is DISABLED so each hop's destination is re-validated
        BEFORE we connect to it. This is the SSRF-via-redirect defence — a public
        URL that 302s to the metadata endpoint is refused at the hop, never
        followed. Shared by the primary fetch and every enrichment attempt so the
        policy cannot be bypassed through a side door.

        Retries live INSIDE the hop (:meth:`_attempt_with_retries`), never around
        this loop: a retry wrapped around the whole chain would re-issue every
        earlier hop and — the part that matters — could not re-validate the hop
        it is actually retrying.
        """
        current = start_url
        for _hop in range(self.settings.max_redirects + 1):
            status, headers, body, complete = await self._attempt_with_retries(
                owner,
                current,
                ceiling,
                budget,
                telemetry,
                max_attempts=max_attempts,
                allow_escalation=allow_escalation,
            )
            location = headers.get("location")
            if status in (301, 302, 303, 307, 308) and location:
                current = urljoin(current, location)
                continue
            return current, status, headers, body, complete
        raise FetchError(
            f"too many redirects (> {self.settings.max_redirects}) starting from {start_url!r}"
        )

    async def _attempt_with_retries(
        self,
        owner: WebReadIO,
        url: str,
        ceiling: int,
        budget: _Budget,
        telemetry: _Telemetry,
        *,
        max_attempts: int,
        allow_escalation: bool,
    ) -> tuple[int, dict[str, str], bytes, bool]:
        """One hop, with retries and at most one browser-profile escalation.

        Two orthogonal mechanisms, and keeping them distinct is what bounds the
        request count at 4 per hop:

        - **Retries** repeat the SAME request identity, for classes where the
          origin's answer says another try could work (a reset, a 502 from a
          rolling deploy, a 429 that named its own interval).
        - **The escalation** changes the request identity ONCE, and only after a
          refusal or a stall — the two classes a repeat cannot help, because the
          origin has already decided about this client.

        Only GET is ever issued here (``client.stream("GET", …)`` is the single
        request site in this package), so retrying is safe by HTTP semantics.
        **If a non-idempotent method is ever added, this predicate must gate on
        the method** — an automatic retry of a POST is a duplicate side effect.
        """
        response, failure = await self._attempt_series(
            owner, url, ceiling, budget, telemetry, profile="default", max_attempts=max_attempts
        )
        if failure is None:
            assert response is not None  # a None failure means a response arrived
            return response

        escalate = (
            allow_escalation
            and failure.escalatable
            and budget.escalation_available
            and budget.remaining() > 0
        )
        if escalate:
            # Spend the one-per-fetch escalation. Marked consumed BEFORE the
            # attempt so a redirect chain or a second failing hop cannot spend
            # it again even if this attempt itself fails.
            budget.escalation_available = False
            esc_response, esc_failure = await self._attempt_series(
                owner, url, ceiling, budget, telemetry, profile="browser", max_attempts=1
            )
            if esc_failure is None:
                assert esc_response is not None
                return esc_response
            # The escalated outcome is the more informative one and is what gets
            # reported: canadiantire goes from "timed out with no message" to
            # "403 from Akamai with a reference id". Fall back to the original
            # response only when the escalation produced none at all (it threw),
            # since a real 403 beats an exception either way.
            failure = esc_failure
            if esc_response is not None:
                response = esc_response

        telemetry.failure = failure
        if response is not None:
            # A classified NON-2xx response still flows to the render step: the
            # status, the headers and (for every class but ``blocked``) the body
            # are what the caller sees. Only a failure with no response at all
            # is an exception.
            return response
        raise FetchError(
            describe(
                failure,
                attempts=telemetry.attempts,
                profiles=telemetry.profiles,
                url=url,
            ),
            failure=failure,
            attempts=telemetry.attempts,
            profiles=telemetry.profiles,
        )

    async def _attempt_series(
        self,
        owner: WebReadIO,
        url: str,
        ceiling: int,
        budget: _Budget,
        telemetry: _Telemetry,
        *,
        profile: str,
        max_attempts: int,
    ) -> tuple[tuple[int, dict[str, str], bytes, bool] | None, FetchFailure | None]:
        """Up to ``max_attempts`` requests with ONE identity, with backoff.

        Returns ``(response, failure)``: a ``None`` failure means the response is
        a success (2xx/3xx); a ``None`` response means every attempt threw before
        a status arrived.
        """
        response: tuple[int, dict[str, str], bytes, bool] | None = None
        failure: FetchFailure | None = None

        for attempt in range(1, max_attempts + 1):
            remaining = budget.remaining()
            if remaining <= 0:
                # The deadline is spent. Report what we already know rather than
                # opening a connection that cannot finish.
                if failure is None:
                    failure = FetchFailure(
                        kind="stall",
                        retryable=False,
                        detail=f"ran out of time after {budget.total:.1f}s",
                    )
                break

            # EVERY attempt gets the full remaining budget — including the first
            # one. There used to be a `min(remaining, 0.6 * total)` cap here for
            # the first attempt of a hop; it turned a slow-but-working origin into
            # a failure, and the reasoning for its removal is at the top of this
            # module. The escalation is opportunistic: it is funded by whatever
            # the hop's own outcome left behind, so a refusal still leaves
            # ~all of T for the browser-shaped probe while a stall that spends the
            # budget simply reports the stall. The attempt's timeout is derived
            # from the budget INSIDE ``_request_once``, after the resolver, which
            # is what keeps the number honest (see that method).
            telemetry.attempts += 1
            telemetry.profiles.append(profile)

            try:
                status, headers, body, complete = await self._request_once(
                    owner, url, ceiling, budget, profile=profile
                )
            except FetchError as error:
                if error.failure is None:
                    # A POLICY refusal — SSRF, an unresolvable host, a scheme we
                    # do not speak. Never retried: it is the guard working, and
                    # asking again cannot change the answer.
                    raise
                failure = error.failure
                response = None
            else:
                # Markers are scanned over the first 8 KB only (see
                # ``failure.MARKER_SCAN_BYTES``): a challenge page is always
                # small, and a marker 3 MB into a real article is a false
                # positive we do not want.
                head = body[:MARKER_SCAN_BYTES].decode("utf-8", errors="replace")
                failure = classify_response(status, headers, head)
                response = (status, headers, body, complete)
                if failure is None:
                    # A retried SUCCESS is a success: clear the earlier attempt's
                    # classification so a 500-then-200 does not report itself as
                    # a server failure that happens to have content. The attempt
                    # COUNT is kept, because that is the honest explanation for
                    # the longer duration the user saw.
                    telemetry.failure = None
                    telemetry.retry_after_s = None
                    return response, None

            telemetry.failure = failure
            # Set UNCONDITIONALLY, including to ``None``: a 429's Retry-After
            # must not ride into the details of a later, different failure. A
            # caller reading ``retry_after_s`` is being told what the REPORTED
            # outcome asked for, and a 503 that named no interval has nothing to
            # say about the 2 s an earlier attempt was told to wait.
            telemetry.retry_after_s = failure.retry_after_s
            if not failure.retryable or attempt >= max_attempts:
                break

            delay = _backoff_delay(attempt, failure.retry_after_s)
            if delay is None or delay > budget.remaining() - _RETRY_MIN_MARGIN_S:
                # Either the origin asked for longer than this turn can honour,
                # or the backoff would eat the budget the next attempt needs.
                # Returning now with the number in the text lets the agent come
                # back later; sleeping would just manufacture a timeout.
                break
            await _backoff_sleep(delay)

        return response, failure

    async def _request_once(
        self,
        owner: WebReadIO,
        url: str,
        ceiling: int,
        budget: _Budget,
        *,
        profile: str,
    ) -> tuple[int, dict[str, str], bytes, bool]:
        """Validate, pin, and issue ONE request.

        **Validation and pinning happen here, inside the attempt** — never
        hoisted into the caller's loop. Hoisting them would be the one way to get
        this badly wrong: a host that resolves public on attempt 1 and private on
        attempt 2 must be refused on attempt 2, and a retry that reused attempt
        1's vetted address without re-checking would reopen exactly the rebinding
        window the pin exists to close.

        The deadline is the ceiling for the VALIDATION too, not just the request:
        ``getaddrinfo`` has no timeout of its own in the stdlib, so an unbounded
        lookup would let a hung resolver push the call past its stated timeout and
        make the per-attempt timeout computed before it stale. So the lookup runs
        under ``asyncio.wait_for(remaining)`` and the request's own timeout is
        recomputed from what is left AFTER it (see ``_attempt_series``).
        """
        # Validate returns the vetted IP to PIN this request's connection to, so
        # httpx cannot re-resolve to a different (internal) address between this
        # check and the socket open (M1). Re-run on EVERY request — every hop,
        # every retry, every enrichment probe, and the escalated attempt.
        # DNS can block even before HTTP starts, which is why it runs in a thread.
        #
        # The abandoned-thread caveat: a timed-out ``to_thread`` cannot be
        # cancelled, so the worker keeps running until getaddrinfo returns on its
        # own. That thread cannot extend THIS call (the wait_for is what the call
        # awaits) — it can only linger in the executor, where the resolver's own
        # timeout ends it. The guarantee this buys is therefore about the fetch,
        # not about the thread: no configuration of retries can make the CALL
        # exceed its stated timeout.
        resolver_budget = budget.remaining()
        if resolver_budget <= 0:
            raise FetchError(
                f"ran out of time after {budget.total:.1f}s before resolving {url}",
                failure=FetchFailure(
                    kind="stall",
                    retryable=False,
                    detail=f"ran out of time after {budget.total:.1f}s",
                ),
            )
        try:
            pinned_ip = await asyncio.wait_for(
                asyncio.to_thread(
                    validate_public_url, url, allow_private=self.settings.allow_private
                ),
                timeout=resolver_budget,
            )
        except asyncio.TimeoutError:
            # A policy refusal (private host, bad scheme) raises out of the
            # thread as its own FetchError; only a lookup that never returns lands
            # here. Classified as ``transport`` so the caller's retry decision is
            # unchanged, and read as a stall by the model — which is what it is.
            raise FetchError(
                f"resolving {url} took longer than the {resolver_budget:.1f}s left "
                "of this fetch's time budget",
                failure=FetchFailure(
                    kind="transport",
                    retryable=True,
                    detail=(
                        "the host name could not be resolved in time — the lookup "
                        f"did not finish within the {resolver_budget:.1f}s left of "
                        "the fetch budget"
                    ),
                ),
            ) from None
        # Recompute against the TRUE remaining time: the lookup above just spent
        # an unbounded amount of it, and the timeout the caller computed before
        # that is now stale (it could exceed the deadline by the lookup's
        # duration). A negative value is clamped to a hair above zero rather than
        # passed through, so the attempt still costs one fast, classified failure
        # instead of an httpx ``Timeout`` built from a negative number.
        attempt_timeout = max(budget.remaining(), 0.001)
        origin = httpx.URL(url)
        async with owner.client(
            # The pool key keeps the CALL's timeout, not the attempt's: the
            # per-attempt budget rides on the request (see ``_stream_once``), so
            # every attempt of one call shares one pooled connection instead of
            # opening a fresh pool entry each time and defeating keep-alive.
            (
                "fetch",
                origin.scheme,
                origin.host,
                origin.port,
                budget.total,
                id(self.transport),
            ),
            transport=self.transport,
            follow_redirects=False,
            timeout=budget.total,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            extra = BROWSER_PROFILE_HEADERS if profile == "browser" else None
            return await self._stream_once(
                client,
                url,
                ceiling,
                pinned_ip,
                extra_headers=extra,
                attempt_timeout=attempt_timeout,
            )

    async def _try_enrichment(
        self, owner: WebReadIO, url: str, ceiling: int, budget: _Budget
    ) -> FetchResult | None:
        """Attempt the cheap enrichment candidates; return a result or ``None``.

        Each candidate rides the same SSRF-guarded :meth:`_follow`, so an
        enrichment fetch is exactly as safe as the primary one. A candidate wins
        only when it returns 2xx with substantial, non-HTML text — otherwise fall
        through so a site without these affordances pays nothing but the bounded,
        best-effort probe. Network errors on a probe are swallowed: enrichment is
        an optimization, never a reason to fail the real fetch.

        Cost (m2): a fresh non-``raw`` fetch pays up to two extra requests (the
        ``.md`` twin and, for a site root, ``/llms.txt``) before the real page.
        Accepted for the common docs-site win because the loop SHORT-CIRCUITS on
        the first substantial candidate (the ``return`` below), the candidate set
        is now tightly path-scoped (M2), and every 2xx result is cached — so the
        cost is paid once per URL+variant, not on every read. A per-call opt-out
        beyond the global ``enrich`` switch was judged not worth the param surface
        for a bounded two-probe cost.
        """
        # Probes get a bounded SLICE of the call's budget, never the whole
        # thing: enrichment is speculative, and a probe that stalls must not be
        # able to starve the request the user actually made (or the escalation
        # that would have diagnosed it).
        probe_budget = _Budget(
            deadline=min(
                budget.deadline,
                _now() + budget.total * _ENRICHMENT_BUDGET_FRACTION,
            ),
            total=budget.total * _ENRICHMENT_BUDGET_FRACTION,
            max_attempts=1,
            escalation_available=False,
        )
        for candidate in _enrichment_candidates(url):
            if probe_budget.remaining() <= 0:
                # The slice is spent. Stop probing rather than queue a request
                # that has no time to complete.
                return None
            try:
                # A probe gets ONE attempt and never the escalation: enrichment
                # is an optimisation, and spending retries or a browser-shaped
                # request on a speculative ``.md`` twin multiplies request volume
                # for no user-visible gain. It still shares the call's deadline,
                # so a stalling probe cannot eat the real fetch's budget.
                final_url, status, headers, body, complete = await self._follow(
                    owner,
                    candidate,
                    ceiling,
                    probe_budget,
                    _Telemetry(),
                    max_attempts=1,
                    allow_escalation=False,
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
        self,
        client: httpx.AsyncClient,
        url: str,
        ceiling: int,
        pinned_ip: str | None = None,
        *,
        extra_headers: Mapping[str, str] | None = None,
        attempt_timeout: float | None = None,
    ) -> tuple[int, dict[str, str], bytes, bool]:
        """One request, reading at most ``ceiling`` bytes off the wire.

        The cap is enforced DURING streaming: we stop pulling chunks once the cap
        is reached and flag the body truncated, so a hostile or huge endpoint can
        never balloon memory (the ``resp.text`` trap the design calls out).

        ``pinned_ip`` is the vetted address from :func:`validate_public_url`. When
        set, the connection is forced to that exact IP while the ``Host`` header
        and TLS SNI stay the ORIGINAL hostname — so the socket lands on the
        address we validated (closing the rebinding window, M1) yet the
        certificate is still verified against the real hostname (verification is
        NEVER disabled). ``None`` means the check was bypassed (``allow_private``)
        or the host was already an IP literal, and httpx connects normally.
        """
        request_url, request_headers, extensions = _pin_request(url, pinned_ip)
        if extra_headers:
            # The pin's ``Host`` header (and the SNI extension) are merged LAST,
            # so an escalated attempt carrying its own headers cannot accidentally
            # unpin itself by overwriting the hostname the certificate is verified
            # against.
            merged = dict(extra_headers)
            merged.update(request_headers)
            request_headers = merged
        buffer = bytearray()
        complete = True
        # The per-attempt budget rides on the REQUEST, not the client, so attempts
        # share one pooled connection while each gets its own slice of the
        # deadline. Connect is capped independently: a handshake that has not
        # completed in 5 s is not going to, and spending the whole budget on it
        # would leave nothing for the body. ``None`` (a direct caller that named
        # no budget) leaves the client's own timeout in force, which is the
        # pre-deadline behaviour.
        request_timeout = (
            httpx.Timeout(
                connect=min(_CONNECT_CAP_S, attempt_timeout),
                read=attempt_timeout,
                write=attempt_timeout,
                pool=min(_CONNECT_CAP_S, attempt_timeout),
            )
            if attempt_timeout is not None
            else httpx.USE_CLIENT_DEFAULT
        )
        try:
            async with client.stream(
                "GET",
                request_url,
                headers=request_headers,
                extensions=extensions,
                timeout=request_timeout,
            ) as response:
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
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            # The message is built from the CLASSIFIER, never from ``str(exc)``:
            # ``httpx.ReadTimeout.__str__()`` is the empty string, which is how
            # the shipped error came to read ``timed out fetching '<url>': ``
            # with nothing after the colon. ``str(exc)`` is appended only when it
            # actually says something.
            failure = classify_exception(exc, timeout_s=attempt_timeout)
            detail = str(exc).strip()
            message = f"{failure.detail or 'the request failed'}: {url}"
            if detail:
                message += f" ({detail})"
            raise FetchError(message, failure=failure) from exc
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
        telemetry: _Telemetry | None = None,
    ) -> FetchResult:
        """Classify the body and render it, producing the full FetchResult."""
        content_type = headers.get("content-type", "").lower()
        byte_count = len(body)
        telemetry = telemetry or _Telemetry(attempts=1, profiles=["default"])
        diagnostics = _diagnostics(telemetry)

        failure = telemetry.failure
        if failure is not None and failure.kind == "blocked":
            # The challenge body is REPLACED, not inlined — and only for this
            # class. It is markup whose entire purpose is to be executed by a
            # browser: 5.5 KB of Cloudflare interstitial (measured on
            # medium.com) that costs context to tell the agent nothing, under a
            # lead that invites the misread that it is page content.
            #
            # Every other non-2xx class still renders its body verbatim below.
            # A 404's body, a 451's and a 500's often genuinely explain
            # themselves, so the narrowing stops here.
            explanation = describe(
                failure, attempts=telemetry.attempts, profiles=telemetry.profiles
            )
            return FetchResult(
                url=request_url,
                final_url=final_url,
                status=status,
                content_type=content_type or "text/html",
                render_method="text",
                content=explanation,
                bytes=byte_count,
                complete=complete,
                **diagnostics,
            )

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
                **diagnostics,
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
            **diagnostics,
        )


class _Diagnostics(TypedDict):
    """The retry/block fields, typed so ``**`` into :class:`FetchResult` keeps
    its types instead of widening every field to ``object``."""

    attempts: int
    profile: str
    profiles: list[str]
    failure_kind: str | None
    block_vendor: str | None
    block_reference: str | None
    retry_after_s: float | None


def _diagnostics(telemetry: _Telemetry) -> _Diagnostics:
    """The retry/block fields every :class:`FetchResult` carries.

    One builder so a new construction site cannot forget half of them, and so a
    SUCCESS reports the honest ``attempts``/``profile`` (a 500-then-200 that took
    two tries says two) without inventing a ``failure_kind`` it does not have.
    """
    failure = telemetry.failure
    return {
        "attempts": max(telemetry.attempts, 1),
        "profile": telemetry.profile,
        "profiles": list(telemetry.profiles) or ["default"],
        "failure_kind": failure.kind if failure is not None else None,
        "block_vendor": failure.vendor if failure is not None else None,
        "block_reference": failure.reference if failure is not None else None,
        "retry_after_s": telemetry.retry_after_s,
    }


def _enrichment_candidates(url: str) -> list[str]:
    """Cheap, high-value alternate URLs to try before scraping the HTML page.

    A proportionate subset of omp's enrichment chain (design §6 step 3), each
    candidate scoped to the EXACT requested resource so the result can never be
    the wrong content for the URL the agent asked about:

    - the ``.md`` suffix twin (``/docs/guide`` → ``/docs/guide.md``) that GitHub
      and many docs generators honour — legitimately path-scoped: it is the same
      page in a cleaner format;
    - ``/llms.txt`` ONLY when the requested URL is the site ROOT (path empty or
      ``/``). ``/llms.txt`` is a SITE-WIDE index, so substituting it for an
      arbitrary subpage (``/docs/guide``) hands the model the whole site's index
      attributed to a URL it did not request — the M2 defect. Restricting it to
      the root means it is only ever returned when it genuinely IS the requested
      resource's best representation.

    Query strings and fragments are dropped from the ``.md`` candidate since a
    ``.md`` twin is a path concept. Only http(s) URLs are enriched.
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
    # Gated to the root: for a subpage it would be the wrong content (M2).
    if path in ("", "/"):
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

    Header first, then a NUL-byte sniff of the head: a mislabeled
    ``application/octet-stream`` that is really text still renders, and a
    ``text/*`` header that is really binary (rare, but possible) is caught by the
    sniff.

    ``octet-stream`` is deliberately NOT in the hard-binary set (m1): it is the
    generic "bytes of unknown type" header that misconfigured servers slap on
    plain-text and markdown responses, so treating it as unconditionally binary
    would return a useless notice for a readable body. It falls through to the
    NUL sniff, which classifies the ACTUAL content — the behaviour this
    docstring has always promised. PDF/zip stay hard-binary because their headers
    are specific and reliable.
    """
    primary = content_type.split(";", 1)[0].strip()
    if primary.startswith(("image/", "audio/", "video/", "font/")):
        return True
    if primary in ("application/pdf", "application/zip"):
        return True
    if primary.startswith("text/") or _matches(content_type, _JSON_TYPES):
        return False
    # No decisive header (octet-stream, unknown, or empty): a NUL byte in the
    # head is the classic binary tell; its absence means render it as text.
    return b"\x00" in body[:1024]
