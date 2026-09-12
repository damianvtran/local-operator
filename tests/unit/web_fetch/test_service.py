"""Service-layer tests: SSRF policy, streaming cap, redirect loop, cache index.

Uses httpx ``MockTransport`` for determinism. The SSRF checks resolve real
hostnames, so those tests monkeypatch ``_resolve_host_ips`` to pin an address
rather than depending on live DNS.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.web_fetch import service
from local_operator.web_fetch.models import WebFetchSettings
from local_operator.web_fetch.service import (
    FetchError,
    WebFetchService,
    coerce_fetch_settings,
    normalize_url,
    validate_public_url,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))


def _service(transport: httpx.MockTransport, **overrides) -> WebFetchService:
    settings = WebFetchSettings(enrich=False, **overrides)
    return WebFetchService(settings, transport=transport)


# --- URL normalization -----------------------------------------------------


def test_normalize_bare_host_defaults_https() -> None:
    assert normalize_url("example.com") == "https://example.com"


def test_normalize_collapsed_scheme() -> None:
    assert normalize_url("https:/example.com/x") == "https://example.com/x"


def test_normalize_rejects_non_http_scheme() -> None:
    with pytest.raises(FetchError, match="only http"):
        normalize_url("file:///etc/passwd")


# --- SSRF policy: direct ---------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1", "::ffff:169.254.169.254"],
)
def test_validate_refuses_private_targets(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: [ip])
    with pytest.raises(FetchError, match="private/loopback/reserved"):
        validate_public_url("http://evil.example/", allow_private=False)


def test_validate_allows_public_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    validate_public_url("https://example.com/", allow_private=False)  # no raise


def test_validate_allow_private_bypasses(monkeypatch: pytest.MonkeyPatch) -> None:
    # With allow_private, DNS is not even consulted — the switch is the escape.
    validate_public_url("http://localhost:3000/", allow_private=True)


def test_file_scheme_refused_at_normalize() -> None:
    with pytest.raises(FetchError):
        normalize_url("file:///etc/passwd")


# --- SSRF policy: via redirect (the case a scheme-only check misses) -------


@pytest.mark.asyncio
async def test_ssrf_via_redirect_is_refused_at_the_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public URL that 302s to the metadata endpoint must be refused, not
    followed. The redirect target is re-validated at the hop.

    The connection is now PINNED to the vetted IP (M1), so the request URL host
    is the IP literal; the intended hostname rides in the ``Host`` header, which
    is what the handler dispatches on.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("host", "").startswith("public.example"):
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})
        return httpx.Response(200, text="SHOULD NOT REACH")

    # public.example resolves public; the metadata IP is judged by its literal.
    monkeypatch.setattr(
        service,
        "_resolve_host_ips",
        lambda host: ["93.184.216.34"] if host == "public.example" else [host],
    )
    svc = _service(httpx.MockTransport(handler))
    with pytest.raises(FetchError, match="private/loopback/reserved"):
        await svc.fetch("http://public.example/")


@pytest.mark.asyncio
async def test_connection_pinned_to_vetted_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """M1: the socket must target the exact IP the validator vetted, not a
    hostname httpx re-resolves. Simulates DNS rebinding by having the validator
    see a public IP while asserting the connection carries THAT IP, never the
    hostname — so a second (malicious) resolution can never be used.

    The captured request URL host is the vetted IP literal; the ``Host`` header
    and (for https) the SNI extension preserve the original hostname so TLS still
    verifies against the name.
    """
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url_host"] = request.url.host
        captured["host_header"] = request.headers.get("host")
        captured["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    await svc.fetch("https://rebind.example/path")

    # The connection went to the vetted IP, NOT the hostname (rebinding closed).
    assert captured["url_host"] == "93.184.216.34"
    assert captured["host_header"] == "rebind.example"
    assert captured["sni"] == "rebind.example"


@pytest.mark.asyncio
async def test_rebinding_second_resolution_never_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that resolves public at validation but whose later lookups return a
    private address is still safe: the connection is pinned to the FIRST vetted
    address, so the private one is never reachable.

    Concretely: the validator returns a public IP; the connection must carry that
    IP. Even if httpx were to re-resolve (it cannot, because the URL is already an
    IP literal), it would land on the vetted public address.
    """
    resolutions = iter([["93.184.216.34"], ["10.0.0.5"]])  # 2nd (private) never used
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: next(resolutions))

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url_host"] = request.url.host
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    svc = _service(httpx.MockTransport(handler))
    await svc.fetch("https://rebind.example/")
    assert captured["url_host"] == "93.184.216.34"


# --- streaming byte cap ----------------------------------------------------


@pytest.mark.asyncio
async def test_max_bytes_enforced_during_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body larger than max_bytes stops downloading and is flagged truncated;
    the buffer never exceeds the ceiling (memory does not balloon)."""
    big = "A" * (2 * 1024 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=big, headers={"content-type": "text/plain"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler), max_bytes=64 * 1024)
    result = await svc.fetch("https://example.com/big")
    assert result.complete is False
    assert result.bytes <= 64 * 1024


@pytest.mark.asyncio
async def test_redirect_updates_final_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "https://example.com/new"})
        return httpx.Response(200, text="landed", headers={"content-type": "text/plain"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/old")
    assert result.final_url == "https://example.com/new"
    assert "landed" in result.content


@pytest.mark.asyncio
async def test_too_many_redirects_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/loop"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler), max_redirects=2)
    with pytest.raises(FetchError, match="too many redirects"):
        await svc.fetch("https://example.com/loop")


@pytest.mark.asyncio
async def test_timeout_names_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    with pytest.raises(FetchError, match="timed out"):
        await svc.fetch("https://example.com/slow")


@pytest.mark.asyncio
async def test_non_2xx_status_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html><body>Not found</body></html>")

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/missing")
    assert result.status == 404


@pytest.mark.asyncio
async def test_pdf_returns_binary_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"%PDF-1.4 fake", headers={"content-type": "application/pdf"}
        )

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/doc.pdf")
    assert result.render_method == "binary"
    assert "application/pdf" in result.content


@pytest.mark.asyncio
async def test_octet_stream_text_body_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    """m1: a text body mislabeled application/octet-stream renders as text (the
    NUL sniff decides), matching the docstring — it is not a binary notice."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"# A markdown doc\n\nReal readable text, no NUL bytes here.",
            headers={"content-type": "application/octet-stream"},
        )

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/data")
    assert result.render_method != "binary"
    assert "readable text" in result.content


@pytest.mark.asyncio
async def test_octet_stream_binary_body_is_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """m1 boundary: octet-stream with a NUL-containing body is still a binary
    notice — the sniff, not the header, is what classifies it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"\x00\x01\x02binary\x00payload",
            headers={"content-type": "application/octet-stream"},
        )

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/blob")
    assert result.render_method == "binary"


# --- config coercion & cache index -----------------------------------------


def test_coerce_clamps_bad_values() -> None:
    settings = coerce_fetch_settings(
        {"timeout_seconds": 0, "max_bytes": -5, "max_redirects": 999, "cache_ttl_seconds": -1}
    )
    assert settings.timeout_seconds >= 1.0
    assert settings.max_bytes >= 1024
    assert settings.max_redirects <= 20
    assert settings.cache_ttl_seconds == 0


def test_coerce_defaults_on_garbage() -> None:
    settings = coerce_fetch_settings("not a mapping")
    assert settings.enabled is True
    assert settings.max_redirects == 5


def test_cache_roundtrip() -> None:
    entry = service.CacheEntry(
        url="https://example.com/",
        final_url="https://example.com/",
        spill_handle="spill://" + "a" * 32,
        fetched_at_ms=123,
        status=200,
        content_type="text/html",
        render_method="markdownify",
        complete=True,
        low_quality=False,
    )
    service.write_cache_entry(entry)
    loaded = service.read_cache_entry("https://example.com/")
    assert loaded is not None
    assert loaded.spill_handle == entry.spill_handle
    assert loaded.status == 200


def test_cache_prune_keeps_newest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "CACHE_MAX_ENTRIES", 3)
    for i in range(6):
        service.write_cache_entry(
            service.CacheEntry(
                url=f"https://example.com/{i}",
                final_url=f"https://example.com/{i}",
                spill_handle="spill://" + str(i) * 32,
                fetched_at_ms=i,
                status=200,
                content_type="text/html",
                render_method="stdlib",
                complete=True,
                low_quality=False,
            )
        )
    remaining = list(service.cache_dir().iterdir())
    assert len(remaining) <= 3


# --- retries, deadline, and the blocked-profile escalation ------------------
#
# These cover §3-§4 of docs/design/web_fetch_robustness.md. The recurring shape
# is a MockTransport that counts its calls, because "how many requests did we
# actually issue" is the property every one of these rests on and a mock's
# return value cannot show it.


class _Recorder:
    """Counts requests and records the identity each one carried.

    The profile is read off the User-Agent rather than asserted through an
    internal hook: what an origin sees is the only thing that matters here, and
    a test that inspected our own call arguments would keep passing if the
    headers stopped reaching the wire.
    """

    def __init__(self, responder) -> None:
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(len(self.requests), request)

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def profiles(self) -> list[str]:
        return [
            "browser" if "Chrome/" in r.headers.get("user-agent", "") else "default"
            for r in self.requests
        ]


CHALLENGE_HEADERS = {"content-type": "text/html", "cf-mitigated": "challenge"}
CHALLENGE_BODY = "<html><head><title>Just a moment...</title></head><body></body></html>"


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff sleeps instead of serving them.

    Per AGENTS.md ("wait on the event, never on the clock"), the assertions
    below are on WHAT was slept, never on elapsed wall time — so these tests
    cannot flake under load and do not spend a second of suite time proving a
    backoff happened.
    """
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(service, "_backoff_sleep", fake_sleep)
    return slept


@pytest.mark.asyncio
async def test_transient_500_then_200_recovers(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The headline case: a 500 from a rolling deploy no longer fails the call."""

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        if n == 1:
            return httpx.Response(500, text="broke")
        return httpx.Response(200, text="recovered", headers={"content-type": "text/plain"})

    recorder = _Recorder(responder)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    result = await svc.fetch("https://example.com/x")

    assert result.status == 200
    assert "recovered" in result.content
    assert result.attempts == 2
    assert recorder.calls == 2
    assert len(_no_sleep) == 1  # exactly one backoff, between the two attempts
    # A retried SUCCESS is a success: the earlier 500 must not be reported as
    # the outcome's failure class.
    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_persistent_500_stops_at_max_attempts(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The attempt count is bounded and asserted on the TRANSPORT, not inferred."""
    recorder = _Recorder(lambda n, req: httpx.Response(500, text="still broke"))
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder), max_attempts=3)
    result = await svc.fetch("https://example.com/x")

    assert result.status == 500
    assert recorder.calls == 3
    assert result.attempts == 3
    assert result.failure_kind == "server"


@pytest.mark.asyncio
async def test_404_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 will not become a 200; retrying spends the user's turn for nothing."""
    recorder = _Recorder(lambda n, req: httpx.Response(404, text="gone"))
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    result = await svc.fetch("https://example.com/missing")

    assert recorder.calls == 1
    assert result.attempts == 1
    assert result.failure_kind == "client"


@pytest.mark.asyncio
async def test_retry_after_small_value_is_honoured(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The origin named its own interval and it fits inside the turn: sleep it."""

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        if n == 1:
            return httpx.Response(429, text="slow down", headers={"retry-after": "2"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    recorder = _Recorder(responder)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    result = await svc.fetch("https://example.com/x")

    assert result.status == 200
    # Slept EXACTLY the origin's number, not our backoff curve.
    assert _no_sleep == [2.0]


@pytest.mark.asyncio
async def test_retry_after_large_value_is_reported_not_slept(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """Blocking a live turn for 15 minutes to obey a header is worse for the user
    than handing the agent the number and letting it come back later."""
    recorder = _Recorder(
        lambda n, req: httpx.Response(429, text="later", headers={"retry-after": "900"})
    )
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    result = await svc.fetch("https://example.com/x")

    assert recorder.calls == 1
    assert _no_sleep == []  # nothing was slept
    assert result.retry_after_s == 900.0


@pytest.mark.asyncio
async def test_validate_public_url_runs_once_per_attempt(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """THE rebinding guard across retries (§8 risk 1).

    Hoisting ``validate_public_url`` out of the attempt loop "for efficiency"
    would let a host that turns private between attempts be re-contacted without
    a check. The count is asserted directly: one validation per network attempt,
    never one per hop.
    """
    validations: list[str] = []
    real_validate = service.validate_public_url

    def counting_validate(url: str, *, allow_private: bool = False) -> str | None:
        validations.append(url)
        return real_validate(url, allow_private=allow_private)

    monkeypatch.setattr(service, "validate_public_url", counting_validate)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])

    recorder = _Recorder(lambda n, req: httpx.Response(503, text="down"))
    svc = _service(httpx.MockTransport(recorder), max_attempts=3)
    await svc.fetch("https://example.com/x")

    assert recorder.calls == 3
    assert len(validations) == 3
    assert validations == ["https://example.com/x"] * 3


@pytest.mark.asyncio
async def test_retry_refuses_a_host_that_turns_private(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The consequence of the above, end to end: attempt 1 resolves public and
    fails; attempt 2 resolves to loopback and must be REFUSED, not followed."""
    resolutions = iter([["93.184.216.34"], ["127.0.0.1"], ["127.0.0.1"]])
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: next(resolutions))

    recorder = _Recorder(lambda n, req: httpx.Response(500, text="broke"))
    svc = _service(httpx.MockTransport(recorder))
    with pytest.raises(FetchError, match="private/loopback/reserved"):
        await svc.fetch("https://rebind.example/x")

    # The second attempt never reached the transport.
    assert recorder.calls == 1


@pytest.mark.asyncio
async def test_every_attempt_including_the_escalation_stays_pinned(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The pin must survive the browser profile: the escalated request carries
    its own headers, and merging them over the pin's ``Host`` would unpin it."""
    recorder = _Recorder(
        lambda n, req: httpx.Response(403, text=CHALLENGE_BODY, headers=CHALLENGE_HEADERS)
    )
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    await svc.fetch("https://pinned.example/x")

    assert recorder.calls == 2
    assert recorder.profiles == ["default", "browser"]
    for request in recorder.requests:
        assert request.url.host == "93.184.216.34"  # the vetted IP, never the name
        assert request.headers["host"] == "pinned.example"
        assert request.extensions.get("sni_hostname") == "pinned.example"


@pytest.mark.asyncio
async def test_blocked_retry_flips_a_challenge_to_success(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The measured medium.com win, reproduced deterministically: the honest
    profile is refused and the browser-shaped one is served."""

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        if "Chrome/" in request.headers.get("user-agent", ""):
            return httpx.Response(200, text="the real page", headers={"content-type": "text/html"})
        return httpx.Response(403, text=CHALLENGE_BODY, headers=CHALLENGE_HEADERS)

    recorder = _Recorder(responder)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    result = await svc.fetch("https://walled.example/x")

    assert result.status == 200
    assert "the real page" in result.content
    assert result.attempts == 2
    assert result.profile == "browser"
    assert recorder.profiles == ["default", "browser"]


@pytest.mark.asyncio
async def test_blocked_retry_fires_exactly_once(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """A refusal costs exactly 2 requests: the honest one and the escalation.
    There is no third, and the block class is never retried with the same
    identity (the origin already decided about this client)."""
    recorder = _Recorder(
        lambda n, req: httpx.Response(403, text=CHALLENGE_BODY, headers=CHALLENGE_HEADERS)
    )
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder), max_attempts=3)
    result = await svc.fetch("https://walled.example/x")

    assert recorder.calls == 2
    assert result.failure_kind == "blocked"
    assert result.block_vendor == "cloudflare"


@pytest.mark.asyncio
async def test_blocked_retry_disabled_costs_one_request(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The switch exists for an operator who wants the client to stay honest
    even in the face of a refusal."""
    recorder = _Recorder(
        lambda n, req: httpx.Response(403, text=CHALLENGE_BODY, headers=CHALLENGE_HEADERS)
    )
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder), blocked_retry=False)
    result = await svc.fetch("https://walled.example/x")

    assert recorder.calls == 1
    assert recorder.profiles == ["default"]
    assert result.failure_kind == "blocked"


@pytest.mark.asyncio
async def test_escalation_is_once_per_fetch_not_once_per_hop(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """A redirect chain must not be able to multiply the escalation budget."""

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://example.com/b"})
        return httpx.Response(403, text=CHALLENGE_BODY, headers=CHALLENGE_HEADERS)

    recorder = _Recorder(responder)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder))
    await svc.fetch("https://example.com/a")

    # hop 1 (the 302) + hop 2 default + hop 2 escalation == 3, never 4.
    assert recorder.calls == 3
    assert recorder.profiles.count("browser") == 1


@pytest.mark.asyncio
async def test_deadline_bounds_the_whole_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hard invariant (§3.3): retries divide ONE budget rather than each
    getting a fresh copy of the timeout.

    Asserted structurally on a fake clock rather than on wall time, per AGENTS.md
    "prefer a structural invariant to a numeric one": the timeout each attempt is
    granted must never exceed what is left of the call's budget.
    """
    granted: list[float] = []
    clock = {"t": 0.0}
    monkeypatch.setattr(service, "_now", lambda: clock["t"])

    async def fake_sleep(delay: float) -> None:
        clock["t"] += delay

    monkeypatch.setattr(service, "_backoff_sleep", fake_sleep)

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout") or {}
        granted.append(float(timeout.get("read", 0.0)))
        # Each attempt "spends" most of what it was granted before failing.
        clock["t"] += granted[-1] * 0.9
        return httpx.Response(503, text="down")

    recorder = _Recorder(responder)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder), max_attempts=5)
    await svc.fetch("https://example.com/x", timeout_seconds=10.0)

    assert granted, "no attempt recorded its timeout"
    # Every attempt's budget fits inside what remained, and the total never
    # exceeds the call's timeout.
    assert all(t > 0 for t in granted)
    assert clock["t"] <= 10.0
    # The deadline, not max_attempts, is what stopped it.
    assert recorder.calls < 5


@pytest.mark.asyncio
async def test_first_attempt_is_shortened_while_a_fallback_remains(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """§3.3's stall shortening: attempt 1 gets at most 0.6·T so a black-holing
    origin leaves time for the browser-shaped probe that actually answers.

    The LAST attempt still gets the full remaining budget — a genuinely slow
    origin must not be penalised on its final try — which is the second half of
    the assertion.
    """
    granted: list[float] = []

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout") or {}
        granted.append(float(timeout.get("read", 0.0)))
        raise httpx.ReadTimeout("", request=request)

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(_Recorder(responder)))
    with pytest.raises(FetchError):
        await svc.fetch("https://stalls.example/x", timeout_seconds=20.0)

    assert len(granted) == 2  # the stall earns the escalation, not a blind repeat
    assert granted[0] <= 20.0 * service._STALL_FIRST_ATTEMPT_FRACTION + 0.01
    # The escalated attempt is the last one and is not artificially shortened.
    assert granted[1] > granted[0] * 0.5


@pytest.mark.asyncio
async def test_stall_message_names_the_class_and_is_never_empty(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """The §1.3 regression, end to end: ``httpx.ReadTimeout`` stringifies to ""
    and the old message ended in ``': '`` with nothing after it."""

    def responder(n: int, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(_Recorder(responder)))
    with pytest.raises(FetchError) as excinfo:
        await svc.fetch("https://stalls.example/x", timeout_seconds=5.0)

    message = str(excinfo.value)
    assert "timed out" in message
    assert "never sent a response" in message
    assert "https://stalls.example/x" in message
    assert not message.rstrip().endswith(":")
    assert not message.rstrip().endswith("': '")


@pytest.mark.asyncio
async def test_enrichment_probe_never_spends_the_escalation(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    """A speculative ``.md`` twin is an optimisation, and must not pay a second
    request to diagnose itself — nor consume the escalation the real fetch may
    need."""
    recorder = _Recorder(
        lambda n, req: (
            httpx.Response(403, text=CHALLENGE_BODY, headers=CHALLENGE_HEADERS)
            if req.url.path.endswith(".md")
            else httpx.Response(200, text="the page", headers={"content-type": "text/html"})
        )
    )
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    settings = WebFetchSettings(enrich=True)
    svc = WebFetchService(settings, transport=httpx.MockTransport(recorder))
    result = await svc.fetch("https://docs.example.com/guide/page")

    assert result.status == 200
    # The blocked probe cost exactly one request, and the escalation was still
    # available (unspent) for the real fetch.
    md_calls = [r for r in recorder.requests if r.url.path.endswith(".md")]
    assert len(md_calls) == 1


@pytest.mark.asyncio
async def test_max_attempts_one_reproduces_the_pre_retry_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_attempts: 1`` with the escalation off is byte-for-byte today's
    engine: one request, one result, no diagnostics invented."""
    recorder = _Recorder(
        lambda n, req: httpx.Response(200, text="hello", headers={"content-type": "text/plain"})
    )
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(recorder), max_attempts=1, blocked_retry=False)
    result = await svc.fetch("https://example.com/x")

    assert recorder.calls == 1
    assert result.attempts == 1
    assert result.profile == "default"
    assert result.failure_kind is None
    assert "hello" in result.content


def test_coerce_clamps_attempts() -> None:
    assert coerce_fetch_settings({"max_attempts": 99}).max_attempts == 5
    assert coerce_fetch_settings({"max_attempts": 0}).max_attempts == 1
    assert coerce_fetch_settings({}).max_attempts == 3
    assert coerce_fetch_settings({}).blocked_retry is True
