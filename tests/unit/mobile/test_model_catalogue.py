"""A cold phone picker must list OAuth aggregators through real discovery.

Only the HTTP transport is replaced. AuthStore, expiry refresh, catalogue disk
I/O, provider parser, mobile authentication, and response projection are real.
The suite's fresh HOME also isolates the catalogue (config-dir alone does not).
"""

import contextlib
import json
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient

from local_operator.credentials import CredentialManager
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.model.catalogue import default_cache_dir
from local_operator.paths import config_dir
from local_operator.providers.auth_store import AuthStore


@pytest.mark.parametrize("expired", [False, True])
def test_cold_radient_oauth_models_refresh_parse_and_cache_off_loop(monkeypatch, expired):
    with contextlib.closing(AuthStore()) as store:
        credential = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "fixture-access",
                "refresh": "fixture-refresh",
                "expires": 0 if expired else int(time.time() * 1000) + 3_600_000,
            },
        )
    assert not default_cache_dir().exists()
    # The daemon may inherit other tooling's env. It must use this stored login
    # and never expose those unrelated providers or override it with an env key.
    monkeypatch.setenv("RADIENT_API_KEY", "ambient-radient-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-openrouter-key")
    requests = []
    refreshes = []
    loop_threads = set()

    def models(_transport, request):
        assert threading.get_ident() not in loop_threads
        assert str(request.url) == "https://api.radienthq.com/v1/models"
        expected = "fixture-refreshed" if expired else "fixture-access"
        assert request.headers["authorization"] == f"Bearer {expected}"
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/mobile-fixture", "name": "Mobile fixture"},
                    {"id": "auto", "name": "Automatic"},
                ]
            },
        )

    async def refresh(_transport, request):
        assert threading.get_ident() not in loop_threads
        assert str(request.url) == "https://api.radienthq.com/v1/auth/oauth/token"
        assert json.loads(request.content)["refresh_token"] == "fixture-refresh"
        refreshes.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "fixture-refreshed",
                "refresh_token": "fixture-rotated",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refresh)
    relay = build_app(MobileDaemon(password="fixture-password", dial_registrants=False))

    async def app(scope, receive, send):
        loop_threads.add(threading.get_ident())
        await relay(scope, receive, send)

    with TestClient(app, follow_redirects=False) as client:
        assert client.get("/api/models").status_code == 401
        assert not requests
        client.post("/login", data={"password": "fixture-password"})
        first = client.get("/api/models")
        assert first.status_code == 200, first.json()
        # The payload carries what the phone RENDERS, and the rows arrive in the
        # picker's own order rather than the registry's. ``name`` is the
        # listing's OWN name even for an aggregator — the row's provider slot
        # already says which route answers, so suppressing the name there (which
        # is right for the TUI, whose row paints a separate selector column) left
        # the phone showing a slug. ``label`` still degrades to the selector for
        # a reseller, unchanged, because that is the string the TUI got.
        assert first.json() == {
            "models": [
                {
                    "selector": "radient/anthropic/mobile-fixture",
                    "provider": "radient",
                    "model_id": "anthropic/mobile-fixture",
                    "name": "Mobile fixture",
                    "label": "radient/anthropic/mobile-fixture",
                    "connected": True,
                    "aggregated": True,
                },
                {
                    "selector": "radient/auto",
                    "provider": "radient",
                    "model_id": "auto",
                    "name": "Automatic",
                    "label": "radient/auto",
                    "connected": True,
                    "aggregated": True,
                },
            ]
        }
        assert client.get("/api/models").json() == first.json()
    assert len(requests) == 1  # Second picker open reads the real fresh cache.
    assert len(refreshes) == int(expired)
    assert "fixture-access" not in first.text
    assert "fixture-refreshed" not in first.text
    with contextlib.closing(AuthStore()) as store:
        row = store.get_credential(credential.id)
        assert row is not None
        assert row.disabled_cause is None
        if expired:
            assert row.data["refresh"] == "fixture-rotated"


@pytest.mark.parametrize("storage", ["login", "legacy"])
def test_openrouter_stored_key_lists_real_catalogue(monkeypatch, storage):
    if storage == "login":
        with contextlib.closing(AuthStore()) as store:
            store.upsert_credential("openrouter", {"type": "api_key", "key": "fixture-key"})
    else:
        CredentialManager(config_dir()).set_credential("OPENROUTER_API_KEY", "fixture-key")

    def models(_transport, request):
        assert str(request.url) == "https://openrouter.ai/api/v1/models"
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(200, json={"data": [{"id": "vendor/model"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")
    assert response.status_code == 200
    assert [row["selector"] for row in response.json()["models"]] == ["openrouter/vendor/model"]


def test_failed_cold_catalogue_is_not_reported_as_empty_inventory(monkeypatch):
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-secret"})
    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda *_: httpx.Response(503, text="upstream diagnostic with fixture-secret"),
    )
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")
    assert response.status_code == 502
    assert "retry or log in again" in response.json()["error"]
    assert "fixture-secret" not in response.text


def test_no_login_or_disabled_login_never_fetches_public_catalogue(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *args: calls.append(args))
    with contextlib.closing(AuthStore()) as store:
        row = store.upsert_credential("radient", {"type": "oauth", "access": "disabled"})
        store.disable_credential(row.id, "test")
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        assert client.get("/api/models").json() == {"models": []}
    assert calls == []


def test_provider_outside_the_stale_enumeration_is_listed_when_persisted(monkeypatch):
    """A login the OLD enumeration could not name must reach the phone's sheet.

    ``alibaba-token-plan`` is in ``providers.registry.PROVIDER_REGISTRY`` and not
    in ``model.registry.SupportedHostingProviders``, which is what this endpoint
    used to walk — so a fully logged-in owner saw none of its models on the phone
    while ``/model`` offered them on the desktop. It stands in here for the five
    ids in that gap (``openai-device``, ``radient-key``, ``xai-oauth``,
    ``zai-oauth`` and the two alibaba flavours).
    """
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("alibaba-token-plan", {"type": "api_key", "key": "fixture-key"})

    def models(_transport, request):
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(200, json={"data": [{"id": "qwen3-max", "name": "Qwen3 Max"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")
    assert response.status_code == 200, response.json()
    rows = response.json()["models"]
    assert "alibaba-token-plan/qwen3-max" in {row["selector"] for row in rows}


def test_rows_arrive_ranked_direct_first_with_aggregators_last(monkeypatch):
    """The bug this endpoint was rebuilt for: the phone got REGISTRY order.

    Two providers hold the same model. The direct route must lead — it is the
    one the owner meant after logging in to xAI — and every aggregated row must
    sit behind every direct one, which is precisely what ``rank_rows``
    guarantees and what registry order destroyed by putting ~445 Radient rows
    ahead of the first direct provider.

    ``xai`` rather than ``anthropic`` as the direct route on purpose: it sorts
    AFTER ``openrouter`` alphabetically, so this cannot pass on the
    provider-name rung alone if the aggregator tier were lost.
    """
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("xai", {"type": "api_key", "key": "xai-key"})
        store.upsert_credential("openrouter", {"type": "api_key", "key": "openrouter-key"})

    def models(_transport, request):
        if request.url.host == "api.x.ai":
            return httpx.Response(200, json={"data": [{"id": "grok-4"}]})
        return httpx.Response(200, json={"data": [{"id": "x-ai/grok-4"}, {"id": "vendor/other"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        rows = client.get("/api/models").json()["models"]

    selectors = [row["selector"] for row in rows]
    # The same model by two routes: the DIRECT one leads.
    assert selectors.index("xai/grok-4") < selectors.index("openrouter/x-ai/grok-4")
    assert rows[0]["provider"] == "xai"
    assert rows[0]["aggregated"] is False
    aggregated = [index for index, row in enumerate(rows) if row["aggregated"]]
    direct = [index for index, row in enumerate(rows) if not row["aggregated"]]
    assert min(aggregated) > max(direct)


def test_ambient_env_key_never_joins_a_remote_picker(monkeypatch):
    """A service manager's environment is not the phone owner's consent.

    The daemon is typically launched by launchd/systemd, whose environment the
    phone's user never chose and cannot inspect. ``usable_providers`` counts an
    env key (correctly, for a local turn); ``persisted_providers`` does not, so
    an inherited ``ANTHROPIC_API_KEY`` must neither appear in the sheet nor
    cause a single request to Anthropic.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-access"})
    hosts = []

    def models(_transport, request):
        hosts.append(request.url.host)
        return httpx.Response(200, json={"data": [{"id": "vendor/model"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")

    assert response.status_code == 200, response.json()
    assert {row["provider"] for row in response.json()["models"]} == {"radient"}
    assert "api.anthropic.com" not in hosts
    assert "ambient-anthropic-key" not in response.text


def test_admitted_empty_set_fetches_nothing_at_all(monkeypatch):
    """``providers=set()`` is honoured literally, not read as "everything".

    An owner who has logged in to nothing is a real state with a real answer:
    no listing is authorized, so no provider may be contacted. A narrowing that
    fell back to the whole registry on an empty collection would turn the
    strictest case into the loosest one.
    """
    import asyncio

    from local_operator.credentials import CredentialManager
    from local_operator.providers.controller import ProviderController

    calls = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *args: calls.append(args))
    with contextlib.closing(AuthStore()) as store:
        controller = ProviderController(store, CredentialManager(config_dir=config_dir()))
        entries, statuses = asyncio.run(controller.live_catalogue(providers=set()))
    assert entries == []
    assert statuses == {}
    assert calls == []


def test_an_aggregated_row_shows_its_human_name_and_keeps_the_desktop_label(monkeypatch):
    """The phone renders the listing's NAME; ``label`` stays what the TUI got.

    The row has two slots — name and provider — so the provider slot already
    says which route answers. Sourcing the name from ``label`` therefore paid
    for that disambiguation twice and spent the whole name slot doing it:
    ``model_label`` refuses a reseller's listing name, so 916 of 996 rows
    rendered ``anthropic/claude-opus-5`` where the desktop rendered ``Claude
    Opus 5``. ``label`` is the parity contract and must NOT move.
    """
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-access"})

    def models(_transport, request):
        return httpx.Response(
            200,
            json={"data": [{"id": "anthropic/claude-opus-5", "name": "Claude Opus 5"}]},
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        rows = client.get("/api/models").json()["models"]

    row = next(r for r in rows if r["selector"] == "radient/anthropic/claude-opus-5")
    assert row["aggregated"] is True
    assert row["name"] == "Claude Opus 5"
    # Unchanged: a reseller's label still degrades to the selector, which is
    # exactly the string the desktop picker receives for this row.
    assert row["label"] == "radient/anthropic/claude-opus-5"


def test_models_are_served_gzipped_when_the_client_accepts_it(monkeypatch):
    """The catalogue is the one large one-shot body on this daemon.

    Per-route rather than ``GZipMiddleware`` because that wraps the SSE stream
    too, and gzip buffers — the phone's live turn output would arrive in blocks
    instead of event-by-event. The compressed body must decode to the identical
    JSON: a transfer encoding that changes the payload is not a transfer
    encoding.
    """
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-access"})

    def models(_transport, request):
        return httpx.Response(
            200,
            json={"data": [{"id": f"vendor/model-{n}", "name": f"Model {n}"} for n in range(200)]},
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        compressed = client.get("/api/models", headers={"Accept-Encoding": "gzip"})
        plain = client.get("/api/models", headers={"Accept-Encoding": "identity"})

    assert compressed.status_code == 200
    assert compressed.headers["content-encoding"] == "gzip"
    assert compressed.headers["vary"] == "Accept-Encoding"
    # httpx decodes the transfer encoding itself, so a readable body here IS the
    # round trip: the gzip stream decompressed to valid JSON. What it cannot show
    # is the size on the wire, which is what `content-length` carries.
    assert compressed.json() == plain.json()
    assert "content-encoding" not in plain.headers
    wire = int(compressed.headers["content-length"])
    assert wire < int(plain.headers["content-length"])
    assert wire < len(plain.content)


def test_gzip_cannot_touch_a_streaming_response():
    """The structural reason the gzip above is applied per-route.

    ``GZipMiddleware`` would wrap ``/api/sessions/{id}/events`` too, and gzip
    buffers — a phone watching a turn would stop seeing events arrive one at a
    time. The helper is only called from ``api_models``, and this pins the
    second line of defence: handed a streaming body it has nothing to compress
    and must add no encoding header, so wiring it somewhere it does not belong
    degrades to a no-op rather than to a silently buffered stream.
    """
    from starlette.responses import StreamingResponse

    from local_operator.mobile.daemon import _maybe_gzip

    async def body():
        yield b"event: sessions\n\n"

    class _Request:
        headers = {"accept-encoding": "gzip"}

    response = StreamingResponse(body(), media_type="text/event-stream")
    assert _maybe_gzip(_Request(), response) is response
    assert "content-encoding" not in response.headers


def test_an_unopenable_store_serves_the_cached_catalogue_not_a_502(monkeypatch):
    """ "I could not look" is not "you own nothing", on the path that actually raises.

    ``persisted_providers`` documents a ``None`` "cannot tell" rung, but
    ``AuthStore.__init__`` connects EAGERLY, so an unreadable ``auth.db`` raised
    out of the constructor before that method ran and the phone got a 502
    carrying a raw SQLite string. The degradation now happens where the read is.
    """
    import sqlite3

    from local_operator.providers import auth_store as auth_store_module

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(auth_store_module.AuthStore, "__init__", explode)
    calls = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *a: calls.append(a))

    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")

    assert response.status_code == 200, response.text
    rows = response.json()["models"]
    # The shipped catalogue still describes the models; claiming an empty
    # inventory would assert something the app never established.
    assert len(rows) > 0
    # And no provider was contacted: which accounts we may speak for is exactly
    # the question that just failed to resolve.
    assert calls == []
    assert "unable to open database file" not in response.text


@pytest.mark.parametrize(
    ("accept_encoding", "compressed"),
    [
        # Asking for it.
        ("gzip", True),
        ("gzip;q=1.0", True),
        ("gzip, deflate, br", True),
        ("GZIP", True),
        ("deflate, gzip;q=0.5", True),
        # REFUSING it. RFC 9110 §12.5.3 gives qvalue 0 the meaning "not
        # acceptable", and a substring test cannot tell this from consent — it
        # served a gzip body to a client that had explicitly declined one, which
        # a non-decoding client (``urllib``) meets as a UnicodeDecodeError on the
        # gzip magic bytes rather than as JSON.
        ("gzip;q=0", False),
        ("gzip;q=0.0", False),
        ("gzip;q=0, deflate", False),
        ("deflate, gzip;q=0", False),
        # Never compressed before this fix and deliberately still not: treating
        # the wildcard as consent would newly compress for clients this daemon
        # has always answered in the clear.
        ("*", False),
        ("*;q=0, identity", False),
        ("identity", False),
        ("deflate", False),
        ("br", False),
        ("", False),
    ],
)
def test_gzip_honours_the_accept_encoding_qvalue(monkeypatch, accept_encoding, compressed):
    """``Accept-Encoding`` is parsed, not substring-matched."""
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-access"})

    def models(_transport, request):
        return httpx.Response(
            200,
            json={"data": [{"id": f"vendor/model-{n}", "name": f"Model {n}"} for n in range(200)]},
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models", headers={"Accept-Encoding": accept_encoding})

    assert response.status_code == 200
    assert (response.headers.get("content-encoding") == "gzip") is compressed
    # The document is the same either way; only its transfer encoding differs.
    assert len(response.json()["models"]) == 200
    # Every representation carries it, not just the compressed one — a cache
    # keys on the headers of the variant it STORED, so announcing it only on the
    # gzip leg leaves the identity response looking like the single valid answer
    # for this URL.
    assert response.headers["vary"] == "Accept-Encoding"


def test_a_refused_gzip_body_is_readable_without_a_decoder():
    """QA's exact repro: a client that does not auto-decode must get text.

    ``httpx`` and ``curl`` transparently decompress, which is what let this
    survive review; ``urllib`` does not, so it is the honest reader here.
    """
    import gzip as gzip_module

    from local_operator.mobile.daemon import _accepts_gzip

    # The decision function, at the level the bug lived at.
    assert _accepts_gzip("gzip;q=0") is False
    assert _accepts_gzip("gzip") is True

    # And the consequence it used to have, made concrete: a UTF-8 decode of a
    # gzip stream fails on the magic bytes at position 1.
    packed = gzip_module.compress(b'{"models": []}')
    with pytest.raises(UnicodeDecodeError):
        packed.decode("utf-8")


def test_gzip_leaves_a_head_request_and_an_incompressible_body_alone():
    """Two ways compressing is wrong even when the client asked for it.

    HEAD: Starlette strips the body after the handler returns, so compressing
    would advertise the COMPRESSED length for a body the client never receives.
    Incompressible: gzip's header makes an already-packed payload bigger, and
    spending CPU to enlarge a response is never right — JSON never reaches this,
    but the helper must not depend on its only caller's payload shape.
    """
    import os

    from starlette.responses import Response

    from local_operator.mobile.daemon import _GZIP_MIN_BYTES, _maybe_gzip

    class _Request:
        def __init__(self, method="GET"):
            self.method = method
            self.headers = {"accept-encoding": "gzip"}

    # The two halves need DIFFERENT bodies, and that is the whole point rather
    # than tidiness. Sharing one random body made this test vacuous: gzip always
    # expands random bytes, so the incompressible early-return satisfied the HEAD
    # assertion by itself and the HEAD guard could be deleted with the suite
    # still green. A guard asserted for a reason unrelated to the guard is not
    # coverage. So HEAD is checked against a body that WOULD compress — leaving
    # the early return the only thing that can keep the encoding off it.
    compressible = b"a" * (_GZIP_MIN_BYTES + 100)
    head = Response(content=compressible)
    assert _maybe_gzip(_Request("HEAD"), head) is head
    assert "content-encoding" not in head.headers
    assert head.headers["vary"] == "Accept-Encoding"
    # The same body over GET must compress, which is what proves the assertion
    # above is about the METHOD and not about this payload being unshrinkable.
    getted = Response(content=compressible)
    _maybe_gzip(_Request("GET"), getted)
    assert getted.headers["content-encoding"] == "gzip"

    incompressible = Response(content=os.urandom(_GZIP_MIN_BYTES + 100))
    original = len(incompressible.body)
    _maybe_gzip(_Request(), incompressible)
    assert "content-encoding" not in incompressible.headers
    assert len(incompressible.body) == original
