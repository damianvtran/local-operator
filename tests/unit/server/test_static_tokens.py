"""The Phase A local-surface credentials: token, signed URLs, generation counter.

Two layers, like the static-roots suite, because each can regress alone:

* the pure contract (``server/utils/local_tokens.py``) -- HKDF derivation, token
  format/classification, the canonical signed byte string and its tamper set,
  the generation counter;
* the live routes through the real app, because the acceptance wiring exists
  only in ``routes/static.py`` and the sign endpoint's door spans two planes.

Phase A's defining property is asserted here as a MATRIX, not a sentence:
nothing on the static routes refuses for a credential reason -- absent, invalid,
stale, valid, signed-valid and signed-tampered all serve. The flip to refusing
is Phases B/C's change (RFC §4.2); when it lands, these rows are what says
exactly which credential states changed behaviour.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import stat
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.server import desktop
from local_operator.server.utils import local_tokens
from local_operator.server.utils.local_tokens import SurfaceKeys

PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

LOG_NAME = "local_operator.server.utils.local_tokens"

#: Shapes an attacker can send that no boot could have minted. Every one must
#: classify "invalid" WITHOUT raising, and identically whatever generation the
#: daemon serves (R1-1/S-1: two of these used to raise -- a 500 on a route
#: whose contract is "served regardless" -- and the non-ASCII one raised only
#: when its prefix MATCHED the serving generation, which made it a generation
#: oracle).
MALFORMED_TOKENS = (
    "",
    ".",
    "..",
    "7",
    "x.y",
    "7.",
    ".abc",
    "\u0667.abc",
    "-1.abc",
    " 7.abc",
    "7.a\t",
    "7.a b",
    "7.a.b",
    "7.a!",
    "9" * 5000 + ".x",  # past the interpreter's str-digit wall: int() would raise
    "9" * 4300 + ".x",  # below the wall, still bounded out before conversion
    "1.\u00e9",  # non-ASCII secret: the digest comparison would raise, and the old
    "2.\u00e9",  # raise distinguished "prefix == serving generation"
)


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Per-test key/log/counter state; the module is process-global by design.

    ``mint_process_keys``' first-call-wins is the production contract, so tests
    replace the singleton directly (the way the desktop tests replace
    ``desktop._CLAIMED``) rather than pretending a second mint can happen. The
    desktop plane is also closed per test: the sign endpoint's door spans both
    planes, and a claim left by another server test must not leak into it.
    """
    monkeypatch.setattr(local_tokens, "_KEYS", None)
    monkeypatch.setattr(local_tokens, "_STALE_LOGGED", set())
    monkeypatch.setattr(local_tokens, "_STATS", {key: 0 for key in local_tokens._STATS})
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    monkeypatch.delenv(desktop.ORIGINS_ENV, raising=False)


def _keys(generation: int = 1) -> SurfaceKeys:
    return local_tokens.mint_process_keys(generation=generation)


def _verify(
    url: str, *, route: str = "images", path: str | None = None, now: float | None = None
) -> bool:
    """Verify a minted URL the way the route does: decoded params, route, path."""
    query = parse_qs(urlsplit(url).query)
    return local_tokens.verify_signed_url(
        route=route,
        path=query["path"][0] if path is None else path,
        exp_raw=query["exp"][0],
        sig_raw=query["sig"][0],
        now=now,
    )


# ------------------------------------------------------------------- the crypto


def test_hkdf_matches_the_cryptography_library():
    """The stdlib derivation against the battle-tested dependency: same salt,
    info and length, so a hand-rolled extract/expand cannot drift silently."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    ikm, info = bytes(range(32)), b"cross-check-info"
    expected = HKDF(algorithm=hashes.SHA256(), length=32, salt=bytes(32), info=info).derive(ikm)
    assert local_tokens._hkdf_sha256(ikm=ikm, info=info) == expected


def test_hkdf_is_pinned_to_a_known_vector():
    """A behaviour pin, computed once against the library, so a refactor cannot
    quietly change every token and signing key this process would derive."""
    vector = local_tokens._hkdf_sha256(ikm=bytes(range(32)), info=b"hkdf-vector-pin-v1")
    assert vector.hex() == "a3725c18dd61b4baae26deed78f8d24563eecd259d7cf8f3b0842b1ba3dac6b1"


def test_token_and_signing_key_info_labels_are_distinct():
    """One seed, two purposes: the same IKM under each label must diverge."""
    assert local_tokens._hkdf_sha256(ikm=b"seed", info=local_tokens._TOKEN_INFO) != (
        local_tokens._hkdf_sha256(ikm=b"seed", info=local_tokens._SIGNING_INFO)
    )


# --------------------------------------------------------------- token format


def test_token_status_classifies_valid_stale_and_invalid():
    keys = _keys(generation=7)
    assert keys.token_status(keys.access_token) == "valid"
    # A generation mismatch is "stale" whatever direction it points: the
    # contract needs inequality after a restart, not ordering (a fabricated
    # future number is still a mismatch, and capped logging bounds the noise).
    assert keys.token_status("6.anything") == "stale"
    assert keys.token_status("99.x") == "stale"
    for malformed in MALFORMED_TOKENS:
        assert keys.token_status(malformed) == "invalid", malformed
    assert keys.token_status("7." + "z" * 43) == "invalid"  # right generation, wrong secret


def test_malformed_tokens_never_raise_and_never_depend_on_the_generation():
    """R1-1/S-1: classification is TOTAL and generation-independent for shapes
    no boot could have minted -- a crafted token can neither error a served
    route nor learn the serving generation from a difference between shapes."""
    serving = SurfaceKeys(generation=5, access_token="5.abc_def", signing_key=b"k" * 32)
    other = SurfaceKeys(generation=9, access_token="9.ghi-jkl", signing_key=b"k" * 32)
    for shape in MALFORMED_TOKENS:
        assert serving.token_status(shape) == "invalid", shape
        assert other.token_status(shape) == "invalid", shape


def test_mint_process_keys_is_first_call_wins():
    first = local_tokens.mint_process_keys(generation=3)
    assert local_tokens.mint_process_keys(generation=9) is first
    assert local_tokens.process_keys() is first
    assert first.generation == 3


# ------------------------------------------------------------------- signing


def test_the_signed_bytes_are_exactly_the_documented_string():
    """``f"{route}\\n{path}\\n{exp}"`` under HMAC-SHA256 -- the canonical string
    both ends must build; asserted against an independently composed HMAC."""
    now = 1_700_000_000
    url, exp = local_tokens.sign_url("images", "/tmp/a b/c.png", ttl_s=60, now=now)
    assert exp == now + 60
    query = parse_qs(urlsplit(url).query)
    assert url.startswith("/v1/static/images?")
    assert query["exp"] == [str(exp)]
    keys = local_tokens.process_keys()
    expected = hmac.new(
        keys.signing_key, f"images\n/tmp/a b/c.png\n{exp}".encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert query["sig"] == [expected]


def test_the_signature_binds_route_path_and_expiry():
    """The full tamper set: swapped path and route, tampered exp and sig, expired."""
    ok = local_tokens.sign_url("images", "/tmp/a.png", ttl_s=60, now=1_700_000_000)
    q = parse_qs(urlsplit(ok[0]).query)
    sig, exp_raw = q["sig"][0], q["exp"][0]
    assert _verify(ok[0], now=1_700_000_010)
    assert not _verify(ok[0], route="html", now=1_700_000_010)  # cross-route reuse
    assert not _verify(ok[0], path="/tmp/b.png", now=1_700_000_010)  # the a.html/b.html case
    assert not local_tokens.verify_signed_url(
        route="images",
        path="/tmp/a.png",
        exp_raw=str(int(exp_raw) + 1),
        sig_raw=sig,
        now=1_700_000_010,
    )
    assert not local_tokens.verify_signed_url(
        route="images", path="/tmp/a.png", exp_raw=exp_raw, sig_raw="0" * 64, now=1_700_000_010
    )
    flipped = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    assert not local_tokens.verify_signed_url(
        route="images", path="/tmp/a.png", exp_raw=exp_raw, sig_raw=flipped, now=1_700_000_010
    )
    # Expired: one second past the expiry, the same URL stops verifying.
    assert not _verify(ok[0], now=int(exp_raw) + 1)


@pytest.mark.parametrize(
    "exp_raw,sig_raw",
    [
        (None, None),
        ("123", None),
        (None, "a" * 64),
        ("12x", "a" * 64),
        ("１２３", "a" * 64),  # non-ASCII digits: isdigit() alone would admit these
        ("", "a" * 64),
        ("123", "zz"),
        ("123", "AB" * 32),  # uppercase hex is not the contract's spelling
        ("123", "ab"),
        ("123", "a" * 63),
        ("123", "a" * 65),
    ],
)
def test_malformed_signature_values_are_a_calm_false(exp_raw, sig_raw):
    """Every malformed shape classifies as False -- never a raise, never a 500."""
    assert not local_tokens.verify_signed_url(
        route="images", path="/x.png", exp_raw=exp_raw, sig_raw=sig_raw, now=0
    )


def test_equivalent_encodings_verify_and_different_decoded_bytes_do_not():
    """The signed byte string is the DECODED value; percent-encoding is transport.

    A client spelling a character as ``%2E`` and one sending ``.`` produce the
    same decoded path and therefore verify against the same signature; a
    DIFFERENT decoded path never verifies, however it is written on the wire.
    """
    url, _ = local_tokens.sign_url("images", "/tmp/a.b.png", ttl_s=60, now=1_700_000_000)
    spelled = url.replace("a.b.png", "a%2Eb%2Epng")
    query = parse_qs(urlsplit(spelled).query)
    assert query["path"][0] == "/tmp/a.b.png"  # decoded back to the signed form
    assert _verify(spelled, now=1_700_000_010)
    assert not local_tokens.verify_signed_url(
        route="images",
        path="/tmp/ab.png",
        exp_raw=query["exp"][0],
        sig_raw=query["sig"][0],
        now=1_700_000_010,
    )


def test_a_space_and_a_unicode_path_round_trip():
    path = "/tmp/dir one/ünïcode-x.png"
    url, _ = local_tokens.sign_url("images", path, ttl_s=60, now=1_700_000_000)
    query = parse_qs(urlsplit(url).query)
    assert query["path"][0] == path
    assert _verify(url, now=1_700_000_010)


# --------------------------------------------------------- generation counter


def test_bump_generation_persists_increments_and_is_private(tmp_path: Path):
    target = tmp_path / "serve" / local_tokens.GENERATION_FILENAME
    assert local_tokens.bump_generation(target) == 1
    assert local_tokens.bump_generation(target) == 2
    assert target.read_text().strip() == "2"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_bump_generation_treats_a_torn_file_as_zero(tmp_path: Path):
    """A torn counter must not wedge a boot: it reads as 0 (so: 1)."""
    parent = tmp_path / "gen"
    parent.mkdir()
    target = parent / local_tokens.GENERATION_FILENAME
    target.write_text("not-a-number\n")
    assert local_tokens.bump_generation(target) == 1


# -------------------------------------------------------- Phase A observation


def test_observation_counts_a_valid_token(caplog):
    keys = _keys(generation=4)
    with caplog.at_level(logging.INFO, logger=LOG_NAME):
        local_tokens.observe_credentials(
            route="images", path="/x.png", token=keys.access_token, exp_raw=None, sig_raw=None
        )
    assert local_tokens.stats()["tokens_valid"] == 1
    assert not caplog.records  # not even at INFO, where the stale line used to sit


def test_a_stale_token_logs_once_per_distinct_generation(caplog):
    _keys(generation=4)
    for _ in range(3):
        local_tokens.observe_credentials(
            route="images", path="/x.png", token="3.old", exp_raw=None, sig_raw=None
        )
    local_tokens.observe_credentials(
        route="images", path="/x.png", token="2.older", exp_raw=None, sig_raw=None
    )
    assert local_tokens.stats()["tokens_stale"] == 4
    # NO ``caplog.at_level`` here, and that is the point (S-2/Q1): the line must
    # be visible at the DEFAULT capture level. INFO was invisible on a default
    # daemon's console; WARNING is what the operator's log level shows.
    assert len(caplog.records) == 2  # once per DISTINCT generation, not per request
    assert all(record.levelno == logging.WARNING for record in caplog.records)


def test_the_stale_log_is_capped_against_fabricated_generations(caplog):
    _keys(generation=100)
    for generation in range(1, 40):
        local_tokens.observe_credentials(
            route="images",
            path="/x.png",
            token=f"{generation}.x",
            exp_raw=None,
            sig_raw=None,
        )
    assert len(caplog.records) == local_tokens._STALE_LOG_CAP


def test_an_invalid_token_is_counted_not_logged(caplog):
    _keys(generation=4)
    with caplog.at_level(logging.INFO, logger=LOG_NAME):
        local_tokens.observe_credentials(
            route="images", path="/x.png", token="4.nope", exp_raw=None, sig_raw=None
        )
    assert local_tokens.stats()["tokens_invalid"] == 1
    assert not caplog.records


def test_observation_counts_a_valid_signature_only():
    url, _ = local_tokens.sign_url("images", "/x.png", ttl_s=60)
    query = parse_qs(urlsplit(url).query)
    local_tokens.observe_credentials(
        route="images",
        path="/x.png",
        token=None,
        exp_raw=query["exp"][0],
        sig_raw=query["sig"][0],
    )
    local_tokens.observe_credentials(
        route="images",
        path="/other.png",
        token=None,
        exp_raw=query["exp"][0],
        sig_raw=query["sig"][0],
    )
    assert local_tokens.stats()["signatures_valid"] == 1


# ---------------------------------------------------------------- the record


def test_the_record_carries_the_surface_fields_and_old_records_default():
    """The additive contract at the record layer: new fields round-trip, a
    record written before them parses with empty defaults (never a reap)."""
    from local_operator.server import registry as serve_registry

    record = serve_registry.build_record(
        instance_id="surface-keys",
        announced=("127.0.0.1", 1111),
        desktop_governed=False,
        access_token="5.abc",
        boot_generation=5,
    )
    payload = record.to_json()
    assert payload["access_token"] == "5.abc"
    assert payload["boot_generation"] == 5

    del payload["access_token"]
    del payload["boot_generation"]
    restored = serve_registry.ServeRecord.from_json(payload)
    assert restored.access_token == ""
    assert restored.boot_generation == 0


# --------------------------------------------------------------- live routes


def _loopback_client() -> AsyncClient:
    """The live app reached as a LOOPBACK-ACCEPTED connection.

    Same shape as ``test_static_roots._loopback_client``: ``scope["server"]``
    is what the gate reads, and httpx builds it from the request URL, so a
    ``127.0.0.1`` base URL is loopback without a socket and the default
    ``http://test`` is a DNS name (rooted, fail-closed).
    """
    from local_operator.server.app import app

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:1111")
    client.headers["Host"] = "127.0.0.1:1111"
    return client


@pytest.fixture
def configured(test_app_client, tmp_path):
    """The live app with ``workspace`` in ``static.roots`` (the rooted fallback)."""
    from local_operator.server.app import app

    directory = tmp_path / "workspace"
    directory.mkdir()
    app.state.config_manager.set_config_value("static", {"roots": [str(directory)]})
    # The app's own renderer dials ``http://127.0.0.1:<port>``; httpx's default
    # ``Host: test`` is a DNS name, which the rebinding guard rightly refuses.
    test_app_client.headers["Host"] = "127.0.0.1:1111"
    return directory


def _token_headers() -> dict[str, str]:
    return {local_tokens.ACCESS_TOKEN_HEADER: local_tokens.process_keys().access_token}


def _outside_png(configured: Path) -> Path:
    outside = configured.parent / "outside" / "x.png"
    outside.parent.mkdir(exist_ok=True)
    outside.write_bytes(PNG)
    return outside


def _tamper_one_sig_char(url: str) -> str:
    match = re.search(r"sig=([0-9a-f]{64})", url)
    assert match is not None
    sig = match.group(1)
    flipped = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    return url[: match.start(1)] + flipped + url[match.end(1) :]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential",
    ["absent", "valid", "invalid", "stale", "signed-valid", "signed-tampered"],
)
async def test_phase_a_serves_regardless_of_credential_state(
    test_app_client, configured, credential
):
    """The permissive contract as a matrix: every credential state serves."""
    outside = _outside_png(configured)
    client = _loopback_client()
    try:
        if credential == "signed-valid":
            url, _ = local_tokens.sign_url("images", str(outside))
            response = await client.get(url)
            assert local_tokens.stats()["signatures_valid"] == 1
        elif credential == "signed-tampered":
            url, _ = local_tokens.sign_url("images", str(outside))
            response = await client.get(_tamper_one_sig_char(url))
            assert local_tokens.stats()["signatures_valid"] == 0
        else:
            headers: dict[str, str] = {}
            if credential == "valid":
                headers = _token_headers()
            elif credential == "invalid":
                headers = {local_tokens.ACCESS_TOKEN_HEADER: "999.deadbeef"}
            elif credential == "stale":
                headers = {
                    local_tokens.ACCESS_TOKEN_HEADER: (
                        f"{local_tokens.process_keys().generation + 1}.deadbeef"
                    )
                }
            response = await client.get(
                "/v1/static/images", params={"path": str(outside)}, headers=headers
            )
    finally:
        await client.aclose()
    assert response.status_code == 200, response.text
    assert response.content == PNG


@pytest.mark.asyncio
async def test_a_pre_token_client_is_unaffected(test_app_client, configured):
    """§4.3's "new (Phase A) <-> old" row, named: plain URLs, no token -- served."""
    outside = _outside_png(configured)
    client = _loopback_client()
    try:
        response = await client.get("/v1/static/images", params={"path": str(outside)})
    finally:
        await client.aclose()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_a_stale_token_through_the_route_is_logged_once(test_app_client, configured, caplog):
    outside = _outside_png(configured)
    stale = f"{local_tokens.process_keys().generation + 1}.beef"
    client = _loopback_client()
    response = None
    try:
        for _ in range(3):
            response = await client.get(
                "/v1/static/images",
                params={"path": str(outside)},
                headers={local_tokens.ACCESS_TOKEN_HEADER: stale},
            )
    finally:
        await client.aclose()
    assert response is not None
    assert response.status_code == 200
    logged = [record for record in caplog.records if "generation" in record.getMessage()]
    assert len(logged) == 1
    # Default capture level (no at_level): a WARNING is what a daemon without
    # LOG_LEVEL shows -- the compat signal has to be visible there (S-2/Q1).
    assert logged[0].levelno == logging.WARNING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", [("9" * 5000 + ".x").encode("ascii"), "1.\u00e9".encode("latin-1")]
)
async def test_a_crafted_token_is_served_not_errored(test_app_client, configured, shape):
    """R1-1/S-1 at the route layer: the two shapes that used to 500 now
    classify as absent -- 200, the file, and no exception text anywhere."""
    outside = _outside_png(configured)
    client = _loopback_client()
    try:
        response = await client.get(
            "/v1/static/images",
            params={"path": str(outside)},
            headers={local_tokens.ACCESS_TOKEN_HEADER: shape},
        )
    finally:
        await client.aclose()
    assert response.status_code == 200
    assert response.content == PNG
    assert b"Exceeds the limit" not in response.content
    assert b"non-ASCII" not in response.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", [("9" * 5000 + ".x").encode("ascii"), "1.\u00e9".encode("latin-1")]
)
async def test_the_sign_door_classifies_crafted_tokens_to_the_stable_401(
    test_app_client, configured, shape
):
    """The sign door's contract for a malformed token is the SAME single
    refusal as every other credential failure -- never a 500 (R1-1/S-1)."""
    response = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png"},
        headers={local_tokens.ACCESS_TOKEN_HEADER: shape},
    )
    assert response.status_code == 401
    assert (
        response.json()["detail"]["text"] == "A local access token or desktop bearer is required."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", [("9" * 5000 + ".x").encode("ascii"), "1.\u00e9".encode("latin-1")]
)
async def test_the_sign_door_crafted_tokens_do_not_pre_empt_a_valid_bearer(
    test_app_client, configured, monkeypatch, shape
):
    """R1-1/S-1: the door is token OR bearer -- a malformed token riding along
    must not stop a valid bearer from opening it (it used to raise first)."""
    bearer = "bearer-for-crafted-token-row"
    monkeypatch.setenv(desktop.TOKEN_ENV, bearer)
    response = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png"},
        headers={
            local_tokens.ACCESS_TOKEN_HEADER: shape,
            "Authorization": f"Bearer {bearer}",
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_a_minted_url_round_trips_through_the_route(test_app_client, configured):
    """POST /v1/static/sign -> GET the URL -> the file, end to end."""
    outside = _outside_png(configured)
    signed = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": str(outside)},
        headers=_token_headers(),
    )
    assert signed.status_code == 200, signed.text
    payload = signed.json()
    assert payload["url"].startswith("/v1/static/images?")
    assert payload["exp"] > time.time()
    client = _loopback_client()
    try:
        served = await client.get(payload["url"])
    finally:
        await client.aclose()
    assert served.status_code == 200
    assert served.content == PNG
    assert local_tokens.stats()["signatures_valid"] == 1


# ------------------------------------------------------- the sign endpoint door


@pytest.mark.asyncio
async def test_sign_refuses_without_any_credential(test_app_client, configured):
    response = await test_app_client.post(
        "/v1/static/sign", json={"route": "images", "path": "/tmp/x.png"}
    )
    assert response.status_code == 401
    assert (
        response.json()["detail"]["text"] == "A local access token or desktop bearer is required."
    )


@pytest.mark.asyncio
async def test_sign_refuses_an_invalid_token(test_app_client, configured):
    response = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png"},
        headers={local_tokens.ACCESS_TOKEN_HEADER: "404.zz"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_sign_accepts_the_local_token(test_app_client, configured):
    response = await test_app_client.post(
        "/v1/static/sign", json={"route": "images", "path": "/tmp/x.png"}, headers=_token_headers()
    )
    assert response.status_code == 200
    assert response.json()["url"].startswith("/v1/static/images?")


@pytest.mark.asyncio
async def test_sign_accepts_the_desktop_bearer(test_app_client, configured, monkeypatch):
    monkeypatch.setenv(desktop.TOKEN_ENV, "desk-secret")
    response = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png"},
        headers={"Authorization": "Bearer desk-secret"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_sign_normalises_every_desktop_refusal_to_one_401(
    test_app_client, configured, monkeypatch
):
    """Wrong bearer, foreign origin and closed plane are one stable refusal --
    the endpoint must not become a way to tell them apart."""
    closed = await test_app_client.post(
        "/v1/static/sign", json={"route": "images", "path": "/tmp/x.png"}
    )
    assert closed.status_code == 401

    monkeypatch.setenv(desktop.TOKEN_ENV, "desk-secret")
    wrong_bearer = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png"},
        headers={"Authorization": "Bearer not-the-bearer"},
    )
    foreign_origin = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png"},
        headers={"Authorization": "Bearer desk-secret", "Origin": "https://evil.example"},
    )
    assert wrong_bearer.status_code == foreign_origin.status_code == 401
    assert (
        wrong_bearer.json()["detail"] == foreign_origin.json()["detail"] == closed.json()["detail"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"route": "wat", "path": "/x.png"},
        {"route": "images", "path": ""},
        {"route": "images", "path": "/x.png", "ttl_s": 0},
        {"route": "images", "path": "/x.png", "ttl_s": 3601},
    ],
)
async def test_sign_refuses_malformed_requests(test_app_client, configured, body):
    response = await test_app_client.post("/v1/static/sign", json=body, headers=_token_headers())
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_sign_accepts_the_ttl_cap_exactly(test_app_client, configured):
    response = await test_app_client.post(
        "/v1/static/sign",
        json={"route": "images", "path": "/tmp/x.png", "ttl_s": local_tokens.TTL_MAX_S},
        headers=_token_headers(),
    )
    assert response.status_code == 200
    assert response.json()["exp"] > time.time() + local_tokens.TTL_MAX_S - 5


# ------------------------------------------------------------- the blast radius


@pytest.mark.asyncio
async def test_non_static_routes_do_not_read_any_of_this(test_app_client, configured):
    """A token is inert outside the static family and the sign endpoint."""
    with_token = await test_app_client.get(
        "/health", headers={local_tokens.ACCESS_TOKEN_HEADER: "9.zz"}
    )
    plain = await test_app_client.get("/health")
    assert with_token.status_code == plain.status_code == 200
