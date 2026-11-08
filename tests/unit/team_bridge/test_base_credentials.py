"""A Team Bridge hub base carrying ``user:password@`` is refused before any request.

WHY (security round 2 on #3587, S-r2-1). A non-canonical hub is reachable
through ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE``, and such a base may be spelled
with URL credentials. Two things went wrong with that spelling:

* httpx logs ``HTTP Request: POST <full url>`` at INFO, credentials included,
  and the local server did not pin httpx below INFO, so ``LOG_LEVEL=INFO``
  printed the password.
* httpx turns URL credentials into ``Authorization: Basic``, which REPLACES the
  pinned ``Bearer`` this client sets (measured on httpx 0.28.1), so such a base
  never authenticated as the signed-in account anyway — it sent the URL's
  password to the hub instead.

THE RIG is the real ``TeamBridgeClient`` over ``httpx.MockTransport`` with
httpx itself at INFO, so the request-line logger is live. The marker is built
by concatenation and is deliberately not password-shaped: a log redactor that
masked it would let the absence assertions pass without testing anything, so
every case also asserts that no ``@`` survives.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.radient_credentials import (
    INVALID_HUB_BASE_PLACEHOLDER,
    org_destination_refused_sentence,
)
from local_operator.team_bridge import client as tb_client
from tests.unit.team_bridge.test_client import FakeStore, _ok

_MARKER = "zz" + "secret" + "zz"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # The opt-in a credentialed base needs to be reachable at all: with the
    # destination guard closed the base would be refused for being
    # non-canonical, and the test would not reach the path that leaked.
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.mark.parametrize(
    ("base_url", "bare"),
    [
        (f"https://ops:{_MARKER}@hub.example/v1", "https://hub.example/v1"),
        (f"https://{_MARKER}@hub.example/v1", "https://hub.example/v1"),
        (f"https://ops:a@{_MARKER}@Hub.Example:8443/v1", "https://Hub.Example:8443/v1"),
        (f"http://ops:{_MARKER}@[2001:db8::1]:9000/v1", "http://[2001:db8::1]:9000/v1"),
        # EMPTY userinfo KEEPS the bearer on httpx 0.28.1 (measured, security
        # round 1 on #3588); it is refused because a hub base never carries ``@``.
        ("https://@hub.example/v1", "https://hub.example/v1"),
    ],
)
@pytest.mark.asyncio
async def test_a_credentialed_base_is_refused_and_never_logged(
    caplog: pytest.LogCaptureFixture, base_url: str, bare: str
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    # Every logger, httpx included, at INFO or below: the level at which the
    # request line carried the credential.
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="httpx")
    client = tb_client.TeamBridgeClient(
        None, base_url=base_url, store=FakeStore(), transport=httpx.MockTransport(handler)
    )

    with pytest.raises(tb_client.BridgeRefusal) as refused:
        await client.status()

    assert refused.value.code == "bridge_destination_refused"
    assert seen == []
    assert client.base_url == bare
    surfaced = [str(refused.value), client.base_url]
    surfaced += [record.getMessage() for record in caplog.records]
    for text in surfaced:
        assert _MARKER not in text
        # The bare base never carries ``@``; neither may any record or the refusal.
        assert "@" not in text


@pytest.mark.asyncio
async def test_a_base_without_credentials_still_reaches_the_hub() -> None:
    """The negative control: the refusal keys on userinfo, nothing broader."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    client = tb_client.TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    await client.status()
    assert [str(request.url) for request in seen] == ["https://hub.example/v1/messaging/status"]
    assert seen[0].headers["Authorization"].startswith("Bearer ")


# ---------------------------------------------------------------------------
# Security round 1 on #3588: S-1 (any ``@`` is credentials, sanitized sentence),
# S-2 (the refusal at the dispatch every request passes), S-3 (discipline key).
# ---------------------------------------------------------------------------

#: Passwords spelled with a raw ``/``, ``?`` or ``#``. RFC 3986 ends the
#: authority at that character, so a split that looks for ``@`` only inside the
#: authority missed them (S-1): the base was not flagged, the generic refusal
#: echoed it whole, and the opt-in path raised a raw ``InvalidURL`` naming part
#: of the password.
_DELIMITED_PASSWORD_BASES = [
    "https://ops:pa" + mark + _MARKER + "@hub.example/v1" for mark in ("/", "?", "#")
] + ["https://ops:" + _MARKER + "/x@hub.example:8443/v1"]


def _assert_clean(texts: list[str]) -> None:
    for text in texts:
        assert _MARKER not in text
        assert "@" not in text
        assert "ops" not in text


@pytest.mark.parametrize("base_url", _DELIMITED_PASSWORD_BASES)
@pytest.mark.parametrize("destination_open", [True, False])
@pytest.mark.asyncio
async def test_a_password_with_an_authority_delimiter_is_still_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    base_url: str,
    destination_open: bool,
) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: destination_open)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="httpx")
    client = tb_client.TeamBridgeClient(
        None, base_url=base_url, store=FakeStore(), transport=httpx.MockTransport(handler)
    )

    with pytest.raises(tb_client.BridgeRefusal) as refused:
        await client.status()

    assert refused.value.code == "bridge_destination_refused"
    assert str(refused.value) == tb_client.CREDENTIALED_BASE_REFUSED_SENTENCE
    assert seen == []
    _assert_clean(
        [str(refused.value), repr(refused.value), client.base_url]
        + [record.getMessage() for record in caplog.records]
    )


@pytest.mark.parametrize(
    "base_url",
    _DELIMITED_PASSWORD_BASES
    + [
        "https://ops:" + _MARKER + "@hub.example:8443/v1",
        "  HTTPS://ops:" + _MARKER + "@hub.example:8443/v1?q=1",
    ],
)
def test_the_noncanonical_sentence_quotes_only_a_sanitized_origin(base_url: str) -> None:
    """The renderer is shared with the CLI and server routes, which pass the raw base."""

    sentence = org_destination_refused_sentence(base_url, "https://api.radienthq.com/v1")
    _assert_clean([sentence])
    assert "hub.example" in sentence


def test_the_noncanonical_sentence_keeps_a_plain_origin_readable() -> None:
    sentence = org_destination_refused_sentence(
        "http://127.0.0.1:9/v1", "https://api.radienthq.com/v1"
    )
    assert "(http://127.0.0.1:9)" in sentence


@pytest.mark.asyncio
async def test_a_ready_access_does_not_bypass_the_credentialed_refusal() -> None:
    """S-2: a caller handing ``access=`` skips ``current_access``; the dispatch still refuses."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    client = tb_client.TeamBridgeClient(
        None,
        base_url="https://ops:" + _MARKER + "@hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    access = OAuthAccess(access_token="x", credential_id=1, account_id="acc_1", kind="oauth")

    with pytest.raises(tb_client.BridgeRefusal) as via_status:
        await client.status(access=access)
    with pytest.raises(tb_client.BridgeRefusal) as via_dispatch:
        await client._dispatch("GET", "/messaging/status", "x", params=None, body=None)

    assert via_status.value.code == via_dispatch.value.code == "bridge_destination_refused"
    assert seen == []


def test_the_shared_discipline_is_keyed_on_the_stripped_base() -> None:
    """S-3: the process-wide discipline map must not hold the credential as a key."""

    tb_client.reset_status_disciplines()
    try:
        tb_client.get_status_discipline(None, "https://ops:" + _MARKER + "@hub.example/v1")
        keys = list(tb_client._DISCIPLINES)
        assert keys == [("", "https://hub.example/v1", True)]
        assert all(_MARKER not in str(part) for key in keys for part in key)
    finally:
        tb_client.reset_status_disciplines()


# ---------------------------------------------------------------------------
# Security round 2 on #3588: S-4 (an ``@`` BEFORE the first ``://``) and S-5
# (a credentialed and a clean base must never share one discipline).
# ---------------------------------------------------------------------------

#: An ``@`` in the would-be scheme half. Splitting on the first ``://`` before
#: looking for ``@`` put the credential in the "scheme" and kept it (S-4): a
#: missing-slashes typo with a later URL in the query, and a bare
#: ``<pw>@https://`` prefix.
_PRE_SCHEME_AT_BASES = [
    "https:ops:" + _MARKER + "@hub.example/?r=https://x.example",
    _MARKER + "@https://hub.example/v1",
    # No scheme at all: nothing trustworthy precedes the credentials.
    "ops:" + _MARKER + "@hub.example:8443/v1",
]


@pytest.mark.parametrize("base_url", _PRE_SCHEME_AT_BASES)
@pytest.mark.asyncio
async def test_an_at_before_the_scheme_separator_is_still_stripped(
    caplog: pytest.LogCaptureFixture, base_url: str
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="httpx")
    tb_client.reset_status_disciplines()
    try:
        client = tb_client.TeamBridgeClient(
            None, base_url=base_url, store=FakeStore(), transport=httpx.MockTransport(handler)
        )
        tb_client.get_status_discipline(None, base_url)
        with pytest.raises(tb_client.BridgeRefusal) as refused:
            await client.status()
        keys = list(tb_client._DISCIPLINES)
    finally:
        tb_client.reset_status_disciplines()

    assert refused.value.code == "bridge_destination_refused"
    assert seen == []
    sentence = org_destination_refused_sentence(base_url, "https://api.radienthq.com/v1")
    _assert_clean(
        [str(refused.value), repr(refused.value), client.base_url, sentence]
        + [str(part) for key in keys for part in key]
        + [record.getMessage() for record in caplog.records]
    )


def test_a_scheme_is_kept_only_when_no_at_precedes_it() -> None:
    """The shown value never includes a byte from before any ``@``."""

    assert tb_client._split_base_credentials("https://ops:" + _MARKER + "@hub.example/v1") == (
        "https://hub.example/v1",
        True,
    )
    for base_url in _PRE_SCHEME_AT_BASES:
        stripped, had_credentials = tb_client._split_base_credentials(base_url)
        assert had_credentials is True
        assert stripped == INVALID_HUB_BASE_PLACEHOLDER
        sentence = org_destination_refused_sentence(base_url, "https://api.radienthq.com/v1")
        assert INVALID_HUB_BASE_PLACEHOLDER in sentence


@pytest.mark.parametrize("credentialed_first", [True, False])
def test_a_credentialed_and_a_clean_base_never_share_a_discipline(
    credentialed_first: bool,
) -> None:
    """S-5: the credentials flag is part of the key, so neither ordering merges."""

    credentialed = "https://ops:" + _MARKER + "@hub.example/v1"
    clean = "https://hub.example/v1"
    order = [credentialed, clean] if credentialed_first else [clean, credentialed]
    tb_client.reset_status_disciplines()
    try:
        first, second = (tb_client.get_status_discipline(None, base) for base in order)
        by_base = dict(zip(order, (first, second)))
        assert first is not second
        assert sorted(tb_client._DISCIPLINES) == [
            ("", clean, False),
            ("", clean, True),
        ]
        assert by_base[credentialed]._client._base_had_credentials is True
        assert by_base[clean]._client._base_had_credentials is False
    finally:
        tb_client.reset_status_disciplines()


# ---------------------------------------------------------------------------
# Security round 3 on #3588: S-6 (a compatibility ``@`` is still credentials).
# ---------------------------------------------------------------------------

#: ``@`` spelled as a compatibility character: FULLWIDTH COMMERCIAL AT
#: (U+FF20) and SMALL COMMERCIAL AT (U+FE6B), both NFKC-equal to ``@``. An
#: ASCII-only check passed them through unflagged, so the refusal and the
#: sentence quoted the password; a later NFKC step anywhere downstream (IDNA
#: host mapping does one) would turn either back into a real userinfo split.
_COMPAT_AT_BASES = [
    "https://ops:" + _MARKER + at + "hub.example/v1" for at in ("\uff20", "\ufe6b")
] + [
    # An ASCII ``@`` earlier, the compatibility one after it: the remainder
    # after the last ASCII ``@`` still carries the marker.
    "https://x@ops:"
    + _MARKER
    + "\uff20hub.example/v1",
]


@pytest.mark.parametrize("base_url", _COMPAT_AT_BASES)
@pytest.mark.asyncio
async def test_a_compatibility_at_sign_is_still_credentials(
    caplog: pytest.LogCaptureFixture, base_url: str
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="httpx")
    assert tb_client._split_base_credentials(base_url) == (INVALID_HUB_BASE_PLACEHOLDER, True)
    client = tb_client.TeamBridgeClient(
        None, base_url=base_url, store=FakeStore(), transport=httpx.MockTransport(handler)
    )

    with pytest.raises(tb_client.BridgeRefusal) as refused:
        await client.status()

    assert refused.value.code == "bridge_destination_refused"
    assert seen == []
    sentence = org_destination_refused_sentence(base_url, "https://api.radienthq.com/v1")
    assert INVALID_HUB_BASE_PLACEHOLDER in sentence
    texts = [str(refused.value), repr(refused.value), client.base_url, sentence]
    texts += [record.getMessage() for record in caplog.records]
    _assert_clean(texts)
    for text in texts:
        assert "\uff20" not in text and "\ufe6b" not in text


def test_a_percent_encoded_at_without_a_real_one_is_untouched() -> None:
    """The negative control: ``%40`` is not an ``@`` under NFKC, so nothing is flagged."""

    base_url = "https://hub.example/v1/team%40ops"
    assert tb_client._split_base_credentials(base_url) == (base_url, False)
