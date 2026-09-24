"""Check a pasted API key with ONE cheap authenticated request before storing it.

Why this exists: the desktop's key-save route stored whatever it was handed and
reported "Signed in", so a mistyped or revoked key looked connected until the
user's first message came back as a raw provider 401 -- the worst moment to learn
it, and one that reads as the app being broken rather than the key being wrong.
Asking the provider "is this key accepted?" at save time costs one GET, and turns
that failure into an inline error beside the field the user is still looking at.

The verdict is three-valued, and the third value is the important design point:

* ``True``  -- the provider accepted the key (a 2xx from an authenticated route).
* ``False`` -- the provider DEFINITIVELY rejected it: HTTP 401, a 403 whose body
  talks about the key or authentication, or the 400 some providers use for a
  malformed key (xAI "Incorrect API key provided", Google "API key not valid").
  Measured 2026-09-24 against every key-accepting provider in the registry with a
  fabricated key; each answered one of those.
* ``None``  -- we could not tell: a timeout, no network, a 5xx, a rate limit, or a
  route that does not exist for this account. Saving must NOT be blocked on this:
  an offline laptop or a provider outage is not evidence the key is wrong, and
  refusing a good key is worse than storing an unchecked one.

Security constraints, which the tests pin: the key is sent only to the provider's
own registry-declared host, in a header (never a query string, which proxies and
server logs record), and nothing derived from the response BODY reaches the
caller. Several providers echo the submitted key back in their 401 body --
DeepSeek quotes it in full -- so the body is read only to classify, never
returned or logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from local_operator.providers.registry import (
    ProviderDefinition,
    credential_provider_id,
    get_provider_definition,
)

logger = logging.getLogger(__name__)

#: The whole check's budget. Long enough for a cold TLS handshake to a distant
#: region, short enough that a save the user is watching never feels hung; past
#: it the verdict is ``None`` and the key is saved anyway.
KEY_CHECK_TIMEOUT_S = 8.0

#: Anthropic's API version header, the same value discovery sends.
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class KeyCheck:
    """The verdict and a sentence for the user; ``reason`` is ``None`` when valid."""

    valid: bool | None
    reason: str | None


def _label(definition: ProviderDefinition) -> str:
    # "OpenAI (ChatGPT Plus/Pro)" names a login route; a sentence about a key
    # wants the company ("OpenAI rejected this API key").
    return definition.name.split(" (", 1)[0]


def _base_url(definition: ProviderDefinition) -> str | None:
    if definition.id == "radient":
        # The one provider whose host is configurable (staging vs production).
        # Checking a key against a host it was not issued for returns a
        # DEFINITIVE 401, which would refuse a good key -- so ask the same
        # resolver every other Radient surface asks.
        from local_operator.env import resolve_radient_api_base_url

        return resolve_radient_api_base_url()
    return definition.base_url


def _request(definition: ProviderDefinition, key: str) -> tuple[str, dict[str, str]] | None:
    """The URL and headers of the cheapest authenticated request, or ``None``."""
    base = _base_url(definition)
    if not base:
        return None
    base = base.rstrip("/")
    if definition.wire == "anthropic":
        url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        return url, {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
    if definition.wire == "google":
        # The header form, not ``?key=``: a query string is what access logs keep.
        return f"{base}/v1beta/models?pageSize=1", {"x-goog-api-key": key}
    if definition.id == "openrouter":
        # OpenRouter's /models is PUBLIC -- it answers 200 to any key or none
        # (measured), so it cannot tell a good key from a bad one. /key is the
        # authenticated read of the key's own metadata.
        return f"{base}/key", {"Authorization": f"Bearer {key}"}
    return f"{base}/models", {"Authorization": f"Bearer {key}"}


def _mentions_key(body: str) -> bool:
    lowered = body.casefold()
    return "api key" in lowered or "api_key" in lowered or "apikey" in lowered


#: Wording that says the KEY is fine but lacks the permission to read the route we
#: probe. OpenAI answers a valid restricted key ("Model capabilities: Request"
#: only, Models permission off) with 401 "You have insufficient permissions for
#: this operation. Missing scopes: api.model.read" -- a key that chats fine. A
#: status code alone cannot tell it from a revoked key, so the body decides, and
#: this check runs BEFORE the 401 rule: refusing a good key is the worse error.
_PERMISSION_MARKERS = ("insufficient permissions", "missing scope", "insufficient_scope")

#: What a 403 must say to count as a verdict on the key itself. Deliberately not
#: the bare ``auth`` stem: "not authorized to access this model" and "unauthorized
#: region" are entitlement/region blocks, which say nothing about the key.
_AUTHENTICATION_MARKERS = ("authentication", "unauthenticated", "invalid api key", "credential")

#: A key is printable ASCII: every provider issues base64/hex-ish tokens, and an
#: HTTP header cannot carry anything else (httpx raises ``UnicodeEncodeError``).
#: What actually arrives here outside that range is paste debris -- a zero-width
#: space, a typographic ellipsis from a truncated display -- so it is named as a
#: re-copy problem before any request is attempted.
_KEY_CHARACTERS = frozenset(chr(code) for code in range(0x21, 0x7F))


def has_invalid_characters(key: str) -> bool:
    """Whether ``key`` holds anything but printable, non-space ASCII."""
    return any(character not in _KEY_CHARACTERS for character in key)


#: The sentence for :func:`has_invalid_characters`; shared so every refusal of a
#: pasted key says the same thing.
INVALID_CHARACTERS_REASON = (
    "This key contains characters an API key cannot have. Re-copy it and try again."
)


def classify(definition: ProviderDefinition, status: int, body: str) -> KeyCheck:
    """The verdict for one response. Split out so it is testable without a socket."""
    label = _label(definition)
    rejected = KeyCheck(False, f"{label} rejected this API key. Check it and try again.")
    if 200 <= status < 300:
        return KeyCheck(True, None)
    lowered = body.casefold()
    if status in (401, 403) and any(marker in lowered for marker in _PERMISSION_MARKERS):
        # Authenticated, just not allowed to list models: a verdict of "unknown".
        return _unverified(label, status)
    if status == 401:
        return rejected
    if status == 403 and (
        _mentions_key(body) or any(marker in lowered for marker in _AUTHENTICATION_MARKERS)
    ):
        return rejected
    if status == 400 and _mentions_key(body):
        return rejected
    # Anything else -- a region block, a rate limit, an outage, a route this
    # account cannot read -- says nothing definite about the key.
    return _unverified(label, status)


def _unverified(label: str, status: int) -> KeyCheck:
    return KeyCheck(
        None,
        f"{label} could not check this key right now (HTTP {status}). "
        "It was saved without being verified.",
    )


async def check_api_key(
    provider_id: str,
    key: str,
    *,
    timeout: float = KEY_CHECK_TIMEOUT_S,
    client: httpx.AsyncClient | None = None,
) -> KeyCheck:
    """Ask ``provider_id`` whether it accepts ``key``. Never raises.

    ``client`` is injectable for tests (an ``httpx.MockTransport``); production
    builds one per call, which is fine for a user-initiated save.
    """
    if has_invalid_characters(key):
        # Before anything else, including the provider lookup: this is a definite
        # "not a key" for every provider, and sending it would raise in httpx.
        return KeyCheck(False, INVALID_CHARACTERS_REASON)
    definition = get_provider_definition(credential_provider_id(provider_id))
    if definition is None or definition.allows_missing_api_key:
        return KeyCheck(None, None)
    request = _request(definition, key)
    if request is None:
        return KeyCheck(None, None)
    url, headers = request
    label = _label(definition)
    owned = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(url, headers=headers, timeout=timeout)
        return classify(definition, response.status_code, response.text[:2048])
    except Exception as error:  # noqa: BLE001 -- "never raises" is the route's contract
        # Broad on purpose: the save route has no guard of its own, and an
        # unexpected failure here (an encoding error, a proxy misconfiguration)
        # must degrade to "saved unverified", not a 500. The exception TYPE only:
        # an error's message can carry the URL or header material, and logging
        # less is free.
        logger.debug("API key check for %s could not complete: %s", definition.id, type(error))
        return KeyCheck(
            None,
            f"Could not reach {label} to check this key. It was saved without being verified.",
        )
    finally:
        if owned:
            await http.aclose()
