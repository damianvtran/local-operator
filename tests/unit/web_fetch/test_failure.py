"""Classifier tests: what each failure class is detected from, and what it says.

The marker bodies here are REAL captures taken through the live path on
2026-09-12, not invented markup — a test written against a plausible-looking
Cloudflare page proves nothing about the page Cloudflare actually serves. The
Akamai body in particular is kept with its HTML entities intact, because that
encoding is exactly what defeats a naive reference-id regex.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.web_fetch.failure import (
    _is_stock_block_page,
    attempt_summary,
    classify_exception,
    classify_response,
    describe,
    extract_reference,
    parse_retry_after,
)

#: Captured verbatim from ``https://medium.com/`` (403, 5,507 bytes). Trimmed to
#: the markers; the real page is mostly challenge JavaScript.
CLOUDFLARE_BODY = (
    '<!DOCTYPE html><html lang="en-US"><head><title>Just a moment...</title>'
    '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
    '<meta http-equiv="content-security-policy" content="default-src \'none\'; '
    "script-src 'nonce-sxCBDQGvcCXBCxZ9Xdwr5S' https://challenges.cloudflare.com\">"
    '</head><body><div class="main-wrapper" role="main"><noscript>'
    '<span id="challenge-error-text">Enable JavaScript and cookies to continue</span>'
    "</noscript></div></body></html>"
)

#: Captured verbatim from ``https://www.shoppersdrugmart.ca/?lang=en&query=power+bar``
#: (403, 382 bytes). The entities are NOT decorative: Akamai emits the reference
#: id as ``Reference&#32;&#35;18&#46;…`` and a regex over the raw bytes misses it.
AKAMAI_BODY = (
    "<HTML><HEAD>\n<TITLE>Access Denied</TITLE>\n</HEAD><BODY>\n"
    "<H1>Access Denied</H1>\n \n"
    "You don't have permission to access &#34;http&#58;&#47;&#47;www&#46;"
    "shoppersdrugmart&#46;ca&#47;&#63;&#34; on this server.<P>\n"
    "Reference&#32;&#35;18&#46;44182117&#46;1789250166&#46;2be4ae76\n"
    "<P>https&#58;&#47;&#47;errors&#46;edgesuite&#46;net&#47;18&#46;44182117&#46;"
    "1789250166&#46;2be4ae76</P>\n</BODY>\n</HTML>"
)


def test_cloudflare_header_marker_is_authoritative() -> None:
    """``cf-mitigated: challenge`` alone classifies, with no body at all.

    Cloudflare sets it specifically so a non-browser client can distinguish a
    challenge from a real 403, which is why it is checked before any body text.
    """
    failure = classify_response(403, {"cf-mitigated": "challenge", "server": "cloudflare"}, "")
    assert failure is not None
    assert failure.kind == "blocked"
    assert failure.vendor == "cloudflare"
    assert failure.retryable is False


def test_cloudflare_body_marker_without_branded_header() -> None:
    """The body fallback catches a challenge whose headers say nothing."""
    failure = classify_response(403, {"content-type": "text/html"}, CLOUDFLARE_BODY)
    assert failure is not None
    assert failure.vendor == "cloudflare"


def test_akamai_header_and_reference_extraction() -> None:
    """``server: AkamaiGHost`` names the vendor, and the entity-encoded
    ``Reference #`` id survives — that id is what a human quotes to support."""
    headers = {"server": "AkamaiGHost", "content-type": "text/html"}
    failure = classify_response(403, headers, AKAMAI_BODY)
    assert failure is not None
    assert failure.kind == "blocked"
    assert failure.vendor == "akamai"
    assert failure.reference == "18.44182117.1789250166.2be4ae76"
    assert extract_reference(AKAMAI_BODY, headers) == "18.44182117.1789250166.2be4ae76"


def test_datadome_header_marker() -> None:
    failure = classify_response(403, {"x-datadome": "protected"}, "")
    assert failure is not None
    assert failure.vendor == "datadome"


def test_unmatched_403_is_blocked_but_never_labelled_a_bot_wall() -> None:
    """The honesty rule (§2.2): an unsigned 403 earns the escalation but must
    NOT be described as bot protection.

    Claiming "bot protection" on a 403 that really means "you are not a
    subscriber" sends the agent to `browser` when it should be asking the user
    for credentials. The word "bot" is the assertion under test.
    """
    failure = classify_response(403, {"content-type": "text/html"}, "<html>nope</html>")
    assert failure is not None
    assert failure.kind == "blocked"
    assert failure.vendor is None
    assert failure.escalatable is True

    text = describe(failure)
    # The assertion is that no bot wall is CLAIMED — not that the token "bot"
    # is absent. §2.2's own prescribed wording ("No anti-bot vendor signature
    # was found … rather than bot protection") contains the word while denying
    # the claim, and that denial is the more useful sentence: it tells the agent
    # the refusal might be an access restriction. §6.6's literal "must not
    # contain 'bot'" would have forbidden the design's own text.
    # Flattened: `describe` pre-wraps its prose to the TUI card's row width (a
    # card paints one row per line and clips the overflow), so the assertion is
    # about the sentence, not about where it happens to break.
    lowered = " ".join(text.lower().split())
    assert "blocked by" not in lowered
    assert "no anti-bot vendor signature was found" in lowered
    assert "access restriction" in lowered
    # It still names the escalation — an unsigned refusal is exactly the case
    # where the agent has to decide between `browser` and asking the user.
    assert "browser" in text


def test_signed_block_says_bot_protection_and_names_the_vendor() -> None:
    failure = classify_response(403, {"server": "AkamaiGHost"}, AKAMAI_BODY)
    assert failure is not None
    text = " ".join(describe(failure, attempts=2, profiles=("default", "browser")).split())
    assert "bot protection" in text
    assert "18.44182117.1789250166.2be4ae76" in text
    assert "browser" in text
    # It says the escalation was already spent, so the agent does not read
    # "try a different client" into it.
    assert "browser-shaped retry was also refused" in text


def test_404_is_client_and_not_retryable() -> None:
    failure = classify_response(404, {}, "<html>gone</html>")
    assert failure is not None
    assert failure.kind == "client"
    assert failure.retryable is False
    assert failure.escalatable is False


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_statuses_retry(status: int) -> None:
    failure = classify_response(status, {}, "")
    assert failure is not None
    assert failure.kind == "server"
    assert failure.retryable is True


def test_503_with_challenge_marker_is_blocked_not_server() -> None:
    """Cloudflare's "under attack" interstitial is a 503 — and repeating the
    request is not what clears it, so it must classify as a block."""
    failure = classify_response(503, {"cf-mitigated": "challenge"}, CLOUDFLARE_BODY)
    assert failure is not None
    assert failure.kind == "blocked"


def test_429_carries_retry_after() -> None:
    failure = classify_response(429, {"retry-after": "7"}, "")
    assert failure is not None
    assert failure.kind == "ratelimit"
    assert failure.retry_after_s == 7.0


def test_retry_after_http_date_form_is_parsed() -> None:
    """CDNs do emit the date form; a client that only understands seconds
    silently falls back to its own backoff and ignores the origin."""
    import datetime

    when = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=30)
    stamp = when.strftime("%a, %d %b %Y %H:%M:%S GMT")
    parsed = parse_retry_after(stamp)
    assert parsed is not None
    assert 20 <= parsed <= 40


@pytest.mark.parametrize("value", ["", "soon", "-5", "0"])
def test_retry_after_garbage_falls_back_to_none(value: str) -> None:
    assert parse_retry_after(value) is None


def test_read_timeout_is_a_stall_and_connect_error_is_transport() -> None:
    request = httpx.Request("GET", "https://example.com/")
    stall = classify_exception(httpx.ReadTimeout("", request=request), timeout_s=12.0)
    assert stall.kind == "stall"
    assert stall.escalatable is True

    transport = classify_exception(httpx.ConnectError("boom", request=request))
    assert transport.kind == "transport"
    assert transport.retryable is True


def test_empty_exception_string_still_produces_a_legible_message() -> None:
    """The §1.3 regression: ``httpx.ReadTimeout.__str__()`` is the EMPTY STRING,
    which is how the shipped error read ``timed out fetching '<url>': `` with
    nothing after the colon. The text is built from the classification, so it
    names the class and how long it waited whatever the exception says."""
    request = httpx.Request("GET", "https://example.com/")
    exc = httpx.ReadTimeout("", request=request)
    assert str(exc) == ""  # the defect's root cause, asserted so it cannot drift

    failure = classify_exception(exc, timeout_s=20.0)
    text = " ".join(describe(failure, attempts=2, profiles=("default", "browser")).split())
    # Sentence case (design review round 1, DN3): this text is the card body's
    # FIRST line in the terminal case, where a lowercase opening read as a
    # fragment with nothing in front of it.
    assert "Read timed out after 20.0s" in text
    assert "never sent a response" in text
    assert not text.rstrip().endswith(":")


def test_marker_scan_window_ignores_a_late_false_positive() -> None:
    """A 5 MB page containing "access denied" 3 MB in is NOT a block.

    Guards §8 risk 6: the phrase is ordinary prose on a real page, and the 8 KB
    window plus the stock-page size bound are what keep it from being read as a
    refusal.
    """
    body = ("x" * (3 * 1024 * 1024)) + "<h1>Access Denied</h1><title>Access Denied</title>"
    # The engine only ever passes the head, which is the behaviour under test.
    from local_operator.web_fetch.failure import MARKER_SCAN_BYTES

    failure = classify_response(403, {"content-type": "text/html"}, body[:MARKER_SCAN_BYTES])
    assert failure is not None
    assert failure.vendor is None  # unsigned: no marker was in the window


def test_stock_waf_page_needs_both_title_and_heading_and_a_small_body() -> None:
    """An article ABOUT access denials must not classify as a block."""
    article = "<html><title>Access Denied</title>" + ("<p>a real article. " * 400) + "</html>"
    failure = classify_response(403, {"content-type": "text/html"}, article)
    assert failure is not None
    assert failure.vendor is None  # no vendor claimed for a long, unsigned body


def test_the_stock_page_bound_is_bytes_not_characters() -> None:
    """Review round 1, N2: ``_STOCK_PAGE_MAX_BYTES`` claimed 2 KB and compared
    CHARACTERS.

    ``body_head`` is decoded text, so the old ``len(body_head)`` was a character
    count: 700 euro signs are 700 characters and 2 100 bytes of UTF-8, i.e. a
    body a third larger than the bound could still be read as a "tiny stock
    page". The unit in the name has to be the unit the comparison uses, or the
    next person tunes this against the wrong quantity. Exercised on the predicate
    itself: both branches of ``classify_response`` for a 403 yield
    ``vendor=None``, so a test at that level cannot tell the two apart.
    """
    small_stock = "<title>Access Denied</title><h1>Access Denied</h1>"
    assert _is_stock_block_page(small_stock) is True

    big_but_few_chars = small_stock + ("\u20ac" * 700)
    assert len(big_but_few_chars) < 2048  # what a character count sees
    assert len(big_but_few_chars.encode("utf-8")) > 2048  # what the wire carries
    assert _is_stock_block_page(big_but_few_chars) is False


def test_attempt_summary_collapses_a_repeated_identity() -> None:
    """Review round 1, Q3: ``3 attempts (default, default, default)`` is accurate
    and reads like a bug.

    The sequence exists to show a CHANGE of identity — that is the fact a reader
    cannot get from the count — so consecutive repeats collapse while the mixed
    case (the one that matters) is untouched.
    """
    assert attempt_summary(3, ("default", "default", "default")) == "3 attempts (default)"
    assert attempt_summary(2, ("default", "browser")) == "2 attempts (default, browser-profile)"
    assert attempt_summary(2, ("default", "default")) == "2 attempts (default)"
    # One attempt says nothing, and an absent identity list is not a lie.
    assert attempt_summary(1, ("default",)) == ""
    assert attempt_summary(2, ()) == "2 attempts"


def test_2xx_and_3xx_are_not_failures() -> None:
    assert classify_response(200, {}, "") is None
    assert classify_response(302, {"location": "/x"}, "") is None
