"""Failure taxonomy for ``web_fetch``: what went wrong, and what to do about it.

One classifier, so the engine's retry decision, the model-facing text and the
tests cannot drift into three different opinions about the same response. It is
deliberately PURE — no I/O, no settings, no httpx client — which is what makes
the marker table cheap to test against real captured bodies.

Three questions it answers, in the order the engine asks them:

1. **What class of failure is this?** (:func:`classify_exception` for a
   transport-level throw, :func:`classify_response` for a status code.) The
   class is what decides whether another attempt could possibly help.
2. **Is this a bot-protection refusal, and whose?** The marker tables below are
   header-first because headers are the stable signal; body strings are the
   fallback for vendors that do not brand their headers.
3. **What does the agent read?** (:func:`describe`.) Building the terminal text
   from the CLASSIFICATION rather than from ``str(exc)`` is what fixes the
   defect where ``httpx.ReadTimeout.__str__()`` is the empty string and the
   error message ended in ``': '`` with nothing after it.

Why the honesty distinction in §2.2 of ``docs/design/web_fetch_robustness.md``
is load-bearing: an UNMATCHED 403 earns the same escalation attempt (a plain
WAF with no branding is common and the retry is cheap) but is NEVER labelled
with a vendor. Telling the agent "bot protection" about a 403 that really means
"you are not a subscriber" sends it to ``browser`` when it should be asking the
user for credentials.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Literal, Mapping, Sequence

import httpx

#: The failure classes. ``ok`` is not modelled — a 2xx/3xx returns ``None`` from
#: :func:`classify_response`, so "no failure" is the absence of one of these
#: rather than a value that has to be checked for everywhere.
FailureKind = Literal["transport", "stall", "server", "ratelimit", "blocked", "client"]

#: Vendors we can name from a signature. Anything unmatched stays ``None`` and
#: is described without a vendor claim (§2.2) — the anti-mislabelling rule.
BlockVendor = Literal["cloudflare", "akamai", "datadome", "perimeterx", "imperva"]

#: Markers are matched against the first 8 KB of the decoded body only. A
#: challenge page is always small; scanning a 5 MB body for substrings is wasted
#: work, and a marker appearing 3 MB into a real article is a false positive we
#: do not want.
MARKER_SCAN_BYTES = 8 * 1024

#: Statuses that can carry a challenge. A 403 is the common refusal; 401 appears
#: on some WAFs; 503 is Cloudflare's "under attack" interstitial. Anything else
#: is classified by its own class, never as a block.
_BLOCKABLE_STATUSES = frozenset({401, 403, 503})

#: Header signatures, vendor by vendor. Each entry is ``(header, substring)``
#: matched case-insensitively against the response headers, which are the STABLE
#: signal — a vendor renames body copy far more often than it renames a header.
#:
#: Every one of these was captured live on 2026-09-12 through this package's own
#: pinned request path, not copied from memory:
#:
#: - ``cf-mitigated: challenge`` verbatim on ``https://medium.com/`` (403, 5,507
#:   bytes). Cloudflare sets it SPECIFICALLY so a non-browser client can tell a
#:   challenge apart from a real 403, which makes it the most stable marker we
#:   have — and the reason it is checked before any body string.
#: - ``server: AkamaiGHost`` verbatim on
#:   ``https://www.shoppersdrugmart.ca/?lang=en&query=power+bar`` (403, 382
#:   bytes) and on ``canadiantire.ca``.
#: - ``x-datadome`` / ``datadome=`` cookie: DataDome's own documented markers.
#: - ``x-iinfo``: Imperva/Incapsula's request-trace header.
_HEADER_MARKERS: tuple[tuple[BlockVendor, str, str], ...] = (
    ("cloudflare", "cf-mitigated", "challenge"),
    ("akamai", "server", "akamaighost"),
    ("datadome", "x-datadome", ""),
    ("datadome", "x-dd-b", ""),
    ("datadome", "set-cookie", "datadome="),
    ("perimeterx", "set-cookie", "_px"),
    ("imperva", "server", "imperva"),
    ("imperva", "x-iinfo", ""),
)

#: Body signatures, used only when no header matched. Lower confidence by
#: construction, which is why they come second and why each is a string a
#: challenge page carries and an ordinary page does not.
_BODY_MARKERS: tuple[tuple[BlockVendor, str], ...] = (
    ("cloudflare", "<title>just a moment..."),
    ("cloudflare", "attention required! | cloudflare"),
    ("cloudflare", "challenges.cloudflare.com"),
    ("akamai", "errors.edgesuite.net"),
    ("datadome", "captcha-delivery.com"),
    ("perimeterx", "px-captcha"),
    ("perimeterx", "perimeterx.net"),
    ("perimeterx", "_pxhd"),
)

#: The Akamai stock page: ``<TITLE>Access Denied</TITLE>`` plus ``<H1>Access
#: Denied</H1>``. Checked as a PAIR, and only on a small body, because "access
#: denied" alone is ordinary prose on a real page (§8 risk 6).
_ACCESS_DENIED_TITLE = re.compile(r"<title>\s*access denied\s*</title>", re.I)
_ACCESS_DENIED_HEADING = re.compile(r"<h1>\s*access denied\s*</h1>", re.I)

#: A generic WAF stock page is short. 2 KB is the ceiling below which "access
#: denied" with no other content is a refusal rather than an article about one.
_STOCK_PAGE_MAX_BYTES = 2048

#: Akamai's ``Reference #18.47182117.1789248907.62349a41`` — the piece a human
#: support agent asks for, and the one durable fact on an otherwise useless
#: page. Akamai HTML-escapes the punctuation (``Reference&#32;&#35;18&#46;…``),
#: so the body is unescaped before this runs or the id never matches.
_REFERENCE_RE = re.compile(r"reference\s*#\s*([0-9a-z.\-]+)", re.I)
#: The edgesuite URL carries the same id and appears even when the ``Reference``
#: line is worded differently; used as the fallback.
_EDGESUITE_RE = re.compile(r"errors\.edgesuite\.net/([0-9a-z.\-]+)", re.I)


@dataclass(frozen=True)
class FetchFailure:
    """One classified failure. Frozen because it is shared across the retry loop,
    the render step and the tool text, and none of them may edit it.

    ``retryable`` means "another attempt WITH THE SAME REQUEST IDENTITY could
    plausibly succeed" — it is not the same question as "is an escalation worth
    it" (:attr:`escalatable`). A ``blocked`` response is emphatically not
    retryable: the origin has made a decision about this client and repeating
    the request unchanged just spends the user's turn.
    """

    kind: FailureKind
    retryable: bool
    vendor: BlockVendor | None = None
    reference: str | None = None
    detail: str = ""
    status: int | None = None
    retry_after_s: float | None = None

    @property
    def escalatable(self) -> bool:
        """Whether a browser-shaped retry (§4) is the right next attempt.

        ``blocked`` is the obvious case, and the one the measured win rests on:
        the origin answered FAST (a 403), so almost the whole call budget is
        still unspent when the probe runs, and on medium.com that probe gets 200
        and 53 KB of the real page.

        ``stall`` is subtler now, and worth stating plainly. A read timeout
        consumes the WHOLE slice it was granted, so a genuine black-hole spends
        the call's budget on the first, honest attempt and there is nothing left
        to fund a probe — which is the intended behaviour, not a gap: the probe
        would have to be paid for out of a budget the origin already burned, and
        capping the first attempt to reserve it is exactly the change that was
        reverted (see the note at the top of ``service.py``). The escalation is
        therefore OPPORTUNISTIC: it runs whenever a stall left budget behind,
        which happens when the stall was truncated by a later hop's short
        remainder rather than by the full call timeout.
        """
        return self.kind in ("blocked", "stall")


def classify_exception(exc: BaseException, *, timeout_s: float | None = None) -> FetchFailure:
    """Classify a transport-level throw into a failure class.

    ``timeout_s`` is the budget the attempt was given, carried into the detail
    text so the message can say *how long* it waited. That number is the caller's
    to know: httpx does not put it on the exception, and
    ``httpx.ReadTimeout.__str__()`` is the EMPTY STRING, which is precisely how
    the shipped error message came to read ``timed out fetching '<url>': `` with
    nothing after the colon.
    """
    waited = f" after {timeout_s:.1f}s" if timeout_s else ""
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout)):
        return FetchFailure(
            kind="stall",
            retryable=False,  # an escalation, not a repeat of the same request
            detail=(
                f"read timed out{waited} — the origin accepted the connection but "
                "never sent a response"
            ),
        )
    if isinstance(exc, httpx.ConnectTimeout):
        return FetchFailure(
            kind="transport",
            retryable=True,
            detail=(
                f"could not open a connection{waited} — the origin never completed " "the handshake"
            ),
        )
    if isinstance(exc, httpx.PoolTimeout):
        return FetchFailure(
            kind="transport",
            retryable=True,
            detail=f"timed out waiting for a free connection{waited}",
        )
    if isinstance(exc, httpx.ConnectError):
        return FetchFailure(
            kind="transport",
            retryable=True,
            detail="the connection was refused or the host could not be reached",
        )
    if isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return FetchFailure(
            kind="transport",
            retryable=True,
            detail="the connection dropped mid-transfer",
        )
    if isinstance(exc, httpx.TimeoutException):
        # A timeout subclass we have not enumerated: treat it as a stall, which
        # is the conservative reading (one escalation, no blind repeat).
        return FetchFailure(kind="stall", retryable=False, detail=f"the request timed out{waited}")
    # Any other httpx.HTTPError. Retryable because the alternatives — a DNS or
    # SSRF refusal — never reach here: those raise FetchError before a request
    # is made.
    return FetchFailure(
        kind="transport",
        retryable=True,
        detail=f"the transport failed ({type(exc).__name__})",
    )


def _header_value(headers: Mapping[str, str], name: str) -> str:
    """Case-insensitive header read. Headers arrive lower-cased from the engine,
    but this module is also called directly from tests with whatever a capture
    happened to carry."""
    if name in headers:
        return headers[name]
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def detect_vendor(headers: Mapping[str, str], body_head: str) -> BlockVendor | None:
    """Name the anti-bot vendor behind a refusal, or ``None`` when unsigned.

    Headers first, body second — see the module docstring. ``None`` is a real
    answer here, not a failure to classify: it is what keeps an unbranded 403
    from being described as bot protection (§2.2).
    """
    for vendor, header, needle in _HEADER_MARKERS:
        value = _header_value(headers, header)
        if not value:
            continue
        if not needle or needle in value.lower():
            return vendor
    lowered = body_head.lower()
    for vendor, needle in _BODY_MARKERS:
        if needle in lowered:
            return vendor
    return None


def _is_stock_block_page(body_head: str) -> bool:
    """The long tail of WAF stock pages: a tiny body that is an ``Access
    Denied`` title AND heading and essentially nothing else.

    Both halves are required, and so is the size bound: a real article that
    quotes "Access Denied" in its text must not be classified as a block, and
    an article is never 2 KB of nothing but that phrase twice.

    The bound is compared against the ENCODED length because ``body_head`` is
    decoded text, and a character count over a str is not the "2 KB" the
    constant claims: the same 2 048 characters can be up to ~8 KB of UTF-8 on the
    wire. The unit in the name is the unit the comparison must use, or a reader
    tunes this against the wrong quantity.
    """
    if len(body_head.encode("utf-8", errors="replace")) > _STOCK_PAGE_MAX_BYTES:
        return False
    return bool(_ACCESS_DENIED_TITLE.search(body_head)) and bool(
        _ACCESS_DENIED_HEADING.search(body_head)
    )


def extract_reference(body_head: str, headers: Mapping[str, str]) -> str | None:
    """The origin's own reference for this refusal, or ``None``.

    This is the single piece of a block page worth keeping: ``Reference
    #18.47182117.…`` is what a human quotes to the site's support desk, and
    Cloudflare's ``cf-ray`` plays the same role. Everything else on the page is
    markup whose purpose is to be executed by a browser.

    The body is HTML-unescaped first because Akamai emits the id as
    ``Reference&#32;&#35;18&#46;44182117&#46;…`` — verified on a live capture,
    and the reason a naive regex over the raw bytes finds nothing.
    """
    unescaped = html.unescape(body_head)
    match = _REFERENCE_RE.search(unescaped) or _EDGESUITE_RE.search(unescaped)
    if match:
        return match.group(1).strip(".")
    ray = _header_value(headers, "cf-ray")
    return ray.strip() or None


def parse_retry_after(value: str) -> float | None:
    """``Retry-After`` as seconds, or ``None`` when absent/unparseable.

    Both wire forms are accepted: delta-seconds (the common one) and the
    HTTP-date form, which CDNs do emit. A zero or negative value returns
    ``None`` so the caller falls back to its normal backoff rather than
    treating "retry immediately" as a hostile instruction to hammer the origin.
    """
    raw = value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        import datetime as _dt

        now = _dt.datetime.now(_dt.timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        seconds = (when - now).total_seconds()
    return seconds if seconds > 0 else None


def classify_response(
    status: int, headers: Mapping[str, str], body_head: str
) -> FetchFailure | None:
    """Classify a RESPONSE, or return ``None`` for a 2xx/3xx.

    ``body_head`` must already be bounded to :data:`MARKER_SCAN_BYTES`; this
    function does not re-bound it, so passing a whole 5 MB body would defeat the
    scan window the false-positive guard rests on.
    """
    if status < 400:
        return None

    retry_after = parse_retry_after(_header_value(headers, "retry-after"))

    if status in _BLOCKABLE_STATUSES:
        vendor = detect_vendor(headers, body_head)
        if vendor is not None or (status == 403 and _is_stock_block_page(body_head)):
            return FetchFailure(
                kind="blocked",
                retryable=False,
                vendor=vendor,
                reference=extract_reference(body_head, headers),
                status=status,
            )
        if status == 403:
            # An UNMATCHED 403 still earns the escalation (a plain WAF with no
            # branding is common and the extra request is cheap), but carries no
            # vendor — so `describe` cannot claim bot protection it did not see.
            return FetchFailure(kind="blocked", retryable=False, status=status)

    if status == 429:
        return FetchFailure(
            kind="ratelimit", retryable=True, status=status, retry_after_s=retry_after
        )
    if status in (500, 502, 503, 504):
        return FetchFailure(kind="server", retryable=True, status=status, retry_after_s=retry_after)
    if 500 <= status < 600:
        return FetchFailure(kind="server", retryable=True, status=status, retry_after_s=retry_after)
    # Every other 4xx: a 404 will not become a 200, and a 451 will not become
    # legal. Retrying spends the user's turn for nothing.
    return FetchFailure(kind="client", retryable=False, status=status)


#: The escalation sentence. Named as a NEXT STEP rather than performed:
#: ``web_fetch`` is read-tier and ``browser`` is write-tier, so driving the
#: real browser from inside a fetch would launder an approval-gated action
#: through an auto-approved call (design §8.1). The agent takes this step
#: itself, through the tool's own approval prompt.
#:
#: A single unwrapped sentence on purpose. It used to be hard-wrapped to 76
#: cells "so a 150-column sentence would not lose its own point off the right
#: edge" — but the wrap was the thing losing it: at 80 columns the 76-cell line
#: was clipped mid-word by the card's own painter, and at 150 the block stopped
#: half-way across. The card now fits these lines to its real width at paint
#: time (``tool_card.py::_append_fetch_body``), which is right at every width.
_BROWSER_NEXT_STEP = (
    "Next step: use the `browser` tool on this URL. It drives the real browser "
    "(a write-tier, approval-gated action), which is the only path that clears "
    "an interactive challenge."
)

#: Vendor label for prose. Kept separate from the enum so the enum stays a
#: machine key (it rides in ``details["block_vendor"]``) and the prose can be
#: capitalised without the two drifting.
_VENDOR_LABELS: dict[str, str] = {
    "cloudflare": "Cloudflare",
    "akamai": "Akamai",
    "datadome": "DataDome",
    "perimeterx": "PerimeterX/HUMAN",
    "imperva": "Imperva",
}


def vendor_label(vendor: str | None) -> str | None:
    """The prose form of a vendor key (``akamai`` → ``Akamai``), or ``None``.

    ``None`` in, ``None`` out, on purpose: the caller's "no signature" case must
    stay distinguishable rather than degrading to a label that reads like a
    detection (§2.2).
    """
    if not vendor:
        return None
    return _VENDOR_LABELS.get(vendor, vendor)


def block_lead(vendor: str | None) -> str:
    """The one line that states WHY a refusal happened, shared by every surface.

    This is the single source for a sentence that otherwise exists three times —
    the tool preview's warning lead (``tool.py``), the TUI card's danger row, and
    the tests that assert both. It took a review round to notice that the copy had
    already drifted between two of those copies, which is the whole argument for
    not hand-rolling it a third time.

    Two shapes, and the difference is the whole point (§2.2): a SIGNED refusal
    names the vendor and says "bot protection"; an unsigned one says only that
    the origin refused, and explicitly raises the possibility that this is an
    access restriction. The unsigned form never claims a bot wall — claiming one
    on a 403 that really means "you are not a subscriber" would send the agent to
    ``browser`` when it should be asking the user for credentials.

    Takes the VENDOR rather than a :class:`FetchFailure` because both callers
    hold the vendor (from ``details["block_vendor"]``) and not the classification
    — the tool and the card rebuild their text from the persisted ``details``
    shape, which is what lets a stored transcript render a year later.
    """
    label = vendor_label(vendor)
    if label:
        return f"blocked by {label} bot protection, not page content"
    return (
        "the origin refused this request, which may be bot protection or an " "access restriction"
    )


#: The §3.4 sentence for a ``Retry-After`` the call would not sleep on. Named
#: because BOTH surfaces need it and they disagree about nothing: the model-facing
#: text for a response-bearing 429 is built by ``tool.py::_header_line`` (never by
#: :func:`describe`, which needs a failure with no response), so a single shared
#: sentence is the only way the number reaches the agent in both shapes.
#:
#: `describe` says the same thing in its own words for the terminal case; this is
#: the response-bearing one.
def retry_after_note(retry_after_s: float) -> str:
    """``The origin asked us to wait 600s before retrying; …``

    The number is the one the call already returned in ``details``. Stating it in
    the text is what makes §3.4's "do not sleep, tell the agent" useful rather
    than a silent one-attempt stop.
    """
    return (
        f"The origin asked us to wait {retry_after_s:.0f}s before retrying; the "
        "wait was not spent inside this call."
    )


def describe(
    failure: FetchFailure,
    *,
    attempts: int = 1,
    profiles: Sequence[str] = (),
    url: str | None = None,
) -> str:
    """The model-facing explanation for ``failure``.

    The ONE place these sentences live, so the tool preview, the CLI and the
    card cannot drift. It is built entirely from the classification — never from
    ``str(exc)``, which for ``httpx.ReadTimeout`` is the empty string — so the
    text can never be empty and never ends in a dangling colon.

    ``profiles`` is the identity each attempt used, in order, so the reader can
    tell "we asked three times and it kept failing" from "we asked twice, the
    second time wearing a browser's headers, and it still refused" — which are
    different facts with different next steps.

    **The prose is deliberately NOT pre-wrapped.** It used to be hand-wrapped to
    76 cells, which is a width that is wrong at 80 columns (the sentence was cut
    mid-word) and wrong at 150 (the block stopped half-way across the card) — and
    a constant cannot be right at both. The TUI card re-wraps these lines to the
    card's own width at paint time (``tool_card.py::_append_fetch_body``), so the
    text is a paragraph here and a fitted block on screen. One consequence worth
    naming: the model-facing preview now carries long lines rather than 76-cell
    ones, which costs no tokens and loses no characters.
    """
    lines: list[str] = []
    used_browser = "browser" in tuple(profiles)

    if failure.kind == "blocked":
        if failure.vendor is not None:
            first = "The origin's bot protection refused this request."
            if used_browser:
                first += (
                    " A browser-shaped retry was also refused, so no headless "
                    "fetch of this URL will succeed."
                )
        else:
            # No signature: state the refusal and the ambiguity, and let the
            # agent decide between `browser` and asking the user for access.
            first = (
                "The origin refused this request. No anti-bot vendor signature was "
                "found, so this may be an access restriction (login, region, or "
                "policy) rather than bot protection."
            )
            if used_browser:
                first += " A browser-shaped retry was refused the same way."
        lines.append(first)
        if failure.reference:
            lines.append(f"Origin reference: {failure.reference}")
        lines.append(_BROWSER_NEXT_STEP)
    elif failure.kind == "stall":
        lines.append(_with_attempts(failure.detail or "the request timed out", attempts, profiles))
        if used_browser:
            lines.append(
                "A browser-shaped retry was tried as well, since a stalled request "
                "is frequently a silent block."
            )
            lines.append(_BROWSER_NEXT_STEP)
    elif failure.kind == "transport":
        lines.append(_with_attempts(failure.detail or "the transport failed", attempts, profiles))
    elif failure.kind == "ratelimit":
        text = "the origin is rate-limiting this client (HTTP 429)"
        if failure.retry_after_s:
            text += f" and asked us to wait {failure.retry_after_s:.0f}s"
        lines.append(_with_attempts(text, attempts, profiles))
        if failure.retry_after_s:
            lines.append(retry_after_note(failure.retry_after_s))
    elif failure.kind == "server":
        status = failure.status or 500
        lines.append(_with_attempts(f"the origin returned HTTP {status}", attempts, profiles))
    else:
        status = failure.status or 400
        lines.append(f"the origin returned HTTP {status}")

    if url:
        lines.append(url)
    # Sentence case on the lead, uniformly. These sentences were written to be
    # CHAINED after the tool's own error prefix, so they opened lowercase — but a
    # card paints this text as its own body, where the first line is read with
    # nothing in front of it and a lowercase opening reads as a fragment (design
    # review round 1, DN3). Doing it once here rather than per-branch keeps every
    # terminal class in one voice.
    if lines and lines[0][:1].islower():
        lines[0] = lines[0][:1].upper() + lines[0][1:]
    return "\n".join(lines)


def _with_attempts(text: str, attempts: int, profiles: Sequence[str]) -> str:
    """Append the attempt count and the identities used, when there was more
    than one. A single attempt says nothing — the count only carries
    information once it is not 1."""
    if attempts <= 1:
        return text
    names = ", ".join(_profile_label(p) for p in _collapsed(profiles)) if profiles else ""
    suffix = f" ({attempts} attempts: {names})" if names else f" ({attempts} attempts)"
    return text + suffix


def _profile_label(profile: str) -> str:
    return "browser-profile" if profile == "browser" else profile


def _collapsed(profiles: Sequence[str]) -> list[str]:
    """Consecutive duplicate identities collapsed to one.

    ``2 attempts (default, default)`` is accurate and reads like a bug: the
    sequence exists to show a CHANGE of identity, and repeating the only identity
    there was adds nothing to it. ``(default, browser-profile)`` — the shape that
    matters — is untouched, because those two entries differ.
    """
    out: list[str] = []
    for profile in profiles:
        if not out or out[-1] != profile:
            out.append(profile)
    return out


def attempt_summary(attempts: int, profiles: Sequence[str]) -> str:
    """``2 attempts (default, browser-profile)`` for the header's meta line, or
    ``""`` when a single attempt makes the count uninformative."""
    if attempts <= 1:
        return ""
    if profiles:
        names = ", ".join(_profile_label(p) for p in _collapsed(profiles))
        return f"{attempts} attempts ({names})"
    return f"{attempts} attempts"
