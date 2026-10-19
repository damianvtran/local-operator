"""What a lost guest leaves behind: a precise transport verdict and a console capture.

WHY THIS EXISTS. In an OSWorld-2.1 EC2 episode (task 043, guest ``:5000``) the
guest's control server stopped listening while EC2's status checks, CPU, network
and disk stayed healthy. The harness then spent three batches x ~84 s and three
model turns on instant ``ECONNREFUSED`` answers, and the mechanism INSIDE the
guest is unrecoverable because the instance was terminated before anything
diagnostic was saved (two such losses in ~19 guest-hours). This module closes
the two generic gaps that made that loss opaque. Neither is OSWorld-specific,
and neither changes what an episode measures.

1. ``classify_transport_failure`` -- WAS THE BATCH SENT?

   ``AwsProvider.execute`` used to flatten every transport failure into "outcome
   is unknown". That is the right answer for a failure AFTER the request was
   written (a read timeout, a dropped or reset connection, a 5xx): the guest may
   already have typed the text, and replaying a click/type batch is not
   idempotent, so those stay fail-closed. It is the WRONG answer for a failure
   to ESTABLISH the connection: if the TCP handshake never completed, no byte of
   the request left this host, so the batch certainly did not apply. The two are
   provably different from the exception chain ``requests`` hands back, and
   saying which one happened turns "the guest went away" from a guess into a
   fact on the wedge record.

   The verdict classifies ONE request, not the whole batch: earlier statements
   may already have applied. ``AwsProvider.execute`` may retry only that unsent
   request, bounded to three attempts after one successful readiness probe.
   A failed readiness probe ends recovery immediately, since a dead control
   server cannot accept input. Anything not PROVEN connect-phase remains
   ``ambiguous`` -- there is deliberately no third "probably not sent", and
   even a successful readiness probe cannot make ambiguous input replayable.

2. ``capture_guest_console`` -- READ-ONLY EVIDENCE BEFORE THE GUEST IS GONE.

   ``ec2:GetConsoleOutput`` and ``ec2:GetConsoleScreenshot`` need no guest
   channel (the control server is the thing that is dead), cost a few API calls
   and tens of KB, and work for any benchmark guest. The screenshot shows
   whether the desktop was at a login screen or a crashed session; the console
   tail shows what the kernel and init were still saying. Both are persisted
   beside ``guest-preparation.json`` in the episode's cache root.

   Constraints, each load-bearing:

   * NEVER RAISES, NEVER BLOCKS TEARDOWN. The caller is already on a failure
     path; evidence is strictly best-effort. Each AWS call runs in a daemon
     thread under a shared deadline, so a hung API call is abandoned rather than
     awaited, and every error becomes a recorded status.
   * DEGRADES ON A MISSING PERMISSION. The documented scoped key predates these
     calls; an ``UnauthorizedOperation`` is recorded as ``denied`` and the
     episode proceeds exactly as before.
   * NO SECRETS ON DISK. Console text can carry cloud-init/user-data output and
     anything a service printed. It is scrubbed with the harness's shared
     credential-shape table PLUS the provider's own known values, BEFORE it is
     bounded (clipping first would sever a secret across the cut and defeat the
     match -- the same ordering ``worker._redacted`` enforces), and then
     re-checked for the known values. If the scrubber is unavailable (an older
     harness without ``local_operator.redaction_shapes``) or the re-check
     fails, the text is WITHHELD and only its status and size are recorded. A
     console screenshot is a picture of the guest desktop, the same class of
     content as the observation frames the episode already persists.
   * BOUNDED. Console text keeps its TAIL (the end is the freshest) up to
     ``MAX_CONSOLE_TEXT_CHARS``; a screenshot above ``MAX_SCREENSHOT_BYTES`` is
     recorded by size only.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Transport classification
# ---------------------------------------------------------------------------

#: The TCP connection was provably never established: no byte of the request was
#: sent, so this request did NOT apply (earlier batch statements may have).
DELIVERY_NOT_SENT = "not_sent"
#: Anything else. The request may have been written, and the guest may have
#: acted on it. Fail-closed: never replayed.
DELIVERY_AMBIGUOUS = "ambiguous"

# Bounds the walk over an exception graph. Real chains are 3-5 links deep; the
# bound only protects against a cyclic or adversarially long ``__context__``.
_MAX_CHAIN = 32


@dataclass(frozen=True)
class TransportVerdict:
    """The classification of one failed guest request.

    ``kind`` is a closed vocabulary (``refused`` / ``timeout`` / ``dns`` /
    ``connect_failed``) for a ``not_sent`` verdict and empty for ``ambiguous``.
    It never carries exception text: transport errors echo the URL and can echo
    the command, which is why ``execute`` has never exposed them.
    """

    delivery: str
    kind: str = ""


def _walk(error: BaseException) -> list[BaseException]:
    """Every exception reachable from ``error`` through the links ``requests`` uses.

    ``__cause__`` and ``__context__`` are the Python links; urllib3's
    ``MaxRetryError`` keeps the real fault in ``.reason`` and ``requests``'
    ``ConnectionError`` keeps the urllib3 error in ``args[0]``. None of the
    three is guaranteed to be mirrored in another, so all are followed.
    """

    seen: set[int] = set()
    found: list[BaseException] = []
    pending: list[BaseException | None] = [error]
    while pending and len(found) < _MAX_CHAIN:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        pending.append(current.__cause__)
        pending.append(current.__context__)
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            pending.append(reason)
        for argument in getattr(current, "args", ()):
            if isinstance(argument, BaseException):
                pending.append(argument)
    return found


def _targets_only(chain: list[BaseException], origin: str) -> bool:
    """True when at least one link's request targets ``origin`` and none targets elsewhere.

    Compared on parsed scheme and ``host:port``, not as a string prefix, so
    ``http://h:5000`` does not match ``http://h:50001``.
    """

    from urllib.parse import urlsplit

    expected = urlsplit(origin)
    seen = False
    for link in chain:
        url = getattr(getattr(link, "request", None), "url", None)
        if not isinstance(url, str):
            continue
        target = urlsplit(url)
        if (target.scheme, target.netloc) != (expected.scheme, expected.netloc):
            return False
        seen = True
    return seen


def classify_transport_failure(error: BaseException, *, origin: str | None) -> TransportVerdict:
    """Decide whether a failed guest request was certainly never sent.

    ``not_sent`` requires POSITIVE evidence that the connect phase failed (a
    urllib3 ``NewConnectionError`` -- which also covers a refused port, an
    unreachable host and a name-resolution failure -- or a connect timeout) and
    NO evidence of anything past the handshake. Everything else is
    ``ambiguous``: a read timeout, ``RemoteDisconnected``, a reset or aborted
    connection, a chunked-read failure, an HTTP error status, a bare builtin
    ``ConnectionError`` with no urllib3 chain. A bare ``ConnectionRefusedError``
    is deliberately NOT enough on its own -- an OS can report it on a ``send``
    to a connection that had already been established.

    REDIRECTS ARE PROOF OF DELIVERY, NOT OF ITS ABSENCE. ``requests`` follows a
    3xx by default, and when the redirect TARGET refuses the connection the
    exception that escapes carries only that final attempt: the original request
    -- which the first server received and may have acted on -- is invisible in
    the chain. A real loopback server that read the full POST and answered 307 to
    a closed port produced exactly ``not_sent/refused`` here (review round 1,
    C1). Two independent guards, because neither alone is sufficient:

    * the guest transport no longer follows redirects (``aws.build_clients``), so
      this shape cannot arise from production code; and
    * this function refuses ``not_sent`` when ANY link carries a response (the
      server answered, so the request was delivered), when any failing request
      targets a different scheme/host/port than ``origin`` (an earlier hop may
      have been delivered), and -- fail closed -- when NO link shows the failing
      request targeting ``origin`` at all. Every exception ``requests`` raises
      for a connect failure carries its ``PreparedRequest``, so a genuine
      refusal always supplies that evidence; an exception that does not is not
      one this function can vouch for. ``origin`` is ``scheme://host:port`` of
      the guest and is REQUIRED so a new caller has to decide; None (before
      allocate) fails closed as well.

    This relies on the guest transport opening a FRESH connection per request
    (``requests.post`` at module level: no pool, no keep-alive). A pooled
    connection that went stale would fail on its first WRITE, which is
    post-handshake and could not be told apart from a dropped request; if the
    transport ever pools, this classifier must be re-derived.
    """

    try:
        import requests.exceptions as req_exc  # type: ignore[import-not-found]
        import urllib3.exceptions as u3_exc  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - requests is a declared dependency
        return TransportVerdict(DELIVERY_AMBIGUOUS)

    try:
        chain = _walk(error)
        import http.client as http_client

        post_send = (
            req_exc.ReadTimeout,
            req_exc.ChunkedEncodingError,
            req_exc.ContentDecodingError,
            req_exc.HTTPError,
            u3_exc.ReadTimeoutError,
            u3_exc.ProtocolError,
            u3_exc.IncompleteRead,
            http_client.HTTPException,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
        )
        # ``NewConnectionError`` is a ``ConnectTimeoutError`` subclass in
        # urllib3 2.x, so one isinstance covers both; named separately to say so.
        connect = (
            u3_exc.NewConnectionError,
            u3_exc.ConnectTimeoutError,
            req_exc.ConnectTimeout,
        )
        if any(isinstance(link, post_send) for link in chain):
            return TransportVerdict(DELIVERY_AMBIGUOUS)
        if not any(isinstance(link, connect) for link in chain):
            return TransportVerdict(DELIVERY_AMBIGUOUS)
        if any(
            getattr(link, "response", None) is not None or getattr(link, "history", None)
            for link in chain
        ):
            return TransportVerdict(DELIVERY_AMBIGUOUS)
        if origin is None or not _targets_only(chain, origin):
            return TransportVerdict(DELIVERY_AMBIGUOUS)

        name_resolution = getattr(u3_exc, "NameResolutionError", None)
        if any(
            (name_resolution is not None and isinstance(link, name_resolution))
            or isinstance(link, socket.gaierror)
            for link in chain
        ):
            return TransportVerdict(DELIVERY_NOT_SENT, "dns")
        if any(isinstance(link, ConnectionRefusedError) for link in chain):
            return TransportVerdict(DELIVERY_NOT_SENT, "refused")
        if any(
            isinstance(link, req_exc.ConnectTimeout)
            or type(link) is u3_exc.ConnectTimeoutError
            or isinstance(link, TimeoutError)
            for link in chain
        ):
            return TransportVerdict(DELIVERY_NOT_SENT, "timeout")
        return TransportVerdict(DELIVERY_NOT_SENT, "connect_failed")
    except Exception:  # pragma: no cover - a classifier that raises must fail closed
        return TransportVerdict(DELIVERY_AMBIGUOUS)


_KIND_PHRASE = {
    "refused": "connection refused",
    "timeout": "connect timed out",
    "dns": "host name did not resolve",
    "connect_failed": "connection could not be established",
}


def describe_failed_action(index: int, verdict: TransportVerdict) -> str:
    """The one-line, value-free message ``execute`` raises for a transport failure.

    ``index`` is the 0-based position of the failing statement in its batch. The
    ``delivery=`` token is machine-greppable and sits in the HEAD of the line,
    because the harness keeps only the head of a wire message (512 characters
    through ``rpc.MAX_DETAIL_MESSAGE``, 500 on the wedge record).

    For a ``not_sent`` verdict the failing statement did not apply, but this is
    one statement of a batch: any EARLIER statement (``index > 0``) was already
    answered with success and DID apply, so the batch as a whole is partial and
    the sentence says so rather than implying a clean no-op.
    """

    number = index + 1
    if verdict.delivery != DELIVERY_NOT_SENT:
        return (
            f"guest action {number} outcome is unknown; batch not retried; "
            f"delivery={DELIVERY_AMBIGUOUS}"
        )
    phrase = _KIND_PHRASE.get(verdict.kind, _KIND_PHRASE["connect_failed"])
    if index == 0:
        effect = "no input of this batch was applied"
    else:
        effect = f"the {index} earlier action(s) of this batch applied, input may be partial"
    return (
        f"guest action {number} was not sent ({phrase}; {effect}); batch not retried; "
        f"delivery={DELIVERY_NOT_SENT}"
    )


# ---------------------------------------------------------------------------
# 2. Console capture
# ---------------------------------------------------------------------------

#: One artifact pair PER STAGE, so the two moments a lost guest can first be
#: noticed never overwrite each other. ``observe``: the first frameless
#: screenshot read, which is when the guest was FIRST seen failing (the 043 run
#: noticed ~4 minutes before the batch that finally ended it). ``execute``: the
#: transport failure that ends the batch and wedges the bridge. Comparing the two
#: shows whether the guest was still changing between them.
STAGE_OBSERVE = "observe"
STAGE_EXECUTE = "execute"
_STAGES = (STAGE_OBSERVE, STAGE_EXECUTE)
CONSOLE_SCHEMA = 1


def console_filename(stage: str) -> str:
    return f"guest-console-{stage}.json"


def screenshot_filename(stage: str) -> str:
    return f"guest-console-{stage}-screenshot.jpg"


#: Console text kept (the TAIL). EC2 itself returns at most the last 64 KiB, so
#: this is a guard against a changed service limit, not a trim of normal output.
MAX_CONSOLE_TEXT_CHARS = 64 * 1024
#: Largest decoded console screenshot persisted. A 1280x720 JPEG is far below it.
MAX_SCREENSHOT_BYTES = 1024 * 1024
#: Wall-clock allowance for BOTH calls together. The caller is ending an episode;
#: evidence must not become the reason teardown is late.
CAPTURE_BUDGET_S = 10.0

# Statuses a capture can record, per artifact. Closed so a reader can switch on it.
STATUS_CAPTURED = "captured"
STATUS_DENIED = "denied"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_EMPTY = "empty"
STATUS_OVERSIZE = "oversize"
STATUS_WITHHELD = "withheld"
STATUS_NO_DESTINATION = "no_destination"

_DENIED_CODES = frozenset(
    {
        "UnauthorizedOperation",
        "AccessDenied",
        "AccessDeniedException",
        "AuthFailure",
        "OperationNotPermitted",
    }
)


def _error_code(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


def _failure_status(error: BaseException) -> dict[str, Any]:
    """A recorded failure: a closed status plus the AWS error CODE, never its text.

    The code (``UnauthorizedOperation``) is what an operator needs to fix an IAM
    policy; the message can embed an ARN or an account id and adds nothing.
    """

    code = _error_code(error)
    if code in _DENIED_CODES:
        return {"status": STATUS_DENIED, "error_code": code}
    return {
        "status": STATUS_ERROR,
        "error_code": code or type(error).__name__,
    }


#: Console text longer than this is withheld outright rather than scanned. EC2
#: returns at most 64 KiB, so this is a guard against a changed service limit
#: that keeps the blob pass's cost bounded by construction.
_MAX_SCAN_CHARS = 1024 * 1024

#: A base64-looking run, possibly WRAPPED over several lines (MIME/`base64`
#: default is 76 columns, and a secret can straddle the break). An initial
#: >= 8 alphabet characters followed by wrapped alphabet lines, including a short
#: FINAL line: dropping that short tail can leave a decodable credential fragment.
_B64_RUN = re.compile(
    r"[A-Za-z0-9+/_\-]{8,}(?:=*[ \t]*\r?\n[ \t]*" r"[A-Za-z0-9+/_\-]+(?==*[ \t]*(?:\r?\n|$)))*=*"
)
#: A contiguous hex run. Hex is byte-aligned, so a known value's own hex spelling
#: is already caught by the shared exact-value pass; this catches mixed-case or
#: otherwise unregistered hex-encoded blobs, which cannot be vouched for.
_HEX_RUN = re.compile(r"\b[0-9a-fA-F]{32,}\b")
#: A base64 run at least this long is opaque and withheld even when nothing
#: recognisable decodes out of it. Conservative by design: an occasional long
#: identifier or path can be withheld too. Keeping an unvouched encoded value
#: is worse than losing that identifier; unit names/status sentences survive.
_OPAQUE_B64_MIN = 40
#: Shape rules only run on decodes of runs at least this long: shorter runs are
#: ordinary words, and the shape table needs more context than that to fire.
_SHAPE_CHECK_MIN = 24


#: Four encoded bytes already form an opaque object, not an isolated URL-space
#: or one Unicode symbol. Fully percent-encoded containing objects evade a
#: standalone-value spelling list because quote() leaves alphanumerics literal
#: (QA round 2); JSON's all-character unicode escapes have the same neighbour
#: hazard. Withhold the ENTIRE console text when this spelling occurs instead
#: of decoding layers or leaving a potentially reversible fragment at a clip.
#: The scan strips whitespace first so wrapping inside a %HH/\uHHHH token does
#: not evade it. This intentionally sacrifices ordinary encoded URLs/blobs; it
#: does not claim to recognise every possible encoding or encryption scheme.
_OPAQUE_ESCAPES = re.compile(r"(?:%[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}){4,}")


def _decoded_views(run: str) -> list[bytes]:
    """Every byte-alignment decode of a base64-looking ``run``.

    A secret inside a LARGER encoded object is spelled differently depending on
    where it falls in the 3-byte groups, so standalone ``base64(secret)`` (what
    the shared exact-value pass matches) misses it (review round 1, C2).
    Decoding from char offsets 0-3 reaches all three byte alignments; both the
    standard and URL-safe alphabets are tried by normalising to standard.
    """

    stripped = re.sub(r"[ \t\r\n=]", "", run).replace("-", "+").replace("_", "/")
    views: list[bytes] = []
    for offset in range(4):
        chunk = stripped[offset:]
        # PAD the final partial group instead of dropping it: truncating to a
        # multiple of 4 clips the tail of a secret that ends the blob. One
        # leftover character (len % 4 == 1) carries no whole byte and is dropped.
        if len(chunk) % 4 == 1:
            chunk = chunk[:-1]
        chunk += "=" * (-len(chunk) % 4)
        if len(chunk) <= 4:
            continue
        try:
            views.append(base64.b64decode(chunk, validate=False))
        except (binascii.Error, ValueError):
            continue
    return views


def _withhold_encoded_blobs(text: str, known_values: tuple[str, ...]) -> str | None:
    """``text`` with encoded blobs that hold, or cannot be vouched not to hold, a secret removed.

    The shared scrubber matches a known value's own spellings, which is blind to
    the value inside a larger encoding. Rather than extend a spelling list that
    can never be complete, this refuses to persist what it cannot read:

    * an opaque percent/JSON-unicode escaped object withholds the entire text,
      including whitespace-wrapped spellings, rather than retaining bytes a
      later reader can trivially decode;
    * a base64 run whose decode, at any alignment, contains a known value or
      trips the credential-shape table is replaced by a marker;
    * a base64 run of >= ``_OPAQUE_B64_MIN`` characters, and a hex run of >=
      32 digits, is replaced by a marker whatever it decodes to --
      an encoded blob of that size in a console is user-data or a key, and the
      diagnostic value of its bytes is nil next to the cost of persisting a
      credential.

    Only a payload that survives these checks reaches the shared scrub. Returns
    None (withhold the whole text) when it contains opaque escape encodings or
    is too large to scan.
    A secret under a layer this does not decode (gzip, a second base64 level, a
    cipher) is NOT covered; the long-run rule is what bounds that residual.
    """

    if len(text) > _MAX_SCAN_CHARS:
        return None
    if _OPAQUE_ESCAPES.search("".join(text.split())):
        return None
    try:
        from local_operator.redaction_shapes import scrub_secrets
    except ImportError:
        return None
    needles = [value.encode("utf-8") for value in known_values if value]
    # A base64 run carrying an N-byte secret has at least ceil(4N/3) characters.
    shortest = min((len(n) for n in needles), default=0)
    known_min = max(8, -(-4 * shortest // 3)) if shortest else 0

    def marker(match: re.Match[str]) -> str:
        return f"[encoded blob withheld: {len(match.group(0))} chars]"

    def replace_b64(match: re.Match[str]) -> str:
        run = match.group(0)
        compact = re.sub(r"[ \t\r\n=]", "", run)
        if (
            len(compact) >= _OPAQUE_B64_MIN
            and any(c.islower() for c in compact)
            and any(c.isupper() for c in compact)
        ):
            return marker(match)
        if known_min and len(compact) >= known_min:
            for view in _decoded_views(run):
                if any(needle in view for needle in needles):
                    return marker(match)
                if len(compact) >= _SHAPE_CHECK_MIN:
                    decoded = view.decode("latin-1")
                    if scrub_secrets(decoded) != decoded:
                        return marker(match)
        return run

    text = _B64_RUN.sub(replace_b64, text)
    return _HEX_RUN.sub(marker, text)


def _scrub(text: str, known_values: tuple[str, ...]) -> str | None:
    """``text`` with credentials removed, or None when it cannot be vouched for.

    None means WITHHOLD: the shared scrubber is not importable (an older
    harness), the text is too large to scan, or a known value survived it.
    Persisting unscrubbed console text would write whatever the guest printed --
    possibly a credential -- to a durable directory, which is strictly worse than
    recording its absence.
    """

    try:
        from local_operator.redaction_shapes import scrub_secrets
    except ImportError:
        return None
    try:
        blobless = _withhold_encoded_blobs(text, known_values)
        if blobless is None:
            return None
        scrubbed = scrub_secrets(blobless, known_values)
    except Exception:
        return None
    # The scrub is exact-value plus shape; a belt-and-braces re-check turns a
    # future scrubber regression into a withheld artifact instead of a leak.
    if any(value and value in scrubbed for value in known_values):
        return None
    return scrubbed


def _run_bounded(
    calls: dict[str, Callable[[], Any]], budget_s: float
) -> dict[str, tuple[str, Any]]:
    """Run each call in its own daemon thread under ONE shared deadline.

    Returns ``{name: ("ok", value) | ("error", exception) | ("timeout", None)}``.
    A call still running at the deadline is ABANDONED (daemon thread), not
    awaited: botocore's own read timeout is a minute, and the caller is on a
    teardown path. The two calls run concurrently so a slow console does not
    spend the screenshot's share of the budget.
    """

    results: dict[str, tuple[str, Any]] = {}
    threads: dict[str, threading.Thread] = {}

    def worker(name: str, call: Callable[[], Any]) -> None:
        try:
            results[name] = ("ok", call())
        except Exception as error:
            results[name] = ("error", error)

    for name, call in calls.items():
        thread = threading.Thread(
            target=worker, args=(name, call), name=f"guest-console-{name}", daemon=True
        )
        threads[name] = thread
        thread.start()
    deadline = time.monotonic() + budget_s
    for thread in threads.values():
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return {name: results.get(name, ("timeout", None)) for name in calls}


def _console_output_record(
    outcome: tuple[str, Any], known_values: tuple[str, ...]
) -> dict[str, Any]:
    state, value = outcome
    if state == "timeout":
        return {"status": STATUS_TIMEOUT}
    if state == "error":
        return _failure_status(value)
    text = value.get("Output") if isinstance(value, dict) else None
    if not isinstance(text, str) or not text.strip():
        return {"status": STATUS_EMPTY}
    scrubbed = _scrub(text, known_values)
    if scrubbed is None:
        return {"status": STATUS_WITHHELD, "original_chars": len(text)}
    kept = scrubbed[-MAX_CONSOLE_TEXT_CHARS:]
    return {
        "status": STATUS_CAPTURED,
        "original_chars": len(text),
        "truncated": len(kept) < len(scrubbed),
        "text": kept,
    }


def _screenshot_record(outcome: tuple[str, Any], stage: str) -> tuple[dict[str, Any], bytes | None]:
    state, value = outcome
    if state == "timeout":
        return {"status": STATUS_TIMEOUT}, None
    if state == "error":
        return _failure_status(value), None
    encoded = value.get("ImageData") if isinstance(value, dict) else None
    if not isinstance(encoded, str) or not encoded:
        return {"status": STATUS_EMPTY}, None
    # Bound BEFORE decoding: base64 inflates by 4/3, so an over-limit payload is
    # refused without allocating its decoded form.
    if len(encoded) > (MAX_SCREENSHOT_BYTES * 4) // 3 + 4:
        return {"status": STATUS_OVERSIZE, "encoded_chars": len(encoded)}, None
    try:
        image = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return {"status": STATUS_ERROR, "error_code": "undecodable"}, None
    if len(image) > MAX_SCREENSHOT_BYTES:
        return {"status": STATUS_OVERSIZE, "bytes": len(image)}, None
    return {
        "status": STATUS_CAPTURED,
        "bytes": len(image),
        "file": screenshot_filename(stage),
    }, image


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a sibling temp file so a reader never sees a half-written artifact."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def capture_guest_console(
    ec2: Any,
    instance_id: str,
    destination: Path | None,
    *,
    stage: str,
    trigger: str,
    known_values: Iterable[str] = (),
    budget_s: float | None = None,
) -> dict[str, Any]:
    """Persist the guest's console tail and screenshot, once, best-effort.

    Returns the record that was written (also useful to a test). ``stage`` picks
    the artifact pair (``STAGE_OBSERVE`` / ``STAGE_EXECUTE``); ``trigger`` names why
    the capture ran (``observe-frameless``, ``execute-not_sent``,
    ``execute-ambiguous``) so a reader can tell what the caller had just seen.
    ``known_values`` are the provider's own secret values (credentials, the guest
    client password) for the exact-value half of the scrub.

    NEVER RAISES: any failure, including one in this function's own bookkeeping,
    is returned as ``{"status": "error", ...}``. The only exceptions allowed past
    it are ``KeyboardInterrupt``/``SystemExit``, which are not ``Exception``.
    """

    try:
        values = tuple(value for value in known_values if isinstance(value, str) and value)
        record: dict[str, Any] = {
            "schema": CONSOLE_SCHEMA,
            "instance_id": instance_id,
            "stage": stage,
            "trigger": trigger,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if stage not in _STAGES:
            record["status"] = STATUS_ERROR
            record["error_code"] = "unknown_stage"
            return record
        if destination is None:
            record["status"] = STATUS_NO_DESTINATION
            return record

        outcomes = _run_bounded(
            {
                # ``Latest=True`` asks for the most recent output instead of
                # the cached boot-time copy EC2 would otherwise serve.
                "console_output": lambda: ec2.get_console_output(
                    InstanceId=instance_id, Latest=True
                ),
                # WakeUp is left at its default (False): it sends a keystroke
                # to the guest, and this capture must stay strictly read-only.
                "console_screenshot": lambda: ec2.get_console_screenshot(InstanceId=instance_id),
            },
            CAPTURE_BUDGET_S if budget_s is None else budget_s,
        )
        record["console_output"] = _console_output_record(outcomes["console_output"], values)
        screenshot, image = _screenshot_record(outcomes["console_screenshot"], stage)
        record["console_screenshot"] = screenshot
        record["status"] = STATUS_CAPTURED

        destination.mkdir(parents=True, exist_ok=True)
        if image is not None:
            _write_atomic(destination / screenshot_filename(stage), image)
        _write_atomic(
            destination / console_filename(stage),
            json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
        )
        return record
    except Exception as error:
        logger.warning("guest console capture failed (%s)", type(error).__name__)
        return {"status": STATUS_ERROR, "error_code": type(error).__name__}
