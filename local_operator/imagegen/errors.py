"""The image lane's failure mapping — its own copy, deliberately not an STT import.

Why a copy rather than ``stt/errors``'s mapper: that module classifies
TRANSCRIPTION failures (its rung vocabulary is ``AudioPath``, its details are
built for the speech surfaces) and it carries a body-scrubbing contract tuned
to those routes. The image rules diverge the moment either lane moves: this
module owns the LANE-SPECIFIC half — the body-scrubbing error builder shaped
by the image routes' payloads — while the closed reason-class vocabulary and
its classifier now live in the generic artifact layer
(:mod:`local_operator.artifacts.errors`, media wave-2) and are re-exported
here under these same names, because the lane's pinned imports read them from
this path.

The reason classes exist because the tool's error report and the PR/QA
evidence both need to GROUP failures ("every rung refused for credits") without
parsing prose, and prose is the one thing this codebase keeps rewording.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from local_operator.artifacts.errors import REASON_CLASSES, failure_reason_class
from local_operator.clients._http import (
    APIError,
    error_payload,
    scrub_details,
    scrub_secrets,
)

__all__ = ["REASON_CLASSES", "api_error_from_httpx_response", "failure_reason_class"]

#: The closed reason-class vocabulary and its classifier moved to the generic
#: artifact layer in media wave-2; re-exported above so the lane's pinned
#: imports (``from local_operator.imagegen.errors import REASON_CLASSES``)
#: keep resolving here.


def api_error_from_httpx_response(
    response: "Any",
    *,
    fallback_message: str,
    secrets: Iterable[Optional[str]] = (),
) -> APIError:
    """Build an :class:`APIError` from a failed ``httpx`` response.

    The httpx twin of the requests-side builder, and the same shape ``stt``
    uses: designed ``error`` string, status/code/details, mask-on-the-way-in —
    plus the SCRUBBED raw body on :attr:`APIError.body`, because an upstream's
    own sentence is the whole diagnostic for a refusal that carries no
    designed vocabulary, and dropping it would make those refusals read "with
    no body" while the data was in hand.
    """
    raw = response.content.decode(errors="replace")
    message, code, details = error_payload(raw)
    return APIError(
        scrub_secrets(message or f"{fallback_message} (HTTP {response.status_code})", secrets),
        status_code=response.status_code,
        code=code,
        details=scrub_details(details, secrets),
        body=scrub_secrets(raw, secrets) if raw.strip() else None,
    )
