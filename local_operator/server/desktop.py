"""Authenticated desktop adapters, separate from the legacy local API.

**Two sources of truth, in this order** (and only :func:`desktop_posture` reads
either, so the ordering cannot drift between callers):

1. The **environment** — ``LOCAL_OPERATOR_DESKTOP_TOKEN``, the capability the
   Electron main process injects into a backend IT started. That is a
   process-lifetime secret passed beside the spawn and is authoritative for the
   whole life of the process; a claim is refused while it is set.
2. A **claim** — ``POST /v1/desktop/claim``, the way a daemon that SOMEBODY
   ELSE started (a TUI, a terminal, launchd) hands the plane to the desktop app.
   The key is minted by :mod:`local_operator.server.registry` at startup and
   published ONLY in that daemon's ``0600`` record under the ``0700`` run
   directory, so reading it already requires the account's own file
   permissions — the same boundary the session registry's ``control_key``
   relies on. It is never returned in a response, logged, or written anywhere
   else, because the record's permissions are the entire authorization story:
   anything that can read the record can already attach to the user's sessions,
   read ``auth.db`` and run ``lop``, so the claim hands that principal no new
   class of secret. A page in a browser cannot read it (it has no filesystem
   access), and it cannot guess 256 bits. A sandboxed renderer never learns it
   either: privileged calls ride ``window.api.desktop.request`` through main,
   so only the token's SOURCE changes for the renderer — nothing about what it
   may reach.

The latch is **one-way**: ``_CLAIMED`` is set once and never cleared while the
process lives. A claim is not a session and not a lease — the desktop app that
claimed the plane keeps using the same bearer for the daemon's whole life
(routes answer 401/403 on capability failures, never by re-opening the plane),
and a second claim would silently invalidate the first app's bearer or, worse,
let a second local caller take over a plane another principal is driving: the
"one app owns this daemon" invariant the daemon-started certificate already
encodes. So a second claim is refused even when it presents the correct key.

**What a claim does to the legacy surface — it TIGHTENS it.** A standalone
daemon historically kept ``allow_origins=["*"]`` with ``allow_credentials=True``
and left the legacy control paths ungated (see ``server/app.py``), because the
boundary it tested was the same environment variable the desktop app sets. A
daemon that a TUI started therefore sat on a predictable loopback port readable
by any page the user visited. Accepting a claim puts this process into the same
managed posture the app imposes when it starts the backend itself, so
``/v1/agents``, ``/v1/jobs``, ``/v1/schedules``, ``/v1/config``,
``/v1/credentials`` and ``/v1/models`` become bearer-gated for every other
local caller, and the wildcard CORS echo is dropped for foreign origins once
the claim has installed an allowlist (a claim that presented no Origin — a
native main-process caller — leaves the historical wildcard-echo behaviour on
NON-control paths, which the gated families above no longer include). The
accepted cost is named in the design (rollout risk 1): a local ``curl`` script
against a claimed daemon starts seeing 401.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import NamedTuple

from fastapi import HTTPException, Request

#: The environment variable the desktop app injects into a backend it spawned.
#: Read HERE and nowhere else — see the module docstring, and the source-grep
#: test that keeps a future caller from growing a private second opinion.
TOKEN_ENV = "LOCAL_OPERATOR_DESKTOP_TOKEN"

#: The comma-separated origins such a backend trusts. Read here and nowhere
#: else, for the same reason. A claim ADDS to whatever this names.
ORIGINS_ENV = "LOCAL_OPERATOR_DESKTOP_ORIGINS"


class DesktopPosture(NamedTuple):
    """The desktop plane's live authorization state, as one value.

    ``token`` is the bearer that opens the plane (the env capability when the
    app started this daemon, the accepted claim key otherwise) and ``""`` when
    the plane is closed. ``origins`` is the Origin allowlist in force, which is
    the environment's list UNIONED with a claimed caller's own origin.

    A pair rather than two functions so that a caller cannot read the bearer
    and the allowlist from two different moments — the boundary middleware and
    the Origin check both need this to describe ONE posture.
    """

    token: str
    origins: frozenset[str]

    @property
    def enabled(self) -> bool:
        """Whether anything governs the plane (env capability or accepted claim)."""
        return bool(self.token)


@dataclass(frozen=True)
class _Claim:
    """The one accepted claim: the key it was proved with, and its Origin."""

    key: str
    origins: frozenset[str]


#: One-way latch (see the module docstring). ``None`` until a claim is accepted.
_CLAIMED: _Claim | None = None


def _parse_origins(raw: str) -> frozenset[str]:
    """The allowlist an environment string names, in the historical dialect.

    ``"null"`` and blanks are dropped: a null origin is the literal string a
    sandboxed document (``srcdoc``, a ``data:`` URL) gets, and admitting it
    would hand every opaque-origin document the plane.
    """
    return frozenset(
        item.strip() for item in raw.split(",") if item.strip() and item.strip() != "null"
    )


def desktop_posture() -> DesktopPosture:
    """The bearer and Origin allowlist governing the desktop plane RIGHT NOW.

    The ONE predicate for "is the desktop plane open", read by every gate:
    the routers' ``require_desktop`` dependency, the legacy control boundary,
    the validation-error shaper, the CORS echo suppressor and the public
    ``/v1/capabilities`` signal. Before this existed, each of them tested the
    environment variable itself, so a claim that turned the routers on but not
    the legacy gate (or the CORS suppressor) was a live possibility — the shape
    of the bug this replaces: the same question answered in seven places.

    The environment wins when it is set: a backend the app itself started is
    already under its control, and a claim against it is refused rather than
    silently installed beside it. Otherwise an accepted claim governs, and the
    env origins (if any) stay in force alongside the claimed origin.
    """
    env_token = os.environ.get(TOKEN_ENV, "")
    env_origins = _parse_origins(os.environ.get(ORIGINS_ENV, ""))
    if env_token:
        return DesktopPosture(token=env_token, origins=env_origins)
    claimed = _CLAIMED
    if claimed is not None:
        return DesktopPosture(token=claimed.key, origins=env_origins | claimed.origins)
    return DesktopPosture(token="", origins=env_origins)


def _is_non_browser_caller(request: Request) -> bool:
    """Whether this request demonstrably did NOT come from a web page.

    ``Sec-Fetch-Site`` is added by the browser fetch/XHR stack itself and sits
    on the forbidden-header list, so page script can neither remove nor forge
    it. Its absence is therefore positive evidence of a native caller (Electron
    main, the dev proxy, curl), which is what lets an Origin-less request be
    admitted without reopening the allowlist bypass to a browser.
    """
    return "sec-fetch-site" not in request.headers


def _require_allowed_origin(request: Request, allowed: frozenset[str]) -> None:
    """The Origin / ``Sec-Fetch-Site`` rule, shared by every way into the plane."""
    origin = request.headers.get("origin")
    if origin is not None:
        if origin not in allowed:
            raise HTTPException(403, "This origin cannot access desktop controls.")
    elif allowed and not _is_non_browser_caller(request):
        # An ABSENT Origin used to skip the check entirely, so a caller that
        # simply omitted the header bypassed the allowlist (code review 4).
        #
        # It cannot become an unconditional "Origin required", because the two
        # first-class callers legitimately send none: Electron main fetches
        # from the main process, and the dev proxy forwards server-side. Both
        # are non-browser agents, for whom Origin is not a security signal.
        #
        # What a BROWSER cannot forge is the absence of `Sec-Fetch-Site`: it is
        # attached by the fetch/XHR stack to every request a page makes and is
        # on the forbidden-header list, so script cannot remove or spoof it. A
        # request carrying it is browser-originated and must present an
        # allowed Origin; one without it is a native client, which the bearer
        # below still authenticates.
        raise HTTPException(403, "This origin cannot access desktop controls.")


def require_desktop(request: Request) -> None:
    """Open the desktop plane to this request, or raise the refusal.

    Written on :func:`desktop_posture`, so an accepted claim opens every route
    that depends on this exactly as the environment capability does. The Origin
    allowlist check and the ``Sec-Fetch-Site`` distinction are unchanged: a
    claim adds its caller's origin to the allowlist, it does not remove the
    requirement to be on it.
    """
    posture = desktop_posture()
    if not posture.enabled:
        raise HTTPException(503, "Desktop controls require a backend started by the desktop app.")
    _require_allowed_origin(request, posture.origins)
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {posture.token}".encode("utf-8")
    if not secrets.compare_digest(supplied.encode("utf-8"), expected):
        raise HTTPException(401, "Desktop authorization is required.")


def _claim_origin(request: Request) -> str | None:
    """The Origin a claim installs, or the refusal that stops it.

    Deliberately NOT :func:`_require_allowed_origin`, because that rule refuses
    any origin it does not already trust and the claim is the one request whose
    whole purpose is to ADD one: requiring the app's origin to be pre-trusted
    would refuse every desktop app that did not start this daemon — the only
    caller this route exists for. The gate here is the record's 256-bit key,
    which a page cannot read; the Origin is the VALUE being installed, not the
    credential, and the holder of the key could open the plane with the bearer
    alone in any case.

    What is still refused, and each for its own reason:

    * ``"null"`` — an opaque origin names EVERY ``srcdoc`` document and ``data:``
      URL rather than one application, so installing it would admit strictly
      more than the app that asked;
    * a browser-originated request with no origin at all — ``Sec-Fetch-Site``
      proves a page and there is nothing to install;
    * an origin the operator's own environment list excludes — that list is a
      deliberate narrowing by whoever started the daemon.
    """
    origin = request.headers.get("origin")
    if origin is None:
        if not _is_non_browser_caller(request):
            raise HTTPException(403, "This origin cannot access desktop controls.")
        return None
    allowed = desktop_posture().origins
    if origin == "null" or (allowed and origin not in allowed):
        raise HTTPException(403, "This origin cannot access desktop controls.")
    return origin


def accept_claim(request: Request, *, published_key: str) -> None:
    """Accept THE single claim on this process's desktop plane, or raise.

    Called only by ``POST /v1/desktop/claim``, which is deliberately NOT behind
    :func:`require_desktop` — it is the way in while the plane is unclaimed,
    and a claim route gated on an unclaimed plane is a deadlock the design
    calls out by name. The Origin rule it applies is therefore its own
    (:func:`_claim_origin`), stricter about what may be INSTALLED than
    :func:`require_desktop` is about what may be reached, and identical about
    the unforgeable ``Sec-Fetch-Site`` signal.

    The key is compared with :func:`secrets.compare_digest` against the value
    published in this daemon's record, exactly as a session's ``control_key``
    is checked. What the caller may see on refusal is a status and nothing
    else: no partial match, no length, no echo of the submitted value.

    Refusals, in the order they are decided:

    * ``403`` — an origin that cannot be installed at all (see
      :func:`_claim_origin`).
    * ``409`` — the plane is ALREADY governed: by the environment (the app
      started this daemon, so there is nothing to claim) or by an accepted
      claim (the latch is one-way; see the module docstring). Deliberately not
      a ``401``: the caller is not being asked for a different key, it is being
      told this plane is not its to claim.
    * ``503`` — no key was published, which means this process booted without
      announcing an address and so wrote no record at all. Without the record
      there is no way to hand a caller the key, and therefore no claim: the
      key's only lawful channel is the record.
    * ``401`` — the presented key does not match the published one.
    """
    global _CLAIMED
    origin = _claim_origin(request)
    if desktop_posture().enabled:
        raise HTTPException(409, "This desktop plane is already controlled.")
    if not published_key:
        raise HTTPException(503, "This backend published no desktop claim key.")
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {published_key}".encode("utf-8")
    if not secrets.compare_digest(supplied.encode("utf-8"), expected):
        raise HTTPException(401, "Desktop claim authorization is required.")
    # The caller's own Origin joins the allowlist — the literal "null" and an
    # origin off a configured list never get this far. A native caller
    # (Electron main) sends none; it is then allowlisted by the
    # ``Sec-Fetch-Site`` rule above, not by an Origin it does not have, and
    # every later request it makes is admitted by the bearer.
    _CLAIMED = _Claim(
        key=published_key,
        origins=frozenset() if origin is None else frozenset({origin}),
    )
