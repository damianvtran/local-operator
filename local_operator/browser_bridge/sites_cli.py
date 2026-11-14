"""``lop browser sites [--revoke ORIGIN]``: see and take back stored site decisions.

WHY THIS EXISTS. A headless node's site card says Approve lasts "until someone
runs ``lop browser sites --revoke`` on that machine" (PR #3938 review D2/U4):
the extension's Settings page is the only other place a grant can be removed,
and a node's Chrome has no screen. This lists what THIS machine's browser has
stored — exact-origin grants, the operator's signed denials (which stop an
agent asking again, D3), broad grants and the all-sites switch — and removes
one exact origin's grant or denial through the bridge's ``revoke_access``.

CONSTRAINTS. It talks to the local bridge only (the same daemon the agent's
browser tool uses), so it works on any machine with a paired extension, node
or not. Removal only ever REDUCES what is allowed (or lets an agent ask again
about a denied site), so it needs no operator signature; a revoke takes effect
on the next navigation, because the extension re-reads its grants per request.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any
from urllib.parse import urlsplit

#: Continuation lines of a wrapped sentence are inset further than the 2-space
#: data rows, so a wrapped sentence never reads as one more listed origin (D10).
_CONTINUATION_INDENT = "    "


def _say(message: Any) -> None:
    from local_operator.i18n.messages import render_resolved

    # Never break inside a word (D10): an origin, a socket path or a command
    # split at a hyphen or mid-token arrives broken when copied back into
    # ``--revoke``. A token longer than the line overflows it instead.
    print(
        textwrap.fill(
            render_resolved(message),
            width=78,
            subsequent_indent=_CONTINUATION_INDENT,
            break_on_hyphens=False,
            break_long_words=False,
        )
    )


def _origin_of(raw: str) -> str | None:
    """``scheme://host[:port]`` for an http(s) URL or origin, else ``None``.

    Only a pre-filter: the extension normalises with ``URL.origin`` (userinfo
    and default port dropped, host punycoded) and reports the origin it acted
    on, which is what the output names (R3-2).
    """
    try:
        parts = urlsplit(str(raw or "").strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return None
        port = f":{parts.port}" if parts.port else ""
    except ValueError:  # a malformed port
        return None
    return f"{parts.scheme}://{parts.hostname}{port}"


async def _call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    from local_operator.browser_bridge.backend import BridgeClient

    return dict(await BridgeClient().call(method, params))


def _fail(message: Any, *, as_json: bool, code: str) -> int:
    """Report a refusal or bridge failure; JSON on ``--json`` so a script parsing
    the output never meets prose (U17)."""
    if as_json:
        from local_operator.i18n.messages import render_resolved

        print(json.dumps({"ok": False, "error": code, "message": render_resolved(message)}))
    else:
        _say(message)
    return 1


def _unreachable(error: Exception, *, as_json: bool) -> int:
    from local_operator.browser_bridge import backend
    from local_operator.i18n.keys import browser_site_approval as _keys
    from local_operator.i18n.messages import render_resolved

    # The bridge's own message ends with desktop advice ("Run 'lop browser
    # status'; 'lop browser install' starts it."); keep only its reason so the
    # remedy is said once, and the node's remedy on a node (D11/U17).
    detail = (str(error) or type(error).__name__).split(" Run 'lop browser status'")[0]
    on_node = backend._on_headless_node()  # noqa: SLF001
    remedy = _keys.sites_remedy_node() if on_node else _keys.sites_remedy_desktop()
    message = _keys.sites_unreachable(detail=detail.rstrip(" ."), remedy=render_resolved(remedy))
    return _fail(message, as_json=as_json, code="unreachable")


def main(revoke: str | None = None, *, as_json: bool = False) -> int:
    """Run the command; ``0`` on success, ``1`` on any refusal or bridge failure."""
    from local_operator.i18n.keys import browser_site_approval as _keys

    if revoke is not None:
        origin = _origin_of(revoke)
        if origin is None:
            bare = "://" not in revoke and _origin_of(f"https://{revoke}") is not None
            message = (
                _keys.sites_bad_origin_no_scheme(origin=revoke)
                if bare
                else _keys.sites_bad_origin(origin=revoke)
            )
            return _fail(message, as_json=as_json, code="bad_origin")
        try:
            result = asyncio.run(_call("revoke_access", {"url": origin}))
        except Exception as error:  # noqa: BLE001 — any bridge failure is reported the same way
            return _unreachable(error, as_json=as_json)
        if as_json:
            print(json.dumps(result, sort_keys=True))
            return 0
        # The origin the extension acted on (R3-2), not the typed spelling.
        origin = str(result.get("origin") or origin)
        removed = [str(item) for item in result.get("removed") or ()]
        if "allowed" in removed:
            _say(_keys.sites_revoked(origin=origin))
        if "denied" in removed:
            _say(_keys.sites_revoked_denial(origin=origin))
        if not removed:
            _say(_keys.sites_revoke_nothing(origin=origin))
        # The revoke already happened: a listing that fails now only loses the
        # broad-grant note, never the exit status (R3-3).
        try:
            listing = asyncio.run(_call("list_access", {}))
        except Exception:  # noqa: BLE001
            return 0
        broad = [str(item) for item in listing.get("broad") or ()]
        host = urlsplit(origin).hostname or ""
        covering = [item for item in broad if host == item or host.endswith(f".{item}")]
        if covering:
            # The exact row is gone, but a domain/host-wide grant from a desktop
            # popup would still admit it: say so rather than imply it is blocked.
            _say(_keys.sites_still_broad(origin=origin, grants=", ".join(covering)))
            _say(_keys.sites_revoke_limits())
        return 0
    try:
        listing = asyncio.run(_call("list_access", {}))
    except Exception as error:  # noqa: BLE001
        return _unreachable(error, as_json=as_json)
    if as_json:
        print(json.dumps(listing, sort_keys=True))
        return 0
    allowed = [str(item) for item in listing.get("allowed") or ()]
    denied = [str(item) for item in listing.get("denied") or ()]
    broad = [str(item) for item in listing.get("broad") or ()]
    allow_all = bool(listing.get("allow_all"))
    if not (allowed or denied or broad or allow_all):
        _say(_keys.sites_none())
        return 0
    if allow_all:
        _say(_keys.sites_allow_all())
    for heading, items in (
        (_keys.sites_allowed(), allowed),
        (_keys.sites_denied(), denied),
        (_keys.sites_broad(), broad),
    ):
        if items:
            _say(heading)
            for item in items:
                print(f"  {item}")  # i18n: ignore data row: an origin or domain, verbatim
    if allowed or denied:
        _say(_keys.sites_revoke_hint())
    if broad or allow_all:
        # --revoke cannot clear either (D14/U17): say where they ARE changed,
        # and that a node has no screen for it, rather than imply the hint covers them.
        _say(_keys.sites_revoke_limits())
    return 0
