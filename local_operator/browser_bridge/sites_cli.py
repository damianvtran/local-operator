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


def _say(message: Any) -> None:
    from local_operator.i18n.messages import render_resolved

    print(textwrap.fill(render_resolved(message), width=78, subsequent_indent="  "))


def _origin_of(raw: str) -> str | None:
    """``scheme://host[:port]`` for an http(s) URL or origin, else ``None``."""
    parts = urlsplit(str(raw or "").strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return f"{parts.scheme}://{parts.netloc.lower()}"


async def _call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    from local_operator.browser_bridge.backend import BridgeClient

    return dict(await BridgeClient().call(method, params))


def main(revoke: str | None = None, *, as_json: bool = False) -> int:
    """Run the command; ``0`` on success, ``1`` on any refusal or bridge failure."""
    from local_operator.i18n.keys import browser_site_approval as _keys

    if revoke is not None:
        origin = _origin_of(revoke)
        if origin is None:
            _say(_keys.sites_bad_origin(origin=revoke))
            return 1
        try:
            result = asyncio.run(_call("revoke_access", {"url": origin}))
            listing = asyncio.run(_call("list_access", {}))
        except Exception as error:  # noqa: BLE001 — any bridge failure is reported the same way
            _say(_keys.sites_unreachable(detail=str(error) or type(error).__name__))
            return 1
        if as_json:
            print(json.dumps(result, sort_keys=True))
            return 0
        if result.get("removed"):
            _say(_keys.sites_revoked(origin=origin))
        else:
            _say(_keys.sites_revoke_nothing(origin=origin))
        broad = [str(item) for item in listing.get("broad") or ()]
        host = urlsplit(origin).hostname or ""
        covering = [item for item in broad if host == item or host.endswith(f".{item}")]
        if covering:
            # The exact row is gone, but a domain/host-wide grant from a desktop
            # popup would still admit it: say so rather than imply it is blocked.
            _say(_keys.sites_still_broad(origin=origin, grants=", ".join(covering)))
        return 0
    try:
        listing = asyncio.run(_call("list_access", {}))
    except Exception as error:  # noqa: BLE001
        _say(_keys.sites_unreachable(detail=str(error) or type(error).__name__))
        return 1
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
    _say(_keys.sites_revoke_hint())
    return 0
