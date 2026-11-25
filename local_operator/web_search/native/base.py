"""The contract every native-search vendor module fills in.

WHY A SECOND DATACLASS beside ``providers.NativeVendor``. The band code (PR1)
needs only three facts about a native vendor -- its id, which legs it can serve
here and their credential class, and its per-search fee for ordering -- and
``providers.NativeVendor`` is exactly that. A vendor MODULE also has to supply
its transport and its status-row copy, which the band code must never read.
:class:`NativeSearchVendor` carries the whole module contract; ``providers``
projects it onto ``NativeVendor`` (bands) and ``ProviderDefinition`` (transport
and copy) at import time, so adding a vendor never edits band code.

IMPORT RULE. Nothing under ``web_search/native/`` may import
``web_search.providers`` at MODULE level: ``providers`` imports this package
at its own top to register the vendors, so a module-level import back would be
a cycle. Helpers that live in ``providers`` (``_source``, ``LEG_MODE``) are
imported inside functions, where ``providers`` is already fully loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

import httpx

from local_operator.i18n.messages import Msg
from local_operator.web_search.models import SearchResponse, WebSearchSettings

#: ``plan``: included in a subscription the user already pays for (no marginal
#: cash) -> the NATIVE band, before Radient. ``key``: bills a pay-as-you-go
#: credential per search -> ``providers.NATIVE_MODEL_KEY_BAND`` (after Radient,
#: operator decision 2026-11-24). Spelled identically to ``providers.NativeClass``
#: (a test pins the two equal) because this module cannot import that one.
NativeClass = Literal["plan", "key"]

#: Signature of a native transport. The same five arguments as
#: ``providers.ProviderSearch``; the leg's mode arrives through
#: ``providers.LEG_MODE`` (see :func:`leg_mode`), never as an argument, so a
#: vendor transport slots into ``PROVIDERS`` unchanged.
NativeSearch = Callable[
    [httpx.AsyncClient, Path | None, WebSearchSettings, str, int],
    Awaitable[SearchResponse],
]
NativeModes = Callable[[Path | None, WebSearchSettings], list[tuple[str, NativeClass]]]


@dataclass(frozen=True, slots=True)
class NativeSearchVendor:
    """One model vendor's built-in web search, as a vendor module declares it.

    ``modes`` MUST be cheap and side-effect free (a bare store SELECT, an env
    read): the status view calls it per provider per render. It returns the
    legs this vendor can serve on this install, strongest first, as
    ``(mode, class)`` pairs, and ``[]`` when no credential is stored.

    ``fee_usd`` is the published per-search tool fee (excluding tokens), used
    by the band code only to order paid legs cheapest-first. ``label`` is the
    brand name, which is never translated; ``access``/``detail`` are keyed copy
    for the `search list` row.
    """

    id: str
    label: str
    access: Msg
    detail: Msg
    modes: NativeModes
    search: NativeSearch
    fee_usd: float | None = None
    #: Env/store key names that also unlock the key leg (shown by `search list`
    #: setup hints). Empty for a vendor that only signs in.
    credential_keys: tuple[str, ...] = ()


def leg_mode(vendor: str, available: list[tuple[str, NativeClass]]) -> str:
    """The mode THIS call serves: the walked leg's, else the strongest available.

    A walked leg pins its mode through ``providers.LEG_MODE``; a vendor with a
    plan AND a key leg must serve the leg the chain is walking, or the plan leg
    would spend the key (or the reverse) at the wrong band position. Outside a
    walk (a direct call) the first available mode serves. A pinned mode that is
    no longer available -- the credential was removed between resolve and call
    -- raises, so the chain records a failure and moves on rather than silently
    spending a different credential class.
    """
    from local_operator.web_search.providers import LEG_MODE

    modes = [mode for mode, _cls in available]
    pinned = LEG_MODE.get()
    if pinned is None:
        if modes:
            return modes[0]
    elif pinned in modes:
        return pinned
    from local_operator.i18n.keys import web_search as web_search_copy
    from local_operator.i18n.text import tr

    raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=vendor)))


def error_text(provider: str, response: httpx.Response) -> RuntimeError:
    """A transport failure carrying the vendor's own (bounded) error body.

    Delegates to ``providers._http_error`` so every transport reports HTTP
    failures in one shape; the body is the vendor's text, never a credential.
    """
    from local_operator.web_search.providers import _http_error

    return _http_error(provider, response)
