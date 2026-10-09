"""The forge-adapter registry: one lookup from a :class:`Ref` to its fetcher.

The whole registry is two FORGE TABLES and one function, because the design's
add-a-forge story is "one file + fixtures":

* a new adapter module exposes an ``ADAPTER`` singleton and its ``kind``;
* the module is imported in :data:`_FULL_ADAPTERS` (a full adapter) or left
  unimported (detect-and-link: the row still exists, no state is fetched).

GitHub and GitLab are the two full adapters this slice ships. Gitea/Forgejo
(PR3), Bitbucket, Azure DevOps and Gerrit are detect-and-link in ``refs.py`` —
:func:`adapter_for` returns ``None`` for them, and every caller already treats
``None`` as "keep the row link-only", so adding a third adapter later changes
no call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from local_operator.code_requests.adapters.base import (
    Comment,
    FetchOutcome,
    Forge,
    ForgeHTTPError,
    merge_comments,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.code_requests.refs import Ref

#: Forge -> adapter singleton. Imported eagerly: both modules are stdlib+httpx
#: and the fetch path is already off the event loop when this package loads.
_FULL_ADAPTERS: dict[str, Forge] = {}


def _register() -> None:
    from local_operator.code_requests.adapters.github import ADAPTER as _github
    from local_operator.code_requests.adapters.gitlab import ADAPTER as _gitlab

    _FULL_ADAPTERS[_github.kind] = _github
    _FULL_ADAPTERS[_gitlab.kind] = _gitlab


_register()

#: The forge kinds a fetch can actually read state for. The route and the tool
#: ask this (not :data:`_FULL_ADAPTERS` directly) so "full" has one spelling.
FULL_FORGES: frozenset[str] = frozenset(_FULL_ADAPTERS)


def adapter_for(ref: "Ref") -> Forge | None:
    """The adapter that can fetch ``ref``, or ``None`` for detect-and-link."""
    if not ref.full:
        return None
    return _FULL_ADAPTERS.get(ref.forge)


def state_of(adapter: Forge, pieces: Mapping[str, object]) -> str:
    """``adapter.state`` behind the "only when there is data" guard.

    An empty summary (a piece that has never been fetched) is not a state, and
    the caller must not store one: ``""`` tells the service to keep whatever it
    had (usually nothing) rather than render a made-up ``open``.
    """
    if not pieces.get("summary"):
        return ""
    return adapter.state(pieces)


__all__ = [
    "Comment",
    "FULL_FORGES",
    "FetchOutcome",
    "Forge",
    "ForgeHTTPError",
    "adapter_for",
    "merge_comments",
    "state_of",
]
