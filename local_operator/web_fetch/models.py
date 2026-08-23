"""Shared web-fetch configuration and result models.

The engine, the tool, the CLI status view, and the TUI card all read one small
contract from here so they cannot drift into subtly different shapes — the same
boundary discipline :mod:`local_operator.web_search.models` keeps for search.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

#: How the body was turned into text. Surfaced in ``details`` and the card so a
#: reader can tell a good markdown render from the degraded stdlib fallback or a
#: pass-through, and so tests can assert which backend actually ran.
RenderMethod = Literal[
    "markdownify",  # the [fetch] extra rendered the HTML
    "stdlib",  # html.parser fallback rendered the HTML (extra absent)
    "json",  # pretty-printed JSON
    "text",  # plain text / markdown pass-through
    "binary",  # PDF/image/other: a notice, never inlined
]

#: Render backend the config asks for. ``auto`` uses markdownify when the
#: ``[fetch]`` extra is importable and silently degrades to the stdlib parser
#: when it is not; ``stdlib`` forces the fallback even when the extra is present
#: (useful for reproducing the bare-install path).
RenderBackend = Literal["auto", "stdlib"]


class FetchResult(BaseModel):
    """Normalized outcome of one fetch, before spill/preview shaping.

    ``content`` is the full rendered text (what gets spilled); the tool builds
    a bounded preview from it. ``complete`` is False when ``max_bytes`` stopped
    the download mid-stream, so the reader knows the tail is missing rather than
    absent from the source.
    """

    model_config = ConfigDict(extra="ignore")

    url: str
    final_url: str
    status: int
    content_type: str
    render_method: RenderMethod
    content: str
    bytes: int  # bytes downloaded (post-decode length is derived from content)
    complete: bool = True
    low_quality: bool = False
    cache: Literal["hit", "miss"] = "miss"


class WebFetchSettings(BaseModel):
    """Validated view of the loose ``values.web_fetch`` YAML mapping."""

    enabled: bool = True
    timeout_seconds: float = 20.0
    max_bytes: int = 5 * 1024 * 1024  # download ceiling, enforced during streaming
    max_redirects: int = 5
    cache_ttl_seconds: int = 900  # 0 disables the URL cache entirely
    allow_private: bool = False  # SSRF: allow loopback/private/link-local targets
    render_backend: RenderBackend = "auto"  # auto = markdownify if [fetch] present
    enrich: bool = True  # try .md / llms.txt / content-negotiation before scraping HTML


DEFAULT_WEB_FETCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "timeout_seconds": 20.0,
    "max_bytes": 5 * 1024 * 1024,
    "max_redirects": 5,
    "cache_ttl_seconds": 900,
    "allow_private": False,
    "render_backend": "auto",
    "enrich": True,
}
