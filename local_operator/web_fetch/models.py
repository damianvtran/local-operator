"""Shared web-fetch configuration and result models.

The engine, the tool, the CLI status view, and the TUI card all read one small
contract from here so they cannot drift into subtly different shapes — the same
boundary discipline :mod:`local_operator.web_search.models` keeps for search.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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

    # --- retry / block diagnostics (all defaulted: every existing construction
    # site stays valid, and a cached entry written by an older lop reads back
    # with these absent rather than failing validation).
    #
    # These exist because a failed fetch used to tell the agent only THAT it
    # failed. `attempts` explains a longer duration, `failure_kind` says whether
    # another try could help, and the block fields say who refused and what to
    # quote to them. They ride in `details`, which never reaches the provider,
    # so they cost no tokens.
    attempts: int = 1  # network attempts spent; 0 on a cache hit (none were)
    failure_kind: str | None = None  # FailureKind, absent on success
    block_vendor: str | None = None  # None on an UNSIGNED refusal — never guessed
    block_reference: str | None = None  # the origin's own reference / cf-ray
    profile: str = "default"  # request identity that produced this outcome
    # Identity used by each attempt, in order. Carried as a sequence rather than
    # derived from `profile` + `attempts` because "three tries, all honest" and
    # "two tries, the second wearing a browser's headers" are different facts and
    # only the sequence distinguishes them.
    profiles: list[str] = Field(default_factory=lambda: ["default"])
    retry_after_s: float | None = None  # honoured or reported Retry-After


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
    # Network attempts per redirect hop, INCLUDING the first. 1 reproduces the
    # pre-retry behaviour exactly. 3 is where added coverage stops paying for
    # added wall-clock: the value of a retry is concentrated in the first one,
    # and this tool runs inside a live turn the user is watching.
    max_attempts: int = 3
    # One browser-shaped attempt after the origin has already refused an honest,
    # self-identifying request. Default on because the measured win is real
    # (medium.com: 403 with the lop UA, 200 with the browser profile, 3/3) and
    # the cost is one request on an already-failed fetch. Off is for an operator
    # who wants the client to stay honest even in the face of a refusal.
    blocked_retry: bool = True


DEFAULT_WEB_FETCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "timeout_seconds": 20.0,
    "max_bytes": 5 * 1024 * 1024,
    "max_redirects": 5,
    "cache_ttl_seconds": 900,
    "allow_private": False,
    "render_backend": "auto",
    "enrich": True,
    "max_attempts": 3,
    "blocked_retry": True,
}
