"""Response models for the per-session code-request surface.

``GET /v1/desktop/sessions/{session_id}/code-requests`` reads the derived index
(``code_requests/ledger.py``), which this slice populates from the transcript alone.
Every row is therefore **link-only**: it names a code request, says how this session
related to it, and opens it in the browser. The ``summary`` and ``lanes`` fields exist
and are ``None`` because fetching a PR's state, its comments and its CI is the NEXT
slice's job — a field that is present-and-null is how a client negotiates that
difference without a second capability key, and it is why those two are the only
``Optional`` members with no default value.

What is deliberately NOT on the wire:

* **Comment bodies.** The round parser reads them (``code_requests/rounds.py``), and a
  listing that shipped every comment on every PR would send megabytes to draw a status
  word.
* **Credentials of any kind.** Not a field, not a flag, not a boolean about whether one
  exists — "link only" is the whole statement, and a later slice's copy names the CLI to
  sign in with.
* **The scanner's own machinery.** ``revision`` (monotone, per session) is the currency
  a client compares against the feed frame; byte counts, offsets and schema versions stay
  on disk where they belong.

``extra="allow"`` and the per-field defaults follow ``MonitorRow``'s stance: an added
field is additive for an older client, so 1b can widen a row without a contract break.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CodeRequestMention(BaseModel):
    """How the session's TEXT referred to one code request.

    ``sources`` is a subset of ``user``/``assistant``/``peer``/``tool`` and is a list
    rather than four booleans because a client renders it as chips in one pass. ``tool``
    only appears on a row that is otherwise visible — refs seen only in tool output are
    collapsed, and appear here only when the caller asked for them.
    """

    model_config = ConfigDict(extra="allow")

    sources: list[str] = Field(default_factory=list)
    count: int = 0
    first_at: float | None = None
    last_at: float | None = None


class CodeRequestRow(BaseModel):
    """One GitHub PR, GitLab MR, or detect-and-link sibling, as this session saw it."""

    model_config = ConfigDict(extra="allow")

    key: str
    url: str
    forge: str
    host: str
    project: str
    number: int
    #: ``opened`` | ``acted`` | ``mentioned`` | ``unknown`` | ``inherited``. The word is
    #: the whole claim, and the copy for each lives in the client.
    relation: str
    #: Every relation this ref accumulated, strongest first: a row can be both opened and
    #: acted on, and a client that hides the second loses the "you commented" marker.
    relations: list[str] = Field(default_factory=list)
    #: ``comment``/``merge``/``review``/``push``/``edit``/``close``, in the order seen.
    acted: list[str] = Field(default_factory=list)
    mention: CodeRequestMention | None = None
    #: True for every row in this slice, and it becomes False per row when an adapter
    #: exists for the forge AND a credential was found. Named here rather than derived
    #: from ``summary is None`` because a fetched row can still be stale-and-not-fetched.
    link_only: bool = True
    #: Why a row is only a link, when there is something to say (an unconfirmed host, a
    #: script-created URL, a fork's inheritance). Shown verbatim; never a guess.
    reason: str | None = None
    #: The one-line remedy a UI shows for a link-only row, per FORGE (cross-round
    #: finding X3): gh/glab with the ``--hostname`` form off the canonical host, and
    #: NO CLI at all for detect-and-link-only forges ("Link only — this host isn't
    #: tracked yet."). A client must render this rather than deriving a CLI from
    #: ``forge`` — the derivation said "gh" for Codeberg (QA round 2, Q11: the field
    #: existed on the view but never reached the wire until it was listed here).
    link_only_hint: str | None = None
    #: Epoch seconds until a cooling host accepts requests again, when the row's host
    #: is rate-limited: the row is tracked and WILL be fetched, so a reader must see
    #: the wait rather than an unqualified "Link only" (QA round 2, Q13).
    cooling_until: float | None = None
    #: ``{job_id,label,agent_role,child_session_id,path}`` when a subagent opened it.
    via: dict[str, Any] | None = None
    #: The parent session id a forked/inherited row came from.
    inherited_from: str | None = None
    #: The rule that classified the row, newest last — the audit trail behind ``relation``.
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    first_at: float | None = None
    last_at: float | None = None
    #: Filled by the adapter slice: ``{state, draft, title, head_sha, ci, comments,
    #: updated_at}`` — ``comments`` is the host's comment count, null when not reported.
    summary: dict[str, Any] | None = None
    #: The parsed review lanes, once comments have been fetched.
    lanes: list[dict[str, Any]] | None = None
    #: When the forge last answered with NEW content (a 304 does not move it).
    fetched_at: float | None = None
    #: True when the last refresh failed and this row keeps older data.
    stale: bool = False
    refresh_error: str | None = None


class CodeRequestListing(BaseModel):
    """The listing's envelope: the rows, what they leave out, and how fresh it is."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    #: Monotone per session, bumped whenever the index is rewritten. This is the value
    #: the ``code_requests`` feed frame compares, so a client can tell "my list is stale"
    #: from "the stream stuttered".
    revision: int = 0
    rows: list[CodeRequestRow] = Field(default_factory=list)
    #: How many refs the scan collapsed because they appeared ONLY in tool output. Rendered
    #: as "N more seen in tool output"; ``?include=mentions_tool`` expands them.
    tool_output_only_count: int = 0
    #: True when that count is the STORED number rather than the session's whole set —
    #: the scan caps the collapsed list, and a client must not present a capped count as
    #: exact.
    tool_output_truncated: bool = False
    #: ``git push`` "create a pull request" links: a fact about a BRANCH, kept out of the
    #: rows on purpose (the link names no code request).
    hints: list[dict[str, Any]] = Field(default_factory=list)
    #: Hosts whose refresh is cooling down, mapped to the instant it lifts
    #: (epoch seconds). ``{}`` when no host is cooling. A cooling host's rows
    #: keep their last known data and say ``stale``; force-refresh never
    #: bypasses this (a force must not defeat the host's own rate limit).
    cooling: dict[str, float] = Field(default_factory=dict)
    #: The scan's own state: ``ready`` when the index is current for the journal,
    #: ``refreshing`` when a scan is running for a journal that has moved, ``error`` when
    #: the last scan failed. A client shows a spinner for the middle one only.
    scan_state: str = "ready"
    updated_at: float | None = None


class CodeRequestRefreshReceipt(BaseModel):
    """``POST …/code-requests/refresh`` — a 202 receipt for queued work.

    The scan half ran; the FETCH half is scheduled (reads never block on the
    network, so the fetch completes behind this receipt and its result arrives
    through the feed frame + the next GET). ``note`` is the honest sentence
    about both halves and about rows that cannot be fetched at all.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    accepted: bool
    keys: list[str] = Field(default_factory=list)
    force: bool = False
    #: One sentence about what was done and queued. Written here rather than
    #: composed by the client so the copy lives beside the behaviour it
    #: describes.
    note: str = ""
