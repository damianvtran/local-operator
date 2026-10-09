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
    #: ``{job_id,label,agent_role,child_session_id,path}`` when a subagent opened it.
    via: dict[str, Any] | None = None
    #: The parent session id a forked/inherited row came from.
    inherited_from: str | None = None
    #: The rule that classified the row, newest last — the audit trail behind ``relation``.
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    first_at: float | None = None
    last_at: float | None = None
    #: Filled by the adapter slice: ``{state, draft, title, head_sha, ci, updated_at}``.
    summary: dict[str, Any] | None = None
    #: Filled by the round parser once comments are fetched.
    lanes: list[dict[str, Any]] | None = None
    fetched_at: float | None = None
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
    #: Hosts whose refresh is cooling down, mapped to the instant it lifts. Always empty
    #: in this slice (nothing fetches yet) and present so 1b does not change the shape.
    cooling: dict[str, float] = Field(default_factory=dict)
    #: The scan's own state: ``ready`` when the index is current for the journal,
    #: ``refreshing`` when a scan is running for a journal that has moved, ``error`` when
    #: the last scan failed. A client shows a spinner for the middle one only.
    scan_state: str = "ready"
    updated_at: float | None = None


class CodeRequestRefreshReceipt(BaseModel):
    """``POST …/code-requests/refresh`` — a 202 receipt for work that has not happened."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    accepted: bool
    keys: list[str] = Field(default_factory=list)
    force: bool = False
    #: What the caller should do instead, in one sentence. This route exists so the UI's
    #: refresh affordance has a stable address; until the adapter slice lands it does no
    #: work, and this field is the honest statement of that rather than a silent no-op.
    note: str = ""
