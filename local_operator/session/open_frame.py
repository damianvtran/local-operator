"""The open frame: the page every surface paints, in the unit it paints.

WHY THIS MODULE EXISTS. A conversation's open is one read performed by four
surfaces (desktop, terminal, relay web, native app) and each of them re-derives
the same three things from whatever rows happen to be loaded: which rows a
transcript SHOWS, where a turn begins and ends, and what a collapsed turn's bar
says. The wire unit is the journal ENTRY, so the answers arrive late and change
after paint — the operator's own case was a bar reading "30 actions" at open and
"Took 2h23m · 423 actions" thirteen pages later — and the payload carried bytes
no renderer paints (on this machine's 40 largest journals: 39.1% checkpoint
rows, 10.0% compaction summaries, 11.3% of assistant-row bytes in provider
replay material).

This module is the ONE derivation, shared by the desktop plane and available to
the relay's fold: a pure function of (rows, index facts, limits) with no HTTP
route, no bridge and no session object in it. ``strip_entry`` says what a
surface can paint; ``read_frame_page`` walks the journal backward and stops on
PAINTABLE rows rather than journal entries; ``publish_runs`` joins the run facts
the transcript index derived from the whole journal.

THE HONESTY RULES, which are the reason this is not just a size optimisation:

- a page that could not reach its oldest run's head says so (``head_cut``),
  rather than looking identical to a page that did;
- run facts are stated for SETTLED runs only. A live tail's counts would be
  corrected by the next row, and a corrected number is exactly the after-paint
  change this contract exists to remove;
- a run whose row body the index could not read is ``complete: false``: its
  counts are a lower bound and the client is told so. (Measured on this machine:
  0 of 65,755 tool rows across the twelve largest journals exceed the index's
  keep limit, so in practice this flag never fires — it exists so that a
  pathological row cannot be reported as an exact count.)
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from local_operator.session.transcript_index import RunRecord, TranscriptIndex

logger = logging.getLogger(__name__)

#: The hard cap on the run-extension, in PAINTABLE rows. Chosen against the
#: measured shape of the operator's sessions: the reported case is a settled run
#: of about 600 rows, and a page that grew to hold it whole would carry roughly
#: 1.2-2 MB through main -> renderer IPC — a payload diet that pays for itself
#: is not one that triples the payload to save a round trip. 400 rows keeps the
#: extension useful for the ordinary tool-heavy run while the cap, not the run,
#: decides the worst case.
OPEN_FRAME_MAX_ROWS = 400

#: The hard cap on the run-extension, in served bytes. The page is measured
#: AFTER the strip (that is what crosses the IPC hop), so this is the number a
#: renderer actually pays: 1.5 MiB is about the p90 of today's UNSTRIPPED page
#: on this machine's real journals, and a stripped page of the same row count is
#: an order of magnitude below it.
OPEN_FRAME_MAX_BYTES = 1_572_864

#: How long a frame may WAIT for its run facts when no index is resident, before
#: answering ``runs_state: "building"``.
#:
#: The same budget ``checkpoints_view`` gives its own first paint, and the same
#: reasoning: a refresh that has a usable cache file costs 3/19/62 ms at
#: 5.9/35/118 MB (measured), so awaiting it is cheaper than making the client
#: condense a run twice; a genuinely COLD scan costs 28/169/559 ms at those sizes
#: (measured), which lands inside this budget for everything up to about 35 MB
#: and does not for the largest journals — those answer ``building`` and the next
#: frame carries the facts. Paying it once per session per process is the price of
#: an exact bar on the first paint, which is what this whole contract is for.
OPEN_FRAME_FACTS_WAIT_S = 0.2

#: How many rows the head hunt may ADD to the page beyond ``limit``.
#:
#: A ROW BUDGET, NOT A PAGE COUNT, and the first draft's page count was measured
#: wrong on this machine's own fixtures: with "one more page" the walk reached a
#: run's opening user row for almost nothing (runs of about five rows need a few
#: rows of slack, not a hundred) while every OTHER page paid 1.6-2.9x today's
#: bytes and still reported ``head_cut``. Stated in rows, the ordinary shape
#: reaches its head and the payload grows by the few rows it took; the cap is
#: what keeps a pathologically long run from pulling the whole journal.
OPEN_FRAME_MAX_EXTRA_ROWS = 100

#: A hard ceiling on the walk, whatever the budget: the page cannot cost more
#: reads than this for ONE frame. It cannot bind while the caps hold — the row
#: and byte caps below stop the walk far sooner — and it exists so no combination
#: of a large ``limit`` and a long run can turn one open into an unbounded walk.
OPEN_FRAME_MAX_RAW_PAGES = 6

#: ``provider_payload`` keys that no surface paints, and the byte share each one
#: holds on the real tail pages. The desktop reducer reads ``details``,
#: ``duration_s``, ``useless`` and ``harness_injected`` from this envelope and
#: nothing else; the three dropped keys have no reader in any of the three
#: client repositories (checked by grep, and named in the PR).
_DROPPED_PROVIDER_KEYS = ("native_replay", "system_fingerprint", "id")

#: Rows no surface paints, in the desktop reducer's own terms.
#:
#: HOW THIS LIST WAS BUILT, because it is a READER survey and not a byte-share
#: one. The reducer projects a durable row in ONE function
#: (``transcript-reducer.ts::durableRecord``), and that function ends every
#: non-message row it does not recognise: only ``completion_attention`` survives
#: among ``type: "custom"`` rows (``if (entry.type !== "message") return null``).
#: So a ``type: "custom"`` row whose type is not ``completion_attention`` cannot
#: paint, whatever it holds — and these are the ones that exist in real journals:
#: the frontend checkpoint (39.1% of the tail bytes on this machine's 40 largest
#: journals, in rows of about 349 KB), the spend receipts, ``session_state``,
#: ``system_prefix`` (3.2%) and ``selected_model``, and ``attention_started``.
#:
#: ``system_prefix`` IS ON THIS LIST DESPITE A COMMENT THAT SAYS OTHERWISE. A
#: module comment in ``transcript-rows.ts`` claims the prefix rows "paint" — the
#: audit's own strip list repeated it — but the gate above refuses them, and the
#: claim is about a shape this build does not write (a message-row custom). The
#: reducer is the authority; the check that caught it is the reason the
#: enumeration is done from each client's code rather than from a byte table.
#:
#: MESSAGE-row customs are a different shape with different rules (``kind:
#: "custom"``): the receipts, the job results, todo snapshots and
#: ``session_mcp_unavailable`` are all PAINTED, and the reducer's
#: ``SILENT_CUSTOM_TYPES`` names the ones that are not. Both spellings are here
#: because both appear in real journals for the same logical row.
_DROPPED_CUSTOM_TYPES = frozenset(
    {
        # type: "custom" rows (the harness's own bookkeeping envelopes)
        "frontend_state_checkpoint_v1",
        "session_spend.v1",
        "session_state",
        "system_prefix",
        "selected_model",
        "attention_started",
        # message-row customs the reducer lists as silent
        "hub_communication",
        "wake_schedule",
        "prune",
    }
)

#: ``type: "custom"`` rows that MUST be served whatever else is dropped: the one
#: custom the desktop paints from this shape. Named as an allow-list so the list
#: above can never take it out by accident.
_KEPT_CUSTOM_TYPE = "completion_attention"

#: A compaction's one-line preview cap. The marker's summary is a full model
#: output (about 700 KB on this machine) and the desktop renders a constant line
#: for the row; the preview exists for a surface that wants a caption.
_COMPACTION_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class FrameResult:
    """One open frame's page, and what the backend had to do to produce it."""

    entries: list[dict[str, Any]]
    has_more: bool
    cursor_missing: bool
    has_newer: bool | None
    #: Per-run facts for the runs intersecting the page (see ``publish_runs``).
    runs: list[dict[str, Any]]
    #: ``ready`` / ``building`` / ``unavailable`` / ``unsupported``.
    runs_state: str
    #: True when the run extension was refused by a cap: the page's oldest run
    #: has no opening user row on it, and ``runs`` is where its true size lives.
    head_cut: bool
    #: True when a cap ended the page at its OLDEST end: rows that were read
    #: (and are still on disk) are not in ``entries``. The caller MUST fold this
    #: into ``has_more`` — otherwise a reader is told the conversation begins
    #: where the cap cut and never pages back to rows it was not given.
    capped: bool = False
    #: Diagnostics for the evidence table and for tests: how many journal rows
    #: were decoded to produce this page, and how many were dropped by the strip.
    raw_rows: int = 0
    dropped_rows: int = 0


def strip_entry(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """One wire row reduced to what a surface paints, or ``None`` to drop it.

    The input is the desktop envelope (``{id, ts, ts_source, type, payload}``)
    and the output keeps that shape — a stripped row is the same kind of thing as
    an unstripped one, so a client's row reader is unchanged. Only FIELDS are
    removed, never renamed: every key the three clients read survives (the table
    in the PR names each read and the file it was found in).
    """
    etype = str(entry.get("type") or "")
    payload = entry.get("payload")
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    if etype == "custom":
        custom_type = str(payload.get("custom_type") or "")
        if custom_type in _DROPPED_CUSTOM_TYPES and custom_type != _KEPT_CUSTOM_TYPE:
            return None
        return {**entry, "payload": payload}
    if etype == "compaction":
        # The reducer pairs a live pass with this row by ``tokens_before`` (its
        # fingerprint: ``append_compaction`` writes no after-figure), renders a
        # constant line for it, and reads nothing else. The summary, the kept
        # window and the preserved user turns are replay material for the model,
        # not for a reader.
        preview = payload.get("summary")
        stripped: dict[str, Any] = {}
        if isinstance(payload.get("tokens_before"), (int, float)) and not isinstance(
            payload.get("tokens_before"), bool
        ):
            stripped["tokens_before"] = payload["tokens_before"]
        if isinstance(preview, str) and preview:
            stripped["preview_text"] = " ".join(preview.split())[:_COMPACTION_PREVIEW_CHARS]
        return {**entry, "payload": stripped}
    if etype != "message":
        # Every other entry type (a `prune` marker, an unknown future type) is
        # served verbatim: an unknown row is not this module's to judge, and a
        # client already ignores what it does not recognise.
        return {**entry, "payload": payload}
    provider = payload.get("provider_payload")
    if isinstance(provider, Mapping):
        trimmed = {k: v for k, v in provider.items() if k not in _DROPPED_PROVIDER_KEYS}
        if trimmed:
            payload["provider_payload"] = trimmed
        else:
            # A row whose envelope held ONLY replay material carries no envelope
            # at all: ``{}`` is not a value any client reads, and leaving it would
            # make "the producer sent nothing" and "the producer sent only
            # material no surface paints" indistinguishable on the wire.
            payload.pop("provider_payload", None)
    # ``usage`` is the provider's accounting envelope: no surface reads it from
    # a history row (the spend panel reads the projection's own usage and the
    # `session_spend.v1` rows the daemon keeps), and it is 1.1% of real tail
    # bytes.
    payload.pop("usage", None)
    return {**entry, "payload": payload}


def paintable(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The rows of ``rows`` a transcript paints, in order, stripped.

    The ONE definition of a paintable row on this plane, so the count a page is
    cut by and the rows the page serves cannot disagree: a row is paintable when
    it survives :func:`strip_entry`. The server-side visibility rule
    (``harness.rows.visible_transcript_rows``) has already been applied by the
    caller, which is why this is a pure filter over the strip and not a second
    predicate beside it.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        stripped = strip_entry(row)
        if stripped is not None:
            out.append(stripped)
    return out


def tail_head_reachable(
    index: TranscriptIndex,
    *,
    limit: int,
    extra_rows: int = OPEN_FRAME_MAX_EXTRA_ROWS,
) -> bool:
    """Whether the TAIL run's opening user row can be reached by the head hunt.

    THE CHEAPEST GUARD IN THE FRAME, and the one that keeps a long run from
    taxing every page. Without it the walk is blind and pays its whole budget
    discovering that a run of 600 rows has no head within reach — measured on the
    S3 fixture as 231 paintable rows and 264 KB against today's 100 rows and
    110 KB, for a ``head_cut: true`` answer either way. With it, the index (which
    already counts every row) answers the question in one subtraction BEFORE the
    extra reads: if the tail run's head is further above the journal's end than
    the walk could go, the frame serves today's row count and lets ``runs`` carry
    the run's true size — which is exact, and free.

    ``False`` for a tail run with no opening user row (a wake or hub run): there
    is no head to reach, so extending is pure cost. ``scan.rows`` is the
    journal's own row count in the same ordinal space as a run's ``first_seq``.
    """
    if not index.runs:
        return False
    tail = index.runs[-1]
    if not tail.opening_user_id:
        return False
    rows_above_head = max(0, index.scan.rows - tail.first_seq)
    # One row of slack for the ordinals themselves: ``scan.rows`` is a COUNT, so
    # the last row's ordinal is one less, and a run whose head sits exactly at the
    # budget's edge should be reached rather than refused.
    return rows_above_head <= limit + extra_rows + 1


def head_reached(paintable_rows: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the page's oldest row is a run's opening user row.

    The extension's stop condition, stated in ROWS rather than in the index's
    turns on purpose: it is a fact about the page the client will paint, it holds
    for a conversation whose index has never been built, and it is the client's
    own run opener (``walkTurns``). A page whose oldest row is anything else may
    be a run's middle, which is what ``head_cut`` exists to say out loud.
    """
    if not paintable_rows:
        return True
    oldest = paintable_rows[0]
    payload = oldest.get("payload")
    return (
        str(oldest.get("type") or "") == "message"
        and isinstance(payload, Mapping)
        and str(payload.get("role") or "") == "user"
    )


def trim_to_limit(rows: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    """The newest ``limit`` PAINTABLE rows of ``rows``, oldest first.

    THE HONEST ANSWER WHEN THE HEAD HUNT FAILS. A page whose oldest run could not
    be completed — the operator's 600-row run, whose head is 500 rows above
    anything a sane budget reaches — is a page whose oldest run is a fragment
    either way, so the rows the extension added buy the client nothing: it still
    cannot condense that run from them, and ``runs`` states its true size. Serving
    them anyway was measured at 2.3x the payload of the page the client asked for
    on the S3 fixture (231 paintable rows, 264 KB against 100 rows and 114 KB) for
    no boundary and no exactness.

    The trim runs from the NEWEST end for the same reason the caps do: the rows a
    reader cannot lose are the recent ones.
    """
    kept: list[Mapping[str, Any]] = []
    for row in reversed(list(rows)):
        if strip_entry(row) is None:
            continue
        kept.append(row)
        if len(kept) >= limit:
            break
    kept.reverse()
    return kept


def served_bytes(entries: Sequence[Mapping[str, Any]]) -> int:
    """The bytes a page costs on the wire, measured the way the wire measures.

    Compact separators and no ``sort_keys`` — the same ``json.dumps`` shape the
    route serialises with — so a cap stated in bytes cannot be defeated by the
    encoder's whitespace. One row at a time so a pathological row cannot make
    the accounting itself the cost.
    """
    return sum(len(json.dumps(entry, separators=(",", ":"))) for entry in entries)


def _run_start_ordinal(runs: Sequence[RunRecord], seq: int) -> int:
    """The index of the last run whose first row is at or before ``seq``."""
    firsts = [run.first_seq for run in runs]
    return bisect_right(firsts, seq) - 1


def publish_runs(
    index: TranscriptIndex,
    *,
    first_seq: int | None,
    last_seq: int | None,
    limit: int = OPEN_FRAME_MAX_ROWS,
) -> list[dict[str, Any]]:
    """The facts of every run that intersects the page, oldest first.

    ``first_seq``/``last_seq`` are the page's own row ordinals, and either may be
    ``None`` when the caller cannot name one — a page read through the shared
    page cache does not hand out ordinals, so the desktop passes the ordinals it
    can derive from the index (the oldest checkpoint row it holds and the newest
    run it reaches). The window is then a SUPERSET of the page's runs rather than
    an exact intersection, which is deliberate: a client looks a run up by an id
    it is holding, so an extra run in the list costs bytes and never a wrong
    answer, while a missing one costs the feature. The list is capped at
    ``limit`` runs, and the cap cannot bind in practice — a page is bounded to
    ``OPEN_FRAME_MAX_ROWS`` paintable rows, so it cannot span more runs than
    that unless its runs are one row each.
    """
    if not index.runs:
        return []
    runs = index.runs
    if first_seq is None:
        start = 0
    else:
        start = max(0, _run_start_ordinal(runs, first_seq) - 1)
    out: list[dict[str, Any]] = []
    for run in runs[start:]:
        if last_seq is not None and run.first_seq > last_seq:
            break
        out.append(run_payload(run))
        if len(out) >= limit:
            break
    return out


def run_payload(run: RunRecord) -> dict[str, Any]:
    """One run record as the wire states it, field by field.

    BUILT FIELD BY FIELD rather than by dumping the dataclass, which is the same
    rule the checkpoints manifest follows: the record carries internal
    bookkeeping the incremental scan resumes from (``last_painter``,
    ``saw_work``), and a ``model_dump`` here would publish it as a client
    contract the day someone adds one more.
    """
    payload: dict[str, Any] = {
        # The client's own run identity: the closing answer's id when the run has
        # one, else its last row's id (``transcript-rows.ts::runsOf``). It is the
        # key a bar is remembered by and the first thing a client matches on.
        "run_key": run.closing_answer_id or run.last_id,
        "opening_user_id": run.opening_user_id or None,
        "closing_answer_id": run.closing_answer_id or None,
        "settled": run.settled,
        "outcome": run.outcome,
        "complete": run.complete,
        "started_ts": run.start_ts,
        "ended_ts": run.end_ts,
    }
    if run.settled:
        # A live tail states NO counts: its rows are still arriving, and a
        # number taken now is one the client would have to correct — the
        # after-paint change this whole contract removes.
        payload.update(
            {
                "action_count": run.action_count,
                "failed_count": run.failed_count,
                "worked_seconds": round(run.worked_seconds, 3),
            }
        )
    else:
        payload.update({"action_count": None, "failed_count": None, "worked_seconds": None})
    return payload


def build_frame(
    raw_entries: Iterable[Mapping[str, Any]],
    *,
    index: TranscriptIndex | None = None,
    bounds: tuple[int | None, int | None] | None = None,
    runs_state: str | None = None,
    head_cut: bool = False,
    has_more: bool = True,
    cursor_missing: bool = False,
    has_newer: bool | None = None,
) -> FrameResult:
    """Strip, cap and publish one page from rows already read, newest last.

    The caps apply to the SERVED page — the bytes that cross the IPC hop — and
    ``head_cut`` is set by the caller when it asked for the extension and could
    not reach the head. Rows are dropped, never truncated: a half-row would be a
    shape no client's reader has ever seen.
    """
    ordered = list(raw_entries)
    raw_rows = len(ordered)
    dropped = sum(1 for raw in ordered if strip_entry(raw) is None)
    # FROM THE NEWEST END, because the caps decide what a page can hold and the
    # rows a reader cannot lose are the recent ones: a page truncated at its
    # OLDEST end is a page with a short tail, which is the defect this contract
    # exists to remove. The single-row exception is deliberate — a first row that
    # exceeds the cap by itself is still served, because a page with no rows is
    # not a smaller page, it is a blank transcript, and today's reader serves
    # that row too.
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    capped = False
    for raw in reversed(ordered):
        stripped = strip_entry(raw)
        if stripped is None:
            continue
        size = len(json.dumps(stripped, separators=(",", ":")))
        if entries and (
            len(entries) >= OPEN_FRAME_MAX_ROWS or total_bytes + size > OPEN_FRAME_MAX_BYTES
        ):
            capped = True
            break
        entries.append(stripped)
        total_bytes += size
    entries.reverse()
    runs: list[dict[str, Any]] = []
    state = runs_state or "unavailable"
    if index is not None:
        # ``bounds is None`` with an index in hand is not a failure: it is a page
        # that holds no run a client could match a fact by (see
        # ``page_seq_bounds``). It answers ``ready`` with an empty list, because
        # the client's behaviour for an unmatched run is to keep its own
        # condensation — the same thing it does for a run with no facts — while
        # ``building`` would promise facts the next frame cannot have.
        runs = (
            publish_runs(index, first_seq=bounds[0], last_seq=bounds[1])
            if bounds is not None
            else []
        )
        state = "ready"
    return FrameResult(
        entries=entries,
        has_more=has_more,
        cursor_missing=cursor_missing,
        has_newer=has_newer,
        runs=runs,
        runs_state=state,
        head_cut=head_cut,
        capped=capped,
        raw_rows=raw_rows,
        dropped_rows=dropped,
    )


def page_seq_bounds(
    index: TranscriptIndex, entries: Sequence[Mapping[str, Any]], *, reaches_eof: bool
) -> tuple[int | None, int | None] | None:
    """The row-ordinal window the runs of this page are found in.

    THE PAGE DOES NOT CARRY ORDINALS, so the bounds come from the rows on it that
    the index does know: a checkpoint row's ``seq`` is its ordinal, and the ids on
    a page are its user rows and its completions — the two rows a client uses to
    match a run. ``None`` means NO FACTS ARE POSSIBLE for this page, and the
    caller states that honestly rather than guessing: a page holding no
    checkpoint row is a slice entirely inside runs, so its fragments have no id a
    fact could be matched by, and emitting the runs of some other region would be
    noise wearing the name of data.

    The window is a SUPERSET by one run on each side (see :func:`publish_runs`):
    the page's oldest row can be the middle of a run whose head is above it, and
    its newest can be inside a run that started after the newest checkpoint. A
    client looks a run up by an id it already holds, so an extra run costs bytes
    and a missing one costs the feature.
    """
    if not index.runs:
        return None
    known = {checkpoint.id: checkpoint.seq for checkpoint in index.checkpoints}
    ids = (str(entry.get("id") or "") for entry in entries)
    seqs = [known[row_id] for row_id in ids if row_id in known]
    if not seqs:
        return None
    first_seq = min(seqs)
    if reaches_eof:
        return (first_seq, None)
    containing = _run_start_ordinal(index.runs, max(seqs))
    following = containing + 1
    if following < len(index.runs):
        return (first_seq, index.runs[following].first_seq)
    return (first_seq, None)


def run_head_of(index: TranscriptIndex, row_id: str) -> str | None:
    """The opening user row of the run that ``row_id`` belongs to, or ``None``.

    THE INDEX DOES NOT CARRY ROW MEMBERSHIP, so this answers the question the
    extension needs from the rows it does carry: the run whose span contains a
    CHECKPOINT row is found by ordinal, and its opening user id is its head. A
    row that is not a checkpoint (a tool row, a mid-run assistant row) has no
    ordinal here and answers ``None`` — the caller then simply does not extend,
    which is today's page.
    """
    seq = _checkpoint_seq(index, row_id)
    if seq is None:
        return None
    for run in index.runs:
        if run.first_seq <= seq <= run.last_seq:
            return run.opening_user_id or None
    return None


def _checkpoint_seq(index: TranscriptIndex, row_id: str) -> int | None:
    """The ordinal of a checkpoint row with this id, or ``None``.

    A linear scan over the manifest is the right cost here: it runs once per
    extension attempt (never per row), and the manifest is small by
    construction — one entry per user turn and per completion, not per row.
    """
    for checkpoint in index.checkpoints:
        if checkpoint.id == row_id:
            return checkpoint.seq
    return None
