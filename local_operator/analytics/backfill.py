"""One-time recovery of session names the ledger never heard about.

Until now the analytics ledger learned a session's human name from exactly one
place — ``Session.set_conversation_name`` — so a name only reached it when it
was set LIVE in that process. Two ordinary paths never went through it (a
``--resume`` restores the title by writing the holder's fields directly, and the
opener-derived stand-in is deliberately host-side), and a session whose naming
call failed had no name to mirror at all. Those gaps are closed at the source
now, but the fix only applies going FORWARD: a ledger that has been accumulating
for months still renders its most expensive rows as bare 12-hex ids, which is
the state the operator actually reported.

This sweep repairs the existing rows. It is the same shape, and exists for the
same reason, as ``resume.backfill_session_titles``: a correct change that
appears to do nothing until each session's next rename is not much of a fix, so
one bounded pass at startup makes the history readable immediately.

Where the name comes from: ``resume.session_name`` — the SAME function that
labels a ``/resume`` picker row, which is what makes the two surfaces agree by
construction rather than by coincidence. It prefers the journalled title and
falls back to the opening user message, so a session that was never successfully
named still recovers the label a human would recognise it by.

EXCEPT FOR SUBAGENTS, which are most of the unnamed rows and which that function
alone labels uselessly here. A delegated session's opening message is its ROLE
PROMPT, and those are boilerplate: on the operator's ledger, naively labelling
the 473 recoverable sessions produced 466 rows that shared their rendered label
with another row — 93 of them reading ``[team: lopdev] You are reviewer``. That
is worse than the hex ids it replaces, because two identical rows cannot be told
apart at all while two hex ids at least address different sessions. So a
subagent is labelled from what actually distinguishes it: its ROLE, plus the
TITLE OF THE CONVERSATION THAT DELEGATED IT, which the ledger already knows via
``calls.parent_session_id``. 461 of the 467 subagent rows have a parent carrying
a real title, giving ``reviewer · Fix usage data not updating`` — a row a reader
can place. The remainder fall back to the opener excerpt.

Written at ``SESSION_NAME_RANK_BACKFILL``, which cannot displace a real title.
The sweep reads a session's title off disk and cannot tell whether it was
user-set, so ranking it any higher would let a startup pass overwrite a rename
that had not yet been journalled.

Measured on the operator's real ledger (871 sessions, 475k calls): 608 sessions
had no ledger name, 473 of them recoverable from disk, covering $9,389 of spend
that rendered as anonymous hex ids. The remaining 135 have had their session
directory pruned and are unrecoverable by any means — the ledger outlives the
transcripts by design (90-day ledger retention against a session store the user
prunes on their own schedule), which is the argument for mirroring the name at
the time it is known rather than relying on a sweep like this one.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from local_operator.analytics.store import SESSION_NAME_RANK_BACKFILL, AnalyticsStore

logger = logging.getLogger("local_operator.analytics.backfill")

#: Pulls the ROLE out of a delegated session's opening message. Both spellings
#: the harness emits are covered: the ``[role: coder]`` / ``[team: lopdev]``
#: header a task prompt opens with, and the ``You are qa-tester on this team``
#: sentence a team brief uses when the header names the team instead of the job.
#: Deliberately a small regex over the opener rather than a read of the roster
#: sidecar: this runs over hundreds of directories at startup, and the opener is
#: already in hand from ``session_name``.
_ROLE_RE = re.compile(
    r"\[role:\s*([^\]]{1,32}?)\s*\]|You are (?:an? |the )?([a-z][a-z\-]{1,20}) on this team",
    re.IGNORECASE,
)

#: Cap on the parent title quoted into a child's label, so the composed name
#: still says something after ``short_session_label`` truncates the row to 32
#: characters for the table.
_PARENT_TITLE_CHARS = 40

#: How many names one pass may WRITE. The sweep runs on the store-maintenance
#: thread beside the origin/title backfills and shares their discipline: bound
#: the work, never the reach. Sized so a first run on a large ledger finishes in
#: one pass (the operator's worst case was 473) while a pathological store still
#: yields the thread promptly; whatever is left is picked up next launch, since
#: every pass re-derives its own worklist.
DEFAULT_BACKFILL_LIMIT = 1000


def backfill_analytics_session_names(
    config_dir: Path,
    *,
    limit: int = DEFAULT_BACKFILL_LIMIT,
    store: AnalyticsStore | None = None,
) -> int:
    """Name the ledger's unnamed sessions from their transcripts; return how many.

    Bounded by the LEDGER, not by the session store: the worklist is the set of
    session ids that have call rows but no name, so the sweep never opens a
    directory for a session that cost nothing and never mints a row for a
    session analytics has never seen. A session whose directory is gone is
    simply skipped — there is nothing left on disk to recover a name from.

    Best-effort in the strongest sense, like every other maintenance pass: this
    may fail in any way at all without disturbing the session that triggered it.
    """
    # Imported here rather than at module scope: ``resume`` is import-guarded to
    # stay free of the engine, and this keeps the dependency pointing one way
    # (analytics may read resume; resume must never import analytics).
    from local_operator.resume import session_name

    store = store if store is not None else AnalyticsStore()
    try:
        pending = store.sessions_missing_names()
    except Exception:  # noqa: BLE001 — an unreadable ledger is a no-op sweep
        logger.debug("analytics: could not read the backfill worklist", exc_info=True)
        return 0
    if not pending:
        return 0
    try:
        parents = store.session_parents()
        known = store.session_names_map()
    except Exception:  # noqa: BLE001 — without the tree, labels are opener-only
        logger.debug("analytics: could not read the session tree", exc_info=True)
        parents, known = {}, {}

    sessions = config_dir / "sessions"
    written = 0
    # Sorted so a store too large for one pass makes deterministic progress
    # rather than re-drawing a random subset each launch.
    for session_id in sorted(pending):
        if written >= limit:
            break
        directory = sessions / session_id
        try:
            if not directory.is_dir():
                continue
            opener = session_name(directory)
            if not opener:
                continue
            name = _compose_label(
                session_id, opener, parents=parents, known_names=known, sessions=sessions
            )
        except Exception:  # noqa: BLE001 — a corrupt transcript costs one row
            logger.debug("analytics: could not name %s", session_id, exc_info=True)
            continue
        if not name:
            continue
        store.upsert_session_name(session_id, name, rank=SESSION_NAME_RANK_BACKFILL)
        written += 1
    return written


def _compose_label(
    session_id: str,
    opener: str,
    *,
    parents: dict[str, str],
    known_names: dict[str, str],
    sessions: Path,
) -> str:
    """The label a recovered row should carry.

    An ordinary session keeps its own opener/title — that is the name its user
    would search for. A DELEGATED session gets ``<role> · <parent title>``
    instead, because its own opener is a shared role prompt and would render
    identically to every sibling's (see the module docstring).

    Degrades in steps rather than all at once: no parent title → role alone; no
    role → the opener excerpt, which is still better than a hex id.
    """
    parent_id = parents.get(session_id, "")
    role = _role_from_opener(opener)
    if not parent_id and not role:
        return opener
    parent_title = known_names.get(parent_id, "")
    if parent_title:
        # Read from the ledger, not from disk: the parent's own row is the name
        # every other surface shows it under, and the parent's directory may
        # well have been pruned while its ledger rows survive.
        parent_title = parent_title[:_PARENT_TITLE_CHARS].strip()
    if role and parent_title:
        return f"{role} · {parent_title}"
    if role:
        return f"{role} · {parent_id[:12]}" if parent_id else role
    if parent_title:
        return f"subagent · {parent_title}"
    return opener


def _role_from_opener(opener: str) -> str:
    """The delegated role named in an opening message, or ``""``.

    Returning ``""`` for an ordinary conversation is what keeps this from
    rewriting a real user session's label: only a role prompt matches.
    """
    match = _ROLE_RE.search(opener)
    if not match:
        return ""
    role = (match.group(1) or match.group(2) or "").strip().lower()
    # A ``[team: lopdev]`` header names the TEAM, not the job, and the job then
    # appears in the sentence the second alternative matches. Reject anything
    # with whitespace or punctuation: a real role is a single bare word.
    if not role or not role.replace("-", "").isalpha():
        return ""
    return role
