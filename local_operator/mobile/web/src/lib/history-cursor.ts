/**
 * The first `GET /history` cursor for a projection, and the rules it rests on.
 *
 * WHY THIS IS A MODULE (QA round 2, Q2-1 on PR #4584). Two readers page the
 * journal backwards from the projection: the transcript's back-fill and the
 * composer's moved-tail wake confirmation. The composer took
 * `transcript[0].id`, which on a capped projection is the PINNED OPENER — older
 * than every row in the gap — so its first page came back empty and a
 * delivered wake read as "unconfirmed". The transcript had already learnt both
 * halves of the rule (skip the pinned head; skip ids the journal never holds).
 * One definition, so the two cannot drift apart again.
 */
import type { TranscriptEntry } from "../types";

/* The daemon's ``PROJECTION_TRANSCRIPT_LIMIT`` (``local_operator/mobile/types.py``):
   the projection is the pinned opener plus the newest 79 rows, so a transcript
   this long is a CAPPED window whose head row is not the tail's chronological
   neighbour. Mirrored rather than fetched because the rule is about the SHAPE of
   what arrived, and ``agent-view.tsx`` mirrors the same number for the child
   sheet. It was compared against ``PAGE`` (120) before — a number the daemon
   never sends — so the pinned-opener branch never ran on a real projection and
   the first cursor was the opener itself: the page came back as the opener's own
   neighbour plus the journal BELOW the cut, and the turns in between were never
   requested (measured with the S6 first-paint fixture: 22 user turns, 272 rows;
   on S3, 28 turns and 839 rows). ``>=`` rather than ``===`` is safe because the
   head row is already held either way — paging from the row after it cannot lose
   it, only re-offer it to the merge's id de-dup. */
export const PROJECTION_ROWS = 80;

/** Every row id the LIVE fold mints, which no journal row can ever match. The
 *  fold creates them for content that is not a journal entry: reasoning
 *  (`rz-`, never persisted), tool rows (`tc-`), notices (`nt-`, `rt-`, `err-`,
 *  `rej-`), compaction (`cx-`) and the peer families' `peer-`/`nonpeer-`
 *  fallbacks when a custom message carried no id — all timestamp- or
 *  call-keyed (`local_operator/mobile/projection.py`, the `id=f"<p>-…"`
 *  sites). The child sheet also synthesises a head row `prompt:<job_id>`
 *  (`agent-view.tsx`). A cursor naming any of these is answered as
 *  end-of-history. */
const LIVE_ONLY_ROW_PREFIXES = ["rz-", "tc-", "nt-", "rt-", "err-", "rej-", "cx-", "peer-", "nonpeer-", "prompt:"];

/** Whether a rendered row's id can name a journal entry — the only kind a
 *  history cursor may use. Anything else is a cursor the daemon can only
 *  answer as end-of-history. */
export function journalAddressableRow(entry: TranscriptEntry | undefined): boolean {
	return entry !== undefined
		&& entry.kind !== "reasoning"
		&& !LIVE_ONLY_ROW_PREFIXES.some((prefix) => entry.id.startsWith(prefix));
}

/** Whether the projection's head row is the pinned opener: the wire's own
 *  `transcript_head_pinned` when the daemon sends it, else (older daemon) the
 *  length-and-kind fallback — a false positive there costs one row of overlap,
 *  which every reader's id de-dup discards. */
export function projectionHeadPinned(entries: readonly TranscriptEntry[], headPinned?: boolean): boolean {
	return headPinned ?? (entries.length >= PROJECTION_ROWS && entries[0]?.kind === "user");
}

/** The id to ask for history BEFORE: the oldest journal-addressable row, below
 *  the head when `skipHead` (the pinned opener is not the tail's neighbour).
 *  Falls back to the first row — a cursor the daemon answers as end of history,
 *  which every caller already treats as "nothing older". */
export function firstHistoryCursor(rows: readonly TranscriptEntry[], skipHead: boolean): string | null {
	return rows.slice(skipHead ? 1 : 0).find(journalAddressableRow)?.id ?? rows[0]?.id ?? null;
}
