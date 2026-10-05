/**
 * The queued-ask vocabulary — the phone's half of the shared copy contract.
 *
 * WHY ONE MODULE. Every ask state (open / answered / timed out / late /
 * declined / dismissed / withdrawn / expired) is spelled in exactly one string
 * per state, and four different places on this surface need to say it: the
 * minimized bar above the composer, the ask card in the sheet, a session row's
 * outstanding chip, and the response card in the transcript. Four call sites
 * each composing its own sentence is how one state grows two wordings — and
 * the wording is the whole feature here, because the user's next action depends
 * on whether an ask is still waiting, already answered, or past its deadline.
 *
 * THE WORDS COME FROM `docs/design/ask-nonblocking.md` §5, verbatim, and that
 * is deliberate rather than lazy: the TUI and the desktop card show the same
 * states, and a reader moving between surfaces must not have to work out
 * whether "delivering" and "sent" mean the same thing. The design states the
 * strings once and puts them in each surface's PR; this file is this surface's
 * copy of that one table.
 *
 * TWO RULES THE STRINGS ENFORCE, both from §5:
 *
 *  * **No surface may claim a notification that did not happen.** `delivered`
 *    is the runtime's statement that the response rows reached the transcript —
 *    not that a human read anything — so an answered-but-undelivered ask says
 *    "delivering", and only a `delivered` row says "the agent was told".
 *  * **An expiry is not a failure.** The expired line names the remedy ("ask the
 *    agent again") and the card disables its controls without an error register.
 *
 * `open` is not "waiting for you": the agent keeps working while an ask is
 * queued, so the open line says so in as many words ("the agent is
 * continuing"). The one place the phone is allowed to sound like a decision is
 * the minimized bar's count, and that bar is labeled by the design as an ask
 * summary, not a "needs you" badge.
 */
import type { AskQuestion, PendingAsk } from "../types";

/** The statuses the wire carries (design §4's frozen `PendingAsk.status`). */
export type AskStatus =
	| "open"
	| "answered"
	| "declined"
	| "timed_out"
	| "late"
	| "dismissed"
	/* The agent retracted the question (design §12): settled, never outstanding. */
	| "withdrawn"
	| "expired";

/** The ink a state line is drawn in. Four states, five readings: `waiting` is
 *  an ask whose deadline has passed locally but which the runtime still reports
 *  `open` — the honest intermediate, never shown as settled. */
export type AskTone = "waiting" | "settled" | "gone" | "attention";

/** Whether the card offers answer controls for this state.
 *
 *  `timed_out` IS answerable and that is the point of the state: the agent moved
 *  on, but the user's answer still reaches it as a `late` response. Everything
 *  already settled (answered/late/declined/dismissed/withdrawn) and everything
 *  too old to carry (expired) is not.
 */
export function isAnswerable(status: string): boolean {
	return status === "open" || status === "timed_out";
}

/** Whether an ask still belongs in the "outstanding" surfaces — the minimized
 *  bar and the session row's chip. Answered/declined asks leave at once: their
 *  work is done, and keeping them would make the chip a log.
 *
 *  `expired` and `dismissed` are excluded for the same reason as the aggregate
 *  route excludes them (`asks/store.index_asks`): neither can be acted on. */
export function isOutstanding(status: string): boolean {
	return isAnswerable(status);
}

/** Whether an ask's conversation is DEAD — every op on it is a terminal refusal.
 *
 *  THE TWO FIELDS ARE NOT INTERCHANGEABLE, and reading one alone has now caused
 *  a bug in each direction. `durable: false` says nothing will ever READ an
 *  answer (the transcript is gone); `runtime_live: false` says nothing can be
 *  DELIVERED to it right now. A live conversation with no transcript is both
 *  false, yet answering it works — the route's cold arm is scoped
 *  `not _session_is_live(...)`, so a live row never reaches the refusal — and a
 *  non-durable row that is merely cold is refused. Only the conjunction is the
 *  state the route answers with `ask_session_gone`, so this is the ONE
 *  predicate the card's pre-emptive terminal state, the sheet's cold strip and
 *  its `open` control all share.
 *
 *  `runtime_live !== true` rather than `=== false`: the route's own test is
 *  "not live", so an absent field (an older daemon) belongs on the same side as
 *  a false one. Absent `durable` is NOT dead — a reader must never withdraw an
 *  affordance it cannot vouch against. */
export function isDeadConversation(row: PendingAsk): boolean {
	return row.durable === false && row.runtime_live !== true;
}

/** How long until this ask's deadline, in milliseconds — negative once past.
 *  Epoch MILLISECONDS on both sides (`created_at`/`expires_at` are `now_ms()`),
 *  unlike the seconds-based `formatRelative`. */
export function remainingMs(row: PendingAsk, nowMs: number): number {
	const expires = Number(row.expires_at) || 0;
	return expires - nowMs;
}

/** A deadline as a short span: `42 m`, `3 h`, `2 d`.
 *
 *  Spaced units, matching §5's own spelling ("expires in 42 m") rather than
 *  `formatElapsed`'s compact TUI durations: this is a countdown a thumb reads at
 *  a glance, not an elapsed-time column that has to line up in a table. The
 *  boundaries are chosen so the number never reads as more precise than it is —
 *  minutes under 90, hours under 48, days after that.
 */
export function durationLabel(ms: number): string {
	const seconds = Math.floor(ms / 1000);
	/* SUB-SECOND IS NOT NOTHING (agent review round 1, N1). This returned ""
	   for a deadline inside the next second, so the open line read "expires in"
	   with no value at all — a sentence whose whole point is the number. The
	   floor is stated as the bound it is rather than rounded to "1 m", which
	   would claim a minute that has already nearly passed. */
	/* A deadline already past is the caller's other branch (""), not a countdown. */
	if (ms <= 0) return "";
	if (seconds < 60) return "<1 m";
	if (seconds < 90 * 60) return `${Math.floor(seconds / 60)} m`;
	const hours = seconds / 3600;
	if (hours < 48) return `${Math.floor(hours)} h`;
	return `${Math.floor(hours / 24)} d`;
}

/** The rows in the order every ask surface should show them: the ask the CHIP
 *  NAMES first, then the wire's own order.
 *
 *  WHY THE ORDER IS THE CLIENT'S TO DECIDE (UX round 1, U8). The chip names the
 *  head ask, and the published list leads with the NEWEST — so tapping a chip
 *  that says "Which sequencing…" opened a sheet whose first card was a different
 *  question, with the named one below the fold. A thumb arrives expecting what
 *  it just read. The design's A2 addendum already names this divergence and its
 *  fix ("a later PR may make the list lead with the oldest too"); this is that
 *  fix, applied where the reader stands, with no wire change.
 *
 *  THE LIFTED ROW IS `dockAsk`, NOT `headAsk` (agent review round 2, N2): with
 *  nothing open, the chip names the first still-answerable TIMEOUT (U7's fix),
 *  so lifting only the open-ask head left the two surfaces naming different rows
 *  in exactly the state U7 exists for. One rule, both callers. */
export function orderedForDisplay(rows: PendingAsk[] | undefined | null): PendingAsk[] {
	const list = Array.isArray(rows) ? rows : [];
	const first = dockAsk(list);
	if (first === null) return [...list];
	return [first, ...list.filter((row) => row.ask_id !== first.ask_id)];
}

/** The one line a queued ask's own state is stated in (§5's shared copy).
 *
 *  `nowMs` is the CLIENT's clock and the countdown is rendered from it, per §5 —
 *  the phone may be minutes away from the runtime's clock, and a countdown the
 *  server computed at push time would freeze at whatever it said then. */
export function askStateLine(row: PendingAsk, nowMs: number): { text: string; tone: AskTone } {
	const status = String(row.status || "open");
	const left = remainingMs(row, nowMs);
	switch (status) {
		case "open": {
			if (left <= 0) {
				/* The runtime has not folded the deadline yet (it folds on its own
				   ≤60s tick), so this is a race window measured in seconds — but it
				   is also the user's real situation, and claiming "expires in 0 m"
				   or silently showing the pre-deadline line would both be untrue.
				   The ask is still answerable either way, so the remedy is the same
				   answer the timed-out line gives. */
				return { text: "Queued — the agent is continuing; deadline passed", tone: "attention" };
			}
			return {
				text: `Queued — the agent is continuing; expires in ${durationLabel(left)}`,
				tone: "waiting",
			};
		}
		case "answered":
			/* `delivered` is the runtime's own statement that the response rows
			   exist in the transcript; until then the honest word is "delivering"
			   (see the module note). */
			return row.delivered
				? { text: "Answered — the agent was told", tone: "settled" }
				: { text: "Answered — delivering", tone: "settled" };
		case "late":
			return { text: "Answered late — the agent was told", tone: "settled" };
		case "declined":
			return { text: "Declined — the agent was told", tone: "settled" };
		case "dismissed":
			return { text: "Dismissed — no reply was sent", tone: "gone" };
		case "withdrawn":
			/* DESIGN §12's word, verbatim — the ONE new copy this status needs on
			   this surface. Like `dismissed`, the ask is over with nothing to send;
			   unlike it, the asker retracted the question rather than the user. */
			return { text: "Withdrawn — no longer needed", tone: "gone" };
		case "expired":
			return {
				text: "Expired — this ask is too old to answer; ask the agent again",
				tone: "gone",
			};
		case "timed_out":
			return {
				text: "Timed out — the agent moved on; you can still answer",
				tone: "attention",
			};
		default:
			/* A status from a newer runtime. Passing it through as its own word is
			   the same rule the projects sheet follows for an unknown project
			   status: an unknown state must not be silently rendered as a known
			   one, and it must not crash the card either. */
			return { text: status.replace(/_/g, " "), tone: "waiting" };
	}
}

/** The asks worth showing on the outstanding surfaces, newest first (the wire's
 *  own order is open-first newest-first — `index_asks`' sort — and this keeps a
 *  single order rather than re-ranking). */
export function outstandingAsks(rows: PendingAsk[] | undefined | null): PendingAsk[] {
	if (!Array.isArray(rows)) return [];
	return rows.filter((row) => isOutstanding(String(row?.status || "open")));
}

/** The ask the bar should NAME: the oldest OPEN ask, or — when nothing is open
 *  — the first still-answerable one (a timed-out ask is answerable, §5).
 *
 *  WHY THE FALLBACK (UX round 1, U7). With only a timed-out ask left, the bar
 *  said "? 1 question waiting" with no preview at all: the count includes the
 *  answerable timeout while the preview did not, so the one state R7 keeps
 *  answerable was also the one state that named nothing. The head rule is
 *  unchanged where an open ask exists. */
export function dockAsk(rows: PendingAsk[] | undefined | null): PendingAsk | null {
	const head = headAsk(rows);
	if (head !== null) return head;
	const outstanding = outstandingAsks(rows);
	return outstanding.length > 0 ? outstanding[0] : null;
}

/** The HEAD ask: the OLDEST still-open ask.
 *
 *  Deliberately NOT `rows[0]`. The published list leads with the NEWEST open
 *  ask, while a bar that jumped to each new arrival would move under the user's
 *  finger mid-tap; the design names this divergence (§4's A2 addendum, and
 *  `asks/render.mirror_card`, which picks the same head for the legacy mirror).
 *  A `timed_out` ask is not a head: it is still answerable but it is no longer
 *  the thing the agent is waiting on, so the bar's count and its headline must
 *  not be built from it.
 */
export function headAsk(rows: PendingAsk[] | undefined | null): PendingAsk | null {
	const list = Array.isArray(rows) ? rows : [];
	let head: PendingAsk | null = null;
	for (const row of list) {
		if (String(row?.status || "") !== "open") continue;
		if (
			head === null ||
			Number(row.created_at) < Number(head.created_at) ||
			(Number(row.created_at) === Number(head.created_at) &&
				String(row.ask_id) < String(head.ask_id))
		) {
			head = row;
		}
	}
	return head;
}

/** The questions still to answer on this ask: the ones the log does not hold and
 *  the legacy draft has not taken. Mirrors `render.mirror_card`'s rule, because
 *  a phone that offered a question another surface already drafted would collect
 *  a second answer to it. */
export function unansweredQuestions(row: PendingAsk): AskQuestion[] {
	const questions = Array.isArray(row.questions) ? row.questions : [];
	const taken = new Set(Object.keys(row.answers ?? {}));
	for (const id of row.draft_question_ids ?? []) taken.add(String(id));
	return questions.filter((q) => !taken.has(String(q?.id || "")));
}

/** The labels chosen for one question, as the answer map holds them. A secret
 *  answer holds the KEY the runtime stored (`[<key>]`), never the value — this
 *  renders that key, so the card can say which credential was supplied without
 *  ever having held it. */
export function answerLabels(row: PendingAsk, questionId: string): string[] {
	const value = row.answers?.[questionId];
	return Array.isArray(value) ? value.map((v) => String(v)) : [];
}

/** Which surface settled this ask, when another one beat this phone to it
 *  (§4's single-winner rule: the loser is told "already answered by <surface>").
 *  Empty when the runtime did not say — an older core, or a settlement this
 *  phone made itself. */
export function answeredBySurface(row: PendingAsk): string {
	const by = row.answered_by;
	if (!by) return "";
	const surface = by.surface;
	return typeof surface === "string" ? surface : "";
}

/** The whole response as one line per question — what the response card shows
 *  behind its disclosure, and what a session row's summary would use. */
export function answeredPairs(
	questions: AskQuestion[] | undefined,
	answers: Record<string, string[]> | undefined,
): { question: string; answer: string }[] {
	const list = Array.isArray(questions) ? questions : [];
	const map = answers ?? {};
	return list.map((q) => {
		const chosen = Array.isArray(map[String(q?.id || "")]) ? map[String(q.id)] : [];
		return {
			question: String(q?.question || ""),
			answer: chosen
				.map((label) => {
					const option = (q.options ?? []).find((o) => o.label === label);
					return option?.description ? `${label} — ${option.description}` : label;
				})
				.join(", "),
		};
	});
}

/** How many questions an ask still has to answer — the bar's count and the
 *  card's "Question 1 of 3" both read it, so they agree by construction. */
export function questionProgress(row: PendingAsk): { index: number; total: number } {
	const total = Array.isArray(row.questions) ? row.questions.length : 0;
	const done = Math.max(0, total - unansweredQuestions(row).length);
	return { index: Math.min(done, Math.max(0, total - 1)), total: Math.max(1, total) };
}

/** The projection's BLOCKING pending request, with the legacy ask mirror
 *  removed (design §4's client rule N3).
 *
 *  For one release a queued ask is ALSO published as today's single-slot
 *  `pending` card, so an old client can still see and answer it. A client that
 *  has the `asks` field must IGNORE that card when its `kind` is `"ask"` —
 *  otherwise the same ask renders twice (the mirror card *and* the queued row)
 *  and a card the user answers is a second answer the queue refuses.
 *
 *  The presence of `asks` IS the capability proxy (§4): the field exists
 *  exactly while the runtime publishes queued asks, so `undefined` means "this
 *  runtime predates them" and the mirrored card is then the only view of the
 *  ask — which is precisely the case the mirror exists for. */
export function blockingPending<T extends { kind: string }>(
	pending: T | null | undefined,
	asks: unknown,
): T | null {
	if (pending == null) return null;
	if (asks !== undefined && pending.kind === "ask") return null;
	return pending;
}
