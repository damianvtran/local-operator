/**
 * When the asks sheet opens BY ITSELF — the phone half of the shared open policy.
 *
 * The TUI's ``local_operator/tui/ask_open_policy.py`` is the reference
 * implementation and the clause numbers below are THE SAME clauses; read its
 * docstring for the argument behind each. This file restates only what is
 * different on a phone, and why.
 *
 * WHY THIS EXISTS. The minimized dock above the composer is one line, and a
 * first-time user who opens a conversation with questions waiting on them may
 * never discover it. The operator asked every ask surface to open its primary
 * interaction by default when a conversation with PENDING asks is opened. On the
 * phone that surface is the ``AsksSheet``.
 *
 * THE SHARED OPEN-POLICY CONTRACT (all four surfaces implement these six):
 *
 *  1. No asks on open -> closed, as before.
 *  2. Pending asks on open -> open ONCE for that view of that conversation. A
 *     "view" begins when the conversation's screen mounts.
 *  3. Everything already addressed on open -> closed, and a view that decided
 *     "closed" never auto-opens afterwards. A new ask arriving later is covered
 *     by the dock and the header entry, which is the discoverability this feature
 *     is not meant to replace.
 *  4. A deliberate close while asks remain is respected. It records the ASK IDS
 *     it waved off against the conversation id, in memory, and survives
 *     re-renders, queue refreshes, asks arriving or changing, and switching away
 *     and back within the same page lifetime. It is FORGOTTEN once none of those
 *     ids is still outstanding. A page reload starts clean and may open it again.
 *  5. Never steal the keyboard, never trap, never open on a guess. Nothing opens
 *     while the user is typing or another surface owns the screen; the sheet can
 *     always be closed; and a frame that does not carry the rows (an old daemon,
 *     a projection that has not loaded, a tally with the rows dropped) is NOT
 *     "pending asks".
 *  6. Auto-open is not a user pressing the door. The dock and the header entry
 *     keep their own semantics; opening by policy is a separate path.
 *
 * WHAT DIFFERS FROM THE TUI, AND WHY
 *
 *  * NO "ARRIVED" CLASSIFICATION. The TUI compares each ask's ``created_at`` with
 *    the instant the view began, to tell "was waiting when I arrived" from "was
 *    asked while I watched", with a 5 s tolerance for the owner machine's clock.
 *    A phone cannot honour that tolerance: ``created_at`` is stamped by the
 *    machine that owns the queue and the phone compares it with its own clock,
 *    and a phone that is minutes off would call a long-pending ask an arrival and
 *    never open it, which is the failure the feature exists to remove. Here
 *    "pending on open" is "named by the FIRST frame that resolves the queue after
 *    the screen mounted". The residue is an ask raised between the mount and that
 *    first snapshot also opening the sheet, which is benign: it is what the user
 *    would see first anyway, and the frame is one SSE snapshot away.
 *
 *  * NO "SETTLING". A conversation switch on the phone is a route change that
 *    REMOUNTS the screen (``app.tsx`` keys it by session and job), so there is no
 *    window in which a conversation is on screen while its composer still holds
 *    the previous one's draft. The TUI's wait-for-the-switch has no equivalent.
 *
 *  * NO ``asks_truncated``. The relay's daemon never forwards that flag (the
 *    projection builds ``asks``/``asks_open`` only), so a dismissal is judged on
 *    the TALLY alone: a frame names every outstanding ask exactly when its rows
 *    cover ``asks_open``. See :func:`namesEveryOutstanding`.
 *
 *  * THE MODULE RECORD OUTLIVES THE SCREEN. The screen remounts on every switch,
 *    so a dismissal kept in component state would be gone the moment the user
 *    switched away, and clause 4 ("...or switching away and back") would be
 *    broken by construction. The record is module state, which lives exactly as
 *    long as the page, which is clause 4's own lifetime.
 *
 * PURE ON PURPOSE. The decision is a function of a handful of facts and a tiny
 * in-memory record; it imports nothing from React, the store or the DOM, so the
 * whole matrix is unit-testable without rendering a screen, and the screen's
 * wiring stays a few lines of glue rather than a second copy of the rules.
 */

/** What ONE projection frame says about a conversation's queue, at view open. */
export type QueueReading =
	/** No usable rows: the runtime does not publish the queue (absence is the
	 *  capability proxy), the projection has not loaded, or only the tally rode
	 *  because the wire bound dropped the rows. NOT "no asks" and NOT "pending
	 *  asks" — a decision taken on it would be a guess (clause 5). */
	| "unresolved"
	/** A live queue with nothing in it (``asks_open: 0``): clause 1. */
	| "empty"
	/** Rows are published and every one is already addressed: clause 3. */
	| "settled"
	/** At least one answerable ask is named: clause 2. */
	| "pending";

/** What the screen should do about the sheet for this frame. */
export type OpenDecision =
	/** Open the sheet now. Returned at most once per view. */
	| "open"
	/** Nothing yet — the queue is not resolved. Ask again on the next frame. */
	| "wait"
	/** Leave it closed. The view has decided; no later frame reopens the question. */
	| "skip";

/**
 * How long after a view began a late frame can still be called "on open".
 *
 * The same 45 s as the TUI: waiting for a resolved frame needs a bound, or a
 * screen that never gets one (a daemon that is slow to attach the session) would
 * open the sheet at minute ten over whatever the user is doing then. On a phone
 * the first snapshot normally lands inside a second or two; the window is the
 * ceiling for a cold session being woken, not the expectation.
 */
export const OPEN_WINDOW_MS = 45_000;

/** The minimum a row has to carry for the policy to judge it. */
export interface PolicyAskRow {
	ask_id: string;
	status?: string;
}

/** An ask the user can still act on: open, or timed out and still answerable.
 *
 *  The same set ``lib/asks.ts`` calls outstanding, restated over the two
 *  statuses rather than imported so this module stays free of the wire types. */
function isAnswerableStatus(status: string | undefined): boolean {
	const value = status ?? "open";
	return value === "open" || value === "timed_out";
}

/**
 * Classify one projection frame's ask fields.
 *
 * ``rows`` is the frame's ``asks`` (``undefined`` while the runtime does not
 * publish them) and ``tally`` its ``asks_open`` (``undefined`` when the runtime
 * does not publish it, ``0`` for a live queue with nothing to fold, ``N`` for N
 * outstanding asks).
 *
 * The rows decide whenever they name an answerable ask. With none, only an
 * explicit ``0`` tally is an answer ("nothing is queued"); ``undefined``
 * (unsupported) and ``N > 0`` (the asks exist but their rows were dropped to fit
 * the frame) both leave the question open, because neither one lets a surface
 * draw anything — opening on the tally alone would mount a sheet over a queue the
 * phone cannot yet show. The same holds for rows that are ALL settled beside a
 * positive tally: the outstanding asks are the ones the bound dropped, and
 * calling that "settled" would spend the view's one decision on a claim the frame
 * contradicts.
 */
export function readQueue(
	rows: readonly PolicyAskRow[] | undefined | null,
	tally: number | undefined | null,
): QueueReading {
	const published = Array.isArray(rows) ? rows : [];
	if (published.length > 0) {
		if (published.some((row) => isAnswerableStatus(row?.status))) return "pending";
		// Rows, and none of them outstanding. That is "everything addressed" only if
		// the tally agrees: ``asks_open`` counts outstanding asks BEFORE the wire's
		// text bound drops rows, and the drop order keeps newer settled rows over an
		// older timed-out one.
		if (typeof tally === "number" && tally > 0) return "unresolved";
		return "settled";
	}
	if (tally === 0) return "empty";
	return "unresolved";
}

/**
 * Whether a frame's rows name EVERY outstanding ask — what a dismissal is judged on.
 *
 * A dismissal is forgotten only on evidence that every ask the user waved off has
 * left the queue, and that evidence is "the frame names all the outstanding asks
 * and none of them is one of mine". A frame that cannot name them all proves
 * nothing: a dropped row may be exactly the one still outstanding.
 *
 * THE TEST IS THE TALLY. ``asks_open`` is summed over the same projected rows the
 * wire carries and BEFORE the text bound drops any, so ``tally <= named`` says the
 * bound dropped nothing outstanding. ``undefined`` (the runtime publishes no
 * tally) cannot vouch for completeness and fails closed, which can only ever
 * under-open, never override a refusal.
 *
 * ONE RESIDUAL LIMIT, not closable on the client: the runtime projects at most 20
 * rows before it computes the tally, so an outstanding timed-out ask that sorts
 * behind 20 newer rows is on neither field and no client can name it. The residue
 * is at most one extra auto-open, which is the surface the feature exists to show.
 */
export function namesEveryOutstanding(
	tally: number | undefined | null,
	namedOutstanding: number,
): boolean {
	return typeof tally === "number" && tally <= namedOutstanding;
}

/**
 * The in-memory record behind the policy — one per page.
 *
 * Three pieces of state and nothing else: the CURRENT VIEW (which conversation is
 * on screen and when it began), whether that view has taken its one decision, and
 * the dismissals — for each conversation the user waved off, WHICH ASKS they
 * waved off. All of it dies with the page, which is clause 4's "a reload may open
 * it again": persisting a dismissal would turn a courtesy into a setting nobody
 * asked for.
 */
export class AskOpenPolicy {
	private readonly windowMs: number;
	private enabled: boolean;
	/** conversation id -> the ask ids the user waved off by closing the sheet
	 *  while they were pending (clause 4). Evaluated and forgotten in
	 *  :meth:`decide`, the only place the queue's current ids are known. */
	private readonly dismissed = new Map<string, ReadonlySet<string>>();
	private view = "";
	private openedAtMs = 0;
	/** True until a view arms, and again once it has decided. A view with no
	 *  decision left to take is inert: every later frame is a no-op. */
	private decided = true;

	constructor(options: { windowMs?: number; enabled?: boolean } = {}) {
		this.windowMs = options.windowMs ?? OPEN_WINDOW_MS;
		this.enabled = options.enabled ?? true;
	}

	/** Turn the whole policy off (and the next view arms nothing). The seam the
	 *  older suites use to keep testing the door; production never calls it. */
	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
		if (!enabled) this.decided = true;
	}

	/**
	 * A conversation's screen mounted — arm its one decision.
	 *
	 * Called on EVERY mount, so a re-entry is a new view even for the same
	 * conversation: the user left and came back, which is exactly the "switching
	 * away and back" clause 4 speaks of, and the dismissal record (not this
	 * call) is what carries the refusal across. A view with no conversation id is
	 * never armed: its dismissal could not be keyed, so opening it would break
	 * clause 4 for exactly the sessions this record cannot remember.
	 */
	beginView(conversationId: string, nowMs: number): void {
		this.view = conversationId;
		this.openedAtMs = nowMs;
		this.decided = !conversationId || !this.enabled;
	}

	/** Whether this view still owes a decision — the screen's cheap per-frame gate.
	 *
	 *  Projection frames land at the stream's rate, so the screen asks this before
	 *  doing any work: once a view has decided, every later frame costs one
	 *  boolean. */
	awaiting(conversationId: string): boolean {
		return Boolean(conversationId) && conversationId === this.view && !this.decided;
	}

	/**
	 * The view's one decision, taken from the first frame that can take it.
	 *
	 * ``occupied`` is everything that already has the user's hands or the screen:
	 * a focused text field, a sheet or an approval card already up. Clause 5:
	 * auto-open yields to all of them, and it yields for GOOD — a user who was busy
	 * when the conversation opened is not interrupted a minute later by a sheet
	 * that appears when they stop. A draft merely SITTING in the composer is not
	 * on that list on a phone (the screen explains why, where it gathers the
	 * facts): focus is observable here, and a restored draft is not being typed.
	 *
	 * ``surfaceOpen`` is whether the sheet is already up: the user got to the
	 * door first, so there is nothing left to do (clause 6).
	 *
	 * ``outstandingIds`` is the whole answerable set when the frame names every
	 * outstanding ask, or ``null`` when it cannot (:func:`namesEveryOutstanding`).
	 * ``null`` HOLDS a dismissal: an ask the bound dropped may be exactly the one
	 * still outstanding, and releasing on that guess would open a sheet the user
	 * refused.
	 */
	decide(
		conversationId: string,
		reading: QueueReading,
		options: {
			nowMs: number;
			occupied: boolean;
			surfaceOpen: boolean;
			outstandingIds: readonly string[] | null;
		},
	): OpenDecision {
		if (!this.awaiting(conversationId)) return "skip";
		if (
			options.surfaceOpen || // clause 6: the door got there first
			options.nowMs - this.openedAtMs > this.windowMs // clause 3: too late to be "on open"
		) {
			this.decided = true;
			return "skip";
		}
		if (reading === "unresolved") return "wait"; // clause 5: not a guess
		this.decided = true;
		const wavedOff = this.dismissed.get(conversationId);
		if (wavedOff !== undefined) {
			const stillHere =
				options.outstandingIds === null ||
				options.outstandingIds.some((id) => wavedOff.has(id));
			if (stillHere) return "skip"; // clause 4: something they refused is still there
			// Every ask they waved off has left the queue. The refusal was about
			// THOSE asks, so the conversation is a fresh one from here on.
			this.dismissed.delete(conversationId);
		}
		if (reading === "pending" && !options.occupied) return "open"; // clause 2
		return "skip"; // clauses 1, 3 and 5
	}

	/**
	 * The user deliberately closed the sheet — remember WHICH asks (clause 4).
	 *
	 * ``pendingIds`` is the pending set at the moment of the close: the asks the
	 * user looked at and waved off. Only while asks REMAIN — closing a sheet of
	 * nothing but settled rows was a glance at history, not a refusal of anything.
	 *
	 * A second dismissal ADDS to the record rather than replacing it: with whole
	 * frames the two are equivalent (an id that left the queue cannot return), and
	 * with a frame that dropped rows the union keeps the ids the shorter list could
	 * not show.
	 *
	 * The record is STICKY for the page: it is not cleared if the user later opens
	 * the sheet by hand. "I closed it once" is the fact the contract asks to be
	 * respected; it is forgotten only by :meth:`decide`, when none of the
	 * waved-off asks is outstanding any more.
	 */
	noteUserClosed(conversationId: string, pendingIds: Iterable<string>): void {
		const ids = new Set(pendingIds);
		if (!conversationId || ids.size === 0) return;
		const merged = new Set(this.dismissed.get(conversationId) ?? []);
		for (const id of ids) merged.add(id);
		this.dismissed.set(conversationId, merged);
		if (conversationId === this.view) this.decided = true;
	}

	/**
	 * The user opened the sheet through a door — the policy has nothing to add.
	 *
	 * Settles the current view's decision, so a policy still waiting on an
	 * unresolved frame cannot open a sheet the user has just opened themselves
	 * (clause 6). Does NOT touch the dismissal record (see :meth:`noteUserClosed`).
	 */
	noteUserOpened(conversationId: string): void {
		if (conversationId === this.view) this.decided = true;
	}

	/** Whether a dismissal is ON RECORD for the conversation — "on record", not
	 *  "still in force": a record is forgotten by :meth:`decide`, the only place the
	 *  queue's current ids are known. */
	isDismissed(conversationId: string): boolean {
		return this.dismissed.has(conversationId);
	}

	/** Forget the view, the decision and every dismissal. Tests only: a page
	 *  reload is the production reset (clause 4's lifetime), and nothing in the app
	 *  calls this. */
	reset(): void {
		this.dismissed.clear();
		this.view = "";
		this.openedAtMs = 0;
		this.decided = true;
	}
}

/**
 * THE ONE POLICY OF THE PAGE. Module state, deliberately: the session screen
 * remounts on every switch, and the dismissal record has to outlive it (see the
 * header). Tests that need a clean record build their own :class:`AskOpenPolicy`
 * or call :func:`resetAskOpenPolicy`.
 */
export const askOpenPolicy = new AskOpenPolicy();

/** Forget everything — views, decisions and dismissals. For tests only (see
 *  :meth:`AskOpenPolicy.reset`): the page reload that production uses is not
 *  available inside a test file, and a record that leaked from one test into the
 *  next would make a clause-4 cell pass or fail on the previous cell's close. */
export function resetAskOpenPolicy(): void {
	askOpenPolicy.reset();
}
