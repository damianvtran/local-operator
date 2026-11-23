/**
 * Asks sheet — the queued asks of the conversation it was opened from (design
 * §5.3, R7's EXPANDED state).
 *
 * ONE CONVERSATION, ON PURPOSE. The sheet used to list every conversation's
 * asks, so opening one session's questions showed every session's (Ben's
 * report, 2026-10-15): the entry sits on a conversation's own screen and reads
 * as that conversation's questions, and other conversations' rows under it
 * were noise at best and a wrong answer at worst. It now shows only the rows whose `session_id` is the opening conversation's.
 * Other conversations' asks stay visible where they belong: on the session
 * list, as each row's own ask chip.
 *
 * IT STILL READS THE AGGREGATE (`GET /api/asks`) and filters client-side,
 * because the aggregate is index-backed and needs no runtime: it answers the
 * same way whether the owning runtime is alive or long gone, which a read of
 * the live projection cannot.
 *
 * THE ANSWER SURFACE IS HERE. While this sheet is open, the phone's ask
 * composer is the sheet's own controls — each ask's form, with its own send —
 * and the conversation composer underneath is unreachable (the sheet is modal
 * over the phone column; see `ui/sheet.tsx`). That is §5.0's routing invariant
 * kept structurally rather than by a flag: with the surface EXPANDED the only
 * field that can receive a keystroke is an answer field, so a chat draft can
 * never become an answer, and with it MINIMIZED (this sheet closed) the
 * composer underneath is an ordinary conversation composer, so an answer can
 * never be sent as chat. The two drafts live in different components and
 * neither is ever copied into the other.
 *
 * LIVE WITHOUT A MANUAL REFRESH, and bounded. The sheet re-reads the aggregate
 * when it opens, whenever the daemon's own list frames change the outstanding
 * ask population (the store's `asksRevision` — a new ask or a settlement moves
 * it) or this conversation's own ask rows change (how a lapsed ask's dismissal
 * or late answer arrives: the list frame's count is 0 either side, #2388), and
 * on a 20 s backstop while it is open. The backstop is there for the
 * one event the list cannot announce: a DEADLINE passing in a conversation
 * whose runtime is gone, where nothing publishes a frame at all. It is a read
 * from an open sheet rather than a poll feeding a push, which is the bound
 * `asks/store.index_asks` asks a caller to keep in mind.
 */
import { contentDir } from "../lib/content-direction";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getAsks } from "../api";
import { AskCard, askStateOf, askToneClass } from "./ask-card";
import { Chevron } from "./ui/disclosure";
import { Sheet } from "./ui/sheet";
import {
	isAnswerable,
	askRowsSignature,
	isDeadConversation,
	isLapsed,
	isOutstanding,
	orderedForDisplay,
	questionTally,
} from "../lib/asks";
import { messageRuntime, useAsksRevision, useProjection, useSessions } from "../store";
import type { PendingAsk } from "../types";

/** How often the open sheet re-reads the aggregate as a deadline backstop. */
const BACKSTOP_MS = 20000;
/** How long a single aggregate read may hang before the sheet says so.
 *
 *  WHY A CLOCK IS THE RIGHT ANSWER FOR A HUNG READ (UX round 1, U6). A read that
 *  FAILS was already handled (the daemon's sentence, or "could not reach the
 *  daemon"); a read that never settles — a connection that is merely flaky,
 *  never reset — left `reading asks…` on screen forever, with the 20 s backstop
 *  re-issuing the same stalled request and no way forward but closing the sheet.
 *  The wait is the failure here, so the wait is what must be bounded. */
const READ_TIMEOUT_MS = 8000;
/** How often the countdowns re-render while the sheet is open.
 *
 *  Deadlines are stated in minutes, so a per-second tick would buy nothing; a
 *  coarse one keeps the crossing within half a minute of the truth (agent review
 *  round 1, N2 — before this the countdown was sampled only when a re-read
 *  landed, so a crossing could be stated up to 20 s late). */
const TICK_MS = 15000;

/** The sentence a dismissal receipt shows, per reason.

    A LOOKUP RATHER THAN A TEMPLATE. This used to be
    `` `relay.ask.dismiss.receipt_${reason}` `` — a key no static check can
    verify (a renamed or missing key would surface as the key text on screen)
    and a shape the relay copy gate counts as unextracted literals. Spelling
    each key here keeps them greppable and type-checked against the generated
    key union; ``lapsed`` is the door with no recorded reason, so it names the
    generic receipt. */
const RECEIPT_KEY = {
	lapsed: "relay.ask.dismiss.receipt",
	moot: "relay.ask.dismiss.receipt_moot",
	answered_in_chat: "relay.ask.dismiss.receipt_answered_in_chat",
} as const;

/** The body's own padding (`p-3`), so "scroll to the card" lands the card where
    it sits at rest instead of flush against the sheet's header. */
const BODY_PAD_PX = 12;

/** The ask to open: the first that can still take an answer, else the first
 *  lapsed one (whose only op is Dismiss). `orFirst` falls back to the head of the
 *  band, so a queue of nothing but dead conversations still shows its sentence;
 *  after an answer it does not, because an ask that cannot be acted on is not
 *  somewhere to walk to. `exceptId` is the ask just settled. */
function pickOpen(rows: PendingAsk[], exceptId: string, orFirst: boolean): PendingAsk | undefined {
	const others = rows.filter((row) => row.ask_id !== exceptId);
	return (
		others.find((row) => String(row.status || "open") === "open" && !isDeadConversation(row)) ??
		others.find(isLapsed) ??
		(orFirst ? others[0] : undefined)
	);
}

/** The start of an ask's first question, one line of it, for a collapsed row. */
function questionStart(row: PendingAsk): string {
	const first = Array.isArray(row.questions) ? row.questions[0]?.question : "";
	return String(first ?? "").split("\n")[0].trim();
}

function failureText(error: unknown): string {
	if (error instanceof TypeError) return messageRuntime.t("relay.asks.error.reach_failed");
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return messageRuntime.t("relay.asks.error.no_reason");
	return message;
}

export function AsksSheet({
	open,
	onClose,
	currentSessionId,
}: {
	open: boolean;
	onClose: () => void;
	/** The conversation this sheet was opened from, or empty for the
	    cross-conversation view the navigation drawer's Questions destination
	    opens (UX review round 1, U1). */
	currentSessionId: string;
}) {
	/* EMPTY MEANS ALL CONVERSATIONS, and that is a real mode rather than a
	   missing argument: the drawer's destination is machine-wide. Passing "" into
	   the per-conversation filter matched no row at all, so a destination that
	   exists to show what is waiting painted "nothing waiting" over three
	   outstanding asks. In this mode every row carries its own conversation's
	   name, so an answer is never attributed to the wrong conversation. */
	const crossConversation = currentSessionId === "";
	const [allRows, setRows] = useState<PendingAsk[]>([]);
	const [loaded, setLoaded] = useState(false);
	/** The receipt for a dismissal this sheet has just seen, KEYED TO THE ASK it
	    was about (agent review round 2, M1 = UX round 2, U9).

	    The daemon answers `{"ok": true, "detail": "dismissed"}`, and the card
	    leaves the aggregate — so the receipt is the only trace of the act. Two
	    things the first cut got wrong, both fixed here: it rendered only when the
	    sheet was EMPTY (so the ordinary multi-ask queue acknowledged nothing), and
	    it survived the operator's NEXT act (answering a second ask left "Dismissed
	    — no longer needed" over an answer, which is the class of claim this lane
	    exists to remove). The ask id is kept so a later dismissal replaces this
	    one rather than being confused with it. */
	const [receipt, setReceipt] = useState<
		{ askId: string; reason: "moot" | "answered_in_chat" | "lapsed" } | null
	>(null);
	const [error, setError] = useState("");
	const [nowMs, setNowMs] = useState(() => Date.now());
	const revision = useAsksRevision();
	const { sessions } = useSessions();
	/* THIS CONVERSATION'S OWN ROWS MOVE WHEN A LAPSED ASK LEAVES. The list
	   frame's `asks_open` does not (a timed-out ask is not counted, #2388), so
	   a dismissal or a late answer on another surface would otherwise wait for
	   the backstop. */
	/* THE HOOK IS UNCONDITIONAL (code review round 2, C2-3). It used to be
	   `crossConversation ? undefined : useProjection(...)`, which changes the hook
	   count if one mounted sheet ever moved between the two modes. Today's callers
	   pass a fixed mode, so nothing breaks now — but a rules-of-hooks violation
	   that is currently unreachable is still a trap for the next caller, and the
	   cost of obeying the rule is one discarded value. */
	const ownProjection = useProjection(currentSessionId).projection;
	const rowsSignature = askRowsSignature(crossConversation ? undefined : ownProjection?.asks);

	const load = useCallback(async () => {
		/* THE READ IS BOUNDED (U6). `AbortSignal.timeout` is the platform's own
		   answer and needs no bookkeeping of its own; where the engine lacks it
		   (the happy-dom test environment does), the read is simply unbounded
		   rather than unavailable — a missing abort must not break the sheet. */
		const signal =
			typeof AbortSignal !== "undefined" && "timeout" in AbortSignal
				? AbortSignal.timeout(READ_TIMEOUT_MS)
				: undefined;
		try {
			const answer = await getAsks(signal);
			setRows(Array.isArray(answer.asks) ? answer.asks : []);
			setError("");
		} catch (failure) {
			if (failure instanceof DOMException && failure.name === "TimeoutError") {
				setError(messageRuntime.t("relay.asks.error.read_timeout"));
				return;
			}
			/* The daemon's sentence, or the plainest honest line when it gave
			   none — the projects sheet's rule, which exists because `request`
			   falls back to the bare status ("503" explains nothing) and a
			   fetch-level failure arrives as the browser's own TypeError. */
			setError(failureText(failure));
		} finally {
			setLoaded(true);
		}
	}, []);

	/* A NEW OPENING RE-READS, and so does a changed population (a new ask, or
	   one settled on another surface, lapsed ones included). ONE effect for all
	   triggers: two effects keyed on `open` would each fire on the opening
	   transition and fetch the aggregate twice for one tap. */
	useEffect(() => {
		if (!open) {
			/* A RECEIPT BELONGS TO ONE OPENING: reopening later must not present
			   a dismissal from the last visit as though it just happened. */
			setReceipt(null);
			return;
		}
		void load();
	}, [open, revision, rowsSignature, load]);

	/* The deadline backstop, while the sheet is open only. */
	useEffect(() => {
		if (!open) return;
		const timer = setInterval(() => void load(), BACKSTOP_MS);
		return () => clearInterval(timer);
	}, [open, load]);

	/* The countdown's own clock (N2): the deadline lines are rendered from
	   `expires_at` on the CLIENT's clock (§5), and this is what makes them
	   advance while the sheet sits open rather than only when a read lands. */
	useEffect(() => {
		if (!open) return;
		setNowMs(Date.now());
		const timer = setInterval(() => setNowMs(Date.now()), TICK_MS);
		return () => clearInterval(timer);
	}, [open]);

	/* Only the ended/not-running strip reads the session list now: whether the
	   opening conversation has ended changes the strip's wording. */
	const ended = useMemo(
		() =>
			!crossConversation &&
			sessions.some((session) => session.session_id === currentSessionId && session.ended === true),
		[sessions, currentSessionId, crossConversation],
	);

	/* THIS CONVERSATION'S ROWS ONLY, before anything orders or counts them, so
	   the title, the empty state and the list all describe the same set. */
	const rows = useMemo(
		() =>
			crossConversation
				? allRows
				: allRows.filter((row) => String(row.session_id) === currentSessionId),
		[allRows, currentSessionId, crossConversation],
	);

	/* THE HEAD FIRST, then the wire's own order (`orderedForDisplay`) — the
	   order the minimized chip names (UX round 1, U8). */
	const listed = useMemo(() => orderedForDisplay(rows), [rows]);
	/* TWO BANDS, in display order (ask-panel parity, the native panel's `askBands`):
	   the asks that still show a card (waiting, lapsed, and an `expired` one drawn
	   as a receipt), and the settled ones under "Settled". Splitting AFTER
	   `orderedForDisplay` keeps the head ask first in its band. Settled asks are a
	   band of their own because one that has nothing left to do belongs below
	   everything that still does — it used to sit between the open card's options
	   and its buttons. */
	const bands = useMemo(() => {
		const current: PendingAsk[] = [];
		const done: PendingAsk[] = [];
		for (const row of listed) {
			const status = String(row.status || "open");
			(isOutstanding(status) || status === "expired" ? current : done).push(row);
		}
		return { current, done };
	}, [listed]);
	/* ONE ASK IS OPEN AT A TIME, and it is the one the pinned bar answers. The
	   reader's own pick (`choice`) wins; until they make one, the first ask that
	   can still be answered is open from the very first render (an effect would
	   paint one collapsed frame first), and is then LATCHED, so a re-rank from a
	   later read cannot move the card from under a half-typed answer. `null` is a
	   deliberate "nothing open" (an answer cleared the last ask that could take
	   one). */
	const [choice, setChoice] = useState<string | null | undefined>(undefined);
	/** The ask whose first question takes focus on mount: set when an answer moves
	    the reader on to the next ask. */
	const [focusAsk, setFocusAsk] = useState<string | null>(null);
	const defaultOpen = pickOpen(bands.current, "", true);
	const choiceLive =
		choice !== undefined && (choice === null || listed.some((row) => row.ask_id === choice));
	const openId = choiceLive ? choice : (defaultOpen?.ask_id ?? null);
	useEffect(() => {
		if (!open || !loaded) return;
		/* A choice whose row left the queue (settled elsewhere) is dropped, which
		   lets the default re-latch. */
		if (choice !== undefined && !choiceLive) setChoice(undefined);
		else if (choice === undefined && defaultOpen) setChoice(defaultOpen.ask_id);
	}, [open, loaded, choice, choiceLive, defaultOpen]);
	/* THE PINNED BAR'S ELEMENT. The open card portals its answering controls into
	   it (`AskCard`'s `footerSlot`); a state rather than a ref so the card is
	   re-rendered with the element once the bar mounts. */
	const [barEl, setBarEl] = useState<HTMLElement | null>(null);
	const scroller = useRef<HTMLDivElement>(null);
	/* THE OPEN CARD STARTS AT THE TOP: when the sheet opens, and when the reader
	   moves on to the next ask, so a long previous card does not leave them
	   scrolled past the question they now have to read. Keyed on the open ask's
	   id, never on a re-read, so the backstop poll does not move a reader who has
	   scrolled on purpose. */
	useEffect(() => {
		if (!open || openId === null) return;
		const region = scroller.current;
		const card = Array.from(
			region?.querySelectorAll<HTMLElement>('[data-testid="ask-card"]') ?? [],
		).find((element) => element.dataset.askId === openId);
		if (!region || !card) return;
		region.scrollTop +=
			card.getBoundingClientRect().top - region.getBoundingClientRect().top - BODY_PAD_PX;
	}, [open, openId]);
	const sessionOf = (row: PendingAsk) =>
		crossConversation ? String(row.session_id) : currentSessionId;
	const nameOf = (row: PendingAsk) => {
		const key = String(row.session_id);
		return sessions.find((session) => session.session_id === key)?.conversation_name || key;
	};
	/* THE WAIT IS STATED BEFORE THE TAP, AND ONLY WHERE THERE IS ONE (design round 1,
	   D4 = UX U4; round-2 corrections below), now on the open card.

	   ANSWERABLE, because a settled receipt offers nothing to answer, so a wait
	   stated above it is noise about a tap that does not exist (UX round 2, U6).
	   `runtime_live` widened it from `ended` alone to the merely-not-running
	   conversation the operator actually hits.

	   DURABILITY, because `runtime_live === false` is ALSO true of a conversation
	   with no transcript at all — where the op is a terminal refusal and the promise
	   of a ~30 s bring-up is a claim about a backend that does not exist (design
	   round 2, D6). It is `isDeadConversation`, the CONJUNCTION of the two fields —
	   not `durable` alone, which would silence the strip on a live conversation that
	   answers fine (design round 3, D7).

	   NOT LAPSED, because a lapsed card offers no answer (`isLapsed`), so "answering
	   will bring it up" would promise a tap the card does not have. */
	const waitsFor = (row: PendingAsk) =>
		isAnswerable(String(row.status || "open")) &&
		!isLapsed(row) &&
		!isDeadConversation(row) &&
		(ended || row.runtime_live === false);
	/* WHAT SITS UNDER THE STATE LINE of the open card: the conversation's name
	   (data, not copy — the projection's own name; UX round 2, U6) only when the
	   sheet holds several conversations' asks, and the wait where there is one. */
	const leadOf = (row: PendingAsk) => {
		const name = crossConversation ? nameOf(row) : "";
		const wait = waitsFor(row)
			? ended
				? messageRuntime.t("relay.asks.sheet.ended_wait")
				: messageRuntime.t("relay.asks.sheet.not_running_wait")
			: "";
		if (name === "" && wait === "") return undefined;
		return (
			<>
				{name ? (
					<p dir={contentDir(name)} className="truncate text-meta font-medium text-ink-dim">
						{name}
					</p>
				) : null}
				{wait ? <p className="text-meta text-ink-dim">{wait}</p> : null}
			</>
		);
	};
	/* THE RECEIPT IS THE INSTANT ECHO, AND IT STANDS DOWN WHEN ITS ROW HAS
	   SETTLED (SA-D8, design D1 = UX U2; review round 2, M2). Since §17 a
	   dismissal taken before the ask's deadline KEEPS the row as a settled
	   receipt, so once that row is here the sheet would state one fact twice:
	   this unattributed line above the list, and the same catalogue sentence on
	   the card below it — two type sizes, two ink tokens, one `gap-3` apart, and
	   the upper one reads as a status of whatever sits directly under it.

	   PRESENCE OF THE ID IS NOT THE TEST. The read that follows a tap is a round
	   trip, and until it lands `rows` is the OLD list, which still carries the
	   ask as `open` — the exact gap this line exists to cover. Suppressing on the
	   id alone hid the acknowledgement precisely when it was needed and left a
	   live-looking card after a successful dismissal, so the test is the ROW's
	   own status: only a row that says `dismissed` replaces the sentence.

	   THE LAPSED DOOR FALLS OUT OF THAT RULE rather than needing its own: a
	   dismissal at or after the deadline removes the row (case 2 of §17), so the
	   read either omits it or still shows it as `timed_out` — never `dismissed` —
	   and the receipt stays as that door's only acknowledgement. */
	const shownReceipt =
		receipt &&
		!listed.some(
			(row) => row.ask_id === receipt.askId && String(row.status) === "dismissed",
		)
			? receipt
			: null;
	/* QUESTIONS, the same unit the bar and the header entry state (agent review
	   round 1, R3): three surfaces answering "how much is waiting" in two units is
	   the defect. All three now count this one conversation's rows.

	   ONLY OFFERED QUESTIONS ARE WAITING (design round 1, D1), the dock's and the
	   header's rule: a lapsed question is shown read-only, so "asks · 1 question"
	   above a card reading Timed out stated a live decision one element above the
	   card that says there is none. A lapsed-only sheet is titled for what it
	   holds.

	   DEAD ROWS' QUESTIONS ARE NOT WAITING ON ANYONE (agent review round 2,
	   MINOR-3). A dead conversation's ask still carries `open`, so it enters
	   `outstandingAsks` — but every op on it is refused with `ask_session_gone`,
	   and a count that promises answers the backend will refuse is the same class
	   of claim this whole lane exists to remove. It is the sheet's own rule
	   ("no contradiction one element apart") applied to the title, one element
	   above the strip that already honours it. */
	const { waiting, lapsed } = questionTally(rows.filter((row) => !isDeadConversation(row)));
	const title =
		waiting > 0
			? messageRuntime.t("relay.asks.sheet.title_waiting", { count: waiting })
			: lapsed > 0
				? messageRuntime.t("relay.asks.sheet.title_lapsed", { count: lapsed })
				: messageRuntime.t("relay.asks.sheet.title_plain");

	/* The open ask's row, if it is in the current band (an expanded "Settled" row
	   is drawn in place, not up here). */
	const openRow = bands.current.find((row) => row.ask_id === openId);
	const settled = (row: PendingAsk, kind: "respond" | "decline" | "dismiss") => {
		/* A NON-DISMISSAL SETTLEMENT RETIRES THE RECEIPT (M1): after answering, an
		   unretired "Dismissed — …" would describe the operator's previous act as
		   though it were their last. */
		if (kind !== "dismiss") setReceipt(null);
		/* ANSWERING WALKS ON (design §5.3): the next ask that can still take an
		   answer opens and takes the reader's place; none leaves the sheet showing
		   what is settled. The re-read below is a round trip, so the next ask is
		   chosen from the rows already here. */
		const next = pickOpen(bands.current, row.ask_id, false);
		setChoice(next ? next.ask_id : null);
		setFocusAsk(next ? next.ask_id : null);
		void load();
	};
	const dismissed = (askId: string, reason: "moot" | "answered_in_chat" | undefined) =>
		setReceipt({ askId, reason: reason ?? "lapsed" });
	/** A collapsed ask: ONE line — the state sentence, then the start of the
	    question, then the caret. A tap opens it (and closes the current one). A
	    settled ask expands in place into its receipt, under the line. */
	const collapsedRow = (row: PendingAsk, expanded: boolean) => {
		const state = askStateOf(row, nowMs);
		const asked = questionStart(row);
		return (
			<div key={`${sessionOf(row)}:${row.ask_id}`} className="flex flex-col gap-1">
				<button
					type="button"
					aria-expanded={expanded}
					data-testid="ask-row"
					data-ask-id={row.ask_id}
					onClick={() => setChoice(expanded ? undefined : row.ask_id)}
					className="flex min-h-11 w-full items-start gap-2 rounded-sm py-1 text-start active:bg-surface"
				>
					{/* THE QUESTION LEADS, the state sentence follows on its own line and is
					    never cut: a waiting row is read for WHAT it asks first, and the state
					    sentence is the part that must stay whole. The conversation's name
					    (data) is the row's own first line when the sheet holds several
					    conversations. The caret stays on the first line, beside the text it
					    opens, and mirrors in a right-to-left layout (`lo-directional`). */}
					<span className="flex min-w-0 flex-1 flex-col">
						{crossConversation ? (
							<span dir={contentDir(nameOf(row))} className="truncate text-start text-meta text-ink-muted">
								{nameOf(row)}
							</span>
						) : null}
						{asked === "" ? null : (
							<span
								dir={contentDir(asked)}
								className="line-clamp-2 text-start text-body-sm font-medium break-words text-ink"
							>
								{asked}
							</span>
						)}
						<span className="text-start text-meta">
							<span className={askToneClass(state.tone)}>{state.text}</span>
							{row.urgent && !expanded ? (
								<span className="font-medium text-warning"> {messageRuntime.t("relay.asks.card.urgent")}</span>
							) : null}
						</span>
					</span>
					<Chevron open={expanded} className="mt-0.5" />
				</button>
				{expanded ? (
					<AskCard
						row={row}
						sessionId={sessionOf(row)}
						nowMs={nowMs}
						hideState
						onSettled={(kind) => settled(row, kind)}
						onDismissed={dismissed}
					/>
				) : null}
			</div>
		);
	};

	return (
		<Sheet
			open={open}
			onClose={onClose}
			title={title}
			scrollerRef={scroller}
			/* THE ANSWERING CONTROLS ARE PINNED, outside the scroller: content may
			   scroll, the control that answers the card may not. The bar is empty (and
			   collapsed) unless the open ask has something to answer with. */
			footer={
				<div
					ref={setBarEl}
					data-testid="asks-bar"
					className="border-t border-hairline px-3 py-2 empty:hidden"
				/>
			}
		>
			<div className="flex flex-col gap-3 p-3">
				{error ? (
					<div className="flex flex-col items-start gap-2">
						<p className="text-start text-body-sm text-danger">{error}</p>
						{/* A RETRY, because a hung or failed read must not leave the sheet
						   with closing as its only move (U6). */}
						<button
							type="button"
							onClick={() => void load()}
							className="flex min-h-11 items-center rounded-sm border border-control px-3 text-body-sm text-ink active:bg-surface"
						>
							{messageRuntime.t("relay.asks.sheet.retry")}
						</button>
					</div>
				) : null}
				{/* THE LOADING LINE IS NOT DECORATION: without it the sheet paints its
				    empty state while the read is still in flight, so "nothing waiting"
				    is shown about an answer that has not arrived yet. It is one line and
				    it disappears the moment the read settles. */}
				{!loaded && !error ? (
					<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.asks.sheet.loading")}</p>
				) : null}
				{/* THE RECEIPT IS NOT THE EMPTY STATE (agent review round 2, M1 = UX
				    round 2, U9). It used to render only when the list was empty, so a
				    dismissal from a queue of two acknowledged nothing at all; it now
				    sits above whatever remains — the place the card left — with the
				    empty state below it saying only that nothing is waiting.

				    AND IT STANDS DOWN ONCE ITS OWN ROW ARRIVES (SA-D8): the settled
				    card below states the same sentence, so keeping both would say one
				    thing twice. See `shownReceipt`. */}
				{shownReceipt ? (
					<p className="text-body-sm text-ink-muted" data-testid="ask-receipt">
						{messageRuntime.t(RECEIPT_KEY[shownReceipt.reason])}
					</p>
				) : null}
				{loaded && rows.length === 0 && !error && !shownReceipt ? (
					<p className="text-body-sm text-ink-muted" data-testid="ask-sheet-empty">
						{messageRuntime.t("relay.asks.sheet.empty")}
					</p>
				) : null}
				{/* THE OPEN ASK, FIRST: one card, whichever position the queue holds it in —
				    the reader opened it to read it, and it is the one the pinned bar answers. */}
				{openRow ? (
					<AskCard
						key={`${sessionOf(openRow)}:${openRow.ask_id}`}
						row={openRow}
						sessionId={sessionOf(openRow)}
						nowMs={nowMs}
						footerSlot={barEl}
						lead={leadOf(openRow)}
						focusQuestion={focusAsk === openRow.ask_id}
						onFocused={() => setFocusAsk(null)}
						onSettled={(kind) => settled(openRow, kind)}
						onDismissed={dismissed}
					/>
				) : null}
				{/* The other asks still waiting (or lapsed), one line each. Opening one
				    closes the current one. */}
				{bands.current
					.filter((row) => row.ask_id !== openId)
					.map((row) => collapsedRow(row, false))}
				{/* SETTLED ASKS go under their own heading, below everything that still
				    waits, and expand IN PLACE into their receipt. */}
				{bands.done.length > 0 ? (
					<>
						<h3 className="pt-1 text-meta font-medium text-ink-muted">
							{messageRuntime.t("relay.asks.sheet.settled")}
						</h3>
						{bands.done.map((row) => collapsedRow(row, openId === row.ask_id))}
					</>
				) : null}
			</div>
		</Sheet>
	);
}
