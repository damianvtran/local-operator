/**
 * Session view (`#/s/:sessionId`) — the core screen. Layout contract:
 *
 *   header / transcript (flex-1, scrolls) / todos / subagents /
 *   pending card / banners / composer
 *
 * The whole column is `100dvh` capped, and a visualViewport listener keeps
 * the composer above the iOS keyboard: when the keyboard opens, the
 * viewport shrinks and the root element is re-pinned to its height. This
 * works around iOS Safari's habit of scrolling the page instead of
 * shrinking the layout when `interactive-widget` is not honoured.
 *
 * That pin is also why the column publishes `--lo-vvh`. Every bounded region
 * inside it (the ask card, the todos and subagent panels) has to be measured
 * against THIS column, and `dvh` is not that column: a virtual keyboard is an
 * overlay, so it shrinks `visualViewport.height` and leaves the dynamic
 * viewport untouched (index.html sets no `interactive-widget`, so the default
 * `resizes-visual` applies). A `60dvh` cap therefore stayed at 468px while the
 * column it lived in fell to 480px — measured on a 360x780 phone with a 300px
 * keyboard, which put `send`, the only way to submit a free-text answer, below
 * the column's clipped foot with no gesture that recovered it. Publishing the
 * pinned height as a custom property gives those caps the same unit as the
 * box they are bounded by, so they tighten exactly when the space does.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { resumeSession, setSessionPin } from "../api";
import { AskDock } from "../components/ask-dock";
import { AsksSheet } from "../components/asks-sheet";
import { ModelSheet } from "../components/model-sheet";
import { Composer } from "../components/composer";
import { GateSheet } from "../components/gate-sheet";
import { WideViewButton } from "../components/wide-view-button";
import { PendingCard } from "../components/pending-card";
import { SubagentsPanel } from "../components/subagents-panel";
import { TodosPanel } from "../components/todos-panel";
import { SessionStatus } from "../components/session-status";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { cn } from "../lib/cn";
import { COLUMN_HEIGHT_VAR, COLUMN_TOP_VAR } from "../lib/column";
import { navigate } from "../router";
import { consumePendingFocus } from "../lib/pending-focus";
import { pinRefusalText } from "../lib/pin-refusal";
import { resumeRefusalText } from "../lib/refusal";
import { useCompletionView } from "../use-completion-view";
import { usePendingEchoes } from "../pending-echo";
import { AgentScreen } from "./agent-view";
import {
	applySessionPin,
	clearSessionPinMark,
	retainProjectionStream,
	retainSessionListStream,
	usePinMarks,
	useProjection,
	useRouteTitle,
	useSessions,
} from "../store";
import { blockingPending, outstandingAsks } from "../lib/asks";
import type { SessionProjection } from "../types";

/** The header strip's control box: a real 44x44 target.

    This was a 32px box (`min-h-8 min-w-8`) plus a 6px hit-slop pseudo-element,
    chosen so the painted strip kept its height. The mobile-UX audit's sweep
    (D2) reads element boxes, and its measured arithmetic showed the slop never
    reached the floor anyway: the controls sit `gap-2` (8px) apart, so two 6px
    slops overlap by 4px between neighbours and the effective horizontal target
    was ~40px, not 44 — while every header control still measured as a sub-44
    box, back/★ included.

    The floor is now the PAINTED box: `min-h-11 min-w-11`, the same idiom the
    past-sessions and agent headers already use. The strip grows
    32 -> 44 (this header lands at ~52px, the height the agent screen's header
    states outright); that is the accepted cost of a target that measures what
    it is, and it unifies the back-button treatments D7 recorded (44 /
    32+slop / 32-no-slop). The slop pseudo-element is gone with it — it existed
    to widen a box that is now wide enough on its own.

    One constant rather than the classes repeated inline, for the reason every
    shared class here is shared: three copies drift. */
const HEADER_CONTROL =
	"flex min-h-11 min-w-11 items-center justify-center rounded-sm active:bg-elevated";

function Header({
	projection,
	sessionId,
	onOpenAsks,
	askQuestionCount,
}: {
	projection: SessionProjection;
	sessionId: string;
	/** Summon the asks sheet. The header entry is the SECOND way into the
	    EXPANDED state (§5.0's "explicit /asks / header action"); the minimized
	    bar above the composer is the first. */
	onOpenAsks: () => void;
	/** The session's outstanding questions, derived ONCE by the screen and handed
	    to both doors (the header entry and the minimized bar), so the two cannot
	    state one queue two ways. */
	askQuestionCount: number;
}) {
	const [gateOpen, setGateOpen] = useState(false);
	/* THE LOOSENING RECEIPT, held HERE rather than in the sheet (design round 6,
	   D2). The sheet unmounts the moment it closes, and a report set inside it was
	   never painted — measured: panel gone, header still "needs you", the user told
	   nothing. The header is the surface the sheet closes back onto, so the receipt
	   belongs to it and survives.

	   IT IS CLEARED BY THE NEXT GATE-CHANGING GESTURE ON THIS SCREEN: `onReceipt("")`
	   comes from the sheet's tighten path, so a `keep asking` from the sheet leaves
	   no header claiming the gate is auto (UX round 8, U8-3 — the flow the round-7
	   comment below was wrong about: it asserted there was nothing stale to clear,
	   and a tighten one tap away is exactly that). THE BOUND, stated rather than
	   implied (agent review round 9, R9-5): the receipt is component state, not
	   derived from the projection, so a gate tightened from ANOTHER surface — the
	   machine's TUI, the desktop app — leaves this header asserting the old state
	   until this screen's next gate-changing gesture. Deriving it from
	   `projection.gate` would remove the staleness window entirely and is the
	   obvious next change if this screen ever shows a gate the phone did not set;
	   today every path that sets it runs through the sheet below. The
	   round-7 design note is kept because it is the reason the receipt lives HERE:
	   an earlier version set it inside the sheet, which unmounted before it could
	   paint (design round 6, D2). */
	const [gateReceipt, setGateReceipt] = useState("");
	/* THE PIN REFUSAL'S OWN RECEIPT (mobile UX batch 2, U2). The ★ used to flip
	   back with no word — measured on a folder-less session: POST `/pin` → 409
	   `no saved messages yet — pin it after you send one`, and the only
	   user-visible outcome was "tap does nothing". The sentence is the daemon's
	   own, composed by the SAME helper the list's sheet uses (`lib/pin-refusal`)
	   so one refusal cannot grow two wordings.

	   IT CARRIES THE CLAIM IT WAS RAISED UNDER, so it can retire exactly when
	   that claim stops being true (U17). The 409 above is a statement about the
	   transcript being EMPTY — "pin it after you send one" — and the reader who
	   follows it saw the stale line sit above their new message until the next
	   pin gesture. `empty` records whether this refusal was that one; the effect
	   below clears it on the next projection that carries a message, and leaves
	   refusals that were never about emptiness (a network failure on a
	   conversation with history) alone. */
	const [pinRefusal, setPinRefusal] = useState<{ text: string; empty: boolean } | null>(
		null,
	);
	/* THE PIN STATE COMES FROM THE LIST STORE, which is the same row the daemon
	   serves on the list frame — so this control and the list's ★ agree by
	   construction rather than by two reads of the pin file. `undefined` (an
	   older daemon, or a session the list has not carried yet) reads as unpinned,
	   which is the honest default: the button then offers to pin, and the next
	   list repaint corrects it if that was wrong.

	   The MARK the user just made wins over the confirmed flag for what this
	   control RENDERS, exactly as it does for the list row's ★: the press must
	   answer immediately. Only the list's own sectioning waits for the daemon —
	   see the mark/section split in `store.ts` for why a row must not move until
	   the pin is confirmed. */
	const { sessions } = useSessions();
	const pinMarks = usePinMarks();
	const row = sessions.find((r) => r.session_id === sessionId);
	const pinned = pinMarks.get(sessionId) ?? Boolean(row?.pinned);
	const togglePin = async () => {
		const next = !pinned;
		setPinRefusal(null);
		/* Optimistic mark, then confirmed: the ★ flips at once and the daemon's
		   next repaint is the authority for the list. */
		applySessionPin(sessionId, next);
		try {
			const saved = await setSessionPin(sessionId, next);
			/* The route answers with the state it READ BACK, so a 200 that disagrees
			   is the daemon saying it did not pin the row. The mark must go now:
			   `settlePinMarks` retires a mark only when a later frame AGREES with it,
			   so a disagreeing one never settles and the ★ would stay on a row the
			   daemon never pinned. */
			if (saved.pinned !== next) clearSessionPinMark(sessionId);
		} catch (error) {
			/* A refusal takes the mark back with it AND says why, in the strip
			   under this header: the list already renders the daemon's reason for
			   the same refusal, and a silent flip-back here read as a dead
			   control (U2). The mark still has to go now — the list would
			   otherwise keep showing a ★ the daemon never accepted until its next
			   repaint, which is the one thing this screen cannot promise. */
			clearSessionPinMark(sessionId);
			setPinRefusal({
				text: pinRefusalText(error),
				empty: projection.transcript.length === 0,
			});
		}
	};
	/* The 409's own exit, taken as soon as it is true (U17): the next projection
	   whose transcript carries a row — the message the refusal asked for. */
	useEffect(() => {
		if (pinRefusal?.empty && projection.transcript.length > 0) setPinRefusal(null);
	}, [pinRefusal, projection.transcript.length]);
	return (
		<>
		<header className="flex items-center gap-2 border-b border-hairline px-1 py-1 pt-[max(env(safe-area-inset-top),0.25rem)]">
			<button
				type="button"
				onClick={() => navigate("/")}
				aria-label="back to sessions"
				className={cn(HEADER_CONTROL, "text-ink-muted")}
			>
				‹
			</button>
			<span className="min-w-0 flex-1 truncate text-body-sm font-medium">
				{projection.conversation_name || "untitled"}
			</span>
			{/* THE DISCOVERABLE PIN. The list also pins on a long-press, but a gesture
			    with no affordance is undiscoverable on its own — this header control is
			    where a reader finds the feature, and it is the SAME shared store, so
			    pinning here moves the list's ★ Pinned section too. `★`/`☆` rather than a
			    word: the header is width-starved and the star is the mark the section
			    heading already uses, so the two cannot be read as different things. */}
			<button
				type="button"
				onClick={() => void togglePin()}
				aria-label={pinned ? "unpin this session" : "pin this session"}
				aria-pressed={pinned}
				className={cn(
					HEADER_CONTROL,
					pinned ? "text-accent" : "text-ink-muted",
				)}
			>
				{pinned ? "★" : "☆"}
			</button>
			{/* THE ASKS ENTRY (§5.0). Present only while this conversation has
			    something waiting, so the header never grows chrome for a state the
			    session is not in — and absent at zero, exactly like the minimized
			    bar. The COUNT is here rather than on the gate control because the
			    two are different promises: a gate holds the turn, an ask does not.

			    THE NUMBER IS VISIBLE, and it is the bar's own number (design round
			    1, N3; agent review round 1, R3): it used to live only in the
			    aria-label, so the eye saw "? asks" while the bar beside it said
			    "4 questions waiting" — one screen stating one queue two ways. Both
			    now read the outstanding set in questions. */}
			{askQuestionCount > 0 ? (
				<button
					type="button"
					onClick={onOpenAsks}
					aria-label={`queued asks in this session (${askQuestionCount} question${
						askQuestionCount === 1 ? "" : "s"
					} waiting)`}
					className={cn(HEADER_CONTROL, "!min-w-0 px-2 text-meta text-accent")}
				>
					? {askQuestionCount}
				</button>
			) : null}
			{/* THE GATE CONTROL (stage D). On the phone this is the LOOSEN surface:
			    `/approvals auto` is authority-increasing, so it asks the runtime for a
			    per-action challenge and signs it with this phone's non-extractable key.
			    Before the redesign a phone could not loosen in ANY session, and the
			    refusal told the reader to find "the window that started this session" —
			    a window a phone cannot become. The label says what the control is about
			    (approvals) rather than naming the command; the sheet names the command.

			    `blockingPending`, not `projection.pending`: once the runtime publishes
			    `asks`, a mirrored ask is not a gate and must not make the approvals
			    control claim this session "needs you" (design §4, client rule N3). */}
			{/* THE READING WIDTH, where the reading happens (design round 1, D2).
			    The toggle used to exist only on the conversation list, so a reader who
			    noticed the narrow column had to leave the session, toggle it, and
			    reopen — the state that motivated issue #1870 is discovered HERE. It is
			    the same `WideViewButton` the list footer renders, so the two cannot
			    disagree about the label or the pressed state. */}
			<WideViewButton className="px-1.5" />
			<button
				type="button"
				onClick={() => setGateOpen(true)}
				aria-label="approvals in this session"
				className={cn(HEADER_CONTROL, "!min-w-0 px-2 text-meta text-ink-muted")}
			>
				{blockingPending(projection.pending, projection.asks) ? "needs you" : "approvals"}
			</button>
			<GateSheet
				open={gateOpen}
				onClose={() => setGateOpen(false)}
				sessionId={sessionId}
				onReceipt={setGateReceipt}
			/>
		</header>
		{gateReceipt ? (
			/* Cleared by the next tightening/loosening gesture rather than on a timer:
			   a receipt that vanishes while the user is looking at it is the defect
			   this exists to fix. */
			<p
				role="status"
				className="border-b border-hairline bg-elevated px-2 py-1 text-meta text-ink-muted"
			>
				{gateReceipt}
			</p>
		) : null}
		{pinRefusal ? (
			/* The same in-flow strip pattern as the receipt above, in the danger ink
			   the list's refusal uses: it takes layout space so it cannot cover a
			   control, and `role="alert"` announces it when it appears. */
			<p
				role="alert"
				className="border-b border-hairline bg-elevated px-2 py-1 text-meta break-words text-danger"
			>
				{pinRefusal.text}
			</p>
		) : null}
		</>
	);
}

/** The documented resume affordance for an ENDED session (mobile UX batch 2,
    U7). docs/mobile.md: "the phone card flips to *ended*, offering resume" and
    "the session is shown as ended (its history stays resumable)". A tap reopens
    the conversation as a NEW live session — the same route the past-sessions
    screen uses (`POST /api/sessions/resume`: the daemon spawns a child that
    resumes the transcript) — and the router takes the phone to it. Local state
    so the button can say `resuming…` and a refusal renders its sentence instead
    of a dead control. */

/* THE ACCEPTANCE LINE'S TWO SPELLINGS (UX round 2, U24). The first is what a
   cold child start honestly looks like; the second is the same fact once "a
   few seconds" has stopped describing it — the session still has not come
   back, which is what the reader needs to decide to wait or retry. 20s is
   generous for the ordinary respawn (the live leg's own real child took ~2s)
   and short enough that a stuck one is not dressed up as a normal start. */
const REOPENING = "reopening — this can take a few seconds";
const STILL_REOPENING = "still reopening — it has not come up yet";
const REOPEN_REPORT_MS = 20_000;

function EndedSessionStrip({ sessionId }: { sessionId: string }) {
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");
	/* THE ACCEPTANCE LINE (UX round 2, U24). A resume whose POST succeeds but
	   whose session does not come back live ends the same way it started
	   otherwise — one click, no word — and the reader cannot tell accepted
	   from failed from still-coming-up. The line says which; the strip
	   unmounting (a live frame flips `ended`, which is what removes this
	   component and its notice) is what ends it. It is also capped: "a few
	   seconds" stops being true long before the strip does, and a stale
	   acceptance reads like a promise the daemon is not keeping — past the
	   cap the sentence says what is actually true instead. */
	const [notice, setNotice] = useState("");
	const resume = async () => {
		if (busy) return;
		setBusy(true);
		setError("");
		setNotice("");
		try {
			const r = await resumeSession(sessionId);
			/* THE SAME-ROUTE NO-OP (UX round 1, U16 / agent-review MINOR 1). The
			   route echoes the id it was given, so this navigation usually lands on
			   the route that is ALREADY mounted — this component, and its `busy`,
			   survive it. Until this reset, a POST that succeeded against a session
			   that did not come back live left `resuming…` disabled forever, with no
			   way out short of leaving the screen. Retrying is safe: `spawn_session`
			   coalesces concurrent resumes onto one constructor and a later attempt
			   reattaches the same owner. */
			setBusy(false);
			setNotice(REOPENING);
			navigate(`/s/${encodeURIComponent(r.session_id)}`);
		} catch (e) {
			/* One refusal voice (design round 1, D5): prefixed like the pin's,
			   instead of the raw daemon string. */
			setError(resumeRefusalText(e));
			setBusy(false);
		}
	};
	useEffect(() => {
		if (notice !== REOPENING) return;
		const t = setTimeout(() => setNotice(STILL_REOPENING), REOPEN_REPORT_MS);
		return () => clearTimeout(t);
	}, [notice]);
	return (
		<>
			<div className="flex items-center gap-2 border-b border-hairline bg-elevated px-2 py-1">
				<div className="min-w-0 flex-1">
					{/* THE TITLE, ONE TEXT LINE AT EVERY WIDTH (design round 3, D10).
					    `this session has ended — its history is kept` dropped its last
					    word onto a second line at 320 and made the strip 63.17px where
					    one line does the job; the shorter sentence keeps the promise
					    (`history kept`) in the same register. */}
					<p role="status" className="text-meta text-ink-muted">
						session ended — history kept
					</p>
					{/* WHERE IT REOPENS (UX round 1, U18; the words in round 2, U25 =
					    D8). The daemon resumes the transcript in the owner's home — the
					    durable directory does not record a cwd to resume into — so the
					    strip says so before the tap rather than letting a project session
					    quietly come back in `~`. The PATH IS SPELLED OUT: a bare `~` is
					    shell shorthand a phone reader should not have to decode
					    (measured at 390 fitting beside the button, and at 320 on its own
					    line inside the row). Inside the row's text cell: at 390 it costs
					    no height at all beside the 44px button, and the row's growth at
					    320 is bounded by the sentence it belongs to.

					    AND THAT A SEND DOES IT TOO (issue #1875): the daemon wakes a host for
					    a prompt to an ended session, so resume is the way to reopen WITHOUT
					    composing, not a prerequisite for continuing. */}
					<p className="mt-0.5 text-meta text-ink-dim">
						a send or resume reopens it in your home folder
					</p>
				</div>
				{/* A real 44px target inside a `pointer-events-none` overlay: the
				    row's dead space passes touches through to the transcript, and
				    only this control claims them. */}
				<button
					type="button"
					disabled={busy}
					onClick={() => void resume()}
					className="pointer-events-auto min-h-11 shrink-0 rounded-sm border border-control px-3 text-body-sm active:bg-surface disabled:opacity-50"
				>
					{busy ? "resuming…" : "resume"}
				</button>
			</div>
			{notice ? (
				<p
					role="status"
					className="border-b border-hairline bg-elevated px-2 py-1 text-meta text-ink-dim"
				>
					{notice}
				</p>
			) : null}
			{error ? (
				<p
					role="alert"
					className="border-b border-hairline bg-elevated px-2 py-1 text-meta break-words text-danger"
				>
					{error}
				</p>
			) : null}
		</>
	);
}

export function SessionScreen({
	sessionId,
	jobId,
}: {
	sessionId: string;
	jobId?: string;
}) {
	const { projection, connected } = useProjection(sessionId);
	/* THE ROUTE'S OWN TAB TITLE (U4, batch 2). A phone's task switcher and share
	 * sheet read `document.title`, and a session wearing the list's `(3) local
	 * operator` described the wrong screen. The session route names the
	 * conversation; the agent route names the child (the roster row's label is
	 * the best identity available before the detail fetch lands). `null` while
	 * nothing is known yet, which leaves the list aggregate in place — the store
	 * owns that half and releases it when this unmounts (see `useRouteTitle`). */
	const agentLabel = jobId
		? projection?.subagents.find((row) => row.job_id === jobId)?.label
		: undefined;
	useRouteTitle(
		jobId
			? `${agentLabel || "agent"} — local operator`
			: projection
				? `${projection.conversation_name || "untitled"} — local operator`
				: null,
	);
	/* Commands this device has sent that the session has not written a row for
	 * yet. Read above the `!projection` return because hooks cannot be called
	 * after it, and resolved against the transcript by id — see `pending-echo.ts`. */
	const pendingEchoes = usePendingEchoes(sessionId, projection?.transcript);
	/* THE BLOCKING PENDING REQUEST, with the legacy ask mirror removed (§4's
	   client rule N3). One derivation, read by the gate control's label, the two
	   panels' `forceCollapsed` and the card's own render site: an ask that rides
	   the single-slot mirror must not collapse the panels, must not make the
	   header say "needs you", and must not be rendered as a second copy of a row
	   the queued surface already shows. */
	const blocking = blockingPending(projection?.pending, projection?.asks);
	/* ONE NUMBER FOR THE WHOLE SCREEN: the header entry and the minimized bar are
	   two doors to one sheet, so they state the same population in the same unit
	   (the outstanding set — open plus still-answerable timeouts — counted in
	   QUESTIONS, which is what §5.0's own bar copy counts). */
	const askQuestionCount = outstandingAsks(projection?.asks).reduce(
		(total, row) => total + (Array.isArray(row.questions) ? row.questions.length : 0),
		0,
	);
	const [modelsOpen, setModelsOpen] = useState(false);
	const [effortOpen, setEffortOpen] = useState(false);
	/* THE ONE-SHOT FOCUS INTENT (see ``lib/pending-focus.ts``). Consumed at MOUNT
	   so it cannot leak into a later mount of the same session, and consumed only
	   for the conversation ROOT: the agent route renders no composer, and burning
	   the flag there would lose the intent the tap expressed. */
	const [autoFocusComposer] = useState(() =>
		(jobId ? false : consumePendingFocus(sessionId)),
	);
	/* The asks sheet — the EXPANDED half of R7 (§5.0). Client-local interaction
	   state with no wire meaning, exactly as the design's §5.0 states. */
	const [asksOpen, setAsksOpen] = useState(false);
	/* BACK CLOSES THE SHEET, it does not leave the conversation (UX round 1, U2).
	   On a phone the system back gesture is the primary escape from a modal, and
	   on this surface it popped the ROUTE — `#/s/asks` → `#/` — taking the answer
	   draft with it. The sheet therefore claims one history entry while it is
	   open, and any back that lands on it collapses the sheet first; a second
	   back leaves the conversation as it always did.

	   The entry carries no URL change (the hash router owns the address), so a
	   pop re-renders the SAME route and the sheet's own listener is what reacts. */
	const openAsks = useCallback(() => {
		window.history.pushState({ ...window.history.state, askSheet: true }, "");
		setAsksOpen(true);
	}, []);
	const closeAsks = useCallback(() => {
		setAsksOpen(false);
		/* Give the entry back when the user closes by hand (✕, scrim, Escape), so
		   the next Back is not eaten by a sheet that is already closed. */
		if (window.history.state?.askSheet) window.history.back();
	}, []);
	/* LEAVING THE SHEET FOR A FOREIGN CONVERSATION REPLACES ITS ENTRY.
	 *
	 * The sheet's own entry is the current one when this fires, so replacing it
	 * puts the target route exactly where the sheet was: no extra history stop,
	 * and Back from the target returns to where the reader was before opening the
	 * sheet. Closing by hand (above) still gives the entry back with `back()`,
	 * which is the right thing for a dismissal — but NOT for a navigation, which
	 * is why this is a separate path and not `closeAsks()` plus a push: the pop
	 * and the push raced (agent review round 2, M1), and the entry the push had
	 * created was the one the pop then discarded.
	 *
	 * THE FLAG IS CLEARED WITH THE ENTRY. `navigate` spreads the current state
	 * into the route it pushes, so an `askSheet` left set here would ride into
	 * the target's entry and let a later dismissal pop a route the reader is
	 * standing on. */
	const openForeignConversation = useCallback((target: string) => {
		setAsksOpen(false);
		window.history.replaceState({ ...window.history.state, askSheet: false }, "");
		navigate(`/s/${target}`, { replace: true, hasInAppPredecessor: true });
	}, []);
	useEffect(() => {
		if (!asksOpen) return;
		const onPop = () => setAsksOpen(false);
		window.addEventListener("popstate", onPop);
		return () => window.removeEventListener("popstate", onPop);
	}, [asksOpen]);
	const rootRef = useRef<HTMLDivElement>(null);
	/* THE RUNG'S HEIGHT, MEASURED BECAUSE IT IS THE OVERLAY'S (round 2,
	   U23 = D7). The ladder is an overlay (round 1, U19/D4) so the column's
	   geometry never moves when a rune appears — but an overlay that nothing
	   compensates for is a strip painted OVER the transcript, and on a
	   transcript that does not scroll (a short ended session: `scrollHeight ==
	   clientHeight`) no gesture can reveal what it covers: measured 28 of the
	   first row's 35px at 390 and 38 of 56px at 320, under a strip whose own
	   sentence says "its history is kept". The rune's height is handed to the
	   transcript, which reserves the same amount INSIDE its scroller (see
	   `topInset` there), so the first row sits under the strip and the strip
	   alone moves nothing. A callback ref rather than an effect on mount,
	   because the ladder only exists once a projection does — `[ladderEl]`
	   attaches the observer exactly when the node arrives. Measured instead
	   of pinned to a constant: the ended strip wraps (52px at 390, 63px at
	   320) and its refusal/acceptance lines have heights of their own. */
	const [ladderEl, setLadderEl] = useState<HTMLDivElement | null>(null);
	const [ladderInset, setLadderInset] = useState(0);
	useLayoutEffect(() => {
		if (!ladderEl) return;
		const measure = () => setLadderInset(ladderEl.getBoundingClientRect().height);
		measure();
		if (typeof ResizeObserver === "undefined") return;
		const ro = new ResizeObserver(measure);
		ro.observe(ladderEl);
		return () => ro.disconnect();
	}, [ladderEl]);

	useEffect(() => retainProjectionStream(sessionId), [sessionId]);
	/* THE LIST STREAM TOO, so the header's ☆/★ reflects the shared pin
	   authoritatively rather than optimistically forever (review round 1, MINOR 1).
	   This header reads its pin state from the LIST store (`useSessions`); without
	   retaining that stream a session opened directly by URL would never receive a
	   repaint, so a pin set here — or cleared on another surface — would not
	   correct itself until navigation.

	   STATED RATHER THAN CALLED "FREE": the stream is refcounted per MOUNT, and
	   React destroys before it creates, so navigating between the list and a
	   session closes and reopens the SSE (measured: `{opened:1,closed:0}` ->
	   `{opened:2,closed:1}`). That is acceptable — one reconnect on a user-driven
	   navigation, over a short-lived socket — but it is not free, and a reader
	   should not plan around it being. (Review round 2, M2-3.) */
	useEffect(() => retainSessionListStream(), []);

	useCompletionView(sessionId, projection, rootRef,
		!connected || Boolean(jobId) || modelsOpen || effortOpen);

	/* Keep the composer above the iOS keyboard: pin the layout column to
	   the visual viewport's height and offset while the keyboard is open. */
	useEffect(() => {
		const vv = window.visualViewport;
		const el = rootRef.current;
		if (!vv || !el) return;
		const sync = () => {
			/* Height + top, never transform: a transform on this column
			   creates a containing block that traps `position: fixed`
			   sheets (slash, model, effort) and clips them to a sliver. */
			el.style.height = `${vv.height}px`;
			el.style.top = `${vv.offsetTop}px`;
			/* Same number, published as a length the children's caps can be
			   written in. Written from THIS handler on purpose: a cap fed by a
			   second source of truth would drift from the pin the moment one
			   of them changed, which is the `dvh`-vs-`visualViewport`
			   divergence this property exists to end. Consumers read it as
			   `var(--lo-vvh, 100dvh)`, so a surface outside this column — or a
			   browser without `visualViewport` — still gets the viewport-
			   relative bound it had before.

			   The TOP is published beside the height for the sheets: a `fixed`
			   overlay resolves against the LAYOUT viewport, and the keyboard
			   shrinks only the VISUAL one, so an overlay anchored to the layout
			   viewport would hold its foot under the keyboard. `lib/column.ts`'
			   `columnBox()` reads both, which puts the overlay on exactly the
			   box this column is pinned to. */
			el.style.setProperty(COLUMN_HEIGHT_VAR, `${vv.height}px`);
			el.style.setProperty(COLUMN_TOP_VAR, `${vv.offsetTop}px`);
		};
		sync();
		vv.addEventListener("resize", sync);
		vv.addEventListener("scroll", sync);
		return () => {
			vv.removeEventListener("resize", sync);
			vv.removeEventListener("scroll", sync);
			el.style.height = "";
			el.style.top = "";
			el.style.removeProperty(COLUMN_HEIGHT_VAR);
			el.style.removeProperty(COLUMN_TOP_VAR);
		};
	}, []);

	if (!projection) {
		return (
			<div
				ref={rootRef}
				className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden"
			>
				<header className="flex items-center gap-2 border-b border-hairline px-1 py-1 pt-[max(env(safe-area-inset-top),0.25rem)]">
					<button
						type="button"
						onClick={() => navigate("/")}
						aria-label="back to sessions"
						className={cn(HEADER_CONTROL, "text-ink-muted")}
					>
						‹
					</button>
				</header>
				<div className="flex flex-1 items-center justify-center">
					<p className="text-body-sm text-ink-dim">
						{connected
							? "waiting for projection…"
							: "connecting to session…"}
					</p>
				</div>
			</div>
		);
	}

	/* An empty session shows the placeholder only while it is genuinely empty. A
	   prompt this device just sent makes the column non-empty the instant it
	   leaves the composer, and hiding its row behind "no messages yet" would put
	   the placeholder in front of the very message that proves the send worked. */
	const showEmptyState =
		projection.transcript.length === 0 && !projection.streaming && pendingEchoes.length === 0;

	return (
		<div
			ref={rootRef}
			className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden"
		>
			{jobId ? (
				<AgentScreen
					sessionId={sessionId}
					jobId={jobId}
					projection={projection}
					connected={connected}
				/>
			) : <>
			{/* THE HEADER, THE GLANCE ROW AND THE STATE LADDER SHARE ONE RELATIVE
			    WRAPPER, because the ladder is an OVERLAY that rides UNDER both (UX
			    round 1, U19 = design round 1, D4; anchor moved in round 2, D9).
			    Every rung used to be an in-flow row, so each appearance/clear pushed
			    the whole column down and back — before the overlay: +26px when the
			    degraded/reconnect strip appeared and +53px for the ended strip, on
			    every flap of a flaky link. Reserved space would have paid the same
			    pixels permanently (52px on a 320-wide phone, for a strip most
			    sessions never show); the overlay keeps the column's geometry fixed
			    for all three rungs. Round 2 D9 moved the anchor from the header to
			    the WRAPPER'S bottom: at the header it painted over the spend/context
			    glance (6.2%/200k · $1.25 — the numbers a reader most wants when a
			    session is struggling) — and round 2 U23 = D7 pays for the rest of
			    the cover: the transcript reserves the rung's height inside its own
			    scroller, so nothing under a rung is unreachable. `pointer-events-none`
			    hands touches back to the transcript underneath — scrolling from the
			    strip's own row must keep working — and the rung's interactive part
			    (the ended strip's resume) opts back in with `pointer-events-auto`.
			    Sheets are `fixed z-50`; this overlay sits at `z-10`, under them. */}
			<div className="relative">
				<Header
					projection={projection}
					sessionId={sessionId}
					onOpenAsks={openAsks}
					askQuestionCount={askQuestionCount}
				/>
				{/* The spend + context glance (phase 1), read-only and self-hiding:
				    it renders nothing until either reading has something to state. */}
				<SessionStatus projection={projection} />
				{/* SESSION HEALTH, AS ONE LADDER (mobile UX batch 2, U7 + U11). Each
				    rung is a different fact and the later ones are only worth stating
				    while the earlier are untrue, so at most ONE strip shows — three
				    stacked on a 320-wide phone would spend the vertical budget the
				    batch-1 work just bought back. `ended` (the process is gone; the
				    resume affordance lives in the strip) outranks `degraded` (the
				    relay's dial is down — sends will fail until it answers) outranks
				    the phone's own link being down (`connected === false`, the store's
				    flag; the retained view is what the reader is looking at). Each
				    clears itself when the daemon's next projection (or the SSE's own
				    reopen) says otherwise. */}
				<div
					ref={setLadderEl}
					className="pointer-events-none absolute inset-x-0 top-full z-10"
				>
					{projection.ended ? (
						<EndedSessionStrip sessionId={sessionId} />
					) : projection.degraded ? (
						<p
							role="status"
							className="border-b border-hairline bg-warning-wash px-2 py-1 text-meta text-warning"
						>
							not answering — showing its last synced view
						</p>
					) : !connected ? (
						<p
							role="status"
							className="border-b border-hairline bg-elevated px-2 py-1 text-meta text-ink-muted"
						>
							reconnecting — showing the last synced view
						</p>
					) : null}
				</div>
			</div>
			{showEmptyState ? (
				/* A just-started session has no messages yet. An empty scroll
				   area reads as "did it break?"; this placeholder says the
				   session is ready and what to do next. Hidden the moment a
				   turn begins streaming (the transcript fills from the user
				   row up). */
				<div className="flex flex-1 flex-col items-center justify-center gap-1 px-8 text-center">
					<p className="text-body-sm text-ink-muted">no messages yet</p>
					<p className="text-meta text-ink-dim">
						send a message below to get started
					</p>
				</div>
			) : (
				<Transcript
					pid={sessionId}
					entries={projection.transcript}
					pending={pendingEchoes}
					streaming={projection.streaming}
					topInset={ladderInset}
				/>
			)}

			{/* The aggregate working line — pinned at the foot of the transcript
			    like the TUI's WorkingBlock, above the panels and composer. */}
			{projection.streaming ? (
				<WorkingLine
					activity={projection.activity}
					startedS={projection.activity_started_s}
				/>
			) : null}

			{/* The panel budget (D1). These two panels and the pending card are
			    unshrinkable-ish siblings in a column that is `overflow-hidden`, so
			    what they claim together comes off the bottom and is CLIPPED, not
			    scrolled. Measured with both panels expanded beside an approval at
			    390x844: the column reported `clientH 844 / scrollH 1427`, approve
			    and deny were 120px below the fold with the card's own scroller
			    already at its end, and at 360x780 the card's top landed at y=781 in
			    a 780px viewport — entirely off screen. Real touch drags over the
			    panels and over the card moved nothing (`colScrollTop=0` throughout);
			    only a script assigning `scrollTop` to an `overflow:hidden` element
			    appeared to recover it, which a finger cannot do.

			    So while a request is pending the panels render COLLAPSED and hold
			    shut. A question outranks a task list and a roster (branding §7):
			    the card is the only thing on screen that needs a decision, and the
			    panels are the only things that can push it off. This is a budget
			    rather than a cap because caps compose badly — three individually
			    bounded regions still sum past the column, which is how 40%+40%+60%
			    overran it. Their own open state is kept, so answering restores what
			    the user had open. With `min-h-0` on both panels (see each) the
			    column can also always fit its children rather than clipping them. */}
			{projection.todos.some((p) => p.items.length > 0) ? (
				<TodosPanel
					todos={projection.todos}
					forceCollapsed={Boolean(blocking)}
				/>
			) : null}
			{projection.subagents.length > 0 ? (
				<SubagentsPanel
					pid={sessionId}
					subagents={projection.subagents}
					forceCollapsed={Boolean(blocking)}
				/>
			) : null}

			{blocking ? (
				/* Key the WHOLE card on request_id + question_index so React
				   remounts it for each question of a multi-part ask. A
				   multi-question ask keeps the SAME request_id pending and only
				   advances question_index (see mobile/projection.py set_pending/
				   _sync_pending) — projection.pending never goes null between
				   questions, so without this key React reuses the one
				   PendingCard instance and its transient useState (busy/error/
				   remember/free-text draft) leaks across questions. That leak is
				   the greyed-out-buttons bug: busy stayed true after answering
				   Q1, so every Q2 option rendered disabled. The remount is the
				   honest fix — it makes the invariant structural rather than
				   relying on the card to reset itself. kind is included as
				   cheap hardening: request_ids are unique per push today, so
				   an approval→ask flip at the same index can't collide in
				   practice, but if the daemon ever reuses an id across kinds
				   the card must not inherit the other kind's state. Known
				   trade-off, deferred from review round 1 (A2): for parallel
				   approvals the remount also resets the `remember` checkbox
				   on the next card. Harmless today — tui_handle's
				   approval_answer ignores `remember` entirely — and no clean
				   key shape fixes it without also breaking the ask remount;
				   revisit only if the daemon ever wires `remember` through
				   to a per-tool store. */
				<PendingCard
					key={`${blocking.request_id}:${blocking.kind}:${blocking.question_index}`}
					pid={sessionId}
					pending={blocking}
					count={projection.pending_count}
				/>
			) : null}

			{/* THE MINIMIZED ASK BAR (§5.0, R7). It sits directly above the composer
			    and is the only ask affordance on this screen: tapping it opens the
			    asks sheet, which is where answering happens. It is absent at zero
			    asks, and its presence CHANGES NOTHING about the composer beneath it —
			    with the bar showing, the composer is an ordinary conversation
			    composer, so a message typed there can never be sent as an answer.
			    That is the half of §5.0's routing rule this screen enforces; the other
			    half ("while EXPANDED the composer sends the answer") is enforced by the
			    sheet being modal over this column: while it is open, the sheet's own
			    answer fields are the only inputs that can receive a keystroke. */}
			<AskDock rows={projection.asks} onOpen={openAsks} />

			<Composer
				pid={sessionId}
				projection={projection}
				onOpenModels={() => setModelsOpen(true)}
				onOpenEffort={() => setEffortOpen(true)}
				effortOpen={effortOpen}
				onCloseEffort={() => setEffortOpen(false)}
				autoFocus={autoFocusComposer}
			/>

			<ModelSheet
				open={modelsOpen}
				onClose={() => setModelsOpen(false)}
				pid={sessionId}
				projection={projection}
			/>

			<AsksSheet
				open={asksOpen}
				onClose={closeAsks}
				currentSessionId={sessionId}
				onOpenConversation={openForeignConversation}
			/>
			</>}
		</div>
	);
}
