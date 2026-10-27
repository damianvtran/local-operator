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
import { flushSync } from "react-dom";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { setSessionPin } from "../api";
import { AskDock } from "../components/ask-dock";
import { AsksSheet } from "../components/asks-sheet";
import { ModelSheet } from "../components/model-sheet";
import { Composer, ComposerFrame } from "../components/composer";
import { GateSheet } from "../components/gate-sheet";
import { SessionMenu } from "../components/session-menu";
import { SessionStatus } from "../components/session-status";
import { PendingCard } from "../components/pending-card";
import { SubagentsPanel } from "../components/subagents-panel";
import { TodosPanel } from "../components/todos-panel";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { SessionHealthControl, useResume } from "../components/session-health";
import { sessionHealth } from "../lib/session-health";
import { AboveComposerStack } from "../components/above-composer";
import { Sheet } from "../components/ui/sheet";
import { Chip } from "../components/ui/chip";
import { cn } from "../lib/cn";
import { COLUMN_HEIGHT_VAR, COLUMN_TOP_VAR, SHORT_COLUMN_PX } from "../lib/column";
import { navigate, parseHash } from "../router";
import { consumePendingFocus } from "../lib/pending-focus";
import { pinRefusalText } from "../lib/pin-refusal";
import { useCompletionView } from "../use-completion-view";
import { usePendingEchoes } from "../pending-echo";
import { AgentScreen } from "./agent-view";
import {
	applySessionPin,
	clearSessionPinMark,
	messageRuntime,
	retainProjectionStream,
	retainSessionListStream,
	usePinMarks,
	useProjection,
	useRouteTitle,
	useSessions,
} from "../store";
import { blockingPending, outstandingAsks } from "../lib/asks";
import { askOpenPolicy, namesEveryOutstanding, readQueue } from "../lib/ask-open-policy";
import type { SessionProjection } from "../types";

/** A field the keyboard types into. The phone's reading of "the user is typing" is
    FOCUS, not a draft sitting in storage (see the open-by-default block in
    `SessionScreen`): a focused text field is a raised keyboard, and a modal sheet
    opening over it would drop that keyboard out from under a sentence in progress.
    Checkboxes and buttons are excluded because focusing one is not typing. */
const TEXT_ENTRY_SELECTOR =
	'textarea, input:not([type="checkbox"]):not([type="radio"]):not([type="button"]):not([type="submit"]):not([type="reset"]), [contenteditable=""], [contenteditable="true"]';

function aTextFieldHoldsTheKeyboard(): boolean {
	const active = document.activeElement;
	return active instanceof HTMLElement && active.matches(TEXT_ENTRY_SELECTOR);
}

/** Whether some other sheet is already up. Every picker on the phone is a `Sheet`
    (model, effort, slash, gate), which renders `role="dialog"`, so asking the DOM
    covers the ones whose open state lives in a child component this screen cannot
    see. A second modal stacked over the first is never what a user asked for. */
function anotherSheetIsOpen(): boolean {
	return document.querySelector('[role="dialog"]') !== null;
}

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
	connected,
	ended,
	gateOpen,
	setGateOpen,
	onOpenSpend,
	fitLevel,
}: {
	projection: SessionProjection;
	sessionId: string;
	connected: boolean;
	/** The screen's one `sessionHealth(...) === "ended"` (review R3-5): the
	    composer reads the same value, so the dot and the message box can never
	    disagree about whether a send reopens the session. */
	ended: boolean;
	gateOpen: boolean;
	setGateOpen: (open: boolean) => void;
	onOpenSpend: () => void;
	fitLevel: number;
}) {
	const [menuOpen, setMenuOpen] = useState(false);
	/* THE HEALTH SHEET AND THE ONE RESUME PATH, held here (UX round 1, U3) so the
	   ⋮ menu can offer Resume while the session is ended without a second copy
	   of `useResume`: the menu runs it, then opens the health sheet, which is
	   where its acceptance line or refusal is painted. */
	const [healthOpen, setHealthOpen] = useState(false);
	const resume = useResume(sessionId, ended);
	/* The opener the menu's sheet returns focus to. Held here because the ⋮
	   button lives here and the sheet is the only consumer that needs it. */
	const menuButtonRef = useRef<HTMLButtonElement>(null);
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
	/** True when the daemon accepted the pin, false when it refused — the menu
	    uses that answer to step aside so this header's refusal strip is visible
	    instead of sitting behind the open sheet. */
	const togglePin = async (): Promise<boolean> => {
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
			return true;
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
			return false;
		}
	};
	/* The 409's own exit, taken as soon as it is true (U17): the next projection
	   whose transcript carries a row — the message the refusal asked for. */
	useEffect(() => {
		if (pinRefusal?.empty && projection.transcript.length > 0) setPinRefusal(null);
	}, [pinRefusal, projection.transcript.length]);
	return (
		<>
		<header className="flex h-[calc(56px+env(safe-area-inset-top))] shrink-0 items-center gap-2 px-1 pt-[env(safe-area-inset-top)]">
			<button
				type="button"
				onClick={() => navigate("/")}
				aria-label={messageRuntime.t("relay.chrome.session.back")}
				className={cn(HEADER_CONTROL, "text-ink-muted")}
			>
				‹
			</button>
			<div className="min-w-0 flex-1">
				{/* THE STATUS DOT (operator request, replacing the ended banner): the
				    title block is the control that opens the health sheet, so its
				    44 px target costs the 56 px header nothing (see
				    `components/session-health.tsx`). */}
				<SessionHealthControl
					projection={projection}
					connected={connected}
					open={healthOpen}
					onOpenChange={setHealthOpen}
					resume={resume}
				>
					{/* THE NAME TRUNCATES, THE FACT DOES NOT (operator finding B): a long
					    conversation title is cut with an ellipsis here, so the whole string
					    stays recoverable as the element's `title` and the document title
					    already carries it. Local truncation with the full value still
					    reachable is truthful; a name painted in full that pushes the phone
					    sideways is not. */}
					<span
						className="block truncate text-heading font-semibold"
						title={projection.conversation_name || undefined}
					>{projection.conversation_name || "untitled"}</span>
				</SessionHealthControl>
			</div>
			{/* THE OVERFLOW. Pin, wide view and approvals all have labelled rows in
			    one sheet now: the `★`, `? N` and `[wide]` controls were each a
			    second unlabelled door on a width-starved row. The ask dock above
			    the composer is the labelled ask door and is always present when
			    asks exist, so the `? N` duplicate is gone rather than moved. */}
			<button
				ref={menuButtonRef}
				type="button"
				onClick={() => setMenuOpen(true)}
				aria-label="session menu"
				className={cn(HEADER_CONTROL, "text-ink-muted")}
			>
				⋮
			</button>
			<SessionMenu
				open={menuOpen}
				onClose={() => setMenuOpen(false)}
				pinned={pinned}
				onTogglePin={() => togglePin()}
				onOpenApprovals={() => setGateOpen(true)}
				onOpenSpend={onOpenSpend}
				showSpend={fitLevel >= 2 && Boolean(projection.usage && Object.keys(projection.usage).length > 0)}
				onResume={ended ? () => {
					void resume.resume();
					setHealthOpen(true);
				} : undefined}
				resuming={resume.busy}
				returnFocusRef={menuButtonRef}
			/>
			<GateSheet
				open={gateOpen}
				onClose={() => setGateOpen(false)}
				sessionId={sessionId}
				onReceipt={setGateReceipt}
				/* THE STATE THE SHEET SHOWS COMES FROM THE RUNTIME. The projection
				   publishes the effective mode beside a pending approval, so that is
				   where this screen reads it — never from `gateReceipt`, which reports
				   only what THIS screen last did and would call a session auto after
				   the terminal tightened it. Absent while nothing is pending, which the
				   sheet renders as no line rather than a guess. */
				approvalMode={projection.pending?.approval_mode}
				approvalProvenance={projection.pending?.approval_provenance}
				approvalGlobalMode={projection.pending?.approval_global_mode}
			/>
		</header>
		{gateReceipt ? (
			/* Cleared by the next tightening/loosening gesture rather than on a timer:
			   a receipt that vanishes while the user is looking at it is the defect
			   this exists to fix. */
			<p
				role="status"
				className="bg-elevated px-2 py-1 text-meta text-ink-muted"
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
				className="bg-elevated px-2 py-1 text-meta break-words text-danger"
			>
				{pinRefusal.text}
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
	/* ONE DOOR FOR THE WHOLE SCREEN: the minimized bar above the composer is the
	   labelled ask entry (`? N` in the header was the same queue stated a second
	   time, without a word to learn it from). `answerableAsks` is the whole
	   outstanding set: it is what the open policy's dismissal memory records. */
	const answerableAsks = outstandingAsks(projection?.asks);
	const [modelsOpen, setModelsOpen] = useState(false);
	/* The model sheet's search text lives HERE (UX round 2, U5): a fresh open
	   clears it, while the effort view's back control returns to the same sheet
	   with the query intact. */
	const [modelQuery, setModelQuery] = useState("");
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
	const [gateOpen, setGateOpen] = useState(false);
	const [tasksOpen, setTasksOpen] = useState(false);
	const [agentsOpen, setAgentsOpen] = useState(false);
	/* BACK CLOSES THE SHEET, it does not leave the conversation (UX round 1, U2).
	   On a phone the system back gesture is the primary escape from a modal, and
	   on this surface it popped the ROUTE — `#/s/asks` → `#/` — taking the answer
	   draft with it. The sheet therefore claims one history entry while it is
	   open, and any back that lands on it collapses the sheet first; a second
	   back leaves the conversation as it always did.

	   The entry carries no URL change (the hash router owns the address), so a
	   pop re-renders the SAME route and the sheet's own listener is what reacts. */
	/* THE SHEET'S HISTORY ENTRY, claimed from one place so the door and the
	   open-by-default path cannot spell it two ways. */
	const claimAsksEntry = useCallback(() => {
		window.history.pushState({ ...window.history.state, askSheet: true }, "");
	}, []);
	/* THIS SCREEN'S ANSWERABLE ASK IDS, as of the latest render, for the two
	   callbacks below that run on a gesture rather than on a render. Written during
	   render (the idiom `ui/sheet.tsx` uses for its own `onCloseRef`) so a close
	   always names the set the user was looking at when they closed. */
	const answerableIdsRef = useRef<string[]>([]);
	answerableIdsRef.current = answerableAsks.map((row) => row.ask_id);
	/* OPEN BY DEFAULT, clause 4 of the shared open-policy contract (see
	   `lib/ask-open-policy.ts`): a close the USER made while asks remain is
	   remembered, per conversation and by ask id, so a later view of this
	   conversation does not open the sheet over a question they already waved off.
	   Recorded only from the three gestures below that are the user turning the
	   sheet away — never from unmounting, never from navigating to another
	   conversation (by the sheet's own `open` control or from outside), which say
	   nothing about this one's asks. */
	const noteAsksDismissed = useCallback(() => {
		askOpenPolicy.noteUserClosed(sessionId, answerableIdsRef.current);
	}, [sessionId]);
	/* THE DOOR (the dock and the header entry). A user pressing it also settles the
	   view's pending decision, so a frame that lands afterwards cannot open a sheet
	   over them a second time (clause 6: opening by policy is not pressing the
	   door, and pressing the door is not something the policy gets to repeat). */
	const openAsks = useCallback(() => {
		askOpenPolicy.noteUserOpened(sessionId);
		claimAsksEntry();
		setAsksOpen(true);
	}, [sessionId, claimAsksEntry]);
	/* THE POLICY'S OWN OPEN. The same sheet and the same Back behaviour as the
	   door, minus the "a user pressed it" bookkeeping. The entry is claimed only if
	   this history entry does not already carry one: React's StrictMode (the dev
	   server) re-runs mount effects, which can reach here twice for one view, and a
	   second entry would make Back close the sheet and then close it again.

	   A KNOWN LIMIT OF CLAIMING THE ENTRY HERE, from the platform and not from this
	   code: unlike the door, this `pushState` runs from an effect with no gesture
	   behind it. Chromium's history-manipulation intervention marks a document's
	   entries "skippable" for the browser's Back button when the document adds an
	   entry without a user activation, and lifts that the moment the document
	   receives one (chromium `docs/history_manipulation_intervention.md`, invariants
	   2-4; it never touches `history.back()`, which is what ✕ and the scrim use).
	   So on a page nobody has tapped yet (a cold deep link from a notification), the
	   very first Back press can leave the page instead of closing the sheet; after
	   any tap it closes the sheet as designed. Nothing is lost either way: the ask
	   is still pending and the sheet has its own always-present close control.
	   Claiming no entry would be worse in the common case (a tap on the session
	   list came first), because Back would then leave the conversation with the
	   sheet still up. Not reproduced here: a headless browser cannot press the
	   browser's Back button. */
	const openAsksByPolicy = useCallback(() => {
		if (window.history.state?.askSheet !== true) claimAsksEntry();
		setAsksOpen(true);
	}, [claimAsksEntry]);
	const closeAsks = useCallback(() => {
		noteAsksDismissed();
		setAsksOpen(false);
		/* Give the entry back when the user closes by hand (✕, scrim, Escape), so
		   the next Back is not eaten by a sheet that is already closed. */
		if (window.history.state?.askSheet) window.history.back();
	}, [noteAsksDismissed]);
	useEffect(() => {
		if (!asksOpen) return;
		const onPop = () => {
			/* WHERE THE POP LANDED DECIDES WHETHER IT WAS A DISMISSAL. The phone's Back
			   gesture is the primary escape from a modal (U2), and the sheet's entry
			   carries no URL change, so a Back out of the sheet lands on THIS
			   conversation's own route: the user turned the sheet away. A pop that lands
			   anywhere else is a navigation, and it fires `popstate` too: the browser's
			   when a notification tap or a deep link changes the hash behind an open sheet.
			   None of those is the user waving this conversation's asks off, and recording
			   one would switch the open-by-default policy off for a conversation whose
			   sheet nobody closed. The sheet still closes in every case, as it always did. */
			const landed = parseHash(window.location.hash);
			const here =
				landed.name === "session" && landed.sessionId === sessionId && landed.jobId === undefined;
			if (here) noteAsksDismissed();
			setAsksOpen(false);
		};
		window.addEventListener("popstate", onPop);
		return () => window.removeEventListener("popstate", onPop);
	}, [asksOpen, noteAsksDismissed, sessionId]);
	/* OPEN BY DEFAULT — the phone's half of the shared open-policy contract. The six
	   clauses are numbered in `lib/ask-open-policy.ts`; this block only gathers the
	   facts and obeys the answer, so the screen carries no second copy of the rules.

	   A VIEW BEGINS HERE, on every mount of the conversation root. `app.tsx` keys
	   this screen by session and job, so opening a conversation, switching to
	   another and coming back each mount it afresh; the policy lives in module
	   state precisely so that what the user waved off survives those remounts. The
	   agent route renders no asks sheet, so it never arms a view. */
	useEffect(() => {
		if (jobId) return;
		askOpenPolicy.beginView(sessionId, Date.now());
	}, [sessionId, jobId]);
	/* THE ONE DECISION, taken from the first frame that can take it. `awaiting` is a
	   single boolean once the view has decided, so the cost of running on every
	   projection frame (the stream's rate, not the user's) is that read.

	   `!connected` is a wait and not a refusal: a retained projection kept across a
	   dropped link is the LAST good frame, not the queue's present state, and
	   deciding on it would open the sheet over asks that may already be settled.

	   `occupied` is clause 5's keyboard test, read here rather than cached because
	   it is a fact about this instant:
	     * a blocking APPROVAL card is up: a run is held, it outranks a question,
	       and the modal sheet would hide it;
	     * a text field holds focus: the keyboard is up, and a sheet that opens over
	       it drops the keyboard out from under a sentence in progress (the sheet
	       focuses its own close control and makes the column behind it inert);
	     * another sheet is already open.

	   A DRAFT SITTING IN STORAGE IS NOT "TYPING", and that is where this differs
	   from the TUI. There the composer holds the caret all the time, so its text
	   is the only observable sign of a sentence in progress. Here focus is
	   observable, and a draft restored from a previous visit is not being typed
	   into: opening the sheet takes nothing from it (drafts are written through to
	   storage, so it is intact when the sheet closes). Vetoing on it would hide the
	   sheet from every returning user who ever left a half-written message, which
	   is the discoverability failure this feature exists to remove. */
	useEffect(() => {
		if (jobId || !projection || !connected) return;
		if (!askOpenPolicy.awaiting(sessionId)) return;
		const ids = answerableAsks.map((row) => row.ask_id);
		const decision = askOpenPolicy.decide(
			sessionId,
			readQueue(projection.asks, projection.asks_open),
			{
				nowMs: Date.now(),
				occupied:
					blocking !== null || aTextFieldHoldsTheKeyboard() || anotherSheetIsOpen(),
				surfaceOpen: asksOpen,
				outstandingIds: namesEveryOutstanding(projection.asks_open, ids.length) ? ids : null,
			},
		);
		if (decision === "open") openAsksByPolicy();
	});
	const rootRef = useRef<HTMLDivElement>(null);
	/* The short layout and the footer's fit are ONE rendering decision. Keeping
	   this only as a DOM attribute let CSS flip before React had fitted its facts
	   row, painting one clipped frame on keyboard changes. Initialising from the
	   visual viewport also makes a cold landscape load compact on its first paint. */
	const [short, setShort] = useState(() =>
		typeof window !== "undefined" && (window.visualViewport?.height ?? Infinity) <= SHORT_COLUMN_PX,
	);
	const [fitLevel, setFitLevel] = useState(0);
	const [spendOpen, setSpendOpen] = useState(false);
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
	useLayoutEffect(() => {
		const vv = window.visualViewport;
		const el = rootRef.current;
		if (!vv || !el) return;
		const sync = (event?: Event) => {
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
			/* React owns data-short so this and the fit level commit together. */
			/* Later native events must commit before paint. During this layout
			   pass React already owns the flush; calling it here would warn. */
			if (event) flushSync(() => setShort(vv.height <= SHORT_COLUMN_PX));
			else setShort(vv.height <= SHORT_COLUMN_PX);
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

	/* THE RESERVED FRAME (first-paint lane T2: finding F8).

	   The projection arrives over SSE, so the first frames of a cold open have
	   no session data at all. This branch used to paint a header and one centred
	   sentence, then mount the header, the transcript, the working line and the
	   composer in a single commit when the projection landed — a full-layout
	   swap in which the composer APPEARED rather than being filled. Reserve the
	   column instead: the same header row, the same transcript box (`flex-1`,
	   holding a status line instead of rows) and the composer's box
	   (`<ComposerFrame>`, drawn from the composer's own class strings). What lands
	   is then content in boxes that were already there, and nothing below the
	   header moves.

	   WHAT IS DELIBERATELY NOT HERE: a title, a glance row, asks, panels or a
	   live composer. Every one of those reads the projection, and rendering them
	   from a fabricated stand-in would put a fiction behind the same branches
	   real data flows through — the honest empty state is a reserved box. The
	   header row's height comes from its 44 px control, so an empty title slot
	   occupies exactly the room the conversation's name will.

	   AND NOT THE STATE LADDER EITHER, which looks like an omission and is not:
	   the frame's own sentence already reports the link, and with no data on
	   screen a "showing the last synced view" strip would be noise about noise
	   (`session-view.health.test.tsx` pins that decision — U11). Nothing can be
	   "synced" yet, so the ladder keeps its existing rule: it speaks once there
	   is something to have fallen back from. */
	if (!projection) {
		return (
			<div
				ref={rootRef}
				className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden"
			>
				<div className="relative">
					<header className="flex h-[calc(56px+env(safe-area-inset-top))] shrink-0 items-center gap-2 px-1 pt-[env(safe-area-inset-top)]">
						<button
							type="button"
							onClick={() => navigate("/")}
							aria-label={messageRuntime.t("relay.chrome.session.back")}
							className={cn(HEADER_CONTROL, "text-ink-muted")}
						>
							‹
						</button>
						{/* The title's own slot, empty: `truncate` + `flex-1` in a row whose
						    height is set by the control beside it. */}
						<span className="min-w-0 flex-1 truncate text-body-sm font-medium" />
					</header>
				</div>
				{/* THE TRANSCRIPT BOX. `min-h-0 flex-1` is the whole contract: it is the
				    space `<Transcript>` will fill, so rows arriving take room that was
				    already theirs. The sentence inside is the reader's only signal that
				    the wait is a wait — it leaves with the first row. */}
				<div className="flex min-h-0 flex-1 flex-col items-center justify-center px-3">
					<p className="text-body-sm text-ink-dim">
						{connected
							? messageRuntime.t("relay.shell.session.waiting")
							: messageRuntime.t("relay.shell.session.connecting")}
					</p>
				</div>
				<ComposerFrame />
			</div>
		);
	}

	/* An empty session shows the placeholder only while it is genuinely empty. A
	   prompt this device just sent makes the column non-empty the instant it
	   leaves the composer, and hiding its row behind "no messages yet" would put
	   the placeholder in front of the very message that proves the send worked. */
	/* THE SCREEN'S ONE "ENDED" (review round 3, R3-5): derived once here and
	   handed to the header and the composer, so the header's ⋮ Resume, the
	   message box's placeholder and its failure copy cannot drift apart. The
	   dot and health sheet still derive their own labels; this is the shared
	   ended action state, not a second health rule. */
	const ended = sessionHealth(projection, connected) === "ended";
	const needsApproval = blocking?.kind === "approval";
	/* The pending card's identity: its React key (one remount per question of
	   a multi-part ask) and the slot's `restOn`, so a new decision is
	   scrolled into view on a slot too short to show it whole (D6 = R2-1). */
	const blockingKey = blocking ? `${blocking.request_id}:${blocking.kind}:${blocking.question_index}` : null;
	const taskItems = projection.todos.flatMap(phase => phase.items);
	const tasksDone = taskItems.filter(task => task.status === "done" || task.status === "dropped").length;

	/* THE ROSTER CHIP, ONCE (code review round 2, C2-5): defined here and rendered
	   from the slot this conversation's state owns — before tasks when something
	   failed, after tasks when the roster is healthy (design round 1, D2).
	   THE FULL EXISTING SENTENCE CARRIES THE NOUN AND THE FAILURE (design D5, UX
	   U3): the old bare failure mark (`X` plus a numeral) read as a dismiss control beside two counters and never named what
	   failed. "Agents 1/3 · 1 failed" says the noun, the running count and the
	   failure in the pill's own danger register, with no ✕-lookalike glyph; its
	   accessible name is that same sentence, so nothing is lost to sighted readers
	   or to assistive tech. No new copy — the two `tray.agents*` keys already exist. */
	/* The roster counters the pill renders. */
	const agentsFailed = projection.subagents.filter(agent => agent.status === "failed").length;
	const agentsRunning = projection.subagents.filter(agent => agent.status === "running").length;
	/* THE ROSTER CHIP, ONCE (code review round 2, C2-5): defined here and rendered
	   from the slot this conversation's state owns — before tasks when something
	   failed, after tasks when the roster is healthy (design round 1, D2).

	   THE FULL SENTENCE, AT EVERY WIDTH (design D5 = UX U3, and UX round 3 U7).
	   A bare failure mark plus a numeral read as a dismiss control beside two other
	   counters and never named what failed; the shortened noun form then dropped the
	   count that made the pill urgent on a 320 phone. Both forms cost information,
	   so the pill shows the full existing sentence everywhere — the accessible name
	   too — in the danger register, and the tray makes room by WRAPPING at 320
	   (gaps 4 px below 390). No new copy: the two `tray.agents*` keys already exist. */
	const agentsChip = projection.subagents.length ? (
		<Chip
			className={cn("shrink-0", agentsFailed > 0 ? "!bg-danger-wash !text-danger" : blocking && "!text-ink-dim")}
			disabled={Boolean(blocking)}
			aria-expanded={agentsOpen}
			aria-label={agentsFailed > 0
				? messageRuntime.t("relay.chrome.session.tray.agents_failed", {
						running: agentsRunning,
						total: projection.subagents.length,
						failed: agentsFailed,
					})
				: undefined}
			title={blocking ? messageRuntime.t("relay.asks.disclosure.held_shut") : undefined}
			onClick={() => setAgentsOpen(true)}
		>
			{agentsFailed > 0
				? messageRuntime.t("relay.chrome.session.tray.agents_failed", {
						running: agentsRunning,
						total: projection.subagents.length,
						failed: agentsFailed,
					})
				: messageRuntime.t("relay.chrome.session.tray.agents", {
						running: agentsRunning,
						total: projection.subagents.length,
					})}
		</Chip>
	) : null;
	const showEmptyState =
		projection.transcript.length === 0 && !projection.streaming && pendingEchoes.length === 0;

	return (
		<div
			ref={rootRef}
			data-short={short ? "" : undefined}
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
			<Header projection={projection} sessionId={sessionId} connected={connected} ended={ended} gateOpen={gateOpen} setGateOpen={setGateOpen} onOpenSpend={() => setSpendOpen(true)} fitLevel={fitLevel} />
			{showEmptyState ? (
				/* A just-started session has no messages yet. An empty scroll
				   area reads as "did it break?"; this placeholder says the
				   session is ready and what to do next. Hidden the moment a
				   turn begins streaming (the transcript fills from the user
				   row up). */
				<div className="flex flex-1 flex-col items-center justify-center gap-1 px-8 text-center">
					<p className="text-body-sm text-ink-muted">
						{messageRuntime.t("relay.chrome.session.empty_title")}
					</p>
					<p className="text-meta text-ink-dim">
						{messageRuntime.t("relay.chrome.session.empty_hint")}
					</p>
				</div>
			) : (
				<Transcript
					pid={sessionId}
					entries={projection.transcript}
					peerCycles={projection.peer_cycles === 1}
					pending={pendingEchoes}
					streaming={projection.streaming}
					topInset={0}
					tailContent={projection.streaming ? <WorkingLine activity={projection.activity} startedS={projection.activity_started_s} /> : undefined}
					supplements={projection.supplements}
					projectionPid={projection.pid}
				/>
			)}

			{/* EVERYTHING BETWEEN THE TRANSCRIPT AND THE COMPOSER sits in ONE slot
			    with one gap and one inset (`components/above-composer.tsx`), never
			    its own padding — the stacked independent insets that left the old
			    ended banner flush on the composer card. */}
			<AboveComposerStack restOn={blockingKey}>
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
					key={blockingKey ?? undefined}
					pid={sessionId}
					pending={blocking}
					count={projection.pending_count}
				/>
			) : null}

			{/* One thumb-height tray replaces three stacked disclosures. Its panels
			    are the same content inside the shared in-flow Sheet, not new lists;
			    a long task/agent read cannot halve the transcript permanently. */}
			{/* NO HEALTH BANNER IN THIS SLOT ANY MORE (operator request): ended and
			    not-answering are the header's status dot, whose sheet carries the
			    sentences and the Resume action. The tray still renders on its own
			    merits in every health state — reading finished work is not a live
			    action, so the Tasks and Agents doors must not depend on liveness
			    (agent review round 1, C1-1 = QA Q1-2). */}
			{(needsApproval || (projection.asks?.length ?? 0) > 0 || projection.todos.some(p => p.items.length) || projection.subagents.length) ?
				/* THE TRAY NEVER SCROLLS SIDEWAYS (operator finding B). It was an
				   `overflow-x: auto` rail, so the last pill was reachable only by a
				   swipe most phone users never try. Now: PRIORITY ORDER (approve,
				   questions, a failing roster, tasks, then a healthy roster), with the
				   pills sized so the worst state still fits one 44 px line at 360 and
				   390. `flex-wrap` is the safety net for a longer translation: a second
				   line costs 44 px of transcript, but hides nothing and adds no tap;
				   `gap-x-1`/`gap-y-3` keep neighbouring hit bands from touching. */
				<div aria-label={messageRuntime.t("relay.chrome.session.tray.label")} className="flex min-h-11 shrink-0 flex-wrap items-center gap-x-1 gap-y-3 py-1.5 min-[390px]:gap-x-2">
					{needsApproval ? <Chip onClick={() => setGateOpen(true)} className="shrink-0 !bg-danger-wash !text-danger" aria-label={messageRuntime.t("relay.chrome.session.tray.approve_label")}>{messageRuntime.t("relay.chrome.session.tray.approve")}</Chip> : null}
					<AskDock rows={projection.asks} onOpen={openAsks} compact />
					{/* ONE CHIP, TWO PLACES (code review round 2, C2-5). It used to be two
					    full copies, each carrying the other's unreachable label ternary
					    and class branch; here the definition exists once and the state
					    decides which of the two SLOTS renders it. A failing roster sits
					    BEFORE tasks because the word that says why the roster matters
					    must not be the last thing on a line (design round 1, D2).
					    THE NOUN AND THE COUNT, NOT A LONE MARK (design D5, UX U3): a bare mark plus a numeral
					    read as a dismiss control beside two other counters and never said
					    what failed. The full existing `agents_failed` sentence — "Agents
					    1/3 · 1 failed" — carries the noun, the running count and the
					    failure, in the danger register the pill already used, with no
					    glyph that could be mistaken for ✕. No new copy. */}
					{agentsFailed > 0 ? agentsChip : null}
					{projection.todos.some(p => p.items.length) ? <Chip className={cn("shrink-0", blocking && "!text-ink-dim")} disabled={Boolean(blocking)} aria-expanded={tasksOpen} title={blocking ? messageRuntime.t("relay.asks.disclosure.held_shut") : undefined} onClick={() => setTasksOpen(true)}>{messageRuntime.t("relay.chrome.session.tray.tasks", {
						done: tasksDone,
						total: taskItems.length,
					})}</Chip> : null}
					{agentsFailed === 0 ? agentsChip : null}
				</div> : null}
			</AboveComposerStack>
			<Sheet open={tasksOpen} onClose={() => setTasksOpen(false)} title={messageRuntime.t("relay.chrome.session.tasks_title")}>
				<TodosPanel todos={projection.todos} sheet />
			</Sheet>
			<Sheet open={agentsOpen} onClose={() => setAgentsOpen(false)} title={messageRuntime.t("relay.chrome.session.agents_title")}>
				<SubagentsPanel pid={sessionId} subagents={projection.subagents} sheet />
			</Sheet>

			<Composer
				pid={sessionId}
				projection={projection}
				onOpenModels={() => {
					/* A FRESH OPEN IS A FRESH LIST: clear the query, as the sheet's own
					   comment requires (a short stale subset on reopen reads as a bug).
					   The effort view's back control does NOT clear it — that is the same
					   sheet's sub-view return, not a new opening (UX round 2, U5). */
					setModelQuery("");
					setModelsOpen(true);
				}}
				onReturnToModels={() => setModelsOpen(true)}
				onOpenEffort={() => setEffortOpen(true)}
				effortOpen={effortOpen}
				onCloseEffort={() => setEffortOpen(false)}
				autoFocus={autoFocusComposer}
				/* ONE "ended", the header dot's (review round 2, R2-2 = U7). */
				ended={ended}
				short={short}
				onFitLevel={setFitLevel}
			/>
			<Sheet open={spendOpen} onClose={() => setSpendOpen(false)} title={messageRuntime.t("relay.chrome.spend.title")}>
				<SessionStatus projection={projection} detail />
			</Sheet>

			<ModelSheet
				open={modelsOpen}
				onClose={() => setModelsOpen(false)}
				pid={sessionId}
				projection={projection}
				onOpenEffort={() => { setModelsOpen(false); setEffortOpen(true); }}
				query={modelQuery}
				onQuery={setModelQuery}
			/>

			<AsksSheet
				open={asksOpen}
				onClose={closeAsks}
				currentSessionId={sessionId}
			/>
			</>}
		</div>
	);
}
