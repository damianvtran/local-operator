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
import { useEffect, useRef, useState } from "react";
import { setSessionPin } from "../api";
import { ModelSheet } from "../components/model-sheet";
import { Composer } from "../components/composer";
import { GateSheet } from "../components/gate-sheet";
import { PendingCard } from "../components/pending-card";
import { SubagentsPanel } from "../components/subagents-panel";
import { TodosPanel } from "../components/todos-panel";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { cn } from "../lib/cn";
import { COLUMN_HEIGHT_VAR } from "../lib/column";
import { navigate } from "../router";
import { useCompletionView } from "../use-completion-view";
import { AgentScreen } from "./agent-view";
import {
	applySessionPin,
	retainProjectionStream,
	useProjection,
	useSessions,
} from "../store";
import type { SessionProjection } from "../types";

function Header({
	projection,
	sessionId,
}: {
	projection: SessionProjection;
	sessionId: string;
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
	/* THE PIN STATE COMES FROM THE LIST STORE, which is the same row the daemon
	   serves on the list frame — so this control and the list's ★ agree by
	   construction rather than by two reads of the pin file. `undefined` (an
	   older daemon, or a session the list has not carried yet) reads as unpinned,
	   which is the honest default: the button then offers to pin, and the next
	   list repaint corrects it if that was wrong. */
	const { sessions } = useSessions();
	const pinned = Boolean(
		sessions.find((row) => row.session_id === sessionId)?.pinned,
	);
	const [pinError, setPinError] = useState("");
	const togglePin = async () => {
		setPinError("");
		const next = !pinned;
		/* Optimistic, then confirmed — the list row moves at once and the daemon's
		   next repaint is the authority. */
		applySessionPin(sessionId, next);
		try {
			await setSessionPin(sessionId, next);
		} catch (e) {
			setPinError(String((e as Error).message ?? e));
		}
	};
	return (
		<>
		<header className="flex items-center gap-2 border-b border-hairline px-1 py-1 pt-[max(env(safe-area-inset-top),0.25rem)]">
			<button
				type="button"
				onClick={() => navigate("/")}
				aria-label="back to sessions"
				className="flex min-h-8 min-w-8 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
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
					"flex min-h-8 min-w-8 items-center justify-center rounded-sm active:bg-elevated",
					pinned ? "text-accent" : "text-ink-muted",
				)}
			>
				{pinned ? "★" : "☆"}
			</button>
			{/* THE GATE CONTROL (stage D). On the phone this is the LOOSEN surface:
			    `/approvals auto` is authority-increasing, so it asks the runtime for a
			    per-action challenge and signs it with this phone's non-extractable key.
			    Before the redesign a phone could not loosen in ANY session, and the
			    refusal told the reader to find "the window that started this session" —
			    a window a phone cannot become. The label says what the control is about
			    (approvals) rather than naming the command; the sheet names the command. */}
			<button
				type="button"
				onClick={() => setGateOpen(true)}
				aria-label="approvals in this session"
				className="flex min-h-8 items-center justify-center rounded-sm px-2 text-meta text-ink-muted active:bg-elevated"
			>
				{projection.pending ? "needs you" : "approvals"}
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
		{pinError ? (
			/* A failed pin POST is SAID, not swallowed: a button whose press changed
			   nothing must not read as success. The optimistic list write is corrected
			   by the daemon's next repaint, so this is the only place the failure is
			   visible at all. */
			<p role="alert" className="border-b border-hairline px-2 py-1 text-meta text-danger">
				Could not save the pin: {pinError}
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
	const [modelsOpen, setModelsOpen] = useState(false);
	const [effortOpen, setEffortOpen] = useState(false);
	const rootRef = useRef<HTMLDivElement>(null);

	useEffect(() => retainProjectionStream(sessionId), [sessionId]);

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
			   relative bound it had before. */
			el.style.setProperty(COLUMN_HEIGHT_VAR, `${vv.height}px`);
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
		};
	}, []);

	if (!projection) {
		return (
			<div
				ref={rootRef}
				className="relative mx-auto flex h-dvh w-full max-w-md flex-col overflow-hidden"
			>
				<header className="flex items-center gap-2 border-b border-hairline px-1 py-1 pt-[max(env(safe-area-inset-top),0.25rem)]">
					<button
						type="button"
						onClick={() => navigate("/")}
						aria-label="back to sessions"
						className="flex min-h-8 min-w-8 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
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

	return (
		<div
			ref={rootRef}
			className="relative mx-auto flex h-dvh w-full max-w-md flex-col overflow-hidden"
		>
			{jobId ? (
				<AgentScreen
					sessionId={sessionId}
					jobId={jobId}
					projection={projection}
					connected={connected}
				/>
			) : <>
			<Header projection={projection} sessionId={sessionId} />

			{projection.transcript.length === 0 && !projection.streaming ? (
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
				<Transcript pid={sessionId} entries={projection.transcript} />
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
					forceCollapsed={Boolean(projection.pending)}
				/>
			) : null}
			{projection.subagents.length > 0 ? (
				<SubagentsPanel
					pid={sessionId}
					subagents={projection.subagents}
					forceCollapsed={Boolean(projection.pending)}
				/>
			) : null}

			{projection.pending ? (
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
					key={`${projection.pending.request_id}:${projection.pending.kind}:${projection.pending.question_index}`}
					pid={sessionId}
					pending={projection.pending}
					count={projection.pending_count}
				/>
			) : null}

			<Composer
				pid={sessionId}
				projection={projection}
				onOpenModels={() => setModelsOpen(true)}
				onOpenEffort={() => setEffortOpen(true)}
				effortOpen={effortOpen}
				onCloseEffort={() => setEffortOpen(false)}
			/>

			<ModelSheet
				open={modelsOpen}
				onClose={() => setModelsOpen(false)}
				pid={sessionId}
				projection={projection}
			/>
			</>}
		</div>
	);
}
