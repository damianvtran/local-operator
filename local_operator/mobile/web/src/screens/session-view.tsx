/**
 * Session view (`#/s/:pid`) — the core screen. Layout contract:
 *
 *   header / transcript (flex-1, scrolls) / todos / subagents /
 *   pending card / banners / composer
 *
 * The whole column is `100dvh` capped, and a visualViewport listener keeps
 * the composer above the iOS keyboard: when the keyboard opens, the
 * viewport shrinks and the root element is re-pinned to its height. This
 * works around iOS Safari's habit of scrolling the page instead of
 * shrinking the layout when `interactive-widget` is not honoured.
 */
import { useEffect, useRef, useState } from "react";
import { ModelSheet } from "../components/model-sheet";
import { Composer } from "../components/composer";
import { PendingCard } from "../components/pending-card";
import { SubagentsPanel } from "../components/subagents-panel";
import { TodosPanel } from "../components/todos-panel";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { navigate } from "../router";
import { AgentScreen } from "./agent-view";
import { retainProjectionStream, useProjection } from "../store";
import type { SessionProjection } from "../types";

function Header({
	projection,
}: {
	projection: SessionProjection;
}) {
	return (
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
		</header>
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
		};
		sync();
		vv.addEventListener("resize", sync);
		vv.addEventListener("scroll", sync);
		return () => {
			vv.removeEventListener("resize", sync);
			vv.removeEventListener("scroll", sync);
			el.style.height = "";
			el.style.top = "";
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
			<Header projection={projection} />

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

			{projection.todos.some((p) => p.items.length > 0) ? (
				<TodosPanel todos={projection.todos} />
			) : null}
			{projection.subagents.length > 0 ? (
				<SubagentsPanel pid={sessionId} subagents={projection.subagents} />
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
