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
import { cn } from "../lib/cn";
import { navigate } from "../router";
import { retainProjectionStream, useProjection } from "../store";
import type { SessionProjection } from "../types";

function Header({
	projection,
}: {
	projection: SessionProjection;
}) {
	const status = projection.ended
		? "bg-ink-disabled"
		: projection.degraded
			? "bg-warning"
			: projection.streaming
				? "lo-pulse bg-accent"
				: "bg-success";
	return (
		<header className="flex items-center gap-2 border-b border-hairline px-1 py-1 pt-[max(env(safe-area-inset-top),0.25rem)]">
			<button
				type="button"
				onClick={() => navigate("/")}
				aria-label="back to sessions"
				className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
			>
				‹
			</button>
			<span
				className={cn("size-2 shrink-0 rounded-full", status)}
				aria-hidden
			/>
			<span className="min-w-0 flex-1 truncate text-body-sm font-medium">
				{projection.conversation_name || "untitled"}
			</span>
		</header>
	);
}

export function SessionScreen({ pid }: { pid: number }) {
	const { projection, connected } = useProjection(pid);
	const [modelsOpen, setModelsOpen] = useState(false);
	const [effortOpen, setEffortOpen] = useState(false);
	const rootRef = useRef<HTMLDivElement>(null);

	useEffect(() => retainProjectionStream(pid), [pid]);

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
						className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
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
			<Header projection={projection} />

			<Transcript entries={projection.transcript} />

			{projection.todos.length > 0 ? (
				<TodosPanel todos={projection.todos} />
			) : null}
			{projection.subagents.length > 0 ? (
				<SubagentsPanel subagents={projection.subagents} />
			) : null}

			{projection.pending ? (
				<PendingCard pid={pid} pending={projection.pending} />
			) : null}

			{projection.degraded && !projection.ended ? (
				<p className="mx-3 rounded-sm bg-warning-wash px-3 py-2 text-center text-body-sm text-warning">
					reconnecting to this session…
				</p>
			) : null}
			{projection.ended ? (
				<p className="mx-3 rounded-sm bg-sunken px-3 py-2 text-center text-body-sm text-ink-dim">
					this session has ended
				</p>
			) : null}

			<Composer
				pid={pid}
				projection={projection}
				onOpenModels={() => setModelsOpen(true)}
				onOpenEffort={() => setEffortOpen(true)}
				effortOpen={effortOpen}
				onCloseEffort={() => setEffortOpen(false)}
			/>

			<ModelSheet
				open={modelsOpen}
				onClose={() => setModelsOpen(false)}
				pid={pid}
				projection={projection}
			/>
		</div>
	);
}
