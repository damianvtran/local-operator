/**
 * The transcript: user bubbles, assistant prose (markdown), one-line tool
 * rows, quiet notices and compaction markers. Rendered as a tail window of
 * the projection's array — ~120 entries — with a "load earlier" affordance
 * at the top that pages the SAME array client-side; the daemon's snapshot
 * already caps the history it sends.
 *
 * Auto-scroll: the view follows the tail only while the user is already at
 * the bottom. Scrolling up to read must never be yanked back by a repaint.
 */
import { useEffect, useRef, useState } from "react";
import { Markdown } from "./markdown"
import { ToolRow } from "./tool-row"
import { RowBoundary } from "./row-boundary";
import { cn } from "../lib/cn";
import type { TranscriptEntry } from "../types";

const PAGE = 120;

function Entry({ entry }: { entry: TranscriptEntry }) {
	switch (entry.kind) {
		case "user":
			/* The user's own words. Right-aligned like the desktop app's bubble,
			   but the marker of identity is the accent edge on the leading
			   side: a user turn is the one thing in the transcript the human
			   said, and the accent is reserved for exactly that kind of "this
			   is what the turn is on" signal (branding §7). Surface ground +
			   hairline keeps it quiet next to the answer that follows. */
			return (
				<div className="flex justify-end">
					<div className="max-w-[85%] rounded-md border border-hairline border-l-2 border-l-accent bg-surface px-3 py-1.5 text-body leading-normal whitespace-pre-wrap">
						{entry.text}
					</div>
				</div>
			);
		case "steer":
			return (
				<div className="flex justify-end">
					<div className="max-w-[85%] rounded-md border border-hairline px-3 py-1 text-body-sm text-ink-muted whitespace-pre-wrap">
						{entry.text}
					</div>
				</div>
			);
		case "assistant":
			/* No per-row caret: the aggregate WorkingLine at the foot of the
			   transcript is the turn's ONE in-progress indicator (branding §7 —
			   never two animations for the same thing). The streaming row just
			   grows; the working line says it's alive, what it's doing, and for
			   how long. */
			return (
				<div className="text-body leading-normal">
					<Markdown text={entry.text} />
				</div>
			);
		case "tool":
			return <ToolRow entry={entry} />;
		case "notice":
		case "compaction":
			return (
				<p className="text-meta text-ink-dim">
					{entry.text}
				</p>
			);
		default:
			return null;
	}
}

export function Transcript({ entries }: { entries: TranscriptEntry[] }) {
	const [windowSize, setWindowSize] = useState(PAGE);
	const scrollRef = useRef<HTMLDivElement>(null);
	const pinnedRef = useRef(true);
	/* The auto-scroll trigger. It must fire ONLY when the transcript actually
	   grew or the tail streamed more text — never on a same-content re-render.
	   The projection SSE sends a fresh `entries` array identity on every
	   repaint, so an effect that depends on the array runs constantly; an
	   expansion that re-rendered then snapped scrollTop to the very bottom
	   and the tapped row flew off-screen (read as "the screen went blank"). */
	const tail = entries[entries.length - 1];
	const growthSignal = `${entries.length}:${tail?.id ?? ""}:${tail?.text?.length ?? 0}:${tail?.final ?? ""}`;

	const visible =
		entries.length > windowSize ? entries.slice(-windowSize) : entries;
	const hiddenCount = entries.length - visible.length;

	/* Follow the tail on new content, but only when already at the bottom. */
	useEffect(() => {
		const el = scrollRef.current;
		if (el && pinnedRef.current) {
			el.scrollTop = el.scrollHeight;
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [growthSignal]);

	const onScroll = () => {
		const el = scrollRef.current;
		if (!el) return;
		pinnedRef.current =
			el.scrollHeight - el.scrollTop - el.clientHeight < 48;
	};

	return (
		<div
			ref={scrollRef}
			onScroll={onScroll}
			className={cn(
				"lo-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-3 py-2",
			)}
		>
			{hiddenCount > 0 ? (
				<button
					type="button"
					onClick={() => setWindowSize((n) => n + PAGE)}
					className="mx-auto rounded-sm border border-control bg-surface px-3 py-1.5 text-meta text-ink-muted active:bg-elevated"
				>
					load earlier ({hiddenCount} more)
				</button>
			) : null}
			{visible.map((e) => (
				/* A boundary per row: one malformed entry must not unmount the
				   whole app (the "tap → blank screen" failure). */
				<RowBoundary key={e.id}>
					<Entry entry={e} />
				</RowBoundary>
			))}
		</div>
	);
}
