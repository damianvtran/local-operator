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
import { cn } from "../lib/cn";
import type { TranscriptEntry } from "../types";

const PAGE = 120;

function Entry({ entry }: { entry: TranscriptEntry }) {
	switch (entry.kind) {
		case "user":
			return (
				<div className="flex justify-end">
					<div className="max-w-[85%] rounded-md border border-hairline bg-surface px-3 py-1.5 text-body leading-normal whitespace-pre-wrap">
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
			return (
				<div className="text-body leading-normal">
					<Markdown text={entry.text} />
					{!entry.final ? (
						<span className="lo-caret" aria-hidden />
					) : null}
				</div>
			);
		case "tool":
			return <ToolRow entry={entry} />;
		case "notice":
		case "compaction":
			return (
				<p className="text-center text-meta text-ink-dim">
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

	const visible =
		entries.length > windowSize ? entries.slice(-windowSize) : entries;
	const hiddenCount = entries.length - visible.length;

	/* Follow the tail on new content, but only when already at the bottom. */
	useEffect(() => {
		const el = scrollRef.current;
		if (el && pinnedRef.current) {
			el.scrollTop = el.scrollHeight;
		}
	}, [entries.length, entries[entries.length - 1]?.text]);

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
				<Entry key={e.id} entry={e} />
			))}
		</div>
	);
}
