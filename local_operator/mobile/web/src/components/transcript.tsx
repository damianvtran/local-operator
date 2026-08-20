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
import { getHistory } from "../api";
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

export function Transcript({
	pid,
	entries,
}: {
	pid: number;
	entries: TranscriptEntry[];
}) {
	const [windowSize, setWindowSize] = useState(PAGE);
	/* Older entries the daemon served, PREPENDED above the live window. The
	   live projection is a tail the fold caps, so a long session's history
	   never arrives over SSE — it is paged in from the transcript on disk as
	   the user scrolls up. */
	const [older, setOlder] = useState<TranscriptEntry[]>([]);
	const [hasMore, setHasMore] = useState(true);
	const [loadingOlder, setLoadingOlder] = useState(false);
	const scrollRef = useRef<HTMLDivElement>(null);
	const pinnedRef = useRef(true);
	/* Auto-load trigger guard: one in-flight page at a time. */
	const loadingRef = useRef(false);
	/* Reset the back-filled history when the session changes. */
	useEffect(() => {
		setOlder([]);
		setHasMore(true);
		setWindowSize(PAGE);
	}, [pid]);

	/* The auto-scroll trigger. It must fire ONLY when the transcript actually
	   grew or the tail streamed more text — never on a same-content re-render.
	   The projection SSE sends a fresh `entries` array identity on every
	   repaint, so an effect that depends on the array runs constantly; an
	   expansion that re-rendered then snapped scrollTop to the very bottom
	   and the tapped row flew off-screen (read as "the screen went blank"). */
	const tail = entries[entries.length - 1];
	const growthSignal = `${entries.length}:${tail?.id ?? ""}:${tail?.text?.length ?? 0}:${tail?.final ?? ""}`;

	/* De-dupe: an older page can overlap the live window's head when the fold
	   re-caps between the fetch and the render. Key on id, older first. */
	const merged = (() => {
		const seen = new Set(older.map((e) => e.id));
		const live = entries.filter((e) => !seen.has(e.id));
		return [...older, ...live];
	})();

	const visible =
		merged.length > windowSize ? merged.slice(-windowSize) : merged;
	const hiddenCount = merged.length - visible.length;
	const oldestId = visible.length > 0 ? visible[0].id : null;

	/* Follow the tail on new content, but only when already at the bottom. */
	useEffect(() => {
		const el = scrollRef.current;
		if (el && pinnedRef.current) {
			el.scrollTop = el.scrollHeight;
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [growthSignal]);

	/* Prepending older rows must NOT move the viewport: capture the scroll
	   offset relative to the top of the OLD content, then restore it after the
	   prepend so the row the user was reading stays put. */
	const prependPage = (page: TranscriptEntry[]) => {
		const el = scrollRef.current;
		const prevHeight = el?.scrollHeight ?? 0;
		const prevTop = el?.scrollTop ?? 0;
		setOlder((cur) => [...page, ...cur]);
		/* Restore after React commits the taller content. */
		requestAnimationFrame(() => {
			const el2 = scrollRef.current;
			if (el2) {
				el2.scrollTop = prevTop + (el2.scrollHeight - prevHeight);
			}
		});
	};

	const loadOlder = async () => {
		if (loadingRef.current || !hasMore || !oldestId) return;
		loadingRef.current = true;
		setLoadingOlder(true);
		try {
			const { entries: page, has_more } = await getHistory(pid, oldestId, PAGE);
			if (page.length > 0) prependPage(page);
			setHasMore(has_more);
		} catch {
			/* A failed page is not fatal — leave hasMore so a retry can load it. */
		} finally {
			loadingRef.current = false;
			setLoadingOlder(false);
		}
	};

	const onScroll = () => {
		const el = scrollRef.current;
		if (!el) return;
		pinnedRef.current =
			el.scrollHeight - el.scrollTop - el.clientHeight < 48;
		/* Near the top with more history to fetch: auto-load so scrolling up
		   just keeps going, no button needed. */
		if (el.scrollTop < 120 && hasMore && !loadingRef.current) {
			void loadOlder();
		}
	};

	return (
		<div
			ref={scrollRef}
			onScroll={onScroll}
			className={cn(
				"lo-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-3 py-2",
			)}
		>
			{hasMore || loadingOlder ? (
				<button
					type="button"
					onClick={() => void loadOlder()}
					disabled={loadingOlder}
					className="mx-auto rounded-sm border border-control bg-surface px-3 py-1.5 text-meta text-ink-muted active:bg-elevated disabled:opacity-60"
				>
					{loadingOlder ? "loading…" : "load earlier"}
				</button>
			) : null}
			{hiddenCount > 0 ? (
				<button
					type="button"
					onClick={() => setWindowSize((n) => n + PAGE)}
					className="mx-auto rounded-sm border border-control bg-surface px-3 py-1.5 text-meta text-ink-muted active:bg-elevated"
				>
					show more ({hiddenCount} loaded)
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
