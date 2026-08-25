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
import { getHistory, getSubagentHistory, imageUrl } from "../api";
import { cn } from "../lib/cn";
import type { TranscriptEntry } from "../types";

const PAGE = 120;

/**
 * One inline attachment thumbnail with designed loading and failure states.
 *
 * A phone loads images over a flaky link, so the two off-happy-path states are
 * common, not edge cases, and both are designed rather than left to the
 * browser: a bare <img> would paint the native broken-image glyph on a 404
 * (which reads as a bug) and would reflow the bubble taller the instant bytes
 * decode (motion branding §7 rules out). Both are avoided by reserving a
 * fixed box up front and swapping a muted placeholder in on error.
 */
function AttachmentImage({
	pid,
	entryId,
	index,
}: {
	pid: string;
	entryId: string;
	index: number;
}) {
	const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
	/* The reserved frame: a fixed height so the row never jumps when the
	   bytes arrive, capped width so a wide image cannot push the bubble past
	   the viewport. object-contain keeps aspect within the frame. */
	if (state === "error") {
		return (
			<div className="flex h-40 w-40 flex-col items-center justify-center gap-1 rounded-sm border border-hairline bg-sunken text-ink-dim">
				<span aria-hidden className="text-body">
					⊘
				</span>
				<span className="text-meta">image unavailable</span>
			</div>
		);
	}
	return (
		<span
			className={cn(
				"relative block h-40 overflow-hidden rounded-sm border border-hairline",
				state === "loading" && "w-40 bg-sunken",
			)}
		>
			{state === "loading" ? (
				<span
					aria-hidden
					className="absolute inset-0 flex items-center justify-center text-meta text-ink-dim"
				>
					loading…
				</span>
			) : null}
			<img
				src={imageUrl(pid, entryId, index)}
				alt="attachment"
				onLoad={() => setState("loaded")}
				onError={() => setState("error")}
				className={cn(
					"h-40 max-w-full rounded-sm object-contain",
					state === "loading" && "invisible",
				)}
			/>
		</span>
	);
}

function Entry({ entry, pid }: { entry: TranscriptEntry; pid: string }) {
	switch (entry.kind) {
		case "user": {
			/* The user's own words. Right-aligned like the desktop app's bubble,
			   but the marker of identity is the accent edge on the leading
			   side: a user turn is the one thing in the transcript the human
			   said, and the accent is reserved for exactly that kind of "this
			   is what the turn is on" signal (branding §7). Surface ground +
			   hairline keeps it quiet next to the answer that follows. */
			const images = entry.images ?? [];
			return (
				<div className="flex min-w-0 justify-end">
					<div className="flex max-w-[85%] flex-col gap-1.5 rounded-md border border-hairline border-l-2 border-l-accent bg-surface px-3 py-1.5">
						{/* Attachments render inline like the TUI's image block: the
						   picture the user sent is part of the turn, not a stripped
						   "[image attached]" note. AttachmentImage owns the loading
						   and failure states (reserved box, designed placeholder) so
						   a flaky-link 404 or a slow decode never shows a broken
						   glyph or reflows the bubble. */}
						{images.length > 0 ? (
							<div className="flex flex-wrap gap-1.5">
								{images.map((img) => (
									<AttachmentImage
										key={img.index}
										pid={pid}
										entryId={entry.id}
										index={img.index}
									/>
								))}
							</div>
						) : null}
						{entry.text ? (
							<div className="text-body leading-normal break-words whitespace-pre-wrap">
								{entry.text}
							</div>
						) : null}
					</div>
				</div>
			);
		}
		case "steer":
			return (
				<div className="flex min-w-0 justify-end">
					<div className="max-w-[85%] rounded-md border border-hairline px-3 py-1 text-body-sm text-ink-muted break-words whitespace-pre-wrap">
						{entry.text}
					</div>
				</div>
			);
		case "parent_message":
			return (
				<div className="flex min-w-0 justify-end">
					<div className="max-w-[85%] rounded-md border border-hairline border-l-2 border-l-accent bg-surface px-3 py-1.5">
						<span className="block text-meta text-ink-dim">Parent</span>
						<p className="text-body-sm text-ink whitespace-pre-wrap break-words">{entry.text}</p>
					</div>
				</div>
			);
		case "subagent_message":
			return (
				<div className="min-w-0 rounded-sm border-l-2 border-l-hairline pl-3">
					<span className="block text-meta text-ink-dim">Subagent</span>
					<p className="text-body-sm text-ink whitespace-pre-wrap break-words">{entry.text}</p>
				</div>
			);
		case "assistant":
			/* No per-row caret: the aggregate WorkingLine at the foot of the
			   transcript is the turn's ONE in-progress indicator (branding §7 —
			   never two animations for the same thing). The streaming row just
			   grows; the working line says it's alive, what it's doing, and for
			   how long. min-w-0 + break-words keep a long URL/path/code span
			   from pushing the row past the viewport (the horizontal-scroll
			   report). */
			return (
				<div className="min-w-0 text-body leading-normal break-words">
					<Markdown text={entry.text} />
				</div>
			);
		case "tool":
			return <ToolRow entry={entry} />;
		case "notice":
		case "compaction":
			return (
				<p className="text-meta text-ink-dim break-words">
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
	jobId,
	scrollKey = `${pid}:${jobId ?? "root"}`,
}: {
	pid: string;
	entries: TranscriptEntry[];
	jobId?: string;
	scrollKey?: string;
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
	/* Each hash route owns its history and scroll position. Browser Back/Forward
	   remounts a route, so preserving it outside React state is what returns the
	   reader to the row they left instead of the latest token. */
	useEffect(() => {
		setOlder([]);
		setHasMore(true);
		setWindowSize(PAGE);
		const saved = Number(sessionStorage.getItem(`lo-mobile-scroll:${scrollKey}`));
		requestAnimationFrame(() => {
			const el = scrollRef.current;
			if (el && Number.isFinite(saved) && saved > 0) el.scrollTop = saved;
		});
	}, [scrollKey]);

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
	/* Long projections pin the opening user row ahead of a disjoint tail. It is
	   already retained, but it is not the tail's chronological history cursor;
	   anchor at the next row so the API can return the missing middle. The
	   merge's id de-dup keeps the opener exactly once when that page reaches it. */
	const oldestId =
		older.length === 0 && entries.length === PAGE && entries[0]?.kind === "user"
			? entries[1]?.id ?? entries[0]?.id ?? null
			: visible.length > 0
				? visible[0].id
				: null;

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
			const { entries: page, has_more } = jobId
				? await getSubagentHistory(pid, jobId, oldestId, PAGE)
				: await getHistory(pid, oldestId, PAGE);
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
		sessionStorage.setItem(`lo-mobile-scroll:${scrollKey}`, String(el.scrollTop));
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
				/* overflow-x-hidden as the backstop: break-words on the rows should
				   wrap everything, but a table or pre that still overflows scrolls
				   INSIDE itself, never the whole chat sideways. */
				"lo-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overflow-x-hidden px-3 py-2",
			)}
		>
			{/* History loads automatically as the user scrolls up — no button. A
			   subtle top indicator is the only chrome: a thin accent bar that
			   fills while a page is in flight, plus a hairline when more history
			   exists. Nothing tappable, nothing blocky. */}
			{loadingOlder ? (
				<div className="flex justify-center py-1" aria-hidden>
					<span className="lo-loadbar h-0.5 w-16 overflow-hidden rounded-full bg-sunken">
						<span className="lo-loadbar-fill block h-full w-1/2 rounded-full bg-accent" />
					</span>
				</div>
			) : hasMore ? (
				<div className="flex justify-center py-1" aria-hidden>
					<span className="h-px w-10 bg-hairline" />
				</div>
			) : null}
			{hiddenCount > 0 ? (
				<button
					type="button"
					onClick={() => setWindowSize((n) => n + PAGE)}
					className="mx-auto text-meta text-ink-dim underline-offset-2 active:underline"
				>
					show {hiddenCount} more loaded
				</button>
			) : null}
			{visible.map((e) => (
				/* A boundary per row: one malformed entry must not unmount the
				   whole app (the "tap → blank screen" failure). */
				<RowBoundary key={e.id}>
					<Entry entry={e} pid={pid} />
				</RowBoundary>
			))}
		</div>
	);
}
