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
import { useEffect, useRef, useState, type ReactNode } from "react";
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

/* Severity glyphs, matching the TUI's NOTICE_GLYPHS exactly (transcript.py).
   Severity was HUE-ONLY on this surface: measured against the theme tokens,
   `danger` and `ink-dim` sit at 1.30:1, and across all 31 themes those two are
   within 1.6:1 on 30 of them (exactly 1.00:1 on six). Desaturated — grayscale,
   a cheap screen, low vision, bright sun — "turn failed", "compaction failed"
   and a routine wake receipt collapsed into one indistinguishable grey, while
   the TUI stayed readable because it fronts every notice with a symbol.

   The glyph is the redundant channel that makes the distinction survive losing
   colour, and `tool-row.tsx` already leads with a state glyph (✓/✗), so
   without this the surface was speaking two visual languages for one idea. */
const NOTICE_GLYPHS = {
	error: "✗",
	warning: "!",
	info: "·",
} as const;

function NoticeRow({ entry }: { entry: TranscriptEntry }) {
	const severity = entry.details.severity;
	const isWake = entry.details.notice_kind === "wake";
	/* A wake is a DELIVERY RECEIPT, not a failure or a warning, so it takes a
	   neutral marker rather than a severity one — but it still gets one,
	   because a row with no glyph in a column of glyphed rows reads as a
	   rendering bug.

	   `○` is the TUI's own ASCII wake glyph (`glyphs.py`, "a clock face"),
	   deliberately not the ⏰ emoji: an emoji renders from the colour font, so
	   it keeps its hue under the grayscale test this glyph exists to pass and
	   carries far more visual weight than the ✗/!/· it sits beside — two
	   inks and two weights for one column. */
	const glyph = isWake ? "○" : severity ? NOTICE_GLYPHS[severity] : NOTICE_GLYPHS.info;
	return (
		/* Severity ink mirrors the TUI's NoticeBlock kind. A refusal, a failed
		   turn and a failed compaction are things the user has to know
		   happened; an unattended gate timeout is something they may have to
		   act on. Everything else keeps the quiet default — the loudest ink in
		   the palette is worth nothing once routine receipts are wearing it. */
		<p
			className={cn(
				"text-meta flex gap-1.5 break-words",
				severity === "error" && "text-danger",
				severity === "warning" && "text-warning",
				/* `info` and absent are ONE case, not two. Written as three
				   branches this fell through all of them for an explicit
				   `severity: "info"` and rendered with no ink class at all —
				   while NOTICE_GLYPHS above does handle `info`, so the table
				   and the renderer disagreed about the same tier. Latent (no
				   producer emits it today) and exactly the shape of the
				   typed-but-unread field this delta exists to remove. */
				(!severity || severity === "info") && "text-ink-dim",
			)}
		>
			{/* aria-hidden: the glyph is a redundant encoding of the severity
			    already carried by the text, so announcing "✗" adds noise for a
			    screen reader rather than information. `shrink-0` keeps the
			    marker on the first line when the text wraps.

			    `w-4 text-center font-mono` is the glyph-column recipe
			    `tool-row.tsx` already uses, and it is what makes the column a
			    column: bare `shrink-0` sizes each marker to its own advance
			    width in the PROPORTIONAL body font, so the text beside it
			    started anywhere across a 7.6px range (21.6→29.1px) — worst on
			    the highest-severity rows, which are the ones the eye should
			    catch fastest. A fixed monospace box makes every notice's text
			    start on one edge. */}
			<span aria-hidden="true" className="w-4 shrink-0 text-center font-mono">
				{glyph}
			</span>
			<span className="min-w-0">{entry.text}</span>
		</p>
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
		case "peer_message": {
			/* An inbound message from another local lop session (`lop send`).
			   It must read as cross-session (not the user's own turn and not a
			   hub parent): a ↔ glyph and a sender label name who reached in.
			   The sender fields are advisory, so the label degrades to a bare
			   "Peer session" when the sender omitted them. */
			const sender = entry.details?.sender ?? {};
			const parts: string[] = [];
			if (sender.conversation_name) parts.push(sender.conversation_name);
			if (sender.pid != null) parts.push(`pid ${sender.pid}`);
			if (sender.model_label) parts.push(sender.model_label);
			const label = parts.length > 0 ? parts.join(" · ") : "Peer session";
			return (
				<div className="min-w-0 rounded-md border border-hairline border-l-2 border-l-accent bg-surface px-3 py-1.5">
					<span className="block text-meta text-ink-dim">↔ {label}</span>
					<p className="text-body-sm text-ink whitespace-pre-wrap break-words">{entry.text}</p>
				</div>
			);
		}
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
			return <NoticeRow entry={entry} />;
		default:
			return null;
	}
}

export function Transcript({
	pid,
	entries,
	jobId,
	scrollKey = `${pid}:${jobId ?? "root"}`,
	tailContent,
	emptyContent,
}: {
	pid: string;
	entries: TranscriptEntry[];
	jobId?: string;
	scrollKey?: string;
	/** Lifecycle outcomes belong in the conversation's one discoverable scroll
	 * surface, not in a clipped nested footer beneath it. */
	tailContent?: ReactNode;
	emptyContent?: ReactNode;
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

	/* Follow the tail when the SCROLLER ITSELF is resized, not just when content
	   arrives. The effect above triggers on new content, and a container that
	   shrinks under stationary content is not new content — so expanding a panel
	   beside the transcript silently cost the live tail: the scroller went from
	   612px to 274px and the newest messages slid 156px below the fold with no
	   scroll of the user's own, visible rows going [0..5] → [0..3] (U6).
	   Collapsing the panel restored it, which made "close the thing you opened"
	   the only way back to the conversation.

	   A ResizeObserver here rather than a callback from each panel: the panels
	   are one cause among several (the pending card claiming its space, the
	   working line appearing, the keyboard re-pinning the column), and a
	   per-caller notification fixes whichever cause someone remembered to wire.
	   Gated on `pinnedRef` exactly like the content path, so a user reading
	   history is never yanked to the bottom by a resize either. */
	useEffect(() => {
		const el = scrollRef.current;
		if (!el || typeof ResizeObserver === "undefined") return;
		const ro = new ResizeObserver(() => {
			if (pinnedRef.current) el.scrollTop = el.scrollHeight;
		});
		ro.observe(el);
		return () => ro.disconnect();
	}, []);

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
					<div data-completion-anchor={e.id} data-completion-complete={e.final && e.text_complete === true}>
						<Entry entry={e} pid={pid} />
					</div>
				</RowBoundary>
			))}
			{visible.length === 0 ? emptyContent : null}
			{tailContent}
		</div>
	);
}
