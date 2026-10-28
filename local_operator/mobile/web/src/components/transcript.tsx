import { useScrollCues } from "../lib/use-scroll-cues";
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
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Markdown } from "./markdown"
import { AskRow } from "./ask-row";
import { AttachmentImage } from "./attachment-image";
import { ImageGenCard } from "./image-gen-card";
import { SupplementBlock } from "./supplement-block";
import { ToolRow } from "./tool-row"
import { RowBoundary } from "./row-boundary";
import { Disclosure } from "./ui/disclosure";
import { followScrollTop } from "../lib/scroll-follow";
import { getHistory, getSubagentHistory } from "../api";
import { messageRuntime } from "../store";
import { cn } from "../lib/cn";
import { IMAGE_GEN_TOOLS } from "../lib/image-gen";
import {
	hubReceiptLabel,
	isHubReceipt,
	quietGroupClauseFacts,
	quietGroupLabel,
	type QuietGroup,
} from "../lib/quiet-groups";
import { contextualQuietGroupsOf } from "../lib/peer-cycles";
import { beatLive, type SupplementBeat } from "../lib/supplement";
import { pendingEchoCaption, type PendingEcho } from "../pending-echo";
import type { TranscriptEntry } from "../types";
import { rowKey } from "../lib/condense-records";
import { splitFirstLine } from "../lib/condense-rows";
import { composeCondensedEntries, type CondenseEntry } from "../lib/condense-view";
import { FoldBar, TurnBar } from "./condense-rows";

const PAGE = 120;

/* The daemon's ``PROJECTION_TRANSCRIPT_LIMIT`` (``local_operator/mobile/types.py``):
   the projection is the pinned opener plus the newest 79 rows, so a transcript
   this long is a CAPPED window whose head row is not the tail's chronological
   neighbour. Mirrored rather than fetched because the rule is about the SHAPE of
   what arrived, and ``agent-view.tsx`` mirrors the same number for the child
   sheet. It was compared against ``PAGE`` (120) before — a number the daemon
   never sends — so the pinned-opener branch never ran on a real projection and
   the first cursor was the opener itself: the page came back as the opener's own
   neighbour plus the journal BELOW the cut, and the turns in between were never
   requested (measured with the S6 first-paint fixture: 22 user turns, 272 rows;
   on S3, 28 turns and 839 rows). ``>=`` rather than ``===`` is safe because the
   head row is already held either way — paging from the row after it cannot lose
   it, only re-offer it to the merge's id de-dup. */
const PROJECTION_ROWS = 80;

/** Whether a rendered row's id can name a journal entry — the only kind a
 *  history cursor may use. Reasoning rows live only in the projection (the
 *  fold's own comment: reasoning "is never persisted anywhere"), and the live
 *  fold's tool rows carry `tc-`, while a served page names a tool row
 *  `<message>:call_<call>`. Anything else is a cursor the daemon can only
 *  answer as end-of-history. */
function journalAddressableRow(entry: TranscriptEntry | undefined): boolean {
	return entry !== undefined && entry.kind !== "reasoning" && !entry.id.startsWith("tc-");
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

function Entry({ entry, pid, condensed = false }: { entry: TranscriptEntry; pid: string; condensed?: boolean }) {
	switch (entry.kind) {
		case "user": {
			/* The user's own words. Right-aligned like the desktop app's bubble,
			   but the marker of identity is the accent edge on the leading
			   side: a user turn is the one thing in the transcript the human
			   said, and the accent is reserved for exactly that kind of "this
			   is what the turn is on" signal (branding §7). Surface ground +
			   hairline keeps it quiet next to the answer that follows. */
			const images = entry.images ?? [];
			/* A condensed turn keeps the user's opening line beside its outcome.
			   Opening the turn removes `condensed`, restoring the identical full
			   bubble and attachments rather than a separately reconstructed row. */
			const firstLine = condensed ? splitFirstLine(entry.text).headline : entry.text;
			return (
				<div className="flex min-w-0 justify-end">
					<div className="flex max-w-[85%] flex-col gap-1.5 rounded-md bg-surface px-3 py-1.5">
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
						{firstLine ? (
							<div className={cn("text-body leading-normal break-words", condensed ? "truncate" : "whitespace-pre-wrap")}>
								{firstLine}
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
		case "subagent_message": {
			// Media belongs to the entry, not its fold eligibility or group anchor.
			const media = (entry.images ?? []).map(img => <AttachmentImage key={img.index} pid={pid} entryId={entry.id} index={img.index} />);
			// The child launch/user task also uses this kind during loading/error.
			// Without producer facts it is a visible request boundary, not hub traffic.
			if (!isHubReceipt(entry)) return entry.kind === "parent_message" ? (
				<div className="flex min-w-0 justify-end"><div className="max-w-[85%] rounded-md bg-surface px-3 py-1.5"><span className="block text-meta text-ink-dim">Parent</span><p className="text-body-sm text-ink whitespace-pre-wrap break-words">{entry.text}</p>{media}</div></div>
			) : (
				<div className="min-w-0 rounded-sm border-s-2 border-s-hairline ps-3"><span className="block text-meta text-ink-dim">Subagent</span><p className="text-body-sm text-ink whitespace-pre-wrap break-words">{entry.text}</p>{media}</div>
			);
			// A receipt is internal traffic, not a end-aligned human turn. Reuse
			// the existing disclosure so the full body remains reachable on demand.
			const label = hubReceiptLabel(entry);
			const intent = [entry.details?.expects_reply === true ? "Reply requested" : "", entry.details?.steer === true ? "Steered" : ""].filter(Boolean);
			return <Disclosure header={<span className="text-meta text-ink-dim">Hub · From {label}{intent.length ? ` · ${intent.join(" · ")}` : ""}</span>}>
				<div className="flex flex-col gap-2 ps-3">
					<p className="text-meta text-ink-muted">{label}{entry.details?.direction ? ` · ${entry.details.direction}` : ""}{entry.details?.job_id && entry.details?.label ? ` · Job ${entry.details.job_id}` : ""}</p>
					<p className="text-body-sm text-ink whitespace-pre-wrap break-words">{typeof entry.details?.body === "string" ? entry.details.body : entry.text}</p>
					{media}
				</div>
			</Disclosure>;
		}
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
				<div className="min-w-0 rounded-md bg-surface px-3 py-1.5">
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
			/* Image-generation calls get the progress card; every other tool
			   keeps the plain one-line row. The detection constant
			   (`lib/image-gen.ts`) is the ONE place the tool name lives —
			   normalised at the lookup the way `tool-row.tsx` normalises its
			   own set — so this branch and the card's tests cannot disagree
			   about what an image-gen tool is. */
			return IMAGE_GEN_TOOLS.has(entry.tool_name.toLowerCase()) ? (
				<ImageGenCard entry={entry} pid={pid} />
			) : (
				<ToolRow entry={entry} />
			);
		case "notice":
		case "compaction":
			return <NoticeRow entry={entry} />;
		/* THE QUEUED ASK SETTLING (design §4/§5). Both kinds carry structured
		   details — the Q&A for a response, the notice's own text for a deadline —
		   so they render through their own card rather than the notice row, which
		   has nowhere to put a disclosure. */
		case "ask_response":
		case "ask_timeout":
			return <AskRow entry={entry} />;
		default:
			return null;
	}
}

/** The row for one command that is in flight — what the user just sent.
 *
 * It exists because the receipt is not instant: before it, the user's words were
 * only ever in the composer's textarea, so a send on a phone link looked like
 * nothing had happened until the daemon answered and the message appeared out of
 * nowhere. This row is painted at submit and replaced by the real one under the
 * same id (`pending-echo.ts`).
 *
 * It takes the box of the row its OP settles into — the user bubble for a
 * prompt, the quieter steer row for a steer — so that reconciliation is the
 * caption leaving rather than a row changing register. Before this, a steer's
 * pending row was the prompt bubble (surface ground, 14px, 77.4px tall) settling
 * into the steer row (no ground, 13px, 49.0px), which the design round measured
 * and refused (D2). The two deliberate departures from a settled row are the
 * same in both variants and both are the pending signal itself: no accent
 * leading edge for a prompt (the accent means "this is what the turn is on", and
 * an unadmitted message is not that yet) and dimmer ink.
 *
 * It carries NO spinner and no shimmer. `WorkingLine` is this surface's ONE
 * in-progress animation (D25), and a second one per row is exactly the
 * competing-patterns defect that rule exists for — so the pending state is said
 * in words and ink, not in motion. */
function CondensedRows({
	items, pid, supplements, supplementLive, openRuns, onToggle,
}: {
	items: readonly CondenseEntry[];
	pid: string;
	supplements: Record<string, SupplementBeat>;
	supplementLive: (beat: SupplementBeat) => boolean;
	openRuns: ReadonlySet<string>;
	onToggle: (entry: Extract<CondenseEntry, { kind: "turn" | "fold" }>, open: boolean) => void;
}) {
	return items.map((item) => {
		if (item.kind === "row") {
			const beat = supplements[item.entry.id];
			const showBeat = beat && item.entry.kind === "assistant" && !item.condensed;
			return <RowBoundary key={item.key}>
				<div data-completion-anchor={item.entry.id} data-completion-complete={item.entry.final && item.entry.text_complete === true ? "true" : "false"}>
					<Entry entry={item.entry} pid={pid} condensed={item.condensed} />
				</div>
				{showBeat ? <SupplementBlock beat={beat} sessionId={pid} jobLive={supplementLive(beat)} /> : null}
			</RowBoundary>;
		}
		const children = <CondensedRows items={item.children} pid={pid} supplements={supplements}
			supplementLive={supplementLive} openRuns={openRuns} onToggle={onToggle} />;
		return item.kind === "turn" ? (
			<TurnBar key={item.key} entry={item} open={openRuns.has(item.segment.key)}
				onToggle={() => onToggle(item, !openRuns.has(item.segment.key))}>{children}</TurnBar>
		) : (
			<FoldBar key={item.key} entry={item} open={openRuns.has(item.key)}
				onToggle={() => onToggle(item, !openRuns.has(item.key))}>{children}</FoldBar>
		);
	});
}

function PendingEchoRow({ echo, streaming }: { echo: PendingEcho; streaming: boolean }) {
	const count = echo.imageCount;
	const attachments = count > 0 ? (
		<span className="text-meta text-ink-dim">
			{count === 1 ? "1 image attached" : `${count} images attached`}
		</span>
	) : null;
	const caption = pendingEchoCaption(echo, streaming);
	return (
		/* `data-pending-echo` is the row's identity for anything that needs to ask
		   "is this command's row still pending" from outside React — the same use
		   `data-completion-anchor` has below, and the only stable hook here that is
		   not a presentation class. */
		<div className="flex min-w-0 justify-end" data-pending-echo={echo.commandId}>
			{echo.op === "steer" ? (
				/* The steer row's own box: `transcript.tsx`'s `case "steer"`. */
				<div className="max-w-[85%] rounded-md border border-hairline px-3 py-1 text-body-sm text-ink-muted">
					{/* Attachments ride as a count, like the TUI's own prompt receipt. The
					   BYTES are never duplicated here: the composer keeps its previews and
					   revokes them on acknowledgement, so a second holder of those object
					   URLs would be a revoke this row cannot see coming. */}
					{attachments}
					{echo.text ? (
						<div className="break-words whitespace-pre-wrap">{echo.text}</div>
					) : null}
					<span className="text-meta text-ink-dim">{caption}</span>
				</div>
			) : (
				/* The user bubble's own box: `transcript.tsx`'s `case "user"`, minus the
				   accent leading edge and with `text-ink-muted` in place of the settled
				   row's full ink. Its `gap-1.5` is the settled bubble's gap, matched
				   rather than harmonised to the 4px ramp, so the two agree (design D5). */
				<div className="flex max-w-[85%] flex-col gap-1.5 rounded-md bg-surface px-3 py-1.5 text-body">
					{attachments}
					{echo.text ? (
						<div className="leading-normal break-words whitespace-pre-wrap text-ink-muted">
							{echo.text}
						</div>
					) : null}
					<span className="text-meta text-ink-dim">{caption}</span>
				</div>
			)}
		</div>
	);
}

/** The bar of ONE quiet group and, when opened, the receipts it holds
 * (quiet-turn design §5, slice S4).
 *
 * A group is a client-derived fold of >= 2 delivery receipts — peer messages
 * and wake receipts on this wire (`lib/quiet-groups`, where the definition
 * lives). The bar states the family's word and the count; the press reveals
 * the receipts with their tool rows, exactly as they paint at top level, and
 * the expansion survives the open tail's appends because the group's key
 * (`qg:<first row id>`) never moves. `partial` is a group the window's top
 * edge cuts: its count is a minimum, so the line states `N+` and the "at
 * least" claim rides the title — the desktop's own reading of the same fact.
 *
 * The bar deliberately carries NO `data-completion-anchor`: it is not a
 * text-bearing row, so it stays out of the scroll-follow ledger. A reader
 * whose nearest row folded into a collapsed bar falls to the nearest
 * surviving row — the fallback the follow effect documents for every
 * vanished row.
 */
function QuietGroupRow({
	group,
	partial,
	children,
}: {
	group: QuietGroup;
	partial: boolean;
	children: ReactNode;
}) {
	const countTitle = partial
		? `At least ${group.count} messages — earlier rows are not loaded`
		: undefined;
	return (
		<Disclosure
			header={
				<span className="flex min-w-0 items-center gap-1 text-meta text-ink-dim">
					<span className="shrink-0">{quietGroupLabel(group.family)}</span>
					<span aria-hidden className="shrink-0">·</span>
					<span className="shrink-0" title={countTitle} aria-label={countTitle}>
						{partial ? `${group.count}+` : `${group.count}`}
					</span>
					{/* The clause set (presence AND order) comes from the SHARED
					    derivation, never from inline conditions here: the desktop and
					    the phone disagreed for exactly as long as each decided its own
					    gate, so the one place that decides is `quietGroupClauseFacts`
					    - the same function the parity fixture pins. `count:` is the
					    line's leading number and is not rendered as a clause, so it is
					    filtered out rather than duplicated. */}
					{quietGroupClauseFacts(group, { partial })
						.filter((fact) => !fact.startsWith("count:"))
						.map((fact) => (
							<span key={fact}>· {clauseText(fact)}</span>
						))}
				</span>
			}
		>
			{children}
		</Disclosure>
	);
}

/** A clause FACT rendered as this surface's words.
 *
 * The facts (the shared derivation) carry the identifier and its count; the
 * wording lives with the surface that says it, exactly as the family word does
 * (`quietGroupLabel`). The plural is handled HERE because the actions clause is
 * reachable at 1: the TUI says `1 action` / `2 actions`, and a bar that printed
 * `1 actions` would be the kind of copy defect the design round exists to
 * catch. The catalogue has no `tui.transcript.group_actions` string on this
 * wire, so the words are the module's own, matching `quietGroupLabel`'s
 * precedent rather than inventing a lookup path this surface does not have.
 */
function clauseText(fact: string): string {
	const [name, rawCount] = fact.split(":");
	const count = Number(rawCount);
	switch (name) {
		case "actions":
			return `${count} action${count === 1 ? "" : "s"}`;
		case "reply_requested":
			return `${count} reply requested`;
		case "steered":
			return `${count} steered`;
		default:
			return fact;
	}
}

export function Transcript({
	pid,
	entries,
	jobId,
	scrollKey = `${pid}:${jobId ?? "root"}`,
	pending = [],
	streaming = false,
	topInset = 0,
	supplements,
	projectionPid = 0,
	peerCycles = false,
	tailContent,
	emptyContent,
}: {
	pid: string;
	entries: TranscriptEntry[];
	jobId?: string;
	scrollKey?: string;
	/** Turn supplements ("Highlights", memo §3.1 row 18): the projection's
	 *  sibling `{anchor: beat}` map. The block renders under the answer row
	 *  whose `id` is the beat's anchor — outside the completion anchor div (the
	 *  block is the answer's supplement, not part of the message the completion
	 *  receipt samples), inside the same row boundary so one malformed beat
	 *  cannot blank the screen. Absent map = no blocks, which is what every
	 *  session without beats and every older daemon mean. */
	supplements?: Record<string, SupplementBeat>;
	/** The projection's own process identity, for the reader rule's `job_live`
	 *  (`lib/supplement.ts`, `beatLive`): 0 is a disk rebuild. */
	projectionPid?: number;
	/** Explicit owner-channel negotiation; never inferred from an entry. */
	peerCycles?: boolean;
	/** Commands this device sent that the session has not written a row for yet.
	 *  Rendered as the LAST rows of the conversation, which is where the real rows
	 *  will land, and boxed like the row each op settles into (`PendingEchoRow`)
	 *  — so reconciliation keeps the row where it is and drops its caption, rather
	 *  than moving it or changing its register (design round 1, D2). */
	pending?: PendingEcho[];
	/** Whether a turn is live, which the pending rows' caption's tense needs —
	 *  a steer queued when its turn has already ended is waiting for the NEXT
	 *  message rather than for the step running now. Defaults to false, which is
	 *  what the subagent transcript (no pending rows) means by it. */
	streaming?: boolean;
	/** The height of anything OVERLAYING this scroller's top edge (the session
	 *  view's state ladder), reserved INSIDE the scroller so the first row can
	 *  never sit under a strip (round 2, U23 = D7). The strip is an overlay so
	 *  the column never jumps (round 1, U19/D4); the price was that the top of
	 *  the history was painted over, and on a transcript that does not scroll
	 *  there was no gesture that could reveal it — measured: 28 of the first
	 *  row's 35px hidden at 390, 38 of 56px at 320, in the exact shape that
	 *  says "its history is kept". The owner measures the strip and hands its
	 *  height here; zero (the default) means nothing overlays this scroller and
	 *  the spacer is not rendered at all. */
	topInset?: number;
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
	const [heldLive, setHeldLive] = useState<TranscriptEntry[]>(entries);
	const [hasMore, setHasMore] = useState(true);
	const [loadingOlder, setLoadingOlder] = useState(false);
	const [olderError, setOlderError] = useState(false);
	const [openRuns, setOpenRuns] = useState<ReadonlySet<string>>(new Set());
	const heldRowsRef = useRef<import("../lib/condense-rows").Row[]>([]);
	const foldLatchRef = useRef<Set<string>>(new Set());
	const condensedRunsRef = useRef<Set<string>>(new Set());
	const watchedRunRef = useRef<string | null>(null);
	const [watchedRunId, setWatchedRunId] = useState<string | null>(null);
	const latchSeenRef = useRef<{ seen: boolean; tailId: string | null }>({ seen: false, tailId: null });
	const headCompletionRef = useRef(0);
	const userPagedRef = useRef(false);
	const pressedHeaderRef = useRef<{ id: string; top: number } | null>(null);
	const scrollRef = useRef<HTMLDivElement>(null);
	const scrollCue = useScrollCues();
	const pinnedRef = useRef(true);
	/* Auto-load trigger guard: one in-flight page at a time. */
	const loadingRef = useRef(false);
	/* Each hash route owns its history and scroll position. Browser Back/Forward
	   remounts a route, so preserving it outside React state is what returns the
	   reader to the row they left instead of the latest token. */
	useEffect(() => {
		setOlder([]);
		setHeldLive(entries);
		setOpenRuns(new Set());
		foldLatchRef.current.clear();
		condensedRunsRef.current.clear();
		watchedRunRef.current = null;
		setWatchedRunId(null);
		latchSeenRef.current = { seen: false, tailId: null };
		headCompletionRef.current = 0;
		userPagedRef.current = false;
		heldRowsRef.current = [];
		setOlderError(false);
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
	/* The pending row is part of the tail for the SAME reason a new entry is: a
	   row that appeared below the fold with no scroll is a send the user watches
	   happen off-screen. Keyed by id, so resolution (which swaps the pending row
	   for the real one, one entry longer) is a change here too. */
	const pendingKey = pending.map((echo) => echo.commandId).join(",");
	const growthSignal = `${entries.length}:${tail?.id ?? ""}:${tail?.text?.length ?? 0}:${tail?.final ?? ""}:${pendingKey}`;

	/* One ordered held list, upserted by the live/replay-stable row key. Rows
	   falling out of the daemon's sliding projection stay held: deleting them
	   would create a hole and make a turn summary claim a shorter run. */
	const heldMap = new Map<string, TranscriptEntry>();
	for (const entry of [...older, ...heldLive]) heldMap.set(rowKey(entry), entry);
	const cappedProjection = entries.length >= PROJECTION_ROWS;
	const pinnedOpener = cappedProjection && entries.length > 1 && entries[0]?.kind === "user" ? entries[0] : null;
	const liveTail = pinnedOpener ? entries.slice(1) : entries;
	for (const entry of liveTail) heldMap.set(rowKey(entry), entry);
	const openerJoined = pinnedOpener !== null && older.some((entry) => rowKey(entry) === rowKey(pinnedOpener));
	const detachedOpener = pinnedOpener && !openerJoined && older.length > 0 ? pinnedOpener : null;
	if (detachedOpener) heldMap.delete(rowKey(detachedOpener));
	/* Only unsolicited live retention is capped. Explicitly fetched history
	   must survive the join: trimming its oldest page here both hides the
	   rows the reader requested and pins the next cursor to that same page. */
	const merged = [...heldMap.values()];
	/* Long projections pin the opening user row ahead of a disjoint tail. It is
	   already retained, but it is not the tail's chronological history cursor;
	   anchor at the next row so the API can return the missing middle. The
	   merge's id de-dup keeps the opener exactly once when that page reaches it.

	   ONCE A PAGE HAS BEEN FETCHED THE CURSOR IS THE OLDEST ROW HELD, not the
	   oldest row RENDERED. `visible[0]` is the oldest row inside the mounted
	   window, and that window only grows when the reader taps `show N more
	   loaded` — so while fetches outpace taps the cursor did not move and every
	   request returned the SAME page. Measured on the S6 fixture after deep
	   history became reachable: 40 taps produced 139 requests and 4,680 mounted
	   rows holding 240 distinct ones, twenty copies of one page, which the
	   reader then scrolls through as a conversation that repeats itself.
	   `older[0]` is the oldest row this view actually holds, so the next page is
	   the one immediately before it and paging is linear in the reader's taps. */
	/* THE FIRST CURSOR MUST NAME A JOURNAL ROW, and not every rendered row does.
	   The live fold mints ``rz-`` reasoning rows and ``tc-`` tool rows
	   (``local_operator/mobile/projection.py``); reasoning is never written to
	   the journal at all, and a served page names tool rows
	   ``<message>:call_<call>``. The daemon answers a cursor it cannot locate as
	   END OF HISTORY (``durable.py:_cursor_candidates`` / ``journal_rows_older_than``),
	   so a cursor taken from one of these rows stops back-filling for good: the
	   reader sees the pinned opening message, a gap, and the newest rows.

	   A USER-HEADED PROJECTION IS THE PINNED-OPENER SHAPE, at any length. The
	   daemon pins the opening user message ahead of a disjoint tail, and the
	   FRAME CAP can push a pinned projection below ``PROJECTION_ROWS`` (16 rows
	   at its floor) — where the old length test fell through to
	   ``merged[0].id``, the opener itself, so every fetch asked for history
	   older than the first message of the conversation and got nothing. Skipping
	   the head costs at most one already-held row on a projection that merely
	   STARTS with a user turn, which the merge's rowKey de-dup discards. */
	const cursorStart = entries.length > 1 && entries[0]?.kind === "user" ? 1 : 0;
	const oldestId =
		older.length > 0
			? older[0].id
			: merged.slice(cursorStart).find(journalAddressableRow)?.id ?? merged[0]?.id ?? null;

	/* THE QUIET GROUPS over the rows on hand (design §5, slice S4): pure per
	   render — rows only append, so a closed group's facts cannot move and the
	   open tail grows in place under its stable key. The definition lives once
	   in `lib/quiet-groups` and is pinned across clients by the shared parity
	   fixture. */
	/* Quiet grouping speaks wire IDs; the condense join speaks canonical tool
	   identities (`call:<tool_call_id>`). Normalize ownership once so both the
	   fold exclusion and disclosure children refer to the exact same rows. */
	const canonicalIds = new Map(merged.map((entry) => [entry.id, rowKey(entry)]));
	const groups = contextualQuietGroupsOf(merged, pid, peerCycles).map((group) => ({
		...group,
		rowIds: group.rowIds.map((id) => canonicalIds.get(id) ?? id),
	}));
	const quietRowIds = new Set(groups.flatMap((group) => group.rowIds));
	const live = streaming || pending.length > 0;
	const condensed = useMemo(() => composeCondensedEntries(merged, {
		live, openRuns, olderRowsExist: hasMore || detachedOpener !== null,
		watchedRunId,
		previousRows: heldRowsRef.current,
		latchedFoldIds: live || watchedRunId ? foldLatchRef.current : new Set(),
		quietRowIds,
		condensedRunIds: condensedRunsRef.current,
	}), [merged, live, openRuns, hasMore, detachedOpener, watchedRunId]);
	if (!live) {
		for (const item of condensed.entries) if (item.kind === "turn") condensedRunsRef.current.add(item.run.key);
	}
	heldRowsRef.current = condensed.rows;
	if (live) foldLatchRef.current = condensed.foldIds;
	else foldLatchRef.current.clear();
	const visibleItems = condensed.entries.slice(-windowSize);
	const hiddenCount = Math.max(0, condensed.entries.length - visibleItems.length);
	const groupsByFirstId = new Map(groups.map((group) => [group.rowIds[0], group]));
	const headCut = hiddenCount > 0 || hasMore;
	const renderItems: Array<CondenseEntry | { kind: "quiet"; key: string; group: QuietGroup }> = [];
	const claimedQuiet = new Set<string>();
	for (const item of condensed.entries) {
		if (item.kind === "row") {
			const group = groupsByFirstId.get(rowKey(item.entry));
			if (group && !claimedQuiet.has(group.key)) {
				claimedQuiet.add(group.key);
				for (const id of group.rowIds) claimedQuiet.add(id);
				renderItems.push({ kind: "quiet", key: group.key, group });
				continue;
			}
			if (claimedQuiet.has(rowKey(item.entry))) continue;
		}
		renderItems.push(item);
	}
	const pagedItems = renderItems.slice(-windowSize);

	/* One row's box + anchor attrs, shared by the plain rows and a group's
	   receipts so the two can never disagree about the follow contract. A
	   boundary per row: one malformed entry must not unmount the whole app
	   (the "tap → blank screen" failure). The supplement seat rides HERE, at
	   the row renderer, so a grouped row and a plain row carry it by
	   construction — the fold (origin/main) and the quiet groups (S4) cannot
	   drift on where a beat paints. */
	const transcriptRow = (entry: TranscriptEntry) => (
		<RowBoundary key={entry.id}>
			<div data-completion-anchor={entry.id} data-completion-complete={entry.final && entry.text_complete === true}>
				<Entry entry={entry} pid={pid} />
			</div>
			{(() => {
				/* The supplement block's seat: under the FINAL answer, outside
				   the completion-anchor div, inside the row boundary. The anchor
				   is the display entry id, so only the answer the beat names
				   renders it; the beat's own `beatLive` decides the stale rule. */
				const beat =
					entry.kind === "assistant" ? supplements?.[entry.id] : undefined;
				if (!beat) return null;
				return (
					<SupplementBlock
						beat={beat}
						sessionId={pid}
						jobLive={beatLive(beat, projectionPid)}
					/>
				);
			})()}
		</RowBoundary>
	);

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

	/* Prepending older rows must NOT move the viewport: the row the user was
	   reading stays put. That is now simply one case of the follow effect above
	   — rows inserted above the reader move the anchor row down and the write
	   follows it — so the rAF restore that used to live here is gone (round 4):
	   it was a second, blind hand on `scrollTop` (a `scrollHeight` delta that
	   also counted any tail growth in the same commit, at a frame's delay),
	   exactly the shape the opt-out exists to prevent. */
	const prependPage = (page: TranscriptEntry[]) => {
		setOlder((cur) => {
			const held = new Set([...cur, ...heldLive, ...entries].map(rowKey));
			return [...page.filter((entry) => !held.has(rowKey(entry)) ||
			(pinnedOpener !== null && rowKey(entry) === rowKey(pinnedOpener))), ...cur];
		});
	};

	useEffect(() => {
		setHeldLive((current) => {
			const next = new Map(current.map((entry) => [rowKey(entry), entry]));
			for (const entry of entries) next.set(rowKey(entry), entry);
			return [...next.values()].slice(-2000);
		});
	}, [entries]);

	const toggleCondensed = (item: Extract<CondenseEntry, { kind: "turn" | "fold" }>, open: boolean) => {
		const id = item.kind === "turn" ? item.segment.key : item.key;
		const header = scrollRef.current?.querySelector(`[data-condense-header="${CSS.escape(item.key)}"]`);
		if (header && scrollRef.current) pressedHeaderRef.current = {
			id: item.key,
			top: header.getBoundingClientRect().top - scrollRef.current.getBoundingClientRect().top,
		};
		pinnedRef.current = false;
		setOpenRuns((current) => {
			const next = new Set(current);
			if (open) next.add(id); else next.delete(id);
			return next;
		});
	};

	const loadOlder = async () => {
		if (loadingRef.current || !hasMore || !oldestId) return;
		loadingRef.current = true;
		setLoadingOlder(true);
		setOlderError(false);
		try {
			const { entries: page, has_more } = jobId
				? await getSubagentHistory(pid, jobId, oldestId, PAGE)
				: await getHistory(pid, oldestId, PAGE);
			if (page.length > 0) {
				prependPage(page);
				/* A page that lands while the reader is parked at the top has to MOUNT
				   in the same commit, or they see the idle hairline and nothing else
				   until they gesture again — one extra flick per page, and the page is
				   already held (review round 2, R2-3). Growing by the page's own
				   length keeps one mechanism per gesture: this adds exactly what just
				   arrived, and `onScroll` still owns everything after that. */
				const el = scrollRef.current;
				if (el && el.scrollTop < 120) setWindowSize((n) => n + page.length);
			}
			setHasMore(has_more);
		} catch {
			/* Keep the cursor and hasMore intact: the explicit retry must request
			   the failed page, not skip it or silently strand the reader. */
			setOlderError(true);
		} finally {
			loadingRef.current = false;
			setLoadingOlder(false);
		}
	};

	useEffect(() => {
		const newest = condensed.runs.at(-1);
		if (!newest) return;
		const tailId = newest.recordIds.at(-1) ?? null;
		const seen = latchSeenRef.current;
		const grew = seen.seen && tailId !== null && tailId !== seen.tailId;
		latchSeenRef.current = { seen: true, tailId };
		if (live || (grew && newest.answerId === null)) {
			const id = condensed.rows[newest.run.openingIndex]?.record.id ?? null;
			watchedRunRef.current = id;
			setWatchedRunId((current) => current === id ? current : id);
		}
	}, [condensed, live]);

	const topRun = condensed.runs[0];
	const topHeadCut = topRun !== undefined && !topRun.run.opensWithUserRow;
	useEffect(() => {
		if (!topHeadCut || live || !topRun?.collapses || !hasMore || loadingOlder || userPagedRef.current) return;
		if (headCompletionRef.current >= 3) return;
		const timer = window.setTimeout(() => {
			if (userPagedRef.current || loadingRef.current || headCompletionRef.current >= 3) return;
			headCompletionRef.current += 1;
			void loadOlder();
		}, 0);
		return () => window.clearTimeout(timer);
	}, [topHeadCut, live, topRun?.collapses, hasMore, loadingOlder, oldestId]);

	const onScroll = () => {
		userPagedRef.current = true;
		const el = scrollRef.current;
		if (!el) return;
		pinnedRef.current =
			el.scrollHeight - el.scrollTop - el.clientHeight < 48;
		sessionStorage.setItem(`lo-mobile-scroll:${scrollKey}`, String(el.scrollTop));
		/* Near the top: reveal what is already held, then fetch what is not.
		   ONE mechanism at a time, in that order, and no button.

		   The order is the fix for two rounds of the same complaint. A page
		   arriving while rows sat hidden behind `show N more loaded` fetched
		   further pages nobody could see (13 pages, 2.24 MB), and the button's
		   own label described a count it did not deliver (`show 1,425 more
		   loaded` revealed 120, with the next tap about 18 screens above the
		   reader). Growing first means the reader's upward gesture always moves
		   the transcript, and a fetch only ever happens once there is nothing
		   left held above them. */
		if (el.scrollTop < 120 && !loadingRef.current) {
			if (hiddenCount > 0) setWindowSize((n) => n + PAGE);
			else if (hasMore) void loadOlder();
		}
	};

	/* THE READER'S ROW, HELD ACROSS EVERY CHANGE ABOVE THEM (round 3 fixed the
	   reserve's own edges; round 4 extends the same hand to every other row
	   change, because the opt-out means no other hand exists). The live window
	   drops its oldest row on each append at the cap (U28 — native anchoring
	   used to cover that removal silently), scrolling up expands the window
	   upward (the tap target that used to do it is gone — review round 1, D2/D3),
	   a page of older rows can prepend, and the reserve grows or clears. So rather than compensating one cause, this measures its EFFECT:
	   EVERY rendered row is remembered with its viewport offset, and the next
	   commit re-measures the row the reader was closest to — the first row at
	   the viewport's top edge — and follows how far it moved.

	   The row's own movement already has the reader's scrolling divided out: a
	   pure scroll moves the row and `scrollTop` by equal and opposite amounts
	   and the two terms cancel, so a streamed frame that lands after a drag
	   does not undo the drag. The write itself is `lib/scroll-follow`, where
	   the two positions that are NOT "hold this row" live: the tail (the
	   browser's clamp already followed a shrink — round 4, reviewer MAJOR 1 =
	   U29) and the top with the reserve's edge (the reveal — round 2, U23).

	   ALL rows are tracked, not just the visible ones, because the change being
	   measured can push the reader's row out of the viewport in the same commit
	   (a 30-row prepend) — a topmost-visible-only anchor would lose its row and
	   write nothing, which is the round-3 bug in a new costume (measured: a
	   34px rung insert changed the topmost visible row's identity and the
	   reader moved by the rung's height). When the row the reader was nearest
	   is gone (it was the one the window dropped), the next-nearest surviving
	   row answers instead; when nothing survives, nothing is written. */
	const anchorRows = useRef<{ id: string; top: number }[] | null>(null);
	const prevFollowScrollTop = useRef(0);
	const prevReserve = useRef(0);
	useLayoutEffect(() => {
		const el = scrollRef.current;
		if (!el) return;
		const box = el.getBoundingClientRect();
		const scrollTop = el.scrollTop;
		const max = el.scrollHeight - el.clientHeight;
		/* The reserve's size read off the DOM, not off `topInset`: the edge
		   belongs to the same two frames the delta is measured between, and a
		   prop can land in a commit the spacer has not moved in yet (measured:
		   a run flagged the edge, skipped the top guard, and a later frame then
		   carried the move as an un-edged delta through the guard). The spacer
		   is not the whole insertion — it is a flex child, so it also opens the
		   scroller's row gap before the next child — and both are read here. */
		let reserve = 0;
		const spacer = el.querySelector<HTMLElement>("[data-scroll-top-inset]");
		if (spacer) {
			const spacerBox = spacer.getBoundingClientRect();
			const next = spacer.nextElementSibling;
			reserve =
				spacerBox.height +
				(next ? Math.max(0, next.getBoundingClientRect().top - spacerBox.bottom) : 0);
		}
		const reserveEdge = reserve !== prevReserve.current;
		prevReserve.current = reserve;
		const rows: { id: string; top: number }[] = [];
		for (const candidate of el.querySelectorAll("[data-completion-anchor]")) {
			rows.push({
				id: candidate.getAttribute("data-completion-anchor") ?? "",
				top: candidate.getBoundingClientRect().top - box.top,
			});
		}
		const pressed = pressedHeaderRef.current;
		if (pressed) {
			pressedHeaderRef.current = null;
			const header = el.querySelector(`[data-condense-header="${CSS.escape(pressed.id)}"]`);
			if (header) {
				const delta = header.getBoundingClientRect().top - box.top - pressed.top;
				if (Math.abs(delta) > 0) el.scrollTop = Math.max(0, Math.min(max, scrollTop + delta));
				/* The next render's generic follow uses these rows. Store them in
				   the corrected frame, not the pre-click frame, or that render
				   undoes this scroll and sends the pressed bar off-screen. Read the
				   actual displacement so boundary clamping stays honest. */
				const shift = el.scrollTop - scrollTop;
				for (const row of rows) row.top -= shift;
			}
		}
		const prev = anchorRows.current;
		if (!pressed && prev) {
			const now = new Map(rows.map((r) => [r.id, r.top]));
			const nearEdge = [...prev].sort((a, b) => Math.abs(a.top) - Math.abs(b.top));
			for (const row of nearEdge) {
				const top = now.get(row.id);
				if (top === undefined) continue;
				const next = followScrollTop({
					domDelta: top - row.top + (scrollTop - prevFollowScrollTop.current),
					scrollTop,
					max,
					reserveEdge,
				});
				if (next !== null) {
					el.scrollTop = next;
					/* The recorded offsets describe the frame the writer just
					   left: a scroll write moves every row by the same amount. */
					const shift = next - scrollTop;
					for (const r of rows) r.top -= shift;
				}
				break;
			}
		}
		anchorRows.current = rows;
		prevFollowScrollTop.current = el.scrollTop;
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [topInset, older, windowSize, entries, pending, hasMore, loadingOlder, visibleItems, openRuns]);

	return (
		<div
			ref={node => { scrollRef.current = node; return scrollCue(node); }}
			onScroll={onScroll}
			className={cn(
				/* overflow-x-hidden as the backstop: break-words on the rows should
				   wrap everything, but a table or pre that still overflows scrolls
				   INSIDE itself, never the whole chat sideways. */
				"lo-scroll lo-fade-y flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overflow-x-hidden px-3 py-2",
				/* ONE HAND ON `scrollTop` (round 3; extended in round 4). The reserve,
				   the live window's oldest-row removal at the cap, a prepended page and
				   the window's own expansion are all "height above the reader" — and
				   both the platform and the follow effect above want to answer for each,
				   together moving a mid-history reader by a whole row per event. The
				   opt-out leaves the follow effect as the only hand: it holds the
				   reader's row structurally (see it above) instead of splitting the job
				   with native anchoring, which is Chrome-only — Safari has none, so the
				   manual path is the whole mechanism there. The session LIST keeps
				   anchoring ON (`screens/session-list.tsx`, QA round 4, Q6): it has no
				   manual path of its own and needs the platform for insertions from
				   other clients — the two answers differ on purpose. */
				"[overflow-anchor:none]",
			)}
		>
			{/* THE OVERLAY'S HEIGHT, RESERVED (round 2, U23 = D7). First child on
			    purpose: everything the reader can reach through this scroller —
			    the load indicator, the "show N more" control, the rows — has to
			    start below anything painted over the scroller's top edge. The
			    height is the owner's measurement of the rung (see `topInset`); it
			    is CONTENT and not padding, because `scrollHeight` has to carry it
			    or the reserved space itself could not be scrolled past. */}
			{topInset > 0 ? (
				<div
					aria-hidden
					data-scroll-top-inset=""
					className="shrink-0"
					style={{ height: `${topInset}px` }}
				/>
			) : null}
			{/* History loads automatically as the user scrolls up — no button. A
			   subtle top indicator is the only chrome: a thin accent bar that
			   fills while a page is in flight, plus a hairline when more history
			   exists. Nothing tappable, nothing blocky.

			   THE TWO STATES ARE THE SAME HEIGHT (2px), and that is load-bearing
			   rather than tidy. The bar used to be 2px and the hairline 1px, so
			   each swap moved the rows below it by a pixel — and the anchor-hold
			   wrote `scrollTop` to compensate, which fired `scroll`, which saw
			   `scrollTop < 120` and fetched again: measured on the S6 deep cell,
			   23 of 27 transitions in one open were that 1px oscillation against
			   a rig scrolling once every 300 ms, and the cascade pulled 13 pages
			   (2.24 MB) into a window the reader could not see. */}
			{detachedOpener ? <div data-detached-opener="true"><Entry entry={detachedOpener} pid={pid} /></div> : null}
			{olderError ? (
				<div className="flex min-h-11 items-center justify-center gap-1.5 text-meta text-ink-dim" role="status">
					<span className="min-w-0 truncate">{messageRuntime.t("relay.transcript.older.failed")}</span>
					<button type="button" className="min-h-11 shrink-0 px-1.5 text-ink-muted" onClick={() => void loadOlder()}>
						{messageRuntime.t("relay.account_policy.retry")}
					</button>
				</div>
			) : null}
			{loadingOlder ? (
				<div className="flex h-0.5 justify-center" aria-hidden>
					<span className="lo-loadbar h-full w-16 overflow-hidden rounded-full bg-sunken">
						<span className="lo-loadbar-fill block h-full w-1/2 rounded-full bg-accent" />
					</span>
				</div>
			) : hasMore ? (
				<div className="flex h-0.5 items-center justify-center" aria-hidden>
					<span className="h-px w-10 bg-hairline" />
				</div>
			) : null}
			{pagedItems.map((item) => {
				if (item.kind === "quiet") {
					return <RowBoundary key={item.key}>
						<QuietGroupRow group={item.group} partial={headCut && item.group.rowIds[0] === (merged[0] ? rowKey(merged[0]) : undefined)}>
							{item.group.rowIds.map((id) => {
								const member = merged.find((entry) => rowKey(entry) === id);
								return member ? transcriptRow(member) : null;
							})}
						</QuietGroupRow>
					</RowBoundary>;
				}
				if (item.kind !== "row") {
					return <CondensedRows key={item.key} items={[item]} pid={pid} supplements={supplements ?? {}}
						supplementLive={(beat) => beatLive(beat, projectionPid ?? pid)} openRuns={openRuns} onToggle={toggleCondensed} />;
				}
				return <CondensedRows key={item.key} items={[item]} pid={pid} supplements={supplements ?? {}}
					supplementLive={(beat) => beatLive(beat, projectionPid ?? pid)} openRuns={openRuns} onToggle={toggleCondensed} />;
			})}
			{pagedItems.length === 0 && pending.length === 0 ? emptyContent : null}
			{/* After the last projected row, because that is where the session will
			    write them. */}
			{pending.map((echo) => (
				<RowBoundary key={echo.commandId}>
					<PendingEchoRow echo={echo} streaming={streaming} />
				</RowBoundary>
			))}
			{tailContent}
		</div>
	);
}
