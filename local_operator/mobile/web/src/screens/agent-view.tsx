import { useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { getSubagentDetail, getSubagentHistory } from "../api";
import { AgentsSheet } from "../components/agents-sheet";
import {
	AGENT_GLYPH,
	AgentRoster,
	agentStatusClass,
} from "../components/subagents-panel";
import { Markdown } from "../components/markdown";
import { TodosPanel } from "../components/todos-panel";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { DetailRequestCoordinator } from "../detail-loader";
import { cn } from "../lib/cn";
import { formatEffort } from "../lib/format";
import { navigate, navigateUp } from "../router";
import type {
	SessionProjection,
	SubagentDetail,
	SubagentRow,
	SubagentStatus,
	TranscriptEntry,
} from "../types";

/** The identity a stable header needs before the full detail lands. A tapped
 * roster row already carries all of it, so the header can paint the real
 * label/agent/effort/status immediately and let only the transcript BODY show
 * the loading state, instead of flickering to a generic "Agent" placeholder
 * and reflowing in a status pill after the fetch (design D1). */
interface HeaderIdentity {
	label: string;
	agent: string;
	effort: string;
	status: SubagentStatus;
}

function identityFromRow(row: SubagentRow): HeaderIdentity {
	return { label: row.label, agent: row.agent, effort: row.effort, status: row.status };
}

const CACHE_MAX = 24;
const detailCache = new Map<string, SubagentDetail>();

function cacheDetail(key: string, detail: SubagentDetail) {
	detailCache.delete(key);
	detailCache.set(key, detail);
	while (detailCache.size > CACHE_MAX) {
		const oldest = detailCache.keys().next().value as string | undefined;
		if (!oldest) break;
		detailCache.delete(oldest);
	}
}

function agentPath(sessionId: string, jobId: string): string {
	return `/s/${encodeURIComponent(sessionId)}/a/${encodeURIComponent(jobId)}`;
}

function normalizeText(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

/** Bound the daemon applies to a row's `prompt` before it reaches the wire.
 * Mirrors `SUBAGENT_PROMPT_PREVIEW_CHARS` in `local_operator/mobile/projection.py`,
 * where `_compact` emits `text[: limit - 1] + "…"` — so a preview that was
 * actually truncated is exactly this long, and a SHORTER prompt ending in `…`
 * ends that way because its author typed the character. */
const SUBAGENT_PROMPT_PREVIEW_CHARS = 1_000;

/** The row text plus each of its paragraph-delimited suffixes, longest first.
 *
 * A launch message is structurally `{preamble}\n\n{prompt}`, and every
 * preamble builder (`AgentProfile.preamble`, `Team.member_preamble`,
 * `SCOUT_PREAMBLE`, the specialist join) terminates on a blank line — so the
 * launch task always begins right after a paragraph break, or at offset 0 when
 * there is no preamble at all. Splitting there is what lets the comparison stay
 * anchored to a real structural boundary: a steer that merely CLOSES by
 * restating the task ("Actually ignore that and Implement the route") has no
 * blank line in front of the restatement and so never yields a matching
 * candidate, while a genuine launch row does.
 *
 * Whitespace inside each candidate is normalised because the wire `prompt` has
 * already been flattened by `_compact`; only the paragraph boundaries
 * themselves must survive long enough to be split on.
 *
 * Returned as ONE normalised string plus the offset each candidate starts at,
 * rather than as a list of suffix strings, to keep the work linear. Building
 * and normalising every suffix separately re-scans the tail of the row once
 * per paragraph break — (breaks x length), measured at 9.4s for a 580KB row
 * with 2000 breaks. The wire `text` for `user`, `assistant` and
 * `parent_message` rows is NOT length-bounded (only `notice`, tool args/output
 * and outcome fields go through `_compact`), so a long pasted row is a
 * reachable input, and the phone renders this on the main thread.
 *
 * The rewrite is exact rather than approximate: normalising collapses every
 * whitespace run to one space, so the normalised full text is just its words
 * joined by single spaces, and the normalised form of a suffix beginning at a
 * paragraph break is the same string from that word onward. Segments that
 * normalise to nothing (consecutive breaks) contribute no words and so are
 * dropped — they can only duplicate the following candidate's offset. Each
 * comparison then costs the needle's length instead of the row's. */
function launchCandidates(text: string): { normalized: string; offsets: number[] } {
	const offsets: number[] = [];
	const parts: string[] = [];
	let length = 0;
	for (const segment of text.split(/\n[ \t]*\n/)) {
		const part = normalizeText(segment);
		if (!part) continue;
		// Offset of this segment's first word in the joined string, which is where
		// the candidate starting at the preceding break begins.
		offsets.push(length);
		parts.push(part);
		length += part.length + 1; // +1 for the single space the join inserts.
	}
	return { normalized: parts.join(" "), offsets };
}

/** Does this `parent_message` row already carry the child's LAUNCH task?
 *
 * Identity first: an id equal to `launch_message_id` is conclusive. Legacy and
 * summary-stripped rows have no such id, so fall back to the prompt text — but
 * anchored to the paragraph boundary the launch task starts at, never as a
 * loose substring, because a steer that quotes the task ("Implement the route
 * more narrowly") would satisfy a substring test and wrongly suppress the head.
 *
 * `prompt` arrives as a bounded PREVIEW, so a genuinely truncated one is
 * matched as a PREFIX of its candidate rather than by equality. That looser
 * comparison is gated on the preview bound as well as the trailing ellipsis:
 * keying on the character alone put an author's own "…" — which costs a human
 * one keystroke — on the loose path and reopened the false positive the
 * anchoring exists to close. */
function isLaunchHead(entry: TranscriptEntry, detail: SubagentDetail): boolean {
	if (detail.launch_message_id && entry.id === detail.launch_message_id) return true;
	const prompt = normalizeText(detail.prompt);
	if (!prompt) return false;
	// `- 1` tolerates a preview whose flattening trimmed a character; the cap is
	// ~1000, so the loose path still demands that much agreement to be reached.
	const truncated = prompt.endsWith("…") && prompt.length >= SUBAGENT_PROMPT_PREVIEW_CHARS - 1;
	const needle = truncated ? prompt.slice(0, -1) : prompt;
	if (!needle) return false;
	const { normalized, offsets } = launchCandidates(entry.text);
	return offsets.some((offset) =>
		// `startsWith(needle, offset)` is the candidate's prefix test without
		// materialising the suffix; equality additionally demands that the
		// candidate end where the needle does.
		truncated
			? normalized.startsWith(needle, offset)
			: normalized.length - offset === needle.length && normalized.startsWith(needle, offset),
	);
}

/** The launch prompt is durable child history, while `prompt` is the raw
 * parent task retained for queued/legacy rows. Correlation identity lets the
 * opening role-expanded user row carry that task's visual semantics without
 * text matching or hiding later user/parent steering turns. */
export function agentConversationEntries(detail: SubagentDetail): TranscriptEntry[] {
	if (detail.launch_message_id) {
		const launchIndex = detail.transcript.findIndex(
			(entry) => entry.id === detail.launch_message_id && entry.kind === "user",
		);
		if (launchIndex >= 0) {
			return detail.transcript.map((entry, index) =>
				index === launchIndex ? { ...entry, kind: "parent_message" } : entry,
			);
		}
	}
	// Only a row carrying the LAUNCH task suppresses the synthesized head. The
	// guard used to test for `parent_message` as such, which broke once a
	// persisted hub steer started folding to that kind server-side: a legacy
	// child (no `launch_message_id`, or a summary row whose id the daemon
	// stripped) that had ever been steered lost the row naming what the whole
	// conversation is about. A steer is a mid-conversation redirection and
	// never stands in for the launch task, so match on the launch text rather
	// than on the kind.
	const hasLaunchHead = detail.transcript.some(
		(entry) => entry.kind === "parent_message" && isLaunchHead(entry, detail),
	);
	if (!detail.prompt || hasLaunchHead) {
		return detail.transcript;
	}
	return [
		{
			id: `prompt:${detail.job_id}`,
			kind: "parent_message",
			text: detail.prompt,
			tool_call_id: "",
			tool_name: "",
			tool_state: "done",
			summary: "",
			intent: "",
			diff_added: 0,
			diff_removed: 0,
			elapsed_s: 0,
			error: "",
			details: {},
			final: true,
		},
		...detail.transcript,
	];
}

function Outcome({ detail }: { detail: SubagentDetail }) {
	if (detail.status === "completed") {
		return (
			<section aria-live="polite" className="mt-2 border-l-2 border-success pl-3">
				<p className="mb-1 text-meta font-medium text-success">
					✓ Result from {detail.label}
				</p>
				{detail.result_text ? (
					<Markdown text={detail.result_text} />
				) : (
					<p className="text-body-sm text-ink-muted">Agent completed without a result.</p>
				)}
			</section>
		);
	}
	if (detail.status === "failed") {
		return (
			<section role="alert" className="mt-2 border-l-2 border-danger pl-3">
				<p className="mb-1 text-meta font-medium text-danger">✕ Agent failed</p>
				{detail.error_text ? (
					<Markdown text={detail.error_text} />
				) : (
					<p className="text-body-sm text-ink-muted">No failure details were recorded.</p>
				)}
			</section>
		);
	}
	if (detail.status === "cancelled" || detail.status === "parked") {
		return (
			<p className="mt-2 text-body-sm text-ink-muted">
				{detail.status === "cancelled" ? "Agent stopped." : "Agent paused."}
			</p>
		);
	}
	return null;
}

function ConversationTail({
	detail,
	sessionId,
	projection,
}: {
	detail: SubagentDetail;
	sessionId: string;
	projection: SessionProjection;
}) {
	const children = projection.subagents.filter((row) => row.parent_job_id === detail.job_id);
	return (
		<>
			{detail.status === "running" ? (
				<WorkingLine
					activity={detail.activity || detail.progress || "Waiting for the first response…"}
					startedS={detail.elapsed_s}
				/>
			) : (
				<Outcome detail={detail} />
			)}
			{detail.todos.some((phase) => phase.items.length > 0) ? (
				<div className="mt-2">
					<TodosPanel todos={detail.todos} embedded />
				</div>
			) : null}
			{children.length > 0 ? (
				<div className="mt-1">
					<AgentRoster
						sessionId={sessionId}
						subagents={projection.subagents}
						parentJobId={detail.job_id}
						collapsible
						embedded
						label="child agents"
					/>
				</div>
			) : null}
		</>
	);
}

function AgentHeader({
	detail,
	identity,
	parentPath,
	onOpenAgents,
	agentsButtonRef,
	fallbackSubtitle = "Loading activity",
}: {
	detail: SubagentDetail | null;
	/** Known label/agent/effort/status from the tapped roster row, used to keep
	 * the header stable while the full detail is still loading (design D1). When
	 * a real ``detail`` is present it wins; otherwise this paints the identity
	 * the user already saw instead of a generic placeholder. */
	identity?: HeaderIdentity | null;
	parentPath: string;
	onOpenAgents?: () => void;
	agentsButtonRef?: RefObject<HTMLButtonElement | null>;
	fallbackSubtitle?: string;
}) {
	const shown: HeaderIdentity | null = detail
		? { label: detail.label, agent: detail.agent, effort: detail.effort, status: detail.status }
		: identity ?? null;
	return (
		<header className="flex min-h-[52px] items-center gap-1 border-b border-hairline bg-surface px-1 pt-[max(env(safe-area-inset-top),0.25rem)]">
			<button
				type="button"
				onClick={() => navigateUp(parentPath)}
				aria-label="back to parent conversation"
				className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
			>
				‹
			</button>
			<div className="min-w-0 flex-1">
				<p className="truncate text-body-sm font-medium">{shown?.label || "Agent"}</p>
				<p className="truncate text-meta text-ink-dim">
					{shown
						? `${shown.agent}${shown.effort ? ` · ${formatEffort(shown.effort)}` : ""}`
						: fallbackSubtitle}
				</p>
			</div>
			{shown ? (
				<span className={cn("max-w-20 shrink truncate text-meta", agentStatusClass(shown.status))}>
					<span aria-hidden className="font-mono">{AGENT_GLYPH[shown.status]}</span>{" "}
					{shown.status}
				</span>
			) : null}
			{/* The Agents navigator needs the full detail (path/peers/children), so
			   it only appears once detail lands — the identity-only header still
			   keeps its label/status stable in the meantime. */}
			{detail && onOpenAgents ? (
				<button
					ref={agentsButtonRef}
					type="button"
					onClick={onOpenAgents}
					aria-label="open agent navigation"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-sm px-2 text-body-sm text-ink-muted active:bg-elevated"
				>
					Agents
				</button>
			) : null}
		</header>
	);
}

function InlineState({ children }: { children: ReactNode }) {
	return <main className="flex min-h-0 flex-1 flex-col gap-2 px-3 py-4">{children}</main>;
}

export function AgentUnavailable({
	sessionId,
	parentPath,
	onRetry,
}: {
	sessionId: string;
	parentPath?: string;
	onRetry?: () => void;
}) {
	const rootPath = `/s/${encodeURIComponent(sessionId)}`;
	const fallbackPath = parentPath || rootPath;
	return (
		<>
			<AgentHeader detail={null} parentPath={fallbackPath} fallbackSubtitle="Unavailable" />
			<InlineState>
				<p role="alert" className="text-body-sm font-medium text-ink">This agent is no longer available.</p>
				<p className="text-body-sm text-ink-dim">It may have expired or been removed.</p>
				<div className="mt-2 flex flex-wrap gap-2">
					{onRetry ? <button type="button" onClick={onRetry} className="min-h-11 rounded-sm border border-control px-3 text-body-sm">Retry</button> : null}
					<button type="button" onClick={() => navigateUp(fallbackPath)} className="min-h-11 rounded-sm border border-control px-3 text-body-sm">Back to parent</button>
					<button type="button" onClick={() => navigate(rootPath)} className="min-h-11 rounded-sm px-3 text-body-sm text-ink-muted active:bg-elevated">View root</button>
				</div>
			</InlineState>
		</>
	);
}

export function AgentLoading({
	sessionId,
	connected,
	identity = null,
	parentPath,
}: {
	sessionId: string;
	connected: boolean;
	identity?: HeaderIdentity | null;
	parentPath?: string;
}) {
	const rootPath = `/s/${encodeURIComponent(sessionId)}`;
	return (
		<>
			<AgentHeader detail={null} identity={identity} parentPath={parentPath ?? rootPath} />
			<InlineState>
				<TranscriptLoading connected={connected} />
			</InlineState>
		</>
	);
}

/** The body-level "loading transcript" affordance, shared by the full-screen
 * detail load and the open→transcript gap after detail lands (U2). The bars use
 * ``bg-elevated`` rather than ``bg-sunken`` because sunken sits at ~1.3:1
 * against the page and all but vanishes in a still or under phone glare (design
 * D2); elevated lifts them a clear step while the shimmer still animates on top.
 * A visible ``lo-spinner`` accompanies the label so the state reads as "loading"
 * even with motion disabled and even before the shimmer registers. */
function TranscriptLoading({ connected }: { connected: boolean }) {
	return (
		<>
			<div className="flex items-center gap-2" role="status">
				<span aria-hidden className="lo-spinner h-4 w-4 rounded-full" />
				<p className="text-body-sm text-ink-dim">
					{connected ? "Loading agent activity…" : "Connecting to saved agent activity…"}
				</p>
			</div>
			<div aria-hidden className="mt-2 flex flex-col gap-3">
				<span className="lo-shimmer-bar h-4 w-3/4 rounded-sm bg-elevated" />
				<span className="lo-shimmer-bar h-4 w-full rounded-sm bg-elevated" />
				<span className="lo-shimmer-bar h-4 w-2/3 rounded-sm bg-elevated" />
			</div>
		</>
	);
}

/** The tail of a child transcript the sheet paints without scrolling. Matches
 * the daemon's PROJECTION_TRANSCRIPT_LIMIT so the initial fetch is the same
 * window the projection used to inline, before it was removed from the wire. */
const SUBAGENT_TRANSCRIPT_TAIL = 80;
/** How often an OPEN sheet for a still-running child re-pulls its transcript.
 * The list projection keeps status/progress/todos live over SSE with no poll;
 * only the full transcript is fetched, and only while the child is running and
 * its sheet is on screen. 1.5s is frequent enough to read as real-time without
 * turning one open sheet into a tight request loop against the child's file. */
const SUBAGENT_TRANSCRIPT_POLL_MS = 1500;

/** Lazily hydrate a child's transcript for the open sheet.
 *
 * Subagent transcripts are no longer carried in the projection wire (they blew
 * past the daemon's 1 MB control-frame cap and wedged real-time updates for the
 * whole session — see ``ProjectionFold.set_subagent_hydrated_details``). So when
 * a modern daemon sends an empty ``detail.transcript``, the sheet fetches the
 * newest page from the lineage-checked history endpoint instead, and — while the
 * child is still running — re-fetches on a modest interval so the open sheet
 * stays live. A legacy daemon that still inlines ``detail.transcript`` is honored
 * as-is (no fetch), which also keeps the component's unit tests transcript-driven.
 *
 * Older pages are still paged in by ``<Transcript>`` itself via the same
 * endpoint as the user scrolls up; this hook only owns the newest tail.
 */
/** Cheap equality for two transcript tails: same length and, per row, same id
 * and same text length. A streaming assistant row grows its text, so comparing
 * text length (not full text) catches live growth without an O(chars) compare,
 * while a settled poll that returns the identical page short-circuits the
 * setState and preserves array identity (U3). */
function sameTail(a: TranscriptEntry[], b: TranscriptEntry[]): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (a[i].id !== b[i].id || a[i].text.length !== b[i].text.length) return false;
	}
	return true;
}

interface LazyTranscript {
	entries: TranscriptEntry[];
	/** A fetch is in flight and nothing has landed yet: the body shows its
	 * loading affordance instead of a blank window (U2). */
	loading: boolean;
	/** The (single, for a settled child) fetch failed and no entries are shown:
	 * the body must surface an error + retry rather than a silent blank (U1). */
	failed: boolean;
	/** Imperative re-pull for the retry affordance. */
	retry: () => void;
}

function useLazySubagentTranscript(
	sessionId: string,
	jobId: string,
	detail: SubagentDetail,
	connected: boolean,
): LazyTranscript {
	// A daemon that still inlines the transcript wins outright: never fetch. This
	// gates the fetch effect below (in its dep list) so legacy payloads render
	// straight from the wire with no network round trip.
	const inlined = detail.transcript.length > 0;
	const [fetched, setFetched] = useState<TranscriptEntry[]>([]);
	const [loading, setLoading] = useState(!inlined);
	const [failed, setFailed] = useState(false);
	// A monotonic nonce the retry button bumps to force the fetch effect to
	// re-run even when nothing else in its dep list changed (a settled child on
	// the same connection). Mirrors the detail loader's explicit retry signal.
	const [attempt, setAttempt] = useState(0);
	const running = detail.status === "running";
	useEffect(() => {
		// Reset when the addressed child changes so a stale page never flashes
		// under a newly-opened sibling before its own fetch lands.
		setFetched([]);
		setLoading(true);
		setFailed(false);
	}, [sessionId, jobId]);
	useEffect(() => {
		if (inlined) return;
		let alive = true;
		let controller: AbortController | null = null;
		const pull = async () => {
			controller?.abort();
			controller = new AbortController();
			try {
				const { entries } = await getSubagentHistory(
					sessionId,
					jobId,
					null,
					SUBAGENT_TRANSCRIPT_TAIL,
					controller.signal,
				);
				if (alive) {
					// Keep the previous array identity when a poll tick returns the
					// same tail (id + text length unchanged), so a long-running
					// child does not re-diff/re-render the whole window every
					// 1.5s for no visible change (U3). A genuine growth or edit
					// still swaps in the fresh array.
					setFetched((prev) => (sameTail(prev, entries) ? prev : entries));
					setLoading(false);
					setFailed(false);
				}
			} catch (err) {
				/* An aborted request is a teardown/replacement, not a failure:
				   the successor pull owns the state. A real drop for a RUNNING
				   child self-heals on the next poll tick, so it only reflects as
				   loading; for a SETTLED child there is no poll, so the single
				   dropped fetch is terminal and must surface a retry (U1). */
				if (!alive || (err instanceof DOMException && err.name === "AbortError")) return;
				setLoading(false);
				setFailed(true);
			}
		};
		void pull();
		// Only an actively-running child needs the poll; a settled child's
		// transcript is final, so one fetch suffices and the interval is skipped.
		const timer = running
			? window.setInterval(() => void pull(), SUBAGENT_TRANSCRIPT_POLL_MS)
			: undefined;
		return () => {
			alive = false;
			controller?.abort();
			if (timer !== undefined) window.clearInterval(timer);
		};
		// ``connected`` is a dep so a restored link re-pulls a settled child that
		// missed its one fetch while offline — the detail loader already retries
		// on reconnect this way (U1); ``attempt`` covers the manual retry button.
	}, [sessionId, jobId, inlined, running, connected, attempt]);
	const retry = () => {
		setLoading(true);
		setFailed(false);
		setAttempt((n) => n + 1);
	};
	if (inlined) {
		return { entries: detail.transcript, loading: false, failed: false, retry };
	}
	return { entries: fetched, loading, failed, retry };
}

/** The body shown when a settled child's single transcript fetch failed and no
 * entries are on screen (U1). Without this the body was a permanent blank on a
 * transient link drop, with no signal anything failed and no way to recover
 * short of navigating away — the exact flaky-link case this surface targets.
 * The Outcome/todos tail still renders below via ``tailContent``; this only
 * fills the empty transcript window with a reason and an in-place retry. */
function TranscriptFetchError({
	connected,
	onRetry,
}: {
	connected: boolean;
	onRetry: () => void;
}) {
	return (
		<div role="alert" className="flex flex-col items-start gap-2 py-4">
			<p className="text-body-sm font-medium text-ink">Couldn't load the transcript.</p>
			<p className="text-meta text-ink-dim">
				{connected
					? "The conversation steps didn't load. Retry to pull them again."
					: "You're offline. Reconnecting will retry automatically."}
			</p>
			<button
				type="button"
				onClick={onRetry}
				className="min-h-11 rounded-sm border border-control px-3 text-body-sm active:bg-elevated"
			>
				Retry
			</button>
		</div>
	);
}

/** Keep loading/cache ownership outside the visual component so its complete
 * state can be exercised independently without introducing a second route. */
export function AgentConversation({
	sessionId,
	jobId,
	projection,
	connected,
	detail,
	initialAgentsOpen = false,
}: {
	sessionId: string;
	jobId: string;
	projection: SessionProjection;
	connected: boolean;
	detail: SubagentDetail;
	initialAgentsOpen?: boolean;
}) {
	const [agentsOpen, setAgentsOpen] = useState(initialAgentsOpen);
	const agentsButtonRef = useRef<HTMLButtonElement>(null);
	const rootPath = `/s/${encodeURIComponent(sessionId)}`;
	const parentPath = detail.parent_job_id
		? agentPath(sessionId, detail.parent_job_id)
		: rootPath;
	const { entries: transcript, loading, failed, retry } = useLazySubagentTranscript(
		sessionId,
		jobId,
		detail,
		connected,
	);
	// ``agentConversationEntries`` reads ``detail.transcript``; feed it the
	// lazily-fetched tail when the wire no longer carries one. Identity-stable
	// via useMemo so the transcript array only rebuilds when its inputs change.
	const detailForRender = useMemo(
		() => (detail.transcript.length > 0 ? detail : { ...detail, transcript }),
		[detail, transcript],
	);
	const entries = useMemo(() => agentConversationEntries(detailForRender), [detailForRender]);
	// The transcript body owns three off-happy-path states when the wire carries
	// no entries: a fetch in flight (loading affordance, not a blank window —
	// U2), a settled child whose one fetch failed (visible error + retry so it is
	// recoverable in place, never a silent blank — U1), or genuinely empty. The
	// header/outcome above stay put; only the BODY reflects these.
	const emptyBody = failed ? (
		<TranscriptFetchError connected={connected} onRetry={retry} />
	) : loading ? (
		<div className="py-4">
			<TranscriptLoading connected={connected} />
		</div>
	) : null;
	return (
		<>
			<AgentHeader
				detail={detail}
				parentPath={parentPath}
				onOpenAgents={() => setAgentsOpen(true)}
				agentsButtonRef={agentsButtonRef}
			/>
			{!connected ? (
				<div role="status" className="border-b border-warning-border bg-warning-wash px-3 py-2 text-meta text-warning">
					Reconnecting. Showing saved agent activity.
				</div>
			) : null}
			<Transcript
				pid={sessionId}
				jobId={jobId}
				entries={entries}
				tailContent={<ConversationTail detail={detail} sessionId={sessionId} projection={projection} />}
				emptyContent={emptyBody}
			/>
			{detail.status === "running" ? (
				<footer className="flex min-h-11 items-center justify-between border-t border-hairline bg-surface px-3 pb-[max(env(safe-area-inset-bottom),0.25rem)] text-meta">
					<span className="text-ink-dim">Read-only</span>
					<button type="button" onClick={() => navigate(parentPath)} className="min-h-11 rounded-sm px-2 text-info active:bg-elevated">Open parent to steer</button>
				</footer>
			) : null}
			<AgentsSheet
				open={agentsOpen}
				onClose={() => setAgentsOpen(false)}
				returnFocusRef={agentsButtonRef}
				sessionId={sessionId}
				detail={detail}
				roster={projection.subagents}
			/>
		</>
	);
}

export function AgentScreen({
	sessionId,
	jobId,
	projection,
	connected,
}: {
	sessionId: string;
	jobId: string;
	projection: SessionProjection;
	connected: boolean;
}) {
	const key = `${sessionId}:${jobId}`;
	const [detail, setDetail] = useState<SubagentDetail | null>(() => detailCache.get(key) ?? null);
	const [error, setError] = useState("");
	const loaderRef = useRef<DetailRequestCoordinator | null>(null);
	useEffect(() => {
		setDetail(detailCache.get(key) ?? null);
		setError("");
		const loader = new DetailRequestCoordinator(
			() => getSubagentDetail(sessionId, jobId),
			(next) => {
				cacheDetail(key, next);
				setDetail((current) => !current || next.version >= current.version ? next : current);
				setError("");
			},
			(reason) => setError(reason instanceof Error ? reason.message : "failed to load agent"),
		);
		loaderRef.current = loader;
		loader.request(projection.version);
		return () => {
			loader.dispose();
			loaderRef.current = null;
		};
	}, [jobId, key, sessionId]);
	useEffect(() => {
		/* Reconnection can restore a transiently unavailable detail endpoint while
		   the retained projection version stays unchanged, so connectivity is an
		   explicit retry signal rather than waiting for another projection frame. */
		if (connected) loaderRef.current?.request(projection.version);
	}, [connected, projection.version]);

	if (error && !detail) {
		const summary = projection.subagents.find((row) => row.job_id === jobId);
		const parentPath = summary?.parent_job_id
			? agentPath(sessionId, summary.parent_job_id)
			: `/s/${encodeURIComponent(sessionId)}`;
		return (
			<AgentUnavailable
				sessionId={sessionId}
				parentPath={parentPath}
				onRetry={() => {
					setError("");
					loaderRef.current?.request(projection.version);
				}}
			/>
		);
	}
	if (!detail) {
		/* The user tapped a roster row that already carried the label, agent,
		   effort and status. Feed that known identity into the loading header so
		   only the BODY shows the skeleton — the header no longer flickers to a
		   generic "Agent" placeholder and reflows a status pill in after the
		   fetch lands (design D1). The metadata is still on the wire in the
		   projection roster even though the transcript is not. */
		const summary = projection.subagents.find((row) => row.job_id === jobId);
		const parentPath = summary?.parent_job_id
			? agentPath(sessionId, summary.parent_job_id)
			: `/s/${encodeURIComponent(sessionId)}`;
		return (
			<AgentLoading
				sessionId={sessionId}
				connected={connected}
				identity={summary ? identityFromRow(summary) : null}
				parentPath={parentPath}
			/>
		);
	}

	return (
		<AgentConversation
			sessionId={sessionId}
			jobId={jobId}
			projection={projection}
			connected={connected}
			detail={detail}
		/>
	);
}
