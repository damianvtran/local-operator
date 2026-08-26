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
import { navigate, navigateUp } from "../router";
import type { SessionProjection, SubagentDetail, TranscriptEntry } from "../types";

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
	if (!detail.prompt || detail.transcript.some((entry) => entry.kind === "parent_message")) {
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
	parentPath,
	onOpenAgents,
	agentsButtonRef,
	fallbackSubtitle = "Loading activity",
}: {
	detail: SubagentDetail | null;
	parentPath: string;
	onOpenAgents?: () => void;
	agentsButtonRef?: RefObject<HTMLButtonElement | null>;
	fallbackSubtitle?: string;
}) {
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
				<p className="truncate text-body-sm font-medium">{detail?.label || "Agent"}</p>
				<p className="truncate text-meta text-ink-dim">
					{detail ? `${detail.agent}${detail.effort ? ` · ${detail.effort}` : ""}` : fallbackSubtitle}
				</p>
			</div>
			{detail ? (
				<span className={cn("max-w-20 shrink truncate text-meta", agentStatusClass(detail.status))}>
					<span aria-hidden className="font-mono">{AGENT_GLYPH[detail.status]}</span>{" "}
					{detail.status}
				</span>
			) : null}
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

export function AgentLoading({ sessionId, connected }: { sessionId: string; connected: boolean }) {
	const rootPath = `/s/${encodeURIComponent(sessionId)}`;
	return (
		<>
			<AgentHeader detail={null} parentPath={rootPath} />
			<InlineState>
				<p className="text-body-sm text-ink-dim">{connected ? "Loading agent activity…" : "Connecting to saved agent activity…"}</p>
				<div aria-hidden className="mt-2 flex flex-col gap-3">
					<span className="h-4 w-3/4 rounded-sm bg-sunken" />
					<span className="h-4 w-full rounded-sm bg-sunken" />
					<span className="h-4 w-2/3 rounded-sm bg-sunken" />
				</div>
			</InlineState>
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
function useLazySubagentTranscript(
	sessionId: string,
	jobId: string,
	detail: SubagentDetail,
): TranscriptEntry[] {
	// A daemon that still inlines the transcript wins outright: never fetch. This
	// gates the fetch effect below (in its dep list) so legacy payloads render
	// straight from the wire with no network round trip.
	const inlined = detail.transcript.length > 0;
	const [fetched, setFetched] = useState<TranscriptEntry[]>([]);
	const running = detail.status === "running";
	useEffect(() => {
		// Reset when the addressed child changes so a stale page never flashes
		// under a newly-opened sibling before its own fetch lands.
		setFetched([]);
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
				if (alive) setFetched(entries);
			} catch {
				/* A dropped page is not fatal: the next poll (or projection
				   frame remount) retries, and the working line stays live from
				   the SSE projection regardless. */
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
	}, [sessionId, jobId, inlined, running]);
	return inlined ? detail.transcript : fetched;
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
	const transcript = useLazySubagentTranscript(sessionId, jobId, detail);
	// ``agentConversationEntries`` reads ``detail.transcript``; feed it the
	// lazily-fetched tail when the wire no longer carries one. Identity-stable
	// via useMemo so the transcript array only rebuilds when its inputs change.
	const detailForRender = useMemo(
		() => (detail.transcript.length > 0 ? detail : { ...detail, transcript }),
		[detail, transcript],
	);
	const entries = useMemo(() => agentConversationEntries(detailForRender), [detailForRender]);
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
				emptyContent={null}
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
	if (!detail) return <AgentLoading sessionId={sessionId} connected={connected} />;

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
