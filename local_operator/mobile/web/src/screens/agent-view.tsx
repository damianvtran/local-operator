import { useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { getSubagentDetail } from "../api";
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

/** A prompt is legacy fallback context. Current projections also carry the
 * authored parent_message in the transcript; rendering both makes one request
 * look like two distinct turns. */
export function agentConversationEntries(detail: SubagentDetail): TranscriptEntry[] {
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

export function AgentUnavailable({ sessionId, onRetry }: { sessionId: string; onRetry?: () => void }) {
	const rootPath = `/s/${encodeURIComponent(sessionId)}`;
	return (
		<>
			<AgentHeader detail={null} parentPath={rootPath} fallbackSubtitle="Unavailable" />
			<InlineState>
				<p role="alert" className="text-body-sm font-medium text-ink">This agent is no longer available.</p>
				<p className="text-body-sm text-ink-dim">It may have expired or been removed.</p>
				<div className="mt-2 flex flex-wrap gap-2">
					{onRetry ? <button type="button" onClick={onRetry} className="min-h-11 rounded-sm border border-control px-3 text-body-sm">Retry</button> : null}
					<button type="button" onClick={() => navigateUp(rootPath)} className="min-h-11 rounded-sm border border-control px-3 text-body-sm">Back to parent</button>
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
	const entries = useMemo(() => agentConversationEntries(detail), [detail]);
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
		return (
			<AgentUnavailable
				sessionId={sessionId}
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
