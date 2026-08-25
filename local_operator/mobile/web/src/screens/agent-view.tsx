import { useEffect, useMemo, useRef, useState } from "react";
import { getSubagentDetail } from "../api";
import {
	AGENT_GLYPH,
	AgentRoster,
	agentStatusClass,
} from "../components/subagents-panel";
import { TodosPanel } from "../components/todos-panel";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { Markdown } from "../components/markdown";
import { DetailRequestCoordinator } from "../detail-loader";
import { cn } from "../lib/cn";
import { navigate, navigateUp } from "../router";
import type { SessionProjection, SubagentDetail, SubagentRow } from "../types";

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

function Lineage({
	sessionId,
	detail,
	roster,
}: {
	sessionId: string;
	detail: SubagentDetail;
	roster: SubagentRow[];
}) {
	const byId = useMemo(() => new Map(roster.map((row) => [row.job_id, row])), [roster]);
	const parent = detail.parent_job_id ? byId.get(detail.parent_job_id) : null;
	const ancestors = detail.ancestor_ids
		.map((id) => byId.get(id))
		.filter(Boolean) as SubagentRow[];
	const peers = detail.peer_ids.map((id) => byId.get(id)).filter(Boolean) as SubagentRow[];
	return (
		<nav
			aria-label="agent lineage"
			className="lo-scroll flex min-h-11 items-center gap-1 overflow-x-auto border-b border-hairline px-2 py-1"
		>
			<button type="button" onClick={() => navigate(`/s/${encodeURIComponent(sessionId)}`)} className="min-h-11 shrink-0 rounded-sm border border-hairline px-3 text-meta active:bg-elevated">root</button>
			{ancestors.slice(0, -1).map((ancestor) => <button key={ancestor.job_id} type="button" onClick={() => navigate(agentPath(sessionId, ancestor.job_id))} className="min-h-11 max-w-40 shrink-0 truncate rounded-sm border border-hairline px-3 text-meta active:bg-elevated">ancestor · {ancestor.label}</button>)}
			{parent ? <button type="button" onClick={() => navigate(agentPath(sessionId, parent.job_id))} className="min-h-11 shrink-0 rounded-sm border border-hairline px-3 text-meta active:bg-elevated">parent · {parent.label}</button> : null}
			{peers.map((peer) => <button key={peer.job_id} type="button" onClick={() => navigate(agentPath(sessionId, peer.job_id))} className="min-h-11 max-w-40 shrink-0 truncate rounded-sm border border-hairline px-3 text-meta active:bg-elevated">peer · {peer.label}</button>)}
		</nav>
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
		loaderRef.current?.request(projection.version);
	}, [projection.version]);

	if (error && !detail) {
		return <main className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"><p role="alert" className="text-body-sm text-danger">agent unavailable · {error}</p><button type="button" onClick={() => navigate(`/s/${encodeURIComponent(sessionId)}`)} className="min-h-11 rounded-sm border border-hairline px-4 text-body-sm">return to root</button></main>;
	}
	if (!detail) {
		return <main className="flex flex-1 items-center justify-center"><p className="text-body-sm text-ink-dim">{connected ? "loading agent…" : "reconnecting to agent…"}</p></main>;
	}
	const directChildren = projection.subagents.filter((row) => row.parent_job_id === jobId);
	const parentPath = detail.parent_job_id
		? agentPath(sessionId, detail.parent_job_id)
		: `/s/${encodeURIComponent(sessionId)}`;
	const outcome = detail.status === "completed" && detail.result_text ? (
		<section aria-live="polite" className="mt-2 border-t border-success-border bg-success-wash px-3 py-3">
			<p className="mb-2 text-meta font-medium text-success">Handoff</p>
			<Markdown text={detail.result_text} />
		</section>
	) : detail.status === "failed" && detail.error_text ? (
		<section role="alert" className="mt-2 border-t border-danger-border bg-danger-wash px-3 py-3">
			<p className="mb-2 text-meta font-medium text-danger">Failure</p>
			<Markdown text={detail.error_text} />
		</section>
	) : null;
	return <>
		<header className="flex min-h-11 items-center gap-2 border-b border-hairline px-2 pt-[max(env(safe-area-inset-top),0.25rem)]">
			<button type="button" onClick={() => navigateUp(parentPath)} aria-label="back to parent conversation" className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated">‹</button>
			<div className="min-w-0 flex-1"><p className="truncate text-body-sm font-medium">{detail.label}</p><p className="truncate font-mono text-mono-sm text-ink-dim">{detail.agent}{detail.effort ? ` · ${detail.effort}` : ""}</p></div>
			<span className={cn("shrink-0 font-mono text-mono-sm", agentStatusClass(detail.status))}>{AGENT_GLYPH[detail.status]} {detail.status}</span>
		</header>
		<Lineage sessionId={sessionId} detail={detail} roster={projection.subagents} />
		{!connected ? <div role="status" className="border-b border-warning-border bg-warning-wash px-3 py-2 text-meta text-warning">connection lost · reconnecting with cached agent detail…</div> : null}
		{detail.prompt ? <section className="border-b border-hairline px-3 py-2"><p className="text-meta text-ink-dim">Parent request</p><p className="max-h-28 overflow-y-auto whitespace-pre-wrap break-words text-body-sm">{detail.prompt}</p></section> : null}
		<Transcript
			pid={sessionId}
			jobId={jobId}
			entries={detail.transcript}
			tailContent={outcome}
			emptyContent={outcome ? null : <div className="flex flex-1 items-center justify-center px-8 text-center"><p className="text-body-sm text-ink-dim">no agent messages recorded</p></div>}
		/>
		{detail.status === "running" ? <WorkingLine activity={detail.activity || detail.progress || "thinking"} startedS={detail.elapsed_s} /> : null}
		{detail.todos.some((phase) => phase.items.length > 0) ? <TodosPanel todos={detail.todos} /> : null}
		{directChildren.length > 0 ? <AgentRoster sessionId={sessionId} subagents={projection.subagents} parentJobId={jobId} /> : null}
		<footer className="border-t border-hairline px-3 py-2 pb-[max(env(safe-area-inset-bottom),0.5rem)] text-center text-meta text-ink-dim">read-only · send commands from the root conversation</footer>
	</>;
}
