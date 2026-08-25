import { useEffect, useMemo, useState } from "react";
import { getSubagentDetail } from "../api";
import {
	AGENT_GLYPH,
	AgentRoster,
	agentStatusClass,
} from "../components/subagents-panel";
import { TodosPanel } from "../components/todos-panel";
import { Transcript } from "../components/transcript";
import { WorkingLine } from "../components/working-line";
import { cn } from "../lib/cn";
import { navigate } from "../router";
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
			<button type="button" onClick={() => navigate(`/s/${encodeURIComponent(sessionId)}`)} className="min-h-9 shrink-0 rounded-sm border border-hairline px-3 text-meta active:bg-elevated">root</button>
			{ancestors.slice(0, -1).map((ancestor) => <button key={ancestor.job_id} type="button" onClick={() => navigate(agentPath(sessionId, ancestor.job_id))} className="min-h-9 max-w-40 shrink-0 truncate rounded-sm border border-hairline px-3 text-meta active:bg-elevated">ancestor · {ancestor.label}</button>)}
			{parent ? <button type="button" onClick={() => navigate(agentPath(sessionId, parent.job_id))} className="min-h-9 shrink-0 rounded-sm border border-hairline px-3 text-meta active:bg-elevated">parent · {parent.label}</button> : null}
			{peers.map((peer) => <button key={peer.job_id} type="button" onClick={() => navigate(agentPath(sessionId, peer.job_id))} className="min-h-9 max-w-40 shrink-0 truncate rounded-sm border border-hairline px-3 text-meta active:bg-elevated">peer · {peer.label}</button>)}
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
	useEffect(() => {
		const controller = new AbortController();
		setError("");
		void getSubagentDetail(sessionId, jobId, controller.signal)
			.then((next) => {
				cacheDetail(key, next);
				setDetail((current) => !current || next.version >= current.version ? next : current);
			})
			.catch((reason: unknown) => {
				if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "failed to load agent");
			});
		return () => controller.abort();
	}, [jobId, key, projection.version, sessionId]);

	if (error && !detail) {
		return <main className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"><p role="alert" className="text-body-sm text-danger">agent unavailable · {error}</p><button type="button" onClick={() => navigate(`/s/${encodeURIComponent(sessionId)}`)} className="min-h-11 rounded-sm border border-hairline px-4 text-body-sm">return to root</button></main>;
	}
	if (!detail) {
		return <main className="flex flex-1 items-center justify-center"><p className="text-body-sm text-ink-dim">{connected ? "loading agent…" : "reconnecting to agent…"}</p></main>;
	}
	const directChildren = projection.subagents.filter((row) => row.parent_job_id === jobId);
	return <>
		<header className="flex min-h-11 items-center gap-2 border-b border-hairline px-2 pt-[max(env(safe-area-inset-top),0.25rem)]">
			<button type="button" onClick={() => history.back()} aria-label="back" className="flex min-h-10 min-w-10 items-center justify-center rounded-sm text-ink-muted active:bg-elevated">‹</button>
			<div className="min-w-0 flex-1"><p className="truncate text-body-sm font-medium">{detail.label}</p><p className="truncate font-mono text-mono-sm text-ink-dim">{detail.agent}{detail.effort ? ` · ${detail.effort}` : ""}</p></div>
			<span className={cn("shrink-0 font-mono text-mono-sm", agentStatusClass(detail.status))}>{AGENT_GLYPH[detail.status]} {detail.status}</span>
		</header>
		<Lineage sessionId={sessionId} detail={detail} roster={projection.subagents} />
		{detail.prompt ? <section className="border-b border-hairline px-3 py-2"><p className="text-meta text-ink-dim">Parent request</p><p className="max-h-28 overflow-y-auto whitespace-pre-wrap break-words text-body-sm">{detail.prompt}</p></section> : null}
		{detail.transcript.length === 0 && detail.status !== "running" ? <div className="flex flex-1 items-center justify-center px-8 text-center"><p className="text-body-sm text-ink-dim">no agent messages recorded</p></div> : <Transcript pid={sessionId} jobId={jobId} entries={detail.transcript} />}
		{detail.status === "running" ? <WorkingLine activity={detail.activity || detail.progress || "thinking"} startedS={detail.elapsed_s} /> : null}
		{detail.status === "completed" && detail.result_text ? <section aria-live="polite" className="max-h-32 overflow-y-auto border-t border-hairline bg-elevated px-3 py-2"><p className="text-meta text-success">Handoff</p><p className="whitespace-pre-wrap break-words text-body-sm">{detail.result_text}</p></section> : null}
		{detail.status === "failed" && detail.error_text ? <section role="alert" className="max-h-32 overflow-y-auto border-t border-danger-border bg-danger-wash px-3 py-2"><p className="text-meta text-danger">Failure</p><p className="whitespace-pre-wrap break-words text-body-sm text-danger">{detail.error_text}</p></section> : null}
		{detail.todos.some((phase) => phase.items.length > 0) ? <TodosPanel todos={detail.todos} /> : null}
		{directChildren.length > 0 ? <AgentRoster sessionId={sessionId} subagents={projection.subagents} parentJobId={jobId} /> : null}
		<footer className="border-t border-hairline px-3 py-2 pb-[max(env(safe-area-inset-bottom),0.5rem)] text-center text-meta text-ink-dim">read-only · send commands from the root conversation</footer>
	</>;
}
