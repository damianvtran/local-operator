import { useMemo } from "react";
import { navigate } from "../router";
import type { SubagentDetail, SubagentRow } from "../types";
import { AgentRow } from "./subagents-panel";
import { Sheet } from "./ui/sheet";

export interface AgentGroups {
	path: SubagentRow[];
	peers: SubagentRow[];
	children: SubagentRow[];
}

/** Resolve hierarchy groups once so the app bar stays constant at every depth.
 * Missing retained rows are skipped because the selected detail remains useful
 * while the bounded roster cache catches up after a reconnect. */
export function groupAgents(
	detail: SubagentDetail,
	roster: SubagentRow[],
): AgentGroups {
	const byId = new Map(roster.map((row) => [row.job_id, row]));
	const pathIds = [...detail.ancestor_ids];
	if (detail.parent_job_id && !pathIds.includes(detail.parent_job_id)) {
		pathIds.push(detail.parent_job_id);
	}
	return {
		path: pathIds.map((id) => byId.get(id)).filter(Boolean) as SubagentRow[],
		peers: detail.peer_ids
			.map((id) => byId.get(id))
			.filter(Boolean) as SubagentRow[],
		children: roster.filter((row) => row.parent_job_id === detail.job_id),
	};
}

function RootRow({ sessionId, onNavigate }: { sessionId: string; onNavigate: () => void }) {
	return (
		<button
			type="button"
			onClick={() => {
				onNavigate();
				navigate(`/s/${encodeURIComponent(sessionId)}`);
			}}
			className="flex min-h-11 w-full items-center gap-2 rounded-sm px-1 text-left active:bg-surface"
		>
			<span aria-hidden className="w-4 shrink-0 text-center text-ink-dim">⌂</span>
			<span className="min-w-0 flex-1 truncate text-body-sm text-ink">Root conversation</span>
		</button>
	);
}

function Group({
	label,
	sessionId,
	agents,
	onNavigate,
}: {
	label: string;
	sessionId: string;
	agents: SubagentRow[];
	onNavigate: () => void;
}) {
	if (agents.length === 0) return null;
	return (
		<section className="border-t border-hairline px-3 py-2 first:border-t-0">
			<h3 className="mb-1 text-meta font-medium text-ink-dim">{label}</h3>
			{agents.map((agent) => (
				<AgentRow
					key={agent.job_id}
					sessionId={sessionId}
					agent={agent}
					onNavigate={onNavigate}
					showMetadata
				/>
			))}
		</section>
	);
}

export function AgentsSheet({
	open,
	onClose,
	sessionId,
	detail,
	roster,
}: {
	open: boolean;
	onClose: () => void;
	sessionId: string;
	detail: SubagentDetail;
	roster: SubagentRow[];
}) {
	const groups = useMemo(() => groupAgents(detail, roster), [detail, roster]);
	return (
		<Sheet open={open} onClose={onClose} title="Agents">
			<section className="px-3 pb-2">
				<h3 className="mb-1 text-meta font-medium text-ink-dim">Path</h3>
				<RootRow sessionId={sessionId} onNavigate={onClose} />
				{groups.path.map((agent) => (
					<AgentRow
						key={agent.job_id}
						sessionId={sessionId}
						agent={agent}
						onNavigate={onClose}
						showMetadata
					/>
				))}
				<div className="flex min-h-11 items-center gap-2 px-1" aria-current="page">
					<span aria-hidden className="w-4 shrink-0 text-center text-accent">●</span>
					<span className="min-w-0 flex-1 truncate text-body-sm font-medium text-ink">
						{detail.label}
					</span>
					<span className="text-meta text-ink-dim">current</span>
				</div>
			</section>
			<Group label="Peers" sessionId={sessionId} agents={groups.peers} onNavigate={onClose} />
			<Group label="Children" sessionId={sessionId} agents={groups.children} onNavigate={onClose} />
		</Sheet>
	);
}
