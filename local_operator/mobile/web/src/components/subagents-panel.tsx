import { cn } from "../lib/cn";
import { formatElapsed } from "../lib/format";
import { navigate } from "../router";
import type { SubagentRow } from "../types";
import { Disclosure } from "./ui/disclosure";

export const AGENT_GLYPH: Record<SubagentRow["status"], string> = {
	running: "⟳",
	completed: "✓",
	failed: "✗",
	cancelled: "–",
	parked: "‖",
};

export function agentStatusClass(status: SubagentRow["status"]): string {
	if (status === "running") return "lo-pulse text-accent";
	if (status === "completed") return "text-success";
	if (status === "failed") return "text-danger";
	return "text-ink-dim";
}

/** One touch-safe row shared by the root roster and every descendant roster. */
export function AgentRow({
	sessionId,
	agent,
}: {
	sessionId: string;
	agent: SubagentRow;
}) {
	return (
		<button
			type="button"
			onClick={() =>
				navigate(
					`/s/${encodeURIComponent(sessionId)}/a/${encodeURIComponent(agent.job_id)}`,
				)
			}
			className="flex min-h-11 w-full items-center gap-2 rounded-sm px-1 text-left active:bg-elevated"
		>
			<span
				className={cn(
					"w-4 shrink-0 text-center font-mono text-mono-sm",
					agentStatusClass(agent.status),
				)}
			>
				{AGENT_GLYPH[agent.status]}
			</span>
			<span className="min-w-0 flex-1 truncate text-body-sm text-ink">
				{agent.label}
			</span>
			<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
				{agent.elapsed_s > 0 ? formatElapsed(agent.elapsed_s) : ""}
			</span>
		</button>
	);
}

export function AgentRoster({
	sessionId,
	subagents,
	parentJobId,
	collapsible = false,
}: {
	sessionId: string;
	subagents: SubagentRow[];
	parentJobId: string | null;
	collapsible?: boolean;
}) {
	const direct = subagents.filter((agent) => agent.parent_job_id === parentJobId);
	if (direct.length === 0) return null;
	const running = direct.filter((agent) => agent.status === "running").length;
	const rows = (
		<div className="flex w-full flex-col gap-1 pb-2">
			{direct.map((agent) => (
				<AgentRow key={agent.job_id} sessionId={sessionId} agent={agent} />
			))}
		</div>
	);
	if (!collapsible) return <section className="border-t border-hairline px-3">{rows}</section>;
	return (
		<Disclosure
			defaultOpen={running > 0}
			className="border-t border-hairline px-3"
			header={
				<span className="text-body-sm text-ink-muted">
				subagents{" "}
					<span className="font-mono text-mono-sm text-ink-dim">
						{running}/{direct.length} running
					</span>
				</span>
			}
		>
			{rows}
		</Disclosure>
	);
}

/** Legacy export retained for callers compiled against the sheet-era name. */
export function SubagentsPanel({
	subagents,
	pid = "",
}: {
	subagents: SubagentRow[];
	pid?: string;
}) {
	return (
		<AgentRoster
			sessionId={pid}
			subagents={subagents}
			parentJobId={null}
			collapsible
		/>
	);
}
