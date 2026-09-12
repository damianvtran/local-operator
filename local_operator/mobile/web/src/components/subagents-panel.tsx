/**
 * Subagent roster: collapsible, one line per agent, shared by the session
 * screen's root roster and every descendant roster on the agent screen.
 *
 * Same phone constraint the todos panel documents, and the same two guards.
 * The roster sits ABOVE the transcript in the session column, so a roster
 * expanded on arrival pushes the conversation off the top: a real session with
 * `subagents 1/22 running` painted 22 rows and left the transcript a 16px
 * sliver with the composer off screen entirely. It therefore starts COLLAPSED
 * (the header's `n/m running` count is the at-a-glance signal; tap to work the
 * list) and its expanded body is capped to ~40% of the viewport with internal
 * scrolling, so even a long fan-out can never crowd out the messages.
 *
 * The previous default was `running > 0`, which made "a subagent is working"
 * — the normal state of a coordinating session — the trigger for covering the
 * screen. The count in the header carries that signal without the cost.
 */
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
	onNavigate,
	showMetadata = false,
}: {
	sessionId: string;
	agent: SubagentRow;
	onNavigate?: () => void;
	showMetadata?: boolean;
}) {
	return (
		<button
			type="button"
			onClick={() => {
				onNavigate?.();
					navigate(
					`/s/${encodeURIComponent(sessionId)}/a/${encodeURIComponent(agent.job_id)}`,
				);
			}}
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
			<span className="min-w-0 flex flex-1 flex-col">
				<span className="truncate text-body-sm text-ink">{agent.label}</span>
				{showMetadata ? (
					<span className="truncate text-meta text-ink-dim">
						{agent.agent}{agent.effort ? ` · ${agent.effort}` : ""}
					</span>
				) : null}
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
	embedded = false,
	label = "subagents",
}: {
	sessionId: string;
	subagents: SubagentRow[];
	parentJobId: string | null;
	collapsible?: boolean;
	embedded?: boolean;
	label?: string;
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
	/* The non-collapsible roster is always the whole point of the surface
	   hosting it, so it is left uncapped: a nested scroller inside a page that
	   already scrolls is the two-competing-patterns bug the Disclosure docstring
	   warns about. The cap belongs where the roster is a SECONDARY panel above a
	   transcript, which is exactly the collapsible case below. */
	if (!collapsible) return <section className={cn("border-t border-hairline", embedded ? "pt-1" : "px-3")}>{rows}</section>;
	return (
		<Disclosure
			/* Collapsed by default, including when agents are running: see the
			   module docstring. `running > 0` opened the roster on arrival for
			   every coordinating session. */
			defaultOpen={false}
			className="border-t border-hairline px-3"
			header={
				<span className="text-body-sm text-ink-muted">
				{label}{" "}
					<span className="font-mono text-mono-sm text-ink-dim">
						{running}/{direct.length} running
					</span>
				</span>
			}
		>
			{/* Capped to ~40% of the viewport and scrolls internally, the same
			   bound the todos panel uses: a 22-row roster can never crowd out the
			   messages, even fully expanded. */}
			<div className="lo-scroll max-h-[40dvh] overflow-y-auto">{rows}</div>
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
