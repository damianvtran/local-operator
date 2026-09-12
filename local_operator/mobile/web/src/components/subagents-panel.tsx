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
 *
 * The header carries FAILURES as well as the running count, which the glyphs
 * used to carry on their own. Collapsing by default is right, but it hid the
 * one status a user must not miss: 3 of 22 agents failed rendered exactly like
 * a healthy fan-out, `subagents 1/22 running`, with no ✗ anywhere on screen and
 * the word "failed" absent from the page (U5). `· 3 failed` in `text-danger`
 * restores that at a glance and costs no vertical space.
 */
import { useRef } from "react";
import { cn } from "../lib/cn";
import { PANEL_FRACTION, columnCap } from "../lib/column";
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
	embedded = false,
	label = "subagents",
	forceCollapsed = false,
}: {
	sessionId: string;
	subagents: SubagentRow[];
	parentJobId: string | null;
	embedded?: boolean;
	label?: string;
	/** Collapse and refuse to expand while something needs a decision — see
	    the session view's panel-budget comment (D1). */
	forceCollapsed?: boolean;
}) {
	const direct = subagents.filter((agent) => agent.parent_job_id === parentJobId);
	/* The body's own scroll offset, kept OUTSIDE the collapsed body. `Disclosure`
	   unmounts its children, so a user who scrolled to row 21, collapsed the
	   panel to read the conversation and re-expanded was returned to row 1 with
	   22 rows to re-scroll (U6). A ref, not state: restoring it must not repaint,
	   and it is per mounted roster rather than per session, which is the right
	   lifetime — a route change should not resurrect a stale offset. */
	const scrollTopRef = useRef(0);
	const bodyRef = useRef<HTMLDivElement>(null);
	if (direct.length === 0) return null;
	const running = direct.filter((agent) => agent.status === "running").length;
	const failed = direct.filter((agent) => agent.status === "failed").length;
	const rows = (
		<div className="flex w-full flex-col gap-1 pb-2">
			{direct.map((agent) => (
				<AgentRow key={agent.job_id} sessionId={sessionId} agent={agent} />
			))}
		</div>
	);
	return (
		<Disclosure
			/* Collapsed by default, including when agents are running: see the
			   module docstring. `running > 0` opened the roster on arrival for
			   every coordinating session. */
			defaultOpen={false}
			forceClosed={forceCollapsed}
			className={cn(
				"border-t border-hairline",
				/* `min-h-0` so this panel can give space back to the column
				   instead of pushing a sibling past its clipped foot (D1). */
				"min-h-0",
				embedded ? "pt-1" : "px-3",
			)}
			header={
				<span className="text-body-sm text-ink-muted">
				{label}{" "}
					<span className="font-mono text-mono-sm text-ink-dim">
						{running}/{direct.length} running
					</span>
					{failed > 0 ? (
						<span className="font-mono text-mono-sm text-danger">
							{" "}· {failed} failed
						</span>
					) : null}
				</span>
			}
		>
			{/* Capped to a fraction of the COLUMN and scrolls internally, the same
			   bound the todos panel uses: a 22-row roster can never crowd out the
			   messages, even fully expanded. Column units rather than `dvh` — see
			   `lib/column.ts`; the two diverge while the keyboard is open, and a cap
			   that does not tighten with the column is not a cap. */}
			<div
				ref={(el) => {
					bodyRef.current = el;
					if (el) el.scrollTop = scrollTopRef.current;
				}}
				onScroll={() => {
					scrollTopRef.current = bodyRef.current?.scrollTop ?? 0;
				}}
				style={columnCap(PANEL_FRACTION)}
				className="lo-scroll overflow-y-auto"
			>
				{rows}
			</div>
		</Disclosure>
	);
}

/** Legacy export retained for callers compiled against the sheet-era name. */
export function SubagentsPanel({
	subagents,
	pid = "",
	forceCollapsed = false,
}: {
	subagents: SubagentRow[];
	pid?: string;
	forceCollapsed?: boolean;
}) {
	return (
		<AgentRoster
			sessionId={pid}
			subagents={subagents}
			parentJobId={null}
			forceCollapsed={forceCollapsed}
		/>
	);
}
