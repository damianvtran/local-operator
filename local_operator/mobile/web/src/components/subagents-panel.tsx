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
import { Disclosure, HELD_DIM } from "./ui/disclosure";

export const AGENT_GLYPH: Record<SubagentRow["status"], string> = {
	running: "⟳",
	completed: "✓",
	failed: "✗",
	cancelled: "–",
	parked: "‖",
	/* `…` and NOT the spinner: a child parked waiting for a free slot is not
	   spending, so the whole treatment is the non-spinning dim gutter
	   (`agentStatusClass` falls through to `text-ink-dim`). The ellipsis is
	   chosen for coverage — it reads as "waiting" in every font on every
	   phone, where `⏸` and `⋯` render as tofu on some. */
	queued: "…",
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
				{agent.model_fallback === true ? (
					/* The pin-integrity badge (PR review round 1, reviewer MAJOR): the wire
					   has carried `model_label` as the `A → B ⚠ fallback` string since the
					   projection was written, and no phone component painted it — a
					   substituted child read exactly like a never-pinned one. Directly
					   under the label in warning ink: the string itself carries the `⚠`
					   and the word `fallback`, so the signal survives grayscale and
					   NO_COLOR alike, matching the notice treatment. `break-words` rather
					   than `truncate`: the tail of this string IS the alarm, and an
					   ellipsis that eats `⚠ fallback` would leave the two model names
					   with nothing saying they disagree. Only children OFF their pin grow
					   this line, so a healthy roster spends no vertical space on it.
					   `=== true`, not truthy: a payload from a runtime that predates the
					   field must read as `no substitution`. */
					<span className="break-words text-meta text-warning">{agent.model_label}</span>
				) : null}
				{showMetadata ? (
					<span className="truncate text-meta text-ink-dim">
						{agent.agent}{agent.effort ? ` · ${agent.effort}` : ""}
					</span>
				) : null}
			</span>
			<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
				{/* `null` is "this roster has no age for the child" and contributes
				 * nothing, exactly as a zero- or sub-second age does in this compact
				 * list; the drill-in is where an age that IS known is shown from the
				 * first frame, including a known `0s`. */}
				{agent.elapsed_s !== null && agent.elapsed_s > 0
					? formatElapsed(agent.elapsed_s)
					: ""}
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
	const queued = direct.filter((agent) => agent.status === "queued").length;
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
			/* THE HINT STANDS DOWN below the width the row needs to keep its label
			   READABLE (mobile UX batch 2, 320x568; retuned in round 1: design D2,
			   D6, agent-review NIT 2). This row's fixed parts — chevron 16 + gaps
			   12 + `1/5 running` 79.5 + `· 1 queued` 72.3 + `· 1 failed` 72.3 +
			   the hint 72.9 — are ~330px, and the row is viewport - 24, so with
			   the hint shown the LABEL takes what is left: measured at 360 it was
			   6px wide (`s`), i.e. the middle phone read worse than the 320 one,
			   where the old 352px stand-down kept 41px. The threshold is now 385:
			   hidden below it, shown at it and above — measured with this class,
			   the label is 30px at 385 (hint on), 63px at 384 (hint hidden), 41px
			   at 320; and 390 keeps both. The tasks row carries the SAME
			   threshold even though its own phrase fits everywhere — the design
			   round flagged the two adjacent held-shut rows disagreeing at 320
			   (one hint, one none), and one rule for the pair is what makes them
			   read as one state; see `todos-panel.tsx`. */
			hintClassName="max-[385px]:hidden"
			className={cn(
				"border-t border-hairline",
				/* `min-h-11`, not `min-h-0` — see the todos panel's fuller note:
				   the collapsed header is the floor, because a container squeezed
				   below its 44px row made the row overflow onto the pending card
				   at 320x568 (mobile UX batch 2, D1). The card yields instead. */
				"min-h-11",
				embedded ? "pt-1" : "px-3",
			)}
			header={
				/* A FLEX line, not inline text, so the row has an explicit order of
				   who yields first. Held shut, this row carries three claims on one
				   44px line — label + running count, the failure count, and the
				   `· answer first` hint — and at 360px wide (a supported viewport)
				   they add up to within ~6px of the width. Something has to give. As
				   inline text nothing could: `truncate` needs a block box to clip, so
				   the label just wrapped, and the LAST inline content — the danger
				   count — was what broke across two lines at a two-digit count
				   (`· 10 failed`), the one glyph U5 and D4 exist to protect. Flex lets
				   the label absorb the pressure instead.

				   `relative` restores the PAINT ORDER that inline text got for free.
				   Below the viewport ladder the column is height-starved, this panel's
				   container collapses to ~12px and the pending card — a LATER sibling —
				   overlaps the row. As inline content the header always drew above that
				   card's background, because CSS paints in-flow block backgrounds before
				   inline content. Flex blockifies these children and moves them into the
				   block phase, where tree order decides and the later card wins:
				   measured at 320x568, the count went to 0 painted danger-red pixels
				   while keeping its box. Positioning lifts the row back above in-flow
				   block backgrounds; no z-index, since paint phase is the whole
				   problem. */
				<span className="relative flex min-w-0 items-baseline gap-1 text-body-sm text-ink-muted">
					{/* The held-shut dim is applied per PART, and the failure count
					    is deliberately a SIBLING of the dimmed span rather than a
					    child of it: opacity composites the whole subtree, so a dim
					    any higher takes the count with it — 7.08:1 down to 3.30:1,
					    measured from the painted frame (design D4). The label and
					    running count may fade, because a pending card already
					    implies the roster is held; a failed fan-out may not, and it
					    matters most while a decision is waiting.

					    `min-w-0 truncate` makes the LABEL the span that YIELDS: it is
					    the only one here that degrades gracefully, because a clipped
					    label still reads and the tail it loses is recoverable by
					    opening the panel. This is the mechanism the hint's `shrink-0`
					    in `disclosure.tsx` already assumes exists.

					    D6 (mobile UX batch): the label yields ON ITS OWN. It used to
					    share one truncating span with the count, which cut the count
					    group mid-phrase at 390 — "subagents 1/5 r…", an unreadable
					    fragment. The count is now a `shrink-0` sibling, so it survives
					    WHOLE (worst case "sub… 1/5 running"); a clipped LABEL is the
					    loss this row can afford, and it stays recoverable by opening
					    the panel. */}
					<span className={cn("min-w-0 truncate", forceCollapsed && HELD_DIM)}>
						{label}
					</span>
					<span
						className={cn(
							"shrink-0 whitespace-nowrap font-mono text-mono-sm text-ink-dim",
							forceCollapsed && HELD_DIM,
						)}
					>
						{running}/{direct.length} running
					</span>
					{queued > 0 ? (
						/* The same `shrink-0 whitespace-nowrap` rule as the failure count, and
						   for the same reason: the number must not break between `3` and
						   `queued`, and the label is the span that yields. It is here because a
						   capacity-parked child is deliberately NOT in the `running`
						   numerator — without this the header would just read a smaller
						   fraction and never say why. */
						<span className="shrink-0 whitespace-nowrap font-mono text-mono-sm text-ink-dim">
							· {queued} queued
						</span>
					) : null}
					{failed > 0 ? (
						/* `shrink-0 whitespace-nowrap`: the count is the row's least
						   expendable token, so it neither shrinks nor breaks between
						   `10` and `failed`. Undimmed at 7.08:1 per D4. */
						<span className="shrink-0 font-mono text-mono-sm whitespace-nowrap text-danger">
							· {failed} failed
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
