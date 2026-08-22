/**
 * Todos panel: collapsible, phase-grouped, one line per item.
 *
 * On a phone the panel sits ABOVE the transcript, so a long list expanded by
 * default pushed the actual conversation off the top of the screen (the
 * reported "todos pop up very high"). Two guards fix that: it starts
 * COLLAPSED (the count in the header is enough at a glance; tap to see the
 * items), and when expanded its body is capped to ~40% of the viewport and
 * scrolls internally, so even a 20-item list can never crowd out the
 * messages.
 *
 * The store is PHASED (mirrors the TUI's phase model): items are grouped under
 * named phases rendered as a muted `name · done/total` header with the phase's
 * items INDENTED beneath it. Back-compat: a single implicit `"Todos"` phase —
 * how a flat/legacy list is carried — renders HEADERLESS and identical to the
 * pre-phase flat list, matching the TUI's `_IMPLICIT_PHASE` rule.
 */
import { cn } from "../lib/cn";
import type { TodoItem, TodoPhase } from "../types";
import { Disclosure } from "./ui/disclosure";

const GLYPH: Record<TodoItem["status"], string> = {
	pending: "☐",
	done: "☑",
	blocked: "~",
	dropped: "-",
};

/** Keep in sync with the server's `IMPLICIT_TODO_PHASE` / the tool's
    `_IMPLICIT_PHASE`: a lone phase with this exact name is the flat-list
    carrier and renders without a header. */
const IMPLICIT_TODO_PHASE = "Todos";

const CLOSED = new Set(["done", "dropped"]);

function TodoRow({ t }: { t: TodoItem }) {
	return (
		<div className="flex items-baseline gap-2">
			<span
				className={cn(
					"shrink-0 font-mono text-mono-sm",
					t.status === "done" && "text-success",
					t.status === "blocked" && "text-warning",
					(t.status === "pending" || t.status === "dropped") &&
						"text-ink-dim",
				)}
				aria-hidden
			>
				{GLYPH[t.status]}
			</span>
			<span
				className={cn(
					"min-w-0 text-body-sm",
					t.status === "done" && "text-ink-dim",
					t.status === "dropped" && "text-ink-dim line-through",
					t.status === "pending" && "text-ink",
					t.status === "blocked" && "text-ink-muted",
				)}
			>
				{t.text}
				{t.status === "blocked" && t.reason ? (
					<span className="text-warning">
						{" "}
						— {t.reason}
					</span>
				) : null}
			</span>
		</div>
	);
}

export function TodosPanel({ todos }: { todos: TodoPhase[] }) {
	const items = todos.flatMap((p) => p.items);
	const done = items.filter((t) => CLOSED.has(t.status)).length;
	// Headerless flat-list case: exactly one phase, and it is the implicit
	// carrier. Anything else (a named phase, or several phases) shows headers.
	const headerless =
		todos.length === 1 && todos[0].name === IMPLICIT_TODO_PHASE;
	return (
		<Disclosure
			/* Collapsed by default on the phone: the panel is above the
			   transcript, and an auto-expanded list pushed the conversation off
			   screen. The header's count is the at-a-glance signal; tap to work
			   the list. */
			defaultOpen={false}
			className="border-t border-hairline px-4"
			header={
				<span className="text-body-sm text-ink-muted">
					tasks{" "}
					<span className="font-mono text-mono-sm text-ink-dim">
						{done}/{items.length}
					</span>
				</span>
			}
		>
			{/* Capped to ~40% of the viewport and scrolls internally: a long
			   list can never crowd out the messages, even fully expanded. */}
			<div className="lo-scroll flex max-h-[40dvh] flex-col gap-1 overflow-y-auto pb-2">
				{headerless
					? todos[0].items.map((t, i) => (
							/* pl-5 aligns items with the panel gutter, same as
							   the pre-phase flat list. */
							<div key={i} className="pl-5">
								<TodoRow t={t} />
							</div>
						))
					: todos.map((phase, pi) => {
							const phaseDone = phase.items.filter((t) =>
								CLOSED.has(t.status),
							).length;
							return (
								<div key={pi} className="flex flex-col gap-1">
									{/* Phase header aligns with the disclosure's
									   own `tasks n/n` summary text column, not the
									   panel's left edge (D2). The summary sits
									   after the chevron (`w-4`) + `gap-1` = 20px,
									   i.e. `pl-5`; at `pl-1` the header poked out
									   LEFT of the control that owns it while items
									   aligned under the summary, so the group read
									   as floating outside the panel. Header at
									   `pl-5` nests it under the summary; items sit
									   one step deeper (`pl-7`) so they read as
									   belonging to the phase, not as siblings of
									   its header. */}
									<div className="pl-5 pt-1 text-body-sm text-ink-muted">
										{phase.name}{" "}
										<span className="font-mono text-mono-sm text-ink-dim">
											{phaseDone}/{phase.items.length}
										</span>
									</div>
									{phase.items.map((t, i) => (
										<div key={i} className="pl-7">
											<TodoRow t={t} />
										</div>
									))}
								</div>
							);
						})}
			</div>
		</Disclosure>
	);
}
