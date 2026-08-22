/**
 * Todos panel: collapsible, one line per item.
 *
 * On a phone the panel sits ABOVE the transcript, so a long list expanded by
 * default pushed the actual conversation off the top of the screen (the
 * reported "todos pop up very high"). Two guards fix that: it starts
 * COLLAPSED (the count in the header is enough at a glance; tap to see the
 * items), and when expanded its body is capped to ~40% of the viewport and
 * scrolls internally, so even a 20-item list can never crowd out the
 * messages.
 */
import { cn } from "../lib/cn";
import type { TodoItem } from "../types";
import { Disclosure } from "./ui/disclosure";

const GLYPH: Record<TodoItem["status"], string> = {
	pending: "☐",
	done: "☑",
	blocked: "~",
	dropped: "-",
};

export function TodosPanel({ todos }: { todos: TodoItem[] }) {
	const done = todos.filter(
		(t) => t.status === "done" || t.status === "dropped",
	).length;
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
						{done}/{todos.length}
					</span>
				</span>
			}
		>
			{/* Capped to ~40% of the viewport and scrolls internally: a long
			   list can never crowd out the messages, even fully expanded. */}
			<div className="lo-scroll flex max-h-[40dvh] flex-col gap-1 overflow-y-auto pb-2 pl-5">
				{todos.map((t, i) => (
					<div key={i} className="flex items-baseline gap-2">
						<span
							className={cn(
								"shrink-0 font-mono text-mono-sm",
								t.status === "done" && "text-success",
								t.status === "blocked" && "text-warning",
								(t.status === "pending" ||
									t.status === "dropped") &&
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
								t.status === "dropped" &&
									"text-ink-dim line-through",
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
				))}
			</div>
		</Disclosure>
	);
}
