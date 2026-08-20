/**
 * Todos panel: collapsible, one line per item. Default-collapsed once most
 * items are done — a finished list is context, not work in progress.
 */
import { cn } from "../../lib/cn";
import type { TodoItem } from "../../types";
import { Disclosure } from "../ui/disclosure";

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
	const defaultOpen = !(done > 4 && done >= todos.length - 1);
	return (
		<Disclosure
			defaultOpen={defaultOpen}
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
			<div className="flex flex-col gap-1 pb-2 pl-5">
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
