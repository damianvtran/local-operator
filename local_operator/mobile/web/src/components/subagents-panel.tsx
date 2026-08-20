/**
 * Subagents panel: collapsible roster with an aggregate header. Tapping a
 * row opens the detail sheet — everything known about that subagent, with
 * the sheet's own close as the back affordance to the parent session.
 */
import { useState } from "react";
import { cn } from "../lib/cn";
import { formatElapsed } from "../lib/format";
import type { SubagentRow } from "../types";
import { Disclosure } from "./ui/disclosure";
import { Sheet } from "./ui/sheet";

const GLYPH: Record<SubagentRow["status"], string> = {
	running: "⟳",
	completed: "✓",
	failed: "✗",
	cancelled: "–",
	parked: "‖",
};

function statusClass(status: SubagentRow["status"]): string {
	switch (status) {
		case "running":
			return "lo-pulse text-accent";
		case "completed":
			return "text-success";
		case "failed":
			return "text-danger";
		default:
			return "text-ink-dim";
	}
}

export function SubagentsPanel({ subagents }: { subagents: SubagentRow[] }) {
	const [selected, setSelected] = useState<SubagentRow | null>(null);
	const running = subagents.filter((s) => s.status === "running").length;
	return (
		<>
			<Disclosure
				defaultOpen={running > 0}
				className="border-t border-hairline px-4"
				header={
					<span className="text-body-sm text-ink-muted">
						subagents{" "}
						<span className="font-mono text-mono-sm text-ink-dim">
							{running}/{subagents.length} running
						</span>
					</span>
				}
			>
				<div className="flex flex-col gap-1 pb-2 pl-5">
					{subagents.map((s) => (
						<button
							key={s.job_id}
							type="button"
							onClick={() => setSelected(s)}
							className="flex min-h-11 w-full flex-col justify-center rounded-sm px-1 text-left active:bg-elevated"
						>
							<span className="flex w-full items-center gap-2">
								<span
									className={cn(
										"w-4 shrink-0 text-center font-mono text-mono-sm",
										statusClass(s.status),
									)}
									aria-hidden
								>
									{GLYPH[s.status]}
								</span>
								<span className="min-w-0 flex-1 truncate text-body-sm text-ink">
									{s.label}
								</span>
								{s.model_label ? (
									<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
										{s.model_label}
									</span>
								) : null}
								{s.elapsed_s > 0 ? (
									<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
										{formatElapsed(s.elapsed_s)}
									</span>
								) : null}
							</span>
							{s.status === "running" && s.progress ? (
								<span className="block truncate pl-6 text-body-sm text-ink-dim">
									{s.progress}
								</span>
							) : null}
							{s.status !== "running" && s.result_text ? (
								<span className="block truncate pl-6 text-body-sm text-ink-dim">
									{s.result_text}
								</span>
							) : null}
						</button>
					))}
				</div>
			</Disclosure>

			<Sheet
				open={selected !== null}
				onClose={() => setSelected(null)}
				title={selected?.label || "subagent"}
			>
				{selected ? (
					<div className="flex flex-col gap-3 p-4">
						<div className="flex items-center gap-2">
							<span
								className={cn(
									"font-mono text-mono",
									statusClass(selected.status),
								)}
							>
								{GLYPH[selected.status]} {selected.status}
							</span>
							<span className="font-mono text-mono-sm text-ink-dim">
								{selected.agent}
							</span>
							{selected.elapsed_s > 0 ? (
								<span className="font-mono text-mono-sm text-ink-dim">
									{formatElapsed(selected.elapsed_s)}
								</span>
							) : null}
						</div>
						{selected.model_label ? (
							<p className="text-body-sm text-ink-muted">
								model{" "}
								<span className="font-mono text-mono-sm">
									{selected.model_label}
								</span>
							</p>
						) : null}
						{selected.progress ? (
							<div className="flex flex-col gap-1">
								<span className="text-meta text-ink-dim">
									progress
								</span>
								<p className="text-body-sm text-ink-muted whitespace-pre-wrap">
									{selected.progress}
								</p>
							</div>
						) : null}
						{selected.result_text ? (
							<div className="flex flex-col gap-1">
								<span className="text-meta text-ink-dim">
									result
								</span>
								<p className="text-body-sm text-ink whitespace-pre-wrap">
									{selected.result_text}
								</p>
							</div>
						) : null}
						{selected.error_text ? (
							<div className="flex flex-col gap-1">
								<span className="text-meta text-ink-dim">
									error
								</span>
								<p className="text-body-sm text-danger whitespace-pre-wrap">
									{selected.error_text}
								</p>
							</div>
						) : null}
					</div>
				) : null}
			</Sheet>
		</>
	);
}
