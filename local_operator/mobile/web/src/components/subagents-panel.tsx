import { useEffect, useMemo, useState } from "react";
import { cn } from "../lib/cn";
import { formatElapsed } from "../lib/format";
import type { SubagentRow } from "../types";
import { Disclosure } from "./ui/disclosure";
import { Sheet } from "./ui/sheet";
import { TodosPanel } from "./todos-panel";
import { Transcript } from "./transcript";
import { WorkingLine } from "./working-line";

const GLYPH: Record<SubagentRow["status"], string> = {
	running: "⟳", completed: "✓", failed: "✗", cancelled: "–", parked: "‖",
};
function statusClass(status: SubagentRow["status"]): string {
	if (status === "running") return "lo-pulse text-accent";
	if (status === "completed") return "text-success";
	if (status === "failed") return "text-danger";
	return "text-ink-dim";
}

export function SubagentsPanel({ subagents, pid = "" }: { subagents: SubagentRow[]; pid?: string }) {
	const [selectedId, setSelectedId] = useState<string | null>(null);
	const byId = useMemo(() => new Map(subagents.map((row) => [row.job_id, row])), [subagents]);
	const selected = selectedId ? byId.get(selectedId) ?? null : null;
	useEffect(() => { if (selectedId && !byId.has(selectedId)) setSelectedId(null); }, [byId, selectedId]);
	const running = subagents.filter((s) => s.status === "running").length;
	const relatives = (ids: string[]) => ids.map((id) => byId.get(id)).filter((row): row is SubagentRow => Boolean(row));
	// These controls are the phone's only path through recursive lineage. Keep
	// each target at the same 44 px minimum as the sheet close button so wrapped
	// peer/child rows cannot turn a small miss into opening the wrong transcript.
	const hierarchyControl = "inline-flex min-h-11 items-center rounded-sm border border-hairline px-3 py-2 text-left text-meta active:bg-elevated";
	return <>
		<Disclosure defaultOpen={running > 0} className="border-t border-hairline px-4" header={<span className="text-body-sm text-ink-muted">subagents <span className="font-mono text-mono-sm text-ink-dim">{running}/{subagents.length} running</span></span>}>
			<div className="flex flex-col gap-1 pb-2 pl-5">{subagents.filter((s) => !s.parent_job_id).map((s) => <button key={s.job_id} type="button" onClick={() => setSelectedId(s.job_id)} className="flex min-h-11 w-full items-center gap-2 rounded-sm px-1 text-left active:bg-elevated"><span className={cn("w-4 text-center font-mono text-mono-sm", statusClass(s.status))}>{GLYPH[s.status]}</span><span className="min-w-0 flex-1 truncate text-body-sm text-ink">{s.label}</span><span className="font-mono text-mono-sm text-ink-dim">{s.elapsed_s > 0 ? formatElapsed(s.elapsed_s) : ""}</span></button>)}</div>
		</Disclosure>
		<Sheet open={selected !== null} onClose={() => setSelectedId(null)} title={selected?.label || "subagent"}>
			{selected ? <div className="flex h-full min-h-0 flex-col">
				<div className="border-b border-hairline px-4 py-2">
					<p className="truncate text-meta text-ink-dim">Conversation {selected.ancestors.map((label) => `› ${label} `)}› {selected.label}</p>
					<div className="mt-1 flex items-center gap-2"><span className={cn("font-mono text-mono-sm", statusClass(selected.status))}>{GLYPH[selected.status]} {selected.status}</span><span className="font-mono text-mono-sm text-ink-dim">{selected.agent}{selected.effort ? ` · ${selected.effort}` : ""}</span></div>
					{selected.status === "failed" && selected.error_text ? <div role="alert" aria-live="assertive" className="mt-2 rounded-sm border border-danger-border bg-danger-wash px-3 py-2"><p className="text-meta text-danger">Failure</p><p className="whitespace-pre-wrap text-body-sm text-danger">{selected.error_text}</p></div> : null}
					{selected.status === "completed" && selected.result_text ? <div aria-live="polite" className="mt-2 rounded-sm border border-hairline bg-elevated px-3 py-2"><p className="text-meta text-success">Result</p><p className="whitespace-pre-wrap text-body-sm text-ink">{selected.result_text}</p></div> : null}
					<div className="mt-3 flex flex-wrap gap-2">
						<button type="button" className={hierarchyControl} onClick={() => setSelectedId(null)}>root</button>
						{selected.parent_job_id && byId.get(selected.parent_job_id) ? <button type="button" className={hierarchyControl} onClick={() => setSelectedId(selected.parent_job_id)}>parent</button> : null}
						{relatives(selected.peer_ids).map((row) => <button key={row.job_id} type="button" className={hierarchyControl} onClick={() => setSelectedId(row.job_id)}>peer · {row.label}</button>)}
						{relatives(selected.child_ids).map((row) => <button key={row.job_id} type="button" className={hierarchyControl} onClick={() => setSelectedId(row.job_id)}>child · {row.label}</button>)}
					</div>
				</div>
				{selected.prompt ? <div className="border-b border-hairline px-4 py-2"><span className="text-meta text-ink-dim">Parent request</span><p className="text-body-sm whitespace-pre-wrap">{selected.prompt}</p></div> : null}
				<div className="min-h-0 flex-1"><Transcript pid={pid} entries={selected.transcript} /></div>
				{selected.status === "running" ? <WorkingLine activity={selected.activity || selected.progress || "thinking"} startedS={0} /> : null}
				{selected.todos.some((phase) => phase.items.length > 0) ? <TodosPanel todos={selected.todos} /> : null}
			</div> : null}
		</Sheet>
	</>;
}
