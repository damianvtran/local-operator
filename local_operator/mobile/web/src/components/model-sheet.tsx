/**
 * Model sheet: the server's ranked /api/models, filtered, current model marked.
 * Choosing POSTs set_model; the next projection repaint shows the truth, so
 * there is no local optimistic state.
 *
 * ONE FLAT LIST, IN THE SERVER'S ORDER. This used to group by provider, which
 * silently re-sorted the payload: the daemon ranks direct-connected providers
 * first and aggregators last (the same `rank_rows` the desktop `/model` uses),
 * and grouping restored registry order — ~445 aggregated Radient rows, roughly
 * 45 phone screens, before the first direct provider. The provider therefore
 * rides on each row instead of heading a section.
 */
import { useEffect, useMemo, useState } from "react";
import { getModels, sendCommand } from "../api";
import { cn } from "../lib/cn";
import type { ModelEntry, SessionProjection } from "../types";
import { Sheet } from "./ui/sheet";

export function ModelSheet({
	open,
	onClose,
	pid,
	projection,
}: {
	open: boolean;
	onClose: () => void;
	/** Route pid — the discovery record's, not the fold's (which stamps 0). */
	pid: string;
	projection: SessionProjection;
}) {
	const [models, setModels] = useState<ModelEntry[]>([]);
	const [filter, setFilter] = useState("");
	const [error, setError] = useState("");

	useEffect(() => {
		if (!open) return;
		getModels()
			.then((r) => setModels(r.models))
			.catch(() => setModels([]));
	}, [open]);

	/* An order-PRESERVING substring filter over the selector — `Array.filter`
	   keeps the server's ranking, so the best route for a query still leads.
	   Matched on `selector` because that is the string the user is typing
	   toward (`opus`, `anthropic/`, `glm`) and the same string the server ranks
	   on; the name is included so a model findable by its display name stays
	   findable.

	   The desktop picker's SUBSEQUENCE fallback (which resolves `anthopus` and
	   `sonnet4`) is deliberately not ported. It is the tie-breaking half of a
	   ranking, and re-implementing scoring here would be a second, drifting
	   copy of `model/ranking.py`; substring alone already leads every query
	   with the direct route because the server ordered the array. A query that
	   needs fuzzy matching is one keystroke from a substring match on a phone
	   keyboard, which is not the trade a duplicated ranker is worth. */
	const filtered = useMemo(() => {
		const q = filter.trim().toLowerCase();
		if (!q) return models;
		return models.filter((m) =>
			`${m.selector} ${m.name}`.toLowerCase().includes(q),
		);
	}, [models, filter]);

	const choose = async (m: ModelEntry) => {
		try {
			await sendCommand(pid, {
				op: "set_model",
				provider: m.provider,
				model_id: m.model_id,
			});
			onClose();
		} catch (e) {
			setError(String((e as Error).message ?? e));
		}
	};

	return (
		<Sheet open={open} onClose={onClose} title="model">
			<div className="flex flex-col gap-1 p-2">
				<input
					value={filter}
					onChange={(e) => setFilter(e.target.value)}
					placeholder="filter models"
					spellCheck={false}
					autoCapitalize="off"
					autoCorrect="off"
					className="mb-1 min-h-9 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
				/>
				{error ? (
					<p className="px-3 py-1 text-body-sm text-danger">
						{error}
					</p>
				) : null}
				{filtered.map((m) => {
					const current = m.selector === projection.model_selector;
					return (
						<button
							key={m.selector}
							type="button"
							onClick={() => void choose(m)}
							className="flex min-h-8 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
						>
							<span
								className={cn(
									"size-2 shrink-0 rounded-full",
									current ? "bg-accent" : "bg-hairline",
								)}
								aria-hidden
							/>
							<span className="min-w-0 flex-1 truncate text-body">
								{m.name}
							</span>
							{/* The provider moves onto the row now that there
							    is no section header to carry it — without it
							    two routes to one model are indistinguishable,
							    and which route answers is what differs in
							    price and quota. */}
							<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
								{m.provider}
							</span>
						</button>
					);
				})}
				{filtered.length === 0 ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						no matching models
					</p>
				) : null}
			</div>
		</Sheet>
	);
}
