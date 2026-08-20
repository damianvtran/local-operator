/**
 * Model sheet: fuzzy search over /api/models grouped by provider, the
 * current model marked. Choosing POSTs set_model; the next projection
 * repaint shows the truth, so there is no local optimistic state.
 */
import { useEffect, useMemo, useState } from "react";
import { getModels, sendCommand } from "../api";
import { cn } from "../lib/cn";
import type { ModelEntry, SessionProjection } from "../types";
import { Sheet } from "./ui/sheet";

export function ModelSheet({
	open,
	onClose,
	projection,
}: {
	open: boolean;
	onClose: () => void;
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

	const groups = useMemo(() => {
		const q = filter.trim().toLowerCase();
		const filtered = q
			? models.filter((m) =>
					`${m.provider} ${m.name} ${m.model_id}`
						.toLowerCase()
						.includes(q),
				)
			: models;
		const byProvider = new Map<string, ModelEntry[]>();
		for (const m of filtered) {
			const list = byProvider.get(m.provider) ?? [];
			list.push(m);
			byProvider.set(m.provider, list);
		}
		return [...byProvider.entries()];
	}, [models, filter]);

	const choose = async (m: ModelEntry) => {
		try {
			await sendCommand(projection.pid, {
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
				{groups.map(([provider, list]) => (
					<div key={provider} className="flex flex-col">
						<span className="px-3 pt-2 font-mono text-mono-sm text-ink-dim">
							{provider}
						</span>
						{list.map((m) => {
							const current =
								m.selector === projection.model_selector;
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
											current
												? "bg-accent"
												: "bg-hairline",
										)}
										aria-hidden
									/>
									<span className="min-w-0 flex-1 truncate text-body">
										{m.name}
									</span>
									<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
										{m.model_id}
									</span>
								</button>
							);
						})}
					</div>
				))}
				{groups.length === 0 ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						no matching models
					</p>
				) : null}
			</div>
		</Sheet>
	);
}
