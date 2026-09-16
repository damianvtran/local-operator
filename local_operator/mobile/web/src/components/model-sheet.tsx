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
import { filterModels } from "../lib/model-filter";
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
	/* `null` means NOT RESOLVED YET, which is a different fact from "resolved
	   and empty" and has to be told apart from it — see the render ladder below.
	   Spelled as a nullable list rather than a separate boolean because
	   `past-sessions.tsx` already states "still loading" that way; a second
	   spelling of one idea beside an established one is the defect. */
	const [models, setModels] = useState<ModelEntry[] | null>(null);
	const [filter, setFilter] = useState("");
	const [error, setError] = useState("");

	useEffect(() => {
		if (!open) return;
		/* A reopened sheet starts from the FULL ranked list. The filter is
		   component state that used to survive a close, so selecting a model and
		   reopening restored the previous query and its short subset — which on a
		   phone reads as the very bug this surface exists to fix (a short,
		   unrepresentative list), arrived at by another route. */
		setFilter("");
		setError("");
		/* Back to unresolved on every open, because this sheet REFETCHES per open:
		   keeping the previous list would paint a stale catalogue as though it were
		   the answer to the request now in flight. */
		setModels(null);
		getModels()
			.then((r) => {
				setModels(r.models);
			})
			.catch((e) => {
				/* The daemon's message, not an empty list. It composes a precise,
				   actionable one — "Model catalogue unavailable for Radient; retry
				   or log in again" — and discarding it rendered a 502 as "no
				   matching models", telling a user whose token expired that their
				   filter matched nothing. */
				setModels([]);
				setError(String((e as Error).message ?? e));
			});
	}, [open]);

	/* Order-PRESERVING: `Array.filter` keeps the server's ranking, so the best
	   route for a query still leads. The predicate itself lives in
	   `lib/model-filter` because `#/new` needs the identical one. */
	const filtered = useMemo(
		() => filterModels(models ?? [], filter),
		[models, filter],
	);

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
				{/* ONE exclusive ladder: error → loading → empty-filtered, and the rows
				    below it. The precedence runs that way round because each rung is a
				    claim about a DIFFERENT fact, and the later rungs are only true once
				    the earlier ones are false.

				    `error` first: a fetch that failed says nothing about the user's
				    query, so "no matching models — try a provider…" under a 502 is a
				    false statement that also blames the user for an expired token.
				    These two used to render TOGETHER — the daemon's real message with
				    the contradictory filter advice directly beneath it — so the error
				    branch has to REPLACE the empty state, not join it.

				    `loading` before empty: `models` is unresolved until the response
				    lands, and an unresolved list is not an empty one. Painting the
				    no-match verdict during the in-flight window told a user with an
				    EMPTY filter field to change their filter — advice for a query they
				    never typed, on every open, since this sheet refetches per open.

				    Reversing any of this re-creates a lie, which is why the order is
				    stated here rather than left to the reader of three sibling
				    conditionals. */}
				{error ? (
					<p className="px-3 py-1 text-body-sm text-danger">
						{error}
					</p>
				) : models === null ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">loading…</p>
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
							{/* The slot is always reserved so nothing shifts, but only
							    the CURRENT row paints a dot. The inert `bg-hairline`
							    dot sat a few values off the sheet surface — barely
							    separable from the background, so the column read as an
							    8px indent rather than a state column — while giving
							    the one meaningful dot 995 decoys to compete with. */}
							<span
								className={cn(
									"size-2 shrink-0 rounded-full",
									current && "bg-accent",
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
				{!error && models !== null && filtered.length === 0 ? (
					/* Name the recovery rather than stating a verdict: with the
					   provider headers gone there is no visible inventory left to
					   scan as a fallback, so "no matching models" alone leaves
					   guessing at another query as the only way out. Reachable ONLY
					   once the catalogue resolved and the fetch succeeded, so it is
					   the one state where the filter really is what excluded
					   everything. */
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						no matching models — try a provider (anthropic, xai) or a
						model name (opus, glm)
					</p>
				) : null}
			</div>
		</Sheet>
	);
}
