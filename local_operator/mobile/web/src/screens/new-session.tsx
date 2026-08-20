/**
 * New session sheet (`#/new`): working directory (home, recents, or free
 * input), optional model, Start. On `{ok, pid}` the router takes the user
 * straight into the new session view.
 */
import { useEffect, useMemo, useState } from "react";
import { getDirectories, getModels, startSession } from "../api";
import { Button } from "../components/ui/button";
import { Sheet } from "../components/ui/sheet";
import { cn } from "../lib/cn";
import { basename, shortenHome } from "../lib/format";
import { navigate } from "../router";
import type { Directories, ModelEntry } from "../types";

export function NewSessionScreen() {
	const [dirs, setDirs] = useState<Directories | null>(null);
	const [models, setModels] = useState<ModelEntry[]>([]);
	const [cwd, setCwd] = useState("");
	const [model, setModel] = useState<ModelEntry | null>(null);
	const [modelSheetOpen, setModelSheetOpen] = useState(false);
	const [filter, setFilter] = useState("");
	const [error, setError] = useState("");
	const [starting, setStarting] = useState(false);

	useEffect(() => {
		getDirectories()
			.then((d) => {
				setDirs(d);
				setCwd((cur) => cur || d.home);
			})
			.catch((e) => setError(String(e.message ?? e)));
		getModels()
			.then((m) => setModels(m.models))
			.catch(() => {
				/* Models are optional here; the daemon's default applies. */
			});
	}, []);

	const filtered = useMemo(() => {
		const q = filter.trim().toLowerCase();
		if (!q) return models;
		return models.filter((m) =>
			`${m.provider} ${m.name} ${m.model_id}`.toLowerCase().includes(q),
		);
	}, [models, filter]);

	const start = async () => {
		if (!cwd.trim() || starting) return;
		setStarting(true);
		setError("");
		try {
			const res = await startSession({
				cwd: cwd.trim(),
				provider: model?.provider,
				model_id: model?.model_id,
			});
			if (res.ok) {
				navigate(`/s/${res.pid}`);
			} else {
				setError("the daemon refused to start the session");
			}
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setStarting(false);
		}
	};

	return (
		<div className="mx-auto flex min-h-full w-full max-w-md flex-col">
			<header className="flex items-center gap-2 px-2 pt-[max(env(safe-area-inset-top),0.75rem)] pb-3">
				<button
					type="button"
					onClick={() => navigate("/")}
					aria-label="back"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
				>
					‹
				</button>
				<h1 className="text-heading">new session</h1>
			</header>
			<main className="flex flex-1 flex-col gap-4 px-4 pb-4">
				<section className="flex flex-col gap-2">
					<label
						htmlFor="cwd-input"
						className="text-body-sm text-ink-muted"
					>
						working directory
					</label>
					<input
						id="cwd-input"
						value={cwd}
						onChange={(e) => setCwd(e.target.value)}
						spellCheck={false}
						autoCapitalize="off"
						autoCorrect="off"
						className="min-h-11 rounded-sm border border-control bg-surface px-3 font-mono text-mono text-ink outline-none"
					/>
					{dirs ? (
						<div className="flex flex-col gap-1">
							{[dirs.home, ...dirs.recent]
								.filter(
									(p, i, all) =>
										p && all.indexOf(p) === i,
								)
								.map((p) => (
									<button
										key={p}
										type="button"
										onClick={() => setCwd(p)}
										className={cn(
											"flex min-h-11 items-center gap-2 rounded-sm px-2 text-left active:bg-elevated",
											cwd === p && "bg-accent-wash",
										)}
									>
										<span className="shrink-0 font-mono text-mono-sm text-ink">
											{basename(p)}
										</span>
										<span className="min-w-0 truncate font-mono text-mono-sm text-ink-dim">
											{shortenHome(p, dirs.home)}
										</span>
									</button>
								))}
						</div>
					) : null}
				</section>

				<section className="flex flex-col gap-2">
					<span className="text-body-sm text-ink-muted">model</span>
					<button
						type="button"
						onClick={() => setModelSheetOpen(true)}
						className="flex min-h-11 items-center justify-between rounded-sm border border-control bg-surface px-3 text-left active:bg-elevated"
					>
						<span className="min-w-0 truncate text-body-sm text-ink">
							{model ? model.name : "default"}
						</span>
						<span className="text-ink-dim">▾</span>
					</button>
				</section>

				{error ? (
					<p className="text-body-sm text-danger">{error}</p>
				) : null}

				<div className="mt-auto">
					<Button
						variant="primary"
						className="w-full"
						disabled={!cwd.trim() || starting}
						onClick={start}
					>
						{starting ? "starting…" : "start"}
					</Button>
				</div>
			</main>

			<Sheet
				open={modelSheetOpen}
				onClose={() => setModelSheetOpen(false)}
				title="model"
			>
				<div className="flex flex-col gap-1 p-2">
					<input
						value={filter}
						onChange={(e) => setFilter(e.target.value)}
						placeholder="filter models"
						spellCheck={false}
						autoCapitalize="off"
						autoCorrect="off"
						className="mb-1 min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
					/>
					<button
						type="button"
						onClick={() => {
							setModel(null);
							setModelSheetOpen(false);
						}}
						className="flex min-h-11 items-center rounded-sm px-3 text-left text-ink-muted active:bg-surface"
					>
						default
					</button>
					{filtered.map((m) => (
						<button
							key={m.selector}
							type="button"
							onClick={() => {
								setModel(m);
								setModelSheetOpen(false);
							}}
							className="flex min-h-11 items-center gap-2 rounded-sm px-3 text-left active:bg-surface"
						>
							<span className="min-w-0 flex-1 truncate text-body">
								{m.name}
							</span>
							<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
								{m.provider}
							</span>
						</button>
					))}
				</div>
			</Sheet>
		</div>
	);
}
