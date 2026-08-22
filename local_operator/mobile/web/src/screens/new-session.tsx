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

	/* The tagged quick-pick rows: home first, then the temp dir (a common
	   scratch root the daemon now admits), then recent working directories.
	   Deduped so a recent dir that IS home or tmp does not appear twice. Each
	   row carries an explicit ``name`` because a bare basename is a poor label
	   for some roots — the macOS temp dir resolves to
	   ``/private/var/folders/…/T`` whose basename is a lone "T", so tmp is
	   named "tmp" outright. */
	const quickPicks = useMemo(() => {
		if (!dirs) return [] as { path: string; name: string; tag: string }[];
		const rows: { path: string; name: string; tag: string }[] = [
			{ path: dirs.home, name: basename(dirs.home), tag: "home" },
		];
		if (dirs.tmp) rows.push({ path: dirs.tmp, name: "tmp", tag: "tmp" });
		for (const p of dirs.recent) {
			rows.push({ path: p, name: basename(p), tag: "recent" });
		}
		const seen = new Set<string>();
		return rows.filter(({ path }) => {
			if (!path || seen.has(path)) return false;
			seen.add(path);
			return true;
		});
	}, [dirs]);

	/* Whether the current selection IS one of the quick-picks — when it is,
	   the free-text fallback stays empty so it does not duplicate the picked
	   row's path as an apparent fourth option (D3). */
	const cwdIsQuickPick = useMemo(
		() => quickPicks.some(({ path }) => path === cwd),
		[quickPicks, cwd],
	);

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
		<div className="relative mx-auto flex h-dvh w-full max-w-md flex-col">
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
			<main className="flex flex-1 flex-col gap-3 px-3 pb-3">
				<section className="flex flex-col gap-2">
					<label
						htmlFor="cwd-input"
						className="text-body-sm text-ink-muted"
					>
						working directory
					</label>
					{/* The quick-pick list is the primary path — tapping a row
					   fills the field, so most sessions start with no typing at
					   all. Each row is TAGGED (home / tmp / recent) so the
					   choices read as a menu rather than an undifferentiated
					   list of paths. The free-text field below stays for the
					   uncommon "somewhere else" case. */}
					{dirs ? (
						<div className="flex flex-col gap-1">
							{quickPicks.map(({ path, name, tag }) => (
								<button
									key={path}
									type="button"
									onClick={() => setCwd(path)}
									className={cn(
										"flex min-h-11 items-center gap-2 rounded-sm border border-control px-3 text-left active:bg-elevated",
										cwd === path
											? "border-accent bg-accent-wash"
											: "bg-surface",
									)}
								>
									<span className="shrink-0 font-mono text-mono-sm text-ink">
										{name}
									</span>
									<span className="min-w-0 flex-1 truncate font-mono text-mono-sm text-ink-dim">
										{shortenHome(path, dirs.home)}
									</span>
									<span className="shrink-0 rounded-sm bg-sunken px-1.5 py-0.5 text-meta text-ink-dim">
										{tag}
									</span>
								</button>
							))}
						</div>
					) : null}
					{/* The free-text fallback for a directory not in the picks.
					   Demoted below the quick-picks in both order and weight,
					   and it mirrors ``cwd`` ONLY when the selection is a custom
					   path — when a quick-pick is active the field stays empty
					   with its placeholder, so it never reads as a fourth option
					   duplicating the home row's absolute path (D3). */}
					<input
						id="cwd-input"
						value={cwdIsQuickPick ? "" : cwd}
						onChange={(e) => setCwd(e.target.value)}
						placeholder="or type another path…"
						spellCheck={false}
						autoCapitalize="off"
						autoCorrect="off"
						className="min-h-9 rounded-sm border border-hairline bg-transparent px-3 font-mono text-mono-sm text-ink-muted outline-none placeholder:text-ink-dim focus:border-control focus:text-ink"
					/>
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
						className="flex min-h-8 items-center rounded-sm px-2 text-left text-ink-muted active:bg-surface"
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
							className="flex min-h-8 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
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
