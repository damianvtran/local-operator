/**
 * Projects sheet — the phone's view of the `project` store.
 *
 * WHAT THIS IS. A sheet over the sessions screen that lists workstreams and
 * drills into one: a flat list and a status-grouped compact board (on a phone,
 * grouped sections read as a board; side-by-side columns do not fit honestly),
 * a detail view (description, progress + age, milestones with a completion
 * toggle, linked sessions with state), and create/delete forms. **Timeline is
 * deliberately not built here** (§V2.B.3): a time axis needs pan/zoom gestures
 * the sheet system does not have, and the daemon already serves the composed
 * payload it would need, so adding it later is additive with no server work.
 *
 * ONE SHEET, VIEWS INSIDE IT. The sheet system draws one panel at a time
 * (`ui/sheet.tsx`); nesting a second Sheet would double the scrim and make
 * Back mean two things. Views swap inside this panel instead, each sub-view
 * carrying a `‹ projects` control back to the browse view; closing the sheet
 * (scrim, ✕, Escape) always ends the whole surface.
 *
 * WHERE THE NUMBERS COME FROM. Every value rendered here is the daemon's —
 * the same wire models the desktop serves, whose milestone statuses and
 * `progress_stale` verdict are computed server-side. This sheet derives no
 * status, no ordering and no staleness of its own, so the phone cannot
 * disagree with the tool or the desktop about one project. After every
 * mutation the affected reads are re-fetched — the list always, the detail
 * from the write's own answer — rather than patched locally, because a local
 * guess is a second derivation with a drift window.
 *
 * WHAT A FAILURE SAYS. A refusal renders the daemon's own sentence (the same
 * rule `pinRefusalReason` in the session list follows): the route's error body
 * is written for this reader, and rewording it here would be a second copy of
 * one rule. A bare status or an empty message falls back to a plain line,
 * because "409" under a button explains nothing.
 */
import {
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
	type ReactNode,
	type RefObject,
} from "react";
import { createProject, deleteProject, getProject, getProjects, setProjectMilestone } from "../api";
import { cn } from "../lib/cn";
import { formatRelative } from "../lib/format";
import type {
	ProjectLinkedSession,
	ProjectMilestone,
	ProjectSummary,
	ProjectView,
} from "../types";
import { Button } from "./ui/button";
import { Sheet } from "./ui/sheet";

/** The board's section order — the daemon's own rank, mirrored so a section
    the client draws and the order the daemon sorted cannot disagree. The one
    Python source is `STATUS_RANK` in `local_operator/server/models/
    desktop_projects.py`; a script cannot import it, so this copy carries the
    pointer. An unknown status (a row from a newer build) gets its own trailing
    section rather than being dropped. */
const STATUS_ORDER = ["active", "paused", "done", "archived"];

type View =
	| { name: "browse" }
	| { name: "detail"; key: string }
	| { name: "create" }
	| { name: "confirm"; key: string; projectName: string };

/** The daemon's sentence for a refusal, or a plain line when it gave none.

    Three failure classes meet here, and none of them may surface raw:
    a bare-status message (`api.request` falls back to the status when a
    failing response's body is not JSON — "409" under a button the reader just
    pressed explains nothing), and a FETCH-level failure, which arrives as the
    browser's own TypeError ("Failed to fetch") when the daemon is not
    listening at all (round-1 design, D3). */
function refusalReason(error: unknown): string {
	if (error instanceof TypeError) return "could not reach the daemon";
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return "the daemon did not say why";
	return message;
}

/** The milestone count vocabulary the row and the section header share, so the
    two cannot count one list differently. */
function milestoneCount(project: { milestones_completed: number; milestones_total: number }): string {
	return `${project.milestones_completed}/${project.milestones_total}`;
}

/** The sessions-column text: a count, the live subset, and nothing else.

    `0 sessions` is stated rather than omitted — an unlinked project is a real
    state (create never auto-links), and a blank would read as either "none" or
    "not loaded". `live` is only named when there is at least one. */
function sessionsText(project: ProjectSummary): string {
	if (project.sessions === 0) return "no sessions";
	const sessions = `${project.sessions} session${project.sessions === 1 ? "" : "s"}`;
	return project.live_sessions > 0 ? `${sessions} · ${project.live_sessions} live` : sessions;
}

/** The state word for one linked session, and its ink.

    `missing` (the linked session's directory is gone; the store marks it and
    never auto-removes) outranks the runtime word: a record can outlive the
    conversation it names, and saying "stopped" about a directory that no
    longer exists would hide the one fact the reader can act on. */
function sessionState(row: ProjectLinkedSession): { word: string; ink: string } {
	if (!row.exists) return { word: "missing", ink: "text-warning" };
	/* An unknown state word from a newer build passes through rather than being
	   flattened into a known one. */
	const base = row.runtime.state;
	const ink =
		base === "live"
			? "text-accent"
			: base === "wedged"
				? "text-warning"
				: "text-ink-dim";
	const word = row.archived ? `${base} · archived` : base;
	return { word, ink };
}

function Section({ title, children }: { title: string; children: ReactNode }) {
	return (
		<section className="border-t border-hairline px-3 py-2">
			<h3 className="mb-1 text-meta font-medium text-ink-dim">{title}</h3>
			{children}
		</section>
	);
}

/** One project row, shared by the list and the board. `showStatus` is off in
    the board because the section heading IS the status — printing it twice in
    one glance is the shape this app avoids elsewhere (the todos panel's
    implicit phase). */
function ProjectRow({
	project,
	showStatus,
	onOpen,
}: {
	project: ProjectSummary;
	showStatus: boolean;
	onOpen: () => void;
}) {
	const meta: string[] = [sessionsText(project)];
	if (project.milestones_total > 0) meta.push(`${milestoneCount(project)} milestones`);
	const progress =
		project.progress_updated_at !== null
			? `progress ${formatRelative(project.progress_updated_at)}`
			: "no progress";
	return (
		<button
			type="button"
			onClick={onOpen}
			className="flex w-full flex-col gap-0.5 rounded-sm px-2 py-2 text-left active:bg-surface"
		>
			<span className="flex min-w-0 items-baseline gap-2">
				<span className="min-w-0 flex-1 truncate text-body font-medium text-ink">
					{project.name}
				</span>
				{showStatus ? (
					<span className="shrink-0 text-meta text-ink-dim">{project.status}</span>
				) : null}
			</span>
			<span className="flex min-w-0 items-baseline gap-2 text-meta">
				{/* Stale only tints a REPORT that exists: `progress_stale` is true by
				    construction for a record with none, and painting every fresh
				    project's "no progress" as a warning would spend the loudest ink
				    on the honest default. */}
				<span
					className={cn(
						"min-w-0 truncate",
						project.progress_updated_at !== null && project.progress_stale
							? "text-warning"
							: "text-ink-muted",
					)}
				>
					{progress}
				</span>
				<span className="ml-auto shrink-0 text-ink-dim">{meta.join(" · ")}</span>
			</span>
		</button>
	);
}

/** The `‹ projects` control sub-views carry back to the browse view. */
function BackRow({ onClick, label = "projects" }: { onClick: () => void; label?: string }) {
	return (
		<button
			type="button"
			onClick={onClick}
			className="flex min-h-11 items-center gap-1 px-3 text-body-sm text-ink-muted active:bg-surface"
		>
			<span aria-hidden>‹</span>
			{label}
		</button>
	);
}

function ErrorBlock({ message, onRetry }: { message: string; onRetry: () => void }) {
	return (
		<div className="flex flex-col items-start gap-2 px-3 py-2">
			<p role="alert" className="text-body-sm break-words text-danger">
				{message}
			</p>
			<Button variant="quiet" onClick={onRetry}>
				retry
			</Button>
		</div>
	);
}

export function ProjectsSheet({
	open,
	onClose,
	returnFocusRef,
}: {
	open: boolean;
	onClose: () => void;
	returnFocusRef?: RefObject<HTMLElement | null>;
}) {
	const [view, setView] = useState<View>({ name: "browse" });
	/* The list/board choice survives a close — it is a way of reading, not a
	   transient state — while everything scoped to one open resets below. */
	const [board, setBoard] = useState(false);
	const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
	const [listError, setListError] = useState("");
	const [detail, setDetail] = useState<{ project: ProjectView; links: ProjectLinkedSession[] } | null>(
		null,
	);
	const [detailError, setDetailError] = useState("");
	const [formError, setFormError] = useState("");
	const [notice, setNotice] = useState("");
	const [busy, setBusy] = useState(false);
	const [name, setName] = useState("");
	const [description, setDescription] = useState("");
	/* The key the in-flight detail read belongs to. A response for a project
	   the reader has already left must not paint under the next one. */
	const detailKeyRef = useRef<string | null>(null);

	const loadList = useCallback(async () => {
		setListError("");
		try {
			const { projects: rows } = await getProjects();
			setProjects(rows);
		} catch (error) {
			setListError(refusalReason(error));
		}
	}, []);

	const loadDetail = useCallback(async (key: string) => {
		detailKeyRef.current = key;
		setDetailError("");
		try {
			const payload = await getProject(key);
			if (detailKeyRef.current !== key) return;
			setDetail(payload);
		} catch (error) {
			if (detailKeyRef.current !== key) return;
			setDetailError(refusalReason(error));
		}
	}, []);

	/* Every open starts at the browse view over a fresh listing. The store is
	   shared with the tool and the desktop app, so "fresh" is the only honest
	   starting state. */
	useEffect(() => {
		if (!open) {
			detailKeyRef.current = null;
			return;
		}
		setView({ name: "browse" });
		setNotice("");
		setFormError("");
		setDetail(null);
		setDetailError("");
		void loadList();
	}, [open, loadList]);

	useEffect(() => {
		if (view.name !== "detail") {
			detailKeyRef.current = null;
			return;
		}
		void loadDetail(view.key);
	}, [view, loadDetail]);

	/* Entering the create view starts a FRESH form. A draft the reader
	   abandoned (cancel, sheet close) must not come back pre-filled on the
	   next "new project" — the reader did not type it this time, and a
	   distracted one can create the abandoned name (round-1 UX, U1). */
	useEffect(() => {
		if (view.name === "create") {
			setName("");
			setDescription("");
		}
	}, [view]);

	const openDetail = (key: string) => {
		setDetail(null);
		setDetailError("");
		setFormError("");
		setView({ name: "detail", key });
	};

	const submitCreate = async () => {
		if (busy) return;
		setBusy(true);
		setFormError("");
		try {
			const { project } = await createProject({
				name: name.trim(),
				description: description.trim(),
			});
			setNotice(`created ${project.name}`);
			setName("");
			setDescription("");
			setView({ name: "browse" });
			await loadList();
		} catch (error) {
			setFormError(refusalReason(error));
		} finally {
			setBusy(false);
		}
	};

	const confirmDelete = async (key: string, projectName: string) => {
		if (busy) return;
		setBusy(true);
		setFormError("");
		try {
			await deleteProject(key, projectName);
			setNotice(`deleted ${projectName}`);
			setView({ name: "browse" });
			setDetail(null);
			await loadList();
		} catch (error) {
			setFormError(refusalReason(error));
		} finally {
			setBusy(false);
		}
	};

	const toggleMilestone = async (project: ProjectView, milestone: ProjectMilestone) => {
		if (busy) return;
		setBusy(true);
		setFormError("");
		try {
			const { project: updated } = await setProjectMilestone(project.id, {
				name: milestone.name,
				completed: milestone.completed_at === null,
			});
			/* The write's own answer replaces the view; the links stay as they
			   were read, because a milestone edit does not touch them. */
			setDetail((current) => (current ? { ...current, project: updated } : current));
			void loadList();
		} catch (error) {
			setFormError(refusalReason(error));
		} finally {
			setBusy(false);
		}
	};

	const groups = useMemo(() => {
		const buckets = new Map<string, ProjectSummary[]>();
		for (const row of projects ?? []) {
			const rows = buckets.get(row.status) ?? [];
			rows.push(row);
			buckets.set(row.status, rows);
		}
		const known = STATUS_ORDER.filter((status) => buckets.has(status));
		const unknown = [...buckets.keys()].filter((status) => !STATUS_ORDER.includes(status)).sort();
		return [...known, ...unknown].map((status) => ({ status, rows: buckets.get(status) ?? [] }));
	}, [projects]);

	const title =
		view.name === "detail"
			? (detail?.project.name ?? "project")
			: view.name === "create"
				? "new project"
				: view.name === "confirm"
					? "delete project"
					: "projects";

	return (
		<Sheet open={open} onClose={onClose} title={title} returnFocusRef={returnFocusRef}>
			{view.name === "browse" ? (
				<div className="flex flex-col pb-3">
					{projects !== null && projects.length > 0 ? (
						<div className="flex items-center gap-2 px-3 pt-1 pb-2">
							<div
								className="flex overflow-hidden rounded-sm border border-control"
								role="group"
							>
								{/* Two readings of ONE array: the list keeps the daemon's
								    order, the board groups it by status. Neither re-sorts.
								    The selected segment spends the ACCENT for its fill: the
								    same-ground fills round 1 tried (bg-elevated, then the
								    accent wash) both measured ~1:1 against the track — 1.009:1
								    in the default theme, under 3:1 in all 31 — so the state
								    leaned on a 1.88:1 label delta alone. accent-on-track is
								    7.24:1 and clears 3:1 in every theme (on-accent label
								    8.58:1). */}
								{(["list", "board"] as const).map((mode) => (
									<button
										key={mode}
										type="button"
										aria-pressed={board === (mode === "board")}
										onClick={() => setBoard(mode === "board")}
										className={cn(
											"min-h-11 min-w-14 px-2 text-mono-sm",
											board === (mode === "board")
												? "bg-accent text-on-accent"
												: "text-ink-muted",
										)}
									>
										{mode}
									</button>
								))}
							</div>
							<Button
								variant="outline"
								className="ml-auto"
								onClick={() => {
									setFormError("");
									setView({ name: "create" });
								}}
							>
								new project
							</Button>
						</div>
					) : null}
					{notice ? (
						<p role="status" className="px-3 pb-1 text-meta text-ink-muted">
							{notice}
						</p>
					) : null}
					{listError ? (
						<ErrorBlock
							message={listError}
							onRetry={() => {
								void loadList();
							}}
						/>
					) : projects === null ? (
						<p className="px-3 py-2 text-body-sm text-ink-dim">loading projects…</p>
					) : projects.length === 0 ? (
						<div className="flex flex-col items-center gap-3 px-6 py-8 text-center">
							<p className="text-body text-ink-muted">no projects yet</p>
							<p className="text-body-sm text-ink-dim">
								create one here, or ask an agent: “create a project and link this session”.
							</p>
							<Button variant="primary" onClick={() => setView({ name: "create" })}>
								new project
							</Button>
						</div>
					) : board ? (
						<div className="flex flex-col gap-1">
							{groups.map((group) => (
								<section key={group.status}>
									<h3 className="px-3 py-1 text-meta font-medium text-ink-muted">
										{group.status}
									</h3>
									{group.rows.map((project) => (
										<ProjectRow
											key={project.id}
											project={project}
											showStatus={false}
											onOpen={() => openDetail(project.id)}
										/>
									))}
								</section>
							))}
						</div>
					) : (
						<div className="flex flex-col">
							{projects.map((project) => (
								<ProjectRow
									key={project.id}
									project={project}
									showStatus
									onOpen={() => openDetail(project.id)}
								/>
							))}
						</div>
					)}
				</div>
			) : null}

			{view.name === "detail" ? (
				<div className="flex flex-col pb-3">
					<BackRow onClick={() => setView({ name: "browse" })} />
					{detailError ? (
						<ErrorBlock
							message={detailError}
							onRetry={() => {
								void loadDetail(view.key);
							}}
						/>
					) : detail === null ? (
						<p className="px-3 py-2 text-body-sm text-ink-dim">loading project…</p>
					) : (
						<>
							<div className="flex flex-col gap-1 px-3 pb-2">
								<p className="text-meta text-ink-muted">
									{[
										detail.project.status,
										detail.project.estimate !== null
											? `${detail.project.estimate} ${detail.project.estimate_unit}`
											: null,
										detail.project.start_date ? `start ${detail.project.start_date}` : null,
										detail.project.target_date
											? `target ${detail.project.target_date}`
											: null,
										detail.project.completed_at
											? `completed ${detail.project.completed_at}`
											: null,
									]
										.filter(Boolean)
										.join(" · ")}
								</p>
								{detail.project.tags.length > 0 ? (
									<p className="text-meta text-ink-dim">
										{detail.project.tags.map((tag) => `#${tag}`).join(" ")}
									</p>
								) : null}
								{detail.project.description ? (
									<p className="text-body-sm break-words whitespace-pre-wrap text-ink">
										{detail.project.description}
									</p>
								) : null}
							</div>
							<Section title="progress">
								{detail.project.progress ? (
									<p className="text-body-sm break-words whitespace-pre-wrap text-ink">
										{detail.project.progress}
									</p>
								) : (
									<p className="text-body-sm text-ink-dim">no progress reported yet</p>
								)}
								{/* The age line belongs to a REPORT: with no snippet it could only
								    restate the line above in the negative (round-1 UX, U3). The
								    reporter comes before the stale marker, because "stale"
								    modifies the report, not the person who filed it (U2). */}
								{detail.project.progress ? (
									<p
										className={cn(
											"mt-1 text-meta",
											detail.project.progress_stale &&
												detail.project.progress_updated_at !== null
												? "text-warning"
												: "text-ink-muted",
										)}
									>
										{detail.project.progress_updated_at !== null
											? `reported ${formatRelative(detail.project.progress_updated_at)}` +
												(detail.project.progress_reported_by
													? ` by ${detail.project.progress_reported_by}`
													: "") +
												(detail.project.progress_stale ? " · stale" : "")
											: "none recorded"}
									</p>
								) : null}
							</Section>
							<Section
								title={
									detail.project.milestones.length > 0
										? `milestones (${detail.project.milestones.filter((m) => m.completed_at !== null).length}/${detail.project.milestones.length})`
										: "milestones"
								}
							>
								{formError ? (
									<p role="alert" className="mb-1 text-meta break-words text-danger">
										{formError}
									</p>
								) : null}
								{detail.project.milestones.length === 0 ? (
									<p className="text-body-sm text-ink-dim">no milestones</p>
								) : (
									detail.project.milestones.map((milestone) => (
										<button
											key={milestone.name}
											type="button"
											disabled={busy}
											aria-pressed={milestone.completed_at !== null}
											onClick={() => void toggleMilestone(detail.project, milestone)}
											className="flex min-h-11 w-full items-center gap-2 rounded-sm px-1 text-left active:bg-surface disabled:opacity-50"
										>
											<span
												aria-hidden
												className={cn(
													"w-4 shrink-0 text-center",
													milestone.status === "completed"
														? "text-success"
														: milestone.status === "overdue"
															? "text-warning"
															: "text-ink-dim",
												)}
											>
												{milestone.completed_at !== null ? "☑" : "☐"}
											</span>
											<span
												className={cn(
													"min-w-0 flex-1 truncate text-body-sm",
													milestone.completed_at !== null ? "text-ink-dim" : "text-ink",
												)}
											>
												{milestone.name}
											</span>
											{milestone.target_date ? (
												<span
													className={cn(
														"shrink-0 font-mono text-mono-sm",
														milestone.status === "overdue"
															? "text-warning"
															: "text-ink-dim",
													)}
												>
													{milestone.target_date}
												</span>
											) : null}
										</button>
									))
								)}
							</Section>
							<Section title={`sessions (${detail.links.length})`}>
								{detail.links.length === 0 ? (
									<p className="text-body-sm text-ink-dim">no linked sessions</p>
								) : (
									detail.links.map((row) => {
										const state = sessionState(row);
										return (
											<div
												key={row.session_id}
												className="flex min-h-8 items-center gap-2 px-1"
											>
												<span className="min-w-0 flex-1 truncate text-body-sm text-ink">
													{row.title || row.session_id}
												</span>
												<span className={cn("shrink-0 text-meta", state.ink)}>
													{state.word}
												</span>
											</div>
										);
									})
								)}
							</Section>
							<div className="px-3 pt-2">
								<Button
									variant="danger"
									disabled={busy}
									onClick={() => {
										setFormError("");
										setView({
											name: "confirm",
											key: detail.project.id,
											projectName: detail.project.name,
										});
									}}
								>
									delete project
								</Button>
							</div>
						</>
					)}
				</div>
			) : null}

			{view.name === "create" ? (
				<div className="flex flex-col gap-3 px-3 pb-4">
					<BackRow onClick={() => setView({ name: "browse" })} />
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">name</span>
						<input
							value={name}
							onChange={(event) => setName(event.target.value)}
							placeholder="e.g. payments-migration"
							spellCheck={false}
							autoCapitalize="off"
							autoCorrect="off"
							className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
						/>
						<span className="text-meta text-ink-dim">
							letters, digits, dot, underscore and hyphen; no spaces
						</span>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">description (optional)</span>
						<input
							value={description}
							onChange={(event) => setDescription(event.target.value)}
							maxLength={240}
							className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
						/>
					</label>
					{formError ? (
						<p role="alert" className="text-body-sm break-words text-danger">
							{formError}
						</p>
					) : null}
					<div className="flex gap-2">
						<Button
							variant="primary"
							disabled={busy || name.trim() === ""}
							aria-busy={busy ? true : undefined}
							onClick={() => void submitCreate()}
						>
							{busy ? "creating…" : "create"}
						</Button>
						<Button variant="quiet" disabled={busy} onClick={() => setView({ name: "browse" })}>
							cancel
						</Button>
					</div>
				</div>
			) : null}

			{view.name === "confirm" ? (
				<div className="flex flex-col gap-3 px-3 pb-4">
					<BackRow
						label="back"
						onClick={() => setView({ name: "detail", key: view.key })}
					/>
					<p className="text-body-sm break-words text-ink">
						delete <span className="font-medium">{view.projectName}</span>? the project row
						is removed permanently; its linked sessions are not touched.
					</p>
					{formError ? (
						<p role="alert" className="text-body-sm break-words text-danger">
							{formError}
						</p>
					) : null}
					<div className="flex gap-2">
						<Button
							variant="danger"
							disabled={busy}
							aria-busy={busy ? true : undefined}
							onClick={() => void confirmDelete(view.key, view.projectName)}
						>
							delete
						</Button>
						<Button
							variant="quiet"
							disabled={busy}
							onClick={() => setView({ name: "detail", key: view.key })}
						>
							cancel
						</Button>
					</div>
				</div>
			) : null}
		</Sheet>
	);
}
