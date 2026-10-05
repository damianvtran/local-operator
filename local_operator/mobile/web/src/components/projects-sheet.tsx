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
import {
	createProject,
	deleteProject,
	getProject,
	getProjects,
	HttpError,
	linkProjectSession,
	patchProject,
	removeProjectMilestone,
	setProjectMilestone,
	unlinkProjectSession,
} from "../api";
import { getSessions } from "../api";
import { cn } from "../lib/cn";
import { formatRelative } from "../lib/format";
import { STATUS_ORDER } from "../projects-status.generated";
import type {
	ProjectLinkedSession,
	ProjectMilestone,
	ProjectSummary,
	ProjectView,
	SessionSummary,
} from "../types";
import { Button } from "./ui/button";
import { Sheet } from "./ui/sheet";

/* The board's section order is GENERATED, not copied: `STATUS_ORDER` comes from
   `src/projects-status.generated.ts`, which `scripts/generate-projects-status.mjs`
   builds out of the daemon's own `STATUS_RANK` (the one source the desktop
   routes and the phone's relay both import). A hand-copied array lived here
   before and had already drifted — the store grew four statuses into seven and
   every unknown one landed in a trailing section instead of its lifecycle
   position. `src/projects-status.test.ts` fails when the two diverge. */

/** The one skin the sheet's text, date, number and select controls share — a
    field added to a form must not look like a foreign control. */
const FIELD_CLASS =
	"min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim";

type View =
	| { name: "browse" }
	| { name: "detail"; key: string }
	| { name: "create" }
	| { name: "edit"; key: string }
	| { name: "link"; key: string }
	| { name: "milestone"; key: string; editing: string | null }
	| { name: "confirm"; key: string; projectName: string };

/** The edit form's values, held as TEXT: a form holds what the reader typed,
    and the daemon's own grammar is what refuses it (a client-side copy of the
    store's rules would be a second validator, drifting one release later). */
interface ProjectDraft {
	name: string;
	description: string;
	status: string;
	tags: string;
	start_date: string;
	target_date: string;
	estimate: string;
	estimate_unit: string;
}

/** The milestone editor's values: `editing === null` means the name is the
    reader's to choose (an absent name is ADDED — the route is add-or-update
    keyed by name), otherwise the name is fixed and only the date changes. */
interface MilestoneDraft {
	name: string;
	target_date: string;
}

/** The edit form, seeded from the row the daemon served. Dates are `""` when
    unset (never `null`): a date input's vocabulary is a string, and `""` is
    the same spelling the PATCH body uses to CLEAR one. */
function draftFrom(project: ProjectView): ProjectDraft {
	return {
		name: project.name,
		description: project.description,
		status: project.status,
		tags: project.tags.join(", "),
		start_date: project.start_date ?? "",
		target_date: project.target_date ?? "",
		estimate: project.estimate === null ? "" : String(project.estimate),
		estimate_unit: project.estimate_unit,
	};
}

/** The tags field's text, split the way a reader types a list — commas or
    spaces, the same two separators the tag grammar itself cannot contain. The
    values are sent VERBATIM: upper-case or a leading `#` is the store's to
    refuse (with its own sentence), not this field's to silently rewrite. */
function parseTags(text: string): string[] {
	return text
		.split(/[\s,]+/)
		.map((tag) => tag.trim())
		.filter((tag) => tag !== "");
}

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
	/* The create form's two extra vocabulary fields. `status` starts at the
	   daemon's own default for a create (`active`) rather than at an empty
	   choice, so the form's resting state posts what the store would have chosen
	   anyway. */
	const [createStatus, setCreateStatus] = useState("active");
	const [createTags, setCreateTags] = useState("");
	/* The edit and milestone editors' values. `null` while their sub-view is not
	   open — the forms are seeded on entry (`draftFrom`, `openMilestone`) rather
	   than by an effect, so an abandoned draft can never be re-shown. */
	const [draft, setDraft] = useState<ProjectDraft | null>(null);
	const [milestoneDraft, setMilestoneDraft] = useState<MilestoneDraft | null>(null);
	/* The link sub-view's candidates: the daemon's sessions, `null` while the
	   fetch is in flight, with its own error line because a catalogue failure
	   must not read as "there is nothing to link". */
	const [candidates, setCandidates] = useState<SessionSummary[] | null>(null);
	const [linkError, setLinkError] = useState("");
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
			setCreateStatus("active");
			setCreateTags("");
		}
	}, [view]);

	/* The link sub-view's candidates are fetched on entry and never cached: the
	   list is "the sessions this daemon currently knows", which a session that
	   started or ended since the sheet opened would make stale. Re-entering the
	   view re-runs this (the view object is new each time), which is also how
	   the error block's retry works. */
	useEffect(() => {
		if (view.name !== "link") return;
		let cancelled = false;
		setCandidates(null);
		setLinkError("");
		getSessions()
			.then(({ sessions }) => {
				if (!cancelled) setCandidates(sessions);
			})
			.catch((error) => {
				if (!cancelled) setLinkError(refusalReason(error));
			});
		return () => {
			cancelled = true;
		};
	}, [view]);

	const openDetail = (key: string) => {
		setDetail(null);
		setDetailError("");
		setFormError("");
		/* A receipt belongs to the view it was earned in: opening a fresh row
		   from the list must not inherit the last row's "updated …" line. */
		setNotice("");
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
				status: createStatus,
				tags: parseTags(createTags),
			});
			setNotice(`created ${project.name}`);
			setName("");
			setDescription("");
			setCreateStatus("active");
			setCreateTags("");
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

	/** A write's refusal, rendered where the reader can act on it.

	    The 404 arm is "the row vanished under the sheet": another surface (the
	    tool, the desktop app, a second phone) deleted the project between the
	    read and the write, and a refusal sentence standing under a detail view
	    for a row that no longer exists is a dead end — there is nothing left to
	    retry against. The sheet says what happened at the browse view and
	    re-reads the list, so the reader sees the store's actual contents.
	    Every other refusal stays in place, with the reader's input intact. */
	const refuseWrite = async (error: unknown, key?: string) => {
		const message = refusalReason(error);
		if (key && error instanceof HttpError && error.code === "project_not_found") {
			setFormError("");
			setDetail(null);
			setNotice(message);
			setView({ name: "browse" });
			await loadList();
			return;
		}
		setFormError(message);
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
			await refuseWrite(error, project.id);
		} finally {
			setBusy(false);
		}
	};

	/** Enter the edit form, seeded from the row the daemon just served. */
	const openEdit = (project: ProjectView) => {
		setDraft(draftFrom(project));
		setFormError("");
		setView({ name: "edit", key: project.id });
	};

	/** Enter the milestone editor; `editing === null` adds a new milestone. */
	const openMilestone = (key: string, editing: ProjectMilestone | null) => {
		setMilestoneDraft({
			name: editing?.name ?? "",
			target_date: editing?.target_date ?? "",
		});
		setFormError("");
		setView({ name: "milestone", key, editing: editing?.name ?? null });
	};

	const submitEdit = async (key: string) => {
		if (busy || !draft) return;
		setBusy(true);
		setFormError("");
		try {
			const { project } = await patchProject(key, {
				name: draft.name.trim(),
				description: draft.description.trim(),
				status: draft.status,
				tags: parseTags(draft.tags),
				/* A field the reader EMPTIED is a deliberate value, not an
				   omission: `""` clears a date (the daemon's own tri-state). */
				start_date: draft.start_date,
				target_date: draft.target_date,
				/* The ESTIMATE is the one exception, and it is the store's rule, not
				   a choice here: its apply arm is `estimate is not None`, so no
				   request can clear it. An emptied box therefore OMITS the key
				   (leave it as it is) — sending `null` would be a silent no-op that
				   left the old number on the row while the form showed it blank. */
				...(draft.estimate.trim() === "" ? {} : { estimate: Number(draft.estimate) }),
				estimate_unit: draft.estimate_unit,
			});
			setNotice(`updated ${project.name}`);
			setView({ name: "detail", key });
			await loadList();
		} catch (error) {
			await refuseWrite(error, key);
		} finally {
			setBusy(false);
		}
	};

	const submitMilestone = async (key: string, editing: string | null) => {
		if (busy || !milestoneDraft) return;
		const milestoneName = (editing ?? milestoneDraft.name).trim();
		if (!milestoneName) return;
		setBusy(true);
		setFormError("");
		try {
			const { project } = await setProjectMilestone(key, {
				name: milestoneName,
				/* Sent even when empty: on an existing milestone `""` CLEARS the
				   date, which is the other half of "set a milestone's target
				   date". On a new one an empty date is simply not a date. */
				target_date: milestoneDraft.target_date,
			});
			setDetail((current) => (current ? { ...current, project } : current));
			setNotice(
				editing === null ? `added milestone ${milestoneName}` : `updated milestone ${milestoneName}`,
			);
			setView({ name: "detail", key });
			void loadList();
		} catch (error) {
			await refuseWrite(error, key);
		} finally {
			setBusy(false);
		}
	};

	const removeMilestone = async (key: string, name: string) => {
		if (busy) return;
		setBusy(true);
		setFormError("");
		try {
			const { project } = await removeProjectMilestone(key, name);
			setDetail((current) => (current ? { ...current, project } : current));
			setNotice(`removed milestone ${name}`);
			setView({ name: "detail", key });
			void loadList();
		} catch (error) {
			await refuseWrite(error, key);
		} finally {
			setBusy(false);
		}
	};

	const linkSession = async (key: string, sessionId: string) => {
		if (busy) return;
		setBusy(true);
		setFormError("");
		try {
			await linkProjectSession(key, sessionId);
			setNotice(`linked ${sessionId}`);
			/* The link route answers with a SUMMARY — no linked-session rows — so
			   the composed view is RE-READ rather than patched from a document
			   this call never returned. */
			setView({ name: "detail", key });
			await loadList();
		} catch (error) {
			await refuseWrite(error, key);
		} finally {
			setBusy(false);
		}
	};

	const unlinkSession = async (key: string, sessionId: string) => {
		if (busy) return;
		setBusy(true);
		setFormError("");
		try {
			await unlinkProjectSession(key, sessionId);
			setNotice(`unlinked ${sessionId}`);
			/* Same reason as the link: the answer is a summary, so the view the
			   reader is looking at comes from a fresh read. */
			await loadDetail(key);
			void loadList();
		} catch (error) {
			await refuseWrite(error, key);
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
		/* `known` rides along so the section heading can SAY when a status is not
		   one this build ranks. Without that word the section looks like every
		   other lifecycle heading while sitting in a position nothing on the
		   server ever chose — the silent mis-group, made visible. */
		return [...known, ...unknown].map((status) => ({
			status,
			known: STATUS_ORDER.includes(status),
			rows: buckets.get(status) ?? [],
		}));
	}, [projects]);

	const title =
		view.name === "detail"
			? (detail?.project.name ?? "project")
			: view.name === "create"
				? "new project"
				: view.name === "edit"
					? "edit project"
					: view.name === "link"
						? "link a session"
						: view.name === "milestone"
							? view.editing === null
								? "add milestone"
								: "edit milestone"
							: view.name === "confirm"
								? "delete project"
								: "projects";

	/* The milestone the editor is on (`null` while ADDING). Held in a local rather
	   than read off `view` inside the JSX's callbacks: `view` is a discriminated
	   union, and a property read inside a callback loses the narrowing. */
	const editingMilestone = view.name === "milestone" ? view.editing : null;

	/* The link picker's rows: the daemon's sessions minus the ones this project
	   already carries. Linking one it already has is a server-side no-op, and
	   offering it would read as a tap that did nothing. */
	const linkRows = useMemo(() => {
		if (view.name !== "link" || candidates === null) return [];
		const linked = new Set(
			detail && detail.project.id === view.key ? detail.project.sessions : [],
		);
		return candidates.filter((session) => !linked.has(session.session_id));
	}, [view, candidates, detail]);

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
									<h3
										className={cn(
											"px-3 py-1 text-meta font-medium",
											group.known ? "text-ink-muted" : "text-warning",
										)}
									>
										{group.known ? group.status : `${group.status} (unknown status)`}
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
					{/* The receipt for the write that just landed. It lives in the DETAIL
					    view as well as the browse view because a milestone edit, a link
					    and an unlink all land here — a confirmation the reader never
					    sees is not a confirmation. */}
					{notice ? (
						<p role="status" className="px-3 pb-1 text-meta text-ink-muted">
							{notice}
						</p>
					) : null}
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
								<div className="flex items-start gap-2">
									<p className="min-w-0 flex-1 text-meta text-ink-muted">
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
									{/* The edit form is the PATCH vocabulary reached from the phone:
									    before it, a row could be created and deleted but never
									    changed. */}
									<Button
										variant="outline"
										className="shrink-0"
										disabled={busy}
										onClick={() => openEdit(detail.project)}
									>
										edit
									</Button>
								</div>
								{/* ONE refusal line for the whole detail view (toggles, unlink,
								    milestone writes): it sits above every section that can raise
								    one, so the sentence is never filed under a section the reader
								    did not touch. */}
								{formError ? (
									<p role="alert" className="break-words text-meta text-danger">
										{formError}
									</p>
								) : null}
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
								{detail.project.milestones.length === 0 ? (
									<p className="text-body-sm text-ink-dim">no milestones</p>
								) : (
									detail.project.milestones.map((milestone) => (
										<div key={milestone.name} className="flex items-center gap-1">
											<button
												type="button"
												disabled={busy}
												aria-pressed={milestone.completed_at !== null}
												onClick={() => void toggleMilestone(detail.project, milestone)}
												className="flex min-h-11 min-w-0 flex-1 items-center gap-2 rounded-sm px-1 text-left active:bg-surface disabled:opacity-50"
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
											{/* The date and the removal live in the milestone editor rather
											    than as two more controls on this row: the row's own control
											    stays the COMPLETION toggle, and stacking three tap targets
											    on a 44px row is how a thumb presses the wrong one. */}
											<button
												type="button"
												disabled={busy}
												aria-label={`edit ${milestone.name}`}
												onClick={() => openMilestone(detail.project.id, milestone)}
												className="min-h-11 shrink-0 rounded-sm px-2 text-meta text-ink-muted active:bg-surface disabled:opacity-50"
											>
												edit
											</button>
										</div>
									))
								)}
								<Button
									variant="outline"
									className="mt-1"
									disabled={busy}
									onClick={() => openMilestone(detail.project.id, null)}
								>
									add milestone
								</Button>
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
												className="flex min-h-11 items-center gap-2 px-1"
											>
												<span className="min-w-0 flex-1 truncate text-body-sm text-ink">
													{row.title || row.session_id}
												</span>
												<span className={cn("shrink-0 text-meta", state.ink)}>
													{state.word}
												</span>
												{/* Unlinking is this row's own act: the link family was the
												    one thing the phone could not do at all, and a
												    session linked by mistake had no way out. */}
												<button
													type="button"
													disabled={busy}
													aria-label={`unlink ${row.title || row.session_id}`}
													onClick={() =>
														void unlinkSession(detail.project.id, row.session_id)
													}
													className="min-h-11 shrink-0 rounded-sm px-2 text-meta text-ink-muted active:bg-surface disabled:opacity-50"
												>
													unlink
												</button>
											</div>
										);
									})
								)}
								<Button
									variant="outline"
									className="mt-1"
									disabled={busy}
									onClick={() => {
										setFormError("");
										setView({ name: "link", key: detail.project.id });
									}}
								>
									link a session
								</Button>
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
							className={FIELD_CLASS}
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
							/* The store's own DESCRIPTION_MAX. The form used to stop the
							   reader at 240, which silently made a phone-created
							   description shorter than the same field on every other
							   surface. */
							maxLength={2000}
							className={FIELD_CLASS}
						/>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">status</span>
						<select
							value={createStatus}
							onChange={(event) => setCreateStatus(event.target.value)}
							className={FIELD_CLASS}
						>
							{/* The daemon's own status list, not a second copy of it: the
							    same generated order the board groups by. */}
							{STATUS_ORDER.map((status) => (
								<option key={status} value={status}>
									{status}
								</option>
							))}
						</select>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">tags (optional)</span>
						<input
							value={createTags}
							onChange={(event) => setCreateTags(event.target.value)}
							placeholder="payments, q4"
							spellCheck={false}
							autoCapitalize="off"
							autoCorrect="off"
							className={FIELD_CLASS}
						/>
						<span className="text-meta text-ink-dim">
							lowercase letters, digits, underscore and hyphen; separated by commas
						</span>
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

			{view.name === "edit" && draft !== null ? (
				<div className="flex flex-col gap-3 px-3 pb-4">
					<BackRow
						label="project"
						onClick={() => setView({ name: "detail", key: view.key })}
					/>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">name</span>
						<input
							value={draft.name}
							onChange={(event) => setDraft({ ...draft, name: event.target.value })}
							spellCheck={false}
							autoCapitalize="off"
							autoCorrect="off"
							className={FIELD_CLASS}
						/>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">description</span>
						<input
							value={draft.description}
							onChange={(event) => setDraft({ ...draft, description: event.target.value })}
							/* The store's own DESCRIPTION_MAX, the same bound the create
							   form carries — an edit form with a shorter cap could not
							   append a word to a row another surface wrote. */
							maxLength={2000}
							className={FIELD_CLASS}
						/>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">status</span>
						<select
							value={draft.status}
							onChange={(event) => setDraft({ ...draft, status: event.target.value })}
							className={FIELD_CLASS}
						>
							{/* A status this build does not rank rides as its own option
							    rather than being dropped: a <select> whose value matches no
							    option SHOWS THE FIRST ONE, so rewriting the options without
							    it would silently turn "blocked" into "planning" on save. */}
							{(STATUS_ORDER.includes(draft.status)
								? STATUS_ORDER
								: [...STATUS_ORDER, draft.status]
							).map((status) => (
								<option key={status} value={status}>
									{status}
								</option>
							))}
						</select>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">tags</span>
						<input
							value={draft.tags}
							onChange={(event) => setDraft({ ...draft, tags: event.target.value })}
							placeholder="payments, q4"
							spellCheck={false}
							autoCapitalize="off"
							autoCorrect="off"
							className={FIELD_CLASS}
						/>
						<span className="text-meta text-ink-dim">
							lowercase letters, digits, underscore and hyphen; separated by commas
						</span>
					</label>
					<div className="flex gap-2">
						<label className="flex flex-1 flex-col gap-1">
							<span className="text-body-sm text-ink-muted">start date</span>
							<input
								type="date"
								value={draft.start_date}
								onChange={(event) =>
									setDraft({ ...draft, start_date: event.target.value })
								}
								className={FIELD_CLASS}
							/>
						</label>
						<label className="flex flex-1 flex-col gap-1">
							<span className="text-body-sm text-ink-muted">target date</span>
							<input
								type="date"
								value={draft.target_date}
								onChange={(event) =>
									setDraft({ ...draft, target_date: event.target.value })
								}
								className={FIELD_CLASS}
							/>
						</label>
					</div>
					<div className="flex gap-2">
						<label className="flex flex-1 flex-col gap-1">
							<span className="text-body-sm text-ink-muted">estimate</span>
							<input
								type="number"
								inputMode="decimal"
								step="any"
								min="0"
								value={draft.estimate}
								onChange={(event) =>
									setDraft({ ...draft, estimate: event.target.value })
								}
								className={FIELD_CLASS}
							/>
							{/* Said out loud because the field really cannot do it: the store's
							    apply arm ignores a null estimate, so an emptied box means "keep
							    the estimate", never "clear it". */}
							<span className="text-meta text-ink-dim">
								a new estimate replaces the old; it cannot be cleared from here
							</span>
						</label>
						<label className="flex flex-1 flex-col gap-1">
							<span className="text-body-sm text-ink-muted">unit</span>
							<select
								value={draft.estimate_unit}
								onChange={(event) =>
									setDraft({ ...draft, estimate_unit: event.target.value })
								}
								className={FIELD_CLASS}
							>
								<option value="points">points</option>
								<option value="days">days</option>
							</select>
						</label>
					</div>
					{formError ? (
						<p role="alert" className="text-body-sm break-words text-danger">
							{formError}
						</p>
					) : null}
					<div className="flex gap-2">
						<Button
							variant="primary"
							disabled={busy || draft.name.trim() === ""}
							aria-busy={busy ? true : undefined}
							onClick={() => void submitEdit(view.key)}
						>
							{busy ? "saving…" : "save"}
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

			{view.name === "milestone" && milestoneDraft !== null ? (
				<div className="flex flex-col gap-3 px-3 pb-4">
					<BackRow
						label="project"
						onClick={() => setView({ name: "detail", key: view.key })}
					/>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">name</span>
						<input
							value={milestoneDraft.name}
							onChange={(event) =>
								setMilestoneDraft({ ...milestoneDraft, name: event.target.value })
							}
							placeholder="e.g. beta cut"
							disabled={editingMilestone !== null}
							spellCheck={false}
							className={cn(FIELD_CLASS, editingMilestone !== null ? "text-ink-muted" : "")}
						/>
						<span className="text-meta text-ink-dim">
							{editingMilestone === null
								? "a name that does not exist yet is added"
								: "the name is the milestone's key; milestones are not renamed here"}
						</span>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-body-sm text-ink-muted">target date (optional)</span>
						<input
							type="date"
							value={milestoneDraft.target_date}
							onChange={(event) =>
								setMilestoneDraft({ ...milestoneDraft, target_date: event.target.value })
							}
							className={FIELD_CLASS}
						/>
						{/* Leaving the box empty is the whole "clear it" affordance: `""` is
						    the spelling the daemon's tri-state reads as CLEARED, so an
						    emptied date input is not the same act as never having set one. */}
					</label>
					{formError ? (
						<p role="alert" className="text-body-sm break-words text-danger">
							{formError}
						</p>
					) : null}
					<div className="flex flex-wrap gap-2">
						<Button
							variant="primary"
							disabled={
								busy || (editingMilestone === null && milestoneDraft.name.trim() === "")
							}
							aria-busy={busy ? true : undefined}
							onClick={() => void submitMilestone(view.key, editingMilestone)}
						>
							{busy ? "saving…" : editingMilestone === null ? "add" : "save"}
						</Button>
						{editingMilestone !== null ? (
							<Button
								variant="danger"
								disabled={busy}
								onClick={() => void removeMilestone(view.key, editingMilestone)}
							>
								remove
							</Button>
						) : null}
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

			{view.name === "link" ? (
				<div className="flex flex-col gap-1 px-3 pb-4">
					<BackRow
						label="project"
						onClick={() => setView({ name: "detail", key: view.key })}
					/>
					{linkError ? (
						<ErrorBlock
							message={linkError}
							/* Re-entering the view is the retry: the candidates effect keys
							   on the view object, and this is a new one. */
							onRetry={() => setView({ name: "link", key: view.key })}
						/>
					) : candidates === null ? (
						<p className="px-1 py-2 text-body-sm text-ink-dim">loading sessions…</p>
					) : linkRows.length === 0 ? (
						<p className="px-1 py-2 text-body-sm text-ink-dim">
							no other sessions to link
						</p>
					) : (
						<div className="flex flex-col">
							{linkRows.map((session) => (
								<button
									key={session.session_id}
									type="button"
									disabled={busy}
									onClick={() => void linkSession(view.key, session.session_id)}
									className="flex min-h-11 w-full items-center gap-2 rounded-sm px-1 text-left active:bg-surface disabled:opacity-50"
								>
									<span className="min-w-0 flex-1 truncate text-body-sm text-ink">
										{session.conversation_name || session.session_id}
									</span>
									<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
										{session.session_id}
									</span>
								</button>
							))}
						</div>
					)}
					{formError ? (
						<p role="alert" className="text-body-sm break-words text-danger">
							{formError}
						</p>
					) : null}
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
