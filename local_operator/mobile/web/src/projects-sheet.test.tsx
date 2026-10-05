// @vitest-environment happy-dom
//
// The Projects sheet: the store's answers, rendered. Every test drives the
// sheet the way the surface is actually used — open it, tap a row, toggle a
// milestone, submit a form — against mocked API calls whose *shapes* are the
// daemon's real ones (the desktop wire models' field sets).
//
// Three behaviours are the reason this file exists rather than trusting the
// screenshots:
//
//   * the LIST and the BOARD are two readings of ONE daemon-ordered array —
//     neither re-sorts, and a status with no rows contributes no section
//     header (an empty heading claims a column that does not exist);
//   * a mutation is followed by a RE-READ (the list is re-fetched; the detail
//     is replaced by the write's own answer), never by a local guess — a
//     regression that patched local state would keep passing every rendering
//     assertion and still drift from the store;
//   * a refusal renders the daemon's own sentence, in place, with the form or
//     the row still standing (the reader's input is not thrown away).
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./api";
import { ProjectsSheet } from "./components/projects-sheet";
import { SESSION_LINK_CAP } from "./projects-status.generated";
/* The populated and board states are pinned to VERBATIM relay bodies captured
   over HTTP against an isolated config root (`projects.list.json` / 
   `projects.detail.json`), and the refusals to the daemon's own error
   envelopes (`projects.refusals.json`). Hand-written objects cannot catch a
   contract change the way a captured one does. */
import detailBody from "./fixtures/projects.detail.json";
import listBody from "./fixtures/projects.list.json";
import refusalBodies from "./fixtures/projects.refusals.json";
import writeBodies from "./fixtures/projects.writes.json";
import type { ProjectLinkedSession, ProjectSummary, ProjectView, SessionSummary } from "./types";

const getProjects = vi.fn();
const getProject = vi.fn();
const createProject = vi.fn();
const deleteProject = vi.fn();
const setProjectMilestone = vi.fn();
const removeProjectMilestone = vi.fn();
const patchProject = vi.fn();
const linkProjectSession = vi.fn();
const unlinkProjectSession = vi.fn();
const getSessions = vi.fn();

/** One refusal, built from the daemon's captured envelope — status, machine
    code and sentence exactly as the wire carried them. */
function refusal(name: keyof typeof refusalBodies): HttpError {
	const body = refusalBodies[name];
	return new HttpError(body.status, body.error, body.code);
}

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getProjects: (...args: unknown[]) => getProjects(...args),
		getProject: (...args: unknown[]) => getProject(...args),
		createProject: (...args: unknown[]) => createProject(...args),
		deleteProject: (...args: unknown[]) => deleteProject(...args),
		setProjectMilestone: (...args: unknown[]) => setProjectMilestone(...args),
		removeProjectMilestone: (...args: unknown[]) => removeProjectMilestone(...args),
		patchProject: (...args: unknown[]) => patchProject(...args),
		linkProjectSession: (...args: unknown[]) => linkProjectSession(...args),
		unlinkProjectSession: (...args: unknown[]) => unlinkProjectSession(...args),
		getSessions: (...args: unknown[]) => getSessions(...args),
	};
});

function summary(over: Partial<ProjectSummary> = {}): ProjectSummary {
	return {
		id: "p1",
		name: "payments-migration",
		description: "",
		status: "active",
		tags: [],
		start_date: null,
		target_date: null,
		completed_at: null,
		estimate: null,
		estimate_unit: "points",
		milestones_completed: 0,
		milestones_total: 0,
		sessions: 0,
		live_sessions: 0,
		progress_stale: true,
		progress_updated_at: null,
		updated_at: 0,
		...over,
	};
}

function view(over: Partial<ProjectView> = {}): ProjectView {
	return {
		id: "p1",
		name: "payments-migration",
		description: "",
		status: "active",
		progress: "",
		progress_updated_at: null,
		progress_reported_by: "",
		progress_stale: true,
		tags: [],
		sessions: [],
		created_at: 0,
		updated_at: 0,
		start_date: null,
		target_date: null,
		completed_at: null,
		estimate: null,
		estimate_unit: "points",
		milestones: [],
		...over,
	};
}

function link(over: Partial<ProjectLinkedSession> = {}): ProjectLinkedSession {
	return {
		session_id: "4e92693767fa",
		exists: true,
		title: null,
		created_at: null,
		archived: false,
		runtime: { state: "stopped" },
		subagents: null,
		todos: null,
		...over,
	};
}

function renderSheet() {
	return render(<ProjectsSheet open onClose={() => {}} />);
}

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

describe("browse", () => {
	it("shows the empty-state sentence and opens the create form from its button", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		renderSheet();
		expect(await screen.findByText("no projects yet")).toBeTruthy();
		expect(screen.getByText(/create a project and link this session/)).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "new project" }));
		expect(screen.getByPlaceholderText("e.g. payments-migration")).toBeTruthy();
	});

	it("renders the daemon's order, then groups the same rows under status headings on the board", async () => {
		getProjects.mockResolvedValue({
			projects: [
				summary({ milestones_total: 3, milestones_completed: 1, sessions: 4, live_sessions: 2 }),
				summary({ id: "p2", name: "audit", status: "paused" }),
			],
		});
		renderSheet();
		const rows = await screen.findAllByRole("button", { name: /payments-migration|audit/ });
		// The list keeps the daemon's own order — it never re-sorts.
		expect(rows.map((row) => row.textContent)).toEqual([
			expect.stringContaining("payments-migration"),
			expect.stringContaining("audit"),
		]);
		// The row carries the counts the daemon computed, not ones this view derived.
		expect(screen.getByText(/1\/3 milestones/)).toBeTruthy();
		expect(screen.getByText(/4 sessions · 2 live/)).toBeTruthy();

		fireEvent.click(screen.getByRole("button", { name: "board" }));
		expect(screen.getByRole("heading", { name: "active" })).toBeTruthy();
		expect(screen.getByRole("heading", { name: "paused" })).toBeTruthy();
		// A status with no rows contributes no heading: no column is claimed.
		expect(screen.queryByRole("heading", { name: "archived" })).toBeNull();
	});

	it("groups the lifecycle statuses under their own headings, in the daemon's rank order", async () => {
		getProjects.mockResolvedValue({
			projects: [
				summary({ id: "p1", name: "review-thing", status: "qa" }),
				summary({ id: "p2", name: "plan-thing", status: "planning" }),
				summary({ id: "p3", name: "ship-thing", status: "validation" }),
			],
		});
		renderSheet();
		await screen.findByText("review-thing");
		fireEvent.click(screen.getByRole("button", { name: "board" }));
		// The headings follow STATUS_ORDER (the daemon's rank): planning, qa and
		// validation each get their own section — none is dropped into the
		// trailing unknown bucket.
		const headings = screen
			.getAllByRole("heading")
			.map((heading) => heading.textContent)
			.filter((text) =>
				["planning", "active", "qa", "validation", "paused", "done", "archived"].includes(
					text ?? "",
				),
			);
		expect(headings).toEqual(["planning", "qa", "validation"]);
	});

	it("keeps the surface mounted on a fetch failure and retries in place", async () => {
		getProjects.mockRejectedValueOnce(new HttpError(503, "Timed out waiting for the projects registry lock", "project_store_busy"));
		renderSheet();
		expect(await screen.findByText(/Timed out waiting for the projects registry lock/)).toBeTruthy();

		getProjects.mockResolvedValueOnce({ projects: [summary()] });
		fireEvent.click(screen.getByRole("button", { name: "retry" }));
		expect(await screen.findByText("payments-migration")).toBeTruthy();
	});

	it("renders an unreachable daemon as prose, not the browser's TypeError", async () => {
		getProjects.mockRejectedValueOnce(new TypeError("Failed to fetch"));
		renderSheet();
		// The browser's own "Failed to fetch" explains nothing (round-1 design, D3).
		expect(await screen.findByText("could not reach the daemon")).toBeTruthy();
		expect(screen.queryByText(/Failed to fetch/)).toBeNull();

		getProjects.mockResolvedValueOnce({ projects: [] });
		fireEvent.click(screen.getByRole("button", { name: "retry" }));
		expect(await screen.findByText("no projects yet")).toBeTruthy();
	});

	it("re-reads the store on every open, so another surface's write is never missed", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		const { rerender } = render(<ProjectsSheet open={false} onClose={() => {}} />);
		rerender(<ProjectsSheet open onClose={() => {}} />);
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(1));
		rerender(<ProjectsSheet open={false} onClose={() => {}} />);
		rerender(<ProjectsSheet open onClose={() => {}} />);
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(2));
	});
});

describe("detail", () => {
	it("shows progress with its age, milestones and linked sessions, and toggles a milestone", async () => {
		const reported = Date.now() / 1000 - 7200;
		getProjects.mockResolvedValue({
			projects: [summary({ sessions: 1, live_sessions: 1, milestones_total: 1 })],
		});
		getProject.mockResolvedValue({
			project: view({
				progress: "cutover done",
				progress_updated_at: reported,
				progress_reported_by: "operator",
				progress_stale: false,
				milestones: [{ name: "beta cut", target_date: "2026-10-01", completed_at: null, status: "upcoming" }],
			}),
			links: [link({ title: "Payments cutover", runtime: { state: "live" } })],
		});
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		await waitFor(() => expect(getProject).toHaveBeenCalledWith("p1"));
		expect(await screen.findByText("cutover done")).toBeTruthy();
		// Reporter before any stale marker (round-1 UX, U2).
		expect(screen.getByText("reported 2h ago by operator")).toBeTruthy();
		expect(screen.getByText("Payments cutover")).toBeTruthy();
		expect(screen.getByText("live")).toBeTruthy();

		setProjectMilestone.mockResolvedValue({
			ok: true,
			project: view({
				milestones: [
					{ name: "beta cut", target_date: "2026-10-01", completed_at: "2026-09-26", status: "completed" },
				],
			}),
		});
		// The row's TOGGLE, not the `edit …` control beside it: the toggle's own
		// accessible name starts with the milestone's name.
		fireEvent.click(screen.getByRole("button", { name: /^beta cut/ }));
		await waitFor(() =>
			expect(setProjectMilestone).toHaveBeenCalledWith("p1", { name: "beta cut", completed: true }),
		);
		// The sheet re-renders from the WRITE'S OWN ANSWER (the returned view),
		// not from a local guess at what changed.
		await waitFor(() =>
			expect(screen.getByRole("button", { name: /^beta cut/ }).getAttribute("aria-pressed")).toBe("true"),
		);
		// ... and the list is re-read, because the counts it shows moved.
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(2));
	});

	it("toggles a completed milestone back off", async () => {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({
			project: view({
				milestones: [
					{ name: "beta cut", target_date: null, completed_at: "2026-09-20", status: "completed" },
				],
			}),
			links: [],
		});
		setProjectMilestone.mockResolvedValue({
			ok: true,
			project: view({
				milestones: [{ name: "beta cut", target_date: null, completed_at: null, status: "upcoming" }],
			}),
		});
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: /^beta cut/ }));
		await waitFor(() =>
			expect(setProjectMilestone).toHaveBeenCalledWith("p1", { name: "beta cut", completed: false }),
		);
	});

	it("keeps the daemon's refusal on the surface when a toggle is refused", async () => {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({
			project: view({
				milestones: [{ name: "beta cut", target_date: null, completed_at: null, status: "upcoming" }],
			}),
			links: [],
		});
		setProjectMilestone.mockRejectedValue(
			new HttpError(409, "project 'payments-migration' was written by a newer local-operator (schema 2); update this build to change it", "project_schema_newer"),
		);
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: /^beta cut/ }));
		expect(await screen.findByText(/update this build/)).toBeTruthy();
	});

	it("binds a stale report to the report, not the reporter, and states an empty one once", async () => {
		const reported = Date.now() / 1000 - 5 * 86400;
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({
			project: view({
				progress: "flights booked",
				progress_updated_at: reported,
				progress_reported_by: "operator",
				progress_stale: true,
			}),
			links: [],
		});
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		// "reported 5d ago by operator · stale" — never "stale by operator".
		expect(await screen.findByText("reported 5d ago by operator · stale")).toBeTruthy();

		getProject.mockResolvedValue({ project: view({ progress: "" }), links: [] });
		fireEvent.click(screen.getByRole("button", { name: "projects" }));
		fireEvent.click(await screen.findByText("payments-migration"));
		expect(await screen.findByText("no progress reported yet")).toBeTruthy();
		// The age line belongs to a REPORT — with none, it would only restate
		// the line above in the negative (round-1 UX, U3).
		expect(screen.queryByText("none recorded")).toBeNull();
	});

	it("shows the detail fetch's failure with a working retry", async () => {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockRejectedValueOnce(new HttpError(404, "no project with id or name 'p1'", "project_not_found"));
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		expect(await screen.findByText(/no project with id or name 'p1'/)).toBeTruthy();

		getProject.mockResolvedValueOnce({ project: view(), links: [] });
		fireEvent.click(screen.getByRole("button", { name: "retry" }));
		expect(await screen.findByText(/no progress reported yet/)).toBeTruthy();
	});
});

describe("create", () => {
	it("creates from the form, then re-reads the list and reports the receipt", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		createProject.mockResolvedValue({ ok: true, project: summary({ name: "alpha" }) });
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));
		fireEvent.change(screen.getByPlaceholderText("e.g. payments-migration"), {
			target: { value: "alpha" },
		});
		fireEvent.click(screen.getByRole("button", { name: "create" }));
		await waitFor(() =>
			expect(createProject).toHaveBeenCalledWith({
				name: "alpha",
				description: "",
				// The two keys the control used to drop on the floor: the create
				// body's own status and tags vocabulary.
				status: "active",
				tags: [],
			}),
		);
		expect(await screen.findByText("created alpha")).toBeTruthy();
		// Back at the browse view, list re-read.
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(2));
	});

	it("starts a fresh form on every entry, so a cancelled draft does not come back", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));
		const field = () => screen.getByPlaceholderText("e.g. payments-migration") as HTMLInputElement;
		fireEvent.change(field(), { target: { value: "abandoned-draft" } });
		fireEvent.click(screen.getByRole("button", { name: "cancel" }));
		fireEvent.click(screen.getByRole("button", { name: "new project" }));
		// The reader did not type it this time (round-1 UX, U1).
		expect(field().value).toBe("");
	});

	it("keeps the form and the daemon's sentence when the name is taken", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		createProject.mockRejectedValue(
			new HttpError(409, "project 'alpha' already exists", "project_name_exists"),
		);
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));
		fireEvent.change(screen.getByPlaceholderText("e.g. payments-migration"), {
			target: { value: "alpha" },
		});
		fireEvent.click(screen.getByRole("button", { name: "create" }));
		expect(await screen.findByText("project 'alpha' already exists")).toBeTruthy();
		// The input the reader typed still stands.
		expect((screen.getByPlaceholderText("e.g. payments-migration") as HTMLInputElement).value).toBe("alpha");
	});
});

describe("delete", () => {
	it("confirms by name, deletes, and reports the receipt from the browse view", async () => {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({ project: view(), links: [] });
		deleteProject.mockResolvedValue({ ok: true, deleted: true });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: "delete project" }));
		// The confirmation names the project (the name is the confirmation the
		// daemon requires) and states the one guarantee: sessions survive it.
		expect(screen.getByText("payments-migration", { selector: "span" })).toBeTruthy();
		expect(screen.getByText(/is removed permanently/)).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "delete" }));
		await waitFor(() =>
			expect(deleteProject).toHaveBeenCalledWith("p1", "payments-migration"),
		);
		expect(await screen.findByText("deleted payments-migration")).toBeTruthy();
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(2));
	});

	it("keeps the confirmation standing when the daemon refuses", async () => {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({ project: view(), links: [] });
		deleteProject.mockRejectedValue(
			new HttpError(409, "project 'payments-migration' was written by a newer local-operator", "project_schema_newer"),
		);
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: "delete project" }));
		fireEvent.click(screen.getByRole("button", { name: "delete" }));
		expect(await screen.findByText(/newer local-operator/)).toBeTruthy();
		expect(screen.getByRole("button", { name: "delete" })).toBeTruthy();
	});
});

/* ---- the captured relay bodies ------------------------------------------
 *
 * The populated and board states are pinned to the daemon's OWN bytes
 * (`src/fixtures/projects.*.json`, captured over HTTP against an isolated
 * config root), not to a hand-built object: a field the relay renames or a
 * shape it changes shows up here as a rendering failure rather than as two
 * hand-written literals drifting apart in the same direction.
 */
describe("captured relay bodies", () => {
	it("renders the daemon's own listing, and groups the same rows on the board", async () => {
		getProjects.mockResolvedValue(listBody);
		renderSheet();
		expect(await screen.findByText("payments-migration")).toBeTruthy();
		// The row's numbers are the daemon's, off the wire.
		expect(screen.getByText(/1 session · 1 live/)).toBeTruthy();
		expect(screen.getByText(/1\/4 milestones/)).toBeTruthy();

		fireEvent.click(screen.getByRole("button", { name: "board" }));
		// Board order follows the daemon's rank: active, then qa, then done.
		expect(screen.getAllByRole("heading").map((heading) => heading.textContent)).toEqual([
			"active",
			"qa",
			"done",
		]);
	});

	it("renders a captured detail body: progress, milestones and the linked row", async () => {
		getProjects.mockResolvedValue(listBody);
		getProject.mockResolvedValue(detailBody);
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));

		expect(await screen.findByText("cutover done")).toBeTruthy();
		expect(screen.getByText(/reported .* by operator/)).toBeTruthy();
		// A completed milestone arrives pressed; an overdue one keeps its date.
		expect(
			screen.getByRole("button", { name: /^schema frozen/ }).getAttribute("aria-pressed"),
		).toBe("true");
		expect(screen.getByText("beta cut")).toBeTruthy();
		expect(screen.getByText("2026-09-30")).toBeTruthy();
		// The linked session row, named by its id when the record has no title.
		expect(screen.getByText("4e92693767fa")).toBeTruthy();
	});

	it("pins the write answers the sheet's re-read policy depends on", () => {
		/* A create/patch/link/unlink answer is a SUMMARY — counts, no milestone
		   list, no linked-session rows — which is exactly why the sheet RE-READS
		   the composed view instead of patching one from a document that never
		   carried it. A milestone write answers with the whole view, which is
		   why that one is applied directly. Those bytes are the daemon's own. */
		for (const key of ["create", "patch", "link", "unlink"] as const) {
			const body = writeBodies[key];
			expect(body.ok).toBe(true);
			expect(body.project).toHaveProperty("milestones_total");
			expect(body.project).not.toHaveProperty("milestones");
		}
		expect(writeBodies.milestone_add.project).toHaveProperty("milestones");
		expect(writeBodies.milestone_add.project).not.toHaveProperty("milestones_total");
	});
});

/* ---- edit --------------------------------------------------------------- */
describe("edit", () => {
	function detail(over: Partial<ProjectView> = {}) {
		return { project: view(over), links: [] as ProjectLinkedSession[] };
	}

	async function openEditForm() {
		getProjects.mockResolvedValue({ projects: [summary()] });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: "edit project" }));
	}

	it("sends the form's fields and re-reads the detail the reader lands on", async () => {
		getProject.mockResolvedValue(detail({ description: "old line", tags: ["q4"] }));
		patchProject.mockResolvedValue({ ok: true, project: summary({ name: "renamed" }) });
		await openEditForm();

		// Seeded from the row the daemon served, not from a blank form.
		expect((screen.getByLabelText(/^name/) as HTMLInputElement).value).toBe("payments-migration");
		expect((screen.getByLabelText(/^tags/) as HTMLInputElement).value).toBe("q4");
		expect((screen.getByLabelText(/^status/) as HTMLSelectElement).value).toBe("active");

		fireEvent.change(screen.getByLabelText(/^name/), { target: { value: "payments-cutover" } });
		fireEvent.change(screen.getByLabelText(/^status/), { target: { value: "qa" } });
		fireEvent.change(screen.getByLabelText(/^tags/), { target: { value: "q4, payments" } });
		fireEvent.change(screen.getByLabelText(/^target date/), { target: { value: "2026-11-30" } });
		fireEvent.change(screen.getByLabelText(/^estimate/), { target: { value: "8" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		await waitFor(() =>
			expect(patchProject).toHaveBeenCalledWith("p1", {
				name: "payments-cutover",
				description: "old line",
				status: "qa",
				tags: ["q4", "payments"],
				// An untouched date travels as "" (NOT omitted): "" is the
				// daemon's own clear spelling, and the form's empty box means
				// cleared.
				start_date: "",
				target_date: "2026-11-30",
				estimate: 8,
				estimate_unit: "points",
			}),
		);
		expect(await screen.findByText("updated renamed")).toBeTruthy();
		// The detail the reader lands on is a fresh read, not the summary.
		await waitFor(() => expect(getProject).toHaveBeenCalledTimes(2));
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(2));
	});

	it("clears what the store can clear and OMITS what it cannot", async () => {
		getProject.mockResolvedValue(detail({ description: "old line", estimate: 5 }));
		patchProject.mockResolvedValue({ ok: true, project: summary() });
		await openEditForm();

		// An emptied description is a clear (the apply arm accepts "").
		fireEvent.change(screen.getByLabelText(/^description/), { target: { value: "" } });
		// An emptied estimate is NOT: the store ignores a null estimate, so the
		// key must be omitted rather than sent as null (a silent no-op that
		// would leave 5 on the row while the form showed it blank).
		fireEvent.change(screen.getByLabelText(/^estimate/), { target: { value: "" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		await waitFor(() => expect(patchProject).toHaveBeenCalledTimes(1));
		const [, body] = patchProject.mock.calls[0] as [string, Record<string, unknown>];
		expect(body.description).toBe("");
		expect("estimate" in body).toBe(false);
	});

	it("keeps the form and shows the daemon's sentence when the store refuses", async () => {
		getProject.mockResolvedValue(detail());
		patchProject.mockRejectedValue(refusal("patch_invalid_date"));
		await openEditForm();

		fireEvent.change(screen.getByLabelText(/^target date/), { target: { value: "2026-11-30" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		expect(await screen.findByText("target_date must be an ISO YYYY-MM-DD date")).toBeTruthy();
		// The reader's input is still there to fix.
		expect(screen.getByLabelText(/^name/)).toBeTruthy();
		expect(screen.queryByText("updated renamed")).toBeNull();
	});

	it("returns to the list when the project vanished under the write", async () => {
		getProject.mockResolvedValue(detail());
		// The captured 404 body; the name it echoes is the one the capture used.
		patchProject.mockRejectedValue(refusal("vanished_project"));
		await openEditForm();

		fireEvent.click(screen.getByRole("button", { name: "save" }));

		expect(await screen.findByText(/no project with id or name/)).toBeTruthy();
		// The list is re-read, which is what shows the store as it now is.
		await waitFor(() => expect(getProjects).toHaveBeenCalledTimes(2));
		expect(screen.getByRole("button", { name: "new project" })).toBeTruthy();
	});

	it("says it is saving while the write is in flight", async () => {
		getProject.mockResolvedValue(detail());
		let settle!: (value: unknown) => void;
		patchProject.mockReturnValue(
			new Promise((resolve) => {
				settle = resolve;
			}),
		);
		await openEditForm();
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		const saving = await screen.findByRole("button", { name: "saving…" });
		expect((saving as HTMLButtonElement).disabled).toBe(true);
		settle({ ok: true, project: summary() });
	});
});

/* ---- link / unlink ------------------------------------------------------ */
describe("link", () => {
	function detail(over: Partial<ProjectView> = {}, links: ProjectLinkedSession[] = []) {
		return { project: view(over), links };
	}

	function sessions(rows: { session_id: string; conversation_name: string }[]) {
		return { sessions: rows };
	}

	async function openPicker() {
		getProjects.mockResolvedValue({ projects: [summary()] });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: "link a session" }));
	}

	it("offers the daemon's sessions minus the ones already linked, and links the tap", async () => {
		getProject.mockResolvedValue(detail({ sessions: ["4e92693767fa"] }));
		getSessions.mockResolvedValue(
			sessions([
				{ session_id: "4e92693767fa", conversation_name: "already linked" },
				{ session_id: "3b1c0d9e8f77", conversation_name: "payments cutover" },
			]),
		);
		linkProjectSession.mockResolvedValue({ ok: true, project: summary() });
		await openPicker();

		expect(await screen.findByText("payments cutover")).toBeTruthy();
		// A session the project already carries is not offered: the daemon would
		// no-op the link, and a tap that does nothing reads as broken.
		expect(screen.queryByText("already linked")).toBeNull();

		fireEvent.click(screen.getByRole("button", { name: /payments cutover/ }));
		await waitFor(() =>
			expect(linkProjectSession).toHaveBeenCalledWith("p1", "3b1c0d9e8f77"),
		);
		// Back on the detail, re-read from the store.
		await waitFor(() => expect(getProject).toHaveBeenCalledTimes(2));
		expect(await screen.findByText("linked 3b1c0d9e8f77")).toBeTruthy();
	});

	it("keeps the picker standing and shows the daemon's refusal", async () => {
		getProject.mockResolvedValue(detail());
		getSessions.mockResolvedValue(sessions([{ session_id: "3b1c0d9e8f77", conversation_name: "x" }]));
		linkProjectSession.mockRejectedValue(refusal("link_cap_reached"));
		await openPicker();

		fireEvent.click(await screen.findByRole("button", { name: /x/ }));
		// The refusal a picker of well-formed ids can really meet: the cap.
		expect(await screen.findByText(/already has 64 linked sessions/)).toBeTruthy();
		expect(screen.getByRole("button", { name: /x/ })).toBeTruthy();
	});

	it("reports an unreachable session catalogue rather than an empty picker", async () => {
		getProject.mockResolvedValue(detail());
		getSessions.mockRejectedValue(new TypeError("Failed to fetch"));
		await openPicker();

		expect(await screen.findByText("could not reach the daemon")).toBeTruthy();
		expect(screen.queryByText("no other sessions to link")).toBeNull();
	});

	it("unlinks the tapped session and re-reads the composed view", async () => {
		getProject
			.mockResolvedValueOnce(detail({ sessions: ["4e92693767fa"] }, [link()]))
			.mockResolvedValue(detail({ sessions: [] }, []));
		unlinkProjectSession.mockResolvedValue({ ok: true, project: summary() });
		getProjects.mockResolvedValue({ projects: [summary()] });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));

		fireEvent.click(await screen.findByRole("button", { name: /^unlink/ }));
		await waitFor(() =>
			expect(unlinkProjectSession).toHaveBeenCalledWith("p1", "4e92693767fa"),
		);
		expect(await screen.findByText("no linked sessions")).toBeTruthy();
	});
});

/* ---- milestones --------------------------------------------------------- */
describe("milestones", () => {
	const upcoming = {
		name: "GA",
		target_date: "2026-11-15",
		completed_at: null,
		status: "upcoming" as const,
	};

	async function openDetail(rows: ProjectView["milestones"] = []) {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({ project: view({ milestones: rows }), links: [] });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
	}

	it("adds a milestone, name and date together", async () => {
		const added = { ...upcoming, name: "beta cut" };
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject
			.mockResolvedValueOnce({ project: view(), links: [] })
			.mockResolvedValue({ project: view({ milestones: [added] }), links: [] });
		setProjectMilestone.mockResolvedValue({ ok: true, project: view({ milestones: [added] }) });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: "add milestone" }));

		fireEvent.change(screen.getByLabelText(/^name/), { target: { value: "beta cut" } });
		fireEvent.change(screen.getByLabelText(/^target date/), { target: { value: "2026-11-15" } });
		fireEvent.click(screen.getByRole("button", { name: "add" }));

		await waitFor(() =>
			expect(setProjectMilestone).toHaveBeenCalledWith("p1", {
				name: "beta cut",
				target_date: "2026-11-15",
			}),
		);
		expect(await screen.findByText("added milestone beta cut")).toBeTruthy();
		expect(await screen.findByText("beta cut")).toBeTruthy();
	});

	it("clears an existing milestone's date from its editor (the name stays the key)", async () => {
		await openDetail([upcoming]);
		setProjectMilestone.mockResolvedValue({
			ok: true,
			project: view({ milestones: [{ ...upcoming, target_date: null }] }),
		});

		fireEvent.click(await screen.findByRole("button", { name: "edit GA" }));
		// A milestone is keyed by name: renaming here would ADD a second one.
		expect((screen.getByLabelText(/^name/) as HTMLInputElement).disabled).toBe(true);

		fireEvent.change(screen.getByLabelText(/^target date/), { target: { value: "" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		await waitFor(() =>
			expect(setProjectMilestone).toHaveBeenCalledWith("p1", {
				name: "GA",
				target_date: "",
			}),
		);
	});

	it("removes a milestone behind its own confirm, and refuses to save an empty new name", async () => {
		await openDetail([upcoming]);
		// An ADD with no name has nothing to key on: the button is inert.
		fireEvent.click(await screen.findByRole("button", { name: "add milestone" }));
		expect((screen.getByRole("button", { name: "add" }) as HTMLButtonElement).disabled).toBe(true);
		fireEvent.click(screen.getByRole("button", { name: "cancel" }));

		removeProjectMilestone.mockResolvedValue({ ok: true, project: view({ milestones: [] }) });
		getProject.mockResolvedValue({ project: view({ milestones: [] }), links: [] });
		fireEvent.click(await screen.findByRole("button", { name: "edit GA" }));

		/* REMOVAL IS TWO TAPS, and the first one sends NOTHING: it used to be a
		   `danger` button sitting beside `save`, with no confirm and no undo,
		   while deleting the project asks for a whole view (design round 6, D2). */
		fireEvent.click(screen.getByRole("button", { name: "remove milestone" }));
		expect(removeProjectMilestone).not.toHaveBeenCalled();
		expect(screen.getByText(/the project's sessions and its history are untouched/)).toBeTruthy();
		// Backing out is a real option, and it also sends nothing.
		fireEvent.click(screen.getByRole("button", { name: "keep it" }));
		expect(removeProjectMilestone).not.toHaveBeenCalled();
		fireEvent.click(screen.getByRole("button", { name: "remove milestone" }));
		fireEvent.click(screen.getByRole("button", { name: "remove" }));

		await waitFor(() => expect(removeProjectMilestone).toHaveBeenCalledWith("p1", "GA"));
		expect(await screen.findByText("removed milestone GA")).toBeTruthy();
		expect(await screen.findByText("no milestones")).toBeTruthy();
	});

	it("keeps the editor and the daemon's sentence when the store refuses", async () => {
		await openDetail([upcoming]);
		setProjectMilestone.mockRejectedValue(refusal("milestone_bad_date"));

		fireEvent.click(await screen.findByRole("button", { name: "edit GA" }));
		fireEvent.change(screen.getByLabelText(/^target date/), { target: { value: "2026-12-01" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		expect(
			await screen.findByText("milestone target_date must be an ISO YYYY-MM-DD date"),
		).toBeTruthy();
		expect(screen.getByRole("button", { name: "save" })).toBeTruthy();
	});
});

/* ---- create: the two fields the control used to drop -------------------- */
describe("create", () => {
	it("carries the chosen status and tags", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		createProject.mockResolvedValue({ ok: true, project: summary() });
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));

		fireEvent.change(screen.getByPlaceholderText("e.g. payments-migration"), {
			target: { value: "audit-sweep" },
		});
		fireEvent.change(screen.getByLabelText(/^status/), { target: { value: "qa" } });
		fireEvent.change(screen.getByPlaceholderText("payments, q4"), {
			target: { value: "audit, q4" },
		});
		fireEvent.click(screen.getByRole("button", { name: "create" }));

		await waitFor(() =>
			expect(createProject).toHaveBeenCalledWith({
				name: "audit-sweep",
				description: "",
				status: "qa",
				tags: ["audit", "q4"],
			}),
		);
	});

	it("shows the daemon's sentence when a tag is not in the store's grammar", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		createProject.mockRejectedValue(refusal("create_name_invalid"));
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));
		fireEvent.change(screen.getByPlaceholderText("e.g. payments-migration"), {
			target: { value: "audit sweep" },
		});
		fireEvent.click(screen.getByRole("button", { name: "create" }));

		expect(
			await screen.findByText(/must be 1-64 characters of letters, digits/),
		).toBeTruthy();
	});
});

/* ---- round-6 remediation -------------------------------------------------
 *
 * One test per finding from the review and design rounds, where a regression
 * would be silent: an accessible name that swallows its helper, a milestone the
 * phone could create but never remove, a destructive tap next to `save`, a
 * picker that would MOVE a filed link into the work set, a cap the reader only
 * learned about from a refusal, and an estimate that could be sent as a silent
 * no-op. Each one fails against the code as it stood before this round.
 */
describe("round-6 remediation", () => {
	const filed = "4e92693767fa";

	function candidate(over: Partial<SessionSummary> = {}): SessionSummary {
		return {
			session_id: "3b1c0d9e8f77",
			section: "active",
			conversation_name: "audit sweep",
			cwd: "/Users/x/audit",
			model_label: "opus",
			streaming: false,
			needs_attention: false,
			pending_kind: null,
			todos_open: 0,
			mtime: 0,
			...over,
		};
	}

	async function openDetail(over: Partial<ProjectView> = {}, links: ProjectLinkedSession[] = []) {
		getProjects.mockResolvedValue({ projects: [summary()] });
		getProject.mockResolvedValue({ project: view(over), links });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
	}

	it("D9: a helper is DESCRIPTION, not part of the control's name", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));

		/* Exact names: text inside a <label> becomes the control's accessible
		   name, so these used to read "tags (optional) lowercase letters, digits,
		   underscore and hyphen; separated by commas". */
		expect(screen.getByLabelText("name")).toBeTruthy();
		expect(screen.getByLabelText("description (optional)")).toBeTruthy();
		expect(screen.getByLabelText("status")).toBeTruthy();
		expect(screen.getByLabelText("tags (optional)")).toBeTruthy();
		expect(screen.queryByLabelText(/lowercase letters, digits/)).toBeNull();
		expect(
			screen.getByLabelText("tags (optional)").getAttribute("aria-describedby"),
		).toBe("create-tags-helper");
	});

	it("D1: the description is a textarea, in both forms", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		renderSheet();
		fireEvent.click(await screen.findByRole("button", { name: "new project" }));
		expect(screen.getByLabelText("description (optional)").tagName).toBe("TEXTAREA");
		// And the bound is the store's own DESCRIPTION_MAX, not the old 240.
		expect(
			(screen.getByLabelText("description (optional)") as HTMLTextAreaElement).maxLength,
		).toBe(2000);

		fireEvent.click(screen.getByRole("button", { name: "cancel" }));
		await openDetail();
		fireEvent.click(await screen.findByRole("button", { name: "edit project" }));
		expect(screen.getByLabelText("description").tagName).toBe("TEXTAREA");
	});

	it("m2: a non-finite estimate is refused with a sentence, never sent as null", async () => {
		/* `1e999` is what a number field hands back for a value that overflows a
		   double; `Number()` makes it Infinity and `JSON.stringify` writes that as
		   `null` — the silent no-op the estimate's own note warns about. */
		patchProject.mockResolvedValue({ ok: true, project: summary() });
		await openDetail();
		fireEvent.click(await screen.findByRole("button", { name: "edit project" }));
		fireEvent.change(screen.getByLabelText(/^estimate/), { target: { value: "1e999" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		expect(await screen.findByText("estimate must be a number")).toBeTruthy();
		expect(patchProject).not.toHaveBeenCalled();
	});

	it("m2b: an out-of-bounds estimate is SENT, not pre-empted by invented copy", async () => {
		/* 0 is a NUMBER, and the store owns the bound and its wording (its refusal
		   names ESTIMATE_MAX, which this client must not hand-copy). So the phone
		   sends it; the store's own sentence comes back through the refusal path
		   every other write already uses. */
		patchProject.mockResolvedValue({ ok: true, project: summary() });
		await openDetail();
		fireEvent.click(await screen.findByRole("button", { name: "edit project" }));
		fireEvent.change(screen.getByLabelText(/^estimate/), { target: { value: "0" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		await waitFor(() =>
			expect(patchProject).toHaveBeenCalledWith(
				"p1",
				expect.objectContaining({ estimate: 0 }),
			),
		);
	});

	it("M1: a milestone name with a slash is refused, and nothing is sent", async () => {
		await openDetail();
		fireEvent.click(await screen.findByRole("button", { name: "add milestone" }));
		fireEvent.change(screen.getByLabelText("name"), { target: { value: "ship/v2" } });

		expect(screen.getByText(/cannot contain a slash/)).toBeTruthy();
		expect((screen.getByRole("button", { name: "add" }) as HTMLButtonElement).disabled).toBe(true);
		fireEvent.click(screen.getByRole("button", { name: "add" }));
		expect(setProjectMilestone).not.toHaveBeenCalled();
	});

	it("M1b: a slash-named milestone says why it cannot be removed here", async () => {
		/* Created by another surface, so it EXISTS — and the relay's delete route
		   cannot address it. The honest control is the explanation, not a button
		   that 404s. */
		await openDetail({
			milestones: [
				{ name: "ship/v2", target_date: null, completed_at: null, status: "upcoming" },
			],
		});
		fireEvent.click(await screen.findByRole("button", { name: "edit ship/v2" }));

		expect(screen.getByText(/cannot contain a slash/)).toBeTruthy();
		expect(screen.queryByRole("button", { name: "remove milestone" })).toBeNull();
	});

	it("m3/D4: the picker hides FILED links and shows enough to choose", async () => {
		getProject.mockResolvedValue({
			project: view({ sessions: [], coordination_sessions: [filed] }),
			links: [],
		});
		getSessions.mockResolvedValue({
			sessions: [
				candidate({ session_id: filed, conversation_name: "filed one" }),
				candidate({ session_id: "3b1c0d9e8f77", conversation_name: "work one" }),
			],
		});
		getProjects.mockResolvedValue({ projects: [summary()] });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));
		fireEvent.click(await screen.findByRole("button", { name: "link a session" }));

		/* The filed link is NOT offered: linking it here would move it into the
		   work set under a "linked …" receipt, because the relay's link body
		   carries no role (m3). */
		expect(await screen.findByText("work one")).toBeTruthy();
		expect(screen.queryByText("filed one")).toBeNull();
		expect(screen.getByText(/1 session filed against this project is not offered/)).toBeTruthy();
		// D4: the facts the sessions list shows, so the reader can choose.
		expect(screen.getByText(/active · opus · \/Users\/x\/audit/)).toBeTruthy();
	});

	it("m3b: a FILED row is labelled, and its null runtime does not crash the sheet", async () => {
		/* The composed view carries both roles in `links`, and a coordination row's
		   `runtime` is null (the key is present, its value is null — not absent).
		   Reading `.state` off it took the whole sheet down; and one heading that
		   added the two roles together said "sessions (2)" for a project the card
		   still described as "1 session". */
		await openDetail(
			{},
			[
				link({ session_id: "4e92693767fa", role: "work", exists: true, runtime: { state: "live" } }),
				link({ session_id: "aa11bb22cc33", role: "coordination", exists: true, runtime: null }),
			],
		);

		expect(await screen.findByText("filed")).toBeTruthy();
		expect(screen.getByRole("heading", { name: "sessions (1 · 1 filed)" })).toBeTruthy();
		expect(screen.getByRole("button", { name: "unlink aa11bb22cc33" })).toBeTruthy();
	});

	it("M1c: a slash-named EXISTING milestone is still editable — only its removal is impossible", async () => {
		/* The add/set route carries the name in its body, so the phone was
		   refusing work it could do (review round 7, M1 narrowed). The name field
		   is fixed while editing; the date is not. */
		setProjectMilestone.mockResolvedValue({ ok: true, project: view() });
		getProject.mockResolvedValue({ project: view(), links: [] });
		await openDetail({
			milestones: [
				{ name: "ship/v2", target_date: null, completed_at: null, status: "upcoming" },
			],
		});
		fireEvent.click(await screen.findByRole("button", { name: "edit ship/v2" }));
		fireEvent.change(screen.getByLabelText("target date (optional)"), {
			target: { value: "2026-12-01" },
		});
		fireEvent.click(screen.getByRole("button", { name: "save" }));

		await waitFor(() =>
			expect(setProjectMilestone).toHaveBeenCalledWith("p1", {
				name: "ship/v2",
				target_date: "2026-12-01",
			}),
		);
	});

	it("D7: the link cap is stated before the tap", async () => {
		const many = Array.from({ length: SESSION_LINK_CAP }, (_, index) =>
			String(index).padStart(12, "0"),
		);
		getProject.mockResolvedValue({
			project: view({ sessions: many }),
			links: many.map((id) => link({ session_id: id })),
		});
		getProjects.mockResolvedValue({ projects: [summary()] });
		renderSheet();
		fireEvent.click(await screen.findByText("payments-migration"));

		const entry = (await screen.findByRole("button", {
			name: "link a session",
		})) as HTMLButtonElement;
		expect(entry.disabled).toBe(true);
		expect(
			screen.getByText(new RegExp(`reached the ${SESSION_LINK_CAP}-session cap`)),
		).toBeTruthy();
	});
});
