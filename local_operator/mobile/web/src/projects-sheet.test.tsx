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
import type { ProjectLinkedSession, ProjectSummary, ProjectView } from "./types";

const getProjects = vi.fn();
const getProject = vi.fn();
const createProject = vi.fn();
const deleteProject = vi.fn();
const setProjectMilestone = vi.fn();

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getProjects: (...args: unknown[]) => getProjects(...args),
		getProject: (...args: unknown[]) => getProject(...args),
		createProject: (...args: unknown[]) => createProject(...args),
		deleteProject: (...args: unknown[]) => deleteProject(...args),
		setProjectMilestone: (...args: unknown[]) => setProjectMilestone(...args),
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
		fireEvent.click(screen.getByRole("button", { name: /beta cut/ }));
		await waitFor(() =>
			expect(setProjectMilestone).toHaveBeenCalledWith("p1", { name: "beta cut", completed: true }),
		);
		// The sheet re-renders from the WRITE'S OWN ANSWER (the returned view),
		// not from a local guess at what changed.
		await waitFor(() =>
			expect(screen.getByRole("button", { name: /beta cut/ }).getAttribute("aria-pressed")).toBe("true"),
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
		fireEvent.click(await screen.findByRole("button", { name: /beta cut/ }));
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
		fireEvent.click(await screen.findByRole("button", { name: /beta cut/ }));
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
		await waitFor(() => expect(createProject).toHaveBeenCalledWith({ name: "alpha", description: "" }));
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
