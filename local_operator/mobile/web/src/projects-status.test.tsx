// @vitest-environment happy-dom
//
// THE STATUS-ORDER GUARD. The Projects sheet groups its board by project
// status, and a section the client draws must sit where the daemon ranked it.
// That order used to be a hand-copied array in `projects-sheet.tsx`, and it had
// already drifted: the store grew four statuses into seven between 0.63.13 and
// 0.67.4, and every status the copy did not know fell into an "unknown" section
// at the END of the board instead of its lifecycle position — a silent
// mis-group that no test noticed, because the rows still rendered.
//
// `STATUS_ORDER` is now generated from the daemon's `STATUS_RANK` by
// `scripts/generate-projects-status.mjs`. These tests are what make that
// generation a gate rather than a convention:
//
//   * the generated file must equal what the generator produces from the
//     Python source RIGHT NOW — a status added on the server, in either
//     direction, reddens this file;
//   * the divergence detector is proven non-vacuous by feeding the parser a
//     mutated source, so "the test would have caught the drift" is measured
//     rather than asserted;
//   * and a status this build genuinely does not know (a newer daemon) is
//     REPORTED on the surface — the fallback is visible, not silent.
import { readFileSync } from "node:fs";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	readSessionLinkCap,
	readStatusOrder,
	renderModule,
	OUT,
	PY_SOURCE,
	PY_STORE,
} from "../scripts/generate-projects-status.mjs";
import { ProjectsSheet } from "./components/projects-sheet";
import { SESSION_LINK_CAP, STATUS_ORDER } from "./projects-status.generated";
import type { ProjectSummary } from "./types";

const getProjects = vi.fn();

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getProjects: (...args: unknown[]) => getProjects(...args),
	};
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

describe("the generated status order", () => {
	it("is the daemon's own board order, and the committed module is current", () => {
		const source = readFileSync(PY_SOURCE, "utf8");
		const store = readFileSync(PY_STORE, "utf8");
		/* Both directions in one line: the module the sheet imports equals the
		   order the Python source states today, and the committed FILE equals
		   what the generator would write — so editing either one alone fails. */
		expect(STATUS_ORDER).toEqual(readStatusOrder(source));
		expect(readFileSync(OUT, "utf8")).toBe(
			renderModule(readStatusOrder(source), readSessionLinkCap(store)),
		);
	});

	it("carries the store's own link cap, and is current about it too", () => {
		const store = readFileSync(PY_STORE, "utf8");
		/* The cap is what lets the sheet say "the cap is reached" BEFORE the tap.
		   A hand-copied 64 would go stale the day the store moves it — and the
		   sheet would then offer a tap the store refuses (review round 6, D7). */
		expect(SESSION_LINK_CAP).toBe(readSessionLinkCap(store));
		const moved = store.replace(/^SESSIONS_MAX = \d+$/m, "SESSIONS_MAX = 99");
		expect(moved).not.toBe(store);
		expect(readSessionLinkCap(moved)).toBe(99);
		expect(
			renderModule(readStatusOrder(readFileSync(PY_SOURCE, "utf8")), readSessionLinkCap(moved)),
		).not.toBe(readFileSync(OUT, "utf8"));
		expect(() => readSessionLinkCap("nothing here")).toThrow();
	});

	it("would notice a status the daemon added or reordered", () => {
		const source = readFileSync(PY_SOURCE, "utf8");
		const store = readFileSync(PY_STORE, "utf8");
		const current = () => renderModule(readStatusOrder(source), readSessionLinkCap(store));
		/* The mutation is the real-world one: a status appended to the rank
		   dict. The parser must see it, and the generated file must then be
		   recognised as stale — otherwise the guard is theatre. */
		const drifted = source.replace(/"archived": 6,/, '"archived": 6,\n    "blocked": 7,');
		expect(drifted).not.toBe(source);
		expect(readStatusOrder(drifted)).toEqual([...STATUS_ORDER, "blocked"]);
		expect(renderModule(readStatusOrder(drifted), readSessionLinkCap(store))).not.toBe(
			readFileSync(OUT, "utf8"),
		);

		/* And a reordering, which the appended-status case cannot catch. */
		const reordered = source
			.replace(/"qa": 2,/, '"qa": 3,')
			.replace(/"validation": 3,/, '"validation": 2,');
		expect(readStatusOrder(reordered)).toEqual([
			...STATUS_ORDER.slice(0, 2),
			"validation",
			"qa",
			...STATUS_ORDER.slice(4),
		]);
		expect(current()).toBe(readFileSync(OUT, "utf8"));
	});

	it("refuses a source it cannot parse rather than emitting an empty order", () => {
		// A gate that silently generates nothing is worse than no gate: this
		// generator must throw when the dict it reads is gone or mangled.
		expect(() => readStatusOrder("STATUS_RANK = {}\n")).toThrow();
		expect(() => readStatusOrder("nothing here")).toThrow();
	});
});

describe("a status this build does not know", () => {
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

	it("is REPORTED in a trailing section, not silently grouped as lifecycle", async () => {
		/* A newer daemon's status. The row must still render (dropping it would
		   hide a project), and its heading must SAY it is unrecognised — the
		   silent mis-group this sheet used to produce is the defect. */
		getProjects.mockResolvedValue({
			projects: [
				summary({ id: "p1", name: "known-row", status: "active" }),
				summary({ id: "p2", name: "newer-row", status: "blocked" }),
			],
		});
		render(<ProjectsSheet open onClose={() => {}} />);
		await screen.findByText("known-row");
		fireEvent.click(screen.getByRole("button", { name: "board" }));

		const headings = screen.getAllByRole("heading").map((heading) => heading.textContent);
		expect(headings).toEqual(["active", "blocked (unknown status)"]);
		// The row itself is still offered.
		expect(screen.getByText("newer-row")).toBeTruthy();
	});
});
