// @vitest-environment happy-dom
//
// The pickers' non-populated states, on BOTH surfaces. Separate from
// `model-sheet.order.test.tsx` because these need `getModels` to reject or to
// stay PENDING, and that file's module mock resolves for every case in it.
//
// Two regressions are pinned here, and they are the same defect seen from two
// ends. The empty-state branch used to be a bare `filtered.length === 0`, which
// is true whenever the list is empty FOR ANY REASON:
//
//   * a failed fetch rendered the daemon's real message ("Model catalogue
//     unavailable for Radient; retry or log in again") with "no matching
//     models — try a provider…" stacked directly beneath it, so the surface
//     said two contradictory things at once; on `#/new`, which had no error
//     state at all, only the false half survived.
//   * an IN-FLIGHT fetch rendered that same advice over an empty filter field,
//     telling a user to change a query they had never typed — on every open,
//     since the sheet refetches per open.
//
// The earlier version of this file covered the rejecting case only, which is
// exactly why the pending case shipped: a test that never leaves the promise
// unresolved cannot see the state the copy is wrong in. Both ends are covered
// now, and each is asserted NEGATIVELY as well (the no-match copy must be
// ABSENT while unresolved, and present only after an empty resolution) so the
// gate itself is observable.
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModelSheet } from "./components/model-sheet";
import { NewSessionScreen } from "./screens/new-session";
import type { ModelEntry, SessionProjection } from "./types";

const getModels = vi.fn();

vi.mock("./api", () => ({
	getModels: (...args: unknown[]) => getModels(...args),
	sendCommand: vi.fn(async () => ({ ok: true })),
	getDirectories: vi.fn(async () => ({ home: "/Users/x", recent: [], tmp: "" })),
	startSession: vi.fn(),
}));

const projection = {
	model_selector: "anthropic/claude-opus-5",
} as unknown as SessionProjection;

/** The daemon's own 502 body, which `api.ts` throws as an `HttpError`. */
const DAEMON_MESSAGE =
	"Model catalogue unavailable for Radient; retry or log in again";

/** The recovery copy — the sentence that must not appear except when the
    catalogue resolved, the fetch succeeded, and the filter really is what
    excluded everything. */
const NO_MATCH = /no matching models/;

const ROWS: ModelEntry[] = [
	{
		selector: "anthropic/claude-opus-5",
		provider: "anthropic",
		model_id: "claude-opus-5",
		name: "Claude Opus 5",
		label: "Claude Opus 5",
		connected: true,
		aggregated: false,
	},
];

/** Open the in-session sheet. */
function renderSheet() {
	return render(
		<ModelSheet open onClose={() => {}} pid="1" projection={projection} />,
	);
}

/** Open `#/new`'s picker, which lives behind the `default` model button. */
async function renderNewSessionPicker() {
	render(<NewSessionScreen />);
	(await screen.findByText("default")).click();
	return screen.findByPlaceholderText("filter models");
}

beforeEach(() => {
	vi.clearAllMocks();
	document.body.innerHTML = "";
});

describe("the in-session sheet", () => {
	it("does not claim nothing matched while the fetch is still in flight", async () => {
		// Deliberately never settled: this is the state the sheet spends its
		// first second in on a mobile link, and the one the previous test file
		// could not reach.
		getModels.mockReturnValue(new Promise(() => {}));

		renderSheet();

		expect(await screen.findByText("loading…")).toBeTruthy();
		expect(screen.queryByText(NO_MATCH)).toBeNull();
	});

	it("says nothing matched only once an EMPTY catalogue has actually resolved", async () => {
		getModels.mockResolvedValue({ models: [] });

		renderSheet();

		expect(await screen.findByText(NO_MATCH)).toBeTruthy();
		expect(screen.queryByText("loading…")).toBeNull();
	});

	it("surfaces the daemon's message INSTEAD OF, not beside, the no-match copy", async () => {
		getModels.mockRejectedValue(new Error(DAEMON_MESSAGE));

		renderSheet();

		await waitFor(() => {
			expect(screen.getByText(DAEMON_MESSAGE)).toBeTruthy();
		});
		// The two are different facts about different things; stacking them told
		// the user their token had expired AND that they should retype a query.
		expect(screen.queryByText(NO_MATCH)).toBeNull();
	});

	it("filters a resolved catalogue down to the no-match state", async () => {
		getModels.mockResolvedValue({ models: ROWS });

		renderSheet();
		const input = await screen.findByPlaceholderText("filter models");
		expect(screen.queryByText(NO_MATCH)).toBeNull();

		input.setAttribute("value", "zzzznomatch");
		const { fireEvent } = await import("@testing-library/react");
		fireEvent.change(input, { target: { value: "zzzznomatch" } });

		expect(screen.getByText(NO_MATCH)).toBeTruthy();
	});
});

describe("the #/new picker", () => {
	it("does not claim nothing matched while the fetch is still in flight", async () => {
		getModels.mockReturnValue(new Promise(() => {}));

		await renderNewSessionPicker();

		expect(await screen.findByText("loading…")).toBeTruthy();
		expect(screen.queryByText(NO_MATCH)).toBeNull();
	});

	it("surfaces the daemon's message on a failed fetch, like the sheet", async () => {
		// Round 1 gave this picker the empty state but left its `.catch` an empty
		// swallow, so the identical 502 produced opposite accounts on the two
		// surfaces: the sheet named the cause, `#/new` blamed the user's filter.
		getModels.mockRejectedValue(new Error(DAEMON_MESSAGE));

		await renderNewSessionPicker();

		await waitFor(() => {
			expect(screen.getByText(DAEMON_MESSAGE)).toBeTruthy();
		});
		expect(screen.queryByText(NO_MATCH)).toBeNull();
	});

	it("keeps `default` selectable through both states", async () => {
		// Neither a failed catalogue nor one in flight may take away the ability
		// to start a session on the daemon's default.
		getModels.mockRejectedValue(new Error(DAEMON_MESSAGE));

		await renderNewSessionPicker();

		await waitFor(() => {
			expect(screen.getByText(DAEMON_MESSAGE)).toBeTruthy();
		});
		// `getAllByText`: "default" names both the trigger button on the screen
		// behind and the row inside the open sheet.
		expect(screen.getAllByText("default").length).toBeGreaterThan(0);
	});

	it("says nothing matched only once an EMPTY catalogue has actually resolved", async () => {
		getModels.mockResolvedValue({ models: [] });

		await renderNewSessionPicker();

		expect(await screen.findByText(NO_MATCH)).toBeTruthy();
		expect(screen.queryByText("loading…")).toBeNull();
	});
});
