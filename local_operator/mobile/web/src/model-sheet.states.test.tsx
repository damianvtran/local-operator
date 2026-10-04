// @vitest-environment happy-dom
//
// The in-session model picker's non-populated states. Separate from
// `model-sheet.order.test.tsx` because these need `getModels` to reject or to
// stay PENDING, and that file's module mock resolves for every case in it.
//
// These cases used to be run against BOTH pickers, the in-session sheet and the
// `#/new` screen's own copy of it. The new-session screen is gone (starting a
// session is one tap on the list and the directory is resolved for the user),
// so the sheet is the only surface left — and these cases are the ones that
// pinned the defect below, not the screen they were reached through.
//
// Two regressions are pinned here, and they are the same defect seen from two
// ends. The empty-state branch used to be a bare `filtered.length === 0`, which
// is true whenever the list is empty FOR ANY REASON:
//
//   * a failed fetch rendered the daemon's real message ("Model catalogue
//     unavailable for Radient; retry or log in again") with "no matching
//     models — try a provider…" stacked directly beneath it, so the surface
//     said two contradictory things at once.
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

/** The empty-catalogue copy (U9, batch 2): resolved, non-error, zero rows.
    Distinct from NO_MATCH — filter advice cannot succeed against an empty
    inventory, and the two facts must not render as one sentence. */
const EMPTY_CATALOGUE = /no models available from this machine/;

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

	it("a resolved catalogue with NO models says so, instead of blaming the filter (U9, batch 2)", async () => {
		getModels.mockResolvedValue({ models: [] });

		renderSheet();

		expect(await screen.findByText(EMPTY_CATALOGUE)).toBeTruthy();
		expect(screen.queryByText(NO_MATCH)).toBeNull();
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
