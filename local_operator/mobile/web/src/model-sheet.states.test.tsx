// @vitest-environment happy-dom
//
// The sheet's non-populated states. Separate from `model-sheet.order.test.tsx`
// because these need `getModels` to REJECT, and that file's module mock resolves
// for every case in it.
//
// The regression: the sheet caught a failed fetch with `.catch(() =>
// setModels([]))`, so a 502 rendered as "no matching models" — the daemon
// composes a precise, actionable message ("Model catalogue unavailable for
// Radient; retry or log in again") and the client threw it away, telling a user
// whose token had expired that their filter matched nothing. The `error` state
// was already declared and already rendered; only `choose()` ever set it.
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModelSheet } from "./components/model-sheet";
import type { SessionProjection } from "./types";

const getModels = vi.fn();

vi.mock("./api", () => ({
	getModels: (...args: unknown[]) => getModels(...args),
	sendCommand: vi.fn(async () => ({ ok: true })),
}));

const projection = {
	model_selector: "anthropic/claude-opus-5",
} as unknown as SessionProjection;

describe("the model sheet when the catalogue cannot be fetched", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		document.body.innerHTML = "";
	});

	it("surfaces the daemon's message rather than claiming nothing matched", async () => {
		const message = "Model catalogue unavailable for Radient; retry or log in again";
		getModels.mockRejectedValue(new Error(message));

		render(
			<ModelSheet
				open
				onClose={() => {}}
				pid="1"
				projection={projection}
			/>,
		);

		await waitFor(() => {
			expect(screen.getByText(message)).toBeTruthy();
		});
	});
});
