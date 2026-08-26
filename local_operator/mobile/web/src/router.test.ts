import { afterEach, describe, expect, it, vi } from "vitest";
import { navigateUp, parseHash } from "./router";

afterEach(() => vi.unstubAllGlobals());

describe("parseHash", () => {
	it("keeps the root conversation route exact", () => {
		expect(parseHash("#/s/root-session")).toEqual({
			name: "session",
			sessionId: "root-session",
		});
	});

	it("round-trips a deep-linked descendant route", () => {
		expect(parseHash("#/s/root%2Fsafe/a/job%3Achild")).toEqual({
			name: "session",
			sessionId: "root/safe",
			jobId: "job:child",
		});
	});

	it("does not accept trailing path fragments as a root route", () => {
		expect(parseHash("#/s/root/a/job/unknown")).toEqual({ name: "list" });
	});
});

describe("navigateUp", () => {
	it("keeps repeated direct-link fallback inside the hierarchy", () => {
		let state: Record<string, unknown> | null = null;
		const replaceState = vi.fn((next: Record<string, unknown>) => { state = next; });
		const back = vi.fn();
		vi.stubGlobal("history", {
			get state() { return state; },
			replaceState,
			back,
		});
		vi.stubGlobal("window", { dispatchEvent: vi.fn() });
		vi.stubGlobal("PopStateEvent", class { constructor(..._args: unknown[]) {} });

		navigateUp("/s/root/a/parent");
		navigateUp("/s/root");

		expect(replaceState).toHaveBeenNthCalledWith(
			1,
			expect.objectContaining({ loMobileHasInAppPredecessor: false }),
			"",
			"#/s/root/a/parent",
		);
		expect(replaceState).toHaveBeenNthCalledWith(
			2,
			expect.objectContaining({ loMobileHasInAppPredecessor: false }),
			"",
			"#/s/root",
		);
		expect(back).not.toHaveBeenCalled();
	});

	it("uses chronological Back for an in-app route", () => {
		const back = vi.fn();
		vi.stubGlobal("history", { state: { loMobileHasInAppPredecessor: true }, back });
		navigateUp("/s/root");
		expect(back).toHaveBeenCalledOnce();
	});
});
