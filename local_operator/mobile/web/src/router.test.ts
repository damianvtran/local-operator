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
	it("replaces a direct deep link with its hierarchy fallback", () => {
		const replaceState = vi.fn();
		vi.stubGlobal("history", { state: null, replaceState, back: vi.fn() });
		vi.stubGlobal("window", { dispatchEvent: vi.fn() });
		vi.stubGlobal("PopStateEvent", class { constructor(..._args: unknown[]) {} });
		navigateUp("/s/root/a/parent");
		expect(replaceState).toHaveBeenCalledWith(
			expect.objectContaining({ loMobileRoute: true }),
			"",
			"#/s/root/a/parent",
		);
	});

	it("uses chronological Back for an in-app route", () => {
		const back = vi.fn();
		vi.stubGlobal("history", { state: { loMobileRoute: true }, back });
		navigateUp("/s/root");
		expect(back).toHaveBeenCalledOnce();
	});
});
