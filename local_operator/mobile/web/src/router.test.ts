import { describe, expect, it } from "vitest";
import { parseHash } from "./router";

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
