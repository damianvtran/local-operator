// @vitest-environment happy-dom
//
// The asks-population signature: the rule that decides WHEN the aggregate is
// re-read. Agent review round 1 (R4) found the mechanism untested — the sheet
// mocks `useAsksRevision`, so the one new live-update path had no assertion.
// The rule is pure, so it is tested directly.
import { describe, expect, it } from "vitest";
import { asksPopulationSignature } from "./store";
import type { SessionSummary } from "./types";

function row(patch: Partial<SessionSummary> = {}): SessionSummary {
	return { session_id: "a", asks_open: 1, ...patch } as SessionSummary;
}

describe("asksPopulationSignature", () => {
	it("moves when one session's count moves", () => {
		expect(asksPopulationSignature([row({ asks_open: 2 })])).not.toBe(
			asksPopulationSignature([row({ asks_open: 1 })]),
		);
	});

	it("does not move on an unrelated repaint — same counts, new objects", () => {
		expect(asksPopulationSignature([row({ asks_open: 2 })])).toBe(
			asksPopulationSignature([row({ asks_open: 2 })]),
		);
	});

	it("is order-insensitive — the daemon's ranking is not new information", () => {
		expect(
			asksPopulationSignature([row({ session_id: "a" }), row({ session_id: "b" })]),
		).toBe(asksPopulationSignature([row({ session_id: "b" }), row({ session_id: "a" })]));
	});

	it("treats a runtime that does not publish asks as contributing nothing", () => {
		/* Absence is the capability proxy (§4), so it must not read as a zero —
		   a zero would be a claim the runtime cannot make, and it would move the
		   signature every time such a session repainted. */
		expect(asksPopulationSignature([row({ asks_open: undefined })])).toBe("");
		expect(asksPopulationSignature([row({ asks_open: 0 })])).toBe("");
	});
});
