// @vitest-environment happy-dom
//
// The ask vocabulary, and the two decisions that are easy to get subtly wrong:
// which ask is the HEAD (the OLDEST open one, never the list's first row — the
// wire leads with the newest, and a bar that jumped to each arrival moves under
// a thumb), and whether the legacy mirrored card is still a blocking gate
// (design §4's client rule N3: once `asks` is present, kind === "ask" is not).
import { describe, expect, it } from "vitest";
import {
	answeredBySurface,
	askStateLine,
	blockingPending,
	dockAsk,
	durationLabel,
	headAsk,
	isAnswerable,
	isLapsed,
	orderedForDisplay,
	outstandingAsks,
	questionProgress,
	questionTally,
	unansweredQuestions,
} from "./asks";
import type { PendingAsk } from "../types";

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a1",
		created_at: 1_000,
		expires_at: 1_000_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [{ id: "q1", question: "ship it?", options: [], multi: false, secret: false, persist: false }],
		...patch,
	};
}

describe("askStateLine", () => {
	it("states the open ask as queued, with a locally rendered countdown", () => {
		const line = askStateLine(ask({ expires_at: 1000 + 42 * 60 * 1000 }), 1000);
		expect(line.text).toBe("Queued — the agent is continuing; expires in 42 m");
		expect(line.tone).toBe("waiting");
	});

	it("says delivering until the runtime has delivered the response rows", () => {
		expect(askStateLine(ask({ status: "answered", delivered: false }), 1).text).toBe(
			"Answered — delivering",
		);
		expect(askStateLine(ask({ status: "answered", delivered: true }), 1).text).toBe(
			"Answered — the agent was told",
		);
	});

	it("states a timed-out ask in Ben's word: the agent goes ahead without the answer", () => {
		/* Ben's report (2026-10-15): "you can still answer" invited an answer to a
		   question the agent had already gone ahead without. The wording holds for
		   every timeout notice, including a secret ask's, which has no default
		   (design round 1, D3). */
		const line = askStateLine(ask({ status: "timed_out" }), 1);
		expect(line.text).toBe("Timed out — the agent goes ahead without your answer");
		expect(line.tone).toBe("gone");
	});

	it("names a dismissal as no reply at all, and an expiry as unfixable", () => {
		expect(askStateLine(ask({ status: "dismissed" }), 1).text).toBe(
			"Dismissed — no reply was sent",
		);
		expect(askStateLine(ask({ status: "expired" }), 1).text).toBe(
			"Expired — this ask is too old to answer; ask the agent again",
		);
	});

	it("names a withdrawal as the agent's own retraction, never a failure", () => {
		const line = askStateLine(ask({ status: "withdrawn" }), 1);
		expect(line.text).toBe("Withdrawn — the agent no longer needs an answer");
		expect(line.tone).toBe("gone");
		expect(isAnswerable("withdrawn")).toBe(false);
	});

	it("does not claim an open ask past its deadline on the phone's clock is lapsed", () => {
		/* The phone's clock may run ahead, and the runtime's host is what judges
		   the deadline (a stale `open` row arrives already re-derived to
		   `timed_out`), so an `open` row stays answerable: the line says the
		   deadline has passed locally without calling it over (review round 1,
		   R3). */
		const line = askStateLine(ask({ expires_at: 500 }), 1000);
		expect(line.text).toBe("Queued — the agent is continuing; deadline passed");
		expect(line.tone).toBe("attention");
	});
});

describe("durationLabel", () => {
	it("spaces its units, and never reads as more precise than it is", () => {
		expect(durationLabel(42 * 60 * 1000)).toBe("42 m");
		expect(durationLabel(3 * 3600 * 1000)).toBe("3 h");
		expect(durationLabel(50 * 3600 * 1000)).toBe("2 d");
		expect(durationLabel(0)).toBe("");
		/* SUB-SECOND IS A BOUND, NOT A BLANK (agent review round 1, N1): the open
		   line used to read "expires in" with no value at all for a deadline
		   inside the next second. */
		expect(durationLabel(400)).toBe("<1 m");
		expect(durationLabel(30_000)).toBe("<1 m");
		expect(durationLabel(60_000)).toBe("1 m");
	});
});

describe("orderedForDisplay", () => {
	it("lifts the ask the chip names — the head when one is open", () => {
		const older = ask({ ask_id: "older", created_at: 100 });
		const newer = ask({ ask_id: "newer", created_at: 900 });
		expect(orderedForDisplay([newer, older]).map((row) => row.ask_id)).toEqual([
			"older",
			"newer",
		]);
	});

	it("lifts the answerable TIMEOUT when nothing is open — the same row the chip names", () => {
		/* r2-N2 (agent review round 2): lifting only the open-ask head left the
		   chip naming a timed-out ask while the sheet kept wire order, in exactly
		   the state U7 exists for. */
		const deadline = ask({ ask_id: "deadline", created_at: 100, status: "timed_out" });
		const settled = ask({ ask_id: "settled", created_at: 50, status: "answered" });
		expect(orderedForDisplay([settled, deadline]).map((row) => row.ask_id)).toEqual([
			"deadline",
			"settled",
		]);
		expect(dockAsk([settled, deadline])?.ask_id).toBe("deadline");
	});

	it("leaves the wire's order alone when there is nothing outstanding", () => {
		const a = ask({ ask_id: "a", status: "answered" });
		const b = ask({ ask_id: "b", status: "declined" });
		expect(orderedForDisplay([a, b]).map((row) => row.ask_id)).toEqual(["a", "b"]);
	});
});

describe("headAsk", () => {
	it("is the OLDEST open ask, not the wire list's first row", () => {
		const newest = ask({ ask_id: "new", created_at: 900 });
		const oldest = ask({ ask_id: "old", created_at: 100 });
		expect(headAsk([newest, oldest])?.ask_id).toBe("old");
	});

	it("ignores a timed-out ask — the agent is no longer waiting on it", () => {
		expect(headAsk([ask({ status: "timed_out" })])).toBeNull();
	});
});

describe("outstanding / answerable", () => {
	it("counts the asks still needing an answer, and only those", () => {
		const rows = [
			ask({ ask_id: "o" }),
			ask({ ask_id: "t", status: "timed_out" }),
			ask({ ask_id: "a", status: "answered" }),
			ask({ ask_id: "d", status: "dismissed" }),
		];
		expect(outstandingAsks(rows).map((row) => row.ask_id)).toEqual(["o", "t"]);
	});

	it("reads a lapse from the status alone — timed_out, and nothing else", () => {
		expect(isLapsed(ask({ status: "timed_out" }))).toBe(true);
		/* An open row past its deadline on the phone's clock is NOT lapsed here:
		   the runtime's host judges the deadline (review round 1, R3). */
		expect(isLapsed(ask({ expires_at: 500 }))).toBe(false);
		/* A settled ask past its deadline is settled, not lapsed. */
		expect(isLapsed(ask({ status: "answered", expires_at: 500 }))).toBe(false);
	});

	it("splits the outstanding questions into waiting and lapsed", () => {
		const two = [
			{ id: "q1", question: "a", options: [], multi: false, secret: false, persist: false },
			{ id: "q2", question: "b", options: [], multi: false, secret: false, persist: false },
		];
		expect(
			questionTally([
				ask({ ask_id: "o", questions: two }),
				ask({ ask_id: "t", status: "timed_out" }),
				ask({ ask_id: "a", status: "answered" }),
			]),
		).toEqual({ waiting: 2, lapsed: 1 });
		expect(questionTally(undefined)).toEqual({ waiting: 0, lapsed: 0 });
	});

	it("treats a timed-out ask as answerable and an answered one as not", () => {
		expect(isAnswerable("timed_out")).toBe(true);
		expect(isAnswerable("open")).toBe(true);
		expect(isAnswerable("answered")).toBe(false);
		expect(isAnswerable("expired")).toBe(false);
	});
});

describe("unansweredQuestions", () => {
	it("drops settled answers AND the legacy path's drafts", () => {
		const row = ask({
			questions: [
				{ id: "q1", question: "a", options: [], multi: false, secret: false, persist: false },
				{ id: "q2", question: "b", options: [], multi: false, secret: false, persist: false },
				{ id: "q3", question: "c", options: [], multi: false, secret: false, persist: false },
			],
			answers: { q1: ["yes"] },
			draft_question_ids: ["q2"],
		});
		expect(unansweredQuestions(row).map((q) => q.id)).toEqual(["q3"]);
		expect(questionProgress(row)).toEqual({ index: 2, total: 3 });
	});
});

describe("blockingPending (N3)", () => {
	it("ignores a mirrored ask once the asks field is present", () => {
		expect(blockingPending({ kind: "ask" }, [])).toBeNull();
	});

	it("keeps an approval, and keeps the mirror when the runtime cannot publish asks", () => {
		const approval = { kind: "approval" };
		expect(blockingPending(approval, [])).toBe(approval);
		const mirror = { kind: "ask" };
		expect(blockingPending(mirror, undefined)).toBe(mirror);
	});
});

describe("answeredBySurface", () => {
	it("names the surface that beat this one to the answer", () => {
		expect(answeredBySurface(ask({ answered_by: { surface: "tui" } }))).toBe("tui");
		expect(answeredBySurface(ask({}))).toBe("");
	});
});
