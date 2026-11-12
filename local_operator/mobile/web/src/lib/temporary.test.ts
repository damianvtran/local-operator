// Temporary chats: the relay's pure decisions (contract §2.3). The capability
// gate fails closed, a malformed mark reads as "not temporary", the partition
// keeps the daemon's order, and a row says either its countdown or its hold.
import { describe, expect, it } from "vitest";
import {
	holdLabelKey,
	isTemporary,
	makeTemporaryNeedsConfirm,
	minutesUntilDeletion,
	partitionTemporary,
	readTemporary,
	sessionTemporary,
	temporaryChatsEnabled,
	temporaryGroupOpen,
	temporaryRowLabel,
} from "./temporary";
import { TEMPORARY_HOLD_CODES, type TemporaryInfo } from "../types";

const NOW_MS = 1_800_000_000_000;
const mark = (over: Partial<TemporaryInfo> = {}): TemporaryInfo => ({
	marked_at: NOW_MS / 1000 - 600,
	idle_minutes: 30,
	deletes_at: NOW_MS / 1000 + 12 * 60,
	hold: null,
	...over,
});

describe("temporaryChatsEnabled (fail closed)", () => {
	it("is on only for a numeric flag >= 1", () => {
		expect(temporaryChatsEnabled({ features: { temporary_chats: 1 } })).toBe(true);
		expect(temporaryChatsEnabled({ features: { temporary_chats: 2 } })).toBe(true);
	});
	it.each([
		["no capabilities", null],
		["no features", {}],
		["absent flag", { features: {} }],
		["zero", { features: { temporary_chats: 0 } }],
		["string", { features: { temporary_chats: "1" } }],
		["boolean", { features: { temporary_chats: true } }],
		["NaN", { features: { temporary_chats: Number.NaN } }],
	])("is off for %s", (_label, caps) => {
		expect(temporaryChatsEnabled(caps as never)).toBe(false);
	});
});

describe("readTemporary", () => {
	it("reads the frozen wire object", () => {
		expect(readTemporary(mark())).toEqual(mark());
	});
	it.each([null, undefined, 1, "x", {}, { marked_at: "now" }, { marked_at: Number.POSITIVE_INFINITY }])(
		"reads %j as not temporary",
		(value) => expect(readTemporary(value)).toBeNull(),
	);
	it("drops the deadline while held, and reads an unknown hold as cannot_verify", () => {
		expect(readTemporary(mark({ hold: "running" }))?.deletes_at).toBeNull();
		const unknown = readTemporary({ ...mark(), hold: "future_code" });
		expect(unknown?.hold).toBe("cannot_verify");
		expect(unknown?.deletes_at).toBeNull();
	});
});

describe("isTemporary / sessionTemporary", () => {
	it("never reads a row's mark without the capability", () => {
		expect(isTemporary({ temporary: mark() }, false)).toBe(false);
		expect(isTemporary({ temporary: mark() }, true)).toBe(true);
		expect(isTemporary({ temporary: null }, true)).toBe(false);
		expect(sessionTemporary(false, mark(), { temporary: mark() })).toBeNull();
	});
	it("prefers the events frame, an explicit null included, over the list row", () => {
		expect(sessionTemporary(true, null, { temporary: mark() })).toBeNull();
		expect(sessionTemporary(true, undefined, { temporary: mark() })).toEqual(mark());
		expect(sessionTemporary(true, mark({ idle_minutes: 15 }), { temporary: null })?.idle_minutes).toBe(15);
	});
});

describe("partitionTemporary", () => {
	const rows = [
		{ session_id: "a", temporary: null },
		{ session_id: "b", temporary: mark() },
		{ session_id: "c" },
		{ session_id: "d", temporary: mark({ hold: "on_screen" }) },
	];
	it("splits in the daemon's order", () => {
		const { normal, temporary } = partitionTemporary(rows, true);
		expect(normal.map((r) => r.session_id)).toEqual(["a", "c"]);
		expect(temporary.map((r) => r.session_id)).toEqual(["b", "d"]);
	});
	it("is the identity without the capability", () => {
		const { normal, temporary } = partitionTemporary(rows, false);
		expect(normal.map((r) => r.session_id)).toEqual(["a", "b", "c", "d"]);
		expect(temporary).toEqual([]);
	});
});

describe("temporaryGroupOpen", () => {
	it("opens only while the group holds the current chat", () => {
		const rows = [{ session_id: "b" }];
		expect(temporaryGroupOpen("b", rows)).toBe(true);
		expect(temporaryGroupOpen("a", rows)).toBe(false);
		expect(temporaryGroupOpen(null, rows)).toBe(false);
	});
});

describe("countdown and holds", () => {
	it("rounds up and never shows zero minutes", () => {
		expect(minutesUntilDeletion(mark(), NOW_MS)).toBe(12);
		expect(minutesUntilDeletion(mark({ deletes_at: NOW_MS / 1000 + 61 }), NOW_MS)).toBe(2);
		expect(minutesUntilDeletion(mark({ deletes_at: NOW_MS / 1000 - 30 }), NOW_MS)).toBe(1);
		expect(minutesUntilDeletion(mark({ hold: "running", deletes_at: null }), NOW_MS)).toBeNull();
	});
	it("warns only in the last five minutes", () => {
		expect(temporaryRowLabel(mark(), NOW_MS)).toEqual({ key: "relay.chrome.temporary_deletes_in", minutes: 12, warning: false });
		expect(temporaryRowLabel(mark({ deletes_at: NOW_MS / 1000 + 4 * 60 }), NOW_MS).warning).toBe(true);
	});
	it("names the hold instead of a countdown, and a missing deadline as cannot_verify", () => {
		expect(temporaryRowLabel(mark({ hold: "open_ask", deletes_at: null }), NOW_MS).key).toBe("relay.chrome.temporary_hold_open_ask");
		expect(temporaryRowLabel(mark({ deletes_at: null }), NOW_MS).key).toBe("relay.chrome.temporary_hold_cannot_verify");
	});
	it("has one key per frozen hold code", () => {
		expect(TEMPORARY_HOLD_CODES).toHaveLength(11);
		expect(new Set(TEMPORARY_HOLD_CODES.map(holdLabelKey)).size).toBe(11);
	});
});

describe("makeTemporaryNeedsConfirm", () => {
	it("asks only when there is something to delete", () => {
		expect(makeTemporaryNeedsConfirm(0)).toBe(false);
		expect(makeTemporaryNeedsConfirm(3)).toBe(true);
	});
});
