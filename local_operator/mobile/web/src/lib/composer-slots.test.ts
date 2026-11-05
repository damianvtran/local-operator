import { describe, expect, it } from "vitest";
import {
	FLICKER_MS,
	GUARD_MS,
	micSlot,
	pressIsReflex,
	primaryRole,
	trackArrival,
} from "./composer-slots";

const idle = { alive: false, stopping: false, sending: false, hasPayload: false };

describe("primaryRole: send, steer and stop share one slot", () => {
	it.each([
		["idle, empty box", { ...idle }, "send"],
		["idle, text", { ...idle, hasPayload: true }, "send"],
		["working, empty box", { ...idle, alive: true }, "stop"],
		["working, text or attachment", { ...idle, alive: true, hasPayload: true }, "steer"],
		["working, stopping, empty", { ...idle, alive: true, stopping: true }, "stopping"],
		/* D6: the pressed state outranks typed text. */
		["working, stopping, text", { ...idle, alive: true, stopping: true, hasPayload: true }, "stopping"],
		/* A send in flight locks the role: the box has just been cleared, so an
		   unlocked slot would flip Steer -> Stop under the double-tapping finger. */
		["send in flight, box already cleared", { ...idle, alive: true, sending: true }, "connecting"],
		["send in flight, idle", { ...idle, sending: true }, "connecting"],
		/* A stale latch with no live turn must not draw Stop. */
		["stopping flag but no live turn", { ...idle, stopping: true }, "send"],
	] as const)("%s -> %s", (_name, input, expected) => {
		expect(primaryRole(input)).toBe(expected);
	});

	it("never yields two roles for one input: stop and send are exclusive by construction", () => {
		for (const alive of [false, true])
			for (const stopping of [false, true])
				for (const sending of [false, true])
					for (const hasPayload of [false, true]) {
						const role = primaryRole({ alive, stopping, sending, hasPayload });
						expect(["connecting", "stopping", "stop", "steer", "send"]).toContain(role);
						// steer only while a turn is alive, send only while not
						if (role === "steer") expect(alive).toBe(true);
						if (role === "send") expect(alive && hasPayload).toBe(false);
					}
	});
});

describe("pressIsReflex: the Stop arrival guard", () => {
	it("swallows a pointer press inside the window", () => {
		expect(pressIsReflex(1000, 1000 + 120, 1)).toBe(true);
		expect(pressIsReflex(1000, 1000 + GUARD_MS - 1, 1)).toBe(true);
	});
	it("lets a press at or after the window through", () => {
		expect(pressIsReflex(1000, 1000 + GUARD_MS, 1)).toBe(false);
		expect(pressIsReflex(1000, 1000 + 5000, 3)).toBe(false);
	});
	it("never guards keyboard or assistive activation (click detail 0)", () => {
		expect(pressIsReflex(1000, 1000, 0)).toBe(false);
	});
});

describe("trackArrival: a flicker of the input must not re-arm the guard", () => {
	it("records the first role and keeps it while unchanged", () => {
		const a = trackArrival(null, "send", 10);
		expect(a).toMatchObject({ role: "send", arrivedAt: 10 });
		expect(trackArrival(a, "send", 900)).toBe(a);
	});
	it("stamps a real change", () => {
		const a = trackArrival(trackArrival(null, "send", 0), "stop", 5000);
		expect(a.arrivedAt).toBe(5000);
	});
	it("keeps the original arrival on a quick there-and-back", () => {
		let a = trackArrival(null, "stop", 1000);
		a = trackArrival(a, "send", 5000); // the turn flag dips
		a = trackArrival(a, "stop", 5000 + FLICKER_MS - 1); // and returns
		expect(a.arrivedAt).toBe(1000);
	});
	it("re-arms when the return is not a flicker", () => {
		let a = trackArrival(null, "stop", 1000);
		a = trackArrival(a, "send", 5000);
		a = trackArrival(a, "stop", 5000 + FLICKER_MS);
		expect(a.arrivedAt).toBe(5000 + FLICKER_MS);
	});
});

describe("micSlot: reserved while unknown, collapsed when it can never exist", () => {
	it("collapses from the first frame when the browser cannot record", () => {
		expect(micSlot(false, null)).toBe("none");
		expect(micSlot(false, { stt: { available: true, path: "x" } })).toBe("none");
	});
	it("reserves the box until the daemon answers", () => {
		expect(micSlot(true, null)).toBe("reserved");
	});
	it("draws the mic when a path can run, else collapses", () => {
		expect(micSlot(true, { stt: { available: true, path: "cloud" } })).toBe("mic");
		expect(micSlot(true, { stt: { available: false, path: null } })).toBe("none");
		expect(micSlot(true, {})).toBe("none");
	});
});
