import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../api";
import { resetTurnStopForTests, STOPPING_BOUND_MS, stopTurn } from "./turn-stop";

vi.mock("../api", async (importOriginal) => ({
	...(await importOriginal<typeof import("../api")>()),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
}));

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
	resetTurnStopForTests();
	vi.useRealTimers();
	vi.clearAllMocks();
});

describe("stopTurn: one abort per pending stop", () => {
	it("sends one abort, and a second call while pending is a no-op", async () => {
		await stopTurn("s1");
		await stopTurn("s1");
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
		expect(api.sendCommand).toHaveBeenCalledWith("s1", { op: "abort" });
	});

	it("is per session", async () => {
		await stopTurn("s1");
		await stopTurn("s2");
		expect(api.sendCommand).toHaveBeenCalledTimes(2);
	});

	it("releases the latch after the 15 s bound, so a lost turn-end cannot leave Stop dead", async () => {
		await stopTurn("s1");
		vi.advanceTimersByTime(STOPPING_BOUND_MS - 1);
		await stopTurn("s1");
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
		vi.advanceTimersByTime(1);
		await stopTurn("s1");
		expect(api.sendCommand).toHaveBeenCalledTimes(2);
	});

	it("resolves to the failure message and releases the latch when the abort is rejected", async () => {
		vi.mocked(api.sendCommand).mockRejectedValueOnce(new Error("relay unreachable"));
		expect(await stopTurn("s1")).toBe("relay unreachable");
		// released at once: the user can try again without waiting out the bound
		expect(await stopTurn("s1")).toBeNull();
		expect(api.sendCommand).toHaveBeenCalledTimes(2);
	});
});
