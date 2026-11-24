import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { busyDeliverySupported, deliveryForSend, getDeliveryMode, isDeliveryMode, nextDeliveryMode, setDeliveryMode, DEFAULT_DELIVERY_MODE, DELIVERY_MODES } from "./delivery";
import type { Capabilities } from "../types";

function caps(features?: Record<string, unknown>): Capabilities {
	return { features: features ?? {} } as unknown as Capabilities;
}

describe("delivery modes", () => {
	it("cycles steer → queue → both → steer", () => {
		expect(nextDeliveryMode("steer")).toBe("queue");
		expect(nextDeliveryMode("queue")).toBe("both");
		expect(nextDeliveryMode("both")).toBe("steer");
	});

	it("validates the vocabulary and defaults", () => {
		expect(isDeliveryMode("queue")).toBe(true);
		expect(isDeliveryMode("sometimes")).toBe(false);
		expect(isDeliveryMode(undefined)).toBe(false);
		expect(DEFAULT_DELIVERY_MODE).toBe("steer");
		expect(DELIVERY_MODES).toEqual(["steer", "queue", "both"]);
	});

	it("gates on the feature key", () => {
		expect(busyDeliverySupported(caps({ busy_delivery: 1 }))).toBe(true);
		expect(busyDeliverySupported(caps({ busy_delivery: 0 }))).toBe(false);
		expect(busyDeliverySupported(caps())).toBe(false);
		expect(busyDeliverySupported(null)).toBe(false);
	});

	it("sends nothing for steer, idle, or unsupported", () => {
		expect(deliveryForSend("steer", true, true)).toBeUndefined();
		expect(deliveryForSend("queue", false, true)).toBeUndefined();
		expect(deliveryForSend("queue", true, false)).toBeUndefined();
		expect(deliveryForSend("queue", true, true)).toBe("queue");
		expect(deliveryForSend("both", true, true)).toBe("both");
	});

	describe("persistence", () => {
		beforeEach(() => {
			localStorage.clear();
		});
		afterEach(() => {
			localStorage.clear();
		});

		it("round-trips each mode and clears the default", () => {
			expect(getDeliveryMode()).toBe("steer");
			setDeliveryMode("queue");
			expect(getDeliveryMode()).toBe("queue");
			setDeliveryMode("both");
			expect(getDeliveryMode()).toBe("both");
			setDeliveryMode("steer");
			expect(getDeliveryMode()).toBe("steer");
			expect(localStorage.getItem("lo-mobile-delivery-mode")).toBeNull();
		});

		it("falls back to steer on a corrupt value or storage outage", () => {
			localStorage.setItem("lo-mobile-delivery-mode", "sometimes");
			expect(getDeliveryMode()).toBe("steer");
			const original = localStorage.getItem.bind(localStorage);
			vi.spyOn(localStorage, "getItem").mockImplementationOnce(() => {
				throw new Error("denied");
			});
			expect(getDeliveryMode()).toBe("steer");
			expect(typeof original).toBe("function");
		});
	});
});
