// @vitest-environment happy-dom
//
// The grouping preference's contract: default OFF, "on" is the literal `1`,
// and turning it off REMOVES the key so the released default is restored.
import { afterEach, describe, expect, it } from "vitest";
import { getGroupTags, setGroupTags } from "./session-group-pref";

const KEY = "lo-mobile-group-tags";

afterEach(() => {
	localStorage.clear();
});

describe("the local grouping preference", () => {
	it("defaults off when the key is absent", () => {
		expect(getGroupTags()).toBe(false);
	});

	it("round-trips through localStorage", () => {
		setGroupTags(true);
		expect(localStorage.getItem(KEY)).toBe("1");
		expect(getGroupTags()).toBe(true);
	});

	it("removes the key when turned off", () => {
		setGroupTags(true);
		setGroupTags(false);
		expect(localStorage.getItem(KEY)).toBeNull();
		expect(getGroupTags()).toBe(false);
	});

	it("reads only the literal 1 as on", () => {
		localStorage.setItem(KEY, "0");
		expect(getGroupTags()).toBe(false);
	});
});
