// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from "vitest";
import { clearPrivateSessionStorage } from "./private-storage";

afterEach(() => {
	localStorage.clear();
});

describe("clearPrivateSessionStorage", () => {
	it("removes every private draft and command envelope but keeps preferences", () => {
		/* U2: logout/401 must leave no uncertain instruction or draft behind for
		   the next user on the device, while non-private preferences survive. */
		localStorage.setItem("lo-mobile-command:session-a", "envelope-a");
		localStorage.setItem("lo-mobile-command:session-b", "envelope-b");
		localStorage.setItem("lo-mobile-draft:session-a", "draft-a");
		localStorage.setItem("lo-theme", "brand-dark");
		localStorage.setItem("unrelated-pref", "keep");

		clearPrivateSessionStorage();

		expect(localStorage.getItem("lo-mobile-command:session-a")).toBeNull();
		expect(localStorage.getItem("lo-mobile-command:session-b")).toBeNull();
		expect(localStorage.getItem("lo-mobile-draft:session-a")).toBeNull();
		// Preferences are not private content and must persist across sign-out.
		expect(localStorage.getItem("lo-theme")).toBe("brand-dark");
		expect(localStorage.getItem("unrelated-pref")).toBe("keep");
	});

	it("re-login after a clear starts with no private state (nothing survives)", () => {
		/* Simulate: user A leaves an envelope, signs out (clear), user B logs in.
		   B must see an empty private surface. */
		localStorage.setItem("lo-mobile-command:session-a", "envelope-a");
		localStorage.setItem("lo-mobile-draft:session-a", "draft-a");

		clearPrivateSessionStorage(); // logout / 401

		const survivors = Object.keys(localStorage).filter(
			(key) =>
				key.startsWith("lo-mobile-command:") || key.startsWith("lo-mobile-draft:"),
		);
		expect(survivors).toEqual([]);
	});
});
