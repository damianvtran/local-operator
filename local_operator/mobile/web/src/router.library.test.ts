// @vitest-environment happy-dom
//
// The two Library routes are parsed here rather than in a screen test: the
// router is the only place a deep link and the footer's `navigate("/library")`
// can disagree, and an unknown path must keep landing on the list.
import { describe, expect, it } from "vitest";
import { navigate, parseHash } from "./router";

describe("library routes", () => {
	it("parses the list and a deep-linked asset", () => {
		expect(parseHash("#/library")).toEqual({ name: "library" });
		expect(parseHash("#/library/a/la_0123456789abcdef")).toEqual({
			name: "library-asset",
			assetId: "la_0123456789abcdef",
		});
	});

	it("decodes an id and still parses the surrounding routes", () => {
		expect(parseHash("#/library/a/la_0123456789abcdef%2Ftail")).toEqual({
			name: "library-asset",
			assetId: "la_0123456789abcdef/tail",
		});
		expect(parseHash("#/past")).toEqual({ name: "past" });
		expect(parseHash("#/nonsense")).toEqual({ name: "list" });
	});

	it("navigates by hash and dispatches the event the route effect listens for", () => {
		let fired = 0;
		const listener = () => {
			fired += 1;
		};
		window.addEventListener("popstate", listener);
		navigate("/library");
		expect(location.hash).toBe("#/library");
		expect(fired).toBe(1);
		window.removeEventListener("popstate", listener);
	});
});
