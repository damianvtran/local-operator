// @vitest-environment happy-dom
//
// The session-list footer's wide toggle is absent in an installed app, where
// wide view is never applied (lib/viewport.ts `isStandalone`): a dead button
// that is easy to hit by accident is how an installed app got stuck small.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WideViewButton } from "./wide-view-button";

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	localStorage.clear();
});

function displayMode(standalone: boolean): void {
	vi.stubGlobal("matchMedia", (query: string) => ({
		matches: standalone && query === "(display-mode: standalone)",
		media: query,
	}));
}

describe("WideViewButton", () => {
	it("renders in a browser tab", () => {
		displayMode(false);
		render(<WideViewButton />);
		expect(screen.getByRole("button", { name: "wide view" })).toBeTruthy();
	});

	it("renders nothing in an installed app", () => {
		displayMode(true);
		const { container } = render(<WideViewButton />);
		expect(screen.queryByRole("button", { name: "wide view" })).toBeNull();
		expect(container.innerHTML).toBe("");
	});
});
