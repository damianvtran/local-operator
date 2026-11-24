// @vitest-environment happy-dom
//
// The two shapes of an ask id on this surface: the quiet copy foot under a
// card or settled row, and the chip inside an agent's sentence. What matters
// here is the rule, not the paint: the foot copies and says so (and selects
// the id when the page has no clipboard, as a plain-HTTP LAN page does not);
// a chip exists only for an id THIS session can answer for and opens exactly
// that ask; an id it cannot answer for stays plain text.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AskChipHostProvider, AskIdChip, AskIdFoot } from "./components/ask-id";
import type { AskChipFact } from "./lib/ask-chip";

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
	delete (navigator as { clipboard?: unknown }).clipboard;
});

function host(facts: [string, AskChipFact][], open = vi.fn()) {
	return { host: { facts: new Map(facts), open }, open };
}

describe("AskIdFoot", () => {
	it("copies the id and confirms it by name", async () => {
		const writeText = vi.fn(async () => {});
		Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
		render(<AskIdFoot askId="a-745e" />);
		const button = screen.getByRole("button", { name: "Copy ask id a-745e" });
		fireEvent.click(button);
		await waitFor(() => expect(writeText).toHaveBeenCalledWith("a-745e"));
		await waitFor(() => expect(screen.getByText("Copied")).toBeTruthy());
		expect(screen.getByText("Copied a-745e")).toBeTruthy();
	});

	it("selects the id and explains when the clipboard is unavailable", async () => {
		Object.defineProperty(navigator, "clipboard", {
			value: { writeText: vi.fn(async () => { throw new Error("insecure context"); }) },
			configurable: true,
		});
		render(<AskIdFoot askId="a-745e" />);
		fireEvent.click(screen.getByRole("button", { name: "Copy ask id a-745e" }));
		await waitFor(() =>
			expect(screen.getByText("Couldn't copy — the id is selected; copy it from there")).toBeTruthy(),
		);
		expect(window.getSelection()?.toString()).toBe("a-745e");
	});
});

describe("AskIdChip", () => {
	it("shows the state word beside the dot and opens its ask", () => {
		const { host: value, open } = host([["a-745e", { status: "open", question: "Which region?" }]]);
		render(
			<AskChipHostProvider value={value}>
				<AskIdChip id="a-745e" />
			</AskChipHostProvider>,
		);
		const chip = screen.getByRole("button", { name: "Ask a-745e, open: Which region?. Activate to show." });
		expect(chip.getAttribute("data-ask-status")).toBe("open");
		expect(chip.textContent).toContain("a-745e");
		expect(chip.textContent).toContain("open");
		fireEvent.click(chip);
		expect(open).toHaveBeenCalledWith("a-745e");
	});

	it("names a settled state with its own word, never colour alone", () => {
		const { host: value } = host([["a-745e", { status: "withdrawn", question: "" }]]);
		render(
			<AskChipHostProvider value={value}>
				<AskIdChip id="a-745e" />
			</AskChipHostProvider>,
		);
		expect(screen.getByRole("button", { name: "Ask a-745e, withdrawn. Activate to show." }).textContent).toContain("withdrawn");
	});

	it("leaves an id this session cannot answer for as plain text", () => {
		const { host: value } = host([]);
		render(
			<AskChipHostProvider value={value}>
				<AskIdChip id="a-9999" />
			</AskChipHostProvider>,
		);
		expect(screen.queryByRole("button")).toBeNull();
		expect(screen.getByTestId("ask-id-plain").textContent).toBe("a-9999");
	});

	it("stays plain with no host at all (a surface that draws no ask state)", () => {
		render(<AskIdChip id="a-745e" />);
		expect(screen.queryByRole("button")).toBeNull();
		expect(screen.getByTestId("ask-id-plain")).toBeTruthy();
	});
});
