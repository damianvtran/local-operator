// @vitest-environment happy-dom
//
// Where a chip may appear inside an agent's answer, and where it may not. The
// id is a citation in prose (bare or in a code span of its own); a fence, a
// link label and a code span that also holds something else are text the
// reader was SHOWN, not a control. And a surface that did not ask for chips
// never gets one, even when a host is present.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AskChipHostProvider } from "./components/ask-id";
import { Markdown } from "./components/markdown";

afterEach(cleanup);

function host() {
	const open = vi.fn();
	return {
		open,
		value: {
			facts: new Map([["a-745e", { status: "open" as const, question: "Which region?" }]]),
			open,
		},
	};
}

describe("Markdown ask chips", () => {
	it("draws a chip for an id in the agent's prose and opens it", () => {
		const { value, open } = host();
		render(
			<AskChipHostProvider value={value}>
				<Markdown text={"the old one, a-745e, is withdrawn"} askChips />
			</AskChipHostProvider>,
		);
		const chip = screen.getByRole("button", { name: /Ask a-745e, open/ });
		expect(screen.getByText(/the old one,/)).toBeTruthy();
		fireEvent.click(chip);
		expect(open).toHaveBeenCalledWith("a-745e");
	});

	it("draws a chip for a code span that holds only the id", () => {
		const { value } = host();
		render(
			<AskChipHostProvider value={value}>
				<Markdown text={"see `a-745e`"} askChips />
			</AskChipHostProvider>,
		);
		expect(screen.getByRole("button", { name: /Ask a-745e/ })).toBeTruthy();
	});

	it("keeps a code span that holds more than the id as code", () => {
		const { value } = host();
		const { container } = render(
			<AskChipHostProvider value={value}>
				<Markdown text={"run `lop ask a-745e`"} askChips />
			</AskChipHostProvider>,
		);
		expect(screen.queryByRole("button")).toBeNull();
		expect(container.querySelector("code")?.textContent).toBe("lop ask a-745e");
	});

	it("keeps a fenced block plain, even when it names an id", () => {
		const { value } = host();
		const { container } = render(
			<AskChipHostProvider value={value}>
				<Markdown text={"```\nAsk a-745e queued\n```"} askChips />
			</AskChipHostProvider>,
		);
		expect(screen.queryByRole("button")).toBeNull();
		expect(container.querySelector("pre")?.textContent).toContain("a-745e");
	});

	it("keeps a link's label plain and its target intact", () => {
		const { value } = host();
		const { container } = render(
			<AskChipHostProvider value={value}>
				<Markdown text={"[a-745e](https://example.test/asks)"} askChips />
			</AskChipHostProvider>,
		);
		expect(screen.queryByRole("button")).toBeNull();
		expect(container.querySelector("a")?.getAttribute("href")).toBe("https://example.test/asks");
	});

	it("leaves an id inside a URL as text (a path is not a citation)", () => {
		const { value } = host();
		render(
			<AskChipHostProvider value={value}>
				<Markdown text={"see https://x.test/a-745e"} askChips />
			</AskChipHostProvider>,
		);
		expect(screen.queryByRole("button")).toBeNull();
	});

	it("draws a chip in a table cell, where an agent often lists asks", () => {
		const { value } = host();
		const table = "| id | note |\n|---|---|\n| a-745e | x |";
		render(
			<AskChipHostProvider value={value}>
				<Markdown text={table} askChips />
			</AskChipHostProvider>,
		);
		expect(screen.getByRole("button", { name: /Ask a-745e/ })).toBeTruthy();
	});

	it("leaves italic markup alone when chips are off", () => {
		// Parsing the italic body recursively is needed for a chip inside it,
		// but it must not turn backticks into code for every other consumer.
		const { container } = render(<Markdown text={"*a`b`*"} />);
		expect(container.querySelector("em")?.textContent).toBe("a`b`");
		expect(container.querySelector("code")).toBeNull();
	});

	it("never chips when the surface did not ask for them", () => {
		const { value } = host();
		render(
			<AskChipHostProvider value={value}>
				<Markdown text={"a-745e"} />
			</AskChipHostProvider>,
		);
		expect(screen.queryByRole("button")).toBeNull();
	});
});
