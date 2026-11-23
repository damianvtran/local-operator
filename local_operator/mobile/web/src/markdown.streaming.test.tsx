// @vitest-environment happy-dom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { Markdown } from "./components/markdown";

afterEach(cleanup);

it("keeps settled blocks and a focused link mounted while the answer grows", () => {
	const prefix = "## A heading\n\n[Reference](https://example.test) and **bold** text.\n\n";
	const { container, rerender } = render(<Markdown text={`${prefix}Streaming`} />);
	const heading = screen.getByText("A heading");
	const link = screen.getByRole("link");
	const tail = container.querySelector("p:last-child");
	link.focus();

	rerender(<Markdown text={`${prefix}Streaming reply`} />);

	expect(screen.getByText("A heading")).toBe(heading);
	expect(screen.getByRole("link")).toBe(link);
	expect(document.activeElement).toBe(link);
	expect(container.querySelector("p:last-child")).toBe(tail);
});

it("reparses the unfinished seam like a full parse, including markup changing kind on append", () => {
	const head = "## Heading\n\nSettled **first** block.\n\nSettled [second](https://example.test).\n\n";
	const tail = "Paragraph\n---wrong rule\n\n| Item | Code |\n| --- | --- |\n| x | `a` |\n\n" +
		"> note\n> another\n- one\n- **two**\n\n```sh\necho  hi\n```\n\nend";
	const streaming = render(<Markdown text={head} />);
	for (let i = 1; i <= tail.length; i++) {
		const text = head + tail.slice(0, i);
		streaming.rerender(<Markdown text={text} />);
		const baseline = render(<Markdown text={text} />);
		expect(streaming.container.innerHTML, `snapshot at ${i}`).toBe(baseline.container.innerHTML);
		baseline.unmount();
	}
});

it("preserves early table cells when another streamed row arrives", () => {
	const table = "| Name | Value |\n| --- | --- |\n| [One](https://example.test) | good |";
	const { rerender } = render(<Markdown text={table} />);
	const link = screen.getByRole("link");
	link.focus();
	rerender(<Markdown text={`${table}\n| Two | also good |`} />);
	expect(screen.getByRole("link")).toBe(link);
	expect(document.activeElement).toBe(link);
});
