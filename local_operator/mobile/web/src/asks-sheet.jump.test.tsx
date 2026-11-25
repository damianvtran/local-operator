// @vitest-environment happy-dom
//
// A chip's tap promises one question. The sheet must therefore open ON that
// ask — not merely show a list containing it: a settled (withdrawn) row that
// has no other trace expands in place, the body scrolls to the card, and the
// reader's place moves onto it. A jump to an ask that is still open lands on
// its first question instead.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AsksSheet } from "./components/asks-sheet";
import type { PendingAsk, SessionProjection, SessionSummary } from "./types";

vi.mock("./api", () => ({
	getAsks: vi.fn(async () => ({ asks: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answered" })),
}));

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: [] as SessionSummary[], connected: true, revision: 0 }),
		useProjection: () => ({ projection: null as SessionProjection | null, connected: true }),
		useAsksRevision: () => 0,
	};
});

const { getAsks } = await import("./api");

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a-745e",
		session_id: "mine",
		cwd: "/tmp/w",
		created_at: 1,
		expires_at: Date.now() + 600_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [
			{ id: "q1", question: "Which region?", options: [], multi: false, secret: false, persist: false },
		],
		...patch,
	};
}

beforeEach(() => {
	(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask()] });
});

afterEach(() => {
	cleanup();
	vi.useRealTimers();
	vi.clearAllMocks();
	vi.restoreAllMocks();
	delete (navigator as { clipboard?: unknown }).clipboard;
	delete (document as { execCommand?: unknown }).execCommand;
});

describe("a chip's jump into the asks sheet", () => {
	it("expands a settled ask in place and moves focus onto its card", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "a-e53b", status: "open" }), ask({ ask_id: "a-745e", status: "withdrawn" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" jumpAsk="a-745e" />);

		const card = await waitFor(() => {
			const found = screen.getAllByTestId("ask-card").find((node) => node.getAttribute("data-ask-id") === "a-745e");
			if (!found) throw new Error("the jumped-to card is not open");
			return found;
		});
		await waitFor(() => expect(document.activeElement).toBe(card));
		// The other ask is collapsed to its one-line row, not drawn as a card.
		expect(screen.getAllByTestId("ask-row").map((row) => row.getAttribute("data-ask-id"))).toContain("a-e53b");
		// The settled row states its id once, beside the row; the expanded card
		// under it does not repeat the same six characters. The other ask is an
		// open card and carries its own foot.
		const feet = screen.getAllByTestId("ask-id-foot").map((foot) => foot.getAttribute("data-ask-id-copy"));
		expect(feet).toContain("a-745e");
		expect(feet.filter((id) => id === "a-745e")).toHaveLength(1);
	});

	it("lands on the first question of an ask that is still open", async () => {
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" jumpAsk="a-745e" />);
		const question = await screen.findByTestId("ask-question-text");
		await waitFor(() => expect(document.activeElement).toBe(question));
	});

	it("scrolls the body so the jumped-to card sits at its rest offset", async () => {
		const card = ask({ ask_id: "a-e53b", status: "open" });
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "a-745e", status: "withdrawn" }), card],
		});
		vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
			// The card sits 200 px below the scroller's top edge.
			if (this.getAttribute("data-ask-id") === "a-745e") return { top: 200 } as DOMRect;
			return { top: 0 } as DOMRect;
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" jumpAsk="a-745e" />);
		const region = await waitFor(() => {
			const node = document.querySelector<HTMLElement>('[role="dialog"] .lo-scroll');
			if (!node) throw new Error("no sheet scroller");
			return node;
		});
		await waitFor(() => expect(region.scrollTop).toBeGreaterThan(0));
	});

	it("does not move focus when the sheet was opened without a jump", async () => {
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await screen.findByTestId("ask-question-text");
		await waitFor(() => expect(document.activeElement).not.toBe(screen.getByTestId("ask-question-text")));
	});

	it("spends the jump once, so a later re-read cannot pull the reader back", async () => {
		vi.useFakeTimers({ shouldAdvanceTime: true });
		/* A FRESH array per read, as the daemon sends: reusing one object lets
		   React bail out of `setRows` and hides the defect this test exists for. */
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({
			asks: [ask({ ask_id: "a-e53b", status: "open" }), ask({ ask_id: "a-745e", status: "withdrawn" })],
		}));
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" jumpAsk="a-745e" />);
		await waitFor(() => expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a-745e"));
		// The reader moves to the other ask.
		const other = screen.getAllByTestId("ask-row").find((row) => row.getAttribute("data-ask-id") === "a-e53b");
		fireEvent.click(other as HTMLElement);
		await waitFor(() => expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a-e53b"));
		const reads = (getAsks as ReturnType<typeof vi.fn>).mock.calls.length;
		// One backstop tick is a re-read: the exact shape that used to re-apply.
		await act(async () => {
			vi.advanceTimersByTime(21_000);
		});
		expect((getAsks as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThan(reads);
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a-e53b");
	});

	it("keeps a failed copy's notice on its own line and the row still opens", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "a-745e", status: "withdrawn" })],
		});
		Object.defineProperty(navigator, "clipboard", {
			value: { writeText: vi.fn(async () => { throw new Error("insecure context"); }) },
			configurable: true,
		});
		Object.defineProperty(document, "execCommand", { value: vi.fn(() => false), configurable: true });
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await screen.findByTestId("ask-row");
		fireEvent.click(screen.getByRole("button", { name: "Copy ask id a-745e" }));
		const notice = await screen.findByText("Couldn't copy — the id is selected; copy it from there");
		// The full-line basis is what stops it becoming the flex item that
		// squeezed the row's button to zero width; the parent wraps so it can.
		expect(notice.className).toContain("w-full");
		expect(screen.getByTestId("ask-row").parentElement?.className).toContain("flex-wrap");
		fireEvent.click(screen.getByTestId("ask-row"));
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
	});
});
