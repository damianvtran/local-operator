// @vitest-environment happy-dom
//
// The asks sheet — the EXPANDED surface (design §5.3). It reads the aggregate
// from `GET /api/asks` (so it works for a conversation the phone is not
// addressing, including one woken by a wake or a monitor), labels a foreign
// row with its conversation, and re-reads when the outstanding population
// moves rather than waiting for a manual refresh.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AsksSheet } from "./components/asks-sheet";
import type { PendingAsk, SessionSummary } from "./types";

vi.mock("./api", () => ({
	getAsks: vi.fn(async () => ({ asks: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answered" })),
}));

let rows: SessionSummary[] = [];
let revision = 0;
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: rows, connected: true }),
		useAsksRevision: () => revision,
	};
});

const { getAsks } = await import("./api");

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a1",
		session_id: "other",
		cwd: "/tmp/w",
		created_at: 1,
		expires_at: Date.now() + 600_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [
			{ id: "q1", question: "ship the fix?", options: [], multi: false, secret: false, persist: false },
		],
		...patch,
	};
}

function summary(patch: Partial<SessionSummary> = {}): SessionSummary {
	return {
		session_id: "other",
		section: "active",
		conversation_name: "woken-by-monitor",
		cwd: "/tmp/w",
		model_label: "",
		streaming: false,
		needs_attention: false,
		pending_kind: "",
		subagents_running: 0,
		todos_open: 0,
		mtime: 0,
		...patch,
	};
}

beforeEach(() => {
	rows = [];
	revision = 0;
	(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask()] });
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

describe("AsksSheet", () => {
	it("renders nothing until it is opened, and reads the aggregate when it is", async () => {
		const { container } = render(<AsksSheet open={false} onClose={() => {}} />);
		expect(container.firstChild).toBeNull();
		expect(getAsks).not.toHaveBeenCalled();
	});

	it("lists a foreign conversation's ask under its own name, with a way in", async () => {
		rows = [summary()];
		const navigate = vi.fn();
		const close = vi.fn();
		render(
			<AsksSheet open onClose={close} currentSessionId="mine" onOpenConversation={navigate} />,
		);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		expect(screen.getByText("woken-by-monitor")).toBeTruthy();
		const link = screen.getByRole("button", { name: "open" });
		fireEvent.click(link);
		expect(navigate).toHaveBeenCalledWith("other");
		/* THE SHEET DOES NOT CLOSE ITSELF HERE (agent review round 2, M1 = design
		   D7's neighbour): a close after the route change popped the entry the
		   navigation had just pushed. The parent owns the whole transition. */
		expect(close).not.toHaveBeenCalled();
		/* And it is a 44 px control like every other on this sheet (D7 = U9),
		   not the 28x17 bare link it was. */
		expect(link.className).toMatch(/min-h-11/);
		expect(link.className).not.toMatch(/underline/);
	});

	it("answers against the row's OWN session route, not the screen's", async () => {
		const { sendCommand } = await import("./api");
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		fireEvent.change(screen.getByPlaceholderText("your answer"), {
			target: { value: "yes — dark behind the flag" },
		});
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("other", {
				op: "ask_respond",
				ask_id: "a1",
				answers: { q1: ["yes — dark behind the flag"] },
			}),
		);
	});

	it("says nothing is waiting when the aggregate is empty", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [] });
		render(<AsksSheet open onClose={() => {}} />);
		await waitFor(() => expect(screen.getByText(/nothing waiting/)).toBeTruthy());
	});

	it("re-reads when the outstanding population moves, without a manual refresh", async () => {
		const { rerender } = render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(getAsks).toHaveBeenCalledTimes(1));
		revision = 1;
		rerender(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(getAsks).toHaveBeenCalledTimes(2));
	});

	it("states the daemon's refusal reason when the read fails", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("the ask index could not be read"),
		);
		render(<AsksSheet open onClose={() => {}} />);
		await waitFor(() =>
			expect(screen.getByText("the ask index could not be read")).toBeTruthy(),
		);
	});

	it("leads with the head ask — the order the minimized chip names", async () => {
		/* U8 (UX round 1): the chip named the oldest open ask while the sheet led
		   with the newest, so a tap landed on a question the reader had not just
		   read. The sheet now shares `orderedForDisplay` with the chip. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "a-new", created_at: 900 }), ask({ ask_id: "a-old", created_at: 1 })],
		});
		render(<AsksSheet open onClose={() => {}} />);
		await waitFor(() => expect(screen.getAllByTestId("ask-card").length).toBe(2));
		const cards = screen.getAllByTestId("ask-card");
		expect(cards[0].getAttribute("data-ask-id")).toBe("a-old");
	});

	it("offers a retry when the read fails, instead of closing being the only move", async () => {
		/* U6 (UX round 1): a hung or failed aggregate read left the sheet with no
		   way forward but closing it. The hung case is bounded in the component
		   (`AbortSignal.timeout`); the failed case gets this control. */
		(getAsks as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("could not reach the daemon"),
		);
		render(<AsksSheet open onClose={() => {}} />);
		await waitFor(() => expect(screen.getByText("could not reach the daemon")).toBeTruthy());
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask()] });
		fireEvent.click(screen.getByRole("button", { name: /try again/ }));
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
	});

	it("warns before the tap when the owning conversation has ended", async () => {
		rows = [summary({ ended: true })];
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() =>
			expect(screen.getByText(/this conversation has ended/)).toBeTruthy(),
		);
	});
});
