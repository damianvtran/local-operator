// @vitest-environment happy-dom
//
// Two layout contracts for the phone's session column, both regressions.
//
// 1. The subagent roster starts COLLAPSED. It used to open whenever any agent
//    was running (`defaultOpen={running > 0}`), so a coordinating session with
//    22 rows painted the whole roster on arrival and left the transcript a
//    16px sliver with the composer off screen (measured at 390x844).
//
// 2. Every ask option stays REACHABLE. The card is an unshrinkable sibling of
//    the transcript inside a `h-dvh overflow-hidden` column, so an uncapped
//    card simply grew past the viewport and the tail was clipped with nothing
//    to scroll — a 10-option ask measured 1426px against an 844px viewport.
//
// These are asserted against the REAL SessionScreen, so a reverted default or
// a dropped cap fails here. happy-dom does no layout (every box is 0x0), so
// the assertions are on the structural contract that produces the layout —
// the roster's `aria-expanded`, and the presence of the cap + scroller classes
// on the card's own elements — not on measured pixels, which this environment
// cannot supply. The pixel evidence lives on the PR.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type {
	PendingRequest,
	SessionProjection,
	SubagentRow,
} from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answer accepted" })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
}));

let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

function row(jobId: string, status: SubagentRow["status"]): SubagentRow {
	return {
		job_id: jobId,
		label: jobId,
		agent: "coder",
		status,
		progress: "",
		elapsed_s: 1,
		model_label: "",
		result_text: "",
		error_text: "",
		parent_job_id: null,
		session_id: `${jobId}-session`,
		prompt: "",
		launch_message_id: "",
		effort: "high",
		ancestors: [],
		ancestor_ids: [],
		child_ids: [],
		peer_ids: [],
		transcript: [],
		todos: [],
		activity: "",
	};
}

function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Overflow",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
		...over,
	} satisfies SessionProjection;
}

/** The operator's reported screen: 22 rows, exactly one of them running. */
function roster(): SubagentRow[] {
	return [
		row("viewport-audit-running", "running"),
		...Array.from({ length: 21 }, (_, i) =>
			row(`surface-audit-${String(i + 1).padStart(2, "0")}`, "completed"),
		),
	];
}

function askPending(optionCount: number): PendingRequest {
	return {
		request_id: "req-long",
		kind: "ask",
		title: "A deliberately long question about rollout sequencing".repeat(4),
		detail: "",
		options: Array.from({ length: optionCount }, (_, i) => ({
			label: `option-${String(i + 1).padStart(2, "0")}`,
			description: "A paragraph-length consequence line for this option. ".repeat(4),
		})),
		secret: false,
		question_index: 0,
		question_total: 1,
	};
}

/** The card's own element: the accent-bordered block the render site mounts. */
function cardRoot(): HTMLElement {
	const el = document.querySelector<HTMLElement>(".border-accent");
	if (!el) throw new Error("pending card not mounted");
	return el;
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	slot = { projection: null, connected: true };
});

describe("subagent roster default", () => {
	it("opens a conversation with the roster collapsed even while an agent runs", () => {
		slot = { projection: projection({ subagents: roster() }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// The header line stays as the at-a-glance signal...
		expect(screen.getByText("1/22 running")).toBeTruthy();
		// ...but its rows are not painted. `running > 0` used to open this.
		const header = screen.getByRole("button", { name: /subagents/ });
		expect(header.getAttribute("aria-expanded")).toBe("false");
		expect(screen.queryByRole("button", { name: /surface-audit-01/ })).toBeNull();
	});

	it("expands on tap into a capped, internally scrolling body", () => {
		slot = { projection: projection({ subagents: roster() }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		const header = screen.getByRole("button", { name: /subagents/ });
		fireEvent.click(header);

		// The a11y contract survives the new default.
		expect(header.getAttribute("aria-expanded")).toBe("true");
		const firstRow = screen.getByRole("button", { name: /surface-audit-01/ });
		expect(firstRow).toBeTruthy();
		// Tap-target convention: every row keeps its 44px minimum.
		expect(firstRow.className).toContain("min-h-11");

		// The expanded body is bounded and scrolls itself, so 22 rows cannot
		// push the transcript out of the column the way they used to.
		const scroller = firstRow.closest(".lo-scroll");
		expect(scroller).not.toBeNull();
		expect(scroller?.className).toContain("max-h-[40dvh]");
		expect(scroller?.className).toContain("overflow-y-auto");
	});

	it("keeps the roster collapsed when nothing is running", () => {
		slot = {
			projection: projection({ subagents: [row("done-01", "completed")] }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);
		expect(
			screen.getByRole("button", { name: /subagents/ }).getAttribute("aria-expanded"),
		).toBe("false");
	});
});

describe("ask card reachability", () => {
	it("caps the card and scrolls its body, so the last of ten options is reachable", () => {
		slot = { projection: projection({ pending: askPending(10), pending_count: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// Every option is mounted — nothing is dropped to make the card fit.
		expect(screen.getByRole("button", { name: /option-10/ })).toBeTruthy();

		// The card is bounded against the viewport rather than growing past it.
		const card = cardRoot();
		expect(card.className).toContain("max-h-[60dvh]");
		// ...and it does not shrink to nothing when the transcript is long.
		expect(card.className).toContain("shrink-0");

		// The bound only helps if the overflow is scrollable: the last option
		// must sit inside a scroller that the card owns. Without this the cap
		// would clip the tail exactly as `overflow-hidden` on the column did.
		const scroller = screen
			.getByRole("button", { name: /option-10/ })
			.closest(".lo-scroll");
		expect(scroller).not.toBeNull();
		expect(scroller?.className).toContain("overflow-y-auto");
		expect(card.contains(scroller)).toBe(true);
		// `min-h-0` is what lets the flex child shrink below its content; a
		// flex child defaults to `min-height:auto` and would refuse to.
		expect(scroller?.className).toContain("min-h-0");
	});

	it("scrolls the question too, since a long question can outgrow the cap alone", () => {
		const pending = askPending(2);
		slot = { projection: projection({ pending, pending_count: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// The question shares the scroller with the options rather than being
		// pinned above it, or it would reintroduce the unreachable tail.
		const question = screen.getByText(pending.title);
		expect(question.closest(".lo-scroll")).not.toBeNull();
	});

	it("bounds the approval variant, whose approve/deny pair overflowed the same way", () => {
		slot = {
			projection: projection({
				pending: {
					request_id: "req-approval",
					kind: "approval",
					title: "bash",
					detail: "A long command plus its consequences. ".repeat(40),
					options: [],
					secret: false,
					question_index: 0,
					question_total: 1,
				},
				pending_count: 1,
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		const approve = screen.getByRole("button", { name: "approve" });
		const card = cardRoot();
		expect(card.className).toContain("max-h-[60dvh]");
		const scroller = approve.closest(".lo-scroll");
		expect(scroller).not.toBeNull();
		expect(card.contains(scroller)).toBe(true);
		// The remember checkbox rides the same scroller as the buttons, so a
		// long detail cannot strand it above the fold either.
		expect(screen.getByRole("checkbox").closest(".lo-scroll")).toBe(scroller);
	});

	it("bounds the free-text/secret variant so its input and send stay reachable", () => {
		slot = {
			projection: projection({
				pending: {
					request_id: "req-secret",
					kind: "ask",
					title: "Paste the staging token".repeat(10),
					detail: "Why this credential is needed, at length. ".repeat(20),
					options: [],
					secret: true,
					question_index: 0,
					question_total: 1,
				},
				pending_count: 1,
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		// Scoped to the card: the composer owns a send control of its own, and
		// the contract under test is the CARD's, not the column's.
		const card = cardRoot();
		const send = [...card.querySelectorAll("button")].find(
			(b) => b.textContent === "send",
		);
		if (!send) throw new Error("card has no send button");
		expect(card.className).toContain("max-h-[60dvh]");
		expect(card.contains(send.closest(".lo-scroll"))).toBe(true);
		// The masked input must be inside the same scroller as its button; a
		// split would let one scroll away from the other.
		const input = document.querySelector('input[type="password"]');
		expect(input?.closest(".lo-scroll")).toBe(send.closest(".lo-scroll"));
	});
});
