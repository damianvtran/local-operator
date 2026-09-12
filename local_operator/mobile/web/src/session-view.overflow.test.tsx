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
// a dropped cap fails here.
//
// WHAT THIS LAYER CANNOT PROVE, and which layer does.
//
// happy-dom does no layout: every box is 0x0, so "the last option is visible",
// "approve is 44px tall on arrival" and "one swipe reaches the foot" are all
// unanswerable here. Round 1 learned this the expensive way — four of the seven
// assertions were `className` echoes (`toContain("max-h-[60dvh]")`), which
// compare a string in the test to the same string in the source and therefore
// pass for a cap that is PRESENT BUT INEFFECTIVE. They passed for the U2
// regression that shipped approve/deny 0px visible at 360x780, and for the U1
// keyboard divergence, because a class name says nothing about what the class
// resolves to.
//
// So the assertions below are of two kinds, both of which can actually fail:
//
//  * RESOLVED STYLE, not class name. happy-dom does resolve `var()` against an
//    ancestor's custom property, so a cap written against the column's pinned
//    height resolves to a px calc while one written in `dvh` does not. That is
//    the U1/C1 property — the cap tracks the column rather than the dynamic
//    viewport — and a revert to `60dvh` fails it.
//  * CONTAINMENT, in both directions. The scroller must contain the option
//    list, and must NOT contain the controls or the error line. The second half
//    is U2/U3: a control inside the scroller can be scrolled out of reach, and
//    a class echo cannot see the difference.
//
// The GEOMETRIC property — that a finger reaches the last option, and that
// approve arrives at its full 44px — is proved one layer up, by
// `scripts/mobile_reachability_check.py`, which drives the real bundle in
// headless Chrome with real touch input and asserts those pixels. Do not read a
// green run of this file as proof of reachability.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { COLUMN_HEIGHT_VAR } from "./lib/column";
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

/** The card's own element. Selected by test id rather than `.border-accent`,
    which composer.tsx (drag-over) and new-session.tsx (selection) also apply,
    so the class would pick the wrong node the first time one of those states
    renders beside a card (C5). */
function cardRoot(): HTMLElement {
	const el = document.querySelector<HTMLElement>('[data-testid="pending-card"]');
	if (!el) throw new Error("pending card not mounted");
	return el;
}

/** The card's scrolling region — the one thing inside it that may scroll. */
function cardScroller(): HTMLElement {
	const el = cardRoot().querySelector<HTMLElement>(".lo-scroll");
	if (!el) throw new Error("card has no scroller");
	return el;
}

/** Pin the column the way the session view's visualViewport handler does, and
    return what `el`'s cap actually RESOLVES to at that column height.

    This is the U1/C1 property expressed as something that can fail. The column
    is pinned to `visualViewport.height` in px while a `dvh` cap follows the
    dynamic viewport, and a virtual keyboard shrinks the first and not the
    second — so the two diverge exactly when the keyboard is open. A cap in
    column units resolves against `--lo-vvh` and tightens with it; a `dvh` cap
    resolves to a viewport unit and does not.

    happy-dom lays nothing out, so the returned string is not a height. It is
    the resolved `calc()`, which is the most this layer can see and is enough to
    tell a column-relative cap from a viewport-relative one. The actual pixels
    are asserted by `scripts/mobile_reachability_check.py`. */
function resolvedCapWithColumnAt(el: HTMLElement, pinnedPx: number): string {
	const column = cardRoot().closest<HTMLElement>(".h-dvh");
	if (!column) throw new Error("card is not inside the session column");
	column.style.setProperty(COLUMN_HEIGHT_VAR, `${pinnedPx}px`);
	return getComputedStyle(el).maxHeight;
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
		const scroller = firstRow.closest<HTMLElement>(".lo-scroll");
		expect(scroller).not.toBeNull();
		expect(scroller?.className).toContain("overflow-y-auto");

		// The bound is measured against the COLUMN, not the dynamic viewport:
		// pin the column the way the visualViewport handler does and the cap
		// resolves against that number. A `40dvh` cap resolves to a viewport
		// unit instead and fails here — which is the keyboard-open divergence.
		const column = scroller?.closest<HTMLElement>(".h-dvh");
		column?.style.setProperty(COLUMN_HEIGHT_VAR, "480px");
		const cap = getComputedStyle(scroller as HTMLElement).maxHeight;
		expect(cap).toContain("480px");
		expect(cap).toContain("0.4");
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

		const card = cardRoot();
		// It does not shrink to nothing when the transcript is long.
		expect(card.className).toContain("shrink-0");

		// The cap TRACKS THE COLUMN. With the column pinned to a keyboard-open
		// 480px, the card's bound resolves against that 480 rather than against
		// the untouched dynamic viewport. This is the assertion that fails for
		// the `60dvh` cap of round 1: `dvh` does not shrink when a keyboard
		// overlays the visual viewport, so the cap stayed at 468px inside a
		// 480px column and put the controls under its clipped foot.
		expect(resolvedCapWithColumnAt(card, 480)).toContain("480px");
		// ...and it is a fraction of it, not the whole column: a card that may
		// claim the entire column has no cap in any useful sense.
		expect(resolvedCapWithColumnAt(card, 480)).toContain("0.6");

		// The bound only helps if the overflow is scrollable: the last option
		// must sit inside a scroller that the card owns. Without this the cap
		// would clip the tail exactly as `overflow-hidden` on the column did.
		const scroller = screen
			.getByRole("button", { name: /option-10/ })
			.closest(".lo-scroll");
		expect(scroller).toBe(cardScroller());
		expect(scroller?.className).toContain("overflow-y-auto");
		// `min-h-0` is what lets the flex child shrink below its content; a
		// flex child defaults to `min-height:auto` and would refuse to.
		expect(scroller?.className).toContain("min-h-0");
		// Arbitrary agent text cannot scroll the card sideways (C7).
		expect(scroller?.className).toContain("overflow-x-hidden");
	});

	it("pins the card's meta row above the scroller so it keeps its identity", () => {
		slot = {
			projection: projection({ pending: askPending(10), pending_count: 2 }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		// D2: scrolled to the option being tapped, the kind label and the "1 of
		// N" counter were both 0% visible — the card became an unlabelled list of
		// buttons at the exact moment of the decision. The row is one fixed line,
		// so pinning it costs no reachability.
		const kind = screen.getByText(/^question/);
		const counter = screen.getByText(/1 of 2/);
		const scroller = cardScroller();
		expect(scroller.contains(kind)).toBe(false);
		expect(scroller.contains(counter)).toBe(false);
		expect(cardRoot().contains(kind)).toBe(true);

		// U4: the option total is stated rather than discovered, since the cap
		// cut first-glance options from 6 to 3 at 390x844.
		expect(screen.getByText(/10 options/)).toBeTruthy();
	});

	it("scrolls the question too, since a long question can outgrow the cap alone", () => {
		const pending = askPending(2);
		slot = { projection: projection({ pending, pending_count: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// The question shares the scroller with the options rather than being
		// pinned above it, or it would reintroduce the unreachable tail. This is
		// the deliberate asymmetry with the meta row above: a title is unbounded
		// prose, a kind label is one line.
		const question = screen.getByText(pending.title);
		expect(question.closest(".lo-scroll")).toBe(cardScroller());
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
		const deny = screen.getByRole("button", { name: "deny" });
		const card = cardRoot();
		const scroller = cardScroller();

		// The cap tracks the column, as above.
		expect(resolvedCapWithColumnAt(cardRoot(), 480)).toContain("480px");

		// U2/Q1, the regression round 1 shipped: the DECISION is outside the
		// scroller. Inside it, an approval with a long detail arrived with
		// approve showing 30 of 44px at 390x844 and 0 of 44px at 360x780 — the
		// card's primary action below the fold on arrival, where the uncapped
		// card before it had shown both buttons whole. A control that can be
		// scrolled away is a control that can be missed.
		expect(scroller.contains(approve)).toBe(false);
		expect(scroller.contains(deny)).toBe(false);
		expect(scroller.contains(screen.getByRole("checkbox"))).toBe(false);
		expect(card.contains(approve)).toBe(true);
		// The detail is what scrolls instead — the content, not the controls.
		expect(screen.getByText(/A long command plus its consequences/).closest(".lo-scroll")).toBe(
			scroller,
		);
		// The action row ends clear of the overlay scrollbar's paint band, which
		// was drawn inside `deny` at gapToContentEdge 0 (D3).
		expect(approve.parentElement?.parentElement?.className).toContain("pr-1.5");
	});

	it("pins the stale-tap error outside the scroller, where it cannot land below the fold", async () => {
		const { sendCommand } = await import("./api");
		vi.mocked(sendCommand).mockRejectedValueOnce(
			new Error("that question moved on"),
		);
		slot = {
			projection: projection({ pending: askPending(10), pending_count: 1 }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		fireEvent.click(screen.getByRole("button", { name: /option-10/ }));
		const error = await screen.findByText(/That question moved on/);

		// U3: the error used to be the scroller's last child, so it was appended
		// ~26px BELOW where the user was standing — at the foot of the option
		// list, because that is where they had just tapped. Every option greyed
		// out and nothing explained why. Pinned, its position does not depend on
		// the user's scroll offset at all.
		expect(cardScroller().contains(error)).toBe(false);
		expect(cardRoot().contains(error)).toBe(true);

		// And the options must still read as inert while it shows, or a stale
		// option sits live-looking under the message (D7).
		expect(
			screen.getByRole("button", { name: /option-01/ }).hasAttribute("disabled"),
		).toBe(true);
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

		// The cap tracks the column, which for THIS variant is the whole point:
		// the keyboard is open exactly when a free-text answer is being typed,
		// and `send` is the only way to submit one (there is no Enter-to-send
		// path on this card). At a 300px keyboard on a 360x780 phone the old
		// `60dvh` cap left it below the column's clipped foot (U1/C1).
		expect(resolvedCapWithColumnAt(cardRoot(), 480)).toContain("480px");

		// The input and its send button are pinned together OUTSIDE the
		// scroller, so a long question cannot scroll either away.
		const input = document.querySelector<HTMLElement>('input[type="password"]');
		const scroller = cardScroller();
		expect(scroller.contains(send)).toBe(false);
		expect(scroller.contains(input as HTMLElement)).toBe(false);
		// ...and they stay together, or one could move without the other.
		expect(send.parentElement).toBe(input?.parentElement);
		expect(send.parentElement?.parentElement?.className).toContain("pr-1.5");
	});
});

describe("column budget while a request is pending", () => {
	it("holds the panels collapsed so they cannot push the decision off the column", () => {
		slot = {
			projection: projection({
				subagents: roster(),
				todos: [{ name: "Todos", items: [{ text: "audit", status: "pending", reason: "" }] }],
				pending: askPending(10),
				pending_count: 1,
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		// D1: the panels, the card and the composer are siblings in a column
		// that is `overflow-hidden`, so what they claim together comes off the
		// bottom and is CLIPPED rather than scrolled. Measured with both panels
		// expanded beside an approval: approve/deny 120px below the fold at
		// 390x844 with the card's own scroller already at its end, and at
		// 360x780 the card's top at y=781 in a 780px viewport. Real touch drags
		// recovered none of it. A question outranks a task list (branding §7),
		// so while one is pending the panels are held shut.
		const rosterHeader = screen.getByRole("button", { name: /subagents/ });
		const todosHeader = screen.getByRole("button", { name: /tasks/ });
		expect(rosterHeader.getAttribute("aria-expanded")).toBe("false");
		expect(todosHeader.getAttribute("aria-expanded")).toBe("false");

		// Held shut means held: a tap must not open them either, or the user
		// can still put the decision off screen in one gesture.
		fireEvent.click(rosterHeader);
		fireEvent.click(todosHeader);
		expect(rosterHeader.getAttribute("aria-expanded")).toBe("false");
		expect(todosHeader.getAttribute("aria-expanded")).toBe("false");
		expect(screen.queryByRole("button", { name: /surface-audit-01/ })).toBeNull();

		// The counts stay legible, so nothing is hidden — only held.
		expect(screen.getByText("1/22 running")).toBeTruthy();
	});

	it("releases the panels once nothing is pending", () => {
		slot = {
			projection: projection({ subagents: roster(), pending: null }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		const rosterHeader = screen.getByRole("button", { name: /subagents/ });
		fireEvent.click(rosterHeader);
		expect(rosterHeader.getAttribute("aria-expanded")).toBe("true");
	});

	it("surfaces a failing fan-out in the collapsed header", () => {
		// U5: collapsing by default hid the status glyphs, and `1/22 running` is
		// exactly what a healthy session shows — 3 failed agents rendered with
		// no ✗ on screen and the word "failed" absent from the page entirely.
		const withFailures = [
			...roster().slice(0, 19),
			row("surface-audit-19", "failed"),
			row("surface-audit-20", "failed"),
			row("surface-audit-21", "failed"),
		];
		slot = { projection: projection({ subagents: withFailures }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("1/22 running")).toBeTruthy();
		const failed = screen.getByText(/3 failed/);
		expect(failed).toBeTruthy();
		// In the danger colour, or it reads as one more neutral count.
		expect(failed.className).toContain("text-danger");
	});

	it("says nothing about failures when there are none", () => {
		slot = { projection: projection({ subagents: roster() }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText(/failed/)).toBeNull();
	});
});
