// @vitest-environment happy-dom
//
// The image-generation card's RENDERED half: the transcript branch, every
// state's body, the cancel gating, and the (unwired) restart/steer slots.
//
// WHY THIS FILE EXISTS
// --------------------
// The adapter's mapping is pinned in `lib/image-gen.test.ts`; what a mapping
// test cannot see is whether the RENDERER does something honest with each
// view state — a fold that emits a state no view reads is a change that does
// not reach the user (the same reason `tool-row.queued.test.tsx` exists).
// Four properties matter most here, each one a thing a person would notice if
// it slipped:
//
//   * the card mounts for `generate_image` and ONLY for it — every other
//     tool keeps its plain one-line row;
//   * `running` carries the tile, an indeterminate bar by default and the
//     determinate branch only on a carried fraction;
//   * Cancel rides `{op:"abort"}` (the composer's own turn-interrupt path)
//     and the card holds "cancelling…" until the SETTLE — never painting
//     "cancelled" off the press;
//   * the restart/steer slots exist but are wired by no caller, so the app's
//     own render shows neither control.
import {
	cleanup,
	fireEvent,
	render,
	screen,
	waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { sendCommand } from "./api";
import { ImageGenCard } from "./components/image-gen-card";
import { Transcript } from "./components/transcript";
import type { TranscriptEntry, TranscriptEntryDetails } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn((_pid: string, entryId: string, index: number) => `img:${entryId}:${index}`),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
}));

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

function entry(over: Partial<TranscriptEntry>): TranscriptEntry {
	return {
		id: "e1",
		kind: "tool",
		text: "",
		tool_call_id: "call_img",
		tool_name: "generate_image",
		tool_state: "running",
		summary: "a red panda in a spacesuit",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {},
		final: false,
		...over,
	};
}

function cardOf(container: HTMLElement): HTMLElement {
	const card = container.querySelector<HTMLElement>('[data-testid="image-gen-card"]');
	if (!card) throw new Error("image-gen card did not render");
	return card;
}

/** Live-detail fields are NOT frozen yet, and their only home in src/ is the
    adapter (see `lib/image-gen.ts`); the type deliberately does not declare
    them, so the tests build the raw bag here. */
function liveDetails(fields: Record<string, unknown>): TranscriptEntryDetails {
	return fields as TranscriptEntryDetails;
}

describe("the transcript branch", () => {
	it("mounts the card for generate_image and leaves other tools as plain rows", () => {
		const { container } = render(<Transcript pid="9" entries={[entry({})]} />);
		expect(cardOf(container)).toBeTruthy();
		cleanup();

		const { container: plain } = render(
			<Transcript
				pid="9"
				entries={[entry({ tool_name: "read", tool_state: "done" })]}
			/>,
		);
		expect(plain.querySelector('[data-testid="image-gen-card"]')).toBeNull();
		/* The row itself still renders — this branch removes nothing from
		   non-image tools. */
		expect(screen.getByText("read")).toBeTruthy();
	});
});

describe("queued", () => {
	it("states queued, and the queue position only when the feed carries one", () => {
		const { container } = render(
			<Transcript pid="9" entries={[entry({ tool_state: "queued" })]} />,
		);
		expect(cardOf(container).textContent).toContain("queued");
		cleanup();

		const { container: withPosition } = render(
			<Transcript
				pid="9"
				entries={[
					entry({ tool_state: "queued", details: liveDetails({ queue_position: 2 }) }),
				]}
			/>,
		);
		expect(cardOf(withPosition).textContent).toContain("queued · position 2");
	});
});

describe("running", () => {
	it("carries the tile, the indeterminate bar and the cancel control", () => {
		const { container } = render(<Transcript pid="9" entries={[entry({})]} />);
		expect(cardOf(container).querySelector(".lo-gen-tile")).toBeTruthy();
		const bar = screen.getByRole("progressbar");
		/* No fraction carried means NO fraction announced — the bar is
		   indeterminate, not a 0%. */
		expect(bar.getAttribute("aria-valuenow")).toBeNull();
		expect(screen.getByRole("button", { name: "cancel" })).toBeTruthy();
	});

	it("paints the determinate branch at the same number it announces", () => {
		render(
			<Transcript
				pid="9"
				entries={[entry({ details: liveDetails({ progress_fraction: 0.42 }) })]}
			/>,
		);
		const bar = screen.getByRole("progressbar");
		expect(bar.getAttribute("aria-valuenow")).toBe("42");
		const fill = bar.firstElementChild as HTMLElement;
		expect(fill.style.width).toBe("42%");
	});

	it("shows the log tail only when present, and only the tail", () => {
		const { container } = render(
			<Transcript
				pid="9"
				entries={[
					entry({
					details: liveDetails({
						log_lines: [
							{ message: "step 1", timestamp: "2026-10-09T00:00:00Z" },
							{ message: "step 2", timestamp: "2026-10-09T00:00:01Z" },
							{ message: "step 3", timestamp: "2026-10-09T00:00:02Z" },
							{ message: "step 4", timestamp: "2026-10-09T00:00:03Z" },
						],
					}),
					}),
				]}
			/>,
		);
		const text = cardOf(container).textContent ?? "";
		expect(text).toContain("step 2");
		expect(text).toContain("step 4");
		expect(text).not.toContain("step 1");
	});
});

describe("settled states", () => {
	it("done: the artifact renders through the existing attachment path", () => {
		const { container } = render(
			<Transcript
				pid="9"
				entries={[
					entry({
						tool_state: "done",
						images: [{ index: 0, mime_type: "image/png" }],
					}),
				]}
			/>,
		);
		const img = container.querySelector("img");
		expect(img?.getAttribute("src")).toBe("img:e1:0");
		expect(cardOf(container).textContent).toContain("image ready");
	});

	it("done with no artifact still states the state, without a broken frame", () => {
		const { container } = render(
			<Transcript pid="9" entries={[entry({ tool_state: "done" })]} />,
		);
		expect(screen.getByText("image ready")).toBeTruthy();
		expect(container.querySelector("img")).toBeNull();
	});

	it("failed: the platform sentence verbatim, with no lead sentence of the card's own", () => {
		render(
			<Transcript
				pid="9"
				entries={[
					entry({
						tool_state: "failed",
						error: "This generation failed before producing output.",
					}),
				]}
			/>,
		);
		expect(
			screen.getByText("This generation failed before producing output."),
		).toBeTruthy();
		/* The generic lead sentence is gone on purpose: the frozen provider
		   contract says `error` IS the sanctioned text, and substituting this
		   surface's wording for it is exactly the defect. */
		expect(screen.queryByText("image generation failed")).toBeNull();
	});

	it("the cancel conflict reads as 'already finished', never as an error", () => {
		const { container } = render(
			<Transcript
				pid="9"
				entries={[
					entry({
						tool_state: "failed",
						details: liveDetails({ error_type: "media_already_completed" }),
						error: "This generation had already finished.",
					}),
				]}
			/>,
		);
		expect(screen.getByText("already finished")).toBeTruthy();
		/* Nothing failed. The conflict's own text must not be painted as a
		   failure message, and no failure sentence may appear from this card. */
		expect(
			screen.queryByText("This generation had already finished."),
		).toBeNull();
		expect(screen.queryByText("failed")).toBeNull();
		/* THE ROW AGREES WITH THE BODY: no failure palette (✗ / danger
		   wash) even though the wire settle was failure-shaped, and the
		   neutral settle glyph instead. */
		expect(cardOf(container).querySelector(".bg-danger-wash")).toBeNull();
		/* EN dash (U+2013), the same glyph `tool-row.tsx`'s GLYPH table uses
		   for the neutral settle — asserted by codepoint so a visually similar
		   dash cannot pass this by accident. */
		expect(cardOf(container).textContent).toContain("\u2013");
		expect(cardOf(container).textContent).not.toContain("✗");
	});

	it("cancelled is a plain state; the restart/steer slots are unwired in the app", () => {
		render(
			<Transcript
				pid="9"
				entries={[entry({ tool_state: "interrupted" })]}
			/>,
		);
		expect(screen.getByText("cancelled")).toBeTruthy();
		/* The app passes neither slot, so neither control renders. The
		   named surface op does not exist yet; a visible dead control
		   would be worse than none. */
		expect(screen.queryByRole("button", { name: "restart" })).toBeNull();
		expect(screen.queryByRole("button", { name: "steer" })).toBeNull();
	});

	it("a cancel that landed (error-shaped, stage cancelled) is not a failure", () => {
		/* The canonical cancel settle: the tool result is error-shaped while
		   its details name the cancel — the card must state `cancelled`, not
		   paint a deliberate stop in the failure ink. */
		render(
			<Transcript
				pid="9"
				entries={[
					entry({ tool_state: "failed", details: liveDetails({ stage: "cancelled" }) }),
				]}
			/>,
		);
		expect(screen.getByText("cancelled")).toBeTruthy();
	});

	it("the restart and steer slots call through when a caller wires them", () => {
		const onRestart = vi.fn();
		const onSteer = vi.fn();
		render(
			<ImageGenCard
				entry={entry({ tool_state: "interrupted" })}
				pid="9"
				onRestart={onRestart}
				onSteer={onSteer}
			/>,
		);
		fireEvent.click(screen.getByRole("button", { name: "restart" }));
		fireEvent.click(screen.getByRole("button", { name: "steer" }));
		expect(onRestart).toHaveBeenCalledTimes(1);
		expect(onSteer).toHaveBeenCalledTimes(1);
	});
});

describe("cancel gating", () => {
	it("cancel rides the turn-interrupt path and holds 'cancelling' until the settle", async () => {
		const { rerender } = render(<Transcript pid="9" entries={[entry({})]} />);
		fireEvent.click(screen.getByRole("button", { name: "cancel" }));
		expect(vi.mocked(sendCommand)).toHaveBeenCalledWith("9", { op: "abort" });
		await screen.findByText("cancelling…");
		/* The hold carries the harness hook the capture rig's geometry dump
		   reads (review round 1, F3). */
		expect(screen.getByTestId("image-gen-hold")).toBeTruthy();
		/* Not pressable twice, and never painted "cancelled" off the press. */
		expect(screen.queryByRole("button", { name: "cancel" })).toBeNull();

		/* The confirmation lands: the row settles as interrupted. */
		rerender(<Transcript pid="9" entries={[entry({ tool_state: "interrupted" })]} />);
		expect(screen.getByText("cancelled")).toBeTruthy();
		expect(screen.queryByText("cancelling…")).toBeNull();

		/* And the settle RETIRED the press (review round 1, F2): if the wire
		   ever re-reported this call as live, the card must render `running`
		   with a fresh control, never a "cancelling…" resurrected by a dead
		   press. No wire path does that today; this pins the invariant. */
		rerender(<Transcript pid="9" entries={[entry({})]} />);
		expect(screen.queryByText("cancelling…")).toBeNull();
		expect(screen.getByRole("button", { name: "cancel" })).toBeTruthy();
	});

	it("a refused abort puts the control back rather than holding a false 'cancelling'", async () => {
		vi.mocked(sendCommand).mockRejectedValueOnce(
			new Error("session not connected"),
		);
		render(<Transcript pid="9" entries={[entry({})]} />);
		fireEvent.click(screen.getByRole("button", { name: "cancel" }));
		await waitFor(() =>
			expect(screen.queryByText("cancelling…")).toBeNull(),
		);
		expect(screen.getByRole("button", { name: "cancel" })).toBeTruthy();
	});
});

describe("the canonical stage word", () => {
	it("the wire's own cancelling holds the card before any press", () => {
		/* The press-driven hold is captured by the cancel-gating tests above;
		   this one arrives only from the feed (`stage: "cancelling"`) and
		   must render the same hold — no press, no cancel control to press
		   twice. */
		render(
			<Transcript
				pid="9"
				entries={[
					entry({
						tool_state: "running",
						details: liveDetails({ stage: "cancelling" }),
					}),
				]}
			/>,
		);
		expect(screen.getByText("cancelling…")).toBeTruthy();
		expect(screen.getByTestId("image-gen-hold")).toBeTruthy();
		expect(screen.queryByRole("button", { name: "cancel" })).toBeNull();
	});
});
