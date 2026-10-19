// @vitest-environment happy-dom
//
// The MINIMIZED ask affordance (design §5.0, R7): absent at zero asks, head +
// count when there are some, and one tap to the answer surface.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AskDock } from "./components/ask-dock";
import type { PendingAsk } from "./types";

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a1",
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

afterEach(() => cleanup());

describe("AskDock", () => {
	it("is absent at zero asks — never a zero badge", () => {
		const { container } = render(<AskDock rows={[]} onOpen={() => {}} />);
		expect(container.firstChild).toBeNull();
	});

	it("says nothing while the runtime cannot publish asks at all", () => {
		const { container } = render(<AskDock rows={undefined} onOpen={() => {}} />);
		expect(container.firstChild).toBeNull();
	});

	it("counts the QUESTIONS waiting and names the head ask's question", () => {
		render(
			<AskDock
				rows={[
					ask({ ask_id: "new", created_at: 900, questions: [
						{ id: "q1", question: "second?", options: [], multi: false, secret: false, persist: false },
					] }),
					ask({ ask_id: "old", created_at: 100 }),
				]}
				onOpen={() => {}}
			/>,
		);
		expect(screen.getByText("2 questions waiting")).toBeTruthy();
		/* The HEAD is the oldest open ask, so the preview must be its question. */
		expect(screen.getByText(/ship the fix\?/)).toBeTruthy();
	});

	it("opens the answer surface on tap", () => {
		const open = vi.fn();
		render(<AskDock rows={[ask()]} onOpen={open} />);
		fireEvent.click(screen.getByTestId("ask-dock"));
		expect(open).toHaveBeenCalledTimes(1);
	});

	it("still NAMES a timed-out ask when it is the only one left", () => {
		/* U7 (UX round 1): with only a timed-out ask left the chip counted it but
		   previewed nothing. The count is now the lapsed one (design round 1, D1,
		   pinned below); the preview must still name the question. */
		render(<AskDock rows={[ask({ status: "timed_out" })]} onOpen={() => {}} />);
		expect(screen.getByText(/ship the fix\?/)).toBeTruthy();
	});

	it("does not call a queue of only lapsed asks waiting: a quiet entry that still opens", () => {
		/* Design round 1, D1: a stopped conversation whose only ask has lapsed
		   showed an accent chip reading "1 question waiting" — the most prominent
		   mark on the screen calling a lapsed ask a live decision. The entry stays
		   (it is the way to Dismiss), in the dim register, saying what it holds. */
		const open = vi.fn();
		render(<AskDock rows={[ask({ status: "timed_out" })]} onOpen={open} />);
		expect(screen.getByText("1 question timed out")).toBeTruthy();
		expect(screen.queryByText(/waiting/)).toBeNull();
		const dock = screen.getByTestId("ask-dock");
		expect(dock.className).not.toMatch(/border-accent|bg-accent-wash/);
		fireEvent.click(dock);
		expect(open).toHaveBeenCalledTimes(1);
	});

	it("counts only the questions still offered as waiting when lapsed ones sit beside them", () => {
		render(
			<AskDock
				rows={[ask({ ask_id: "o" }), ask({ ask_id: "t", status: "timed_out" })]}
				onOpen={() => {}}
			/>,
		);
		expect(screen.getByText("1 question waiting")).toBeTruthy();
	});

	it("is a chip, not a full-bleed strip", () => {
		/* D1 (design round 1): the approved shape is inset + rounded + accent
		   bordered, matching `pending-card.tsx`; a wash spanning the whole width
		   read as a banner, which §5.0 names as the thing it must not be. */
		render(<AskDock rows={[ask()]} onOpen={() => {}} />);
		const dock = screen.getByTestId("ask-dock");
		expect(dock.className).toMatch(/mx-2/);
		expect(dock.className).toMatch(/rounded-md/);
		expect(dock.className).toMatch(/border-accent/);
	});
});
