// @vitest-environment happy-dom
//
// The session status glance: what it shows, and — as important — when it shows
// NOTHING.
//
// The two refusals this component inherits from the Python are the ones that
// matter here: a zero spend renders no segment (a confident `$0.0000` over
// billed tokens is the lie `_spend_text` exists to prevent), and a session
// with no reading yet renders no context segment. When neither has anything to
// say the row is absent entirely, so the header's layout does not carry a
// permanent empty strip. What jsdom can prove is this structure and the exact
// spellings; the pixel evidence (row height, the 390-px fit) is the before/
// after browser frames recorded on the PR.
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { SessionProjection } from "../types";
import { SessionStatus } from "./session-status";

afterEach(() => {
	cleanup();
});

/** Only the fields this row reads; the rest of the projection is irrelevant to
 * it, so the cast goes through `unknown` rather than fabricating a session. */
function projection(overrides: Record<string, unknown> = {}): SessionProjection {
	return {
		cumulative_parent_cost: 1.25,
		child_costs: {},
		subagent_cost: null,
		subagent_cost_knowledge: null,
		cost_knowledge: "exact",
		context_tokens: 12_400,
		context_window: 200_000,
		context_is_estimate: false,
		usage: {},
		...overrides,
	} as unknown as SessionProjection;
}

describe("SessionStatus", () => {
	it("shows the spend and the context reading side by side", () => {
		const { container } = render(<SessionStatus projection={projection()} />);
		expect(container.textContent).toContain("$1.25");
		expect(container.textContent).toContain("6.2%/200k");
	});

	it("marks a lower-bound spend with the band's \u2265", () => {
		const { container } = render(
			<SessionStatus projection={projection({ cost_knowledge: "floor" })} />,
		);
		expect(container.textContent).toContain("\u2265$1.25");
	});

	it("renders $— for billed tokens at an unpriceable total", () => {
		const { container } = render(
			<SessionStatus
				projection={projection({
					cumulative_parent_cost: null,
					cost_knowledge: "unknown",
					usage: { input_tokens: 9_000, output_tokens: 100 },
				})}
			/>,
		);
		expect(container.textContent).toContain("$\u2014");
	});

	it("flags an estimated context figure and leaves a measured one unmarked", () => {
		const { container } = render(
			<SessionStatus projection={projection({ context_is_estimate: true })} />,
		);
		const [spend, context] = Array.from(container.querySelectorAll("span"));
		expect(spend.textContent).toBe("$1.25");
		expect(context.textContent).toBe("~6.2%/200k");
	});

	it("renders no row at all when there is nothing to state", () => {
		const fresh = render(
			<SessionStatus
				projection={projection({ cumulative_parent_cost: null, context_tokens: null })}
			/>,
		);
		expect(fresh.container.innerHTML).toBe("");

		// A real zero is a stated fact the band still refuses to paint: the
		// segment disappears, it does not become `$0.0000`.
		const zero = render(
			<SessionStatus
				projection={projection({ cumulative_parent_cost: 0, context_tokens: 0 })}
			/>,
		);
		expect(zero.container.innerHTML).toBe("");
	});
});
