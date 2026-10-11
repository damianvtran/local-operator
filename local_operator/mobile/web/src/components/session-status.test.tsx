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
	it("reads context before spend, each cell keeping its own edge", () => {
		const { container } = render(<SessionStatus projection={projection()} />);
		// Reading order is the one both references use — the TUI band's right
		// group and the desktop strip read context before spend (design round
		// 1, D3).
		const order = Array.from(container.querySelectorAll("span[data-testid]")).map((el) =>
			el.getAttribute("data-testid"),
		);
		expect(order).toEqual(["session-status-context", "session-status-spend"]);
		const context = container.querySelector('[data-testid="session-status-context"]');
		const spend = container.querySelector('[data-testid="session-status-spend"]');
		expect(context?.textContent).toBe("6.2%/200k");
		expect(spend?.textContent).toBe("$1.25");
		// `ml-auto` sits on the SECOND cell, so each keeps a fixed edge and a
		// lone reading neither reflows nor floats: context hugs the left,
		// spend the right — the single-cell states mirror the pair.
		expect(spend?.className).toContain("ml-auto");
		expect(context?.className).not.toContain("ml-auto");

		const contextOnly = render(
			<SessionStatus projection={projection({ cumulative_parent_cost: null })} />,
		);
		const loneContext = contextOnly.container.querySelector(
			'[data-testid="session-status-context"]',
		);
		expect(loneContext?.textContent).toBe("6.2%/200k");
		expect(loneContext?.className).not.toContain("ml-auto");

		const spendOnly = render(<SessionStatus projection={projection({ context_tokens: null })} />);
		const loneSpend = spendOnly.container.querySelector('[data-testid="session-status-spend"]');
		expect(loneSpend?.textContent).toBe("$1.25");
		expect(loneSpend?.className).toContain("ml-auto");
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

	it("appends the estimate marker as a dim WORD beside the reading", () => {
		// The desktop's convention, per the design round's D2 ruling: a word,
		// not a `~` glyph or a tinted number — a hint the user would have had
		// to be taught. It renders in its own dim segment so it never inherits
		// the reading's rung colour, and the spelling itself is untouched.
		const { container } = render(
			<SessionStatus projection={projection({ context_is_estimate: true })} />,
		);
		const context = container.querySelector('[data-testid="session-status-context"]');
		expect(context?.textContent).toBe("6.2%/200kestimate");
		const marker = context?.querySelector("span");
		expect(marker?.textContent).toBe("estimate");
		expect(marker?.className).toContain("text-ink-dim");

		// A measured figure carries no marker at all.
		const measured = render(
			<SessionStatus projection={projection({ context_is_estimate: false })} />,
		);
		const measuredCell = measured.container.querySelector(
			'[data-testid="session-status-context"]',
		);
		expect(measuredCell?.textContent).toBe("6.2%/200k");
		expect(measuredCell?.querySelector("span")).toBeNull();
	});

	it("carries the TUI's colour-vision weight on the warm rungs only", () => {
		// The band bolds the context reading on every rung but the base one
		// (`status_line.py`: `bold=semantic != CONTEXT_COLOR_BASE`) because hue
		// alone cannot carry the step under colour-vision deficiency; the
		// phone mirrors that with 600 on label/danger, regular on signal.
		const base = render(<SessionStatus projection={projection()} />);
		const baseCell = base.container.querySelector('[data-testid="session-status-context"]');
		expect(baseCell?.className).toContain("text-ink-muted");
		expect(baseCell?.className).not.toContain("font-semibold");

		const label = render(
			<SessionStatus projection={projection({ context_tokens: 110_001 })} />,
		);
		const labelCell = label.container.querySelector('[data-testid="session-status-context"]');
		expect(labelCell?.className).toContain("text-accent");
		expect(labelCell?.className).toContain("font-semibold");

		const danger = render(
			<SessionStatus projection={projection({ context_tokens: 160_001 })} />,
		);
		const dangerCell = danger.container.querySelector('[data-testid="session-status-context"]');
		expect(dangerCell?.className).toContain("text-danger");
		expect(dangerCell?.className).toContain("font-semibold");
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

describe("SessionStatus — the published spend object (PR-2)", () => {
	it("renders the published total rather than a local re-sum, mark included", () => {
		// Legacy fields deliberately disagree with the published object: the
		// published figure must win, because re-summing the parts is exactly
		// how the surfaces came to disagree (design §5.1). `partial` keeps its
		// `≥`, which is the backend's own knowledge rung.
		const { container } = render(
			<SessionStatus
				projection={projection({
					cumulative_parent_cost: 1.25,
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 961_000,
						knowledge: "partial",
						rows: [],
					},
				})}
			/>,
		);
		expect(container.querySelector('[data-testid="session-status-spend"]')?.textContent).toBe(
			"\u2265$0.961",
		);
	});

	it("marks a pending voice charge as the floor it is", () => {
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 100_000,
						knowledge: "exact",
						rows: [],
						pending: { voice: "pending_owner_append", calls: 2 },
					},
				})}
			/>,
		);
		expect(container.querySelector('[data-testid="session-status-spend"]')?.textContent).toBe(
			"\u2265$0.100",
		);
	});

	it("lists channel rows with their own amounts and the sub-equiv marker", () => {
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 1_005_000,
						knowledge: "partial",
						rows: [
							{
								channel: "inference",
								label: "anthropic/claude-sonnet-5-5",
								amount_micro: 900_000,
								basis: ["billed"],
							},
							{ channel: "image", amount_micro: 100_000, basis: ["estimated"] },
							{ channel: "tts", amount_micro: null, basis: [] },
							{
								channel: "inference",
								label: "openai/gpt-5.4",
								amount_micro: 5_000,
								basis: ["subscription_api_equivalent"],
							},
						],
					},
				})}
			/>,
		);
		const chips = Array.from(
			container.querySelectorAll('[data-testid="session-status-channel"]'),
		).map((el) => el.textContent);
		expect(chips).toEqual([
			"claude-sonnet-5-5 $0.900",
			"image $0.100",
			"tts $\u2014",
			"gpt-5.4 $0.0050 sub-equiv",
		]);
		// The sub-equiv qualifier is its own dim span — the money must never be
		// read as billed dollars.
		const marker = container.querySelector('[data-testid="session-status-channels"] .text-ink-dim');
		expect(marker?.textContent).toBe(" sub-equiv");
	});

	it("states an untracked session in plain words, never an implied $0", () => {
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: false,
						total_micro: 0,
						knowledge: "partial",
						rows: [],
					},
				})}
			/>,
		);
		expect(
			container
				.querySelector('[data-testid="session-status-channels-untracked"]')
				?.textContent?.trim(),
		).toBe("channels not tracked");
	});

	it("keeps the untracked disclosure silent when nothing is missing (M-4)", () => {
		// A marker-less session whose own figure is `exact` has no evidence of
		// missing spend, so it reads like any other — the desktop gate's rule.
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: false,
						total_micro: 210_000,
						knowledge: "exact",
						rows: [],
					},
				})}
			/>,
		);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		expect(container.querySelector('[data-testid="session-status-spend"]')?.textContent).toBe(
			"$0.210",
		);
	});

	it("draws RECOVERED rows on a marker-less session, notice beside them", () => {
		// The dropped-rows bug: a backfilled session has no marker but real
		// recovered rows, and they must render — with the disclosure that the
		// recovered set may be incomplete.
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: false,
						total_micro: 53_000,
						knowledge: "partial",
						rows: [{ channel: "image", amount_micro: 53_000, basis: ["estimated"] }],
					},
				})}
			/>,
		);
		expect(
			container.querySelector('[data-testid="session-status-channel"]')?.textContent?.trim(),
		).toBe("image $0.053");
		expect(
			container
				.querySelector('[data-testid="session-status-channels-untracked"]')
				?.textContent?.trim(),
		).toBe("channels not tracked");
	});

	it("names the remainder with the TUI's separator: `· +N more` (D1)", () => {
		const many = Array.from({ length: 6 }, (_, index) => ({
			channel: "image",
			provider: `p${index}`,
			amount_micro: 1_000,
			basis: ["estimated"],
		}));
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: { version: 1, tracked: true, total_micro: 6_000, rows: many },
				})}
			/>,
		);
		const channels = container.querySelector('[data-testid="session-status-channels"]');
		expect(channels?.textContent).toContain("· +2 more");
	});

	it("shows no breakdown line at all without a published object", () => {
		const { container } = render(<SessionStatus projection={projection()} />);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		expect(container.querySelector('[data-testid="session-status-spend"]')?.textContent).toBe(
			"$1.25",
		);
	});
});
