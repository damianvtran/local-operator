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
import { cleanup, fireEvent, render } from "@testing-library/react";
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

	it("lists channel rows with their own amounts and the included marker", () => {
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
		const chipsButton = container.querySelector(
			'[data-testid="session-status-spend"]',
		) as HTMLButtonElement;
		/* The chips are a DRILL-DOWN now: collapsed they must not exist, so a
		   test that reads them has to take the same press the reader takes. */
		fireEvent.click(chipsButton);
		const chips = Array.from(
			container.querySelectorAll('[data-testid="session-status-channel"]'),
		).map((el) => el.textContent);
		expect(chips).toEqual([
			"claude-sonnet-5-5 $0.900",
			"image $0.100",
			"tts $\u2014",
			"gpt-5.4 $0.0050 included",
		]);
		// The included qualifier is its own dim span — the money must never be
		// read as billed dollars.
		const marker = container.querySelector('[data-testid="session-status-channels"] .text-ink-dim');
		expect(marker?.textContent).toBe(" included");
	});

	it("states an untracked session behind `cost detail`, never an implied $0", () => {
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
		/* Zero total, so there is no spend figure to press; the disclosure is
		   still reachable through a control that names the facts, and it is
		   absent from the row until that control is pressed. */
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		expect(container.querySelector('[data-testid="session-status-spend"]')).toBeNull();
		const detail = container.querySelector('[data-testid="session-status-cost-detail"]');
		expect(detail?.tagName).toBe("BUTTON");
		expect(detail?.textContent).toContain("cost detail");
		fireEvent.click(detail as HTMLElement);
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
		fireEvent.click(container.querySelector('[data-testid="session-status-spend"]') as HTMLElement);
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
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 6_000,
						knowledge: "exact",
						rows: many,
					},
				})}
			/>,
		);
		fireEvent.click(container.querySelector('[data-testid="session-status-spend"]') as HTMLElement);
		const channels = container.querySelector('[data-testid="session-status-channels"]');
		expect(channels?.textContent).toContain("· +2 more");
	});

	it("shows no breakdown line and no control without a published object", () => {
		const { container } = render(<SessionStatus projection={projection()} />);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		const spend = container.querySelector('[data-testid="session-status-spend"]');
		expect(spend?.textContent).toBe("$1.25");
		/* A legacy daemon publishes no object, so there is nothing behind a
		   control: the spend cell stays a plain SPAN, not a dead button. */
		expect(spend?.tagName).toBe("SPAN");
		expect(container.querySelector('[data-testid="session-status-cost-detail"]')).toBeNull();
	});
});

describe("SessionStatus — the spend drill-down (state, not telemetry)", () => {
	/** The operator's shape: one line, with three `image` rows that used to be
	 * indistinguishable chips. */
	function populated() {
		return {
			version: 1,
			tracked: false,
			total_micro: 447_700_000,
			knowledge: "partial",
			rows: [
				{ channel: "image", provider: "openai", model: "gpt-image-1", amount_micro: 80_000, basis: ["estimated"] },
				{ channel: "image", provider: "openai", model: "gpt-image-2", amount_micro: 80_000, basis: ["estimated"] },
				{ channel: "image", provider: "radient", model: "dall-e-3", amount_micro: null, basis: [] },
				{ channel: "search", provider: "brave", amount_micro: 0, basis: [] },
				{ channel: "read", provider: "browser", amount_micro: 120, basis: ["estimated"] },
			],
		};
	}

	it("renders one line by default — the channel element does not exist at all", () => {
		const { container } = render(
			<SessionStatus projection={projection({ spend_channels: populated() })} />,
		);
		expect(container.querySelector('[data-testid="session-status"]')).not.toBeNull();
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		const spend = container.querySelector('[data-testid="session-status-spend"]');
		expect(spend?.tagName).toBe("BUTTON");
		expect(spend?.getAttribute("aria-expanded")).toBe("false");
		/* No dangling reference while the panel does not exist (R1-7). */
		expect(spend?.getAttribute("aria-controls")).toBeNull();
		expect(spend?.className).toContain("min-h-11");
		/* The accessible name carries the figure the eye sees and what the press
		   opens — the chevron is decoration, not the name. */
		expect(spend?.getAttribute("aria-label")).toContain("spend breakdown");
		expect(spend?.getAttribute("aria-label")).toContain("\u2265$447.70");
	});

	it("toggles the inline panel and reports aria-expanded both ways", () => {
		const { container } = render(
			<SessionStatus projection={projection({ spend_channels: populated() })} />,
		);
		const spend = () => container.querySelector('[data-testid="session-status-spend"]') as HTMLElement;
		fireEvent.click(spend());
		expect(spend().getAttribute("aria-expanded")).toBe("true");
		const panel = container.querySelector('[data-testid="session-status-channels"]');
		expect(panel).not.toBeNull();
		/* aria-controls names the panel the button opened. */
		expect(spend().getAttribute("aria-controls")).toBe(panel?.id);
		const text = panel?.textContent ?? "";
		expect(text).toContain("image/gpt-image-1 $0.080");
		expect(text).toContain("image/gpt-image-2 $0.080");
		expect(text).toContain("image/dall-e-3 $\u2014");
		expect(text).toContain("channels not tracked");
		expect(text).toContain("· +1 more");

		fireEvent.click(spend());
		expect(spend().getAttribute("aria-expanded")).toBe("false");
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
	});

	it("ignores layout-driven scroll — tail-follow must not close the panel", () => {
		const { container } = render(
			<SessionStatus projection={projection({ spend_channels: populated() })} />,
		);
		const spend = container.querySelector('[data-testid="session-status-spend"]') as HTMLElement;
		fireEvent.click(spend);
		expect(container.querySelector('[data-testid="session-status-channels"]')).not.toBeNull();

		/* THE REGRESSION R1-1/U3/Q1: at the bottom of a long transcript the
		   panel's own 61px insertion changes the scroll extent, the browser's
		   tail-follow emits `scroll`, and a scroll listener closed the panel
		   before it could be read. A layout-driven scroll is not the reader
		   scrolling away, so it must not dismiss anything. */
		const scroller = container.querySelector('[data-testid="session-status-channels"]');
		fireEvent.scroll(scroller as Element);
		fireEvent.scroll(window);
		expect(container.querySelector('[data-testid="session-status-channels"]')).not.toBeNull();
		expect(spend.getAttribute("aria-expanded")).toBe("true");
		expect(spend.getAttribute("aria-controls")).toBe(
			container.querySelector('[data-testid="session-status-channels"]')?.id,
		);
	});

	it("collapses on a touch drag, the reader's own gesture", () => {
		const { container } = render(
			<SessionStatus projection={projection({ spend_channels: populated() })} />,
		);
		const spend = container.querySelector('[data-testid="session-status-spend"]') as HTMLElement;
		fireEvent.click(spend);
		fireEvent.touchMove(window);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		expect(spend.getAttribute("aria-expanded")).toBe("false");
	});

	it("collapses on a wheel gesture, but not on a page key typed into the composer", () => {
		const { container } = render(
			<SessionStatus projection={projection({ spend_channels: populated() })} />,
		);
		const spend = container.querySelector('[data-testid="session-status-spend"]') as HTMLElement;
		fireEvent.click(spend);
		/* Typing in a text field is not a page-scroll intent even when the key
		   is `End`; the listener excludes text entry before it considers keys. */
		const field = document.createElement("input");
		document.body.appendChild(field);
		fireEvent.keyDown(field, { key: "End" });
		expect(container.querySelector('[data-testid="session-status-channels"]')).not.toBeNull();

		fireEvent.keyDown(window, { key: "PageDown" });
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		fireEvent.click(spend);
		fireEvent.wheel(window);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		field.remove();
	});

	it("qualifies a shared channel label with the row's model, then provider", () => {
		const byProvider = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 160_000,
						knowledge: "exact",
						rows: [
							{ channel: "image", provider: "openai", amount_micro: 80_000 },
							{ channel: "image", provider: "radient", amount_micro: 80_000 },
						],
					},
				})}
			/>,
		);
		fireEvent.click(
			byProvider.container.querySelector(
				'[data-testid="session-status-spend"]',
			) as HTMLElement,
		);
		expect(
			Array.from(
				byProvider.container.querySelectorAll('[data-testid="session-status-channel"]'),
			).map((el) => el.textContent?.trim()),
		).toEqual(["image/openai $0.080", "image/radient $0.080"]);
	});

	it("does not offer a control when there is nothing behind it", () => {
		const { container } = render(
			<SessionStatus
				projection={projection({
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 210_000,
						knowledge: "exact",
						rows: [],
					},
				})}
			/>,
		);
		const spend = container.querySelector('[data-testid="session-status-spend"]');
		expect(spend?.tagName).toBe("SPAN");
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
	});

	it("keeps a zero-total breakdown behind `cost detail`, never on by default", () => {
		/* R1-2/U1: a published breakdown with no figure to print — a zero total
		   here — used to render its panel permanently because there was no spend
		   figure to put the control on. The facts must stay reachable, but the
		   default row is still one line. */
		const { container } = render(
			<SessionStatus
				projection={projection({
					cumulative_parent_cost: null,
					context_tokens: null,
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 0,
						knowledge: "exact",
						rows: [{ channel: "image", amount_micro: 0, basis: ["estimated"] }],
					},
				})}
			/>,
		);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		const detail = container.querySelector('[data-testid="session-status-cost-detail"]');
		expect(detail?.tagName).toBe("BUTTON");
		expect(detail?.className).toContain("min-h-11");
		expect(detail?.getAttribute("aria-expanded")).toBe("false");
		/* With no context either, the row contains only that control. */
		expect(container.querySelector('[data-testid="session-status-context"]')).toBeNull();
		expect(container.querySelector('[data-testid="session-status-spend"]')).toBeNull();
		fireEvent.click(detail as HTMLElement);
		expect(
			container.querySelector('[data-testid="session-status-channel"]')?.textContent?.trim(),
		).toBe("image $0.0000");
	});

	it("keeps an unpriceable breakdown behind `cost detail` too", () => {
		/* The other no-figure shape: unknown knowledge with no billed tokens, so
		   `publishedSpend` has no `$—` to print. Rows still exist and still need
		   a labelled door. */
		const { container } = render(
			<SessionStatus
				projection={projection({
					cumulative_parent_cost: null,
					spend_channels: {
						version: 1,
						tracked: true,
						total_micro: 0,
						knowledge: "unknown",
						rows: [
							{ channel: "tts", provider: "radient", amount_micro: null, basis: [] },
						],
					},
				})}
			/>,
		);
		expect(container.querySelector('[data-testid="session-status-channels"]')).toBeNull();
		const detail = container.querySelector('[data-testid="session-status-cost-detail"]');
		/* `cumulative_parent_cost: null` leaves context as the only other cell; the
		   control sits on its own.
		 */
		expect(detail).not.toBeNull();
		fireEvent.click(detail as HTMLElement);
		expect(
			container.querySelector('[data-testid="session-status-channel"]')?.textContent?.trim(),
		).toBe("tts $—");
	});
});
