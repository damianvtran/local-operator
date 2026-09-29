// The phone's spend and context spellings, pinned to the Python sources.
//
// WHY THIS FILE EXISTS
// --------------------
// `sessionSpend` ports `FrontendSessionState.cumulative_cost` /
// `cumulative_cost_knowledge` plus the band's `_spend_text` rules; the money
// ladder ports `tui/costs.py::format_usd` / `micro_from_usd`; `contextSpelling`
// / `contextSemanticColor` / `formatContextTokens` / `formatWindow` port
// `tui/widgets/status_line.py` and `session/frontend_state.py`. The desktop
// strip already keeps one copy of these rules for its own surface
// (`session-cost.ts` / `session-context.ts`); this bundle keeps one for the
// phone, and the SAME generated artifact pins both directions:
// `spend-context.parity.json`, written by
// `scripts/generate_spend_context_parity.py`, asserted here and against the
// live Python by `tests/unit/mobile/test_tui_bridge.py`. A change to either
// side fails a suite in ITS OWN tree rather than diverging with both green.
//
// The fixture's spend cases construct a real `FrontendSessionState(**inputs)`
// on the Python side and record its property answers, so the ledger
// combinations (owner ledger AND compatibility rows present, the double-count
// case) and the `$—` / zero / floor spellings are the Python's, not this
// suite's opinion.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { pyFixed } from "./fixed-point";
import {
	contextEstimateMarker,
	contextReading,
	contextSemanticColor,
	contextSpelling,
	formatContextTokens,
	formatUsd,
	formatWindow,
	microFromUsd,
	sessionSpend,
	type SessionSpendInput,
} from "./spend-context";

interface ParityFixture {
	pyFixed: [number, number, string][];
	money: [number, string][];
	micro: [number, number | null][];
	spend: [
		unknown[],
		[number, number] | null,
		[number | null, string, boolean, string],
	][];
	context: [number, number, string, string][];
}

/** The spend cases carry their five inputs POSITIONALLY, in the projection's
 * field order — remapping here is what fails when a field is dropped or
 * reordered rather than letting the wrong value feed the wrong name. */
const SPEND_KEYS = [
	"cumulative_parent_cost",
	"child_costs",
	"subagent_cost",
	"subagent_cost_knowledge",
	"cost_knowledge",
] as const;

function spendInput(values: unknown[]): SessionSpendInput {
	const input: Record<string, unknown> = {};
	for (const [index, key] of SPEND_KEYS.entries()) input[key] = values[index];
	return input as unknown as SessionSpendInput;
}

/** The usage pair is positional too: `[input_tokens, output_tokens]`. */
function usageOf(values: [number, number] | null): Record<string, number> | null {
	return values ? { input_tokens: values[0], output_tokens: values[1] } : null;
}

/** The generated fixture, read from disk rather than imported: the vitest node
 * environment resolves it the same way whatever the bundler's JSON settings
 * do, and a moved or deleted fixture fails here loudly instead of silently
 * shrinking the table. */
const parity: ParityFixture = JSON.parse(
	readFileSync(join(__dirname, "spend-context.parity.json"), "utf8"),
) as ParityFixture;

describe("pyFixed", () => {
	it("matches Python's format() on the shared parity fixture", () => {
		// Real ties, near-ties and both signs — the exact-binary rounding rule
		// `toFixed` gets wrong.
		expect(parity.pyFixed.length).toBeGreaterThan(70);
		for (const [value, digits, expected] of parity.pyFixed) {
			expect(pyFixed(value, digits), `${value} @ ${digits}dp`).toBe(expected);
		}
	});

	it("rounds exact ties half-to-even, as CPython does", () => {
		// 0.125 is exactly representable; Python: format(0.125, ".2f") == "0.12".
		expect(pyFixed(0.125, 2)).toBe("0.12");
		expect(pyFixed(0.375, 2)).toBe("0.38");
		// 2.675 is NOT a tie (its double sits just below) — half-even on the
		// exact value still rounds down, which is the case `Intl` gets wrong.
		expect(pyFixed(2.675, 2)).toBe("2.67");
		expect(pyFixed(-2.675, 2)).toBe("-2.67");
	});

	it("renders a non-finite value as nothing rather than inventing one", () => {
		expect(pyFixed(Number.NaN, 2)).toBe("");
		expect(pyFixed(Number.POSITIVE_INFINITY, 2)).toBe("");
	});
});

describe("the money ladder", () => {
	it("matches format_usd on the shared parity fixture", () => {
		expect(parity.money.length).toBeGreaterThan(15);
		for (const [micro, expected] of parity.money) {
			expect(formatUsd(micro), `${micro}µ$`).toBe(expected);
		}
	});

	it("refuses to round a nonzero sub-half-micro figure to $0.0000", () => {
		expect(formatUsd(1)).toBe("<$0.0001");
		expect(formatUsd(49)).toBe("<$0.0001");
		expect(formatUsd(50)).toBe("$0.0001");
		// A genuine zero keeps the honest spelling; the band drops the segment.
		expect(formatUsd(0)).toBe("$0.0000");
	});

	it("matches micro_from_usd's float→micro step, half-to-even included", () => {
		expect(parity.micro.length).toBeGreaterThanOrEqual(10);
		for (const [cost, expected] of parity.micro) {
			expect(microFromUsd(cost), String(cost)).toBe(expected);
		}
	});

	it("returns null for a value the ladder cannot take", () => {
		expect(microFromUsd(Number.NaN)).toBeNull();
		expect(microFromUsd(Number.POSITIVE_INFINITY)).toBeNull();
		expect(microFromUsd("1.25" as unknown as number)).toBeNull();
		expect(microFromUsd(null)).toBeNull();
	});
});

describe("sessionSpend", () => {
	it("matches cumulative_cost / _spend_text on the shared parity fixture", () => {
		// ≥35 cases: every ledger combination, the zero policy, the floor mark
		// and the billed-but-unpriceable `$—`.
		expect(parity.spend.length).toBeGreaterThanOrEqual(35);
		for (const [values, usage, [total, knowledge, isFloor, text]] of parity.spend) {
			const spend = sessionSpend(spendInput(values), usageOf(usage));
			expect(
				[spend.total, spend.knowledge, spend.isFloor, spend.text],
				JSON.stringify(values),
			).toEqual([total, knowledge, isFloor, text]);
		}
	});

	it("degrades to the defaults for a pre-upgrade wire (no block at all)", () => {
		const spend = sessionSpend({} as SessionSpendInput, undefined);
		expect(spend).toEqual({ total: null, knowledge: "unknown", isFloor: false, text: "" });
	});

	it("never sums the owner ledger and the compatibility rows together", () => {
		// The double-count rule, stated on its own: rows alone would say 99.5,
		// the owner ledger says 2.0, and the ledger is the authority.
		const spend = sessionSpend(
			{
				cumulative_parent_cost: 1,
				child_costs: { "job-a": 99.5 },
				subagent_cost: 2,
				subagent_cost_knowledge: "exact",
				cost_knowledge: "exact",
			},
			null,
		);
		expect(spend.total).toBe(3);
	});
});

describe("the context reading", () => {
	it("matches context_spelling and context_semantic_color on the fixture", () => {
		expect(parity.context.length).toBeGreaterThan(40);
		for (const [tokens, window, spelling, rung] of parity.context) {
			expect(contextSpelling(tokens, window), `${tokens}/${window}`).toBe(spelling);
			expect(contextSemanticColor(tokens, window), `${tokens}/${window}`).toBe(rung);
		}
	});

	it("keeps both ladders strictly greater-than", () => {
		// On a boundary the calmer colour holds, so a hovering number cannot
		// flicker between two hues.
		expect(contextSemanticColor(110_000, 200_000)).toBe("signal");
		expect(contextSemanticColor(110_001, 200_000)).toBe("label");
		expect(contextSemanticColor(200_000, 0)).toBe("signal");
		expect(contextSemanticColor(200_001, 0)).toBe("label");
		expect(contextSemanticColor(500_000, 0)).toBe("label");
		expect(contextSemanticColor(500_001, 0)).toBe("danger");
	});

	it("spells a window-unknown reading against an explicit unknown", () => {
		expect(contextSpelling(12_400, 0)).toBe("12.4k/\u2014");
		expect(formatContextTokens(12_400)).toBe("12.4k");
		expect(formatWindow(1_000_000)).toBe("1M");
		expect(formatWindow(200_000)).toBe("200k");
	});

	it("returns a READING, with the estimate flag rather than a rounded number", () => {
		const estimate = contextReading({
			context_tokens: 12_400,
			context_window: 200_000,
			context_is_estimate: true,
		});
		expect(estimate.status).toBe("estimate");
		expect(estimate.spelling).toBe("6.2%/200k");
		// The marker is a WORD, composed beside the spelling — never baked into
		// it (design round 1, D2; see `contextEstimateMarker` for both
		// precedents).
		expect(contextEstimateMarker(estimate)).toBe("estimate");

		const measured = contextReading({
			context_tokens: 12_400,
			context_window: 200_000,
			context_is_estimate: false,
		});
		expect(measured.status).toBe("measured");
		expect(measured.spelling).toBe("6.2%/200k");
		expect(contextEstimateMarker(measured)).toBe("");

		const windowUnknown = contextReading({ context_tokens: 12_400, context_window: 0 });
		expect(windowUnknown.status).toBe("window-unknown");
		expect(windowUnknown.spelling).toBe("12.4k/\u2014");
		expect(contextEstimateMarker(windowUnknown)).toBe("");

		const none = contextReading({});
		expect(none.status).toBe("no-reading");
		expect(none.spelling).toBe("");
		expect(contextEstimateMarker(none)).toBe("");
	});
});
