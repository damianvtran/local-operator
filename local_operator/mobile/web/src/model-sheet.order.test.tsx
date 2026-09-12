// @vitest-environment happy-dom
//
// Regression: the phone's model sheet only ever showed Radient and OpenRouter.
// `GET /api/models` served 962 rows in REGISTRY order (radient 445, then the
// direct providers, then openrouter 445), and the sheet then grouped them by
// provider — which re-sorts, so even a correctly ranked payload would have been
// thrown away on arrival. A user hunting for `anthropic/claude-opus-5` scrolled
// ~445 rows, roughly 45 phone screens, past a router's catalogue to reach it.
//
// Both halves of the fix are asserted HERE, against a real captured slice of the
// daemon's ranked payload rather than a hand-written one: the server's order is
// preserved by the render, and filtering preserves it too. This is also the
// substitute for a pixel screenshot — see the PR: the browser extension is not
// connected on this host, so the rendered DOM ORDER is what was actually
// inspected, and no screenshot is claimed.
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModelSheet } from "./components/model-sheet";
import ranked from "./fixtures/models.ranked.json";
import { NewSessionScreen } from "./screens/new-session";
import type { ModelEntry, SessionProjection } from "./types";

const models = ranked as ModelEntry[];

const AGGREGATORS = new Set(["openrouter", "radient", "radient-key"]);

vi.mock("./api", () => ({
	getModels: vi.fn(async () => ({ models })),
	sendCommand: vi.fn(async () => ({ ok: true })),
	getDirectories: vi.fn(async () => ({ home: "/Users/x", recent: [], tmp: "" })),
	startSession: vi.fn(),
}));

const projection = {
	model_selector: "anthropic/claude-opus-5",
} as unknown as SessionProjection;

/** Every row button's selector, in the order the DOM actually lays them out.
 *
 * Keyed off the `key`-bearing buttons rather than text, because the row renders
 * the NAME (which repeats across routes) — the point of the test is which route
 * comes first, so the identity has to be the selector. */
function renderedOrder(): string[] {
	const buttons = [...document.querySelectorAll("button")];
	const order: string[] = [];
	for (const button of buttons) {
		const text = button.textContent ?? "";
		const match = models.find(
			(m) =>
				text.includes(m.name) &&
				text.includes(m.provider) &&
				!order.includes(m.selector),
		);
		if (match) order.push(match.selector);
	}
	return order;
}

/** Index bounds of the direct/aggregated split.
 *
 * Written with an explicit loop rather than `findLastIndex`, which needs an
 * es2023 lib this project does not target — widening the target for one test
 * assertion is not a trade worth making. An absent side returns a bound that
 * keeps `last < first` true, so a list of only direct rows still passes. */
function lastDirectIndex(order: string[]): number {
	let last = -1;
	order.forEach((s, i) => {
		if (!AGGREGATORS.has(s.split("/")[0])) last = i;
	});
	return last;
}

function firstAggregatedIndex(order: string[]): number {
	const found = order.findIndex((s: string) =>
		AGGREGATORS.has(s.split("/")[0]),
	);
	return found === -1 ? order.length : found;
}

describe("the phone's model sheet", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		document.body.innerHTML = "";
	});

	it("renders one flat list in the server's order, not regrouped by provider", async () => {
		render(
			<ModelSheet
				open
				onClose={() => {}}
				pid="1"
				projection={projection}
			/>,
		);
		await screen.findByText(models[0].name);

		const order = renderedOrder();
		expect(order).toEqual(models.map((m) => m.selector));
	});

	it("puts every direct row ahead of every aggregated one", async () => {
		render(
			<ModelSheet
				open
				onClose={() => {}}
				pid="1"
				projection={projection}
			/>,
		);
		await screen.findByText(models[0].name);

		const order = renderedOrder();
		expect(lastDirectIndex(order)).toBeLessThan(firstAggregatedIndex(order));
	});

	it("shows the provider on the row now that no header carries it", async () => {
		render(
			<ModelSheet
				open
				onClose={() => {}}
				pid="1"
				projection={projection}
			/>,
		);
		await screen.findByText(models[0].name);

		// Two routes to claude-opus-5 exist in the fixture; without the
		// per-row provider they are indistinguishable, and the route is what
		// differs in price and quota.
		expect(screen.getAllByText("anthropic").length).toBeGreaterThan(0);
		expect(screen.getAllByText("openrouter").length).toBeGreaterThan(0);
	});
});

describe("filtering the sheet", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		document.body.innerHTML = "";
	});

	// The query a user actually types for a model both a direct provider and a
	// router can serve. Which DIRECT provider leads is not the claim (several
	// serve glm); that a router never does is.
	it.each(["glm", "opus"])(
		"leads %s with a direct route, never the router",
		async (query) => {
		render(
			<ModelSheet
				open
				onClose={() => {}}
				pid="1"
				projection={projection}
			/>,
		);
		await screen.findByText(models[0].name);

		fireEvent.change(screen.getByPlaceholderText("filter models"), {
			target: { value: query },
		});

		const order = renderedOrder();
		expect(order.length).toBeGreaterThan(0);
		expect(AGGREGATORS.has(order[0].split("/")[0])).toBe(false);
		// Every direct hit still precedes every aggregated one after filtering.
		expect(lastDirectIndex(order)).toBeLessThan(firstAggregatedIndex(order));
		// And the filter did not re-sort: what survives is a SUBSEQUENCE of the
		// server's array, which is what `Array.filter` guarantees and what a
		// client-side ranker would quietly break.
		const full = models.map((m) => m.selector);
		expect(order).toEqual(full.filter((s) => order.includes(s)));
		},
	);
});

describe("the new-session model picker", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		document.body.innerHTML = "";
	});

	it("also lists the server's order rather than leading with the router", async () => {
		render(<NewSessionScreen />);

		// The picker lives behind the "default" model button.
		fireEvent.click(await screen.findByText("default"));
		await screen.findByPlaceholderText("filter models");

		const order = renderedOrder();
		expect(order).toEqual(models.map((m) => m.selector));
		expect(AGGREGATORS.has(order[0].split("/")[0])).toBe(false);
	});
});
