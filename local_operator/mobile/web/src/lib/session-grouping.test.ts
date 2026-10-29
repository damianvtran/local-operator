// @vitest-environment happy-dom
//
// The pure grouping matrix for S2b: catalogue order, unknown/deleted primary,
// no primary, empty groups, stable daemon order and pinned exclusion. Kept on
// the pure function so the screen only has to prove integration.
import { describe, expect, it } from "vitest";
import {
	groupByPrimaryTag,
	hasGroupedRows,
	UNTAGGED_GROUP_KEY,
	UNTAGGED_GROUP_LABEL,
} from "./session-grouping";
import type { SessionSummary, TagCatalogueEntry } from "../types";

function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s",
		section: "active",
		conversation_name: "Session",
		cwd: "",
		model_label: "",
		streaming: false,
		needs_attention: false,
		unseen: false,
		pending_kind: "",
		subagents_running: 0,
		subagents_queued: 0,
		todos_open: 0,
		mtime: 0,
		pinned: false,
		...over,
	};
}

function row(id: string, primary: string | null, over: Partial<SessionSummary> = {}) {
	return summary({ session_id: id, conversation_name: id, primary, ...over });
}

const CATALOGUE: TagCatalogueEntry[] = [
	{ id: "work", name: "Work", order: 2, conversations: 2 },
	{ id: "research", name: "Research", order: 0, conversations: 1 },
	{ id: "home", name: "Home", order: 1, conversations: 1 },
];

function ids(groups: ReturnType<typeof groupByPrimaryTag>): string[][] {
	return groups.map((group) => group.rows.map((entry) => entry.session_id));
}

describe("groupByPrimaryTag", () => {
	it("orders groups by catalogue order, not payload order", () => {
		const groups = groupByPrimaryTag(
			[row("a", "work"), row("b", "research"), row("c", "home")],
			CATALOGUE,
		);
		expect(groups.map((group) => group.key)).toEqual(["research", "home", "work"]);
	});

	it("keeps catalogue payload order for equal order values", () => {
		const catalogue: TagCatalogueEntry[] = [
			{ id: "second", name: "Second", order: 1, conversations: 1 },
			{ id: "first", name: "First", order: 1, conversations: 1 },
		];
		const groups = groupByPrimaryTag([row("a", "first"), row("b", "second")], catalogue);
		expect(groups.map((group) => group.key)).toEqual(["second", "first"]);
	});

	it("keeps daemon row order inside each group", () => {
		const groups = groupByPrimaryTag(
			[row("z", "work"), row("a", "work"), row("m", "work")],
			CATALOGUE,
		);
		expect(ids(groups)).toEqual([["z", "a", "m"]]);
	});

	it("buckets an unknown or deleted primary as Untagged, last", () => {
		const groups = groupByPrimaryTag(
			[row("gone", "deleted"), row("live", "work"), row("none", null)],
			CATALOGUE,
		);
		expect(groups.map((group) => group.key)).toEqual(["work", UNTAGGED_GROUP_KEY]);
		expect(groups.at(-1)?.label).toBe(UNTAGGED_GROUP_LABEL);
		expect(ids(groups).at(-1)).toEqual(["gone", "none"]);
	});

	it("omits groups with no rows", () => {
		const groups = groupByPrimaryTag([row("a", "home")], CATALOGUE);
		expect(groups.map((group) => group.key)).toEqual(["home"]);
	});

	it("returns no Untagged bucket when every row resolves", () => {
		const groups = groupByPrimaryTag([row("a", "work"), row("b", "home")], CATALOGUE);
		expect(groups.some((group) => group.key === UNTAGGED_GROUP_KEY)).toBe(false);
	});

	it("returns one Untagged bucket when no row has tags", () => {
		const groups = groupByPrimaryTag([row("a", null), row("b", null)], []);
		expect(groups).toHaveLength(1);
		expect(groups[0].key).toBe(UNTAGGED_GROUP_KEY);
		expect(ids(groups)).toEqual([["a", "b"]]);
	});

	it("excludes pinned rows — ★ Pinned is its own top section", () => {
		const groups = groupByPrimaryTag(
			[row("pin", "work", { pinned: true }), row("a", "work")],
			CATALOGUE,
		);
		expect(ids(groups)).toEqual([["a"]]);
	});
});

describe("hasGroupedRows", () => {
	it("is false when only Untagged would render — the flat fallback", () => {
		const groups = groupByPrimaryTag([row("a", null)], CATALOGUE);
		expect(hasGroupedRows(groups)).toBe(false);
	});

	it("is true when at least one row resolves to a catalogue primary", () => {
		const groups = groupByPrimaryTag([row("a", "work"), row("b", null)], CATALOGUE);
		expect(hasGroupedRows(groups)).toBe(true);
	});
});
