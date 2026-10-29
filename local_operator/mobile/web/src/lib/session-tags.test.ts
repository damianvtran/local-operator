/**
 * Conversation tags — the pure rules, tested without a DOM.
 *
 * Every case here is a rule the SHEET and the ROW LINE both depend on: a label
 * budget that preserves the store's order, a desired-state toggle whose clear
 * and primary fallback match what the relay applies, and the merge that keeps a
 * deleted-but-assigned category visible long enough to remove it.
 */
import { describe, expect, it } from "vitest";

import {
	assignmentFrom,
	atTagCap,
	catalogueRows,
	choosePrimary,
	clearedAssignment,
	compressedTags,
	MAX_ASSIGNED_TAGS,
	TAG_LINE_MAX,
	toggleAssignment,
} from "./session-tags";
import type { TagCatalogueEntry, TagRef } from "../types";

const tag = (id: string, name = id.toUpperCase()): TagRef => ({ id, name });

const entry = (
	id: string,
	order: number,
	name = id.toUpperCase(),
): TagCatalogueEntry => ({ id, name, order, conversations: 0 });

describe("compressedTags", () => {
	it("draws nothing for an untagged conversation, so no line exists", () => {
		expect(compressedTags([])).toEqual({ shown: [], overflow: 0 });
	});

	it("draws one label whole", () => {
		const { shown, overflow } = compressedTags([tag("a")]);
		expect(shown.map((t) => t.id)).toEqual(["a"]);
		expect(overflow).toBe(0);
	});

	it("draws two labels whole", () => {
		const { shown, overflow } = compressedTags([tag("a"), tag("b")]);
		expect(shown).toHaveLength(2);
		expect(overflow).toBe(0);
	});

	it("compresses past two to a count, preserving assignment order", () => {
		const { shown, overflow } = compressedTags([tag("a"), tag("b"), tag("c"), tag("d")]);
		expect(shown.map((t) => t.id)).toEqual(["a", "b"]);
		expect(overflow).toBe(2);
	});

	it("carries a long name through untouched: the row truncates, the data does not", () => {
		const long = "A category name long enough to run past any phone row";
		const { shown } = compressedTags([tag("a", long)]);
		expect(shown[0].name).toBe(long);
	});

	it("uses the two-label budget the row line is built for", () => {
		expect(TAG_LINE_MAX).toBe(2);
	});
});

describe("toggleAssignment", () => {
	it("appends a new category, in assignment order, without touching the primary", () => {
		expect(toggleAssignment({ tags: ["a"], primary: "a" }, "b")).toEqual({
			tags: ["a", "b"],
			primary: "a",
		});
	});

	it("leaves a null primary null when a second tag joins", () => {
		expect(toggleAssignment({ tags: ["a"], primary: null }, "b")).toEqual({
			tags: ["a", "b"],
			primary: null,
		});
	});

	it("removes a non-primary category and keeps the primary", () => {
		expect(toggleAssignment({ tags: ["a", "b"], primary: "b" }, "a")).toEqual({
			tags: ["b"],
			primary: "b",
		});
	});

	it("falls back to the first surviving assignment when the primary is removed", () => {
		expect(toggleAssignment({ tags: ["a", "b", "c"], primary: "b" }, "b")).toEqual({
			tags: ["a", "c"],
			primary: "a",
		});
	});

	it("falls back to no primary when the last tag is removed", () => {
		expect(toggleAssignment({ tags: ["a"], primary: "a" }, "a")).toEqual({
			tags: [],
			primary: null,
		});
	});

	it("refuses to add past the cap", () => {
		const full = {
			tags: Array.from({ length: MAX_ASSIGNED_TAGS }, (_, i) => `t${i}`),
			primary: "t0",
		};
		expect(atTagCap(full)).toBe(true);
		expect(toggleAssignment(full, "one-more")).toEqual(full);
	});

	it("still permits removal at the cap", () => {
		const full = {
			tags: Array.from({ length: MAX_ASSIGNED_TAGS }, (_, i) => `t${i}`),
			primary: "t0",
		};
		expect(toggleAssignment(full, "t3").tags).toHaveLength(MAX_ASSIGNED_TAGS - 1);
	});

	it("clears to the empty desired state the route takes", () => {
		expect(clearedAssignment()).toEqual({ tags: [], primary: null });
	});
});

describe("choosePrimary", () => {
	it("marks a selected category primary", () => {
		expect(choosePrimary({ tags: ["a", "b"], primary: null }, "b")).toEqual({
			tags: ["a", "b"],
			primary: "b",
		});
	});

	it("ignores an unselected category: the relay refuses primary outside tags", () => {
		const current = { tags: ["a"], primary: "a" };
		expect(choosePrimary(current, "b")).toEqual(current);
	});
});

describe("catalogueRows", () => {
	it("renders the registry in its own order, not the payload's array order", () => {
		const rows = catalogueRows(
			[entry("b", 1), entry("a", 0)],
			[],
			clearedAssignment(),
		);
		expect(rows.map((row) => row.id)).toEqual(["a", "b"]);
	});

	it("keeps payload order for a tie in `order` (stable)", () => {
		const rows = catalogueRows(
			[entry("first", 0), entry("second", 0)],
			[],
			clearedAssignment(),
		);
		expect(rows.map((row) => row.id)).toEqual(["first", "second"]);
	});

	it("reflects the desired state's selection and primary", () => {
		const rows = catalogueRows(
			[entry("a", 0), entry("b", 1)],
			[],
			{ tags: ["b"], primary: "b" },
		);
		expect(rows.map((row) => [row.id, row.selected, row.primary])).toEqual([
			["a", false, false],
			["b", true, true],
		]);
	});

	it("merges an assigned category the registry no longer carries, still selectable", () => {
		const rows = catalogueRows(
			[entry("a", 0)],
			[tag("a"), tag("gone", "Deleted category")],
			{ tags: ["a", "gone"], primary: "gone" },
		);
		expect(rows.map((row) => row.id)).toEqual(["a", "gone"]);
		const missing = rows[1];
		expect(missing.missing).toBe(true);
		expect(missing.name).toBe("Deleted category");
		expect(missing.selected).toBe(true);
		expect(missing.primary).toBe(true);
	});

	it("does not duplicate an assigned category the registry does carry", () => {
		const rows = catalogueRows([entry("a", 0)], [tag("a")], {
			tags: ["a"],
			primary: null,
		});
		expect(rows).toHaveLength(1);
		expect(rows[0].missing).toBe(false);
	});
});

describe("assignmentFrom", () => {
	it("copies ids in assignment order and the primary as given", () => {
		expect(assignmentFrom([tag("a"), tag("b")], "b")).toEqual({
			tags: ["a", "b"],
			primary: "b",
		});
	});

	it("reads an untagged row as the empty assignment", () => {
		expect(assignmentFrom([], null)).toEqual({ tags: [], primary: null });
	});
});
