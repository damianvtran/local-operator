/**
 * Conversation tags — the PURE half of the phone's tags surface.
 *
 * One implementation for the row line (session list and past sessions, through
 * `components/session-tag-line.tsx`) and the assignment sheet, so the two can
 * never disagree about what a row claims. Nothing here reads or writes storage:
 * the relay's assignment is the only store (see `SessionTagsSheet`), and a
 * local cache would be a second derivation with a drift window.
 *
 * DESIRED STATE, NOT DELTAS. Every mutation returns the complete
 * `{tags, primary}` the route takes, because that is the contract
 * (`PUT /api/sessions/{id}/tags`): a retried request lands on the same
 * assignment, and the clear is just the empty one. The order is the store's
 * record and is preserved verbatim — the row line renders it, and the primary
 * fallback depends on it.
 */
import type { TagCatalogueEntry, TagRef } from "../types";

/** The relay's cap (`local_operator/session/session_tags.py`). Enforced here so
    the UI disables rows BEFORE a write the relay would refuse with a 422; the
    relay remains the authority and its sentence is what a refusal renders. */
export const MAX_ASSIGNED_TAGS = 8;

/** Labels the list row shows before it compresses the rest to `+N`. Two fits a
    phone-width row beside the title without pushing the row's own metadata out;
    it is the same budget the brief set for both lists. */
export const TAG_LINE_MAX = 2;

/** A complete desired-state assignment: ids in assignment order, and the one
    marked primary (nullable — a tagged conversation may have no primary). */
export interface TagAssignment {
	tags: string[];
	primary: string | null;
}

export function assignmentFrom(
	tags: readonly TagRef[],
	primary: string | null,
): TagAssignment {
	return { tags: tags.map((tag) => tag.id), primary };
}

export function clearedAssignment(): TagAssignment {
	return { tags: [], primary: null };
}

export function atTagCap(assignment: TagAssignment): boolean {
	return assignment.tags.length >= MAX_ASSIGNED_TAGS;
}

/** Add or remove one category, returning the complete desired state.
 *
 *  Removing the PRIMARY assignment falls back to the first surviving one, in
 *  assignment order — the same rule the store applies when it prunes a deleted
 *  category. Removing a non-primary tag leaves the primary alone.
 *
 *  Adding past the cap is a NO-OP: the sheet disables those rows, and this is
 *  the second guard so no path can post a ninth assignment the relay refuses. */
export function toggleAssignment(
	current: TagAssignment,
	id: string,
): TagAssignment {
	if (current.tags.includes(id)) {
		const tags = current.tags.filter((tagId) => tagId !== id);
		return {
			tags,
			primary: current.primary === id ? (tags[0] ?? null) : current.primary,
		};
	}
	if (atTagCap(current)) return current;
	return { tags: [...current.tags, id], primary: current.primary };
}

/** Mark one SELECTED category primary. An unselected id cannot be primary (the
    relay refuses that body), so it is a no-op rather than a local invention.

    There is deliberately no "clear the primary" arm: the store resolves a null
    primary to the first assigned tag, so a locally-cleared primary would be a
    state the relay never serves — the assignment surface offers no such verb. */
export function choosePrimary(current: TagAssignment, id: string): TagAssignment {
	if (!current.tags.includes(id)) return current;
	return { tags: current.tags, primary: id };
}

export interface CompressedTags {
	shown: TagRef[];
	/** How many labels the row did NOT draw, for the `+N` mark. */
	overflow: number;
}

/** The row's label budget: the first `max` labels in assignment order plus a
    count of the rest. Order is the store's record, never re-sorted here. */
export function compressedTags(
	tags: readonly TagRef[],
	max: number = TAG_LINE_MAX,
): CompressedTags {
	const shown = tags.slice(0, Math.max(0, max));
	return { shown: [...shown], overflow: Math.max(0, tags.length - shown.length) };
}

/** One row of the assignment sheet. */
export interface CatalogueRow {
	id: string;
	name: string;
	order: number;
	conversations: number;
	selected: boolean;
	primary: boolean;
	/** Assigned to this conversation but ABSENT from the registry snapshot —
	    a category deleted (or renamed away) since the row was painted, or a
	    catalogue read that raced the assignment. It stays selectable so the
	    reader can remove it; the store prunes such ids on read, but the sheet
	    must not hide the id it is about to send. */
	missing: boolean;
}

/** The sheet's rows: the registry in ITS order, then any assigned id the
    registry no longer carries, in assignment order. `desired` is the draft (or
    the last authoritative frame), so the checkboxes follow the same state the
    next write would send. */
export function catalogueRows(
	catalogue: readonly TagCatalogueEntry[],
	assigned: readonly TagRef[],
	desired: TagAssignment,
): CatalogueRow[] {
	const known = new Set(catalogue.map((entry) => entry.id));
	const rows: CatalogueRow[] = catalogue
		.map((entry, index) => ({ entry, index }))
		.sort((a, b) => a.entry.order - b.entry.order || a.index - b.index)
		.map(({ entry }) => ({
			id: entry.id,
			name: entry.name,
			order: entry.order,
			conversations: entry.conversations,
			selected: desired.tags.includes(entry.id),
			primary: desired.primary === entry.id,
			missing: false,
		}));
	for (const tag of assigned) {
		if (known.has(tag.id)) continue;
		rows.push({
			id: tag.id,
			name: tag.name,
			order: Number.MAX_SAFE_INTEGER,
			conversations: 0,
			selected: desired.tags.includes(tag.id),
			primary: desired.primary === tag.id,
			missing: true,
		});
	}
	return rows;
}
