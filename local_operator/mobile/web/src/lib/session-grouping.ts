/**
 * View-only conversation grouping by PRIMARY TAG — the pure half of S2b.
 *
 * Grouping is a LOCAL VIEW PREFERENCE, not a store: nothing here writes, and
 * the relay's assignment stays the only source of truth about what a row is
 * tagged with. The input is the daemon's own row order (already ranked on the
 * shared catalog key), so this module only partitions it; it never re-sorts
 * rows. That is the same contract the ungrouped screen has with the daemon,
 * and the reason a tag edit (SSE row update) regroups live without a refetch.
 *
 * ORDER OF GROUPS is the registry's, not the rows': catalogue `order` first,
 * ties left in the catalogue payload's own order. `order` is the field the
 * assignment sheet's `catalogueRows` also sorts on, so the phone's grouped home
 * and its tag sheet list categories in the same sequence.
 *
 * UNTAGGED IS ONE BUCKET, LAST. A row whose primary is absent, unknown to this
 * catalogue (a stale read), or deleted from it belongs there. Rendering one
 * bucket rather than dropping such rows is deliberate: a stale catalogue must
 * not make conversations disappear from the home screen.
 *
 * PINNED ROWS ARE NOT GROUPED. ★ Pinned is its own top section (the sidebar's
 * rule), so a pinned row passed in is skipped here even though the screen
 * already hands this function the unpinned split. Keeping the rule in one pure
 * place is what makes the matrix testable.
 */
import type { SessionSummary, TagCatalogueEntry } from "../types";

export interface TagGroup {
	/** Render key: the catalogue id, or the untagged sentinel. */
	key: string;
	/** Heading text — the registry's own name, or `Untagged`. */
	label: string;
	/** The catalogue entry this group is keyed by, or null for Untagged. */
	tag: TagCatalogueEntry | null;
	rows: SessionSummary[];
}

export const UNTAGGED_GROUP_KEY = "untagged";

/** The heading for rows with no resolvable primary. Advisory copy, like S2a:
    not an i18n catalogue key, so this surface stays inside the existing
    allowance. */
export const UNTAGGED_GROUP_LABEL = "Untagged";

/** Partition `rows` into catalogue-ordered groups plus a final Untagged bucket.
 *
 *  Only NON-EMPTY groups are returned: a category with no conversations
 *  contributes no heading, the same rule the Active/Previous sections follow.
 *  Pinned rows are ignored — they are rendered in ★ Pinned above the groups.
 */
export function groupByPrimaryTag(
	rows: readonly SessionSummary[],
	catalogue: readonly TagCatalogueEntry[],
): TagGroup[] {
	const ordered = catalogue
		.map((entry, index) => ({ entry, index }))
		.sort((a, b) => a.entry.order - b.entry.order || a.index - b.index);
	const known = new Set(ordered.map(({ entry }) => entry.id));
	const buckets = new Map<string, SessionSummary[]>();
	const untagged: SessionSummary[] = [];
	for (const row of rows) {
		if (row.pinned) continue;
		const primary = row.primary ?? null;
		if (primary && known.has(primary)) {
			const bucket = buckets.get(primary);
			if (bucket) bucket.push(row);
			else buckets.set(primary, [row]);
		} else {
			untagged.push(row);
		}
	}
	const groups: TagGroup[] = [];
	for (const { entry } of ordered) {
		const bucket = buckets.get(entry.id);
		if (bucket && bucket.length > 0) {
			groups.push({ key: entry.id, label: entry.name, tag: entry, rows: bucket });
		}
	}
	if (untagged.length > 0) {
		groups.push({
			key: UNTAGGED_GROUP_KEY,
			label: UNTAGGED_GROUP_LABEL,
			tag: null,
			rows: untagged,
		});
	}
	return groups;
}

/** Whether the grouping should REPLACE Active/Previous.
 *
 *  FLAT FALLBACK: a list where no unpinned row resolves to a catalogue primary
 *  is not a grouped list — it would be one `Untagged` heading over every row,
 *  which is the normal flat view with extra chrome. Rendering the ordinary
 *  Active/Previous sections there keeps the screen's own information (the
 *  daemon's active/previous partition) instead. */
export function hasGroupedRows(groups: readonly TagGroup[]): boolean {
	return groups.some((group) => group.tag !== null);
}
