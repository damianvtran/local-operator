/**
 * The model pickers' shared filter predicate.
 *
 * ONE implementation for both surfaces — the in-session sheet
 * (`components/model-sheet.tsx`) and the `#/new` picker
 * (`screens/new-session.tsx`). They had the same predicate written out twice,
 * which is how they came to diverge from each other and from what users type.
 */
import type { ModelEntry } from "../types";

/** The text a query is matched against: everything that identifies the row. */
function haystack(m: ModelEntry): string {
	return `${m.selector} ${m.name} ${m.provider}`.toLowerCase();
}

/**
 * Tokens of `query`, lowercased; empty when the query is blank.
 *
 * Whitespace-tolerant BY DESIGN, and this is the finding that forced it: the
 * predicate used to be a single `includes` over `` `${selector} ${name}` ``,
 * and `selector` is `provider/model_id` with a SLASH. So the form a user
 * naturally types on a headerless 996-row list — provider, space, model —
 * matched nothing at all: `xai grok` returned 0 rows where the previous release
 * returned 30, `anthropic claude` 0 where it returned 19. The user is not one
 * keystroke from a match; they have to DELETE a word they had good reason to
 * type, with nothing on screen saying so.
 */
export function filterTokens(query: string): string[] {
	return query.trim().toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * Whether `m` matches every token of the query (case-insensitive substring).
 *
 * EVERY token must appear, so each word the user adds narrows the list — which
 * is what a user typing `anthropic claude` means by it. Order is irrelevant, so
 * `claude anthropic` finds the same rows.
 *
 * Scoring stays OUT of this deliberately. The desktop picker's SUBSEQUENCE tier
 * (which resolves `anthopus` and `sonnet4`) is the tie-breaking half of a
 * ranking, and re-implementing it here would be a second, drifting copy of
 * `model/ranking.py` — the server already ordered the array, and callers use
 * `Array.filter`, which is order-preserving, so the best route still leads every
 * query. Token matching fixes a class of query that returned NOTHING; fuzzy
 * matching would only re-rank queries that already return something.
 */
export function matchesModel(m: ModelEntry, tokens: string[]): boolean {
	if (tokens.length === 0) return true;
	const text = haystack(m);
	return tokens.every((t) => text.includes(t));
}

/** `models` filtered by `query`, in the server's order. */
export function filterModels(
	models: ModelEntry[],
	query: string,
): ModelEntry[] {
	const tokens = filterTokens(query);
	if (tokens.length === 0) return models;
	return models.filter((m) => matchesModel(m, tokens));
}
