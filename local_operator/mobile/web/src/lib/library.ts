/**
 * Library view helpers — the small decisions every Library surface must make
 * the same way.
 *
 * NOTHING HERE DECIDES CONTENT. Ordering, filtering, the effective name and the
 * origin group all come from the relay's own projection (`library/service.py`),
 * and this file must never recompute them: a second derivation is exactly the
 * drift the service exists to prevent. What lives here is presentation only —
 * a byte count, the glyph a kind gets, the prompt an asset composer sends, and
 * the tag-name grammar the frozen metadata contract enforces.
 */
import type { LibraryAsset } from "../types";

/** The tag grammar the relay's store enforces (`_normalise_tag_name`). Kept in
    sync by `TAG_NAME_RE` here AND a test against the relay's refusal, because a
    client that validates more loosely shows a create button that always fails. */
export const TAG_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,23}$/;

/** A human byte count with one decimal only while that decimal says something.
    `null` (a digest row before it is materialised) reads as an empty string, not
    "0 B" — the latter is a claim about a file whose size is unknown. */
export function formatBytes(bytes: number | null): string {
	if (bytes === null || !Number.isFinite(bytes) || bytes < 0) return "";
	if (bytes < 1024) return `${bytes} B`;
	const units = ["KB", "MB", "GB", "TB"];
	let value = bytes / 1024;
	let index = 0;
	while (value >= 1024 && index < units.length - 1) {
		value /= 1024;
		index += 1;
	}
	const rounded = value >= 10 ? Math.round(value) : Math.round(value * 10) / 10;
	return `${rounded} ${units[index]}`;
}

/** One glyph per kind. Text, not an icon font: the bundle ships no icon asset,
    and a glyph reads the same in every theme. */
export function kindGlyph(kind: string): string {
	switch (kind) {
		case "image":
			return "▧";
		case "audio":
			return "♪";
		case "video":
			return "▶";
		case "pdf":
			return "PDF";
		case "pptx":
			return "PPT";
		case "docx":
			return "DOC";
		case "markdown":
			return "MD";
		case "html":
			return "HTML";
		case "spreadsheet":
			return "▦";
		case "code":
			return "</>";
		case "text":
			return "TXT";
		default:
			return "•";
	}
}

/** The kind label a reader would say out loud. */
export function kindLabel(kind: string): string {
	switch (kind) {
		case "markdown":
			return "markdown";
		case "html":
			return "html";
		case "spreadsheet":
			return "spreadsheet";
		case "pptx":
			return "presentation";
		case "docx":
			return "document";
		default:
			return kind;
	}
}

/** The origin badge's word. Raw origins exist too; this is the three-group view
    the design froze, and the service already chose the group. */
export function originLabel(group: string): string {
	switch (group) {
		case "generated":
			return "generated";
		case "created":
			return "created";
		default:
			return "mentioned";
	}
}

/** The first source conversation, with the row's own fallback when a catalog row
    has no occurrence (the service does not produce one today, but a projection
    change must not put "undefined" on a card). */
export function sourceTitle(asset: LibraryAsset): string {
	const occurrence = asset.occurrences[0];
	return occurrence?.session_title || occurrence?.session_id || "";
}

/**
 * The first message an asset composer sends.
 *
 * It carries the asset's IDENTITY (id, name, display path, kind, source) so the
 * session can act without the phone re-sending bytes, and the reader's own words
 * last — the instruction, not a summary of it. The display path is deliberate:
 * absolute paths never reach this surface, and on the owner's machine `~`-form
 * is what the agent can open.
 */
export function assetPrompt(asset: LibraryAsset, question: string): string {
	const lines = [
		"I'm asking about an asset in my Library.",
		`- name: ${asset.name}`,
		`- id: ${asset.id}`,
		`- kind: ${kindLabel(asset.kind)}${asset.size_bytes ? ` (${formatBytes(asset.size_bytes)})` : ""}`,
	];
	if (asset.ref.display_path) lines.push(`- path: ${asset.ref.display_path}`);
	const source = sourceTitle(asset);
	if (source) lines.push(`- source conversation: ${source}`);
	if (asset.tags.length) {
		lines.push(`- tags: ${asset.tags.map((tag) => tag.name).join(", ")}`);
	}
	return `${lines.join("\n")}\n\n${question.trim()}`;
}
