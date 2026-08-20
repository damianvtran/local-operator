/**
 * Theme selection: the active palette id, persisted to localStorage and
 * applied as `document.documentElement.dataset.theme` — the attribute every
 * `--lo-*` variable in themes.generated.css keys off. Default is
 * localOperatorDark, the product's native ground.
 *
 * The list of selectable themes is read from the palette modules themselves
 * so adding a palette file makes it selectable without touching this file.
 */
import { dracula } from "./themes/dracula";
import { dune } from "./themes/dune";
import { iceberg } from "./themes/iceberg";
import { localOperatorDark, localOperatorLight } from "./themes/local-operator";
import { monokai } from "./themes/monokai";
import { neon } from "./themes/neon";
import { obsidian } from "./themes/obsidian";
import type { ThemeDefinition } from "./themes/palette-contract";
import { radient } from "./themes/radient";
import { sage } from "./themes/sage";
import { synth } from "./themes/synth";
import { tokyoNight } from "./themes/tokyo-night";

export const DEFAULT_THEME = "localOperatorDark";

export const THEMES: ThemeDefinition[] = [
	localOperatorDark,
	localOperatorLight,
	dracula,
	dune,
	iceberg,
	monokai,
	neon,
	obsidian,
	radient,
	sage,
	synth,
	tokyoNight,
].sort((a, b) => a.name.localeCompare(b.name));

const KEY = "lo-mobile-theme";

export function getTheme(): string {
	const saved = localStorage.getItem(KEY);
	return saved && THEMES.some((t) => t.id === saved) ? saved : DEFAULT_THEME;
}

export function applyTheme(id: string): void {
	document.documentElement.dataset.theme = id;
	localStorage.setItem(KEY, id);
}

/** Called once at boot, before first paint of React content. */
export function initTheme(): void {
	document.documentElement.dataset.theme = getTheme();
}
