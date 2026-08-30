/**
 * Theme selection: the active palette id, persisted to localStorage and
 * applied as `document.documentElement.dataset.theme` — the attribute every
 * `--lo-*` variable in themes.generated.css keys off. Default is
 * localOperatorDark, the product's native ground.
 *
 * The list of selectable themes is read from the palette modules themselves
 * so adding a palette file makes it selectable without touching this file.
 */
import { alucard } from "./themes/alucard";
import { ayuDark } from "./themes/ayu";
import { ayuLight } from "./themes/ayu-light";
import { ayuMirage } from "./themes/ayu-mirage";
import { catppuccinFrappe } from "./themes/catppuccin-frappe";
import { catppuccinMacchiato } from "./themes/catppuccin-macchiato";
import { dracula } from "./themes/dracula";
import { dune } from "./themes/dune";
import { duskfox } from "./themes/duskfox";
import { everforest } from "./themes/everforest";
import { everforestLight } from "./themes/everforest-light";
import { gruvboxLight } from "./themes/gruvbox-light";
import { iceberg } from "./themes/iceberg";
import { kanagawaLotus } from "./themes/kanagawa-lotus";
import { kanagawaWave } from "./themes/kanagawa-wave";
import { localOperatorDark, localOperatorLight } from "./themes/local-operator";
import { monokai } from "./themes/monokai";
import { neon } from "./themes/neon";
import { nightfox } from "./themes/nightfox";
import { obsidian } from "./themes/obsidian";
import { oneLight } from "./themes/one-light";
import type { ThemeDefinition } from "./themes/palette-contract";
import { palenight } from "./themes/palenight";
import { radient } from "./themes/radient";
import { rosePine, rosePineDawn, rosePineMoon } from "./themes/rose-pine";
import { sage } from "./themes/sage";
import { synth } from "./themes/synth";
import { tokyoNight } from "./themes/tokyo-night";
import { tokyoNightDay } from "./themes/tokyo-night-day";

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
	rosePine,
	rosePineDawn,
	rosePineMoon,
	sage,
	synth,
	tokyoNight,
	alucard,
	ayuDark,
	ayuLight,
	ayuMirage,
	catppuccinFrappe,
	catppuccinMacchiato,
	duskfox,
	everforest,
	everforestLight,
	gruvboxLight,
	kanagawaLotus,
	kanagawaWave,
	nightfox,
	oneLight,
	palenight,
	tokyoNightDay,
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
