import { describe, expect, it } from "vitest";
import { THEMES } from "../theme";
import {
	PELE_COLOUR_ROLES,
	PELE_THEME_STYLE,
	diagramFonts,
	resolvedThemeDeclarations,
} from "./diagram-theme";

const camel = (name: string) => name.replace(/^--lo-/, "").replace(/-([a-z])/g, (_m, c: string) => c.toUpperCase());
const palette = (id: string) => THEMES.find((t) => t.id === id)!.palette as unknown as Record<string, string>;

/* CIE76 colour difference between two #rrggbb values. */
function lab(hex: string): [number, number, number] {
	const v = hex.replace("#", "");
	const lin = [0, 2, 4].map((i) => {
		const c = parseInt(v.slice(i, i + 2), 16) / 255;
		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	const [r, g, b] = lin;
	const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
	const x = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
	const y = f(0.2126 * r + 0.7152 * g + 0.0722 * b);
	const z = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
	return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}
const deltaE = (a: string, b: string) => Math.hypot(...lab(a).map((v, i) => v - lab(b)[i]) as [number, number, number]);

describe("Pele theme mapping", () => {
	it("points every --pele-* variable at a variable, never a literal colour", () => {
		for (const [name, value] of Object.entries(PELE_THEME_STYLE)) {
			expect(name).toMatch(/^--pele-/);
			expect(value).toMatch(/^var\(--[a-z-]+\)$/);
		}
	});

	it("names only --lo-* roles that every one of the shipped palettes defines", () => {
		expect(THEMES.length).toBe(31);
		for (const role of new Set(Object.values(PELE_COLOUR_ROLES))) {
			expect(role).toMatch(/^--lo-/);
			for (const theme of THEMES) {
				expect(palette(theme.id)[camel(role)], `${theme.id} ${role}`).toMatch(/^#[0-9a-fA-F]{6}$/);
			}
		}
	});

	it("keeps the first four series distinct (dE >= 15) in all 31 palettes", () => {
		const first4 = [1, 2, 3, 4].map((n) => camel(PELE_COLOUR_ROLES[`--pele-series-${n}` as keyof typeof PELE_COLOUR_ROLES]));
		let worst = Infinity;
		for (const theme of THEMES) {
			const p = palette(theme.id);
			for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
				const d = deltaE(p[first4[i]], p[first4[j]]);
				worst = Math.min(worst, d);
				expect(d, `${theme.id} ${first4[i]}/${first4[j]}`).toBeGreaterThanOrEqual(15);
			}
		}
		/* Measured when written: 19.9 (palenight, warning/danger). */
		expect(worst).toBeGreaterThan(19);
	});

	it("takes fonts from the app's font variables, with a fallback when unset", () => {
		expect(diagramFonts((n) => (n === "--font-sans" ? "Inter, sans-serif" : ""))).toEqual({
			sans: "Inter, sans-serif",
			mono: "ui-monospace, monospace",
		});
	});

	it("resolves values for a saved SVG without anything that could leave the style attribute", () => {
		const out = resolvedThemeDeclarations((name) =>
			name === "--lo-ink" ? '#fff"; onload="x' : name === "--font-sans" ? 'system-ui, "Segoe UI"' : name === "--lo-surface" ? "#101010" : "",
		);
		const map = Object.fromEntries(out);
		expect(map["--pele-bg"]).toBe("#101010");
		expect(map["--pele-font"]).toBe("system-ui, Segoe UI");
		expect(map["--pele-fg"]).not.toMatch(/["';]/);
		expect(map["--pele-accent"]).toBeUndefined();
	});
});
