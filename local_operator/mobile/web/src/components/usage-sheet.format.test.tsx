// @vitest-environment happy-dom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { getUsage, type PhoneUsageLimit } from "../api";
import { BUNDLED_OWNED } from "../i18n/bundled.gen";
import { createMessageRuntime } from "../i18n/runtime";
import { UsageSheet } from "./usage-sheet";

const context = vi.hoisted(() => ({ backendLocale: "fr", uiLocale: "en", direction: "ltr" as const, revision: 0 }));
// Keep the actual catalogue-selection/ICU runtime; only isolate store ingress.
// This is component/DOM text proof, not an audited French bundle or pixels.
vi.mock("../store", async () => {
	const { createMessageRuntime } = await import("../i18n/runtime");
	const runtime = createMessageRuntime(() => context, () => undefined);
	return { messageRuntime: runtime, useLanguage: () => context.backendLocale, useLocaleSnapshot: () => context };
});
vi.mock("../api", () => ({ getUsage: vi.fn() }));
import { messageRuntime } from "../store";

function select(locale: "en" | "fr" | "es" | "ru" | "vi" | "hi", translated: boolean) {
	context.backendLocale = locale;
	for (const [namespace, messages] of Object.entries(BUNDLED_OWNED)) {
		messageRuntime.registerBundle(namespace, "en", { ...messages });
		if (translated) messageRuntime.registerBundle(namespace, locale, {
			...messages,
			...(namespace === "relay.shell" ? {
				"relay.shell.usage.unit_credits": "crédits",
				"relay.shell.usage.unit_usd": "dollars US",
				"relay.shell.usage.unit_tokens": "jetons",
				"relay.shell.usage.unit_requests": "requêtes",
			} : {}),
		});
	}
	context.uiLocale = messageRuntime.uiLocaleFor(locale);
}
const base: PhoneUsageLimit = {
	amount: { used: null, limit: null, remaining: null, used_fraction: null, unit: "percent" },
	paid_fallback: false, available: null, unlimited: false, supersedes_tiers: false,
	family_cap: false, shared: false, window: "5h", family: "unknown",
	resets_at_ms: null, payable: false, exhausted: false,
};
function load(limits: PhoneUsageLimit[], cached_at?: number) {
	vi.mocked(getUsage).mockResolvedValue({ source: "cached", cached_at,
		reports: [{ provider: "synthetic", unavailable: false, fetched_at: 1, limits }] });
	render(<UsageSheet open onClose={() => undefined} />);
}
afterEach(() => { cleanup(); vi.resetAllMocks(); context.uiLocale = "en"; });

it("rounds .29/.07 through percent style without binary tails", async () => {
	select("en", false);
	load([0.29, 0.07].map((used_fraction) => ({ ...base, amount: { ...base.amount, used_fraction } })));
	expect(await screen.findByText(/29% used/)).toBeTruthy();
	expect(screen.getByText(/7% used/)).toBeTruthy();
	expect(document.body.textContent).not.toContain("28.999999999999996");
	expect(document.body.textContent).not.toContain("7.000000000000001");
});

it("formats FR decimals and unit labels using the served catalogue locale", async () => {
	select("fr", true);
	load(["credits", "usd", "tokens", "requests", "percent", "unknown", "UNTRUSTED_UNIT"].map((unit) =>
		({ ...base, amount: { ...base.amount, remaining: unit === "percent" ? 29 : 123.5, unit } })));
	expect(await screen.findByText(/123,5 crédits left/)).toBeTruthy();
	for (const unit of ["dollars US", "jetons", "requêtes"]) expect(screen.getByText(new RegExp(`123,5 ${unit} left`))).toBeTruthy();
	expect(screen.getByText(/29\s% left/)).toBeTruthy();
	expect(document.body.textContent).not.toMatch(/usd|tokens|requests|UNTRUSTED_UNIT/);
	expect(screen.getAllByText(/5-hour: 123,5 left/)).toHaveLength(2);
});

it("formats used decimals and percent-point amounts without guessing unknown units", async () => {
	select("fr", true);
	load([
		{ ...base, amount: { ...base.amount, used: 12345.6789, unit: "tokens" } },
		{ ...base, amount: { ...base.amount, used: 7, unit: "percent" } },
		{ ...base, amount: { ...base.amount, used: 12.5, unit: "NEW_UNIT" } },
	]);
	expect(await screen.findByText(/12\s345,679 jetons used/)).toBeTruthy();
	expect(screen.getByText(/7\s% used/)).toBeTruthy();
	expect(screen.getByText(/5-hour: 12,5 used/)).toBeTruthy();
	expect(document.body.textContent).not.toContain("NEW_UNIT");
});

it("renders singular English units at one and plural units at other measured counts", async () => {
	select("en", false);
	const counts = [0, 1, 2, 1.5, -1, 1000000];
	load(["credits", "tokens", "requests"].flatMap((unit) => counts.map((remaining) =>
		({ ...base, amount: { ...base.amount, remaining, unit } }))));
	for (const unit of ["credits", "tokens", "requests"]) {
		for (const count of counts) {
			const noun = Math.abs(count) === 1 ? unit.slice(0, -1) : unit;
			expect(await screen.findByText(`Included allowance · 5-hour: ${new Intl.NumberFormat("en").format(count)} ${noun} left`)).toBeTruthy();
		}
	}
});

it.each([
	["en", ["other", "one", "other", "other", "one", "other"]],
	["fr", ["one", "one", "other", "one", "one", "many"]],
	["es", ["other", "one", "other", "other", "one", "many"]],
	["ru", ["many", "one", "few", "other", "one", "many"]],
	["vi", ["other", "other", "other", "other", "other", "other"]],
	["hi", ["one", "one", "other", "other", "one", "other"]],
] as const)("uses actual %s CLDR count rules for all three unit nouns", async (locale, categories) => {
	select(locale, true);
	// Category markers deliberately are NOT native translations: this separates
	// the caller/runtime count proof from the independent native audit obligation.
	const messages = { ...BUNDLED_OWNED["relay.shell"] };
	const valid = new Set(categories);
	for (const unit of ["credits", "tokens", "requests"]) {
		messages[`relay.shell.usage.unit_${unit}`] = `{count, plural, ${[...valid].map((category) => `${category} {${unit}_${category}}`).join(" ")}}`;
	}
	messageRuntime.registerBundle("relay.shell", locale, messages);
	const counts = [0, 1, 2, 1.5, -1, 1000000];
	load(["credits", "tokens", "requests"].flatMap((unit) => counts.map((remaining) =>
		({ ...base, amount: { ...base.amount, remaining, unit } }))));
	for (const unit of ["credits", "tokens", "requests"]) {
		for (const [index, count] of counts.entries()) {
			// Retain NBSP/narrow-NBSP grouping bytes instead of Testing Library's
			// default whitespace collapse, which hides the locale's real output.
			expect(await screen.findByText(`Included allowance · 5-hour: ${new Intl.NumberFormat(locale).format(count)} ${unit}_${categories[index]} left`, { normalizer: (text) => text })).toBeTruthy();
		}
	}
});

it("does not send nonfinite counts to unit plural selection or alter safe branches", async () => {
	select("en", false);
	load([
		{ ...base, amount: { ...base.amount, remaining: NaN, used_fraction: 0.29, unit: "credits" } },
		{ ...base, amount: { ...base.amount, used: Infinity, unit: "tokens" } },
		{ ...base, amount: { ...base.amount, remaining: 1, unit: "UNKNOWN_UNIT" } },
		{ ...base, amount: { ...base.amount, remaining: 1, unit: "percent" } },
		{ ...base, amount: { ...base.amount, remaining: 1, unit: "usd" } },
	]);
	expect(await screen.findByText(/29% used/)).toBeTruthy();
	expect(screen.getByText(/Amount not reported/)).toBeTruthy();
	expect(screen.getByText(/5-hour: 1 left/)).toBeTruthy();
	expect(screen.getByText(/5-hour: 1% left/)).toBeTruthy();
	expect(screen.getByText(/5-hour: 1 USD left/)).toBeTruthy();
	expect(document.body.textContent).not.toMatch(/NaN|Infinity|UNKNOWN_UNIT|relay\.shell\.usage\.unit/);
});

it("formats reset and snapshot dates consistently in the catalogue locale", async () => {
	select("fr", true);
	const stamp = Date.UTC(2026, 0, 2, 15, 4);
	load([{ ...base, resets_at_ms: stamp }], stamp);
	// TZ=UTC is pinned by `test.env` in vite.config.ts, never inferred from the
	// host: without that pin these assertions describe the contributor's clock
	// (2 of 13 fail on a UTC-5 machine, 13/13 on a UTC one).
	expect(await screen.findByText(/Resets at 2 janv\. 2026, 15:04/)).toBeTruthy();
	expect(screen.getByText(/As of 2 janv\. 2026, 15:04\./)).toBeTruthy();
	expect(document.body.textContent).not.toContain("2026-01-02T15:04");
});

it("uses EN formatting when the backend preference has no complete owned bundle", async () => {
	// Own runtime guarantees no French registration leaks from earlier tests.
	const isolated = createMessageRuntime(() => ({ ...context, uiLocale: "en" }), () => undefined);
	for (const [ns, copy] of Object.entries(BUNDLED_OWNED)) isolated.registerBundle(ns, "en", { ...copy });
	expect(isolated.uiLocaleFor("fr")).toBe("en");
	context.backendLocale = "fr"; context.uiLocale = "en";
	load([{ ...base, amount: { ...base.amount, remaining: 123.5, unit: "credits" } }], Date.UTC(2026, 0, 2, 15, 4));
	expect(await screen.findByText(/123.5 credits left/)).toBeTruthy();
	expect(screen.getByText(/As of Jan 2, 2026, 3:04/)).toBeTruthy();
});
