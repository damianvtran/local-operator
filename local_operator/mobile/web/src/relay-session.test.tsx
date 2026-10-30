// @vitest-environment happy-dom
/**
 * relay.session: the extracted session copy, and what the document claims.
 *
 * What these pin, in the order the contract cares about:
 *
 * 1. BEFORE/AFTER ENGLISH IDENTITY. The extraction must not change a single
 *    byte a reader sees: every string the session surfaces paint is asserted to
 *    equal the copy that was hardcoded before this slice. The ellipses are the
 *    U+2026 the catalogue carries, not three dots.
 * 2. WHOLE-SURFACE FALLBACK. A missing, partial or invalid owned bundle for the
 *    resolved locale selects the WHOLE English bundle — never a per-key mix of
 *    two languages under one page language.
 * 3. LOCALE + REVISION. Registering a same-locale change repaints through the
 *    ONE store channel and bumps the snapshot's local revision — not a wire
 *    revision — and the document only claims a locale the owned copy carries:
 *    French ships its relay bundle since #3131 and paints fr, while a supported
 *    locale with no relay bundle falls back to English under its own backend
 *    preference.
 *
 * These are behaviour assertions. They make NO screenshot claim; rendered
 * design QA for the relay SPA is intentionally blocked from this session (the
 * browser tool cannot reach a live host here) and is tracked separately.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BUNDLED_OWNED } from "./i18n/bundled.gen";
import { RESOLVED_LOCALES } from "./i18n/runtime";
import { messageRuntime, useLocaleSnapshot } from "./store";

/** The exact copy the session surfaces hardcoded BEFORE this slice (identity). */
const BEFORE = {
	"relay.session.list.search_placeholder": "Search conversations…",
	"relay.session.list.empty_hint": "start one below, or from the TUI on your machine",
	"relay.session.list.no_matches": "no matching conversations",
	"relay.session.list.start_idle": "new session",
	"relay.session.list.start_busy": "starting…",
	"relay.session.view.back": "back to sessions",
	"relay.session.view.empty_title": "no messages yet",
	"relay.session.view.empty_hint": "send a message below to get started",
	"relay.session.past.back": "back",
	"relay.session.past.heading": "past sessions",
	"relay.session.past.search_placeholder": "search names and conversations…",
	"relay.session.past.loading": "loading…",
	"relay.session.directory.sheet_title": "working directory",
	"relay.session.directory.path_placeholder": "or type another path…",
	"relay.session.model.sheet_title": "model",
	"relay.session.model.filter_placeholder": "filter models",
	"relay.session.model.loading": "loading…",
} as const;

afterEach(() => {
	vi.unstubAllGlobals();
	document.documentElement.lang = "en";
	document.documentElement.dir = "ltr";
});

/** The keys the redesigned list ACTUALLY renders (sentence case, O1). The
    superseded `relay.session.*` values above stay frozen and unused — the
    catalogue freeze forbids editing an existing English value, so a changed
    string is a new `relay.chrome.*` key and both are asserted. */
const CHROME = {
	"relay.chrome.list.no_matches": "No matching conversations",
	"relay.chrome.list.new": "New",
	"relay.chrome.list.new_label": "New session",
	"relay.chrome.list.starting": "Starting…",
} as const;

describe("relay.chrome list copy the redesign renders", () => {
	it("paints the sentence-case English the list surfaces use", () => {
		for (const [key, text] of Object.entries(CHROME)) {
			expect(messageRuntime.t(key as keyof typeof CHROME)).toBe(text);
		}
	});
});

describe("relay.session extraction", () => {
	it("renders byte-identical English for every owned session key", () => {
		for (const [key, text] of Object.entries(BEFORE)) {
			expect(messageRuntime.t(key as keyof typeof BEFORE)).toBe(text);
		}
	});

	it("follows the actual owned-copy language, not the backend preference", () => {
		// The contract from #2297: lang/dir describe the ACTUAL rendered owned-copy
		// language (uiLocale), not the backend preference. Both directions are
		// pinned here: a locale the relay ships paints its own language, and a
		// SUPPORTED locale it does not ship paints the WHOLE English bundle instead
		// of claiming copy nobody translated — the half that goes red if the store
		// ever follows the backend preference blindly.
		//
		// The unshipped locale is DERIVED from the generated manifest rather than
		// named, because the shipped set moves while a branch is open: vi shipped
		// with #3163 and the catalogue waves for es (#3164) and hi (#3166) are
		// still in flight. A fixed name would turn this test red the day one of
		// them lands and invite someone to flip it to the shipped language,
		// deleting the assertion it exists to make. Measured on this tree, the
		// shipped set is en/fr/ru/vi/ur/zh-CN, so the pick is es.
		const unshipped = RESOLVED_LOCALES.find(
			locale => locale !== "en" && !Object.values(BUNDLED_OWNED).some(bundle => locale in bundle),
		);
		expect(messageRuntime.uiLocaleFor("fr")).toBe("fr");
		expect(messageRuntime.uiLocaleFor("ur")).toBe("ur");
		// The fallback arm exists only while a supported locale still lacks a relay
		// catalogue; es (#3164) and hi (#3166) are the last two, so the arm is
		// skipped rather than asserted once they land. A green run must never
		// depend on a gap in the catalogue set, which is the same trap as naming
		// the locale: it goes red on the day the gap closes.
		if (unshipped !== undefined) expect(messageRuntime.uiLocaleFor(unshipped)).toBe("en");
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		// No config frame is applied in this fixture, so the snapshot is the boot
		// state and says nothing about either preference.
		expect(result.current.backendLocale).toBeNull();
		expect(result.current.uiLocale).toBe("en");
		expect(result.current.direction).toBe("ltr");
		unmount();
	});

	it("selects the whole English bundle when the resolved locale is missing or partial", () => {
		// A partial translation of relay.session must NOT be used: requiring the
		// full English key set is what keeps one page from mixing languages.
		act(() => {
			messageRuntime.registerBundle("relay.session", "ur", {
				"relay.session.past.heading": "اردو",
			});
		});
		expect(messageRuntime.uiLocaleFor("ur")).toBe("en");
		// And the "translated" key still paints English, because the bundle is not
		// complete enough to select at all.
		expect(messageRuntime.t("relay.session.past.heading")).toBe("past sessions");
	});

	it("bumps the local revision on a same-locale registration through the one store channel", () => {
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		const before = result.current.revision;
		act(() => {
			messageRuntime.registerBundle("relay.session", "en", {
				...BEFORE,
				"relay.session.list.no_matches": "nothing matched",
			});
		});
		const after = result.current.revision;
		expect(after).toBeGreaterThan(before);
		expect(result.current.backendLocale).toBeNull(); // untouched: not a wire change
		unmount();
	});
});
