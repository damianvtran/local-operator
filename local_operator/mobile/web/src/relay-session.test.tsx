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
import { BUNDLED_CHUNKS } from "./i18n/bundled.gen";
import { RESOLVED_LOCALES } from "./i18n/runtime";
import { localeBundleLoader, messageRuntime, useLocaleSnapshot } from "./store";

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

	it("follows the actual owned-copy language, not the backend preference", async () => {
		// The contract from #2297: lang/dir describe the ACTUAL rendered owned-copy
		// language (uiLocale), not the backend preference. BOTH directions are pinned
		// here, in the halves the lazy tree actually has: a shipped locale's copy is a
		// chunk the store has to REQUEST (`BUNDLED_CHUNKS`), so until it is requested
		// the surface is English, and once it lands the locale is that locale's — the
		// half that goes red if the store ever follows the backend preference
		// blindly.
		//
		// The unshipped pick is DERIVED from the shipped manifest rather than named,
		// because the shipped set moves while a branch is open: vi shipped with
		// #3163, es with #3172, and the hi catalogue (#3166) is still in flight. A
		// fixed name would turn this test red the day one of them lands and invite
		// someone to flip it to the shipped language, deleting the assertion it
		// exists to make. Shipped-ness is read from `BUNDLED_CHUNKS`, never from
		// `BUNDLED_OWNED`: the eager map holds English only, so asking it would call
		// every locale unshipped and pick one that does have copy.
		const unshipped = RESOLVED_LOCALES.find(
			locale => locale !== "en" && !(locale in BUNDLED_CHUNKS),
		);
		// Nothing in this fixture has asked the loader for a chunk yet, so both a
		// shipped LTR (fr) and a shipped RTL (ur) locale fall back to the WHOLE
		// English bundle rather than claiming copy the store has not received.
		expect(messageRuntime.uiLocaleFor("fr")).toBe("en");
		expect(messageRuntime.uiLocaleFor("ur")).toBe("en");
		// Then the real lazy path, through the store's own loader — the only door that
		// registers a chunk — which is what makes French the rendered language.
		await localeBundleLoader.ensure("fr");
		expect(messageRuntime.uiLocaleFor("fr")).toBe("fr");
		// The fallback arm exists only while a supported locale still lacks a
		// catalogue, so it is skipped rather than asserted once hi lands: a green run
		// must never depend on a gap in the catalogue set, which is the same trap as
		// naming the locale — it goes red on the day the gap closes.
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
