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
 *    revision — and the document stays EN/LTR under a French backend, because
 *    the owned copy on screen is English.
 *
 * These are behaviour assertions. They make NO screenshot claim; rendered
 * design QA for the relay SPA is intentionally blocked from this session (the
 * browser tool cannot reach a live host here) and is tracked separately.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
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

describe("relay.session extraction", () => {
	it("renders byte-identical English for every owned session key", () => {
		for (const [key, text] of Object.entries(BEFORE)) {
			expect(messageRuntime.t(key as keyof typeof BEFORE)).toBe(text);
		}
	});

	it("keeps the document EN/LTR under a French backend, because the copy is English", () => {
		// The contract from #2297: lang/dir describe the ACTUAL rendered owned-copy
		// language (uiLocale), not the backend preference. With only English owned
		// bundles, a French (or Urdu) backend still paints EN/LTR.
		expect(messageRuntime.uiLocaleFor("fr")).toBe("en");
		expect(messageRuntime.uiLocaleFor("ur")).toBe("en");
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
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
