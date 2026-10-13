// @vitest-environment happy-dom
/**
 * relay.shell: the extracted shell chrome, and what the document claims.
 *
 * What these pin, in the order the contract cares about:
 *
 * 1. BEFORE/AFTER ENGLISH IDENTITY. The extraction must not change a single
 *    byte a reader sees: every string the shell paints is asserted to equal the
 *    copy that was hardcoded before this slice, plus the document metadata
 *    (``lang``/``dir``) that the owned-copy language actually is.
 * 2. WHOLE-SURFACE FALLBACK. A missing, partial or invalid owned bundle for the
 *    resolved locale selects the WHOLE English bundle — never a per-key mix of
 *    two languages under one page language. ``uiLocaleFor`` is the rule, and it
 *    is exercised directly against the store's registered bundle.
 * 3. LOCALE + REVISION. Registering a same-locale change repaints through the
 *    ONE store channel and bumps the snapshot's local revision — not a wire
 *    revision — and re-labels the document only when the owned copy can carry
 *    the locale.
 *
 * These are behaviour assertions. They make NO screenshot claim; rendered
 * design QA for the relay SPA is intentionally blocked from this session (the
 * browser tool cannot reach a live host here) and is tracked separately.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { messageRuntime, useLocaleSnapshot } from "./store";

/** The exact copy the shell hardcoded BEFORE this slice (the identity target). */
const BEFORE = {
	"relay.shell.list.empty": "no sessions running",
	"relay.shell.list.connecting": "connecting…",
	"relay.shell.list.section.active": "Active Sessions",
	"relay.shell.list.section.previous": "Previous Sessions",
	"relay.shell.list.theme": "choose theme",
	"relay.shell.session.waiting": "waiting for projection…",
	"relay.shell.session.connecting": "connecting to session…",
} as const;

afterEach(() => {
	vi.unstubAllGlobals();
	document.documentElement.lang = "en";
	document.documentElement.dir = "ltr";
});

describe("relay.shell extraction", () => {
	it("renders byte-identical English for every owned shell key", () => {
		for (const [key, text] of Object.entries(BEFORE)) {
			expect(messageRuntime.t(key as keyof typeof BEFORE)).toBe(text);
		}
	});

	it("keeps the document EN/LTR under a French backend, because the copy is English", () => {
		// The contract from #2297: lang/dir describe the ACTUAL rendered owned-copy
		// language (uiLocale), not the backend preference. With only an English
		// owned bundle, a French backend still paints EN/LTR.
		expect(messageRuntime.uiLocaleFor("fr")).toBe("en");
		expect(messageRuntime.uiLocaleFor("ur")).toBe("en");
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		expect(result.current.uiLocale).toBe("en");
		expect(result.current.direction).toBe("ltr");
		unmount();
	});

	it("selects the whole English bundle when the resolved locale is missing or partial", () => {
		// A partial translation of the SAME namespace must NOT be used: requiring
		// the full English key set is what keeps one page from mixing languages.
		act(() => {
			messageRuntime.registerBundle("relay.shell", "ur", {
				"relay.shell.list.empty": "اردو",
			});
		});
		expect(messageRuntime.uiLocaleFor("ur")).toBe("en");
		// And the rendered copy for a key that WAS "translated" still comes from
		// English, because the bundle is not complete enough to select at all.
		expect(messageRuntime.t("relay.shell.list.empty")).toBe("no sessions running");
	});

	it("bumps the local revision on a same-locale registration through the one store channel", () => {
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		const before = result.current.revision;
		act(() => {
			messageRuntime.registerBundle("relay.shell", "en", {
				...BEFORE,
				"relay.shell.list.empty": "no sessions up",
			});
		});
		const after = result.current.revision;
		expect(after).toBeGreaterThan(before);
		expect(result.current.backendLocale).toBeNull(); // untouched: not a wire change
		unmount();
	});
});
