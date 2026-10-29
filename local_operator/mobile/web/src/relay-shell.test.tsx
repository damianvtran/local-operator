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
import { messageRuntime, retainConfigStream, useLocaleSnapshot } from "./store";
import { BUNDLED_CHUNKS, BUNDLED_OWNED } from "./i18n/bundled.gen";
import { directionFor } from "./i18n/runtime";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";

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

/** The section headings the drawer's list view actually renders now. */
const CHROME = {
	"relay.chrome.list.section.active": "Active",
	"relay.chrome.list.section.previous": "Previous",
} as const;

describe("relay.chrome section copy the redesign renders", () => {
	it("paints the sentence-case English the sections use", () => {
		for (const [key, text] of Object.entries(CHROME)) {
			expect(messageRuntime.t(key as keyof typeof CHROME)).toBe(text);
		}
	});
});

describe("relay.shell extraction", () => {
	it("renders byte-identical English for every owned shell key", () => {
		for (const [key, text] of Object.entries(BEFORE)) {
			expect(messageRuntime.t(key as keyof typeof BEFORE)).toBe(text);
		}
	});

	it("follows the actual owned-copy language, not the backend preference", () => {
		// The contract from #2297: lang/dir describe the ACTUAL rendered owned-copy
		// language (uiLocale), not the backend preference. French has no emitted
		// owned chunk yet, so a French backend paints EN/LTR. A locale that DOES
		// ship one arrives with its chunk on the config frame — the lazy path is
		// relay-shell's "loads a shipped locale's chunk" test, asserted there so
		// this one does not depend on whether that chunk has already been loaded.
		expect(messageRuntime.uiLocaleFor("fr")).toBe("en");
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

	it("switches the real document en→fr→en for a locale whose owned copy is synthetic", async () => {
		// These values are fixtures, not French translations. French ships no
		// emitted chunk (`BUNDLED_CHUNKS`), so this exercises the gate, the config
		// event and the root metadata without racing a real dynamic import; the
		// real chunk's own path is the next test.
		/* The config channel is a FETCH-streamed SSE reader, not an EventSource:
		   the tunnel's 60 s lease rotation needs the fetch lifecycle. Drive the
		   same fixture the rest of the config tests use — one probing response
		   settles presence, then the live subscription's own stream carries the
		   frames. */
		vi.useFakeTimers();
		Stream.instances = [];
		vi.stubGlobal("fetch", vi.fn()
			.mockResolvedValueOnce(new Response(null, {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			}))
			.mockImplementation(Stream.fetch));
		const englishBefore = messageRuntime.t("relay.shell.list.empty");
		for (const [namespace, locales] of Object.entries(BUNDLED_OWNED)) {
			const english = locales.en;
			expect(english).toBeTruthy();
			messageRuntime.registerBundle(namespace, "fr", Object.fromEntries(
				Object.keys(english).map(key => [key, `FR-FIXTURE ${key}`]),
			));
		}
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		const release = retainConfigStream();
		await settleStream();
		expect(Stream.instances).toHaveLength(1);
		const source = Stream.instances[0];
		const emit = async (language: string) => {
			await act(async () => {
				source.emit("config", JSON.stringify({ changed_keys: ["language"], language }));
				await settleStream();
			});
		};
		try {
			await emit("fr");
			expect(result.current).toMatchObject({ backendLocale: "fr", uiLocale: "fr", direction: "ltr" });
			expect(document.documentElement.lang).toBe("fr");
			expect(document.documentElement.dir).toBe("ltr");
			expect(messageRuntime.t("relay.shell.list.empty")).toBe("FR-FIXTURE relay.shell.list.empty");
			await emit("en");
			expect(result.current).toMatchObject({ backendLocale: "en", uiLocale: "en", direction: "ltr" });
			expect(document.documentElement.lang).toBe("en");
			expect(messageRuntime.t("relay.shell.list.empty")).toBe(englishBefore);
		} finally {
			await emit("en");
			release();
			unmount();
			vi.useRealTimers();
			vi.unstubAllGlobals();
			messageRuntime.registerBundle("relay.shell", "en", { ...BEFORE });
			Stream.instances = [];
		}
	});

	it("loads a shipped locale's chunk on the config frame and flips the document only then", async () => {
		/* The real lazy path: the eager map is English, so the frame leaves the
		   page English until `bundled.<locale>.gen` lands, then registers through
		   the same `registerBundle` channel. Pre-loading the module only settles
		   WHEN the store's own import resolves; nothing is registered by hand. */
		const shipped = Object.keys(BUNDLED_CHUNKS);
		if (shipped.length === 0) return; // no complete locale to load yet
		const locale = shipped.includes("ur") ? "ur" : shipped[0];
		await BUNDLED_CHUNKS[locale]();
		vi.useFakeTimers();
		Stream.instances = [];
		vi.stubGlobal("fetch", vi.fn()
			.mockResolvedValueOnce(new Response(null, {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			}))
			.mockImplementation(Stream.fetch));
		const englishBefore = messageRuntime.t("relay.session.list.empty_hint");
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		const release = retainConfigStream();
		await settleStream();
		const source = Stream.instances[0];
		const emit = async (language: string) => {
			await act(async () => {
				source.emit("config", JSON.stringify({ changed_keys: ["language"], language }));
				await settleStream();
			});
		};
		try {
			await emit(locale);
			// The chunk resolves outside the timer queue: pump until the snapshot
			// reports the loaded locale, then assert the whole page moved together.
			for (let tick = 0; tick < 20 && result.current.uiLocale !== locale; tick += 1) {
				await act(async () => { await settleStream(); });
			}
			expect(result.current).toMatchObject({
				backendLocale: locale,
				uiLocale: locale,
				direction: directionFor(locale),
			});
			expect(document.documentElement.lang).toBe(locale);
			expect(document.documentElement.dir).toBe(directionFor(locale));
			// The copy actually came from the chunk, not the English fallback.
			expect(messageRuntime.t("relay.session.list.empty_hint")).not.toBe(englishBefore);
		} finally {
			await emit("en");
			release();
			unmount();
			vi.useRealTimers();
			vi.unstubAllGlobals();
			Stream.instances = [];
		}
	});

});
