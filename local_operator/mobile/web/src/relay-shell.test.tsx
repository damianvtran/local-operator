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
import { BUNDLED_OWNED } from "./i18n/bundled.gen";
import { RESOLVED_LOCALES } from "./i18n/runtime";
import { messageRuntime, retainConfigStream, useLocaleSnapshot } from "./store";
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
		expect(unshipped).toBeDefined();
		expect(messageRuntime.uiLocaleFor("fr")).toBe("fr");
		expect(messageRuntime.uiLocaleFor("ur")).toBe("ur");
		expect(messageRuntime.uiLocaleFor(unshipped!)).toBe("en");
		const { result, unmount } = renderHook(() => useLocaleSnapshot());
		// No config frame is applied in this fixture, so the snapshot is the boot
		// state and says nothing about either preference.
		expect(result.current.backendLocale).toBeNull();
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

	it("switches the real document en→ur→en only after every owned synthetic bundle is complete", async () => {
		// These values are fixtures, not Urdu translations. They exercise the same
		// generated manifest, registration, config event and root metadata path.
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
			messageRuntime.registerBundle(namespace, "ur", Object.fromEntries(
				Object.keys(english).map(key => [key, `UR-FIXTURE ${key}`]),
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
			await emit("ur");
			expect(result.current).toMatchObject({ backendLocale: "ur", uiLocale: "ur", direction: "rtl" });
			expect(document.documentElement.lang).toBe("ur");
			expect(document.documentElement.dir).toBe("rtl");
			expect(messageRuntime.t("relay.shell.list.empty")).toBe("UR-FIXTURE relay.shell.list.empty");
			await emit("en");
			expect(result.current).toMatchObject({ backendLocale: "en", uiLocale: "en", direction: "ltr" });
			expect(document.documentElement.lang).toBe("en");
			expect(document.documentElement.dir).toBe("ltr");
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
});
