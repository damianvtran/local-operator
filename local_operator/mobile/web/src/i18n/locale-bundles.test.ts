/** The lazy locale chunks: English is complete first, the chunk only refines it.
 *
 * The window these tests describe is a product decision, not an accident: while
 * a chunk is in flight the page is WHOLLY English (the runtime refuses a locale
 * whose owned namespaces are not all present), and the chunk landing repaints it
 * as one page. The bootstrap's promise handling — awaiting that load inside its
 * budget so a cold deep link paints the right language first — is asserted in
 * `store.locale-bootstrap.test.ts`. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMessageRuntime, directionFor, type LocaleSnapshot } from "./runtime";
import { createLocaleBundleLoader, type LocaleBundleManifest } from "./locale-bundles";

type MutableContext = { -readonly [K in keyof LocaleSnapshot]: LocaleSnapshot[K] };
type OwnedCopy = Record<string, Record<string, Record<string, string>>>;

const ENGLISH: OwnedCopy = {
	"relay.shell": { en: { "relay.shell.list.empty": "no sessions running" } },
};
const URDU: OwnedCopy = {
	"relay.shell": { ur: { "relay.shell.list.empty": "کوئی سیشن نہیں چل رہا" } },
};

/** Several owned namespaces, so "one notification per chunk" is observable. */
const WIDE_URDU: OwnedCopy = {
	"relay.shell": { ur: { "relay.shell.list.empty": "خالی" } },
	"relay.session": { ur: { "relay.session.list.empty_hint": "شروع کریں" } },
	"relay.asks": { ur: { "relay.asks.card.decline": "انکار کریں" } },
};

/** The store's own wiring in miniature: one runtime, one snapshot published
 *  through `changed()` exactly like `publishLocaleSnapshot`, and the eager
 *  English map registered the way `store.ts` does at init. */
function harness(manifest: LocaleBundleManifest, retryDelaysMs?: readonly number[]) {
	const context: MutableContext = { backendLocale: null, uiLocale: "en", direction: "ltr", revision: 0 };
	const publish = () => {
		context.uiLocale = runtime.uiLocaleFor(context.backendLocale);
		context.direction = directionFor(context.uiLocale);
		context.revision++;
	};
	const runtime = createMessageRuntime(() => context, publish, Object.keys(ENGLISH));
	for (const [namespace, locales] of Object.entries(ENGLISH)) {
		for (const [locale, messages] of Object.entries(locales)) {
			runtime.registerBundle(namespace, locale, { ...messages });
		}
	}
	publish();
	const loader = createLocaleBundleLoader({
		activate: (entries) => { runtime.registerBundles(entries); },
		manifest,
		...(retryDelaysMs ? { retryDelaysMs } : {}),
	});
	return { context, runtime, loader };
}

afterEach(() => { vi.useRealTimers(); });

describe("lazy owned-copy chunks", () => {
	it("keeps the page English-complete while a locale chunk is in flight, then flips as a whole", async () => {
		let settle!: (module: { BUNDLED_LOCALE: OwnedCopy }) => void;
		const load = vi.fn(() => new Promise<{ BUNDLED_LOCALE: OwnedCopy }>(resolve => { settle = resolve; }));
		const { context, runtime, loader } = harness({ ur: load });
		context.backendLocale = "ur";
		const pending = loader.ensure("ur");
		expect(load).toHaveBeenCalledTimes(1);
		// THE WINDOW: every owned namespace is English, so the surface is one
		// en/ltr page — never Urdu strings beside English ones.
		expect(runtime.uiLocaleFor("ur")).toBe("en");
		expect(runtime.t("relay.shell.list.empty")).toBe("no sessions running");
		settle({ BUNDLED_LOCALE: URDU });
		await pending;
		expect(runtime.uiLocaleFor("ur")).toBe("ur");
		expect(runtime.t("relay.shell.list.empty")).toBe("کوئی سیشن نہیں چل رہا");
		expect(directionFor(runtime.uiLocaleFor("ur"))).toBe("rtl");
	});

	it("repaints ONCE for a chunk with many namespaces", async () => {
		// Twelve namespaces are one incoming map, so they are one repaint and one
		// `publishLocaleSnapshot` — not twelve paints of the same flip.
		const { context, runtime, loader } = harness({ ur: async () => ({ BUNDLED_LOCALE: WIDE_URDU }) });
		context.backendLocale = "ur";
		const before = context.revision;
		await loader.ensure("ur");
		expect(context.revision).toBe(before + 1);
		expect(runtime.uiLocaleFor("ur")).toBe("ur");
	});

	it("imports a locale once, and never for English or a locale without a chunk", () => {
		const load = vi.fn(async () => ({ BUNDLED_LOCALE: URDU }));
		const { runtime, loader } = harness({ ur: load });
		const first = loader.ensure("ur");
		// A second frame naming the same locale joins the load in flight.
		expect(loader.ensure("ur")).toBe(first);
		expect(load).toHaveBeenCalledTimes(1);
		expect(loader.ensure("en")).toBeNull();
		expect(loader.ensure("fr")).toBeNull();
		expect(loader.ensure(null)).toBeNull();
		expect(load).toHaveBeenCalledTimes(1);
		expect(runtime.uiLocaleFor("fr")).toBe("en");
	});

	it("retries a failed chunk on a widening, bounded schedule", async () => {
		vi.useFakeTimers();
		const load = vi.fn()
			.mockRejectedValueOnce(new Error("chunk 404"))
			.mockRejectedValueOnce(new Error("still down"))
			.mockResolvedValueOnce({ BUNDLED_LOCALE: URDU });
		const { context, runtime, loader } = harness({ ur: load }, [1_000, 4_000]);
		context.backendLocale = "ur";
		await loader.ensure("ur")?.catch(() => undefined);
		expect(load).toHaveBeenCalledTimes(1);
		expect(runtime.uiLocaleFor("ur")).toBe("en");
		// First retry after the first delay, second after the second, then stop.
		await vi.advanceTimersByTimeAsync(999);
		expect(load).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(2);
		expect(load).toHaveBeenCalledTimes(2);
		await vi.advanceTimersByTimeAsync(4_000);
		expect(load).toHaveBeenCalledTimes(3);
		await vi.advanceTimersByTimeAsync(60_000);
		expect(load).toHaveBeenCalledTimes(3); // bounded: no attempt after the schedule
		expect(runtime.uiLocaleFor("ur")).toBe("ur");
	});

	it("leaves the English surface standing while every retry fails", async () => {
		vi.useFakeTimers();
		const load = vi.fn().mockRejectedValue(new Error("offline"));
		const { runtime, loader } = harness({ ur: load }, [1_000]);
		const pending = loader.ensure("ur");
		await pending?.catch(() => undefined);
		await vi.advanceTimersByTimeAsync(5_000);
		expect(load).toHaveBeenCalledTimes(2);
		await vi.runAllTimersAsync();
		expect(runtime.uiLocaleFor("ur")).toBe("en");
		expect(runtime.t("relay.shell.list.empty")).toBe("no sessions running");
	});
});
