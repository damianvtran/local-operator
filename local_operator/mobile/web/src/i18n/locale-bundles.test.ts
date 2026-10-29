/** The lazy locale chunks: English is complete first, the chunk only refines it.
 *
 * The window these tests describe is the product decision, not an accident: a
 * config frame naming a locale repaints English-complete first and that
 * locale's own copy when its chunk lands (`store.ts` `applyConfigFrame`). What
 * must never happen is a MIX — so every assertion here is either "the whole
 * page is English" or "the whole page is that locale". */
import { describe, expect, it, vi } from "vitest";
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

/** The store's own wiring in miniature: one runtime, one snapshot published
 *  through `changed()` exactly like `publishLocaleSnapshot`, and the eager
 *  English map registered the way `store.ts` does at init. */
function harness(manifest: LocaleBundleManifest) {
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
		activate: (namespace, locale, messages) => { runtime.registerBundle(namespace, locale, messages); },
		manifest,
	});
	return { context, runtime, loader };
}

describe("lazy owned-copy chunks", () => {
	it("keeps the page English-complete while a locale chunk is in flight, then flips as a whole", async () => {
		let settle!: (module: { BUNDLED_LOCALE: OwnedCopy }) => void;
		const load = vi.fn(() => new Promise<{ BUNDLED_LOCALE: OwnedCopy }>(resolve => { settle = resolve; }));
		const { context, runtime, loader } = harness({ ur: load });
		context.backendLocale = "ur";
		loader.ensure("ur");
		expect(load).toHaveBeenCalledTimes(1);
		// THE WINDOW: every owned namespace is English, so the surface is one
		// en/ltr page — never Urdu strings beside English ones.
		expect(runtime.uiLocaleFor("ur")).toBe("en");
		expect(runtime.t("relay.shell.list.empty")).toBe("no sessions running");
		settle({ BUNDLED_LOCALE: URDU });
		await vi.waitFor(() => expect(runtime.uiLocaleFor("ur")).toBe("ur"));
		expect(runtime.t("relay.shell.list.empty")).toBe("کوئی سیشن نہیں چل رہا");
		expect(directionFor(runtime.uiLocaleFor("ur"))).toBe("rtl");
	});

	it("imports a locale once, and never for English or a locale without a chunk", () => {
		const load = vi.fn(() => Promise.resolve({ BUNDLED_LOCALE: URDU }));
		const { runtime, loader } = harness({ ur: load });
		loader.ensure("ur");
		loader.ensure("ur");
		expect(load).toHaveBeenCalledTimes(1);
		// English is the eager map; a partial locale (no emitted chunk) must stay
		// English rather than fetch bytes that could not render anyway.
		loader.ensure("en");
		loader.ensure("fr");
		loader.ensure(null);
		expect(load).toHaveBeenCalledTimes(1);
		expect(runtime.uiLocaleFor("fr")).toBe("en");
	});

	it("forgets a rejected chunk so the next frame retries, leaving English standing", async () => {
		const load = vi.fn()
			.mockRejectedValueOnce(new Error("chunk 404"))
			.mockResolvedValueOnce({ BUNDLED_LOCALE: URDU });
		const { runtime, loader } = harness({ ur: load });
		loader.ensure("ur");
		expect(load).toHaveBeenCalledTimes(1);
		// Let the rejection settle: the page is still wholly English.
		await vi.waitFor(() => expect(runtime.t("relay.shell.list.empty")).toBe("no sessions running"));
		expect(runtime.uiLocaleFor("ur")).toBe("en");
		await Promise.resolve();
		loader.ensure("ur");
		await vi.waitFor(() => expect(runtime.uiLocaleFor("ur")).toBe("ur"));
		expect(load).toHaveBeenCalledTimes(2);
	});
});
