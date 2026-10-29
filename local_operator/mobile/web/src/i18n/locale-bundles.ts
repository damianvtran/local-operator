/** Owned-copy chunks only: one eager English map at init, one chunk per shipped
 * locale on first demand. No catalogue fetch, locale inference, storage or
 * second locale truth — `registerBundle` is still the only door.
 *
 * WHY A LOADER AT ALL. Every owned string exists in English, so the eager map
 * is complete on its own; the other shipped locales are 30–70 kB each and only
 * one is ever active (`bundled.gen.ts`, `BUNDLED_CHUNKS`). Loading one is
 * therefore an OPTIMISATION of what is already a finished English surface, not
 * a prerequisite of it: `ensure` is fire-and-forget, and a slow, failed or
 * absent chunk leaves the whole page in English — never half of it in two
 * languages, because the runtime only selects a locale whose owned namespaces
 * are ALL complete (`uiLocaleFor`). */
import { BUNDLED_CHUNKS } from "./bundled.gen";

/** One generated owned-copy map: namespace -> locale -> message key -> string. */
export type OwnedCopy = Readonly<Record<string, Readonly<Record<string, Readonly<Record<string, string>>>>>>;

export type LocaleBundleManifest = Readonly<Record<string, () => Promise<{ readonly BUNDLED_LOCALE: OwnedCopy }>>>;

interface Options {
	/** Registers one namespace's copy for one locale — the store passes
	    `messageRuntime.registerBundle`, keeping the single notification channel. */
	activate: (namespace: string, locale: string, messages: Record<string, string>) => void;
	/** The generated manifest, injectable so a test can hold the chunk open and
	    watch the English window instead of racing a real import. */
	manifest?: LocaleBundleManifest;
}

export function createLocaleBundleLoader(options: Options) {
	const manifest = options.manifest ?? BUNDLED_CHUNKS;
	const requested = new Set<string>();
	return {
		/** Start (once) the chunk for `locale`; no-op for English and for a
		 *  locale that ships no chunk, so a partial locale stays English. */
		ensure(locale: string | null): void {
			if (!locale || locale === "en" || requested.has(locale)) return;
			const load = manifest[locale];
			if (!load) return;
			requested.add(locale);
			void load()
				.then(module => {
					for (const [namespace, locales] of Object.entries(module.BUNDLED_LOCALE)) {
						for (const [tag, messages] of Object.entries(locales)) {
							options.activate(namespace, tag, { ...messages });
						}
					}
				})
				// A failed chunk (offline, a redeploy mid-session) must not poison
				// the locale for the rest of the session: forget it so the next
				// config frame retries, and leave the English surface as it is.
				.catch(() => { requested.delete(locale); });
		},
	};
}
