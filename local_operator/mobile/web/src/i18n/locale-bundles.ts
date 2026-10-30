/** Owned-copy chunks only: one eager English map at init, one chunk per shipped
 * locale on first demand. No catalogue fetch, locale inference, storage or
 * second locale truth — `registerBundles` is the only door.
 *
 * WHY A LOADER AT ALL. Every owned string exists in English, so the eager map
 * is complete on its own; the other shipped locales are 26–36 kB built each and
 * only one is ever active (`bundled.gen.ts`, `BUNDLED_CHUNKS`). Loading one is a
 * SIZE optimisation, never a correctness prerequisite: a chunk that never
 * arrives leaves the whole page in English — never half of it in two languages,
 * because the runtime only selects a locale whose owned namespaces are ALL
 * complete (`uiLocaleFor`).
 *
 * TIMING IS THE STORE'S DECISION, NOT THIS MODULE'S. `ensure` returns the load
 * it started so the pre-mount bootstrap can AWAIT it inside its budget: a chunk
 * that lands in time paints the right language on the first frame; one that
 * misses falls back to the English default instead of hanging the boot. Live
 * config frames ignore the promise and let the flip land when it lands. */
import { BUNDLED_CHUNKS } from "./bundled.gen";

/** One generated owned-copy map: namespace -> locale -> message key -> string. */
export type OwnedCopy = Readonly<Record<string, Readonly<Record<string, Readonly<Record<string, string>>>>>>;

export type LocaleBundleManifest = Readonly<Record<string, () => Promise<{ readonly BUNDLED_LOCALE: OwnedCopy }>>>;

/** Namespace, locale and its message map — the shape `registerBundles` takes. */
export type OwnedCopyEntry = [string, string, Record<string, string>];

interface Options {
	/** Registers a whole owned-copy map with one notification — the store passes
	    `messageRuntime.registerBundles`, keeping the single notification channel. */
	activate: (entries: Iterable<OwnedCopyEntry>) => void;
	/** The generated manifest, injectable so a test can hold the chunk open and
	    observe the bootstrap's budget instead of racing a real import. */
	manifest?: LocaleBundleManifest;
	/** Retry delays after a failed load, in order; an empty list disables retry. */
	retryDelaysMs?: readonly number[];
}

/** A failed chunk is an offline blip or a redeploy mid-session — worth a few
 *  retries on a widening gap. Without them the only retry is the next config
 *  frame, which may never come in a long-lived tab: one blip would leave the
 *  reader in English for the rest of the session. The count therefore never
 *  needs clearing: a successful load keeps its resolved promise in `pending`, so
 *  no later attempt can follow it — the counter only accrues inside one failed
 *  episode. Once this list is spent, only a frame that names the locale starts
 *  another attempt, and that one gets no backoff; the English fallback stands, so
 *  a spent schedule is a reader in English, never a stuck one. */
const RETRY_DELAYS_MS = [1_000, 4_000, 16_000];

export function createLocaleBundleLoader(options: Options) {
	const manifest = options.manifest ?? BUNDLED_CHUNKS;
	const retryDelays = options.retryDelaysMs ?? RETRY_DELAYS_MS;
	/** In-flight (or settled) load per locale, so two frames name one fetch. */
	const pending = new Map<string, Promise<void>>();
	const failures = new Map<string, number>();

	function register(module: { readonly BUNDLED_LOCALE: OwnedCopy }): void {
		// The module repeats its locale tag per namespace; register the tags the
		// chunk itself carries, so a mismatched chunk cannot be labelled with a
		// locale the runtime would then select its copy for.
		const entries: OwnedCopyEntry[] = [];
		for (const [namespace, locales] of Object.entries(module.BUNDLED_LOCALE)) {
			for (const [tag, messages] of Object.entries(locales)) {
				entries.push([namespace, tag, { ...messages }]);
			}
		}
		options.activate(entries);
	}

	function attempt(locale: string, load: () => Promise<{ readonly BUNDLED_LOCALE: OwnedCopy }>): Promise<void> {
		const promise = load().then(module => { register(module); });
		pending.set(locale, promise);
		void promise.catch((error: unknown) => {
			// A persistent 404 (a tab holding an older entry against a rebuilt
			// `dist`) would otherwise leave the device in English with nothing in any
			// log to explain it. Developer log line, never user copy.
			console.debug("[i18n] locale chunk failed", locale, error); // i18n: ignore developer log
			if (pending.get(locale) === promise) pending.delete(locale);
			const failed = (failures.get(locale) ?? 0) + 1;
			failures.set(locale, failed);
			const delay = retryDelays[failed - 1];
			if (delay === undefined) return;
			const timer = setTimeout(() => { void start(locale); }, delay);
			// Never keep a node/test process alive for a retry nobody is waiting on.
			(timer as { unref?: () => void }).unref?.();
		});
		return promise;
	}

	function start(locale: string): Promise<void> | null {
		const inFlight = pending.get(locale);
		if (inFlight) return inFlight;
		const load = manifest[locale];
		// English is already registered, and a partial locale ships no chunk: both
		// stay English rather than fetching bytes that could not render.
		if (locale === "en" || !load) return null;
		return attempt(locale, load);
	}

	return {
		/** Start (or join) the chunk load for `locale`; `null` when there is
		 *  nothing to load. Awaited by the pre-mount bootstrap, ignored by the
		 *  live stream. */
		ensure(locale: string | null): Promise<void> | null {
			if (!locale) return null;
			return start(locale);
		},
	};
}
