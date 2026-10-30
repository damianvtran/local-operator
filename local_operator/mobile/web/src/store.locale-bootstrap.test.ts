// @vitest-environment happy-dom
/** The cold deep link's FIRST CONTENT PAINT (i18n review round 1, QA Q-2 /
 *  design D3; round 2, #3165 major).
 *
 * `main.tsx` awaits `resolveInitialLocale()` before it mounts anything, so the
 * guarantee is about the snapshot a mounted consumer sees first: with the
 * locale's chunk inside the bootstrap budget it must already be that locale —
 * an English first frame under Urdu metadata is the regression this pins. Past
 * the budget the documented fallback stands: complete English now, the locale
 * when the chunk lands. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

type OwnedCopy = Record<string, Record<string, Record<string, string>>>;

/** A complete Urdu owned map for the namespaces the store's owned set covers. */
function urduCopy(namespaces: string[], english: OwnedCopy): OwnedCopy {
	const copy: OwnedCopy = {};
	for (const namespace of namespaces) {
		copy[namespace] = { ur: Object.fromEntries(Object.keys(english[namespace]?.en ?? {}).map(key => [key, `UR-FIXTURE ${key}`])) };
	}
	return copy;
}

function frame(language: string): Response {
	return new Response(`data: ${JSON.stringify({ changed_keys: ["language"], language })}\n\n`, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

/** A fresh store with the Urdu chunk under this test's control. Returns the
 *  module plus the resolver the test calls to hand the chunk over (or never). */
async function freshStore(chunkResolves: boolean, language = "ur") {
	vi.resetModules();
	const generated = await import("./i18n/bundled.gen");
	const english = generated.BUNDLED_OWNED;
	let release!: () => void;
	const gate = new Promise<void>(resolve => { release = resolve; });
	const previous = (generated.BUNDLED_CHUNKS as Record<string, unknown>)[language];
	(generated.BUNDLED_CHUNKS as Record<string, () => Promise<{ BUNDLED_LOCALE: OwnedCopy }>>)[language] =
		async () => {
			if (!chunkResolves) await new Promise<void>(() => { /* never */ });
			await gate;
			return { BUNDLED_LOCALE: urduCopy(Object.keys(english), english) };
		};
	vi.stubGlobal("fetch", vi.fn(async () => frame(language)));
	const store = await import("./store");
	return { store, release, english, restore: () => { (generated.BUNDLED_CHUNKS as Record<string, unknown>)[language] = previous; } };
}

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("cold deep link before the first content paint", () => {
	it("paints the chunk's locale on the first frame when it lands inside the budget", async () => {
		const { store, release, restore } = await freshStore(true);
		try {
			let settled = false;
			const resolving = store.resolveInitialLocale();
			void resolving.then(() => { settled = true; });
			// The boot is WAITING on the chunk, not merely promising to fix the paint
			// later: without this the test would pass on an English first frame too.
			await Promise.resolve();
			await Promise.resolve();
			expect(settled).toBe(false);
			release();
			await resolving;
			// The first snapshot ANY mounted consumer sees is already the locale:
			// there is no English frame to correct afterwards.
			const { result, unmount } = renderHook(() => store.useLocaleSnapshot());
			expect(result.current).toMatchObject({ backendLocale: "ur", uiLocale: "ur", direction: "rtl" });
			expect(document.documentElement.lang).toBe("ur");
			expect(document.documentElement.dir).toBe("rtl");
			expect(store.messageRuntime.t("relay.shell.list.empty")).not.toBe("no sessions running");
			unmount();
		} finally { restore(); }
	});

	it("falls back to the documented English default when the chunk misses the budget", async () => {
		vi.useFakeTimers();
		const { store, restore } = await freshStore(false);
		try {
			let settled = false;
			const resolving = store.resolveInitialLocale();
			void resolving.then(() => { settled = true; });
			// The chunk never arrives: the boot waits — but only for its budget.
			await vi.advanceTimersByTimeAsync(50);
			expect(settled).toBe(false);
			// Past the budget it finishes rather than hanging, and the first frame is
			// the documented complete-English fallback.
			await vi.advanceTimersByTimeAsync(store.LOCALE_BOOTSTRAP_BUDGET_MS);
			await resolving;
			expect(settled).toBe(true);
			const { result, unmount } = renderHook(() => store.useLocaleSnapshot());
			expect(result.current).toMatchObject({ backendLocale: "ur", uiLocale: "en", direction: "ltr" });
			expect(document.documentElement.lang).toBe("en");
			expect(store.messageRuntime.t("relay.shell.list.empty")).toBe("no sessions running");
			unmount();
		} finally { restore(); }
	});
});
