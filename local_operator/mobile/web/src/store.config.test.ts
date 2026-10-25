// @vitest-environment happy-dom
/**
 * The config stream (wave-5 slice D): the relay's resolved language, live.
 *
 * What these pin: a frame lands the language on the store AND mirrors it onto
 * ``<html lang>``; an older daemon's 404 is treated as ABSENT — no
 * subscription, no retry timers, ever; and a transient probe failure retries
 * with backoff rather than being read as absence.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BUNDLED_OWNED } from "./i18n/bundled.gen";
import { initializeLocaleMetadata, messageRuntime, retainConfigStream, useLanguage, useLocaleSnapshot } from "./store";

class FakeEventSource {
	static instances: FakeEventSource[] = [];
	onopen: (() => void) | null = null;
	onerror: (() => void) | null = null;
	listeners = new Map<string, (event: MessageEvent) => void>();
	closed = false;
	constructor(readonly url: string) {
		FakeEventSource.instances.push(this);
	}
	addEventListener(name: string, listener: EventListenerOrEventListenerObject) {
		this.listeners.set(name, listener as (event: MessageEvent) => void);
	}
	close() {
		this.closed = true;
	}
}

const configOk = () =>
	new Response(null, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});

describe("retainConfigStream", () => {
	it("starts English with unknown backend, without device-language inference or catalogue fetching", async () => {
		vi.resetModules();
		const fresh = await import("./store");
		const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
		vi.stubGlobal("navigator", { language: "ur-PK", languages: ["ur-PK"] });
		fresh.initializeLocaleMetadata();
		const { result, unmount } = renderHook(() => fresh.useLocaleSnapshot());
		// Each approved owned namespace/locale pair registers once through the
		// existing repaint channel. Pin ownership identities independently, then
		// derive the initial revision count; never suppress same-locale updates.
		// The rebase unioned both branches' slices: relay.session (main, C2) and
		// relay.account_policy (LOH-12) join relay.shell in OWNED_SPA_NAMESPACES;
		// relay.library is the phone Library's own en-only copy (plural count);
		// relay.session.tags is the LOH-11 tag slice's own namespace.
		// relay.ask is the R5 ask card's own: the dismiss reasons and the
		// stale-ask context line, bundled like every other owned surface.
		// relay.asks is the asks/cards slice's owned copy (ask card, sheet, dock,
		// pending card, approval gate and disclosure). Both are OWNED slices on
		// either side of the rebase; the namespace names differ on purpose.
		// relay.approvals is the approvals-policy slice's own en-only copy (the
		// policy state line, the global scope and the mode-change links).
		const ownedNamespaces = Object.keys(BUNDLED_OWNED).sort();
		expect(ownedNamespaces).toEqual([
			"relay.account_policy",
			"relay.approvals",
			"relay.ask",
			"relay.asks",
			"relay.library",
			"relay.session",
			"relay.session.tags",
			"relay.shell",
		]);
		const registrations = Object.values(BUNDLED_OWNED).reduce(
			(count, locales) => count + Object.keys(locales).length,
			0,
		);
		expect(result.current).toEqual({ backendLocale: null, uiLocale: "en", direction: "ltr", revision: registrations });
		expect(document.documentElement.lang).toBe("en"); expect(document.documentElement.dir).toBe("ltr");
		expect(fetchMock).not.toHaveBeenCalled(); unmount();
	});
	it("keeps backend fr separate from English UI and publishes same-locale registration via existing subscription", async () => {
		vi.stubGlobal("EventSource", FakeEventSource);
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(configOk()));
		initializeLocaleMetadata();
		const languageHook = renderHook(() => useLanguage());
		const snapshotHook = renderHook(() => useLocaleSnapshot());
		const close = retainConfigStream();
		try {
			await vi.waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
			const source = FakeEventSource.instances[0];
			act(() => { source.listeners.get("config")?.({ data: JSON.stringify({ language: "fr" }) } as MessageEvent); });
			expect(languageHook.result.current).toBe("fr");
			expect(snapshotHook.result.current).toMatchObject({ backendLocale: "fr", uiLocale: "en", direction: "ltr" });
			expect(document.documentElement.lang).toBe("en"); expect(document.documentElement.dir).toBe("ltr");
			const previous = snapshotHook.result.current;
			act(() => { messageRuntime.registerWire("wire.exec", "fr", { "wire.exec.error_line": "Fixture {error}" }); });
			expect(snapshotHook.result.current.revision).toBe(previous.revision + 1);
			const registered = snapshotHook.result.current;
			for (const invalid of ["auto", "fr-CA", "bogus", ""]) act(() => { source.listeners.get("config")?.({ data: JSON.stringify({ language: invalid }) } as MessageEvent); });
			expect(snapshotHook.result.current).toBe(registered);
			expect(languageHook.result.current).toBe("fr");
		} finally { close(); languageHook.unmount(); snapshotHook.unmount(); }
		const offline = renderHook(() => useLocaleSnapshot());
		expect(offline.result.current.backendLocale).toBe("fr");
		expect(offline.result.current.uiLocale).toBe("en");
		offline.unmount();
	});
	afterEach(() => {
		vi.useRealTimers();
		FakeEventSource.instances = [];
		vi.unstubAllGlobals();
		document.documentElement.lang = "en";
	});

	it("applies a frame to the store and mirrors it onto <html lang>", async () => {
		vi.stubGlobal("EventSource", FakeEventSource);
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(configOk()));
		const { result, unmount } = renderHook(() => useLanguage());
		const close = retainConfigStream();
		try {
			await vi.waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
			const source = FakeEventSource.instances[0];
			act(() => {
				source.listeners.get("config")?.({
					data: JSON.stringify({ changed_keys: ["language"], language: "en" }),
				} as MessageEvent);
			});
			expect(result.current).toBe("en");
			expect(document.documentElement.lang).toBe("en");
		} finally {
			close();
			unmount();
		}
	});

	it("drops a malformed frame without touching the language", async () => {
		vi.stubGlobal("EventSource", FakeEventSource);
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(configOk()));
		const { result, unmount } = renderHook(() => useLanguage());
		const close = retainConfigStream();
		try {
			await vi.waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
			const source = FakeEventSource.instances[0];
			/* A KNOWN value first — the module store is shared across this file's
			   tests, so "unchanged" is the assertion, not "null". */
			act(() => {
				source.listeners.get("config")?.({
					data: JSON.stringify({ changed_keys: [], language: "en" }),
				} as MessageEvent);
			});
			act(() => {
				source.listeners.get("config")?.({ data: "not json" } as MessageEvent);
			});
			expect(result.current).toBe("en");
		} finally {
			close();
			unmount();
		}
	});

	it("treats an older daemon's 404 as ABSENT: probed ONCE, no subscription, no retries", async () => {
		vi.useFakeTimers();
		vi.stubGlobal("EventSource", FakeEventSource);
		const fetchMock = vi.fn().mockResolvedValue(new Response("not found", { status: 404 }));
		vi.stubGlobal("fetch", fetchMock);
		const close = retainConfigStream();
		try {
			await vi.advanceTimersByTimeAsync(60_000);
			/* Exactly ONE probe (CI round 1, MINOR-2): a re-probing regression
			   must fail here, not just the "no subscription" half. */
			expect(fetchMock).toHaveBeenCalledTimes(1);
			expect(FakeEventSource.instances).toHaveLength(0);
		} finally {
			close();
		}
	});

	it("self-heals a transient 5xx: retry, then subscribe — only 404 is absence", async () => {
		vi.useFakeTimers();
		vi.stubGlobal("EventSource", FakeEventSource);
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(new Response("boom", { status: 503 }))
			.mockResolvedValue(configOk());
		vi.stubGlobal("fetch", fetchMock);
		const close = retainConfigStream();
		try {
			await vi.advanceTimersByTimeAsync(1200);
			expect(fetchMock).toHaveBeenCalledTimes(2);
			expect(FakeEventSource.instances).toHaveLength(1);
		} finally {
			close();
		}
	});

	it("retries a transient probe failure with backoff, then subscribes", async () => {
		vi.useFakeTimers();
		vi.stubGlobal("EventSource", FakeEventSource);
		const fetchMock = vi
			.fn()
			.mockRejectedValueOnce(new Error("offline"))
			.mockResolvedValue(configOk());
		vi.stubGlobal("fetch", fetchMock);
		const close = retainConfigStream();
		try {
			await vi.advanceTimersByTimeAsync(1200);
			expect(fetchMock).toHaveBeenCalledTimes(2);
			expect(FakeEventSource.instances).toHaveLength(1);
		} finally {
			close();
		}
	});
});

// The FIRST CONTENT PAINT (i18n review round 1, QA Q-2 / design D3). A cold
// deep link used to paint English copy before the baseline frame arrived, so
// `main.tsx` resolves the baseline BEFORE mounting. These pin the resolver's
// contract: it takes the first frame of the SAME config route, applies it
// through the one authority, and leaves the English default when the route is
// absent or the stream never speaks.
describe("resolveInitialLocale", () => {
	const frame = (language: string) =>
		new Response(`event: config\ndata: {"changed_keys":[],"language":"${language}"}\n\n`, {
			headers: { "content-type": "text/event-stream" },
		});

	it("publishes the baseline frame so lang/dir and copy agree on the first paint", async () => {
		vi.resetModules();
		const fresh = await import("./store");
		const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => frame("ur"));
		vi.stubGlobal("fetch", fetchMock);
		await fresh.resolveInitialLocale();
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/config/events");
		const { result, unmount } = renderHook(() => fresh.useLocaleSnapshot());
		expect(result.current.uiLocale).toBe("ur");
		expect(result.current.direction).toBe("rtl");
		expect(document.documentElement.lang).toBe("ur");
		expect(document.documentElement.dir).toBe("rtl");
		unmount();
	});

	it("keeps the English default when the route is absent or stalled", async () => {
		const routes: Array<() => Promise<Response>> = [
			() => Promise.resolve(new Response(null, { status: 404 })),
			() => Promise.reject(new Error("offline")),
		];
		for (const route of routes) {
			vi.resetModules();
			const fresh = await import("./store");
			vi.stubGlobal("fetch", vi.fn(route));
			await fresh.resolveInitialLocale();
			const { result, unmount } = renderHook(() => fresh.useLocaleSnapshot());
			expect(result.current).toEqual({ backendLocale: null, uiLocale: "en", direction: "ltr", revision: expect.any(Number) });
			expect(document.documentElement.dir).toBe("ltr");
			unmount();
		}
	});
});
