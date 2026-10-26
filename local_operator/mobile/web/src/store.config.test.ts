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

import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";

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
		// Each approved OWNED English namespace registers once through the
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
			"relay.chrome",
			"relay.library",
			"relay.session",
			"relay.session.tags",
			"relay.shell",
			"relay.transcript",
		]);
		expect(result.current).toEqual({ backendLocale: null, uiLocale: "en", direction: "ltr", revision: ownedNamespaces.length });
		expect(document.documentElement.lang).toBe("en"); expect(document.documentElement.dir).toBe("ltr");
		expect(fetchMock).not.toHaveBeenCalled(); unmount();
	});
	it("keeps backend fr separate from English UI and publishes same-locale registration via existing subscription", async () => {
		vi.useFakeTimers(); Stream.instances = [];
		vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(configOk()).mockImplementation(Stream.fetch));
		initializeLocaleMetadata();
		const languageHook = renderHook(() => useLanguage());
		const snapshotHook = renderHook(() => useLocaleSnapshot());
		const close = retainConfigStream();
		try {
			await settleStream(); expect(Stream.instances).toHaveLength(1);
			const source = Stream.instances[0];
			await act(async () => { source.emit("config", JSON.stringify({ language: "fr" })); await settleStream(); });
			expect(languageHook.result.current).toBe("fr");
			expect(snapshotHook.result.current).toMatchObject({ backendLocale: "fr", uiLocale: "en", direction: "ltr" });
			expect(document.documentElement.lang).toBe("en"); expect(document.documentElement.dir).toBe("ltr");
			const previous = snapshotHook.result.current;
			act(() => { messageRuntime.registerWire("wire.exec", "fr", { "wire.exec.error_line": "Fixture {error}" }); });
			expect(snapshotHook.result.current.revision).toBe(previous.revision + 1);
			const registered = snapshotHook.result.current;
			for (const invalid of ["auto", "fr-CA", "bogus", ""]) await act(async () => { source.emit("config", JSON.stringify({ language: invalid })); await settleStream(); });
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
		Stream.instances = [];
		vi.unstubAllGlobals();
		document.documentElement.lang = "en";
	});

	it("applies a frame to the store and mirrors it onto <html lang>", async () => {
		vi.useFakeTimers(); Stream.instances = [];
		vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(configOk()).mockImplementation(Stream.fetch));
		const { result, unmount } = renderHook(() => useLanguage());
		const close = retainConfigStream();
		try {
			await settleStream(); expect(Stream.instances).toHaveLength(1);
			const source = Stream.instances[0];
			await act(async () => { source.emit("config", JSON.stringify({ changed_keys: ["language"], language: "en" })); await settleStream(); });
			expect(result.current).toBe("en");
			expect(document.documentElement.lang).toBe("en");
		} finally {
			close();
			unmount();
		}
	});

	it("drops a malformed frame without touching the language", async () => {
		vi.useFakeTimers(); Stream.instances = [];
		vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(configOk()).mockImplementation(Stream.fetch));
		const { result, unmount } = renderHook(() => useLanguage());
		const close = retainConfigStream();
		try {
			await settleStream(); expect(Stream.instances).toHaveLength(1);
			const source = Stream.instances[0];
			/* A KNOWN value first — the module store is shared across this file's
			   tests, so "unchanged" is the assertion, not "null". */
			await act(async () => { source.emit("config", JSON.stringify({ changed_keys: [], language: "en" })); await settleStream(); });
			await act(async () => { source.emit("config", "not json"); await settleStream(); });
			expect(result.current).toBe("en");
		} finally {
			close();
			unmount();
		}
	});

	it("treats an older daemon's 404 as ABSENT: probed ONCE, no subscription, no retries", async () => {
		vi.useFakeTimers();
		vi.useFakeTimers(); Stream.instances = [];
		const fetchMock = vi.fn().mockResolvedValue(new Response("not found", { status: 404 }));
		vi.stubGlobal("fetch", fetchMock);
		const close = retainConfigStream();
		try {
			await vi.advanceTimersByTimeAsync(60_000);
			/* Exactly ONE probe (CI round 1, MINOR-2): a re-probing regression
			   must fail here, not just the "no subscription" half. */
			expect(fetchMock).toHaveBeenCalledTimes(1);
			expect(Stream.instances).toHaveLength(0);
		} finally {
			close();
		}
	});

	it("self-heals a transient 5xx: retry, then subscribe — only 404 is absence", async () => {
		vi.useFakeTimers();
		vi.useFakeTimers(); Stream.instances = [];
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(new Response("boom", { status: 503 }))
			.mockResolvedValueOnce(configOk())
			.mockImplementation(Stream.fetch);
		vi.stubGlobal("fetch", fetchMock);
		const close = retainConfigStream();
		try {
			await vi.advanceTimersByTimeAsync(1200);
			expect(fetchMock).toHaveBeenCalledTimes(3);
			expect(Stream.instances).toHaveLength(1);
		} finally {
			close();
		}
	});

	it("retries a transient probe failure with backoff, then subscribes", async () => {
		vi.useFakeTimers();
		vi.useFakeTimers(); Stream.instances = [];
		const fetchMock = vi
			.fn()
			.mockRejectedValueOnce(new Error("offline"))
			.mockResolvedValueOnce(configOk())
			.mockImplementation(Stream.fetch);
		vi.stubGlobal("fetch", fetchMock);
		const close = retainConfigStream();
		try {
			await vi.advanceTimersByTimeAsync(1200);
			expect(fetchMock).toHaveBeenCalledTimes(3);
			expect(Stream.instances).toHaveLength(1);
		} finally {
			close();
		}
	});
});

// An auth failure during capability negotiation is not a transient lack of
// config-stream support. Concurrent probes/streams share one page-wide reload.
it("config probe 401 clears private state once and stops before streaming", async () => {
	vi.useFakeTimers(); vi.resetModules();
	const fresh = await import("./store");
	const { handleUnauthorized } = await import("./api");
	const reload = vi.fn(); vi.stubGlobal("location", { reload });
	localStorage.setItem("lo-mobile-draft:expired", "private");
	localStorage.setItem("local-operator-models", "keep");
	const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 401 }));
	vi.stubGlobal("fetch", fetchMock);
	const release = fresh.retainConfigStream();
	try {
		await vi.advanceTimersByTimeAsync(60000);
		handleUnauthorized(); // another concurrent request was already in flight
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(reload).toHaveBeenCalledTimes(1);
		expect(localStorage.getItem("lo-mobile-draft:expired")).toBeNull();
		expect(localStorage.getItem("local-operator-models")).toBe("keep");
	} finally {
		release(); vi.useRealTimers(); vi.unstubAllGlobals();
	}
});
