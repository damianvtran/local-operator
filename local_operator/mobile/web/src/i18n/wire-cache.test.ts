import { describe, expect, it, vi } from "vitest";
import { createWireLoader } from "./wire-cache";
import { createMessageRuntime } from "./runtime";
import type { Capabilities } from "../types";
const hash = "a".repeat(64);
const etag = `"${hash}"`;
const success = (namespace = "wire.exec", revision = hash) => new Response(JSON.stringify({ status: 200, message: "Catalogue retrieved.", result: { content_sha256: revision, messages: { [`${namespace}.fixture`]: "Fixture text" } } }), { status: 200, headers: { ETag: `"${revision}"` } });
function rig(fetcher = vi.fn<typeof fetch>()) {
	const context: { origin: string; locale: string | null } = { origin: "https://relay.test", locale: "fr" };
	let caps: Capabilities | null = { features: { i18n: 1 }, i18n: { catalogues: 1 } };
	const activate = vi.fn();
	const loader = createWireLoader({ context: () => context, capabilities: () => caps, activate, fetch: fetcher, maxEntries: 2 });
	return { context, loader, fetcher, activate, caps: (value: Capabilities | null) => { caps = value; } };
}

describe("explicit wire loader", () => {
	for (const status of [200, 304]) for (const withdrawn of [null, { features: { i18n: 0 }, i18n: { catalogues: 1 } }, { features: { i18n: 1 }, i18n: { catalogues: 0 } }]) {
		it(`withdrawal ${JSON.stringify(withdrawn)} refuses delayed ${status} and reenables cold`, async () => {
			const r = rig();
			if (status === 304) { r.fetcher.mockResolvedValueOnce(success()); expect(await r.loader.request("wire.exec")).toBe(true); }
			const previous = r.activate.mock.calls.length;
			let finish!: (response: Response) => void;
			r.fetcher.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
			const promise = r.loader.request("wire.exec");
			r.caps(withdrawn);
			finish(status === 200 ? success() : new Response(null, { status: 304, headers: { ETag: etag } }));
			expect(await promise).toBe(false);
			expect(r.loader.size()).toBe(0); expect(r.activate).toHaveBeenCalledTimes(previous);
			r.caps({ features: { i18n: 1 }, i18n: { catalogues: 1 } }); r.fetcher.mockResolvedValueOnce(success());
			expect(await r.loader.request("wire.exec")).toBe(true);
			expect(r.fetcher.mock.calls.at(-1)?.[1]?.headers).toEqual({});
		});
	}
	it("withdrawal removes active served copy and aborts even if reenabled before response", async () => {
		let caps: Capabilities | null = { features: { i18n: 1 }, i18n: { catalogues: 1 } };
		const context = { backendLocale: "fr", uiLocale: "en", direction: "ltr" as const, revision: 0 };
		const runtime = createMessageRuntime(() => context, () => { context.revision++; });
		const fetcher = vi.fn<typeof fetch>().mockResolvedValueOnce(success());
		const loader = createWireLoader({ context: () => ({ origin: "https://relay.test", locale: context.backendLocale }), capabilities: () => caps, fetch: fetcher,
			activate: (namespace, locale, messages) => { runtime.registerWire(namespace, locale, messages); },
			deactivate: (namespace, locale) => { runtime.unregisterWire(namespace, locale); },
		});
		expect(await loader.request("wire.exec")).toBe(true);
		expect(runtime.renderEnvelope("wire.exec.fixture", {}, "Exact fallback").locale).toBe("fr");
		let finish!: (response: Response) => void;
		fetcher.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
		const delayed = loader.request("wire.exec");
		caps = null; expect(loader.invalidateCapabilities()).toBe(false);
		expect(loader.size()).toBe(0);
		expect(runtime.renderEnvelope("wire.exec.fixture", {}, "Exact fallback")).toEqual({ text: "Exact fallback", locale: null, direction: null });
		caps = { features: { i18n: 1 }, i18n: { catalogues: 1 } };
		finish(success()); expect(await delayed).toBe(false);
		expect(runtime.renderEnvelope("wire.exec.fixture", {}, "Exact fallback").locale).toBe(null);
		fetcher.mockResolvedValueOnce(success()); expect(await loader.request("wire.exec")).toBe(true);
		expect(fetcher.mock.calls.at(-1)?.[1]?.headers).toEqual({});
	});
	it("never fetches without both capabilities, valid backend locale and an explicitly requested wire namespace", async () => {
		const r = rig();
		for (const caps of [null, { features: { i18n: 1 } }, { i18n: { catalogues: 1 } }]) {
			r.caps(caps); expect(await r.loader.request("wire.exec")).toBe(false);
		}
		r.caps({ features: { i18n: 1 }, i18n: { catalogues: 1 } });
		for (const name of ["relay.shell", "wire", "wire.exec/../x"]) expect(await r.loader.request(name)).toBe(false);
		r.context.locale = null; expect(await r.loader.request("wire.exec")).toBe(false);
		expect(r.fetcher).not.toHaveBeenCalled();
	});
	it("singleflights, validates opaque revision/ETag and revalidates the exact cached identity", async () => {
		const r = rig(); r.fetcher.mockResolvedValueOnce(success()).mockResolvedValueOnce(new Response(null, { status: 304, headers: { ETag: etag } }));
		expect(await Promise.all([r.loader.request("wire.exec"), r.loader.request("wire.exec")])).toEqual([true, true]);
		expect(r.fetcher).toHaveBeenCalledTimes(1);
		expect(await r.loader.request("wire.exec")).toBe(true);
		expect(r.fetcher.mock.calls[1][1]?.headers).toEqual({ "If-None-Match": etag });
		r.context.origin = "https://different.test"; r.fetcher.mockResolvedValueOnce(success());
		expect(await r.loader.request("wire.exec")).toBe(true);
		expect(r.fetcher.mock.calls[2][1]?.headers).toEqual({});
	});
	it("retries an unexpected 304 exactly once without a validator", async () => {
		const r = rig(); r.fetcher.mockResolvedValueOnce(new Response(null, { status: 304 })).mockResolvedValueOnce(success());
		expect(await r.loader.request("wire.exec")).toBe(true); expect(r.fetcher).toHaveBeenCalledTimes(2);
		const failed = rig(); failed.fetcher.mockResolvedValue(new Response(null, { status: 304 }));
		expect(await failed.loader.request("wire.exec")).toBe(false); expect(failed.fetcher).toHaveBeenCalledTimes(2);
	});
	it("does not activate a stale locale even when a fetch ignores abort", async () => {
		let finish!: (response: Response) => void;
		const fetcher = vi.fn<typeof fetch>().mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
		const r = rig(fetcher); const promise = r.loader.request("wire.exec");
		r.context.locale = "en"; r.loader.invalidateContext();
		expect((fetcher.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
		finish(success()); expect(await promise).toBe(false); expect(r.activate).not.toHaveBeenCalled();
	});
	it("fails closed on schema, namespace, hash or ETag mismatch, and does not loop on 404/offline", async () => {
		const invalid = [
			new Response("{}", { status: 200 }),
			success("wire.other"),
			new Response(JSON.stringify({ status: 200, message: "ok", result: { messages: { "wire.exec.fixture": "text" }, content_sha256: "bad" } }), { headers: { ETag: etag } }),
			new Response(JSON.stringify({ status: 200, message: "ok", result: { messages: { "wire.exec.fixture": "text" }, content_sha256: hash } }), { headers: { ETag: `"${"b".repeat(64)}"` } }),
			new Response(null, { status: 404 }),
		];
		for (const response of invalid) { const r = rig(); r.fetcher.mockResolvedValue(response); expect(await r.loader.request("wire.exec")).toBe(false); expect(r.activate).not.toHaveBeenCalled(); expect(r.fetcher).toHaveBeenCalledTimes(1); }
		const offline = rig(); offline.fetcher.mockRejectedValue(new TypeError("offline")); expect(await offline.loader.request("wire.exec")).toBe(false); expect(offline.fetcher).toHaveBeenCalledTimes(1);
	});
	it("expires a stalled fetch and cancels an oversized body without activating", async () => {
		vi.useFakeTimers();
		try {
			const stalled = rig(); stalled.fetcher.mockImplementation(() => new Promise(() => {}));
			const promise = stalled.loader.request("wire.exec");
			await vi.advanceTimersByTimeAsync(10_001);
			expect(await promise).toBe(false);
			expect((stalled.fetcher.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
			expect(stalled.activate).not.toHaveBeenCalled();
		} finally { vi.useRealTimers(); }
		const cancel = vi.fn();
		const oversized = rig();
		const body = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(1024 * 1024 + 1)); }, cancel });
		oversized.fetcher.mockResolvedValue(new Response(body, { status: 200, headers: { ETag: etag } }));
		expect(await oversized.loader.request("wire.exec")).toBe(false);
		expect(cancel).toHaveBeenCalled(); expect(oversized.activate).not.toHaveBeenCalled();
	});

	it("bounds cache entries without mixing locale or namespace identities", async () => {
		const r = rig();
		for (const namespace of ["wire.exec", "wire.slash", "wire.errors"]) { r.fetcher.mockResolvedValueOnce(success(namespace)); expect(await r.loader.request(namespace)).toBe(true); }
		expect(r.loader.size()).toBe(2);
		r.fetcher.mockResolvedValueOnce(success()); expect(await r.loader.request("wire.exec")).toBe(true);
		expect(r.fetcher.mock.calls[3][1]?.headers).toEqual({});
	});
});
