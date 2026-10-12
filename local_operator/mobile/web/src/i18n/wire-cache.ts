/** Explicit wire delivery only; no bootstrap, startup fetch, storage or locale inference.
 * Lookup identity is origin/locale/namespace; content revision is the server's
 * opaque raw-file SHA, not a client serialization digest. Revalidate every use. */
import type { Capabilities } from "../types";
import { isResolvedLocale } from "./runtime";
interface Context { origin: string; locale: string | null }
interface Entry { hash: string; etag: string; messages: Record<string, string>; bytes: number }
interface Options {
	context: () => Context;
	capabilities: () => Capabilities | null;
	activate: (namespace: string, locale: string, messages: Record<string, string>) => void;
	deactivate?: (namespace: string, locale: string) => void;
	fetch?: typeof fetch;
	maxEntries?: number;
	maxBytes?: number;
}
const MAX_RESPONSE = 1024 * 1024;
const NAMESPACE = /^wire(?:\.[a-z][a-z0-9_]*)+$/;
const HASH = /^[a-f0-9]{64}$/;
const record = (value: unknown): value is Record<string, unknown> =>
	!!value && typeof value === "object" && !Array.isArray(value);
function envelope(value: unknown, namespace: string, etag: string | null): Entry | null {
	if (!record(value) || value.status !== 200 || typeof value.message !== "string" || !record(value.result)) return null;
	const { messages, content_sha256: hash } = value.result;
	if (!record(messages) || typeof hash !== "string" || !HASH.test(hash) || etag !== `"${hash}"`) return null;
	if (!Object.entries(messages).every(([key, text]) => key.startsWith(`${namespace}.`) && typeof text === "string")) return null;
	// Serialization measures weight ONLY; it is never compared to the raw-file hash.
	return { hash, etag, messages: messages as Record<string, string>, bytes: JSON.stringify(messages).length * 2 };
}
async function boundedBody(response: Response): Promise<string> {
	if (!response.body) return "";
	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let size = 0;
	try {
		for (;;) {
			const { value, done } = await reader.read();
			if (done) break;
			size += value.byteLength;
			if (size > MAX_RESPONSE) throw new Error("catalogue response limit");
			chunks.push(value);
		}
	} finally { await reader.cancel(); reader.releaseLock(); }
	const bytes = new Uint8Array(size);
	let offset = 0;
	for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
	return new TextDecoder().decode(bytes);
}

export function createWireLoader(options: Options) {
	const cache = new Map<string, Entry>();
	const active = new Map<string, { namespace: string; locale: string }>();
	const pending = new Map<string, { promise: Promise<boolean>; controller: AbortController }>();
	const fetcher = options.fetch ?? ((...args: Parameters<typeof fetch>) => fetch(...args));
	const limit = Math.max(1, Math.min(options.maxEntries ?? 16, 64));
	const byteLimit = Math.max(1, options.maxBytes ?? 4 * 1024 * 1024);
	let identity = "";
	let generation = 0;
	function invalidateContext(): void {
		const context = options.context();
		const next = JSON.stringify([context.origin, context.locale]);
		if (next === identity) return;
		identity = next;
		generation++;
		for (const item of pending.values()) item.controller.abort();
		pending.clear();
	}
	function invalidateCapabilities(): boolean {
		const caps = options.capabilities();
		if (typeof caps?.features?.i18n === "number" && Number.isInteger(caps.features.i18n) && caps.features.i18n >= 1 &&
			typeof caps.i18n?.catalogues === "number" && Number.isInteger(caps.i18n.catalogues) && caps.i18n.catalogues >= 1) return true;
		// Withdrawal is an authority change even with unchanged locale/origin.
		// Drop owned served maps too: they must not outlive the permission to
		// activate them. Re-enable starts cold, never via an old validator.
		if (cache.size || pending.size || active.size) {
			generation++;
			for (const item of pending.values()) item.controller.abort();
			pending.clear(); cache.clear();
			const withdrawn = [...active.values()]; active.clear();
			for (const item of withdrawn) options.deactivate?.(item.namespace, item.locale);
		}
		return false;
	}
	function put(key: string, entry: Entry): void {
		cache.delete(key);
		if (entry.bytes > byteLimit) return;
		cache.set(key, entry);
		let bytes = [...cache.values()].reduce((sum, item) => sum + item.bytes, 0);
		while (cache.size > limit || bytes > byteLimit) {
			const oldest = cache.keys().next().value!;
			bytes -= cache.get(oldest)!.bytes;
			cache.delete(oldest);
		}
	}
	async function request(namespace: string): Promise<boolean> {
		invalidateContext();
		const context = options.context();
		if (!invalidateCapabilities() || !NAMESPACE.test(namespace) || !isResolvedLocale(context.locale)) return false;
		let origin: string;
		try { const url = new URL(context.origin); if (!["http:", "https:"].includes(url.protocol) || url.origin !== context.origin) return false; origin = url.origin; }
		catch { return false; }
		const locale = context.locale;
		const key = JSON.stringify([origin, locale, namespace]);
		const existing = pending.get(key);
		if (existing) return existing.promise;
		if (pending.size >= limit) return false;
		const epoch = generation;
		const controller = new AbortController();
		const cached = cache.get(key);
		const run = async (): Promise<boolean> => {
			try {
				let validator = cached?.etag;
				for (let attempt = 0; attempt < 2; attempt++) {
					const response = await fetcher(`${origin}/api/i18n/catalogues/${locale}/${namespace}`, {
						signal: controller.signal, headers: validator ? { "If-None-Match": validator } : {},
					});
					invalidateContext();
					if (!invalidateCapabilities() || generation !== epoch || controller.signal.aborted) return false;
					let entry: Entry | null;
					if (response.status === 304) {
						if (!cached || validator !== cached.etag) { validator = void 0; continue; }
						const returned = response.headers.get("etag");
						if (returned !== null && returned !== cached.etag) return false;
						entry = cached;
					} else if (response.status === 200) {
						entry = envelope(JSON.parse(await boundedBody(response)), namespace, response.headers.get("etag"));
					} else return false; // 404/401/offline never create a retry loop
					invalidateContext();
					if (!entry || !invalidateCapabilities() || generation !== epoch || controller.signal.aborted) return false;
					put(key, entry);
					active.delete(key); active.set(key, { namespace, locale });
					if (active.size > limit) {
						const oldest = active.keys().next().value!;
						const removed = active.get(oldest)!; active.delete(oldest);
						options.deactivate?.(removed.namespace, removed.locale);
					}
					if (!invalidateCapabilities() || generation !== epoch || controller.signal.aborted) return false;
					options.activate(namespace, locale, entry.messages);
					return true;
				}
				return false;
			} catch { return false; }
		};
		// A public catalogue lookup must not hold the singleflight slot forever.
		// Race cancellation as well as signalling fetch: even an adapter that
		// ignores AbortSignal cannot activate or pin the caller after expiry.
		let cancelled!: (value: boolean) => void;
		const aborted = new Promise<boolean>(resolve => { cancelled = resolve; });
		const onAbort = () => cancelled(false);
		controller.signal.addEventListener("abort", onAbort, { once: true });
		const timer = setTimeout(() => controller.abort(), 10_000);
		const promise = Promise.race([run(), aborted]).finally(() => {
			clearTimeout(timer);
			controller.signal.removeEventListener("abort", onAbort);
			if (pending.get(key)?.controller === controller) pending.delete(key);
		});
		pending.set(key, { promise, controller });
		return promise;
	}
	return { request, invalidateContext, invalidateCapabilities, size: () => cache.size };
}
