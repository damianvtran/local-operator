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
import { retainConfigStream, useLanguage } from "./store";

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
