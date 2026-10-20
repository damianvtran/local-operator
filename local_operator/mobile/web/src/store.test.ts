// @vitest-environment happy-dom
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openEventStream, retainSessionListStream, useSessions } from "./store";

class FakeEventSource {
	static instances: FakeEventSource[] = [];
	onopen: (() => void) | null = null;
	onerror: (() => void) | null = null;
	listeners = new Map<string, (event: MessageEvent) => void>();
	closed = false;
	constructor(readonly url: string) { FakeEventSource.instances.push(this); }
	addEventListener(name: string, listener: EventListenerOrEventListenerObject) {
		this.listeners.set(name, listener as (event: MessageEvent) => void);
	}
	close() { this.closed = true; }
}

describe("openEventStream", () => {
	afterEach(() => {
		vi.useRealTimers();
		FakeEventSource.instances = [];
		vi.unstubAllGlobals();
	});

	it("ignores data, open and error callbacks from a retired source", () => {
		vi.useFakeTimers();
		vi.stubGlobal("EventSource", FakeEventSource);
		const data: string[] = [];
		const opened = vi.fn();
		const disconnected = vi.fn();
		const close = openEventStream("/events", "projection", value => data.push(value), opened, disconnected);
		const first = FakeEventSource.instances[0];
		first.onopen?.();
		first.onerror?.();
		vi.advanceTimersByTime(1000);
		const current = FakeEventSource.instances[1];
		current.onopen?.();
		first.listeners.get("projection")?.({ data: "retired" } as MessageEvent);
		first.onopen?.();
		first.onerror?.();
		expect(current.closed).toBe(false);
		expect(opened).toHaveBeenCalledTimes(2);
		expect(disconnected).toHaveBeenCalledTimes(1);
		current.listeners.get("projection")?.({ data: "current" } as MessageEvent);
		expect(data).toEqual(["current"]);
		close();
		current.listeners.get("projection")?.({ data: "after close" } as MessageEvent);
		expect(data).toEqual(["current"]);
	});

	it("reports a post-connect disconnect and clears it on reconnect", () => {
		vi.useFakeTimers();
		vi.stubGlobal("EventSource", FakeEventSource);
		const transitions: string[] = [];
		const close = openEventStream("/events", "projection", vi.fn(), () => transitions.push("open"), () => transitions.push("disconnected"));
		const first = FakeEventSource.instances[0];
		first.onopen?.();
		first.onerror?.();
		expect(first.closed).toBe(true);
		expect(transitions).toEqual(["open", "disconnected"]);
		vi.advanceTimersByTime(1000);
		const second = FakeEventSource.instances[1];
		second.onopen?.();
		expect(transitions).toEqual(["open", "disconnected", "open"]);
		close();
	});
});

describe("the session-list frame revision", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		FakeEventSource.instances = [];
	});

	const frame = (ids: string[]) =>
		JSON.stringify({
			sessions: ids.map((session_id) => ({ session_id, pinned: false })),
		});

	it("counts accepted list frames only, and never a component render", () => {
		vi.stubGlobal("EventSource", FakeEventSource);
		const release = retainSessionListStream();
		try {
			const source = FakeEventSource.instances[0];
			const { result, rerender } = renderHook(() => useSessions());
			expect(result.current.revision).toBe(0);

			/* A render driven by anything else must not move the counter: the
			   revision is a fact about frames, not about React work. */
			rerender();
			expect(result.current.revision).toBe(0);

			act(() => {
				source.listeners.get("sessions")?.({
					data: frame(["a"]),
				} as MessageEvent);
			});
			expect(result.current.revision).toBe(1);
			expect(result.current.sessions.map((row) => row.session_id)).toEqual(["a"]);

			rerender();
			expect(result.current.revision).toBe(1);

			/* A malformed frame is DROPPED before it can replace the list, so it
			   must not advance the counter either. */
			act(() => {
				source.listeners.get("sessions")?.({ data: "{" } as MessageEvent);
			});
			expect(result.current.revision).toBe(1);
			expect(result.current.sessions.map((row) => row.session_id)).toEqual(["a"]);

			act(() => {
				source.listeners.get("sessions")?.({
					data: frame(["a", "b"]),
				} as MessageEvent);
			});
			expect(result.current.revision).toBe(2);
			expect(result.current.sessions.map((row) => row.session_id)).toEqual([
				"a",
				"b",
			]);
		} finally {
			release();
		}
	});
});
