import { afterEach, describe, expect, it, vi } from "vitest";
import { openEventStream } from "./store";

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
