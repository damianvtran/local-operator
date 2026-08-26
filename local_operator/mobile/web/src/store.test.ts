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
