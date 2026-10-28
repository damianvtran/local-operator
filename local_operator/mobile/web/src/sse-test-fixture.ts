import { vi } from "vitest";

/** A real byte stream at fetch's boundary, not a second transport mock. */
export class FetchStreamFixture {
	static instances: FetchStreamFixture[] = [];
	readonly response: Response;
	private sink!: ReadableStreamDefaultController<Uint8Array>;
	closed = false;
	constructor(readonly url: string, signal?: AbortSignal | null) {
		const stream = new ReadableStream<Uint8Array>({
			start: controller => { this.sink = controller; },
			cancel: () => { this.closed = true; },
		});
		this.response = new Response(stream, { headers: { "content-type": "text/event-stream" } });
		signal?.addEventListener("abort", () => {
			if (this.closed) return;
			this.closed = true;
			this.sink.error(new DOMException("Aborted", "AbortError"));
		});
		FetchStreamFixture.instances.push(this);
	}
	chunk(value: string) { if (!this.closed) this.sink.enqueue(new TextEncoder().encode(value)); }
	emit(event: string, data: unknown) {
		this.chunk(`event: ${event}\ndata: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`);
	}
	end() { if (!this.closed) { this.closed = true; this.sink.close(); } }
	fail() { if (!this.closed) { this.closed = true; this.sink.error(new Error()); } }
	static fetch = vi.fn(async (url: string, init?: RequestInit) => new FetchStreamFixture(url, init?.signal).response);
	static install() {
		FetchStreamFixture.instances = [];
		FetchStreamFixture.fetch.mockClear();
		vi.stubGlobal("fetch", FetchStreamFixture.fetch);
	}
}

/** Flush stream microtasks before the next paint, without sleeping in real time. */
export async function settleStream() {
	await vi.advanceTimersByTimeAsync(0);
	await vi.advanceTimersByTimeAsync(40);
}
