// The status dot's state map (operator request: replace the ended banner with a
// header dot). Pure, so EVERY combination of the runtime facts the relay
// publishes is enumerated rather than sampled: `ended`, `degraded`,
// `heartbeat_stale` (true / false / absent), `pid` (live owner vs disk rebuild),
// `streaming`, and the phone's own link. A missed cell is how a "busy" session
// once read as broken; enumerating them is the guard.
import { describe, expect, it } from "vitest";
import { sessionHealth, type SessionHealth } from "./lib/session-health";
import type { SessionProjection } from "./types";

type Facts = {
	ended: boolean;
	degraded: boolean;
	heartbeat_stale: boolean | null | undefined;
	pid: number;
	streaming: boolean;
	connected: boolean;
};

/** The rule, restated independently of the implementation so the table checks
    the SPEC (first match wins): ended → degraded/stale → no live owner → the
    phone's link → live. Streaming never appears: busy is green. */
function expected(f: Facts): SessionHealth {
	if (f.ended) return "ended";
	if (f.degraded || f.heartbeat_stale === true) return "not_answering";
	if (f.pid === 0) return "ended";
	if (!f.connected) return "reconnecting";
	return "live";
}

const cells: Facts[] = [];
for (const ended of [false, true])
	for (const degraded of [false, true])
		for (const heartbeat_stale of [undefined, null, false, true])
			for (const pid of [0, 4242])
				for (const streaming of [false, true])
					for (const connected of [true, false])
						cells.push({ ended, degraded, heartbeat_stale, pid, streaming, connected });

describe("sessionHealth — every combination of the runtime facts", () => {
	it.each(cells)("%o", (f) => {
		const input = {
			ended: f.ended,
			degraded: f.degraded,
			pid: f.pid,
			streaming: f.streaming,
			...(f.heartbeat_stale === undefined ? {} : { heartbeat_stale: f.heartbeat_stale }),
		} as Pick<SessionProjection, "ended" | "degraded" | "pid" | "heartbeat_stale">;
		expect(sessionHealth(input, f.connected)).toBe(expected(f));
	});

	it("covers the whole space (2·2·4·2·2·2 cells)", () => {
		expect(cells).toHaveLength(128);
	});

	it("reads an ABSENT or null heartbeat verdict as no evidence, never as stale", () => {
		// An older daemon omits the key; a disk rebuild carries none. Both must
		// leave a live owner green — guessing stale would turn every session on
		// an older relay yellow.
		const live = { ended: false, degraded: false, pid: 7 };
		expect(sessionHealth(live, true)).toBe("live");
		expect(sessionHealth({ ...live, heartbeat_stale: null }, true)).toBe("live");
		expect(sessionHealth({ ...live, heartbeat_stale: false }, true)).toBe("live");
		expect(sessionHealth({ ...live, heartbeat_stale: true }, true)).toBe("not_answering");
	});

	it("keeps a degraded disk rebuild yellow — the runtime is alive, its dial is down", () => {
		// The SSE route serves a live-but-payload-less entry a disk rebuild
		// (pid 0) carrying `degraded`; that owner exists, so it is not ended.
		expect(sessionHealth({ ended: false, degraded: true, pid: 0 }, true)).toBe("not_answering");
	});
});
