// The status dot's state map (operator request: replace the ended banner with a
// header dot). Pure, so EVERY combination of the runtime facts the relay
// publishes is enumerated rather than sampled: `ended`, `degraded`,
// `heartbeat_stale` (true / false / absent), `pid` (live owner vs disk rebuild),
// `streaming`, and the phone's own link. A missed cell is how a "busy" session
// once read as broken; enumerating them is the guard.
import { describe, expect, it } from "vitest";
import { sessionHealth, type SessionHealth } from "./lib/session-health";
import { healthDetailKey } from "./components/session-health";
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
    the SPEC (first match wins): ended → the phone's link → degraded/stale → no
    live owner → live. Streaming never appears: busy is green. */
function expected(f: Facts): SessionHealth {
	if (f.ended) return "ended";
	if (!f.connected) return "reconnecting";
	if (f.degraded || f.heartbeat_stale === true) return "not_answering";
	if (f.pid === 0) return "ended";
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

/* EVERY PRECEDENCE PAIR, named (#3008 review round 1 + the copy sign-off's
   order: ended > the phone's link > degraded > stale > no live owner > live).
   The exhaustive table above proves the order; these say which fact wins each
   head-to-head, so a reorder fails on a sentence rather than on a cell dump. */
describe("sessionHealth — each precedence pair", () => {
	const owner = { ended: false, degraded: false, pid: 4242 };
	it.each<[string, Parameters<typeof sessionHealth>[0], boolean, SessionHealth]>([
		["ended beats the phone's link", { ...owner, ended: true }, false, "ended"],
		["ended beats degraded", { ...owner, ended: true, degraded: true }, true, "ended"],
		["ended beats a stale beat", { ...owner, ended: true, heartbeat_stale: true }, true, "ended"],
		["the phone's link beats degraded", { ...owner, degraded: true }, false, "reconnecting"],
		["the phone's link beats a stale beat", { ...owner, heartbeat_stale: true }, false, "reconnecting"],
		["the phone's link beats a disk rebuild", { ...owner, pid: 0 }, false, "reconnecting"],
		["degraded beats a disk rebuild (an older daemon's live-but-down entry)", { ...owner, degraded: true, pid: 0 }, true, "not_answering"],
		["a stale beat beats a disk rebuild", { ...owner, heartbeat_stale: true, pid: 0 }, true, "not_answering"],
		["a disk rebuild beats live", { ...owner, pid: 0 }, true, "ended"],
	])("%s", (_name, input, connected, want) => {
		expect(sessionHealth(input, connected)).toBe(want);
	});

	/* THE ONE PAIR THE STATE WORD CANNOT TELL APART (review round 2, R2-3).
	   Degraded and a stale beat both read "Not answering", so asserting the
	   state passes whichever wins. What differs is the sheet's sentence — a
	   send fails at once (409) vs. may still arrive (504) — so the pair is
	   asserted on the catalogue KEY the sheet renders, against both
	   single-cause controls. */
	it("degraded beats a stale beat: the sheet says dropped, not quiet", () => {
		const both = { ...owner, degraded: true, heartbeat_stale: true };
		const dropped = { ...owner, degraded: true };
		const quiet = { ...owner, heartbeat_stale: true };
		expect(healthDetailKey(sessionHealth(dropped, true), dropped)).toBe("relay.health.dropped_detail");
		expect(healthDetailKey(sessionHealth(quiet, true), quiet)).toBe("relay.health.quiet_detail");
		expect(healthDetailKey(sessionHealth(both, true), both)).toBe("relay.health.dropped_detail");
	});
});
