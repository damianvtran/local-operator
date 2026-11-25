// REVIEW ROUND 1, m2 (PR #4584): the composer's `WAKE_SETTLED_BOUND_MS` is
// derived from the relay's `PROMPT_WAKE_DEADLINE_S` (`mobile/daemon.py`), and
// nothing failed when one moved without the other. A bound BELOW the relay's
// deadline would report a still-booting wake as failed to a composer that
// missed the one-shot `wake_failed` frame. The constant is read from the
// daemon's source (vitest has no Python), the same source-read pattern as the
// stylesheet cells.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { WAKE_SETTLED_BOUND_MS } from "./components/composer";

/* The relay's work outside its own deadline: `engage_runtime` takes the
   deadline only after `_engage_and_publish`'s pre-engage durable fold, and
   the failure frame is published after it expires. */
const RELAY_PRE_BOOT_SLACK_MS = 30_000;

describe("the composer's wake bound tracks the relay's deadline (m2)", () => {
	it("stays at or above PROMPT_WAKE_DEADLINE_S plus the pre-boot slack", () => {
		const daemon = readFileSync(join(__dirname, "..", "..", "daemon.py"), "utf8");
		const match = /^PROMPT_WAKE_DEADLINE_S = ([0-9.]+)$/m.exec(daemon);
		expect(match, "PROMPT_WAKE_DEADLINE_S not found in mobile/daemon.py").not.toBeNull();
		const deadlineS = Number.parseFloat(match![1]);
		expect(deadlineS).toBeGreaterThan(0);
		expect(WAKE_SETTLED_BOUND_MS).toBeGreaterThanOrEqual(deadlineS * 1000 + RELAY_PRE_BOOT_SLACK_MS);
	});

	it("is back-referenced at the daemon constant", () => {
		const daemon = readFileSync(join(__dirname, "..", "..", "daemon.py"), "utf8");
		const at = daemon.indexOf("PROMPT_WAKE_DEADLINE_S = ");
		expect(daemon.slice(Math.max(0, at - 800), at)).toContain("WAKE_SETTLED_BOUND_MS");
	});
});
