/**
 * The session's health, as the header's status dot reports it — a pure
 * function of runtime facts the relay publishes, so the dot, its words and the
 * subtitle beside it share ONE ladder (`components/session-health.tsx`,
 * `components/session-subtitle.tsx`).
 */
import type { SessionProjection } from "../types";

/** The four readings the dot can give. Three colours: `not_answering` and
    `reconnecting` share yellow because both mean "a message sent now will not
    reach a running session right away"; they differ in WHOSE side is quiet, so
    they keep separate words. */
export type SessionHealth = "live" | "not_answering" | "reconnecting" | "ended";

/** The fields {@link sessionHealth} reads — named so a caller (and a test) can
    see the whole input, and so no other projection field can creep in. */
export type HealthInput = Pick<SessionProjection, "ended" | "degraded" | "pid" | "heartbeat_stale">;

/**
 * THE RULE, first match wins. Every input is a runtime fact the relay daemon
 * publishes (`local_operator/mobile/types.py::SessionProjection`):
 *
 * 1. GREY `ended` — `ended`: the relay's scan proved the pid gone (history is
 *    kept and resumable). A receipt, not a reading: it stays true however old
 *    the frame is, so it outranks the phone's own link.
 * 2. YELLOW `reconnecting` — the phone's own link to the relay is down
 *    (`connected` false). Everything below is a reading taken from the LAST
 *    frame, which may be stale by now, so neither yellow cause nor a green can
 *    be confirmed; the honest thing to say is whose side is down (#3008 review,
 *    N2/U6 and the copy sign-off).
 * 3. YELLOW `not_answering` — `degraded` (the relay cannot reach the owner's
 *    control socket, or its scan classified the record `wedged`), OR
 *    `heartbeat_stale === true` (the owner's heartbeat AND its frames have both
 *    been quiet past the relay's one health window, `SESSION_HEALTH_WINDOW_S`).
 *    `degraded` wins between the two (the sheet says "a send fails straight
 *    away", see `healthDetail`): with the dial down the send meets the 409
 *    first, whatever the heartbeat says. `heartbeat_stale` `null`/absent is
 *    "not reported" — an older daemon, or a disk rebuild — and contributes
 *    NOTHING: guessing stale from absence would turn every session on an older
 *    relay yellow. Checked BEFORE the pid rule below because an older daemon
 *    serves a live entry whose dial is down a disk rebuild (pid 0) carrying
 *    `degraded`, and that runtime is alive, not ended.
 * 4. GREY `ended` — `pid === 0`: a frame folded from disk with no running owner
 *    behind it (`daemon._durable_projection` stamps 0; every live frame —
 *    including a live session's first-paint rebuild since #3008 — carries its
 *    owner's pid, the same liveness rule `lib/supplement.ts` uses). A past
 *    conversation opened from history, or one a send is still waking, is not
 *    running yet, so it must not read as live; its first live frame turns the
 *    dot green.
 * 5. GREEN `live` — a running owner the relay has heard from inside the window.
 *
 * BUSY IS GREEN. A streaming turn is alive and answering — it pushes frames
 * many times a second, which is exactly what keeps `heartbeat_stale` false even
 * while the turn starves the runtime's heartbeat writer. `streaming` and
 * `activity` are therefore deliberately NOT inputs: "working" is the
 * subtitle's word, not a health state.
 *
 * The ladder's order matches the subtitle's (`SessionSubtitle`), which reads
 * this function for its first rungs, so the dot and the words beside it cannot
 * disagree.
 */
export function sessionHealth(projection: HealthInput, connected: boolean): SessionHealth {
	if (projection.ended) return "ended";
	if (!connected) return "reconnecting";
	if (projection.degraded || projection.heartbeat_stale === true) return "not_answering";
	if (!projection.pid) return "ended";
	return "live";
}
