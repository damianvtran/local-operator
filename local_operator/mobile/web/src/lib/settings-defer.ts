/* The deferred-ack contract for settings switches (`set_model`/`set_effort`).

   WHY. A switch used to be answered only once it landed, and landing waits for
   the session's loop, so a pick made mid-tool or mid-eval hung for the whole
   step. With `defer_ok` the runtime answers within its inline window
   (`server.SETTINGS_INLINE_ACK_S`): the real outcome when it landed, else
   `detail: "queued"` — accepted, applied at the next step. Kept out of
   `api.ts` so tests that mock the transport keep these real. */

/** The `detail` of an accepted-but-not-yet-applied switch. Success, never an
    error. Mirrors `attach_client.QUEUED_DETAIL`. */
export const QUEUED_DETAIL = "queued";

/** The optional deferred-ack keys. One `request_id` per pick lets the
    runtime's `settings_result` be matched to this request. */
export function deferredSettingsFields(): { defer_ok: true; request_id: string } {
	return { defer_ok: true, request_id: crypto.randomUUID() };
}
