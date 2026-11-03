/* Submit one pairing code to the local daemon — shared by the popup and the
 * pairing tab (pair/pair.html).
 *
 * WHY IT IS SHARED. The popup was the only place a code could be typed, and a
 * Chrome action popup closes the moment focus leaves it — which is exactly what
 * a person does to read a code in another window (operator report,
 * 2026-11-02). The pairing tab is a normal extension page that survives focus
 * loss; it must submit a code exactly the way the popup does, so the socket
 * dance lives here once instead of twice.
 *
 * THE DANCE, unchanged from the popup's original: open the extension socket,
 * say `hello` with this install's stored token (empty when unpaired), wait for
 * the ack, send `{event: "pair", code}`, read one `pair_result`. The daemon
 * identifies the install from the socket's Origin (chrome-extension://<id>), so
 * the code is checked against THIS install's code and nobody else's.
 *
 * ONE shared rejection is wired to error AND close for every await: a socket
 * that drops after open but before `pair_result` must land in the failure
 * branch — a bare onmessage promise would suspend forever and leave the form
 * disabled (review finding M1 on the popup). */
import { PROTO_VERSION } from "../protocol.gen";
import { pairVerdict } from "../popup/pair-flow";

export type PairSubmitOutcome =
  | { ok: true; token: string }
  /** The daemon answered and refused the code; `message` says what to do. */
  | { ok: false; kind: "rejected"; message: string }
  /** The daemon answered the hello, then closed before a verdict. */
  | { ok: false; kind: "closed" }
  /** Nothing answered on the port. */
  | { ok: false; kind: "unreachable" };

export interface PairSubmitInput {
  code: string;
  port: number;
  token?: string | undefined;
}

export async function submitPairingCode(input: PairSubmitInput): Promise<PairSubmitOutcome> {
  let answered = false;
  let wire: WebSocket | undefined;
  try {
    wire = new WebSocket(`ws://127.0.0.1:${input.port}/extension`);
    const socket = wire;
    let failed: (reason: Error) => void = () => {};
    const failure = new Promise<never>((_, reject) => {
      failed = reject;
    });
    void failure.catch(() => {});
    socket.onerror = () => failed(new Error("socket error"));
    socket.onclose = () => failed(new Error("socket closed"));
    await Promise.race([
      new Promise((resolve) => {
        socket.onopen = resolve;
      }),
      failure,
    ]);
    const nextMessage = (): Promise<MessageEvent> =>
      Promise.race([
        new Promise<MessageEvent>((resolve) => {
          socket.onmessage = resolve;
        }),
        failure,
      ]);
    // Frames the daemon may push UNASKED on this socket (it becomes this
    // identity's link while open): a `show_pairing` nudge from `lop browser
    // pair`, a `role` change, a liveness `ping`. Skipped, so none is read as
    // the ack or the verdict. Anything else is the answer — the ack step accepts
    // whatever the daemon replies to `hello` (as the popup always has), and the
    // verdict step waits for `pair_result` by name.
    const UNSOLICITED = new Set(["show_pairing", "role", "ping"]);
    const nextAnswer = async (name?: string): Promise<Record<string, unknown>> => {
      for (;;) {
        const frame = JSON.parse(String((await nextMessage()).data)) as Record<string, unknown>;
        if (UNSOLICITED.has(String(frame.event))) continue;
        if (name === undefined || frame.event === name) return frame;
      }
    };
    socket.send(
      JSON.stringify({
        event: "hello",
        proto: PROTO_VERSION,
        token: input.token ?? "",
        extension_version: chrome.runtime.getManifest().version,
        browser: navigator.userAgent,
      }),
    );
    await nextAnswer();
    answered = true;
    socket.send(JSON.stringify({ event: "pair", code: input.code.trim() }));
    const outcome = pairVerdict(await nextAnswer("pair_result"));
    if (outcome.ok) return outcome;
    return { ok: false, kind: "rejected", message: outcome.message };
  } catch {
    return answered ? { ok: false, kind: "closed" } : { ok: false, kind: "unreachable" };
  } finally {
    if (wire) {
      // Detach before closing: this close is deliberate, not a failure.
      wire.onerror = null;
      wire.onclose = null;
      try {
        wire.close();
      } catch {
        // Already closed.
      }
    }
  }
}

/** Keep a code field to digits without moving the caret (finding U6 on the
 * popup). Only reassigns when sanitising actually changed the string, and puts
 * the caret back at the same edit point, minus the characters stripped ahead of
 * it — assigning `.value` otherwise collapses the selection to the end. */
export function digitsOnly(field: HTMLInputElement): void {
  const raw = field.value;
  const clean = raw.replace(/\D/g, "");
  if (clean === raw) return;
  const caret = field.selectionStart ?? raw.length;
  const strippedBeforeCaret = raw.slice(0, caret).length - raw.slice(0, caret).replace(/\D/g, "").length;
  field.value = clean;
  const next = Math.max(0, caret - strippedBeforeCaret);
  field.setSelectionRange(next, next);
}
