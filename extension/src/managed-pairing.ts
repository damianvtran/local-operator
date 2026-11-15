/* Pairing from Chrome enterprise policy — the headless-node path.
 *
 * WHY THIS EXISTS. On a headless mesh node (`lop browser setup --node`) the
 * extension runs in a `--headless=new` Chrome that has no screen and no person
 * in front of it. The ordinary pairing ceremony — the CLI prints a code, a
 * person types it into the popup or the pairing tab — cannot happen there:
 * the pairing tab opens where nobody can see it. So `lop browser setup --node`
 * mints the code itself (through the daemon's own `/pairing/codes` route, the
 * one `lop browser pair` calls) and delivers it through Chrome's
 * `3rdparty.extensions.<id>` managed policy, which this module reads.
 *
 * WHY THIS DOES NOT IMPERSONATE A USER, OR WEAKEN THE PROOF.
 *   - The code is still the proof. The daemon never auto-trusts an extension
 *     id; it checks this code against the code it minted for THIS install's
 *     socket Origin, exactly as it does for a typed code.
 *   - The channel is root-written policy (`/etc/opt/chrome/policies/managed`).
 *     `chrome.storage.managed` is readable only by THIS extension (Chrome
 *     scopes `3rdparty.extensions.<id>` to that id); no page and no other
 *     extension can read it, and nothing here ever surfaces the code to a page.
 *   - The party that minted the code is a same-uid `lop` process, which could
 *     already mint a code and read it; nothing new is exposed to anyone.
 *   - Setup removes the key from the policy as soon as `paired: yes`, and a
 *     code expires on the daemon's own TTL regardless.
 *
 * On a user's machine there is no managed policy, so `get` returns nothing
 * and this module does nothing: the person-types-the-code path is unchanged.
 *
 * `chrome.storage.managed` only delivers keys the manifest declares in
 * `storage.managed_schema` (measured on Chrome 157: without a schema the area
 * stays empty), which is why `managed_schema.json` ships with the build. */
import { submitPairingCode, type PairSubmitOutcome } from "./pair/submit";

/** The policy key setup writes. Mirrors `MANAGED_PAIRING_KEY` in
 * `local_operator/browser_bridge/node_host.py`; renaming one without the other
 * silently breaks node pairing. */
export const MANAGED_PAIRING_KEY = "pairingCode";

export interface ManagedPairingDeps {
  /** `chrome.storage.managed`, or undefined where Chrome does not offer it. */
  managed: Pick<chrome.storage.StorageArea, "get"> | undefined;
  /** Reads this install's stored token and daemon port. */
  readLocal: () => Promise<{ token?: string | undefined; port?: number | undefined }>;
  /** Persists the token a successful pair returns (the pairing tab does the same). */
  writeToken: (token: string) => Promise<void>;
  submit?: (input: { code: string; port: number; token?: string | undefined }) => Promise<PairSubmitOutcome>;
  defaultPort: number;
}

export type ManagedPairingResult = "no_policy" | "already_tried" | "busy" | "paired" | "rejected" | "unreachable";

/** A code the daemon ANSWERED (paired or refused) is not submitted again in
 * this worker's lifetime: a refused code stays refused (expired, or not this
 * install's), and retrying it on every alarm tick would only spam the daemon.
 * Setup mints a fresh code and rewrites the policy when it needs another
 * attempt, which changes the value and so re-arms this. A code that never
 * reached the daemon (not up yet, socket dropped) IS retried on the next
 * trigger — the reconnect alarm is that floor. */
let lastAnswered: string | undefined;
let inFlight = false;

/** Test hook: forget the last answered code. */
export function resetManagedPairingForTest(): void {
  lastAnswered = undefined;
  inFlight = false;
}

export async function pairFromManagedPolicy(deps: ManagedPairingDeps): Promise<ManagedPairingResult> {
  if (!deps.managed) return "no_policy";
  let raw: unknown;
  try {
    raw = (await deps.managed.get(MANAGED_PAIRING_KEY))?.[MANAGED_PAIRING_KEY];
  } catch {
    // An area with no schema, or a policy Chrome refused to parse, rejects
    // here. Either way there is no code to act on.
    return "no_policy";
  }
  const code = typeof raw === "string" ? raw.replace(/\D/g, "") : "";
  if (!code) return "no_policy";
  if (code === lastAnswered) return "already_tried";
  // Startup, the policy-change event and the alarm can all fire together; two
  // concurrent submits would evict each other's socket (close 4000).
  if (inFlight) return "busy";
  inFlight = true;
  try {
    const local = await deps.readLocal();
    const port = typeof local.port === "number" && local.port > 0 ? local.port : deps.defaultPort;
    const submit = deps.submit ?? submitPairingCode;
    const outcome = await submit({ code, port, token: local.token });
    if (outcome.ok) {
      lastAnswered = code;
      await deps.writeToken(outcome.token);
      return "paired";
    }
    if (outcome.kind === "rejected") {
      lastAnswered = code;
      return "rejected";
    }
    return "unreachable";
  } finally {
    inFlight = false;
  }
}
