import { CHROME_API_DEADLINE_MS, deadline } from "./driver/deadline";
// From the host-free driver module rather than from `cdp.ts`, which re-exports
// it: `cdp.ts` registers a `chrome.debugger` listener at module scope, and this
// module is imported by the OPTIONS PAGE too — a page that must not acquire a
// debugger dependency (or a listener it can never use) to read two settings.
import { BridgeCommandError } from "./driver/errors";
import { CAPABILITY_SWITCHES } from "./protocol.gen";

/* The two file-transfer capabilities, and the consent that gates each of them.
 *
 * WHY THIS EXISTS AT ALL. Downloads and uploads move files across the boundary
 * between the user's machine and a page. The operator's decision of record is
 * that BOTH are OFF until a human turns them on in the extension's own options:
 *
 *   * `download` also needs a Chrome permission (`downloads`), which is
 *     REQUESTED from the options page at the moment the switch is turned on —
 *     never declared at install/update time, because an update that silently
 *     widens what an installed extension may do is exactly the thing users are
 *     entitled to refuse.
 *   * `upload` needs no permission (it rides the `debugger` grant the extension
 *     already holds), so its switch is the ONLY control — which is why it exists
 *     even though no new capability had to be granted for it.
 *
 * THREE RULES ARE STRUCTURAL, not stylistic:
 *
 *   1. THE OPERATOR IS THE ONLY WRITER. `write` below is called from the
 *      options page on a human gesture and from nowhere else: no daemon frame,
 *      no tool call, no page and no pairing path can reach it. The typescript
 *      signature takes no caller identity because there is no caller to trust —
 *      the module is simply never imported by the worker's command path (the
 *      worker only READS, and reads are what the capability advertisement is
 *      built from).
 *   2. THE STORED FLAG IS NOT THE TRUTH. A flag can outlive the permission that
 *      made it meaningful (the user revokes `downloads` in chrome://extensions,
 *      or an enterprise policy removes it), and a switch that READS ON while the
 *      API is unavailable is the failure mode this module exists to design out.
 *      So the effective answer is always `flag AND permission`, recomputed on
 *      every read — never cached, because the permission can change while the
 *      worker is alive.
 *   3. A REFUSAL NAMES THE SWITCH AND WHERE IT LIVES. A capability that is off
 *      is not a broken build: the model must be able to tell the user which
 *      control to change, in the same words the options page shows them.
 */

/** The storage key of each switch, one key per capability.
 *
 * Spelled out rather than derived from the method name so that a key is greppable
 * from the options page and from the popup without a lookup, and so that renaming
 * a method (a wire change) cannot silently orphan a user's stored consent.
 */
export const CONSENT_KEYS: Record<string, string> = {
  download: "allowDownloads",
  upload: "allowUploads",
};

/** The ONE spelling of where the switches live, for every refusal that points at
 *  them. It names Chrome's own extension page rather than a
 *  `chrome-extension://<id>/options/options.html` URL because the id differs
 *  between an unpacked development install and the store build, and a refusal
 *  that links to the wrong install's id is worse than one that names the path a
 *  user can follow for either. */
export const SWITCH_LOCATION =
  "the Local Operator extension's options page (chrome://extensions -> Details -> Extension options)";

/** The Chrome permission a switch must hold, or `""` when it needs none.
 *
 * Read from the GENERATED table, which Python emits from its own constants: the
 * harness's refusal copy and this module must agree about which capability needs
 * a permission, or the user is told to grant something nothing uses. */
export function switchPermission(method: string): string {
  return CAPABILITY_SWITCHES[method]?.permission ?? "";
}

/** The operator-facing label of a switch, from the same generated table. */
export function switchLabel(method: string): string {
  return CAPABILITY_SWITCHES[method]?.label ?? method;
}

/** Whether this build has a switch for a method at all.
 *
 * False means "no consent exists for this capability on this build", which is
 * NOT the same as "switched off": a method with no switch is as available as its
 * advertisement says, and only the harness's own version arithmetic may call such
 * a method unavailable.
 */
export function hasConsentSwitch(method: string): boolean {
  return method in CONSENT_KEYS;
}

/** Every method this build gates behind an operator switch, sorted. */
export function consentGatedMethods(): string[] {
  return Object.keys(CONSENT_KEYS).sort();
}

/** The stored flag alone — never the answer to "may this run". */
export async function storedSwitch(method: string): Promise<boolean> {
  const key = CONSENT_KEYS[method];
  if (!key) return false;
  const stored = await deadline(
    chrome.storage.local.get([key]),
    CHROME_API_DEADLINE_MS,
    `chrome.storage.local.get(${key})`,
  );
  return stored?.[key] === true;
}

/** Persist a switch. THE OPTIONS PAGE'S CALL, from a human gesture, and nothing
 *  else's: see rule 1 in this module's header. */
export async function writeSwitch(method: string, on: boolean): Promise<void> {
  const key = CONSENT_KEYS[method];
  if (!key) throw new Error(`no consent switch for ${method}`);
  await deadline(
    chrome.storage.local.set({ [key]: on }),
    CHROME_API_DEADLINE_MS,
    `chrome.storage.local.set(${key})`,
  );
}

/** Whether Chrome currently holds a permission.
 *
 * Conservative in the direction that matters: an unavailable `chrome.permissions`
 * API (a test host, a future API surface) reports NOT held, because the only
 * claim this function may never make is that a capability we cannot verify is
 * available. The consequence of the conservative answer is a switch that reads
 * off — a state the user can see and fix — rather than one that reads on while
 * the call would fail with Chrome's own error.
 */
export async function permissionHeld(permission: string): Promise<boolean> {
  if (!permission) return true;
  const api = chrome.permissions;
  if (!api?.contains) return false;
  try {
    return await deadline(
      // The cast is @types/chrome's: the permission names are validated against
      // Chrome's own union, while the string here comes from the GENERATED table
      // whose values are pinned by the Python test that emits it. A name that is
      // not a real permission fails the request, which reads as "not held" and
      // leaves the switch off — the conservative direction.
      api.contains({ permissions: [permission as chrome.runtime.ManifestPermissions] }),
      CHROME_API_DEADLINE_MS,
      `chrome.permissions.contains(${permission})`,
    );
  } catch {
    // A thrown or timed-out check is not evidence of a grant. Same direction as
    // the missing-API branch: unknown reads as "not held".
    return false;
  }
}

/** Whether a capability may actually run right now: flag AND permission.
 *
 * Recomputed per read (rule 2) — there is deliberately no cache, because the
 * permission can be revoked in another window while this worker stays alive.
 */
export async function capabilityEnabled(method: string): Promise<boolean> {
  if (!hasConsentSwitch(method)) return true;
  if (!(await storedSwitch(method))) return false;
  return permissionHeld(switchPermission(method));
}

/** The methods this build serves but the operator has switched off.
 *
 * What travels in the `capability_switches` event, and the reason it is a list of
 * its own rather than an absence from `capabilities`: the harness needs to tell
 * "this build cannot" (update it) from "the operator has not enabled this"
 * (point at the switch). A method missing its permission counts as disabled, so a
 * revoked grant is reported as a consent that must be restored rather than as a
 * capability that vanished.
 */
export async function disabledCapabilities(): Promise<string[]> {
  const gated = consentGatedMethods();
  const states = await Promise.all(gated.map((method) => capabilityEnabled(method)));
  return gated.filter((_, index) => !states[index]);
}

/** The ONE sentence a switched-off capability is refused with, at execution.
 *
 * Carried here rather than at each calling site so the worker's refusal, the
 * popup's notice and the harness's own copy name the same switch in the same
 * words. The closing clause exists because "not enabled" reads like a build
 * limitation unless it is said to be a setting the user can change.
 */
export function consentOffMessage(method: string): string {
  const label = switchLabel(method);
  const permission = switchPermission(method);
  const grant = permission
    ? ` Turning it on asks Chrome for the '${permission}' permission; if you refuse that, the switch stays off.`
    : "";
  return (
    `'${method}' is switched off: the operator has not turned on "${label}" in ${SWITCH_LOCATION}.` +
    `${grant} Ask the user to turn it on there, then retry — this is a setting, not a build limitation.`
  );
}

/** Throw unless the operator has enabled this capability.
 *
 * DEFENCE IN DEPTH, and it is not redundant with the advertisement. The daemon
 * refuses to SEND a method the extension did not advertise (design §6.3), but a
 * daemon that predates the advertisement sends anyway, and a switch can be turned
 * off between the advertisement and the command arriving — the one window in
 * which "off" must still mean off. So the handler itself is the last gate, and it
 * is here rather than in each command so no future handler can forget it.
 */
export async function requireConsent(method: string): Promise<void> {
  if (await capabilityEnabled(method)) return;
  throw new BridgeCommandError("capability_unsupported", consentOffMessage(method));
}
