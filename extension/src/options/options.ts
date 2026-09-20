import { DEFAULT_PORT, getLocal } from "../state";
import { PROTO_VERSION } from "../protocol.gen";
import {
  capabilityEnabled,
  PERMISSION_MISSING_MESSAGE,
  PERMISSION_REQUEST_DEADLINE_MS,
  PERMISSION_REQUEST_PENDING,
  permissionHeld,
  storedSwitch,
  switchPermission,
  writeSwitch,
} from "../consent";
import { deadline } from "../driver/deadline";
import { allowAllView, nextAllowAllView, type AllowAllAction, type AllowAllView } from "./allow-all-flow";
import { grantRows, removeGrantAccessibleName, revokeMessageFor } from "./grant-list";
import { runWorkerMutation } from "./mutation-flow";

const port = document.getElementById("port") as HTMLInputElement;
const allowAll = document.getElementById("allow-all") as HTMLInputElement;
const allowAllDialog = document.getElementById("allow-all-dialog") as HTMLDialogElement;
const allowAllAck = document.getElementById("allow-all-ack") as HTMLInputElement;
const allowAllEnable = document.getElementById("allow-all-enable") as HTMLButtonElement;
const allowAllBanner = document.getElementById("allow-all-banner") as HTMLDivElement;
const sites = document.getElementById("sites") as HTMLUListElement;
const sitesEmpty = document.getElementById("sites-empty") as HTMLParagraphElement;
const sitesSuperseded = document.getElementById("sites-superseded") as HTMLParagraphElement;
const pairStatus = document.getElementById("pair-status") as HTMLParagraphElement;
const confirm = document.getElementById("confirm") as HTMLParagraphElement;

interface Health {
  paired: boolean;
  extension_connected: boolean;
  current_url?: string;
}

async function health(): Promise<Health | null> {
  const { port: saved = DEFAULT_PORT } = await getLocal();
  try {
    const response = await fetch(`http://127.0.0.1:${saved}/health`);
    return response.ok ? ((await response.json()) as Health) : null;
  } catch {
    return null;
  }
}

function flash(message: string): void {
  // A brief confirmation so a destructive or silent action visibly took
  // effect (findings U7/D8).
  confirm.textContent = message;
  confirm.classList.remove("hidden");
  window.setTimeout(() => confirm.classList.add("hidden"), 4000);
}

async function renderStatus(): Promise<void> {
  const probe = await health();
  if (!probe) {
    pairStatus.textContent = "Local Operator isn't reachable on this computer.";
  } else if (probe.paired) {
    pairStatus.textContent = probe.extension_connected
      ? "Paired and connected to Local Operator."
      : "Paired. The browser will reconnect when Local Operator is running.";
  } else {
    pairStatus.textContent = "Not paired. Open the extension popup to pair.";
  }
}

// The all-sites switch's view state. Kept here rather than re-read from
// storage per event because the dialog-open state (switch shown on, nothing
// written) exists only in this page.
let allowAllState: AllowAllView = allowAllView(false);
let allowAllStored = false;

function paintAllowAll(view: AllowAllView): void {
  allowAllState = view;
  allowAll.checked = view.switchOn;
  allowAllAck.checked = view.acked;
  allowAllEnable.disabled = !view.acked;
  allowAllBanner.classList.toggle("hidden", !view.banner);
  if (view.dialogOpen && !allowAllDialog.open) allowAllDialog.showModal();
  if (!view.dialogOpen && allowAllDialog.open) allowAllDialog.close();
}

/** Run one action through the state machine and persist its write, if any.
 * This is the ONLY writer of `allowAllSites`: a plain storage write from the
 * options page, with no worker message or daemon RPC equivalent, so nothing
 * an agent can call is able to flip it. */
async function applyAllowAll(action: AllowAllAction): Promise<void> {
  const view = nextAllowAllView(allowAllStored, action, allowAllState);
  if (view.write !== undefined) {
    await chrome.storage.local.set({ allowAllSites: view.write });
    allowAllStored = view.write;
    flash(view.write ? "All websites are now allowed." : "Site prompts are back on.");
    paintAllowAll(view);
    // paintAllowAll owns the switch, banner and dialog; the Allowed sites card
    // is painted by render(). Without this the superseded strip is absent at
    // the moment of the accidental enable it exists for, and stale in the
    // other direction: after turning the bypass off the card kept asserting
    // "These grants are not in effect" about grants that now are (U8/Q3).
    await render();
    return;
  }
  paintAllowAll(view);
}

async function render(): Promise<void> {
  const local = await getLocal();
  const { port: saved = DEFAULT_PORT, origins = {} } = local;
  port.value = String(saved);
  allowAllStored = local.allowAllSites === true;
  paintAllowAll(allowAllView(allowAllStored));
  await renderStatus();
  await renderConsent();
  sites.replaceChildren();
  const entries = grantRows(origins, local.hostGrants, local.siteGrants);
  // While every website is allowed these rows grant nothing extra, so the
  // empty note would tell a user hunting for the off-switch that nothing is
  // granted while the agent in fact has every site (U2).
  sitesSuperseded.classList.toggle("hidden", !allowAllStored);
  sitesEmpty.classList.toggle("hidden", entries.length > 0 || allowAllStored);
  for (const entry of entries) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = entry.label;
    const remove = document.createElement("button");
    remove.className = "btn";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", removeGrantAccessibleName(entry));
    remove.addEventListener("click", async () => {
      const result = await runWorkerMutation(revokeMessageFor(entry), `Removed ${entry.label}.`);
      flash(result.message);
      if (result.ok) await render();
    });
    row.append(name, remove);
    sites.append(row);
  }
}

port.addEventListener("change", async () => {
  const parsed = Number(port.value);
  if (Number.isInteger(parsed) && parsed >= 1024 && parsed <= 65535) {
    await chrome.storage.local.set({ port: parsed });
    flash(`Daemon port saved as ${parsed}.`);
  } else {
    flash("Port must be a number between 1024 and 65535.");
    await render();
  }
});

allowAll.addEventListener("change", () => void applyAllowAll({ type: "toggle", checked: allowAll.checked }));
allowAllAck.addEventListener("change", () => void applyAllowAll({ type: "ack", checked: allowAllAck.checked }));
allowAllEnable.addEventListener("click", () => void applyAllowAll({ type: "enable" }));
document.getElementById("allow-all-cancel")?.addEventListener("click", () => void applyAllowAll({ type: "cancel" }));
// Escape fires `cancel` on a modal dialog; route it through the same revert
// so the switch never stays on without a write behind it.
allowAllDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  void applyAllowAll({ type: "cancel" });
});
document.getElementById("allow-all-banner-off")?.addEventListener("click", () => void applyAllowAll({ type: "turn_off" }));
// The same exit from inside the Allowed sites card, where a user looking for
// what to revoke actually lands (U2).
document.getElementById("sites-superseded-off")?.addEventListener("click", async () => {
  await applyAllowAll({ type: "turn_off" });
  await render();
});

const allowDownloads = document.getElementById("allow-downloads") as HTMLInputElement;
const allowUploads = document.getElementById("allow-uploads") as HTMLInputElement;
const downloadsNotice = document.getElementById("downloads-notice") as HTMLParagraphElement;
const uploadsNotice = document.getElementById("uploads-notice") as HTMLParagraphElement;
const consentState = document.getElementById("consent-state") as HTMLParagraphElement;

/* The two file-transfer switches.
 *
 * THE PAGE IS THE ONLY WRITER (`consent.ts` rule 1) and the ONLY place the
 * optional `downloads` permission is requested. Both switches paint from the
 * EFFECTIVE state — flag AND permission — never from the flag alone, which is
 * what stops the page showing "on" for a capability Chrome has since revoked.
 */

/* Paint a notice, in one of two WEIGHTS.
 *
 * Success and refusal used to be the same box, so "Downloads are on" and "Chrome
 * did not grant the permission" differed only in their words (round-1 D2) — on the
 * one part of this page that asks the user to go and do something. `attention` is
 * for the notices that need an ACTION (a grant refused, a grant taken away, a
 * dialog still unanswered): same place, same sentence structure, visibly the one to
 * read first.
 *
 * Painting also repaints the STATE LINE, because the two would otherwise say the
 * same thing in the same breath (round-2 D8): the missing-permission notice ends
 * "so downloads are off" and the state line under it opened "Downloads and uploads
 * are both off…", eight pixels apart. The state line now defers to whichever notice
 * is already explaining a capability, and speaks only to what is left.
 */
function notice(
  target: HTMLParagraphElement,
  message: string,
  kind: "info" | "attention" = "info",
  reveal = false,
): void {
  target.textContent = message;
  target.classList.toggle("hidden", message === "");
  target.classList.toggle("consent-note--attention", kind === "attention" && message !== "");
  paintState();
  // BRING IT INTO VIEW (round-2 U1). Measured before this: at 760x520 the sentence
  // sat at viewport y=530-587 in a 520 px viewport and at 420x700 at y=592-688, so a
  // user who had scrolled far enough to press the switch saw the knob flick back
  // with nothing readable saying why. `scrollIntoView` changes the SCROLL OFFSET
  // only — the two switch rows keep their document positions, which is the property
  // D4 measured and which every reflowing fix (a reserved-height slot above the
  // rows, a notice between them) would have destroyed.
  //
  // A CSS `position: sticky; bottom: 12px` block was tried FIRST and is NOT what
  // ships: it pinned the block at 900x620 and 420x700 but left it at y=530 in a
  // 520 px viewport, i.e. it did not fix the case that matters (round-2 U1's own
  // measurement). It also would not survive the cases this one has to: a notice
  // painted while the window is TALLER than the notice's position (nothing scrolls,
  // correctly) followed by a resize to a short window leaves it below the fold with
  // nothing to bring it back, and tabbing away scrolls focus into view and abandons
  // it again. The reveal below re-runs on both.
  if (message && reveal) {
    revealNotice(target);
  }
}

/** The switch row a notice belongs to, so the reveal corrects the row the user
 *  actually pressed.
 *
 * Hard-coding the downloads row was round-4 R1: the correction ran for EVERY revealed
 * notice, and for the uploads notice it scrolled back UP to a row nobody had touched
 * — measured in real Chrome at 380x300, pressing "Allow uploads" left the uploads row
 * at y=194 and the uploads notice at 410–467 in a 300 px viewport, entirely below the
 * fold, where the same reveal without that step put it at 231–288, fully visible. The
 * downloads notice was hidden by the same step (409–505 against a counterfactual
 * 192–288), which is why §17.16's threshold read low as well.
 */
function rowFor(target: HTMLParagraphElement): HTMLElement | null {
  const rows: Array<[HTMLParagraphElement, string]> = [
    [downloadsNotice, "allow-downloads"],
    [uploadsNotice, "allow-uploads"],
  ];
  const entry = rows.find(([notice]) => notice === target);
  return entry ? document.querySelector(`label[for="${entry[1]}"]`) : null;
}

/** Scroll so the sentence AND the control it belongs to are both readable.
 *
 * Three measured requirements, none of them cosmetic:
 *
 *   1. THE LAYOUT MOVES AFTER THE PAINT. The page's own `#confirm` flash banner
 *      ("Downloads are now allowed.", 20 px plus margins) is raised by the same
 *      gesture, so a scroll computed at paint time is ~31 px stale by the time it
 *      settles — measured leaving the notice 7 px of 38 visible for about four
 *      seconds (round-3 D2). So the scroll is applied, then RE-APPLIED on the next
 *      frame, when the geometry is final, and both passes read live geometry.
 *   2. THE NOTICE MUST NOT SIT FLUSH ON THE EDGE. `scroll-margin-bottom` on the
 *      notice supplies the gap the CSS pins; without it the bottom border landed on
 *      the window edge at every size (round-3 U3).
 *   3. THE ROW IN VIEW IS THE ONE THAT WAS PRESSED (round-4 R1), and the notice
 *      still has to be readable after that correction — the two constraints are
 *      solved TOGETHER, in one offset, rather than by two scrolls that fight.
 *
 * Priority when both cannot be satisfied (the short viewports): the row wins. The
 * knob and its state line stay visible and the notice's tail clips, which is the
 * right way round — the user can see what they pressed and read the answer's opening
 * lines, and the threshold where that starts is recorded in §17.16 rather than
 * discovered by a user.
 *
 * FLOOR is the SINGLE source of that 12 px gap: the CSS `scroll-margin-*` these
 * replaced are only honoured by `scrollIntoView`, which this no longer calls, so they
 * were removed rather than left as inert duplicates (round-5 R5-4). Near the threshold
 * the gap is BEST-EFFORT, not exact — measured 3.9 px instead of 12 px in one
 * compressed case, because the row constraint is applied after the notice's and can
 * claw part of the gap back. Below the threshold something has to give and it is the
 * gap, not the control.
 */
function revealNotice(target: HTMLParagraphElement): void {
  const apply = (): void => {
    const FLOOR = 12;
    const row = rowFor(target);
    const targetBottom = target.getBoundingClientRect().bottom;
    // Positive when the notice's bottom (plus the CSS's own scroll margin, which
    // `scrollIntoView` would have honoured) sits below the fold: scroll down by it.
    let delta = Math.max(0, targetBottom + FLOOR - window.innerHeight);
    if (row) {
      // …unless that would carry the pressed row off the top, in which case the row
      // is brought back to its floor and the notice is allowed to clip.
      const rowTop = row.getBoundingClientRect().top - delta;
      if (rowTop < FLOOR) delta -= FLOOR - rowTop;
    }
    if (delta !== 0) window.scrollBy({ top: delta, behavior: "auto" });
  };
  requestAnimationFrame(() => {
    apply();
    // Again, one frame later: the flash banner has landed by now.
    requestAnimationFrame(apply);
  });
}

/** The state of both switches, in words, once each.
 *
 * Reads the two CHECKBOXES rather than re-reading storage: they are the authority
 * `renderConsent` last painted, and repainting a notice must not become a storage
 * round trip. */
function paintState(): void {
  const downloadsOn = allowDownloads.checked;
  const uploadsOn = allowUploads.checked;
  // A VISIBLE notice is the words for the capability it is about — whichever weight
  // it carries. The state line therefore speaks only for the capability no notice
  // covers, and hides when both are covered, so the two never say the same thing
  // twice (round-2 D8 fixed the attention case; round-4 U3 closes the `info` one,
  // where the notice's "Uploads are on." and the line's "Uploads are on." sat eight
  // pixels apart).
  //
  // Deriving the branches from the SWITCHES rather than from the notices also removes
  // the latent defect round-4 R2 found: the old both-explained branch printed "Neither
  // the agent's saving nor its attaching is available on this browser" from a state
  // that could hold with an ON switch 400 px above it, unreachable only because
  // `uploadsNotice` happens to be painted `info` at its single call site. There is no
  // such branch to reach now — a covered capability is simply not restated.
  const covered = (element: HTMLParagraphElement): boolean => element.textContent !== "";
  const downloadsCovered = covered(downloadsNotice);
  const uploadsCovered = covered(uploadsNotice);

  if (downloadsCovered && uploadsCovered) {
    consentState.textContent = "";
    consentState.classList.add("hidden");
    return;
  }
  consentState.classList.remove("hidden");
  if (!downloadsCovered && !uploadsCovered) {
    if (downloadsOn && uploadsOn) {
      consentState.textContent = "Downloads and uploads are both on for this browser.";
    } else if (downloadsOn) {
      consentState.textContent = "Downloads are on. Uploads are off.";
    } else if (uploadsOn) {
      consentState.textContent = "Uploads are on. Downloads are off.";
    } else {
      consentState.textContent =
        "Downloads and uploads are both off, so the agent can neither save nor attach files on this browser.";
    }
    return;
  }
  // Exactly one capability has its own sentence on screen: say the other one, and
  // only the other one. "as well" needs an antecedent, so it is used only when the
  // covering notice reports that capability OFF — a quiet notice reports an ACTION
  // ("Uploads are on. Turn this off…"), and "Downloads are off as well." under it
  // claimed a relation that is not there (round-5 D1/UX U2). The ATTENTION weight is
  // the same test the rest of this file uses for "this capability is off".
  const reportsOff = (element: HTMLParagraphElement): boolean =>
    element.classList.contains("consent-note--attention");
  if (downloadsCovered) {
    const both = reportsOff(downloadsNotice) && !uploadsOn;
    consentState.textContent = uploadsOn ? "Uploads are on." : both ? "Uploads are off as well." : "Uploads are off.";
  } else {
    const both = reportsOff(uploadsNotice) && !downloadsOn;
    consentState.textContent = downloadsOn
      ? "Downloads are on."
      : both
        ? "Downloads are off as well."
        : "Downloads are off.";
  }
}

/** Paint both switches from live state, then the state line. */
async function renderConsent(): Promise<void> {
  const [downloadsOn, uploadsOn] = await Promise.all([
    capabilityEnabled("download"),
    capabilityEnabled("upload"),
  ]);
  allowDownloads.checked = downloadsOn;
  allowUploads.checked = uploadsOn;
  // The state in WORDS as well as in pixels, beside the other cards' own state
  // lines (round-1 D3: the only thing carrying the off switch was a 14×14 knob at
  // 1.13:1 against the card, and no sentence said which way either switch points).
  paintState();
  // A notice is only ever painted by an ACTION (a denial, a revocation, a
  // refusal to drop the grant), so painting clears nothing on a plain render:
  // the state the user has to read must survive a repaint, but a stale notice
  // about an action they have since answered must not. With ONE exception, which
  // is a STATE rather than an action: a stored flag with no grant behind it.
  //
  // Chrome can take the permission away while this page is closed (round-1 D1),
  // and the flag then outlives it — so the page rendered an untouched default
  // (both switches off, empty note) while the stored consent still said "on",
  // identical to a fresh install and identical between GRANTED, REFUSED and
  // REVOKED. The flag is repaired here — it is this page's own storage, and a flag
  // its permission no longer justifies is not consent — and the reason is painted
  // with the same sentence the revocation path owns.
  if ((await storedSwitch("download")) && !downloadsOn) {
    await writeSwitch("download", false);
    // Revealed too, and deliberately: measured at 760x520, a user who scrolls just
    // far enough to reach the switch (y=549) still has this notice at y=829 off the
    // screen, so the repaired state read as the untouched default at the moment it
    // mattered — U1's defect in the one state that exists on LOAD. The reveal scrolls
    // only when the notice is actually outside the viewport, and this paint fires
    // ONCE per page (the stored flag is cleared in the same branch), so it cannot
    // fight the user's own scrolling on an ordinary render.
    notice(downloadsNotice, PERMISSION_MISSING_MESSAGE, "attention", true);
  }
}

/** Turn `Allow downloads` on: the permission request happens HERE, on the click.
 *
 * `chrome.permissions.request` must be called from a user gesture, which is
 * exactly why the switch — and not a background path — is the only way to turn
 * this capability on: there is no code path from the daemon, a page or a tool
 * call to this function's input. A refusal leaves the switch OFF and says so;
 * the stored flag is written only AFTER the grant, so the state the user sees is
 * never the state they were denied (the failure mode consent.ts rule 2 exists
 * for). */
async function enableDownloads(): Promise<void> {
  const permission = switchPermission("download");
  const api = chrome.permissions;
  // The pending window, stated BEFORE the dialog can appear: the switch paints the
  // EFFECTIVE state (off) rather than the click's optimism, and the note says what
  // the page is waiting for. Round-1 U1 measured the old shape — switch ON,
  // disabled, silent, unchanged after 25 s — which tells a user whose dialog
  // opened behind another window that the capability is on when nothing has been
  // granted, and leaves them no way to undo it from this page.
  await renderConsent();
  // ATTENTION, not the quiet weight (round-2 U2/D6, one defect from two streams).
  // This is the one note that sends the user hunting for a dialog that may have
  // opened behind another window, and the lozenge's ring was measured identical to
  // the success note's (`rgb(59,53,39)`, against `rgb(181,175,162)` for denial and
  // revocation) although this file's own rule classes an unanswered dialog as
  // attention-worthy.
  notice(downloadsNotice, PERMISSION_REQUEST_PENDING, "attention", true);
  const generation = ++pendingDownloadRequest;
  downloadRequestPending = true;
  let granted = false;
  if (permission && api?.request) {
    try {
      // BOUNDED, and VERIFIED (round-1 R5). Measured on Chrome 153.0.8013.53 on
      // this host: the call can fail to SETTLE at all — the promise stayed pending
      // past 25 s, and past a 120 s `Runtime.evaluate` wait — so an unbounded await
      // leaves the switch waiting with no note forever. And the resolved value is
      // not evidence of a grant: the user can answer the dialog and have it
      // refused, so the only proof is asking Chrome whether it holds the
      // permission. Truthiness of the request alone stored the flag `true` and
      // flashed "Downloads are now allowed." with nothing behind it.
      const answer = await deadline(
        api.request({ permissions: [permission as chrome.runtime.ManifestPermissions] }),
        PERMISSION_REQUEST_DEADLINE_MS,
        `chrome.permissions.request(${permission})`,
      ).catch(() => false);
      // A CANCELLED wait must not go on to store anything (round-2 U3): the user
      // pressed the switch again to stop waiting, so this stale answer belongs to a
      // question they have withdrawn. Chrome may still grant (or refuse) later, and
      // the `onAdded`/`onRemoved` listeners repaint when it does — the grant is
      // never inferred from a superseded answer.
      if (generation !== pendingDownloadRequest) {
        downloadRequestPending = false;
        return;
      }
      // `deadline` resolving without a truthy answer, or resolving truthy while
      // Chrome holds nothing, are ONE outcome as far as this page is concerned:
      // there is no grant behind the switch (M5 — the earlier copy reported "did not
      // grant" for that second case, which is a claim about Chrome's answer we
      // cannot make when the honest reading is "we could not confirm it").
      granted = answer === true && (await permissionHeld(permission));
    } catch {
      // A thrown request is not a grant. Same direction as a denial, because the
      // only safe reading of "we could not ask" is "not granted".
      granted = false;
    }
  }
  // Whatever happens next, no request is in flight any more — including on the
  // superseded path, which returned above without clearing it.
  downloadRequestPending = false;
  if (generation !== pendingDownloadRequest) return;
  if (!granted) {
    // The switch must not stay where the user put it: Chrome refused, or never
    // answered, so the capability is not available and a switch reading ON would be
    // a lie the very next refusal would contradict.
    await writeSwitch("download", false);
    await renderConsent();
    notice(
      downloadsNotice,
      // Says what the page can SEE, not what Chrome decided (M5), and does not
      // promise a dialog it cannot show (U4: the old copy ended "Chrome will ask
      // once more", which is a promise about a prompt this build cannot put on the
      // screen).
      `Downloads are still off: Chrome holds no '${permission}' permission for this extension, so the request was not granted or did not finish. Turn the switch on again to ask Chrome once more.`,
      "attention",
      true,
    );
    return;
  }
  await writeSwitch("download", true);
  await renderConsent();
  notice(downloadsNotice, "Downloads are on. Turn this off to stop the agent saving files.");
  flash("Downloads are now allowed.");
}

/** Turn `Allow downloads` off, and hand the permission back.
 *
 * The grant is RELEASED, not merely ignored: the switches are the consent
 * boundary, and leaving an extension holding a permission none of its switches
 * justifies is the state a user revoking access in Chrome is trying to leave. A
 * failed removal is reported rather than hidden — the capability is still off
 * (the flag decides), but the browser still lists the grant, and only the user
 * can remove it from there. */
async function disableDownloads(): Promise<void> {
  await writeSwitch("download", false);
  const permission = switchPermission("download");
  const api = chrome.permissions;
  let removed = true;
  if (permission && api?.remove) {
    try {
      removed = await api.remove({
        permissions: [permission as chrome.runtime.ManifestPermissions],
      });
    } catch {
      removed = false;
    }
  }
  await renderConsent();
  notice(
    downloadsNotice,
    removed
      ? "Downloads are off, and the downloads permission has been handed back to Chrome."
      : `Downloads are off. Chrome still lists the '${permission}' permission for this extension — remove it in chrome://extensions if you want it gone as well.`,
    removed ? "info" : "attention",
    true,
  );
  flash("Downloads are no longer allowed.");
}

async function applyUploads(checked: boolean): Promise<void> {
  await writeSwitch("upload", checked);
  await renderConsent();
  notice(
    uploadsNotice,
    checked
      ? "Uploads are on. Turn this off to stop the agent attaching your files."
      : "Uploads are off, so the agent cannot attach any local file to a page.",
    "info",
    true,
  );
  flash(checked ? "Uploads are now allowed." : "Uploads are no longer allowed.");
}

/* The downloads switch, and how a keyboard user gets OUT of a pending request.
 *
 * The earlier shape set `disabled = true` for the whole (bounded, but up to 120 s)
 * wait and nothing else. Round-2 U3 measured what that costs the user this copy is
 * written for: pressing Space dumped focus to `BODY`, an 8-step Tab walk never
 * reached the switch again, and there was no cancel anywhere on the page.
 *
 * So the switch is NEVER disabled: it keeps focus, keeps its place in the tab
 * order, and a second press is the CANCEL. A press that lands while a request is
 * pending supersedes it (`pendingDownloadRequest`), which stops this page waiting
 * and stores nothing — the answer to a withdrawn question is not consent. A grant
 * that arrives after the cancel is still honoured, because it is Chrome's state
 * and `onAdded` repaints from it; it is simply never *inferred* from a superseded
 * promise.
 */
let pendingDownloadRequest = 0;
let downloadRequestPending = false;

/** Stop waiting for Chrome's answer, and say so.
 *
 * Stores nothing: a withdrawn question is not consent, and the grant is never
 * INFERRED from a superseded promise. A grant Chrome has already made is still
 * honoured — `chrome.permissions.onAdded` repaints from Chrome's own state — so
 * cancelling cannot leave the page disagreeing with the browser.
 */
function cancelPendingDownload(): void {
  pendingDownloadRequest += 1;
  downloadRequestPending = false;
  void renderConsent().then(() =>
    // Short, because it reports a pause the user just created (round-3 U6): the
    // 236-character three-clause version was written for a path a real gesture could
    // not even reach, and the state it describes is the one the line below already
    // shows.
    notice(downloadsNotice, "Stopped waiting. Downloads stay off.", "info", true),
  );
}

/* THE CANCEL IS BOUND TO THE INTENT, NOT TO THE STATE (round-3 U1).
 *
 * The round-2 guard was `downloadRequestPending && !allowDownloads.checked`, and it
 * could not fire for a real gesture: `enableDownloads()` first awaits
 * `renderConsent()`, which repaints the knob to the capability's real (off) state —
 * so a human's second press finds the switch OFF, toggles it ON, and the guard fails.
 * Measured with trusted events: press → 1 request; press again 1.2 s later → **2
 * requests**, with the notice, knob and scroll byte-identical; a third press → 3. The
 * cancel only ever appeared for a synthetic `change` carrying `checked=false`, i.e.
 * the shape the code expected and no human produces.
 *
 * A `click` listener is where this belongs, and `preventDefault()` is what makes it
 * work: for a checkbox the activation behaviour (the toggle) runs as the DEFAULT
 * ACTION of the click, so a handler that runs first can cancel it, and the `change`
 * event never fires. Space on a focused switch fires a click too, so the keyboard
 * path is the same code. The switch therefore keeps focus, keeps its place in the
 * tab order, and a second press is the cancel — while a first press still reaches
 * `enableDownloads` through the untouched `change` listener below.
 */
allowDownloads.addEventListener("click", (event) => {
  if (!downloadRequestPending) return;
  event.preventDefault();
  cancelPendingDownload();
});

allowDownloads.addEventListener("change", () => {
  // NO pending-state branch here (round-4 N2): the click listener above
  // `preventDefault()`s the toggle while a request is in flight, so the `change`
  // event never fires in that state and a second guard would be dead code that
  // reads like a live one. A synthetic `change` carrying `checked=false` could still
  // reach it — that is the rig shape round-3 U1 measured, not a human gesture, and it
  // must not be what keeps the cancel honest.
  const action = allowDownloads.checked ? enableDownloads() : disableDownloads();
  void action.finally(() => {
    // Nothing to restore: the switch was never disabled, and `renderConsent` has
    // already painted whatever the outcome was.
    paintState();
  });
});
allowUploads.addEventListener("change", () => void applyUploads(allowUploads.checked));

// A grant can be removed from OUTSIDE this page — chrome://extensions, an
// enterprise policy, another window — and the switch must follow immediately
// rather than keep claiming a capability that is gone. The stored flag is
// repaired too, so a later repaint cannot resurrect the "on" reading.
chrome.permissions?.onRemoved?.addListener((removed) => {
  void (async () => {
    const permission = switchPermission("download");
    if (permission && removed.permissions?.includes(permission as chrome.runtime.ManifestPermissions)) {
      await writeSwitch("download", false);
      await renderConsent();
      // The SAME sentence the load-time repair paints (`PERMISSION_MISSING_MESSAGE`):
      // a grant taken away while this page was open, a grant taken away while it was
      // closed and a grant that was never made are one state, and one spelling of it
      // is what lets a user recognise the state they are in. Revealed, because the
      // grant can be taken away while this page is open and the user may not be
      // looking at the card when it happens.
      notice(downloadsNotice, PERMISSION_MISSING_MESSAGE, "attention", true);
    }
  })();
});
// The other direction, for completeness: a grant added in chrome://extensions
// leaves the switch off (the switch is the consent, not the grant), but the page
// re-reads so its notices cannot describe a state that has moved.
chrome.permissions?.onAdded?.addListener(() => void renderConsent());

document.getElementById("unpair")?.addEventListener("click", async () => {
  const beforeUnpair = await getLocal();
  const cleared = await runWorkerMutation(
    { event: "clear_access_grants" },
    "This browser is unpaired. Local Operator can no longer use it.",
  );
  if (!cleared.ok) {
    flash(cleared.message);
    return;
  }
  // Tell the running daemon so it severs the LIVE socket, not just the next
  // reconnect (findings A5/U1). Best-effort: the daemon may be down.
  const { port: saved = DEFAULT_PORT } = beforeUnpair;
  try {
    const wire = new WebSocket(`ws://127.0.0.1:${saved}/extension`);
    await new Promise((resolve, reject) => {
      wire.onopen = resolve;
      wire.onerror = reject;
    });
    wire.send(
      JSON.stringify({
        event: "hello",
        proto: PROTO_VERSION,
        token: beforeUnpair.token ?? "",
        extension_version: chrome.runtime.getManifest().version,
        browser: navigator.userAgent,
      }),
    );
    await new Promise((resolve) => setTimeout(resolve, 150));
    wire.send(JSON.stringify({ event: "unpair" }));
    await new Promise((resolve) => setTimeout(resolve, 150));
    wire.close();
  } catch {
    // Daemon unreachable; the local token wipe still stands.
  }
  flash(cleared.message);
  await render();
});

void render();
