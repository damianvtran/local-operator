import { DEFAULT_PORT, getLocal } from "../state";
import { PROTO_VERSION } from "../protocol.gen";
import {
  capabilityEnabled,
  permissionHeld,
  switchPermission,
  writeSwitch,
} from "../consent";
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

/* The two file-transfer switches.
 *
 * THE PAGE IS THE ONLY WRITER (`consent.ts` rule 1) and the ONLY place the
 * optional `downloads` permission is requested. Both switches paint from the
 * EFFECTIVE state — flag AND permission — never from the flag alone, which is
 * what stops the page showing "on" for a capability Chrome has since revoked.
 */

function notice(target: HTMLParagraphElement, message: string): void {
  target.textContent = message;
  target.classList.toggle("hidden", message === "");
}

/** Paint both switches (and their notices) from live state. */
async function renderConsent(): Promise<void> {
  const [downloadsOn, uploadsOn] = await Promise.all([
    capabilityEnabled("download"),
    capabilityEnabled("upload"),
  ]);
  allowDownloads.checked = downloadsOn;
  allowUploads.checked = uploadsOn;
  // A notice is only ever painted by an ACTION (a denial, a revocation, a
  // refusal to drop the grant), so painting clears nothing on a plain render:
  // the state the user has to read must survive a repaint, but a stale notice
  // about an action they have since answered must not.
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
  let granted = true;
  if (permission && api?.request) {
    try {
      granted = await api.request({
        permissions: [permission as chrome.runtime.ManifestPermissions],
      });
    } catch {
      // A thrown request is not a grant. Same direction as a denial, because the
      // only safe reading of "we could not ask" is "not granted".
      granted = false;
    }
  } else if (permission) {
    granted = false;
  }
  if (!granted) {
    // The switch must not stay where the user put it: Chrome refused, so the
    // capability is not available and a switch reading ON would be a lie the
    // very next refusal would contradict.
    await writeSwitch("download", false);
    await renderConsent();
    notice(
      downloadsNotice,
      `Chrome did not grant the '${permission}' permission, so downloads stay off. You can turn this on again — Chrome will ask once more.`,
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
  );
  flash(checked ? "Uploads are now allowed." : "Uploads are no longer allowed.");
}

allowDownloads.addEventListener("change", () => {
  // While the request is in flight, and again if it is refused, the switch shows
  // the state the capability is REALLY in rather than the click's optimism.
  allowDownloads.disabled = true;
  const action = allowDownloads.checked ? enableDownloads() : disableDownloads();
  void action.finally(() => {
    allowDownloads.disabled = false;
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
      notice(
        downloadsNotice,
        "Chrome removed the downloads permission, so downloads are off. Turn the switch on again to ask for it once more.",
      );
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
