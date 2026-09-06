import { DEFAULT_PORT, getLocal } from "../state";
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
        proto: 1,
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
