import { DEFAULT_PORT, getLocal } from "../state";
import { grantRows, removeExactGrant } from "./grant-list";

const port = document.getElementById("port") as HTMLInputElement;
const sites = document.getElementById("sites") as HTMLUListElement;
const sitesEmpty = document.getElementById("sites-empty") as HTMLParagraphElement;
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

async function render(): Promise<void> {
  const local = await getLocal();
  const { port: saved = DEFAULT_PORT, origins = {} } = local;
  port.value = String(saved);
  await renderStatus();
  sites.replaceChildren();
  const entries = grantRows(origins, local.hostGrants);
  sitesEmpty.classList.toggle("hidden", entries.length > 0);
  for (const entry of entries) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = entry.label;
    const remove = document.createElement("button");
    remove.className = "btn";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${entry.label} access`);
    remove.addEventListener("click", async () => {
      const local = await getLocal();
      if (entry.scope === "host") {
        const response = (await chrome.runtime.sendMessage({
          event: "host_grant_revoke",
          canonicalKey: entry.key,
        })) as { applied?: boolean } | undefined;
        if (!response?.applied) return;
      } else {
        await chrome.storage.local.set({
          origins: removeExactGrant(entry, local.origins ?? {}),
        });
      }
      flash(`Removed ${entry.label}.`);
      await render();
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

document.getElementById("unpair")?.addEventListener("click", async () => {
  const beforeUnpair = await getLocal();
  const cleared = (await chrome.runtime.sendMessage({ event: "clear_access_grants" })) as
    | { applied?: boolean }
    | undefined;
  if (!cleared?.applied) return;
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
  flash("This browser is unpaired. Local Operator can no longer use it.");
  await render();
});

void render();
