import { DEFAULT_PORT, getLocal } from "../state";

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
  const { port: saved = DEFAULT_PORT, origins = {} } = await getLocal();
  port.value = String(saved);
  await renderStatus();
  sites.replaceChildren();
  const entries = Object.entries(origins).sort(([a], [b]) => a.localeCompare(b));
  sitesEmpty.classList.toggle("hidden", entries.length > 0);
  for (const [origin, verdict] of entries) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = origin;
    const remove = document.createElement("button");
    remove.className = "btn";
    remove.textContent = "Remove";
    remove.addEventListener("click", async () => {
      const next = { ...(await getLocal()).origins };
      delete next[origin];
      await chrome.storage.local.set({ origins: next });
      flash(`Removed ${origin}.`);
      await render();
    });
    row.append(name, remove);
    sites.append(row);
    void verdict;
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
  await chrome.storage.local.remove(["token", "origins"]);
  // Tell the running daemon so it severs the LIVE socket, not just the next
  // reconnect (findings A5/U1). Best-effort: the daemon may be down.
  const { port: saved = DEFAULT_PORT, token } = { ...(await getLocal()), token: undefined };
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
        token: token ?? "",
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
