import { DEFAULT_PORT, getLocal, getSession } from "../state";

const sections = ["connected", "pairing", "disconnected", "origin"].map((id) => document.getElementById(id));

function show(id: string): void {
  for (const section of sections) section?.classList.toggle("hidden", section.id !== id);
}

async function daemonState(): Promise<{ connected: boolean; paired: boolean; browser: string } | null> {
  const { port = DEFAULT_PORT } = await getLocal();
  try {
    const response = await fetch(`http://127.0.0.1:${port}/health`);
    if (!response.ok) return null;
    return (await response.json()) as { connected: boolean; paired: boolean; browser: string };
  } catch {
    return null;
  }
}

async function render(): Promise<void> {
  const { pendingOrigin } = await getSession();
  if (pendingOrigin) {
    const title = document.getElementById("origin-title");
    if (title) title.textContent = `Let the agent use ${pendingOrigin.hostname}?`;
    show("origin");
    return;
  }
  const daemon = await daemonState();
  if (!daemon) show("disconnected");
  else if (daemon.paired) {
    const detail = document.getElementById("connected-detail");
    if (detail) detail.textContent = daemon.browser || "";
    show("connected");
  } else show("pairing");
}

document.getElementById("pair-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("pair-code") as HTMLInputElement | null;
  const error = document.getElementById("pair-error");
  if (!input || !error) return;
  error.classList.add("hidden");
  const { token } = await getLocal();
  const { port = DEFAULT_PORT } = await getLocal();
  try {
    const wire = new WebSocket(`ws://127.0.0.1:${port}/extension`);
    await new Promise((resolve, reject) => {
      wire.onopen = resolve;
      wire.onerror = reject;
    });
    wire.send(JSON.stringify({ event: "hello", proto: 1, token: token ?? "", extension_version: chrome.runtime.getManifest().version, browser: navigator.userAgent }));
    await new Promise((resolve) => { wire.onmessage = resolve; });
    wire.send(JSON.stringify({ event: "pair", code: input.value.trim() }));
    const verdict = await new Promise<MessageEvent>((resolve) => { wire.onmessage = resolve; });
    const frame = JSON.parse(String(verdict.data)) as {
      event: string;
      ok: boolean;
      token?: string;
      message?: string;
    };
    if (frame.ok && frame.token) {
      // Pairing runs on a short popup-owned socket, which deliberately replaces
      // the worker socket. Persist before closing so the worker's reconnect can
      // authenticate instead of returning to the unpaired state.
      await chrome.storage.local.set({ token: frame.token });
      wire.close();
      await new Promise((resolve) => setTimeout(resolve, 250));
      await render();
    } else {
      error.textContent = frame.message ?? "That code didn't match. Codes expire after two minutes — check the app for a fresh one.";
      error.classList.remove("hidden");
    }
  } catch {
    error.textContent = "Could not reach Local Operator on this machine.";
    error.classList.remove("hidden");
  }
});

document.getElementById("retry")?.addEventListener("click", () => void render());
document.getElementById("origin-allow")?.addEventListener("click", async () => {
  const always = (document.getElementById("origin-always") as HTMLInputElement | null)?.checked;
  await decide(always ? "always" : "once");
});
document.getElementById("origin-deny")?.addEventListener("click", () => void decide("deny"));

async function decide(decision: "once" | "always" | "deny"): Promise<void> {
  const { pendingOrigin } = await getSession();
  if (pendingOrigin) {
    await chrome.runtime.sendMessage({ event: "origin_decision", requestId: pendingOrigin.requestId, decision });
  }
  await render();
}

void render();
