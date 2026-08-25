import { DEFAULT_PORT, getLocal } from "../state";

const port = document.getElementById("port") as HTMLInputElement;
const sites = document.getElementById("sites") as HTMLUListElement;

async function render(): Promise<void> {
  const { port: saved = DEFAULT_PORT, origins = {} } = await getLocal();
  port.value = String(saved);
  sites.replaceChildren();
  for (const [origin, verdict] of Object.entries(origins).sort(([a], [b]) => a.localeCompare(b))) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = origin;
    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.addEventListener("click", async () => {
      const next = { ...(await getLocal()).origins };
      delete next[origin];
      await chrome.storage.local.set({ origins: next });
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
  } else {
    await render();
  }
});

document.getElementById("unpair")?.addEventListener("click", async () => {
  await chrome.storage.local.remove(["token", "origins"]);
  await render();
});

void render();
