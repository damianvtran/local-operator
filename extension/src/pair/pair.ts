/* The pairing TAB (pair/pair.html): a code-entry surface that survives focus loss.
 *
 * WHY IT EXISTS. Pairing used to be possible only in the toolbar popup, and a
 * Chrome action popup closes the moment focus leaves it. The code is read in
 * another window — the chat with the agent, or a terminal — so the popup was
 * gone exactly when the person needed it (operator report, 2026-11-02: ~14
 * minutes and six attempts to pair one browser). This page is a normal tab: it
 * keeps the field, the digits typed so far and any result while the person
 * switches windows and comes back.
 *
 * HOW IT IS REACHED.
 *   - `lop browser pair` asks the daemon to send `show_pairing` to every
 *     connected, unpaired install; the worker opens (or re-focuses) this tab.
 *   - A fresh install opens it once (worker.ts, onInstalled "install").
 *   - The popup's pairing card links here ("Open in a tab").
 *
 * WHAT IT DOES NOT CHANGE. The code is checked by the daemon against THIS
 * install's code (the socket's Origin names the install), with the same
 * 5-attempt cap and rotation as the popup; the submit itself is shared
 * (`submitPairingCode`). The page never receives a code from anywhere — the
 * person types it.
 *
 * WHAT IT RE-CHECKS. On load, on "Check again", and whenever the tab becomes
 * visible again (the person coming back from the chat) — so a browser paired
 * meanwhile from the popup shows "Paired." rather than a stale form. A
 * re-check never touches the field's value: digits the person already typed
 * survive every re-render. */
import { DEFAULT_PORT, getLocal } from "../state";
import { PAIR_ASK_PHRASE, PAIR_TTL_MINUTES } from "../protocol.gen";
import { digitsOnly, submitPairingCode } from "./submit";

type View = "checking" | "form-state" | "paired-state" | "unreachable-state";
const VIEWS: View[] = ["checking", "form-state", "paired-state", "unreachable-state"];

interface Health {
  paired?: boolean;
  authorized_extension_ids?: string[];
}

const HEALTH_TIMEOUT_MS = 3000;

const input = document.getElementById("pair-code") as HTMLInputElement;
const error = document.getElementById("pair-error") as HTMLParagraphElement;
const submit = document.getElementById("pair-submit") as HTMLButtonElement;
const card = document.getElementById("card") as HTMLElement;

// The generated phrase and lifetime, so this page, the popup, the daemon's
// result messages and the agent guide say the same thing.
for (const node of Array.from(document.querySelectorAll<HTMLElement>("#pair-ask-phrase, .ask-phrase"))) {
  node.textContent = PAIR_ASK_PHRASE;
}
const ttl = document.getElementById("pair-ttl");
if (ttl) ttl.textContent = String(PAIR_TTL_MINUTES);

let current: View = "checking";
/** Set once THIS page saw `pair_result.ok`. /health reports the pairing only
 * after the worker re-dials with the new token, and a re-check in that window
 * must hold "Paired." instead of putting the form back (the popup's race,
 * pair-flow.ts). */
let pairedHere = false;
let busy = false;

function show(view: View): void {
  for (const id of VIEWS) document.getElementById(id)?.classList.toggle("hidden", id !== view);
  card.style.setProperty(
    "--tone",
    view === "paired-state" ? "var(--success)" : view === "unreachable-state" ? "var(--danger)" : "",
  );
  // Focus the field only when ARRIVING at the form, never on a re-check that
  // finds the form already up — that would yank the caret out of the field.
  if (view === "form-state" && current !== "form-state") input.focus();
  current = view;
}

async function health(port: number): Promise<Health | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
  try {
    const response = await fetch(`http://127.0.0.1:${port}/health`, { signal: controller.signal });
    return response.ok ? ((await response.json()) as Health) : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Whether THIS install is paired: listed in the daemon's allow-list AND
 * holding a token. A listed id with no token (storage cleared) cannot
 * authenticate, so it needs the form, not "Paired." */
function selfPaired(probe: Health, token: string | undefined): boolean {
  const listed = probe.authorized_extension_ids;
  if (!token) return false;
  if (Array.isArray(listed)) return listed.includes(chrome.runtime.id);
  return Boolean(probe.paired);
}

async function render(): Promise<void> {
  if (busy) return;
  const { token, port = DEFAULT_PORT } = await getLocal();
  const probe = await health(port);
  if (busy) return;
  if (!probe) {
    show("unreachable-state");
    return;
  }
  if (pairedHere || selfPaired(probe, token)) {
    show("paired-state");
    return;
  }
  // Was this install unpaired on purpose (the worker records a 4003 close)?
  // A separate sentence, so the form is not indistinguishable from a fresh
  // install after a revoke (the popup's U5 rule).
  const { revoked } = (await chrome.storage.session.get("revoked")) as { revoked?: boolean };
  document.getElementById("pair-unpaired-note")?.classList.toggle("hidden", !revoked);
  show("form-state");
}

function setBusy(on: boolean): void {
  busy = on;
  input.disabled = on;
  submit.disabled = on;
  submit.textContent = on ? "Pairing…" : "Pair";
}

function fail(message: string): void {
  error.textContent = message;
  error.classList.remove("hidden");
  card.style.setProperty("--tone", "var(--danger)");
  // Unlock BEFORE selecting: a disabled input refuses focus. SELECT rather
  // than focus, so the next keystroke or paste replaces the rejected digits
  // instead of being swallowed by maxlength (the popup's U1 rule).
  setBusy(false);
  input.select();
}

input.addEventListener("input", () => {
  digitsOnly(input);
  // A fresh edit clears the last failure, so a red line never sits under a
  // code the person has already corrected.
  if (!error.classList.contains("hidden")) {
    error.classList.add("hidden");
    card.style.setProperty("--tone", "");
  }
});

document.getElementById("pair-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const code = input.value.trim();
  if (code.length !== 6) {
    fail("Type all 6 digits of the code.");
    return;
  }
  error.classList.add("hidden");
  setBusy(true);
  try {
    const { token, port = DEFAULT_PORT } = await getLocal();
    const outcome = await submitPairingCode({ code, port, token });
    if (outcome.ok) {
      await chrome.storage.local.set({ token: outcome.token });
      pairedHere = true;
      setBusy(false);
      show("paired-state");
      return;
    }
    if (outcome.kind === "rejected") fail(outcome.message);
    else if (outcome.kind === "closed") fail("Local Operator closed the connection. Press Pair to try again.");
    else fail("Can't reach Local Operator. Make sure it's running, then press Pair.");
  } finally {
    if (busy) setBusy(false);
  }
});

document.getElementById("retry")?.addEventListener("click", () => {
  show("checking");
  void render();
});

// `lop browser pair` ran again while this tab was open: the worker focuses the
// tab and asks for a re-check rather than reloading it (a reload would lose
// what the person typed). `render` never touches the field's value.
chrome.runtime.onMessage.addListener((message) => {
  if (message?.event === "pairing_recheck") void render();
  return undefined;
});

// The person coming back from the chat is the moment the page must be right.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") void render();
});

void render();
