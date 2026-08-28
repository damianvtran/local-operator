import {
  adjacentEntryId,
  liveQueue,
  selectEntry,
  type AccessQueueEntry,
} from "../access-queue";
import { DEFAULT_PORT, getLocal, getSession, getSurfaces } from "../state";
import { pairVerdict, viewForHealth } from "./pair-flow";
import {
  ackForDecision,
  originPromptView,
  type DecidedOrigin,
  type OriginDecision,
} from "./origin-flow";

type State =
  | "connected"
  | "paired"
  | "pairing"
  | "disconnected"
  | "incompatible"
  | "origin"
  | "origin-ack";
const sections = [
  "connected",
  "paired",
  "pairing",
  "disconnected",
  "incompatible",
  "origin",
  "origin-ack",
].map((id) => document.getElementById(id));

// True once THIS popup saw pair_result.ok. The daemon confirms pairing on the
// popup's own socket before /health reports paired (the worker still has to
// reconnect with the new token), so a health-driven render in that window must
// hold the explicit success view instead of putting the form back — a re-shown
// form invites re-submitting the consumed one-time code, which then fails with
// a misleading "No live pairing code". See pair-flow.ts.
let locallyPaired = false;

// The origin decision THIS popup already made — the consent flow's copy of the
// same race: `origin_decision` goes to the worker, but session storage and
// /health keep echoing the prompt until that round-trip lands, so a render in
// the window would resurrect the three buttons over a click that already took
// effect. Keyed by origin so a different pending origin still prompts (A6).
// See origin-flow.ts.
let decidedOrigin: DecidedOrigin | null = null;

// The origin the prompt is currently showing, whichever source supplied it.
// After a worker restart session storage is empty and the prompt renders from
// /health's pending_origin alone; a decision click must still acknowledge (and
// key its latch) against THAT origin, or the click lands on the one branch
// where the missing-feedback bug survives (review finding m1).
let shownPromptOrigin: string | undefined;
// The prompt GENERATION the visible buttons belong to. The click sends THIS
// id — never a re-read of current state — so a prompt replaced after render
// cannot be approved by a click aimed at the old one (round-2 B1). Empty for
// a /health-fallback render (worker restarted, no session record); the worker
// then still requires the ORIGIN to match its live prompt.
let shownPromptId = "";
// Popup-local selection follows an immutable generation, not an index. Storage
// changes retain it while alive; when it disappears FIFO becomes current.
let selectedEntryId: string | undefined;

interface Health {
  extension_connected: boolean;
  paired: boolean;
  browser: string;
  current_url?: string;
  current_title?: string;
  pending_origin?: string;
}

// The card's 2px top rule is the design system's ONE spend of colour, and it is
// only ever spent on real semantics: success when the agent can drive, danger
// when the user must recover (error/incompatible), and a neutral hairline for
// the transitional states (disconnected/pairing/origin prompt). The identity
// mark is never tinted — see popup.css. Mirrors callback_page.py's _TONE_VARS.
const TONE: Record<State, string> = {
  connected: "var(--success)",
  paired: "var(--success)",
  pairing: "var(--hairline-strong)",
  disconnected: "var(--hairline-strong)",
  incompatible: "var(--danger)",
  origin: "var(--hairline-strong)",
  // Placeholder only: the ack's real tone is per-decision (success for allow,
  // neutral for deny) and showOriginAck overrides it right after show().
  "origin-ack": "var(--hairline-strong)",
};

function show(state: State): void {
  for (const section of sections) section?.classList.toggle("hidden", section.id !== state);
  document.getElementById("card")?.style.setProperty("--tone", TONE[state]);
  if (state === "pairing") {
    const input = document.getElementById("pair-code") as HTMLInputElement | null;
    input?.focus();
  }
}

async function daemonHealth(): Promise<Health | null> {
  const { port = DEFAULT_PORT } = await getLocal();
  try {
    const response = await fetch(`http://127.0.0.1:${port}/health`);
    if (!response.ok) return null;
    return (await response.json()) as Health;
  } catch {
    return null;
  }
}

async function render(): Promise<void> {
  // A pending site decision wins the popup: it is the one thing the user must
  // act on (findings U2/D1). The daemon reports it in /health so the popup
  // shows it even after a worker restart.
  const session = await getSession();
  const health = await daemonHealth();
  const queue = liveQueue(session.accessQueue, Date.now());
  const selected = selectEntry(queue, selectedEntryId);
  selectedEntryId = selected?.entryId;
  // A health-only fallback keeps an older daemon/extension pairing usable, but
  // queue storage is authoritative whenever it has an entry.
  const pendingOriginValue = selected?.origin || session.pendingOrigin?.origin || health?.pending_origin;
  const pendingHost = selected?.displayAuthority || session.pendingOrigin?.hostname || hostnameOf(health?.pending_origin);
  if (!pendingOriginValue) {
    shownPromptOrigin = undefined;
    shownPromptId = "";
    decidedOrigin = null;
  }
  if (pendingHost) {
    // The heading is fixed prose; the host — the string the user must verify
    // before granting a standing browsing grant — goes in the monospace trough
    // where it renders intact and reads unambiguously as data (D11). The
    // heading id stays constant, so no per-host text in the serif face.
    const host = document.getElementById("origin-host");
    if (host) host.textContent = pendingHost;
    shownPromptOrigin = pendingOriginValue;
    shownPromptId = selected?.entryId ?? session.pendingOrigin?.promptId ?? "";
    renderQueueControls(queue, selected);
    // A fresh prompt must arrive with live buttons even if a previous
    // decision's lock is still set.
    setOriginBusy(false);
    show("origin");
    return;
  }

  // The worker records the last close reason so a protocol mismatch renders
  // "Update needed" rather than sending the user back to code entry (D2).
  const { connState } = (await chrome.storage.session.get(["connState"])) as {
    connState?: string;
  };
  if (!health) {
    show(connState === "incompatible" ? "incompatible" : "disconnected");
    return;
  }
  if (connState === "incompatible") {
    show("incompatible");
    return;
  }
  if (health.paired) {
    // Handoff complete: the worker holds the new token and health confirms it,
    // so the pairing latch has done its job. Clearing it here means a LATER
    // unpair (daemon revoke → close 4003) renders the form again instead of a
    // stale success view (review finding m2). A revoke landing inside the
    // pre-confirmation window is not distinguishable from the race itself and
    // stays covered by the latch until the popup reopens.
    locallyPaired = false;
    // Name the site the agent is driving so the user can see it without
    // hunting for a background tab (finding U3). The URL rides in a labelled
    // trough (the callback page's treatment); when nothing is open the label
    // hides and the trough carries a plain note.
    const label = document.getElementById("connected-label");
    const detail = document.getElementById("connected-detail");
    if (detail && label) {
      const url = health.current_url;
      // Parallel sessions can each drive their own tab now; the card stays a
      // one-line status (no list, no redesign), so with several surfaces the
      // label carries the count and the trough the most recently driven URL.
      const surfaceCount = Object.keys(await getSurfaces()).length;
      if (url) {
        label.textContent =
          surfaceCount > 1 ? `Driving ${surfaceCount} tabs` : health.current_title || "Driving";
        detail.textContent = url;
        label.classList.remove("hidden");
      } else {
        label.classList.add("hidden");
        detail.textContent = "No page open yet.";
      }
    }
    show("connected");
    return;
  }
  // Health has not confirmed the new token yet: keep the success view if this
  // popup already paired (the race above), otherwise offer the form.
  show(viewForHealth(false, locallyPaired));
}

function hostnameOf(origin: string | undefined): string {
  if (!origin) return "";
  try {
    return new URL(origin).hostname;
  } catch {
    return origin;
  }
}

// Keep the pairing field to digits so a desktop keyboard cannot enter letters
// (finding U6); the field is also autofocused when the pairing state renders.
const codeInput = document.getElementById("pair-code") as HTMLInputElement | null;
codeInput?.addEventListener("input", () => {
  codeInput.value = codeInput.value.replace(/\D/g, "").slice(0, 6);
});

// Lock the form for the duration of one submission: a second click while the
// first is in flight would spend the same one-time code twice, and the second
// attempt always fails confusingly. The disabled ramp in popup.css plus the
// relabelled button are the in-progress affordance.
function setPairBusy(busy: boolean): void {
  const input = document.getElementById("pair-code") as HTMLInputElement | null;
  const button = document.querySelector<HTMLButtonElement>("#pair-form button[type='submit']");
  if (input) input.disabled = busy;
  if (button) {
    button.disabled = busy;
    button.textContent = busy ? "Pairing…" : "Pair";
  }
}

document.getElementById("pair-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("pair-code") as HTMLInputElement | null;
  const error = document.getElementById("pair-error");
  if (!input || !error) return;
  error.classList.add("hidden");
  setPairBusy(true);
  const { token, port = DEFAULT_PORT } = await getLocal();
  try {
    const wire = new WebSocket(`ws://127.0.0.1:${port}/extension`);
    // ONE shared rejection wired to error AND close for every await below: a
    // socket that drops after open but before pair_result must land in
    // `catch` — a bare onmessage promise would suspend forever, and with the
    // busy lock held that means a form disabled until the popup is reopened
    // (review finding M1). The no-op catch keeps a post-handshake close (our
    // own close() included, if handlers weren't detached) from surfacing as an
    // unhandled rejection.
    let failed: (reason: Error) => void = () => {};
    const failure = new Promise<never>((_, reject) => {
      failed = reject;
    });
    void failure.catch(() => {});
    wire.onerror = () => failed(new Error("socket error"));
    wire.onclose = () => failed(new Error("socket closed"));
    await Promise.race([
      new Promise((resolve) => {
        wire.onopen = resolve;
      }),
      failure,
    ]);
    const nextMessage = (): Promise<MessageEvent> =>
      Promise.race([
        new Promise<MessageEvent>((resolve) => {
          wire.onmessage = resolve;
        }),
        failure,
      ]);
    wire.send(
      JSON.stringify({
        event: "hello",
        proto: 1,
        token: token ?? "",
        extension_version: chrome.runtime.getManifest().version,
        browser: navigator.userAgent,
      }),
    );
    await nextMessage();
    wire.send(JSON.stringify({ event: "pair", code: input.value.trim() }));
    const verdict = await nextMessage();
    const outcome = pairVerdict(JSON.parse(String(verdict.data)));
    if (outcome.ok) {
      await chrome.storage.local.set({ token: outcome.token });
      // Detach before closing: this close is deliberate, not a failure.
      wire.onerror = null;
      wire.onclose = null;
      wire.close();
      // Confirm SUCCESS from the pair_result frame alone, before any health
      // round-trip: the worker has not reconnected with the new token yet, so
      // /health still says unpaired and a render() here would re-show the form
      // (the race documented at `locallyPaired`). The explicit success state is
      // the user's feedback; render() below only upgrades it to the connected
      // view once health confirms.
      locallyPaired = true;
      show("paired");
      await new Promise((resolve) => setTimeout(resolve, 250));
      await render();
    } else {
      error.textContent = outcome.message;
      error.classList.remove("hidden");
      // A live pairing error turns the status rule danger — the one place the
      // pairing state spends colour, and only on a real failure.
      document.getElementById("card")?.style.setProperty("--tone", "var(--danger)");
      // Unlock BEFORE focusing: a disabled input refuses focus, and the
      // `finally` unlock runs only after this handler returns. (The finally
      // re-call is idempotent.)
      setPairBusy(false);
      input.focus();
    }
  } catch {
    error.textContent = "Could not reach Local Operator on this machine.";
    error.classList.remove("hidden");
    document.getElementById("card")?.style.setProperty("--tone", "var(--danger)");
  } finally {
    // Always unlock, success included: if health later drops (daemon restart)
    // the user lands back on this form, and it must not arrive pre-disabled.
    setPairBusy(false);
  }
});

document.getElementById("retry")?.addEventListener("click", () => void render());
document.getElementById("retry-incompatible")?.addEventListener("click", () => void render());
document.getElementById("origin-allow")?.addEventListener("click", () => void decide("once"));
document.getElementById("origin-always")?.addEventListener("click", () => void decide("always"));
document.getElementById("origin-deny")?.addEventListener("click", () => void decide("deny"));
document.getElementById("origin-previous")?.addEventListener("click", () => void moveQueue(-1));
document.getElementById("origin-next")?.addEventListener("click", () => void moveQueue(1));

// Lock the three consent buttons the moment one is clicked: the session-storage
// read below is async, and a second click in that window would double-send the
// decision. render()'s prompt path unlocks for the next genuine prompt.
function setOriginBusy(busy: boolean): void {
  for (const id of ["origin-allow", "origin-always", "origin-deny", "origin-previous", "origin-next"]) {
    const button = document.getElementById(id) as HTMLButtonElement | null;
    if (button) button.disabled = busy;
  }
}

function renderQueueControls(queue: AccessQueueEntry[], selected: AccessQueueEntry | undefined): void {
  const position = selected ? queue.findIndex((entry) => entry.entryId === selected.entryId) + 1 : 1;
  const count = Math.max(1, queue.length);
  const positionEl = document.getElementById("origin-position");
  const waitingEl = document.getElementById("origin-waiting");
  if (positionEl) positionEl.textContent = `${position} of ${count}`;
  if (waitingEl) waitingEl.textContent = `${count} ${count === 1 ? "request" : "requests"} waiting`;
  for (const id of ["origin-previous", "origin-next"]) {
    const button = document.getElementById(id) as HTMLButtonElement | null;
    if (button) button.disabled = queue.length < 2;
  }
  const authority = selected?.displayAuthority ?? shownPromptOrigin ?? "site";
  document
    .getElementById("origin-previous")
    ?.setAttribute("aria-label", `Previous site request before ${authority}, ${position} of ${count}`);
  document
    .getElementById("origin-next")
    ?.setAttribute("aria-label", `Next site request after ${authority}, ${position} of ${count}`);
}

async function moveQueue(delta: -1 | 1): Promise<void> {
  const { accessQueue } = await getSession();
  const queue = liveQueue(accessQueue, Date.now());
  selectedEntryId = adjacentEntryId(queue, selectedEntryId, delta);
  await render();
}

function showOriginAck(decision: OriginDecision): void {
  const ack = ackForDecision(decision);
  const title = document.getElementById("origin-ack-title");
  const sub = document.getElementById("origin-ack-sub");
  if (title) title.textContent = ack.title;
  if (sub) sub.textContent = ack.sub;
  document.getElementById("origin-ack-check")?.classList.toggle("hidden", !ack.check);
  show("origin-ack");
  // Per-decision tone override: allow is a real success (the agent proceeds),
  // deny is a completed choice — neutral, never danger, which is reserved for
  // states the user must recover from.
  document
    .getElementById("card")
    ?.style.setProperty(
      "--tone",
      ack.tone === "success" ? "var(--success)" : "var(--hairline-strong)",
    );
}

async function decide(decision: OriginDecision): Promise<void> {
  setOriginBusy(true);
  // The click answers what the user SAW — shownPromptOrigin/shownPromptId
  // captured at render — never a re-read of current state: re-reading was
  // round-2 B1's consent hole, where a prompt replaced after render made the
  // click approve an origin the user never looked at. The worker validates
  // the generation and rejects a stale one.
  const origin = shownPromptOrigin;
  if (origin) {
    // Acknowledge from the CLICK alone, before the worker round-trip: session
    // storage and /health keep echoing this prompt until `origin_decision`
    // lands, and a render in that window would put the three buttons back with
    // no sign the click worked — the same race as pairing success. The latch
    // holds the ack through stale echoes; render() takes over to Connected
    // once the echo clears.
    decidedOrigin = { origin, decision };
    showOriginAck(decision);
    const response = (await chrome.runtime.sendMessage({
      event: "origin_decision",
      origin,
      decision,
      entryId: shownPromptId,
    })) as { applied?: boolean } | undefined;
    if (!response?.applied) {
      // The worker refused the decision: the prompt was replaced (or expired)
      // after this popup rendered it. Say so instead of pretending the click
      // took effect, clear the latch, and re-render — which shows the CURRENT
      // prompt, whose buttons are live again.
      decidedOrigin = null;
      showOriginNotice(
        "Request changed.",
        "The site request was replaced while this window was open. Review the new request.",
      );
      setOriginBusy(false);
      // Hold the notice long enough to read (same shape as the pairing
      // success hold), then fall through to render(), which draws the
      // CURRENT prompt with live buttons.
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  }
  // A successful decision removes this generation; FIFO becomes current. A
  // stale rejection also reselects deterministically from the live queue.
  selectedEntryId = undefined;
  await render();
}

/** A neutral informational card in the ack slot — used when a decision could
 * NOT be applied (stale prompt generation). No check mark: nothing was
 * granted or denied. */
function showOriginNotice(title: string, sub: string): void {
  const titleEl = document.getElementById("origin-ack-title");
  const subEl = document.getElementById("origin-ack-sub");
  if (titleEl) titleEl.textContent = title;
  if (subEl) subEl.textContent = sub;
  document.getElementById("origin-ack-check")?.classList.add("hidden");
  show("origin-ack");
  document.getElementById("card")?.style.setProperty("--tone", "var(--hairline-strong)");
}

// Re-render whenever the worker changes connection state: the worker learns a
// new token only on its own reconnect (its retry alarm can be ~30s out), and
// without this the "Connecting…" success view sits until the user pokes the
// popup (review finding m3). connState flips exactly when something the popup
// shows has changed, so this is cheaper and quieter than polling /health.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "session" && (changes.connState || changes.pendingOrigin || changes.accessQueue)) void render();
});

void render();
