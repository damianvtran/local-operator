import { DEFAULT_PORT, getLocal, getSession } from "../state";
import { pairVerdict, viewForHealth } from "./pair-flow";

type State = "connected" | "paired" | "pairing" | "disconnected" | "incompatible" | "origin";
const sections = ["connected", "paired", "pairing", "disconnected", "incompatible", "origin"].map(
  (id) => document.getElementById(id),
);

// True once THIS popup saw pair_result.ok. The daemon confirms pairing on the
// popup's own socket before /health reports paired (the worker still has to
// reconnect with the new token), so a health-driven render in that window must
// hold the explicit success view instead of putting the form back — a re-shown
// form invites re-submitting the consumed one-time code, which then fails with
// a misleading "No live pairing code". See pair-flow.ts.
let locallyPaired = false;

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
  const { pendingOrigin } = await getSession();
  const health = await daemonHealth();
  const pendingHost = pendingOrigin?.hostname || hostnameOf(health?.pending_origin);
  if (pendingHost) {
    // The heading is fixed prose; the host — the string the user must verify
    // before granting a standing browsing grant — goes in the monospace trough
    // where it renders intact and reads unambiguously as data (D11). The
    // heading id stays constant, so no per-host text in the serif face.
    const host = document.getElementById("origin-host");
    if (host) host.textContent = pendingHost;
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
    // Name the site the agent is driving so the user can see it without
    // hunting for a background tab (finding U3). The URL rides in a labelled
    // trough (the callback page's treatment); when nothing is open the label
    // hides and the trough carries a plain note.
    const label = document.getElementById("connected-label");
    const detail = document.getElementById("connected-detail");
    if (detail && label) {
      const url = health.current_url;
      if (url) {
        label.textContent = health.current_title || "Driving";
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
    await new Promise((resolve) => {
      wire.onmessage = resolve;
    });
    wire.send(JSON.stringify({ event: "pair", code: input.value.trim() }));
    const verdict = await new Promise<MessageEvent>((resolve) => {
      wire.onmessage = resolve;
    });
    const outcome = pairVerdict(JSON.parse(String(verdict.data)));
    if (outcome.ok) {
      await chrome.storage.local.set({ token: outcome.token });
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

async function decide(decision: "once" | "always" | "deny"): Promise<void> {
  const { pendingOrigin } = await getSession();
  if (pendingOrigin) {
    // Decisions are keyed by ORIGIN so a redirect chain resolves each hop
    // independently (finding A6).
    await chrome.runtime.sendMessage({
      event: "origin_decision",
      origin: pendingOrigin.origin,
      decision,
    });
  }
  await render();
}

void render();
