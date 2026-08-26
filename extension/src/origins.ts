import { BridgeCommandError, cdp } from "./cdp";
import { safeHttpUrl, storedOriginAllowed } from "./origin-policy";
import { ORIGIN_PROMPT_TIMEOUT_MS } from "./protocol.gen";
import { getLocal } from "./state";

export { safeHttpUrl } from "./origin-policy";
export { ORIGIN_PROMPT_TIMEOUT_MS } from "./protocol.gen";

export type OriginDecision = "once" | "always" | "deny";

// The popup resolves a decision by ORIGIN, not by command id: a single command
// can pause on several origins in a redirect chain, and keying the resolver by
// command id let the second origin overwrite the first's resolver so only the
// last was answerable (finding A6). Keying by origin lets each hop resolve
// independently, and the popup only ever shows one origin at a time anyway.
const waiting = new Map<string, (decision: OriginDecision) => void>();

// A hook the worker installs so a pending decision can raise an ambient signal
// (a system notification) the user sees without already having the popup open
// (finding U2). Kept injectable so origins.ts stays free of worker wiring.
let onPendingChange: ((pending: { origin: string; hostname: string } | null) => void) | null = null;
export function setPendingObserver(
  observer: (pending: { origin: string; hostname: string } | null) => void,
): void {
  onPendingChange = observer;
}

export async function originAllowed(url: URL): Promise<boolean> {
  const { origins = {} } = await getLocal();
  return storedOriginAllowed(origins, url);
}

export async function askOrigin(url: URL, requestId: string): Promise<boolean> {
  if (await originAllowed(url)) return true;
  // Record the pending decision for the popup and the ambient observer. Keyed
  // by origin so concurrent hops do not clobber each other.
  await chrome.storage.session.set({
    pendingOrigin: { origin: url.origin, hostname: url.hostname, requestId },
  });
  await chrome.action.setBadgeBackgroundColor({ color: "#e96042" });
  await chrome.action.setBadgeText({ text: "!" });
  await chrome.action.setTitle({ title: `Local Operator wants to open ${url.hostname}` });
  onPendingChange?.({ origin: url.origin, hostname: url.hostname });
  const decision = await new Promise<OriginDecision>((resolve) => {
    waiting.set(url.origin, resolve);
    setTimeout(() => {
      if (waiting.delete(url.origin)) resolve("deny");
    }, ORIGIN_PROMPT_TIMEOUT_MS);
  });
  await chrome.storage.session.remove("pendingOrigin");
  await chrome.action.setBadgeText({ text: "" });
  await chrome.action.setTitle({ title: "Local Operator" });
  onPendingChange?.(null);
  if (decision === "always") {
    const { origins = {} } = await getLocal();
    await chrome.storage.local.set({ origins: { ...origins, [url.origin]: "allow" } });
  }
  return decision !== "deny";
}

export function resolveOrigin(origin: string, decision: OriginDecision): void {
  const resolve = waiting.get(origin);
  if (resolve) {
    waiting.delete(origin);
    resolve(decision);
  }
}

interface PausedRequest {
  requestId: string;
  request: { url: string };
  resourceType: string;
}

export async function withOriginGate<T>(
  tabId: number,
  commandId: string,
  operation: () => Promise<T>,
  alreadyAllowed: string[] = [],
): Promise<T> {
  // Fetch interception is the security boundary for redirects and trusted
  // clicks. webNavigation fires after Chrome has begun a request; pausing the
  // main document here means an unapproved origin receives no cookies or page
  // request before the browser-owned popup decision.
  const once = new Set(alreadyAllowed);
  const decisions: Promise<void>[] = [];
  let denied: BridgeCommandError | undefined;
  const listener = (
    source: chrome.debugger.Debuggee,
    method: string,
    raw: object | undefined,
  ): void => {
    if (source.tabId !== tabId || method !== "Fetch.requestPaused") return;
    const event = raw as PausedRequest;
    const decision = (async () => {
      let url: URL;
      try {
        url = safeHttpUrl(event.request.url);
      } catch (error) {
        denied = new BridgeCommandError("origin_denied", String(error));
        await cdp(tabId, "Fetch.failRequest", { requestId: event.requestId, errorReason: "BlockedByClient" });
        return;
      }
      if (!once.has(url.origin) && !(await originAllowed(url))) {
        if (!(await askOrigin(url, commandId))) {
          denied = new BridgeCommandError("origin_denied", "site permission was denied", { origin: url.origin });
          await cdp(tabId, "Fetch.failRequest", { requestId: event.requestId, errorReason: "BlockedByClient" });
          return;
        }
        once.add(url.origin);
      }
      await cdp(tabId, "Fetch.continueRequest", { requestId: event.requestId });
    })();
    decisions.push(decision);
  };
  await cdp(tabId, "Fetch.enable", { patterns: [{ resourceType: "Document", requestStage: "Request" }] });
  chrome.debugger.onEvent.addListener(listener);
  try {
    const result = await operation();
    await Promise.all(decisions);
    if (denied) throw denied;
    return result;
  } finally {
    chrome.debugger.onEvent.removeListener(listener);
    try { await cdp(tabId, "Fetch.disable"); } catch { /* a closed tab needs no cleanup */ }
  }
}
