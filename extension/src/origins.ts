import { BridgeCommandError, cdp } from "./cdp";
import { safeHttpUrl, storedOriginAllowed } from "./origin-policy";
import { getLocal } from "./state";

export { safeHttpUrl } from "./origin-policy";

export type OriginDecision = "once" | "always" | "deny";
const waiting = new Map<string, (decision: OriginDecision) => void>();

export async function originAllowed(url: URL): Promise<boolean> {
  const { origins = {} } = await getLocal();
  return storedOriginAllowed(origins, url);
}

export async function askOrigin(url: URL, requestId: string): Promise<boolean> {
  if (await originAllowed(url)) return true;
  await chrome.storage.session.set({
    pendingOrigin: { origin: url.origin, hostname: url.hostname, requestId },
  });
  await chrome.action.setBadgeBackgroundColor({ color: "#e96042" });
  await chrome.action.setBadgeText({ text: "!" });
  const decision = await new Promise<OriginDecision>((resolve) => {
    waiting.set(requestId, resolve);
    setTimeout(() => {
      if (waiting.delete(requestId)) resolve("deny");
    }, 60_000);
  });
  await chrome.storage.session.remove("pendingOrigin");
  await chrome.action.setBadgeText({ text: "" });
  if (decision === "always") {
    const { origins = {} } = await getLocal();
    await chrome.storage.local.set({ origins: { ...origins, [url.origin]: "allow" } });
  }
  return decision !== "deny";
}

export function resolveOrigin(requestId: string, decision: OriginDecision): void {
  const resolve = waiting.get(requestId);
  if (resolve) {
    waiting.delete(requestId);
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
