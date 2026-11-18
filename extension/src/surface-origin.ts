import { BridgeCommandError, requireSurface } from "./cdp";
import { ensureTopLevelAccess, originAllowed } from "./origins";
import { CHROME_API_DEADLINE_MS, deadline } from "./settle";
import { noteAdmittedOrigin, type StoredSurface } from "./state";

/*
 * The CURRENT-DOCUMENT origin gate for page-touching commands.
 *
 * WHY: site approval is enforced on navigations the agent starts (open/goto,
 * and clicks/selects via `withOriginGate`'s Fetch interception). A page the
 * user approved can still navigate ITSELF between commands — a timer, a meta
 * refresh, a script — and nothing is intercepting then. Without this gate the
 * next `read`/`snapshot`/`screenshot`/`click`/... ran against whatever site
 * the tab had landed on, so content from a site the user never approved
 * reached the agent. Every page-touching command therefore re-checks the
 * tab's committed URL here, at ONE choke point in the worker's dispatch
 * (`worker.ts`), so a new command cannot forget to.
 *
 * NO NEW FRICTION: an approved origin (persistent grant, allow-all, or an
 * origin this tab was admitted to by a consumed allow-once — see
 * `StoredSurface.admittedOrigins`) passes silently, exactly as before. An
 * unapproved one gets the SAME `origin_not_allowed` answer a `goto` to that
 * origin gets (`ensureTopLevelAccess` is reused, not copied), so the agent runs
 * the existing request_access flow; a once grant raised by that flow is
 * consumed by the retried command just like a retried goto. No prompt is
 * raised from here — a read never blocks on a popup.
 */

/** Commands that do NOT touch the current document and so are not gated.
 * Everything else that names a tab is — default-deny for future commands.
 * `goto`/`open` are the recovery path off an unapproved page (and gate their
 * own destination); `close`/`status`/`tabs` read only the tab's URL/title,
 * which the `tabs` listing already shows for every surface; the rest manage
 * approvals or ownership and never touch a page. */
export const ORIGIN_GATE_EXEMPT = new Set([
  "open",
  "goto",
  "close",
  "status",
  "tabs",
  "request_access",
  "await_access",
  "cancel_access",
  "submit_signed_decision",
  "list_access",
  "revoke_access",
  "retitle",
  "owner_inspect",
  "owner_recover",
  "owner_finish",
  "owner_retain",
  "owner_release",
]);

/** Same requester shape `nav.ts` accepts: only a session-scoped identity may
 * claim a requester-bound once grant. */
function sessionRequester(params: Record<string, unknown>): string {
  const value = typeof params.requester === "string" ? params.requester.trim() : "";
  return value.startsWith("session:") ? value : "";
}

/** Documents with no site of their own: a fresh tab before its first
 * navigation (`open` creates at about:blank). Their content is either empty or
 * inherited from the approved page that created them. */
function isBlankDocument(url: URL): boolean {
  return url.protocol === "about:" && (url.pathname === "blank" || url.pathname === "srcdoc");
}

/**
 * Throw unless `href` (a document URL observed for `surface`) is an origin
 * this surface may expose. `consume` lets the command-entry check spend a
 * once grant (as a retried goto would); the post-execution re-check passes
 * false so it can only confirm, never spend.
 */
export async function assertDocumentAdmitted(
  surface: StoredSurface,
  href: string,
  requestId: string,
  requester: string,
  consume: boolean,
): Promise<void> {
  let url: URL;
  try {
    url = new URL(href);
  } catch {
    // An unreadable URL proves nothing about which site is showing.
    throw new BridgeCommandError("origin_denied", "the tab's current page could not be identified");
  }
  if (isBlankDocument(url)) return;
  // blob: URLs carry their creator's origin (URL.origin unwraps them); the
  // other non-web schemes (file:, chrome:, view-source:, data: with an opaque
  // origin...) are pages the approval flow cannot grant, so refuse them with
  // the recovery spelled out rather than an origin the agent cannot request.
  const web = url.protocol === "blob:" ? new URL(url.origin) : url;
  if (web.protocol !== "http:" && web.protocol !== "https:") {
    throw new BridgeCommandError(
      "origin_denied",
      `the tab is showing a ${url.protocol} page, which cannot be approved — use goto to return to an allowed site`,
      { origin: url.protocol },
    );
  }
  if (surface.admittedOrigins?.includes(web.origin)) return;
  if (!consume) {
    // Confirm-only: a persistent grant or allow-all still admits; anything
    // else is the shape a first-time origin gets.
    if (await originAllowed(web)) return;
    throw new BridgeCommandError("origin_not_allowed", `site ${web.origin} is not allowed yet`, {
      origin: web.origin,
      url: web.href,
    });
  }
  const admission = await ensureTopLevelAccess(web, requestId, requester);
  // A spent once grant must keep the page readable for later commands, like
  // the navigation it would otherwise have admitted.
  if (admission.viaOnceGrant) await noteAdmittedOrigin(surface.tabId, web.origin);
}

/**
 * The dispatch-level gate: resolve the handle (same `tab_closed` answer as
 * every handler) and check the tab's committed URL. A no-op for exempt
 * methods and for calls without a handle (the handler's own `requireSurface`
 * refuses those).
 */
export async function gateSurfaceOrigin(
  method: string,
  params: Record<string, unknown>,
  requestId: string,
): Promise<void> {
  if (ORIGIN_GATE_EXEMPT.has(method)) return;
  if (params.tab === undefined || params.tab === null || params.tab === "") return;
  const surface = await requireSurface(params.tab);
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  await assertDocumentAdmitted(surface, tab.url ?? "", requestId, sessionRequester(params), true);
}
