import { BridgeCommandError, requireSurface } from "./cdp";
import { NODE_DENIALS_KEY, normalizedDenials } from "./node-site-decisions";
import { ensureTopLevelAccess, originAllowed } from "./origins";
import { CHROME_API_DEADLINE_MS, deadline } from "./settle";
import { getSurfaces, noteAdmittedOrigin, surfaceToken, type StoredSurface } from "./state";

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
  // An explicit Deny the user saved AFTER an allow-once admitted this tab
  // outranks that admission (the deny is the newer, stronger statement); the
  // normal path below then refuses unless a fresh grant says otherwise. Read
  // only when the admission would otherwise short-circuit, so the common
  // persistently-approved path keeps its storage-read budget.
  if (surface.admittedOrigins?.includes(web.origin) && !(await explicitlyDenied(web.origin))) return;
  if (!consume) {
    // Confirm-only: a persistent grant or allow-all still admits; anything
    // else is the shape a first-time origin gets.
    if (await originAllowed(web)) return;
    throw notAllowed(web.origin);
  }
  // Judge the ORIGIN, never the page's own URL: grants are origin-keyed, and
  // the refusal's data must not echo a path/query the unapproved page chose
  // (it can carry tokens) — for a goto that URL was the agent's input, here it
  // is not.
  let admission;
  try {
    admission = await ensureTopLevelAccess(new URL(web.origin), requestId, requester);
  } catch (error) {
    if (error instanceof BridgeCommandError && error.code === "origin_not_allowed") throw notAllowed(web.origin);
    throw error;
  }
  // A spent once grant must keep the page readable for later commands, like
  // the navigation it would otherwise have admitted.
  if (admission.viaOnceGrant) await noteAdmittedOrigin(surface.tabId, web.origin);
}

/**
 * Confirm-only admission of a bare origin for `surface` — the rule the gate
 * applies to a live document, for data captured EARLIER (buffered console
 * logs). Never spends a grant and never prompts.
 */
export async function originAdmitted(surface: StoredSurface, origin: string): Promise<boolean> {
  if (!origin) return false;
  try {
    await assertDocumentAdmitted(surface, `${origin}/`, "", "", false);
    return true;
  } catch (error) {
    if (error instanceof BridgeCommandError) return false;
    throw error;
  }
}

function notAllowed(origin: string): BridgeCommandError {
  return new BridgeCommandError("origin_not_allowed", `site ${origin} is not allowed yet`, { origin });
}

/** A saved per-origin Deny (popup/options) or an operator node denial. One
 * local-storage read; both keys are written by the existing decision flows. */
async function explicitlyDenied(origin: string): Promise<boolean> {
  const stored = (await deadline(
    chrome.storage.local.get(["origins", NODE_DENIALS_KEY]),
    CHROME_API_DEADLINE_MS,
    "chrome.storage.local.get(origins, node denials)",
  )) as { origins?: Record<string, string>; [NODE_DENIALS_KEY]?: unknown };
  if (stored.origins?.[origin] === "deny") return true;
  return Object.prototype.hasOwnProperty.call(normalizedDenials(stored[NODE_DENIALS_KEY]), origin);
}

/** Gated commands whose RESULT is page-derived data and so must be discarded
 * when the tab committed another document while they ran (the gate checked
 * the tab before; the capture happens after). `read` checks its own result's
 * `location.href` and is listed for the same tab-level confirmation. Actions
 * (click/type/select/scroll/upload/download) are not: their effect already
 * happened, their navigations run under `withOriginGate`, and turning a done
 * click into an error would only make the agent repeat it. */
const POST_CHECKED = new Set(["read", "snapshot", "screenshot", "styles", "hit_test", "ancestors", "query"]);

/** The origin a document URL belongs to (blob: unwrapped to its creator), or
 * "" when unreadable — the key admitted origins and log entries are compared
 * by. */
export function documentOrigin(href: string): string {
  try {
    const url = new URL(href);
    return url.protocol === "blob:" ? new URL(url.origin).origin : url.origin;
  } catch {
    return "";
  }
}

/**
 * The dispatch-level gate around one handler: resolve the handle (same
 * `tab_closed` answer as every handler), check the tab's committed URL, run
 * the handler, and for page-data commands re-read the tab ONCE afterwards.
 * If the top origin moved while the handler ran, the result is only returned
 * when the new origin is itself admitted (confirm-only — it never spends a
 * grant). A pass-through for exempt methods and for calls without a handle
 * (the handler's own `requireSurface` refuses those).
 *
 * Why a tab re-read and not a per-frame URL for every extractor: the shared
 * geometry functions (driver/, vendored into the desktop host) keep their
 * result shape, and a navigation that commits during the capture leaves the
 * tab on the new origin — the page that navigated away cannot bring it back,
 * because the destination is the site it does not control.
 */
export async function withSurfaceOriginGate<T>(
  method: string,
  params: Record<string, unknown>,
  requestId: string,
  run: () => Promise<T>,
): Promise<T> {
  if (ORIGIN_GATE_EXEMPT.has(method)) return run();
  if (params.tab === undefined || params.tab === null || params.tab === "") return run();
  const surface = await requireSurface(params.tab);
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  const gatedHref = tab.url ?? "";
  await assertDocumentAdmitted(surface, gatedHref, requestId, sessionRequester(params), true);
  const result = await run();
  if (!POST_CHECKED.has(method)) return result;
  const after = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  const afterHref = after.url ?? "";
  if (documentOrigin(afterHref) !== documentOrigin(gatedHref)) {
    // Fresh surface: a revoke while the handler ran may have dropped an
    // admission the copy read before the handler still lists.
    const current = (await getSurfaces())[surfaceToken(surface)] ?? { ...surface, admittedOrigins: [] };
    await assertDocumentAdmitted(current, afterHref, requestId, "", false);
  }
  return result;
}
