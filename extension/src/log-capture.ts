import type { LogEntry } from "./protocol.gen";

/**
 * Per-tab console/runtime log capture for the `logs` command.
 *
 * WHY a ring buffer in the worker rather than querying the page: the whole
 * point of `logs` is to surface real console output AND uncaught exceptions
 * that already happened — a page's own `console` history is not readable after
 * the fact, so we must be listening from the moment the surface opens. We
 * enable the CDP `Log` and `Runtime` domains on attach and buffer three event
 * streams into one ordered list per tab:
 *   - `Runtime.consoleAPICalled`  — every console.log/warn/error/info/debug
 *   - `Runtime.exceptionThrown`   — uncaught exceptions (the debugging payoff)
 *   - `Log.entryAdded`            — browser-level messages (network, security,
 *                                   deprecations) the page never sees
 *
 * CONSTRAINTS:
 *   - The buffer is module-global and does NOT survive service-worker death.
 *     That is acceptable and honest: `logs` is defined as "since this surface
 *     opened", and a worker restart tears the surface's debugger session down
 *     anyway, so there is no window where we claim history we silently lost.
 *   - Capped two ways (entry count AND total bytes) so a page in a tight
 *     console-spam loop cannot grow the worker's memory without bound; the
 *     oldest entries are dropped first (newest-last is what the tool wants).
 *   - No page-provided string is ever evaluated. We read structured CDP event
 *     payloads and stringify argument previews with a fixed formatter, keeping
 *     the no-remote-code posture.
 */

// Keep the newest N entries; a debugging read wants recent output, and an
// unbounded buffer on a noisy page is a memory leak in a long-lived worker.
const MAX_ENTRIES = 200;
// Independent byte ceiling: 200 short lines is nothing, but 200 lines each
// carrying a stringified megabyte object would still blow up. Trim oldest
// until under this regardless of count.
const MAX_BYTES = 256 * 1024;
// One argument preview is itself capped so a single huge logged object cannot
// dominate the buffer; the tool truncates the whole response again to its own
// text ceiling on the Python side.
const MAX_ARG_CHARS = 2000;

/*
 * ORIGIN TAGS. The buffer outlives navigation (it is "since this surface
 * opened"), and a tab can navigate itself to a site the user never approved,
 * log there, and come back. Each entry is therefore stamped, at capture, with
 * the top-level DOCUMENT it was produced under, and `logs` returns only
 * entries whose document origin the tab is admitted to NOW (commands/logs.ts)
 * — the same rule the dispatch gate applies to live page reads
 * (surface-origin.ts). The tag lives beside the entry, not in it, so the wire
 * `LogEntry` shape is unchanged.
 *
 * Documents are numbered GENERATIONS: `Runtime.executionContextsCleared` (a
 * main-frame navigation) starts a new one, and a generation's origin is the
 * origin of the MAIN FRAME's default context (`auxData.frameId` equal to the
 * tab's main frame id, read once with `Page.getFrameTree`). Matching the frame
 * id, not "the first default context after the clear", matters because
 * contexts are created lazily: a same-process iframe with script can get its
 * context before a script-less main document does, and its origin must never
 * label the page (review m1). The origin is resolved at READ time, so entries
 * logged before the main context appeared are still attributed correctly.
 *
 * Every context in the map belongs to the current generation (the map is
 * cleared with it), so console output and exceptions from an iframe are
 * attributed to the TOP page, exactly as read/snapshot include iframe content
 * of an approved page: an approved page's embedded widgets keep logging into
 * `logs` as they did before this check existed, and an unapproved top page's
 * iframes stay hidden with it (review Q3/n1). An event naming a context id the
 * map does not hold belongs to no known document and is tagged with no
 * generation — never returned, never borrowed from the current page (review
 * m2). `Log.entryAdded` carries no context, so it takes the current
 * generation. Origins are taken from the CDP Runtime stream itself, so they are
 * ordered with the events they label (a webNavigation commit arrives on a
 * different channel and could land after the new page's first logs).
 */
interface Tagged {
  entry: LogEntry;
  // Document generation the entry was captured under; -1 = unknown (dropped).
  generation: number;
}

interface Buffer {
  entries: Tagged[];
  bytes: number;
  // Live context id -> its frame id, for the current generation only.
  contexts: Map<number, string>;
  generation: number;
  // Generation -> main-frame document origin ("" or absent = unknown).
  origins: Map<number, string>;
  mainFrameId: string;
}

const buffers = new Map<number, Buffer>();

function bufferFor(tabId: number): Buffer {
  let buffer = buffers.get(tabId);
  if (!buffer) {
    buffer = { entries: [], bytes: 0, contexts: new Map(), generation: 0, origins: new Map(), mainFrameId: "" };
    buffers.set(tabId, buffer);
  }
  return buffer;
}

/** Normalize CDP's context origin: opaque contexts report "null" or "". */
function cleanOrigin(value: unknown): string {
  return typeof value === "string" && value !== "null" ? value : "";
}

/** The generation an event with `contextId` belongs to: the current one for a
 * live context, -1 (never returned) for an id the map does not hold, and the
 * current one for an event that names no context at all. */
function contextGeneration(buffer: Buffer, contextId: unknown): number {
  if (contextId === undefined || contextId === null) return buffer.generation;
  return typeof contextId === "number" && buffer.contexts.has(contextId) ? buffer.generation : -1;
}

function originOf(buffer: Buffer, generation: number): string {
  return buffer.origins.get(generation) ?? "";
}

function push(tabId: number, entry: LogEntry, generation: number): void {
  const buffer = bufferFor(tabId);
  buffer.entries.push({ entry, generation });
  buffer.bytes += entry.text.length;
  // Evict oldest-first until BOTH caps are satisfied. Order matters: the tool
  // contract is newest-last, so we always drop from the front.
  while (buffer.entries.length > MAX_ENTRIES || buffer.bytes > MAX_BYTES) {
    const dropped = buffer.entries.shift();
    if (!dropped) break;
    buffer.bytes -= dropped.entry.text.length;
  }
}

/** Normalize a CDP console API `type` to the tool's level vocabulary. */
function levelForConsole(type: unknown): string {
  switch (type) {
    case "error":
    case "assert":
      return "error";
    case "warning":
      return "warning";
    case "info":
      return "info";
    case "debug":
      return "log";
    default:
      return "log";
  }
}

/** Normalize a CDP `Log.entryAdded` level to the tool vocabulary. */
function levelForLogEntry(level: unknown): string {
  switch (level) {
    case "error":
      return "error";
    case "warning":
      return "warning";
    case "info":
      return "info";
    default:
      return "log";
  }
}

/**
 * Render a `Runtime.RemoteObject` argument to a short string WITHOUT evaluating
 * anything. CDP already sends a value or a preview; we read those fields only.
 */
function renderArg(arg: Record<string, unknown> | undefined): string {
  if (!arg) return "";
  if (arg.type === "string") return String(arg.value ?? "");
  if ("value" in arg && arg.value !== undefined) return String(arg.value);
  if (typeof arg.description === "string") return arg.description;
  if (arg.type === "undefined") return "undefined";
  if (arg.subtype === "null") return "null";
  return String(arg.type ?? "");
}

function clip(text: string): string {
  return text.length > MAX_ARG_CHARS ? `${text.slice(0, MAX_ARG_CHARS)}…` : text;
}

// The single debugger event listener. chrome.debugger delivers every domain
// event for every attached tab through ONE global listener, so we route by the
// source tabId into the right per-tab buffer and ignore events for tabs we are
// not capturing.
function onEvent(source: chrome.debugger.Debuggee, method: string, rawParams?: object): void {
  const tabId = source.tabId;
  if (tabId === undefined || !buffers.has(tabId)) return;
  const params = (rawParams ?? {}) as Record<string, unknown>;
  const buffer = bufferFor(tabId);

  if (method === "Runtime.executionContextCreated") {
    const context = (params.context ?? {}) as Record<string, unknown>;
    const aux = (context.auxData ?? {}) as Record<string, unknown>;
    const frameId = typeof aux.frameId === "string" ? aux.frameId : "";
    if (typeof context.id === "number") buffer.contexts.set(context.id, frameId);
    if (aux.isDefault === true && frameId && frameId === buffer.mainFrameId) {
      buffer.origins.set(buffer.generation, cleanOrigin(context.origin));
    }
  } else if (method === "Runtime.executionContextDestroyed") {
    if (typeof params.executionContextId === "number") buffer.contexts.delete(params.executionContextId);
  } else if (method === "Runtime.executionContextsCleared") {
    // A main-frame navigation: every old context is gone, and the next
    // document's origin is unknown until its context is created.
    buffer.contexts.clear();
    buffer.generation += 1;
  } else if (method === "Runtime.consoleAPICalled") {
    const args = Array.isArray(params.args) ? (params.args as Record<string, unknown>[]) : [];
    const text = clip(args.map(renderArg).join(" ").trim());
    const frame = topFrame(params.stackTrace);
    push(tabId, {
      level: levelForConsole(params.type),
      text,
      source: "console",
      url: frame.url,
      line: frame.line,
      timestamp: typeof params.timestamp === "number" ? params.timestamp : Date.now(),
    }, contextGeneration(buffer, params.executionContextId));
  } else if (method === "Runtime.exceptionThrown") {
    const details = (params.exceptionDetails ?? {}) as Record<string, unknown>;
    // Prefer the thrown value's description (the Error's message + stack); fall
    // back to the bare exception text CDP always provides.
    const exception = (details.exception ?? {}) as Record<string, unknown>;
    const text = clip(
      String(exception.description ?? details.text ?? "uncaught exception").trim(),
    );
    push(tabId, {
      level: "error",
      text,
      source: "exception",
      url: typeof details.url === "string" ? details.url : "",
      line: typeof details.lineNumber === "number" ? details.lineNumber : 0,
      timestamp: typeof params.timestamp === "number" ? params.timestamp : Date.now(),
    }, contextGeneration(buffer, details.executionContextId));
  } else if (method === "Log.entryAdded") {
    const entry = (params.entry ?? {}) as Record<string, unknown>;
    push(tabId, {
      level: levelForLogEntry(entry.level),
      text: clip(String(entry.text ?? "").trim()),
      source: "log-entry",
      url: typeof entry.url === "string" ? entry.url : "",
      line: typeof entry.lineNumber === "number" ? entry.lineNumber : 0,
      timestamp: typeof entry.timestamp === "number" ? entry.timestamp : Date.now(),
    }, buffer.generation);
  }
}

interface StackFrame {
  url?: unknown;
  lineNumber?: unknown;
}

function topFrame(stackTrace: unknown): { url: string; line: number } {
  const frames =
    stackTrace && typeof stackTrace === "object"
      ? ((stackTrace as Record<string, unknown>).callFrames as StackFrame[] | undefined)
      : undefined;
  const frame = frames?.[0];
  return {
    url: typeof frame?.url === "string" ? frame.url : "",
    line: typeof frame?.lineNumber === "number" ? frame.lineNumber : 0,
  };
}

let listenerBound = false;

/**
 * Begin capturing logs for a tab. Called from `open` right after the debugger
 * attaches, so buffering starts before the destination page runs any script.
 * Idempotent: enabling a domain twice is harmless, and re-registering the
 * global listener is guarded so a worker that survived keeps one listener.
 */
export async function startLogCapture(
  tabId: number,
  cdp: (tabId: number, method: string, params?: Record<string, unknown>) => Promise<unknown>,
): Promise<void> {
  if (!listenerBound) {
    chrome.debugger.onEvent.addListener(onEvent);
    listenerBound = true;
  }
  // Create the buffer BEFORE enabling the domains, so an event that fires
  // between the two enable calls is not dropped by the tabId guard in onEvent.
  const fresh = !buffers.has(tabId);
  const buffer = bufferFor(tabId);
  if (fresh) {
    // The main frame id labels each document's origin (ORIGIN TAGS). It is
    // the tab's target id and survives navigations, so one read per buffer.
    // Unreadable -> stays "" and every entry fails closed.
    try {
      const tree = (await cdp(tabId, "Page.getFrameTree")) as { frameTree?: { frame?: { id?: unknown } } } | undefined;
      const id = tree?.frameTree?.frame?.id;
      if (typeof id === "string") buffer.mainFrameId = id;
    } catch { /* fail closed: no document origin, no entries returned */ }
    // A NEW buffer on a session whose Runtime domain is already on (a resume
    // after the worker restarted while Chrome kept the debugger session):
    // V8 answers a repeated Runtime.enable without replaying
    // executionContextCreated, so the buffer would never learn the current
    // document's origin and every log would be dropped until the next
    // navigation (review m3/Q4). Disable first to force the replay. Only for a
    // fresh buffer: on a live one the replay would duplicate console history.
    // On a brand-new session the disable is a harmless no-op.
    try { await cdp(tabId, "Runtime.disable"); } catch { /* not enabled yet */ }
  }
  await cdp(tabId, "Runtime.enable");
  await cdp(tabId, "Log.enable");
}

/**
 * Filter+limit a list of entries. Pure (no chrome, no buffer state) so the
 * level/limit contract is unit-testable without a debugger session. `level`
 * "all" keeps everything; `limit` > 0 keeps the most recent n, preserving
 * oldest→newest order.
 */
export function filterEntries(entries: LogEntry[], level: string, limit: number): LogEntry[] {
  const filtered = level === "all" ? entries : entries.filter((entry) => entry.level === level);
  return limit > 0 && filtered.length > limit ? filtered.slice(filtered.length - limit) : filtered;
}

/** The distinct origins a tab's buffered entries came from, so the caller can
 * decide admission once per origin (an async storage read) rather than per
 * entry. */
export function loggedOrigins(tabId: number): string[] {
  const buffer = buffers.get(tabId);
  if (!buffer) return [];
  return [...new Set(buffer.entries.map((tagged) => originOf(buffer, tagged.generation)))];
}

/** Read filtered entries for a tab, newest-last, keeping only entries whose
 * capture origin is in `admitted` (see ORIGIN TAGS above). `level` "all"
 * keeps every level; `limit` applies after the origin filter. */
export function readLogs(tabId: number, level: string, limit: number, admitted: ReadonlySet<string>): LogEntry[] {
  const buffer = buffers.get(tabId);
  if (!buffer) return filterEntries([], level, limit);
  const entries = buffer.entries
    .filter((tagged) => {
      const origin = originOf(buffer, tagged.generation);
      return origin !== "" && admitted.has(origin);
    })
    .map((tagged) => tagged.entry);
  return filterEntries(entries, level, limit);
}

/** Drop a tab's buffer when its surface closes so a stale tabId cannot leak. */
export function dropLogCapture(tabId: number): void {
  buffers.delete(tabId);
}
