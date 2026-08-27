export type StoredVerdict = "allow" | "deny";

export interface HostGrantOperation {
  canonicalKey: string;
  action: "grant" | "revoke";
  at: number;
}

/** The schema marker is deliberately separate from append-only operations.
 * Each decision writes a unique chrome.storage key, so worker and Settings
 * contexts never race by rewriting shared state. */
export interface HostGrantsState {
  version: 1;
}

export type GrantScope = "origin" | "loopback_all_ports";
export const HOST_GRANT_STORAGE_PREFIX = "hostGrantOp:v1:";

export function safeHttpUrl(raw: unknown): URL {
  if (typeof raw !== "string") throw new Error("URL is required");
  const parsed = new URL(raw);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("only http:// and https:// can be opened");
  }
  // WHATWG URL turns IPv4 shorthand such as 127.1 into 127.0.0.1. That is
  // convenient for navigation but unsafe for a grant classifier whose contract
  // explicitly admits only the literal loopback identities. Compare the raw
  // host token before returning the normalized URL so shorthand cannot acquire
  // the exact host's authority.
  if (parsed.hostname === "127.0.0.1") {
    const authority = raw.match(/^https?:\/\/([^/?#]+)/i)?.[1] ?? "";
    const rawHost = authority.startsWith("[")
      ? authority.slice(0, authority.indexOf("]") + 1)
      : authority.split(":", 1)[0];
    if (rawHost !== "127.0.0.1") throw new Error("IPv4 loopback must use 127.0.0.1 exactly");
  }
  return parsed;
}

/** Deliberately literal, not a DNS or subnet test. Names that happen to resolve
 * to loopback, shorthand IPv4, mapped IPv6, and localhost subdomains do not
 * earn the broader grant. URL parsing lowercases localhost and preserves IPv6
 * brackets; a trailing dot remains distinct and therefore fails closed. */
export function isLoopbackHost(url: URL): boolean {
  return url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "[::1]";
}

/** URL.host is the browser's normalized display authority. It includes a
 * nondefault port and correctly brackets IPv6 without hand-built syntax. */
export function displayAuthority(url: URL): string {
  return url.host;
}

/** JSON's array encoding is a structured, collision-free serializer for the
 * normalized scheme and exact hostname. The port is intentionally absent. */
export function loopbackHostGrantKey(url: URL): string | null {
  if (!isLoopbackHost(url)) return null;
  return JSON.stringify([url.protocol, url.hostname]);
}

export function hostGrantOperationStorageKey(operationId: string): string {
  return `${HOST_GRANT_STORAGE_PREFIX}${operationId}`;
}

export function loopbackHostGrantLabel(key: string): string | null {
  try {
    const parsed: unknown = JSON.parse(key);
    if (!Array.isArray(parsed) || parsed.length !== 2) return null;
    const [protocol, hostname] = parsed;
    if (typeof protocol !== "string" || typeof hostname !== "string") return null;
    const url = new URL(`${protocol}//${hostname}/`);
    if (!isLoopbackHost(url) || loopbackHostGrantKey(url) !== key) return null;
    return `${url.protocol}//${url.host}`;
  } catch {
    return null;
  }
}

export function validHostGrantSchema(value: unknown): value is HostGrantsState {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    (value as { version?: unknown }).version === 1;
}

export function validHostGrantOperation(value: unknown): value is HostGrantOperation {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const operation = value as Partial<HostGrantOperation>;
  return typeof operation.canonicalKey === "string" &&
    loopbackHostGrantLabel(operation.canonicalKey) !== null &&
    (operation.action === "grant" || operation.action === "revoke") &&
    typeof operation.at === "number" && Number.isFinite(operation.at);
}

/** Latest logical operation for one authority. Operation time is captured
 * before async storage begins; therefore a revoke started after an approval
 * wins even if the earlier approval's write physically lands last. Ties fail
 * closed by preferring revoke. */
export function latestHostGrantOperation(
  local: Record<string, unknown>,
  canonicalKey: string,
): HostGrantOperation | null {
  if (!validHostGrantSchema(local.hostGrants)) return null;
  let latest: HostGrantOperation | null = null;
  for (const [storageKey, value] of Object.entries(local)) {
    if (!storageKey.startsWith(HOST_GRANT_STORAGE_PREFIX) || !validHostGrantOperation(value)) continue;
    if (value.canonicalKey !== canonicalKey) continue;
    if (!latest || value.at > latest.at || (value.at === latest.at && value.action === "revoke")) {
      latest = value;
    }
  }
  return latest;
}

export function matchingGrantScope(
  origins: Record<string, StoredVerdict>,
  local: Record<string, unknown>,
  url: URL,
): GrantScope | null {
  if (origins[url.origin] === "allow") return "origin";
  const key = loopbackHostGrantKey(url);
  if (!key) return null;
  return latestHostGrantOperation(local, key)?.action === "grant" ? "loopback_all_ports" : null;
}

export function storedOriginAllowed(
  origins: Record<string, StoredVerdict>,
  url: URL,
  local: Record<string, unknown> = {},
): boolean {
  return matchingGrantScope(origins, local, url) !== null;
}
