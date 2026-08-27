export type StoredVerdict = "allow" | "deny";

export interface HostGrant {
  scope: "all_ports";
  createdAt: number;
}

/** Versioned separately from exact-origin decisions so older extensions keep
 * enforcing the origin allowlist they understand instead of misreading a
 * broader authority. Unknown versions deliberately fail closed. */
export interface HostGrantsState {
  version: 1;
  grants: Record<string, HostGrant>;
}

export type GrantScope = "origin" | "loopback_all_ports";

export function safeHttpUrl(raw: unknown): URL {
  if (typeof raw !== "string") throw new Error("URL is required");
  const parsed = new URL(raw);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("only http:// and https:// can be opened");
  }
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

export function displayAuthority(url: URL): string {
  return url.host;
}

export function loopbackHostGrantKey(url: URL): string | null {
  if (!isLoopbackHost(url)) return null;
  return JSON.stringify([url.protocol, url.hostname]);
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
    (value as { version?: unknown }).version === 1 &&
    !!(value as { grants?: unknown }).grants &&
    typeof (value as { grants?: unknown }).grants === "object" &&
    !Array.isArray((value as { grants?: unknown }).grants);
}

export function matchingGrantScope(
  origins: Record<string, StoredVerdict>,
  hostGrants: unknown,
  url: URL,
): GrantScope | null {
  if (origins[url.origin] === "allow") return "origin";
  if (!validHostGrantSchema(hostGrants)) return null;
  const key = loopbackHostGrantKey(url);
  if (!key) return null;
  const grant = hostGrants.grants[key];
  return grant?.scope === "all_ports" && typeof grant.createdAt === "number"
    ? "loopback_all_ports"
    : null;
}

export function storedOriginAllowed(
  origins: Record<string, StoredVerdict>,
  url: URL,
  hostGrants?: unknown,
): boolean {
  return matchingGrantScope(origins, hostGrants, url) !== null;
}
