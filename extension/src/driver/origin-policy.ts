import { PSL_RULES } from "./psl.gen";

export type StoredVerdict = "allow" | "deny";

/** Legacy (0.1.4 to 0.1.7) loopback all-port grant, keyed
 * `[protocol, hostname]` and same-scheme only. Never rewritten or widened:
 * new approvals write `siteGrants` instead (see SiteGrantsState), and the v1
 * records stay readable so a downgrade does not lose them. */
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

/** A broad grant from the popup's "All pages on this domain" option.
 *
 * `domain`: key is the bare lowercase ASCII registrable domain
 * (`gominerva.com`); covers every subdomain, both schemes, any port.
 * `host`: key is a literal loopback hostname (`localhost`, `127.0.0.1`,
 * `[::1]`); covers any port on both schemes. Loopback has no registrable
 * domain, so it gets its own scope rather than a fake one. */
export interface SiteGrant {
  scope: "domain" | "host";
  createdAt: number;
}

/** Stored under its OWN key (`chrome.storage.local.siteGrants`) rather than
 * as a `hostGrants` v2: an older extension ignores a key it never reads, which
 * is fail-closed for free on downgrade, and the v1 loopback records are never
 * migrated, so no existing grant is silently widened from same-scheme to
 * both schemes. The cost is two broad-grant shapes coexisting read-only. */
export interface SiteGrantsState {
  version: 1;
  grants: Record<string, SiteGrant>;
}

/** The scope a stored grant admitted a URL under, in lookup order. */
export type GrantScope = "origin" | "domain" | "host" | "loopback_all_ports";

/** What the popup's broad option would grant for a URL, computed by the
 * worker at enqueue and stored on the queue entry: the popup never runs the
 * suffix list itself, so an entry without it simply has no domain option. */
export interface BroadGrant {
  scope: SiteGrant["scope"];
  key: string;
}

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

// ---- Registrable domain (Public Suffix List) ---------------------------------

let pslRules: Set<string> | null = null;

/** Parsed lazily: the popup and options bundles never call this, and the
 * worker only needs it on the first broad-grant computation. */
function rules(): Set<string> {
  if (!pslRules) pslRules = new Set(PSL_RULES.split("\n"));
  return pslRules;
}

function isIpLiteral(hostname: string): boolean {
  return hostname.startsWith("[") || /^\d{1,3}(\.\d{1,3}){3}$/.test(hostname);
}

/** The registrable domain (eTLD+1) of a URL's hostname, or null when there is
 * none to offer. Standard PSL matching: the longest matching rule among exact,
 * `*.` wildcard and `!` exception wins, with the implicit `*` rule as the
 * default; the public suffix is that rule (minus its first label for an
 * exception) and the registrable domain is the suffix plus one more label.
 *
 * Returns null, meaning NO domain option, for: IP literals (v4 dotted quad or
 * bracketed v6), a trailing-dot hostname, a single-label hostname, and a
 * hostname that IS a public suffix (`co.uk`, `github.io`, `com`), because a
 * grant keyed on any of those would cover an unbounded set of unrelated
 * sites. Loopback literals are handled by broadGrantFor before this runs.
 * `URL.hostname` is already lowercase punycode, so no normalisation here. */
export function registrableDomain(url: URL): string | null {
  const hostname = url.hostname;
  if (!hostname || hostname.endsWith(".") || isIpLiteral(hostname)) return null;
  const labels = hostname.split(".");
  if (labels.length < 2 || labels.some((label) => !label)) return null;
  const table = rules();
  // Public-suffix label count under the best matching rule. Default rule `*`
  // makes the last label the suffix.
  let suffixLabels = 1;
  for (let index = labels.length - 1; index >= 0; index -= 1) {
    const candidate = labels.slice(index).join(".");
    const matched = labels.length - index;
    // An exception rule names a hostname that is NOT a public suffix even
    // though a wildcard parent says it is: its suffix is one label shorter.
    if (table.has(`!${candidate}`)) {
      suffixLabels = matched - 1;
      break;
    }
    if (table.has(candidate)) suffixLabels = Math.max(suffixLabels, matched);
    // A wildcard rule `*.ck` matches one more label than itself; test it
    // against the parent so `foo.bar.ck` yields suffix `bar.ck`.
    if (index > 0 && table.has(`*.${candidate}`)) suffixLabels = Math.max(suffixLabels, matched + 1);
  }
  if (labels.length <= suffixLabels) return null;
  return labels.slice(labels.length - suffixLabels - 1).join(".");
}

/** The broad grant the popup may offer for a URL: loopback gets a host grant
 * (any port, both schemes), everything else its registrable domain. Null
 * means the popup shows only the exact-site and once options. */
export function broadGrantFor(url: URL): BroadGrant | null {
  if (isLoopbackHost(url)) return { scope: "host", key: url.hostname };
  const domain = registrableDomain(url);
  return domain ? { scope: "domain", key: domain } : null;
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
    if (protocol !== "http:" && protocol !== "https:") return null;
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

export function validSiteGrantSchema(value: unknown): value is SiteGrantsState {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    (value as { version?: unknown }).version === 1 &&
    !!(value as { grants?: unknown }).grants &&
    typeof (value as { grants?: unknown }).grants === "object" &&
    !Array.isArray((value as { grants?: unknown }).grants);
}

function siteGrantScope(grant: SiteGrant | undefined, scope: SiteGrant["scope"]): boolean {
  return grant?.scope === scope && typeof grant.createdAt === "number";
}

/** Lookup order: exact origin, then the new site grants (loopback host, then
 * registrable domain), then the legacy same-scheme loopback grant. An unknown
 * `siteGrants` version hides every site grant, so a record written by a newer
 * extension prompts again here rather than being misread (fail closed).
 * `origins[x] === "deny"` is typed but never written by decideAccess; it is
 * simply not an allow. */
export function matchingGrantScope(
  origins: Record<string, StoredVerdict>,
  hostGrants: unknown,
  url: URL,
  siteGrants?: unknown,
): GrantScope | null {
  if (origins[url.origin] === "allow") return "origin";
  if (validSiteGrantSchema(siteGrants)) {
    if (isLoopbackHost(url)) {
      if (siteGrantScope(siteGrants.grants[url.hostname], "host")) return "host";
    } else {
      const domain = registrableDomain(url);
      if (domain && siteGrantScope(siteGrants.grants[domain], "domain")) return "domain";
    }
  }
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
  siteGrants?: unknown,
): boolean {
  return matchingGrantScope(origins, hostGrants, url, siteGrants) !== null;
}
