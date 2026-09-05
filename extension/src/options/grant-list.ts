import {
  loopbackHostGrantLabel,
  type StoredVerdict,
  validHostGrantSchema,
  validSiteGrantSchema,
} from "../origin-policy";

/** One row in the Settings "Allowed sites" list. `origin` is an exact grant,
 * `domain` a registrable-domain grant, `host` a loopback any-port grant,
 * `legacy_host` a pre-0.1.8 same-scheme loopback all-port grant that is read
 * and revoked but never written any more. */
export interface GrantRow {
  key: string;
  label: string;
  scope: "origin" | "domain" | "host" | "legacy_host";
}

const SCOPE_SUFFIX: Record<GrantRow["scope"], string> = {
  origin: "this site",
  domain: "all subdomains, any port",
  host: "any port",
  legacy_host: "all ports (%s only)",
};

export function grantRows(
  origins: Record<string, StoredVerdict>,
  hostGrants: unknown,
  siteGrants?: unknown,
): GrantRow[] {
  const exact = Object.entries(origins).flatMap(([origin, verdict]) =>
    verdict === "allow"
      ? [{ key: origin, label: `${origin} · ${SCOPE_SUFFIX.origin}`, scope: "origin" as const }]
      : [],
  );
  // Unknown/malformed metadata hides only broad records. Exact grants stay
  // manageable, and no unsupported schema is interpreted or rewritten.
  const site: GrantRow[] = validSiteGrantSchema(siteGrants)
    ? Object.entries(siteGrants.grants).flatMap(([key, grant]): GrantRow[] => {
        if (typeof grant?.createdAt !== "number") return [];
        if (grant.scope === "domain") return [{ key, label: `${key} · ${SCOPE_SUFFIX.domain}`, scope: "domain" as const }];
        if (grant.scope === "host") return [{ key, label: `${key} · ${SCOPE_SUFFIX.host}`, scope: "host" as const }];
        return [];
      })
    : [];
  const legacy: GrantRow[] = validHostGrantSchema(hostGrants)
    ? Object.entries(hostGrants.grants).flatMap(([key, grant]): GrantRow[] => {
        const label = loopbackHostGrantLabel(key);
        if (!label || grant?.scope !== "all_ports" || typeof grant.createdAt !== "number") return [];
        const scheme = label.split(":", 1)[0] ?? "http";
        return [{ key, label: `${label} · ${SCOPE_SUFFIX.legacy_host.replace("%s", scheme)}`, scope: "legacy_host" as const }];
      })
    : [];
  return [...exact, ...site, ...legacy].sort((a, b) => a.label.localeCompare(b.label));
}

const SCOPE_NAME: Record<GrantRow["scope"], string> = {
  origin: "this-site",
  domain: "domain",
  host: "any-port",
  legacy_host: "legacy all-ports",
};

export function removeGrantAccessibleName(row: GrantRow): string {
  const authority = row.label.replace(/ · [^·]*$/, "");
  return `Remove ${SCOPE_NAME[row.scope]} grant for ${authority}`;
}

/** The worker message that revokes a row; each scope lives in its own
 * storage record, so each has its own revoke path. */
export function revokeMessageFor(row: GrantRow): Record<string, unknown> {
  switch (row.scope) {
    case "origin":
      return { event: "origin_grant_revoke", origin: row.key };
    case "legacy_host":
      return { event: "host_grant_revoke", canonicalKey: row.key };
    default:
      return { event: "site_grant_revoke", key: row.key };
  }
}

export function removeExactGrant(
  row: GrantRow,
  origins: Record<string, StoredVerdict>,
): Record<string, StoredVerdict> {
  const next = { ...origins };
  delete next[row.key];
  return next;
}
