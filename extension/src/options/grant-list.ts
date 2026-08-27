import { loopbackHostGrantLabel, type StoredVerdict, validHostGrantSchema } from "../origin-policy";

export interface GrantRow {
  key: string;
  label: string;
  scope: "origin" | "host";
}

export function grantRows(
  origins: Record<string, StoredVerdict>,
  hostGrants: unknown,
): GrantRow[] {
  const exact = Object.entries(origins).flatMap(([origin, verdict]) =>
    verdict === "allow"
      ? [{ key: origin, label: `${origin} · this port`, scope: "origin" as const }]
      : [],
  );
  // Unknown/malformed metadata hides only broad records. Exact grants stay
  // manageable, and no unsupported schema is interpreted or rewritten.
  const broad = validHostGrantSchema(hostGrants)
    ? Object.entries(hostGrants.grants).flatMap(([key, grant]) => {
        const label = loopbackHostGrantLabel(key);
        return label && grant?.scope === "all_ports" && typeof grant.createdAt === "number"
          ? [{ key, label: `${label} · all ports`, scope: "host" as const }]
          : [];
      })
    : [];
  return [...exact, ...broad].sort((a, b) => a.label.localeCompare(b.label));
}

export function removeExactGrant(
  row: GrantRow,
  origins: Record<string, StoredVerdict>,
): Record<string, StoredVerdict> {
  const next = { ...origins };
  delete next[row.key];
  return next;
}
