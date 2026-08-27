import { loopbackHostGrantLabel, type HostGrantsState, type StoredVerdict } from "../origin-policy";

export interface GrantRow {
  key: string;
  label: string;
  scope: "origin" | "host";
}

export function grantRows(
  origins: Record<string, StoredVerdict>,
  hostGrants: HostGrantsState | undefined,
): GrantRow[] {
  const exact = Object.entries(origins).flatMap(([origin, verdict]) =>
    verdict === "allow"
      ? [{ key: origin, label: `${origin} · this port`, scope: "origin" as const }]
      : [],
  );
  // Unknown versions remain invisible and ineffective rather than being
  // interpreted as a shape this extension does not understand.
  const broad = hostGrants?.version === 1
    ? Object.keys(hostGrants.grants).flatMap((key) => {
        const label = loopbackHostGrantLabel(key);
        return label ? [{ key, label: `${label} · all ports`, scope: "host" as const }] : [];
      })
    : [];
  return [...exact, ...broad].sort((a, b) => a.label.localeCompare(b.label));
}

export function removeGrant(
  row: GrantRow,
  origins: Record<string, StoredVerdict>,
  hostGrants: HostGrantsState | undefined,
): { origins?: Record<string, StoredVerdict>; hostGrants?: HostGrantsState } {
  if (row.scope === "origin") {
    const next = { ...origins };
    delete next[row.key];
    return { origins: next };
  }
  if (hostGrants?.version !== 1) return {};
  const grants = { ...hostGrants.grants };
  delete grants[row.key];
  return { hostGrants: { version: 1, grants } };
}
