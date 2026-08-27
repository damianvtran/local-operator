import {
  HOST_GRANT_STORAGE_PREFIX,
  latestHostGrantOperation,
  loopbackHostGrantLabel,
  type StoredVerdict,
  validHostGrantOperation,
  validHostGrantSchema,
} from "../origin-policy";

export interface GrantRow {
  key: string;
  label: string;
  scope: "origin" | "host";
}

export function grantRows(
  origins: Record<string, StoredVerdict>,
  local: Record<string, unknown>,
): GrantRow[] {
  const exact = Object.entries(origins).flatMap(([origin, verdict]) =>
    verdict === "allow"
      ? [{ key: origin, label: `${origin} · this port`, scope: "origin" as const }]
      : [],
  );
  // Unknown/malformed metadata hides only broad records. Exact grants stay
  // manageable, and no unsupported schema is interpreted or rewritten.
  const canonicalKeys = validHostGrantSchema(local.hostGrants)
    ? new Set(
        Object.entries(local).flatMap(([storageKey, value]) =>
          storageKey.startsWith(HOST_GRANT_STORAGE_PREFIX) && validHostGrantOperation(value)
            ? [value.canonicalKey]
            : [],
        ),
      )
    : new Set<string>();
  const broad = [...canonicalKeys].flatMap((key) => {
    if (latestHostGrantOperation(local, key)?.action !== "grant") return [];
    const label = loopbackHostGrantLabel(key);
    return label ? [{ key, label: `${label} · all ports`, scope: "host" as const }] : [];
  });
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
