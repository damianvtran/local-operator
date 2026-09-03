import { getSurfaces, putSurface, resolveSurfaceToken, type StoredSurface } from "./state";

const GROUP_PREFIX = "LO · ";
const FALLBACK_LABEL = "Session";
const MAX_LABEL_CLUSTERS = 30;
const NO_GROUP = -1;

/** Browser grouping is presentation only: every failure is swallowed so a
 * missing, partial, or policy-disabled Chromium API can never block browsing. */
let groupQueue: Promise<unknown> = Promise.resolve();
function serialize<T>(op: () => Promise<T>): Promise<T> {
  const run = groupQueue.catch(() => {}).then(op);
  groupQueue = run;
  return run;
}

function cleanLabel(value: unknown): string {
  const raw = typeof value === "string" ? value : "";
  // Defense in depth for older or non-Python bridge clients. C0/C1/DEL,
  // direction controls, and zero-width format controls are all Unicode Cc/Cf.
  const clean = raw
    .replace(/\p{Cf}/gu, "")
    .replace(/\p{Cc}/gu, (character) => (/\s/u.test(character) ? " " : ""))
    .replace(/\s+/gu, " ")
    .trim();
  if (!clean) return FALLBACK_LABEL;
  const segments = typeof Intl.Segmenter === "function"
    ? [...new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(clean)].map((part) => part.segment)
    : [...clean];
  if (segments.length <= MAX_LABEL_CLUSTERS) return clean;
  const hard = segments.slice(0, MAX_LABEL_CLUSTERS).join("").trimEnd();
  const boundary = hard.lastIndexOf(" ");
  return `${boundary >= 8 ? hard.slice(0, boundary).trimEnd() : hard}…`;
}

function trustedOwner(params: Record<string, unknown>): string {
  const requester = typeof params.requester === "string" ? params.requester.trim() : "";
  return requester.startsWith("session:") ? requester : "";
}

function baseTitle(params: Record<string, unknown>): string {
  return `${GROUP_PREFIX}${cleanLabel(params.session_label)}`;
}

function appliedTitle(base: string, ordinal: number): string {
  return ordinal > 1 ? `${base} (${ordinal})` : base;
}

async function tabOrUndefined(tabId: number): Promise<chrome.tabs.Tab | undefined> {
  try { return await chrome.tabs.get(tabId); } catch { return undefined; }
}

async function isAppliedGroup(surface: StoredSurface, groupId: number): Promise<boolean> {
  if (surface.appliedGroupId !== groupId || !surface.groupAppliedLabel) return false;
  try {
    const group = await chrome.tabGroups.get(groupId);
    // IDs can be recycled after browser/worker churn. Matching the exact title
    // and colour LO last applied prevents a recycled personal group with the
    // same numeric ID from becoming writable through stale session storage.
    return group.title === surface.groupAppliedLabel && group.color === "cyan";
  } catch {
    return false;
  }
}

async function allocateTitle(
  surface: StoredSurface,
  ownerKey: string,
  wantedBase: string,
): Promise<{ base: string; ordinal: number; title: string }> {
  if (surface.ownerKey === ownerKey && surface.groupBaseLabel === wantedBase && surface.groupOrdinal) {
    return {
      base: wantedBase,
      ordinal: surface.groupOrdinal,
      title: appliedTitle(wantedBase, surface.groupOrdinal),
    };
  }

  const surfaces = Object.values(await getSurfaces());
  const sameOwner = surfaces.find((candidate) =>
    candidate !== surface && candidate.ownerKey === ownerKey && candidate.groupBaseLabel === wantedBase && candidate.groupOrdinal
  );
  if (sameOwner?.groupOrdinal) {
    return {
      base: wantedBase,
      ordinal: sameOwner.groupOrdinal,
      title: appliedTitle(wantedBase, sameOwner.groupOrdinal),
    };
  }

  const used = new Set(
    surfaces
      .filter((candidate) => candidate.ownerKey !== ownerKey && candidate.groupBaseLabel === wantedBase)
      .map((candidate) => candidate.groupOrdinal)
      .filter((ordinal): ordinal is number => typeof ordinal === "number" && ordinal > 0),
  );
  let ordinal = 1;
  while (used.has(ordinal)) ordinal += 1;
  return { base: wantedBase, ordinal, title: appliedTitle(wantedBase, ordinal) };
}

async function existingSiblingGroup(
  surface: StoredSurface,
  ownerKey: string,
  windowId: number,
): Promise<number | undefined> {
  for (const candidate of Object.values(await getSurfaces())) {
    if (candidate.tabId === surface.tabId || candidate.ownerKey !== ownerKey) continue;
    const tab = await tabOrUndefined(candidate.tabId);
    if (
      tab?.windowId === windowId
      && typeof tab.groupId === "number"
      && tab.groupId !== NO_GROUP
      && await isAppliedGroup(candidate, tab.groupId)
    ) {
      return tab.groupId;
    }
  }
  return undefined;
}

/** Best-effort ordinary-command hook. Exact token lookup preserves the same
 * capability boundary as command dispatch without changing recency or errors. */
export async function reconcileCommandTab(params: Record<string, unknown>): Promise<void> {
  try {
    const surface = resolveSurfaceToken(params.tab, await getSurfaces());
    if (surface) await reconcileTabGroup(surface, params, false);
  } catch {
    // The real command remains authoritative for stale-handle diagnostics.
  }
}

/**
 * `retitle` — rename this session's group after its title arrives.
 *
 * A conversation names itself asynchronously, usually AFTER the opening
 * command created the group, and a session that only opens, screenshots and
 * closes issues no later command for `reconcileCommandTab` to ride. The host
 * therefore pushes the new label explicitly.
 *
 * NOT `explicit`: this carries no navigation intent, so it must obey the same
 * rule every ordinary command does — a tab the user has since moved into a
 * personal group is left alone rather than being pulled back out by a rename.
 * Only open/resume may re-home a tab.
 *
 * Exact-token lookup is the capability boundary: a session can only rename the
 * group of a surface whose full handle it holds, and `trustedOwner` still
 * requires the daemon-supplied `session:` requester, so the label cannot be
 * pushed onto another session's group.
 *
 * Returns the applied label purely as an observability answer for the daemon;
 * the empty string means "nothing was renamed", never an error, because
 * grouping is presentation and this command has no failure the caller can act
 * on.
 */
export async function retitle(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  try {
    const surface = resolveSurfaceToken(params.tab, await getSurfaces());
    if (!surface) return { title: "" };
    await reconcileTabGroup(surface, params, false);
    // Re-read: reconcileTabGroup persists through its own serialized queue, so
    // the in-hand object may predate the write it just made.
    const updated = resolveSurfaceToken(params.tab, await getSurfaces());
    const applied = updated?.groupAppliedLabel ?? "";
    // Report the label only when it is the one this call asked for. On the
    // user-moved-tab path the reconcile returns early WITHOUT clearing
    // `groupAppliedLabel`, so the stored value is the previous title — echoing
    // it would report a rename that deliberately did not happen, contradicting
    // the contract above (QA round 1, Q3). Comparing against the wanted title
    // keeps "" meaning exactly "nothing was renamed".
    const wanted = appliedTitle(baseTitle(params), updated?.groupOrdinal ?? 1);
    return { title: applied === wanted ? applied : "" };
  } catch {
    return { title: "" };
  }
}

/** Reconcile trusted session metadata with native browser chrome.
 *
 * `explicit` is reserved for open/resume. Ordinary commands update the title
 * only while the live group still matches the advisory ID LO persisted when it
 * created/joined that group. A mismatch means the user moved the tab into a
 * personal group, which must remain untouched. Open/resume may remove the tab
 * from that personal group by creating/joining an LO-owned group, but it never
 * updates the personal group itself. IDs remain advisory because Chromium can
 * recycle them; a stale mismatch therefore fails safely until explicit resume.
 */
export function reconcileTabGroup(
  surface: StoredSurface,
  params: Record<string, unknown>,
  explicit: boolean,
): Promise<void> {
  return serialize(async () => {
    try {
      if (typeof chrome.tabs.group !== "function" || typeof chrome.tabGroups?.update !== "function") return;
      const ownerKey = trustedOwner(params);
      if (!ownerKey) return;
      const tab = await tabOrUndefined(surface.tabId);
      if (!tab || tab.windowId === undefined) return;

      const allocation = await allocateTitle(surface, ownerKey, baseTitle(params));
      surface.ownerKey = ownerKey;
      surface.groupBaseLabel = allocation.base;
      surface.groupOrdinal = allocation.ordinal;

      const grouped = typeof tab.groupId === "number" && tab.groupId !== NO_GROUP;
      const stillOwned = grouped && await isAppliedGroup(surface, tab.groupId);
      if (!explicit && !stillOwned) {
        // Covers both manual ungrouping and a move into a personal group. Never
        // infer ownership from an LO-looking title: titles are user-controlled.
        await putSurface(surface);
        return;
      }

      let groupId = stillOwned ? tab.groupId : undefined;
      let created = false;
      if (groupId === undefined) {
        const sibling = await existingSiblingGroup(surface, ownerKey, tab.windowId);
        groupId = sibling === undefined
          ? await chrome.tabs.group({ tabIds: [surface.tabId] })
          : await chrome.tabs.group({ groupId: sibling, tabIds: [surface.tabId] });
        created = sibling === undefined;
      }

      // Omitting `collapsed` on updates preserves a user's collapsed group.
      await chrome.tabGroups.update(groupId, {
        title: allocation.title,
        color: "cyan",
        ...(created ? { collapsed: false } : {}),
      });
      surface.groupAppliedLabel = allocation.title;
      surface.appliedGroupId = groupId;
      await putSurface(surface);
    } catch {
      // Grouping is never part of the browsing success contract. This also
      // covers group success followed by update rejection on managed browsers.
    }
  });
}
