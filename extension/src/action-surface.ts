import type { QueueSnapshot } from "./approval-store";

export interface ActionSurfaceFailure {
  operation: string;
  reason: unknown;
}

export type ActionSurfaceObserver = (snapshot: QueueSnapshot) => void | Promise<void>;

/** Reconcile each ambient surface independently. Chrome action APIs may reject
 * transiently during worker startup; one failed cosmetic call must never stop
 * the numbered badge, tooltip, or notification observer from being attempted. */
export async function reconcileActionSurface(
  snapshot: QueueSnapshot,
  observer?: ActionSurfaceObserver,
): Promise<ActionSurfaceFailure[]> {
  const count = snapshot.queue.length;
  const badge = count > 9 ? "9+" : count ? String(count) : "";
  const title =
    count === 0
      ? "Local Operator"
      : count === 1
        ? "1 site request waiting"
        : `${count} site requests waiting`;
  const operations: Array<[string, () => Promise<unknown> | unknown]> = [
    ["badge background", () => chrome.action.setBadgeBackgroundColor({ color: "#b23a31" })],
    // Explicit white text keeps the count legible on the semantic danger red
    // across browser/OS themes. Older Chromium types may omit this API, so its
    // absence is treated like any other isolated surface failure.
    [
      "badge text color",
      () =>
        typeof chrome.action.setBadgeTextColor === "function"
          ? chrome.action.setBadgeTextColor({ color: "#ffffff" })
          : undefined,
    ],
    ["badge text", () => chrome.action.setBadgeText({ text: badge })],
    ["action title", () => chrome.action.setTitle({ title })],
    ["pending observer", () => observer?.(snapshot)],
  ];
  const settled = await Promise.allSettled(operations.map(([, operation]) => Promise.resolve().then(operation)));
  const failures: ActionSurfaceFailure[] = [];
  settled.forEach((result, index) => {
    if (result.status === "rejected") {
      const operation = operations[index]![0];
      failures.push({ operation, reason: result.reason });
      console.warn(`approval action surface failed: ${operation}`, result.reason);
    }
  });
  return failures;
}
