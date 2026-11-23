/**
 * Fired (bubbling) on a row whose height changed AFTER it first painted, so the
 * scroller above it can decide whether to keep its reader at the tail. Only the
 * transcript listens; everywhere else it is a no-op.
 */
export const CONTENT_RESIZED_EVENT = "lo:content-resized";
