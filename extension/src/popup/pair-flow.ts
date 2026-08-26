/* Pure decisions for the popup's pairing flow, kept DOM-free so the pure-node
 * test suite can exercise them (popup.ts touches `document` at module scope and
 * cannot be imported under node).
 *
 * The race these functions exist for: the daemon answers `pair_result.ok` on
 * the popup's own WebSocket the moment the code is accepted, but /health only
 * reports `paired: true` after the background worker reconnects with the new
 * token. In that window a render() driven by /health would put the pairing form
 * back on screen; the user, seeing no feedback, re-submits the consumed
 * one-time code and gets a misleading "No live pairing code" error. Success
 * must therefore render from the pair_result frame ALONE and never be
 * downgraded by a stale health probe. */

/** The default mismatch copy (branding doc's pairing-error microcopy). */
export const PAIR_MISMATCH_MESSAGE =
  "That code didn't match. Codes expire after two minutes — check the app for a fresh one.";

export interface PairResultFrame {
  event?: string;
  ok?: boolean;
  token?: string;
  message?: string;
}

export type PairVerdict = { ok: true; token: string } | { ok: false; message: string };

/** Fold a pair_result frame into exactly one of two outcomes. `ok` without a
 * token is treated as a failure: a success we cannot store a credential for
 * would render "Patched in." over a connection that can never authenticate. */
export function pairVerdict(frame: PairResultFrame): PairVerdict {
  if (frame.ok && frame.token) return { ok: true, token: frame.token };
  return { ok: false, message: frame.message ?? PAIR_MISMATCH_MESSAGE };
}

/** Which view a reachable-daemon render shows. `locallyPaired` records that
 * THIS popup already saw pair_result.ok, so a health probe that has not caught
 * up yet ("the race") holds the explicit success state instead of falling back
 * to the form. */
export function viewForHealth(
  healthPaired: boolean,
  locallyPaired: boolean,
): "connected" | "paired" | "pairing" {
  if (healthPaired) return "connected";
  return locallyPaired ? "paired" : "pairing";
}
