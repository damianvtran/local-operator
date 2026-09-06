/* Pure state machine for the Settings "Dangerously allow all websites"
 * switch, DOM- and chrome-free for the pure-node test suite (the same pattern
 * as mutation-flow.ts).
 *
 * Why a state machine for one checkbox: the switch must NOT take effect on
 * the click. Checking it opens a confirmation dialog, and only an explicit
 * Enable after an "I understand the risks" acknowledgement writes the
 * setting; Cancel or Escape reverts the switch and writes nothing. Turning
 * it OFF is immediate, because narrowing access never needs a warning. The
 * write itself is `chrome.storage.local.set({ allowAllSites })` from the
 * options page, deliberately with no worker message or daemon RPC behind it,
 * so nothing an agent can reach flips this. */

export type AllowAllAction =
  | { type: "toggle"; checked: boolean }
  | { type: "ack"; checked: boolean }
  | { type: "enable" }
  | { type: "cancel" }
  | { type: "turn_off" };

export interface AllowAllView {
  /** What the switch shows. */
  switchOn: boolean;
  dialogOpen: boolean;
  /** The dialog's acknowledgement checkbox; Enable is gated on it. */
  acked: boolean;
  /** The persistent danger banner above the cards. */
  banner: boolean;
  /** The value to persist, or undefined when this action writes nothing. */
  write?: boolean;
}

export function allowAllView(stored: boolean, dialogOpen = false, acked = false): AllowAllView {
  return { switchOn: stored || dialogOpen, dialogOpen, acked, banner: stored };
}

export function nextAllowAllView(stored: boolean, action: AllowAllAction, current?: AllowAllView): AllowAllView {
  const view = current ?? allowAllView(stored);
  switch (action.type) {
    case "toggle":
      // Checking opens the dialog and shows the switch on optimistically so
      // the dialog reads as confirming the click; nothing is written yet.
      if (action.checked && !stored) return allowAllView(false, true, false);
      if (!action.checked && stored) return { ...allowAllView(false), write: false };
      return allowAllView(stored);
    case "ack":
      return { ...view, acked: action.checked };
    case "enable":
      if (!view.dialogOpen || !view.acked) return view;
      return { ...allowAllView(true), write: true };
    case "cancel":
      // Escape or Cancel: revert the optimistic switch, write nothing.
      return allowAllView(stored);
    case "turn_off":
      return { ...allowAllView(false), write: false };
  }
}
