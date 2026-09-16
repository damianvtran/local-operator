import type { LocalState, SessionState } from "../state";
import { CHROME_API_DEADLINE_MS, deadline } from "../settle";

// This is a read projection, not another authority/cache. Snapshot refs and
// grant maps can dwarf the UI state and must not cross the storage IPC boundary
// just to name the current card. Keep security mutations on their existing lane.
export type PopupSession = Pick<SessionState, "accessQueue" | "pendingOrigin" | "surfaces"> & {
  connState?: string;
  revoked?: boolean;
};
export async function getPopupSession(): Promise<PopupSession> {
  return deadline(
    chrome.storage.session.get(["accessQueue", "pendingOrigin", "surfaces", "connState", "revoked"]),
    CHROME_API_DEADLINE_MS,
    "chrome.storage.session.get(popup state)",
  );
}
export async function getPopupLocal(): Promise<Pick<LocalState, "port" | "allowAllSites">> {
  return deadline(
    chrome.storage.local.get(["port", "allowAllSites"]),
    CHROME_API_DEADLINE_MS,
    "chrome.storage.local.get(popup state)",
  );
}
