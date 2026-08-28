export interface MutationResult {
  ok: boolean;
  message: string;
}

/** Runtime messages can reject when Chrome is restarting the worker, and a
 * negative acknowledgement means persistence did not happen. Both must render
 * as visible status instead of falling out of an async event handler. */
export async function runWorkerMutation(
  message: Record<string, unknown>,
  success: string,
): Promise<MutationResult> {
  try {
    const response = (await chrome.runtime.sendMessage(message)) as { applied?: boolean } | undefined;
    return response?.applied
      ? { ok: true, message: success }
      : { ok: false, message: "Could not update site access. Try again." };
  } catch {
    return { ok: false, message: "Could not reach the extension worker. Try again." };
  }
}
