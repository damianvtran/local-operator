import type { TranscriptEntry } from "../types";
import { quietGroupsOf, quietSenderLabel, type QuietGroup } from "./quiet-groups";

const object = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Additional CLOSED proof, not a lower global quiet threshold. Only updated
 * owners negotiate this capability and strip the reserved raw-details key.
 * Missing/foreign/partial evidence leaves the unchanged default scanner alone.
 */
export function contextualQuietGroupsOf(
  rows: readonly TranscriptEntry[], sessionId: string, negotiated = false,
): QuietGroup[] {
  const defaults = quietGroupsOf(rows);
  if (!negotiated || !sessionId) return defaults;
  const claimed = new Set<string>();
  const certified: QuietGroup[] = [];
  for (let start = 0; start < rows.length; start++) {
    const first = rows[start];
    const stamp: unknown = first.details?.peer_cycle;
    if (!object(stamp) || stamp.version !== 1 || stamp.session_id !== sessionId ||
        first.kind !== "peer_message" || stamp.first_id !== first.id ||
        typeof stamp.quiet_call_id !== "string" || !stamp.quiet_call_id ||
        !Array.isArray(stamp.member_ids) || stamp.member_ids.length < 2 ||
        stamp.member_ids.length > 4096 ||
        !stamp.member_ids.every((id: unknown) => typeof id === "string" && id.length > 0)) continue;
    const ids: string[] = stamp.member_ids;
    if (new Set(ids).size !== ids.length || stamp.last_id !== ids[ids.length - 1] ||
        ids.some(id => claimed.has(id)) || start + ids.length > rows.length) continue;
    const members = rows.slice(start, start + ids.length);
    if (members.some((row, index) => row.id !== ids[index] || row.text_complete === false ||
        row.images?.length || row.videos?.length || row.error ||
        row.details?.held === true || row.details?.steer === true ||
        row.details?.severity === "error" || row.details?.severity === "warning" ||
        (index > 0 && (row.kind !== "tool" || row.tool_state !== "done" ||
          (row.tool_name !== "send" && row.tool_name !== "eval"))))) continue;
    const calls = members.slice(1).map(row => row.tool_call_id);
    if (calls.some(id => !id) || new Set(calls).size !== calls.length ||
        ids.some(id => rows.filter(row => row.id === id).length !== 1)) continue;
    const sender = object(first.details?.sender) ? first.details.sender : {};
    const label = quietSenderLabel({
      ...(typeof sender.conversation_name === "string" ? { conversation_name: sender.conversation_name } : {}),
      ...(typeof sender.cwd === "string" ? { cwd: sender.cwd } : {}),
      ...(typeof sender.session_id === "string" ? { session_id: sender.session_id } : {}),
      ...(typeof sender.pid === "number" ? { pid: sender.pid } : {}),
    });
    certified.push({ key: `qg:${first.id}`, family: "peer", count: 1,
      firstTs: null, lastTs: null, senders: [{ label, count: 1 }],
      // Eval carriers are physical disclosure rows, not additional sends.
      actions: members.filter(row => row.kind === "tool" && row.tool_name === "send").length,
      failed: 0, open: false, rowIds: ids });
    for (const id of ids) claimed.add(id);
  }
  const positions = new Map(rows.map((row, index) => [row.id, index]));
  return [...defaults.filter(group => !group.rowIds.some(id => claimed.has(id))), ...certified]
    .sort((left, right) => (positions.get(left.rowIds[0]) ?? 0) - (positions.get(right.rowIds[0]) ?? 0));
}
