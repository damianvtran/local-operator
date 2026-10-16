import { describe, expect, it } from "vitest";
import corpus from "./peer-cycles.context.json";
import { contextualQuietGroupsOf } from "./peer-cycles";
import { quietGroupsOf } from "./quiet-groups";
import type { TranscriptEntry } from "../types";

type Raw = { id: string; kind: string; text?: string; tool_call_id?: string; tool_name?: string };
function row(raw: Raw): TranscriptEntry {
  if (raw.kind !== "peer_message" && raw.kind !== "tool" && raw.kind !== "assistant") throw new Error("unknown corpus kind");
  return { id: raw.id, kind: raw.kind, text: raw.text ?? "original payload", tool_call_id: raw.tool_call_id ?? "",
    tool_name: raw.tool_name ?? "", tool_state: "done", summary: "", intent: "", diff_added: 0,
    diff_removed: 0, elapsed_s: 0, error: "", details: {}, final: true, text_complete: true };
}
function owned(test: (typeof corpus.cases)[number]) {
  const rows = test.rows.map(row);
  rows[0].details.peer_cycle = { version: 1, session_id: corpus.session_id, first_id: test.members[0],
    last_id: test.members.at(-1), member_ids: [...test.members], quiet_call_id: "actual-typed-quiet" };
  return rows;
}
describe("owner-certified closed peer graphs", () => {
  for (const test of corpus.cases) {
    it(`${test.name}: exact physical rows, logical actions and public final outside`, () => {
      const rows = owned(test); const original = JSON.stringify(rows);
      const groups = contextualQuietGroupsOf(rows, corpus.session_id, true);
      expect(groups).toHaveLength(1); expect(groups[0].rowIds).toEqual(test.members);
      expect(groups[0].actions).toBe(test.actions); expect(groups[0].count).toBe(1);
      expect(groups[0].rowIds).not.toContain("public-final");
      expect(JSON.stringify(rows)).toBe(original);
    });
    it(`${test.name}: raw marker without negotiation and foreign session do not grant proof`, () => {
      const rows = owned(test);
      expect(contextualQuietGroupsOf(rows, corpus.session_id, false)).toEqual(quietGroupsOf(rows));
      expect(contextualQuietGroupsOf(rows, "foreign-session", true)).toEqual(quietGroupsOf(rows));
    });
    it(`${test.name}: gap, mutation, partial body and attention refuse the contextual exception`, () => {
      const mutations = [
        (rows: TranscriptEntry[]) => rows.splice(1, 1),
        (rows: TranscriptEntry[]) => { rows[1].tool_state = "failed"; },
        (rows: TranscriptEntry[]) => { rows[1].text_complete = false; },
        (rows: TranscriptEntry[]) => { rows[1].details.held = true; },
        (rows: TranscriptEntry[]) => { rows.splice(1, 0, { ...row({id:"boundary",kind:"assistant"}), text:"material" }); },
      ];
      for (const mutate of mutations) {
        const rows = owned(test); mutate(rows);
        // The legacy scanner remains authoritative for its own thresholds;
        // refusing this exception must not claim a new logical-action count.
        expect(contextualQuietGroupsOf(rows, corpus.session_id, true)).toEqual(quietGroupsOf(rows));
      }
    });
  }
});
