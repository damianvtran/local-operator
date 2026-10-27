# Vietnamese Notifications slash help — delta audit nt25a2

## Verdict
**Passed.** One changed key read: `wire.slash.notifications_page.help`. Zero unresolved blocker/major/minor/nit findings in this delta. Previous **V3 is fixed** in the current uncommitted catalogue bytes. This audit does not edit or commit the repair.

- English: `Asks and agent updates; read marks all read`
- Previous Vietnamese: `Câu hỏi và tin tác nhân; read: đánh dấu đã đọc`
- Current Vietnamese: `Câu hỏi, cập nhật; read: đánh dấu tất cả đã đọc`

The current value exactly matches nt25a1's proposed repair. `tất cả` explicitly restores the bulk scope of the literal `read` command. The concise noun list `Câu hỏi, cập nhật` is idiomatic command-help register and uses the UI's Cập nhật term instead of tin; the /notifications context supplies that these are agent-originated updates. The clause remains direct, sentence-case, and free of unnecessary address forms. No plural/format arguments exist in this one-key namespace; parser-derived argument tuples on both sides are empty.

## Measured execution
The repository's actual `parse_message`, `message_arguments`, and `render_message` functions ran on the changed input. The successful invocation exited 0. The rendered result equals the current catalogue value. Width is **47 cells against maxCells 50**; NFC and absence of combining/control codepoints were asserted. Cell count derives from Unicode East Asian width for this NFC Vietnamese/ASCII string, not from pixel measurement or a browser frame.

Four in-memory negative controls were rejected: the original missing-all wording, the repaired wording with literal read removed, an overwidth suffix, and malformed ICU. These targeted assertions validate the known repaired contract, not arbitrary natural-language meaning; linguistic judgment remains this auditor's review. No negative control touched a catalogue.

## Retained scope
Eight input SHA-256 comparisons passed: slash English/context, errors English/context/Vietnamese, and UI English/context/Vietnamese. The previous nt25a1 bundle's companion hashes also matched. Consequently the core errors pass (4 keys) and UI clean-with-findings (130 keys, two existing minor findings) are retained, not re-audited or relabeled clean. This round reads only the one changed help key; no new out-of-scope findings were sought.

## Provenance and limits
Auditor: translation-auditor (audit-vi, lopdev), session cfdff5945838. Actual model this resumed round: **openai/gpt-6-astra**, as explicitly supplied by the session-state update. Previous nt25a1 records openai/gpt-6.1-sol and is left unchanged; the model change is disclosed rather than rewriting history. This auditor is independent of the L1-core-notify coder/translator and authored neither translation nor remediation.

Only this new evidence bundle and own scratch were written. No catalogue, ledger, generated code, or git state was changed. No human native-speaker or browser/native rendered-fit certification is claimed.

## Reproduction
`audit.log` contains the actual successful stdout; `audit-script.py.txt` preserves the exact executed script without making the evidence tree executable discovery input. An isolated HOME/config and `env -i` stripped inherited CMUX/LOP variables; bytecode writing was disabled.

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/vi-nt25a2-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/vi-nt25a2-home/config" PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/workspace/repos/lo-notifications-l1 ~/local-operator/.venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/notifications-vi-nt25a2.py"
```
