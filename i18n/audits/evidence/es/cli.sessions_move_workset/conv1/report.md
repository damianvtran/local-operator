### Translation audit — convergence (merged namespace)

Reviewer: translation-auditor (lopdev team), session `e507d894afb3`, model anthropic/claude-opus-5-5, no fallback used.
Scope: `60dcbe088..7c7541d29`, which is merge `9729908d7` of origin/main `5e860469d` (#4380) plus pin fix `7c7541d29`. Worktree `/home/ec2-user/local-operator/.worktrees/move-dependency-recreate` @ `7c7541d29bddf2981d048b5ae4b6fc5aab030715`.

**Merged source**
- The repo's own `generate._en_sha('cli.sessions_move_workset')` returns `0f5602723eb92b22c93931ddc3f2cbffec94da232249504726202a45aaf4db1b`, matching the manager's value. Context `be34b5769c82454dd0fe0b13177833cd7f3bd209a7e022cdfe7ff4e54a9ccb78`.
- Every catalogue has 111 keys = 87 base + 4 new #4380 rows (`yes_preflight_failed`, `yes_carry_unavailable`, `agent_needs_operator`, `plan_not_honoured`) + 2 #4380 changes (`needs_confirmation`, `help_yes`) + 20 #4378 `deps_*`. The en context has 108 keys.
- No key or context note changed on both sides.

**Byte-for-byte provenance, all 7 locales**
- Every #4380 row equals main `5e860469d`. Main's last receipt per locale is `passed` at `2661ca7d…`, by auditor reviewer `3e52ab86e193`, and its record's `target_sha256` and `context_sha256` match main's catalogue and context bytes.
- Every `deps_*` row equals `60dcbe088`. Its catalogues are byte-identical to my round-3 head `70878bbaf`, and the committed `sd2/record.json` files match my round-3 pins.
- Every other key equals base `792c7e57e` on both sides.
- So no row is outside a certified value and nothing needed a fresh full audit.
- The PR's `audit-history` files equal main's: the merge kept main's suffix, so the `b7035a7c…` rows from `60dcbe088` (`sd2` first slot, `slot2-r3` second slot) are no longer in history. Their evidence files remain in the tree and are cited below.

**Union coherence**
- I read the 6 #4380 rows alongside the 20 `deps_*` rows in each locale.
- Device, agent, conversation and move terms agree across sides in es, fr, hi, ru, ur and vi. "digest" appears only in #4380 rows.
- Mechanically, across all 111 keys:
  - fr: no plain space before `:;!?`.
  - Placeholders: parity holds except the two pre-existing base fr rows noted below, which repeat `{device}` legally.
  - No format characters (Cf).
  - zh-CN: CJK/Latin spaces only in the pre-existing base `worktree_left` (`{path}  留在`).
- Those two items are unchanged base bytes and do not gate.
- `check.py --catalogues-only` was not re-run, since no value is new.

| ID | Sev | Locale | Keys | Current | Proposed | Reason |
|---|---|---|---|---|---|---|
| C-ZH-1 | minor | zh-CN | yes_carry_unavailable, agent_needs_operator, plan_not_honoured | 迁移 for "move" | 移动 (只移动对话, 移动卡片, 移回{source}再重新移动) | `deps_help`/`deps_description` and 17 namespace keys say 移动, and these rows already mix in 没有移动任何内容. This is drift on #4380's certified side: deferred, does not gate. |

**Verdict (`cli.sessions_move_workset`, merged):** PASS for each of es, fr, hi, ru, ur, vi and zh-CN. Overall **PASS**. Coverage: 111 keys × 7 locales = 777 values checked byte-for-byte against their certified side; the 26 rows that differ from base (6 #4380 + 20 #4378) × 7 = 182 values were also read for coherence.

**Receipt proposals:** evidence `conv1`, one per locale, at source `0f5602723eb92b22…`, against the CURRENT history (= main's). `ledger.py append --dry-run` returns rc 0 for all 7, and a stale-hash negative (the `60dcbe088` history) is refused.

| Locale | Target sha256 | Expected history sha |
|---|---|---|
| es | `91b1e6f2cc2927888cb2e8502a8c695f5e6e8370d4abf28cc5130142837dd37d` | `2690901615ca062e…` |
| fr | `e4c6e8af4386ea333370d3ac0a31120983fc28b918b1cf246e9acc5c3b3dc987` | `787558ed4d80e876…` |
| hi | `38058677cf545eb6a7a8bbced2dc0d66e39e87d302fb913660c2db62e0c7bc47` | `9eae86c26d0f1b5a…` |
| ru | `7637456847a6650d53c4ca2b3b2a93b4b208b72b1c609c02d7008b8865d37d05` | `a0ab7c3589b03504…` |
| ur | `28dae9b7b003353426fc8ce95621188224b75f1a88d703ffb8ee5ca9d4ce8684` | `72a48bfbc6fbd14e…` |
| vi | `2666fdf5561ecfcd764cd40f55ab67458ed0c56c0f125ed02e470639b351bd93` | `de818a08df740799…` |
| zh-CN | `c4a1691b1db303347c3ff3592bea23bab6117626580c9cc8870dbe6a85ca60ee` | `2a9a97a85427039b…` |

Translators (verified in session transcripts; provider `anthropic` and model `claude-opus-5-5` only):
- #4380 rows: `d57574b79887` and `8fbb05fa8dc3`, plus `512f23f6b2d0` for zh-CN `plan_not_honoured` only.
- #4378 rows: `8910d769897e` and `28260efb8cc9`.

This audit is agent-independent only: same model as the translators. Before the dropped rows, the #4378 `deps_*` rows had a second-slot audit (`6fd346121602`, evidence `slot2-r3`). On main, the #4380 rows carry one `passed` receipt at `2661ca7d`. Whether the convergence needs a second-slot receipt at the merged hash is the manager's call; this first-slot receipt does not stand in for one.

**Limitations:** no native-speaker review, no rendered terminal frame.
