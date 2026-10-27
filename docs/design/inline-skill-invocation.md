# Inline `$skill` invocation: decision note

Status: ACCEPTED. Author: architect (lopdev). Date: 2026-10-27. Branch `feat/inline-skill-invocation`, base `568868a81`.
Implemented on the branch; the accepted deviations from the draft are recorded in D3 (case) and D10 (the ink's surface rule).

Trigger rule is settled by the operator: keep `$`, and any boundary `$name` that matches a discovered skill fires anywhere in the message — subject to the surface rule in D10 (a line the submit path never expands attaches nothing, so it inks nothing) and the case rule in D3.

**Record carrier:** no PR or ticket existed when this was written, so this file is the carrier. Copy the "Decisions" block into the PR's `## Decisions` when the branch opens. No prior decision rows existed, so nothing is overturned.

## Recommendation

Attach skills as a tail block on the same text content, and strip it on replay with the same pattern the `@path` reference block already uses. The message the user typed goes out and shows unchanged.

- **Trigger:** one pure function, `skill_mentions(text, names)`, in `local_operator/sigils.py`. The submit parser, the composer ink and the picker all call it. Swapping to option B or C later is a change to that one function.
- **Payload:** `<user text, verbatim>` + `\n\n<operator-skills>` + a one-line preamble + one `<skill name="…">body</skill>` per skill + `</operator-skills>`.
- **Replay:** `harness/rows.py::user_row_text` strips complete skill blocks and complete reference blocks. Every surface that paints a user row already goes through it.
- **Composer:** accepting a picker row replaces only the token at the caret. The reassembly code is deleted.
- **Visibility:** every recognised token is inked before send. After send, the user row gets a receipt line, e.g. `skills: research, code-review · ~3.2k tokens`.
- **Order of work:** pure code and tests first (sigils, invoke, rows, references), then app wiring, then composer, then transcript receipt, then i18n.

## Decisions

### D1: Payload representation
- **Taken:** option (a), one `TextContent` with verbatim text, a delimiter, and a skill block appended.
- **Reason:** every consumer of user rows takes a string and calls `user_row_text(message.text)`. Those are `tui/session_presentation.py:2021`, `mobile/projection.py:1690`, `session/runtime/serving.py:4443`, `server/routes/desktop_wakes.py:1035`, `resume.py:3819`, `session/naming.py:1882` and `session/checkpoint_naming.py:313`. A tail block changes none of their signatures.
- **Rejected (b), new content block type:**
  - `Content` has five members, not two (`harness/types.py:643`: text, image, audio, attachment, document).
  - Each provider serializer has its own `isinstance` chain (`providers/clients.py:2274, 2428, 4677, 5270`).
  - The cache-key chain at `providers/context.py:55-100` crashed twice on non-text blocks, per its own comment (QA round 1, Q1/Q2).
  - Mobile projection, transcript storage and the TypeScript client would each need work as well.
- **Rejected (c), `provider_payload["details"]`:** `types.py` documents that key as "harness bookkeeping … wire clients MUST NOT replay". The model would never see the bodies. Making it see them means wrapping the text again, which is what we are removing.
- **Also rejected, two `TextContent` blocks:** `Message.text` joins all text blocks, so the body would leak into every row consumer. Splitting by block index would need a change in all seven hosts.
- **Consequences:**
  - The message persists as text plus block, so the bodies are re-sent every turn until compaction, as they are today for the leading case.
  - The `invocation="…"` attribute and the `typed=` workaround go away.
  - The live row becomes `expand_pastes(text)`, the same as an ordinary prompt. Live and replay then agree by construction, and the "MAJOR-2" branch at `tui/app.py:~25605-25626` is deleted.
  - Titling reads the row via `user_row_text`, so it uses the user's words.
- **Commits us to:** the `<operator-skills>` wire grammar and preamble line (frozen once persisted).
- **Overturned by:** a requirement that desktop and mobile show the attached skills as structured data, not a text receipt. That would justify (b).
- **Owner:** coder.

### D2: Replay stripping that does not truncate a user who types the delimiter
- **Taken:** a `skill_block_spans(text)` that copies `references.reference_block_spans` (`references.py:1041-1100`).
  - A span exists only when opener + exact preamble line + closer are all present.
  - An unclosed or nested opener rescans from the next opener.
  - Body occurrences of the two markers are defused with a zero-width space (as `references.py:1436` does).
  - `user_row_text` strips the union of reference spans and skill spans. It uses one extracted `_strip_spans` helper, instead of copying the join logic at `rows.py:~830-870`.
  - Names for the receipt are read from the `<skill name=` heads inside a span.
- **Reason:**
  - Typing `<operator-skills>` in a sentence has no preamble after it, so it is not a span and nothing is hidden.
  - A complete block pasted by the user is stripped. This is the accepted cost the references code already documents.
  - A tail-anchored rule would be safer. It is impossible because `Session.prompt` appends the reference block after ours (`session/session.py:9560`).
- **Rejected:** a per-message nonce. It would not survive forwarding or compaction, and it adds state that can disagree with the text.
- **Legacy kept:** `typed_line_of`, `_INVOCATION_TAG_RE` and `_unescape_attr` stay. Existing transcripts and the desktop client's own writer (`local-operator-ui` `skill-invocation.ts:223`) still produce the old format.
- **Overturned by:** a measured case of real user text being stripped.
- **Owner:** coder.

### D3: Trigger shape and the pasted/quoted-name consequence
- **Taken:** `skill_mentions` in `sigils.py`.
  - **Boundary:** `is_boundary` (`sigils.py:68`), scanning the whole text, not line by line.
  - **Name class:** `[A-Za-z0-9][A-Za-z0-9._-]*`.
  - **Trailing punctuation:** try the full name first, then the name with trailing `.`, `-` or `_` trimmed. This lets `use $research.` fire while `foo.bar` still works. It also makes a leading `$research.` fire, which it does not today.
  - **Case (as implemented):** an exact name match always fires. A case-insensitive match fires when the token is LEADING (the legacy `$skill` rule, preserved: `$RESEARCH` and `$Research` both fire), and INLINE only when the token contains a lowercase letter (`$Research` fires; `$PATH`/`$DEBUG` do not). This keeps every pre-inline invocation valid while closing the environment-variable collision the picker's evidence rule (`command_picker.py`, `skill_suggestions`) already guarded.
  - **Dedupe:** by resolved name, in first-seen order. No `hide` filter, as today.
- **Escape with no new grammar:** the character before `$` must be whitespace or the start of text. So `$$research`, `` `$research` ``, `($research)` and `"$research"` are all literal.
- **Rejected:** keeping case-insensitive matching everywhere. It would make `echo $PATH` fire a skill named `path`.
- **Behaviour changes to flag:** leading `$research.` starts firing (trailing punctuation is now trimmed at both positions). Leading `$RESEARCH` KEEPS firing — the scoping option below was taken, so the draft's "stops firing" outcome does not apply.
- **Consequences to tell the user:**
  - A name quoted in prose fires. "How do I use `$research`?" attaches the body.
  - A short paste (not collapsed) containing `$research` fires. The paste collapses into a chip only above 20 rows with at least 4 pasted rows (`editor.py:244, 261`).
  - A collapsed paste stays inert, because the scan runs on the typed text with chips, before paste expansion.
  - The ink and the receipt are what keep this visible.
- **Overturned by:** the receipt showing surprise attachments in real use. That would send the question back to option B (provenance) or C (a new sigil).
- **Owner:** coder.

### D4: Token budget and notices
- **Taken:**
  - Skills attach in text order using a cheap `len(bytes)//4` estimate.
  - Skills that would exceed a cap of 8 skills or about 48k estimated tokens are not attached. A warning names them, and the user's message always sends.
  - An empty-body skill is skipped with a notice. The others still attach.
  - A load failure for one skill is a notice for that skill only.
- **Reason:**
  - Skill files on this machine (34 `SKILL.md`s across the standard roots) have a median of 6.5 KB (~1.6k tokens), a mean of 8.6 KB and a maximum of 60.8 KB (~15k tokens).
  - Eight pasted names could add 100k or more tokens. Today's silent fall-through (`app.py:56061`) is the wrong model for that.
- **Not measured:** the 8 and 48k numbers are chosen defaults, not data. Per-model context windows would settle them.
- **Rejected:** a hard refusal that keeps the draft. I could not confirm how a refused draft is restored from `on_editor_submitted`, and refusing risks losing the message (the bargain `_expand_invocation` documents).
- **Also rejected:** a tokenizer count at submit. It would load tiktoken on the message pump.
- **Owner:** coder.

### D5: Composer behaviour, ink and attachment visibility
- **Insert:** `_complete_skill` (`editor.py:9008`) does a span replacement only. In `command_picker._skill_span_replacement` (`:967`), the `.lstrip()` on the text after the token joins `fix $re⏎the bug` into one line. That breaks "don't rewrite the user's text". Fix it so a space is added only when the next character is not whitespace, and nothing is stripped.
- **Picker:** unchanged, one list at the caret. `skill_token_is_leading` and the lowercase-evidence rule in `skill_suggestions` stay, because they decide when Enter completes versus sends.
- **Ink:** in `_skill_runs` (`editor.py:4941`), drop the "only the first token of the buffer" skip (`:5012`). Ink every span that `skill_mentions` accepts. `set_skill_names` (`:2965`) must carry exact names, not lowercased ones. Keep the inert style for a lone unknown token.
- **Indicator:** no new composer chrome, because ink answers "what will fire". After send, `UserBlock` (`widgets/transcript.py:1438`) gets a receipt row beside the existing attachment receipt (`:1754`, `_receipt_rows`). On replay, names and size are read from the persisted block, so live and resumed rows match.
- **Rejected:** a persistent "attached skills" chip in the composer. It duplicates ink and takes layout space.
- **Not covered:** the phone shows verbatim text with no receipt, because `TranscriptEntry` carries only text and images.
- **Owner:** coder, with designer on rendered before/after frames of the ink and the receipt.

### D6: Surfaces
- **Gain inline and multi-skill:**
  - The main composer submit (`on_editor_submitted` → `_expand_invocation`, `app.py:25604`).
  - Prompt-consuming slash commands. `_submit_command_prompt` (`app.py:20856`, which calls `_expand_invocation` at `:~20914`) scans the request tail, so `/team ops fix $research` fires. The slash word stays leading.
  - Steered messages, because they carry `sent`.
- **Unchanged:**
  - Slash commands in general, and `/model $x`.
  - Bang mode: `! echo $PATH` stays shell (`app.py:25455-25465` returns first).
  - The `/btw` aside, which returns before expansion (`app.py:~25421-25447`).
  - `lop exec` and headless mode. I searched `exec_session.py`, `exec_mode.py`, `headless_print.py`, `cli.py` and `server/routes/*.py` and found no use of `skills.invoke`. `$name` stays literal there. What must stay is the legacy replay reader (D2).
- **Owner:** coder.

### D7: References must skip skill blocks
- **Taken:** `references._reference_tokens` (`references.py:1152`) adds skill-block spans to its skip set.
- **Evidence:** I ran `expand_references` on a legacy payload whose skill body contained `@notes.txt`, with a file by that name present. The file content was read into the message (`body-token read file: True`). This already happens for a leading skill. With several bodies it gets worse, and it can trigger approval prompts.
- **Rejected:** leaving it. N bodies multiply a leak that exists today.
- **Owner:** coder.

### D8: i18n
- **New keys** (defined in `i18n/keys/tui_app.py` and `tui_transcript.py`, catalogues under `i18n/catalogues/<locale>/`):
  1. `tui.transcript.skills.receipt`: the user-row receipt, ICU plural on count, with names and token estimate as parameters.
  2. `tui.app.skills.over_budget`: the warning naming skills that were not attached.
  3. `tui.app.skills.empty_body`: replaces the hardcoded English at `app.py:56061`, now per skill.
  4. `tui.app.skills.load_failed`: replaces the hardcoded English at `app.py:56068`.
- **Changed key:** `tui.app.help.skill_note` (`tui_app.py:81`), because the gesture is no longer leading-only. The `/skills` header (`_skills_block`) also names the gesture. The coder must check whether that string is keyed.
- **Gate (AGENTS.md "Internationalization is mandatory"):**
  - Every new or changed key is translated into all 7 non-English locales in `i18n/languages.json`: fr, es, zh-CN, ru, vi, ur, hi.
  - Each key gets a `context.json` entry with role, tone and note.
  - An independent auditor passes the translations before the ledger rows ship.
  - `python scripts/i18n/generate.py --check` and `python scripts/i18n/check.py` must pass.
  - `tests/unit/i18n/test_language_manifest.py` must pass, with no growth in `coverage-allowance.json`.
  - Never hand-edit `shipped.json`.
- **Not keyed:** the model-facing preamble and the block tags stay English.
- **Owner:** coder for keys and en; a separate translator/auditor pass for the other locales.

### D9: Deletions
- **Delete:**
  - `command_picker._reassembled_skill` and `_reassembled_skill_argument` (`:971-1060`).
  - Their two call branches in `completion_for` (`:823-851`).
  - The reassembly paragraphs in `Editor._complete_skill`.
  - `invoke.parse_invocation`, `SkillInvocation`, `render_invocation` and `_escape_attr`. The only caller is `app._expand_invocation`.
  - The `row = text if sent is not None …` branch and the `typed=` plumbing used only for the invocation line.
  - The `invoke.py` module docstring, which argues for anchoring and would be false.
- **Keep:**
  - `skill_token_is_leading`, `_skill_inline` and `skill_suggestions`. They still gate when the picker opens.
  - `_skill_argument_floor`, which lets the picker open inside `/team x $…`.
  - `typed_line_of`, `_INVOCATION_TAG_RE`, `_unescape_attr` and `rows.typed_line_of`, for old transcripts.
  - `_skill_body_has_content`.
  - `skill_token_spans`, kept and pin-tested against the new function, or made to delegate to it.

### D10: Which lines the `$` ink may claim (added in review round 1)

- **Taken:** one pure surface rule, `command_picker.skill_scan_region`, returns the cells a submit would scan (or `None`). `Editor._compute_skill_runs` inks only inside that region, and `_submit_command_prompt` passes the same region to the expansion.
  - **Shell/bang:** a bang-mode buffer, or a recalled `!` line, scans nothing — submit runs it as a command.
  - **Leading slash command:** only the request tails of `/team`, `/teams`, `/agent`, `/agents` and `/goal` are scanned (the commands whose handlers call `_submit_command_prompt`). `/btw` opens the aside, `/loop` and `/fork` start their own flows, and every other command dispatches; none expands a skill, so none inks.
  - **The aside:** while the `/btw` card owns the composer, `_allows_skill_expansion` is off and the ink is empty — the aside's submit returns before the expansion.
  - **The argument tail is its own string:** a `$RESEARCH` opening `/team backend …` is LEADING there, so the legacy case-insensitive rule applies in the ink exactly as it does at submit.
  - An unrecognised leading `/word` is prose (submit sends it as a prompt), so the whole buffer is scanned.
- **Reason:** round 1, M1/m4. The draft's mitigation for surprise attachments is "ink is what will be sent"; an ink painted on a line that never expands is a false claim in the other direction, and in shell mode it paints an environment variable as a skill.
- **Overturned by:** a surface that expands skills but is not covered here — the rule is one function, so extending it is a change in one place.
- **Owner:** coder.

## Implementation plan

1. **`local_operator/sigils.py`.**
   - Add `SkillMention` and `skill_mentions(text, names)`.
   - It must live here because anything under `local_operator/skills/` runs the package `__init__`. I confirmed that import pulls in `discovery`, and the picker avoids that on the TUI boot path.
2. **`local_operator/skills/invoke.py`.**
   - Add `attach_plan(text, skills)`, `render_skill_block(entries)`, `skill_block_spans`, `skill_block_names` and the budget logic.
   - Keep the legacy reader only.
   - Place the shared span finder in a host-free module, and run `test_import_graph` to check.
3. **`local_operator/harness/rows.py`.** Extract `_strip_spans`, and make `user_row_text` strip reference and skill spans, then fall back to `typed_line_of`.
4. **`local_operator/references.py`.** Skip skill spans in `_reference_tokens`.
5. **`local_operator/tui/app.py`.**
   - Rework `_expand_invocation` to return the sent text plus the attached list, with keyed notices.
   - Simplify `on_editor_submitted` (`:25604-25626`) and `_submit_command_prompt` (`:~20914`).
   - Make `_push_skill_names` (`:55908`) push exact names.
   - Pass names to `UserBlock` in `_submit_prompt` (`:~34500`) and in the replay at `tui/session_presentation.py:2021`.
6. **`tui/widgets/command_picker.py` and `editor.py`.** The deletions and fixes in D5 and D9.
7. **`tui/widgets/transcript.py`.** The `UserBlock` receipt row.
8. **i18n.** The keys and catalogues in D8.
9. **`local-operator-ui` (outside this PR).**
   - `skill-row.ts` and `skill-invocation.ts` read and write only the legacy shape.
   - A new-format message viewed in the desktop app may show the body in the user's bubble. I did not trace the desktop paint path.
   - Record this under "not addressed" in the PR, not as an issue.

## Test matrix

| Area | Cases |
|---|---|
| **Parser** (`tests/unit/skills/test_invoke.py`, a new `sigils` test) | Money (`costs $5`, `$100 for it`) and glued (`a$b`, `costs$5`) do not fire. Env vars `echo $PATH` and `$HOME` do not fire, and `$path` fires only if a skill named `path` exists. Mid-sentence, line start and after a newline fire. `$research.`, `$research,` and `$research?` fire, and `foo.bar` and `a-b` names work. `$$research`, a backticked name and `($research)` do not fire. Multi-skill is in text order, and a repeated name is deduped. Case: exact `DEBUG` fires and `$Research` fires. A bare `$`, `$_x`, `$ßeta` and an unknown name do not fire. A collapsed paste chip is not scanned. Hidden skills fire. A token inside an existing skill block is ignored. |
| **Payload and strip** | User text is byte-identical before the block. Body markers are defused. A literal `<operator-skills>` typed in prose is not stripped. An unclosed opener followed by a real block strips only the real block. Skill block plus reference block strips both. A pasted complete block is stripped (documents the accepted cost). Legacy `invocation=` payloads still replay as the typed line. A body containing `@file` does not read the file. |
| **Budget** | Empty body is skipped with a notice and the others attach. More than 8 skills, or over the token cap, attaches in order and names the rest. The message always sends. |
| **Composer** (`tests/unit/tui/test_skill_invocation.py`, `test_skill_highlight.py`, `test_skill_in_command_argument.py`; pilots under `env -u NO_COLOR TERM=xterm-256color`, CMUX variables unset) | Insert at the caret mid-draft with no rewrite and no line-joining. A second pick adds a second token. Enter with the list open completes and does not send. `$PATH` opens no list. Ink on every recognised token, plus the lone-unknown inert style. Inside `/team x $…` the token is inked and inserted in place. The ghost text is empty on mid-text insert. |
| **Submit, replay and titling** | Row equals the verbatim text, with pastes expanded. Receipt shows after send. A resumed session shows the same row and the same receipt. Title comes from the user's words. A queued/steered message carries the block. `lop exec` leaves `$name` literal. Bang mode is untouched. `/btw` is untouched. |
| **i18n** | The four new keys plus the changed `help.skill_note` in all 7 locales. The manifest, `generate.py --check` and `check.py` all pass with no allowance growth. |
| **Visual** | Rendered before/after frames of the ink and the receipt, for the designer. |

## Why this and not the alternatives

The rejected options are listed under each decision. The two I looked at hardest:
- **A new content block (D1b):** the only option that gives structured data, but the serializer chain has crashed on unknown blocks before.
- **A tail-anchored strip (D2):** safer in theory, but impossible because the reference block is appended after ours.

## What this commits us to

- **Contracts frozen:**
  - The `<operator-skills>` wire grammar and its preamble line. Changing the preamble later orphans old transcripts.
  - The legacy `<skill … invocation=…>` reader stays forever.
- **Files owned:** `sigils.py` owns the trigger. `invoke.py` owns the block grammar. `rows.py` stays the one place that decides what a user row is.
- **Work ordered, not parallel:** the pure layers first, then app wiring, then composer and transcript, with i18n last. The translations need an independent audit pass.
- **Hard to reverse:** bodies persisted in the new shape. Reverting later means keeping both readers.
- **Behaviour change to accept:** leading `$research.` starts firing (D3). Leading `$RESEARCH` still fires, by the scoped case rule D3 records.

## Risks and what would change the answer

1. **Desktop UI drift (cross-repo).** The desktop app reads and writes only the legacy shape, and I did not confirm how it paints a new-format message. If it shows the body in the bubble, ship a matching UI change. If that is not possible, the fallback is for the TUI to emit the legacy shape for single leading skills.
2. **Surprise attachments.** An inline match in prose, a short paste or a quoted name silently adds up to ~15k tokens per skill to every later turn.
   - **Evidence on this machine:** 540 session transcripts, 302 non-payload user messages containing `$`, 104 other `$` tokens, and zero boundary `$name` tokens matching any of the 66 skill-directory names I found. I counted but never printed token content.
   - **Limits of that evidence:** one machine, and skill names taken from directory names, not the discovered set.
   - **Mitigations:** ink, receipt and caps.
   - **Watch:** how often the receipt appears on messages that did not intend it. If that is more than occasional, move to option B.
3. **Strip and reference interplay.** Complete-block span stripping can hide a pasted complete block. Reference expansion reads `@paths` inside skill bodies today (measured). Both need the D2 and D7 tests before merge.

## What I verified, and what I could not

**Verified by reading, with line refs above:**
- `invoke.py` in full.
- The `rows.py` strip logic.
- The `references.py` span grammar and its position after our block (`session.py:9560`).
- `command_picker` completion and reassembly.
- `editor.py` ink, completion and Enter routing.
- The `app.py` submit and replay paths, including `_submit_command_prompt`.
- `UserBlock`'s attachment receipt.
- The `Content` union, which has five members.

**Verified by running here:**
- The reference leak into a skill body (probe via `expand_references`).
- That `local_operator.skills` imports `discovery`.
- The skill-size statistics (34 files).
- The transcript scan above.
- Zero overlap between the 66 skill names and a hand-listed set of 15 common environment-variable names. That set is not exhaustive.

**Not verified:**
- No tests were run. This is a design, not an implementation.
- `Message.text` is at `types.py:1155` per grep, but I did not re-read the body.
- How the desktop app paints a user row.
- How a refused draft would be restored from `on_editor_submitted`.
- Whether the `/skills` header copy is keyed.
- The right values for the 8-skill and 48k-token caps.
- Line numbers marked `~` were read from truncated output and should be re-checked by the coder.
