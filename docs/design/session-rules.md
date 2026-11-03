# Design: per-session rules (investigation + POC)

Status: **PROPOSED — POC for peer review, not for merge.** Base: `origin/main` @ `f2187d4f5`
(v0.95.2). Branch: `poc/session-rules-b`. Date: 2026-11-03.

Everything here is behind `LOP_SESSION_RULES=1`. With the variable unset, the session reads and
writes no `rules.json`, and every system block is byte-identical to `origin/main`. The one
flag-off change to prompt handling is listed in §9. `rules.json` is also added to the fork and
move copy sets, which is inert when the file is absent. The flag-off
proof and every run log are in the PR comments, not in this tree (AGENTS.md "Evidence goes on
the PR").

## 1. Recommendation

**Yes, add rules as a thin native primitive.** This is a product judgement informed by the
evidence below, not a result the evidence proves on its own. What the evidence does show:

- the memory/decisions path is the wrong instrument for standing rules, because it is untrusted by
  design and rarely read back;
- with one cheap model and one rule type, a tail `<rules>` section followed every edit, removal,
  restart and compaction, while the same rule held in block 0 followed edits only 1 time in 3.

Most standing instructions in the scan were never written anywhere the scan could see. So the
primitive is only worth graduating together with a capture path.

- **Do not reuse memory/decisions for this.** Memory is untrusted by design
  (`guides/memory/GUIDE.md:20`; the only ambient injection is a ≤180-token block headed
  "Untrusted recall candidates", `memory.py:870/884`). "Decision" is a memory tag that
  compaction carries 20 deep (`compaction/carry.py:259`). Making memory authoritative would
  reverse a trust rule the product states on purpose.
- **Use the shape the POC built:** a `rules.json` sidecar (beside `goal.json`), rendered as an
  authoritative `<rules>` section in the volatile tail next to `<goal>`. Each edit ships as a
  `[session-state]` delta. One precedence sentence goes in block 0 whenever the flag is on.
- **Before graduating**, decide the open questions in §8, chiefly the precedence between a
  rule and the operator's live message, and model-proposed rules.

## 2. The question underneath: does the decisions/memory store decay?

The scan was read-only over every transcript under `~/.local-operator/sessions/` on the
author's machine: 807 sessions, first activity 2026-08-28 to 2026-11-03 UTC, 118,424 tool
calls. The script is attached to the PR.

| Measure | Value |
|---|---|
| user-role turns matching a broad standing-instruction regex (`always`, `never`, `from now on`, …) — **an upper bound, not hand-sampled; it includes agent-written briefs in delegated sessions** | 515 turns in 363 sessions |
| of those 363 sessions, how many ever called `memory store` | **12** |
| `memory store` calls / `memory recall` calls (all sessions) | 63 / 33 |
| sessions that stored something / of those, how many later recalled anything | 13 / **6** |
| median tool calls from first store to next recall, where one happened | 969.5 |
| sessions that compacted / of those, how many recalled after the first compaction | 51 / 7 |

A standing instruction can be lost at three points. The data separates them:

1. **Never captured.** 351 of the 363 matching sessions made no `memory store` call. The scan
   does not see other durable homes (an AGENTS.md edit, `/goal`, a todo, an external task), so
   this is an upper bound on loss, and the largest one the data can show.
2. **Captured but not read back.** 7 of the 13 storing sessions never recalled after storing (5 never
   recalled at all), and the median gap
   to the next recall where one happened is about 970 tool calls. The scan does not show that
   recalls cluster at compaction: only 3 of 33 recalls fall within 20 tool calls of one.
3. **Read back but not obeyed.** The store cannot answer this: recalled text is labelled
   untrusted, so it carries no instruction weight by construction.

So the store is used rarely and read back less. More importantly, it is the wrong instrument
for rules. Fixing recall frequency would not help with point 1 or point 3.

## 3. What a rule is (vs its neighbours)

| Thing | Author | Scope | Trust | Read path | Lifetime |
|---|---|---|---|---|---|
| `custom_instructions` / system-prompt line | operator / profile | machine, profile | authoritative | block 0 (cached head), read once per session | until config edit |
| `/goal` | operator | one session, single | authoritative | volatile tail, every provider call | until replaced/done |
| memory fact | usually the model | session/global/team/agent | **untrusted data** | `recall`, or ≤180-token suggestion | until forgotten |
| decision (memory tag) | usually the model | memory scope | untrusted | carried through compaction (20 newest) | as memory |
| **rule (this POC)** | **operator** | **one session, many** | **authoritative** | **volatile tail, every provider call** | **until removed; survives restart + compaction** |

A rule is a *standing constraint on how to behave*, for example "never push to main" or "answer
in British English". A goal says what to achieve. A decision records an outcome. A memory is a
fact that may be stale.

## 4. Design space: swarm comparison

Four read-only investigators each scored one shape on the same seven axes (1–5, higher is
better), citing `origin/main`. Their full reports are attached to the PR.

| Axis | (a) file injected every turn — block 0 / tail | (b) records + per-turn selection | (c) pre-turn checklist / verifier | (d) tool-callable add/drop state |
|---|---|---|---|---|
| model obeys it | 4 / 3 | 3 | 3 | 3 |
| context cost per turn | 5 / 5 | 4 | 2 (verifier = one extra call per check) | 4 |
| conflict handling | 2 / 3 | 2 | 4 | 3 |
| edit/remove mid-session | 3 (new prefix epoch, cold cache) / 5 | 2 ("removed" ≡ "not matched" on the wire) | 4 | 5 |
| compaction / restart / fork / subagent | 3 / 4 | 2 | 3 | 4 |
| implementation size | 3 / 3 | 2 | 2 | 3 |
| failure modes / security | 2 / 2 | 2 (a missed match is silent) | 2 (second reader of injected content) | 3 (self-modification risk) |
| **verdict** | **tail placement chosen** | rejected: a miss is silent | only its deterministic subset is worth having, later | the lifecycle half; rendering = (a)-tail |

The investigators converged: (d)'s lifecycle (operator-owned state that can be listed, edited
and removed) rendered through (a)'s tail placement. (b) was rejected because a rule exists for
the case nobody foresaw, and selection must foresee it twice. (c) as an LLM verifier costs a
model call per check and shares the checked model's blind spots. Deterministic checks (for
example a forbidden command before dispatch) are a sensible later layer on top of stored rules,
not a substitute for them.

## 5. The POC (what this PR contains)

Contract frozen before implementation (architect decisions D1–D6, in the PR):

- **Flag:** `LOP_SESSION_RULES=1`, read once when the system-blocks provider is built
  (`session/rules.py: rules_enabled`). No `Setting`: a setting needs copy in 8 locales, which
  is out of proportion for a POC.
- **Data:** `<session dir>/rules.json` = `{"v":1,"next":N,"rules":[{id,text,created,updated}]}`.
  Ids are `r1, r2, …` and never reused. At most 20 rules of ≤300 single-line characters each.
  Text containing a tail-section marker is refused. Writes are atomic. A present file that
  fails validation is voided as a whole, with a warning, and never partly applied.
- **Mutation:** operator only, through a dev CLI:
  `python -m local_operator.session.rules <session_dir> list | add <text> | edit <id> <text> | rm <id>`
  with JSON output. There is no model tool, slash command or TUI in the POC.
- **Rendering:** `<rules>` sits after `<goal>` in the tail (`prompts_api.py`). It is registered
  in `_TAIL_SECTION_MARKERS` and `STATE_SECTION_HEADINGS["rules"] = "Rules"`, and it is not
  advisory. With the flag on, block 0's "Session state updates" paragraph gains one sentence,
  present even with zero rules, so the first `add` never moves the cached head.
- **Per step:** the provider re-reads `rules.json`, keyed by (mtime, size). A changed render
  reaches the model as a `[session-state]` delta on the very next provider step. A read failure
  of any kind renders no section and never breaks the turn.
- **Copy sets:** `rules.json` is added to `fork.COPIED_SIDECARS` and `network/sync.COPY_SET_NAMES`.
  It is inert when the file is absent.

### How the winner was chosen (arena)

Two contenders implemented the same frozen spec against the same 60 acceptance tests, in
separate worktrees: A on claude-sonnet-5-5 and B on claude-opus-5-5. Both passed 60/60 and both
showed flag-off byte identity. Both also passed the real-model evidence bar on one repeat. A
cross-judge (reviewer, claude-sonnet-5-5) scored them on six code criteria. **B won** because A
has one major defect: an `add` over a malformed file overwrote it and reissued `r1`, losing
every rule and breaking id uniqueness. B was hardened with three grafts and fixes: a flag-on
session with no directory keeps the sentence; the parser refuses deeply nested JSON (a
`RecursionError`) as a malformed file, with a catch-all backstop so a rules read never breaks
a turn; and a voided file logs one warning per change. The
scorecard is in the PR.

## 6. Evidence summary (detail in PR comments)

Real model calls ran through `lop exec`, one process per turn, in an isolated `HOME` and config
dir. The model was **deepseek/deepseek-flash**, because no OpenAI key is stored on this host.
There were 3 repeats per arm and 11 turns each. All checks are deterministic: does the reply's
last line equal the marker the rule demands? The rule was `End every reply with the exact line:
-- R1`, edited mid-session to `-- R2`, then removed, then re-added, followed by a process
restart and then two forced compaction passes.

| step | no-rule control | alternative: same rule in block 0 (`system_prompt.md`) | **tail `<rules>` (this POC)** |
|---|---|---|---|
| rule followed (t1, t2) | marker 0/6 | 6/6 | **6/6** |
| edited mid-session → next turns follow the new text (t4, t5) | — | **2/6** (kept the old marker) | **6/6** |
| removed → stops (t6)¹ | — | **0/3** (a marker kept) | **3/3** |
| re-added + process restart (t7, t8) | — | 2/6 | **6/6** |
| after forced compaction (t9, t10) | — | 4/6 | **6/6** |
| operator says "no sign-off this time" (t3, recorded, not scored)² | — | rule kept 3/3 | rule yielded 3/3 |

¹ The harness's own check on t6 was `absent:R2`, which false-passes a stale `-- R1` line. The
block-0 replies end `-- R1`, `-- R1`, `-- R2`, so none stopped. The matrix attached to the PR is
re-scored as "no marker of any kind".
² The raw harness output marks t3 PASS for block 0 because the marker was present. It is a
precedence probe, not a pass/fail check (§8.1).

**Why the block-0 alternative failed edits (a hypothesis).** The transcripts show its system
prefix *was* re-frozen with the new text on the edit turn: in all three repeats a
`system_prefix` entry carries `-- R2`. In 2 of 3 repeats the model kept repeating the old marker
from its own history anyway. In the tail arm the edit arrives as an explicit `## Rules`
`[session-state]` message in the conversation, and the model switched every time. One reply on
removal said: "No sign-off line — the standing rule was removed from this session's state".

The data are *consistent with* "an announced change sticks better than a silently rewritten
system prompt". They do not isolate it. The two arms differ in placement and in announcement at
once, and no arm tested an announced block-0 change or a tail change with no delta. N=3, one
model.

**Compaction.** Rules live in a system block, which is never summarised. After the compaction
pass, the re-anchoring delta re-shipped `rules` alongside `knowledge` and `memory_hint`, as seen
in the transcript state trail.

**Token cost.**

- **Static measurement** with the repo's own estimator (`approx_text_tokens`, chars/4) and the
  real `Session._system_state_delta` path:
  - Block 0 grows by 30 tokens once, whatever the rule count.
  - The tail grows by 70 / 148 / 412 tokens for 1 / 5 / 20 rules.
  - An edit re-ships the whole section once (`{'3': ['rules']}`), and that delta then stays in
    history until compaction.
- **Live measurement**, paired turn by turn against the control:
  - Tail: median +254 input tokens per turn (range +83..+610).
  - Block-0 alternative: median +108 (range −277..+342).
  - Both figures include the longer replies the rule itself causes. The tail figure also includes
    the accumulated `[session-state]` deltas.
  - Median uncached delta: −5 (tail) and −14.5 (block 0), so caching absorbs almost all of it in
    both arms.
  - One block-0 edit turn ran cold (repeat 1, t4: 2,560 of 19,540 tokens from cache). The other
    two repeats did not, and no tail turn did. One event is not evidence of a cache advantage.
    The structural point is still real: a block-0 edit starts a new prefix epoch
    (`session.py:6729`), and a tail edit does not.

## 7. Limits of this evidence

- One model, which is cheap and not a frontier model. One rule type, a mechanically checkable
  marker. N=3. Compliance on judgement-type rules ("be concise", "ask before deleting") was not
  measured.
- The adversarial probe is n=3 and unscored. It shows a precedence question (§8), not a
  pass or fail.
- Not measured: the cheapest baseline, "operator states the rule in chat". User turns survive
  compaction verbatim up to a cap (`compaction/cutpoint.py:899`), so the chat-only arm may follow
  well. Its weaknesses are structural rather than behavioural: it cannot be listed, edited or
  removed.
- Each `lop exec` turn is a new process, so "restart" here means process restart with
  transcript replay. A TUI session resumed after a crash takes the same path
  (`Session.__init__`), but that was not separately exercised.
- Compaction took the snapcompact path, so the forced passes did not issue a provider call.
- Prompt-level rules are not enforcement. A rule like "never send email" is only guaranteed by
  the approval layer. The rule text explains intent; the gate enforces it.

## 8. Open questions for the reviewer

1. **Rule vs live operator message.** Today the tail rule yields to "not this time". Should a
   session rule win until edited, or should the model yield *and say so*, as one run did
   unprompted? The POC's precedence sentence is silent on this; graduation must pick.
2. **Model-proposed rules.** Capture is the measured loss (§2), so the model should propose a
   rule when the operator uses standing phrasing. Should the operator confirm each one with a
   card? (The POC is operator-only, because `--yolo` bypasses write approval,
   `session.py:14329`.)
3. **Subagent inheritance.** Off in the POC, following the `/goal` precedent. Should rules seed
   a child's brief, given that a delegated slice usually needs the parent's constraints?
4. **Scope.** Rules are session-only. Promoting a rule to a project or global scope overlaps
   `custom_instructions`. Keep them separate?
5. **Edit cost.** Each edit re-ships the full section into history. With frequent edits, ship a
   per-rule diff instead?

## 9. Graduation checklist (not in this PR)

- Replace the flag with a `session.rules.enabled` `Setting` (8-locale copy).
- Add a `/rules` slash command and a TUI panel, following the todo/wake panel pattern, plus
  mobile and desktop state projection.
- Add a capture path: on standing phrasing, the model proposes a rule and the operator confirms
  it.
- Decide §8.1 and encode it in the rendered text.
- Make `rules.json` non-writable by model tools: the POC accepts that `bash` can rewrite it.
- Surface a voided `rules.json` to the operator and the model. The POC only logs a warning, so a
  rejected file silently stops binding.
- Lock read-modify-write in the CLI. Only the final replace is atomic today, so two concurrent
  `add` calls can lose one update.
- Decide on the delta splitter. Registering `<rules>` in `_TAIL_SECTION_MARKERS` applies even
  with the flag off, so a knowledge paragraph that starts `<rules>` would be split into a
  `## Rules` delta. It round-trips byte-for-byte, and no shipped guide or skill contains one,
  but it is the one flag-off-visible change. `<goal>` sets the precedent.
- Optionally, add deterministic pre-dispatch checks for rules that a machine can check.
