# relay.approvals - zh-CN - re-attestation ap25b1

Revision: 3d0f77762a (locale split), en source unchanged since 4b40d82512
Auditor: translation-auditor (lopdev team) - re-attestation pass by translation-reviewer
Model: deepseek/deepseek-flash (same model as the 2026-10-25 first pass; agent/context independence only)
Date: 2026-10-25

## Composed sentence (what the user reads)

前两个选项只改变 **本会话** 而已；最后一个则把其后 **所有会话的默认值** 改掉。

Bold runs: `本会话` and `所有会话的默认值`.

## Reading

前两个选项只改变 **本会话** 而已；最后一个则把其后 **所有会话的默认值** 改掉。 (the four spaces are injected by the component, not present in any key). Meaning preserved: 只 ... 而已 confines the first two choices to 本会话 - 而已 is the standard restrictive partner of 只 and carries meaning rather than padding; 则 carries the contrast; 把 ... 改掉 is the disposal construction that keeps '所有会话的默认值' a unit and lets the verb close the sentence in 'after', and 改掉 adds a completive 'change away' flavour that en's plain 'changes' lacks (acceptable, mildly colloquial). Not over-emphatic: one restrictive and one contrastive marker. Bold: '本会话' and '所有会话的默认值' - both complete; 其后 stays plain exactly as en's 'that follow it' does. FINDING (minor, open, REGRESSION, component-level): the pre-split single key contained zero spaces; the five-key join inserts four U+0020 between Han characters (改变|本, 话|而, 后|所, 值|改), which Chinese typography does not use and which zh-CN.md's no-space discipline and its cell budget both argue against: four visible gaps in a 33-character sentence, and four extra cells under the guide's own model (ASCII = 1 cell, CJK = 2), 70 cells against the 66 the Han text itself costs. The copy is right; the join is wrong for CJK. Confidence: medium-high on meaning, medium on register. LIMITATION: same model as the first pass; no rendered frame captured - the defect follows from the join rule, not from a screenshot. Evidence: i18n/audits/evidence/zh-CN/relay.approvals/ap25b1/record.json

## Seam

defect: four U+0020 inserted between Han characters, which Chinese does not space

## Findings

- ZH-1 [minor/open] component join (gate-sheet.tsx), all four seams: The split inserts four U+0020 between Han characters where the pre-split key had none: four visible gaps in a 33-character sentence and 4 extra cells under the zh-CN cell model (70 against 66). Copy is correct; the join is wrong for CJK. Regression introduced by the split.

## Confidence

medium-high on meaning, medium on register (Chinese read, not native)

## Limitations

- Same model as the first pass: independence is agent/context, not model diversity.
- No native speaker has reviewed any locale; hi and ur were judged non-native by the two earlier
  passes and this pass does not overturn that.
- No rendered frame was captured; see record.json `rendered_artifact`.
