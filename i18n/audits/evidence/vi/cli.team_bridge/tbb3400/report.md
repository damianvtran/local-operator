# vi cli.team_bridge audit tbb3400 (PR #3400)

Scope: the two added keys `notify.budget_exhausted_title` and `notify.budget_exhausted_body` at PR head `5456b8db1d017c421c7632324ab2065e64e8b85e`, read against the en source, `cli.team_bridge.context.json` notes (budget used up so new agent messages wait; nothing lost, screened automatically as the budget refills; people and messages addressed to the agent screened first; no implication of rejection, deletion or notification), the vi style guide, `i18n/glossary.md`, and the namespace's existing renderings. The other 55 keys are byte-unchanged since the PR #3198 passed row and were not re-audited.

## Translator values (as committed at head)

- title: Tạm dừng sàng lọc tin nhắn của tác tử
- body: Tác tử của bạn đã dùng hết hạn mức sàng lọc trong giờ này, nên các tin nhắn tác tử mới đang chờ. Chúng sẽ được sàng lọc tự động khi hạn mức được bổ sung. Tin nhắn từ người dùng và tin nhắn gửi tới tác tử của bạn được sàng lọc trước.

## Findings and dispositions (all fixed by this auditor in the audit commit)

- **V1 MAJOR** (title+body) — current: `tác tử`. Glossary §2 agent = 'tác nhân'; namespace uses 'tác nhân' (22 uses). Exact repeat of the tbp3198 major. Fix: 'tác nhân' throughout; title uses product name 'tin nhắn tác nhân'.
- **V2 MAJOR** (body) — current: `Tin nhắn từ người dùng`. 'người dùng' = 'users'; agents also act for users, so it blurs the human-vs-agent priority fact the context note requires. Fix: 'Tin nhắn từ con người'.
- **V3 MINOR** (body) — current: `trong giờ này`. 'This hour' implies a clock-hour reset; the budget is hourly/rolling. Fix: 'hạn mức sàng lọc theo giờ'.
- **V4 MINOR** (title) — current: `Tạm dừng sàng lọc…`. Verb-initial title reads as a command/button ('Pause screening'). Fix: 'Sàng lọc tin nhắn tác nhân đang tạm dừng'.

## Shipped values after remediation

- title: Sàng lọc tin nhắn tác nhân đang tạm dừng
- body: Tác nhân của bạn đã dùng hết hạn mức sàng lọc theo giờ, nên các tin nhắn tác nhân mới đang chờ. Chúng sẽ được sàng lọc tự động khi hạn mức được bổ sung. Tin nhắn từ con người và tin nhắn gửi tới tác nhân của bạn được sàng lọc trước.

Accuracy: all three context facts survive; no rejection, deletion or notification implied. Unresolved blocker/major: 0.

## Mechanical

Key parity 57/57; no placeholders, ICU or markup in either key (none in source); NFC; no Cf/bidi control characters; no code tokens. No `maxChars`/`maxCells` hint exists for these keys; both are inbox-notification prose, not width-bound terminal cells.

## Negative control

At head 5456b8db (rows absent) the generator's `_shipped_locales()` returned `['en']`: the stale `cli.team_bridge` hash would have dropped all seven locales. After these rows it returns all eight (see audit.log).

## Limitations

Independent LLM review, not native-speaker certification; no MT or translation service used. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3400 and is recorded as unrecorded, not inferred.
