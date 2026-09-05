# PR #671 — remediation round 1 evidence

Captured against the remediation head in the worktree `/tmp/lop-live-settings`
(branch `dev-live-settings`), on top of review head `979b54b4`.

## Behavioural cells (real processes)

| file | what it proves |
|---|---|
| `twoproc-approvals.log` | The **asymmetric rule**, three cells. Pane B is a real `OwnedSessionHandle` on the production config-watch path; every WRITE is a separate OS process (`python -m local_operator.cli config edit`). Cell 1: a loosening does not move a session that typed `/approvals ask`, emits the keep notice, and the gate is *really* armed — a real `bash` decision PARKS and the human's deny is honoured. Cell 2: a tightening moves a session that explicitly chose `auto` (safety always propagates). Cell 3: a session that never chose follows the file both ways. |
| `page-write-gate.log` | **R1**, the blocker, in its unsafe direction. A `/settings` page write (`settings_io.write_setting`, the exact call `SettingsView._write` makes) now moves *this pane's own* gate: before, a command tool auto-approved after the operator wrote `ask`; now it mounts a card and the denial holds. The write stays silent — the page is its own receipt. |
| `refusal-card.log` | **U3/D4**. A refused `web_fetch` returns no details mapping, so the card renders no `Fetched: <url> · cache miss` row for a call that never opened a connection, matching `web_search`. Both refusals name the key without the stale `: false`. |
| `settings-ladder.log` | **D2/D5**. Every affected help string measured through the real `SettingsView` detail ladder at 80/100/120 cols: nothing clips at any width. |

## Frames (rendered and viewed)

| file | what it shows |
|---|---|
| `keep-notice-80.png` / `-100.png` | The approvals keep notice, and a bare `/approvals` disclosing the divergence (`ask (this session) … config.yml says auto`). Both wrap whole. |
| `parked-card-note-80.png` / `-100.png` | **U6**: the parked card now says `config.yml set approvals to auto; this call still needs your answer, later calls will not`, so the frame no longer contradicts the notice above it. Every option label stays visible. |
| `settings-webfetch-80.png` / `-100.png` | **D2/D5**: the Web fetch footer ends on `call.` with no mid-clause stop and no literal backticks. |

## Known residual (reported, not hidden)

The designer's amended copy removes the clip everywhere, but at **exactly 100
cols** three rows (`hosting`, `model_name`, `tool_approval_mode`) still shed
their YAML key path where the pre-PR copy kept it — the amended strings are
72/72/74 cells against per-row budgets of 71/68/58. The key returns at 120 cols
and is present at 80. Measured in `settings-ladder.log`; called out in the
design-remediation comment rather than silently resolved, since the copy was a
designer decision.
