# QA matrix — hub auto-update (local-operator #1800 @ `69880475d`, local-operator-ui #667 @ `e0e81c696b`)

Run: 2026-09-30, qa-tester (lopdev). Model note: this session ran on `deepseek/deepseek-flash`
(anthropic quota exhausted, provider fallback mid-session). Every cell below is an execution
result read from the tool/bridge output, not a restatement of the design.

## Environment (isolation)

| Piece | Value |
|---|---|
| Backend binary | `~/local-operator-worktrees/hub-auto-update-0929/.venv/bin/local-operator` (PR head `69880475d2a1515bd8cd57f2e640bf432c986a53`) |
| UI build | `~/local-operator-ui-worktrees/hub-update-indicators-0929`, `electron-vite build` at head `e0e81c696b`, built with `VITE_LOCAL_OPERATOR_API_URL=http://127.0.0.1:11522` |
| Config A (author/push side) | `<scratch>/hubqa/a/.local-operator` — port 11521 |
| Config B (pulled/auto-update side) | `<scratch>/hubqa/b/.local-operator` — port 11522 |
| Auth store | a **read-only COPY** of the operator's live store (`copy_auth.py`: radient OAuth + deepseek api_key, row-for-row, 0600, values never printed). The live store was never written: `live calls where session_id like 'hub%' -> 0`. |
| Launch | `env -i HOME=<iso> LOCAL_OPERATOR_CONFIG_DIR=<iso>/.local-operator LOCAL_OPERATOR_DESKTOP_TOKEN=<synthetic> PATH=$PATH TERM=xterm-256color` — every `CMUX_*`/`LOP_*` absent (backend AGENTS.md recipe) |
| Org | `5b72d9f4-8ce7-4821-ac8a-f92c365ed5b3` (operator's), throwaway agent `LO-hub-autoupdate-QA` → hub id `a4e7495a-801a-4628-92bd-e6ccbb9f8214` |
| Merge model | default (config `hosting=deepseek`, `model_name=deepseek-flash`), `hub.merge_model=""`; `resolve_merge_model` → `deepseek/deepseek-flash` |

## 1. Real-org end-to-end — the runner merges a hub update without clobbering local work (**PASS**)

| # | command | actual output |
|---|---|---|
| 1 | `POST /v1/agents` (A) + `PUT …/system-prompt` (v1, 5 sections) + `POST /v1/agents/{id}/publish?visibility=org&tenant_id=…` | 201 / 200 / **200** `{"agent_id":"a4e7495a…","version":"1.0.0","document_version":1,"moderation":{"verdict":"allow"}}` |
| 2 | `lop agents pull --id a4e7495a… --org 5b72d9f4…` (config B) | exit 0, `Successfully pulled agent 'LO-hub-autoupdate-QA' (ID: d6e79086…)`; baseline record written: `{kind: agent, hub_id: a4e7495a…, tenant_id: 5b72d9f4…, recorded_by: pull, fields: [description, instructions]}` |
| 3 | `PUT /v1/agents/{b}/system-prompt` — **local edit** (adds a `## Tooling` sentence) **+ intentional removal** (deletes the whole `## Escalation` section) | 200 |
| 4 | `PUT /v1/agents/{a}/system-prompt` (v2, `## Scope` rewritten — a **different part**) then `PUT /v1/agents/{a}/publish?visibility=org&tenant_id=…` | 200; the hub copy really moved — a fresh pull into a third config shows `QA-HUB-AUTOUPDATE-REVISED` present |
| 5 | runner fired it — startup tick `13:05:58Z` (saw the pre-bump remote: `up-to-date`), then **timer tick `13:11:08Z`** | `last_tick {"reason":"timer","checked":1,"available":1,"applied":1}`, item `state=applied`, `summary {taken-remote:1, kept-local:1, removal-honored:1, combined:0, unresolved:0}`, `applied_backup hub/backups/agent-d6e79086….json` |
| 6 | proof it was the RUNNER, not a call: B's access log | only `GET /v1/desktop/hub/updates` (my store read) and one `POST …/check` that **422'd** on an invalid body — no successful manual check or apply ever reached B |
| 7 | (a) remote landed (b) local edit survived (c) removal not regrown (d) report+backup | merged text carries `QA-HUB-AUTOUPDATE-REVISED` + the new Scope sentence (a); `Always confirm the destination path before writing anything.` present (b); `## Escalation` **absent** (c); backup written with `reason: merge-pull` holding the pre-merge local fields (d) |

## 2. Model-backed merge — a genuine conflict resolved by the model (**PASS**)

Both sides rewrote the SAME sentence differently (`…Local builds prefer the shorter phrasing.` vs `…Remote builds prefer the longer phrasing, with an example.`), staged before the `13:22:01Z` timer tick.

| # | command | actual output |
|---|---|---|
| 1 | timer tick after staging | `{"reason":"timer","applied":1}`, `summary {combined:1, kept-local:1, removal-honored:1, unresolved:0}` |
| 2 | merged text on disk | `…in different regions. Local builds prefer the shorter phrasing; remote builds prefer the longer phrasing, with an example.` — a **rewrite to combine** (semicolon join), not a concatenation of the two sentences |
| 3 | the model that ran it | direct: `resolve_merge_model(cm).label == "deepseek/deepseek-flash"` (override `''`); second, independent route report after the failure round: `engine {'mode':'llm','model':'deepseek/deepseek-flash','attempts':1,'chunks':1}` |
| 4 | spend attribution + isolation | the **isolated** analytics store has exactly one call: `session_id=hub-merge, provider=deepseek, model=deepseek-flash, ok=1, 367 in / 150 out`; the **live** store has `0` rows matching `session_id like 'hub%'` |

This is the pass the earlier rigs could not run — the completion is real, on the real default model.

## 3. Failure path (**PASS**, with a vocabulary correction)

| # | induced cause | command | actual output |
|---|---|---|---|
| 1 | `hub.merge_model="nosuchprovider/nosuchmodel"` | `POST …/apply` | 200, report `outcome=needs-review, applied=false, error_class=provider-error/request`, field `engine {'mode':'deterministic','model':None,'attempts':1,'failure_class':'provider-error/request'}`, region `## Notes` = `unresolved`; store `state=available, error_class=provider-error/request, attempts=1, auto_retry=true, next_retry_at=+13 min`, **no backup, nothing partially written** |
| 2 | `hub.merge_model="garbage"` (malformed, no `provider/model`) | `POST …/apply` | report `error_class=model-unavailable` — the design's named refusal (B2.4) |
| 3 | cause removed (`merge_model=""`), credential re-seeded | `POST …/retry` | 200 `outcome=would-merge, applied=false, error_class=null`; field `outcome=merged`, `combined 1, unresolved 0`, `engine {'mode':'llm','model':'deepseek/deepseek-flash'}`; store lifted to `available` with **no** `error_class` |
| 4 | then `POST …/apply` | `outcome=merged, applied=true`; store `state=applied`, `applied_backup` written |

**Correction to the brief's expectation.** A broken merge model does **not** produce `state=failed`.
Per `store.py:record_failure` (`store.py:417-427`) `failed` is reserved for `prompt-too-long` /
`hub-item-missing`, and for a `provider-error` on an item that never had a successful check; an
item whose check succeeded lands `available` **with** `error_class` — by design (B4.3). That is
still the failure the UI renders as a fault: `hubMarkFor` returns `failedMark(item)` for any
`available` item carrying an `error_class` (hub-updates.ts:236-240), and the frame proves it.

## 4. Route matrix (server B unless noted) (**PASS**)

| surface | command | actual output | verdict |
|---|---|---|---|
| capabilities | `GET /v1/capabilities` | `features.hub_updates = 1` | PASS |
| store read | `GET /v1/desktop/hub/updates` | 200 `{credential:"ok", settings:{auto_agents,auto_teams,interval_min}, counts:{…}, items:[…]}` — no network | PASS |
| unauthorized | same, no header | **401** `Desktop authorization is required.` | PASS |
| unauthorized | same, wrong bearer | **401** (identical sentence — no token oracle) | PASS |
| invalid body | `POST …/check {"request_id":"not-a-uuid"}` | **422** `The request has invalid fields.` (the `RequestID` is a uuid pattern) | PASS |
| unknown item | `POST …/apply {"kind":"agent","name":"ghost…"}` | **404** `no linked agent named 'ghost-agent-that-does-not-exist'` | PASS |
| wrong kind | same `name`, `kind:"team"` | **404** `no linked team named 'LO-hub-autoupdate-QA'` | PASS |
| invalid enum | `"prefer":"bogus"` | **422** `The request has invalid fields.` | PASS |
| destructive guard | `"replace":"remote"` without `confirm_replace` | **422** `replace discards one side of the item; pass confirm_replace: true to confirm` | PASS |
| strict body | extra field `"bogus":1` | **422** (extra="forbid") | PASS |
| apply-all | `POST …/apply-all` with nothing available | 200 `{reports:[], counts:{}}` | PASS |
| wrong-tenant / no-credential | org item + missing OAuth | tick `credential:"none"`, item `error_class=no-credential`, **state preserved** (`available`), no attempt counted, `next_retry_at` cleared — the informational class, not a fault | PASS (observed live when the operator's runtime rotated the OAuth token under my copy) |
| concurrent-edit 409 | not reached: the cross-process lease needs two live writers at the same instant. Design note 12 puts it in `service.apply_items`; the runner's `run_exclusive` lock and the 30 s lease wait are unit-covered. | — | **NOT COVERED (route level)** |
| side effects | store transitions, `hub/backups/*.json` written per apply, `hub/.status.lock`, `hub/baselines/*.json`; no credential value anywhere in `status.json` or the logs | PASS |

## 5. CLI spot (**PASS**)

| command | actual output |
|---|---|
| `lop hub status` | `Agent Hub updates: 0 available, 0 failed, 0 updating (auto: agents on, teams on; checked every 5 min; login ok)` + `agent LO-hub-autoupdate-QA: applied` (exit 0) |
| `lop hub status --json` | the route payload verbatim (`auto_will_apply`, `remote_fingerprint`, `summary`, …) |
| `lop agents sync --check` (diverged item) | `LO-hub-autoupdate-QA: hub update available (both-changed)` — report-only, exit 0 |
| `lop teams sync` (nothing linked) | `no linked teams`, exit 0 |
| `lop teams link` (no args) | argparse usage, exit 2 |

`lop agents sync`'s merge path was exercised through the same `apply_items` the CLI calls
(the route), not as a separate CLI run — see gaps.

## 6. UI frames — live daemon, real Electron, real pointer events (**PASS**, from the built PR head)

Rig: the branch's own `scripts/renderer-driver.mjs` (Electron in `--window-mode=headless`, its own
CDP bridge), assembled with a scene into the scratchpad, run as
`node driver-qa.mjs --scene qa-hub-marks --backend http://127.0.0.1:11522 --backend-records <B>/run/serve`.
The driver's own isolation checks passed on every run: *"the app holds a connection to this run's
backend"* PASS, *"the app holds NO connection to the operator's own backend (http://localhost:1111)"*
PASS, *"the app's logs went to this run's scratch tree"* PASS, *"no process from this run outlived
its boot"* PASS, *"the harness is driving the Electron this branch pins"* PASS.

| state | what the app painted (`data-hub-mark` / `aria-label`) | frame |
|---|---|---|
| update available, **manual mode** (`auto_update.agents=false`, `auto_will_apply:false`, classification `remote-only`) | `mark="available" label="Update LO-hub-autoupdate-QA from the hub"` | `hub-available-manual-before.png` |
| click → applied (same run, real pointer press) | press → `mark="applied"`; store `state=applied`, `summary {taken-remote:1}`, backup written | `hub-available-manual-after.png` |
| needs a decision (`classification: both-changed` — a conflict) | `mark="review" label="Review the hub update for LO-hub-autoupdate-QA"` (pressing opens the detail pane, never applies) | `hub-manual-available-before.png` |
| failed + retry (`error_class=model-unavailable`, retryable) | `mark="failed" label="Retry the hub update for LO-hub-autoupdate-QA"` | `hub-failed-before.png` |
| retry pressed (cause removed) | press → `mark="updating"` (the spinner); the settled answer was `would-merge` — failure retired, offer restored | `hub-retry-recovery-before.png`, `hub-retry-recovery-after.png` |
| merge preservation | the store's `summary {taken-remote:1, kept-local:1, removal-honored:1}` over frames 1–2 is the preserved-work case; the merged text is in §1.7 | — (store evidence) |
| up-to-date baseline (no mark at all) | `mark=null` while the daemon's store held the item `up-to-date`; the row is otherwise unchanged and the reserved slot is empty | `hub-uptodate-before.png` |

Every frame above was taken with the driver's own boot assertions passing in the same run: it holds a
connection to this run's backend (11522) and **none** to the operator's (`http://localhost:1111`), its
logs went to the run's scratch tree, and no process outlived the boot.

## Findings (Q-findings)

- **Q1 [minor, non-blocking] — the org republish does not bump the listing's version.** Three
  `PUT …/publish?visibility=org` calls that demonstrably changed the hub document all answered
  `version:"1.0.0", document_version:1`. Functionally harmless (the design's `remote_fingerprint`
  is computed from content — the runner saw all three), but the response field is not a usable
  "did it move?" signal for a caller.
- **Q2 [minor, non-blocking] — a locally authored + published agent is not tracked by auto-update.**
  After `POST …/publish` on the author row, the row has `tags: []` and no baseline record:
  `record_agent_baseline` is called only from the pull/import path (`agents.py:2537`) and the
  check's repair arm (`check.py:228`). The design (A2.2) lists *publish* as a writer of B, but the
  push side is explicitly the other session's half (Q1/Q5), so this is a scope observation rather
  than a defect in #1800. Worth an explicit sentence in the PR so a reader does not expect a
  published row to appear in the sidebar.
- **Q3 [informational] — `failed` is narrower than the brief's example implies.** See §3; recorded
  so the merge decision is not made against a wrong expectation.
- **Environment note (not a product finding).** The operator's own live runtime rotates the
  Radient OAuth refresh token (~15 min cadence, observed 13:15:01Z). A scratch copy taken at
  t0 stops resolving after that rotation: the runner then records `credential:"none"` /
  `no-credential` and keeps the item's state (correct and non-destructive). Every credential-
  dependent step in this pass re-seeded the copy immediately before it; a future rig should do
  the same or the run will look like a product failure when it is a stale login.

## Gaps — what this pass did NOT run

1. **Teams sync path (task §1's "repeat on a TEAM").** Not run. `lop teams sync` against a store
   with no linked teams was exercised (`no linked teams`, exit 0), and the team arm's code path
   was not driven end to end against the live org. The team publish/pull tooling is a separate
   rig (agent-server + scratch Mongo, `qa-evidence/hub-team-publish/`); bringing it up was out of
   this round's budget.
2. **Concurrent-edit 409** at the route level (two live writers inside one lease window). Unit-
   covered in the PR; not reproduced here.
3. **`lop agents sync`'s own CLI merge run** (the merge was driven through the route/runner, which
   is the same `apply_items`). `--check` was run.
4. **The whole-tree unit suite / pyright** — deliberately left to CI per the compute directive.
5. ~~Up-to-date baseline frame~~ — captured later in the same pass (`hub-uptodate-before.png`).
