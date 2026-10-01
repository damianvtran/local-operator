# ZERO CLASSIFICATION — arm 1830 r1, task_009 and task_013 (records only)

Manager questions: are 009's `getJSON is not defined` and 013's bare `0.0` apparatus-attributable
or capability misses? Nothing changed; arm frozen.

## task_009 — **APPARATUS-ATTRIBUTABLE** (evaluator read a shadowing directory-listing tab)

Chain, from the record and the shipped assets:

1. `getJSON` is a **page helper**, defined in the task's own shipped app:
   `HKU-RIMS-System/common.js:87` (`function getJSON()`), and **every** app page loads `common.js`
   (index.html, research-form.html, research-output.html, …). So a correctly-loaded app page
   always defines it.
2. The evaluator (`gated/tasks/task_009.py:177`) calls `get_activate_tab_json` with
   `tab_prefix = "file:///home/user/Desktop/HKU-RIMS-System/"` — the **bare directory**.
   The getter takes the **first** Playwright page whose URL starts with that prefix
   (`chrome.py:2095`) — which includes every app page *and* the directory itself.
3. The run's final frame (artifacts `7ca13b81…`, the settled state) shows the Chrome window with a
   tab titled **"Index of /home/user/Desktop/HKU-RIMS-System/"** — a **directory listing**, whose
   URL is *exactly* the bare prefix. The same window also holds the real app tabs
   (`Research Output…`, `research-form.html?ty…`), which load `common.js`.
4. The evaluator's stderr shows it **matched that bare-directory URL** and then failed:
   `[ACTIVATE_TAB_JSON] Matched tab url 'file:///home/user/Desktop/HKU-RIMS-System/'` →
   `getJSON is not defined in target page`. A directory listing runs no page scripts, so the
   check is guaranteed to fail there.

**Verdict: the evaluation did not read the model's work — it read a stray directory-listing tab.**
009 must be **excluded from the capability rate** like F1/F2, not counted as a model miss.

**What this does and does not prove.** It proves the zero came from the wrong page, not from the
model's result. It does **not** prove the model's `sessionStorage['researchForms']` held the correct
answers (the evaluator never read the app page, and the guest is gone) — so 009 is
"not evaluated as intended", not "verified correct".

**Repair direction (not applied):** the prefix is too loose. Either pin it to a specific page
(`…/research-form.html` / whatever page holds the submitted data) or make the getter prefer a tab
where `typeof getJSON === 'function'` before falling back — i.e. scan all prefix matches and pick
one that can answer. Reproduced shape: any episode whose Chrome has the folder open.

## task_013 — **CAPABILITY MISS** (the evaluator checked, and the result was wrong)

013's artifact **does** carry diagnostics (the premise of "no diagnostic" is a reading gap), and the
record shows the check ran to completion:

- stderr: `INFO:desktopenv.getters.state:[GET_STATE_WITH_COOKIE] Saved state to …/013/state_fetched.json`
- cache dir contains **both** `state_gt.json` (20,823 B — the expected answers, fetched) and
  `state_fetched.json` (3,879 B — the model's state, fetched).
- `evaluate()` returns `0.0` early only `if not expected or not result` — both were present, so the
  guard did not fire and the comparison ran. `evaluator_result: 0.0`.

**Verdict: genuine "checked and failed" → capability-attributable.** (013 also completed cleanly:
`completed/finish`, 25 steps.) This matches 1748's 013 (completed/26/0/0).

## General answer on "scored but silent" zeros (for the report)

- **Scored vs unscored is always distinguishable**: every scored episode carries a score artifact
  with `evaluator_result`; "we could not score" is a separate `unscored` outcome the runner decides,
  never an adapter zero.
- **"Ran vs did not run" is usually distinguishable** from the diagnostics block — the adapter
  captures the evaluator's `stdout`/`stderr`/logs and a cache manifest, so a getter's log line or a
  fetched cache file is positive evidence it ran (as 013's is). But the block is **additive**: an
  evaluator that emits nothing and fetches nothing produces **no block** — that is the genuinely
  ambiguous case.
- **"Why it failed" is often NOT recoverable**: when the evaluator returns a bare float (013),
  nothing in the record names the failing checkpoint. Only evaluators that print per-checkpoint
  detail (as task_003 does: `City/Filter/LLM/Transp`) leave a reason. So a silent `0.0` with a
  diagnostics block can be classified **ran-and-failed** but not *why*.

## Scoreboard impact

- 009 → apparatus-attributable (exclude from capability rate).
- 013 → capability miss.
- 003 → apparatus-attributable (unusable evaluator-judge provider; separate record).

Nothing in the tranche was changed.
