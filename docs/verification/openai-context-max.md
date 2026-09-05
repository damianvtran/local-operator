# OpenAI active context verification

## Contract

Use the selected ChatGPT account's supported maximum as the active context
window, preserving its provider default separately. Keep API-key limits and all
compaction settings independent. Publish account-rotation metadata through the
request's own stream so parent and child usage do not share a mutable callback.

## Executed local HTTP path

A task-owned `ThreadingHTTPServer` served `/models` and Codex-shaped `/responses`.
The candidate's real `AuthStore`, `SessionStreamFn`, `Session`, discovery/cache,
HTTP wire client, event loop and frontend projection were exercised. Only the
remote provider was replaced; the local account used a fabricated test token.
A fresh isolated `HOME` prevented reads of the operator's config or caches.

```
$ HOME=<isolated-home> LOCAL_OPERATOR_CONFIG_DIR=<isolated-home>/.local-operator \
  .venv/bin/python <evidence-dir>/real_path.py
GET /models: 200, account=local-account
POST /responses: 200, model=gpt-6-astra, account=local-account
max_context_window wire flag: absent
active=872000, default=272000, maximum=872000
300000 input tokens: 34.4%/872k
compaction trigger with explicit 0.8/400000: 400000
local fixture: missing authorization 401, wrong account 403, invalid model 400
```

The HTTP fixture reports 300,000 input tokens for a short request. This proves
accounting and propagation above the old default, **not** acceptance of a
300,000-token real provider request. No huge paid generation was performed.

The coordinating agent separately exercised the candidate resolver against the
real authenticated production catalogue with isolated cache storage and no
inference. Results: Astra and the 5.6 Sol/Terra/Luna variants active/max 872,000,
default 272,000; 5.5 and mini 272,000; Spark 128,000. The explicit 400,000 trigger
remained 400,000 for the 872,000-window models.

## Rendered evidence

Real `OperatorApp.run_test`, loading the application's stylesheet, at 100x34 and
50x34. SVG exports were rendered with the installed `rsvg-convert` and the PNGs
were viewed. Populated transcript, composer, child-job usage, model picker,
changed selection and settled consecutive frames were captured.

- Baseline composer and child: `110.3%/272k` at 300,000 tokens.
- Candidate composer and child: `34.4%/872k`.
- Wide picker: `872k max · provider default 272k`.
- Narrow picker: `872k max` remains while prices drop.
- Opt-out: wide provider-default-active label; narrow `272k active`.
- Equal-limit mini retains its ordinary single 272k label.
- No percentage clamp: genuine overflows remain above 100%.
- Picker content/virtual dimensions stayed 96x3 (wide), 46x3 (narrow), with no
  scrollbar, through settled frames and selection changes.

The negative HTTP statuses above validate the local fixture's failure paths;
provider failover/rotation behavior is separately exercised by regression tests.
They are not represented as real OpenAI authorization responses.

Original pre-edit captures are retained. Matched seeded baseline frames use the
original picker row methods from the base commit in the same app host; no user
work was stashed or overwritten. Fixture prices are representative, not billing
verification. Long-context pricing and surcharges are outside this change.

## Regression coverage

`tests/unit/model/test_openai_context.py` covers default/max parsing and cache
round trips, invalid maxima, complete public registry bypass for OAuth,
account-scoped memoization/token refresh, account rotation including replayable
calls, API-key separation, missing-account/offline fallback, opt-out/default
behavior, session/frontend adoption, primary recovery, cold resume, current-row
metadata, child statistics, unchanged compaction threshold and overflow display.
Settings consumer-default and live-configuration tests include the new key.
