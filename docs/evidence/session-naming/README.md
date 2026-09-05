# Session naming capability repair

## Failure and scope

At base `619ec60a`, GPT-6 inherited GPT-5's effort ladder. The foreground
session could use `high` successfully, but the isolated naming request selected
its advertised lowest rung, `none`. The provider rejected it and the existing
best-effort naming wrapper retained the opener excerpt. Both reported sessions
had no persisted generated title; no private session transcript is included here.

The repair is in the shared capability policy, not a naming-only model override.
GPT-5 retains its prior ladder; GPT-6 uses `low`, `medium`, `high`, `xhigh`, `max`.
Provider discovery still takes precedence. Later generations use the newest
known family fallback, not a guarantee about an unobserved future provider.

Source: <https://developers.openai.com/api/docs/models/gpt-6-astra>.
The official migration guide also explicitly rejects `none` and `minimal`:
<https://developers.openai.com/api/docs/guides/latest-model.md>.

## Live provider evidence

The temporary diagnostic script used `Session.complete_once` through
`create_stream_fn` and the real OpenAI Codex OAuth wire. It selected one unexpired
credential with a read-only SQLite connection, copied only that row into a fresh
isolated HOME/config, and refused to refresh it. No live session was attached,
steered or edited. Only synthetic prompts and request/usage metadata were logged.
The baseline mode loaded the committed base effort module into that isolated
process; no source checkout was reverted.

Commands executed from the implementation worktree:

```sh
LOP_NAMING_BASELINE=1 .venv/bin/python /tmp/lop-naming-probe.py before-clean
.venv/bin/python /tmp/lop-naming-probe.py after
.venv/bin/python /tmp/lop-naming-probe.py after-short 'Fix session auto naming for reasoning models'
.venv/bin/python /tmp/lop-naming-probe.py after-long 'Please diagnose why two conversations keep the first message excerpt as the title rather than a short generated name. Check the actual provider request and preserve independent foreground model settings while repairing the capability metadata.'
```

All calls used `gpt-6-astra`, `max_tokens=1024`, `isolated=true`; selected
foreground effort remained `high`. The default synthetic opener was
“Fix the login redirect loop when a browser session expires”. Actual results:

| Run | Errand effort | Result | Seconds | Output / reasoning tokens |
| --- | --- | --- | ---: | ---: |
| Base | none | HTTP 400; generated title null | 1.112 | not returned |
| Fixed, default opener | low | Fix Expired Session Login Redirect Loop | 2.486 | 16 / 0 |
| Fixed, short opener | low | Fix Session Auto Naming for Reasoning Models | 2.733 | 17 / 0 |
| Fixed, long opener | low | Diagnose Conversation Titles and Capability Metadata | 2.788 | 36 / 18 |

The real baseline error was:

```text
Unsupported value: 'none' is not supported with the 'gpt-6-astra' model.
Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.
```

Every fixed run ended with `stop_reason=stop`, `error=null`. The unchanged
15-second deadline and 1024-token ceiling suffice in these measurements.
An initial QA setup was invalid: its partial OAuth identity copy reached the
wrong auth route and returned HTTP 401 (missing model.request scopes). Repeating
with the full single credential row reproduced the baseline 400 and fixed
success; that setup failure is not evidence of a naming latency problem.
The existing single-attempt, no-failover, later-message retry behavior is unchanged.

QA additionally verified live Claude Sonnet 4.6 naming at low effort (1.302s),
missing credentials (no-key failure), and invalid credentials (HTTP 401, no retry).
Live GPT-5.4 comparison was blocked because the account's Codex route did not
support it; GPT-5 preservation is covered deterministically, not claimed live.

## Real application frames

QA drove the **real OperatorApp**, using Textual `run_test` at 110×32, with its
actual stylesheet. The main answer was deliberately synthetic (and labelled as
such in every populated frame); **the title completion was a real provider call**.
Baseline captures loaded the exact base effort module into a fresh interpreter;
the rest of the application code was unchanged, not a separate baseline checkout.
Empty, loading, settled,
a subsequent settled frame, and `/effort` output were captured as SVGs and
rendered to the committed PNGs. The baseline's settled state is also the naming
error state: the best-effort request fails without interrupting the main turn.

| State | Baseline | Fixed |
| --- | --- | --- |
| Empty | [frame](baseline-real-empty.png) | [frame](after-real-empty.png) |
| Loading / provisional | [frame](baseline-real-loading.png) | [frame](after-real-loading.png) |
| Settled | [frame](baseline-real-settled.png) | [frame](after-real-settled.png) |
| Consecutive settled | [frame](baseline-real-consecutive.png) | [frame](after-real-consecutive.png) |
| Effort command output | [frame](baseline-real-effort-picker.png) | [frame](after-real-effort-picker.png) |
| Effort autocomplete picker | [frame](baseline-autocomplete.png) | [frame](after-autocomplete.png) |

QA's successful real-app run reported a persisted title of
`Fix Expired Session Login Redirect Loop`, no provisional label remaining, and
an isolated `low` request with no temperature override. The foreground still
showed `high`. `/effort` dropped the invalid `none` choice and added `max`.
The actual autocomplete picker was captured separately and shows the same
corrected choices. Captures predate the release-only version bump.

Geometry after submit, for loading/settled/consecutive/effort states, was stable:
status region `[2, 28, 106, 2]`, content size `[105, 1]`, virtual size `[106, 2]`,
no horizontal or vertical scrollbar. Consecutive settled frames showed no reflow.
The empty welcome state has its existing centered geometry, not the docked
post-submit geometry; no stylesheet, layout or scheduling code changed.

## Regression sensitivity

The new six-case naming regression runs the real Session and provider wire
shapers with an HTTP MockTransport (not a live endpoint). It covers public
Responses, Codex Responses, and aggregator chat completions, each with the
configured cheap tier or the effective session model. It deliberately removes
warm listing metadata so the offline fallback is actually under test.

Executing those corrected fixtures with only the effort policy from `619ec60a`
loaded in the test process produced **6 failed** (all titles null). On the fix,
they produced **6 passed**. Requests remain single-attempt, isolated, capped at
1024, without rejected sampling parameters; foreground effort and transcript
remain unchanged. Family tests cover GPT-5 preservation, GPT-6 direct/prefixed
and unlisted variants, future-generation fallback, and none/minimal/max clamping.

## Integration onto updated main

After the original proof above, merged `origin/main` at `3b100234` (PR #628)
with a normal merge, retaining the original history. Reserved patch version is
now **0.46.17** to avoid other coordinated release branches. The original
`619ec60a` reproduction and screenshots remain historical evidence, not claims
that the baseline changed.

On the integrated source, ran the same live naming probe:

```sh
.venv/bin/python /tmp/lop-naming-probe.py integrated
```

Actual result: `low`, `max_tokens=1024`, `isolated=true`; output 16 tokens,
reasoning 0; terminal `stop`, no error; title
`Fix Expired Session Login Redirect Loop` in **5.030s**, selected foreground
`high` unchanged.

Integration regression command:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest \
  tests/unit/model tests/unit/providers tests/unit/session/test_session.py \
  tests/unit/session/test_naming.py tests/unit/tui/test_conversation_naming.py \
  tests/unit/session/runtime/test_owned.py tests/unit/test_session_factory.py -q
```

Result: **1946 passed, 18 failed** in 28.81s. Every failed node ID exactly
matches the 18 already demonstrated baseline resume failures in
[baseline-failures.txt](baseline-failures.txt); no naming/model/provider/lifecycle
failure appeared. Full-tree flake8, black 26.1.0 (891 files), isort 5.13.2,
and pyright were repeated and passed on the integration; pyright reported
zero errors and warnings.
The full unit suite was not repeated locally for unchanged majority code;
current-head CI supplies the full integrated run.

## Original whole-tree gate outcomes

Flake8, black 26.1.0 (871 files), isort 5.13.2, and pyright over the whole tree
passed; pyright reported zero errors and warnings. Final focused naming,
session, effort and TUI lifecycle files: 221 passed in 17.58s.

The full unit command (`env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m
pytest tests/unit -q`) reported **19 failed, 12958 passed, 15 skipped** in 931.99s.
Eighteen failures belong to existing session-factory resume validation behavior;
the other was the background-price revalidation timing test.

A narrow comparison ran `tests/unit/test_session_factory.py` and
`tests/unit/model/test_prices.py` with `-n0 -q` on this tree and in a separate
process loading the changed production module from base `619ec60a` instead.
All other implicated production/test files were verified unchanged. Both runs
returned **18 failed, 138 passed**, with identical failed node IDs; the price
test passed on both isolated runs. See [exact failures](baseline-failures.txt).
The manager accepted these demonstrated unrelated baseline exceptions rather
than expanding this repair into resume/price changes. The full suite is not
claimed green.

No auth, retry, timeout, output extraction, prompt, persistence, or UI layout
implementation changed. The scratch live scripts and isolated credentials are
not committed; the deterministic regression fixtures are in the test suite.
