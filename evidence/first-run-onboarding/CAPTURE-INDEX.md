# Evidence — first-run onboarding (Lane B), PR #2071

Everything here is a rendered artefact of the **real** application, captured by
rigs in the authoring session's scratchpad (never in a repository tree). No
images are committed inside the PR's own tree — this branch is the accepted
home for them, per `AGENTS.md` §7.

## How these were captured

| Set | Rig | What it drives |
| --- | --- | --- |
| `frames/01…14`, `frames/15…16` | `shots/onboarding_shot.py` | the real `OperatorApp` (the app that loads `local_operator.tcss`) through `run_test`, `app.save_screenshot()` → SVG → PNG. No fake host: the setup state is reached by a session factory that raises `HostingNotConfiguredError`, exactly the real first-run path. |
| `frames/15`, `frames/16` | `rig/attended_greet.py` | the real app with a **real session factory** (`session_factory.create_session`) and a **real model turn**, against the provider `RADIENT_API_KEY` from the operator's encrypted store (injected into the rig's environment; never printed, never written to this branch). Isolated `HOME` + config dir. |

## The frames

| # | Frame | Shows |
| --- | --- | --- |
| 01 | `before-setup-80` | setup splash BEFORE — `! /login openai to get started — no provider configured`, hint table `/login · set up a provider`, `v0.68.7 / setup`, tip `/login <provider> sets up a provider (e.g. /login openai)` |
| 02 | `after-setup-80` | setup splash AFTER — `! Connect an AI account first: type /login radient`, checklist `/login radient  sign in (recommended)` / `/login  other providers or an API key` / `then  Aida says hello and helps you set up` / `ctrl/cmd+d  quit`, tip `New here? Radient is one browser sign-in; no key to paste`, composer `Type /login radient to begin` |
| 03, 04 | `before/after-setup-120` | the same pair at 120 columns (the cue renders whole at both widths) |
| 05 | `before-login-picker-120` | `/login ` BEFORE — flat registry order, `openai` first, `radient` ~20th, unexplained `*` marker on flavour rows, blanks in the description column |
| 06 | `after-login-picker-120` | `/login ` AFTER — `radient` first tagged `recommended`, grouped like the desktop, a description on every row |
| 07, 08 | `before/after-login-picker-80` | the picker at 80 columns |
| 09 | `before-radient-login-80` | `/login radient` BEFORE — `opening your browser to authorize…` then a 300-character OAuth URL across five rows |
| 10 | `after-radient-login-80` | `/login radient` AFTER — `Opening your browser to sign in to Radient (…)`, `Didn't open? http://…:54549/launch`, `Finished in the browser? This window continues on its own · ctrl+c cancels`, full URL last in the dimmest ink |
| 11, 12 | `before/after-api-key-login-80` | a pasted-key login — the new chat key rows (`anthropic-key` / `openai-api-key`) with the same copy |
| 17 | `before-radient-login-120` | `/login radient` BEFORE at 120 columns (same URL dump) |
| 18 | `after-radient-login-120` | the same, AFTER |
| 19 | `before-api-key-login-120` | a pasted-key login BEFORE at 120 columns |
| 20 | `after-api-key-login-120` | the same, AFTER |
| 13 | `before-refusal-120` | a message typed at the setup splash, BEFORE — `! your message was not sent: /login openai to get started — no provider configured (/provider lists all). — edit e`, toast `No provider configured` |
| 14 | `after-refusal-120` | the same, AFTER — `! Not sent — connect an AI account first: type /login radient — edit e`, toast `Connect an AI account` (shorter, names the recommended command, and the apology for the length is gone) |
| 15 | `real-run-attended-greeting` | **the real thing**: real provider turn; Aida's reply is the only visible row, no trigger line, no wake receipt card, no user row |
| 16 | `real-run-signed-in-radient` | the same with a Radient identity present (OAuth `id_token` claims): she greets **by name** and confirms the email instead of asking for it |

## Measured logs

| File | Contents |
| --- | --- |
| `geometry.json` | grid geometry behind every frame (cell metrics from the 80/120-column stills; the two real-run stills' viewBox and the caveat that it is the whole composed screen box) |
| `attended-greeting-digest.json` | ledger timeline (`armed → delivered`, 2.4 s apart), `surface=tui`, the trigger never in the visible transcript, the transcript row inventory with `details.hidden = true` on the `wake_prompt` |
| `attended-greeting-identity-digest.json` | the same for the signed-in-with-Radient run |
| `attended-visible-transcript.txt` | the visible rows of the real run, as the TUI painted them |
| `attended-identity-visible-transcript.txt` | the signed-in run's visible rows |
| `install-sh-uv-present.log` | `install.sh` on an empty `UV_CACHE_DIR`, uv already installed: **20 s** total (Step 2 took 15 s) |
| `install-sh-uv-installed.log` | `install.sh` on an empty `UV_CACHE_DIR` with uv absent: **28 s** total (Step 1 3 s, Step 2 18 s, Step 3 6 s) |
| `xplat-login-probes.json` | the two new probe battery rows: `login.list` PASS (`first=radient, 27 rows`), `login.api_key` PASS (`stored=True listed=True getpass_noise=False`) |
| `xplat-tui-tty-preexisting-fail.json` | the `tui.tty` FAIL this host shows — pre-existing, diagnosed in the PR's evidence comment (the probe's own teardown blocks in `waitpid` when it kills mid-boot; it fails identically with this branch's boot route disabled and with `origin/main`'s own `tui/*` files) |
| `check-changed-summary.txt` | `make check-changed`: the scope report (why it escalated to whole-tree), every job's result, the 4 failures and the totals |

## Re-running

```
# frames 01-14 (needs the worktree's venv; writes SVG stills you then rasterise)
env -i HOME=$ISO LOCAL_OPERATOR_CONFIG_DIR=$ISO/.local-operator PATH=$PATH TERM=xterm-256color \
  .venv/bin/python <rig>/onboarding_shot.py . <outdir> <scene> <cols>

# frames 15-16 (a real provider turn; the key comes from the credential store)
env -i HOME=$RIG_HOME PATH=$PATH TERM=xterm-256color \
  RIG_RADIENT_KEY="$(lop secret get RADIENT_API_KEY)" \
  .venv/bin/python <rig>/attended_greet.py --home $RIG_HOME --out <outdir> [--identity]
```

The rigs live in the authoring session's scratchpad rather than `scripts/`
because they are PR evidence, not reusable tooling; `scripts/shot_login.py`,
`scripts/shot_welcome.py` and `scripts/visual_gallery.py` in the tree are the
reusable capture harnesses they build on.

## Round 3f — the `fal` rows (design review, head `b134200e5c`)

Six frames, captured from the PR worktree's real `OperatorApp` in an isolated
`HOME` (`scripts.visual_capture.save_capture`), for the row `main` added after
this branch's last merge: `fal` (media-only, image/video) covered by the login
catalogue's three maps. Rig: `shot_fal.py` in the reviewing session's
scratchpad; every pair is settled (consecutive captures byte-identical).

| # | File | Shows |
| --- | --- | --- |
| 1 | `frames/r3f-fal-setup-picker-tail-80%2Epng` | setup state, `/login` picker scrolled to the bottom: `fal` absent, list ends at `Your endpoint` — no gap, no dangling group |
| 2 | `frames/r3f-fal-setup-listing-80%2Epng` | setup state, bare `/login` listing: no `Not for chat` group |
| 3 | `frames/r3f-fal-picker-tail-80%2Epng` | configured install, `/login` picker at the bottom: `FAL — Media only; not for chat — needs login`, whole, in context |
| 4 | `frames/r3f-fal-picker-tail-120%2Epng` | the same at 120 columns |
| 5 | `frames/r3f-fal-listing-80%2Epng` | bare `/login` listing: `connect an AI account — Not for chat` with `FAL` last |
| 6 | `frames/r3f-fal-listing-120%2Epng` | the same at 120 columns |

`measured-logs/r3f-fal-measurements.json` carries the numbers behind frames
3-5: per-scene screen size, picker width, primary/detail column cells, match
count, and the fal row's painted string (80: 24-cell description in a 37-cell
allowance; 120: 41-cell allowance — both whole, no ellipsis).

One extra frame in the same round-3f set — `frames/r3f-fal-picker-tail-60.png`: the
picker at 60 columns, BELOW the 80-column floor the setup screen is designed
for. Every description degrades by ellipsis uniformly (fal's `Media only; not…`
beside `Speech only; not…` and `Classification o…`), so the row is no worse
than its established peers below the floor.

The round-3f references above encode the final dot as `%2E` (e.g. `r3f-fal-setup-picker-tail-80%2Epng`): some harness content pipelines rewrite literal `-80.png` / `-120.png` spans in authored text, and the encoded form survives all of them while resolving to the same file (verified: every link fetches HTTP 200 `image/png`, byte-identical to the captured frame).
