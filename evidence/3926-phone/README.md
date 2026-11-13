# PR #3926 — phone ask card, suggested command (design round 2, Mac capture)

Evidence only; nothing here ships. Head under test: `3e41637a843a` (feat/ask-suggested-commands).

## How these frames were produced
- Worktree `~/local-operator/.worktrees/ask-cmd-capture` detached at `3e41637a843a`; phone bundle built with
  `vite build` against the primary checkout's `node_modules` (relative symlink, lockfile byte-identical).
- `askcmd_fixture.py` (this folder) imports the PR tree's own `scripts.probe_isolation` (fresh HOME + config dir,
  CMUX_* stripped) and `scripts.mobile_overflow_fixture` helpers, and serves ONE live, answerable runtime on
  `127.0.0.1` with 5 queued asks (no-command "before", short, ~394-char, free-text, 2-question with the command on Q2).
  Locale = `values.language` in the sandbox `config.yml` (en / ur / ru; fixture restarted per locale).
  Startup line printed `Fixture serves tree: <worktree>`.
- Driven with the harness `browser` tool (desktop app host, one background tab, closed at the end). No scripted/headless
  browser engine. Throwaway login password typed via a session secret, deleted afterwards.
- Width: the browser tool's `record_start width/height` resizes the tab's real viewport. **376x812** (375 is odd and the
  tool only accepts even sides) and **320x700**. Recording files were discarded; screenshots are the evidence.
- Theme: the app's own Appearance picker — Local Operator Dark (default) and Local Operator Light.

## Frames
| # | file | state |
|---|---|---|
| before | before-dark-376.png | same question, no command (dark, en) |
| 1 | 01-dark-short-376.png | short command, dark |
| 2 | 02-light-short-376.png | short command, light |
| 3 | 03-long-dark-376.png | 394-char command with unbroken URL/path, dark |
| 4 | 04-0-before-tap-ru-320.png, 04-1-after-tap-ru-320.png, 04-2-plus2s-ru-320.png | Copy tapped (ru, 320, longest label) — button 108.59 px before / after / +2 s |
| 5 | 05-freetext-dark-376.png, 05-freetext-light-376.png | free-text question with a command |
| 6 | 06-copy-failed-dark-376.png | clipboard write rejected (background tab) → selection + status note |
| 7 | 07-multi-q2-dark-376.png | command on question 2/2 |
| 8 | 08-ur-rtl-light-376.png | ur (RTL) |
| 9 | 09-ru-320-light.png, 09b-ru-320-long-light.png | ru at 320 px, short and long |

The "Copied" success state could not be photographed: every Copy press in the agent's background tab rejected
(`navigator.clipboard.writeText` needs a focused document), so frame 4 shows the tap → failure path; width stability is
measured instead (see the PR comment).
