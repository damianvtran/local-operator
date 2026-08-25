# PR body source material (branch held pending cmux re-validation)

Assembled while the evidence was fresh so the PR can be opened without
re-deriving any of it. Not intended to ship as documentation — delete or leave,
but do not treat it as user-facing.

## Title

`feat(multiplexer): resume the right session in each pane after a crash`

## Summary

~15 `lop` sessions run as cmux sidebar workspaces. When cmux crashes the
workspaces return but every surface opens a fresh shell and a fresh `lop`, and
all conversation continuity is lost — the sessions survive on disk under
`~/.local-operator/sessions` but nothing maps a pane to the session it held.

This publishes that missing fact per pane: *this pane holds session `<id>`, and
here is the argv that reopens it*. cmux gets a real resume binding over its
control socket. tmux/wezterm/zellij/screen have no resume API, so they get a
documented, discoverable marker a restore script can consume. Backends sit
behind one interface in a registry, so another multiplexer is a new module.

Minor bump: 0.33.1 -> 0.34.0.

## Design points worth calling out in review

- **`source: agent-hook` over the CLI.** `cmux surface resume set` hardcodes
  `source = "cli"`, which cmux resolves to `approvalPolicy = .manual`,
  `autoResume = false`. Only the socket RPC can reach auto. A future
  simplification to the CLI would still create a binding and still print one,
  and auto-resume would be silently dead. Commented at the top of
  `multiplexer/cmux.py`.
- **The re-assert timer is justified by cmux's SOURCE, not by a measurement
  here.** `SharedLiveAgentIndex.cacheTTL` is 60s; a binding published against
  an index snapshot older than this process is retired by
  `retireAgentHookResumeBinding`, which latches `autoResume = false`
  permanently. `REASSERT_INTERVAL_S = 90` therefore has to stay above that TTL.
  The measurements on this host can neither confirm nor refute the race (see
  caveat below), so the defence is retained on the strength of the Swift.
- **Restore-and-idle is a safety boundary, not a default.** `resume_argv` is
  the single place the command is built and it can only ever emit
  `lop --resume <id>`. Fifteen panes restoring unattended must not resume tool
  execution.
- **Two gates, different lifetimes.** Subagent-ness is permanent and decided
  once (`origin.json`); resumability is transient and re-checked before every
  publish, because a cold session has no transcript until its first turn lands.
  Checking resumability once at startup would mean every cold session — most of
  them — silently never publishes, discovered only after a crash.
- **Headless gate on the app hook.** This suite runs inside cmux and
  `tests/conftest.py` does not clear `CMUX_*`, so without it every pilot test
  would rewrite the resume binding of the session running the tests. A
  paired test proves the gate is what blocks it: same arrangement, gate
  lifted, produces `surface.resume.set`.

## Testing evidence

Full artifacts in `docs/evidence/multiplexer-resume/`. **Read
`cmux-host-caveat.md` first** — it bounds every cmux claim below.

**Strongest evidence — it reaches the file a crash restores from.**
`cmux-on-disk-binding.txt`: read back out of
`~/Library/Application Support/cmux/session-com.cmuxterm.app.json` (written on
cmux's autosave, not on clean shutdown) while a binding was in force:
`kind=local-operator`, `checkpointId=c9a4b15eed9e`, `source=agent-hook`,
`autoResume=true`, `approvalPolicy=auto`. Identity verified rather than
eyeballed: `panels[0].id == CMUX_SURFACE_ID` and
`workspaces[8].workspaceId == CMUX_WORKSPACE_ID`, both exact. An in-memory
binding that never reached disk would be the classic "looks fixed and is not".

**Live publish/retire** (`cmux-publish-retire.txt`): a real session publishes
`source: agent-hook`, `approval_policy: auto`, `auto_resume: true`, with
`checkpoint_id` equal to its real directory name; binding gone after retire.

**The four refusals** (`negative-cases.txt`), all publishing nothing: subagent
session, kill switch, no multiplexer in env, cmux binary unresolvable.

**tmux for real** (`tmux.txt`, reproducible via `tmux-evidence.sh`): against a
private tmux server on its own socket, both pane options written and read back
with `show-options -pv`, and after retire both genuinely *unset* (`rc=1`) rather
than blanked — a blank option would read to a restore script as a session with
an empty id.

## Honest limitations (must survive into the PR body)

- The cmux host was running from a **deleted binary** during capture
  (`Contents/MacOS/` empty; app executing an old in-memory image). Behaviour was
  not self-consistent across the update boundary: bindings held for minutes,
  then retired in 2-8s, and in one control run a binding stayed
  `auto_resume: true` for 75s **after its agent process was killed** — which a
  working `isStaleAgentHookBinding` should have retired. That last result says
  cmux's liveness retirement was not running at all.
- Consequently: the 45/45-samples-true duty cycle measured here is **not** a
  product guarantee, and it was identical with and without the vault
  registration, so this host did **not** isolate the registration's causal role.
- The vault registration is documented as required because cmux's own
  `docs/vault.md` requires it for a custom kind — **not** because it was
  verified here to be what holds the binding. Reviewers should not infer
  otherwise.
- **Outstanding: clean re-validation on a healthy cmux install.** The operator
  restarts cmux at a time of his choosing; that restart is also the first
  real-world test of this feature.

## Gates

- `flake8 .` clean
- `black --check` (26.1.0) clean, `isort --check-only --profile black` clean
- `pyright` 0 errors, 0 warnings
- Unit suite: 6788 passed, 7 skipped. The 2 failures seen
  (`test_comms.py::test_a_resumed_child_replays_the_stopped_ones_transcript`,
  `::test_a_cancelled_child_keeps_the_work_it_had_already_done`) are
  **pre-existing flakes**: clean `origin/main` fails the same files with a
  *different* test on each full run (four distinct failure sets observed across
  runs), and both pass in isolation on both branches.
- The 45 new tests pass 4/4 consecutive runs with no flakiness.
