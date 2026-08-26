# Evidence: multiplexer session resume

Captured on macOS 25.6.0, cmux 0.64.22 (102), tmux 3.7c, on 2026-08-25.

Read `cmux-host-caveat.md` first: the cmux host was in a damaged state during
this capture, and it bounds what the cmux numbers below can be claimed to show.

| File | What it shows |
|---|---|
| `cmux-publish-retire.txt` | A real publish and retire against a live cmux surface |
| `cmux-on-disk-binding.txt` | The binding in the file cmux restores from after a crash |
| `negative-cases.txt` | The four refusals: subagent, kill switch, no multiplexer, no binary |
| `tmux.txt` | tmux exercised against a real server: options written, read back, unset |
| `cmux-host-caveat.md` | Why the cmux host's state limits the cmux claims |

## What is proven

**The publish path works.** `cmux-publish-retire.txt` shows a real session
publishing `source: agent-hook`, `approval_policy: auto`, `auto_resume: true`,
`kind: local-operator`, with `checkpoint_id` equal to the session's actual
directory name under `~/.local-operator/sessions`, and the binding gone again
after retire.

**It survives to disk, for the right surface.** `cmux-on-disk-binding.txt` reads
the binding back out of
`~/Library/Application Support/cmux/session-com.cmuxterm.app.json` — the file a
crash restores from, written on cmux's autosave rather than on clean shutdown —
and shows `autoResume: true`. Identity is verified rather than eyeballed:
`panels[0].id` equals `CMUX_SURFACE_ID` and `workspaces[N].workspaceId` equals
`CMUX_WORKSPACE_ID`, both exact. This is the property that would otherwise
"look fixed and not be": an in-memory binding that never reached disk would be
useless in the only situation it exists for.

**The refusals hold.** `negative-cases.txt` shows all four publishing nothing and
leaving no binding: a subagent session (`origin.json`), the
`LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME` kill switch, an environment with no
multiplexer at all, and a cmux surface whose CLI cannot be resolved.

**tmux is real, not mocked.** `tmux.txt` runs against a private tmux server on
its own socket (so it cannot touch a real session): the backend writes both pane
options, they are read back with `show-options -pv`, and after retire both are
genuinely *unset* — `rc=1`, absent — rather than set to an empty string that a
restore script would misread as a session with a blank id.

## Reproducing

The cmux captures need a live cmux surface, so they are scripts rather than
tests. tmux is reproducible anywhere tmux is installed:

```sh
bash docs/evidence/multiplexer-resume/tmux-evidence.sh
```

The unit suite covers the same contracts without needing either multiplexer:

```sh
.venv/bin/python -m pytest tests/unit/test_multiplexer.py -q
env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest \
  tests/unit/tui/test_multiplexer_broadcast.py -q
```
