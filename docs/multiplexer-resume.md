# Multiplexer session resume

When a terminal multiplexer dies and comes back, its panes come back too — but
each one opens a fresh shell and a fresh `lop`. Every conversation is still on
disk under `~/.local-operator/sessions`; none of them is reachable except by
remembering which pane held which session and retyping `lop --resume <id>`.

`lop` now publishes, per pane, the one fact the multiplexer cannot derive: *this
pane is holding session `<id>`, and here is the argv that reopens it*.

Nothing here is required for `lop` to run. Publication is best-effort by
construction: a missing binary, a socket error, a multiplexer that is not
running, or a timeout is logged at debug and otherwise ignored.

## What gets published

**Always a restore-and-idle command** — `lop --resume <id>`, which replays the
transcript and waits for you. It never continues an interrupted turn. This is a
safety property, not a default: a restore happens to every pane at once,
unattended, usually after a crash you did not choose. The worst case of a
spurious restore has to be an idle session rather than a dozen agents resuming
tool execution with nobody watching.

**Never a subagent's session.** A delegated review, design or scout run creates
a session directory shaped exactly like a real conversation, and it runs inside
its parent's pane. Publishing one would overwrite the pane's binding, so a crash
restore would reopen the delegated run instead of your work. Sessions marked
with `origin.json` are skipped.

**Never before the session can actually be resumed.** `--resume` refuses an id
whose `transcript.jsonl` does not exist yet, so a session publishes nothing
until its first turn has persisted, then publishes within a few seconds.

**Nothing sensitive.** The session id and the working directory, and nothing
else. No environment, no credentials, no prompt text.

## Per multiplexer

| Multiplexer | Mechanism | Auto-restores after a crash |
|---|---|---|
| cmux | resume binding via the control socket | yes, with the registration below |
| tmux | pane options `@lop_session`, `@lop_resume_command` | no — needs a restore script |
| wezterm | per-pane user vars, same two names | no — needs a restore script |
| zellij | state file under `~/.local-operator/multiplexer/` | no — needs a restore script |
| screen | state file under `~/.local-operator/multiplexer/` | no — needs a restore script |

Only cmux has an API for "run this command in this pane on restore". The others
publish a discoverable marker instead, which a restore script consumes.

### Reading the markers

tmux and wezterm store native pane options:

```sh
tmux show-options -pv @lop_session
tmux show-options -pv @lop_resume_command
```

zellij and screen have no per-pane key/value store, so the same two facts go in
`~/.local-operator/multiplexer/<backend>-<pane-id>.json`:

`<pane-id>` is the multiplexer's own pane identity, which may be more than one
value. zellij uses `<session-name>-<pane-id>` (`ZELLIJ_SESSION_NAME` and
`ZELLIJ_PANE_ID`), because a zellij pane id is only unique within its session.
screen uses `<sty>-<window>` (`STY` and `WINDOW`), falling back to `<sty>` alone
where `WINDOW` is not exported — in that fallback the marker is per session
rather than per window, so a second `lop` in the same screen session overwrites
it.

```json
{"session_id": "…", "command": ["…/lop", "--resume", "…"], "cwd": "…"}
```

A restore script's job is the same either way: for each pane, read the session
id, and if it still names a directory under `~/.local-operator/sessions`, run the
resume command in that pane. A clean exit removes the marker; a crash leaves it,
which is the point. Stale markers are harmless — a pane id is reused, so the
next session in that pane overwrites it.

## cmux: the vault registration

cmux only auto-resumes an agent it knows about. `local-operator` is not in its
built-in catalog, so you register it once in `~/.config/cmux/cmux.json`. Per
cmux's `docs/vault.md`, this is what teaches its process scanner to recognise a
running `lop` and how to resume one; without it, cmux may retire the binding as
stale (`Workspace.isStaleAgentHookBinding`).

`lop` publishes its binding whether or not you install this. The registration is
what makes cmux keep it.

```jsonc
{
  "vault": {
    "agents": [
      {
        "id": "local-operator",
        "name": "Local Operator",
        "detect": { "alternateArgvBasenamesAny": ["lop"] },
        "sessionIdSource": { "type": "argvOption", "argvOption": "--resume" },
        "resumeCommand": "lop --resume {{sessionId}}",
        "cwd": "preserve",
        "sessionDirectory": "~/.local-operator/sessions"
      }
    ]
  }
}
```

**`detect` must not be `processName: "lop"`.** A running `lop` is really
`…/uv/tools/local-operator/bin/python3 …/.local/bin/lop`, so `argv[0]` is
`python3` and a process-name rule never matches. `alternateArgvBasenamesAny`
exists for exactly this: cmux detects that `argv[0]` is a Python runtime, walks
argv to find the real script argument, and matches its basename.

On an older cmux that does not support `alternateArgvBasenamesAny`, use the
compatibility form instead:

```jsonc
"detect": { "processName": "python3", "argvContains": "lop" }
```

Verified on cmux 0.64.22 by observation — publishing a binding and reading it
back — rather than by inspecting the build, so treat the version boundary
between the two forms as approximate and confirm on your own install with the
trial below.

### Trial it without editing your real config

cmux merges a **project-local** `.cmux/cmux.json`, walking up from a process's
working directory. So you can test the registration against a scratch directory
and leave `~/.config/cmux/cmux.json` untouched:

```sh
mkdir -p /tmp/lop-vault-trial/.cmux
# put the registration above in /tmp/lop-vault-trial/.cmux/cmux.json
cd /tmp/lop-vault-trial && lop
```

Then, from that session, confirm what cmux stored:

```sh
cmux --json surface resume show
```

You want `source: "agent-hook"`, `approval_policy: "auto"`,
`auto_resume: true`, and a `checkpoint_id` equal to the session's directory
name under `~/.local-operator/sessions`. Once it looks right there, copy the
`vault` block into `~/.config/cmux/cmux.json`.

To confirm it survives a crash rather than only a clean shutdown, read it back
from the file cmux restores from (it is written on a short autosave delay):

```sh
python3 - <<'EOF'
import json, os
p = os.path.expanduser("~/Library/Application Support/cmux/session-com.cmuxterm.app.json")
for w in json.load(open(p))["windows"]:
    for ws in w["tabManager"]["workspaces"]:
        for panel in ws.get("panels", []):
            rb = (panel.get("terminal") or {}).get("resumeBinding")
            if rb and rb.get("kind") == "local-operator":
                print(ws["workspaceId"], panel["id"], rb["checkpointId"], rb["autoResume"])
EOF
```

`autoResume: true` there is what a crash restores from.

You can also exercise the restore without crashing anything:

```sh
cmux restore --surface <surface-ref>
```

## Turning it off

```sh
export LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME=1
```

Mirrors `LOCAL_OPERATOR_NO_TERMINAL_TITLE`. Useful for a recording, a CI job, or
a session opened to read someone else's transcript in a pane that already holds
a real one.

## Troubleshooting

**`cmux --json surface resume show` returns `resume_binding: null`.** The
session has not persisted a turn yet (nothing is published until it can actually
be resumed), or this is a subagent's session, or the kill switch is set.

**`approval_policy` is `manual` and `auto_resume` is `false`.** Something
published a `cli`-sourced binding, or cmux retired ours as stale — install the
vault registration above. Note that `cmux surface resume set` on the command
line can only ever produce a manual binding; `lop` deliberately does not use it.

**The binding disappears when you quit.** That is correct. A clean exit
withdraws it so your next shell in that pane does not replay a closed session.
A crash leaves it, which is what the restore uses.
