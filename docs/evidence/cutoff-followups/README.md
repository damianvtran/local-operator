# Follow-up to #970: U7 give-up, Q3 guard, D7/D8 copy, NIT-1 memo, U6 sidebar

The four round-2 review streams on #970 deliberately did not carry these; each
item below names the finding it closes and the command that produced its
evidence. Everything ran in a `HOME`/config-isolated environment with every
`CMUX_*`, `LOP_MOBILE_CHILD_*` and `LOP_RUNTIME_*` variable stripped.

## U7 (UX round 2, MAJOR, pre-existing) — the message after a watched cut-off

`u7_repro.py` drives the real path: a real runtime subprocess parked in the real
`bash` tool, SIGKILLed while a real `AttachSession.connect` viewer watches, then
a second message typed into the recovering session.

```sh
# BEFORE: a venv installed editable from origin/main 95fccacda
/tmp/u7-main/.venv/bin/python docs/evidence/cutoff-followups/u7_repro.py
# AFTER: the branch's own venv
.venv/bin/python docs/evidence/cutoff-followups/u7_repro.py
```

```
BEFORE  [verdict] t+8.1s  aborted=True cut_off_cause='<absent>'   error=None
        [type] message accepted at t+8.1s
        t+  9.1s  streaming=False recovering=True can_go_cold=False records=1
        ...
        t+ 68.2s  streaming=False recovering=True can_go_cold=False records=0
                  transcript_rows=9
        [result] WEDGED: accepted and never served after 60 s
        [verdict-line] WEDGED                                     exit=1

AFTER   [verdict] t+8.0s  aborted=False cut_off_cause='owner-lost'
                  turn cut off — the session's runtime stopped answering while
                  this turn was running. The transcript holds what it wrote
                  before that and nothing after.
        [type] message accepted at t+8.0s
        t+  9.0s  streaming=False recovering=False can_go_cold=True records=0
        t+ 10.0s  streaming=True  recovering=False can_go_cold=True records=1
                  transcript_rows=12
        [result] the next message was SERVED after 2.0 s (rows 9 -> 12)
        [verdict-line] SERVED                                     exit=0
```

## D8 (design round 2) — the restored dock row's word

`docs/evidence/cutoff-followups/d8-dock-*.{svg,png}` are stills of the real
`OperatorApp` at 100x30 with three restored rows, taken with
`scripts.visual_capture.save_capture` (the faithful capture helper). The rows
printed beside each frame are what `SubagentRow.paint` renders:

```
before                        after
• draft the memo   ↺  1m  interrupted      • draft the memo   ↺  1m  cut off
• rank the leads   ↺  1m  interrupted      • rank the leads   ↺  1m  interrupted
• audit merged MRs ✓  1m  18 findings      • audit merged MRs ✓  1m  18 findings
```

## D7 (design round 2) — the phone's resume button

`d7-phone-{before,after,control-deliberate-stop}.png` render the real
`Transcript` + `Composer` in the column `session-view.tsx` builds, in the
production Tailwind theme, at 390 and 320 CSS px, fed the payload the daemon
serializes (`stop_reason: "aborted"`, plus the new `cut_off` flag, and the
`severity: "error"` notice row from #970).

```
before   ✗ Stopped with an error — the session's runtime stopped answering … /
         interrupted — tap to resume
after    ✗ Stopped with an error — the session's runtime stopped answering … /
         turn cut off — tap to resume
control  · Interrupted / interrupted — tap to resume        (unchanged)
```
