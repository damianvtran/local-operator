# Designer round 3 — the AUDIT-phase pair, and what a pasted notice gets

Two things the published evidence did not settle, so I captured them myself.

Everything here runs against a **byte-identical `cp -c` clone** of the operator's
session `835fbcafdc27` (`md5 dfd5c14903aface4d715f9ffdce36afe`, 15,926 journal
lines — a later snapshot than the coder's, so the row counts in my census are
higher by the rows appended since), in an isolated `HOME` +
`LOCAL_OPERATOR_CONFIG_DIR`, with every `CMUX_*` variable unset, and the live
store never written.

## 1. `audit-after-mine.{svg,png}` — an independent reproduction of the after member

Same command as the PR's, same head (`87de885f7`), same script
(`scripts/audit_notice_census.py <session-dir> <out.svg>`):

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/audit_notice_census.py <copy-of-835fbcafdc27> /tmp/audit-after-mine.svg
rsvg-convert /tmp/audit-after-mine.svg -o audit-after-mine.png
```

Census on my copy: `context rows=414 user=65 notice_rows=5 painted=0`,
`audit rows=15572 user=68 notice_rows=8 painted=0`, and through the assembled app
`on screen (audit): user rows that are notices = 0`. The coder's copy reports
`343 / 15501 / 68`; identical on every figure that carries the claim.

**What it differs from the published after member by — and why that matters.**
The published after frame's first visible row is a red `x session failed to start:
object Session can't be used in 'await' expression` notice, with the band reading
`session error`. That row is not in the replayed history: neither string occurs
anywhere in the 15,926-line journal (`grep -c "session failed to start"` → 0), and
my run mounts the compaction marker first, at `y=2`, with the band reading
`test/e2e-model`. It is capture-session state, and the block geometry confirms it
is the *only* difference — every block after it matches the published frame's row
set, heights and 1-line rhythm, offset by exactly the two lines that row occupies.

## 2. `paste-*.png` — the stated cost, as a person meets it

`paste_shot.py` drives the real `OperatorApp` (so `local_operator.tcss` applies)
over one short conversation in three cases:

| frame | case | painted transcript |
|---|---|---|
| `paste-no_paste.png` | control: the notice never pasted | prompt, answer, follow-up |
| `paste-pasted.png` | the notice pasted as the whole message | **byte-identical SVG to the control** (`md5 da31488718d078be9fe06c317b7bdb6c`) |
| `paste-quoted.png` | the notice quoted inside a sentence | the row survives, gutter and all |

So the pasted message leaves **no trace at all** on the rendered frame — not a
gap, not a placeholder — and the person's next message sits directly under the
answer, reading as if it answered nothing. The quoted form is unaffected, which
locates the limit precisely: a message is dropped when it *begins* with a notice
head and has nothing before it.
