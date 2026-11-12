# Team Bridge egress gate (finding S-38, slice 1)

Code comments cite the sections of this page by name, for example
`team-bridge-egress.md: "Grants"`.

## Status

**Closed for ask mode; accepted risk in auto/yolo.**

That sentence is the only correct summary of what this slice does, and it is a
recorded product decision, not an oversight: sessions running
`tool_approval_mode: auto` as the machine default, `--yolo`, or a
host-declared unattended `--tools` inventory that names the tool keep running
third-party-reaching calls unattended after a teammate's agent message
arrives. Do not describe S-38 as closed without the qualifier.

Users meet Team Bridge as **Agent messages** (the Settings title). Product copy
calls the source "a message from a teammate's agent (Agent messages)"; "Team
Bridge" is the internal name used in code and on this page.

## The problem

Team Bridge delivers text written by people outside this machine into a
session: a `team_bridge_message` card, or the result of any `bridge` tool
call. Screening is probabilistic. When an injected instruction got through,
the read-tier web tools (`web_fetch`, `read https://…`, `web_search`,
`web_read`) never reached the approval gate, so in `ask` mode a model could
put a secret into a URL and nobody was asked.

All of the code lives in `local_operator/harness/bridge_egress.py` (one leaf
module); every other change is a call into it.

## Latch

A session latches the first time bridge content reaches it and never
unlatches. Only `/new` or a fresh session starts clean — not `/clear`,
compaction, pruning, an approval or the end of a turn. The state is a
`bridge_taint_latch` bookkeeping row in `transcript.jsonl` (never rendered to
the model), so it survives resume, fork, `/move` and the prune-fold rewrite of
the file; a full scan of the journal is the backstop.

## Propagation between sessions

The latch follows bridge text wherever it is carried.

- **Push into a child.** A `task` child of a latched session is latched for
  life. A live parent and child are linked: whichever latches first latches
  the other, because a parent can write anything into a running child
  (`hub` send/steer are read-tier) and everything a child reports reaches its
  parent. A `hub` message from a latched session is stamped and latches the
  child when it lands, even if the child is still queued.
- **Push to a peer.** `send`, `sessions` spawn/resume and `network` mutations
  are write-tier, so their cards carry the latch and ask in a latched session.
- **Pull.** Any tool result that carries another session's transcript content
  is a source when that session is latched:
  - `sessions` peek, in every mode (tail, head, cursor, query, regex, digest).
    The target's whole journal is judged, not the window, because the window
    can hold a restatement without the source row.
  - A mesh peek, a mesh `send` reply, a `sessions` resume-with-prompt reply on
    a peer, and a `network sessions` send reply. No peer reports its latch
    today, so these count as latched (fail closed). A future peer-reported
    "clean" flag may narrow this only for peers the operator controls (same
    account, or a device admitted as the operator's own); from any other
    member that flag is attacker-chosen.
  - `read` or `grep` of a latched session's content, in every spelling:
    its directory (`<config>/sessions/<id>/`), the decompressed archive copy
    (`<config>/cache/decompressed/<id>/`), and a spill it owns
    (`<config>/spill/<digest>.*` or `spill://<digest>`). Spills and
    decompressed copies fail closed on provenance: a spill with no known
    owner, or whose owner has no session directory or no transcript (never
    existed, or was cleaned up), counts; a decompressed copy whose session is
    gone is judged by scanning the copy itself. A spill the reading session
    wrote itself is not a pull. A search rooted at any of those stores'
    roots, or at the config directory or `<config>/cache`, counts
    unconditionally. A search rooted ABOVE the config directory counts
    unless the walk provably skips it (see the hidden-directory rule below).
    `grep` with no `path` is judged at its working directory. The pulling
    call's own result is stamped too.
  The stamp is `details["bridge_taint"] = True` on the result. It feeds the
  same latch; it is not a second gate.

## Seam

The loop's one tier decision (`harness/loop.py`, `_runner_result`) calls
`escalate_for_bridge_taint`. In a latched session `web_fetch`, `read <url>`,
`web_read`, a `web_search` whose query carries a URL, a host-shaped token or
an IP literal, and the mutating ops of `project`, `library`, `agent`, `tags`
and `skill` move from `read` to `write`, so `ask` mode shows a card. `team`
account writes keep their own, stricter gate. The seam never lowers a tier.

## Search providers

While latched, `web_search`/`web_read` walk only index-only providers (Brave,
SerpAPI, DuckDuckGo, a SearXNG at a loopback or private literal address).
Answer engines are filtered out in every mode, and no approval widens the set;
a forced answer-engine provider or an empty filtered chain is refused with no
card. A host that has no latch object is judged by its live messages, and the
decision is recorded on the tool context so the provider filter agrees.

## Cards

Every card in a latched session offers Allow and Deny only. In the TUI, `A`,
`S` and `G` are inert, and the slot that shows the approvals policy on a clean
card shows the notice and one line on how to get auto back instead. Pressing
`A`, `S` or `G` there answers nothing and briefly highlights that line. The
question row is the same as on a clean card. For `web_fetch`, `read`,
`web_search`, `web_read`, `browser` and `monitor`/`wake`, the card adds the
full request when its ordinary summary is clipped or does not show every
argument verbatim. No mode change answers a parked card, and a viewer's
background approval leaves it parked. Desktop and phone receive the
`bridge_latched`/`taint_raised` flags and the notice as the first line of the
description, whether the runtime or the TUI owns the gate. A card parked
while the session was clean is re-published with those fields, on the same
request id, the moment the session latches, so every viewer redraws it as the
Allow/Deny card. An attached terminal keeps a mirror of the owner's latch,
marked from that flag, which drives the redraw only; the owner still enforces.

The latch records which kind of source set it, and that kind picks only the
wording of the card notice, the paused-auto notice and the refused-auto
receipt. `agent_message` means a Team Bridge card or `bridge` result reached
this conversation. `other_session` means the content came through another
session or device (a peek, a mesh reply, a cross-session read, a spill or
archive copy, or a parent/child link). An Agent message wins when both apply,
and a latch row with no kind reads as `agent_message`. The kind never changes
a tier, a card's choices or a grant (`test_u1_kind_never_changes_the_decision`).
The plain `[y/N]` terminal prompt has no session object, so it keeps the
`agent_message` wording.

## Grants

A session auto granted before the latch no longer covers calls once the
session latches; the call falls through to the card. In the issuing
conversation the TUI card's `G` and the desktop's "All sessions" count as
session grants too. A deliberate `/approvals auto` typed after the latch is
honoured and recorded (`granted_while_latched` in `approvals.json`), so it
survives resume.

## bridge_ack

A routed `approvals auto` or `approvals default auto` for a latched session
needs the `bridge_ack` marker (sent by a desktop build that knows the latch;
the TUI marks typed commands). Without it the command is refused with a
receipt that says to type `/approvals auto` in the terminal, so an older
desktop's "Set auto" escape cannot re-loosen a latched session. The `default`
form is covered because global auto covers a latched session's cards (the
accepted auto/yolo risk), so a stale card's `G` would otherwise loosen the gate
in one press. A `default` form is judged by what it resolves to, not by its
words (`approvals_default_may_loosen` in `harness/approval.py`): a bare
`approvals default` keeps a TUI owner's current mode, and a suspended pre-latch
session auto keeps that mode auto, so the bare form is refused while latched on
both owner hosts. Name `default ask` to tighten. Today that includes the phone composer's typed `/approvals
auto`: the phone has no `bridge_ack` path yet, so a latched session can be
loosened only from the terminal (fail closed).

## Fail-closed

An error deriving the latch means latched; an error classifying a call means
`write`; an unparseable target means egress; a target session whose state
cannot be read means latched; a gate that cannot receive the flags refuses the
call.

## Acceptance rows

The T-rows are pinned in `tests/unit/harness/test_bridge_egress.py`,
`tests/unit/harness/test_bridge_taint_propagation.py`,
`tests/unit/session/test_bridge_taint_latch.py`,
`tests/unit/session/runtime/test_serving_bridge_taint.py` and
`tests/unit/tui/test_approval_bridge_taint.py`.

## Not covered here

- Auto, yolo and declared unattended inventories (accepted risk, above).
- The desktop presentation (not mounting `ApprovalEscape`, the picker sending
  `bridge_ack`) is a local-operator-ui follow-up under the UI freeze. Core
  enforces regardless: an older desktop's escape can neither answer a latched
  card nor make a post-latch grant.
- Read-tier tools are escalated by name. `code_requests show`,
  `artifact_data get/list`, `agent search/org_list` and `network` reads stay
  `read`; none reaches an attacker-chosen host today. A guard test fails when
  a new static read-tier tool appears so it is classified deliberately.
- Pull paths not closed: a confined `bash` command reading another session's
  files, and session titles in `sessions list`/`info`.
- The hidden-directory rule: a `read` or `grep` rooted above a HIDDEN config
  directory (the default `~/.local-operator`) is not counted only when ALL
  of these hold, because then neither grep engine walks into it: the call
  sets no `include`; no `.ignore`, `.rgignore` or `.gitignore` with a `!`
  line sits in the search root, any parent of it, or a non-hidden directory
  on the way down to the config directory; and none of those directories is
  a git work tree (`.git`, whose `info/exclude` and the global excludes file
  can whitelist too). Any `include` counts because ripgrep treats a glob
  matching the folder's own name (`*`, `*.*`, `**`, `**/*`) as a whitelist.
  The grep tool runs ripgrep with `--no-config`, so `RIPGREP_CONFIG_PATH`
  cannot add `--hidden`; the Python fallback prunes every hidden directory
  regardless of includes and ignore files. A non-hidden config directory
  gets no exemption.
- Stored injection through files and memory, a quarantined reader, and the
  preview CSP.
