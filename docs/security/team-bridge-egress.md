# Team Bridge egress gate (finding S-38, slice 1)

**Status: closed for ask mode; accepted risk in auto/yolo.**

That sentence is the only correct summary of what this slice does. It is the
operator's recorded decision (Option C, ask a-432c, 2026-11-11): sessions
running `tool_approval_mode: auto` as the machine default, `--yolo`, or a
host-declared unattended `--tools` inventory that names the tool keep running
third-party-reaching calls unattended after Team Bridge content arrives. Do
not describe S-38 as closed without the qualifier.

## The problem

Team Bridge delivers text written by people outside this machine into a
session: a `team_bridge_message` card, or the result of any `bridge` tool
call. Screening is probabilistic. When an injected instruction got through,
the read-tier web tools (`web_fetch`, `read https://…`, `web_search`,
`web_read`) never reached the approval gate, so in `ask` mode a model could
put a secret into a URL and nobody was asked.

## What the code does

All of it lives in `local_operator/harness/bridge_egress.py` (one leaf
module); every other change is a call into it.

- **The latch.** A session latches the first time bridge content reaches it
  and never unlatches. Only `/new` or a fresh session starts clean — not
  `/clear`, compaction, pruning, an approval or the end of a turn. The state is
  a `bridge_taint_latch` bookkeeping row in `transcript.jsonl` (never rendered
  to the model), so it survives resume, fork and `/move`; a full scan of the
  journal is the backstop. A `task` child of a latched session is latched for
  life.
- **The seam.** The loop's one tier decision (`harness/loop.py`,
  `_runner_result`) calls `escalate_for_bridge_taint`. In a latched session
  `web_fetch`, `read <url>`, `web_read`, a `web_search` whose query carries a
  URL or host-shaped token, and the mutating ops of `project`, `library`,
  `agent`, `tags` and `skill` move from `read` to `write`, so `ask` mode shows
  a card. It never lowers a tier.
- **Search providers.** While latched, `web_search`/`web_read` walk only
  index-only providers (Brave, SerpAPI, DuckDuckGo, a SearXNG at a loopback or
  private literal address). Answer engines are filtered out in every mode, and
  no approval widens the set; a forced answer-engine provider or an empty
  filtered chain is refused with no card.
- **Every card in a latched session** offers Allow and Deny only: `A`, `S` and
  `G` are inert in the TUI, the card carries a notice line and, for the web
  tools and `monitor`/`wake`, the full target. No mode change answers a parked
  card, and a viewer's background approval leaves it parked.
- **A session auto granted before the latch** no longer covers calls once the
  session latches; the call falls through to the card. A deliberate
  `/approvals auto` typed after the latch is honoured and recorded
  (`granted_while_latched` in `approvals.json`), so it survives resume.
- **Desktop version skew.** A routed `approvals auto` for a latched session
  needs the `bridge_ack` marker (sent by the `/approvals` picker of a desktop
  build that knows the latch; the TUI marks typed commands). Without it the
  command is refused with an "Approvals unchanged" receipt, so an older
  desktop's "Set auto" escape cannot re-loosen a latched session.
- **Fail-closed.** An error deriving the latch means latched; an error
  classifying a call means `write`; an unparseable target means egress; a gate
  that cannot receive the flags refuses the call.

## Not covered here

- Auto, yolo and declared unattended inventories (accepted risk, above).
- The desktop presentation (not mounting `ApprovalEscape`, the picker sending
  `bridge_ack`) is a local-operator-ui follow-up under the UI freeze. Core
  enforces regardless: an older desktop's escape can neither answer a latched
  card nor make a post-latch grant.
- Stored injection through files and memory, a quarantined reader, and the
  preview CSP (memo §10, R4, R5).
