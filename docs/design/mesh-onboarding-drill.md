# Drill runbook v1 — remote onboarding join (workstream E2E)

Owner: the workstream E2E (drill commander = the manager; runner built by coder, slice (b)).
This file is the OPERATIONAL artifact: commands, expected receipts, and what to keep.
Source of truth for intent: `mesh-remote-onboarding.md` §7 (Steps 0–4). Do not re-derive
from memory; the note and the PRs are the references.

## 0. Preconditions

- A release whose tag contains slices (a) + (b) is PUBLISHED (the node is updated to it).
  Measure the window from the repo at drill time — never from a session snapshot:
  `git -C ~/local-operator fetch --tags origin main && git tag --sort=-creatordate | head`
- The operator is available for gestures: Step 0's admin prompt, Step 1's Touch ID.
- The node: `ec2-user@99.79.190.164` (`cloud-node-1`), Amazon Linux, passwordless sudo
  **as a DRILL ASSUMPTION** (OQ16), key `~/.ssh/lop-mesh-nprod.pem` works from this Mac.
- This Mac is measured no-anchor (`spawn-capability-only`, §1.1) — Step 0 exercises that.
- Hygiene: synthetic sessions only; never touch the operator's live/probe sessions (the
  drill commander holds their ids for the drill window — keep them out of this doc and the
  evidence); isolated config roots for every local harness; TUI boots must unset every
  inherited `CMUX_*`; kills scoped to own pids; evidence to a scratchpad matrix file.

## Step 0 — local bootstrap on THIS Mac (§7 Step 0; slice (b)'s `lop operator setup`)

1. Baseline (read-only): `lop operator status` → expect `spawn-capability-only`, no anchor.
2. Agent files the ask / runs the verb (agent-interim path):
   - with the credential prompt: the operator supplies the admin password ONCE via
     `/credential` (a masked capture which reports the generated `LOP_SECRET_…`
     name it is stored under) and PROMOTES that name with `/credential --persist
     LOP_SECRET_…` — the capture writes session memory only, and `--sudo-secret`
     resolves through `lop secret get` — then `lop operator setup --json
     --sudo-secret LOP_SECRET_…`;
   - or a terminal runs `lop operator setup --json` and answers the sudo PROMPT.
3. Operator raises the one admin gesture (sudo prompt here; the native sheet is slice (c)).
4. Assert the receipts: `proposed → consent → generated → installed → verified`, all ok,
   `state: installed`, level ASSERTED afterwards: `lop operator status` shows the anchor
   installed + trusted (`operator-presence` if the keyagent path works here, else
   `operator-file-only` — record which, honestly).
5. Artifacts: `setup --json` output; `lop operator status` before/after; the key's
   `key_id` + `spki_fp`.
6. **Gate: Step 1's Touch ID cells cannot proceed until the level flips.**

## Step 1 — onboard cloud-node-1 (§7 Step 1; slice (b) execution)

1. Credential-free handshake (no secret may be touched): `lop onboard probe`-shaped work
   runs through `network.onboard.probe` — record host, banner, `SHA256:` host-key fp.
   (The request verb + card are slice (a)'s; this runbook drives whatever it calls.)
2. File the request (slice (a)): `lop network approvals request --host 99.79.190.164 --json`
   → card appears; operator approves (Touch ID).
3. Execute: `lop network approvals run <approval_id> --json` (or the desktop's Run).
   Expected receipts, in order: `invite, pre_read, install, join, anchor, grants, relay, verify`.
   - `pre_read` = step zero: OS/arch, lop version, uv, systemctl, linger, sudo, anchor path.
     A contradiction HALTS: record `failed`, fresh card — do not proceed on the wrong facts.
   - `install`: `lop-update <tag>` (build present) or `uv tool install local-operator==<tag>`.
   - `join`: `join @<token> --automated`; the operator's approve is the admit (pre-answered
     confirm); a mismatch refuses and spends the attempt (audit `sas_mismatch`).
     Satisfied when already active: a node already an active member passes this step with
     no re-join and no dial; the step's detail reads "<node> is already an active member
     of <network> (epoch N); admission is satisfied and no re-join was attempted — the
     invite goes unused and expires." Read it in `lop network approvals run <id> --json`:
     it is the join receipt's `detail` (`steps[] | select(.step == "join") | .detail`).
     The human run block shows only the card's state. Test the join mechanism against a
     genuinely non-member target.
   - `anchor`: F4b trio re-derived locally; `install --from` lands the EXACT approved bytes.
   - `grants`: node-side `member grant <net> <mac-id> approve unattended` per the card's ticks.
   - `relay`: install/start the systemd `--user` unit + linger; linger missing = caveat,
     not a failure (OQ11).
   - `verify`: Mac-side `lop network ready --peer cloud-node-1 --json` is the acceptance
     surface; `doctor`/`peers` ride the same receipt. Only all-green folds to `connected`.
4. Node-side spot checks (commands + outputs into the matrix):
   - `lop --version` == the pinned tag; `ls -l /etc/local-operator/operators/`;
   - `systemctl --user status local-operator-network.service`;
   - `loginctl show-user ec2-user -p Linger --value`.
5. Row check is NODE-side (rule 2 keeps the Mac's own copy unchanged): on the NODE,
   `lop network show` carries `approve`/`unattended` on the Mac's member row — or read
   the grants step's `data` from `lop network approvals run <id> --json` (the same row,
   written by the same command). Mac-side: the Mesh tab shows the device.
6. Negative cells (M3): expiry mid-run refuses the next step (`expired` + receipts);
   a forged allow over the wire is refused at the node; replay of a consumed challenge
   is refused end-to-end. Record each refusal's audit line beside the positive cells.

## Steps 2–4 (other slices; referenced so the matrix is one file)

- Step 2: remote create + probe (no dead-end), full-auto variant (unattended grant).
- Step 3: park answered remotely (the ALLOW path; slice (a)'s wire work).
- Step 4: carry-over + queued move (slice (d)).

## Fallback topology (if the real node leg cannot run)

Two local config roots + a second relay on loopback (the mesh suite's own topology):
`tests/unit/network/test_onboard.py` (fake transport) + `test_join_automated.py` (real
ceremony, no SSH). State in the QA report that SSH-transport coverage then comes from
unit cells + the manual run only — never imply coverage you do not have.

## Teardown

- Stop every relay/harness you started, by exact pid; no orphans (rig hygiene).
- Remove scratch roots (`env -i HOME=…` isolates); leave the operator's stores untouched.
- Kill nothing by bare program name.
