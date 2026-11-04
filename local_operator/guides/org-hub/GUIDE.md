---
name: org-hub
description: "Get your organization's shared agents, teams and skills from its private Agent Hub, pull the latest, stay subscribed."
requires: radient-team
---

# Your organization's Agent Hub

Your Radient organization shares agents, teams and skills through its private Agent Hub. This machine follows it automatically; use the commands below to look, pull now, or resolve a name clash.

## Look

- `lop hub org` (alias `lop hub org list`): the catalogue per organization, with a `local` column — `linked` (followed), `conflict` (a local item has the same name but no link), `absent` (not here yet). It also lists *unlinked local copies* and what the last automatic run adopted. `--org <tenant>`, `--kind agent|team|skill` (repeatable), `--json`.
- Agent tool, mid-session: `agent(op='org_list', org?, kinds?, query?)`.

## Pull

- `lop hub org pull` prints the plan (counts and conflicts) and changes nothing; add `--yes` to run it. `--kind`, `--name <id-or-name>` (repeatable) narrow it. With several organizations, `--org` is required.
- Agent tool: `agent(op='org_pull', org?, kinds?, names?, on_conflict?)` — always subscribes; never auto-pushes.
- Order is agents → skills → teams (team rosters name agents). Re-running is safe: linked items are left to the update runner; nothing is duplicated.
- New links are subscribed: updates arrive the same way as for any hub-pulled item (`guide://agents`, "Agent Hub: pull, push, update").

## Same name, no link: `--on-conflict`

- `adopt-link` (default): back up the local copy under `hub/backups/`, link it (subscribed, no auto-push). Identical content: done. Different: reconciled through the normal hub merge; if no merge is possible the local content stays linked, so the drift shows in `lop skills sync --check` / `lop hub status` instead of hiding. Never overwrites silently.
- `adopt`: hub wins, after the same backup.
- `reconcile`: merge now (spends model tokens); a merge that needs a person is reported, not forced.
- `skip`: leave it; the exact command to resolve it is printed.

Pick `adopt-link` unless the user says the org copy should replace theirs (`adopt`). Report the backup paths you were given.

## Automatic enrolment

- On sign-in, daemon start and the hub update tick (at most hourly per organization), new organization items are pulled and same-name local copies are adopt-linked. It never auto-pushes and never uses hub-wins.
- Turn it off: `lop config edit hub.org.auto_enrol false` (or `/settings` → Organization Agent Hub). Already-linked items keep following `hub.auto_update.*`.
