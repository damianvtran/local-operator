---
name: org-hub
description: "Get your organization's shared agents, teams and skills from its private Agent Hub, pull the latest, stay subscribed."
requires: radient-team
---

# Your organization's Agent Hub

Your Radient organization shares agents, teams and skills through its private Agent Hub. This machine follows it automatically; use the commands below to look, pull now, or resolve a name clash.

## Look

- `lop hub org` (alias `lop hub org list`): the catalogue per organization, with a `local` column — `linked` (followed), `conflict` (a local item has the same name but no link), `linked-elsewhere` (a local item has the same name but already follows a different hub item; left alone), `absent` (not here yet). It also lists *unlinked local copies* and what the last automatic run adopted. `--org <name, prefix or tenant id>`, `--kind agent|team|skill` (repeatable), `--json`.
- Agent tool, mid-session: `agent(op='org_list', org?, kinds?, query?)` — `kinds` is comma-separated (`agent,skill`), `query` filters names and descriptions.

## Pull

- `lop hub org pull` prints the plan (counts and conflicts) and changes nothing; add `--yes` to run it. `--kind`, `--name <id-or-name>` (repeatable) narrow it. With several organizations, `--org` is required.
- Agent tool: `agent(op='org_pull', org?, kinds?, name?, on_conflict?)` — `name` is ONE hub id, name or unique name prefix (the result names the id it resolved to; an ambiguous prefix is refused with the candidates); `kinds` is comma-separated. Always subscribes; never auto-pushes.
- Order is agents → skills → teams (team rosters name agents). Re-running is safe: linked items are left to the update runner; nothing is duplicated.
- New links are subscribed: updates arrive the same way as for any hub-pulled item (`guide://agents`, "Agent Hub: pull, push, update").

## Same name, no link: `--on-conflict`

- `adopt-link` (default): back up the local copy under `hub/backups/`, then link it (subscribed, no auto-push) without changing a byte of it. Identical content: `identical`. Different: `kept-local-linked` — the hub version is recorded as the link's baseline, so the difference shows as `local-only` in `lop skills sync --check` / `lop agents sync --check` instead of hiding. Never merges, never calls a model.
- `adopt`: hub wins, after the same backup.
- `reconcile`: merge the two copies now (may spend model tokens); a merge that needs a person is reported, not forced. Only on explicit request — CLI `--on-conflict reconcile`, tool call with `confirm_replace=true`.
- `skip`: leave it; the exact command to resolve it is printed.

Pick `adopt-link` unless the user says the org copy should replace theirs (`adopt`). Report the backup paths you were given.

## A skill sync that needs a decision

- `lop skills sync` exits 1 when any skill is `needs-review`, `refused`, `failed` or `unavailable` (not brought up to date); `--check` only reports and exits 0. The message names each conflicted file and its kind (edit/edit, edit/removal, add/add).
- Settle it from the CLI (never by editing `~/.local-operator/skills`): `lop skills sync --name <skill> --prefer local` or `--prefer remote` (remote = hub). Each file the merge could not settle takes that side's whole file; cleanly merged files still merge. Then `lop skills push <skill>` publishes a kept local side. Pick the side the user asked for; if unclear, ask.

## Automatic enrolment

- On sign-in, daemon start and the hub update tick (at most hourly per organization), new organization items are pulled and same-name local copies are adopt-linked (content kept as it is). It never merges, never calls a model, never auto-pushes and never uses hub-wins.
- Turn it off: `lop config edit hub.org.auto_enrol false` (or `/settings` → Organization Agent Hub). Already-linked items keep following `hub.auto_update.*`.
