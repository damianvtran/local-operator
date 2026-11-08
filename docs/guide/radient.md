<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Radient: automatic model selection and agent sharing

[Radient](https://console.radienthq.com) adds two optional capabilities:

- **Automatic model selection**: `lop --hosting radient` picks
  the best model per step to balance quality and cost, no `--model` needed.
- **Agent sharing**: push your agents to Radient's public registry, pull
  agents others published:

```bash
lop credential update RADIENT_API_KEY
lop agents push --name "My Agent"
lop agents pull --id "<agent_id>"     # no key needed to pull
```

- **Organization sharing**: share agents and teams privately inside a Radient
  organization. These commands act as the signed-in person, so sign in first;
  `--org` names the organization (pass the `tenant_id` the picker prints):

```bash
lop login radient
lop teams push --org <tenant_id> <team>
lop teams pull --org <tenant_id> <team_id>
lop agents push --name "My Agent" --org <tenant_id>
lop agents pull --id "<agent_id>" --org <tenant_id>
```

Without `--org` every command behaves exactly as before (the public registry
path), and `lop teams push` without it lists your organizations and asks for
the flag rather than guessing one. Organization calls carry your signed-in
account's token, so they address the Radient cloud API only -- to point them
at a local or QA hub deliberately, opt in per shell with
`RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1`.
