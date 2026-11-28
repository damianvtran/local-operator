"""Alert responders: the Local Operator half of Radient Team Integrations, Phase 3.

RFC "Radient Team Integrations" (C3, approved 2026-11-17 by Damian Tran), §7.1-§7.3.
An app posts an alert card into a Team Bridge channel (``origin: "integration"``);
the hub wakes ONLY the agent endpoints whose owners registered them as responders
in that channel (agent-server #451). This package decides, on the owner's device,
whether such a wake may turn into work, and fences that work with the hub's claim
lease.

THE AUTHORITY RULE. Inbound alert text is untrusted app data: it never grants
authority and never starts work by itself. The only thing that authorizes alert
work is the owner's own, human-authored ``responders.yaml`` (:mod:`.policy`),
which agent tools cannot write (the guardrail evidence class names it), checked
together with the Team Bridge authority fence (``team_bridge.authority``) and the
hub's uncached kill switches on every claim and heartbeat (:mod:`.lease`).

Modules:

- :mod:`.policy` — the local policy file: schema, strict parse, default OFF,
  ``policy_sha256``, and the per-alert admission decision.
- :mod:`.lease` — the ``lease_epoch`` claim client: claim, heartbeat, release,
  and the closed outcome vocabulary for the hub's coded refusals.
- :mod:`.runner` — the per-alert orchestration in its safety order: authority ->
  policy -> screen -> claim -> run (heartbeating) -> release.
"""

from __future__ import annotations
