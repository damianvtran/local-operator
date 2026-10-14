"""Team Bridge — the harness half of Radient agent messages.

Internal module id ``team_bridge``; the user-visible feature noun is "agent
messages" and nothing local is named ``lop team*`` (RFC §2, §13.11: the word
"team" already belongs to agent teams here, and the human group is "your
organization").

The package is split so each module stays importable where it is read:

- :mod:`local_operator.team_bridge.settings` — the six ``team_bridge.*``
  ``config.yml`` keys, their defaults and fail-closed readers. Stdlib-only, so
  the settings registry's consumer-defaults test can import it cheaply.
- :mod:`local_operator.team_bridge.client` — the authenticated hub transport
  for ``/v1/messaging`` (central Radient OAuth, never the model-routing key
  cascade), the §6.5 refusal vocabulary, the served-text accession rules, and
  the §5.1 status-read discipline (boundary-only, coalesced, cached ≥5 min,
  never a poll loop).
- ``server/routes/desktop_team_bridge.py`` — the desktop route lane's four
  routes over the same client.

The wire contract is ``agent-server/docs/TEAM-BRIDGE.md``; section references
in these modules point at it. The server side may still be dark
(``BRIDGE_ENABLED`` off, §5.5): nothing here flips a flag or assumes launch.
"""
