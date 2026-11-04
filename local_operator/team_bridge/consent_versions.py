"""Team Bridge consent version identifiers and the ONE default-on/engaging set.

WHY A MODULE OF ITS OWN. Every site that decides whether a served or accepted
consent allows an automatic action — ``client`` (default-on, the enrolment echo,
consent-on-record), ``engagement`` (the F8 never-used chief-of-staff exception)
and ``reconcile_plan`` (a pure planner whose import allowlist forbids config
access) — must read the SAME set, so it lives in an import-free leaf any of them
can import without dragging config or client code along.
"""

from __future__ import annotations

#: ``bridge_consent_v1`` (opt-in, no engagement disclosure) stays in ``client``
#: and is deliberately in no set here.

#: The first opt-out disclosure ("Team Bridge is on by default ...").
CONSENT_VERSION_V2 = "bridge_consent_v2"
#: agent-server #258: v2's disclosure (tightened to fit the 2 KiB cap, no fact
#: removed) PLUS the built-in ``#general`` channel — enabling seats the
#: chief-of-staff agent in a room every person and agent in the organization
#: reads, where it wakes only on an @-mention or a followed thread reply.
CONSENT_VERSION_V3 = "bridge_consent_v3"

#: The consent versions whose copy discloses the opt-out default, and so the
#: ONE set every default-on and engagement site reads (``default_on_in_force``,
#: ``enroll_default_on`` and its commit-only recovery, ``_consent_on_record``,
#: ``reconcile_plan``, ``engagement.ENGAGING_CONSENT_VERSIONS``). Only a hub
#: that SERVES one of these (and lists it as accepted) can turn an absent
#: ``team_bridge.enabled`` into "eligible to be enrolled". Mirrors the hub's
#: ``defaultOnConsentVersions`` (agent-server ``internal/bridge/bridge.go``).
#:
#: WHY v3 IS HERE. Both v2 and v3 say "Team Bridge is on by default for members
#: of an active Team plan" and both disclose that a message may reach the agent
#: before the member has used it. v3 adds the ``#general`` carve-out, which v2
#: misdescribes ("agents join shared channels once owners accept"); the hub
#: seats agents in ``#general`` only while serving v3, and a hub serving v3 to a
#: client that knew only v2 would silently stop default-on and engagement for
#: every member. Knowing v3 here is inert until the hub serves it, and SERVING
#: it is the hub's deploy-time decision (``BRIDGE_CONSENT_VERSION``) gated on
#: product/legal sign-off of the exact bytes — never this client's call.
#:
#: CONSTRAINT: exact string membership, never ordering. ``bridge_consent_v4`` or
#: ``v10`` is a new disclosure this build has not read and inherits nothing;
#: adding one is a deliberate edit here after reading its text. ``v1`` (opt-in,
#: no engagement disclosure) is deliberately absent.
DEFAULT_ON_CONSENT_VERSIONS: frozenset[str] = frozenset({CONSENT_VERSION_V2, CONSENT_VERSION_V3})
