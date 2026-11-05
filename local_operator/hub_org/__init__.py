"""The organization (private) Agent Hub: eligibility gate, catalogue, bulk pull, auto-enrol.

Every surface in this package is a Radient Team feature: it exists for an
account signed in to Radient that belongs to at least one organization on an
active team plan, and is invisible to everyone else (no guide, no
recommendation, no tip, no up-sell). :mod:`local_operator.hub_org.eligibility`
is the one, network-free answer to "is the gate open"; every other module asks
it rather than re-deriving the rule.
"""
