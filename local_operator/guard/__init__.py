"""Radient Guard, client side (RFC v0.6, slice L1).

Guard reports *aggregate* guardrail activity for members of a Team org that
has turned it on, after the member's one-time host-level consent. This package
holds the client half. In slice L1 it is deliberately inert:

- :mod:`local_operator.guard.decisions` defines the typed, enum-only
  :class:`~local_operator.guard.decisions.GuardDecisionRecord` that the
  guardrail checkpoint (``guardrails/check.py``) emits for every decision, and
  a process-wide sink that is a NO-OP by default;
- :mod:`local_operator.guard.outbox` is the durable local queue a later slice
  will drain once the install is admitted.

Nothing in this package opens a socket or imports a network client. Admission,
enrolment and transport land in L2a/L2b behind the consent gate; until then a
record goes nowhere, and a non-Team install never even builds the outbox file.

Keep this ``__init__`` import-free: ``guardrails/check.py`` imports the
decisions module on every tool call's path, and a heavy package import here
would land on that path too.
"""
