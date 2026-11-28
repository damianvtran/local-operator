"""Radient Guard, client side (RFC v0.6 slice L1; v0.9 slice L2a).

Guard reports *aggregate* guardrail activity for members of a Team org that
has turned it on, after the member's one-time host-level consent. This package
holds the client half. In slice L1 it is deliberately inert:

- :mod:`local_operator.guard.decisions` defines the typed, enum-only
  :class:`~local_operator.guard.decisions.GuardDecisionRecord` that the
  guardrail checkpoint (``guardrails/check.py``) emits for every decision, and
  a process-wide sink that is a NO-OP by default;
- :mod:`local_operator.guard.outbox` is the durable local queue a later slice
  will drain once the install is admitted.

The L1 modules above open no socket and import no network client. Slice L2a
adds the member's consent (RFC v0.9 §5):

- :mod:`local_operator.guard.consent` -- the durable prompt queue, sticky
  decline/revoke, ``admit_increasing``-only accept, revoke, receipt notices,
  and the admission-point status read, which returns before any call, file or
  credential-store access for an account with no Team membership;
- :mod:`local_operator.guard.consent_op` -- the ``guard_consent`` typed op on
  the desktop approvals plane (accepted only via ``/approve-operation``);
- :mod:`local_operator.guard.install_key` -- the install key, created only
  inside an admitted accept.

Nothing installs the decision sink or drains the outbox yet (L3); a record
still goes nowhere, and a non-Team install builds no Guard file at all.

Keep this ``__init__`` import-free: ``guardrails/check.py`` imports the
decisions module on every tool call's path, and a heavy package import here
would land on that path too.
"""
