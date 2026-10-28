"""Radient Saver: the client half of compress-only context reduction (beta).

WHY THIS PACKAGE EXISTS. A long agent session re-sends its whole history on
every turn. Radient Saver uploads the part of that history that is ready to be
frozen to Radient's stateless ``POST /v1/saver/prepare``, gets compact "epochs"
back, and splices them in place of the original rows before the provider call.
The provider is still called by THIS process with ITS OWN credentials, so
failover, rotation, API keys and subscription OAuth all keep working, and
Radient never receives a provider credential (contract §2, S-32).

THE CONTRACT IS EXTERNAL. Every rule here is pinned by the Radient Saver
production contract (``CONTRACT.md`` rev4 in the radient-context-reduction
repository; the agent-server ``internal/saver`` package is the server half).
Section numbers in comments (§3.1, §6.1, ...) refer to that document.

OFF BY DEFAULT. ``radient.saver.mode`` ships ``off`` and ``LOP_RADIENT_SAVER``
overrides it; with the gate off nothing in this package runs, no client is
built and no request leaves the machine (:mod:`.gate`).

Module map:

- :mod:`.canonical` — the shared canonical byte form and digest (§3.1);
- :mod:`.projection` — the ONE upload serialiser, ``saver_projection``, and
  ``verbatim_wire`` (§3.1, §3.2 step 2);
- :mod:`.validate` — every §6.1 check on an untrusted response;
- :mod:`.client` — the credential-confined request builder (§7, S-31, S-32);
- :mod:`.chain` — the per-session chain: verify, splice, prefetch, adoption,
  back-off, notices and the spend record (§3.2, §6);
- :mod:`.gate` — the mode setting, the env override and ``resolve_saver`` (§9).
"""
