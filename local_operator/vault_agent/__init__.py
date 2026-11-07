"""The harness side of Radient Vault's agent arm.

WHAT THIS IS. A Team-plan person can let THIS machine's agent read a credential a
teammate shared with them in Radient Vault, without the agent ever seeing a
value. The person grants that access in the console (a "delegation"); the grant
binds the agent's Ed25519 public key to the person's own Team Bridge endpoint.
At resolve time the harness proves possession of the matching private key over
the exact operation, and the authority verifies it. Contract: agent-server
``docs/VAULT.md`` §4 ("The agent vault tool"), §4.1 step 4 (the agent proof),
§6.12 (route 1, ``delegations_available``) and §6.20 (delegations, the route-29
pickup).

THE MODULES, and why they are split this way:

* :mod:`.wire` — the byte encodings (commitment, agent-proof message, key
  fingerprint). Pure and dependency-light so the authority's cross-repo vectors
  pin it exactly.
* :mod:`.keys` — the delegation seed, kept in the local encrypted store under a
  reserved namespace no agent-class read can reach (``LOP_VAULT_DELEGATION_*``).
* :mod:`.client` — routes 1, 15, 27 and 29 with the signed-in PERSON's bearer.
* :mod:`.eligibility` — the cached, network-free gate that decides whether the
  ``secret`` tool offers ``vault_resolve`` at all, plus the cached grant.
* :mod:`.resolve` — the one resolve flow the tool calls (mint, prove, call).
* :mod:`.cli` — ``lop vault connect|status|disconnect``.

WHAT THIS IS NOT. The local encrypted store stays the default for every secret
the operator keeps; Vault is for a value a TEAMMATE shared. Agents cannot create
or upload Vault items. And nothing here is a hardware boundary: a process running
as the operator that is willing to run ``lop`` can read the delegation seed, which
is the encrypted store's stated boundary (``AGENTS.md``, "Credentials").
"""
