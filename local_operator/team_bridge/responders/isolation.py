"""The isolation attestation a responder spawn must carry (design §4).

UNWIRED BY DESIGN (see ``jail.py``). Slice C2 makes ``SpawnRequest`` require an
:class:`IsolationAttestation` and has ``ResponderRunner`` refuse a spawn whose
attestation :func:`attestation_refusal` rejects; this slice defines the record
and the check so C2 only plumbs them.

WHY A RECORD AND NOT JUST "THE SPAWNER DID IT". The run's boundary is built
from three independent pieces -- the seatbelt profile, the rebuilt environment
and the pinned STS identity -- plus the trees it was given to read. Logging
exactly those (a profile digest, the env KEY list, the assumed-role ARN that
``GetCallerIdentity`` returned, each repository's commit) makes a responder
run auditable after the fact, and refusing without them makes a spawner that
skipped a piece fail closed instead of spawning a wider run. Env VALUES are
never recorded: they include the STS triple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .jail import ENV_ALLOWLIST, ResponderJail
from .materialize import MaterializedRepo, RepoNotAllowed, pinned_repo_name
from .pinned_credentials import EXPECTED_ASSUMED_ROLE_PREFIX, PinnedCredentials

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")


@dataclass(frozen=True)
class IsolationAttestation:
    """What one responder run was confined by. Contains no secret."""

    jail_profile_sha256: str
    env_keys: tuple[str, ...]
    assumed_role_arn: str
    #: ``(radient-ml/<name>, commit sha)`` per materialized repository.
    repos: tuple[tuple[str, str], ...] = ()


def attest(
    jail: ResponderJail,
    env: Mapping[str, str],
    credentials: PinnedCredentials,
    repos: Sequence[MaterializedRepo] = (),
) -> IsolationAttestation:
    """Build the attestation for a spawn, refusing one that would not verify."""

    attestation = IsolationAttestation(
        jail_profile_sha256=jail.profile_sha256(),
        env_keys=tuple(sorted(env)),
        assumed_role_arn=credentials.assumed_role_arn,
        repos=tuple((repo.entry, repo.commit) for repo in repos),
    )
    refusal = attestation_refusal(attestation)
    if refusal is not None:
        raise ValueError(refusal)
    return attestation


def attestation_refusal(attestation: IsolationAttestation | None) -> str | None:
    """Why a spawn carrying ``attestation`` must be refused, or ``None``."""

    if attestation is None:
        return "isolation_missing: a responder spawn requires an isolation attestation"
    if _HEX64.fullmatch(attestation.jail_profile_sha256 or "") is None:
        return "isolation_invalid: no jail profile digest"
    unexpected = sorted(set(attestation.env_keys) - set(ENV_ALLOWLIST))
    if unexpected:
        return f"isolation_invalid: environment keys outside the allowlist: {', '.join(unexpected)}"
    if not (attestation.assumed_role_arn or "").startswith(EXPECTED_ASSUMED_ROLE_PREFIX):
        return "isolation_invalid: the run's identity is not the pinned responder role"
    for entry, commit in attestation.repos:
        try:
            pinned_repo_name(entry)
        except RepoNotAllowed:
            return f"isolation_invalid: repository {entry!r} is not pinned"
        if _COMMIT.fullmatch(commit or "") is None:
            return f"isolation_invalid: repository {entry!r} has no commit"
    return None


__all__ = ["IsolationAttestation", "attest", "attestation_refusal"]
