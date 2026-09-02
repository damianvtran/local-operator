"""Cleanup refs and plan: the deterministic, descriptor-only teardown contract.

The hard constraint (C2): a rescue worker enters at HANDSHAKEN and has NEVER
run ``prepare``. It receives only the persisted ``RescueDescriptor`` and must
tear down from ``descriptor.cleanup_plan.actions[*].resource_ref`` alone. The
natural EC2 identifier — the ``i-…`` instance ID — does not exist until
``run_instances`` returns inside ``reset_start``, i.e. after the descriptor is
already persisted. So the ref cannot be an instance ID.

Resolution: the ref is a name WE choose, applied to AWS as a tag inside the
``run_instances`` call itself (TagSpecifications, atomic at creation — a
follow-up ``create_tags`` is a review blocker because there is a window where
the instance exists unnamed). Teardown resolves ID→tag with
``describe_instances(Filters=[tag:lop:episode=<id>])``, which needs nothing but
the episode ID the descriptor already carries.

``CleanupRefs.mint(episode_id)`` is deterministic: the same episode ID always
produces the same refs, client token, and cleanup_plan_id. That is what makes
"rebuilt from the descriptor alone" assertable as a structural equality.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan

# Closed evidence_code vocabulary (StrictIdentifier). Each code is a distinct
# outcome a rescuer or auditor can grep for.
EVIDENCE_INSTANCE_TERMINATED = "instance-terminated"
EVIDENCE_INSTANCE_ABSENT = "instance-absent"
EVIDENCE_TERMINATE_UNCONFIRMED = "terminate-unconfirmed"
EVIDENCE_TERMINATE_DENIED = "terminate-denied"
EVIDENCE_SCHEDULE_DELETED = "schedule-deleted"
EVIDENCE_SCHEDULE_ABSENT = "schedule-absent"
# The scheduler refused or failed the delete for a reason other than "no such
# schedule". Maps to ``attempted`` in the adapter: the TTL lease still exists
# and will fire, which is SAFE (it terminates the instance) but leaves a cloud
# object the operator did not retire, so rescue must look again.
EVIDENCE_SCHEDULE_DELETE_FAILED = "schedule-delete-failed"
EVIDENCE_SESSION_CLOSED = "session-closed"
# An action kind this build cannot execute. Distinct from every "we looked"
# code above: it reports that teardown was never attempted at all, which is
# why it pairs with an ``attempted`` status that keeps rescue_required set
# rather than a ``not_needed`` that would clear it.
EVIDENCE_KIND_UNSUPPORTED = "kind-unsupported"


@dataclass(frozen=True)
class CleanupRefs:
    """The three symbolic refs a plan names. Derived purely from episode_id."""

    instance_ref: str
    lease_ref: str
    session_ref: str
    client_token: str

    @classmethod
    def mint(cls, episode_id: str) -> "CleanupRefs":
        return cls(
            instance_ref=f"lop-ep-{episode_id}",
            lease_ref=f"lop-ttl-{episode_id}",
            session_ref=episode_id,
            client_token=hashlib.sha256(f"lop-osworld-v2|{episode_id}".encode()).hexdigest()[:32],
        )

    @classmethod
    def from_descriptor_actions(cls, actions: tuple[Any, ...]) -> "CleanupRefs":
        """Reconstruct refs from a persisted plan's resource_refs alone.

        This is the rescue guarantee stated as a constructor: given ONLY the
        actions a descriptor carries, recover the refs. A rescue worker never
        ran prepare, so this must work without any episode context beyond the
        refs themselves. It parses rather than mints so that a plan authored
        by an older adapter build is still rescuable.
        """

        instance_ref = lease_ref = session_ref = None
        for action in actions:
            if action.kind == "release_instance":
                instance_ref = action.resource_ref
            elif action.kind == "revoke_lease":
                lease_ref = action.resource_ref
            elif action.kind == "close_session":
                session_ref = action.resource_ref
        if instance_ref is None or lease_ref is None or session_ref is None:
            raise ValueError("cleanup plan does not carry the three OSWorld refs")
        episode_id = session_ref
        return cls(
            instance_ref=instance_ref,
            lease_ref=lease_ref,
            session_ref=session_ref,
            client_token=hashlib.sha256(f"lop-osworld-v2|{episode_id}".encode()).hexdigest()[:32],
        )


def build_cleanup_plan(episode_id: str, refs: CleanupRefs) -> CleanupPlan:
    """The declarative plan prepare returns. No action targets an instance ID.

    - ``release-instance`` terminates the EC2 instance named by the tag; the
      root volume dies with it (DeleteOnTermination=true default), so there is
      deliberately no ``delete_volume`` action — it could only ever report
      ``not_needed`` and add noise.
    - ``revoke-ttl-lease`` deletes the EventBridge one-shot OSWorld creates.
    - ``close-session`` drops the DesktopEnv refs; always succeeds, never raises.
    """

    return CleanupPlan(
        episode_id=episode_id,
        actions=(
            CleanupAction(
                action_id="release-instance",
                kind="release_instance",
                resource_ref=refs.instance_ref,
                timeout_ms=60_000,
                max_attempts=2,
            ),
            CleanupAction(
                action_id="revoke-ttl-lease",
                kind="revoke_lease",
                resource_ref=refs.lease_ref,
                timeout_ms=30_000,
                max_attempts=2,
            ),
            CleanupAction(
                action_id="close-session",
                kind="close_session",
                resource_ref=refs.session_ref,
                timeout_ms=10_000,
                max_attempts=2,
            ),
        ),
    )
