"""OSWorld 2.0 evaluation adapter for local-operator.

This is a *separate installable distribution*, deliberately living in the repo
tree but OUTSIDE the ``local_operator`` package. The reasons are structural,
not cosmetic:

- ``discovery.distribution_digest`` pins the adapter by verifying every RECORD
  row of the installed wheel. A package inside ``local_operator`` would be
  part of the harness distribution, so every harness release would invalidate
  every adapter pin.
- The adapter imports ``local_operator.evaluation.adapters.api`` to build its
  ``AdapterMetadata``; keeping the source in-tree means one PR can move both
  sides of a protocol change together, while the wheel + digest + isolated
  worker give the isolation that matters at runtime.

The cloud-free slice — task parsing, requirement derivation, provisioning
resolution, cleanup-ref determinism, action and observation translation,
score mapping, and a ``FakeProvider`` that drives a real ``EpisodeRunner``
end to end with zero AWS spend — is the bulk of the adapter and is what CI
exercises. ``providers/aws.py`` is the one module that spends money; it is
the production default and is unit-tested against botocore's Stubber.
"""

from __future__ import annotations

from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter


def create() -> OSWorldV2Adapter:
    """Entry point named by ``AdapterSelector.entry_point``.

    ``discovery.load_selected_adapter`` requires the factory to be a plain
    module attribute that returns an object satisfying ``EvaluationAdapter``.
    Construction must be side-effect free: the worker calls this before any
    state-transition authority exists, and nothing here may touch the network,
    the filesystem outside the workspace, or a cloud API.
    """

    return OSWorldV2Adapter()


__all__ = ["OSWorldV2Adapter", "create"]
