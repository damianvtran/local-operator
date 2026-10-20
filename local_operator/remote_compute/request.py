"""The run-create body: a network is REQUIRED, and it is the tenant field.

WHY A MODEL INSTEAD OF A DICT. The server keys a remote run by the network it
was submitted through; a request built without one would have to be refused at
the far end, after the work was already addressed to a host. Making
``binding`` a required field means a later slice cannot construct the body at
all without deciding the tenant — the type checker catches the omission, not a
remote 4xx.

WHY A BINDING AND NOT AN ID. ``NetworkBinding`` is issued only by
``resolve_network``, which answers from this device's local mesh store. A bare
``str`` would accept ANY well-formed id, so the membership guarantee would hold
only while every caller happened to resolve; requiring the binding makes it a
type fact. The id on the wire is read from the binding.

THE REMAINING FIELDS ARE DELIBERATELY NOT FROZEN HERE. This slice only settles
the tenant half of the contract; the command, environment, timeout and result
shape belong to the slice that adds ``lop exec --remote``. They therefore travel
as :attr:`RunCreateRequest.body`, merged into the JSON unchanged — with one
exception: a body may not carry its own ``network_id``, because a second
spelling of the tenant field is exactly how two readings of a request come to
disagree.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from local_operator.remote_compute.resolve import (
    CODE_DUPLICATE_NETWORK_ID,
    CODE_INVALID_NETWORK_ID,
    NETWORK_ID_PATTERN,
    NetworkBinding,
)


@dataclasses.dataclass(frozen=True)
class RunCreateRequest:
    """A run-create body bound to one network, ready to serialise.

    Build it from a :class:`~local_operator.remote_compute.resolve.NetworkBinding`
    — the only value that proves membership — and the id travels from there.
    The shape check is repeated here with ``fullmatch`` because this object is
    the last place before the id goes on the wire; it is the same constant, not
    a second rule.
    """

    binding: NetworkBinding
    #: Everything else the run-create body carries, unchanged into the JSON.
    body: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # IMPORTED HERE so this module's import stays stdlib + resolve (which is
        # itself light); a refusal is raised out of the same typed family.
        from local_operator.network.types import MeshRefusal

        # ``isinstance`` rather than a type annotation alone: annotations are
        # not enforced at runtime, and a bare id is the exact mistake this model
        # exists to make unreachable.
        if not isinstance(self.binding, NetworkBinding):
            raise TypeError(
                "RunCreateRequest takes a NetworkBinding issued by "
                "remote_compute.resolve_network(); a bare network id cannot "
                "prove this device is a member of the network it names"
            )
        if not NETWORK_ID_PATTERN.fullmatch(self.binding.network_id):
            raise MeshRefusal(
                CODE_INVALID_NETWORK_ID,
                f"{self.binding.network_id!r} is not a minted mesh network id; "
                "resolve the network first (remote_compute.resolve_network) and "
                "use its binding",
            )
        if "network_id" in self.body:
            raise MeshRefusal(
                CODE_DUPLICATE_NETWORK_ID,
                "the run body may not carry its own `network_id`: the tenant is a "
                "field of this model, and two spellings of it could disagree",
            )

    @property
    def network_id(self) -> str:
        """The tenant id, read from the binding — never from a caller's string."""
        return self.binding.network_id

    def to_json(self) -> dict[str, Any]:
        """The wire object: the caller's fields, then the tenant field on top."""
        return {**self.body, "network_id": self.binding.network_id}
