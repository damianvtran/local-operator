"""The run-create body: a network is REQUIRED, and it is the tenant field.

WHY A MODEL INSTEAD OF A DICT. The server keys a remote run by the network it
was submitted through; a request built without one would have to be refused at
the far end, after the work was already addressed to a host. Making
``network_id`` a required field means a later slice cannot construct the body at
all without deciding the tenant — the type checker catches the omission, not a
remote 4xx.

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
    — the only way this package produces a valid id — or from an id that has the
    minted shape. Either spelling is validated here, because this object is the
    last place before the id goes on the wire.
    """

    network_id: str
    #: Everything else the run-create body carries, unchanged into the JSON.
    body: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # IMPORTED HERE so this module's import stays stdlib + resolve (which is
        # itself light); a refusal is raised out of the same typed family.
        from local_operator.network.types import MeshRefusal

        if not NETWORK_ID_PATTERN.match(self.network_id):
            raise MeshRefusal(
                CODE_INVALID_NETWORK_ID,
                f"{self.network_id!r} is not a minted mesh network id; resolve the "
                "network first (remote_compute.resolve_network) and use its id",
            )
        if "network_id" in self.body:
            raise MeshRefusal(
                CODE_DUPLICATE_NETWORK_ID,
                "the run body may not carry its own `network_id`: the tenant is a "
                "field of this model, and two spellings of it could disagree",
            )

    @classmethod
    def for_binding(
        cls, binding: NetworkBinding, body: Mapping[str, Any] | None = None
    ) -> "RunCreateRequest":
        """The body for a resolved binding — the intended construction path."""
        return cls(network_id=binding.network_id, body=dict(body or {}))

    def to_json(self) -> dict[str, Any]:
        """The wire object: the caller's fields, then the tenant field on top."""
        return {**self.body, "network_id": self.network_id}
