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

THE REMAINING FIELDS ARE FROZEN IN :class:`RunSpec`, NOT IN THE REQUEST. The
request still takes a mapping (:attr:`RunCreateRequest.body`) so its one
invariant stays legible, but the tier-1 fields the wire contract names are
built by ``RunSpec`` and cross into the request through ``RunSpec.bind``. The
split is deliberate: adding typed fields BESIDE the mapping would give the same
body two spellings, and a body that disagrees with itself is exactly what the
``network_id`` rule below already refuses. A body may never carry its own
``network_id``, because that is the second spelling of the tenant field.
"""

from __future__ import annotations

import dataclasses
import math
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


#: The compute sizes ``POST /v1/remote-compute/runs`` accepts.
RUN_SIZES: tuple[str, ...] = ("s", "m", "l")


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """The tier-1 run body: what a short-lived remote run is made of.

    WHY A SECOND MODEL. ``RunCreateRequest`` holds the tenant invariant and
    nothing else; this holds the fields the public contract names, with their
    shapes checked ONCE, here, before a binding exists. ``bind`` is the only
    way from here into a request, so a caller cannot assemble a body whose
    ``network_id`` came from anywhere but ``resolve_network``.

    ``hosting`` AND ``model`` ARE REQUIRED AND DECLARED. A remote run does not
    inherit this device's provider keys, so the far end must be told which
    model to run; the CLI resolves them through the same ladder a local exec
    uses (agent > flag > team > config) and sends what the operator resolved.

    ``budget_usd`` is optional because the server applies its own admission
    default when absent; ``name`` is optional because a run without a label is
    still a run.
    """

    prompt: str
    repo_url: str
    repo_sha: str
    hosting: str
    model: str
    size: str = "m"
    max_hours: float = 2.0
    budget_usd: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        # REFUSED HERE, ONCE: every one of these would otherwise travel to the
        # far end and come back as a 400 the operator cannot act on, after the
        # request had already been given an idempotency key.
        for field_name in ("prompt", "repo_url", "repo_sha", "hosting", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"a remote run needs a non-empty {field_name}")
        if self.size not in RUN_SIZES:
            raise ValueError(
                f"unknown remote run size {self.size!r}; the sizes are {', '.join(RUN_SIZES)}"
            )
        if not math.isfinite(self.max_hours) or self.max_hours <= 0:
            raise ValueError(f"max_hours must be a positive number, got {self.max_hours!r}")
        if self.budget_usd is not None and (
            not math.isfinite(self.budget_usd) or self.budget_usd < 0
        ):
            raise ValueError(f"budget_usd must be zero or more, got {self.budget_usd!r}")

    def to_body(self) -> dict[str, Any]:
        """The wire fields, with the two optional ones OMITTED when unset.

        Omitted rather than sent as ``null``: an explicit null for ``budget_usd``
        reads as a choice (no budget) where absence reads as "use the server's
        admission default", and only one of those is what the caller meant.
        """
        body: dict[str, Any] = {
            "repo_url": self.repo_url,
            "repo_sha": self.repo_sha,
            "prompt": self.prompt,
            "hosting": self.hosting,
            "model": self.model,
            "size": self.size,
            "max_hours": self.max_hours,
        }
        if self.name:
            body["name"] = self.name
        if self.budget_usd is not None:
            body["budget_usd"] = self.budget_usd
        return body

    def bind(self, binding: NetworkBinding) -> RunCreateRequest:
        """Pair this body with a resolved network; the only crossing into a request."""
        return RunCreateRequest(binding, self.to_body())
