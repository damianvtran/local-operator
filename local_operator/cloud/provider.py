"""The provisioning seam: one Protocol, and the records that cross it.

WHY A PROTOCOL AND NOT A BASE CLASS. ``FargateProvider``, ``K8sWorkloadProvider``
and ``FakeComputeProvider`` have to be interchangeable here, and the ledger must
never learn which one it holds — so the seam is STRUCTURAL. A shared base class
would put an import of the cloud's own concrete backends under the ledger and
turn "which provider is this" into a question the ledger could answer.

EVERY METHOD IS ASYNC. The AWS path is I/O-bound and the fake has to drive the
same code path as the real one, which is the argument
``benchmarks/osworld_v2_adapter/src/lop_osworld_v2_adapter/providers/base.py``
makes for its own ``EnvironmentProvider``. Every blocking call inside an
implementation runs under ``asyncio.to_thread`` (that adapter's
``providers/aws.py`` states the discipline), so the event loop stays free and a
``FakeComputeProvider`` sharing the path is a real test rather than a mock.

``status()`` REPORTS AND NEVER MUTATES; THE LEDGER OWNS THE STATE MACHINE.
``status()`` says what AWS or the cluster says and nothing more. The rule it
follows is ``network/move_queue.py``'s ("its writers are the source RELAY's
transitions and ``--cancel-queued`` — never the requester's memory"): one writer
per record. A provider that could also write task state would break it, so the
state machine lives on ``types.ComputeTask`` and is driven only by
``ledger.CloudLedger``.

WHERE ``CloudRefused`` COMES FROM. It follows the shape of ``network/types.py``'s
``MeshRefusal`` (a machine ``code`` plus one human sentence) because the CLI
verbs want the refusal style the mesh verbs already use — but it is its OWN
class, not a subclass: a cloud refusal must not be catchable by an
``except MeshRefusal`` written for the mesh, and this package must not widen the
mesh's import graph to hand a sentence to a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - the seam's own records need no runtime import
    from local_operator.cloud.types import PoolRequest, Tier

#: What a provider's own reading of a reference can be. Four values on purpose:
#: ``pending`` is "created and not yet observable" (ECS answers an empty
#: ``describe_tasks`` for the first fractions of a second after RunTask) and
#: ``absent`` is "this provider does not know the reference at all". They are
#: different facts — the ledger's ``lost`` transition keys on the second, and
#: collapsing them would report a task that is merely young as vanished.
ProviderPhase = Literal["pending", "running", "stopped", "absent"]

#: The phase vocabulary as a runtime set, mirrored by the reader below — the one
#: place a string becomes a :data:`ProviderPhase`.
PROVIDER_PHASES: frozenset[str] = frozenset({"pending", "running", "stopped", "absent"})


@dataclass(frozen=True)
class ProviderCapacity:
    """The capacity/describe call: what this provider can accept right now.

    ``detail`` is provider-specific and inert: for Fargate it carries the cluster
    and the ``<family>:<revision>`` the stack points at. It is a plain
    ``dict[str, str]`` rather than a union of per-provider records so that a
    surface can print it without knowing which provider it came from.
    """

    provider: str
    tier: Tier
    ready: bool
    detail: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeStatus:
    """One reading of a provider's own state. Never mutates anything.

    ``exit_code`` is ``None`` until the provider has one to report, which is NOT
    the same as zero: a task that stopped without an exit code is unknown, and
    the ledger must not read it as success.
    """

    provider: str
    provider_ref: str
    phase: ProviderPhase
    exit_code: int | None = None
    stop_reason: str = ""
    started_at: float | None = None
    stopped_at: float | None = None
    #: The node a pod landed on. Empty for Fargate, which has no such thing.
    node: str = ""
    detail: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_ref": self.provider_ref,
            "phase": self.phase,
            "exit_code": self.exit_code,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "node": self.node,
            "detail": dict(self.detail),
        }

    @staticmethod
    def from_json(data: Any) -> ComputeStatus:
        """Parse a reading, defaulting what the writer left out.

        An absent key is not a claim, so every field has its default; a key that
        is PRESENT and names a phase this build does not know is refused by name,
        because the alternative is reporting an unreadable reading as ``pending``
        and letting a caller wait for a task that already ended.
        """
        # Deferred on purpose: ``types`` imports this module at import time (this
        # module holds the seam's refusal), so the reverse direction has to be
        # deferred to keep the package importable in either order. Same pattern
        # and same reason as ``network/move_queue.py``'s deferred lock import.
        from local_operator.cloud.types import (
            _as_optional_float,
            _as_optional_int,
            _as_str,
            _as_str_map,
            _refuse,
        )

        if not isinstance(data, dict):
            _refuse("version_unknown", "cloud: a compute status must be a JSON object")
        phase = _as_str(data.get("phase")) or "pending"
        if phase not in PROVIDER_PHASES:
            _refuse("version_unknown", f"cloud: unknown provider phase {phase!r}")
        return ComputeStatus(
            provider=_as_str(data.get("provider")),
            provider_ref=_as_str(data.get("provider_ref")),
            # Checked against PROVIDER_PHASES just above; the cast is the seam
            # between a runtime vocabulary and the Literal that mirrors it.
            phase=cast(ProviderPhase, phase),
            exit_code=_as_optional_int(data.get("exit_code")),
            stop_reason=_as_str(data.get("stop_reason")),
            started_at=_as_optional_float(data.get("started_at")),
            stopped_at=_as_optional_float(data.get("stopped_at")),
            node=_as_str(data.get("node")),
            detail=_as_str_map(data.get("detail")),
        )

    @property
    def stopped(self) -> bool:
        """Whether this reading says the workload is over, whatever the exit was.

        ``absent`` counts: a reference the provider no longer knows is not a
        running workload, it is one whose end nobody recorded — the distinction
        the LEDGER draws (``lost`` versus ``expired``), not this reading.
        """
        return self.phase in ("stopped", "absent")


@dataclass(frozen=True)
class DrainOutcome:
    """What a drain attempt actually achieved, and what it got back.

    ``drained`` is the ONLY field that may be read as success: a drain that hit
    its deadline reports ``drained=False`` with ``reason: "deadline"`` even when
    the task did stop, because the provider was asked for a checkpoint it never
    confirmed (§3.4's ``drained_unconfirmed``, which exists for exactly this).
    """

    drained: bool
    waited_s: float
    reason: str
    checkpoint: dict[str, str] = field(default_factory=dict)


class CloudRefused(Exception):
    """A refusal that carries a machine ``code`` and a sentence for the operator.

    The two are separate for the reason ``MeshRefusal`` states: tests and the
    audit record key on the code, a person reads the sentence.
    """

    def __init__(self, code: str, sentence: str) -> None:
        super().__init__(sentence)
        self.code = code
        self.sentence = sentence

    def __str__(self) -> str:  # pragma: no cover - Exception's own repr is enough
        return self.sentence


@runtime_checkable
class ComputeProvider(Protocol):
    """One provisioning backend.

    Implemented by ``FargateProvider``, ``K8sWorkloadProvider`` and
    ``FakeComputeProvider``.
    """

    #: The class attribute the ledger and a ``--json`` body name this provider by.
    name: str

    async def describe(self) -> ProviderCapacity:
        """The capacity/describe call: what this provider can accept right now."""
        ...

    async def submit(self, request: PoolRequest) -> str:
        """Create the workload. The ONLY method that creates. Returns the provider_ref."""
        ...

    async def status(self, provider_ref: str) -> ComputeStatus:
        """One reading of the provider's own state. Never mutates it."""
        ...

    async def cancel(self, provider_ref: str, *, reason: str) -> None:
        """Stop it now, without waiting for a checkpoint."""
        ...

    async def drain(self, provider_ref: str, *, deadline_s: float) -> DrainOutcome:
        """Ask it to stop, wait up to deadline_s for a checkpoint, then stop it."""
        ...
