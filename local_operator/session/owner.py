"""``SessionOwner``: where a session's runtime lives and how a viewer reaches it.

WHY THIS EXISTS. A remote session is the *same object* as a local one — the same
``AttachedSession`` facade, the same control protocol, the same command surface
(``mesh-session-mobility.md`` §0 and §3.1). Everything a viewer does therefore
works remotely by construction rather than by enumeration. The one thing that
genuinely differs is the answer to three questions the local case answers from
``run/mobile``:

* where the owner is (``locate``),
* how to start one (``engage``),
* how to dial it (``make_client``).

Those three are the whole of this protocol, and they are the ONLY mesh seam in
the facade (§3.1: "No other member of ``AttachedSession`` learns about the
mesh"). ``LocalOwner`` below is the default and is exactly today's code paths,
so a caller that supplies nothing gets today's behaviour byte for byte — that
is the zero-peer regression (R16 topology 0).

WHY THIS MODULE IS IMPORT-LIGHT, AND WHY IT IS NOT IN ``network/``. §3.1 says so:
the facade's seam must import without the network package, because
``local_operator/network/**`` is the mesh transport and ``AttachedSession`` is
built on the CLI startup path for every ``lop`` invocation, mesh or not. So the
protocol and the local implementation live here, and the remote implementation —
which does reach the network package — lives in ``network/projection.py``.

Every heavy import below is function-local for that reason: ``asyncio`` (the
module is imported by synchronous CLI code), the attach client, and the launch
machinery (``mobile.attach_client`` and ``session.runtime.launch`` are both far
too heavy to load for a listing).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from local_operator.session.placement import SessionPlacement

if TYPE_CHECKING:  # both are function-local at runtime (see the module docstring)
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.launch import WarmErrand
    from local_operator.session.runtime.types import SessionRecord


#: The placement every pre-mesh session has. One shared frozen value rather than
#: a fresh construction per facade: it is compared (``placement.mode == "local"``)
#: and returned from a property that a TUI polls.
LOCAL_PLACEMENT = SessionPlacement(mode="local", policy="pinned")


@dataclass(frozen=True)
class SessionSeed:
    """What a viewer knows about a session BEFORE it binds.

    §3.4: a remote viewer's blank first frame is a *visible* defect, not a
    cosmetic one, because the sidebar paints before the bind lands. A local
    viewer seeds from the local transcript, so it never noticed; a remote one has
    no local directory and would paint "0 messages". This carries the row's facts
    — as far as the catalogue knows them — so the band reads "connecting to
    <device>…" with the right name and title instead of an empty conversation.

    Every field is optional and ``None`` means "not known", which the facade
    renders exactly as it renders a cold local session with no history.
    """

    name: str = ""
    model_label: str = ""
    cwd: str = ""
    mtime: float | None = None
    history_message_count: int | None = None
    #: The device the session lives on, for the connecting band. Empty for a
    #: local seed, which is what keeps a local viewer's copy unchanged.
    device_name: str = ""


@runtime_checkable
class SessionOwner(Protocol):
    """Where a session's runtime lives and how to reach it. Three questions."""

    @property
    def placement(self) -> SessionPlacement:
        """How this session is placed — the facade's ``runtime_locality`` reads it."""
        ...

    def locate(self) -> tuple[SessionRecord | None, int | None]:
        """The owner's record and pid, or ``(None, pid)`` / ``(None, None)``.

        Same contract as ``mobile.attach_client.find_runtime_record``, whose
        docstring is the specification: ``(None, pid)`` means an owner exists but
        published no usable record, ``(None, None)`` means no owner at all. Both
        states must be reproduced by a remote owner (§3.4), because
        ``_bind_under_lock`` branches on exactly that difference.

        SYNCHRONOUS on purpose: every caller of the local form wraps it in
        ``asyncio.to_thread`` because the registry scan touches the disk, and a
        remote implementation whose answer comes from a socket must keep the same
        blocking contract or the caller's ``to_thread`` would buy nothing.
        """
        ...

    async def engage(self, *, cwd: str, warm: "WarmErrand", **kwargs: Any) -> None:
        """Make an owner exist, or join one that is starting.

        ``kwargs`` carries the bind loop's own budgets (``preempt``,
        ``preempt_budget_s``), which the local implementation forwards to
        ``launch.engage_runtime`` and a remote one ignores — it has no local
        spawn to preempt.
        """
        ...

    def make_client(
        self, on_projection: Any, on_disconnected: Any, **kwargs: Any
    ) -> "AttachClient":
        """The frame-level client for this owner, already configured.

        The signature mirrors ``AttachClient.__init__`` (two positional
        callbacks, keyword-only the rest) rather than taking an opaque
        ``**callbacks`` blob, because the facade builds the callbacks — they
        close over ITS state — and only the transport differs. One client class
        per owner rather than one client with a mode flag, which is what makes
        ``RemoteSessionClient`` able to override ``connect`` and nothing else.
        """
        ...


class LocalOwner:
    """The default owner: exactly the code paths that existed before the mesh.

    Every import is function-local and every call is the same call the facade
    made before this seam existed, so a facade constructed with no ``owner``
    argument runs the identical machine code it always did. That is not
    politeness — it is the property the zero-peer regression test asserts, and
    the reason the seam is three calls rather than a parallel implementation.
    """

    def __init__(self, config_dir: Path, session_id: str) -> None:
        self._config_dir = Path(config_dir)
        self._session_id = session_id

    @property
    def placement(self) -> SessionPlacement:
        return LOCAL_PLACEMENT

    def locate(self) -> tuple["SessionRecord | None", int | None]:
        from local_operator.mobile.attach_client import find_runtime_record

        return find_runtime_record(self._config_dir, self._session_id)

    async def engage(self, *, cwd: str, warm: "WarmErrand", **kwargs: Any) -> None:
        from local_operator.session.runtime.launch import engage_runtime

        await engage_runtime(
            self._session_id,
            cwd,
            warm,
            config_dir=self._config_dir,
            preempt=kwargs.get("preempt"),
            preempt_budget_s=kwargs.get("preempt_budget_s", 0.0),
        )

    def make_client(
        self, on_projection: Any, on_disconnected: Any, **kwargs: Any
    ) -> "AttachClient":
        from local_operator.mobile.attach_client import AttachClient

        return AttachClient(on_projection, on_disconnected, **kwargs)


def owner_for(owner: "SessionOwner | None", config_dir: Path, session_id: str) -> "SessionOwner":
    """The owner to use: the one supplied, else the local one.

    A FUNCTION rather than an ``or`` at the construction site so the default is
    one fact in one place — a facade that defaulted to a remote owner by
    accident, or to none at all, is the failure this seam exists to make
    impossible.
    """
    return owner if owner is not None else LocalOwner(config_dir, session_id)
