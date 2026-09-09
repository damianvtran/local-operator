"""The in-process secret library agents reach from ``eval`` (design §5.3).

    from local_operator.secrets import secrets
    token = secrets["GITHUB_TOKEN"]

**Why a lazy mapping and not injected globals.** The alternative the design
rejected was seeding every secret into the eval worker's namespace at spawn.
That would put every secret in the worker's memory whether or not the cell
wants one, expose them to ``print(globals())`` and any namespace introspection,
and force a decision about WHICH secrets to inject before knowing what the cell
does. A mapping decides nothing up front: each ``__getitem__`` is one store
round trip and one audit row, so the audit trail records retrievals at the
granularity they actually happen rather than "the kernel started".

**Why the value never reaches the model.** The bytes live in the worker
process. They enter the transcript only if the cell prints or returns them, and
:data:`_LEDGER` is the guard for exactly that: every value handed out here is
registered, and :mod:`local_operator.tools.eval_worker` scrubs every registered
value out of stdout, stderr, the trailing-expression result, display output and
the streaming frames that feed ``jobs(op='peek')``. That is a belt-and-braces
sink, not the primary mechanism — the primary mechanism is that a cell has no
reason to print a credential — but the failure it guards is the one that
matters most on this series, so it is unconditional.

**Why the ledger is consulted through ``sys.modules`` rather than imported.**
``eval_worker`` is on the spawn path of every eval kernel and this module pulls
the storage stack behind it. A worker that never touches a secret must not pay
for SQLite and AES, so the worker checks whether this module is ALREADY
imported and skips the whole ledger when it is not. That is sound rather than
merely cheap: if nothing imported this module, nothing obtained a value through
it, and there is nothing to scrub.

**Residual risk, stated because §9 requires it not be overclaimed.** This is
not a sandbox. A cell that deliberately writes a retrieved value to a file, or
posts it somewhere, is doing what the operator asked the agent to be able to
do; the ledger scrubs the transcript, not the world. And anything running as
the operator that is willing to run ``lop`` can obtain these values itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


class SecretValue(str):
    """A retrieved secret that does not paint itself in a ``repr``.

    A ``str`` subclass rather than a wrapper, because the whole point is that
    it drops into the code the agent was going to write anyway:
    ``requests.get(url, headers={"Authorization": f"Bearer {token}"})``,
    ``token.encode()``, ``"a" + token`` all behave exactly as a ``str`` does.

    **Only ``__repr__`` lies, and deliberately not ``__str__``.** A bare
    ``token`` as a cell's trailing expression is rendered with ``repr``, and so
    is a secret sitting inside a printed list or dict — those are the accidents
    worth catching, and catching them costs nothing. Overriding ``__str__``
    would instead break the primary use: ``f"Bearer {token}"`` formats through
    ``__str__``, so a redacting ``__str__`` would silently send the literal text
    ``Bearer [redacted]`` to the API and produce an authentication failure that
    looks like a bad credential rather than a harness bug. The safety net for
    ``print(token)`` is the worker ledger below, which scrubs the bytes out of
    the captured stream whatever route they took — strictly stronger than a
    lying ``__str__``, and it cannot break a working request.
    """

    #: ``name`` must be declared here: ``str`` defines ``__slots__``, so a
    #: subclass with an empty ``__slots__`` has no ``__dict__`` and cannot take
    #: an instance attribute at all.
    __slots__ = ("name",)

    #: The secret's name in the store. Not secret — it is exactly what the
    #: model is told — so it rides on the object for a readable ``repr``.
    name: str

    def __new__(cls, value: str, name: str = "") -> "SecretValue":
        instance = super().__new__(cls, value)
        # `object.__setattr__` is not needed (str is not frozen), but the
        # attribute must be set on the instance rather than the class, or every
        # SecretValue would report the last-retrieved name.
        instance.name = name
        return instance

    def __repr__(self) -> str:
        label = f" {self.name}" if self.name else ""
        return f"<secret{label}: [redacted], {len(self)} chars>"


class _RedactionLedger:
    """Every secret value this process has handed out, for output scrubbing.

    Thread-safe because a cell may retrieve from a worker thread while the
    foreground thread is building the response; a set mutated during iteration
    would raise inside the response path and turn a successful cell into a
    crash.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: set[str] = set()

    def register(self, value: str) -> None:
        if not value:
            # Registering "" would make `str.replace` insert the marker between
            # every character of every output.
            return
        with self._lock:
            self._values.add(value)
        _publish(value)

    def values(self) -> list[str]:
        """Registered values, longest first.

        Longest first so a value that is a prefix of another cannot leave the
        remainder of the longer one visible — the same ordering rule
        :func:`local_operator.variables.redact_secret_values` documents.
        """
        with self._lock:
            return sorted(self._values, key=len, reverse=True)

    def scrub(self, text: str) -> str:
        """Replace every registered value in ``text`` with ``[redacted]``."""
        for value in self.values():
            if value in text:
                text = text.replace(value, "[redacted]")
        return text


#: Process-wide ledger. Module-level because the eval worker's namespace is
#: rebuilt per cell while the kernel (and its retrievals) outlive one cell: a
#: secret fetched in cell 3 must still be scrubbed out of cell 9's output.
_LEDGER = _RedactionLedger()


def registered_values() -> list[str]:
    """Values retrieved through this module, for an output filter to scrub."""
    return _LEDGER.values()


def scrub(text: str) -> str:
    """``text`` with every retrieved secret replaced by ``[redacted]``."""
    return _LEDGER.scrub(text)


#: Optional hook the eval worker installs so the parent process learns the
#: values it must scrub out of the worker's real-fd crash tail. The worker owns
#: the transport (a dedicated fd); this module owns the trigger (registration).
#: ``None`` in every non-worker process, where no such parent exists.
_PUBLISH_HOOK: Any = None


def set_publish_hook(hook: Any) -> None:
    """Install the per-registration publish hook (worker only; see R1)."""
    global _PUBLISH_HOOK
    _PUBLISH_HOOK = hook


def _publish(value: str) -> None:
    """Notify the installed hook of a newly registered value, best-effort.

    A hook failure (the parent's fd closed, an old parent) must never fault
    the retrieval that triggered it — the value is already safely in this
    process's ledger, which covers every in-worker channel.
    """
    hook = _PUBLISH_HOOK
    if hook is None:
        return
    try:
        hook(value)
    except BaseException:  # noqa: BLE001 — publication is advisory, never fatal
        pass


class SecretsMapping(Mapping[str, SecretValue]):
    """Read-through view of the long-term store, one round trip per lookup.

    A ``Mapping`` rather than a bare function so a cell can write the two forms
    an agent naturally reaches for — ``secrets["NAME"]`` and
    ``secrets.get("NAME")`` — and so ``"NAME" in secrets`` answers without
    retrieving (and therefore without an audit row claiming a read that never
    happened).

    **Nothing is cached.** A second lookup is a second audit row, which is the
    point: the audit trail is meant to record what the process actually did.
    Caching would also keep bytes alive in a kernel that outlives the cell that
    wanted them.
    """

    def __init__(self, base: "Path | None" = None) -> None:
        # Only ever non-None in tests, which must not reach the operator's real
        # store. Threaded through to `open_store` rather than read from the
        # environment so a test cannot forget to isolate itself.
        self._base = base

    def _open(self) -> Any:
        """Open the store through the broker seam.

        Imported here and not at module scope: this module is imported into
        every eval worker namespace, and the storage stack behind
        :mod:`local_operator.secrets.access` (SQLite plus the AES binding) is a
        cost only a cell that actually wants a secret should pay.
        """
        from local_operator.secrets.access import open_store

        return open_store(self._base)

    def __getitem__(self, name: str) -> SecretValue:
        from local_operator.secrets.access import session_id
        from local_operator.secrets.errors import SecretNotFound

        key = str(name)
        try:
            raw = self._open().get(key, session_id=session_id())
        except SecretNotFound as exc:
            # A KeyError, because this is a Mapping and `secrets.get(...)` and
            # `"X" in secrets` are built on __getitem__ raising it. The store's
            # own message is kept as the argument so a cell that prints the
            # exception still gets the actionable sentence.
            raise KeyError(str(exc)) from exc
        value = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        # REGISTER BEFORE RETURNING. The cell gets the value only after the
        # scrubber knows about it, so there is no window in which a cell could
        # print a value the response filter has not been told about.
        _LEDGER.register(value)
        return SecretValue(value, key)

    def __contains__(self, name: object) -> bool:
        """Whether a secret exists, WITHOUT retrieving it.

        Deliberately routed through ``describe`` rather than ``get``: an
        existence check is not a read, and recording it as one would fill the
        audit trail with retrievals that never handed out a byte.
        """
        from local_operator.secrets.errors import SecretStoreError

        try:
            self._open().describe(str(name))
        except SecretStoreError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        from local_operator.secrets.errors import SecretStoreError

        try:
            records = self._open().list()
        except SecretStoreError:
            # An absent or unopenable store iterates empty rather than raising:
            # `list(secrets)` in a cell that is exploring should report "none
            # here", and the verbs that actually need the store still raise.
            return iter([])
        return iter([record.name for record in records])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        # Never enumerates: __repr__ runs in incidental places (a traceback, a
        # debugger, an accidental trailing expression) and neither a store hit
        # nor an audit row belongs on any of them.
        return "<local_operator secrets: secrets['NAME'] retrieves one value>"


#: The instance agents use. Created eagerly because construction touches
#: nothing — every store access is inside a method — so importing this module
#: still costs only the stdlib.
secrets = SecretsMapping()
