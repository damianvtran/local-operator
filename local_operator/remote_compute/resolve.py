"""Resolve a run's mesh network against THIS device's local store.

THE GUARANTEE THIS MODULE ENFORCES: a remote request can only ever carry a
network this device's LOCAL RECORD knows it belongs to. The input is a name or
an id a human typed; the output is a :class:`NetworkBinding` — the network's
minted id plus its display name — and that object is constructible NOWHERE
ELSE, so the guarantee is carried by the type rather than by every caller
remembering to resolve. The store lists local records, so an id the device
never joined answers ``unknown`` rather than "not reachable", and a membership
this device lost (removed, refused by peers, untrusted after a panic) is
refused before any request is built. "Local record" is the honest bound: a
removal this device has not yet received is not visible here, and no local read
can show it.

WHY THE RESOLUTION RULES LOOK FAMILIAR: ``network.cli._resolve`` already answers
the same question for the CLI's verbs, with the same "never guess between two
tenants" rule. The primitives it uses are reused here rather than restated —
``store.list_networks`` for the local set and ``store.match_networks`` for
"id first, then name" — so the two surfaces cannot disagree about what a
``<network>`` argument means. The rules themselves (one network and no argument
means that one; zero means name the remedy; two or more means refuse and list
them) are the CLI's, kept identical on purpose.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_operator.network.types import NetworkRecord

#: The minted network id (``network/store.py::new_network_id``).
#:
#: Validated on the RESOLVED record, not merely on user input: the id is what
#: the run-create body carries, and the server keys a tenant by it. A record
#: minted before the ``n_``+hex spelling existed still resolves locally (the
#: store preserves older ids verbatim), but it cannot be sent as this field —
#: so the refusal names the record rather than sending an id the far side
#: cannot key.
#:
#: MATCHED WITH ``fullmatch`` BY CALLERS, never ``$``: ``$`` also matches before
#: a trailing newline, so ``^n_[0-9a-f]{24}$`` admits ``"n_<hex>\n"``.
NETWORK_ID_PATTERN = re.compile(r"n_[0-9a-f]{24}")

#: The construction token for :class:`NetworkBinding`. Module-private on
#: purpose: :func:`resolve_network` is its only issuer, so holding a binding is
#: PROOF that this device's local record vouched for the network. A bare id
#: beside it would let a caller skip that proof — the hole this field closes.
_ISSUE_TOKEN = object()

#: Refusal codes, spelled once because a caller (and a test) branches on the
#: string. They are ``MeshRefusal`` codes, the vocabulary every mesh surface
#: already uses for a machine-readable "no".
CODE_NO_NETWORK = "remote_compute_no_network"
CODE_UNKNOWN_NETWORK = "remote_compute_unknown_network"
CODE_AMBIGUOUS_NETWORK = "remote_compute_ambiguous_network"
CODE_NOT_A_MEMBER = "remote_compute_not_a_member"
CODE_INVALID_NETWORK_ID = "remote_compute_invalid_network_id"
CODE_DUPLICATE_NETWORK_ID = "remote_compute_duplicate_network_id"


@dataclasses.dataclass(frozen=True)
class NetworkBinding:
    """The one network a remote run is bound to: its minted id and display name.

    Frozen because it is the tenant decision a request is built from — a
    binding mutated between two reads could name one network in the body and
    another in the receipt.

    ISSUED ONLY BY :func:`resolve_network`. The private token is compared in
    ``__post_init__``, so ``NetworkBinding(network_id=..., name=...)`` raises
    ``TypeError``: membership is not a property a caller can assert into
    existence, and a request that took a bare id could never prove it either.
    The token is excluded from ``repr`` and equality because it is proof of
    issue, not part of what the binding MEANS.
    """

    network_id: str
    name: str
    _issued_by: object = dataclasses.field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issued_by is not _ISSUE_TOKEN:
            raise TypeError(
                "NetworkBinding is issued only by remote_compute.resolve_network(); "
                "a binding built by hand has not checked this device's membership"
            )


def resolve_network(network: str | None, root: Path | None = None) -> NetworkBinding:
    """The network ``network`` means on THIS device, or a refusal naming why not.

    ``network`` is a name or an id; ``None`` (or empty) asks for "the device's
    network", which is answerable only when there is EXACTLY one USABLE one — a
    record this device is an active member of. With none the remedy is to create
    or join one, and with several the reader's problem is which one, so neither
    is guessed. ``root`` redirects the mesh store's config root; tests pass a
    temp directory and production passes nothing.
    """
    # IMPORTED HERE, not at module scope: see the package docstring. The store
    # reaches the record/identity graph, and the flag reader must not pay for it.
    from local_operator.network import store
    from local_operator.network.types import MeshRefusal

    records = store.list_networks(root)
    if not network:
        # ONLY records this device is an ACTIVE member of are candidates. A
        # removed or untrusted record cannot be a tenant a run is sent through,
        # and counting it would report "ambiguous" — naming a network this
        # device is not in — where the usable set is unambiguous.
        candidates = [row for row in records if _is_active_member(row)]
        if len(candidates) == 1:
            record = candidates[0]
        elif not candidates:
            raise MeshRefusal(
                CODE_NO_NETWORK,
                "remote compute needs a mesh network this device is an active "
                "member of, and it is in none; create or join one with "
                "`lop network init`, then switch remote compute on",
            )
        else:
            raise MeshRefusal(
                CODE_AMBIGUOUS_NETWORK,
                "name a network for this remote run: this device is an active "
                "member of " + ", ".join(f"{row.name} ({row.network_id})" for row in candidates),
            )
    else:
        matches = store.match_networks(records, network)
        if not matches:
            raise MeshRefusal(
                CODE_UNKNOWN_NETWORK,
                f"this device is not in a network called {network!r}",
            )
        if len(matches) > 1:
            raise MeshRefusal(
                CODE_AMBIGUOUS_NETWORK,
                f"{network!r} matches {len(matches)} networks; use the network id",
            )
        record = matches[0]

    _require_active_membership(record)
    if not NETWORK_ID_PATTERN.fullmatch(record.network_id):
        raise MeshRefusal(
            CODE_INVALID_NETWORK_ID,
            f"the network record {record.name!r} carries the id {record.network_id!r}, "
            "which is not a minted mesh network id; re-create or re-pair the network "
            "before sending work to it",
        )
    return NetworkBinding(network_id=record.network_id, name=record.name, _issued_by=_ISSUE_TOKEN)


def _is_active_member(record: "NetworkRecord") -> bool:
    """Whether :func:`_require_active_membership` would ADMIT ``record``.

    The no-argument resolution picks its candidates with this, so the set it
    counts and names is exactly the set a run could be sent through.
    """
    if record.self_member() is None:
        return False
    from local_operator.network.relay import membership_state

    return membership_state(record)["state"] == "active"


def _require_active_membership(record: "NetworkRecord") -> None:
    """Refuse unless ``membership_state`` says this device is an ACTIVE member.

    ``membership_state`` is the network's single reading of "what is this
    device's standing here" — it is what ``ls``, ``show`` and ``doctor`` paint,
    and its docstring exists because three copies of that rule had already
    disagreed. A fourth copy here would be the one nobody was looking at, so it
    is CALLED (lazily: ``network.relay`` is heavy and only a resolution pays for
    it) and every state but ``active`` — removed, refused by peers, untrusted
    after a panic — is a refusal. Untrusted is not "not a member" in name, but it
    is exactly the state in which a peer will not admit this device's work, so
    sending the run would fail at the far end instead of here.

    NO SELF ROW IS ALSO A REFUSAL, checked BEFORE the shared reading because
    that reading answers ``active`` for a record with no row for this device: it
    reports what this device can establish about the TRUST state, and silence in
    the member table is not membership. A removal this device has not received
    looks exactly like that here, so the safe answer is the one that does not
    send work to a peer that may have dropped us.
    """
    from local_operator.network.relay import membership_state
    from local_operator.network.types import MeshRefusal

    if record.self_member() is None:
        raise MeshRefusal(
            CODE_NOT_A_MEMBER,
            f"this device's local record of {record.name!r} has no member row for "
            "it, so its membership cannot be shown; remote compute only sends "
            "work to a network this device is an active member of",
        )
    state = membership_state(record)
    if state["state"] == "active":
        return
    raise MeshRefusal(
        CODE_NOT_A_MEMBER,
        f"{state['sentence']}; remote compute only sends work to a network this "
        "device is an active member of",
    )
