"""``lop operator`` — argument registration for operator authority (design §2.5).

Split from the verbs for the same reason ``lop secret`` is: ``cli.py`` imports
this module on EVERY ``lop`` invocation to build the parser, so nothing here may
reach for ``cryptography`` or the OS keychain. The verbs live in
:mod:`local_operator.operator.handlers`, imported only once an ``operator`` verb
has actually been dispatched.

The verb set is ``init | trust | install | sign | status``:

* ``init``    — create the operator's private key in the best store this host
  offers and stage the anchor; it reports the LEVEL it achieved rather than
  assuming the ladder reached the top;
* ``install`` — the ONE privileged step: move the staged anchor to the
  root-owned path the runtime reads;
* ``trust``   — show whether the installed anchor is trusted (root-owned, ours)
  and exit non-zero when it is not;
* ``sign``    — the one signing entry point every surface uses, on the wire
  shape the runtime verifies;
* ``status``  — the level, with the reasons and the residual.
"""

from __future__ import annotations

import argparse
from typing import Any

#: The backends ``--backend`` accepts. ``auto`` is the presence ladder; the rest
#: are explicit so an operator who wants the file fallback (a CI box, a
#: container) can say so and be told what it costs.
BACKEND_CHOICES = ("auto", "secure-enclave", "cng-presence", "file-only")


def add_parser(subparsers: Any) -> None:
    """Register ``lop operator`` and its verbs. Stdlib-only."""
    parser = subparsers.add_parser(
        "operator",
        help="Operator authority: the key that may loosen a running approval gate",
    )
    actions = parser.add_subparsers(dest="operator_command")

    init_parser = actions.add_parser(
        "init",
        help="Create the operator key in this host's best store and stage the anchor",
    )
    init_parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    init_parser.add_argument("--label", default="", help="A label stored in the anchor")

    install_parser = actions.add_parser(
        "install",
        help="Install the staged anchor as a root-owned file (needs sudo/admin ONCE)",
    )
    install_parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the privileged command instead of running it",
    )

    actions.add_parser("trust", help="Show whether the installed anchor is trusted")
    actions.add_parser("status", help="Report the operator authority level and why")

    devices_parser = actions.add_parser(
        "devices",
        help="List paired phones, pending pairing requests, and revoke/authorise a device",
    )
    devices_parser.add_argument(
        "--revoke",
        default="",
        metavar="DEVICE_ID",
        help=(
            "Add this device to the anchor's revocation list (needs the same ONE "
            "privileged step as installing the anchor)"
        ),
    )
    devices_parser.add_argument(
        "--authorise",
        default="",
        metavar="DEVICE_ID",
        help=(
            "Lift a revocation for this device on this machine: clears the local "
            "record and the anchor's entry (needs the same ONE privileged step as "
            "--revoke). Host-side only — a phone cannot un-revoke itself"
        ),
    )
    devices_parser.add_argument(
        "--print-only",
        action="store_true",
        help=(
            "With --revoke or --authorise, print the privileged command instead of " "running it"
        ),
    )

    sign_parser = actions.add_parser(
        "sign",
        help="Sign a runtime's challenge with the operator key (raises the OS prompt)",
    )
    sign_parser.add_argument("--challenge", required=True, help="The challenge, hex")
    sign_parser.add_argument(
        "--purpose",
        required=True,
        choices=("loosen", "approve"),
        help="Which authority-increasing action this signature is for",
    )
    sign_parser.add_argument("--session", default="", help="The session id to bind to")
    sign_parser.add_argument("--request-id", default="", help="The card id, for approve")


def main(args: argparse.Namespace) -> int:
    """Entry point for ``lop operator``; see :mod:`local_operator.operator.handlers`.

    A deliberately thin shim, function-local for the same reason ``lop secret``'s
    is: the handlers pull in the keychain/``cryptography`` stack, and that cost
    belongs to the run that uses a key rather than to ``lop --version``.
    """
    from local_operator.operator.handlers import dispatch

    return dispatch(args)
