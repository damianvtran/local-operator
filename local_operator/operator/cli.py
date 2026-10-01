"""``lop operator`` — argument registration for operator authority (design §2.5).

Split from the verbs for the same reason ``lop secret`` is: ``cli.py`` imports
this module on EVERY ``lop`` invocation to build the parser, so nothing here may
reach for ``cryptography`` or the OS keychain. The verbs live in
:mod:`local_operator.operator.handlers`, imported only once an ``operator`` verb
has actually been dispatched.

The verb set is ``init | trust | install | anchor export | setup | sign | status``:

* ``init``    — create the operator's private key in the best store this host
  offers and stage the anchor; it reports the LEVEL it achieved rather than
  assuming the ladder reached the top;
* ``install`` — the ONE privileged step: move the staged anchor to the
  root-owned path the runtime reads (``--from`` installs a statement received
  from another machine, verified as canonical first);
* ``anchor export`` — emit the PUBLIC statement for transfer to a peer that will
  hold it as a verify-only root;
* ``setup``   — the agent-runnable self-install for THIS machine (§3.7):
  init → consent → install → verify, with the receipts that name each;
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
    # INSTALL FROM A TRANSFERRED STATEMENT (remote onboarding §3.3 step 7, F4b):
    # the public anchor statement arrives from the operator's own machine, and
    # the bytes that land must be exactly the bytes whose digest the approval
    # minted — so this path refuses a file that is not the canonical form rather
    # than re-serialising it into something "equivalent".
    install_parser.add_argument(
        "--from",
        dest="from_file",
        default="",
        metavar="PATH",
        help=(
            "Install a public anchor statement RECEIVED from the operator's machine "
            "(the file `lop operator anchor export` writes); verified as canonical "
            "before the privileged step"
        ),
    )

    anchor_parser = actions.add_parser(
        "anchor", help="The PUBLIC anchor statement, for transfer to another machine"
    )
    anchor_actions = anchor_parser.add_subparsers(dest="anchor_command")
    anchor_export = anchor_actions.add_parser(
        "export", help="Write the public anchor statement for transfer"
    )
    anchor_export.add_argument(
        "--file",
        default="",
        metavar="PATH",
        help="Write the statement to this path (public data; mode 0644)",
    )
    anchor_export.add_argument("--json", action="store_true")

    setup_parser = actions.add_parser(
        "setup",
        help=(
            "Set up operator authority on THIS machine end to end: create the key, "
            "raise the one admin gesture, install the anchor, verify"
        ),
    )
    setup_parser.add_argument("--json", action="store_true")
    setup_parser.add_argument(
        "--sudo-secret",
        default="",
        metavar="NAME",
        help=(
            "A secret-store name holding this user's admin password, used ONCE for "
            "the privileged step when sudo cannot prompt (never printed or logged)"
        ),
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
    sign_parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        metavar="SECONDS",
        help=(
            "How long the OS presence prompt may take (default 180). On expiry the key "
            "agent is killed and nothing is signed"
        ),
    )


def main(args: argparse.Namespace) -> int:
    """Entry point for ``lop operator``; see :mod:`local_operator.operator.handlers`.

    A deliberately thin shim, function-local for the same reason ``lop secret``'s
    is: the handlers pull in the keychain/``cryptography`` stack, and that cost
    belongs to the run that uses a key rather than to ``lop --version``.
    """
    from local_operator.operator.handlers import dispatch

    return dispatch(args)
