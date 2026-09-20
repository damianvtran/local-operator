"""``lop pair`` — argument registration for device pairing (stage D).

WHY A TOP-LEVEL VERB RATHER THAN ``lop operator pair``. ``lop operator`` is the
operator's own key management: creating it, installing the anchor, signing a
challenge that something else minted. Pairing is a different actor's onboarding —
it is the moment a PHONE becomes a signer — and an operator looking for "add my
phone" is looking under pairing, not under the key that happens to sign the
certificate. The verb is a sibling of ``operator`` for the same reason
``secret`` is: one thing, one name.

This module is the STDLIB-ONLY half, like :mod:`local_operator.secrets.cli`:
``cli.py`` imports it on EVERY ``lop`` invocation to build the parser, so
nothing here may reach for ``cryptography``, Security.framework or the keychain.
The verbs live in :mod:`local_operator.operator.pair_handlers`, imported only
once ``pair`` has actually been dispatched.
"""

from __future__ import annotations

import argparse
from typing import Any


def add_parser(subparsers: Any) -> None:
    """Register ``lop pair``. Stdlib-only."""
    parser = subparsers.add_parser(
        "pair",
        help="Pair a phone as an operator device: your signature authorises it, once",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for a device to claim the code (default: 300)",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Mint and print the pairing code, then exit (no waiting)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Approve a claimed code without asking here. The signature still raises "
            "the OS presence prompt on a host that has one; on a file-only host this "
            "grants without a gesture, and `lop operator status` says which host you are on"
        ),
    )
    parser.add_argument(
        "--device",
        default="",
        help="Only approve the pending request naming this device id (default: the first)",
    )


def main(args: argparse.Namespace) -> int:
    """Entry point for ``lop pair``; see :mod:`local_operator.operator.pair_handlers`."""
    from local_operator.operator.pair_handlers import dispatch

    return dispatch(args)
