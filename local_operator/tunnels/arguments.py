"""Argument registration stays stdlib-only on every CLI startup."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("tunnel", help="Radient remote access to your local harnesses")
    actions = parser.add_subparsers(dest="tunnel_command")
    for name in ("create", "connect", "configure"):
        child = actions.add_parser(name)
        if name == "connect":
            child.add_argument("tunnel_id", nargs="?")
            child.add_argument(
                "--no-start",
                action="store_true",
                help="Save configuration without starting the connector",
            )
        child.add_argument("--credential-id", type=int)
        if name != "connect":
            child.add_argument("--name")
            child.add_argument("--gateway-port", type=int)
            child.add_argument("--mobile-port", type=int)
            child.add_argument("--opencode-port", type=int)
            child.add_argument("--no-mobile", action="store_true")
            child.add_argument("--no-opencode", action="store_true")
        if name == "configure":
            enabled = child.add_mutually_exclusive_group()
            enabled.add_argument(
                "--enable", dest="remote_enabled", action="store_true", default=None
            )
            enabled.add_argument("--disable", dest="remote_enabled", action="store_false")
        child.add_argument(
            "--opencode-auth-file",
            type=Path,
            help="Private JSON {username,password}; kept only on this device",
        )
        child.add_argument(
            "--accept-monthly-price",
            help="Accept the exact USD monthly price from lop tunnel billing",
        )
    setup = actions.add_parser("setup", help="Observe mobile setup; only apply makes changes")
    setup.add_argument("setup_action", nargs="?", choices=("plan", "apply"), default="plan")
    setup.add_argument("--credential-id", type=int)
    setup.add_argument("--accept-monthly-price", help="Accept the exact fresh activation quote")
    setup.add_argument("--json", action="store_true")
    billing = actions.add_parser("billing", help="Show the server's current monthly cost and price")
    billing.add_argument("--credential-id", type=int)
    billing.add_argument(
        "--json", action="store_true", help="Fresh account, balance and setup readiness as JSON"
    )
    activate = actions.add_parser(
        "activate", help="Activate tunnel billing at an explicitly accepted price"
    )
    activate.add_argument("--accept-monthly-price", required=True)
    activate.add_argument("--credential-id", type=int)
    for name in (
        "install",
        "start",
        "stop",
        "restart",
        "status",
        "list",
        "serve",
        "revoke",
        "uninstall",
    ):
        child = actions.add_parser(name)
        if name == "list":
            child.add_argument("--credential-id", type=int)
        if name == "status":
            child.add_argument(
                "--json",
                action="store_true",
                help="Machine-readable connector, cloud and login state as JSON",
            )
