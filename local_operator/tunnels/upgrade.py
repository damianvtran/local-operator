"""One locally authored recovery contract for unsupported tunnel clients.

HTTP 426 is a protocol/bootstrap prerequisite, not a retryable authorization
failure. Keep copy keyed and resolved at the display boundary; provider bodies
must never become an exception, status record or setup receipt.
"""

from local_operator.i18n.keys import tunnel_upgrade
from local_operator.i18n.text import tr

REASON = "client_upgrade_required"
KIND = "update_client"
OVERRIDE = "tunnel.transport: cloudflare"


def label() -> str:
    return tr("tunnel.upgrade")(tunnel_upgrade.client_label())


def detail(surface: str = "cli") -> str:
    return tr("tunnel.upgrade")(
        tunnel_upgrade.client_guidance(
            update_command="/update" if surface == "tui" else "lop update",
            override=OVERRIDE,
            reconnect_command="/mobile enable" if surface == "tui" else "lop tunnel connect",
        )
    )


def plain() -> str:
    return tr("tunnel.upgrade")(tunnel_upgrade.client_plain())
