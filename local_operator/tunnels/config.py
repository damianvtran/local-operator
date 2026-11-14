"""Private connector configuration; cloud records contain no origin secrets."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir

# The browser bridge already owns 4099 and the mobile relay owns 4098. Keep
# this separate listener available when both existing local services are up.
DEFAULT_GATEWAY_PORT = 4100
DEFAULT_API_URL = "https://api.radienthq.com"
HARNESS_PORTS = {"local-operator": 4098, "opencode": 4096}
ORIGIN_ISSUER = "https://tunnels.radienthq.com"
_HOST = re.compile(r"[a-z0-9]+-(?:lop|oc)\.radienthq\.com\Z")


def directory(base: Path | None = None) -> Path:
    """This store's directory.

    ``base`` exists for ONE caller: the LaunchAgent repair renders the unit the
    plist on disk already names, which is the store recorded in that plist
    rather than whatever this process's environment happens to point at — a
    repair brings a unit up to date IN PLACE, it does not migrate it. Every
    other caller keeps reading the ambient config dir.
    """
    return (base if base is not None else config_dir()) / "tunnel"


def private_write(path: Path, value: str, *, exclusive: bool = False) -> bool:
    """Publish a complete private file; exclusive intents elect one CLI writer.

    Linking the fully fsynced temporary file is an atomic create-if-absent.
    An O_EXCL write directly to the destination would let a second process
    observe an empty or partially written intent before the first flushes it.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            os.replace(temporary, path)
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(value: dict[str, Any]) -> None:
    private_write(directory() / "config.json", json.dumps(value, indent=2) + "\n")


def load() -> dict[str, Any]:
    path = directory() / "config.json"
    if not path.exists():
        # `lop login radient`, not `/login radient`: this sentence is printed by a
        # SHELL surface (`lop tunnel status`, and every command that reads the
        # store through here), where `/login` is not a program — the shell answers
        # it with `no such file or directory`. Review round 2's m5 is this string
        # (and the two beside it in `api.py`/`cli.py`); the rule it follows is the
        # one the park's own sentence states at `gateway.TERMINAL_DETAIL`: a
        # surface may only name a command ITS reader can run.
        raise ValueError("No tunnel configured. Run lop tunnel create after lop login radient.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Tunnel configuration must be an object.")
    return value


def port(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1024 <= value <= 65535:
        raise ValueError("Tunnel ports must be integers between 1024 and 65535.")
    return value


def validate_connection(result: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before forwarding: only the documented Radient names.

    Configurable API endpoints would turn an OAuth bearer into an arbitrary
    HTTP-client credential. The public API and origin host suffix stay pinned;
    tests inject transports, never weaken those production trust boundaries.
    """
    access = result.get("origin_auth")
    tunnel = result.get("tunnel")
    if not isinstance(access, dict) or not isinstance(tunnel, dict):
        raise ValueError("Radient returned an incomplete tunnel connection.")
    if access.get("issuer") != ORIGIN_ISSUER:
        raise ValueError("Invalid Radient origin-proof issuer.")
    jwks = access.get("jwks")
    if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list) or not jwks["keys"]:
        raise ValueError("Missing pinned Radient origin-proof public keys.")
    for key in jwks["keys"]:
        if (
            not isinstance(key, dict)
            or key.get("kty") != "RSA"
            or key.get("alg", "RS256") != "RS256"
            or not isinstance(key.get("kid"), str)
            or not key["kid"]
            or "d" in key
        ):
            raise ValueError("Invalid Radient origin-proof public key.")
    if not isinstance(access.get("owner_account_id"), str) or not access["owner_account_id"]:
        raise ValueError("Missing tunnel owner.")
    if access.get("tunnel_id") != tunnel.get("id") or not isinstance(tunnel.get("id"), str):
        raise ValueError("Origin proof must name this tunnel.")
    version = access.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
        or version != tunnel.get("version")
    ):
        raise ValueError("Origin proof must name this tunnel version.")
    gateway = port(result.get("gateway_port", tunnel.get("gateway_port")))
    harnesses = tunnel.get("harnesses")
    if not isinstance(harnesses, list) or not harnesses:
        raise ValueError("Tunnel must declare its harnesses.")
    hosts: set[str] = set()
    ids: set[str] = set()
    for harness in harnesses:
        if not isinstance(harness, dict) or harness.get("id") not in HARNESS_PORTS:
            raise ValueError("Unknown tunnel harness.")
        if harness["id"] in ids:
            raise ValueError("Duplicate tunnel harness.")
        ids.add(harness["id"])
        if not isinstance(harness.get("enabled"), bool):
            raise ValueError("Harness enabled must be a boolean.")
        if port(harness.get("port")) == gateway:
            raise ValueError("A harness cannot proxy back to the tunnel gateway.")
        hostname = harness.get("hostname")
        if not isinstance(hostname, str) or not _HOST.fullmatch(hostname) or hostname in hosts:
            raise ValueError("Invalid or duplicate tunnel hostname.")
        hosts.add(hostname)
    return result


#: The operator's kill switch: `tunnel.transport: cloudflare` in config.yml keeps
#: this device on the cloudflared-only path even when `/connect` offers the AWS
#: relay. Read from config.yml rather than the tunnel's own config.json because
#: `lop tunnel connect` rewrites config.json wholesale, and a kill switch that a
#: routine re-enrolment silently drops is not one.
TRANSPORT_KEY = "tunnel.transport"
TRANSPORT_CLOUDFLARE = "cloudflare"
TRANSPORT_RELAY = "relay"


def transport_override() -> str | None:
    """``"cloudflare"`` when the operator pinned cloudflared, else ``None`` (auto).

    Read-only through `read_config_values`, never a `ConfigManager`: building a
    manager writes a default config.yml when none exists, and the tunnel unit
    must not create the operator's config as a side effect of starting. Both the
    flat dotted key (how the settings writer stores nested keys) and a nested
    ``tunnel: {transport: ...}`` mapping are honoured, so a hand edit in either
    shape works. Any other value is ignored: an unknown word must not be able to
    switch a device OFF the path that is serving it today.
    """
    from local_operator.config import read_config_values

    values = read_config_values(config_dir()) or {}
    chosen = values.get(TRANSPORT_KEY)
    nested = values.get("tunnel")
    if chosen is None and isinstance(nested, dict):
        chosen = nested.get("transport")
    if isinstance(chosen, str) and chosen.strip().lower() == TRANSPORT_CLOUDFLARE:
        return TRANSPORT_CLOUDFLARE
    return None


def _relay_marker() -> Path:
    return directory() / "relay-offered"


def relay_offered() -> bool:
    """Whether the last `/connect` this device ran offered a usable relay.

    Setup, `lop tunnel install` and the desktop setup route decide whether
    cloudflared is a hard prerequisite WITHOUT calling `/connect` themselves (it
    mints credentials). The tunnel service is the one caller that does, so it
    leaves this marker for them; the kill switch outranks it.
    """
    return transport_override() is None and _relay_marker().exists()


def record_relay_offer(offered: bool) -> None:
    marker = _relay_marker()
    if offered:
        private_write(marker, "1\n")
    else:
        marker.unlink(missing_ok=True)
