"""Exact-port TCP relay: the ONLY route out of a paid coding trial's agent phase.

WHY (egress.py): the task container sits on an ``internal`` Docker network with
no default route. This relay runs in a separate gate container attached to that
network AND to an uplink, and forwards a fixed set of container ports, each to
one fixed host-side port (the run's meter, LoP edge or Codex bridge on
``host.docker.internal``). Nothing else is listened on, so a tool that ignores
proxy settings, dials an IP literal or resolves a public name gets no route.

CONSTRAINTS:
* stdlib only: it runs under the frozen toolchain interpreter mounted read-only
  at /opt/bench, inside the already-pulled task image (no extra image pull).
* The port map comes from argv as ``<listen>=<host_port>`` pairs and is
  validated; the upstream host is fixed, never taken from the client.
* Opaque byte relay: TLS (LoP's CONNECT tunnel to the edge) and HTTP pass
  through unchanged, so metering stays entirely on the host.
* Prints one ``ready`` line after every listener is bound (the gate's
  healthcheck reads /tmp/gate.ready).
* ``--package-proxy <port> --allowlist <json> --log <jsonl>`` (required) also
  serves the agent-phase package proxy (``package_egress.py``, copied beside
  this file): exact allowlisted hosts on 80/443, model hosts always denied,
  one log row per connection. It is the ONLY non-fixed destination the gate
  reaches, and it is deliberately not in the relayed port map.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

UPSTREAM_HOST = "host.docker.internal"
READY = Path("/tmp/gate.ready")


def parse_ports(args: list[str]) -> dict[int, int]:
    ports: dict[int, int] = {}
    for arg in args:
        listen, sep, host = arg.partition("=")
        if not sep or not listen.isdigit() or not host.isdigit():
            raise ValueError("port map entries must be <listen>=<host_port>")
        lp, hp = int(listen), int(host)
        if not (0 < lp < 65536 and 0 < hp < 65536) or lp in ports:
            raise ValueError("ports must be unique integers in 1..65535")
        ports[lp] = hp
    if not ports:
        raise ValueError("at least one port mapping is required")
    return ports


_OPTIONS = ("--package-proxy", "--allowlist", "--log")


def parse_args(args: list[str]) -> tuple[dict[str, object], dict[int, int]]:
    """``(options, port map)``; the package-proxy options are all required.

    Required rather than optional so a gate can never come up without the
    package route for one harness and with it for another (equal treatment).
    """
    opts: dict[str, object] = {}
    rest = list(args)
    while rest and rest[0].startswith("--"):
        name = rest.pop(0)
        if name not in _OPTIONS or name.lstrip("-").replace("-", "_") in opts or not rest:
            raise ValueError(f"unknown, repeated or valueless option: {name}")
        opts[name.lstrip("-").replace("-", "_")] = rest.pop(0)
    if set(opts) != {"package_proxy", "allowlist", "log"}:
        raise ValueError("--package-proxy, --allowlist and --log are required")
    ports = parse_ports(rest)
    proxy = str(opts["package_proxy"])
    if not proxy.isdigit() or not 0 < int(proxy) < 65536 or int(proxy) in ports:
        raise ValueError("package proxy port must be a free integer in 1..65535")
    opts["package_proxy"] = int(proxy)
    return opts, ports


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    # EOF is forwarded as a half-close, not a close: the other direction may
    # still be carrying the response (closing here dropped it in tests).
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError, asyncio.CancelledError):
        pass


def _handler(host_port: int):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_connection(UPSTREAM_HOST, host_port)
        except OSError:
            writer.close()
            return
        try:
            await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
        finally:
            for w in (up_writer, writer):
                w.close()

    return handle


async def main(argv: list[str]) -> None:
    opts, ports = parse_args(argv)
    # Imported here: package_egress.py sits beside this file in the gate's
    # support mount; the port-map tests import this module without it.
    import package_egress

    proxy = package_egress.PackageProxy(
        package_egress.Policy.load(str(opts["allowlist"])),
        package_egress.ConnectionLog(str(opts["log"]), via="gate"),
    )
    servers = [
        await asyncio.start_server(_handler(hp), "0.0.0.0", lp) for lp, hp in ports.items()
    ]
    servers.append(await asyncio.start_server(proxy.handle, "0.0.0.0", opts["package_proxy"]))
    READY.write_text("ready\n")
    print("ready " + " ".join(f"{lp}->{hp}" for lp, hp in ports.items())
          + f" package-proxy:{opts['package_proxy']}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
