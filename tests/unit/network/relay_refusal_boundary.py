"""Opt-in causal boundary for the stale-link incident receipt cell, not a relay fake.

A welcome is sent before the listener registers its link. Pair completion and
``contact_peers`` return therefore cannot witness the remote handler's completion.
Track actual handlers and successful dial/receiver identities, then fence further
dials before disposing of setup links. Other network cells keep normal redial.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from local_operator.network import relay


class RefusalBoundary:
    def __init__(
        self, servers: tuple[relay.RelayServer, relay.RelayServer], patch: pytest.MonkeyPatch
    ) -> None:
        self.servers = servers
        self.condition = threading.Condition()
        self.handlers: list[dict[str, Any]] = []
        self.dials: list[dict[str, Any]] = []
        self.admissions: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.pair_threads: set[threading.Thread] = set()
        self.fenced = False
        self.permit: tuple[Any, ...] | None = None
        for server in servers:
            self._install(server, patch)

    def _install(self, server: relay.RelayServer, patch: pytest.MonkeyPatch) -> None:
        inbound, admit, dial = server._handshake_inbound, server._admit_link, server.dial
        pair_listener = server._run_pair_listener

        def pairing(*args: Any, **kwargs: Any) -> Any:
            # Pair admission changes phase to member before _admit_link; phase
            # cannot identify the handler that still owes contact_peers work.
            with self.condition:
                self.pair_threads.add(threading.current_thread())
                self.condition.notify_all()
            return pair_listener(*args, **kwargs)

        def handshake(sock: Any, addr: Any) -> None:
            entry = dict(
                server=server,
                thread=threading.current_thread(),
                done=False,
                endpoints=(sock.getsockname(), sock.getpeername()),
            )
            with self.condition:
                self.handlers.append(entry)
                self.condition.notify_all()
            try:
                inbound(sock, addr)
            except BaseException as exc:
                with self.condition:
                    self.errors.append(f"inbound {entry['endpoints']}: {exc!r}")
                raise
            finally:
                with self.condition:
                    entry["done"] = True
                    self.condition.notify_all()

        def admission(link: relay.PeerLink) -> str:
            endpoints = (link.sock.getsockname(), link.sock.getpeername())
            verdict = admit(link)
            with self.condition:
                self.admissions.append(
                    dict(
                        link=link,
                        server=server,
                        thread=threading.current_thread(),
                        verdict=verdict,
                        endpoints=endpoints,
                    )
                )
                self.condition.notify_all()
            return verdict

        def outgoing(network_id: str, **kwargs: Any) -> Any:
            key = (
                server,
                threading.current_thread(),
                network_id,
                kwargs.get("host"),
                kwargs.get("epoch"),
            )
            with self.condition:
                if self.fenced:
                    if self.permit != key:
                        evidence = f"unexpected post-fence dial: {key!r}"
                        # Some callers deliberately swallow dial failures. The ledger
                        # must survive that catch and fail the owning test anyway.
                        self.errors.append(evidence)
                        raise AssertionError(evidence)
                    self.permit = None
                entry = dict(
                    server=server, thread=threading.current_thread(), key=key, done=False, link=None
                )
                self.dials.append(entry)
                self.condition.notify_all()
            try:
                result = dial(network_id, **kwargs)
                with self.condition:
                    entry["link"] = result[0]
                    if result[0] is None:
                        self.errors.append(f"uncorrelatable failed dial: {key!r}: {result[1]}")
                return result
            except BaseException as exc:
                with self.condition:
                    self.errors.append(f"dial {key!r}: {exc!r}")
                raise
            finally:
                with self.condition:
                    entry["done"] = True
                    self.condition.notify_all()

        # Install before start, not after fixture yield: otherwise a redial may
        # already be in flight when the test thinks it has gated new producers.
        patch.setattr(server, "_redial_loop", lambda: server._stop.wait())
        patch.setattr(server, "_run_pair_listener", pairing)
        patch.setattr(server, "_handshake_inbound", handshake)
        patch.setattr(server, "_admit_link", admission)
        patch.setattr(server, "dial", outgoing)

    def finish(self) -> None:
        """After fixture stop, reap exact tracked handlers even on assertion failure."""
        deadline = time.monotonic() + 60
        with self.condition:
            threads = [handler["thread"] for handler in self.handlers]
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
            assert not thread.is_alive(), f"fixture handler survived stop: {thread!r}"
        self.assert_clean()

    def assert_clean(self) -> None:
        with self.condition:
            assert not self.errors, self.errors

    def _receiver(self, outgoing: dict[str, Any]) -> dict[str, Any] | None:
        link = outgoing["link"]
        if link is None:
            return None
        sender = next(a for a in self.admissions if a["link"] is link)
        expected = tuple(reversed(sender["endpoints"]))
        for admission in self.admissions:
            peer = admission["link"]
            if (
                not admission["verdict"]
                and admission["server"] is not outgoing["server"]
                and peer.link_id == link.link_id
                and peer.device_id == outgoing["server"].identity.device_id
                and link.device_id == admission["server"].identity.device_id
            ):
                for handler in self.handlers:
                    if (
                        handler["thread"] is admission["thread"]
                        and handler["server"] is admission["server"]
                        and handler["endpoints"] == expected
                    ):
                        return handler
        return None

    def settle(self, *, fence: bool = False, timeout: float = 60) -> None:
        """Join the causal closure, not a momentarily empty active counter.

        Every successful outbound requires a positive receiving-handler match.
        Failed dials are fail-closed: we cannot prove their remote accept lifetime.
        A fence is installed atomically with the closure check, before new sockets
        can be opened. Already-entered dials are in the ledger under this same lock.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self.condition:
                self.assert_clean()
                pair_handlers = self.pair_threads
                correlated = all(d["done"] and self._receiver(d) for d in self.dials)
                complete = (
                    bool(pair_handlers)
                    and correlated
                    and all(h["done"] for h in self.handlers)
                    and pair_handlers <= {h["thread"] for h in self.handlers}
                )
                threads = [h["thread"] for h in self.handlers]
                if complete and fence:
                    self.fenced = True
                if not complete:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0, (
                        f"causal setup incomplete: handlers={self.handlers!r}; "
                        f"dials={self.dials!r}; admissions={self.admissions!r}"
                    )
                    self.condition.wait(min(remaining, 0.1))
                    continue
            # ``done`` is set in finally; join also witnesses the actual thread exit.
            for thread in threads:
                thread.join(max(0, deadline - time.monotonic()))
                assert not thread.is_alive(), f"inbound handler survived: {thread!r}"
            with self.condition:
                # A handler can synchronously create descendants. Recheck all
                # identities after joining, not just the original thread snapshot.
                if threads == [h["thread"] for h in self.handlers] and all(
                    d["done"] and self._receiver(d) for d in self.dials
                ):
                    self.assert_clean()
                    return

    def dispose_setup(self, network_id: str) -> None:
        self.settle(fence=True)
        links = {id(a["link"]): a["link"] for a in self.admissions}
        links.update({id(d["link"]): d["link"] for d in self.dials if d["link"]})
        for link in links.values():
            link.close("fixture-setup-complete")
        deadline = time.monotonic() + 60
        for link in links.values():
            for thread in (link._reader, link._writer):
                if thread is not None:
                    thread.join(max(0, deadline - time.monotonic()))
                    assert not thread.is_alive(), f"setup link thread survived: {thread!r}"
        for server in self.servers:
            assert not [item for item in server.links.values() if item.network_id == network_id]
        self.assert_clean()

    def arm(
        self,
        server: relay.RelayServer,
        network_id: str,
        host: str,
        *,
        thread: threading.Thread | None = None,
    ) -> None:
        with self.condition:
            assert self.fenced and self.permit is None
            self.permit = (server, thread or threading.current_thread(), network_id, host, 1)

    def selected_pair(self, link: relay.PeerLink) -> relay.PeerLink:
        self.settle()
        peer = self.servers[0] if link.server is self.servers[1] else self.servers[1]
        receiving = peer.links[link.link_id]
        for server, expected in ((link.server, link), (peer, receiving)):
            live = [
                item
                for item in server.links.values()
                if item.network_id == link.network_id and item.alive
            ]
            assert live == [expected], [(item.link_id, item.phase) for item in live]
            assert not expected.closing and expected.phase == "reconcile"
            assert expected.link_id == link.link_id and expected.epoch == link.epoch
            other = peer if server is link.server else link.server
            assert expected.device_id == other.identity.device_id
        self.assert_clean()
        return receiving
