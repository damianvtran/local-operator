"""The session's record of what MCP discovery actually did.

Why this exists: the composition root used to report MCP failures with a
single ``print(..., file=sys.stderr)``. Under a full-screen Textual app that
line is either swallowed by the alternate screen buffer or painted straight
over the composed frame — so the one thing a user needs when their tools are
missing ("server X did not start, and here is why") was the one thing they
could not read. The outcome is recorded on the session as data instead, and
each front end decides how to surface it: the TUI raises a toast plus a
transcript notice and keeps a live count in the status band, while the
headless callers keep the stderr line they already had.

Deliberately NOT defined inside ``local_operator.mcp``: importing that
package pulls the manager and, through it, the optional MCP SDK. This record
has to be constructible on exactly the path where that import FAILED, which a
type living inside the package it describes cannot be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The failure key that means "the MCP layer itself", not a server. Discovery
#: raising, and a machine whose configured servers all failed for one setup
#: reason, are both recorded under this one key — a front end that renders
#: ``MCP {name} failed`` must never name a server that does not exist, and three
#: spellings of "not a server" (``discovery``, ``.mcp.json``, an install hint
#: repeated per server) would each need their own special case.
MCP_DISCOVERY_KEY = "discovery"


@dataclass(frozen=True)
class McpStartupOutcome:
    """What one discovery round connected, what it failed on, and why.

    ``configured`` is the full set of servers the config files asked for, so
    it is the only field that answers "does this machine use MCP at all" — the
    question the status band's visibility hangs on. It is NOT
    ``connected + failures``: the startup gate settles fast connects and
    leaves the rest connecting in the background, so a server can legitimately
    be in neither list at the moment this snapshot is taken.

    ``failures`` maps server name to the error text, because "2 servers
    failed" is not actionable and "github: command not found: gh" is. The
    mapping is what the durable transcript notice and ``/mcp`` render from.

    Frozen: this is a BOOT SNAPSHOT that the toast and the notice quote after
    the fact. Live state (a server that dropped ten minutes in) is read from
    the manager, which is the only thing that actually knows it.
    """

    configured: tuple[str, ...] = ()
    connected: tuple[str, ...] = ()
    failures: dict[str, str] = field(default_factory=dict)
    tool_count: int = 0
    #: True while servers deferred past the 250 ms startup gate are still
    #: connecting, so this outcome is a PROVISIONAL snapshot, not the final
    #: tally. OAuth HTTP servers do metadata discovery and a possible token
    #: refresh before their transport opens, so they routinely miss the gate;
    #: reporting failures from a settling snapshot names a server as failed
    #: while it is in fact mid-handshake and about to succeed. Front ends
    #: suppress the failure surface while this is True and re-report once the
    #: manager fires ``on_startup_settled`` with a settled (``settling=False``)
    #: outcome. False when nothing was deferred — the gate snapshot was final.
    settling: bool = False
    #: The SUBSET of ``failures`` whose cause was the transport rather than the
    #: server or the config. Carried as its own field rather than folded into
    #: the failure text so a front end can group on the FACT instead of parsing
    #: prose: when every failure is connectivity, the toast says ``(network)``
    #: once instead of naming nine servers, which is the operator-requested
    #: distinction between "your link is down" and "nine MCP servers are
    #: broken". Empty on every manager that does not report it, which renders
    #: exactly the pre-change copy.
    network_failures: frozenset[str] = frozenset()

    @property
    def failed(self) -> bool:
        """True when at least one configured server did not come up."""
        return bool(self.failures)

    @property
    def all_failures_are_network(self) -> bool:
        """True when EVERY reported failure was a transport failure.

        The question the toast's footer asks before it replaces its list of
        bare server names with ``(network)``: when the failures are all
        connectivity, naming them adds nothing the head line's count does not
        already say, while the fact that they share one cause is the only part
        the user can act on (tether, wait, change network). False for an empty
        failure map — there is nothing to group — so a clean boot is never
        labelled.

        ``network_failures`` is a subset of ``failures``' keys by construction
        (the manager writes both through one recording method), so the superset
        test is the whole check. The ``discovery`` key is never counted a
        network failure: it means the CONFIG layer failed, which no host
        explains, and calling that connectivity would send the user to
        diagnose a link that is fine.
        """
        return bool(self.failures) and self.network_failures.issuperset(self.failures)

    @property
    def reportable(self) -> bool:
        """Whether this outcome is worth interrupting the user with.

        A machine with no ``.mcp.json`` produces an empty outcome and must
        stay silent — the whole feature has to be invisible when unused.
        Servers that are still connecting past the startup gate are silent
        too: they are in ``configured`` but in neither result list, and
        announcing "0 connected" while a connect is still in flight would be
        both alarming and wrong. The status band still shows the count in that
        window, and it ticks up when the connect lands.

        A SETTLING snapshot is never reportable: while servers deferred past the
        gate are still connecting, any failure in this snapshot may belong to a
        server that is in fact mid-handshake, and any success tally is
        incomplete. Reporting it would flash "N of M up — failed: X" a beat
        before the slow OAuth servers land. The manager re-reports a settled
        (``settling=False``) outcome once the round drains, and THAT one is what
        the user sees. The live status band covers the in-between window.
        """
        if self.settling:
            return False
        return bool(self.connected) or bool(self.failures)
