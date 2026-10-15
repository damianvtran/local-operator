"""``lop acp``: Local Operator as an Agent Client Protocol (ACP) stdio agent.

An ACP client (an editor, or Qwen Audio Agent's Gateway) spawns ``lop acp`` and
speaks newline-delimited JSON-RPC on its stdin/stdout. Each ACP session is one
in-process lop session opened through :func:`local_operator.sdk.open_session`;
nothing here is a second agent loop. The decision record is
``docs/design/acp-agent.md``.

Module map:

* ``stdio``   — :func:`~local_operator.acp_agent.stdio.claim_stdio`, run before
  anything else so tool children and stray prints cannot touch the protocol;
* ``mapping`` — pure functions from lop events to ACP ``session/update``s;
* ``agent``   — :class:`~local_operator.acp_agent.agent.LopAcpAgent`, the ACP
  method handlers and the approval bridge;
* ``main``    — :func:`~local_operator.acp_agent.main.run`, the CLI entry.

The ``agent-client-protocol`` package is the optional ``acp`` extra, so only
``stdio`` (standard library only) may be imported without it.
"""
