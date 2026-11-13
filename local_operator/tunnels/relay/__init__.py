"""Built-in connector for Radient's AWS tunnel relay (``tunnel-relay``).

The relay replaces cloudflared: the device dials OUT over WSS, one WebSocket
carries one yamux session, and the relay opens one yamux stream per public
request. Each stream is a raw byte pipe to this device's loopback gateway, so
nothing here parses HTTP — the gateway already verifies the relay's origin
assertion exactly as it verified the Cloudflare Worker's.

The wire protocol is frozen in agent-server ``docs/TUNNEL_RELAY.md`` (C1, C2);
the golden transcripts it is tested against are vendored under
``tests/fixtures/tunnel_relay/`` from agent-server
``internal/relay/testdata`` at a8351fc4 (see its PROVENANCE.md).

- :mod:`.yamux` — the yamux v0 server half (the device never opens a request
  stream; it accepts the relay's).
- :mod:`.connector` — the WebSocket session, the control stream, make-before-
  break rotation, and the per-stream gateway pipe.
"""
