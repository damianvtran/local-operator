# Tunnel relay golden fixtures

These files are part of the frozen contract in [`docs/TUNNEL_RELAY.md`](../../../docs/TUNNEL_RELAY.md). The Go
relay (`internal/relay/...`) and the Python connector (local-operator `local_operator/tunnels/relay/`) both test
against them, which is how two implementations written in parallel are shown to interoperate before an E2E
exists. Changing a byte here changes the contract: update the doc in the same PR and tell every consumer.

No file here holds a real credential. HMAC keys are labelled `NOT-A-REAL-SECRET`, the assertion signing key was
ephemeral and is not committed, and token values are placeholders.

| Path | Contract | Consumers | How to use |
|---|---|---|---|
| `yamux/0[1-4]_*.json` | C1.2–C1.5 | relay (A2), connector (C1) | `deterministic: true`: replay one side's frames and assert the other side's bytes **exactly** (`header_hex` + `payload_hex`). |
| `yamux/05_flow_control_window_update.json` | C1.2 flow control | relay, connector | `deterministic: false`: assert the listed `invariants`, never the frame list. Payloads > 4096 bytes are recorded as `payload_sha256`. |
| `yamux/recorder/` | C1.7 | whoever regenerates | Its own Go module (testdata is ignored by `go build ./...`). `cd yamux/recorder && GOFLAGS=-p=2 go run . <outdir>`, then diff against the committed files. |
| `control/messages.json` | C1.4 | relay, connector | `valid[].line` is the exact bytes an emitter MUST produce; `invalid[]` names the required reaction. |
| `connect/*.json` | C2 | B1, connector | Shape of `/connect`, `RunTunnelConfig` and the connector JWT. |
| `registry/*.lua` | C3 | relay, B1 (`revoke.lua`) | Load verbatim; never reformat. `values.json` has key and value examples. |
| `hmac/control_auth.json` | C4.1 | relay (sign), agent-server (verify) | Recompute `s=` from `key_utf8` + `canonical` and assert the header bytes. |
| `hmac/peer_auth.json` | C4.2 | relay | Same, for pod-to-pod. |
| `assertion/origin_assertion.json` | C5 | relay (A1) | Build the assertion from `inputs`; assert `protected_header_json`, `payload_json` and `signing_input` byte-for-byte. |

Provenance:
- yamux transcripts: real `hashicorp/yamux v0.1.2` on both ends over `net.Pipe`, recorded by `yamux/recorder`.
  The RST in `03` is hand-encoded, because that library has no public stream reset. The real relay session read
  it as `yamux.ErrConnectionReset`.
- Assertion: `jose 6.2.11`, the Worker's pinned version, running the exact `SignJWT` chain of
  `edge/tunnel-worker/src/index.ts:371-378` with `iat` and `jti` fixed.
- Lua: executed against `redis-server` 8.x locally. The register → refresh(1) → revoked refresh(-1) →
  supersede refresh(0) → restore(2) → unregister(1) and revoke-monotonic paths returned the documented values.
