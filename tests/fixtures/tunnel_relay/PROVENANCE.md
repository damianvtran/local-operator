# Vendored tunnel-relay contract fixtures

Copied byte-for-byte from radient-ml/agent-server `internal/relay/testdata/` at
commit `a8351fc4a55e9b47cf44a697ef01915570c66766` (PR #361, the frozen contract
`docs/TUNNEL_RELAY.md`). Only the files the Python connector consumes (C1, C2)
are vendored; the Lua scripts, HMAC vectors and assertion fixture belong to the
relay and the control plane, and `README.md` here is upstream's own index.

Do not edit these files here. When the contract changes, re-copy them from the
new agent-server commit, update the SHA above, and re-run
`tests/unit/test_tunnel_relay.py`.
