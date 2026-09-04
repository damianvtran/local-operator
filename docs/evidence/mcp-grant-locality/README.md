# `/mcp` grant verbs on a detached runtime

## The defect

`/mcp reauth notion` in a detached session answered:

```
/mcp reauth opens a browser and stores credentials on the machine running the
terminal — run it from a terminal on that machine
```

The user was already on that machine. The advice was also unfollowable: a
detached session ROUTES the verb to its owner, the owner refused it, and there
was no third place to type it. `/mcp login`, `/mcp logout` and `/mcp reauth`
were unreachable for the whole class of sessions that outlive their terminal —
precisely when an expired credential needs refreshing.

`OwnedSessionHandle._mcp_slash` refused all three verbs unconditionally
(`session/runtime/owned.py`, introduced in 0.45.0 / #576), on the theory that a
routed command arrives from somewhere the runtime's browser cannot reach.

## Why the premise was wrong

The runtime's control socket binds loopback only, which `runtime/server.py`
calls "the security invariant of the whole design":

```python
self._server = await asyncio.start_server(
    self._on_connection, host="127.0.0.1", port=0, limit=_MAX_LINE_BYTES
)
```

A client that can dial a runtime is therefore already on the runtime's machine,
and its default browser is the user's browser. The guard fired on exactly the
case it was meant to protect.

## Two further defects found while fixing it

1. **The grant could not have been awaited inline anyway.** An attach client
   abandons a request after `ACK_TIMEOUT_S` (15 s, `mobile/attach_client.py`)
   while a grant is budgeted at 600 s. The runtime's per-connection reader is
   also strictly serial (`readline()` then `await _on_request(...)`), so an
   inline grant parks every other op on that connection — model switches,
   aborts, prompts — behind a browser tab for up to ten minutes. The TUI handle
   had this bug live: it awaited the grant "unbounded" against a client that
   had already given up.

2. **Duplicated grant logic.** `resolve_server` / `login_allowed` / the
   login-worker body existed in `tui/app.py` only, so the runtime could not
   reuse them even had it wanted to.

## The fix

- `local_operator/mcp/grants.py` — the transport-neutral core, following the
  `mcp/verbs.py` precedent. `start_grant` validates synchronously (unknown
  server, ineligible server, bad arity) and returns immediately, then reports
  the settled outcome as a `NoticeEvent`, which the relay already fans out to
  every attached front end.
- `ClientLocality` (`session/runtime/types.py`) — locality becomes a property
  the CLIENT declares in its auth frame, defaulting to `local`. The runtime no
  longer guesses. This is the seam a future mobile relay uses to say `remote`
  and get the refusal, which is kept for the topology it actually describes.
- `tui/app.py` now delegates to the same core, so the attached and detached
  paths cannot diverge.

## Evidence

`e2e_repro.py` stands up a real `RuntimeServer`, dials it over the real
loopback control socket the way an attach terminal does, and sends the exact
`slash_result` frame a typed `/mcp reauth notion` produces.

```
$ .venv/bin/python docs/evidence/mcp-grant-locality/e2e_repro.py
```

See `e2e_after.txt` for the captured run. Summary:

| Case | Result |
| --- | --- |
| Local attach client, `/mcp reauth notion` | Grant runs; `forgot=['notion'] disconnected=['notion'] connected=['notion']`; outcome delivered as a `NoticeEvent` |
| Client declaring `locality: remote` | Refused, and **nothing** touched — no disconnect, no credential deletion |
| `/mcp reauth` (no name) | `usage: /mcp reauth <name>` |
| `/mcp reauth a b` | `takes one server name — got 'a b'` |
| `/mcp bogus x` | `unknown mcp subcommand: bogus` |

The script patches `mcp_logout_server` before running. Without that it deletes
a real credential from the developer's `~/.local-operator/auth.db` as a side
effect of gathering evidence — which happened on the first run of an earlier
draft, and is why the patch is now in the script rather than in the operator's
memory.

`before.txt` captures the refusal reproduced against the released 0.46.0 build.
