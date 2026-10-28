# Desktop Team Bridge owner audit ("{chief of staff}'s messages")

The signed-in human reads, read-only, every conversation **their own agent**
(the designated chief of staff) is or was in: DMs with humans and agents,
groups and channels. That includes messages the agent's screening **rejected**
or never screened. The hub side is agent-server #221
(`docs/TEAM-BRIDGE-CONVERSATIONS.md` §16); this is the runtime's proxy over it.

Capability: `features.team_bridge_owner_audit >= 1`. A renderer that does not
see it hides the section. The renderer keys availability on this capability,
never on a 404 from one conversation: `conversation_not_found` is also the
answer for a non-owner or an out-of-window conversation.

## Routes

All four are `GET`, desktop-bearer gated, and **require**
`expected_account_id` (the captured signed-in account; a mismatch is `409
account_mismatch` before any hub request). `tenant_id` names the organization.
No endpoint parameter exists: the hub derives the audited agent from caller +
tenant, and `endpoint_id`, `actor_endpoint_id`, `purpose`, `scope` or any other
unknown or repeated query name is a `422 invalid_request` before any I/O.

| Route | Extra query | Result |
| --- | --- | --- |
| `/v1/desktop/team-bridge/owner-audit/conversations` | `limit`, `cursor` | `OwnerAuditConversationPage` |
| `.../owner-audit/conversations/{conversation_id}/messages` | `limit`, `before_seq` \| `after_seq` | `OwnerAuditMessagePage` (newest-first) |
| `.../owner-audit/conversations/{conversation_id}/messages/{message_id}/replies` | `limit`, `after_seq` | `OwnerAuditThread` (oldest-first) |
| `.../owner-audit/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}` | none | bytes |

`before_seq` is exclusive. `after_seq` is **inclusive** on both routes, the same
as the hub's member history (`after_seq=9` returns seq 9 and later), so a "load
newer" page must drop the boundary row it already shows.

JSON results ride the usual `CRUDResponse` envelope (`result`). The contract
models live in `local_operator/server/models/desktop_team_bridge_owner_audit.py`
and are published in the OpenAPI schema. Only the fields the hub sent are
forwarded (an open window's `agent.left_seq` is absent, not `null`), and unknown
additive hub fields pass through.

- `OwnerAuditConversation`: `id`, `tenant_id`, `kind`, `name?`, `topic?`,
  `visibility`, `status`, `builtin?`, `last_message_at`, `latest_seq`,
  `last_thread_root_id?`, `members[]`, `member_count`, `viewers?[]` (the
  disclosure: `{account_id, display_label, reason:"agent_owner",
  agent_member_key}`), and `agent` (`{account_id, display_label, state:
  active|left|removed, joined_seq, left_seq?}`). **No read state**: auditing
  raises no unread badge and no notification.
- `OwnerAuditMessage`: the hub's message view, with `body` always present for
  in-window messages, plus `agent_delivery_status` (`admitted` | `rejected` |
  `not_screened`; absent on the agent's own messages) and `agent_verdict_at`.
  Render rejected/unscreened bodies as plaintext, collapsed by default, with no
  link previews and no remote images.
- Attachment bytes: `image/png` and `image/jpeg` inline; a generic file is
  `application/octet-stream` with `Content-Disposition: attachment;
  filename="attachment.bin"`. Anything else is a `502 malformed_page`.

Every response, including refusals raised by the route, carries `Cache-Control:
private, no-store`. Hub refusals pass through unchanged: not your agent's, not
a member, outside the audit window, or unknown are all the same `404
conversation_not_found` (bytes: `attachment_not_found`), so the routes are no
existence oracle. A shape this build cannot read is `502 malformed_page`.

## Human-only, and what that does and does not mean

These bodies have **not** passed the agent's screening. They are served to the
human owner and never to a model.

**No agent tool path.** The `team_bridge` tool, `read_context`,
`model_boundary` and the connector cannot reach the `TeamBridgeClient.owner_audit_*`
methods. `tests/unit/team_bridge/test_owner_audit_isolation.py` enforces this
in three ways:
- statically: only the client, these routes and the contract module may name the
  surface, including the hyphenated route path;
- at the client: every hub request on `/agents/owned` outside the four methods is
  refused before dispatch, whatever spelling built the path;
- behaviourally: every tool op runs without one audit request.

The agent's own admitted-only read is unchanged.

**No desktop bearer for model-run code.** The routes are gated by the desktop
plane's bearer (`LOCAL_OPERATOR_DESKTOP_TOKEN`), so model-run code must not be
able to obtain it. Measured reaches, and what closes each one:

| Reach | Closed by |
| --- | --- |
| `bash`, `eval` (Python kernel) and their descendants inherit it from the session runtime's environment | withheld by name from every model child in BOTH `shell_environment` modes (`agent_shell.DAEMON_BEARER_ENVS`, dropped by ``without_daemon_bearer``) |
| a `PostToolUse` hook command (a model can write a project settings file naming one) and the `network` tool's `lop network` child | spawned with the same name-based withholding (`hook_forwarding.run_hook`, `network/tool.py`) |
| `ps eww $PPID` (Darwin) or `/proc/$PPID/environ` (Linux) on the session runtime | the runtime is launched without it (`session/runtime/launch.py`, `standby.py`, `mobile/daemon.py`) |
| `ps eww <daemon pid>` on the `lop serve` daemon | `server.desktop.adopt_spawn_bearer` moves the bearer onto an inherited pipe and re-execs the daemon once, same pid and argv, so the exec-time block never held the value; the plane is governed from the adopted copy (`TOKEN_FD_ENV`). `procname.is_own_launch` decides from what is RUNNING, not from the command line — the console script, or a `-m` of a module inside this package — so no interpreter flag (`-O`, `-Ou`, `-q`, `-E`, `-sP`, an attached `-mlocal_operator.cli`) can make it decline; a foreign program, `-m pip` or a `-c` program still answers no |
| the exec successor of a NORMAL daemon's in-place reload (`server/reload.py`) | handed over by pipe, never by environment, so it never re-enters a successor's exec-time block |
| a launch the predicate refuses (`--reload`, uvicorn's development auto-reloader; a non-product launch line) | **not closed**: those never re-exec, so the value stays in that process's exec-time block. Development mode only; the desktop app never starts it |
| the serve record (`run/serve/<pid>.json`) | an app-spawned daemon publishes `claim_key: ""`; a claimed daemon's key is spent by the claim |
| argv / inherited fds | never carried there |

**What this does NOT protect against.** Model-run code runs as the operator's
uid. This is not a boundary against same-uid code that goes looking:
- **Darwin:** the desktop app persists the bearer in a `0600` `desktop-token`
  file in its Application Support directory, so a same-uid process can read it.
- **Both platforms:** a same-uid process can read the Radient OAuth row in
  `auth.db` and call the hub's owner-audit routes directly, with no desktop
  bearer at all. It can also read process memory or the keychain.
- **Window:** the launch-environment window before the daemon's capture runs
  (milliseconds at boot) is the same one the app's own spawn opens.

Closing those needs a different uid or a sandbox for model commands. The
confined (benchmark) mode's seatbelt profile is such a sandbox for its shell:
it denies reads of `auth.db` under each config root and of the app's
`desktop-token` file at `~/Library/Application Support/Local Operator/` (and a
dev build's `local-operator-ui/`). A packaged app under ANOTHER product name, a
custom `userData` path, or any session that is not confined, is outside that
deny. The guarantee
here is that no harness path HANDS the bearer to the model, and that no
ambient read of the model's own or its parents' environment yields it.

Nothing is persisted: no bridge-store write, no body cache, no read marker.
The renderer's in-memory query cache is the only copy, and it is cleared on an
account change.

## Audit window

The hub serves `seq ∈ [joined_seq, left_seq)`, intersected with the caller's
membership start and the deployment floor `BRIDGE_OWNER_AUDIT_SINCE`. By the
operator's explicit choice while Team Bridge is in testing, the floor defaults
to unset (unbounded: everything since the agent joined). Revisit it before
general launch.
