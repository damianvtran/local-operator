# Desktop Team Bridge owner audit ("{chief of staff}'s messages")

The signed-in human reads, read-only, every conversation **their own agent**
(the designated chief of staff) is or was in: DMs with humans and agents,
groups and channels. That includes messages the agent's screening **rejected**
or never screened. The hub side is agent-server #221
(`docs/TEAM-BRIDGE-CONVERSATIONS.md` §16); this is the runtime's proxy over it.

Capability: `features.team_bridge_owner_audit >= 1`. A renderer that does not
see it hides the section.

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

## Human-only

These bodies have **not** passed the agent's screening. They are served to the
human owner and never to a model: the `team_bridge` tool, `read_context`,
`model_boundary`, the connector and every other agent-context path cannot reach
the `TeamBridgeClient.owner_audit_*` methods.
`tests/unit/team_bridge/test_owner_audit_isolation.py` enforces this statically
(only the client, these routes and the contract module may name the surface)
and behaviourally (every tool op runs without a single audit request). The
agent's own admitted-only read is unchanged.

Nothing is persisted: no bridge-store write, no body cache, no read marker.
The renderer's in-memory query cache is the only copy, and it is cleared on an
account change.

## Audit window

The hub serves `seq ∈ [joined_seq, left_seq)`, intersected with the caller's
membership start and the deployment floor `BRIDGE_OWNER_AUDIT_SINCE`. By the
operator's explicit choice while Team Bridge is in testing, the floor defaults
to unset (unbounded: everything since the agent joined). Revisit it before
general launch.
