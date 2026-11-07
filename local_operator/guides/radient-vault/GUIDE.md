---
name: radient-vault
description: "Use a credential a teammate shared in Radient Vault, or one kept there for another machine; your own credentials stay in the local secret store."
requires: radient-vault
---

# Reading a teammate-shared credential from Radient Vault

The local encrypted store (`secret` store/retrieve, `lop secret`) stays the
default for every credential you or the user keep. Use Vault only for a value a
teammate shared, or one the user keeps in Vault to use from more than one
machine.

## Read one

`secret(op='vault_resolve', vault_id='vlt_…', item_id='vit_…', name='DEPLOY_KEY')`
— or pass the console item link (`…/dashboard/vaults/<vlt_>/items/<vit_>`) as
`item_id`. Take both ids from the user or their link; never guess one.

The value lands as a **session credential**: use `$DEPLOY_KEY` in `bash`. It is
never shown to you, lasts for this session only and is not saved locally. Never
echo it, write it to a file, or paste it into a message, a commit or a memory.

## When it refuses

- "no live Radient Vault access": the user runs `lop vault connect`, compares the
  printed fingerprint with the approval page, and approves. Then try once more.
- "isn't available yet": stop and tell the user; do not retry.
- Clock, membership, plan or sign-in refusals name their fix; relay it.

You cannot create or upload Vault items: the user adds values in the console.
