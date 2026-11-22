# Agent badge approved Reader input manifest

Input freeze only — independent A/B copy audits, rendered fit and native RTL/shaping approval remain pending. This index is not an audit receipt, ledger admission or application configuration.

- Repository: `damianvtran/local-operator`
- Frozen copy HEAD: `7d03eb5874ddbf8d3d16aeed8a12d7d204d68603`
- English source commit: `741a45d8465c5f4608e374a1bffeb9d2190065cf`
- Namespace/key: `wire.agent_badges` / `wire.agent_badges.packaged`
- English raw blob SHA-256: `5caa688fa6d6a78f315e32f7bb1db3b8cae08c77f727dd36af22e45a0206a23b`
- English context raw blob SHA-256: `8ab96370595b180339f4a6392daaef10a6f8324f202f3738dfc97574df5591e7`
- Author: actor `8a63c771f926`, actual model `openai/gpt-6.1-sol`; quota fallback disclosed verbatim in `../wire-agent-badges-author-8a63c771f926.json`.
- Raw author receipt SHA-256: `ca539cc1cacb46457d20a7e7fc5f264aaf9431e06720b87557d6abfe6c57307b`
- Source binding receipt: `../source-binding.json` (SHA-256 `69c208fcafeac2d22b3851fb6f8d98eca7445aa5d190a07651cc0fc1ac2b8812`).

## Supported capture contract

Every `*.reader.json` below preserves the actual `read_public_copy` tool result, including its returned `content`, `details`, tool-call identity and error status; no Reader envelope fields were reconstructed. The source contract is `docs/security/redaction-policy.md` §4, especially its committed-blob check and result-details provenance. Both `details.commit` and `details.checkout_head` equal the frozen copy HEAD; repository identity was verified. The JSON payload from the returned content was compared with the corresponding HEAD blob.

The Reader’s `details.sha256` hashes its selected returned text, not raw JSON file bytes. This manifest records raw HEAD-blob hashes separately; do not substitute one hash domain for another. Each actual envelope carries the selected-content digest.

## Exact approved payloads and bindings

| Locale | Approved value for `wire.agent_badges.packaged` | Raw HEAD blob SHA-256 | Actual Reader envelope | Envelope file SHA-256 |
|---|---|---|---|---|
| en | packaged | `5caa688fa6d6a78f315e32f7bb1db3b8cae08c77f727dd36af22e45a0206a23b` | [en.reader.json](en.reader.json) | `61488aee5b09453d9840e764f78138421def295f1de8160751e6cef316d31a3e` |
| zh-CN | 内置 | `7d4ee2c090bf96f8d280e84e057d208b65372000bccb851bba0b88c65666d3f1` | [zh-CN.reader.json](zh-CN.reader.json) | `631888506501bac2e90ad5ebe75bf519208555ffaa620b3f9d3ec8b46955ad91` |
| fr | fourni | `e4312efcb294f012872a50cb9adc1a64ca51b36abd730d10a97e172bd4cf46ab` | [fr.reader.json](fr.reader.json) | `0f37e41155f6d6d0ac8ed3afd23495f6e207d6a3eec0328f609d9b636e2ec973` |
| es | incluido | `393ca75c04e0e28adb8acd766c00850f07578fe54c2b7fe3307557b7eda333dc` | [es.reader.json](es.reader.json) | `f4615ab524c9dcd8f160d12096a2ac8dff3cbf536e54a490c4c1550c98d7a091` |
| ru | встроенный | `329a61c18aeee9cd811586349999e88f16b1e769ec5c54ff5c3e854412d1cff2` | [ru.reader.json](ru.reader.json) | `26741b96762a1b631362c5facaa9f126b04c47486f33ce67c33fa0d9a68a3f68` |
| vi | đi kèm | `ac9a9baac27826ad4966f87b1d26dbd8c9db94b26d7e9ffd3cb8cec56d49a8ca` | [vi.reader.json](vi.reader.json) | `a5acbaadb1db8b70f4cc5882a621b0636f33b130ad5e55daab22a77e0d7d42b4` |
| hi | साथ मिला | `f5e37710fac3b43a63ce5e85da03cd20151302440a29566e40ca83f432cacbc9` | [hi.reader.json](hi.reader.json) | `893a29fbf294aaae09178dc08f4573aac2836174a894c5b41b0f7c7b832005c2` |
| ur | ساتھ فراہم کردہ | `90185ca989752bde7e56c3269609eae6aceca184fca4b8f67845a9b42258365b` | [ur.reader.json](ur.reader.json) | `6cd6a583a7044923330d4a260a70a4668d1d60a1f1d04d1eb231ace62826e75f` |

English context approved Reader envelope: [en-context.reader.json](en-context.reader.json), capture SHA-256 `9e97f066e9ec1a0508708c2a3a025aa0673303be61d633508f77126152e82699`.

Its exact approved context payload is:

```json
{
 "wire.agent_badges.packaged": {
  "role": "agent provenance badge",
  "tone": "short, plain, lowercase",
  "maxCells": 27,
  "note": "Available built-in starter not yet present in the registry. Not installation status, trust, security, or permission. Shares ONE 27-cell facts budget with kind and proactive; not 27 cells per label. Preserve the proactive safety marker whole, spilling it or scope to another facts line when needed."
 }
}
```

## Width and provenance boundary

The 27-cell allowance is for the whole facts line, not each translated badge. The author’s combination estimates are estimates only; some kind/source/proactive combinations exceed that shared width and require whole-marker spill in the still-private implementation. No translation wording was edited here for fit. No generated eligibility, audit history, ledger or baseline was changed. No tests, generation, native dispatch or rendering were run in this metadata-only checkpoint.

Capturer: coder `05ab22c0ed05`, actual model `anthropic/claude-opus-5-5`; no fallback in this metadata checkpoint. The translation author is the distinct actor/model recorded above.
