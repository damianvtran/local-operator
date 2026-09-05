# Isolated runtime dependency policy

The adapter's `[tool.uv].override-dependencies` is authoritative when resolving
its optional OSWorld dependency set. Keep the current harness requirements;
do not downgrade shared libraries to satisfy stale upstream SDK metadata.

| Upstream declaration | Explicit override | Pilot installation |
| --- | --- | --- |
| OSWorld `requests~=2.31.0` | `requests>=2.32.0`, matching the harness | 2.34.2, unchanged from prior pilot |
| ZhipuAI `pyjwt>=2.8.0,<2.9.0` | `pyjwt[crypto]>=2.10,<3`, matching the harness | 2.13.0, matching the development environment |

The requests override was already declared and exercised by the preceding
pilot. Rebuilding against the current harness exposed the second conflict:
the old copied lock had PyJWT2.8.0, which no longer satisfies the harness.
Updating PyJWT satisfies the harness but exposes the optional ZhipuAI SDK's old
upper bound. The installed SDK is `zhipuai==2.1.5.20250825`; its current published
metadata still has that bound ([PyPI metadata](https://pypi.org/pypi/zhipuai/json)).
No upstream task, evaluator, framework source, or wheel metadata is rewritten.

## Actual compatibility evidence

The SDK's real `zhipuai.core._jwt_token.generate_token` helper was executed in
separate isolated interpreters with PyJWT2.8.0 and2.13.0, using identical fixed
time and a synthetic key. Both returned the same token SHA256:

```text
b8a059da3a7ff0ab0a4267332a61e69a5e98cce8e16d96adbbfa9e37503f1f43
```

Claims and the HS256/sign_type header were independently verified under each
version. No token was printed and no network/model request was made. This proves
the SDK's signing path, not an unperformed live ZhipuAI inference request. The
pilot actor and simulator use OpenRouter, not ZhipuAI.

The reusable regression is
`tests/unit/evaluation/adapters/osworld/test_dependency_policy.py`:

- The override specifiers and extras must exactly match the current harness.
- Superseded versions are rejected and the tested modern versions are admitted.
- The real optional SDK signing path must match the retained control digest.

The SDK test may skip in a lean development/CI environment lacking the optional
OSWorld tree; do not describe that skip as runtime compatibility validation.
The paid runtime deliberately has no pytest dependency. Run the exact shared
signing probe directly with that interpreter instead:

```sh
"$BENCHMARK_PYTHON" -I -B benchmarks/osworld_v2_adapter/probes/jwt_signing.py
```

The probe was run successfully with both isolated PyJWT versions above. A direct
attempt to run pytest in the paid environment correctly reported that pytest was
absent; no test dependencies were installed there.

## Honest dependency checks and pins

Plain `uv pip check` reads installed wheel metadata, not uv's project overrides.
It therefore still reports the two intentionally superseded upstream bounds.
Retain that raw output; do **not** describe it as a green pip check. Validate that
these are the only conflicts, that the installed versions satisfy the declared
overrides and current harness, and preserve the actual installed version list.
Any other conflict blocks admission until understood and resolved.

The new pilot environment changes PyJWT2.8.0 to2.13.0; requests remains2.34.2.
Record this alongside the source revision, wheel hashes, interpreter, complete
locked dependency list, and adapter/workspace digests. This policy does not
change normal TUI/mobile startup or install into an already-running environment.
