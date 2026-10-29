# fr / wire.settings — remediation delta audit (le28r2-remed)

Remediation commit `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (parent `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`). Verdict: **PASS** for the 2 corrected key(s).

Scope is the delta only: the unchanged keys keep the `le28r2` judgement; this round does not re-audit them.

## Keys re-audited

### `wire.settings.local_providers.base_url.label` — fixed

Now: endpoint {name}

Drops «de» before a provider display name whose vowel would require elision (Ollama, OpenAI-compatible). Correct: 15 cells, was 18.

### `wire.settings.local_providers.models.label` — fixed

Now: surcharges de modèle pour {name}

Same elision fix via «pour». 32 cells, was 30. No declared maxCells for this key; it was already above the inferred 29-cell row budget before this change (the fr over-budget label count is unchanged at 48).

## Checks

- Key set still equals the English source key set; ICU argument parity checked by `scripts/i18n/check.py --catalogues-only` on this tree.
- Routed `read_public_copy` receipt for the corrected catalogue at `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (session `c754bf63ab2a`, 2026-10-29T00:24:38Z); its digest was recomputed from the commit blob and matches.
- No English source, context sidecar, key module or generated artifact changed in the remediation commit.
