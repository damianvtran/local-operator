# hi / wire.settings — remediation delta audit (le28r2-remed)

Remediation commit `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (parent `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`). Verdict: **PASS** for the 2 corrected key(s).

Scope is the delta only: the unchanged keys keep the `le28r2` judgement; this round does not re-audit them.

## Keys re-audited

### `wire.settings.proactive_patience_episode_ttl_ms.help` — fixed

Now: एक चक्र की कड़ी सीमा, इंतज़ार चाहे कैसे भी तय हुए हों (मिलीसेकंड; 7200000 = 2 h)। इसके बाद आने वाली जागृति चुपचाप बंद हो जाती है।

wake → «जागृति» (glossary + hi guide); feminine agreement with the feminine noun.

### `wire.settings.shell_environment_exclude.help` — fixed

Now: खाली = कुछ नहीं हटाया जाता। ऐसे नाम जो एजेंट के चलाए कमांड को किसी भी मोड में कभी नहीं दिए जाते — सत्र क्रेडेंशियल स्टोर के अपने इंजेक्शन समेत, यानी किसी वेरिएबल को सीधे मना करने का यही तरीक़ा है। सिर्फ़ bash और eval की चाइल्ड प्रोसेस पर लागू; MCP सर्वर और हार्नेस की अपनी प्रक्रियाएँ इसके दायरे में नहीं आतीं।

child processes named as processes; the abandoned «चिल्ड्रन» transliteration is gone.

## Checks

- Key set still equals the English source key set; ICU argument parity checked by `scripts/i18n/check.py --catalogues-only` on this tree.
- Routed `read_public_copy` receipt for the corrected catalogue at `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (session `c754bf63ab2a`, 2026-10-29T00:24:38Z); its digest was recomputed from the commit blob and matches.
- No English source, context sidecar, key module or generated artifact changed in the remediation commit.
